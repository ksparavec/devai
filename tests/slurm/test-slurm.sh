#!/usr/bin/env bash
# test-slurm.sh -- the live checks of platform-rebuild Phase 5 ("devai-slurm
# and the bus", Exit criteria), on the real card, with the stack down.
#
# It starts devai-slurm itself (as compose defines it: --pid=host, podman's
# default user namespace, devai-net only, nothing published, the tmpfs, the
# jobs directory and ~/.config/devai/slurm; first WITHOUT the GPU device, then
# with it), and two devai-control test containers from
# tests/slurm/Dockerfile.control-test, so jobs use the production subjects,
# users and namespaces:
#
#   devai-control-test-engines   default user namespace, root, the GPU;
#                                container `engines`, service `fake`
#                                (scripts/vram-ballast.py, 2 GiB)
#   devai-control-test-operator  --userns=keep-id, user devai, no GPU;
#                                container `operator`, service `actions`
#                                (tests/slurm/control-test/repo/actions.yaml)
#
# Refusals, before anything is touched:
#   - a production devai-control runs (devai-engines or devai-operator, or
#     any `devai-control --container` process on the host): both would answer
#     the same subjects and match the guard's command-line lookup;
#   - the stack is up: run `make cache-down slurm-down` first (and, once
#     devai-operator exists, `podman stop devai-operator`);
#   - the card is not idle, or an image, the secrets or the jobs directory
#     are missing (`make build-slurm slurm-init`; the directory is created
#     by hand, plan E9).
#
# An EXIT trap removes the test containers and the epilog test switch, and
# brings the stack back as compose defines it: `make cache-up slurm-up`
# (Phase 9 moves devai-slurm into cache-up; Phase 11 adds compose.gpu.yaml),
# then `make operator-up` if it found devai-operator stopped.
#
# The unkillable-holder check runs the ballast as host root through `sudo -n`.
# Without passwordless sudo it is skipped, and the script exits 77 when
# everything else passed: the operator runs that check by hand and records
# the result in docs/slurm.md.
#
# Exit: 0 every check passed; 1 a check failed or a refusal; 77 all ran
# checks passed but one was skipped.
#
# Environment: CONTAINER_RUNTIME (podman), DEVAI_GPU_DEVICE
# (nvidia.com/gpu=all), DEVAI_SLURM_CONFIG_DIR (~/.config/devai/slurm),
# SLURM_IMAGE (localhost/devai-slurm:latest), SLURM_TOOLS_DIST
# (devai-tools/bin/static, from `make slurm-tools`), APT_PROXY.
set -uo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
RUNTIME=${CONTAINER_RUNTIME:-podman}
JOBS=/var/cache/devai/jobs
SLURM_CONFIG_DIR=${DEVAI_SLURM_CONFIG_DIR:-$HOME/.config/devai/slurm}
SLURM_IMAGE=${SLURM_IMAGE:-localhost/devai-slurm:latest}
TEST_IMAGE=localhost/devai-control-test:latest
TOOLS=${SLURM_TOOLS_DIST:-$REPO/devai-tools/bin/static}
GPU_DEVICE=${DEVAI_GPU_DEVICE:-nvidia.com/gpu=all}
NET=devai-net
ENG=devai-control-test-engines
OPR=devai-control-test-operator
HOLDER=devai-control-test-holder
TMP=$(mktemp -d)
PASS=0
FAIL=0
SKIPPED=0
OPERATOR_WAS_STOPPED=0
SUDO_BALLAST_PID=

say()  { printf '\n== %s\n' "$*"; }
pass() { PASS=$((PASS + 1)); printf '  PASS  %s\n' "$*"; }
fail() { FAIL=$((FAIL + 1)); printf '  FAIL  %s\n' "$*"; }
skip() { SKIPPED=$((SKIPPED + 1)); printf '  SKIP  %s\n' "$*"; }
die()  { printf 'test-slurm: %s\n' "$*" >&2; exit 1; }
check() { local what=$1; shift; if "$@"; then pass "$what"; else fail "$what"; fi; }
not() { ! "$@"; }

# --- refusals ---------------------------------------------------------------

running() { [ "$("$RUNTIME" container inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" = true ]; }

for c in devai-engines devai-operator; do
    if running "$c"; then
        die "$c runs a production devai-control. Stop it first: make cache-down slurm-down; $RUNTIME stop devai-operator"
    fi
done
if pids=$(pgrep -af '[d]evai-control --container' 2>/dev/null) && [ -n "$pids" ]; then
    die "a devai-control runs on this host (it would answer the test's subjects):
$pids
Stop it first: make cache-down slurm-down; $RUNTIME stop devai-operator"
fi
up=$("$RUNTIME" ps --format '{{.Names}}' | grep -E '^devai-' || true)
if [ -n "$up" ]; then
    die "the stack is up ($(echo "$up" | tr '\n' ' ')). Bring it down first: make cache-down slurm-down"
fi
if "$RUNTIME" container exists devai-operator 2>/dev/null; then
    OPERATOR_WAS_STOPPED=1
fi
"$RUNTIME" image exists "$SLURM_IMAGE" || die "no $SLURM_IMAGE -- make build-slurm"
for f in slurm.key jwt_hs256.key mariadb.pass nats/jobs.pass nats/operator.pass nats/control-engines.pass nats/control-operator.pass; do
    [ -s "$SLURM_CONFIG_DIR/$f" ] || die "no $SLURM_CONFIG_DIR/$f -- make slurm-init"
done
[ -d "$JOBS" ] || die "no $JOBS -- create it (a plain directory; plan E9)"
for b in devai-control devai-bus; do
    [ -x "$TOOLS/$b" ] || die "no $TOOLS/$b -- make slurm-tools"
done
command -v nvidia-smi >/dev/null || die "no nvidia-smi: these checks need the card"
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
gpu_apps() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' '; }
[ -z "$(gpu_apps)" ] || die "the card is in use: $(gpu_apps | tr '\n' ' ')"
BASE=$(gpu_used)
[ "$BASE" -le 512 ] || die "the card shows $BASE MiB used with no process; the checks need it idle"
SUDO_OK=0
sudo -n true 2>/dev/null && SUDO_OK=1

# --- cleanup ----------------------------------------------------------------

cleanup() {
    local rc=$?
    say "cleanup"
    [ -n "$SUDO_BALLAST_PID" ] && sudo -n kill "$SUDO_BALLAST_PID" 2>/dev/null
    "$RUNTIME" exec devai-slurm rm -f /run/devai/test/epilog.conf >/dev/null 2>&1
    for c in "$ENG" "$OPR" "$HOLDER" devai-control-test-client; do
        "$RUNTIME" rm -f -t 5 "$c" >/dev/null 2>&1
    done
    "$RUNTIME" rm -f -t 60 devai-slurm >/dev/null 2>&1
    rm -rf "$TMP"
    echo "  bringing the stack back: make cache-up slurm-up"
    make -s -C "$REPO" cache-up >/dev/null 2>&1 || echo "  WARNING: make cache-up failed; run it by hand"
    make -s -C "$REPO" slurm-up >/dev/null 2>&1 || echo "  WARNING: make slurm-up failed; run it by hand"
    if [ "$OPERATOR_WAS_STOPPED" = 1 ] && make -s -C "$REPO" -n operator-up >/dev/null 2>&1; then
        make -s -C "$REPO" operator-up || echo "  WARNING: make operator-up failed"
    fi
    printf '\ntest-slurm: %d passed, %d failed, %d skipped\n' "$PASS" "$FAIL" "$SKIPPED"
    if [ "$rc" -eq 0 ]; then
        if [ "$FAIL" -gt 0 ]; then rc=1
        elif [ "$SKIPPED" -gt 0 ]; then rc=77
        fi
    fi
    exit "$rc"
}
trap cleanup EXIT

# --- helpers ----------------------------------------------------------------

sx() { "$RUNTIME" exec devai-slurm "$@"; }
djobs() { sx devai-jobs "$@"; }
# devai-bus as the operator, from inside devai-slurm (its password is there).
opbus() {
    sx env DEVAI_BUS_URL=nats://127.0.0.1:4222 DEVAI_BUS_USER=operator \
        DEVAI_BUS_PASSWORD_FILE=/etc/devai/slurm/nats/operator.pass devai-bus "$@"
}
jget() { python3 -c 'import json,sys
try:
    d = json.load(sys.stdin)
except ValueError:  # no answer (a request that timed out while polling)
    d = None
for k in sys.argv[1].split("."):
    d = d.get(k) if isinstance(d, dict) else None
print("" if d is None else d)' "$1"; }

fake_status() { opbus request devai.engines.fake.state '{}' --job 0 --timeout 5s 2>/dev/null | jget state.status; }
operator_answers() { opbus request devai.operator.actions.state '{}' --job 0 --timeout 3s >/dev/null 2>&1; }
node_state() { sx sinfo -h -o %T -n devai 2>/dev/null; }
node_is() { [[ "$(node_state)" =~ ^($1)$ ]]; }
pending_on() { [ "$(sx squeue -h -j "$1" -o %T/%r 2>/dev/null)" = "PENDING/$2" ]; }
exit_is() { [ "$(job_exit "$1")" = "$2" ] && [ -z "$(sx squeue -h -j "$1" 2>/dev/null)" ]; }
no_test_sleep() { ! pgrep -f 'test-sleep.sh' >/dev/null; }

job_state() {
    local s
    s=$(sx squeue -h -j "$1" -o %T 2>/dev/null)
    if [ -n "$s" ]; then echo "$s"; return; fi
    sx sacct -n -X -P -j "$1" -o state 2>/dev/null | head -1 | awk '{print $1}'
}
job_reason() { sx squeue -h -j "$1" -o %r 2>/dev/null; }
job_exit() { sx sacct -n -X -P -j "$1" -o exitcode 2>/dev/null | head -1; }

# wait_for SECONDS COMMAND...: true once COMMAND succeeds.
wait_for() {
    local t=$1; shift
    local end=$((SECONDS + t))
    until "$@"; do
        [ "$SECONDS" -ge "$end" ] && return 1
        sleep 1
    done
}
state_is() { [[ "$(job_state "$1")" =~ ^($2)$ ]]; }
fake_is() { [ "$(fake_status)" = "$1" ]; }
card_at_base() { [ "$(gpu_used)" -le $((BASE + 256)) ]; }
card_holds() { [ "$(gpu_used)" -ge $((BASE + 1800)) ]; }

LOAD='{"launch":{"backend":"fake","model":"fake-2g","ctx":4096},"timeout_s":840}'
submit_engine() { djobs submit engine --backend fake --load "$LOAD"; }
submit_action() { djobs submit action --action test-sleep --params "$1" "${@:2}"; }
submit_bench() {
    djobs submit bench --action test-bench --params "$1" --run t1 --task "${2:-gsm8k}" \
        --backend fake --model fake-2g --ctx 4096 "${@:3}"
}

epilog_off() { sx sh -c 'mkdir -p /run/devai/test && echo DEVAI_TEST_EPILOG=off > /run/devai/test/epilog.conf'; }
epilog_on() { sx rm -f /run/devai/test/epilog.conf; }

start_slurm() {
    local gpu=()
    [ "$1" = gpu ] && gpu=(--device "$GPU_DEVICE")
    "$RUNTIME" rm -f -t 60 devai-slurm >/dev/null 2>&1
    "$RUNTIME" network exists "$NET" 2>/dev/null || "$RUNTIME" network create "$NET" >/dev/null
    "$RUNTIME" run -d --name devai-slurm --hostname devai-slurm --network "$NET" --pid=host \
        --tmpfs /run/devai --security-opt label=disable "${gpu[@]}" \
        -e SLURM_REST_URL=http://127.0.0.1:6820 -e SLURM_JWT_KEY_FILE=/etc/slurm/jwt_hs256.key -e SLURM_USER=devai \
        -v "$JOBS:$JOBS" -v "$SLURM_CONFIG_DIR:/etc/devai/slurm:ro" \
        "$SLURM_IMAGE" >/dev/null || die "devai-slurm did not start"
    wait_for 180 slurm_ready || die "devai-slurm not ready within 180 s: $("$RUNTIME" logs --tail 30 devai-slurm 2>&1)"
}
slurm_ready() {
    [ "$(sx sinfo -h -o %T -n devai 2>/dev/null)" = idle ] &&
        sx sacctmgr -n show cluster format=cluster 2>/dev/null | grep -qw devai &&
        [ "$(sx sacctmgr -n -P show user devai format=adminlevel 2>/dev/null)" = Operator ]
}

control_yaml() {  # container user service-block
    cat <<EOF
container: $1
bus: {url: "nats://devai-slurm:4222", user: control-$1, password_file: /etc/devai/bus/control-$1.pass}
supervisor: {socket: $2, program_dir: $3}
state_file: $4
lease_timeout_s: 60
services:
$5
EOF
}

start_engines() {  # load delay in seconds
    control_yaml engines /run/supervisor.sock /run/devai/supervisor /run/devai/control.json \
        "  fake: {kind: fake, mib: 2048, load_delay_s: $1, script: /usr/local/lib/devai/vram-ballast.py, log_dir: /var/log/devai-fake}" \
        > "$TMP/engines.yaml"
    "$RUNTIME" rm -f -t 5 "$ENG" >/dev/null 2>&1
    "$RUNTIME" run -d --name "$ENG" --network "$NET" --device "$GPU_DEVICE" --security-opt label=disable \
        --tmpfs /run/devai \
        -v "$TMP/engines.yaml:/etc/devai/control.yaml:ro" \
        -v "$SLURM_CONFIG_DIR/nats/control-engines.pass:/etc/devai/bus/control-engines.pass:ro" \
        "$TEST_IMAGE" /usr/bin/supervisord -n -c /etc/devai/test/supervisord-engines.conf >/dev/null \
        || die "$ENG did not start"
    wait_for 30 fake_is idle || die "$ENG's devai-control does not answer: $("$RUNTIME" logs --tail 20 "$ENG" 2>&1)"
}

start_operator() {
    control_yaml operator /tmp/supervisor.sock /tmp/supervisor /tmp/control.json \
        "  actions: {kind: actions, registry: /opt/devai-test/actions.yaml, repo: /opt/devai-test, ops_env: []}" \
        > "$TMP/operator.yaml"
    # ops_env: [] -- test only: the test repository has no Makefile, so no
    # `make ops-env` (C8; Phase 6 adds the target).
    "$RUNTIME" rm -f -t 5 "$OPR" >/dev/null 2>&1
    "$RUNTIME" run -d --name "$OPR" --network "$NET" --userns=keep-id --user devai --security-opt label=disable \
        -v "$TMP/operator.yaml:/etc/devai/control.yaml:ro" \
        -v "$SLURM_CONFIG_DIR/nats/control-operator.pass:/etc/devai/bus/control-operator.pass:ro" \
        -v "$JOBS:$JOBS" \
        "$TEST_IMAGE" /usr/bin/supervisord -n -c /etc/devai/test/supervisord-operator.conf >/dev/null \
        || die "$OPR did not start"
    wait_for 30 operator_answers \
        || die "$OPR's devai-control does not answer: $("$RUNTIME" logs --tail 20 "$OPR" 2>&1)"
}

buslog_has() {  # pattern
    grep -qE "$1" "$JOBS"/bus/*.jsonl 2>/dev/null
}

# --- build the test image ---------------------------------------------------

say "build $TEST_IMAGE"
"$RUNTIME" build --network=host ${APT_PROXY:+--build-arg APT_PROXY=$APT_PROXY} \
    --build-arg DIST_ID="$(cat "$TOOLS/devai-control" "$TOOLS/devai-bus" | sha256sum | cut -c1-16)" \
    -v "$TOOLS:/var/cache/devai-tools-dist:ro" -f "$REPO/tests/slurm/Dockerfile.control-test" \
    -t "$TEST_IMAGE" "$REPO" >"$TMP/build.log" 2>&1 || die "build failed: $(tail -20 "$TMP/build.log")"
pass "built $TEST_IMAGE"

# --- without the GPU device -------------------------------------------------

say "devai-slurm without the GPU device"
start_slurm nogpu
start_operator
id=$(submit_action '{"sleep_s":2}') || fail "submit without a GPU"
check "an action job runs and completes without a GPU" wait_for 60 state_is "$id" COMPLETED
id=$(submit_engine)
check "an engine job's prolog skipped NVML" wait_for 30 test -s "$JOBS/results/$id/guard.json"
check "guard.json says gpu false" grep -q '"gpu": false' "$JOBS/results/$id/guard.json"
djobs cancel "$id" >/dev/null
wait_for 90 state_is "$id" 'CANCELLED|COMPLETED|FAILED'
"$RUNTIME" rm -f -t 5 "$OPR" >/dev/null

# --- with the GPU: Slurm itself ---------------------------------------------

say "devai-slurm with the GPU: node, accounting, REST"
start_slurm gpu
check "sinfo shows devai idle" [ "$(sx sinfo -h -o %T -n devai)" = idle ]
cluster_listed() { sx sacctmgr -n show cluster format=cluster | grep -qw devai; }
check "sacctmgr lists the cluster devai" cluster_listed
check "devai has AdminLevel=Operator" [ "$(sx sacctmgr -n -P show user devai format=adminlevel)" = Operator ]
check "podman port devai-slurm lists nothing" [ -z "$("$RUNTIME" port devai-slurm)" ]
"$RUNTIME" restart -t 60 devai-slurm >/dev/null
wait_for 180 slurm_ready
check "AdminLevel=Operator after podman restart" [ "$(sx sacctmgr -n -P show user devai format=adminlevel)" = Operator ]

client() {  # [-e VAR=value]... devai-jobs args: a container on devai-net that signs its own JWT (pkg/slurmrest)
    local envs=()
    while [ "${1:-}" = -e ]; do envs+=(-e "$2"); shift 2; done
    "$RUNTIME" run --rm --name devai-control-test-client --network "$NET" --security-opt label=disable \
        -v "$SLURM_CONFIG_DIR/jwt_hs256.key:/key:ro" \
        -e SLURM_REST_URL=http://devai-slurm:6820 -e SLURM_JWT_KEY_FILE=/key "${envs[@]}" \
        --entrypoint devai-jobs "$SLURM_IMAGE" "$@"
}
start_engines 0
start_operator
id=$(client submit action --action test-sleep --params '{"sleep_s":1}' --requester 'test "client" {x}' 2>/dev/null)
if [ -n "$id" ]; then
    check "a REST-submitted job runs" wait_for 60 state_is "$id" COMPLETED
    comment=$(sx sacct -n -X -P -j "$id" -o comment%1000 | head -1)
    check "its JSON comment arrives intact in sacct" python3 -c 'import json,sys; c=json.loads(sys.argv[1]); sys.exit(c.get("requester") != "test \"client\" {x}" or c.get("kind") != "action" or len(c.get("hold_key", "")) != 32)' "$comment"
else
    fail "submit through slurmrestd from another container"
fi
for u in root nobody; do
    before=$(sx sacct -n -X -a -P -o jobid | wc -l)
    if client -e SLURM_USER="$u" submit action --action test-sleep --params '{"sleep_s":1}' >/dev/null 2>&1; then
        fail "a JWT for $u was accepted"
    else
        after=$(sx sacct -n -X -a -P -o jobid | wc -l)
        check "a JWT for $u is refused at submission and nothing runs" [ "$before" = "$after" ]
    fi
done

say "two engine jobs, the license, the guard event, devai-jobs on the host"
a=$(submit_engine); b=$(submit_engine)
check "the first engine job loads the fake engine (2 GiB held)" wait_for 60 card_holds
# Right after a submission squeue shows reason None until the scheduler has
# looked at the job; the license is the reason it then gives.
check "the second -L engine job waits PENDING (Licenses)" wait_for 10 pending_on "$b" Licenses
host_list_json() { "$REPO/bin/devai-jobs" list --json | python3 -c 'import json,sys; json.load(sys.stdin)'; }
check "bin/devai-jobs list --json succeeds" host_list_json
check "the prolog's devai.events.slurm.guard event is in the bus log" buslog_has "devai.events.slurm.guard.*\"job\": ?${a}[,}]"
# Hold the first job's model through one devai-gpu sampling interval (5 s):
# a job cancelled within a second of its load has only the sampler's first
# sample, taken before the model was there (2026-10-09: peak 480 MiB).
sleep 6
djobs cancel "$a" >/dev/null
check "scancel unloads the fake engine; the card is back at its baseline" wait_for 90 card_at_base
gpu_summary_ok() { python3 -c 'import json,sys; g=json.load(open(sys.argv[1])); sys.exit(not g["samples"] or g["peak_mib"] < int(sys.argv[2]))' "$JOBS/results/$1/gpu.json" $((BASE + 1800)); }
check "gpu.json holds the GPU summary" wait_for 30 gpu_summary_ok "$a"
check "the next engine job starts once the first one's epilog has ended" wait_for 120 card_holds
b_start=$(sx sacct -n -X -P -j "$b" -o start | head -1)
a_end=$(sx sacct -n -X -P -j "$a" -o end | head -1)
check "the second job started after the first ended (sacct: $a_end <= $b_start)" [ "$(date -d "$a_end" +%s)" -le "$(date -d "$b_start" +%s)" ]

say "the job script killed with SIGKILL"
sx scancel --signal=KILL --batch "$b"
check "the epilog unloads after a SIGKILLed job script" wait_for 90 card_at_base
epilog_off
c=$(submit_engine)
wait_for 60 card_holds
sx scancel --signal=KILL --batch "$c"
check "with the epilog switched off, the lease unloads within 70 s" wait_for 70 card_at_base
epilog_on

say "the load delay: 90 s and 150 s (longer than the 60 s lease)"
start_engines 90
d=$(submit_engine)
check "a 90 s load reaches serving" wait_for 150 fake_is serving
sleep 70
check "and stays loaded while its job leases" fake_is serving
djobs cancel "$d" >/dev/null; wait_for 90 card_at_base
start_engines 150
e=$(submit_engine)
check "a 150 s load reaches serving" wait_for 220 fake_is serving
check "and its devai-bus request exited 0 (the job holds it)" state_is "$e" RUNNING
# The reply itself, not the serving event, must have arrived: devai-control
# may answer for 15 minutes (C1's allow_responses), not nats-server's 2.
reply_not_event() { ! grep -q 'took job .* as the reply' "$JOBS/results/$1/job.log"; }
check "the load was answered by its reply after 150 s (the 15-minute response permission)" reply_not_event "$e"
djobs cancel "$e" >/dev/null; wait_for 90 card_at_base
epilog_off
f=$(submit_engine)
sleep 20
sx scancel --signal=KILL --batch "$f"
check "a job killed during its 150 s load: the model still reaches serving" wait_for 220 fake_is serving
served=$SECONDS
check "and the lease unloads it within 70 s of serving" wait_for 75 card_at_base
[ $((SECONDS - served)) -ge 50 ] || fail "unloaded only $((SECONDS - served)) s after serving"
epilog_on

say "cancel while loading; an epilog's unload during the trap's"
start_engines 30
g=$(submit_engine)
wait_for 30 fake_is loading
djobs cancel "$g" >/dev/null
# The epilog's unload, sent while the trap's runs (as the epilog sends it:
# --job = that job, no force): it returns when that one ends.
t0=$SECONDS
opbus request devai.engines.fake.unload '{"timeout_s":60}' --job "$g" --timeout 70s >"$TMP/u1" 2>&1
t1=$SECONDS
check "an unload during the trap's returns ok with it" grep -q '"ok":true' "$TMP/u1"
opbus request devai.engines.fake.unload '{}' --job "$g" --timeout 70s >"$TMP/u2" 2>&1
check "one after it returns ok at once" [ $((SECONDS - t1)) -le 3 ]
check "it was ok" grep -q '"ok":true' "$TMP/u2"
: "$t0"
check "the trap's unload stops the load; the card is back at its baseline" wait_for 90 card_at_base
check "the cancelled-while-loading job ends 0" wait_for 90 exit_is "$g" 0:0
check "its epilog passed: the node is not drained" not node_is 'drained|draining' 
start_engines 0

say "the operator's supervisor verbs on the fake engine"
h=$(submit_engine)
wait_for 60 fake_is serving
rest=$(opbus request devai.engines.fake.restart '{}' --job 0 --timeout 10s)
check "restart on a serving fake engine is refused busy" grep -q '"code":"busy"' <<<"$rest"
opbus request devai.engines.fake.stop '{}' --job 0 --timeout 90s >/dev/null
check "stop as user operator unloads the fake engine" wait_for 60 card_at_base
completed_0() { [ "$(sx sacct -n -X -P -j "$1" -o state,exitcode | head -1)" = 'COMPLETED|0:0' ]; }
check "its job ends 0" wait_for 90 completed_0 "$h"
check "every event of it is in the bus log" buslog_has "devai.events.engines.fake.*\"job\": ?${h}[,}]"

say "action jobs"
i=$(submit_action '{"sleep_s":600}')
wait_for 30 state_is "$i" RUNNING; sleep 3
sx scancel --signal=KILL --batch "$i"
check "an action job's script SIGKILLed: its process group stops within 70 s" wait_for 75 no_test_sleep
j=$(submit_action '{"sleep_s":600,"stop_s":15}')
wait_for 30 state_is "$j" RUNNING; sleep 3
t0=$SECONDS
djobs cancel "$j" >/dev/null
sleep 8
check "scancel of an action that needs 15 s to stop: the job stays RUNNING or COMPLETING" state_is "$j" 'RUNNING|COMPLETING'
wait_for 60 state_is "$j" 'CANCELLED|COMPLETED|FAILED'
check "and it ends only after the action has exited (>= 15 s)" [ $((SECONDS - t0)) -ge 14 ]

# scancel during the `run` request: devai-control is paused, so the request
# is in flight when TERM comes; once it runs again it gets `run`, then the
# trap's `stop`, and the job ends only with the run's `exited`.
"$RUNTIME" pause "$OPR" >/dev/null
j2=$(submit_action '{"sleep_s":600,"stop_s":15}')
wait_for 30 state_is "$j2" RUNNING; sleep 3
djobs cancel "$j2" >/dev/null
sleep 5
check "scancel during the run request: the job stays RUNNING or COMPLETING" state_is "$j2" 'RUNNING|COMPLETING'
"$RUNTIME" unpause "$OPR" >/dev/null
check "and ends once the run has exited" wait_for 90 state_is "$j2" 'CANCELLED|COMPLETED|FAILED'
no_run_left() { ! grep -q '"status":"running"' <<<"$(opbus request devai.operator.actions.state '{}' --job "$j2" --timeout 5s 2>/dev/null)"; }
check "no run of it is left running" no_run_left

say "bench jobs and the bench license"
k=$(submit_bench '{"sleep_s":600,"stop_s":15}')
wait_for 30 state_is "$k" RUNNING; sleep 3
l=$(submit_bench '{"sleep_s":5}' mmlu)
check "a second -L bench job waits PENDING (Licenses)" wait_for 10 pending_on "$l" Licenses
djobs cancel "$k" >/dev/null
sleep 8
check "while the first bench run stops (15 s), the second still waits" state_is "$l" PENDING
check "the second starts after the first's run exited and its epilog ended" wait_for 90 state_is "$l" 'RUNNING|COMPLETED'
wait_for 60 state_is "$l" COMPLETED
k_end=$(sx sacct -n -X -P -j "$k" -o end | head -1); l_start=$(sx sacct -n -X -P -j "$l" -o start | head -1)
check "sacct: the second bench job started at or after the first ended ($k_end <= $l_start)" [ "$(date -d "$k_end" +%s)" -le "$(date -d "$l_start" +%s)" ]

# The B:INT@120 warning: a 3-minute bench job gets SIGINT after about a
# minute; its run scores its prefix (15 s here), writes result.json
# (truncated) and exits; only then may the next bench job start.
k2=$(submit_bench '{"sleep_s":600,"stop_s":15}' gsm8k --time-limit-min 3)
wait_for 30 state_is "$k2" RUNNING; sleep 2
l2=$(submit_bench '{"sleep_s":5}' mmlu)
check "the next bench job waits for the license" wait_for 10 pending_on "$l2" Licenses
check "the warning ends the first one's run (prefix scored)" wait_for 150 state_is "$k2" 'COMPLETED|CANCELLED|FAILED|TIMEOUT'
check "its run wrote result.json, truncated" grep -q '"status":"truncated"' "$JOBS/results/$k2/result.json"
check "the next bench job then starts" wait_for 90 state_is "$l2" 'RUNNING|COMPLETED'
k2_end=$(sx sacct -n -X -P -j "$k2" -o end | head -1); l2_start=$(sx sacct -n -X -P -j "$l2" -o start | head -1)
check "and not before the first ended ($k2_end <= $l2_start)" [ "$(date -d "$k2_end" +%s)" -le "$(date -d "$l2_start" +%s)" ]
wait_for 60 state_is "$l2" COMPLETED

say "ownership and isolation (I26)"
m=$(submit_action '{"sleep_s":1}')
wait_for 60 state_is "$m" 'COMPLETED|FAILED'
check "action.log and result.json are written in results/$m" test -s "$JOBS/results/$m/action.log" -a -s "$JOBS/results/$m/result.json"
all_group_mine() { [ -z "$(find "$JOBS/results/$1" ! -group "$(id -g)")" ]; }
all_readable() { find "$JOBS/results/$1" -type f -exec cat {} + >/dev/null; }
check "every file carries the host user's group" all_group_mine "$m"
check "the host user can read every file" all_readable "$m"
check "the host user can delete the directory" rm -rf "$JOBS/results/$m"
# A job runs as devai in devai-slurm; these two run as that same user there.
devai_reads_key() { "$RUNTIME" exec -u devai devai-slurm cat /etc/slurm/slurm.key >/dev/null 2>&1; }
check "a job (user devai in devai-slurm) cannot read slurm.key" not devai_reads_key
sleep 600 & hostpid=$!
devai_signals() { "$RUNTIME" exec -u devai devai-slurm kill -0 "$1" 2>/dev/null; }
check "a job (user devai in devai-slurm) cannot signal a host user's process" not devai_signals "$hostpid"
kill "$hostpid" 2>/dev/null

say "the guard and foreign holders"
guard_verdict_is() { grep -q "\"verdict\": \"$2\"" "$JOBS/results/$1/guard.json" 2>/dev/null; }
for who in root 2000; do
    "$RUNTIME" rm -f -t 0 "$HOLDER" >/dev/null 2>&1
    "$RUNTIME" run -d --rm --name "$HOLDER" --device "$GPU_DEVICE" --security-opt label=disable --user "$who" \
        "$TEST_IMAGE" python3 /usr/local/lib/devai/vram-ballast.py 2048 >/dev/null
    wait_for 30 card_holds
    n=$(submit_engine)
    check "the prolog kills a foreign GPU holder (uid $who) in a plain rootless container" wait_for 90 guard_verdict_is "$n" killed
    check "and the engine job loads" wait_for 90 fake_is serving
    djobs cancel "$n" >/dev/null; wait_for 90 card_at_base
done
"$RUNTIME" rm -f -t 0 "$HOLDER" >/dev/null 2>&1

if [ "$SUDO_OK" = 1 ]; then
    sudo -n python3 "$REPO/scripts/vram-ballast.py" 2048 >/dev/null 2>&1 &
    wait_for 30 card_holds
    SUDO_BALLAST_PID=$(sudo -n pgrep -f 'vram-ballast.py 2048' | head -1)
    o=$(submit_engine)
    check "a holder the guard cannot kill drains the node" wait_for 120 node_is 'drained|draining'
    job_held() { sx squeue -h -j "$1" -o %r | grep -qi held; }
    check "and its job is held" wait_for 60 job_held "$o"
    p=$(submit_action '{"sleep_s":5}')
    sleep 5
    check "a license-free action job waits PENDING while drained" state_is "$p" PENDING
    djobs queue | grep -E "^$p " || true
    sudo -n kill "$SUDO_BALLAST_PID"; SUDO_BALLAST_PID=
    wait_for 30 card_at_base
    check "the health check resumes the node within 60 s of the holder exiting" wait_for 60 node_is 'idle|mixed|allocated' 
    check "and the action job starts within 60 s of that" wait_for 60 state_is "$p" 'RUNNING|COMPLETED'
    djobs cancel "$o" >/dev/null 2>&1
else
    skip "the unkillable-holder check needs passwordless sudo: run it by hand and record it in docs/slurm.md"
fi

say "node state and ReturnToService"
sx scontrol update NodeName=devai State=DOWN Reason=test
sx supervisorctl -c /etc/devai/supervisord.conf restart slurmd >/dev/null
check "after DOWN and a slurmd restart the node is idle within 60 s" wait_for 60 node_is idle

say "the bus rides out a restart"
q=$(submit_engine); r=$(submit_action '{"sleep_s":600}')
wait_for 60 fake_is serving; wait_for 30 state_is "$r" RUNNING
sx supervisorctl -c /etc/devai/supervisord.conf restart nats-server >/dev/null
sleep 15
both_running() { [ "$(sx squeue -h -j "$1,$2" -o %T | sort -u)" = RUNNING ]; }
check "after supervisorctl restart nats-server both jobs are RUNNING" both_running "$q" "$r"
check "and the fake engine is still loaded" fake_is serving
sx supervisorctl -c /etc/devai/supervisord.conf stop nats-server >/dev/null
sleep 70
check "with nats-server stopped 70 s, the lease unloads the fake engine" card_at_base
sx supervisorctl -c /etc/devai/supervisord.conf start nats-server >/dev/null
check "once the bus is back the engine job's hold sees it and the job ends 0" wait_for 60 exit_is "$q" 0:0
djobs cancel "$r" >/dev/null 2>&1; wait_for 90 state_is "$r" 'CANCELLED|COMPLETED|FAILED|LOST'

say "devai-slurm restarted while jobs run: NODE_FAIL, never run again"
# The jobs die with the container; node-recover sets the node DOWN before
# slurmd starts, so Slurm ends them NODE_FAIL at once, and slurmd's
# registration returns the node to service. (Slurm alone ends them only at a
# registration more than BatchStartTimeout after they were last seen, tens
# of minutes later: on 2026-10-09 both were still RUNNING 260 s after the
# restart, the engine job holding the engine license.)
s=$(submit_engine); u=$(submit_action '{"sleep_s":900}')
wait_for 60 fake_is serving; wait_for 30 state_is "$u" RUNNING
"$RUNTIME" restart -t 60 devai-slurm >/dev/null
wait_for 180 slurm_ready
check "the engine job ended NODE_FAIL" wait_for 60 state_is "$s" NODE_FAIL
check "the action job ended NODE_FAIL" wait_for 60 state_is "$u" NODE_FAIL
check "the node is back in service" wait_for 60 node_is idle
check "neither ran a second time" [ "$(sx sacct -n -P -X -j "$s,$u" -o restarts 2>/dev/null | sort -u)" = 0 ]
check "the lease unloaded the fake engine" wait_for 90 card_at_base

exit 0
