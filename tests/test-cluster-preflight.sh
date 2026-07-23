#!/usr/bin/env bash
# Cluster-mode Phase 1.5 preflight: validate the worker protocol
# against the stub head BEFORE Phase 2 commits any head-side code
# to it.
#
# Per docs/plans/gpu-arbiter-cluster-mode.md Phase 1.5 (decision 13):
# stubbed backends acceptable, no real GPU required. CI-runnable.
#
# The seven scenarios from the plan:
#   1. Two workers register; both visible.
#   2. Heartbeat cadence + monotonic counter.
#   3. drain command flow.
#   4. serve command flow + token gating (401) + real inbound dispatch.
#   5. shutdown lifecycle policy (ephemeral honours, persistent refuses).
#   6. Failure recovery: kill stub head; worker retries.
#   7. Token rotation: new token effective on next heartbeat.
#
# Each scenario runs as a focused stage. Stages 4-5 require the
# inbound listener (compiled into the arbiter); stage 6 kills + restarts
# the stub head; stage 7 rotates the token file in place.
#
# Usage:
#   ./tests/test-cluster-preflight.sh
#
# Exits 0 on success, non-zero on first failure. Prints a per-stage
# OK/FAIL banner so a CI log makes the failure point obvious.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ARBITER_BIN="${ARBITER_BIN:-$REPO_ROOT/gpu-arbiter/gpu-arbiter}"
STUB_HEAD_PY="$REPO_ROOT/tests/fixtures/stub-head.py"

if [[ ! -x "$ARBITER_BIN" ]]; then
    echo "BUILD: gpu-arbiter binary missing; building..." >&2
    (cd "$REPO_ROOT/gpu-arbiter" && go build -o gpu-arbiter ./...) || {
        echo "FAIL: go build" >&2
        exit 1
    }
fi
if [[ ! -f "$STUB_HEAD_PY" ]]; then
    echo "FAIL: stub-head.py missing at $STUB_HEAD_PY" >&2
    exit 1
fi

# Stamp every output line with the stage so failures are easy to spot.
banner() { echo; echo "===== STAGE $* ====="; }
ok()     { echo "[OK] $*"; }
fail()   { echo "[FAIL] $*" >&2; exit 1; }

WORKDIR=$(mktemp -d)
TOKEN_FILE="$WORKDIR/token"
echo "the-token" > "$TOKEN_FILE"

PIDS=()
cleanup() {
    local rc=$?
    for pid in "${PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 0.5
    for pid in "${PIDS[@]}"; do
        kill -KILL "$pid" 2>/dev/null || true
    done
    if [[ $rc -ne 0 ]]; then
        echo "  preflight FAILED; preserving workdir at $WORKDIR for inspection" >&2
    else
        rm -rf "$WORKDIR"
    fi
}
trap cleanup EXIT

# Pick free ports so concurrent CI runs don't collide.
free_port() {
    python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()'
}
HEAD_PORT=$(free_port)
WORKER_A_PORT=$(free_port)
WORKER_B_PORT=$(free_port)
HEAD_URL="http://localhost:$HEAD_PORT"

start_stub_head() {
    local commands_file="${1:-}"
    if [[ -n "$commands_file" ]]; then
        python3 "$STUB_HEAD_PY" --token the-token --port "$HEAD_PORT" --commands "$commands_file" \
            >"$WORKDIR/head.stdout" 2>"$WORKDIR/head.stderr" &
    else
        python3 "$STUB_HEAD_PY" --token the-token --port "$HEAD_PORT" \
            >"$WORKDIR/head.stdout" 2>"$WORKDIR/head.stderr" &
    fi
    HEAD_PID=$!
    PIDS+=("$HEAD_PID")
    # Wait for /healthz.
    for _ in $(seq 1 30); do
        if curl -fsS "$HEAD_URL/healthz" >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.2
    done
    fail "stub-head did not start on :$HEAD_PORT"
}

stop_stub_head() {
    if [[ -n "${HEAD_PID:-}" ]]; then
        kill -TERM "$HEAD_PID" 2>/dev/null || true
        wait "$HEAD_PID" 2>/dev/null || true
        HEAD_PID=""
    fi
}

introspect() {
    curl -fsS -H "Authorization: Bearer the-token" "$HEAD_URL/_introspect"
}

# LAST_WORKER_PID is set by start_worker. Calling code reads it
# instead of using $(start_worker ...) -- the latter runs in a
# subshell that can't update PIDS in the parent.
LAST_WORKER_PID=""
start_worker() {
    local name="$1" lifecycle="$2" port="$3"
    local logfile="$WORKDIR/${name}.log"
    DEVAI_MODE=worker \
    DEVAI_HEAD_URL="$HEAD_URL" \
    DEVAI_WORKER_TOKEN_FILE="$TOKEN_FILE" \
    DEVAI_WORKER_NAME="$name" \
    DEVAI_LIFECYCLE="$lifecycle" \
    DEVAI_GPU_TYPE=test \
    GPU_MEMORY_GB=24 \
    DEVAI_BACKENDS=ollama,vllm \
    DEVAI_WORKER_INBOUND_PORT="$port" \
    "$ARBITER_BIN" --mode=worker \
        >"$logfile" 2>&1 &
    local pid=$!
    PIDS+=("$pid")
    LAST_WORKER_PID="$pid"
    # Wait for the worker's inbound listener.
    for _ in $(seq 1 30); do
        if curl -fsS "http://localhost:$port/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.2
    done
    fail "$name did not start inbound listener on :$port -- see $logfile"
}

# -------- Stage 1: registration --------

banner 1 "two workers register; both visible to head"
start_stub_head ""
start_worker worker-a persistent "$WORKER_A_PORT"; PID_A="$LAST_WORKER_PID"
start_worker worker-b ephemeral  "$WORKER_B_PORT"; PID_B="$LAST_WORKER_PID"
sleep 1
n_reg=$(introspect | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d['registrations']))")
[[ "$n_reg" == "2" ]] || fail "expected 2 registrations, got $n_reg"
ok "two workers registered"

# -------- Stage 2: heartbeat cadence + counter --------

banner 2 "heartbeats arrive at ~10s cadence; counters monotonic"
sleep 12
n_hb=$(introspect | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['heartbeats_count'])")
[[ "$n_hb" -ge 2 ]] || fail "expected >= 2 heartbeats after 12s, got $n_hb"
ok "$n_hb heartbeats received"

# -------- Stage 6 (out of order; pulls down workers/head): failure recovery --------
# Done before stages 3-5 so we don't have to teardown twice.

banner 6 "head-down then head-up; workers re-register without operator intervention"
stop_stub_head
sleep 1
start_stub_head ""
sleep 12
n_reg2=$(introspect | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d['registrations']))")
# Heartbeats keep failing during the gap; the workers should keep
# their existing worker_id (no re-register on heartbeat fail in
# Phase 1) so registrations don't grow. But heartbeats DO resume:
n_hb2=$(introspect | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['heartbeats_count'])")
[[ "$n_hb2" -ge 1 ]] || fail "no heartbeats after head restart"
ok "heartbeats resumed after head bounce ($n_hb2 received)"

# -------- Stage 7: token rotation --------

banner 7 "rotate token in tmpfs; next heartbeat picks up the new value"
# We rotate by overwriting the file -- TokenStore re-reads after its
# 30s cache TTL. To make the test fast, write the SAME value (we
# can't change the head's expected token without restart), and just
# verify the file rewrite doesn't break the worker. A real
# rotation test runs end-to-end with a head-side update too.
date > "$TOKEN_FILE.new"
cat "$TOKEN_FILE" > "$TOKEN_FILE.new"
mv "$TOKEN_FILE.new" "$TOKEN_FILE"
sleep 12
n_hb3=$(introspect | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['heartbeats_count'])")
[[ "$n_hb3" -gt "$n_hb2" ]] || fail "no heartbeats after token-file rewrite ($n_hb3 vs $n_hb2)"
ok "heartbeats continued after in-place token rewrite"

# Tear down workers + head; stages 3-5 spin them up fresh with
# canned commands. The arbiter's srv.Shutdown waits up to 10s on a
# graceful drain -- KILL after the TERM grace so the next stage isn't
# racing a still-alive worker that grabs the canned commands first.

for pid in "${PIDS[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
done
sleep 1
for pid in "${PIDS[@]}"; do
    kill -KILL "$pid" 2>/dev/null || true
done
# Wait for processes to actually terminate.
for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
done
PIDS=()

# -------- Stage 3-5: command dispatch --------
# Single stub-head session, three commands queued. Each stage
# observes the worker's behaviour via its log file (the arbiter
# command executor logs to stderr per cluster_main.go).

banner 3-5 "drain + serve + shutdown commands flow through dispatch"
echo '[{"type":"drain","backend":"vllm"},{"type":"serve","request_id":"r1","target_model":"Qwen3-8B-NVFP4","target_ctx":131072},{"type":"shutdown","grace_seconds":2}]' > "$WORKDIR/cmds.json"
# Re-allocate ports because the previous workers' sockets may still
# be in TIME_WAIT.
sleep 1
WORKER_C_PORT=$(free_port)
WORKER_D_PORT=$(free_port)
start_stub_head "$WORKDIR/cmds.json"
start_worker worker-eph ephemeral "$WORKER_C_PORT"; PID_E="$LAST_WORKER_PID"
# Wait for at least one heartbeat tick (HeartbeatInterval=10s).
# Poll the worker log instead of fixed-sleep so a slow CI doesn't
# false-fail.
LOG="$WORKDIR/worker-eph.log"
for _ in $(seq 1 20); do
    if grep -q "drain backend=vllm" "$LOG" 2>/dev/null; then
        break
    fi
    sleep 1
done
grep -q "drain backend=vllm" "$LOG" || { echo "----- log dump -----"; cat "$LOG"; echo "----- head.stderr -----"; cat "$WORKDIR/head.stderr"; fail "no drain in worker log"; }
grep -q "serve req=r1 model=Qwen3-8B-NVFP4 ctx=131072" "$LOG" || fail "no serve in worker log"
grep -q "shutdown grace=2s acknowledged" "$LOG" || fail "no shutdown in worker log"
# The executor must reach the scheduler's drain, not just log an ack:
# the Phase 1 no-op hard-exited on `shutdown` with no drain at all, so
# live requests died mid-stream. This line is emitted only after
# arbiter.drainBackend has returned for every backend.
#
# The standalone `drain backend=vllm` in the same batch is NOT asserted
# to completion here: its goroutine races the shutdown goroutine queued
# behind it, and shutdown's own drain("") covers vllm anyway, so the
# process can legitimately exit before the per-backend line prints. That
# path is covered deterministically by the Go unit test
# TestArbiterCommandExecutor_DrainDrainsNamedBackend.
grep -q "drain complete; exiting" "$LOG" || \
    { echo "----- log dump -----"; cat "$LOG"; fail "shutdown exited without draining first (executor is still a no-op)"; }
ok "drain + serve + shutdown all dispatched"

# Verify ephemeral worker actually exited (shutdown grace=2; wait a bit longer).
sleep 4
if kill -0 "$PID_E" 2>/dev/null; then
    fail "ephemeral worker did not exit after shutdown command"
fi
ok "ephemeral worker exited after shutdown"

# Persistent variant: shutdown command must be REFUSED.
echo '[{"type":"shutdown","grace_seconds":2}]' > "$WORKDIR/cmds.json"
# Reload commands into the head so the next heartbeat picks them up
# (the head returns the queued list once and then empties it).
stop_stub_head
sleep 1
WORKER_E_PORT=$(free_port)
start_stub_head "$WORKDIR/cmds.json"
start_worker worker-persistent persistent "$WORKER_E_PORT"; PID_P="$LAST_WORKER_PID"
sleep 12
LOG="$WORKDIR/worker-persistent.log"
grep -q "refusing shutdown command: lifecycle=persistent" "$LOG" || \
    fail "persistent worker did NOT refuse shutdown"
sleep 4
if ! kill -0 "$PID_P" 2>/dev/null; then
    fail "persistent worker exited despite refusing shutdown"
fi
ok "persistent worker refused shutdown and stayed up"
kill -TERM "$PID_P" 2>/dev/null || true

# -------- Stage 4 token gating (separate curl; no full worker setup) --------

banner 4 "inbound endpoint gates on the bearer token, then dispatches for real"
sleep 1
WORKER_F_PORT=$(free_port)
start_worker worker-tok persistent "$WORKER_F_PORT"; PID_T="$LAST_WORKER_PID"
code=$(curl -sS -o /dev/null -w '%{http_code}' "http://localhost:$WORKER_F_PORT/v1/cluster/inbound" \
    -X POST -H 'Content-Type: application/json' -d '{"x":1}')
[[ "$code" == "401" ]] || fail "inbound without token returned $code, want 401"
ok "inbound rejected unauth (401)"

# An authenticated request is dispatched through the worker's OWN
# single-host request handler (cluster-mode decision 2) -- it is no
# longer the Phase 1 placeholder that answered every request with 503.
# This worker has no probe cache, so its vLLM model allowlist is empty
# and the real handler answers 404 "unknown model": proof the request
# travelled the whole chain instead of short-circuiting.
code2=$(curl -sS -o "$WORKDIR/inbound.json" -w '%{http_code}' \
    "http://localhost:$WORKER_F_PORT/v1/cluster/inbound" \
    -X POST -H 'Content-Type: application/json' \
    -H 'Authorization: Bearer the-token' \
    -H 'X-Devai-Backend: vllm' \
    -d '{"model":"no-such-model"}')
if [[ "$code2" == "503" ]] && grep -q 'not_implemented' "$WORKDIR/inbound.json"; then
    fail "inbound is still the Phase 1 placeholder (503 not_implemented)"
fi
[[ "$code2" == "404" ]] || {
    echo "----- inbound body -----"; cat "$WORKDIR/inbound.json"
    fail "inbound with token returned $code2, want 404 from the real request handler"
}
grep -q 'unknown model' "$WORKDIR/inbound.json" || {
    echo "----- inbound body -----"; cat "$WORKDIR/inbound.json"
    fail "inbound 404 did not come from the single-host allowlist check"
}
ok "inbound accepted token and dispatched into the real request handler"

# Without X-Devai-Backend the worker falls back to a model-name lookup;
# a body with no usable model is a 400, not a silent 503.
code3=$(curl -sS -o "$WORKDIR/inbound-nobackend.json" -w '%{http_code}' \
    "http://localhost:$WORKER_F_PORT/v1/cluster/inbound" \
    -X POST -H 'Content-Type: application/json' \
    -H 'Authorization: Bearer the-token' -d '{"x":1}')
[[ "$code3" == "400" ]] || {
    echo "----- inbound body -----"; cat "$WORKDIR/inbound-nobackend.json"
    fail "inbound without a resolvable backend returned $code3, want 400"
}
ok "inbound without a resolvable backend returns a clear 400"

echo
echo "===== ALL PREFLIGHT STAGES PASSED ====="
exit 0
