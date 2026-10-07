#!/bin/bash
# accounting-init -- a one-shot supervisord program of devai-slurm, run at
# every container start (docs/slurm.md Sec. 3.1; platform-rebuild Phase 5, I6).
#
# Waits up to 120 s for slurmdbd to answer, then idempotently creates the
# cluster devai, the account devai and the user devai in it, and gives devai
# Slurm's AdminLevel=Operator (hold, release, node resume). root gets no
# association. Exits non-zero on the timeout or on any other failure, so
# supervisord logs it.
set -u
export PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin

# supervisord restarts this program after an unexpected exit
# (autorestart=unexpected, startsecs=0): the pause keeps a failure that
# repeats from becoming a busy loop.
fail() { echo "accounting-init: $*" >&2; sleep 10; exit 1; }

deadline=$((SECONDS + 120))
until sacctmgr -n show cluster >/dev/null 2>&1; do
    if [ "$SECONDS" -ge "$deadline" ]; then
        fail "slurmdbd did not answer within 120 s"
    fi
    sleep 2
done

# Each step runs only when its result is missing, so a restart changes
# nothing (sacctmgr's messages for "already there" differ per entity and
# exit non-zero). sacctmgr -i: no prompt.
step() {
    local out
    if ! out=$(sacctmgr -i "$@" 2>&1); then
        echo "accounting-init: sacctmgr $*: $out" >&2
        return 1
    fi
    echo "accounting-init: sacctmgr $*: ok"
}
q() { sacctmgr -n -P show "$@" 2>/dev/null; }

new_cluster=0
if ! q cluster format=cluster | grep -qx devai; then
    step add cluster devai || fail "the cluster was not added"
    new_cluster=1
fi
if ! q account devai format=account | grep -qx devai; then
    step add account devai Cluster=devai Description=devai Organization=devai || fail "the account was not added"
fi
if ! q assoc where cluster=devai account=devai user=devai format=user | grep -qx devai; then
    step add user devai Account=devai Cluster=devai DefaultAccount=devai || fail "the user was not added"
fi
if [ "$(q user devai format=adminlevel)" != Operator ]; then
    step modify user devai set AdminLevel=Operator || fail "AdminLevel was not set"
fi
# `add cluster` gives root a user association in account root by itself
# (measured on 24.11.5); root gets none here (C3: root has no association).
if q assoc where user=root format=user | grep -qx root; then
    step remove user root || fail "root's association was not removed"
fi

# A cluster added after slurmctld started registers at its next
# reconfiguration.
if [ "$new_cluster" = 1 ]; then
    scontrol reconfigure || echo "accounting-init: scontrol reconfigure failed" >&2
fi
echo "accounting-init: done"
