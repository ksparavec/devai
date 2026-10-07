#!/bin/bash
# Prolog of devai-slurm (docs/slurm.md Sec. 6; platform-rebuild Phase 5, I7,
# I26). slurmd runs it as root for EVERY job at allocation
# (PrologFlags=Alloc), before the job script opens its log.
#
# 1. For every job: results/<jobid>/ and its artifacts/, owned by devai,
#    group 0 (in podman's default user namespace gid 0 is the host user's
#    group), mode 2775, so devai-operator (keep-id, the host user) writes
#    action.log and result.json there through the group bit (C4, I26).
# 2. For engine, trainer, probe and GPU-action jobs: the GPU guard and the
#    sampler (devai-gpu guard). A non-zero exit fails the prolog: Slurm
#    drains the node and requeues the job held (PrologFlags=ForceRequeueOnFail),
#    so nothing starts on a busy card.
set -u
J=${SLURM_JOB_ID:?prolog: SLURM_JOB_ID not set}
case "$J" in *[!0-9]*|'') echo "prolog: bad job id '$J'" >&2; exit 1 ;; esac
R=/var/cache/devai/jobs/results/$J

if ! install -d -o devai -g 0 -m 2775 "$R" "$R/artifacts"; then
    echo "prolog: cannot create $R (is the jobs volume full? devai-jobs prune)" >&2
    exit 1
fi

# slurmd passes the prolog none of the container's environment: the bus
# settings of C7, the values of C3's job environment.
export PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
export DEVAI_BUS_URL=nats://127.0.0.1:4222
export DEVAI_BUS_USER=jobs
export DEVAI_BUS_PASSWORD_FILE=/run/devai/bus/jobs.pass

exec /usr/local/lib/devai/devai-gpu guard --job "$J"
