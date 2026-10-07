#!/bin/bash
# Epilog of devai-slurm (docs/slurm.md Sec. 6; platform-rebuild Phase 5, C3,
# C4). slurmd runs it as root after every job; Slurm returns the job's
# licenses and starts nothing on the node until it has ended, so its checks
# cannot race the next job.
#
# For engine, trainer, probe and GPU-action jobs (devai-gpu epilog): a probe
# or GPU-action job's action is stopped if it still runs; the sampler is
# stopped and results/<jobid>/gpu.json written; the job's service is ended as
# its script would (`unload`, unless the job ended with `sleep`; `cancel` for
# a trainer run); the card must be idle again (apart from a sleeping model),
# otherwise the epilog exits 1 and the node drains until the health check
# resumes it. Bench and other action jobs leave the card and the engines
# alone. Without a GPU only the unload or cancel is sent.
#
# Phase 5's epilog test switch: DEVAI_TEST_EPILOG=off in
# /run/devai/test/epilog.conf (a tmpfs; only tests/slurm/test-slurm.sh writes
# it, through podman exec, and removes it again) makes the epilog write
# gpu.json only. devai-gpu reads it.
set -u
J=${SLURM_JOB_ID:?epilog: SLURM_JOB_ID not set}
case "$J" in *[!0-9]*|'') echo "epilog: bad job id '$J'" >&2; exit 1 ;; esac

export PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
export DEVAI_BUS_URL=nats://127.0.0.1:4222
export DEVAI_BUS_USER=jobs
export DEVAI_BUS_PASSWORD_FILE=/run/devai/bus/jobs.pass

exec /usr/local/lib/devai/devai-gpu epilog --job "$J"
