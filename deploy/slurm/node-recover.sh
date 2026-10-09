#!/bin/bash
# node-recover -- a one-shot supervisord program of devai-slurm, run at every
# container start before slurmd (platform-rebuild Phase 5).
#
# Every Slurm process runs in this container, so a job slurmctld recovers
# from its saved state as running died with the previous container. Slurm
# ends such a job by itself only when the node re-registers more than
# BatchStartTimeout after the job was last seen, and slurmd re-registers on
# its own only every few tens of minutes: on 2026-10-09 two jobs stayed
# RUNNING, one holding the engine license, through 260 s, a container
# re-creation and a reconfigure.
#
# So: wait up to 120 s for slurmctld; when it lists any job on the node as
# running, set the node DOWN (Slurm ends each job NODE_FAIL, and JobRequeue=0
# keeps it from running again); then start slurmd, whose registration
# returns the node to service (ReturnToService=2). The DOWN comes before
# slurmd registers, so no job can be started on the node in between and
# then be ended with them. slurmd is started on every path. A drained node
# keeps its drain reason: the health check resumes only a drain whose reason
# names the guard (devai-gpu GUARD_REASONS).
#
# Only for a container start: run by hand while jobs really run, it would
# end them too.
set -u
export PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin

start_slurmd() {
    supervisorctl -c /etc/devai/supervisord.conf start slurmd \
        || echo "node-recover: supervisorctl start slurmd failed" >&2
}

deadline=$((SECONDS + 120))
until scontrol ping 2>/dev/null | grep -q 'is UP'; do
    if [ "$SECONDS" -ge "$deadline" ]; then
        echo "node-recover: slurmctld did not answer within 120 s; starting slurmd without the check" >&2
        start_slurmd
        exit 1
    fi
    sleep 1
done

if ! jobs=$(squeue -h -a -w devai -t RUNNING,COMPLETING,CONFIGURING,SUSPENDED,STOPPED -o %i 2>&1); then
    echo "node-recover: squeue failed ($jobs); starting slurmd without the check" >&2
    start_slurmd
    exit 1
fi
jobs=$(echo "$jobs" | paste -sd, -)
if [ -n "$jobs" ]; then
    reason="devai-slurm restarted; its jobs died with it"
    case "$(sinfo -h -n devai -o %T 2>/dev/null)" in
        *drain*)
            old=$(sinfo -h -n devai -o %E 2>/dev/null)
            [ -n "$old" ] && [ "$old" != none ] && reason=$old
            ;;
    esac
    if scontrol update NodeName=devai State=DOWN Reason="$reason"; then
        echo "node-recover: jobs $jobs died with the previous container: node set DOWN, they end NODE_FAIL"
    else
        echo "node-recover: could not set the node DOWN; jobs $jobs stay listed until slurmd re-registers late" >&2
    fi
fi
start_slurmd
