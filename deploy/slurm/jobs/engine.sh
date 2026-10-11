#!/bin/bash
# Engine job (platform-rebuild C3; docs/slurm.md Sec. 4). The router submits
# `exec /usr/local/lib/devai/jobs/engine.sh` with DEVAI_BACKEND and DEVAI_LOAD
# (the body of the `load` request, `gpu-arbiter -print-load`).
umask 0002
S=devai.engines.$DEVAI_BACKEND J=$SLURM_JOB_ID
trap 'devai-bus request $S.unload "{}" --job "$J" --timeout 25s; exit 0' TERM INT
trap 'devai-bus request $S.sleep "{}" --job "$J" --timeout 120s; exit 0' USR1   # refused unless serving (C1); the epilog then unloads
devai-bus request $S.load "$DEVAI_LOAD" --job "$J" --timeout 900s || exit $?   # timeout_s is at most 840 (C1, C2)
devai-bus hold $S --job "$J" --parent $$ &   # a lease every 10 s; ignores TERM and INT; exits 0 when the service stops serving this job, 3 on failed
wait $!
