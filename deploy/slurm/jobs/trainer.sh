#!/bin/bash
# Trainer job (platform-rebuild C3; docs/slurm.md Sec. 4). The router submits
# it with DEVAI_FT_JOB once the laya controller has accepted a fine-tuning job.
umask 0002
T=devai.laya-trainer.trainer J=$SLURM_JOB_ID
trap 'devai-bus request $T.cancel "{\"reason\":\"scancel\"}" --job "$J" --timeout 25s; exit 0' TERM INT
trap 'devai-bus request $T.cancel "{\"reason\":\"time_limit\"}" --job "$J" --timeout 25s; exit 0' USR2   # B:USR2@30
devai-bus request $T.train "{\"ft_job\":\"$DEVAI_FT_JOB\"}" --job "$J" --timeout 30s || exit $?
devai-bus hold $T --job "$J" --parent $$ &   # a lease every 10 s; ignores TERM and INT; exits 0, 1 or 3 with the run
wait $!
