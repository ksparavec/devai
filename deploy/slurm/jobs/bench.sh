#!/bin/bash
# Bench job (platform-rebuild C3; docs/slurm.md Sec. 8): `run` the bench-task
# action in devai-operator and hold the run; -L bench keeps bench tasks one at
# a time (I5). At 30 minutes Slurm sends SIGINT to this script alone
# (B:INT@120): the task scores its unbroken prefix and exits; only then does
# the job end and release the bench license.
umask 0002
A=devai.operator.actions J=$SLURM_JOB_ID H=
stop() {   # stop replies at once (C1); the job ends only once the run has ended
  devai-bus request $A.stop "{\"signal\":\"$1\",\"grace_s\":$2}" --job "$J" --timeout 5s
  [ -n "$H" ] || { devai-bus hold $A --job "$J" --parent $$ & H=$!; }   # the signal came during `run`
  wait "$H"; rc=$?
  [ "$rc" = 1 ] && rc=0     # the run ended because this script stopped it; its own status is in result.json
  exit $rc
}
trap 'stop TERM 20' TERM          # scancel or the time limit; Slurm sends SIGKILL after KillWait (30 s)
trap 'stop INT 110' INT           # B:INT@120, the time-limit warning
devai-bus request $A.run "{\"kind\":\"bench\",\"action\":\"$DEVAI_ACTION\",\"params\":$DEVAI_PARAMS}" --job "$J" --timeout 30s || exit $?
devai-bus hold $A --job "$J" --parent $$ &   # ignores TERM and INT; exits with the run: 0, 1 or 3
H=$!
wait $H
