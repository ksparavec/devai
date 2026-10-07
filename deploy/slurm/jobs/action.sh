#!/bin/bash
# Action job (platform-rebuild C3, C8): `run` a registered action in
# devai-operator and hold the run. DEVAI_ACTION and DEVAI_PARAMS (a JSON
# object) come from the submitter.
umask 0002
A=devai.operator.actions J=$SLURM_JOB_ID H=
stop() {   # stop replies at once (C1); the job ends only once the run has ended
  devai-bus request $A.stop "{\"signal\":\"$1\",\"grace_s\":$2}" --job "$J" --timeout 5s
  [ -n "$H" ] || { devai-bus hold $A --job "$J" --parent $$ & H=$!; }   # TERM came during `run`
  wait "$H"; rc=$?
  [ "$rc" = 1 ] && rc=0     # the run ended because this script stopped it; its own status is in result.json
  exit $rc
}
trap 'stop TERM 20' TERM          # scancel or the time limit; Slurm sends SIGKILL after KillWait (30 s)
devai-bus request $A.run "{\"kind\":\"action\",\"action\":\"$DEVAI_ACTION\",\"params\":$DEVAI_PARAMS}" --job "$J" --timeout 30s || exit $?
devai-bus hold $A --job "$J" --parent $$ &   # ignores TERM and INT; exits with the run: 0, 1 or 3
H=$!
wait $H
