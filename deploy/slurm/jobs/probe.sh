#!/bin/bash
# Probe job (platform-rebuild C3, C8): one model (one model and band for
# Ollama). It leases the model the prober loads with this job id, `run`s the
# per-model probe action (probe-<backend>-model) in devai-operator and holds
# the run. Once the run's hold has returned, on either path, it unloads that
# model (the job owns it: no force; a no-op when the prober already unloaded
# it) and ends the lease. The epilog sends the unload again.
umask 0002
A=devai.operator.actions S=devai.engines.$DEVAI_BACKEND J=$SLURM_JOB_ID H='' L=''
finish() {
  devai-bus request $S.unload '{}' --job "$J" --timeout 25s
  [ -n "$L" ] && kill -HUP "$L" 2>/dev/null
}
stop() {   # stop replies at once (C1); the job ends only once the run has ended
  devai-bus request $A.stop "{\"signal\":\"$1\",\"grace_s\":$2}" --job "$J" --timeout 5s
  [ -n "$H" ] || { devai-bus hold $A --job "$J" --parent $$ & H=$!; }   # TERM came during `run`
  wait "$H"; rc=$?
  [ "$rc" = 1 ] && rc=0     # the run ended because this script stopped it; its own status is in result.json
  finish
  exit $rc
}
trap 'stop TERM 20' TERM          # scancel or the time limit
devai-bus hold $S --job "$J" --lease-only --parent $$ &   # the model the prober loads with this job id; ends on HUP
L=$!
devai-bus request $A.run "{\"kind\":\"probe\",\"action\":\"$DEVAI_ACTION\",\"params\":$DEVAI_PARAMS}" --job "$J" --timeout 30s || { rc=$?; finish; exit $rc; }
devai-bus hold $A --job "$J" --parent $$ &   # ignores TERM and INT; exits with the run: 0, 1 or 3
H=$!
wait $H; rc=$?
finish
exit $rc
