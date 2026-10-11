#!/bin/bash
# test-bench: a bench task. On INT (Slurm's B:INT@120 warning, passed on by
# bench.sh as `stop {signal: INT}`) it "scores its unbroken prefix" for
# STOP_S seconds, writes result.json (status truncated) and exits; on TERM
# it stops after STOP_S seconds without one (devai-control writes the
# fallback, cancelled).
write_result() {
  printf '{"v":1,"job":%s,"kind":"bench","status":"%s","summary":"test-bench","outputs":{}}\n' \
    "$DEVAI_JOB_ID" "$1" > "$DEVAI_RESULTS_DIR/result.json"
}
trap 'echo "INT: scoring the prefix"; sleep "${STOP_S:-0}"; write_result truncated; exit 0' INT
trap 'echo "TERM: stopping"; sleep "${STOP_S:-0}"; exit 143' TERM
echo "test-bench job=$DEVAI_JOB_ID sleep=$SLEEP_S"
sleep "$SLEEP_S" & wait $!
write_result ok
