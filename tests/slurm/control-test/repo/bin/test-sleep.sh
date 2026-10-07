#!/bin/bash
# test-sleep: sleeps SLEEP_S, exits EXIT_CODE; on TERM or INT it takes STOP_S
# seconds to stop (an action that needs time to stop on TERM).
on_stop() { echo "stopping ($1) in ${STOP_S:-0} s"; sleep "${STOP_S:-0}"; echo stopped; exit 143; }
trap 'on_stop TERM' TERM
trap 'on_stop INT' INT
echo "test-sleep job=$DEVAI_JOB_ID sleep=$SLEEP_S"
sleep "$SLEEP_S" & wait $!
echo "done"
exit "${EXIT_CODE:-0}"
