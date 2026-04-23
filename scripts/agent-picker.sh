#!/bin/bash
# Interactive model/agent picker for devai shell sessions.
# Delegates to model-picker; its stdout/stderr and exit code are the user's
# truth. No fallback, no wrapping, no filtering.

exec python3 /usr/local/bin/model-picker "$@"
