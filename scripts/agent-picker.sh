#!/bin/bash
# Interactive model/agent picker for devai shell sessions.
# Delegates to model-picker for fzf-based model → backend → agent selection.
# Falls back to bash on error.
exec python3 /usr/local/bin/model-picker "$@" || exec bash
