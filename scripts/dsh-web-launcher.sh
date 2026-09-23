#!/usr/bin/env bash
# DeepSeek Harness web launcher -- the devai picker's "dsh" agent.
#
# DeepSeek Harness has no terminal UI; its interactive surface is a browser
# UI. The picker has already written the router providers and the default
# model (model-picker.py, _write_dsh_config); this serves the UI on the
# container's port 3080, which `make lab-cpu|lab-gpu` publishes as DSH_PORT,
# and prints the URL to open. Ctrl-C stops it.
#
# Why it needs its own port and not JupyterLab's /proxy/: dsh hardcodes
# `<base href="/">` and loads /plugins/... and /api/remote.mux from the root,
# so it only works at the root of its own origin.
#
# Env contract:
#   DSH_PORT   host-side port published by lab-cpu/lab-gpu (required; the
#              shell containers publish none, and the picker hides this
#              agent there)
#   HOST_IP    host address the browser uses (set by lab-cpu/lab-gpu)
#   DEVAI_ROUTER_API_KEY  exported here; the router providers name it as
#              their apiKeyEnv (local backends ignore the value, dsh needs one)
#   DEVAI_DSH_OVERLAY     overlay path (default /etc/devai/dsh-overlay.yml)
#   DEVAI_DSH_SHELL_DEBUG=1  print the resolved command and exit (test hook)
set -euo pipefail

if [ -z "${DSH_PORT:-}" ]; then
    echo "error: DeepSeek Harness serves a browser UI on a published port, and only" >&2
    echo "       'make lab-cpu' / 'make lab-gpu' publish one (DSH_PORT)." >&2
    exit 1
fi

host="${HOST_IP:-localhost}"
overlay="${DEVAI_DSH_OVERLAY:-/etc/devai/dsh-overlay.yml}"
export DEVAI_ROUTER_API_KEY="${DEVAI_ROUTER_API_KEY:-local}"

# The /api trust fence accepts only these browser authorities.
cmd=(dsh --profile web --patch "$overlay" --no-open
     --trusted-host "${host}:${DSH_PORT}"
     --trusted-host "localhost:${DSH_PORT}"
     --trusted-host "127.0.0.1:${DSH_PORT}")

if [ "${DEVAI_DSH_SHELL_DEBUG:-}" = "1" ]; then
    printf '%q ' "${cmd[@]}"; echo
    exit 0
fi

echo "DeepSeek Harness web UI -- open the URL below in your browser; Ctrl-C stops it."
# dsh prints its in-container address (127.0.0.1:3080, plus the container's
# own LAN IP, which no browser can reach); rewrite it to the published one.
#
# A second line names the localhost form of the same URL. dsh serves its
# Settings pages (models, API keys) only to a page whose address is
# localhost / 127.x / [::1] -- decided in the browser from the URL
# (`$host.isLoopback`); from any other address they fail with "settings are
# unavailable in this browser". Chat works either way.
"${cmd[@]}" 2>&1 | sed -u \
    -e 's# (LAN: [^)]*)##' \
    -e "s#^dsh web: http://127\.0\.0\.1:3080/\(.*\)\$#dsh web: http://${host}:${DSH_PORT}/\1\nsettings (models, API keys) open only at a localhost address: http://localhost:${DSH_PORT}/\1\n  from another machine, tunnel first: ssh -L ${DSH_PORT}:localhost:${DSH_PORT} <this host>#" \
    -e "s#http://127\.0\.0\.1:3080/#http://${host}:${DSH_PORT}/#g"
