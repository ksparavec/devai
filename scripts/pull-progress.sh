#!/usr/bin/env bash
# Live pull-progress tracker for podman image downloads.
# Appends one compact status line per interval so scrollback preserves history.
# Exits when no pull processes remain (or with Ctrl-C).
#
# Usage: scripts/pull-progress.sh [interval_seconds]
#        Default interval is 5 seconds.

set -u

INTERVAL="${1:-5}"
CACHE_DIR="${CACHE_DIR:-/var/cache/devai/registry}"
STAGING_DIR="${CACHE_DIR}/tmp"

default_iface="$(ip -4 route show default 2>/dev/null | awk '/default/ {print $5; exit}')"

human() {
    local b=${1:-0}
    if   [ "$b" -ge 1073741824 ]; then awk "BEGIN{printf \"%.1fG\", $b/1073741824}"
    elif [ "$b" -ge 1048576 ];    then awk "BEGIN{printf \"%.0fM\", $b/1048576}"
    elif [ "$b" -ge 1024 ];       then awk "BEGIN{printf \"%.0fK\", $b/1024}"
    else echo "${b}B"
    fi
}

net_rx_bytes() {
    [ -n "$default_iface" ] || { echo 0; return; }
    local f="/sys/class/net/$default_iface/statistics/rx_bytes"
    [ -r "$f" ] && cat "$f" || echo 0
}

podman_cpu_pct() {
    ps -u "$(id -un)" -o pcpu=,comm= 2>/dev/null \
        | awk '$2=="podman"{s+=$1} END{printf "%.0f", s+0}'
}

staging_state() {
    # One line per staging dir: "<label>=<bytes>:<blobs>"
    find "$STAGING_DIR" -maxdepth 1 -mindepth 1 -name 'container_images_storage*' -type d 2>/dev/null \
        | while read -r d; do
            local b c label
            b=$(du -sb "$d" 2>/dev/null | awk '{print $1}')
            c=$(find "$d" -maxdepth 1 -type f 2>/dev/null | wc -l)
            label=$(basename "$d" | sed 's/container_images_storage//' | cut -c1-5)
            printf '%s=%s:%s\n' "$label" "${b:-0}" "${c:-0}"
        done
}

pull_procs_count() {
    pgrep -cf "podman.*compose.*up|docker-compose.*up|podman.*pull" 2>/dev/null || echo 0
}

image_count() {
    podman images --format '{{.Repository}}' 2>/dev/null \
        | grep -Evc '^localhost/|^<none>' || echo 0
}

declare -A prev_staging
prev_used_k=0
prev_rx=0
prev_ts=0

printf '%-8s %-9s %-9s %-9s %-5s %-5s %-6s  %s\n' \
    TIME LV LV/s RX/s CPU pulls imgs 'STAGING(bytes Δ/s · blobs)'

while true; do
    ts=$(date +%s)
    used_k=$(df --output=used "$CACHE_DIR" 2>/dev/null | tail -1 | tr -d ' ')
    avail_g=$(df --output=avail -BG "$CACHE_DIR" 2>/dev/null | tail -1 | tr -d ' G')
    rx=$(net_rx_bytes)
    cpu=$(podman_cpu_pct)
    pulls=$(pull_procs_count)
    imgs=$(image_count)

    if [ "$prev_ts" -gt 0 ]; then
        dt=$((ts - prev_ts))
        [ "$dt" -lt 1 ] && dt=1
        lv_rate=$(awk "BEGIN{printf \"%.0fM\", ($used_k - $prev_used_k)/1024/$dt}")
        net_rate=$(awk "BEGIN{printf \"%.0fM\", ($rx - $prev_rx)/1048576/$dt}")
    else
        dt=1; lv_rate="—"; net_rate="—"
    fi

    used_gb=$(awk "BEGIN{printf \"%.0fG\", $used_k/1048576}")
    staging_out=""
    declare -A cur_staging
    while IFS= read -r entry; do
        [ -z "$entry" ] && continue
        label="${entry%%=*}"
        rest="${entry#*=}"
        bytes="${rest%%:*}"
        blobs="${rest##*:}"
        cur_staging["$label"]=$bytes
        if [ -n "${prev_staging[$label]:-}" ] && [ "$dt" -gt 0 ]; then
            delta=$(( (bytes - ${prev_staging[$label]}) / dt ))
            d_str=$(human "${delta#-}")
            [ "$delta" -lt 0 ] && d_str="-$d_str"
        else
            d_str="—"
        fi
        staging_out+="#$label $(human "$bytes")(${d_str}/s·$blobs) "
    done < <(staging_state)

    printf '%-8s %-9s %-9s %-9s %-5s %-5s %-6s  %s\n' \
        "$(date +%H:%M:%S)" "$used_gb" "$lv_rate" "$net_rate" "${cpu}%" "$pulls" "$imgs" "${staging_out:-(no in-flight staging dirs)}"

    prev_staging=()
    for k in "${!cur_staging[@]}"; do prev_staging[$k]=${cur_staging[$k]}; done
    prev_used_k=$used_k
    prev_rx=$rx
    prev_ts=$ts

    if [ "$pulls" -eq 0 ]; then
        echo "DONE — no pull processes remain."
        exit 0
    fi

    sleep "$INTERVAL"
done
