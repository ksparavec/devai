#!/usr/bin/env bash
# Idempotent setup of the /run/devai tmpfs mount used by sops-rendered
# secrets (mcp-gateway Phase 2, gpu-arbiter cluster mode, SkyPilot fleet
# provisioner).
#
# Mount options chosen per docs/plans/sops-age-secrets.md decision 3:
#   - tmpfs (RAM-backed; gone on reboot, no plaintext on disk)
#   - nodev,nosuid,noexec (defence in depth)
#   - size=4M (enough for several .env files; tiny by tmpfs standards)
#   - mode=0700 (only the owning user can list contents)
#
# Idempotency: if /run/devai is already a tmpfs mount, exit 0 with a
# one-line confirmation. If it exists as a regular directory, mount
# tmpfs over it. If it does not exist, create it then mount.
#
# Boot persistence is NOT installed here; ship a separate
# deploy/systemd/run-devai.mount unit (or systemd-tmpfiles drop-in)
# when the operator wants the mount to come back across reboots --
# documented in docs/secrets.md.
#
# Usage (typically run as root via sudo):
#   sudo bash deploy/setup-secrets-tmpfs.sh

set -euo pipefail

MOUNTPOINT="${DEVAI_SECRETS_TMPFS:-/run/devai}"
SIZE="${DEVAI_SECRETS_TMPFS_SIZE:-4m}"

is_tmpfs() {
    # mountpoint -q exits 0 iff $1 is a mountpoint; couple with stat -f
    # to confirm the filesystem type. mountpoint alone returns 0 for
    # bind mounts and other surprises.
    if ! command -v mountpoint >/dev/null 2>&1; then
        # Fallback: use /proc/mounts directly.
        awk -v p="$1" '$2 == p && $3 == "tmpfs" {found=1} END {exit !found}' /proc/mounts
        return $?
    fi
    if mountpoint -q "$1" 2>/dev/null; then
        local t
        t=$(stat -f -c '%T' "$1" 2>/dev/null || echo "")
        [[ "${t}" == "tmpfs" ]]
        return $?
    fi
    return 1
}

if is_tmpfs "${MOUNTPOINT}"; then
    echo "${MOUNTPOINT}: already a tmpfs mount; nothing to do."
    exit 0
fi

if [[ ! -d "${MOUNTPOINT}" ]]; then
    mkdir -p "${MOUNTPOINT}"
fi

# Refuse to mount over a non-empty regular directory: that would
# shadow real files. The operator should clean up first.
if [[ -n "$(ls -A "${MOUNTPOINT}" 2>/dev/null)" ]]; then
    echo "ERROR: ${MOUNTPOINT} is non-empty and not a tmpfs mount. Refusing" >&2
    echo "       to shadow existing files. Move them aside, then re-run." >&2
    exit 1
fi

mount -t tmpfs -o "nodev,nosuid,noexec,size=${SIZE},mode=0700" tmpfs "${MOUNTPOINT}"
echo "${MOUNTPOINT}: mounted tmpfs (size=${SIZE}, mode=0700, nodev,nosuid,noexec)."
