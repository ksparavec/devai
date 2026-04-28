#!/usr/bin/env bash
#
# Create a dedicated 100 GB XFS volume at /var/cache/devai/logs.
#
# Aligns with the existing devai storage layout: every other cache subdir
# (apt, npm, pip, registry, ollama, open-webui) is its own LVM logical
# volume in the `vgais` volume group, mounted via /etc/fstab. This script
# adds `cache_logs` alongside them.
#
# Idempotent: re-running is safe — existing LV / mount / fstab entry are
# detected and reused. Existing data at the mountpoint is wiped before
# the LV is mounted, but only after `make cache-down` stops anything
# writing there.
#
# Usage:
#   sudo deploy/setup-logs-volume.sh                  # 100G default
#   sudo SIZE=200G deploy/setup-logs-volume.sh        # custom size
#   sudo VG=other_vg deploy/setup-logs-volume.sh      # different VG
#
# Run AFTER `make cache-down` so no container holds the old logs path.

set -euo pipefail

VG="${VG:-vgais}"
LV="${LV:-cache_logs}"
SIZE="${SIZE:-100G}"
MOUNTPOINT="${MOUNTPOINT:-/var/cache/devai/logs}"
DEVICE="/dev/${VG}/${LV}"

if [[ "$EUID" -ne 0 ]]; then
    echo "error: must run as root (use sudo)" >&2
    exit 1
fi

# Ownership target: copy whatever uid/gid owns the parent directory
# (/var/cache/devai). That's the user who runs rootless podman in this
# project and writes to the other cache LVs.
PARENT="$(dirname "$MOUNTPOINT")"
if [[ ! -d "$PARENT" ]]; then
    echo "error: parent directory $PARENT does not exist" >&2
    exit 1
fi
OWNER_UID="$(stat -c %u "$PARENT")"
OWNER_GID="$(stat -c %g "$PARENT")"

echo "==> setup parameters"
echo "    volume group:   $VG"
echo "    logical volume: $LV"
echo "    size:           $SIZE"
echo "    mountpoint:     $MOUNTPOINT"
echo "    owner:          ${OWNER_UID}:${OWNER_GID}"

# ── Sanity: is anything still using $MOUNTPOINT? ────────────────────────────
if findmnt --target "$MOUNTPOINT" --types xfs >/dev/null 2>&1 \
   && [[ "$(findmnt -no SOURCE --target "$MOUNTPOINT")" == "$DEVICE" ]]; then
    echo "==> $MOUNTPOINT already mounted from $DEVICE — skipping create/format/mount"
    SKIP_CREATE=1
else
    SKIP_CREATE=0
    if [[ -d "$MOUNTPOINT" ]] \
       && fuser -mv "$MOUNTPOINT" 2>/dev/null | grep -q .; then
        echo "error: $MOUNTPOINT is in use by another process" >&2
        echo "       run 'make cache-down' first, then re-run this script." >&2
        exit 1
    fi
fi

# ── Volume group sanity ─────────────────────────────────────────────────────
if ! vgs --noheadings -o vg_name 2>/dev/null | grep -q "^[[:space:]]*${VG}\$"; then
    echo "error: volume group '$VG' not found" >&2
    echo "       available: $(vgs --noheadings -o vg_name | tr -d ' ' | tr '\n' ' ')" >&2
    exit 1
fi

# ── Create LV (idempotent) ──────────────────────────────────────────────────
if [[ "$SKIP_CREATE" -eq 0 ]]; then
    if lvs "${VG}/${LV}" >/dev/null 2>&1; then
        echo "==> ${VG}/${LV} already exists — reusing"
    else
        # Check free space.
        FREE_BYTES="$(vgs --noheadings -o vg_free --units b "$VG" \
                      | tr -d ' B')"
        REQ_BYTES="$(numfmt --from=iec "$SIZE")"
        if [[ "$FREE_BYTES" -lt "$REQ_BYTES" ]]; then
            FREE_HUMAN="$(numfmt --to=iec --suffix=B "$FREE_BYTES")"
            echo "error: VG '$VG' has only $FREE_HUMAN free, need $SIZE" >&2
            exit 1
        fi
        echo "==> creating ${VG}/${LV} (${SIZE})"
        lvcreate -L "$SIZE" -n "$LV" "$VG"
    fi

    # Format if no existing filesystem.
    if blkid -p "$DEVICE" >/dev/null 2>&1; then
        existing_fs="$(blkid -s TYPE -o value "$DEVICE" || true)"
        echo "==> ${DEVICE} already has filesystem '${existing_fs}' — keeping"
    else
        echo "==> formatting ${DEVICE} as xfs"
        mkfs.xfs -q "$DEVICE"
    fi
fi

# ── /etc/fstab entry (idempotent) ───────────────────────────────────────────
UUID="$(blkid -s UUID -o value "$DEVICE")"
if [[ -z "$UUID" ]]; then
    echo "error: cannot read UUID of ${DEVICE}" >&2
    exit 1
fi
echo "==> UUID: $UUID"

# Drop any stale entries for this mountpoint or device, then append a
# canonical entry. Keep a backup of /etc/fstab first.
cp -f /etc/fstab "/etc/fstab.bak.$(date -u +%Y%m%dT%H%M%SZ)"
sed -i.tmp -e "\#\\s${MOUNTPOINT}\\s#d" \
           -e "\#${DEVICE}\\s#d" \
           -e "\#UUID=${UUID}\\s#d" \
           /etc/fstab
rm -f /etc/fstab.tmp
echo "UUID=${UUID}  ${MOUNTPOINT}  xfs  defaults,noatime  0  2" >> /etc/fstab
echo "==> /etc/fstab updated"

# ── Mount it now ────────────────────────────────────────────────────────────
mkdir -p "$MOUNTPOINT"

if findmnt --target "$MOUNTPOINT" --types xfs >/dev/null 2>&1; then
    echo "==> $MOUNTPOINT already mounted"
else
    # Wipe whatever was previously sitting at $MOUNTPOINT before the new
    # filesystem hides it. We're explicitly asked to remove old logs.
    if [[ -n "$(ls -A "$MOUNTPOINT" 2>/dev/null)" ]]; then
        echo "==> wiping old contents of $MOUNTPOINT"
        find "$MOUNTPOINT" -mindepth 1 -delete
    fi
    mount "$MOUNTPOINT"
    echo "==> mounted ${DEVICE} at ${MOUNTPOINT}"
fi

# ── Ownership ───────────────────────────────────────────────────────────────
chown "${OWNER_UID}:${OWNER_GID}" "$MOUNTPOINT"
chmod 0755 "$MOUNTPOINT"

# ── Summary ─────────────────────────────────────────────────────────────────
echo
echo "==> done."
findmnt "$MOUNTPOINT"
df -h "$MOUNTPOINT"
echo
echo "Next: run 'make cache-up' (logger will write to the new volume)."
