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
SIZE="${SIZE:-100G}"           # virtual size for the thin LV
POOL="${POOL:-cachepool}"      # thin pool name in $VG
MOUNTPOINT="${MOUNTPOINT:-/var/cache/devai/logs}"
DEVICE="/dev/${VG}/${LV}"
RECREATE="${RECREATE:-0}"      # 1 → unmount + lvremove existing, then recreate fresh

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
echo "    recreate:       $RECREATE"

# ── Optional clean slate ────────────────────────────────────────────────────
# RECREATE=1 wipes any pre-existing logs LV (e.g. a thick LV from an
# earlier setup attempt) before creating the thin one. Safe to run on a
# fresh host — every step is guarded by an existence check.
if [[ "$RECREATE" == "1" ]]; then
    echo "==> RECREATE=1: removing any existing $MOUNTPOINT mount, fstab"
    echo "    entry, and ${VG}/${LV} logical volume."
    # Use exact-path mount check, not findmnt --target (which walks up
    # to the root filesystem and always matches on a non-mountpoint).
    if mountpoint -q "$MOUNTPOINT" 2>/dev/null; then
        umount "$MOUNTPOINT" || true
    fi
    # Strip any old fstab entries for this mountpoint or device.
    if grep -qE "(\\s${MOUNTPOINT}\\s|^${DEVICE}\\s|UUID=[^[:space:]]+\\s+${MOUNTPOINT}\\s)" /etc/fstab 2>/dev/null; then
        cp -f /etc/fstab "/etc/fstab.bak.$(date -u +%Y%m%dT%H%M%SZ)"
        sed -i.tmp -e "\#\\s${MOUNTPOINT}\\s#d" \
                   -e "\#${DEVICE}\\s#d" \
                   /etc/fstab
        rm -f /etc/fstab.tmp
    fi
    if lvs "${VG}/${LV}" >/dev/null 2>&1; then
        lvremove -f "${VG}/${LV}"
    fi
    if [[ -d "$MOUNTPOINT" ]]; then
        rm -rf "$MOUNTPOINT"
    fi
fi

# ── Sanity: is anything still using $MOUNTPOINT? ────────────────────────────
# Use `mountpoint -q` for an EXACT path match. `findmnt --target` walks
# up the path looking for any enclosing mount (e.g. "/"), which gave
# false positives when /var/cache/devai/logs wasn't a mountpoint yet.
if mountpoint -q "$MOUNTPOINT" \
   && [[ "$(findmnt -no SOURCE --mountpoint "$MOUNTPOINT")" == "$DEVICE" ]]; then
    echo "==> $MOUNTPOINT already mounted from $DEVICE — skipping create/format/mount"
    SKIP_CREATE=1
else
    SKIP_CREATE=0
    # Defend against running while a process holds files OPEN inside
    # $MOUNTPOINT — those would silently keep writing to the pre-mount
    # inode after we mount over them. We probe with `lsof +D` (recurses
    # into the directory). `fuser -mv` was wrong here: -m treats the
    # argument as a mount point and flags every process using the
    # ENCLOSING filesystem (root /), which always trips.
    if [[ -d "$MOUNTPOINT" ]] && command -v lsof >/dev/null 2>&1; then
        in_use="$(lsof +D "$MOUNTPOINT" 2>/dev/null | tail -n +2 || true)"
        if [[ -n "$in_use" ]]; then
            echo "error: $MOUNTPOINT has files open by:" >&2
            echo "$in_use" | head -5 >&2
            echo "       run 'make cache-down' first, then re-run this script." >&2
            exit 1
        fi
    fi
fi

# ── Volume group sanity ─────────────────────────────────────────────────────
if ! vgs --noheadings -o vg_name 2>/dev/null | grep -q "^[[:space:]]*${VG}\$"; then
    echo "error: volume group '$VG' not found" >&2
    echo "       available: $(vgs --noheadings -o vg_name | tr -d ' ' | tr '\n' ' ')" >&2
    exit 1
fi

# ── Locate or validate the thin pool ────────────────────────────────────────
# The new LV must be a thin volume inside a thin pool. Detect the pool by
# segtype=thin-pool; require POOL= if the VG has more than one.
POOLS_RAW="$(lvs --noheadings -o lv_name,segtype "$VG" 2>/dev/null \
             | awk '$2 == "thin-pool" { print $1 }')"
mapfile -t POOLS < <(echo "$POOLS_RAW")
# Strip empty entries that mapfile leaves behind on empty input.
POOLS_NONEMPTY=()
for p in "${POOLS[@]}"; do
    [[ -n "$p" ]] && POOLS_NONEMPTY+=("$p")
done

if [[ -n "$POOL" ]]; then
    found=0
    for p in "${POOLS_NONEMPTY[@]}"; do
        [[ "$p" == "$POOL" ]] && found=1 && break
    done
    if [[ "$found" -ne 1 ]]; then
        echo "error: thin pool '$POOL' not found in VG '$VG'" >&2
        echo "       available pools: ${POOLS_NONEMPTY[*]:-<none>}" >&2
        exit 1
    fi
elif [[ "${#POOLS_NONEMPTY[@]}" -eq 0 ]]; then
    echo "error: VG '$VG' has no thin pool" >&2
    echo "       create one first, e.g.:" >&2
    echo "         sudo lvcreate -L <size> --thinpool <name> $VG" >&2
    exit 1
elif [[ "${#POOLS_NONEMPTY[@]}" -gt 1 ]]; then
    echo "error: VG '$VG' has multiple thin pools: ${POOLS_NONEMPTY[*]}" >&2
    echo "       set POOL=<name> to choose one" >&2
    exit 1
else
    POOL="${POOLS_NONEMPTY[0]}"
fi
echo "==> thin pool: ${VG}/${POOL}"

# ── Create thin LV (idempotent) ─────────────────────────────────────────────
if [[ "$SKIP_CREATE" -eq 0 ]]; then
    if lvs "${VG}/${LV}" >/dev/null 2>&1; then
        # Verify it's a thin LV in our chosen pool.
        existing_pool="$(lvs --noheadings -o pool_lv "${VG}/${LV}" \
                         | tr -d ' ')"
        if [[ "$existing_pool" != "$POOL" ]]; then
            echo "error: ${VG}/${LV} exists but is not in pool '$POOL'" >&2
            echo "       segtype/pool: $(lvs --noheadings -o segtype,pool_lv "${VG}/${LV}")" >&2
            echo "       remove it first or set POOL=$existing_pool to reuse." >&2
            exit 1
        fi
        echo "==> ${VG}/${LV} already exists in thin pool '$POOL' — reusing"
    else
        echo "==> creating thin LV ${VG}/${LV} (virtual size ${SIZE}) in pool '$POOL'"
        lvcreate --thin --virtualsize "$SIZE" --name "$LV" "${VG}/${POOL}"
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

# Exact-path mount check — see above note on findmnt --target gotcha.
if mountpoint -q "$MOUNTPOINT"; then
    src="$(findmnt -no SOURCE --mountpoint "$MOUNTPOINT")"
    if [[ "$src" == "$DEVICE" ]]; then
        echo "==> $MOUNTPOINT already mounted from $DEVICE"
    else
        echo "error: $MOUNTPOINT is mounted from $src, not $DEVICE" >&2
        echo "       unmount it first or set RECREATE=1." >&2
        exit 1
    fi
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
