#!/usr/bin/env bash
#
# Create a dedicated XFS volume at /var/lib/libvirt/images, backed by a
# thin LV in vgais/cachepool, and wire libvirt's `default` storage pool
# to it. This is the host-side storage layer for VFIO test VMs (see
# docs/HOST_VFIO_SETUP.md, phase 7) — it is NOT part of the devai
# production stack and is independent of `setup-logs-volume.sh`.
#
# The script is structurally a sibling of setup-logs-volume.sh: same
# thin-pool model (vgais/cachepool), same /etc/fstab handling, same
# idempotency rules. The differences are mountpoint, owner (root:root
# 0711 — what libvirt-qemu expects), and the libvirt pool dance around
# the mount (destroy → mount → start) so the pool's target dir is not
# held open while we mount over it.
#
# Idempotent: re-running is safe. Existing LV, fstab entry, and pool
# definition are detected and reused. Existing data at the mountpoint
# is preserved unless RECREATE=1.
#
# Usage:
#   sudo deploy/setup-libvirt-images-volume.sh                  # 250G default
#   sudo SIZE=500G deploy/setup-libvirt-images-volume.sh        # custom size
#   sudo VG=other_vg deploy/setup-libvirt-images-volume.sh      # different VG
#   sudo POOL_NAME=test deploy/setup-libvirt-images-volume.sh   # use 'test' instead of libvirt's 'default' pool
#   sudo RECREATE=1 deploy/setup-libvirt-images-volume.sh       # wipe + redo (DESTROYS VM IMAGES)
#
# Pre-conditions:
#   - libvirt-daemon-system + libvirt-clients installed (HOST_VFIO_SETUP.md §6)
#   - vgais/cachepool exists with enough free thin extents
#   - No VMs currently running (the pool gets restarted)

set -euo pipefail

VG="${VG:-vgais}"
LV="${LV:-libvirt_images}"
SIZE="${SIZE:-250G}"            # virtual size for the thin LV
POOL="${POOL:-cachepool}"       # LVM thin pool name in $VG
LIBVIRT_POOL="${POOL_NAME:-default}"   # libvirt storage pool name
MOUNTPOINT="${MOUNTPOINT:-/var/lib/libvirt/images}"
DEVICE="/dev/${VG}/${LV}"
RECREATE="${RECREATE:-0}"

if [[ "$EUID" -ne 0 ]]; then
    echo "error: must run as root (use sudo)" >&2
    exit 1
fi

# ── Tooling sanity ──────────────────────────────────────────────────────────
for tool in lvcreate vgs lvs blkid mkfs.xfs mountpoint findmnt virsh; do
    command -v "$tool" >/dev/null 2>&1 \
        || { echo "error: $tool not found (install lvm2 / xfsprogs / libvirt-clients)" >&2; exit 1; }
done

if ! systemctl is-active --quiet libvirtd 2>/dev/null \
   && ! systemctl is-active --quiet virtqemud 2>/dev/null; then
    echo "error: neither libvirtd nor virtqemud is running" >&2
    echo "       start it first: sudo systemctl start libvirtd" >&2
    exit 1
fi

echo "==> setup parameters"
echo "    LVM volume group:    $VG"
echo "    LVM thin pool:       $POOL"
echo "    LVM logical volume:  $LV (virtual size $SIZE)"
echo "    mountpoint:          $MOUNTPOINT"
echo "    libvirt pool name:   $LIBVIRT_POOL"
echo "    owner:               root:root mode 0711 (libvirt-qemu convention)"
echo "    recreate:            $RECREATE"

# ── Optional clean slate ────────────────────────────────────────────────────
# RECREATE=1 destroys an existing libvirt_images LV (and any VM disk
# images on it). Only run when you're sure there are no VMs you care
# about defined against this pool.
if [[ "$RECREATE" == "1" ]]; then
    echo "==> RECREATE=1: stopping pool, unmounting, removing LV"
    virsh pool-destroy "$LIBVIRT_POOL" 2>/dev/null || true
    if mountpoint -q "$MOUNTPOINT" 2>/dev/null; then
        umount "$MOUNTPOINT" || umount -l "$MOUNTPOINT" || true
    fi
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
fi

# ── Volume group sanity ─────────────────────────────────────────────────────
if ! vgs --noheadings -o vg_name 2>/dev/null | grep -q "^[[:space:]]*${VG}\$"; then
    echo "error: volume group '$VG' not found" >&2
    echo "       available: $(vgs --noheadings -o vg_name | tr -d ' ' | tr '\n' ' ')" >&2
    exit 1
fi

# ── Thin pool sanity ────────────────────────────────────────────────────────
if ! lvs --noheadings -o lv_name,segtype "$VG" 2>/dev/null \
        | awk -v p="$POOL" '$1 == p && $2 == "thin-pool" { f=1 } END { exit !f }'; then
    echo "error: thin pool '${VG}/${POOL}' not found" >&2
    echo "       available pools: $(lvs --noheadings -o lv_name,segtype "$VG" | awk '$2=="thin-pool"{print $1}' | tr '\n' ' ')" >&2
    exit 1
fi
echo "==> thin pool: ${VG}/${POOL}"

# ── Stop libvirt pool so its target dir is not held open ────────────────────
if virsh pool-info "$LIBVIRT_POOL" >/dev/null 2>&1; then
    if [[ "$(virsh pool-info "$LIBVIRT_POOL" | awk '/^State:/{print $2}')" == "running" ]]; then
        echo "==> stopping libvirt pool '$LIBVIRT_POOL' before mount-over"
        virsh pool-destroy "$LIBVIRT_POOL"
    fi
fi

# ── Sanity: target dir empty (only when not already our mount) ──────────────
# If the mountpoint is already our LV, the contents are VM images we want
# to preserve. Only refuse to proceed when there is foreign content
# sitting in a non-mounted directory at the target — those files would
# be hidden by mounting over them.
if mountpoint -q "$MOUNTPOINT" 2>/dev/null \
   && [[ "$(findmnt -no SOURCE --mountpoint "$MOUNTPOINT" 2>/dev/null)" == "$DEVICE" ]]; then
    SKIP_MOUNT=1
    echo "==> $MOUNTPOINT already mounted from $DEVICE — skipping create/format/mount"
else
    SKIP_MOUNT=0
    if [[ -d "$MOUNTPOINT" ]] && [[ -n "$(ls -A "$MOUNTPOINT" 2>/dev/null)" ]]; then
        echo "error: $MOUNTPOINT is non-empty and not mounted from $DEVICE" >&2
        echo "       contents:" >&2
        ls -la "$MOUNTPOINT" | head -20 >&2
        echo "       move or remove the contents, or set RECREATE=1." >&2
        exit 1
    fi
fi

# ── Create thin LV (idempotent) ─────────────────────────────────────────────
if [[ "$SKIP_MOUNT" -eq 0 ]]; then
    if lvs "${VG}/${LV}" >/dev/null 2>&1; then
        existing_pool="$(lvs --noheadings -o pool_lv "${VG}/${LV}" | tr -d ' ')"
        if [[ "$existing_pool" != "$POOL" ]]; then
            echo "error: ${VG}/${LV} exists but is not in pool '$POOL'" >&2
            echo "       segtype/pool: $(lvs --noheadings -o segtype,pool_lv "${VG}/${LV}")" >&2
            exit 1
        fi
        echo "==> ${VG}/${LV} already exists in pool '$POOL' — reusing"
    else
        echo "==> creating thin LV ${VG}/${LV} (virtual size ${SIZE}) in pool '$POOL'"
        lvcreate --thin --virtualsize "$SIZE" --name "$LV" "${VG}/${POOL}"
    fi

    # Format if no existing filesystem (don't pass -f; refuse to overwrite).
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

cp -f /etc/fstab "/etc/fstab.bak.$(date -u +%Y%m%dT%H%M%SZ)"
sed -i.tmp -e "\#\\s${MOUNTPOINT}\\s#d" \
           -e "\#${DEVICE}\\s#d" \
           -e "\#UUID=${UUID}\\s#d" \
           /etc/fstab
rm -f /etc/fstab.tmp
echo "UUID=${UUID}  ${MOUNTPOINT}  xfs  defaults,noatime  0  2" >> /etc/fstab
echo "==> /etc/fstab updated"

# ── Mount ───────────────────────────────────────────────────────────────────
mkdir -p "$MOUNTPOINT"
if [[ "$SKIP_MOUNT" -eq 0 ]]; then
    mount "$MOUNTPOINT"
    echo "==> mounted ${DEVICE} at ${MOUNTPOINT}"
fi

# ── Ownership / perms ───────────────────────────────────────────────────────
# Two modes:
#  - root:libvirt 2775 (default when the libvirt group exists). Members
#    of the libvirt group can wget/curl/cp images directly into the
#    pool; the setgid bit makes new files inherit the libvirt group.
#    libvirt-qemu (the user qemu drops to when running VMs) can still
#    read images via the world-execute bit.
#  - root:root 0711 (when the libvirt group is absent, or RESTRICTED_PERMS=1).
#    Strict isolation: qemu can traverse to a known image path but
#    cannot list the directory or write to it. Operator must use
#    `virsh vol-upload` to bring images in.
RESTRICTED_PERMS="${RESTRICTED_PERMS:-0}"
if [[ "$RESTRICTED_PERMS" != "1" ]] && getent group libvirt >/dev/null 2>&1; then
    chown root:libvirt "$MOUNTPOINT"
    chmod 2775         "$MOUNTPOINT"
    echo "==> $MOUNTPOINT  owner: root:libvirt  mode: 2775 (group-writable)"
else
    chown root:root "$MOUNTPOINT"
    chmod 0711      "$MOUNTPOINT"
    echo "==> $MOUNTPOINT  owner: root:root  mode: 0711 (libvirt-mediated writes only)"
fi

# ── Define libvirt pool if missing, then start + autostart ──────────────────
if ! virsh pool-info "$LIBVIRT_POOL" >/dev/null 2>&1; then
    echo "==> defining libvirt pool '$LIBVIRT_POOL' targeting $MOUNTPOINT"
    virsh pool-define-as "$LIBVIRT_POOL" dir --target "$MOUNTPOINT"
    virsh pool-build "$LIBVIRT_POOL" 2>/dev/null || true
fi

if [[ "$(virsh pool-info "$LIBVIRT_POOL" 2>/dev/null | awk '/^State:/{print $2}')" != "running" ]]; then
    echo "==> starting libvirt pool '$LIBVIRT_POOL'"
    virsh pool-start "$LIBVIRT_POOL"
fi

if [[ "$(virsh pool-info "$LIBVIRT_POOL" 2>/dev/null | awk '/^Autostart:/{print $2}')" != "yes" ]]; then
    echo "==> setting libvirt pool '$LIBVIRT_POOL' autostart"
    virsh pool-autostart "$LIBVIRT_POOL"
fi

# ── Summary ─────────────────────────────────────────────────────────────────
echo
echo "==> done."
findmnt "$MOUNTPOINT"
df -h "$MOUNTPOINT"
echo
virsh pool-list --all
echo
virsh pool-info "$LIBVIRT_POOL"
echo
echo "Next: run docs/HOST_VFIO_SETUP.md §8 (smoke test the GPU handoff)."
