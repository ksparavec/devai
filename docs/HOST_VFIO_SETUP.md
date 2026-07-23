# HOST_VFIO_SETUP.md -- Host configuration for temporary GPU handoff to test VMs

This document is the **one-time host-side procedure** to prepare a
Linux box so QEMU/KVM virtual machines can **temporarily borrow** the
NVIDIA GPU via PCI passthrough (VFIO) for the duration of an
INSTALL.md validation run, and then **automatically give it back** to
the host's nvidia driver when the VM shuts down. Outside of test
runs, the host continues to use the GPU normally -- Dev AI Lab on the
bare metal, CUDA workloads, anything else -- exactly as before.

> **Why a separate doc?** This procedure mutates host-side firmware
> settings, kernel command-line, initramfs, and module configuration.
> Several steps require reboots. None of it is reversible from inside
> a container or VM. It should be done once per physical host and
> never re-run as part of an INSTALL.md run.

The procedure is written for an autonomous coding agent (Claude
Code, OpenAI Codex, or equivalent tier-1 agent) but the verification
commands and recovery rules are equally usable by a human operator.
Each step states its **purpose**, **precondition**, **idempotency
rule**, and **verification**.

---

## 0. Contract

### 0.1 Goal

After successful completion:

- IOMMU is enabled in firmware and in the running kernel.
- The `vfio`, `vfio_iommu_type1`, and `vfio_pci` kernel modules are
  available (loadable on demand) but **do not auto-claim** the
  dedicated NVIDIA GPU at boot. The nvidia driver continues to bind
  it normally.
- `libvirt` + `qemu-system-x86_64` + `virt-install` are installed.
- libvirt's `<hostdev managed='yes'>` mechanism is operational.
  Defining a VM with the GPU as a managed host device is sufficient:
  libvirt detaches the GPU from its current driver (`nvidia`) at VM
  start, binds it to `vfio-pci`, runs the VM, and on VM shutdown
  unbinds `vfio-pci` and re-binds `nvidia`. No manual bind/unbind
  scripts to maintain.
- A single `virsh start <vm-name>` is enough to give a VM the GPU,
  and `virsh destroy <vm-name>` (or guest-initiated shutdown) returns
  it to the host.

### 0.2 What is *not* changed

- The host's nvidia driver continues to load at boot.
- `nvidia-smi` continues to work on the bare host.
- Dev AI Lab installed natively on the host (the production stack)
  continues to work between test runs.
- The dedicated NVIDIA GPU does not lose its host-side identity --
  it is still `Kernel driver in use: nvidia` after every reboot.
  Only during an active test VM run does it briefly switch.

### 0.3 Constraints

1. **Stop GPU consumers before starting the test VM.** Anything on
   the host actively using the GPU (Dev AI Lab containers,
   `nvidia-smi` background loops, CUDA processes, X server if it
   uses the dGPU) blocks libvirt's driver detach. Production stack
   must be `make cache-down` before `virsh start`. The test harness
   automates this; the operator should be aware.
2. **Reboots required.** At least two reboots: one after enabling
   IOMMU (section 3), one after configuring vfio-pci as a loadable module
   (section 5). The agent must surface "reboot required" and wait for the
   operator rather than silently rebooting.
3. **Idempotent re-runs.** Each phase begins with a detection step
   that decides whether work is needed.
4. **Reversible.** section 10 documents how to undo every change in this
   document. Reverting takes one config-file removal plus a reboot.

### 0.4 Conventions

- `$` runs as the invoking user.
- `#` runs as `root` (use `sudo`).
- `[verify]` is a read-only check.
- `${INVOKING_USER}` = `$(id -un)` of the operator.

### 0.5 Phase order

| # | Phase | Mutates | Reboot? | Verification |
|---|---|---|---|---|
| 1 | Detect environment | no | no | `[verify]` only |
| 2 | Confirm UEFI VT-d / VT-x | UEFI firmware | no (manual) | `dmesg`, `/proc/cmdline` |
| 3 | Enable kernel IOMMU | `/etc/default/grub`, initramfs | **yes** | `dmesg \| grep DMAR` |
| 4 | Identify IOMMU groups | no | no | `find /sys/kernel/iommu_groups/` |
| 5 | Make vfio-pci loadable (no auto-claim) | `modules-load.d` | **yes** | `lsmod \| grep vfio`, `lspci -nnk` |
| 6 | Install libvirt + QEMU | apt, services | no | `virsh version`, `virt-host-validate` |
| 7 | Default network + storage pool | libvirt | no | `virsh net-list`, `virsh pool-list` |
| 8 | End-to-end smoke test of managed handoff | one-shot VM | no | `nvidia-smi` round-trip pre/post |

---

## 1. Phase 1 -- Detect environment

### 1.1 CPU vendor and virtualisation features

```bash
[verify] $ lscpu | grep -E '^(Vendor ID|Model name|Virtualization)'
[verify] $ grep -Eqo '(vmx|svm)' /proc/cpuinfo && echo "virt-ext present" || echo "MISSING"
```

Stop here if `vmx` (Intel) or `svm` (AMD) is absent -- the CPU lacks
hardware virtualisation extensions.

### 1.2 IOMMU readiness

```bash
[verify] # dmesg | grep -E 'DMAR|IOMMU|AMD-Vi' | head -10
[verify] $ cat /proc/cmdline
```

If `/proc/cmdline` does not include `intel_iommu=on` (Intel) or
`amd_iommu=on` (AMD), the kernel is not using IOMMU even if the
firmware exposes it. Phase 3 fixes this.

### 1.3 GPU inventory

```bash
[verify] $ lspci -nn | grep -iE 'vga|display|3d controller|audio'
```

Identify three things:

- **`${IGPU_BDF}`** -- bus-device-function of the GPU the **host**
  uses for display (typically integrated, e.g. `00:02.0` Intel
  `[8086:7d67]`).
- **`${DGPU_BDF}`** -- bus-device-function of the GPU dedicated to
  test VMs (typically discrete NVIDIA, e.g. `02:00.0` `[10de:2c34]`).
- **`${DGPU_AUDIO_BDF}`** -- the audio function on the same physical
  card, must be passed through together (typically `02:00.1`
  `[10de:22e9]` for NVIDIA cards).

For this repository's reference host (Intel Arrow Lake-S +
NVIDIA RTX PRO 4000 Blackwell), the values are:

```
IGPU_BDF=0000:00:02.0   IGPU_ID=8086:7d67
DGPU_BDF=0000:02:00.0   DGPU_ID=10de:2c34
DGPU_AUDIO_BDF=0000:02:00.1   DGPU_AUDIO_ID=10de:22e9
```

If the agent's host differs, replace these values throughout the
remainder of the document.

### 1.4 Boot loader

```bash
[verify] $ test -d /sys/firmware/efi && echo "EFI" || echo "BIOS"
[verify] $ test -f /etc/default/grub && echo "GRUB"
[verify] $ test -d /boot/loader      && echo "systemd-boot"
```

The reference host is **EFI + GRUB**. If the host uses systemd-boot,
replace the GRUB-specific commands in section 3 with edits to
`/boot/loader/entries/*.conf`.

### 1.5 Currently bound GPU driver

```bash
[verify] $ lspci -nnk -s "${DGPU_BDF}" | sed -n 's/^[[:space:]]*Kernel driver in use:/&/p'
[verify] $ lsmod | grep -E '^(nvidia|vfio)' || echo "neither nvidia nor vfio loaded"
```

Expected starting state on a host already running Dev AI Lab
natively: `Kernel driver in use: nvidia`. After this entire
procedure completes, the line still reads `nvidia` between test
runs -- only during an active VM run does it temporarily change to
`vfio-pci`.

---

## 2. Phase 2 -- Confirm VT-d / VT-x in UEFI firmware

This phase mutates **firmware**, not the OS, and the agent must
**not** automate it. The operator boots into UEFI setup and confirms:

- **Intel hosts**: `Intel VT-d` (sometimes labelled "VT for Directed
  I/O") is enabled. Distinct from `Intel VT-x` (CPU virt extension),
  which must also be enabled.
- **AMD hosts**: `AMD-Vi` / `IOMMU` is enabled. SVM (CPU virt) must
  also be enabled.
- Recommended: `Resizable BAR` / `Above 4G Decoding` enabled (modern
  NVIDIA cards benefit; required for some).

Setting names vary by motherboard. After saving and rebooting:

```bash
[verify] # dmesg | grep -E 'DMAR.*IOMMU enabled|AMD-Vi.*Initialized'
```

A line like `DMAR: IOMMU enabled` (Intel) or AMD-Vi initialisation
(AMD) confirms the firmware exposes IOMMU to the kernel.

---

## 3. Phase 3 -- Enable kernel IOMMU support

UEFI exposing IOMMU is necessary but not sufficient -- the kernel
needs `intel_iommu=on iommu=pt` (Intel) or `amd_iommu=on iommu=pt`
(AMD) on its command line. `iommu=pt` ("passthrough") improves
performance for non-VFIO devices by letting them bypass the IOMMU.

### 3.1 Edit GRUB

```bash
# cp /etc/default/grub /etc/default/grub.bak.$(date -u +%Y%m%dT%H%M%SZ)
# vi /etc/default/grub
```

Find `GRUB_CMDLINE_LINUX_DEFAULT=` and **append** `intel_iommu=on
iommu=pt` (Intel) or `amd_iommu=on iommu=pt` (AMD). Don't replace
existing parameters. For the reference host, the existing line is:

```
GRUB_CMDLINE_LINUX_DEFAULT="ipv6.disable=1 quiet"
```

Edited:

```
GRUB_CMDLINE_LINUX_DEFAULT="ipv6.disable=1 quiet intel_iommu=on iommu=pt"
```

Regenerate the GRUB config:

```bash
# update-grub                              # Debian / Ubuntu
# grub2-mkconfig -o /boot/grub2/grub.cfg   # Fedora / RHEL / openSUSE
```

### 3.2 Reboot

```bash
# systemctl reboot
```

The agent **must not** reboot autonomously. Surface "reboot required
to enable IOMMU" and wait for the operator.

### 3.3 Verification

After reboot:

```bash
[verify] $ cat /proc/cmdline | grep -oE '(intel|amd)_iommu=on'
[verify] # dmesg | grep -E 'DMAR.*IOMMU enabled|AMD-Vi.*Initialized' | head -3
```

Without these lines, the rest of the procedure cannot work.

### 3.4 Recovery

| Symptom | Recovery |
|---|---|
| `update-grub` reports "command not found" | Non-GRUB loader; edit `/boot/loader/entries/*.conf` (systemd-boot). |
| Kernel param applied but `dmesg` still says no IOMMU | Phase 2 firmware setting was missed. Reboot into UEFI and recheck. |
| Host won't boot after GRUB regenerate | Press `e` at the GRUB menu and remove `iommu` params for one boot, then restore from `/etc/default/grub.bak.*`. |

---

## 4. Phase 4 -- Identify IOMMU groups for the dedicated GPU

The kernel exposes IOMMU groups under `/sys/kernel/iommu_groups/`.
Devices in the same group can only be passed through together. A
clean group for the GPU contains only the GPU and its audio function;
a polluted group contains additional devices (USB, NIC, SATA) that
would have to be given to the VM as well -- usually unacceptable.

```bash
[verify] # for d in /sys/kernel/iommu_groups/*/devices/${DGPU_BDF}; do
            grp=$(basename "$(dirname "$(dirname "$d")")")
            echo "GPU is in IOMMU group $grp"
            echo "Group $grp contents:"
            ls -1 /sys/kernel/iommu_groups/$grp/devices/
          done
```

Expected on the reference host: the group contains exactly two BDFs
(GPU `02:00.0` + audio `02:00.1`).

If extra devices appear:

1. **Move the card to a different PCIe slot.** Many motherboards
   group all devices behind a single root port together; the GPU's
   slot may share a port with onboard NICs or USB controllers. The
   motherboard manual typically labels which slot has its own IOMMU
   group.
2. **ACS override patch.** Some kernels (Proxmox, hand-built) split
   groups artificially. Distro kernels do not include this patch.

For the reference host this group is clean -- no further action.

---

## 5. Phase 5 -- Make vfio-pci loadable, no auto-claim

Unlike a permanent-VFIO setup, this phase **does not** put a
`vfio-pci ids=...` directive in `/etc/modprobe.d/`. We simply ensure
the vfio modules are *available* so libvirt can load and use them
on demand, while leaving the nvidia driver as the boot-time owner
of the GPU.

### 5.1 Module-load configuration

```bash
# cat > /etc/modules-load.d/vfio.conf <<'EOF'
# Make the VFIO module stack available at boot. We deliberately do
# NOT put `options vfio-pci ids=...` here -- that would auto-claim
# the GPU and prevent the nvidia driver from binding it during
# normal operation. With these modules merely loaded (not bound to
# any device), libvirt's `<hostdev managed='yes'>` can detach
# nvidia and bind vfio-pci on demand at VM start, then reverse it
# on VM shutdown.
vfio
vfio_iommu_type1
vfio_pci
EOF
```

If a previous attempt configured permanent VFIO on this host,
remove the offending files first:

```bash
# rm -f /etc/modprobe.d/vfio.conf                         # ids= directive
# rm -f /etc/modprobe.d/vfio-softdep.conf                 # softdep nvidia
# rm -f /etc/modprobe.d/blacklist-nvidia-bare.conf        # nvidia blacklist
```

### 5.2 Update the initramfs

```bash
# update-initramfs -u -k all                # Debian / Ubuntu
# dracut --regenerate-all --force           # Fedora / RHEL / openSUSE
```

### 5.3 Reboot

```bash
# systemctl reboot
```

### 5.4 Verification

After reboot:

```bash
[verify] $ lsmod | grep -E '^vfio' | sort
```

Expected: `vfio`, `vfio_iommu_type1`, `vfio_pci` all present.

```bash
[verify] $ lspci -nnk -s ${DGPU_BDF}
```

Expected: `Kernel driver in use: nvidia`. The GPU is still in
nvidia's hands at idle -- exactly what we want.

```bash
[verify] $ nvidia-smi -L
```

Expected: lists the dedicated GPU and reports its name (`RTX PRO
4000 Blackwell`). The bare-host nvidia stack still works.

---

## 6. Phase 6 -- Install libvirt + QEMU stack

```bash
# Debian / Ubuntu
# apt-get update
# apt-get install -y --no-install-recommends \
        qemu-system-x86 qemu-utils ovmf \
        libvirt-daemon-system libvirt-clients virtinst \
        bridge-utils dnsmasq-base

# Fedora / RHEL
# dnf install -y \
        @virtualization \
        qemu-kvm qemu-img edk2-ovmf \
        libvirt libvirt-client virt-install

# openSUSE
# zypper install -y \
        qemu qemu-x86 qemu-tools qemu-ovmf-x86_64 \
        libvirt libvirt-client virt-install

# Arch
# pacman -S --noconfirm \
        qemu-full edk2-ovmf \
        libvirt virt-install dnsmasq
```

`ovmf` / `edk2-ovmf` provides the UEFI firmware images needed for
OVMF-based VMs. UEFI is strongly preferred over legacy BIOS for GPU
passthrough; modern NVIDIA cards typically refuse to function in
legacy mode.

Enable + start libvirt and add the operator to the `libvirt` group:

```bash
# systemctl enable --now libvirtd
# usermod -aG libvirt ${INVOKING_USER}
$ newgrp libvirt              # or log out / in to pick up the group
```

### 6.1 Verification

```bash
[verify] $ virsh version
[verify] $ qemu-system-x86_64 --version
[verify] $ ls /usr/share/OVMF/        # Debian / Ubuntu
[verify] $ ls /usr/share/edk2-ovmf/   # Fedora / RHEL / Arch
[verify] $ virt-host-validate
```

`virt-host-validate` should report PASS on hardware virtualisation,
KVM presence, IOMMU, and IOMMU groups. cgroup-related WARNs are
typically benign on cgroup-v2 hosts.

---

## 7. Phase 7 -- Default libvirt network and storage pool

```bash
# virsh net-start  default      2>/dev/null || true
# virsh net-autostart default
# virsh pool-start  default     2>/dev/null || true
# virsh pool-autostart default
```

Default storage pool at `/var/lib/libvirt/images/` is fine. If
`/var/lib/libvirt/` lives on the host's main partition and you want
test VM images on a dedicated LV instead, point libvirt at a
different path with `virsh pool-define-as test-images dir
--target /path/to/dir; virsh pool-build test-images; virsh
pool-start test-images; virsh pool-autostart test-images`.

### 7.1 Verification

```bash
[verify] $ virsh net-list --all
[verify] $ virsh pool-list --all
```

Expected: `default` network and `default` pool both `active` +
`autostart yes`.

---

## 8. Phase 8 -- End-to-end smoke test of GPU handoff

Confirm that the bind/unbind dance actually works: the GPU goes from
`nvidia` -> `vfio-pci` before VM start, and back to `nvidia` after VM
stop.

> **Note on syntax.** Debian Trixie ships libvirt 11.x with a
> `virt-install` whose `--hostdev` does not accept the `managed=yes`
> or `driver.name=vfio` sub-options that newer libvirt versions
> introduced. We therefore do the detach/reattach **manually** with
> `virsh nodedev-detach` / `virsh nodedev-reattach` before and after
> the VM run, and pass the device to `virt-install` as a plain
> `--hostdev pci_0000_XX_XX_X`. Functionally identical to
> `managed='yes'`; the test harness in `tests/test-install-vm.sh`
> uses the same explicit pair so production stack lifecycle (stop
> before, restart after) is obvious.

### 8.1 Pre-flight: free the GPU

```bash
[verify] $ nvidia-smi --query-compute-apps=pid --format=csv,noheader
```

Output must be empty. Anything actively using the GPU
(Dev AI Lab containers, CUDA processes, etc.) blocks the detach
with "device is in use".

Common GPU-holders to stop:

```bash
# 1. Production devai stack (if running natively on the host)
$ cd /path/to/devai && make cache-down

# 2. NVIDIA CUDA MPS daemon (sometimes auto-started by HPC tooling)
$ sudo lsof /dev/nvidia* 2>/dev/null
# If nvidia-cuda-mps-control appears, kill it:
$ sudo pkill -x nvidia-cuda-mps-control || sudo kill <pid>

# 3. nvidia-persistenced (kept-warm GPU initialiser)
$ sudo systemctl stop nvidia-persistenced 2>/dev/null || true

# 4. Anything else holding /dev/nvidia*
$ sudo lsof /dev/nvidia* 2>/dev/null   # should be empty after the above
```

If the host runs Dev AI Lab natively, take it down first:

```bash
$ cd /path/to/devai && make cache-down
```

### 8.2 Detach the GPU from `nvidia` and bind it to `vfio-pci`

```bash
$ sudo virsh nodedev-detach pci_0000_02_00_0
$ sudo virsh nodedev-detach pci_0000_02_00_1

# Confirm both functions flipped
[verify] $ lspci -nnk -s 02:00.0 | grep 'Kernel driver in use'   # vfio-pci
[verify] $ lspci -nnk -s 02:00.1 | grep 'Kernel driver in use'   # vfio-pci
```

If either `nodedev-detach` hangs, return to section 8.1: something on the
host is still holding the GPU. The kernel unbind blocks until every
holder closes its file descriptor.

### 8.3 Define a minimal smoke VM with the GPU attached

```bash
$ wget -O /var/lib/libvirt/images/debian-13-generic-amd64.qcow2 \
       https://cloud.debian.org/images/cloud/trixie/latest/debian-13-generic-amd64.qcow2

$ virt-install \
      --name vfio-smoke \
      --memory 2048 --vcpus 2 \
      --osinfo name=debiantesting \
      --boot uefi \
      --disk path=/var/lib/libvirt/images/debian-13-generic-amd64.qcow2,format=qcow2,bus=virtio \
      --network default \
      --hostdev pci_0000_02_00_0 \
      --hostdev pci_0000_02_00_1 \
      --import --noautoconsole
```

The plain `--hostdev pci_0000_XX_XX_X` (no sub-options) is what
Debian Trixie's `virt-install` accepts. The device is already in
`vfio-pci` from section 8.2, so libvirt just claims it.

### 8.4 Verify and tear down

```bash
[verify] $ virsh list --all                                  # vfio-smoke   running
[verify] $ virsh dumpxml vfio-smoke | grep -E 'hostdev|address.*type=.pci' | head

$ sleep 15
$ virsh destroy vfio-smoke
$ virsh undefine vfio-smoke --nvram                          # keep disk image

# Reattach: rebind GPU + audio function back to `nvidia`
$ sudo virsh nodedev-reattach pci_0000_02_00_0
$ sudo virsh nodedev-reattach pci_0000_02_00_1

[verify] $ lspci -nnk -s 02:00.0 | grep 'Kernel driver in use'   # nvidia
[verify] $ nvidia-smi -L                                          # GPU listed normally
```

That's the smoke passing: nodedev-detach flipped `nvidia -> vfio-pci`,
the VM ran with the GPU, `virsh destroy` released it, `nodedev-reattach`
flipped it back, and `nvidia-smi -L` works again.

### 8.5 Recovery

| Symptom | Recovery |
|---|---|
| `virsh start` fails with "device in use" | A process on the host still uses the GPU. Check `nvidia-smi --query-compute-apps`, `lsof /dev/nvidia*`, or stop Dev AI Lab. |
| `virsh start` fails with "vfio: device assignment requires CONFIG_VFIO" | vfio modules not loaded. Run `modprobe vfio_pci` and re-check `lsmod`. |
| GPU stuck in vfio-pci after VM destroy (nvidia-smi reports "no devices") | Some cards don't reset cleanly. Run `virsh nodedev-reattach pci_0000_${DGPU_BDF//[:.]/_}` manually. If still stuck, reboot. Modern Ampere/Ada/Blackwell cards generally handle this gracefully; older cards are worse. |
| `Kernel driver in use:` shows neither nvidia nor vfio-pci | The driver chain failed mid-rebind. `modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia` (in that order) then `modprobe nvidia`. |

---

## 9. State-of-the-world reference

After sections 1-7 succeed, **at idle** (no test VM running):

```
Host (bare metal)
+-- /etc/default/grub                    GRUB_CMDLINE_LINUX_DEFAULT includes
|                                        intel_iommu=on iommu=pt
+-- /etc/modules-load.d/vfio.conf        vfio, vfio_iommu_type1, vfio_pci loaded
+-- (no /etc/modprobe.d/vfio*.conf --     so vfio-pci does NOT auto-claim any device)
|   no ids=, no softdep, no blacklist)
+-- /sys/kernel/iommu_groups/<N>/devices/
|   +-- 0000:02:00.0  (GPU,   driver: nvidia)
|   +-- 0000:02:00.1  (audio, driver: snd_hda_intel)
|
+-- nvidia driver                        loaded, bound to 02:00.* (host display
|                                        unaffected -- host uses iGPU)
+-- vfio_pci module                      loaded, bound to nothing
|
+-- libvirtd.service                     active, autostart
+-- default libvirt net + pool           active, autostart
```

**During a test VM run**, libvirt automatically detaches the GPU
from nvidia and binds it to vfio-pci. After VM shutdown, libvirt
reverses everything and the GPU is back under nvidia. No host-side
action required for the switching.

---

## 10. Tear-down -- remove all VFIO setup

To revert this host to plain bare-metal (no VFIO infrastructure at
all):

```bash
# 1. Remove vfio module-load config
# rm -f /etc/modules-load.d/vfio.conf

# 2. Remove the IOMMU kernel cmdline params (optional; harmless to keep)
#    Edit /etc/default/grub, remove `intel_iommu=on iommu=pt`
#    from GRUB_CMDLINE_LINUX_DEFAULT, then update-grub.

# 3. Regenerate initramfs
# update-initramfs -u -k all       # Debian / Ubuntu
# dracut --regenerate-all --force  # Fedora / RHEL / openSUSE

# 4. Remove libvirt + QEMU (optional -- they're not actively claiming
#    the GPU and don't need to be removed unless disk space matters)
# apt-get remove --purge libvirt-daemon-system libvirt-clients virtinst \
#                        qemu-system-x86 qemu-utils ovmf
# (or distro equivalent)

# 5. Reboot
# systemctl reboot
```

After reboot:

```bash
[verify] $ lsmod | grep vfio          # should be empty
[verify] $ lspci -nnk -s 0000:02:00.0 # Kernel driver in use: nvidia
[verify] $ nvidia-smi -L              # GPU listed as before
```

---

## 11. Decisions left to the operator

The agent must surface these to the operator and wait for an
explicit answer rather than choosing autonomously:

1. **Reboot timing.** Phases 3 and 5 require reboots; the agent
   waits, does not initiate.
2. **Storage pool location** (section 7). Default is libvirt's
   `/var/lib/libvirt/images/`. On a host with the LVM cache pool
   from INSTALL.md present, the operator may prefer a dedicated LV.
3. **Resizable BAR / Above 4G Decoding** in UEFI (section 2). Strongly
   recommended for modern GPUs; some boards lock these behind
   `CSM disabled`.
4. **Pre-flight stop of Dev AI Lab** (section 8.1, before each test VM
   run). The test harness automates this, but if the operator runs
   `virsh start` manually, the production stack must be down first.

For everything else, the agent decides per the procedure above.

---

## 12. Versioning of this file

When the upstream packages or kernel APIs change, regenerate the
relevant section:

- Section 6 package lists -- keep in sync with `apt-cache search libvirt`,
  `dnf search libvirt`, etc.
- Section 8 `virt-install` flag set -- libvirt occasionally renames `--hostdev`
  sub-options across major versions.
- Section 10 sysfs paths and module names -- stable across kernels.

For the actual INSTALL.md test harness that consumes the host
configured here, see `tests/test-install-vm.sh` (to be written
in a separate change after this host setup is in place).
