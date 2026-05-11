# INSTALL.md — VM-based bootstrap procedure for Dev AI Lab

This document is the **single, self-contained installation procedure
for Dev AI Lab onto a QEMU/KVM virtual machine** on a Linux host with
a dedicated NVIDIA GPU. It is written for an autonomous coding agent
(Claude Code, OpenAI Codex, or equivalent tier-1 agent) to read and
execute end-to-end with **no human interaction beyond preparing the
host once** per [`docs/HOST_VFIO_SETUP.md`](docs/HOST_VFIO_SETUP.md)
and granting `sudo` on the host so the agent can drive libvirt.

The procedure has two parts:

1. **§1 — Host readiness gate.** The agent verifies that the host has
   already been prepared for PCI passthrough per
   `docs/HOST_VFIO_SETUP.md`. If any precondition is missing, the
   agent **stops** and prints the exact remediation hints. The host
   must be configured by the operator; the agent does **not** mutate
   firmware, kernel command line, initramfs, or VFIO module config.
2. **§2 onwards — Autonomous VM provisioning.** Once the gate passes,
   the agent creates a Debian 13 (Trixie) VM, hands it the passed-
   through GPU, and runs the full install (storage, container
   runtime, lab build, model probe, smoke test) inside the guest.
   The operator is not consulted again until the final §14 smoke
   test reports green.

Bare-metal installation is out of scope. The guest is always Debian
13 (Trixie) on x86_64 with the host's NVIDIA GPU passed through; no
other guest OS, architecture, or storage layout is supported. For
the runtime architecture the VM ends up running, see
`docs/router.md`, `docs/backends.md`, the project `CLAUDE.md`, and
the in-repo `README.md`.

---

## Decisions recorded in this procedure

The agent applies the following decisions **without asking**. They
narrow the install to one supported configuration so the procedure
can run autonomously. Anything the operator wants to override must
be set in the environment or `.env` **before** the agent starts.

| #   | Decision | Why locked here |
|-----|----------|-----------------|
| D1  | Hypervisor = libvirt + QEMU/KVM. No Hyper-V, VMware, VirtualBox, Proxmox CLI. | `HOST_VFIO_SETUP.md` documents only this stack. |
| D2  | Guest OS = Debian 13 (Trixie) generic cloud image, x86_64. | Project reference platform; every `apt-get`, NVIDIA-driver path, and `docker-compose` step in §4 targets Trixie directly. |
| D3  | Single VM disk: `vda` qcow2, 250 GiB. Cloud-init partitions it into `vda1` = 50 GiB root (ext4) and `vda2` = 200 GiB LVM PV for `vgais`. | Lab images live on `cache_registry` (§5.3) so the root never carries model or image bloat; 50 GiB is enough for the OS, system packages, and a working set. One disk simplifies provisioning and host snapshotting. |
| D4  | VM resources: 4 vCPUs, 16 GiB RAM. | Minimum that completes `make build` + vLLM/SGLang cold-start without OOM on a 24 GiB-class GPU while leaving host headroom for the desktop / hypervisor. |
| D5  | VM name = `devai-vm`. | Single canonical name. If a VM of that name already exists, the agent refuses to clobber it without an explicit `DEVAI_VM_RECREATE=1`. |
| D6  | Network = libvirt `default` NAT. SSH to the VM uses the NAT lease. | Per `HOST_VFIO_SETUP.md` §7. No bridges, no macvtap. |
| D7  | Cloud-init: agent generates a fresh ED25519 keypair on the host at `~/.ssh/devai-vm` and seeds it as `authorized_keys` for guest user `devai` with passwordless sudo. | Agent never logs in interactively; SSH key auth only. |
| D8  | GPU BDFs are **discovered**, not hard-coded. | `lspci -nn` parse yields `${DGPU_BDF}` and `${DGPU_AUDIO_BDF}`. Reference values from `HOST_VFIO_SETUP.md` §1.3 (`0000:02:00.0` / `0000:02:00.1`) are illustrative only. |
| D9  | GPU detach: `virsh nodedev-detach` of both functions immediately before `virsh start`; reattach automatically on `virsh destroy` (§16 tear-down). | Per `HOST_VFIO_SETUP.md` §8.2/§8.4. The agent does this once per provisioning run. |
| D10 | Host GPU pre-flight: agent confirms `nvidia-smi --query-compute-apps` is empty, no `lsof /dev/nvidia*` holders, `nvidia-persistenced` stopped, no MPS daemon. If any holder remains, halt with the offending PID list. | Otherwise `nodedev-detach` blocks forever (§8.5 recovery). |
| D11 | NVIDIA driver path **inside the VM** = CUDA repo `nvidia-open`. | Blackwell-class cards require the open kernel modules; Debian's `nvidia-driver` silently fails on Blackwell. `nvidia-open` works on Ampere/Ada too, so the agent picks it unconditionally. |
| D12 | In-VM reboot count = exactly one, after NVIDIA driver install. Agent waits for SSH to come back via `until ssh ... true; do sleep 5; done`. | Host kernel IOMMU and VFIO-module reboots are the host operator's responsibility per `HOST_VFIO_SETUP.md`; inside the VM only the driver install needs a reboot. |
| D13 | Compose provider = Go `docker-compose` v2 (downloaded into the VM during §4). | Debian Trixie's Python `podman-compose` 1.3.0 does not expand `${VAR:-default}` and breaks the router's on-demand vLLM/SGLang recreate. |
| D14 | `JUPYTER_TOKEN` = random 32-char hex generated by the agent and written to `.env` before §10. Token is echoed once into the run log. | Removes an operator prompt; auto-start in §13 then works without a re-prompt. |
| D15 | HuggingFace token is **not required** for the default procedure. The `NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` repo (and the fallback `nvidia/Qwen3-8B-NVFP4`) are public on HF -- no licence click-through, anonymous downloads work. If the operator does keep a token at `${HOME}/.cache/huggingface/token` or `${HF_TOKEN}` on the host, the agent injects it into the VM via cloud-init `write_files` for higher anonymous rate limits and future-proofing; if not, the agent proceeds. | Verified against the live HF model pages: both targets show no gating banner. The token would only matter if the operator later swaps in a gated model. |
| D16 | First Ollama model = `qwen3.5:9b-q8_0`. NVFP4 model for the mandatory vLLM probe and real end-to-end test = `NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`. SGLang is **not probed** by this procedure (it stays as a sleeping placeholder in the compose stack for runtime use). | Ollama qwen3.5 covers the GGUF path; Nemotron-3-Nano-30B-A3B-NVFP4 (3 B active params over 30 B total) covers the NVFP4 + vLLM path within 24 GiB VRAM. Probing SGLang doubles wall time without adding coverage the vLLM probe doesn't already give. |
| D17 | `devai-agent` is installed inside the VM (not on the host) and supports non-interactive model/agent selection plus one-shot prompts via `--model`, `--agent`, and `--prompt`. The §11.3 e2e test drives the launcher with these flags rather than `podman run` directly. | Host is a pure hypervisor for this procedure; the lab lives in the guest. Using the launcher in the e2e test exercises the same surface real users will hit, instead of a parallel test-only invocation path. |
| D18 | Public exposure of VM services is the operator's job. Agent does not touch host firewall, iptables, nftables, or libvirt forward rules. | Exposing the lab or router beyond the host requires an operator-owned reverse-proxy + auth layer that this procedure does not provision. |
| D19 | Tear-down (§16) destroys the VM, removes its disks, and runs `virsh nodedev-reattach` on the GPU functions. It does **not** revert the host VFIO setup — that is `HOST_VFIO_SETUP.md` §10. | Symmetric with provisioning: this doc owns the VM lifecycle, not the host. |
| D20 | Cloud-init seed ISO built with `cloud-localds` (Debian package `cloud-image-utils`). If absent, fall back to `genisoimage`. | Both are in Debian main; one of them is always available on a host that already meets `HOST_VFIO_SETUP.md` §6. |
| D21 | The agent runs on the host (the libvirt admin). All in-VM commands are executed via `ssh -i ~/.ssh/devai-vm devai@${VM_IP}`. | Single execution model; no remote-mode/local-mode branching. |
| D22 | Single supported guest = Debian 13 x86_64 with GPU. The host is one of the distros supported by `HOST_VFIO_SETUP.md` §6. | One supported guest plus one supported set of host distros keeps the validated code paths small. |

These decisions are recorded inline so a future review can locate
every place this document deliberately narrows behaviour for the
sake of autonomous execution.

---

## Conventions

- `$` runs on the **host** as the invoking (non-root) user.
- `#` runs on the **host** as `root` (use `sudo`).
- `[host]` is an explicit reminder that the command is host-side.
- `[vm]` runs **inside the VM** via
  `ssh -i ~/.ssh/devai-vm devai@${VM_IP} '...'`. The agent prefixes
  every such command with that SSH transport; the snippet shows only
  the inner command for readability.
- `[verify]` is a read-only check whose failure halts the agent.
- `${INVOKING_USER}` is the host user driving the agent (`$(id -un)`).
- `${VM_IP}` is the VM's IPv4 on the libvirt `default` network,
  resolved after first boot from `virsh net-dhcp-leases default`.
- `${REPO_DIR}` is the absolute path of the cloned `devai` repository
  **inside the VM** (default `/home/devai/devai`).

---

## 1. Host environment readiness check

The agent enters this section first. Every command in §1 is read-only
and runs on the host. If any check fails, the agent **prints the
remediation hint listed in §1.4 for that specific check and stops**.
The operator then completes the missing piece of
`docs/HOST_VFIO_SETUP.md` and re-runs the agent.

### 1.1 Why this gate exists

`HOST_VFIO_SETUP.md` is the one-time host procedure: it mutates UEFI,
the kernel command line, initramfs, and module-load configuration,
and requires at least two reboots. None of it is reversible from
inside a container or an unprivileged shell, and several steps need
operator presence in the firmware setup screen. INSTALL.md therefore
**must not** attempt to perform any of it. The agent's job here is
purely to confirm the host is in the expected post-setup state.

### 1.2 Required host state

All of the following must be true. The checks are listed in the
order the agent runs them.

| # | Check | Required value | If wrong, see §1.4 hint |
|---|-------|----------------|-------------------------|
| C1 | CPU virtualization extensions | `vmx` (Intel) or `svm` (AMD) in `/proc/cpuinfo` | H1 |
| C2 | Kernel IOMMU on cmdline | `intel_iommu=on` or `amd_iommu=on` in `/proc/cmdline` | H2 |
| C3 | IOMMU initialized | `DMAR: IOMMU enabled` (Intel) or `AMD-Vi ... Initialized` in `dmesg` | H2 |
| C4 | VFIO modules loaded | `vfio`, `vfio_iommu_type1`, `vfio_pci` all in `lsmod` | H3 |
| C5 | No auto-claim of the NVIDIA GPU | `lspci -nnk -s ${DGPU_BDF}` reports `Kernel driver in use: nvidia` (not `vfio-pci`). If the GPU is on `vfio-pci` at idle and the binding is **not** permanent (no `options vfio-pci ids=...` in `/etc/modprobe.d/`), the agent self-heals with `virsh nodedev-reattach` in §1.3 before failing. | H4 |
| C6 | libvirt installed and running | `virsh version` succeeds; `systemctl is-active libvirtd` returns `active` | H5 |
| C7 | `virt-install` available | `virt-install --version` succeeds | H5 |
| C8 | OVMF firmware available | `/usr/share/OVMF/OVMF_CODE.fd` or `OVMF_CODE_4M.fd` exists | H5 |
| C9 | libvirt `default` network active and autostart | `virsh net-info default` reports active + autostart | H6 |
| C10 | libvirt `default` storage pool active and autostart | `virsh pool-info default` reports active + autostart | H6 |
| C11 | Storage pool has ≥ 260 GiB free | `virsh pool-info default` `Available` ≥ 260 GiB | H7 |
| C12 | GPU not currently in use | `nvidia-smi --query-compute-apps=pid` empty AND `lsof /dev/nvidia*` empty | H8 |
| C13 | GPU IOMMU group is clean | the group containing `${DGPU_BDF}` contains only the GPU and its audio function | H9 |
| C14 | Cloud-init seed-ISO tool | `cloud-localds` (preferred) or `genisoimage` available | H10 |
| C15 | The host user has passwordless `sudo` for the specific commands this procedure invokes (or the agent runs as root) | Per-command probe via `sudo -n -l <cmd>` for the call-surface listed in §1.3 C15 succeeds for every entry | H11 |
| C16 | libvirt default pool dir is libvirt-group-writable with setgid, **and** the invoking host user is in the `libvirt` group | `stat` reports group `libvirt`, mode bits include `g+w` (020) and setgid (02000); `id -nG` lists `libvirt`. Required so §2.4 / §2.5 / §2.6 (`wget`, `cloud-localds`, `qemu-img create`) can write to `/var/lib/libvirt/images` as the unprivileged user and the resulting files are readable by libvirtd. | H12 |

### 1.3 Check commands

```bash
[host] [verify] $ grep -Eqo '(vmx|svm)' /proc/cpuinfo                         # C1
[host] [verify] $ grep -Eqo '(intel|amd)_iommu=on' /proc/cmdline              # C2
[host] [verify] # dmesg | grep -Eq 'DMAR.*IOMMU enabled|AMD-Vi.*Initialized'  # C3
[host] [verify] $ lsmod | awk '$1 ~ /^vfio(_iommu_type1|_pci)?$/ {print $1}' \
                   | sort -u | tr '\n' ' '                                    # C4
                  # must print: vfio vfio_iommu_type1 vfio_pci

# Discover the dedicated NVIDIA GPU and its audio function (D8)
[host] $ DGPU_BDF=$(lspci -nn | awk '
            /NVIDIA/ && /(VGA|3D)/ { gsub(":", " "); print "0000:" $1 ":" $2; exit }
         ')
[host] $ DGPU_AUDIO_BDF=$(lspci -nn | awk -v p="${DGPU_BDF#0000:}" '
            # p looks like "02:00.0"; strip the .function so "want"
            # is "02:00", which substring-matches the audio function
            # at "02:00.1" on the same bus:device.
            BEGIN { sub(/\.[0-9]+$/, "", p); want=p }
            /NVIDIA/ && /Audio/ && $0 ~ want { gsub(":", " "); print "0000:" $1 ":" $2; exit }
         ')
[host] [verify] $ test -n "${DGPU_BDF}" && test -n "${DGPU_AUDIO_BDF}"

# C5 -- GPU must be on nvidia at idle. If a previous run (or any other
# transient detach) left it on vfio-pci, self-heal by reattaching. Only
# halt with H4 when a *permanent* vfio-pci binding exists.
[host] $ current_drv=$(lspci -nnk -s "${DGPU_BDF}" \
                        | awk -F': ' '/Kernel driver in use/ {print $2}')
[host] $ if [ "${current_drv}" = vfio-pci ]; then
            if grep -lE '^[[:space:]]*options[[:space:]]+vfio-pci[[:space:]]+ids' \
                 /etc/modprobe.d/*.conf 2>/dev/null | grep -q .; then
              echo "C5: permanent vfio-pci binding present -- see H4"; exit 1
            fi
            sudo virsh nodedev-reattach pci_$(echo "${DGPU_BDF}"       | tr ':.' '_')
            sudo virsh nodedev-reattach pci_$(echo "${DGPU_AUDIO_BDF}" | tr ':.' '_')
         fi
[host] [verify] $ lspci -nnk -s "${DGPU_BDF}" \
                   | awk -F': ' '/Kernel driver in use/ {print $2}' \
                   | grep -qx nvidia                                          # C5

[host] [verify] $ virsh version                                               # C6
[host] [verify] # systemctl is-active libvirtd
[host] [verify] $ virt-install --version                                      # C7
[host] [verify] $ ls /usr/share/OVMF/OVMF_CODE*.fd >/dev/null 2>&1            # C8
[host] [verify] $ virsh net-info  default | grep -E 'Active:.*yes'            # C9
[host] [verify] $ virsh net-info  default | grep -E 'Autostart:.*yes'
[host] [verify] $ virsh pool-info default | grep -E 'State:.*running'         # C10
[host] [verify] $ virsh pool-info default | grep -E 'Autostart:.*yes'

# C11 — at least 260 GiB free in the default pool
[host] [verify] $ avail_bytes=$(virsh pool-info default --bytes \
                   | awk -F': +' '/Available/ {print $2}')
[host] [verify] $ test "${avail_bytes}" -ge $((260 * 1024**3))

# C12 -- nobody holds the GPU. The lsof target is the exact pair the
# NOPASSWD whitelist in §1.4 H11 grants; nvidia-modeset/uvm/etc are
# secondary nodes and a holder there would also hold nvidia0|nvidiactl.
[host] [verify] $ test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
[host] [verify] $ test -z "$(sudo lsof /dev/nvidia0 /dev/nvidiactl 2>/dev/null)"

# C13 — IOMMU group containment
[host] [verify] $ group=$(basename "$(dirname "$(dirname \
                   "$(ls /sys/kernel/iommu_groups/*/devices/${DGPU_BDF} 2>/dev/null)")")")
[host] [verify] $ test -n "${group}"
[host] [verify] $ allowed="${DGPU_BDF}\n${DGPU_AUDIO_BDF}"
[host] [verify] $ extras=$(ls /sys/kernel/iommu_groups/${group}/devices/ \
                   | grep -vxE "${DGPU_BDF}|${DGPU_AUDIO_BDF}" || true)
[host] [verify] $ test -z "${extras}"

[host] [verify] $ command -v cloud-localds >/dev/null || command -v genisoimage >/dev/null   # C14

# C15 -- per-command NOPASSWD probe. The procedure only invokes a
# narrow whitelist of binaries under sudo; we require NOPASSWD only
# for those. `sudo -n -l <cmd>` exits 0 iff <cmd> is permitted for the
# invoking user without re-authentication. If the agent is already
# running as root, the probe is moot.
[host] [verify] $ if [ "$(id -u)" -ne 0 ]; then
                    for cmd in \
                        /usr/bin/virsh \
                        /usr/bin/virt-install \
                        '/usr/bin/systemctl is-active --quiet libvirtd' \
                        '/usr/bin/lsof /dev/nvidia0 /dev/nvidiactl' ; do
                      sudo -n -l ${cmd} >/dev/null 2>&1 || {
                        echo "C15: NOPASSWD missing for: ${cmd}"; exit 1; }
                    done
                  fi                                                          # C15

# C16 -- pool dir is libvirt-group-writable with setgid AND invoking
# user is in the libvirt group. Needed for §2.4 wget, §2.5
# cloud-localds, and §2.6 qemu-img create to run without sudo and
# produce files libvirtd can read. The agent skips the user-membership
# half when running as root.
[host] $ POOL_PATH=$(virsh pool-dumpxml default \
                       | awk -F'[<>]' '/<path>/ {print $3; exit}')
[host] [verify] $ test -n "${POOL_PATH}"
[host] [verify] $ test "$(stat -c '%G' "${POOL_PATH}")" = libvirt
[host] [verify] $ perms=$(stat -c '%a' "${POOL_PATH}"); \
                  test $(( 8#${perms} & 020   )) -eq $((020))   && \
                  test $(( 8#${perms} & 02000 )) -eq $((02000))
[host] [verify] $ test "$(id -u)" -eq 0 || id -nG | tr ' ' '\n' | grep -qx libvirt   # C16
```

### 1.4 Remediation hints (printed only on failure)

If a check fails, the agent emits the matching hint **verbatim** and
stops. The operator fixes the host, then re-runs the agent from §1.

- **H1 — CPU lacks virt extensions.** This host cannot run KVM VMs.
  Replace the host. No software fix.
- **H2 — IOMMU not on kernel command line.** Follow
  `docs/HOST_VFIO_SETUP.md` §3: append `intel_iommu=on iommu=pt`
  (Intel) or `amd_iommu=on iommu=pt` (AMD) to
  `GRUB_CMDLINE_LINUX_DEFAULT`, run `update-grub`, reboot.
- **H3 — VFIO modules not loaded.** Follow
  `docs/HOST_VFIO_SETUP.md` §5: drop `/etc/modules-load.d/vfio.conf`
  listing `vfio`, `vfio_iommu_type1`, `vfio_pci`, run
  `update-initramfs -u -k all` (or distro equivalent), reboot.
- **H4 — GPU is bound to `vfio-pci` at idle via a *permanent* config.**
  H4 fires only when §1.3 C5 found `options vfio-pci ids=...` in
  `/etc/modprobe.d/*.conf` -- the transient case (stale VM that
  detached and never reattached) is auto-healed by §1.3 calling
  `virsh nodedev-reattach` and is not an operator-visible failure.
  Remove the offending `vfio-pci ids=` line per
  `docs/HOST_VFIO_SETUP.md` §5.1, run `update-initramfs -u -k all`
  (or distro equivalent), then reboot.
- **H5 — libvirt/virt-install/OVMF missing.** Install per
  `docs/HOST_VFIO_SETUP.md` §6 (`apt-get install -y --no-install-
  recommends qemu-system-x86 qemu-utils ovmf libvirt-daemon-system
  libvirt-clients virtinst bridge-utils dnsmasq-base`), then
  `systemctl enable --now libvirtd` and add `${INVOKING_USER}` to
  group `libvirt`.
- **H6 — libvirt `default` net or pool not active.** Run
  `virsh net-start default; virsh net-autostart default;
  virsh pool-start default; virsh pool-autostart default` per
  `docs/HOST_VFIO_SETUP.md` §7.
- **H7 — libvirt storage pool has < 260 GiB free.** Free space, or
  point `default` at a larger directory via `virsh pool-define-as`.
  The agent will not silently use a different pool.
- **H8 — GPU is currently in use.** Stop the holders before
  proceeding. Typical fixes (`docs/HOST_VFIO_SETUP.md` §8.1):
  `make cache-down` if a host-native Dev AI Lab is running,
  `sudo systemctl stop nvidia-persistenced`,
  `sudo pkill -x nvidia-cuda-mps-control`. Re-run `lsof /dev/nvidia*`
  until empty.
- **H9 — IOMMU group is polluted.** Move the GPU to a different
  PCIe slot whose root port has its own IOMMU group. See
  `docs/HOST_VFIO_SETUP.md` §4. This is a hardware change, not a
  software one.
- **H10 — No seed-ISO tool.** `apt-get install -y cloud-image-utils`
  (provides `cloud-localds`) on the host, or
  `apt-get install -y genisoimage` as a fallback.
- **H11 — Operator lacks NOPASSWD coverage for the procedure's
  call-surface.** This procedure invokes only a narrow set of
  binaries under sudo; rather than granting blanket NOPASSWD, drop
  the following file (and only the following):

  ```
  # /etc/sudoers.d/devai-install   (mode 0440, validated with `visudo -cf`)
  ${INVOKING_USER} ALL=(root) NOPASSWD: /usr/bin/virsh
  ${INVOKING_USER} ALL=(root) NOPASSWD: /usr/bin/virt-install
  ${INVOKING_USER} ALL=(root) NOPASSWD: /usr/bin/systemctl is-active --quiet libvirtd
  ${INVOKING_USER} ALL=(root) NOPASSWD: /usr/bin/lsof /dev/nvidia0 /dev/nvidiactl
  ```

  Substitute `${INVOKING_USER}` with the actual host user. Validate
  with `sudo visudo -cf /etc/sudoers.d/devai-install` before saving.
  Alternatively, run the agent as root.
- **H12 — libvirt pool dir not group-writable, or user not in
  `libvirt` group.** Fix both halves:

  ```
  sudo chgrp libvirt /var/lib/libvirt/images
  sudo chmod g+ws    /var/lib/libvirt/images   # group-write + setgid
  sudo usermod -aG libvirt ${INVOKING_USER}
  ```

  The setgid bit (`g+s`) is mandatory -- without it, files created in
  the pool dir inherit the user's primary group and libvirtd may
  refuse to read them. `usermod -aG` does not take effect for the
  current session; log out and back in (or run `newgrp libvirt` in a
  fresh shell) before re-running the gate. If installing the agent's
  systemd unit, ensure it inherits the `libvirt` supplementary group.

### 1.5 Exit criterion

Every `[verify]` in §1.3 succeeded. The agent records `${DGPU_BDF}`,
`${DGPU_AUDIO_BDF}`, and the libvirt pool path. It now proceeds to
§2 without further operator interaction.

---

## 2. Provision the VM

The agent creates `devai-vm`, attaches the GPU, and waits for SSH.
Everything in §2 is host-side.

### 2.1 VM specification (recorded, not negotiable here)

| Property | Value | Source |
|----------|-------|--------|
| Name | `devai-vm` | D5 |
| vCPUs | 4 | D4 |
| Memory | 16 GiB | D4 |
| Firmware | UEFI (OVMF) | `HOST_VFIO_SETUP.md` §6 |
| Network | libvirt `default` (NAT) | D6 |
| Disk (`vda`) | 250 GiB qcow2; cloud-init partitions into `vda1` = 50 GiB root + `vda2` = 200 GiB LVM PV | D3 |
| Host devices | `${DGPU_BDF}`, `${DGPU_AUDIO_BDF}` | D8 / D9 |
| Guest OS | Debian 13 generic cloud image, x86_64 | D2 |
| Guest user | `devai`, passwordless sudo, ED25519 key auth | D7 |
| Seed | cloud-init NoCloud ISO (`seed.iso`) | D20 |

### 2.2 Idempotency: refuse to clobber an existing VM

```bash
[host] [verify] $ virsh dominfo devai-vm >/dev/null 2>&1 && \
                    test "${DEVAI_VM_RECREATE:-0}" = 1 || \
                    ! virsh dominfo devai-vm >/dev/null 2>&1
```

If a `devai-vm` already exists and `DEVAI_VM_RECREATE` is not `1`, the
agent stops and asks the operator to either `export
DEVAI_VM_RECREATE=1` (then re-run) or pick a different name by
exporting `DEVAI_VM_NAME=...` (then re-run; the agent substitutes that
name everywhere in §2). When `DEVAI_VM_RECREATE=1` is set, the agent
runs the §16 tear-down flow before continuing.

### 2.3 Generate SSH keypair (D7)

```bash
[host] $ KEY=${HOME}/.ssh/devai-vm
[host] $ test -f "${KEY}" || ssh-keygen -t ed25519 -f "${KEY}" -N '' \
                              -C "devai-vm $(date -u +%Y%m%d)"
[host] $ PUBKEY=$(cat "${KEY}.pub")
```

### 2.4 Fetch the Debian 13 cloud image

```bash
[host] $ POOL_PATH=$(virsh pool-dumpxml default \
                      | awk -F'[<>]' '/<path>/ {print $3; exit}')
[host] $ IMG="${POOL_PATH}/debian-13-generic-amd64.qcow2"
[host] $ test -f "${IMG}" || wget -O "${IMG}" \
            https://cloud.debian.org/images/cloud/trixie/latest/debian-13-generic-amd64.qcow2
```

The download is idempotent: a present file is reused.

### 2.5 Build cloud-init seed ISO (D20)

```bash
[host] $ WORK=$(mktemp -d)
[host] $ cat > "${WORK}/user-data" <<EOF
#cloud-config
hostname: devai-vm
users:
  - name: devai
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
    lock_passwd: true
    ssh_authorized_keys:
      - ${PUBKEY}

# D3: do not let cloud-init grow root to fill the whole 250 GiB disk.
# We partition manually below so vda1 stays at 50 GiB and vda2 holds
# the 200 GiB LVM PV for vgais.
growpart:
  mode: off
resize_rootfs: false

package_update: true
packages:
  - ca-certificates
  - curl
  - gnupg
  - git
  - make
  - openssh-server
  - parted
  - e2fsprogs

runcmd:
  - [ systemctl, enable, --now, ssh ]
  - |
    set -e
    test -e /var/lib/devai-partitioned && exit 0
    ROOT_DEV=\$(findmnt -no SOURCE /)
    DISK=/dev/\$(lsblk -no PKNAME "\$ROOT_DEV")
    PART_NUM=\$(echo "\$ROOT_DEV" | grep -oE '[0-9]+$')
    parted -s "\$DISK" resizepart "\$PART_NUM" 50GiB
    partprobe "\$DISK" || true
    resize2fs "\$ROOT_DEV"
    parted -s "\$DISK" mkpart primary 50GiB 100%
    partprobe "\$DISK" || true
    touch /var/lib/devai-partitioned
EOF
[host] $ cat > "${WORK}/meta-data" <<EOF
instance-id: devai-vm
local-hostname: devai-vm
EOF

# D15 — inject an HF token only if the operator already keeps one on
# the host. The default models in §11.2 are public, so a missing
# token is not a failure; the VM proceeds with anonymous HF downloads.
[host] $ if [ -s "${HOME}/.cache/huggingface/token" ] || [ -n "${HF_TOKEN:-}" ]; then
            TOKEN=$(cat "${HOME}/.cache/huggingface/token" 2>/dev/null || printf '%s' "${HF_TOKEN}")
            cat >> "${WORK}/user-data" <<EOF
write_files:
  - path: /home/devai/.cache/huggingface/token
    owner: devai:devai
    permissions: '0600'
    content: |
      ${TOKEN}
EOF
         fi

[host] $ SEED="${POOL_PATH}/devai-vm-seed.iso"
[host] $ if command -v cloud-localds >/dev/null; then
            cloud-localds "${SEED}" "${WORK}/user-data" "${WORK}/meta-data"
         else
            genisoimage -output "${SEED}" -volid cidata -joliet -rock \
                "${WORK}/user-data" "${WORK}/meta-data"
         fi
[host] $ rm -rf "${WORK}"
```

### 2.6 Allocate VM disk

```bash
[host] $ VDA="${POOL_PATH}/devai-vm.qcow2"
[host] $ qemu-img create -f qcow2 -F qcow2 -b "${IMG}" "${VDA}" 250G
```

`vda` is a 250 GiB copy-on-write overlay on the upstream cloud image
so the base stays untouched and re-provisioning is cheap. Cloud-init
(§2.5) repartitions the disk on first boot: `vda1` holds the 50 GiB
root filesystem and `vda2` is a fresh 200 GiB partition that §6 turns
into an LVM PV for `vgais`.

### 2.7 Host GPU pre-flight (D10)

Already verified in §1 (C12), but the agent re-checks immediately
before detaching — anything could have started a CUDA process in the
intervening seconds.

```bash
[host] [verify] $ test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
[host] [verify] $ test -z "$(sudo lsof /dev/nvidia0 /dev/nvidiactl 2>/dev/null)"
```

If non-empty, halt with hint H8. (`lsof` target matches the NOPASSWD
whitelist defined in §1.4 H11; see §1.3 C12 for the rationale.)

### 2.8 Detach GPU from nvidia, bind to vfio-pci (D9)

```bash
[host] $ DGPU_NODE=pci_$(echo "${DGPU_BDF}"       | tr ':.' '_')
[host] $ DGPU_AUDIO_NODE=pci_$(echo "${DGPU_AUDIO_BDF}" | tr ':.' '_')
[host] # virsh nodedev-detach "${DGPU_NODE}"
[host] # virsh nodedev-detach "${DGPU_AUDIO_NODE}"

[host] [verify] $ lspci -nnk -s "${DGPU_BDF}"       | grep -q 'Kernel driver in use: vfio-pci'
[host] [verify] $ lspci -nnk -s "${DGPU_AUDIO_BDF}" | grep -q 'Kernel driver in use: vfio-pci'
```

### 2.9 virt-install

```bash
[host] $ virt-install \
            --name devai-vm \
            --memory 16384 --vcpus 4 \
            --cpu host-passthrough \
            --osinfo name=debiantesting \
            --boot uefi \
            --disk path="${VDA}",format=qcow2,bus=virtio \
            --disk path="${SEED}",device=cdrom \
            --network network=default,model=virtio \
            --hostdev "${DGPU_NODE}" \
            --hostdev "${DGPU_AUDIO_NODE}" \
            --graphics none --console pty,target_type=serial \
            --import --noautoconsole
```

`--osinfo name=debiantesting` silences the `osinfo` warning on Trixie.
`--hostdev pci_...` matches the syntax that Debian Trixie's
`virt-install` accepts (per `HOST_VFIO_SETUP.md` §8 note); the GPU is
already in `vfio-pci` from §2.8 so libvirt just claims it.

### 2.10 Wait for SSH

```bash
[host] $ for _ in $(seq 1 60); do
            VM_IP=$(virsh net-dhcp-leases default \
                      | awk '/devai-vm/ {print $5}' | cut -d/ -f1)
            test -n "${VM_IP}" && break
            sleep 5
         done
[host] [verify] $ test -n "${VM_IP}"

[host] $ for _ in $(seq 1 60); do
            ssh -i ~/.ssh/devai-vm \
                -o StrictHostKeyChecking=no \
                -o UserKnownHostsFile=/dev/null \
                -o ConnectTimeout=2 \
                devai@${VM_IP} true && break
            sleep 5
         done
[host] [verify] $ ssh -i ~/.ssh/devai-vm \
                    -o StrictHostKeyChecking=no \
                    -o UserKnownHostsFile=/dev/null \
                    devai@${VM_IP} \
                    'cloud-init status --wait'
[host] [verify] $ ssh -i ~/.ssh/devai-vm \
                    -o StrictHostKeyChecking=no \
                    -o UserKnownHostsFile=/dev/null \
                    devai@${VM_IP} \
                    'lspci -nn | grep -i nvidia && lsblk /dev/vda'
```

`cloud-init status --wait` blocks until first-boot finishes, which
guarantees the §2.5 partitioning runcmd has completed (vda1 = 50 GiB,
vda2 = 200 GiB, unformatted) before the agent moves on.

Expected: the VM's `lspci` lists the NVIDIA GPU and its audio
function, and `lsblk` shows `vda1` (~50G ext4 mounted at `/`) plus
`vda2` (~200G, no filesystem). From here on, the agent uses
`ssh -i ~/.ssh/devai-vm devai@${VM_IP} '...'` for every `[vm]`
command. The seed ISO can stay attached; cloud-init runs only on
first boot.

### 2.11 Verification (exit criterion for §2)

```bash
[host] [verify] $ virsh dominfo devai-vm | grep -E 'State: +running'
[host] [verify] $ virsh dumpxml  devai-vm | grep -E 'hostdev|address.*type=.pci' | head
[vm]   [verify] $ lspci -nnk | grep -A2 NVIDIA | grep -q 'Kernel driver in use'
```

The in-VM `Kernel driver in use:` line will be empty (no nvidia driver
yet) at this point — that's expected; §3 installs the driver.

---

## 3. Detect environment inside the VM

The agent SSHs in and records the facts that drive later phases.

### 3.1 Platform identification

```bash
[vm] [verify] $ uname -m            # must be x86_64
[vm] [verify] $ uname -r
[vm] [verify] $ . /etc/os-release && test "${ID}" = debian && test "${VERSION_ID}" = 13
```

If any check fails, the cloud image is wrong; halt and surface to
the operator. With the recorded D2 decision the only path here is
Debian 13 (Trixie); other distros would mean the operator overrode
the image and accepted responsibility for the deviation.

### 3.2 GPU presence

```bash
[vm] [verify] $ lspci -nn | grep -iE 'NVIDIA.*(VGA|3D)' | head -1
[vm] [verify] $ test -e /dev/kvm || true   # informational only; KVM nesting not required
```

`/dev/nvidia0` does **not** exist yet — the host kernel module is in
the VM image only as `nouveau` (blacklisted on cloud images) or
absent. §4.1 (D11) installs the open NVIDIA driver.

### 3.3 Storage detection

The cache LVs (see §6) will live on `/dev/vda2`, the 200 GiB
partition that cloud-init carved out of `/dev/vda` in §2.5. The agent
records the current state of vda:

```bash
[vm] [verify] $ lsblk -no NAME,SIZE,FSTYPE /dev/vda
                # vda      250G
                # |-vda1   50G    ext4   (root, mounted at /)
                # |-vda14  ...                (BIOS boot, Debian cloud image)
                # |-vda15  ...    vfat        (EFI System Partition)
                # `-vda2   200G                (unformatted; vgais PV in §6)
```

If `/dev/vda2` is missing or already populated with LVM metadata,
cloud-init's partitioning step did not produce the expected layout.
Inspect `/var/log/cloud-init-output.log` inside the VM. The agent
does **not** repartition vda after first boot.

### 3.4 User and rootless podman pre-state

```bash
[vm] [verify] $ id
[vm] [verify] $ grep "^devai:" /etc/subuid | awk -F: '{print $3}'   # >= 65536
[vm] [verify] $ grep "^devai:" /etc/subgid | awk -F: '{print $3}'   # >= 65536
```

Cloud-init creates `devai` via `useradd`, which on Debian populates
subuid/subgid automatically. If the counts are below 65536, §4 fixes
it via `usermod --add-subuids/--add-subgids`.

### 3.5 Repo presence

```bash
[vm] [verify] $ test -d ${REPO_DIR}/.git || echo "missing — phase 7 will clone"
```

---

## 4. Install host packages inside the VM

Single distro = Debian 13 (D22).

### 4.1 Apt + base tooling

```bash
[vm] # apt-get update
[vm] # apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg git make \
        python3 python3-yaml python3-pip python3-venv \
        podman podman-compose \
        uidmap fuse-overlayfs slirp4netns passt \
        netavark aardvark-dns nftables \
        lvm2 xfsprogs thin-provisioning-tools \
        openssl mkcert
```

Why these specific packages:

- `passt` (binary `pasta`) is podman 5.x's default rootless network
  backend on Debian Trixie. Without it `podman run` fails with
  "could not find pasta".
- `netavark` + `aardvark-dns` + `nftables` are required for
  user-defined networks (the compose stack always uses one).
  Debian's `podman` package only **recommends** them;
  `--no-install-recommends` skips them.
- `thin-provisioning-tools` is mandatory for LVM thin-pool
  auto-activation at boot. Without it, the `cache_*` LVs come up
  inactive and every `/var/cache/devai/*` mount fails silently
  (see §6.7 `nofail` requirement).
- `mkcert` is optional for browser-trusted certs; the agent only
  uses it as a fallback if `openssl req` fails (it shouldn't, since
  the package is present).

### 4.2 Install Go-based `docker-compose` v2 (D13)

The Python `podman-compose` 1.3.0 shipped by Trixie does not expand
`${VAR:-default}` in `deploy/docker-compose.yaml`. The Go binary
does. `podman compose` (no hyphen) auto-detects the Go binary and
delegates to it when present.

```bash
[vm] # DC_VER=v2.40.0
[vm] # curl -fsSL \
        "https://github.com/docker/compose/releases/download/${DC_VER}/docker-compose-linux-$(uname -m)" \
        -o /usr/local/bin/docker-compose
[vm] # chmod +x /usr/local/bin/docker-compose

[vm] [verify] $ docker-compose version
[vm] [verify] $ podman compose version    # same version as docker-compose
```

### 4.3 NVIDIA driver — CUDA repo, `nvidia-open` (D11)

```bash
[vm] # wget -qO /tmp/cuda-keyring.deb \
        https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb
[vm] # dpkg -i /tmp/cuda-keyring.deb && rm /tmp/cuda-keyring.deb

[vm] # curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
[vm] # curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
        > /etc/apt/sources.list.d/nvidia-container-toolkit.list

[vm] # apt-get update
[vm] # apt-get install -y nvidia-open nvidia-container-toolkit linux-headers-amd64
```

`nvidia-open` pulls in `nvidia-kernel-open-dkms`, which builds the
open `nvidia.ko` against the running kernel using `linux-headers-
amd64`. On Blackwell the closed kernel modules from `cuda-drivers`
load but fail GPU init with `RmInitAdapter failed!`; on Ampere/Ada
both work. The agent unconditionally picks open (D11) for forward
compatibility.

### 4.4 Reboot the VM (D12)

```bash
[vm] # systemctl reboot
[host] $ until ssh -i ~/.ssh/devai-vm -o ConnectTimeout=2 devai@${VM_IP} true; do
            sleep 5
         done
[vm] [verify] $ nvidia-smi -L          # must list the passed-through GPU
[vm] [verify] $ test -e /dev/nvidia0
```

If `nvidia-smi` reports "couldn't communicate with the NVIDIA driver",
the most common cause is a kernel/header mismatch — `linux-headers-
amd64` lagged the running kernel. Run `apt-get install -y
linux-headers-$(uname -r)`, then `dpkg-reconfigure
nvidia-kernel-open-dkms`, and reboot once more.

### 4.5 subuid/subgid

Skip if §3.4 reported counts ≥ 65536. On a fresh cloud-init `devai`
user, those counts are usually already correct.

```bash
[vm] # usermod --add-subuids 100000-165535 devai
[vm] # usermod --add-subgids 100000-165535 devai
[vm] $ podman system migrate
```

### 4.6 Activate the podman socket

```bash
[vm] $ systemctl --user enable --now podman.socket
[vm] # loginctl enable-linger devai
```

`loginctl enable-linger` keeps `--user` services running after logout,
which the eventual §13 systemd auto-start unit relies on.

### 4.7 Verification

```bash
[vm] [verify] $ podman --version
[vm] [verify] $ podman compose version
[vm] [verify] $ python3 --version       # >= 3.11
[vm] [verify] $ make --version
[vm] [verify] $ nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[vm] [verify] $ test -S /run/user/$(id -u)/podman/podman.sock
```

---

## 5. Configure container runtime + GPU in the VM

### 5.1 NVIDIA Container Device Interface (CDI)

```bash
[vm] # nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
[vm] [verify] $ podman info --format '{{.Host.CgroupVersion}} {{.Host.Security.Rootless}}'
[vm] [verify] $ podman run --rm --device nvidia.com/gpu=all \
                  docker.io/library/debian:trixie nvidia-smi -L
```

The CDI device name `nvidia.com/gpu=all` is what
`deploy/docker-compose.yaml` references. `nvidia-smi -L` inside the
container must list the same GPU as §4.4.

### 5.2 containers.conf

Write only if missing — never clobber a populated file:

```bash
[vm] $ test -f ~/.config/containers/containers.conf || \
         nvidia-ctk runtime configure --runtime=podman \
           --config=~/.config/containers/containers.conf
```

### 5.3 Rootless podman storage relocation (mandatory)

Default graphroot is `~/.local/share/containers/storage`, which lives
on `vda1` (the 50 GiB root). The lab images (~30 GiB) plus the pulled
backend images (~30 GiB more) would fill it. Relocate **before any
podman command that materializes storage** (`pull`, `run`, `build`).

Once podman has written its sqlite state DB at the default path,
switching graphroot requires either an in-place edit of `db.sql` or a
destructive `podman system reset --force`. Setting `storage.conf`
before the first podman call avoids both.

```bash
[vm] $ mkdir -p ~/.config/containers
[vm] $ cat > ~/.config/containers/storage.conf <<'EOF'
[storage]
driver = "overlay"
graphroot = "/var/cache/devai/registry"
EOF
```

Verification is deferred to §6.8 because `/var/cache/devai/registry`
does not exist until §6 mounts it.

### 5.4 Registry mirror routing (optional)

The infra stack runs a `registry:2` pull-through cache on `:5000`.
Routing podman through it speeds up rebuilds.

```bash
[vm] $ mkdir -p ~/.config/containers
[vm] $ test -f ~/.config/containers/registries.conf || \
         cp ${REPO_DIR}/deploy/registries.conf \
            ~/.config/containers/registries.conf
```

The shipped file adds `localhost:5000` as an insecure mirror for
`docker.io` and falls back to `docker.io` directly if the mirror is
down. Safe to install before §10 (when the mirror starts).

### 5.5 Verification

The podman-storage probe is deferred to §6.8. The CDI probe (§5.1)
already ran.

---

## 6. Provision storage on `/dev/vda2`

The 200 GiB `vda2` becomes `vgais`, holding one 200 GiB thin pool and
seven thin LVs (§6.1).

### 6.1 Target layout

| LV name | Size (virtual) | Mountpoint | Filesystem | Owner |
|---|---|---|---|---|
| `cache_ollama` | 200G | `/var/cache/devai/ollama` | xfs | `devai` |
| `cache_registry` | 200G | `/var/cache/devai/registry` | xfs | `devai` |
| `cache_pip` | 30G | `/var/cache/devai/pip` | xfs | `devai` |
| `cache_apt` | 10G | `/var/cache/devai/apt` | xfs | `devai` |
| `cache_npm` | 10G | `/var/cache/devai/npm` | xfs | `devai` |
| `cache_open_webui` | 5G | `/var/cache/devai/open-webui` | xfs | `devai` |
| `cache_logs` | 100G | `/var/cache/devai/logs` | xfs | `devai` |

Sum of virtual sizes ≈ 555 GiB. The thin pool is **exactly 200 GiB**,
intentionally over-committed; thin provisioning charges only written
extents.

### 6.2 Volume group

```bash
[vm] # pvcreate /dev/vda2
[vm] # vgcreate vgais /dev/vda2
[vm] [verify] # vgs vgais --units g
```

### 6.3 Thin pool

```bash
[vm] # lvcreate -L 200G -T vgais/cachepool
[vm] [verify] # lvs vgais/cachepool
```

The pool size is exactly 200 GiB; do not scale.

### 6.4 Thin LVs

```bash
[vm] # for spec in \
        "cache_ollama:200G" \
        "cache_registry:200G" \
        "cache_pip:30G" \
        "cache_apt:10G" \
        "cache_npm:10G" \
        "cache_open_webui:5G" \
        "cache_logs:100G"; do
            lv="${spec%%:*}"; size="${spec##*:}"
            if ! lvs "vgais/${lv}" >/dev/null 2>&1; then
                lvcreate --thin --virtualsize "${size}" --name "${lv}" vgais/cachepool
            fi
        done
```

### 6.5 Filesystems

```bash
[vm] # for lv in cache_ollama cache_registry cache_pip cache_apt cache_npm cache_open_webui cache_logs; do
            dev="/dev/vgais/${lv}"
            blkid -p "${dev}" >/dev/null 2>&1 && continue
            mkfs.xfs -q "${dev}"
        done
```

### 6.6 Mounts and fstab

```bash
[vm] # mkdir -p /var/cache/devai/{ollama,registry,pip,apt,npm,open-webui,logs}
[vm] # for lv in cache_ollama cache_registry cache_pip cache_apt cache_npm cache_open_webui cache_logs; do
            dev="/dev/vgais/${lv}"
            uuid="$(blkid -s UUID -o value "${dev}")"
            case "${lv}" in
                cache_ollama)     mp="/var/cache/devai/ollama" ;;
                cache_registry)   mp="/var/cache/devai/registry" ;;
                cache_pip)        mp="/var/cache/devai/pip" ;;
                cache_apt)        mp="/var/cache/devai/apt" ;;
                cache_npm)        mp="/var/cache/devai/npm" ;;
                cache_open_webui) mp="/var/cache/devai/open-webui" ;;
                cache_logs)       mp="/var/cache/devai/logs" ;;
            esac
            findmnt -no SOURCE --mountpoint "${mp}" 2>/dev/null | grep -q "${dev}$" && continue
            cp -f /etc/fstab "/etc/fstab.bak.$(date -u +%Y%m%dT%H%M%SZ)"
            sed -i.tmp -e "\#\\s${mp}\\s#d" -e "\#${dev}\\s#d" \
                -e "\#UUID=${uuid}\\s#d" /etc/fstab
            rm -f /etc/fstab.tmp
            echo "UUID=${uuid}  ${mp}  xfs  defaults,noatime,nofail  0  2" >> /etc/fstab
            mount "${mp}"
        done
[vm] # chown -R devai:devai /var/cache/devai
```

`nofail` is mandatory: a thin-LV that fails to activate at boot would
otherwise drop systemd into `emergency.target`, and on a cloud-init
image with `root` locked the emergency console loops on
"Press Enter". With `nofail` the host boots and the operator can SSH
in to diagnose.

### 6.7 Auto-activation across reboots

```bash
[vm] [verify] # vgchange -an vgais && vgchange -ay vgais
[vm] [verify] # lvs vgais -o lv_name,attr | grep -E '^ *cache_' | grep -v 'a.tz' \
                  && { echo FAIL; exit 1; } || echo OK
```

If any cache_* LV shows `Vwi---tz--` instead of `Vwi-aotz--`,
`thin-provisioning-tools` is missing (§4.1) — install and re-test.

### 6.8 Deferred podman probes (from §5.3)

```bash
[vm] [verify] $ podman info --format '{{.Store.GraphRoot}}'    # /var/cache/devai/registry
[vm] [verify] $ podman info --format '{{.Host.OCIRuntime.Name}}'
[vm] [verify] $ podman run --rm docker.io/library/debian:trixie true
[vm] [verify] $ podman run --rm --device nvidia.com/gpu=all \
                  docker.io/library/debian:trixie \
                  nvidia-smi --query-gpu=name --format=csv,noheader
```

The `debian:trixie` pull lands inside `/var/cache/devai/registry` —
proof the graphroot relocation took effect before any image was
materialized.

---

## 7. Clone repo, write `.env`, registry routing

### 7.1 Clone

```bash
[vm] $ git clone https://github.com/<owner>/devai.git ${REPO_DIR}
[vm] $ cd ${REPO_DIR}
```

The canonical clone URL is a `${DEVAI_REPO_URL}` environment variable
provided by the operator at agent start. The agent does not guess.

### 7.2 `.env`

```bash
[vm] $ test -f ${REPO_DIR}/.env || cp ${REPO_DIR}/.env.example ${REPO_DIR}/.env

# D14 — generate JUPYTER_TOKEN if absent
[vm] $ grep -q '^JUPYTER_TOKEN=' ${REPO_DIR}/.env || \
         echo "JUPYTER_TOKEN=$(openssl rand -hex 16)" >> ${REPO_DIR}/.env

# D13 — pin compose image versions explicitly, in case some future
# `podman compose` path bypasses docker-compose v2
[vm] $ cat >> ${REPO_DIR}/.env <<'EOF'
VLLM_IMAGE=docker.io/vllm/vllm-openai:latest-cu130-ubuntu2404
SGLANG_IMAGE=docker.io/lmsysorg/sglang:v0.5.10.post1-cu130
DEVAI_REASONING=auto
GPU_MEMORY_GB=24
MAX_CONTEXT_LEN=131072
EOF
```

If the host's GPU has more or less VRAM than 24 GiB, the operator
overrides `GPU_MEMORY_GB` by setting it in the environment before
running the agent (the agent picks that up and writes the override
into `.env` before §10).

### 7.3 Cache ownership sanity

```bash
[vm] [verify] $ stat -c '%U' /var/cache/devai      # devai
```

---

## 8. Pre-pull images and CLI binaries

### 8.1 Pull infrastructure images

```bash
[vm] $ cd ${REPO_DIR}
[vm] $ make pull-images
```

Targets (see `deploy/docker-compose.yaml` for the source of truth):
`debian:trixie`, `docker.io/nvidia/cuda:12.9.1-cudnn-runtime-
ubuntu24.04`, `docker.io/library/golang:1.23-bookworm`,
`gcr.io/distroless/static-debian12`, `sameersbn/apt-cacher-ng:latest`,
`registry:2`, `ollama/ollama:latest`,
`docker.io/vllm/vllm-openai:latest-cu130-ubuntu2404`,
`docker.io/lmsysorg/sglang:v0.5.10.post1-cu130`,
`ghcr.io/open-webui/open-webui:main`,
`docker.io/library/nginx:alpine`, `quay.io/podman/stable`.

If any pull fails, the agent halts. Re-running `make pull-images` is
idempotent (digest match → no-op).

### 8.2 Fetch CLI binaries

```bash
[vm] $ make fetch-cli
```

Downloads to `/var/cache/devai/pip/bin/` (ETag cached):

- Anthropic Claude Code
- OpenAI Codex
- Ollama CLI
- code-server
- `uv` / `uvx`
- Google Gemini CLI
- LATE

These are bind-mounted read-only into the lab image at build time, so
§9 is offline.

### 8.3 Verification

```bash
[vm] [verify] $ podman images --format '{{.Repository}}:{{.Tag}}' \
                | grep -E '(debian|nvidia/cuda|golang|distroless|apt-cacher|registry|ollama|vllm|sglang|open-webui|nginx|podman/stable)' \
                | sort -u
[vm] [verify] $ ls -1 /var/cache/devai/pip/bin/ \
                | grep -E '^(claude|codex|ollama|code-server|uv|uvx|gemini|late)$'
```

---

## 9. Build images

```bash
[vm] $ cd ${REPO_DIR}
[vm] $ make build           # base-cpu, lab-cpu, base-gpu, lab-gpu, router
```

| Target | Image |
|---|---|
| `build-base-cpu` | `devai-base-cpu` |
| `build-cpu`      | `devai-lab-cpu`  |
| `build-base-gpu` | `devai-base-gpu` |
| `build-gpu`      | `devai-lab-gpu`  |
| `build-router`   | `localhost/devai-router` |

The VM always has a GPU (D9), so the GPU variants always build. The
Makefile bind-mounts `/var/cache/devai/pip`, `/var/cache/devai/npm`,
and `/var/cache/devai/pip/bin` into each build, so the network is
hit only if §8 missed something.

### 9.1 Verification

```bash
[vm] [verify] $ podman images --format '{{.Repository}}:{{.Tag}}' \
                | grep -E '^(devai-(base-cpu|base-gpu|lab-cpu|lab-gpu)|localhost/devai-router):latest$' \
                | sort
```

Expected: five entries.

### 9.2 Recovery

| Symptom | Recovery |
|---|---|
| `fetch-cli` produced the binary but the build complains it is missing | Permissions: `chmod +rX -R /var/cache/devai/pip/bin`. |
| Build OOMs the VM during torch install | Reduce parallel build jobs: `BUILDAH_FORMAT=docker BUILD_ARGS="--jobs 1" make build`. |

---

## 10. Start infrastructure

### 10.1 TLS certificates for `devai-webui-proxy`

`nginx:alpine` has no `openssl` binary, so the entrypoint's
"self-signed fallback" silently fails (the `openssl req` call exits
with "command not found", swallowed by `2>/dev/null`). nginx then
loops on "cannot load certificate" and the container restarts
forever. Pre-generate certs **on the VM filesystem** so the
entrypoint takes the "found existing" branch:

```bash
[vm] $ SSL_DIR=${HOME}/devai-home/.jupyter/ssl
[vm] $ mkdir -p "${SSL_DIR}"
[vm] $ test -f "${SSL_DIR}/cert.pem" || openssl req -x509 -nodes -days 365 \
            -newkey rsa:2048 \
            -keyout "${SSL_DIR}/key.pem" -out "${SSL_DIR}/cert.pem" \
            -subj "/CN=devai-webui"
[vm] $ chmod 600 "${SSL_DIR}/key.pem"
[vm] [verify] $ ls "${SSL_DIR}"/cert.pem "${SSL_DIR}"/key.pem
```

### 10.2 Bring the stack up

```bash
[vm] $ make cache-up
```

This:

- Ensures `devai`'s podman socket is enabled (idempotent).
- Creates the `devai-net` network if missing.
- Runs `podman compose -f deploy/docker-compose.yaml up -d` (via
  `docker-compose` v2 — D13), starting `devai-apt-cache`,
  `devai-registry-cache`, `devai-ollama`, `devai-vllm` (placeholder),
  `devai-sglang` (placeholder), `devai-router`, `devai-open-webui`,
  `devai-webui-proxy`, `devai-logger`.

`devai-vllm` and `devai-sglang` launch with `entrypoint: ["sleep",
"infinity"]` and do not serve traffic; the router replaces each
container on demand when a request first hits ports 11435/11436.

### 10.3 Verification

```bash
[vm] [verify] $ podman ps --format '{{.Names}}\t{{.Status}}' | grep -E '^devai-' | sort
[vm] [verify] $ curl -fsS http://localhost:11434/v1/models
[vm] [verify] $ curl -k -fsS https://localhost:8443/
```

The probe-cache files in `deploy/.*-reasoning-cache.json` may not
exist on first `cache-up` — that's fine, §11 populates them.

### 10.4 Recovery

| Symptom | Recovery |
|---|---|
| `Error: name <devai-X> already in use` | `make cache-down && make cache-up`. |
| Router container exits immediately | `podman logs devai-router` — most common cause is missing podman socket; re-run `systemctl --user enable --now podman.socket`. |
| `ollama` repeatedly OOMs at startup | Set `OLLAMA_KEEP_ALIVE=10s`; if the GPU has < 8 GiB free, expect probes to fail. |

---

## 11. Pull and probe initial model

### 11.1 Default first model (D16)

```bash
[vm] $ make model-pull FAMILY=qwen3.5
[vm] $ make probe
```

`make model-pull` reads `deploy/models.yaml` and pulls every fitting
variant of `qwen3.5` for the host's `(VRAM, context)` matrix.

`make probe` exercises every `(VRAM band, context tier, backend)`
cell and writes the result to `deploy/.ollama-reasoning-cache.json`.
First run takes 5–15 minutes per model.

### 11.2 vLLM NVFP4 probe (mandatory)

The vLLM probe pulls and exercises
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` end-to-end. The repo
is public on HuggingFace (NVIDIA Nemotron Open Model License) -- no
licence click-through, no HF token required. If §2.5 happened to
inject one anyway, it's used to raise anonymous rate limits but
isn't load-bearing.

```bash
[vm] $ pip install --user 'huggingface-hub[cli]'
[vm] $ make model-pull FAMILY=NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4
[vm] $ make cache-down
[vm] $ make probe-vllm
[vm] $ make cache-up
```

`make probe-vllm` needs exclusive GPU access -- hence the
`cache-down` first. It writes
`deploy/.vllm-reasoning-cache.json` with the model's
`(fits, reasoning_parser, tool_parser, disable_verified)` row at
every fitting context tier; the router and picker pick the entries
up on the next request, no rebuild.

SGLang is **not** probed by this procedure (D16). The
`devai-sglang` placeholder stays in the compose stack for runtime
use but is never exercised by INSTALL.md.

### 11.3 Real end-to-end agent test

Final correctness gate. The agent drives `devai-agent` itself with
its non-interactive options (D17): `--model`, `--agent claude`,
`--workdir ${REPO_DIR}` (mounted as `/home/devai/work` inside the
container), and `--prompt "<question>"` for one-shot Q&A. Three
questions about the repo are asked; each answer is grepped for an
expected substring.

```bash
# Step 1 -- install the launcher and dummy Claude credentials so the
# e2e test does not depend on §12 having run yet. Both steps are
# idempotent; §12 re-runs them as no-ops.
[vm] $ cd ${REPO_DIR}
[vm] $ make install
[vm] $ devai-agent --init
[vm] $ TARGET=${HOME}/devai-home/.claude/.credentials.json
[vm] $ mkdir -p "$(dirname "${TARGET}")"
[vm] $ python3 - "${TARGET}" <<'PY'
import json, pathlib, sys
prefix = "sk-ant-DUMMY_LOCAL_DEVAI_NOT_VALID_FOR_REAL_API_USE_"
dummy = {"claudeAiOauth": {
    "accessToken":      prefix + "A" * 56,
    "refreshToken":     prefix + "B" * 56,
    "expiresAt":        4070908800000,
    "scopes": ["user:file_upload","user:inference",
               "user:mcp_servers","user:profile",
               "user:sessions:claude_code"],
    "subscriptionType": "max",
    "rateLimitTier":    "default_claude_max_20x"}}
target = pathlib.Path(sys.argv[1])
target.write_text(json.dumps(dummy, indent=2))
target.chmod(0o600)
PY

# Step 2 -- warm the vLLM container so the first claude -p call does
# not race the 10-minute NVFP4 cold-start timeout. `devai-agent
# --show` prints the constructed podman command without launching;
# we still rely on direct curl to actually warm the backend because
# --show is dry-run only.
[vm] $ MODEL='NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4@131072'
[vm] $ devai-agent --model "${MODEL}" --agent claude --show     # sanity-print
[vm] $ curl -fsS http://localhost:11435/v1/chat/completions \
            -H 'content-type: application/json' \
            -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"stream\":false,\"max_tokens\":4}" \
            > /dev/null

# Step 3 -- three Q&A pairs. devai-agent runs claude inside the lab
# container with the repo mounted at /home/devai/work, the router
# pinned via ANTHROPIC_BASE_URL by the in-container picker, and the
# answer printed to stdout. Each expected substring must appear in
# the answer (case-insensitive grep).
[vm] $ run_q () {
            local q="$1" expected="$2"
            local answer
            answer=$(devai-agent \
                --workdir "${REPO_DIR}" \
                --model "${MODEL}" \
                --agent claude \
                --prompt "$q")
            printf 'Q: %s\nA: %s\n' "$q" "$answer"
            if printf '%s' "$answer" | grep -qiF -- "$expected"; then
                printf 'PASS (saw "%s")\n\n' "$expected"
            else
                printf 'FAIL (missing "%s")\n' "$expected"
                return 1
            fi
        }

[vm] $ capture_q () {
            # Same launcher invocation as run_q but returns the raw answer on
            # stdout without doing a substring check; used for the Q4 summary
            # whose validation needs the full text.
            devai-agent \
                --workdir "${REPO_DIR}" \
                --model "${MODEL}" \
                --agent claude \
                --prompt "$1"
        }

[vm] [verify] $ run_q \
        "Read CLAUDE.md and tell me the relative path of the Go source file that implements the GPU arbiter router daemon. Reply with only the path." \
        "gpu-arbiter/main.go"

[vm] [verify] $ run_q \
        "Read CLAUDE.md and tell me which inference backend the router uses to serve NVFP4 safetensors weights. Reply with only the backend name in lowercase." \
        "vllm"

[vm] [verify] $ run_q \
        "Read CLAUDE.md and tell me the host port the router exposes for the vLLM backend. Reply with only the port number." \
        "11435"

# Q4 -- complex: write a full repo summary, then verify all its claims.
# The verification has two layers:
#   (a) a keyword sweep on the script side -- any correct summary of this
#       repo MUST mention these core terms; if it doesn't, the model is
#       either generating filler or has hallucinated a different project;
#   (b) an LLM-as-judge pass that re-reads the actual files and is told
#       to emit a fixed verification token only if every claim grounds
#       in those files. Anything else fails the grep.

[vm] $ SUMMARY=$(capture_q "$(cat <<'PROMPT'
You are operating inside a freshly cloned copy of the 'devai' (Dev AI
Lab) repository, mounted at your current working directory. Use the
Read tool to load CLAUDE.md, README.md (if present), the top-level
Makefile, deploy/docker-compose.yaml, gpu-arbiter/main.go (just the
first 80 lines), and scripts/model-picker.py (just the header).

Then write a 250-400 word factual summary of the project covering:
  1. What the project is and who runs it.
  2. The three inference backends (Ollama, vLLM, SGLang), how they
     are reached via the router's host ports, and the GPU mutual-
     exclusion invariant the router enforces.
  3. The lab images (devai-lab-cpu / devai-lab-gpu) and the two-
     layer base+lab build pattern.
  4. The on-disk cache layout under /var/cache/devai/ and the
     thin-pool / LV scheme.
  5. The role of model-picker.py and the devai-agent launcher.

Constraints: stick to claims you can verify from the files you read.
Do NOT speculate about missing details. Do NOT invent file paths,
ports, or container names. Use plain prose, no bullet lists, no
markdown headings, no code fences -- one continuous block of text.
PROMPT
)")
[vm] $ printf '=== generated summary ===\n%s\n=========================\n' "${SUMMARY}"

# Verification layer (a) -- required-content sweep.
[vm] [verify] $ for kw in router ollama vllm sglang NVFP4 podman JupyterLab GPU \
                          gpu-arbiter cachepool 11434 11435 11436 devai-net; do
                    if ! printf '%s' "${SUMMARY}" | grep -qiF -- "${kw}"; then
                        printf 'FAIL: summary missing required keyword "%s"\n' "${kw}"
                        exit 1
                    fi
                done
                printf 'PASS: summary covers all required keywords\n\n'

# Verification layer (b) -- LLM-as-judge fact-check. Write the summary to
# a file the verifier model can read with its Read tool; we don't try to
# stuff multi-KB of text through shell quoting.
[vm] $ SUMMARY_FILE="${REPO_DIR}/.devai-e2e-summary.tmp"
[vm] $ printf '%s\n' "${SUMMARY}" > "${SUMMARY_FILE}"

[vm] [verify] $ run_q "$(cat <<'PROMPT'
Inside the current working directory there is a file named
.devai-e2e-summary.tmp containing a summary of this repository
written by an earlier model. Your job is to fact-check it.

Step 1: Read .devai-e2e-summary.tmp.
Step 2: Read CLAUDE.md, README.md (if it exists), the top-level
        Makefile, deploy/docker-compose.yaml, gpu-arbiter/main.go,
        and scripts/model-picker.py.
Step 3: For every factual claim in the summary (project purpose,
        backend names, host ports, container names, file paths,
        cache layout, build flow, picker behaviour), check it
        against the files you read.

If and only if every claim is supported by those files, reply with
exactly the single literal token ALL_CLAIMS_VERIFIED on its own line
and nothing else. If any claim is wrong, unsupported, or invented,
do NOT emit that token; instead list each problematic claim as a
bulleted item with a short reason.
PROMPT
)" "ALL_CLAIMS_VERIFIED"

[vm] $ rm -f "${SUMMARY_FILE}"
```

Four `PASS` lines = `devai-agent`'s non-interactive surface, the
lab image, the router, the vLLM backend, the NVFP4 weights, and
the Claude Code agent are all wired up correctly against the real
repo, and the model can both produce **and self-validate** a
multi-claim summary against the on-disk files. Any `FAIL` halts
the agent; the operator inspects `podman logs devai-router`,
`podman logs devai-vllm`, the captured summary text, and the
verifier's bulleted list of rejected claims to triage.

### 11.4 Verification of the Ollama path

```bash
[vm] [verify] $ test -s ${REPO_DIR}/deploy/.ollama-reasoning-cache.json
[vm] [verify] $ test -s ${REPO_DIR}/deploy/.vllm-reasoning-cache.json
[vm] [verify] $ podman exec devai-ollama ollama list | tail -n +2
[vm] [verify] $ curl -fsS http://localhost:11434/api/chat \
                  -H 'content-type: application/json' \
                  -d '{"model":"qwen3.5:9b-q8_0","messages":[{"role":"user","content":"reply with the single word PONG"}],"stream":false}' \
                  | python3 -c "import sys,json;print(json.load(sys.stdin)['message']['content'][:200])"
```

Expected: `PONG`. If the Ollama model tag differs, use the first
row of `ollama list`.

### 11.5 Recovery

| Symptom | Recovery |
|---|---|
| `make model-pull` fails on HuggingFace 401 | Unexpected for the default public model. Either NVIDIA has changed the repo's gating, or the operator swapped in a gated alternative -- check the model's HF page in a browser, accept any licence shown, and (if needed) put a valid token at `~/.cache/huggingface/token` on the host before re-running. |
| `make probe-vllm` writes `fits=false` for every cell | GPU VRAM is too small for the Nemotron-3-Nano MoE at any context tier on this host. This is a hardware constraint; the install target does not fit. |
| `make probe-vllm` aborts with "router/vllm container running" | Run `make cache-down` first, then `make probe-vllm`, then `make cache-up`. |
| §11.3 `claude -p` returns empty / hangs | The vLLM container is still cold-starting (5--10 min on Blackwell for the first request). Re-run the warm-up curl, then retry the failed `run_q` call. |
| §11.3 PASS but answer text is rambling around the expected substring | Acceptable. The check is a substring grep, not equality. |
| §11.3 Q4 keyword sweep fails on a term that's clearly in the repo | The model produced a thin / partial summary. Re-run `capture_q` once -- NVFP4 sampling is non-deterministic. If it repeats, the model isn't loading CLAUDE.md (check whether the work-dir mount succeeded inside the lab container: `podman exec ... ls /home/devai/work`). |
| §11.3 Q4 judge pass lists hallucinated claims | Real failure mode: the summary contained a wrong fact and the judge correctly caught it. Inspect the judge's bulleted list -- if the listed claims are genuinely wrong, the model under test is misreading CLAUDE.md (could indicate a bad NVFP4 download). Re-run from §11.2 `make probe-vllm`. |
| §11.3 Q4 judge emits `ALL_CLAIMS_VERIFIED` plus extra text | Treated as PASS by the grep, but worth eyeballing the extra text once. The instruction says "nothing else"; a model that adds prose anyway is otherwise functional but slightly off-instruction. |

---

## 12. Install `devai-agent` launcher

§11.3 already ran `make install` and `devai-agent --init`. This
phase re-runs them defensively (idempotent: re-running `make
install` replaces the symlinks, `devai-agent --init` resets
`~/.devai/preferences.yaml` to defaults) and adds the PATH export
in case the operator's shell rc didn't pick up `~/.local/bin`.

```bash
[vm] $ cd ${REPO_DIR}
[vm] $ make install
[vm] $ grep -q '\.local/bin' ~/.bashrc || \
         echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
[vm] $ devai-agent --init
```

`make install` symlinks `${REPO_DIR}/bin/devai-agent` to
`~/.local/bin/devai-agent` and the picker + probe caches to `~/.devai/`.

The launcher supports both interactive and non-interactive use:

- `devai-agent` (no args) -- fzf picker, then interactive shell.
- `devai-agent --model <name> --agent <id>` -- skip the picker but
  still start an interactive session.
- `devai-agent --model <name> --agent claude --prompt "<text>"` --
  one-shot: pipe the prompt to `claude -p`, print its stdout, exit
  with the agent's return code. No TTY, no picker, no shell.
- `devai-agent --show` -- dry-run the podman command without
  launching.

The §11.3 e2e test uses the `--prompt` form (D17).
`devai-agent --init` writes a default `~/.devai/preferences.yaml`.

### 12.1 Verification

```bash
[vm] [verify] $ command -v devai-agent
[vm] [verify] $ devai-agent --show
```

`--show` must print the resolved preferences and the constructed
`podman run` command without launching anything.

### 12.2 Claude Code dummy credentials (mandatory if `--agent claude`)

```bash
[vm] $ TARGET=${HOME}/devai-home/.claude/.credentials.json
[vm] $ mkdir -p "$(dirname "${TARGET}")"
[vm] $ python3 - "${TARGET}" <<'PY'
import json, pathlib, sys
prefix = "sk-ant-DUMMY_LOCAL_DEVAI_NOT_VALID_FOR_REAL_API_USE_"
dummy = {
    "claudeAiOauth": {
        "accessToken":      prefix + "A" * 56,
        "refreshToken":     prefix + "B" * 56,
        "expiresAt":        4070908800000,
        "scopes": [
            "user:file_upload", "user:inference",
            "user:mcp_servers", "user:profile",
            "user:sessions:claude_code",
        ],
        "subscriptionType": "max",
        "rateLimitTier":    "default_claude_max_20x",
    }
}
target = pathlib.Path(sys.argv[1])
target.write_text(json.dumps(dummy, indent=2))
target.chmod(0o600)
print("wrote", target, "mode 0600")
PY
[vm] [verify] $ test -f ${HOME}/devai-home/.claude/.credentials.json
[vm] [verify] $ stat -c '%a' ${HOME}/devai-home/.claude/.credentials.json     # 600
```

The token strings are cryptographically invalid; the picker pins
`ANTHROPIC_BASE_URL` to the local router so no cloud call is ever
attempted. The file's **presence** suppresses Claude Code's
hardcoded `claude-haiku-4-5-20251001` startup probe that would
otherwise phantom-launch a vLLM container and hang the session for
10 minutes.

This file is per-user, per-VM, mode 0600. Never copy a real
credentials file into a shared VM.

---

## 13. Auto-start at boot (optional)

```bash
[vm] $ make install-systemd
```

This stages `deploy/docker-compose.yaml` and `deploy/registry-config.yaml`
to `~/.config/devai/`, installs `deploy/systemd/devai-infra.service`
as a `--user` unit, enables `podman.socket` and
`devai-infra.service`, and calls `loginctl enable-linger devai` (a
no-op if §4.6 already ran).

```bash
[vm] [verify] $ systemctl --user is-enabled devai-infra.service
[vm] [verify] $ systemctl --user is-active  devai-infra.service
```

The agent runs §13 unconditionally because the VM is dedicated to
Dev AI Lab. If the operator overrides via `DEVAI_NO_SYSTEMD=1`, the
agent skips this section.

---

## 14. End-to-end smoke test

The agent has already passed the real e2e gate in §11.3. This phase
is a final sanity-check that the post-§12 / §13 system surfaces are
all up.

```bash
[vm] $ make test-router
[vm] $ make test-ollama
```

```bash
[vm] [verify] $ curl -fsS http://localhost:11434/api/chat \
                  -H 'content-type: application/json' \
                  -d '{"model":"qwen3.5:9b-q8_0","messages":[{"role":"user","content":"reply with the single word PONG"}],"stream":false}'
[vm] [verify] $ curl -k -fsS https://localhost:8443/      | head -c 200
[vm] [verify] $ curl -fsS http://localhost:8888/ -o /dev/null && echo lab_up
```

A green smoke test plus four §11.3 PASS lines (Q1, Q2, Q3, plus Q4's
keyword sweep and judge pass each printing PASS) means the agent has
finished correctly.

For a manual external test from the host: `ssh -L
8888:localhost:8888 -L 11434:localhost:11434 -L 8443:localhost:8443
-i ~/.ssh/devai-vm devai@${VM_IP}` then hit
`http://localhost:8888/` etc. from a browser on the host.

The full `make test` suite (~30–60 minutes) runs `cache-down` for
probe smokes and then `cache-up` again. Run only for exhaustive
validation.

---

## 15. State-of-the-world reference

### 15.1 Filesystem map inside the VM

```
/var/cache/devai/                       (all backed by vgais thin LVs)
├── apt/             ← cache_apt
├── npm/             ← cache_npm
├── pip/             ← cache_pip (uv cache + CLI binaries in pip/bin/)
├── registry/        ← cache_registry (podman graphroot + registry:2 mirror)
├── ollama/          ← cache_ollama
│   └── models/
│       ├── blobs/         (Ollama GGUF + manifests)
│       ├── manifests/
│       └── vllm/          (HF safetensors per repo dir)
├── open-webui/      ← cache_open_webui
└── logs/            ← cache_logs

${REPO_DIR}/
├── .env
├── deploy/
│   ├── .ollama-reasoning-cache.json
│   ├── .vllm-reasoning-cache.json
│   ├── .sglang-reasoning-cache.json
│   ├── .bench-cache.json
│   ├── docker-compose.yaml
│   ├── models.yaml
│   ├── registries.conf
│   └── registry-config.yaml
└── scripts/

${HOME}/                                (= /home/devai)
├── .config/containers/{registries.conf,storage.conf,containers.conf}
├── .config/devai/                      (systemd-managed copies, §13)
├── .config/systemd/user/devai-infra.service
├── .devai/                             (devai-agent state + cache symlinks)
├── .local/bin/devai-agent
├── .cache/huggingface/token            (only if D15 fired)
└── devai-home/                         (lab container's persistent home)
```

### 15.2 Network

The VM lives on `192.168.122.0/24` (libvirt `default` NAT) at
`${VM_IP}`. All containers inside the VM share `devai-net`:

```
host[clients] ── NAT ── vm[${VM_IP}]:11434 ─► devai-router:11434 ─► devai-ollama:11434
                                       :11435 ─► devai-router:11435 ─► devai-vllm:11434
                                       :11436 ─► devai-router:11436 ─► devai-sglang:11434
                                       :8443  ─► devai-webui-proxy:443 ─► devai-open-webui:8080 ─► devai-router:11434
                                       :8888  ─► devai-lab-gpu:8888
```

Only those VM ports are exposed; backends are reachable only from
inside `devai-net`. Whether the host's IP exposes the VM ports to the
LAN/internet is the operator's responsibility (D18).

### 15.3 GPU mutual exclusion

Only one of `devai-ollama`, `devai-vllm`, `devai-sglang` holds the
passed-through GPU at a time. The router (`gpu-arbiter/main.go`)
enforces this by draining the active backend before recreating
another. Talking directly to any backend bypasses the invariant — use
only the router ports (`11434/11435/11436`).

### 15.4 Data classes and what is safe to delete

| Path | Class | Safe to delete? |
|---|---|---|
| `/var/cache/devai/registry/docker/` | regenerable mirror cache | yes |
| `/var/cache/devai/apt/` / `pip/` / `npm/` | regenerable build caches | yes (forces full rebuild) |
| `/var/cache/devai/ollama/models/blobs/` | downloaded GGUFs | costly to re-pull |
| `/var/cache/devai/ollama/models/vllm/` | downloaded NVFP4 weights | costly to re-pull |
| `/var/cache/devai/open-webui/` | chat history & users | preserve unless intentional reset |
| `/var/cache/devai/logs/` | container stdout logs | yes |
| `${REPO_DIR}/deploy/.*-reasoning-cache.json` | probe results | yes (next probe regenerates) |
| `${REPO_DIR}/deploy/.bench-cache.json` | bench scores | yes |

`/var/cache/devai/registry/` is **never** safe to wipe wholesale
(shared with podman graphroot — see `make cache-clean`).

---

## 16. Tear-down

The agent's tear-down owns the VM lifecycle. It does **not** revert
the host's VFIO setup — that lives in `HOST_VFIO_SETUP.md` §10.

### 16.1 Inside-VM cleanup (optional)

If the operator wants to remove the in-VM state before destroying
the domain (rare; usually the whole VM is just deleted):

```bash
[vm] $ make cache-down
[vm] $ make clean
[vm] $ make uninstall
[vm] # systemctl --user disable --now devai-infra.service || true
[vm] # rm -f ~/.config/systemd/user/devai-infra.service
[vm] # rm -rf ~/.config/devai ~/.devai
```

### 16.2 Destroy the VM (D19)

```bash
[host] # virsh destroy   devai-vm 2>/dev/null || true
[host] # virsh undefine  devai-vm --nvram --remove-all-storage
```

`--remove-all-storage` deletes the qcow2 disk and the seed ISO.
The Debian base cloud image (`debian-13-generic-amd64.qcow2`) is
preserved for re-provisioning.

### 16.3 Reattach the GPU to nvidia (D9)

```bash
[host] # virsh nodedev-reattach "${DGPU_NODE}"
[host] # virsh nodedev-reattach "${DGPU_AUDIO_NODE}"

[host] [verify] $ lspci -nnk -s "${DGPU_BDF}" | grep -q 'Kernel driver in use: nvidia'
[host] [verify] $ nvidia-smi -L
```

After this the host is back to its idle state per
`HOST_VFIO_SETUP.md` §9.

### 16.4 What is **not** torn down

- The cached cloud image at `${POOL_PATH}/debian-13-generic-amd64.qcow2`.
- The SSH keypair `~/.ssh/devai-vm{,.pub}`.
- Anything in `HOST_VFIO_SETUP.md` §1–§7 (IOMMU, vfio modules,
  libvirt, default net/pool).

Re-running the agent re-creates a fresh `devai-vm` against the same
cached base, in roughly the time it takes to run §3–§14.

---

## 17. Decisions left to the operator

Almost everything is now pre-decided (see "Decisions recorded" at the
top). The remaining operator inputs are:

1. **Reboot of the host** when the §1 gate fails because IOMMU or
   VFIO modules are missing. The agent prints the hint and stops.
2. **Slot move** when §1 (C13) reports a polluted IOMMU group. This
   is a physical change to the host.
3. **HuggingFace token** (D15). Optional. The default §11.2 model
   (`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`) is public and
   downloads anonymously. The operator only needs to provide a
   token if they swap §11.2 in for a gated model.
4. **Override env vars** if the recorded defaults don't fit:
   `DEVAI_VM_RECREATE`, `DEVAI_VM_NAME`, `DEVAI_REPO_URL`,
   `GPU_MEMORY_GB`, `DEVAI_NO_SYSTEMD`.
5. **Public exposure of VM services** (D18) — operator-owned reverse
   proxy + auth layer, the agent does nothing here.

For everything else, the agent decides per the procedure above.

---

## 18. Validating this procedure

This document is "doc as test": if a tier-1 coding agent reading
INSTALL.md cannot produce a working VM, the doc is the bug.

Validation method: an operator (or driver agent) runs the procedure
end-to-end against a fresh, just-prepared host (per
`HOST_VFIO_SETUP.md` §1–§7) and confirms both the §11.3 real e2e
agent test and the §14 smoke test pass.
There is no in-repo automated harness — that abstraction was removed
because the failure modes it hid (env-var substitution, dummy claude
credentials, LVM auto-activation, fstab `nofail`, etc.) were exactly
the gaps this doc itself needed to close. Each of those is called
out explicitly in the relevant phase.

---

## 19. Versioning of this file

This document tracks the repository at the time of writing. When
`deploy/docker-compose.yaml`, `deploy/Dockerfile.{base,lab,router}`,
`Makefile`, or the `.env.example` shape changes, regenerate the
relevant section:

- §8.1 image list — keep in sync with `podman compose -f
  deploy/docker-compose.yaml config --images`.
- §6.1 volume table — keep in sync with the storage section of
  `README.md` and `deploy/setup-logs-volume.sh`.
- §7.2 `.env` keys — keep in sync with `.env.example` and the `?=`
  defaults at the top of `Makefile`.
- §11.x model-pull / probe targets — keep in sync with the `MODELS`
  block in `make help`.

The agent should treat this file as authoritative for **bootstrap**;
for runtime architecture, defer to `docs/router.md`,
`docs/backends.md`, `CLAUDE.md`, and the in-repo `README.md`.
