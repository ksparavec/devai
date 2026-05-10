# INSTALL.md — Bootstrap procedure for AI agents

This document is the single, self-contained installation procedure for
**Dev AI Lab** on **Linux**. It is written for an autonomous coding
agent (Claude Code, OpenAI Codex, or equivalent tier-1 agent) to read
and execute end-to-end on a fresh host with **no human interaction**
beyond granting `sudo` once and (optionally) supplying API keys. A
human operator can also follow it, but the phrasing, verification
commands, and idempotency rules are tuned for an agent.

The repository deliberately does NOT ship a monolithic Bash or Make
scaffold for host bootstrap. Instead, the agent reads this file and
issues one shell command at a time. Each command has a **purpose**, a
**precondition**, an **idempotency rule**, and a **verification**
command. If a verification fails, the agent must follow the per-section
recovery rule rather than retrying blindly.

---

## Entry point — read this first

**Every agent starts from this file regardless of platform.** Before
executing any command in this document, run the following platform
check:

```bash
[verify] $ uname -s
```

- If the result is **`Darwin`** (macOS), **stop reading this file**
  and switch to [`INSTALL_macOS.md`](INSTALL_macOS.md). That document
  is the complete macOS procedure: it covers Apple Silicon's unified
  memory architecture, native Ollama via Metal, `podman machine`,
  and the macOS-specific compose override. The Linux phases below do
  not apply on macOS — applying them produces a working but slow
  stack that wastes the project's GPU-arbitration plumbing.
- If the result is **`Linux`**, continue with §0 below. The
  remainder of this file covers Debian/Ubuntu, RHEL/Fedora/Rocky/
  Alma, openSUSE/SLES, Arch/EndeavourOS, and (best-effort) Alpine.
- Anything else (`FreeBSD`, `OpenBSD`, `SunOS`, …) is unsupported —
  halt and surface to the operator.

The redirect is one-way: `INSTALL_macOS.md` is self-contained and
does not refer back here for installation steps.

---

## 0. Contract

### 0.1 Goal

Take a host that has only:

- One of the supported Linux distributions in §0.5, with rootless
  container support.
- Local administrative privileges (`sudo`).
- Optionally an NVIDIA GPU (Blackwell, Ada, or Ampere class) on
  `x86_64`, with the matching kernel driver already installed.
- Network reachability to the package manager's repositories,
  `docker.io`, `huggingface.co`, `registry.ollama.ai`, `github.com`,
  `storage.googleapis.com`, `astral.sh`, `npmjs.org`, `nodejs.org`.

…and produce a host that runs:

- The `devai-router` plus `devai-ollama`, `devai-vllm` (placeholder),
  `devai-sglang` (placeholder), `devai-open-webui`, `devai-webui-proxy`,
  `devai-apt-cache`, `devai-registry-cache`, `devai-logger` services.
- The `devai-lab-cpu` and (if GPU is present) `devai-lab-gpu` images
  built and tagged locally.
- A populated probe cache for at least one fitting model on at least
  one backend.
- A `devai-agent` launcher on the user's `PATH`.

### 0.2 Constraints

1. **No interactive prompts.** Every command runs non-interactively.
   When a tool would prompt by default, pass the equivalent flag
   (`-y`, `--non-interactive`, `--yes`, `<&-`).
2. **Idempotent.** Re-running this procedure top to bottom on an
   already-installed host must not destroy data, change UUIDs,
   re-format filesystems, or evict running containers (except where the
   step explicitly says it does).
3. **Verify before mutating.** Each phase begins with a detection step
   that decides whether work is needed.
4. **Fail loudly, recover narrowly.** On any verification failure, do
   the smallest scoped recovery that the section documents. Do not
   silently retry the same command.
5. **Preserve user data.** Never `rm -rf` or `lvremove` an existing LV
   without an explicit instruction in this file or an explicit human
   command.

### 0.3 Conventions used below

- Commands prefixed with `$` run as the invoking (non-root) user.
- Commands prefixed with `#` run as `root` (use `sudo`).
- Commands prefixed with `[verify]` are read-only checks.
- `${INVOKING_USER}` resolves to `$(id -un)` of the user who
  ultimately runs `devai-agent` and `make`. That same user must own
  `/var/cache/devai/*` and the rootless podman state.
- `${HOST_IP}` resolves to `$(hostname -I | awk '{print $1}')`.
- `${REPO_DIR}` is the absolute path to the cloned `devai` repository.
- `${HOME_DIR}` is `${INVOKING_USER}`'s home (`getent passwd
  ${INVOKING_USER} | cut -d: -f6`).

### 0.4 Phase order

Phases are sequential. Do not start phase N+1 before phase N's exit
verification passes.

| # | Phase | Mutates host? | Network? | Verification |
|---|---|---|---|---|
| 1 | Detect environment | no | no | `[verify]` only |
| 2 | Install host packages | yes (apt) | yes | `dpkg -l` |
| 3 | Configure container runtime + GPU | yes (config files only) | no | CDI list (no podman runs — see §3 ordering note) |
| 4 | Provision LVM cache volumes | yes (lvm/fs) | yes (first podman pull) | `findmnt`, `lsblk`, then deferred §3.4 podman probes |
| 5 | Clone repo + write `.env` + registries.conf | yes (files) | yes | `test -f` |
| 6 | Pre-pull base images and CLI binaries | yes (image store, cache) | yes | `podman images`, `ls` |
| 7 | Build images (`make build`) | yes (images) | maybe | `podman images` |
| 8 | Start infrastructure (`make cache-up`) | yes (containers) | yes (model image first time) | `podman ps`, HTTP probe of router |
| 9 | Pull and probe initial model | yes (model store, caches) | yes | probe-cache file content |
| 10 | Install `devai-agent` launcher | yes (`~/.local/bin`) | no | `command -v devai-agent` |
| 11 | (optional) systemd auto-start | yes | no | `systemctl --user is-active` |
| 12 | End-to-end smoke test | yes (one chat completion) | no | HTTP 200 + non-empty body |

If the agent stops mid-procedure and is later resumed, it must
re-execute the detection steps in §1 to determine which phases are
already done; idempotency rules in each phase make re-execution safe.

### 0.5 Platform support matrix

| Platform | Lab + agents | Ollama (CPU) | Ollama (GPU) | vLLM / SGLang | Notes |
|---|---|---|---|---|---|
| Debian 12+ / Ubuntu 22.04+ x86_64 | yes | yes | yes (NVIDIA) | yes (NVIDIA, Blackwell+ for NVFP4) | reference platform; everything in this doc was developed against it |
| Fedora 39+ / RHEL 9+ / Rocky 9+ / Alma 9+ x86_64 | yes | yes | yes (NVIDIA) | yes | needs SELinux exceptions or `--security-opt label=disable` (already in compose) |
| openSUSE Leap 15.5+ / Tumbleweed x86_64 | yes | yes | yes (NVIDIA) | yes | zypper paths in §2 |
| Arch / EndeavourOS x86_64 | yes | yes | yes (NVIDIA) | yes | rolling; pin nothing, accept upgrades |
| Debian/Ubuntu/Fedora aarch64 (no NVIDIA) | yes | yes | no | no | CPU lab only; some images lack arm64 manifests — check `podman pull` results |
| Alpine Linux | best-effort | yes | yes (with manual NVIDIA setup) | partial | musl + rootless podman work; NVIDIA Container Toolkit packaging is community-maintained — verify before relying on it |
| macOS 13+ (Apple Silicon or Intel) | — | — | — | — | **Use [`INSTALL_macOS.md`](INSTALL_macOS.md), not this file.** Different architecture (host-native Ollama with Metal/UMA + lab in `podman machine`). |
| Windows / WSL2 | not in scope here | — | — | — | technically possible (WSL2 + nvidia-cdi-hook); not part of this procedure |

When the agent identifies a platform that is not in this matrix and
not obviously close to one that is, it must halt and surface the OS,
release, and architecture to the operator rather than improvising.

The rest of this document uses **`${OS_FAMILY}`** to abbreviate the
detected platform group:

- `debian`   — Debian, Ubuntu, Pop!_OS, Mint Debian Edition
- `rhel`     — RHEL, Fedora, CentOS Stream, Rocky, AlmaLinux, Amazon Linux 2023
- `suse`     — openSUSE Leap, openSUSE Tumbleweed, SLES
- `arch`     — Arch, EndeavourOS, Manjaro
- `alpine`   — Alpine Linux 3.19+ (best-effort)

(macOS is excluded from this list — it has its own document.)

### 0.6 Execution context — local agent or remote agent

This document describes work performed on a **target machine**. The
agent executing the procedure may run in either of two execution
contexts:

- **Local** — the agent's shell **is** the target's shell. Every
  `sudo apt-get install`, `lvcreate`, `podman build`, etc. operates
  in-place. This is the natural mode for an operator setting up Dev
  AI Lab on their own workstation.
- **Remote** — the agent runs on a separate **control host** and
  issues every install command against the target via SSH. The agent
  does not log into an interactive session; each command is a single
  `ssh user@target 'command'` invocation, return-coded, with stdout
  and stderr captured back to the agent's transport. Reboots
  disconnect SSH transiently; the agent waits for SSH to come back
  and continues. The typical remote-mode target is a
  freshly-provisioned libvirt VM with a dedicated GPU passed through
  via VFIO — see [`docs/HOST_VFIO_SETUP.md`](docs/HOST_VFIO_SETUP.md)
  for the host-side preparation.

INSTALL.md is **identical in both modes** — the work to be done on
the target is the same. Each command shown in this document is to be
read as "execute this on the target." The agent translates that to
either bare-shell (local) or SSH-prefixed (remote) before invocation.

In **remote mode**, the agent must obey these conventions:

- **Every** command from this document targets the remote machine.
  The agent's `bash` tool, when used unprefixed, runs on the
  control host — which is **not** what INSTALL.md is talking about.
  Always prefix with `ssh -i <key> -o StrictHostKeyChecking=no
  <user>@<target> 'command'`. The only legitimate uses of
  unprefixed bash on the control host are reading this document,
  scp/rsync of files into the target, polling SSH availability, and
  inspecting harness state (transcripts, logs).
- `sudo` runs as root **on the target** — the SSH user is in
  passwordless sudoers.
- Reboots: when a phase requires reboot (§2.1 NVIDIA driver, §3
  kernel IOMMU on bare metal), the agent issues
  `ssh ... 'sudo systemctl reboot'`, then loops `until ssh -o
  ConnectTimeout=2 ... true; do sleep 5; done` before resuming. No
  human waits between reboots.
- This document does **not** cover host-side preparation for the
  remote-mode target. Provisioning the VM, configuring GPU
  passthrough on the control host, and dispatching the agent are
  the operator's responsibility — see
  [`docs/HOST_VFIO_SETUP.md`](docs/HOST_VFIO_SETUP.md). When the
  agent enters this document in remote mode, the target is already
  booted, SSH-reachable, and has the GPU attached via VFIO.

When this document says "the agent decides" or "the operator
confirms", that authority is the agent regardless of execution
mode — only the *transport* of commands changes.

---

## 1. Phase 1 — Detect environment

The agent must record these facts before mutating anything. They drive
later decisions (CPU vs GPU build, which volumes already exist, whether
a thin pool needs creating).

### 1.1 Platform identification

The entry-point check (top of file) already confirmed `uname -s` is
`Linux`. Now identify the distribution:

```bash
[verify] $ uname -m              # x86_64 | aarch64
[verify] $ uname -r
[verify] $ . /etc/os-release && echo "${ID} ${ID_LIKE:-} ${VERSION_ID}"
```

Map `ID` (and `ID_LIKE` as a fallback) to `${OS_FAMILY}`:

| `ID` / `ID_LIKE` contains | `${OS_FAMILY}` |
|---|---|
| `debian`, `ubuntu`, `linuxmint`, `pop` | `debian` |
| `rhel`, `fedora`, `centos`, `rocky`, `almalinux`, `amzn` | `rhel` |
| `opensuse`, `sles`, `suse` | `suse` |
| `arch`, `manjaro`, `endeavouros` | `arch` |
| `alpine` | `alpine` |
| anything else | unsupported — halt |

Cross-reference §0.5 to confirm the detected platform is supported.
Best-effort platforms (Alpine, exotic distros) require the operator
to opt in explicitly before the agent proceeds.

### 1.2 GPU presence

```bash
[verify] $ test -e /dev/nvidia0 && echo gpu_present || echo gpu_absent
[verify] $ command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
```

Set `${HAS_GPU}=1` only if **all** of the following hold:

- `uname -m` is `x86_64`.
- `/dev/nvidia0` exists AND `nvidia-smi` runs.

Otherwise `${HAS_GPU}=0`. On Linux aarch64, always `${HAS_GPU}=0` —
the NVIDIA Container Toolkit and the vLLM/SGLang container images
do not target that platform. The CPU-only path still produces a
working JupyterLab; only GPU inference is skipped.

If the kernel module is missing but the operator wants GPU, the
agent must install the NVIDIA driver (Phase 2) and reboot before
continuing. The agent must NOT silently reboot the host — it must
surface a "reboot required" message to the operator and stop.

### 1.3 Storage detection

LVM2 thin-provisioned volumes under `/var/cache/devai/` are
**mandatory**. Dev AI Lab does not run on plain directories on the
root filesystem: a runaway model download must not be able to fill
the host's `/`, and per-volume size accounting + isolation
(`make cache-clean`, `make setup-logs`, model storage independent
of build cache) all assume LVM thin pools.

Record:

```bash
[verify] # lvm --version            # confirm LVM2 tooling is present
[verify] # vgs --noheadings -o vg_name,vg_free | sort -u
[verify] # lvs --noheadings -o lv_name,vg_name,segtype,pool_lv | grep -E 'thin-pool|thin' || true
[verify] # findmnt -no SOURCE,TARGET,FSTYPE /var/cache/devai/{ollama,registry,pip,apt,npm,open-webui,logs} 2>/dev/null || true
[verify] # lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT
```

- `${VG_NAME}` — the volume group with enough free space (≥ 600 GB
  recommended) for the cache pool. Default if creating fresh: `vgais`.
  If the host already has a VG with `cachepool` in it, **reuse it**.
- `${POOL_NAME}` — `cachepool` is conventional; reuse if present.
- `${EXISTING_VOLUMES}` — set of `cache_*` LVs already provisioned.
  Do NOT recreate any of them.
- `${EXISTING_MOUNTS}` — `/var/cache/devai/*` mountpoints already
  active. Skip mount steps for these in Phase 4.

If the host has **no LVM at all and no candidate disk/partition** for
the cache pool (e.g. a small VPS with only a single boot partition,
a workstation installed without LVM and no spare disk, Alpine on a
single partition), the agent **must halt and surface to the
operator**. The operator must either:

- Attach a second physical or virtual disk (typical: 600 GB+ for the
  thin pool's physical extent), then re-run from §1; or
- Free a partition on the existing disk and provision LVM on it; or
- Install Dev AI Lab on a different host that meets the prerequisite.

INSTALL.md does not provide a non-LVM fallback path. The trade-offs
of plain directories on root (no per-volume sizing, no `make
setup-logs`, full root-fs at risk of model bloat) are not
acceptable for a project that routinely manages 100+ GB of model
weights and container caches.

### 1.4 User and rootless podman

```bash
[verify] $ id
[verify] $ subuid_count=$(grep "^${INVOKING_USER}:" /etc/subuid | head -n1 | awk -F: '{print $3}')
[verify] $ subgid_count=$(grep "^${INVOKING_USER}:" /etc/subgid | head -n1 | awk -F: '{print $3}')
[verify] $ echo "subuid=${subuid_count} subgid=${subgid_count}"
```

`subuid_count` and `subgid_count` must each be ≥ 65536. Most modern
Linux installs satisfy this automatically when the user was created
through the standard `adduser` / `useradd -m` path. If either is
missing, Phase 2 will fix it via `usermod --add-subuids/--add-subgids`
(or distro-specific equivalents — see §2).

### 1.5 Repo presence

```bash
[verify] $ test -d "${REPO_DIR}/.git" && git -C "${REPO_DIR}" rev-parse --abbrev-ref HEAD
```

If `${REPO_DIR}` is missing, Phase 5 clones it. If present, verify the
remote and branch but do not re-clone.

---

## 2. Phase 2 — Install host packages

This phase is platform-aware. Pick exactly one of §2.1.* matching
`${OS_FAMILY}`, then run §2.2 (subuid/subgid, Linux only), §2.3
(socket / `podman machine`), and §2.4 (verification) — those final
three subsections are platform-aware in turn.

### 2.0 Package responsibilities (cross-distro)

The same logical roles are needed on every platform. The per-platform
sections below resolve each role to a concrete package name.

| Role | Provided by |
|---|---|
| Container engine | `podman` (preferred) or `docker` |
| Compose provider (Go, **mandatory**) | `docker-compose` v2+ (Go binary), installed at `/usr/local/bin/docker-compose`. The Python `podman-compose` 1.3.0 shipped by Debian Trixie/Ubuntu **does not** expand `${VAR:-default}` substitutions used throughout `deploy/docker-compose.yaml` — the router then receives literal strings as image references and fails to recreate the on-demand vLLM/SGLang containers. `podman compose` (a wrapper) defers to `docker-compose` when present, so installing the Go binary fixes the substitution without changing the Makefile. |
| GNU make | `make` |
| Git | `git` |
| Python ≥ 3.11 + YAML + venv | `python3`, `python3-yaml`, `python3-venv`/`python3-pip` (or distro equivalents) |
| Curl + ca-certs + gnupg | `curl`, `ca-certificates`, `gnupg` |
| Rootless container helpers (Linux) | `uidmap`/`shadow-utils-newxidmap`, `fuse-overlayfs`, `slirp4netns` (or `passt`) |
| LVM2 + XFS tools (Linux storage) | `lvm2`, `xfsprogs` |
| LVM thin tooling (**mandatory** on Linux) | `thin-provisioning-tools` (provides `/usr/sbin/thin_check`). Without it, LVM refuses to auto-activate thin LVs at boot — every `cache_*` mount fails silently (see §4.7's `nofail` requirement) and the stack starts with no `/var/cache/devai/*` storage. |
| TLS for `devai-webui-proxy` | `openssl` on the host (alpine `nginx:alpine` ships no openssl binary, so the container's "self-signed fallback" is a no-op; certs must be pre-generated on the host — see §8.0). `mkcert` is optional for browser-trusted certs. |
| (optional, GPU only) NVIDIA kernel driver | distro-specific (see §2.1.*) |
| (optional, GPU only) NVIDIA Container Toolkit | `nvidia-container-toolkit` (every supported distro publishes a package) |

### 2.1.a Debian / Ubuntu (`${OS_FAMILY}=debian`)

Idempotent: `apt-get install -y` is a no-op when packages are present.

```bash
# apt-get update
# apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg git make \
        python3 python3-yaml python3-pip python3-venv \
        podman podman-compose \
        uidmap fuse-overlayfs slirp4netns passt \
        netavark aardvark-dns nftables \
        lvm2 xfsprogs thin-provisioning-tools \
        openssl mkcert
```

Then install **`docker-compose` v2 (Go binary)** as the compose provider —
without it `podman compose` falls through to the Python `podman-compose`
1.3.0 in Debian, which does not expand `${VAR:-default}` and breaks the
router's on-demand vLLM/SGLang recreation:

```bash
# DC_VER=v2.40.0   # any v2.x release
# curl -fsSL "https://github.com/docker/compose/releases/download/${DC_VER}/docker-compose-linux-$(uname -m)" \
        -o /usr/local/bin/docker-compose
# chmod +x /usr/local/bin/docker-compose
[verify] $ docker-compose version       # must print "Docker Compose version vX.Y.Z"
[verify] $ podman compose version       # must print the SAME version (proves the wrapper picked it up)
```

Network-stack notes for podman 5.x rootless on Debian Trixie:

- `passt` (binary `pasta`) is podman's default rootless network
  backend; without it, `podman run` fails with `could not find pasta,
  the network namespace can't be configured`. `slirp4netns` is the
  fallback used on older kernels; installing both lets podman pick
  the one its defaults prefer.
- `netavark` + `aardvark-dns` + `nftables` are required when
  containers join a user-defined network (as compose does — every
  service joins the project's bridge). Without them, `podman start`
  on those containers fails with `netavark: nftables error: unable to
  execute nft` and `aardvark-dns binary not found`. Debian's `podman`
  package only **recommends** these; `--no-install-recommends`
  skips them.

If `${HAS_GPU}=1`, the **driver source matters** for newer GPUs:

- **Debian Trixie's `nvidia-driver`** (currently 550.163.01) supports
  Ampere (RTX 30xx, A-series) and Ada (RTX 40xx, L-series). It
  **does not support Blackwell** (RTX PRO 4000/5000/6000 Blackwell,
  RTX 50xx) — the driver fails its probe with
  `NVRM: ... not supported by the NVIDIA 550.163.01 driver release`
  and `/dev/nvidia0` never appears.
- **NVIDIA's CUDA APT repo** ships latest production drivers
  (570+ at the time of writing) which support Ampere through
  Blackwell. Use this path on Blackwell hosts; either path works
  for older cards.

Path A — Debian's `nvidia-driver` (Ampere/Ada only):

```bash
# Enable contrib + non-free-firmware (Trixie cloud images ship main only)
# sed -i 's|^Components: main$|Components: main contrib non-free-firmware non-free|' \
       /etc/apt/sources.list.d/debian.sources

# NVIDIA Container Toolkit repo
# curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
       | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
# curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
       | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
       > /etc/apt/sources.list.d/nvidia-container-toolkit.list

# apt-get update
# apt-get install -y nvidia-driver nvidia-smi nvidia-container-toolkit \
                     linux-headers-amd64
```

Path B — NVIDIA's CUDA repo (Blackwell or any newer GPU):

```bash
# CUDA repo keyring
# wget -qO /tmp/cuda-keyring.deb \
       https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb
# dpkg -i /tmp/cuda-keyring.deb && rm /tmp/cuda-keyring.deb

# NVIDIA Container Toolkit repo (same as Path A)
# curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
       | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
# curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
       | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
       > /etc/apt/sources.list.d/nvidia-container-toolkit.list

# apt-get update
# apt-get install -y nvidia-open nvidia-container-toolkit linux-headers-amd64
```

**Use `nvidia-open`, not `cuda-drivers`, on Blackwell.** Blackwell
GPUs (PCI ID `10de:2c34` and similar) require NVIDIA's **open
kernel modules**; the closed-source modules pulled in by
`cuda-drivers` will load but fail GPU init with
`NVRM: GPU 0000:XX:00.0: RmInitAdapter failed!` and the dmesg
hint `requires use of the NVIDIA open kernel modules`. The
`nvidia-open` metapackage pulls in `nvidia-kernel-open-dkms`,
which builds the open `nvidia.ko` against the running kernel.

For Ampere/Ada, both `cuda-drivers` (closed) and `nvidia-open`
(open) work; pick `nvidia-open` for forward-compatibility with
future GPU generations.

Why both:

- `linux-headers-amd64` is required for DKMS to build `nvidia.ko`
  against the running kernel; without it the module is missing
  post-reboot and `nvidia-smi` reports "couldn't communicate with
  the NVIDIA driver".
- `nvidia-smi` is its own package on Debian (not pulled in by
  `nvidia-driver`) but is needed by §3.1 verification.

For Ubuntu the same paths apply; only the Debian
`sources.list.d/debian.sources` edit is unnecessary (Ubuntu's
`restricted` and `multiverse` are typically already enabled).

### 2.1.b RHEL / Fedora / Rocky / Alma / Amazon Linux 2023 (`${OS_FAMILY}=rhel`)

```bash
# dnf install -y \
        ca-certificates curl gnupg2 git make \
        python3 python3-pyyaml python3-pip \
        podman podman-compose \
        shadow-utils fuse-overlayfs slirp4netns \
        lvm2 xfsprogs \
        mkcert
```

On RHEL/Rocky/Alma 9, `podman-compose` lives in EPEL —
`dnf install -y epel-release` first if it is missing. Amazon Linux
2023 ships podman without compose; install via PyPI:

```bash
# pip3 install podman-compose
```

If `${HAS_GPU}=1`:

```bash
# dnf config-manager --add-repo \
       https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo
# dnf install -y nvidia-driver nvidia-container-toolkit
```

The exact NVIDIA driver package name varies — Fedora uses
`akmod-nvidia` (RPM Fusion), RHEL uses the CUDA repo's `nvidia-driver`
or `kmod-nvidia-latest-dkms`. Defer to the operator if the
straightforward `dnf install nvidia-driver` does not resolve.

SELinux: the project's compose file already passes
`--security-opt label=disable` to GPU containers because rootless
podman + nvidia-container-toolkit + enforcing SELinux historically
clashes. No additional SELinux work is required for the default
path. If the operator runs in `Enforcing` mode and prefers labels,
they must add the matching `container_t` rules manually.

### 2.1.c openSUSE Leap / Tumbleweed / SLES (`${OS_FAMILY}=suse`)

```bash
# zypper install -y \
        ca-certificates curl gpg2 git make \
        python3 python3-PyYAML python3-pip \
        podman podman-compose \
        shadow fuse-overlayfs slirp4netns \
        lvm2 xfsprogs \
        mkcert
```

If `${HAS_GPU}=1`:

```bash
# zypper addrepo https://download.nvidia.com/opensuse/leap/15.5 NVIDIA
# zypper --gpg-auto-import-keys refresh
# zypper install -y nvidia-driver-G06 nvidia-container-toolkit
```

Adjust `15.5` to the actual Leap version; on Tumbleweed use the
`tumbleweed` path. The NVIDIA Container Toolkit RPM repo
(`https://nvidia.github.io/libnvidia-container/stable/rpm/`) also
works on openSUSE.

### 2.1.d Arch / EndeavourOS / Manjaro (`${OS_FAMILY}=arch`)

```bash
# pacman -Syu --noconfirm \
        ca-certificates curl gnupg git make \
        python python-yaml python-pip \
        podman podman-compose \
        shadow fuse-overlayfs slirp4netns \
        lvm2 xfsprogs \
        mkcert
```

Arch's package names are unversioned (always latest). If
`${HAS_GPU}=1`:

```bash
# pacman -S --noconfirm nvidia nvidia-container-toolkit
```

Use `nvidia-dkms` instead of `nvidia` when running a non-stock kernel
(e.g. `linux-zen`, `linux-lts`).

### 2.1.e Alpine 3.19+ (`${OS_FAMILY}=alpine`, best-effort)

```bash
# apk add --no-cache \
        ca-certificates curl gnupg git make \
        python3 py3-yaml py3-pip \
        podman podman-compose \
        shadow-uidmap fuse-overlayfs slirp4netns \
        lvm2 xfsprogs \
        mkcert
```

The musl + busybox combination is **not** the project's reference
platform. Known caveats:

- The CUDA + NVIDIA Container Toolkit stack on Alpine relies on
  community packaging that lags behind glibc distros; verify with
  `nvidia-ctk --version` before continuing.
- `make probe-vllm` may fail on Alpine if the upstream vLLM image
  was compiled against glibc-only glibc-dependent CUDA bindings;
  this is a vLLM-side limitation, not the project's.

If the operator has not explicitly opted into Alpine, halt and
surface the platform.

### 2.2 subuid/subgid

Skip if §1.4 reported counts ≥ 65536.

```bash
# usermod --add-subuids 100000-165535 ${INVOKING_USER}
# usermod --add-subgids 100000-165535 ${INVOKING_USER}
$ podman system migrate
```

On distros where `usermod` lacks `--add-subuids` (older util-linux),
edit `/etc/subuid` and `/etc/subgid` directly:

```bash
# echo "${INVOKING_USER}:100000:65536" >> /etc/subuid
# echo "${INVOKING_USER}:100000:65536" >> /etc/subgid
$ podman system migrate
```

`podman system migrate` resets the rootless storage to pick up the new
mappings. It is destructive of any existing rootless state, so the
agent must only run it after detecting that the new ranges are
genuinely needed.

### 2.3 Container runtime activation

The router invokes podman remotely to recreate vLLM/SGLang containers,
so the user's podman socket must be active.

```bash
$ systemctl --user enable --now podman.socket
$ loginctl enable-linger ${INVOKING_USER}
```

`loginctl enable-linger` keeps `--user` services running after logout —
required for systemd-managed `devai-infra.service` (Phase 11).

On Alpine and other non-systemd distros, the equivalent is to run
`podman system service --time=0 unix:///run/user/$(id -u)/podman/podman.sock &`
under a long-lived supervisor (OpenRC, runit, s6). The exact unit is
distro-specific — document and surface to the operator if Alpine is
the target.

### 2.4 Verification

```bash
[verify] $ podman --version
[verify] $ podman compose version
[verify] $ python3 --version           # ≥ 3.11
[verify] $ make --version
[verify] $ test -S "/run/user/$(id -u)/podman/podman.sock"
```

---

## 3. Phase 3 — Configure container runtime + GPU

> **Ordering note.** §3 only writes config files and (on GPU hosts)
> generates the CDI spec; it deliberately defers every `podman run`
> probe to §4.8 because §3.2.1's `graphroot = /var/cache/devai/registry`
> requires that mount point to exist. Sequence: §3.1 → §3.2 → §3.2.1
> → §3.3 → §4 → §3.4 (which is now a no-op pointer to §4.8).

### 3.0 Subsection applicability

| Subsection | Linux + GPU | Linux CPU-only |
|---|---|---|
| 3.1 NVIDIA CDI | required | skip |
| 3.2 containers.conf | recommended | recommended |
| 3.3 Registry mirror routing | optional | optional |
| 3.4 Verification | required | partial |

### 3.1 NVIDIA Container Device Interface (CDI)

Skip if `${HAS_GPU}=0`.

The repo's containers reference GPU via the CDI device name
`nvidia.com/gpu=all` (see `deploy/docker-compose.yaml`). CDI must be
generated and stored where rootless podman finds it:

```bash
# nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
[verify] # podman info --format '{{.Host.CgroupVersion}} {{.Host.Security.Rootless}}'
[verify] $ podman run --rm --device nvidia.com/gpu=all \
              docker.io/library/debian:trixie nvidia-smi -L
```

The `nvidia-smi -L` output must list at least one GPU. If it fails
with "no such device", regenerate CDI after confirming `nvidia-smi`
works on the host directly.

### 3.2 Containers config

The repo expects rootless podman to use the user's runtime socket.
Generate (or update) `~/.config/containers/containers.conf` ONLY if it
is missing or lacks the runtime config; never overwrite a populated
file:

```bash
$ test -f ${HOME_DIR}/.config/containers/containers.conf || \
    nvidia-ctk runtime configure --runtime=podman \
        --config=${HOME_DIR}/.config/containers/containers.conf
```

#### 3.2.1 Rootless podman storage relocation (mandatory)

Default rootless graphroot is `${HOME_DIR}/.local/share/containers/storage`,
which lives on the **root filesystem**. The lab images (~30 GB total)
plus pulled NVFP4 / GGUF backend images (~30 GB more) will quickly fill
`/` on any host whose `${HOME_DIR}` is not on a large dedicated volume.
Relocate the graphroot to the LVM-backed `cache_registry` LV **before
running any `podman` command that touches storage** (`pull`, `run`,
`build`, etc.). Once podman has initialised its sqlite state DB at the
default path, switching graphroot requires either an in-place sqlite
edit of `db.sql` (`UPDATE DBConfig SET StaticDir=…, GraphRoot=…,
VolumeDir=…;` plus the same `REPLACE()` over `VolumeConfig.JSON`) or a
destructive `podman system reset --force`. Configuring storage.conf
before the first podman call avoids both.

```bash
$ mkdir -p ${HOME_DIR}/.config/containers
$ cat > ${HOME_DIR}/.config/containers/storage.conf <<EOF
[storage]
driver = "overlay"
graphroot = "/var/cache/devai/registry"
EOF

[verify] $ podman info --format '{{.Store.GraphRoot}}'
# Expected: /var/cache/devai/registry
```

The `cache_registry` LV is created in §4 with this exact path; the
ordering here means storage.conf is in place before §3.4's
`podman run` verification call. Using the same `graphroot` path that
the host uses (the project's standard) keeps any future image transfers
between hosts straightforward.

### 3.3 Registry mirror routing

The infra stack runs a `registry:2` pull-through cache on `:5000`.
Routing rootless podman through it is optional but speeds up
re-builds. Install the supplied config only if no `registries.conf`
exists yet (do not clobber a customised one):

```bash
$ mkdir -p ${HOME_DIR}/.config/containers
$ test -f ${HOME_DIR}/.config/containers/registries.conf || \
      cp ${REPO_DIR}/deploy/registries.conf \
         ${HOME_DIR}/.config/containers/registries.conf
```

The shipped `registries.conf` adds `localhost:5000` as an insecure
mirror for `docker.io`. It falls back to `docker.io` directly if the
mirror is down, so installing it before §8 (when the mirror starts)
is safe.

### 3.4 Verification

Deferred to §4.8 — see the ordering note at the top of §3. The probes
that test `podman run` need `graphroot = /var/cache/devai/registry`
to exist as a mounted directory, which §4 provides.

The non-podman probe (CDI list) is safe to run here:

```bash
[verify] # nvidia-ctk cdi list 2>/dev/null | head        # GPU hosts only
```

---

## 4. Phase 4 — Provision storage for `/var/cache/devai/`

This phase creates seven LVM2 thin-provisioned volumes under
`${VG_NAME}/cachepool`, formats them XFS, and mounts them at
`/var/cache/devai/<name>`. **LVM is mandatory**; if §1.3 reported
no usable VG, the agent has already halted and the operator has
attached a disk or freed a partition before re-entering this phase.

### 4.0 Why LVM thin pool

> One physical extent backs many sparse LVs, so unused capacity is
> shared but accounting is per-LV. Snapshots are cheap. A runaway
> model download fills only its own LV, never the host's `/`. The
> repo's `make cache-clean` and `make setup-logs` targets assume
> this layout verbatim (see `deploy/setup-logs-volume.sh`).

### 4.1 Target layout

| LV name | Size (virtual) | Mountpoint | Filesystem | Owner |
|---|---|---|---|---|
| `cache_ollama` | 200G–500G | `/var/cache/devai/ollama` | xfs | `${INVOKING_USER}` |
| `cache_registry` | 200G | `/var/cache/devai/registry` | xfs | `${INVOKING_USER}` |
| `cache_pip` | 30G | `/var/cache/devai/pip` | xfs | `${INVOKING_USER}` |
| `cache_apt` | 10G | `/var/cache/devai/apt` | xfs | `${INVOKING_USER}` |
| `cache_npm` | 10G | `/var/cache/devai/npm` | xfs | `${INVOKING_USER}` |
| `cache_open_webui` | 5G | `/var/cache/devai/open-webui` | xfs | `${INVOKING_USER}` |
| `cache_logs` | 100G | `/var/cache/devai/logs` | xfs | `${INVOKING_USER}` |

Sum of virtual sizes ≈ 855 GB; the underlying thin pool only needs to
hold what is actually written. Recommended thin-pool physical size:
≥ 600 GB (room for one or two large NVFP4 models plus mirror).

`cache_ollama` is the largest because it holds **all** model weights
(Ollama GGUF blobs under `models/blobs/` AND vLLM/SGLang safetensors
trees under `models/vllm/<repo>/`). A single 30B NVFP4 checkpoint can
be 18–25 GB; plan accordingly.

### 4.2 Decision tree

The agent reads `${VG_NAME}`, `${POOL_NAME}`, and `${EXISTING_VOLUMES}`
from §1.3.

```
if ${VG_NAME} unset (no usable VG):
    → create VG on a free disk or partition (§4.3)
    → create thin pool (§4.4)
elif ${POOL_NAME} present in ${VG_NAME}:
    → reuse the existing pool
else:
    → create thin pool inside ${VG_NAME} (§4.4)

for lv in cache_ollama cache_registry cache_pip cache_apt cache_npm cache_open_webui cache_logs:
    if lv in ${EXISTING_VOLUMES}:
        skip (do not recreate, do not reformat)
    else:
        lvcreate (§4.5)
        mkfs.xfs (§4.6)

for mountpoint:
    if already mounted from the matching device:
        skip
    else:
        ensure /etc/fstab entry, mkdir, mount (§4.7)
```

### 4.3 Volume group (only when one does not already exist)

Skip if `${VG_NAME}` is set.

The agent must NOT pick a disk autonomously. If no VG exists, halt and
surface this to the operator with a list of candidate block devices
(`lsblk -dp -o NAME,SIZE,TYPE,MOUNTPOINTS`). Human approval of the
target device is required.

When the operator picks `${DEVICE}` (e.g. `/dev/nvme1n1` or
`/dev/nvme0n1p4`):

```bash
# pvcreate ${DEVICE}
# vgcreate vgais ${DEVICE}
```

Set `${VG_NAME}=vgais`.

### 4.4 Thin pool

Skip if a `thin-pool` LV already exists in `${VG_NAME}`.

```bash
# lvcreate -L 600G -T ${VG_NAME}/cachepool
```

Set `${POOL_NAME}=cachepool`. The pool size is bounded by free space
in `${VG_NAME}`; pick a value ≥ 1.5x the sum of intended LV virtual
sizes that fit in `vgs --units g --noheadings -o vg_free ${VG_NAME}`.
On a small disk, scale all LV sizes proportionally.

### 4.5 Thin LVs

Idempotency: each `lvcreate` is wrapped in an existence check so a
re-run is a no-op.

```bash
# for spec in \
      "cache_ollama:200G" \
      "cache_registry:200G" \
      "cache_pip:30G" \
      "cache_apt:10G" \
      "cache_npm:10G" \
      "cache_open_webui:5G" \
      "cache_logs:100G"; do
      lv="${spec%%:*}"; size="${spec##*:}"
      if ! lvs "${VG_NAME}/${lv}" >/dev/null 2>&1; then
          lvcreate --thin --virtualsize "${size}" --name "${lv}" \
              "${VG_NAME}/${POOL_NAME}"
      fi
  done
```

### 4.6 Filesystem

Format ONLY when the LV reports no signature:

```bash
# for lv in cache_ollama cache_registry cache_pip cache_apt cache_npm cache_open_webui cache_logs; do
      dev="/dev/${VG_NAME}/${lv}"
      if blkid -p "${dev}" >/dev/null 2>&1; then
          continue
      fi
      mkfs.xfs -q "${dev}"
  done
```

### 4.7 Mountpoints, /etc/fstab, mount

```bash
# mkdir -p /var/cache/devai/{ollama,registry,pip,apt,npm,open-webui,logs}

# for lv in cache_ollama cache_registry cache_pip cache_apt cache_npm cache_open_webui cache_logs; do
      dev="/dev/${VG_NAME}/${lv}"
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
      # If the device is already mounted at the right place, skip
      if findmnt -no SOURCE --mountpoint "${mp}" 2>/dev/null | grep -q "${dev}$"; then
          continue
      fi
      # Replace any stale fstab line for this mountpoint or device
      cp -f /etc/fstab "/etc/fstab.bak.$(date -u +%Y%m%dT%H%M%SZ)"
      sed -i.tmp -e "\#\\s${mp}\\s#d" -e "\#${dev}\\s#d" \
          -e "\#UUID=${uuid}\\s#d" /etc/fstab
      rm -f /etc/fstab.tmp
      echo "UUID=${uuid}  ${mp}  xfs  defaults,noatime,nofail  0  2" >> /etc/fstab
      mount "${mp}"
  done

# chown -R ${INVOKING_USER}:${INVOKING_USER} /var/cache/devai
```

**`nofail` is mandatory.** Without it, a mount that fails at boot (e.g.
LVM thin pool not yet activated because `thin-provisioning-tools` is
missing — see §2.0) drops systemd into `emergency.target`. On a
cloud-init image with the root account locked, the emergency console's
`sulogin` then loops on "Press Enter" with no way to log in. With
`nofail` the failed mount is skipped, the system boots, and the
operator can ssh in to diagnose. This is the difference between a
recoverable fault and an unrecoverable VM.

The repo ships `deploy/setup-logs-volume.sh` which performs the
`cache_logs` step exactly this way and is safe to use instead of the
loop above for that one volume:

```bash
$ cd ${REPO_DIR}
$ make setup-logs        # idempotent, prompts only via sudo
```

`make setup-logs` cannot replace this whole phase because it only
manages `cache_logs`; the other six volumes still need the loop.

### 4.8 Verification

LVM + mounts:

```bash
[verify] # lvs ${VG_NAME} --noheadings -o lv_name,size,segtype,pool_lv | sort
[verify] # findmnt /var/cache/devai/{ollama,registry,pip,apt,npm,open-webui,logs}
[verify] $ stat -c '%U:%G %a %n' /var/cache/devai
```

Expected: every mountpoint resolves to its `/dev/${VG_NAME}/cache_*`
device, mounted xfs, owned by `${INVOKING_USER}`.

**Auto-activation across reboots:** the LVs must come up active on
their own at boot, not require `vgchange -ay`. Confirm:

```bash
[verify] # systemctl reboot       # or, less invasive, simulate:
[verify] # vgchange -an ${VG_NAME} && vgchange -ay ${VG_NAME}
[verify] # lvs ${VG_NAME} -o lv_name,attr | grep -E 'cache_' | grep -v 'a.tz' && \
              echo "FAIL: at least one cache_* LV is not active" || \
              echo "OK: all cache_* LVs are active"
```

If any cache_* LV shows `Vwi---tz--` (no `a` flag) instead of
`Vwi-aotz--`, `thin-provisioning-tools` is missing — install it
(see §2.0) and re-test.

Now run the deferred §3.4 podman probes — `graphroot` from §3.2.1
finally points at a real mount:

```bash
[verify] $ podman info --format '{{.Store.GraphRoot}}'
# must equal /var/cache/devai/registry
[verify] $ podman info --format '{{.Host.OCIRuntime.Name}}'
[verify] $ podman run --rm docker.io/library/debian:trixie true
[verify] $ ${HAS_GPU} -eq 0 || podman run --rm \
              --device nvidia.com/gpu=all \
              docker.io/library/debian:trixie nvidia-smi --query-gpu=name --format=csv,noheader
```

The `debian:trixie` pull lands inside `/var/cache/devai/registry`,
proving graphroot relocation took effect before any image was
materialised.

### 4.9 Recovery

| Symptom | Recovery |
|---|---|
| `lvcreate` fails with "Volume group has insufficient free space" | Reduce target sizes proportionally and re-run §4.5. Do NOT extend the VG without operator approval. |
| `mkfs.xfs` reports "device contains a known filesystem" | The LV is already populated. Confirm it is one of ours via `blkid` and skip the format. Do NOT pass `-f`. |
| `mount` fails with "wrong fs type" | The fstab line UUID is stale (LV was recreated). Fix by re-reading UUID with `blkid`. |
| `chown` reports permission denied on a path containing already-running container subuid files | Run `podman unshare chown -R 0:0 /var/cache/devai/<path>` from `${INVOKING_USER}`'s shell instead. |

---

## 5. Phase 5 — Repo, configuration, registry mirror

### 5.1 Clone

Skip if `${REPO_DIR}` exists and is the right repo.

```bash
$ git clone https://github.com/<owner>/devai.git ${REPO_DIR}
$ cd ${REPO_DIR}
```

The agent must be told the canonical clone URL by the operator if
multiple forks exist. Do not assume a default.

### 5.2 `.env`

The repo ships `.env.example`. Copy only if `.env` is absent
(idempotency: never overwrite an existing `.env`):

```bash
$ test -f ${REPO_DIR}/.env || cp ${REPO_DIR}/.env.example ${REPO_DIR}/.env
```

**`.env` and `${VAR:-default}` compose substitution.** `deploy/docker-compose.yaml`
references several values via `${KEY:-fallback}` syntax (notably
`VLLM_IMAGE`, `SGLANG_IMAGE`, `DEVAI_REASONING`, `GPU_MEMORY_GB`,
`MAX_CONTEXT_LEN`, `VLLM_PLUGINS_HOST_DIR`). The Go-based
`docker-compose` v2 binary (installed in §2.1.a) expands these
correctly and a sparse `.env` is enough. The Python `podman-compose`
1.3.0 in Debian apt does **not** expand `${VAR:-default}` — those
strings reach the router container as literal `${VLLM_IMAGE:-…}` text
and the router fails to recreate vLLM/SGLang on demand. If for any
reason `docker-compose` v2 is not available, set the values
explicitly in `.env`:

```text
VLLM_IMAGE=docker.io/vllm/vllm-openai:latest-cu130-ubuntu2404
SGLANG_IMAGE=docker.io/lmsysorg/sglang:v0.5.10.post1-cu130
DEVAI_REASONING=auto
GPU_MEMORY_GB=24
MAX_CONTEXT_LEN=131072
VLLM_PLUGINS_HOST_DIR=
```

Tunable keys (see comments inside `.env.example`):

- `LAB_PORT` — JupyterLab port (default 8888).
- `WEBUI_PORT` — Open WebUI HTTPS port (default 8443).
- `CONTAINER_RUNTIME` — `podman` (default) or `docker`.
- `HOST_HOME_DIR` — host directory whose `.gitconfig` and `.ssh` are
  copied into the lab container's home on first run.
- `HOME_VOLUME` — persistent home directory for the lab user
  (default `${HOME_DIR}/devai-home`).
- `JUPYTER_TOKEN` — recommended; if unset, JupyterLab generates a
  random token printed at startup. Set this for systemd auto-start.
- `GPU_MEMORY_GB` — total VRAM the router should advertise (default
  24). Match real card; the picker filters by this band.
- `MAX_CONTEXT_LEN` — upper bound for per-request context (default
  131072 = 128K). Routes are capped at `min(model_max_ctx,
  MAX_CONTEXT_LEN)`.
- `DEVAI_REASONING` — global reasoning policy
  (`auto|off|low|medium|high`); per-request overrides are described in
  the project README.
- Proxy: `HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`, `APT_PROXY` —
  set if the host is behind a corporate proxy. The Makefile threads
  them into both build args and runtime env.

### 5.3 Permissions on cache (defensive)

`make cache-up` creates files inside `/var/cache/devai/*` as the
`${INVOKING_USER}` rootless mapping. If §4 was followed exactly, the
top-level dirs are already owned correctly and no further action is
needed. Verify:

```bash
[verify] $ stat -c '%U' /var/cache/devai
```

Must equal `${INVOKING_USER}`.

---

## 6. Phase 6 — Pre-pull base images and CLI binaries

This is split out from `make build` for two reasons:

1. The base/runtime images and CLI tarballs are large; doing them
   first lets the agent fail fast if a registry is unreachable.
2. `fetch-cli` populates `/var/cache/devai/pip/bin/` which is bind-
   mounted into the build, so the actual `make build` runs offline.

### 6.1 Pull infrastructure images

`make pull-images` pulls every image referenced by
`deploy/docker-compose.yaml` plus the base images for the lab build.
Idempotent — `podman pull` no-ops on a digest match.

```bash
$ cd ${REPO_DIR}
$ make pull-images
```

Image set (current as of repo HEAD; consult `deploy/docker-compose.yaml`
for the source of truth):

- `debian:trixie` — base for CPU lab and router build stage.
- `docker.io/nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04` — base for
  GPU lab.
- `docker.io/library/golang:1.23-bookworm` — router build stage.
- `gcr.io/distroless/static-debian12` — router runtime stage.
- `sameersbn/apt-cacher-ng:latest` — apt cache.
- `registry:2` — Docker Hub mirror.
- `ollama/ollama:latest` — Ollama backend.
- `docker.io/vllm/vllm-openai:latest-cu130-ubuntu2404` — vLLM
  backend (placeholder + on-demand recreate).
- `docker.io/lmsysorg/sglang:v0.5.10.post1-cu130` — SGLang backend.
- `ghcr.io/open-webui/open-webui:main` — chat UI.
- `docker.io/library/nginx:alpine` — TLS termination for the chat UI.
- `quay.io/podman/stable` — logger sidecar (tails podman logs).

If a pull fails for one image but succeeds for others, the agent must
NOT proceed to §7 — `make build` will fail mid-way and leave a partial
image. Recover by re-running `make pull-images` and checking network
or auth on the failing registry.

### 6.2 Fetch CLI binaries

```bash
$ make fetch-cli
```

This downloads (with ETag-based caching under
`/var/cache/devai/pip/bin/.etags/`):

- Anthropic Claude Code (release CDN)
- OpenAI Codex (GitHub releases)
- Ollama CLI (GitHub releases)
- code-server (GitHub releases)
- `uv` and `uvx` (GitHub releases)
- Google Gemini CLI (npm registry tarball)
- LATE (GitHub releases)

Each tool is staged to `/var/cache/devai/pip/bin/<name>` and bind-
mounted read-only into the lab image at build time. Re-running the
target only re-downloads tools whose ETag changed.

### 6.3 Verification

```bash
[verify] $ podman images --format '{{.Repository}}:{{.Tag}}' | grep -E '(debian|nvidia/cuda|golang|distroless|apt-cacher|registry|ollama|vllm|sglang|open-webui|nginx|podman/stable)' | sort -u
[verify] $ ls -1 /var/cache/devai/pip/bin/ | grep -E '^(claude|codex|ollama|code-server|uv|uvx|gemini|late)$'
```

---

## 7. Phase 7 — Build images

```bash
$ cd ${REPO_DIR}
$ make build           # builds: base-cpu, lab-cpu, base-gpu, lab-gpu, router
```

Targets (all called by `make build`):

| Target | Image | When to skip |
|---|---|---|
| `build-base-cpu` | `devai-base-cpu` | always required (router not affected; lab-cpu depends on it) |
| `build-cpu` | `devai-lab-cpu` | always required |
| `build-base-gpu` | `devai-base-gpu` | `${HAS_GPU}=0` |
| `build-gpu` | `devai-lab-gpu` | `${HAS_GPU}=0` |
| `build-router` | `localhost/devai-router` | always required |

If `${HAS_GPU}=0`, run only the CPU builds:

```bash
$ make build-cpu build-router
```

On Linux aarch64, the CPU lab image works but the build must use
arm64 base images. Most upstream images (`debian:trixie`,
`ollama/ollama:latest`, `quay.io/podman/stable`,
`ghcr.io/open-webui/open-webui:main`,
`docker.io/library/nginx:alpine`, `docker.io/library/golang:1.23-bookworm`,
`gcr.io/distroless/static-debian12`, `registry:2`,
`sameersbn/apt-cacher-ng:latest`) ship arm64 manifests.
**`docker.io/vllm/vllm-openai:*` and `docker.io/lmsysorg/sglang:*` do
not ship arm64 manifests** — `make cache-up` will leave their
placeholders in `Created` state on aarch64 hosts. This is harmless
because the router only recreates them on demand, and on aarch64
nothing should be probing them. If a probe is attempted, it will
fail with an architecture-mismatch error; do not retry.

The repo's `Makefile` `fetch-cli` recipe also hardcodes the `uv`
tarball filename to `uv-x86_64-unknown-linux-gnu.tar.gz`, which
breaks on aarch64. Patch in place before running `make build-cpu`:

```bash
$ test "$(uname -m)" != aarch64 || \
      sed -i.bak 's|uv-x86_64-unknown-linux-gnu|uv-aarch64-unknown-linux-gnu|g' \
          ${REPO_DIR}/Makefile
```

The Makefile bind-mounts `/var/cache/devai/pip` (uv cache),
`/var/cache/devai/npm` (npm cache), and `/var/cache/devai/pip/bin`
(CLI binaries) into each build, so the build is mostly offline after
§6 succeeded.

### 7.1 Verification

```bash
[verify] $ podman images --format '{{.Repository}}:{{.Tag}}' | grep -E '^(devai-(base-cpu|base-gpu|lab-cpu|lab-gpu)|localhost/devai-router):latest$' | sort
```

Expected (GPU host): five entries. CPU-only host: three (no
`-gpu` suffix).

### 7.2 Recovery

| Symptom | Recovery |
|---|---|
| `fetch-cli` produced the binary but the build complains it is missing | Permissions: `/var/cache/devai/pip/bin/<name>` must be readable by the rootless build subuid. Run `chmod +rX -R /var/cache/devai/pip/bin`. |
| Build fails on `pip install` step with network errors | Re-run `make pull-images` and verify the proxy env is exported into the build with `make build HTTP_PROXY=... HTTPS_PROXY=...`. |
| GPU build OOMs the host during torch install | Reduce parallel build threads via `BUILDAH_FORMAT=docker BUILD_ARGS="--jobs 1" make build-gpu`. |

---

## 8. Phase 8 — Start infrastructure

### 8.0 TLS certificates for `devai-webui-proxy` (mandatory)

The `nginx:alpine` image used by `devai-webui-proxy` does **not** ship
the `openssl` binary, so `deploy/webui-proxy/entrypoint.sh`'s
"self-signed fallback" silently fails (the `openssl req` call exits
with `command not found`, swallowed by the entrypoint's
`2>/dev/null`). nginx then loops on
`cannot load certificate "/etc/nginx/ssl/cert.pem"` and the container
restarts forever. Pre-generate certs **on the host** so the entrypoint
takes the "found existing" branch:

```bash
$ SSL_DIR=${HOME_DIR}/devai-home/.jupyter/ssl
$ mkdir -p "$SSL_DIR"
$ test -f "$SSL_DIR/cert.pem" || openssl req -x509 -nodes -days 365 \
      -newkey rsa:2048 \
      -keyout "$SSL_DIR/key.pem" -out "$SSL_DIR/cert.pem" \
      -subj "/CN=devai-webui"
$ chmod 600 "$SSL_DIR/key.pem"
[verify] $ ls "$SSL_DIR"/cert.pem "$SSL_DIR"/key.pem
```

(For browser-trusted certs use `mkcert <hostname>` instead of openssl
and copy the resulting `<host>.pem` / `<host>-key.pem` into `$SSL_DIR`.)

### 8.1 Bring the stack up

```bash
$ make cache-up
```

(The Makefile creates the `devai-net` external network if missing,
then runs `podman compose -f deploy/docker-compose.yaml up -d` — which
in turn dispatches to the `docker-compose` v2 binary installed in
§2.1.a. If you bypass the Makefile and call compose directly, you
must `podman network create devai-net` first.)

This:

- Ensures `${INVOKING_USER}`'s podman socket is enabled (idempotent).
- Creates the `devai-net` network if missing.
- `podman compose -f deploy/docker-compose.yaml up -d` brings up:
  `devai-apt-cache`, `devai-registry-cache`, `devai-ollama`,
  `devai-vllm` (placeholder), `devai-sglang` (placeholder),
  `devai-router`, `devai-open-webui`, `devai-webui-proxy`,
  `devai-logger`.

`devai-vllm` and `devai-sglang` are launched with
`entrypoint: ["sleep", "infinity"]` and DO NOT serve traffic; the
router replaces each container on demand when a request first arrives
on its dedicated port (11435 / 11436). This is by design — see
`docs/backends.md` and the project CLAUDE.md.

### 8.2 Verification

```bash
[verify] $ podman ps --format '{{.Names}}\t{{.Status}}' | grep -E '^devai-' | sort
[verify] $ curl -fsS http://localhost:11434/v1/models   # router → ollama
[verify] $ curl -k -fsS https://localhost:8443/         # webui-proxy
```

The first time `make cache-up` runs, the router probe-cache files
(`deploy/.ollama-reasoning-cache.json`,
`deploy/.vllm-reasoning-cache.json`,
`deploy/.sglang-reasoning-cache.json`) may not yet exist on disk —
that is fine. The router treats missing caches as "no models known
yet" and the bind mounts in compose are conditional. Phase 9
populates them.

### 8.3 Recovery

| Symptom | Recovery |
|---|---|
| `Error: name <devai-X> already in use` | `make cache-down && make cache-up`. The router occasionally leaves drifted vllm/sglang containers behind; `cache-down` force-removes them. |
| Router container exits immediately | `podman logs devai-router` — most common cause is a missing podman socket bind (`XDG_RUNTIME_DIR` unset in the user environment). Re-run `systemctl --user enable --now podman.socket`. |
| `ollama` container repeatedly OOMs at startup | Set `OLLAMA_KEEP_ALIVE=10s` in the environment and `OLLAMA_MAX_LOADED_MODELS=1` (already in compose). If the GPU has < 8 GB free, no Ollama model is loadable; expect probes to fail. |

---

## 9. Phase 9 — Pull and probe initial model

The system is functional after §8 but has no models. The minimum to
prove end-to-end behaviour is one Ollama GGUF model. NVFP4 / vLLM /
SGLang are optional and only useful on Blackwell-class hardware.

### 9.1 Default first model

```bash
$ make model-pull FAMILY=qwen3.5
$ make probe
```

`make model-pull` reads `deploy/models.yaml` and pulls every fitting
variant of the `qwen3.5` family for the host's `(VRAM, context)`
matrix. On a 24 GB card, this typically pulls 1–3 quantised tags
(e.g. `qwen3.5:9b-q8_0`).

`make probe` exercises every `(VRAM band, context tier, backend)` cell
on each downloaded Ollama digest and writes the result to
`deploy/.ollama-reasoning-cache.json`. The first run takes 5–15 min
per model depending on context tier.

### 9.2 vLLM / SGLang (GPU only, optional)

Skip on Linux aarch64 — no upstream image support (see §7).

NVFP4 weights are in HuggingFace format; pulling them requires the
`huggingface-hub` Python package on the host:

```bash
$ pip install --user 'huggingface-hub[cli]'
$ huggingface-cli login        # only if the model is gated
$ make model-pull FAMILY=Qwen3-8B-NVFP4    # example
```

Probing vLLM and SGLang requires **exclusive** GPU access, so the
infra stack must be down first:

```bash
$ make cache-down
$ make probe-vllm
$ make probe-sglang
$ make cache-up
```

`make probe-vllm` and `make probe-sglang` write to
`deploy/.vllm-reasoning-cache.json` and
`deploy/.sglang-reasoning-cache.json` respectively. The router and
picker pick up the new entries on the next request — no rebuild
needed.

### 9.3 Verification

```bash
[verify] $ test -s ${REPO_DIR}/deploy/.ollama-reasoning-cache.json
[verify] $ podman exec devai-ollama ollama list | tail -n +2
```

For full end-to-end (sends a real prompt to the loaded model):

```bash
[verify] $ curl -fsS http://localhost:11434/api/chat -H 'content-type: application/json' \
              -d '{"model":"qwen3.5:9b-q8_0","messages":[{"role":"user","content":"reply with the single word PONG"}],"stream":false}' \
              | python3 -c "import sys,json; print(json.load(sys.stdin)['message']['content'][:200])"
```

Expected: `PONG` or a short response containing `PONG`. If the model
name `qwen3.5:9b-q8_0` does not exist in the local Ollama cache, use
the first row of `ollama list` instead.

### 9.4 Recovery

| Symptom | Recovery |
|---|---|
| `make model-pull` fails on HuggingFace 401 | Run `huggingface-cli login`; for gated models the operator must accept the licence on the HF web UI first. |
| Probe writes `fits=false` for every cell | The host VRAM is genuinely too small for that model at any tier. Pick a smaller family (`qwen3.5`, `llama3.2`) or a higher-quant tag (q4 instead of q8). |
| `make probe-vllm` aborts with "router/vllm/sglang container running" | Run `make cache-down` first. Probing needs exclusive GPU. |

---

## 10. Phase 10 — Install `devai-agent` launcher

```bash
$ cd ${REPO_DIR}
$ make install
$ echo 'export PATH="$HOME/.local/bin:$PATH"' >> ${HOME_DIR}/.bashrc   # only if missing
$ devai-agent --init
```

`make install` (defined in the Makefile):

- Symlinks `${REPO_DIR}/bin/devai-agent` to
  `${HOME_DIR}/.local/bin/devai-agent`.
- Symlinks the picker and probe caches into `${HOME_DIR}/.devai/`
  so the launcher can bind-mount them into the lab container without
  rebuilding the image.

`devai-agent --init` writes a default
`${HOME_DIR}/.devai/preferences.yaml` with `vram=24`,
`context=131072`, etc. Edit if the host's VRAM differs.

### 10.1 Verification

```bash
[verify] $ command -v devai-agent
[verify] $ devai-agent --show
```

The `--show` invocation must print the resolved preferences and the
constructed `podman run` command without launching anything. If it
errors with "image not found", re-run §7.

### 10.2 Claude Code dummy credentials (mandatory if `--agent claude` is used)

Claude Code (any version baked into the lab image) refuses to run
without a credentials file at `${HOME_DIR}/.claude/.credentials.json`.
On a real workstation that file is populated by `claude auth` against
api.anthropic.com. In the local-router setup there is no Anthropic
account to authenticate against — but if the file is **absent**,
Claude fires a startup probe with its hardcoded model id
`claude-haiku-4-5-20251001`. The router has no entry for that name,
so it phantom-launches a vLLM container that never serves, the
foreground turn starves on the 10-minute health timeout, and the
session hangs. The picker's `ANTHROPIC_DEFAULT_HAIKU_MODEL` /
`ANTHROPIC_SMALL_FAST_MODEL` env vars do **not** suppress this probe
on the Claude Code versions currently in the lab image.

Fix: drop a syntactically valid but locally inert credentials file
at the path Claude reads inside the lab container (which is the
host path `${HOME_DIR}/devai-home/.claude/.credentials.json`,
bind-mounted by `devai-agent` to `/home/devai/.claude/...`).

```bash
$ TARGET=${HOME_DIR}/devai-home/.claude/.credentials.json
$ mkdir -p "$(dirname "$TARGET")"
$ python3 - "$TARGET" <<'PY'
import json, os, pathlib, sys
prefix = "sk-ant-DUMMY_LOCAL_DEVAI_NOT_VALID_FOR_REAL_API_USE_"   # 52 chars
dummy = {
    "claudeAiOauth": {
        "accessToken":      prefix + "A" * 56,           # 108 chars total
        "refreshToken":     prefix + "B" * 56,
        "expiresAt":        4070908800000,                # year 2099 (ms)
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

[verify] $ test -f ${HOME_DIR}/devai-home/.claude/.credentials.json
[verify] $ stat -c '%a' ${HOME_DIR}/devai-home/.claude/.credentials.json   # must be 600
```

The token strings are cryptographically invalid — they will be
rejected immediately by api.anthropic.com if Claude ever falls back
to it — but the picker pins `ANTHROPIC_BASE_URL` to the local router,
so no cloud call is ever attempted. The file's *presence* alone is
what suppresses the OAuth probe.

This file is **per-user, per-host, mode 0600**. Do not check it into
the repo. Do not copy a real `.credentials.json` from a workstation
into a shared host — that token would be valid for the workstation
operator's account and would leak quota.

---

## 11. Phase 11 — (optional) auto-start at boot

### 11.1 systemd (most distros)

To bring the cache stack up on boot:

```bash
$ make install-systemd
```

This:

- Copies `deploy/docker-compose.yaml` and
  `deploy/registry-config.yaml` to `${HOME_DIR}/.config/devai/`.
- Installs `deploy/systemd/devai-infra.service` as a `--user` unit.
- Enables `podman.socket` and `devai-infra.service`.
- Calls `loginctl enable-linger ${INVOKING_USER}` so the user services
  keep running after logout.

Idempotent: re-running overwrites the staged compose file (intentional
— picks up upstream changes) but does not destroy any container state.

Verification:

```bash
[verify] $ systemctl --user is-enabled devai-infra.service
[verify] $ systemctl --user is-active  devai-infra.service
```

### 11.2 Non-systemd distros (Alpine, etc.)

Translate `devai-infra.service` to the host's service manager.
For OpenRC, place an init script in `/etc/init.d/devai-infra`
running `cd ${HOME_DIR}/.config/devai && podman compose up -d`,
then `rc-update add devai-infra default`. The agent should not
generate this file autonomously — surface the request to the
operator with the equivalent `systemd` unit content as a starting
point.

---

## 12. Phase 12 — End-to-end smoke test

A green smoke test means the agent has finished correctly.

```bash
$ make test-router    # Go unit tests; no GPU; ~1 min
$ make test-ollama    # Live Ollama integration; needs §9 to have pulled at least one model
```

For a manual single-shot:

```bash
[verify] $ curl -fsS http://localhost:11434/api/chat -H 'content-type: application/json' \
              -d '{"model":"qwen3.5:9b-q8_0","messages":[{"role":"user","content":"reply with the single word PONG"}],"stream":false}'
[verify] $ curl -k -fsS https://localhost:8443/                | head -c 200
[verify] $ curl -fsS http://localhost:8888/ -o /dev/null && echo lab_up || true
```

The full `make test` suite (`test-router`, probe smokes, vLLM/SGLang
live integration, matrix tests) takes 30–60 min and requires the
infra to be cycled (`cache-down` for probe smokes, then `cache-up`).
Run only when the operator wants exhaustive validation.

---

## 13. State-of-the-world reference

For the agent reading this file later, here is what the *running*
host looks like once Phases 1–11 succeed.

### 13.1 Filesystem map

```
/var/cache/devai/
├── apt/             ← LV cache_apt        (apt-cacher-ng)
├── npm/             ← LV cache_npm        (npm cache for builds)
├── pip/             ← LV cache_pip        (uv cache + CLI binaries in pip/bin/)
├── registry/        ← LV cache_registry   (podman graphroot + registry:2 mirror)
├── ollama/          ← LV cache_ollama     (Ollama blobs + vLLM/SGLang weight trees)
│   └── models/
│       ├── blobs/         (Ollama GGUF + manifests)
│       ├── manifests/
│       └── vllm/          (HF safetensors per repo dir)
├── open-webui/      ← LV cache_open_webui (chat UI app data)
├── logs/            ← LV cache_logs       (per-container stdout, written by devai-logger)
└── bench/                                  (optional bench run artefacts; created by make bench)

${REPO_DIR}/
├── .env                                  (host config — never overwrite)
├── deploy/
│   ├── .ollama-reasoning-cache.json      (probe cache, schema v3, digest-keyed)
│   ├── .vllm-reasoning-cache.json        (probe cache, schema v2, repo+sha-keyed)
│   ├── .sglang-reasoning-cache.json      (probe cache, schema v2)
│   ├── .bench-cache.json                 (optional bench scores; populated by make bench)
│   ├── docker-compose.yaml               (infra spec — source of truth)
│   ├── models.yaml                       (catalog — auto-generated)
│   ├── registries.conf                   (rootless podman registry routing)
│   ├── registry-config.yaml              (registry:2 config)
│   ├── recovery-flags.json               (per-model launch overrides)
│   └── vllm-plugins.json                 (custom parser plugin registry)
└── scripts/                              (probe drivers, pickers, bench harness)

${HOME_DIR}/
├── .config/containers/registries.conf    (mirror routing — only if installed in §3.3)
├── .config/devai/                        (systemd-managed copies, only if §11)
├── .config/systemd/user/devai-infra.service  (only if §11)
├── .devai/                               (devai-agent state)
│   ├── preferences.yaml
│   ├── sessions/
│   ├── model-picker.py                   (symlink → repo)
│   ├── .ollama-reasoning-cache.json      (symlink → repo)
│   ├── .vllm-reasoning-cache.json        (symlink → repo)
│   ├── .sglang-reasoning-cache.json      (symlink → repo)
│   └── .bench-cache.json                 (symlink → repo)
├── .local/bin/devai-agent                (symlink → repo/bin/devai-agent)
└── devai-home/                           (lab container's persistent home; bind-mounted)
```

### 13.2 Network

All containers share the user-defined bridge `devai-net`. Inter-
container traffic uses container names as DNS:

```
clients (host port 11434) ─► devai-router:11434 ─► devai-ollama:11434
clients (host port 11435) ─► devai-router:11435 ─► devai-vllm:11434
clients (host port 11436) ─► devai-router:11436 ─► devai-sglang:11434
browser (host port 8443)  ─► devai-webui-proxy:443 ─► devai-open-webui:8080 ─► devai-router:11434
browser (host port 8888)  ─► devai-lab-{cpu,gpu}:8888                  (lab)
```

Only the router's listening ports, the webui-proxy port, the apt
cache port (3142), the registry mirror port (5000), and the lab port
(8888) are exposed on the host. Backends are reachable only from
inside the network.

### 13.3 GPU mutual exclusion (router invariant)

Only one of `devai-ollama`, `devai-vllm`, `devai-sglang` holds the
GPU at a time. The router (`gpu-arbiter/main.go`) enforces this by
draining the active backend before recreating another. Any deployment
that bypasses the router (e.g. by talking to `devai-ollama:11434`
directly) breaks the invariant — do not do this. Use only the
router's host-exposed ports `11434/11435/11436`.

### 13.4 Data classes and what is safe to delete

| Path | Class | Safe to delete? |
|---|---|---|
| `/var/cache/devai/registry/docker/` | regenerable mirror cache | yes |
| `/var/cache/devai/apt/` | regenerable apt cache | yes |
| `/var/cache/devai/pip/` | regenerable build cache | yes (forces full rebuild) |
| `/var/cache/devai/npm/` | regenerable build cache | yes |
| `/var/cache/devai/ollama/models/blobs/` | downloaded models | costly; deleting forces re-pull (10–100 GB per model) |
| `/var/cache/devai/ollama/models/vllm/` | downloaded NVFP4 weights | same caveat |
| `/var/cache/devai/open-webui/` | chat history & users | preserve unless intentional reset |
| `/var/cache/devai/logs/` | container stdout logs | yes (rotated implicitly) |
| `${REPO_DIR}/deploy/.*-reasoning-cache.json` | probe results | yes (next probe regenerates) |
| `${REPO_DIR}/deploy/.bench-cache.json` | bench scores | yes (next bench run regenerates) |

`${REPO_DIR}/deploy/registry/` (note: shared with podman graphroot)
is **never** safe to wipe wholesale — see the `cache-clean` Makefile
target for the correct selective approach via `podman unshare`.

---

## 14. Tear-down

If the agent is asked to fully remove Dev AI Lab from the host:

```bash
$ make cache-down               # stop and remove containers
$ make clean                    # remove built images
$ make uninstall                # remove devai-agent launcher + symlinks
# systemctl --user disable --now devai-infra.service     (only if §11 ran)
# rm -f ${HOME_DIR}/.config/systemd/user/devai-infra.service
# rm -rf ${HOME_DIR}/.config/devai ${HOME_DIR}/.devai
```

To also reclaim the LVM volumes (DESTROYS DOWNLOADED MODELS):

```bash
# umount /var/cache/devai/{ollama,registry,pip,apt,npm,open-webui,logs}
# sed -i.bak '/\/var\/cache\/devai\//d' /etc/fstab
# for lv in cache_ollama cache_registry cache_pip cache_apt cache_npm cache_open_webui cache_logs; do
      lvremove -f ${VG_NAME}/${lv}
  done
# lvremove -f ${VG_NAME}/${POOL_NAME}        # only if no other LVs depend on it
```

Do not perform the LVM step without explicit operator confirmation.

---

## 15. Decisions left to the operator

The agent must surface these to the operator and wait for an explicit
answer rather than choosing autonomously:

1. **Platform confirmation** when §0.5 lists the detected platform as
   "best-effort" (Alpine), or when the platform is not in the matrix
   at all.
2. **Block device for the volume group** (§4.3) when no `vgais` (or
   equivalent) exists.
3. **Thin pool size** (§4.4) when free space in the VG is between
   100 GB and 800 GB — anything above 600 GB is comfortable, anything
   below requires shrinking the LV size table in §4.1.
4. **Disk to back the LVM cache pool** (§1.3) when the host has no
   existing VG and no spare disk/partition. INSTALL.md halts here
   until the operator attaches a second disk (typical: ≥ 600 GB) or
   frees a partition. There is no plain-directory fallback —
   /var/cache/devai/* must live on dedicated LVs.
5. **HuggingFace credentials** (§9.2) when downloading gated NVFP4
   weights (any NVIDIA-* repo, Llama-3.x, etc.).
6. **JUPYTER_TOKEN** value (§5.2) when auto-start is requested —
   without a fixed token, every service restart invalidates browser
   bookmarks.
7. **Public exposure** — none of the host ports listed in §13.2
   should be exposed to the public internet without an explicit
   reverse-proxy + auth layer the operator owns. The agent must NOT
   add a port-forward to a router or firewall rule unless asked.
8. **Reboot to load NVIDIA driver** (§2.1.*) when the kernel module
   is absent post-install. *GPU only.*
9. **Auto-start on non-systemd distros** (§11.2) — requires a
   manually authored OpenRC/runit/s6 service that the agent must
   propose and the operator must approve before installing.

For everything else, the agent decides per the procedure above.

---

## 16. Validating this procedure

This document is "doc as test": if a tier-1 coding agent (Claude
Code, Codex, etc.) reading INSTALL.md cannot produce a working
stack, the doc is the bug.

The validation method is for the operator (or a separate driver
agent) to run the procedure end-to-end against a fresh host and
confirm the §12 smoke test passes. There is no in-repo automated
harness — that abstraction was removed because the failure modes it
hid (env-var substitution, dummy claude credentials, LVM
auto-activation, fstab `nofail`, etc.) were exactly the gaps the
doc itself needed to close. Each of those is now called out
explicitly in the relevant phase.

If you want to validate against an isolated VM rather than touch
the production host, see [`docs/HOST_VFIO_SETUP.md`](docs/HOST_VFIO_SETUP.md)
for the host-side preparation needed to temporarily hand the GPU
to a libvirt VM via PCI passthrough. The VM provisioning + cloud-init
+ agent dispatch is then driven by the operator with `virt-install`
and an SSH session, not by a hidden bash harness.

---

## 17. Versioning of this file

This document tracks the repository at the time of writing. When
`deploy/docker-compose.yaml`, `deploy/Dockerfile.{base,lab,router}`,
`Makefile`, or the `.env.example` shape changes, regenerate the
relevant section here. Specifically:

- §6.1 image list — keep in sync with
  `podman compose -f deploy/docker-compose.yaml config --images`.
- §4.1 volume table — keep in sync with the storage section of
  `README.md` and `deploy/setup-logs-volume.sh`.
- §5.2 `.env` keys — keep in sync with `.env.example` and
  the `?=` defaults at the top of `Makefile`.
- §9.x model-pull / probe targets — keep in sync with the
  `MODELS` block in `make help`.

The agent should treat this file as authoritative for **bootstrap**;
for runtime architecture, defer to `docs/router.md`,
`docs/backends.md`, `CLAUDE.md`, and the in-repo `README.md`.
