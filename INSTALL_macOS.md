# INSTALL_macOS.md — macOS bootstrap procedure for AI agents

This document is the **macOS-only** companion to [`INSTALL.md`](INSTALL.md).
Agents reach this file through the redirect in §0 of `INSTALL.md` after
detecting `Darwin` from `uname -s`. Do not interleave this file with
the Linux procedure; the architectural choices on macOS diverge enough
that mixing the two will produce a broken stack.

---

## 0. Why a separate document

macOS is not "Linux with a different package manager." Three things
make the inference path qualitatively different from the Linux
reference platform:

1. **Unified memory architecture (UMA).** On Apple Silicon (M1/M2/M3/
   M4), CPU and GPU share the same physical RAM through a single
   memory controller. There is no discrete VRAM, no PCIe transfer to
   move tensors onto a GPU, and the GPU's working set can be a large
   fraction of total system RAM. A 64 GB M3 Max can hold a
   70B-parameter model entirely "on GPU" without copy overhead, which
   is impossible on any consumer NVIDIA card. §11 below explains how
   to exploit this.
2. **No CUDA, no NVIDIA Container Toolkit.** vLLM and SGLang are CUDA-
   only. On macOS they are removed from the running stack — there is
   no Metal port and no plan from the upstream projects to add one.
   The fast inference paths on macOS are **Ollama** (llama.cpp + Metal)
   and **MLX-LM** (Apple's first-party ML framework, native to UMA).
3. **Containers run inside a Linux VM.** `podman machine` ships a
   small Linux VM (Apple Virtualization framework on M-series, QEMU
   on Intel). Anything inside that VM has no Metal access — the VM's
   "GPU" is virtual and uninteresting. This forces a hybrid topology:
   the inference engine runs **natively on the macOS host**, the
   developer-tooling lab runs **inside the VM**, and the two talk over
   the host network bridge.

The redirect from `INSTALL.md` exists because applying the Linux
procedure on macOS produces a stack that is functional but slow (CPU
inference inside the VM, no UMA exploitation) and that uses the
project's GPU-arbitration plumbing for nothing.

---

## 1. Contract

### 1.1 Goal

Take a macOS 13+ host (Apple Silicon strongly preferred; Intel
supported but degraded — see §2.2) with sudo and produce:

- **Ollama** running natively on the host with Metal acceleration,
  using unified memory directly.
- A `podman machine` Linux VM hosting the **devai lab container** and
  optionally the **devai-router**.
- Lab container's AI CLIs (Claude Code, Codex, Gemini, Aider, etc.)
  configured to reach host-side Ollama via `host.containers.internal`.
- A `devai-agent` host launcher on `PATH`.
- Optional: **MLX-LM** as a second native inference server for users
  who want the Apple-native model formats.

The infrastructure subset that DOES NOT run on macOS:

- `devai-ollama` container (replaced by host-native Ollama)
- `devai-vllm`, `devai-sglang` (CUDA-only)
- The bench harness's vLLM/SGLang phases (`make bench-vllm`,
  `make bench-sglang`)

### 1.2 Constraints

Same as `INSTALL.md` §0.2: non-interactive, idempotent, verify before
mutating, fail loudly, preserve user data. Re-stated here so this file
is self-contained.

### 1.3 Conventions

- `$` runs as the macOS desktop user (admin group).
- `[verify]` is a read-only check.
- `${INVOKING_USER}` = `$(id -un)` on the macOS host.
- `${REPO_DIR}` = absolute path to the cloned `devai` repo on the
  macOS host (e.g. `~/git/devai`).
- `${HOME_DIR}` = `~` of `${INVOKING_USER}`.
- `${VM_NAME}` = the `podman machine` name. Default: `devai`.
- `${VM_USER}` = the username inside the VM. Default: `core`.
- "VM-side path" means the path exists inside the `podman machine`
  Linux VM and is reachable via `podman machine ssh ${VM_NAME}`. It is
  NOT visible on the macOS host's Finder.

### 1.4 Phase order

| # | Phase | Mutates host? | Mutates VM? | Verification |
|---|---|---|---|---|
| 1 | Detect environment | no | no | `[verify]` only |
| 2 | Install Homebrew + base CLI tools | yes | no | `brew list` |
| 3 | Install Ollama natively | yes | no | `curl http://localhost:11434/api/tags` |
| 4 | (optional) Install MLX-LM | yes | no | `python3 -m pip show mlx-lm` |
| 5 | Configure host Ollama for cross-VM reach | yes | no | `curl http://0.0.0.0:11434/api/tags` |
| 6 | `podman machine init` | yes | yes (creates) | `podman machine list` |
| 7 | Clone repo, write `.env`, write compose override | yes | no | `test -f` |
| 8 | Build CPU lab + router images | yes | yes (image store) | `podman images` |
| 9 | Start macOS infra subset | yes | yes (containers) | `podman ps` |
| 10 | Pull initial Ollama model on host | yes | no | `ollama list` |
| 11 | Install `devai-agent` launcher | yes | no | `command -v devai-agent` |
| 12 | End-to-end smoke test | yes | yes (one chat completion) | HTTP 200 + non-empty body |
| 13 | (optional) Unified-memory tuning | yes (sysctl) | no | `sysctl iogpu.wired_limit_mb` |

---

## 2. Phase 1 — Detect environment

### 2.1 Confirm macOS

```bash
[verify] $ uname -s                          # Darwin
[verify] $ sw_vers -productVersion           # ≥ 13.0
[verify] $ sw_vers -productName              # macOS
```

Stop if `uname -s` is not `Darwin` — wrong document. Stop if macOS
< 13 — `podman machine` and Homebrew dependencies require 13+.

### 2.2 Architecture

```bash
[verify] $ uname -m                          # arm64 | x86_64
[verify] $ sysctl -n machdep.cpu.brand_string
```

**Apple Silicon (`arm64`)** is the supported reference platform for
this document. Every M-series CPU has the integrated GPU and the
unified memory architecture that make local inference worth doing on
macOS at all.

**Intel (`x86_64`)** Macs are supported only for the developer-tools
side (lab + agent CLIs). They lack:

- Apple Silicon's unified memory — system RAM and any GPU memory are
  separate pools.
- A meaningful GPU for ML — recent Intel Macs use integrated Intel
  GPUs that have no usable Metal-accelerated LLM stack. Discrete AMD
  GPUs in older iMacs/MBPs can run Metal Performance Shaders but
  llama.cpp's Metal backend is sized for Apple-Silicon-class GPUs;
  expect 5–10x slower inference than even a modest M-series chip.
- eGPU NVIDIA support (Apple removed it in macOS 12).

If `${OS_ARCH}=x86_64`, the agent should surface this to the operator
with a recommendation: do the Linux path on a separate Linux box and
use Ollama on the Mac only as a CPU-only secondary backend, OR
proceed with the Apple Silicon procedure here knowing that the
performance bullets in §11 will not apply. Do not silently downgrade.

### 2.3 RAM and disk

```bash
[verify] $ sysctl -n hw.memsize | awk '{printf "%.0f GB\n", $1/1024/1024/1024}'
[verify] $ df -h /                            # boot volume free space
[verify] $ df -h "${HOME_DIR}"                # only differs if data volume separate
```

Record:

- `${HOST_RAM_GB}` — total physical RAM. Drives model-size
  recommendations in §12 and §11.
- `${HOST_FREE_GB}` — free disk on the boot volume. The `podman
  machine` VM disk lives here unless you point it elsewhere; budget
  ≥ 100 GB free for a useful install (200 GB+ if pulling 70B-class
  models).

### 2.4 Tooling presence

```bash
[verify] $ command -v brew     && brew --version
[verify] $ command -v podman   && podman --version
[verify] $ command -v ollama   && ollama --version
[verify] $ command -v git      && git --version
[verify] $ command -v python3  && python3 --version
```

Each missing tool is installed in the appropriate later phase. Do not
install eagerly here.

### 2.5 GPU / Metal sanity (Apple Silicon only)

```bash
[verify] $ system_profiler SPDisplaysDataType | grep -E "Chipset Model|VRAM|Metal"
```

Expected line: `Metal Support: Metal 3` (or higher) on every supported
M-series chip. If absent, Metal-accelerated Ollama will fall back to
CPU and the entire premise of macOS-native inference is lost — surface
to the operator.

---

## 3. Phase 2 — Install Homebrew and base CLI tools

### 3.1 Install Homebrew if missing

```bash
$ command -v brew || /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
$ eval "$(/opt/homebrew/bin/brew shellenv)"   # arm64; on Intel use /usr/local/bin/brew
```

The agent should append the appropriate `eval "$(brew shellenv)"`
line to the operator's shell init file (`~/.zshrc` or `~/.bash_profile`)
**only** if `brew` is not yet on `PATH` for new shells. Do not
duplicate an existing line.

### 3.2 Base packages

```bash
$ brew install \
        coreutils gnu-sed gawk findutils \
        git make jq \
        python@3.13 \
        podman podman-compose \
        mkcert
```

Why each:

- `coreutils gnu-sed gawk findutils` — the project's `Makefile`
  recipes assume GNU semantics (`stat -c`, `sed -i.tmp`, GNU `find`
  predicates). Without these, several Makefile targets silently
  produce wrong output. The agent must prepend their `gnubin`
  directories to `PATH` for every `make` invocation:

  ```bash
  export PATH="$(brew --prefix coreutils)/libexec/gnubin:$(brew --prefix gnu-sed)/libexec/gnubin:$(brew --prefix gawk)/libexec/gnubin:$(brew --prefix findutils)/libexec/gnubin:$PATH"
  ```

- `python@3.13` — symlinked as `python3`. Used by `scripts/*.py`.
- `podman podman-compose` — container engine + compose plugin.
  `make cache-up` calls `podman compose` (subcommand), which
  `podman-compose` from Homebrew satisfies.
- `mkcert` — optional, only needed if the operator wants trusted local
  TLS certs for Open WebUI.

### 3.3 Verification

```bash
[verify] $ brew --version
[verify] $ podman --version          # ≥ 4.0 for host.containers.internal support
[verify] $ podman compose version
[verify] $ python3 --version         # 3.13.x
[verify] $ make --version            # GNU Make
[verify] $ stat --version            # GNU coreutils (not BSD)
```

If `stat --version` reports BSD or unknown, the gnubin `PATH` shim was
not applied. Re-export and retry.

---

## 4. Phase 3 — Install Ollama natively

This is the core of the macOS architecture. Ollama runs on the **host**
with full Metal access, not inside the VM.

### 4.1 Install

```bash
$ brew install ollama
```

This installs the `ollama` CLI plus a `brew services` definition that
runs the `ollama serve` daemon under `launchd`.

Alternative: download the Ollama `.app` bundle from `ollama.com`. The
`.app` includes a menu-bar UI but is otherwise equivalent. The brew
formula is preferred for headless operation and for the
`brew services` lifecycle hooks.

### 4.2 Start the service

```bash
$ brew services start ollama
[verify] $ curl -fsS http://localhost:11434/api/tags
```

Expected: `{"models":[]}` on a clean install. Anything else means the
daemon failed to start — `brew services info ollama` and
`tail -f /opt/homebrew/var/log/ollama.log` for diagnostics.

### 4.3 Verification

```bash
[verify] $ pgrep -f 'ollama serve' >/dev/null && echo running
[verify] $ curl -fsS http://localhost:11434/api/version | jq -r .version
```

---

## 5. Phase 4 — (optional) Install MLX-LM

Skip unless the operator explicitly wants Apple's first-party ML
framework alongside Ollama.

### 5.1 Why MLX in addition to Ollama?

| Aspect | Ollama (llama.cpp + Metal) | MLX-LM |
|---|---|---|
| Maturity | very mature, large model catalogue (GGUF) | newer; smaller catalogue (`mlx-community/*` on HF) |
| Speed (decode) | excellent | competitive, sometimes faster on small models |
| Speed (prompt prefill) | excellent | usually faster on Apple Silicon |
| Memory footprint | tightly controlled | similar |
| Quantisation | Q3/Q4/Q5/Q6/Q8/F16 | int4 / int8 / fp16 (different scheme) |
| Project integration | first-class (router, picker, agents) | not integrated; standalone HTTP server |

The project's lab/router stack is built around Ollama. MLX-LM is for
operators who want a second engine for benchmarking or for models
only available in MLX format. Treat it as additive.

### 5.2 Install

```bash
$ python3 -m pip install --user 'mlx-lm[server]'
```

### 5.3 Run

```bash
$ python3 -m mlx_lm.server \
        --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
        --host 0.0.0.0 --port 8081 &
```

The first invocation downloads weights from `huggingface.co` to
`~/.cache/huggingface/hub/`.

### 5.4 Verification

```bash
[verify] $ curl -fsS http://localhost:8081/v1/models | jq -r '.data[].id'
```

`mlx_lm.server` exposes the OpenAI-compatible `/v1/chat/completions`
endpoint. Agents can be pointed at `http://host.containers.internal:8081`
in the same way as host Ollama.

---

## 6. Phase 5 — Configure host Ollama for cross-VM reach

The lab container runs inside `podman machine`. To call host Ollama,
it must reach the macOS host across the VM's network bridge. Two
things must be true:

1. Ollama listens on `0.0.0.0`, not just `127.0.0.1`.
2. The lab container's outbound DNS resolves `host.containers.internal`
   to the host. Podman ≥ 4.0 sets this up automatically — verified
   in §7.

### 6.1 Bind Ollama to all interfaces

```bash
$ launchctl setenv OLLAMA_HOST "0.0.0.0:11434"
$ brew services restart ollama
```

`launchctl setenv` sets a global environment variable visible to
GUI-launched processes. The `brew services` launchd plist inherits it.

### 6.2 Persistence across reboots

The `launchctl setenv` call above is **not** persistent. On reboot,
re-apply or place the export in a launchd login agent. A minimal
agent:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>io.devai.ollama-host</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/launchctl</string>
    <string>setenv</string>
    <string>OLLAMA_HOST</string>
    <string>0.0.0.0:11434</string>
  </array>
  <key>RunAtLoad</key><true/>
</dict></plist>
```

Save as `~/Library/LaunchAgents/io.devai.ollama-host.plist` and load
with `launchctl load -w ~/Library/LaunchAgents/io.devai.ollama-host.plist`.
The agent must NOT install this autonomously — surface to the
operator.

### 6.3 Firewall

The macOS Application Firewall is OFF by default. If the operator has
turned it on (`System Settings → Network → Firewall`), Ollama must be
allowed. The brew binary lives at `$(brew --prefix)/bin/ollama`; add
it explicitly via `Firewall Options → Add ollama → Allow incoming`.
Do not disable the firewall globally.

### 6.4 Verification

```bash
[verify] $ curl -fsS http://localhost:11434/api/tags
[verify] $ HOST_IP=$(ipconfig getifaddr en0); curl -fsS http://${HOST_IP}:11434/api/tags
```

Both must return `{"models":[]}` (or a populated list once §10 ran).

---

## 7. Phase 6 — `podman machine init`

### 7.1 Sizing

The VM disk holds the lab image, the router image, and any
auxiliary container caches. It does **not** hold model weights —
those live in `~/.ollama/models/` on the macOS host.

| Resource | Minimum | Recommended |
|---|---|---|
| `--cpus` | 4 | 6 (M3+) or 8 (M3 Max+) |
| `--memory` | 4096 MB | 8192–12288 MB |
| `--disk-size` | 60 GB | 100 GB |

The lab image is ~6–8 GB; the router is ~10 MB; PyTorch CPU adds ~2
GB if used. The rest is build cache and headroom for incremental
rebuilds.

> **Memory note.** Do not allocate more than 25–30 % of `${HOST_RAM_GB}`
> to the VM. The whole point of the macOS architecture is to keep RAM
> available to host Ollama for unified-memory inference (§11). On a
> 32 GB Mac with `--memory 16384`, you starve Ollama.

### 7.2 Init + start

```bash
$ podman machine init \
        --cpus 6 \
        --memory 8192 \
        --disk-size 100 \
        --rootful=false \
        ${VM_NAME}
$ podman machine start ${VM_NAME}
$ podman system connection default ${VM_NAME}
```

Idempotency: if `podman machine list --format '{{.Name}}'` already
contains `${VM_NAME}`, do NOT re-init. Verify the existing size with
`podman machine inspect ${VM_NAME}`. If too small, surface to the
operator — growing the disk or RAM requires destroying and recreating
the VM, which loses container state and built images.

### 7.3 Verification

```bash
[verify] $ podman machine list --format '{{.Name}}\t{{.Running}}\t{{.Default}}'
[verify] $ podman info --format '{{.Host.OS}}/{{.Host.Arch}}'
[verify] $ podman machine ssh ${VM_NAME} 'getent hosts host.containers.internal'
```

Last line must resolve `host.containers.internal` to the host gateway
IP (typically `192.168.127.254` on M-series machines). If empty, the
podman version is too old or the network mode is wrong; upgrade
podman.

---

## 8. Phase 7 — Clone repo, write `.env`, write compose override

### 8.1 Clone

Skip if `${REPO_DIR}` already exists.

```bash
$ git clone https://github.com/<owner>/devai.git ${REPO_DIR}
$ cd ${REPO_DIR}
```

### 8.2 `.env`

Copy `.env.example` to `.env` only if `.env` is absent. Then set
macOS-appropriate values:

```bash
$ test -f ${REPO_DIR}/.env || cp ${REPO_DIR}/.env.example ${REPO_DIR}/.env
```

Recommended `.env` values for macOS:

```ini
LAB_PORT=8888
WEBUI_PORT=8443
CONTAINER_RUNTIME=podman
HOST_HOME_DIR=$(HOME)
HOME_VOLUME=$(HOME)/devai-home

# Disable VRAM-band heuristics — host Ollama uses unified memory.
GPU_MEMORY_GB=0
MAX_CONTEXT_LEN=131072
```

`GPU_MEMORY_GB=0` short-circuits the picker's VRAM-band filter so
Ollama models the host can run are not hidden as "won't fit." The
host's actual usable memory budget is whatever is left after macOS
and other apps — see §11.

### 8.3 Compose override

The default `deploy/docker-compose.yaml` boots `devai-ollama`,
`devai-vllm`, and `devai-sglang` containers that have no purpose on
macOS. Write an override that disables them and points the router at
host Ollama:

```bash
$ cat > ${REPO_DIR}/deploy/docker-compose.macos.yaml <<'EOF'
# macOS override. Apply with:
#   podman compose -f deploy/docker-compose.yaml \
#                  -f deploy/docker-compose.macos.yaml up -d
#
# Removes Linux-only GPU services and rewires the router to talk to
# host-native Ollama via host.containers.internal.

services:
  ollama:
    profiles: ["disabled"]
  vllm:
    profiles: ["disabled"]
  sglang:
    profiles: ["disabled"]

  router:
    environment:
      - OLLAMA_URL=http://host.containers.internal:11434
      - OLLAMA_PORT=11434
      # vLLM/SGLang ports must remain set so the router's listener
      # binds, but their backends will never come up. The router
      # rejects requests on those ports cleanly.
      - VLLM_URL=http://127.0.0.1:1
      - SGLANG_URL=http://127.0.0.1:1

  open-webui:
    environment:
      - OLLAMA_BASE_URL=http://router:11434
EOF
```

`profiles: ["disabled"]` is a podman-compose idiom that registers the
service but never starts it unless `--profile disabled` is passed —
which the macOS procedure never does.

### 8.4 Verification

```bash
[verify] $ test -f ${REPO_DIR}/.env
[verify] $ test -f ${REPO_DIR}/deploy/docker-compose.macos.yaml
[verify] $ grep -c '^GPU_MEMORY_GB=0' ${REPO_DIR}/.env
```

---

## 9. Phase 8 — Build CPU lab + router images

### 9.1 Apple Silicon `arm64` build caveat

The repo's `Makefile` `fetch-cli` recipe hardcodes the `uv` tarball
filename to `uv-x86_64-unknown-linux-gnu.tar.gz`. On `arm64` podman
machine VMs (default on M-series), this fetches the wrong arch and
the build fails when `cp /var/cache/bin/uv ...` runs against a binary
that does not execute.

Two workarounds:

**A. Patch the Makefile in place** (recommended; one line).

```bash
$ sed -i.bak 's|uv-x86_64-unknown-linux-gnu|uv-aarch64-unknown-linux-gnu|g' \
       ${REPO_DIR}/Makefile
```

The repo's other CLI fetches (codex, ollama, code-server, claude,
late) already detect `dpkg --print-architecture` and pick the right
arch automatically. Only `uv` is hardcoded.

**B. Force an x86_64 VM via Rosetta 2.** `podman machine init
--arch=amd64` on Apple Silicon creates an x86_64 VM that runs under
Rosetta 2. Slower (~60 % of native arm64 throughput) but every
upstream image works without patching. Use only when the operator
explicitly wants a cross-arch VM.

The agent must apply exactly one workaround. Default to A.

### 9.2 Pull and build

```bash
$ cd ${REPO_DIR}
$ make pull-images       # populates podman machine VM image store
$ make fetch-cli         # downloads CLI tarballs to /var/cache/devai/pip/bin (VM-side)
$ make build-cpu build-router
```

**Do NOT run `make build-gpu` or `make build`.** Both target
`devai-lab-gpu` and `devai-base-gpu`, which pull
`docker.io/nvidia/cuda:*` — that image has no arm64 manifest and is
useless without an NVIDIA GPU anyway.

### 9.3 Verification

```bash
[verify] $ podman images --format '{{.Repository}}:{{.Tag}}' | grep -E '^(devai-(base-cpu|lab-cpu)|localhost/devai-router):latest$'
```

Expected: three entries.

---

## 10. Phase 9 — Start the macOS infra subset

### 10.1 Start

```bash
$ cd ${REPO_DIR}
$ podman compose \
        -f deploy/docker-compose.yaml \
        -f deploy/docker-compose.macos.yaml \
        up -d
```

This brings up:

- `devai-apt-cache`, `devai-registry-cache` — build accelerators (kept).
- `devai-router` — protocol bridge + reasoning policy + tool stripping.
  Pointed at host Ollama.
- `devai-open-webui`, `devai-webui-proxy` — chat UI.
- `devai-logger` — per-container stdout sink.

NOT brought up: `devai-ollama`, `devai-vllm`, `devai-sglang`. That is
correct — host-native Ollama is doing the inference.

### 10.2 Verification

```bash
[verify] $ podman ps --format '{{.Names}}\t{{.Status}}' | grep -E '^devai-' | sort
[verify] $ podman ps --format '{{.Names}}' | grep -E '^devai-(ollama|vllm|sglang)$' && echo SHOULD_BE_EMPTY || true
[verify] $ curl -fsS http://localhost:11434/v1/models       # router → host ollama
[verify] $ curl -k -fsS https://localhost:8443/             # webui-proxy
```

The first command should list six services. The second must print
nothing. The third proves the router talks to host Ollama through
`host.containers.internal`.

### 10.3 Recovery

| Symptom | Recovery |
|---|---|
| `curl http://localhost:11434/v1/models` returns connection refused | Router is not reaching host Ollama. From inside the VM: `podman machine ssh ${VM_NAME} 'curl -v http://host.containers.internal:11434/api/tags'`. If that fails, re-check §6.1 (`OLLAMA_HOST=0.0.0.0:11434`). |
| Router container restarts repeatedly | `podman logs devai-router`. Most common: `OLLAMA_URL` resolved at compose time picked up the wrong value — confirm by inspecting `podman inspect devai-router | jq '.[].Config.Env'`. |
| Open WebUI shows "no models" | Open WebUI caches the model list; click the refresh button in the model dropdown after Phase 10 pulls a model. |

---

## 11. Phase 10 — Pull initial Ollama model on the host

```bash
$ ollama pull qwen3.5:9b-q8_0
[verify] $ ollama list
```

Use `ollama pull` directly on the macOS host — NOT `make model-pull`.
The Makefile's `model-pull` target talks to the in-VM `devai-ollama`
container, which on macOS has been disabled. Pulling on the host
puts weights in `~/.ollama/models/` where Metal can mmap them.

For arm64 Apple Silicon, prefer Q4_K_M / Q5_K_M / Q8_0 quantisations.
F16 and BF16 have no Apple-specific advantage and double memory use
without commensurate quality. See §13 for per-RAM-tier recommendations.

### 11.1 Verification

```bash
[verify] $ ollama list | tail -n +2
[verify] $ curl -fsS http://localhost:11434/api/chat \
              -H 'content-type: application/json' \
              -d '{"model":"qwen3.5:9b-q8_0","messages":[{"role":"user","content":"reply with the single word PONG"}],"stream":false}' \
              | jq -r .message.content
```

Expected: a string containing `PONG`. First-call latency includes
model load (Metal mmap of weights into unified memory) — typically
2–10 s for an 8B model on M-series; subsequent calls are sub-second.

---

## 12. Phase 11 — Install `devai-agent` launcher

```bash
$ cd ${REPO_DIR}
$ make install
$ test -d "${HOME_DIR}/.local/bin" && grep -q "${HOME_DIR}/.local/bin" <<<"${PATH}" || \
      echo 'export PATH="$HOME/.local/bin:$PATH"' >> ${HOME_DIR}/.zshrc
$ devai-agent --init
```

`make install` symlinks `${REPO_DIR}/bin/devai-agent` into
`${HOME_DIR}/.local/bin/devai-agent` and stages config under
`${HOME_DIR}/.devai/`. The launcher is a Python script that runs
`podman run` — it works the same on macOS as on Linux.

### 12.1 Patch `OLLAMA_HOST` for the lab container

`bin/devai-agent` defaults the lab container's `OLLAMA_HOST` to
`http://devai-router:11434`, which on macOS is correct (the router IS
running and routes to host Ollama). No patch needed. If the operator
chose to skip the router (an architectural variant not covered here),
they must export `OLLAMA_HOST=http://host.containers.internal:11434`
before invoking `devai-agent`.

### 12.2 Verification

```bash
[verify] $ command -v devai-agent
[verify] $ devai-agent --show
```

`--show` prints the resolved `podman run` invocation without launching
anything. It must reference `devai-lab-cpu` (NOT `-gpu`).

---

## 13. Phase 12 — End-to-end smoke test

```bash
[verify] $ curl -fsS http://localhost:11434/api/chat \
              -H 'content-type: application/json' \
              -d '{"model":"qwen3.5:9b-q8_0","messages":[{"role":"user","content":"reply with the single word PONG"}],"stream":false}'
[verify] $ curl -k -fsS https://localhost:8443/ -o /dev/null && echo webui_up
$ devai-agent
```

Inside the launched lab shell:

```bash
[verify] $ ollama list
[verify] $ echo 'Tell me a haiku about caches.' | claude --print 2>/dev/null
```

The first `ollama list` reaches host Ollama through the router across
`host.containers.internal`. The second invokes Claude Code with the
default `ANTHROPIC_BASE_URL` pointing at the router → host Ollama.

---

## 14. Unified-memory tuning (advanced, operator decision)

Apple Silicon's GPU normally gets a kernel-managed share of unified
memory — roughly 67 % of total RAM, capped well below 100 % to leave
room for the OS, page cache, and other apps. For very large models
(70B Q4 ≈ 40 GB, 120B Q4 ≈ 65 GB) this default is the bottleneck.
The kernel exposes a tunable that raises the GPU's wired-memory
ceiling:

```bash
[verify] $ sysctl iogpu.wired_limit_mb            # current; 0 = kernel default
$ sudo sysctl iogpu.wired_limit_mb=<value_in_MB>
```

The agent must NOT change this autonomously. It must surface to the
operator with the trade-offs:

- Setting too high (more than ~85 % of total RAM) starves macOS and
  causes UI freezes, Spotlight grinds, and in extreme cases
  kernel-panic-class lock-ups.
- The setting is non-persistent. Persist via `/etc/sysctl.conf`:
  ```
  iogpu.wired_limit_mb=57344
  ```
- It interacts unpredictably with non-LLM GPU workloads (ProRes
  decode, Final Cut, Logic Pro plugins). Do not raise on a workstation
  the operator also uses for media production.
- Apple does not document this knob; behaviour can change in a macOS
  point release. Re-verify after every macOS upgrade.

Recommended starting points (subject to operator confirmation):

| Total RAM | Default GPU ceiling | Suggested `iogpu.wired_limit_mb` | Headroom for system |
|---|---|---|---|
| 16 GB | ~10.6 GB | leave default | 5.4 GB |
| 32 GB | ~21.3 GB | `26624` (26 GB) | 6 GB |
| 36 GB | ~24 GB | `30720` (30 GB) | 6 GB |
| 48 GB | ~32 GB | `40960` (40 GB) | 8 GB |
| 64 GB | ~42.6 GB | `57344` (56 GB) | 8 GB |
| 96 GB | ~64 GB | `86016` (84 GB) | 12 GB |
| 128 GB | ~85 GB | `114688` (112 GB) | 16 GB |
| 192 GB | ~128 GB | `172032` (168 GB) | 24 GB |

These are conservative ceilings; community-reported safe values run
~5 % higher. Validate by running the largest expected model for 10
minutes and watching `Activity Monitor → Memory → Memory Pressure`
stay green.

---

## 15. Memory sizing recommendations per RAM tier

Apple Silicon makes "Will model X fit?" simpler than on discrete GPUs:
the answer is "if the quantised file is smaller than your wired GPU
ceiling minus the KV cache, yes." The KV cache for an 8B model at
128K context is ~3 GB; at 32K it is ~750 MB.

| RAM | Sweet-spot models | Largest practical | Notes |
|---|---|---|---|
| 8 GB | Llama-3.2-3B-Q4, Phi-3-mini-Q4 | 7B Q3 (degraded) | barely useful for serious work; lab + agent CLIs eat 1–2 GB |
| 16 GB | Llama-3.1-8B-Q4, Qwen3-8B-Q4_K_M, Mistral-7B-Q5 | 13B Q4 | the floor for "real" local inference |
| 24 GB (M3 Pro base) | Qwen3-14B-Q4, Llama-3.1-8B-Q8 | 32B Q3 | comfortable single-user dev rig |
| 32 GB | Qwen3-32B-Q4, Mixtral-8x7B-Q3 | 32B Q5 / 47B MoE Q3 | sweet spot for Q4 instruct models |
| 36 GB (M3 Pro Max) | Qwen3-32B-Q4, Llama-3.1-8B-Q8 + KV-heavy workloads | 70B Q2 (poor quality) | as 32 GB plus headroom |
| 48 GB | Qwen3-30B-A3B-Q4, Mixtral-8x7B-Q4 | 70B Q3 | first tier where 70B is on the table |
| 64 GB | Llama-3.1-70B-Q4_K_M, Qwen3-72B-Q4 | 70B Q5 | quality-grade 70B serving |
| 96 GB (M3 Max) | Llama-3.1-70B-Q5_K_M, Mixtral-8x22B-Q4 | 110B Q4 | comfortable two-model + bench |
| 128 GB (M3 Max) | Llama-3.1-70B-Q8, Qwen3-110B-Q4 | 405B Q2 (very degraded) | 70B at near-fp16 quality |
| 192 GB (M2 Ultra / M3 Ultra) | Llama-3.1-405B-Q4 | 405B Q4 | true frontier-class single-host inference |

Caveats:

- These rows assume `iogpu.wired_limit_mb` raised per §11. Default
  ceilings drop one tier.
- MoE active-vs-total parameters mean a 47B Mixtral runs at ~13B
  speed but takes 47B of RAM; budget by RAM, not by speed.
- Long contexts inflate KV cache linearly. A 128K Qwen3-72B-Q4
  context needs ~10 GB for KV alone — drop one row in the table.

---

## 16. State-of-the-world reference

Once Phases 1–13 succeed:

```
macOS host
├── /Applications/                          (untouched by this procedure)
├── /opt/homebrew/                          (Homebrew prefix on Apple Silicon)
│   ├── bin/podman, ollama, python3, ...
│   └── var/log/ollama.log
├── ~/.ollama/                              (Ollama state — REGULAR mac filesystem)
│   ├── models/blobs/                       (GGUF weights; can be huge)
│   └── models/manifests/
├── ~/Library/LaunchAgents/                 (optional persistence agents)
│   └── io.devai.ollama-host.plist          (only if §6.2 was applied)
├── ~/.devai/                               (devai-agent state, same as Linux)
│   ├── preferences.yaml
│   ├── sessions/
│   └── *.json (probe-cache symlinks)
├── ~/.local/bin/devai-agent
└── ${REPO_DIR}/                            (the cloned repo)
    ├── .env
    └── deploy/
        ├── docker-compose.yaml             (Linux-default infra spec)
        └── docker-compose.macos.yaml       (macOS override authored in §8.3)

podman machine VM (${VM_NAME})
├── /var/cache/devai/                       (build/log caches, VM-internal)
│   ├── apt/, npm/, pip/, registry/, logs/
│   └── (NO ollama/ — host-side instead)
└── containers:
    devai-apt-cache, devai-registry-cache,
    devai-router (→ host Ollama),
    devai-open-webui, devai-webui-proxy,
    devai-logger
```

Network flow:

```
Browser (8443)  ─►  devai-webui-proxy  ─►  devai-open-webui  ─►  devai-router
Agent (lab)     ─►  devai-router (in VM)  ─►  host.containers.internal:11434
                                            ─►  macOS host: ollama serve (Metal, UMA)
```

Inference never crosses the VM boundary in the hot path — only the
HTTP request and the streamed response do.

---

## 17. Tear-down

```bash
$ cd ${REPO_DIR}
$ podman compose -f deploy/docker-compose.yaml -f deploy/docker-compose.macos.yaml down --remove-orphans
$ make clean
$ make uninstall
$ brew services stop ollama
$ launchctl unsetenv OLLAMA_HOST
$ launchctl unload ~/Library/LaunchAgents/io.devai.ollama-host.plist 2>/dev/null || true
$ rm -f  ~/Library/LaunchAgents/io.devai.ollama-host.plist
```

To also remove the VM (DESTROYS BUILT IMAGES AND VM-SIDE CACHES):

```bash
$ podman machine stop ${VM_NAME}
$ podman machine rm   ${VM_NAME}
```

To also remove host Ollama and pulled weights:

```bash
$ brew uninstall ollama
$ rm -rf ~/.ollama
```

To revert `iogpu.wired_limit_mb` if §14 was applied:

```bash
$ sudo sysctl iogpu.wired_limit_mb=0
$ sudo sed -i.bak '/^iogpu\.wired_limit_mb/d' /etc/sysctl.conf
```

Do not perform any of these tear-down steps without explicit operator
confirmation.

---

## 18. Decisions left to the operator

Surface and wait for explicit answer:

1. **Apple Silicon vs Intel** (§2.2). On Intel, no UMA, no fast Metal
   for LLMs. The operator should be told the unified-memory bullets
   in §11–§15 do not apply.
2. **VM resource sizing** (§7.1). The trade-off is direct: every GB
   given to the VM is taken away from host Ollama's memory budget.
3. **`iogpu.wired_limit_mb`** (§14). System-stability sensitive; do
   not raise without operator approval. If approved, choose a value
   from §14's table and validate under sustained load.
4. **MLX-LM** (§5). Optional; only install if the operator wants a
   second native engine.
5. **Persistent `OLLAMA_HOST` LaunchAgent** (§6.2). Reasonable to
   install for headless servers, optional for desktops.
6. **Compose override editing** (§8.3) — confirm the operator wants
   the documented "no in-VM ollama, router → host" architecture vs
   the alternative of bypassing the router entirely (lighter; loses
   the reasoning policy and tool-stripping features).
7. **arm64 Makefile patch vs amd64-via-Rosetta VM** (§9.1). Default to
   the patch unless the operator has a reason to want amd64.
8. **Public exposure** — no port listed in §16 should be exposed to
   the public internet without an explicit reverse-proxy + auth layer
   the operator owns.
9. **Tear-down scope** (§17) — the VM, the host weights, and the
   `iogpu` setting are independent; tear-down should ask which apply.

---

## 19. Versioning

This document tracks the repository at the time of writing. When the
following change, regenerate the relevant section:

- `deploy/docker-compose.yaml` service set (§8.3, §10.1, §16).
- `Makefile` `fetch-cli` recipe arch handling (§9.1).
- `bin/devai-agent` `OLLAMA_HOST` default (§12.1).
- macOS unified-memory ceiling behaviour (§14) — re-check after each
  macOS major release.
- Apple Silicon SoC introductions — extend §15 if a new tier ships.

For runtime architecture details and protocol-level behaviour, defer
to `docs/router.md`, `docs/backends.md`, and the in-repo `README.md`.
