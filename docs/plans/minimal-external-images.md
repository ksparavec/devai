# Minimal external images

_Pull exactly one image from upstream -- Debian's `trixie-slim` -- and build everything else here, with every engine compiled for CUDA 13.1 and all backends in one image._

## Status

Draft (2026-09-30). Operator decisions M1-M15 recorded; no open questions.

## Dependencies

None. [Plan: slurm-gatekeeper](./slurm-gatekeeper.md) builds on this one: its jobs start engines in the `devai-engines` image built in Phase 4 here, and its control-plane image uses the Debian pin from Phase 2. `devai-engines` contains no Slurm part (M7). The Slurm documents (slurm-gatekeeper.md, docs/slurm.md) are owned by the session that wrote them and are not edited by this plan.

## Enables / Unblocks

- One Slurm package build instead of one per distribution: every devai image is Debian trixie after Phases 3-4.
- A re-bench that runs once, on final images. The re-bench of 2026-09-27/28 is stopped; if it resumed before Phase 4, every vLLM and SGLang row would be measured twice.
- No floating upstream tags in the running stack. Today `latest` or `main` is what runs for the apt cache, the egress firewall, Open WebUI and 12 MCP servers.
- Five containers fewer: registry-cache, vllm-devai, open-webui, webui-proxy, mcp-gateway (plus up to 12 MCP containers started per call).
- Two images fewer (`devai-base-cpu`, `devai-lab-cpu`), and one set of images for every host: a host without an NVIDIA GPU runs the same images on the CPU (M14, M15).
- About 117 GB of upstream images out of the local store (Phase 5).

## Out of scope

- Slurm itself: the `devai-slurm` image and how its jobs start engines are [Plan: slurm-gatekeeper](./slurm-gatekeeper.md). M7 is recorded here only because it fixes the image count.
- Building PyTorch, FlashInfer, Triton or other generic dependencies from source (M3).
- Upgrading any engine version. Ollama, vLLM 0.28.0 and SGLang 0.5.16 keep their versions; only how they are built changes (M5).
- CPU builds of vLLM or SGLang. Without a GPU, only Ollama and the laya trainer run (M15).
- GPUs other than this host's. The engines are compiled for compute capability 12.0 (sm120) only; a host with another NVIDIA GPU needs a rebuild for its architecture.
- Developer tooling run ad hoc from images (gitleaks, hadolint, golangci-lint, actionlint in docs/security-ci.md). They mirror CI and are not part of the stack.
- Bootstrapping Debian itself from the archive (mmdebstrap) to reach zero pulled images. The trust root would be the same Debian archive keys; it can be added later if an air-gapped host needs it.

## Operator decisions (2026-09-30)

- **M1 -- Only Debian's official image comes from upstream,** pinned by digest. Everything else is built here from Debian packages, from hash-locked PyPI packages, or from source at a pinned tag -- the pattern `devai-ollama` and `vllm-devai` already follow.
- **M2 -- CUDA 13.1 only.** `cuda-nvcc-13-1` is the only nvcc the host's apt sources offer (`apt-cache policy`, 2026-09-30: no candidate for 12.9 or 13.0). A backend that cannot be built and run with it is dropped or upgraded. No stock builds of anything.
- **M3 -- What is compiled here, and where the rest comes from.** Compiled here: the engines and their own CUDA kernel packages -- Ollama, vLLM, and SGLang with its kernel packages. Every other dependency: the Debian package when its version satisfies the engine's pin, otherwise the pinned, hash-locked PyPI wheel (as `scripts/build-vllm.sh` does for `torch==2.13.0` today). In practice nearly all of the ML stack comes from PyPI: Debian main carries torch 2.6.0 CPU-only (SGLang pins `torch==2.11.0` with CUDA) and does not package transformers; FlashInfer and Triton are not in the host's apt sources.
- **M4 -- One vLLM backend:** vLLM 0.28.0 + HyperQwen (today's `vllm-devai`), named `vllm`, on port 11435. Stock vLLM 0.22.1 (CUDA 12.9) is dropped; the `vllm-devai` backend name and port 11437 are retired.
- **M5 -- SGLang stays at 0.5.16 and is compiled here.** Only the build method changes, so a regression points at our build, not at a new release. `deploy/backend-flags.yaml` and the SGLang recovery flags stay valid. A version bump is a separate step.
- **M6 -- All backends in one image, `devai-engines`:** Ollama, vLLM, SGLang. The router and the laya trainer keep their own images.
- **M7 -- All Slurm components in one image, `devai-slurm`, run as one container** (slurmctld, slurmdbd, slurmrestd, slurmd, MariaDB), built by the Slurm plan. **No Slurm in the backends image:** jobs start engines inside `devai-engines` (the operator's answer to the Slurm session, 2026-09-30); the details are that plan's.
- **M8 -- No per-model engine images.** The `image` field in `deploy/recovery-flags.json` (vLLM 0.25.1 for NVIDIA-Nemotron-Nano-9B-v2-NVFP4, the vLLM "gemma" build for diffusiongemma-26B-A4B-it-NVFP4) has no place in one image. A model that does not run on its backend's single version is dropped.
- **M9 -- Open WebUI is dropped,** with its nginx TLS proxy. Browser chat is the dsh web UI, which reaches every backend; Open WebUI only saw the Ollama port.
- **M10 -- MCP is dropped:** the gateway, its per-call servers and devai's own `devai-model-status` server. Deleted, not moved to `attic/`; git history keeps it.
- **M11 -- The AMD/ROCm overlay moves to `attic/`.** It never ran on ROCm hardware, and the rocm6.4 index has no torch for Python 3.14. devai is NVIDIA-only until it is restored.
- **M12 -- The sops/age secrets scaffold moves to `attic/`.** Its only remaining consumer was the MCP gateway's two servers that needed secrets. The Slurm plan can restore it if it wants its JWT key and MariaDB password encrypted.
- **M13 -- One Debian tag, `trixie-slim`,** for every image. The two tags differ by one file: `trixie-slim` carries `/etc/dpkg/dpkg.cfg.d/docker`, which keeps apt from installing docs, man pages, info pages and translations. Man pages inside containers do not matter.
- **M14 -- One lab image, no CPU variants.** `devai-base-cpu` and `devai-lab-cpu` and their targets are dropped. The remaining images lose the `-gpu` suffix (`devai-base`, `devai-lab`), since nothing distinguishes them any more. Only GPU-capable images are built (CUDA torch; the engines compiled for CUDA 13.1).
- **M15 -- GPU if the host has one, otherwise CPU.** One host check decides at launch: at least one NVIDIA GPU usable through CDI -> the containers get the GPU; none -> the same images run on the CPU, which works reasonably on unified-memory hosts. What runs on the CPU follows from the builds: Ollama (its build includes llama.cpp's CPU backend, preset `cpu` in `scripts/build-ollama.sh`) and the laya trainer (torch runs on the CPU). vLLM and SGLang are compiled for CUDA only and are not offered without a GPU. On macOS nothing changes in kind: Ollama runs natively on the host with Metal (INSTALL_macOS.md) and the lab runs in `podman machine`, now from the one lab image built for arm64.

## Open questions

None.

## Context

On 2026-09-30 the operator asked to reduce the external containers to a minimum. The inventory that day (default NVIDIA setup, `attic/` excluded; sizes from `podman images`):

| Upstream image | Used for | Size |
| -------------- | -------- | ---- |
| `sameersbn/apt-cacher-ng:latest` | apt-cache | 87 MB, a personal image built 22 months ago |
| `registry:2` | registry-cache (Docker Hub pull-through) | 26 MB, 2 years old |
| `ghcr.io/open-webui/open-webui:main` | open-webui | 5.06 GB, floating `main` |
| `nginx:alpine` | webui-proxy | 64 MB |
| `quay.io/podman/stable` | logger (`podman --remote logs` only) | 641 MB |
| `ghcr.io/luckypipewrench/pipelock:latest` | pipelock, `make pipelock-ca-init` | 32 MB, floating `latest`; the binary is v3.0.0 (commit 9d2e37d) |
| `vllm/vllm-openai:v0.22.1-x86_64-cu129-ubuntu2404` | vllm backend, `QUANT_IMAGE` | 30.3 GB |
| `vllm/vllm-openai:v0.25.1-x86_64-cu129-ubuntu2404` | one model (recovery-flags `image`) | 24.5 GB |
| `vllm/vllm-openai:gemma-x86_64-cu130` | one model (recovery-flags `image`) | 20.2 GB |
| `lmsysorg/sglang:v0.5.16-cu130` | sglang backend | 30.1 GB |
| `nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04` | base of devai-base-gpu -> lab-gpu, laya trainer | 4.91 GB |
| `golang:1.27-bookworm` | build stage of router, mcp-modelstatus | 871 MB |
| `gcr.io/distroless/static-debian12` | runtime of router, mcp-modelstatus | 3 MB |
| `debian:trixie`, `debian:trixie-slim` | base-cpu; devai-ollama, vllm-devai | `trixie-slim` kept (M1, M13) |

Also: `docker/mcp-gateway:v0.43.3` plus up to 12 `docker.io/mcp/*` images pulled per call, all `latest` (opt-in profile); three AMD images (`rocm/dev-ubuntu-24.04`, `vllm-openai-rocm`, `sglang:latest-rocm`); and four that slurm-gatekeeper Phase 1 planned under its D4, "use a published image" (SchedMD's slurmctld/slurmdbd/slurmrestd, `mariadb:11.8`).

Three things made this more than tidying. The component that decrypts all of the lab's egress (pipelock) ran a floating `latest` from a personal namespace. The registry mirror serves every docker.io pull over plain HTTP with no signature check (its own comment in `deploy/registries.conf`). And the engines sat on two distributions and two CUDA majors -- measured inside the images:

| Engine | Distribution | torch | FlashInfer | Triton | CUDA |
| ------ | ------------ | ----- | ---------- | ------ | ---- |
| stock vLLM 0.22.1 | Ubuntu 24.04 | 2.11.0+cu129 | 0.6.11.post2 | 3.6.0 | 12.9 |
| vllm-devai 0.28.0 | Debian 13 | 2.13.0 | 0.6.16.post3 | 3.7.1 | 13.0 runtime, 13.1 nvcc |
| SGLang 0.5.16 | Ubuntu 24.04 | 2.11.0+cu130 | 0.6.14 | 3.6.0 | 13.0 |

They cannot share one Python environment, and FlashInfer compiles its kernels with nvcc at first use (measured for vllm-devai on 2026-09-21), so stock vLLM would have needed a CUDA 12.9 toolkit the host's apt sources do not carry. That is what M2 and M4 settle.

## Approach

First remove what is no longer wanted (Open WebUI, MCP) and move the unverified or consumer-less parts to `attic/` (AMD overlay, sops/age), so nothing below is rebuilt only to be deleted. Then pin `debian:trixie-slim` by digest as the only pulled image. Infrastructure services move to images built here from Debian packages (apt-cacher-ng, podman) or from source (pipelock); the router is compiled on the host, as Ollama already is, and copied into an empty image. The GPU base moves from NVIDIA's Ubuntu image to Debian, since torch brings its own CUDA 13 libraries, and becomes the only base: the CPU images go, and one host check attaches the GPU when there is one. The three engines become one image, `devai-engines`: Ollama from the host build, vLLM 0.28 from the existing host build, SGLang 0.5.16 from a new host build, each Python engine in its own environment, sharing one CUDA 13.1 toolkit, started through one launcher script. Probe and bench staleness is keyed to each engine's build rather than to the image, so rebuilding one engine does not invalidate the others. Everything is re-probed once, before the re-bench and before the Slurm plan starts engines from the image. Finally, the stack learns to run without a GPU: Ollama and the laya trainer on the CPU, with a RAM-based probe band so the picker has something to offer. A guard test keeps it this way.

End state:

| Image | Built from | Containers |
| ----- | ---------- | ---------- |
| `debian:trixie-slim` | pulled, digest-pinned | -- |
| `devai-engines` | trixie-slim + host builds | today's ollama / vllm / sglang containers; after Slurm, one permanent idle container (Slurm plan) |
| `devai-slurm` | trixie-slim + devai's Slurm packages (all daemons) + Debian MariaDB (Slurm plan) | one |
| `devai-router` | empty image + host-built static binary | router |
| `devai-laya-trainer` | devai-base | laya-trainer |
| `devai-base`, `devai-lab` | trixie-slim | the lab, with or without the GPU |
| `devai-apt-cache`, `devai-logger`, `devai-pipelock` | trixie-slim + Debian package / source | one each |

---

## Phase 1 -- Remove and move to attic

### Goal

Open WebUI and MCP are gone; the AMD overlay and the sops/age scaffold sit in `attic/` with restore notes. No GPU, no probe or bench data touched.

### Deliverables

```
deploy/docker-compose.yaml     modify -- open-webui, webui-proxy, mcp-gateway (+ mcp-state volume) removed
deploy/webui-proxy/            remove
deploy/mcp-catalog-devai.yaml, deploy/mcp-gateway.env, deploy/mcp-secrets.sops.env.example,
deploy/Dockerfile.mcp-modelstatus, scripts/mcp-health.sh
                               remove
devai-tools/cmd/devai-mcp-modelstatus, devai-tools/internal/modelcache, devai-tools/internal/routerclient
                               remove -- routerclient and modelcache have no other importer
tests/test-mcp.sh, tests/test-mcp-modelstatus.sh, tests/python/test_mcp_gateway_phase1.py,
tests/python/test_mcp_gateway_phase2.py, tests/fixtures/modelstatus/
                               remove
docs/mcp.md, docs/mcp-model-status.md
                               remove
attic/amd-rocm/                new    -- devai-tools/cmd/devai-gpu-vendor, devai-tools/internal/envfile
                                         (its only user), tests/test-gpu-vendor.sh,
                                         tests/python/test_devai_agent_gpu_vendor.py, docs/gpu-vendors.md,
                                         the removed Makefile targets and AMD branches, RESTORE.md
attic/sops-age/                new    -- .sops.yaml, scripts/age-keygen-host.sh, scripts/render-secret.sh,
                                         deploy/setup-secrets-tmpfs.sh, docs/secrets.md,
                                         tests/python/test_sops_age_scaffold.py, the removed Makefile targets, RESTORE.md
attic/README.md                modify -- index both
Makefile                       modify -- mcp-*, build-mcp-modelstatus-image, test-mcp-modelstatus, gpu-vendor,
                                         test-gpu-vendor, secrets-*, age-keygen-host removed; AMD branch of
                                         GPU_BASE_IMAGE removed; MCP names out of PIPELOCK_NO_PROXY
deploy/Dockerfile.lab          modify -- ROCm branch of the torch step removed
bin/devai-agent                modify -- --gpu-vendor removed; MCP names out of NO_PROXY_HOSTS
devai-tools/internal/backup    modify -- the age-key entry (manifest.go, restore.go) removed, or it warns on every backup
docs/plans/README.md           modify -- mcp-gateway: Retired; sops-age-secrets: Frozen (attic)
```

### Detailed steps

1. Open WebUI: remove both services. Its data under `/var/cache/devai/open-webui` stays until the operator removes it (it may be its own volume, per the mount-point convention). The mkcert certificates JupyterLab uses are unaffected.
2. MCP: delete the listed files; remove MCP mentions from the pipelock allowlist if any exist only for MCP servers.
3. AMD overlay to `attic/amd-rocm/`, the way `attic/cluster-mode/` was done: files moved, not deleted; `DEVAI_GPU_DEVICE` stays (it is the CDI device string for NVIDIA too). Tests that pin the AMD branches elsewhere (`gpu-arbiter/main_test.go`, `test_lab_python_and_torch.py`, `test_hf_store_linking.py`, `test_ollama_image_cutover.py`) keep only their NVIDIA cases.
4. sops/age to `attic/sops-age/`; `docs/plans/sops-age-secrets.md` becomes Frozen.

### Exit criteria

- `make cache-up` brings the stack up without open-webui, webui-proxy and mcp-gateway.
- `make test-router`, `make test-devai-tools`, `make test-python`, `make test-backup-restore` pass.
- No reference to the removed files outside `attic/` and history docs.

### Phase 1 risks

| Risk | Mitigation |
| ---- | ---------- |
| someone still uses Open WebUI or an MCP client config | operator decisions M9, M10; the dsh web UI covers browser chat |
| devai-backup warns about a missing age key forever | its age-key entry is removed in this phase |

---

## Phase 2 -- Infrastructure built here

### Goal

Every always-on service except the engines runs from an image built here; `debian:trixie-slim` is the only pulled image.

### Deliverables

```
deploy/Dockerfile.apt-cache         new    -- trixie-slim + Debian apt-cacher-ng (3.7.5)
deploy/Dockerfile.logger            new    -- trixie-slim + Debian podman (5.4.2, the host's server version)
scripts/build-pipelock.sh           new    -- host Go build of pipelock at v3.0.0 (commit 9d2e37d), retried fetch, tag checked
deploy/Dockerfile.pipelock          new    -- trixie-slim (CA bundle) + the binary at /pipelock
deploy/Dockerfile.router            modify -- FROM scratch + host-built static binary
deploy/Dockerfile.base              modify -- default BASE_IMAGE = the pinned trixie-slim
deploy/docker-compose.yaml          modify -- localhost/devai-* images; registry-cache removed
deploy/registries.conf              modify -- mirror stanza removed
Makefile                            modify -- DEBIAN_IMAGE digest pin passed to every build; build-apt-cache,
                                              build-logger, build-pipelock; router built with host Go;
                                              pull-images = Debian only; pipelock-ca-init uses the local image;
                                              registry-cache out of CACHE_SERVICES and PIPELOCK_NO_PROXY
tests/python/test_upstream_images.py new   -- the guard (below)
```

### Detailed steps

1. Pin Debian: one Makefile variable, `docker.io/library/debian:trixie-slim@sha256:<digest>`, passed as `BASE_IMAGE` to every build, including `Dockerfile.base` (which moves from `trixie` to `trixie-slim`, M13). Bumping it is a deliberate edit followed by a rebuild of every image.
2. apt-cache and logger images. `logging.sh` is unchanged; Debian's `podman` binary serves `--remote`. Check the apt-cacher-ng user id against the existing `/var/cache/devai/apt` volume and `chown` once if it differs.
3. pipelock from source at the version already running, so behaviour does not change; the bump to a newer release (v3.5.0 is current) is a separate step. The image keeps the paths compose and the healthcheck use (`/pipelock`, `/config/...`).
4. Router: a `CGO_ENABLED=0` build with the host's Go (already required by `build-ollama` and `make test-devai-tools`), copied into `FROM scratch`. The router makes no HTTPS calls (no `https://` URL in `gpu-arbiter/*.go` outside tests) and sets no `USER`.
5. Remove registry-cache; remove the mirror stanza from `deploy/registries.conf` and tell existing hosts to do the same in `~/.config/containers/registries.conf` (INSTALL.md).
6. Guard test: reads `deploy/docker-compose.yaml`, every `deploy/Dockerfile*` and the Makefile's image variables. Allowed: `localhost/devai-*` and the pinned Debian reference. No exceptions. Anything else fails with the file and line.

### Exit criteria

- `make pull-images` pulls only the pinned Debian image.
- `make cache-up` brings the stack up; the logger writes a file for every devai container.
- From the lab, the pipelock checks in docs/pipelock.md behave as documented (allowed host 200, fake secret POST 403).
- An image build's apt traffic is served by apt-cache.
- `make test-router`, `make test-devai-tools`, `make test-python` pass.

### Phase 2 risks

| Risk | Mitigation |
| ---- | ---------- |
| apt-cacher-ng runs as a different uid than the old image | one `chown` of the cache volume, in the step itself |
| the router needs something distroless provided (CA bundle, tzdata, passwd) | the exit criteria run it; fall back to trixie-slim as the runtime base |
| pipelock's upstream image carries files the binary expects | compare the old image's filesystem before switching |

---

## Phase 3 -- One lab image on Debian

### Goal

One base and one lab image, built on `debian:trixie-slim` instead of `nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04`, run with the GPU when the host has one and on the CPU otherwise (M14, M15). The laya trainer follows the base.

### Deliverables

```
scripts/host-gpu.sh             new    -- the host check: prints the CDI device string when at least one
                                          NVIDIA GPU is usable (`nvidia-smi -L` lists one and the CDI spec
                                          names nvidia.com/gpu=all), nothing otherwise; an explicit
                                          DEVAI_GPU_DEVICE overrides it (empty = force CPU)
Makefile                        modify -- devai-base / devai-lab only; build-base-cpu, build-cpu, lab-cpu,
                                          shell-cpu removed; `lab` and `shell` attach the GPU per the host check;
                                          base = the pinned trixie-slim
deploy/Dockerfile.lab           modify -- CPU torch index branch removed; the workaround for the Ubuntu
                                          base's `ubuntu` user at uid 1000 removed
deploy/Dockerfile.laya-trainer  modify -- FROM devai-base; tini (Debian package) as PID 1; default command
                                          an idle loop (Slurm plan)
bin/devai-agent                 modify -- --cpu removed; image devai-lab; GPU per the host check
scripts/model-picker.py, scripts/dsh-web-launcher.sh, packages/jupyter-ai-launchers
                                modify -- messages naming lab-cpu|lab-gpu
INSTALL.md, INSTALL_macOS.md, README.md, AGENTS.md, docs/pipelock.md, docs/skypilot-user-guide.md,
scripts/sky-setup.sh            modify -- one lab image; macOS builds it for arm64
```

### Detailed steps

1. Build `devai-base` and `devai-lab` on trixie-slim with the CUDA torch. torch 2.14.0 from PyPI brings the CUDA 13.0 runtime libraries itself (`laya-trainer/requirements.lock` pins `nvidia-cuda-runtime==13.0.96`), so the old base's system CUDA 12.9 may be unused. If something needs system CUDA, it comes from NVIDIA's debian13 apt repository (the host's CUDA 13.1 packages), never from an image.
2. The host check decides the device for `make lab`, `make shell` and devai-agent. The same image runs in both cases; CUDA torch runs on the CPU when no device is attached.
3. The laya trainer image gets `tini` and an idle default command, so the Slurm plan can run it as a permanent container; the router's current launch overrides the command, as today.
4. macOS: `INSTALL_macOS.md` builds the one lab image natively for arm64 in `podman machine` instead of the CPU image.

### Exit criteria

- On this host: in `devai-lab`, `torch.cuda.is_available()`, a matmul and a cuDNN convolution run on the card; `nvtop` shows it.
- The same image with `DEVAI_GPU_DEVICE=` (forced CPU) starts, torch runs on the CPU, the picker and an agent work against the router.
- `make test-laya-trainer` passes; `make laya-check` passes (GPU window, evicts the teacher about 3 min).
- The arm64 build on a Mac succeeds, or is recorded as not yet verified.

### Phase 3 risks

| Risk | Mitigation |
| ---- | ---------- |
| a lab package links against the base's system CUDA or cuDNN | install the CUDA 13.1 packages from NVIDIA's debian13 repository |
| the hash locks lack aarch64 wheels, so the arm64 build fails | regenerate the locks for both platforms (`make lab-lock`) |
| the CUDA torch makes the lab image larger on hosts that never have a GPU | accepted (M14) |

---

## Phase 4 -- devai-engines

### Goal

One image holds every inference engine, each compiled here for CUDA 13.1 on sm120; the router serves from it; every model is re-probed on it.

### Deliverables

```
scripts/build-sglang.sh              new    -- host build of SGLang 0.5.16 and its kernel packages (M3, M5), like build-vllm.sh
deploy/Dockerfile.engines            new    -- trixie-slim; tini; Ollama, vLLM env, SGLang env, one CUDA 13.1 toolkit,
                                               launcher, per-engine labels; default command an idle loop (Slurm plan)
scripts/devai-engine                 new    -- the launcher: `devai-engine <backend> <args>`
gpu-arbiter/*.go                     modify -- entrypoints via the launcher; one engines image; vllm-devai removed; per-engine drift
scripts/_probe_core.py, _probe_hf_common.py, _probe_load.py, probe-check.py,
scripts/bench/bench_runner.py, _bench_core.py, scripts/bench-sync.py
                                     modify -- per-engine identity instead of image digest
scripts/probe-vllm-devai-reasoning.py remove -- merged into the vllm prober
scripts/model-picker.py, bin/devai-agent, config/*, scripts/generate-catalog.py, deploy/models.yaml,
deploy/recovery-flags.json, deploy/docker-compose.yaml, Makefile, tests/
                                     modify -- vllm-devai -> vllm (M4); `image` field removed (M8)
```

### Detailed steps

1. **4a -- SGLang build spike.** List SGLang 0.5.16's dependencies with compiled code and sort them by M3. Its pins include `sglang-kernel==0.4.5`, `sgl-deep-gemm==0.1.4.post1`, `humming-kernels[cu13]==0.1.10`, `flash-attn-4>=4.0.0b18`, `quack-kernels>=0.6.1` and `flashinfer_python[cu13]==0.6.14` (PyPI metadata); those that are SGLang's own kernel packages are compiled, the rest come from Debian if its version satisfies the pin, else PyPI. Build for sm120 with nvcc 13.1 against the pinned `torch==2.11.0`, install in a trixie-slim environment with the host toolkit, serve one model. The spike also settles how a Debian package satisfies a pin inside the engine's environment (Debian's python3 with system site-packages, or none used). If SGLang cannot be built and no upgrade builds either, it is dropped (M2) and the rest of this phase proceeds without it.
2. **4b -- `scripts/build-sglang.sh`** on the pattern of `build-vllm.sh`: sources sha256-pinned, external repositories fetched once at the revisions the source names, every network step tried 3 times, sm120 only, output a wheelhouse plus a content hash, and a record of where each dependency came from (compiled, Debian, PyPI).
3. **4c -- `Dockerfile.engines`.** One stage per engine, copied into the final stage least-changed first. The CUDA 13.1 compiler pieces are copied once from the host toolkit, as `Dockerfile.vllm` does today (FlashInfer needs nvcc at run time). Each engine gets an image label, `devai.engine.<backend>`, set to its build's content hash (the `DIST_ID` idea already used for vLLM). `tini` is PID 1 and the default command is an idle loop, so the Slurm plan can run the image as one permanent container; the router's current launch overrides the command. The Slurm plan's own scripts (`devai-run`, `devai-kill`) are added to this image and to the laya trainer image by that plan.
4. **4d -- Launcher.** `devai-engine <backend>` sets `PATH`, `CUDA_HOME` and `LD_LIBRARY_PATH` for that engine and `exec`s it. The router's `vllmEntrypoint` / `sglangEntrypoint` (`gpu-arbiter/main.go:1381`, `:1523`) and Ollama's launch call it instead of a bare `python3`; so do the probers. One `ENGINES_IMAGE` replaces `OLLAMA_IMAGE`, `VLLM_IMAGE`, `VLLM_DEVAI_IMAGE` and `SGLANG_IMAGE`.
5. **4e -- `vllm-devai` -> `vllm`.** Router backend table, compose, the picker's backend lists and agent provider sets (`router-vllm-devai`), the probers, derived catalog rows (`backend: [vllm]`), tests that pin the vllm-devai registration points. Port 11437 retired.
6. **4f -- Per-engine staleness.** Today the HF probe caches stamp `_meta.current_image_digest`, and `make probe-check`, the router's drift check (`probeCachePathByBackend`, `readProbedImageDigest`) and `make bench-plan` (`stale_image`) compare it. With one image for every backend, rebuilding it for an Ollama change would mark every vLLM and SGLang cell and bench row stale. Stamp and compare the engine's `devai.engine.<backend>` label instead; Ollama's cache gets the same stamp (it has none today).
7. **4g -- Re-probe everything** on the new image: `make probe`, `probe-vllm`, `probe-sglang`, `probe-load-vllm`, `probe-load-sglang`. Failures go to the exclusion ledger and the model is dropped (M8), including the two models that had their own images. `make backup-create` first.
8. **4h -- `QUANT_IMAGE`** points at `devai-engines` (vLLM environment); `scripts/quant/quant_smoke.py` already adds llm-compressor on top of the image's torch.

### Exit criteria

- `make probe-check` is clean against `devai-engines`; the picker lists re-probed rows only.
- A picker-launched agent completes a turn on Ollama, vLLM and SGLang (if kept).
- `make test-router`, `make test-python`, `make test-vllm`, `make test-sglang`, `make test-e2e` pass.
- Rebuilding only the Ollama part leaves `make probe-check` clean for vLLM and SGLang.
- Done before the re-bench resumes and before slurm-gatekeeper starts engines from the image.

### Phase 4 risks

| Risk | Mitigation |
| ---- | ---------- |
| SGLang does not build or run with nvcc 13.1 on sm120 | 4a answers it first; drop or upgrade (M2) |
| the upstream images carried components the source builds lack (SGLang's recovery flags for Gemma-4 rely on `flashinfer_cutlass`) | the re-probe finds them; the model is dropped or the component is added to the build |
| models lost to the single vLLM version (Nemotron-Nano-9B-v2, diffusiongemma) | accepted (M8); the re-probe lists them |
| a rebuild of one engine rewrites the layers after it | order stages least-changed first; vLLM (patched most often) last |
| the image grows past what the layer store handles well | measure; vllm-devai alone is 9.76 GB and SGLang's size built this way is unknown |
| a full re-probe takes GPU time the operator wants elsewhere | schedule it in the window the stopped re-bench would have used |

---

## Phase 5 -- Running without a GPU

### Goal

On a host without an NVIDIA GPU the stack comes up from the same images: Ollama serves on the CPU, the laya trainer trains on the CPU, and vLLM and SGLang say that they need a GPU (M15).

### Deliverables

```
deploy/docker-compose.yaml        modify -- the `devices:` entries move out
deploy/compose.gpu.yaml           new    -- the GPU device for the engines and the laya trainer; `make cache-up`
                                            adds this file when the host check finds a GPU
gpu-arbiter/*.go                  modify -- empty DEVAI_GPU_DEVICE = CPU mode: engines and trainer created without
                                            a device; vLLM and SGLang requests refused with an error naming the reason
scripts/probe-ollama-reasoning.py, Makefile (probe)
                                  modify -- a `cpu` band: models loaded with no GPU layers, resident memory measured
                                            against the host's RAM, cells stamped with the band
scripts/model-picker.py           modify -- without a GPU: reads the `cpu` band, hides vLLM and SGLang rows
tests/                            modify -- CPU mode of the router, picker and probe band
```

### Detailed steps

1. Device handling: compose cannot make a device conditional, so the device list moves to `deploy/compose.gpu.yaml`, which `make cache-up` includes only when the host check prints a device.
2. Router: today an unset `DEVAI_GPU_DEVICE` falls back to `nvidia.com/gpu=all` for every container it recreates. It changes to "exactly what compose hands over", and empty means CPU mode.
3. CPU probe band: Ollama on the CPU keeps weights and KV cache in RAM. The band's budget is the host's RAM minus a named reserve for everything else on the host (a unified-memory host shares that RAM with the GPU it may have). The probe decides each model's context as on the GPU; the compose default `OLLAMA_CONTEXT_LENGTH=262144` is not assumed to fit.
4. Picker: without a GPU it reads the `cpu` band. The bench stays GPU-only, so CPU rows show no bench columns.

### Exit criteria

- On this host with the GPU forced off (`DEVAI_GPU_DEVICE=`): `make cache-up`; `make probe` fills the `cpu` band; the picker lists Ollama rows only; an agent completes a turn on the CPU; a request on port 11435 or 11436 returns the no-GPU error.
- `make test-laya-trainer` passes without a GPU (its tests already run on the CPU).
- With the GPU present, nothing changes: `make test-router`, `make test-python`, `make test-e2e` pass.

### Phase 5 risks

| Risk | Mitigation |
| ---- | ---------- |
| CPU inference is slow on hosts without unified memory | accepted; M15 targets unified-memory hosts |
| the Slurm plan assumes a GPU resource that a CPU host does not have | raised with the Slurm plan; its jobs must not request a GPU there |
| a forced-CPU run on this host is not the same as a real unified-memory host | recorded as such; verified on real hardware when one is available |

---

## Phase 6 -- Cleanup

Remove `deploy/Dockerfile.ollama` and `deploy/Dockerfile.vllm` (merged into `Dockerfile.engines`) and the vllm-devai prober. List the retired images with their sizes -- the upstream ones and the old `devai-base-cpu`, `devai-lab-cpu`, `devai-base-gpu`, `devai-lab-gpu`, `devai-ollama`, `vllm-devai` -- and remove them from the local store only after the operator confirms: the store is shared with other projects on this host. Update CLAUDE.md, docs/backends.md, docs/router.md, docs/pipelock.md, INSTALL.md. Exit: the guard test passes with no exceptions; `make build` from a store holding only the pinned Debian image succeeds.

---

## Combined risk register

| Risk | Phase | Mitigation |
| ---- | ----- | ---------- |
| every upstream bump becomes a rebuild here | all | accepted; same as Ollama and vLLM today |
| security fixes in Debian arrive only when the digest is bumped | all | deliberate periodic bump and rebuild, documented next to the pin |
| per-engine staleness code touches probers, bench and router at once | 4 | one change with tests; the Ollama-only rebuild exit criterion |
| renaming the lab images (`-gpu` suffix dropped) breaks scripts and habits that name them | 3 | every reference in the tree changes in the same phase; the old names leave the store only in Phase 6 |

## Migration / rollback story

- Phase 1: MCP and Open WebUI come back by reverting the commit; the AMD overlay and sops/age by their `attic/` RESTORE.md.
- Phases 2-3: the compose and Makefile image variables can point back at the upstream images and the old lab images, which stay in the local store until Phase 6.
- Phase 4: `make backup-create` before the re-probe; rollback is restoring the probe caches and pointing the image variable back. Removing `vllm-devai` changes agent provider names (`router-vllm-devai`); the picker rewrites those sets at every launch.
- Phase 5: CPU mode is only entered when the host check finds no GPU (or `DEVAI_GPU_DEVICE=` forces it), so a GPU host behaves as before.
- Existing hosts delete the registry mirror stanza from `~/.config/containers/registries.conf`.

## Estimated effort

| Phase | Wall-clock |
| ----- | ---------- |
| 1 | 1 day |
| 2 | 1-2 days |
| 3 | 1-2 days + a GPU window |
| 4 | 4-7 days + a full re-probe |
| 5 | 2-3 days |
| 6 | 1 day |

Estimates, not measurements.

## References

- [Plan: slurm-gatekeeper](./slurm-gatekeeper.md) -- D4, D6; under revision 2026-09-30
- attic/README.md, attic/cluster-mode/RESTORE.md -- the pattern for moving work to the attic
- scripts/build-vllm.sh, scripts/build-ollama.sh, deploy/Dockerfile.vllm, deploy/Dockerfile.ollama -- the host-build pattern
- pipelock: github.com/luckyPipewrench/pipelock (Go, Apache-2.0)
- SGLang 0.5.16 package metadata on PyPI (pinned dependencies)
- NVIDIA CUDA repository for Debian 13 (the host's CUDA 13.1 packages)
