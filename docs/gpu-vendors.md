# GPU vendors: NVIDIA (default) and AMD/ROCm

devai is NVIDIA-only by default. This doc covers the vendor overlay
that lets a host switch to AMD/ROCm, what's actually been verified
for each vendor, and the manual checklist for the parts that need
real ROCm hardware to confirm.

## The switch

One knob, `DEVAI_GPU_VENDOR` (`nvidia` default, or `amd`), drives four
derived values. Flip it with:

```bash
make gpu-vendor VENDOR=amd     # or VENDOR=nvidia to switch back
```

This runs `devai-tools/cmd/devai-gpu-vendor`, which add-or-replaces
these four keys in `.env` (not a blind `sed` -- a fresh checkout's
`.env` may not have any of them yet):

| `DEVAI_GPU_VENDOR` | `DEVAI_GPU_DEVICE` | `VLLM_IMAGE` | `SGLANG_IMAGE` |
|---|---|---|---|
| `nvidia` (default) | `nvidia.com/gpu=all` | `docker.io/vllm/vllm-openai:v0.22.1-x86_64-cu129-ubuntu2404` | `docker.io/lmsysorg/sglang:v0.5.10.post1-cu130` |
| `amd` | `amd.com/gpu=all` | `docker.io/vllm/vllm-openai-rocm:latest` (**placeholder -- verify current tag**) | `docker.io/lmsysorg/sglang:latest-rocm` (**placeholder -- verify current ROCm-tagged release**) |

The AMD image tags in `devai-tools/cmd/devai-gpu-vendor/main.go`'s
`vendorValues` table are hardcoded constants, not fetched -- this
session had no network access to Docker Hub to confirm the current
`vllm/vllm-openai-rocm` / `lmsysorg/sglang` ROCm release tags (see
"What's actually been verified" below). Check Docker Hub before
relying on them.

### Every place the device string had to change

`nvidia.com/gpu=all` was hardcoded in 9 places across the repo.
`DEVAI_GPU_DEVICE` now covers 7 of them; the other 2 are deliberately
left NVIDIA-only:

```
gpu-arbiter/main.go:2193             <- backend container recreate (buildContainerSpec)
deploy/docker-compose.yaml:60,80,93  <- ollama + vllm + sglang compose `devices:`
Makefile GPU_FLAGS (~line 185)   <- lab-gpu, shell-gpu, + probe/test targets (7 call sites, one variable)
bin/devai-agent gpu_flags()      <- the standalone devai-agent launcher
--- left NVIDIA-only, not in scope this pass ---
scripts/verify-backend-flags.py:45   <- probe/bench harness
scripts/_probe_hf_common.py:614      <- probe/bench harness
```

(Line anchors are from commit 57c4052 and drift with edits -- grep for
`DEVAI_GPU_DEVICE` / `nvidia.com/gpu=all` rather than trusting them.)

The last two are the probe/bench harness itself -- extending them to
ROCm is real work (different flag surface, different failure modes)
that needs a real ROCm host to develop against, not just to test
against. Deferred until one exists.

**`deploy/docker-compose.yaml`** substitutes `DEVAI_GPU_DEVICE` (with
the NVIDIA default as fallback) into the three backend services'
`devices:` lists -- no `DEVAI_GPU_VENDOR` awareness needed at that
layer, just the one derived string.

**`gpu-arbiter`** reads `DEVAI_GPU_DEVICE` in code (`containerRecreate`)
and, **since 2026-07-27, the router service's `environment:` block in
`deploy/docker-compose.yaml` passes it in.**

Until then it did not, and the consequence was silent: a composed router
fell back to the built-in `nvidia.com/gpu=all` whenever it recreated a
backend container, whatever `.env` said. The `amd` overlay reached only
the backends compose started itself -- and since vLLM and SGLang start
as `sleep infinity` placeholders and are *always* recreated by the
router on first request, that meant the overlay never reached the two
services it matters most for. Passing `-e DEVAI_GPU_DEVICE=...` to a
hand-started arbiter was the only path where the arbiter's own read had
any effect.

The router does not inherit the shell that ran compose, so
interpolating `${DEVAI_GPU_DEVICE}` into the *backend* services' device
lists (which compose already did) never reached the router's own
process environment. That is the general shape of this bug, and it bit
twice: the SSE keepalive knobs landed with the same defect on the same
day. `tests/python/test_hf_store_linking.py` now asserts that every
`DEVAI_*` name the router pulls from its environment is forwarded by
compose, so a third instance fails a test instead of shipping.

**`Makefile`'s `GPU_FLAGS`** reads the same `DEVAI_GPU_DEVICE` (via
`.env`, which the Makefile already `-include`s) -- one variable change
fixes `lab-gpu`, `shell-gpu`, and every other target built on
`GPU_FLAGS`.

**`bin/devai-agent`** is different: it deliberately does not read the
repo's `.env` (state lives in `~/.devai/preferences.yaml`, independent
of repo cwd -- see the launcher's own docstring). Its vendor knob is a
`gpu_vendor` preference field instead, set via `--gpu-vendor amd` once
(persists across runs):

```bash
devai-agent --gpu-vendor amd     # first run: sets and persists gpu_vendor
devai-agent                      # later runs: reuses the persisted value
```

Makefile users and `devai-agent` users therefore each flip their own
knob once (`make gpu-vendor VENDOR=amd` vs `devai-agent --gpu-vendor
amd`) -- there's no single switch that covers both launch paths,
because they deliberately don't share config storage.

## The `devai-lab-gpu` image

`make build-gpu`'s base-image selection is also `DEVAI_GPU_VENDOR`-conditional:
`docker.io/rocm/dev-ubuntu-24.04:6.4.3-complete` for `amd`,
the existing `nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04` for
`nvidia` (an explicit `GPU_BASE_IMAGE=...` in `.env`/the shell still
wins over this default either way). `6.4.3`, not the newer `7.2.x`
ROCm line, paired with PyTorch's `rocm6.4` wheel index -- the more
conservative, already-proven pairing for a first cut.

`Dockerfile.lab`'s torch-install step gained a third branch, gated on
a new `ARG GPU_VENDOR=nvidia` (separate from the existing `ARG
GPU_BUILD=false`, which stays "is this a GPU build at all"): AMD gets
`--index-url https://download.pytorch.org/whl/rocm6.4` -- a full
index-url **replacement**, not `--extra-index-url` like the CPU
branch. That distinction matters: `--extra-index-url` still consults
default PyPI first, which could resolve a CUDA wheel on an AMD host.

Two things that turned out to need **no** change, confirmed by reading
the actual Dockerfile logic (not just by lineage reasoning):

- The `render` GID-naming block (~line 186-198) is already
  vendor-neutral -- a build-time `ARG RENDER_GID`, not hardcoded to
  NVIDIA's GID.
- The UID-1000 rename block (~line 200-219) only branches on "does uid
  1000 already exist" (true for both `nvidia/cuda`'s baked-in `ubuntu`
  user and `rocm/dev-ubuntu-24.04`'s, since both are `FROM
  ubuntu:24.04`), never on vendor.

## What's actually been verified

Be precise about this -- the two vendors are not equally proven:

- **NVIDIA**: fully verified. Existing probe/bench data and live
  hardware back every claim in this repo about NVIDIA GPU behavior.
- **AMD/ROCm**: **not build-verified either.** The plan for this
  change called for actually running the ROCm `Dockerfile.base` +
  `Dockerfile.lab` builds on this host to prove they succeed without
  needing real ROCm hardware (an image build never needs the target
  accelerator). That could not be done in this session: the sandbox's
  network policy blocks Docker Hub's CDN
  (`production.cloudfront.docker.com` returns 403 on every `docker.io`
  pull -- confirmed via the build proxy's failure log, a deliberate
  policy denial, not a transient error). `gcr.io` pulls work fine;
  nothing on `docker.io` does, which blocks pulling both
  `rocm/dev-ubuntu-24.04` and the existing `nvidia/cuda` base image
  the same way.
  - What *was* verified: `Dockerfile.lab`'s new `ARG GPU_VENDOR` +
    3-way torch-install branch parses correctly and BuildKit begins
    executing it (confirmed by building with `BASE_IMAGE=scratch` as a
    stand-in -- it fails on `scratch` having no shell, past the point
    where a Dockerfile syntax error would surface, proving the new
    conditional logic is at least well-formed).
  - What was **not** verified: the actual `docker.io/rocm/dev-ubuntu-24.04`
    pull, the ROCm PyTorch wheel install, or anything downstream of
    that.
  - `gpu-arbiter`'s `DEVAI_GPU_DEVICE` unit test and
    `devai-tools/cmd/devai-gpu-vendor`'s round-trip tests are real,
    passing, code-level verification -- independent of the Docker Hub
    block.

Treat the AMD/ROCm path as: code and Dockerfile logic written and
internally consistent, config-flip tooling fully tested, but the
actual image build is unverified pending either network access to
Docker Hub or a host that already has the ROCm base image cached
locally.

## Manual verification checklist (future ROCm host)

Run these on a host with real AMD GPU hardware and unrestricted
network access:

1. `make gpu-vendor VENDOR=amd`
2. `make build-gpu` -- confirm both the base and lab image build to
   completion (expect several GB of image pulls; this alone was
   blocked in the sandbox this feature was developed in, see above).
3. `make lab-gpu` -- confirm the container starts, `--device
   amd.com/gpu=all` attaches without error.
4. Inside the container: `rocm-smi` sees the GPU; `/dev/kfd` and
   `/dev/dri/renderD*` are present (CDI device injection worked).
5. `python -c "import torch; print(torch.cuda.is_available())"` ->
   `True` (ROCm's PyTorch build reuses the `torch.cuda` namespace,
   this is expected and correct, not a CUDA/ROCm mixup).
6. A real `make bench` pass, compared against the NVIDIA baseline in
   `docs/bench-results.md`.
7. Extend `scripts/verify-backend-flags.py` / `scripts/_probe_hf_common.py`
   to ROCm once there's a real host to develop the probe-flag
   differences against.
