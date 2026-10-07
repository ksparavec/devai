# AMD/ROCm GPU-vendor overlay -- restore notes

Moved here on 2026-10-07 by platform-rebuild Phase 1, operator decision
M11: "The AMD/ROCm overlay moves to `attic/`. It never ran on ROCm
hardware, and the rocm6.4 index has no torch for Python 3.14. devai is
NVIDIA-only until it is restored."

Read this before thawing anything here. The overlay was frozen in an
unverified state: its Go tool and its tests pass, but nothing in it has
ever run against an AMD GPU, and parts of it could not build at freeze
time.

## State at freeze time

1. **Never run on ROCm hardware, and never even built.** `docs/gpu-vendors.md`
   ("What's actually been verified") records that the ROCm base image was
   never pulled and the ROCm torch wheel never installed.
2. **The lab's ROCm torch branch cannot build.** It installs from PyTorch's
   `rocm6.4` index, which has no cp314 torch at all; the lab image is on
   Python 3.14.7 (checked 2026-09-27; `rocm7.2` had 2.14.0 then).
3. **The AMD image tags are placeholders.** `vendorValues` in
   `devai-tools/cmd/devai-gpu-vendor/main.go` names
   `vllm/vllm-openai-rocm:latest` and `lmsysorg/sglang:latest-rocm`; nobody
   checked that either exists.
4. **Two call sites were never covered:** `scripts/verify-backend-flags.py`
   and `scripts/_probe_hf_common.py` hard-code `nvidia.com/gpu=all`.
5. **The overlay switches images that the platform rebuild removes.** It
   swapped the stock `VLLM_IMAGE` / `SGLANG_IMAGE` and the CUDA base image.
   Platform-rebuild M2-M6 replace those with one `devai-engines` image
   compiled here for CUDA 13.1. After that rebuild, ROCm means building the
   engines for ROCm, not flipping an image tag.

## What was moved here

```
amd-rocm/
  devai-tools/cmd/devai-gpu-vendor/   main.go, main_test.go
  devai-tools/internal/envfile/       envfile.go, envfile_test.go (its only user)
                                      -- all four tagged //go:build devai_frozen_amd
  tests/test-gpu-vendor.sh
  tests/python/test_devai_agent_gpu_vendor.py
  docs/gpu-vendors.md
  Makefile.frozen-targets   gpu-vendor, test-gpu-vendor, build-gpu-vendor-tool,
                            the GPU_BASE_IMAGE branch, build-gpu's GPU_VENDOR
                            arg, and the names in .PHONY / build-devai-tools /
                            the `test` aggregate, verbatim
  frozen-branches.txt       the AMD branches cut out of live files, verbatim:
                            deploy/Dockerfile.lab (ARG GPU_VENDOR, the ROCm
                            torch branch), bin/devai-agent (--gpu-vendor, the
                            gpu_vendor preference), .env.example, comments in
                            compose and requirements-torch.txt, and the AMD
                            test cases of gpu-arbiter/main_test.go and
                            tests/python/test_lab_python_and_torch.py
```

`attic/` sits outside every Go module, and the build tag keeps the files
out of any build that does include them.

## What stayed live

- `DEVAI_GPU_DEVICE`: it is the CDI device string for NVIDIA too. The
  Makefile still exports it (default `nvidia.com/gpu=all`) and uses it in
  `GPU_FLAGS`; compose still interpolates it into the backend services'
  `devices:` and forwards it to the router; `gpu-arbiter`'s
  `buildContainerSpec` still reads it. `TestBuildContainerSpecGPUDevice`
  now tests the override with `nvidia.com/gpu=0`.
- `bin/devai-agent` reads an old `~/.devai/preferences.yaml` that still has
  a `gpu_vendor` key and drops it (`RETIRED_PREF_KEYS`), so the next save
  omits it. Pinned by `tests/python/test_devai_agent_nvidia_only.py`.

## How to thaw

1. Fix items 2, 3 and 5 above first, on paper: a ROCm torch index with
   cp314 wheels, real image tags (or ROCm engine builds), and a ROCm host
   to test on.
2. Move the files back, then strip the `//go:build devai_frozen_amd` line
   and the blank line after it from each of the four Go files:

   ```
   git mv attic/amd-rocm/devai-tools/cmd/devai-gpu-vendor devai-tools/cmd/
   git mv attic/amd-rocm/devai-tools/internal/envfile devai-tools/internal/
   git mv attic/amd-rocm/tests/test-gpu-vendor.sh tests/
   git mv attic/amd-rocm/tests/python/test_devai_agent_gpu_vendor.py tests/python/
   git mv attic/amd-rocm/docs/gpu-vendors.md docs/
   ```

3. Paste the Makefile pieces back from `Makefile.frozen-targets`, and the
   live-file branches from `frozen-branches.txt`. In `bin/devai-agent`,
   take `gpu_vendor` out of `RETIRED_PREF_KEYS`; then
   `tests/python/test_devai_agent_nvidia_only.py`, which asserts there is
   no `--gpu-vendor`, has to go or change.
4. `make test-devai-tools`, `bash tests/test-gpu-vendor.sh`,
   `make test-python`; then the manual checklist at the end of
   `docs/gpu-vendors.md`, on the ROCm host.

The history before the move: `git log --follow -- attic/amd-rocm/docs/gpu-vendors.md`.
