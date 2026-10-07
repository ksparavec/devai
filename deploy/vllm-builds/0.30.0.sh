# shellcheck shell=bash disable=SC2034
# Build record: vLLM 0.30.0 with HyperQwen 7af097b and devai's own series
# (platform-rebuild E1, Phase 4 step 4v). Sourced by scripts/build-vllm.sh
# (VLLM_VERSION=0.30.0, its default); it sets nothing else.
#
# Runtime set: 0.30.0.in beside this file (written by build-vllm.sh --write-in:
# the release's requirements/cuda.txt + common.txt, then HyperQwen's
# docker/requirements.txt, without vllm, flashinfer-cubin and index lines),
# hash-locked as 0.30.0.lock by `make vllm-lock VLLM_VERSION=0.30.0`, PyPI alone.
# flashinfer-cubin is not installed: since 0.6.14 it is published only on
# flashinfer.ai's index, and FlashInfer JIT-compiles its kernels with the
# image's nvcc instead (measured on 0.28.0, 2026-09-21; 4v checks it on 0.30.0).

# The PyPI sdist: pypi.org/pypi/vllm/0.30.0/json, read 2026-10-06 (the plan) and
# 2026-10-07, equal to the downloaded file's sha256 (2026-10-07).
VLLM_SDIST_SHA256=5f8f4e890c042ffa1c3e103f81c35e2d96f60a0a175ac43a4adbae7006bef62b

# torch for the build venv; equals the release's requirements/cuda.txt pin, and
# build-vllm.sh refuses a runtime set that pins another.
VLLM_TORCH_PIN=2.13.0

# github.com/syv-ai/HyperQwen: the newest commit on main whose
# docker/requirements.txt still pins vllm==0.30.0 when 4v started -- main's
# head on 2026-10-07, 7af097b of 2026-10-06T17:53:06Z (the 0.30 port is
# d88544b of 2026-09-28; the pin is unchanged since). Upstream applies its whole
# series at this commit (patches/apply.sh; dflash2-backport left the series in
# e1459c7), so nothing is skipped. Its KVarN patches (kvarn/) are not part of
# the series and are not applied (4/2-bit KV was ruled out on quality grounds).
HYPERQWEN_COMMIT=7af097b5ea3acfecd60aabbff5bc457e503b2dc5
HYPERQWEN_SKIP=()

# devai's own series (repository-relative), applied after HyperQwen's.
DEVAI_PATCHES=deploy/vllm-patches

# Read from the 0.30.0 sdist's CMakeLists.txt (cutlass, :487 and :503) and
# cmake/external_projects/, 2026-10-07. Changed since 0.28.0: cutlass v4.7.1,
# flash-attention, FlashMLA, MSA, tml-fa4, FlashKDA and DeepGEMM revisions;
# DeepGEMM moved to the vllm-project fork (vllm#56876); DeepSelect is new
# (CMakeLists.txt:1575, cloned for every CUDA build), and 0.30.0 honours
# DEEPSELECT_SRC_DIR (deepselect.cmake:7-8). qutlass and triton are unchanged.
#
# name | env var vLLM's CMake reads | repository | pinned ref | file naming the
# pin and the repository | optional subdirectory the variable must point at
# (triton: vLLM wants the triton_kernels package directory itself)
EXTERNALS=(
  "cutlass|VLLM_CUTLASS_SRC_DIR|https://github.com/nvidia/cutlass.git|v4.7.1|CMakeLists.txt"
  "flash-attention|VLLM_FLASH_ATTN_SRC_DIR|https://github.com/vllm-project/flash-attention.git|506341a143fcabd4bb79052a7605ada727d6b3f5|cmake/external_projects/vllm_flash_attn.cmake"
  "FlashMLA|FLASH_MLA_SRC_DIR|https://github.com/vllm-project/FlashMLA|0eee43b12f034b657133cf2afca6a72ebb6efccf|cmake/external_projects/flashmla.cmake"
  "MSA|FMHA_SM100_SRC_DIR|https://github.com/vllm-project/MSA.git|f355c37eb4e1413f21ee2ad8bbad25079e6bef9d|cmake/external_projects/fmha_sm100.cmake"
  "tml-fa4|TML_FA4_SRC_DIR|https://github.com/vllm-project/tml-fa4.git|75765e76a9c2c012c1f6ecd64577eb646eb4d303|cmake/external_projects/tml_fa4.cmake"
  "FlashKDA|FLASH_KDA_SRC_DIR|https://github.com/vllm-project/FlashKDA.git|b59532f1f464fbd536272780e30df5bf6a2ccc02|cmake/external_projects/flashkda.cmake"
  "qutlass|QUTLASS_SRC_DIR|https://github.com/IST-DASLab/qutlass.git|e74319e3405ce6d71965732880f5dc1f52371f64|cmake/external_projects/qutlass.cmake"
  "DeepGEMM|DEEPGEMM_SRC_DIR|https://github.com/vllm-project/DeepGEMM.git|e1f418c2a4f20818221f6b0e578b4c2f634d4c3f|cmake/external_projects/deepgemm.cmake"
  "DeepSelect|DEEPSELECT_SRC_DIR|https://github.com/vllm-project/DeepSelect.git|d96d33afe1fab0d6066da49cdc91e64c2bee65ea|cmake/external_projects/deepselect.cmake"
  "triton|TRITON_KERNELS_SRC_DIR|https://github.com/triton-lang/triton.git|v3.5.1|cmake/external_projects/triton_kernels.cmake|python/triton_kernels/triton_kernels"
)
