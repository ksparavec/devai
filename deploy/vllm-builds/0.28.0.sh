# shellcheck shell=bash disable=SC2034
# Build record: vLLM 0.28.0 with HyperQwen c0c81bb and devai's own series.
# Sourced by scripts/build-vllm.sh (VLLM_VERSION=0.28.0); it sets nothing else.
#
# These are the values build-vllm.sh hardcoded before platform-rebuild Phase 4
# step 4v, unchanged, so a 0.28.0 build is the build that made
# docker.io/devai/vllm-devai:v0.28.0-cu131-debian13-hq.c0c81bb. One thing differs
# from that build: the runtime set is now the hash lock beside this file
# (0.28.0.in -> 0.28.0.lock, `make vllm-lock VLLM_VERSION=0.28.0`), where it used
# to be HyperQwen's docker/requirements.txt with transitive dependencies resolved
# freely at fetch time. Kept as the rollback of 4v (E1); 0.28.0's wheelhouse
# built before 4v serves until 4g passes.

# The PyPI sdist (pypi.org/pypi/vllm/0.28.0/json; build-vllm.sh verifies it).
VLLM_SDIST_SHA256=ac96dd0ec5be9c13f2aa4bfb50498c4727110406f6e4980b58a4097e4d18634a

# torch for the build venv; equals the release's requirements/cuda.txt pin, and
# build-vllm.sh refuses a runtime set that pins another.
VLLM_TORCH_PIN=2.13.0

# github.com/syv-ai/HyperQwen at the commit that pins vllm==0.28.0, and the
# patches of its series this record skips, as HyperQwen's own Dockerfile did:
# dflash2-backport is native in 0.28.0.
HYPERQWEN_COMMIT=c0c81bbbbf91f11b54af7b95f49bd6d1570c2ba0
HYPERQWEN_SKIP=(dflash2-backport.patch)

# devai's own series (repository-relative), applied after HyperQwen's. One
# directory serves both records: its patch is dry-run at --fuzz 0 against each.
DEVAI_PATCHES=deploy/vllm-patches

# name | env var vLLM's CMake reads | repository | pinned ref | file naming the
# pin and the repository | optional subdirectory the variable must point at
# (triton: vLLM wants the triton_kernels package directory itself)
EXTERNALS=(
  "cutlass|VLLM_CUTLASS_SRC_DIR|https://github.com/nvidia/cutlass.git|v4.4.2|CMakeLists.txt"
  "flash-attention|VLLM_FLASH_ATTN_SRC_DIR|https://github.com/vllm-project/flash-attention.git|f3e1a4f74c99145c0717709860bf765de1703779|cmake/external_projects/vllm_flash_attn.cmake"
  "FlashMLA|FLASH_MLA_SRC_DIR|https://github.com/vllm-project/FlashMLA|a8f794d1251cbfd88a5011445dd5582289c727e4|cmake/external_projects/flashmla.cmake"
  "MSA|FMHA_SM100_SRC_DIR|https://github.com/vllm-project/MSA.git|087c161814d4d9c735b46c21212a09e5f8eb92fa|cmake/external_projects/fmha_sm100.cmake"
  "tml-fa4|TML_FA4_SRC_DIR|https://github.com/vllm-project/tml-fa4.git|b206834606ed5b5f21f8eed6b0683f528ea9cf7d|cmake/external_projects/tml_fa4.cmake"
  "FlashKDA|FLASH_KDA_SRC_DIR|https://github.com/vllm-project/FlashKDA.git|053de1b716ef3255873e02d2d28f4adf09951978|cmake/external_projects/flashkda.cmake"
  "qutlass|QUTLASS_SRC_DIR|https://github.com/IST-DASLab/qutlass.git|e74319e3405ce6d71965732880f5dc1f52371f64|cmake/external_projects/qutlass.cmake"
  "DeepGEMM|DEEPGEMM_SRC_DIR|https://github.com/deepseek-ai/DeepGEMM.git|8b1392b978f5a03c828dd1711090d7fb50958b8a|cmake/external_projects/deepgemm.cmake"
  "triton|TRITON_KERNELS_SRC_DIR|https://github.com/triton-lang/triton.git|v3.5.1|cmake/external_projects/triton_kernels.cmake|python/triton_kernels/triton_kernels"
)
