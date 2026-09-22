#!/usr/bin/env bash
# Compile vLLM FROM SOURCE on this host, with the HyperQwen patch series
# applied, for this machine's GPU only. Output: a wheelhouse that
# deploy/Dockerfile.vllm installs offline into debian:trixie-slim. No upstream
# vLLM image is used anywhere, not even as a build input (operator decision,
# same as devai-ollama -- see scripts/build-ollama.sh).
#
# What is built, and why it is shaped this way:
#
#   * vLLM $VLLM_VERSION from the PyPI sdist, sha256-verified. HyperQwen
#     (github.com/syv-ai/HyperQwen) pins exactly this release and applies its
#     patches with `--fuzz 0`, so the version is not a free choice.
#   * The HyperQwen series is applied to the SOURCE TREE before compiling, in
#     the order patches/series gives (a few patches carry context an earlier
#     one adds, so glob order is wrong). `dflash2-backport` is skipped, as
#     HyperQwen's own Dockerfile does: it is native in 0.28.0. Their KVarN
#     KV-cache plugin is NOT installed -- 4/2-bit KV was ruled out here on
#     quality grounds (see docs/backends.md).
#   * devai's own patches (deploy/vllm-patches/series) are applied after
#     HyperQwen's, same rules. The first one stops the MTP drafter allocating
#     two vocabulary-sized layers vLLM deletes anyway -- the 0.5 GiB shortfall
#     that kept MTP from loading on 24 GiB.
#     All 38 patches touch Python only. Compiling therefore produces the same
#     kernels the stock wheel ships, minus every GPU generation but ours.
#   * TORCH_CUDA_ARCH_LIST=12.0 (Blackwell): native code for this card, no
#     other architecture compiled.
#   * vLLM's CMake clones nine external kernel repositories DURING CONFIGURE,
#     outside any retry. They are fetched here instead, each pinned to the
#     revision the source tree names (a drifted pin is refused), and handed to
#     CMake through the *_SRC_DIR variables vLLM already honours.
#   * The Rust front end's crates are fetched up front for the same reason.
#
# Every network step is tried 3 times and then FAILS the script (operator
# rule: every failure is an error, and a download is repeated 3 times before
# bailing out). A failed fetch leaves no directory behind.
#
# Host prerequisites -- Debian packages, installed once by the operator:
#   cmake ninja-build gcc g++ git patch python3 python3-dev        (Debian main)
#   cuda-nvcc-13-1 cuda-cudart-dev-13-1 cuda-cccl-13-1 libcublas-dev-13-1
#   libcusparse-dev-13-1 libcusolver-dev-13-1 libcufft-dev-13-1
#   libcurand-dev-13-1 cuda-nvrtc-dev-13-1 libnvjitlink-dev-13-1
#   libcufile-dev-13-1 cuda-nvtx-13-1 cuda-profiler-api-13-1  (NVIDIA debian13)
# plus uv (creates the build venv; python3-venv is not needed) and a rustup
# toolchain matching the source tree's rust-toolchain.toml.
#
# Builds under ~/.cache/devai/vllm-build, NOT /var/cache/devai (a new
# top-level directory there is not volume-backed).
set -euo pipefail

VLLM_VERSION="${VLLM_VERSION:-0.28.0}"
VLLM_SDIST_SHA256="${VLLM_SDIST_SHA256:-ac96dd0ec5be9c13f2aa4bfb50498c4727110406f6e4980b58a4097e4d18634a}"
HYPERQWEN_REPO="https://github.com/syv-ai/HyperQwen.git"
HYPERQWEN_COMMIT="${HYPERQWEN_COMMIT:-c0c81bbbbf91f11b54af7b95f49bd6d1570c2ba0}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.1}"
CUDA_ARCH_LIST="${CUDA_ARCH_LIST:-12.0}"
BUILD_ROOT="${VLLM_BUILD_ROOT:-$HOME/.cache/devai/vllm-build}"
# nvcc peaks at several GiB per translation unit on the attention kernels;
# 16 keeps a 64 GiB host out of swap where $(nproc) does not.
JOBS="${JOBS:-16}"
FETCH_RETRY_DELAY="${FETCH_RETRY_DELAY:-5}"
PYTHON="${PYTHON:-/usr/bin/python3}"
DEVAI_PATCH_DIR="${DEVAI_PATCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/deploy/vllm-patches}"

# name | env var vLLM's CMake reads | repository | pinned ref | file naming the pin
# | optional subdirectory the variable must point at (triton: vLLM wants the
#   triton_kernels package directory itself, not the repository root)
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

die() { echo "error: $*" >&2; exit 1; }
say() { echo "==> $*"; }

retry3() {
    local label="$1"; shift
    local n=1
    until "$@"; do
        if [ "$n" -ge 3 ]; then
            echo "error: $label failed after 3 attempts" >&2
            return 1
        fi
        echo "    $label failed (attempt $n/3); retrying in ${FETCH_RETRY_DELAY}s" >&2
        n=$((n + 1))
        sleep "$FETCH_RETRY_DELAY"
    done
}

# One attempt at fetching exactly `ref` (tag or full commit id) with its
# submodules, shallow. Builds into "$dest.partial" so an interrupted fetch
# never looks like a finished one.
fetch_ref() {
    local repo="$1" ref="$2" dest="$3"
    rm -rf "$dest.partial"
    mkdir -p "$dest.partial"
    ( cd "$dest.partial" \
        && git init --quiet \
        && git remote add origin "$repo" \
        && git fetch --quiet --depth 1 origin "$ref" \
        && git checkout --quiet FETCH_HEAD \
        && git submodule --quiet update --init --recursive --depth 1 ) || {
        rm -rf "$dest.partial"; return 1; }
    echo "$ref" > "$dest.partial/.devai-ref"
    mv "$dest.partial" "$dest"
}

fetch_source() {
    local label="$1" repo="$2" ref="$3" dest="$4"
    if [ -f "$dest/.devai-ref" ]; then
        say "$label ${ref:0:12} already fetched"
    else
        rm -rf "$dest"
        say "fetching $label ${ref:0:12}"
        retry3 "fetching $label" fetch_ref "$repo" "$ref" "$dest"
    fi
    local have
    have="$(cat "$dest/.devai-ref")"
    [ "$have" = "$ref" ] || die "$dest is at '$have', expected $label $ref"
}

fetch_sdist() {
    local tarball="$BUILD_ROOT/src/vllm-$VLLM_VERSION.tar.gz"
    mkdir -p "$BUILD_ROOT/src"
    if [ ! -f "$tarball" ]; then
        say "fetching vLLM $VLLM_VERSION sdist"
        retry3 "fetching the vLLM sdist" "$PYTHON" -m pip download "vllm==$VLLM_VERSION" \
            --no-binary=:all: --no-deps --no-build-isolation --no-cache-dir -q \
            -d "$BUILD_ROOT/src"
    fi
    local have
    have="$(sha256sum "$tarball" | cut -d' ' -f1)"
    [ "$have" = "$VLLM_SDIST_SHA256" ] \
        || die "$tarball sha256 is $have, expected $VLLM_SDIST_SHA256"
}

fetch_wheels() {
    # Runtime set: exactly what HyperQwen pins. Build set: what pyproject's
    # build-system asks for and the runtime set does not already carry.
    mkdir -p "$BUILD_ROOT/wheels" "$BUILD_ROOT/wheels-build"
    say "fetching runtime wheels (HyperQwen docker/requirements.txt)"
    retry3 "fetching runtime wheels" "$PYTHON" -m pip download \
        -r "$BUILD_ROOT/HyperQwen/docker/requirements.txt" -d "$BUILD_ROOT/wheels" \
        --only-binary=:all: --no-cache-dir --progress-bar off -q
    say "fetching build-only wheels"
    retry3 "fetching build wheels" "$PYTHON" -m pip download \
        "setuptools-scm>=8" "setuptools-rust>=1.9.0" wheel build \
        -d "$BUILD_ROOT/wheels-build" --only-binary=:all: --no-cache-dir \
        --progress-bar off -q
}

preflight() {
    local t f
    for t in git cmake ninja gcc g++ patch uv cargo sha256sum "$PYTHON"; do
        command -v "$t" >/dev/null || die "'$t' not found on PATH"
    done
    [ -x "$CUDA_HOME/bin/nvcc" ] || die "no nvcc at $CUDA_HOME/bin/nvcc (set CUDA_HOME)"
    for f in cublas_v2.h cusparse.h cusolverDn.h cufft.h curand.h nvrtc.h \
             nvJitLink.h cufile.h cuda_profiler_api.h nvtx3/nvToolsExt.h; do
        [ -f "$CUDA_HOME/include/$f" ] \
            || die "no $f under $CUDA_HOME/include (see the package list at the top of this script)"
    done
}

# Refuse to build against a pin the source tree no longer names: the table
# above is only correct for the vLLM release it was written for.
check_pins() {
    local src="$1" row name ref file
    for row in "${EXTERNALS[@]}"; do
        IFS='|' read -r name _ _ ref file _ <<< "$row"
        grep -q -- "$ref" "$src/$file" \
            || die "$name pin $ref is not named in $file -- update EXTERNALS for vLLM $VLLM_VERSION"
    done
}

# Apply one patch series to the vLLM package directory. `--fuzz 0`: a patch
# that needs fuzz to apply is a patch written for different code.
apply_series() {
    local src="$1" dir="$2" label="$3" skip="${4:-}" name
    [ -f "$dir/series" ] || die "no series file in $dir"
    sed -e 's/#.*//' -e 's/^[[:space:]]*//;s/[[:space:]]*$//' -e '/^$/d' "$dir/series" \
    | while IFS= read -r name; do
        if [ -n "$skip" ] && [ "$name" = "$skip" ]; then
            echo "    skip $name (native in vLLM $VLLM_VERSION)"
            continue
        fi
        echo "    $name"
        patch -p1 --fuzz 0 --no-backup-if-mismatch --quiet -d "$src/vllm" < "$dir/$name" \
            || die "$label/$name did not apply cleanly to vLLM $VLLM_VERSION"
        echo "$label/$name" >> "$src/PATCHES.applied"
    done
}

apply_patches() {
    local src="$1"
    : > "$src/PATCHES.applied"
    apply_series "$src" "$BUILD_ROOT/HyperQwen/patches" hyperqwen dflash2-backport.patch
    grep -q '^hyperqwen/' "$src/PATCHES.applied" \
        || die "no patch was applied -- empty patches/series?"
    apply_series "$src" "$DEVAI_PATCH_DIR" devai
}

main() {
    preflight
    mkdir -p "$BUILD_ROOT/ext"
    fetch_sdist
    fetch_source "HyperQwen" "$HYPERQWEN_REPO" "$HYPERQWEN_COMMIT" "$BUILD_ROOT/HyperQwen"
    fetch_wheels

    local src="$BUILD_ROOT/src/vllm-$VLLM_VERSION"
    say "unpacking a pristine source tree"
    rm -rf "$src"
    tar -xzf "$BUILD_ROOT/src/vllm-$VLLM_VERSION.tar.gz" -C "$BUILD_ROOT/src"
    check_pins "$src"

    local row name var repo ref sub
    for row in "${EXTERNALS[@]}"; do
        IFS='|' read -r name var repo ref _ sub <<< "$row"
        fetch_source "$name" "$repo" "$ref" "$BUILD_ROOT/ext/$name"
        [ -d "$BUILD_ROOT/ext/$name/$sub" ] || die "$name has no '$sub' directory"
        export "$var=$BUILD_ROOT/ext/$name${sub:+/$sub}"
    done

    say "fetching Rust crates"
    ( cd "$src/rust" && retry3 "cargo fetch" cargo fetch --locked --quiet )

    say "applying the HyperQwen series, then devai's own (--fuzz 0)"
    apply_patches "$src"

    say "creating the build venv"
    rm -rf "$BUILD_ROOT/venv"
    uv venv --quiet --python "$PYTHON" "$BUILD_ROOT/venv"
    # shellcheck disable=SC1091
    . "$BUILD_ROOT/venv/bin/activate"
    uv pip install --quiet --offline --no-index \
        --find-links "$BUILD_ROOT/wheels" --find-links "$BUILD_ROOT/wheels-build" \
        "torch==2.13.0" numpy ninja packaging "setuptools>=77.0.3,<81.0.0" \
        "setuptools-scm>=8" "setuptools-rust>=1.9.0" wheel jinja2 regex build protobuf

    say "compiling vLLM $VLLM_VERSION for sm$CUDA_ARCH_LIST (-j$JOBS); this takes a while"
    export PATH="$CUDA_HOME/bin:$PATH" CUDA_HOME
    export TORCH_CUDA_ARCH_LIST="$CUDA_ARCH_LIST" MAX_JOBS="$JOBS" NVCC_THREADS=1
    export VLLM_TARGET_DEVICE=cuda CARGO_NET_OFFLINE=true
    export SETUPTOOLS_SCM_PRETEND_VERSION="$VLLM_VERSION"
    rm -rf "$src/dist"
    # --skip-dependency-check: pyproject lists the `cmake` PyPI package as a
    # build requirement, but setup.py only ever runs the `cmake` executable,
    # and Debian's is on PATH (preflight checks it).
    ( cd "$src" && python -m build --wheel --no-isolation --skip-dependency-check \
          --outdir dist . )

    local wheel
    wheel="$(ls "$src"/dist/vllm-*.whl 2>/dev/null | head -1)"
    [ -n "$wheel" ] || die "the build produced no wheel"
    "$PYTHON" - "$wheel" <<'PY' || die "the wheel failed verification"
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1]); names = z.namelist()
assert any(n.startswith("vllm/_C") and n.endswith(".so") for n in names), "no compiled vllm/_C*.so"
mtp = z.read("vllm/model_executor/models/qwen3_5_mtp.py").decode()
assert "draft_vocab_ids" in mtp, "HyperQwen qwen3_5-mtp-draft-vocab patch is not in the wheel"
assert "shares_target_vocab" in mtp, "devai qwen3_5-mtp-share-vocab-at-init patch is not in the wheel"
PY

    say "publishing the wheelhouse"
    local dist="$BUILD_ROOT/dist"
    rm -rf "$dist"
    mkdir -p "$dist/wheels"
    cp "$BUILD_ROOT"/wheels/*.whl "$dist/wheels/"
    rm -f "$dist"/wheels/vllm-*.whl          # the PyPI build; ours replaces it
    cp "$wheel" "$dist/wheels/"
    cp "$BUILD_ROOT/HyperQwen/docker/requirements.txt" "$dist/requirements.txt"
    cp "$src/PATCHES.applied" "$dist/PATCHES.applied"
    {
        echo "vllm=$VLLM_VERSION"
        echo "hyperqwen=$HYPERQWEN_COMMIT"
        echo "cuda_arch=$CUDA_ARCH_LIST"
        echo "cuda_home=$CUDA_HOME"
    } > "$dist/BUILD_INFO"
    say "done: $dist"
    du -sh "$dist" "$wheel" | sed 's/^/    /'
    echo "    $(wc -l < "$dist/PATCHES.applied") patches applied"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main "$@"
fi
