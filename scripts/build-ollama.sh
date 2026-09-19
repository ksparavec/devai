#!/usr/bin/env bash
# Build Ollama FROM SOURCE on the host, for this machine's GPU only, and lay
# out the result for deploy/Dockerfile.ollama to copy into debian:trixie-slim.
#
# Why from source: the prebuilt routes are huge and almost entirely unused
# here. The stock ollama/ollama image is ~4.4 GiB (three bundled GPU runtimes;
# the `ollama` binary is 36 MiB of it) and the release tarball is 1361 MB,
# because both carry CUDA 12 AND 13 compiled for every GPU generation. This
# host is amd64 + one NVIDIA Blackwell card, so exactly one backend compiled
# for exactly one architecture is needed.
#
# Host prerequisites (Debian packages; NVIDIA's apt repo for the CUDA ones):
#   cmake  cuda-nvcc-13-1  libcublas-dev-13-1  cuda-cudart-dev-13-1  cuda-cccl-13-1
#   plus gcc/g++, make, git and Go >= the version in Ollama's go.mod.
#
# It mirrors upstream's own Dockerfile step for step -- two CMake builds out
# of llama/server (presets `cpu` and `llama_cuda_v13_linux`), each installed
# with `--component llama-server --strip`, then `go build` -- with TWO
# deliberate differences:
#   1. CMAKE_CUDA_ARCHITECTURES=120-real instead of upstream's twelve
#      `-virtual` targets. `-virtual` is PTX the driver must JIT at start-up;
#      `120-real` is native Blackwell code. Smaller, no JIT, and it does not
#      depend on the driver being able to JIT a newer toolkit's PTX.
#   2. llama.cpp is fetched ONCE here and handed to CMake, instead of CMake's
#      FetchContent cloning it during configure (once per build dir, i.e.
#      twice, outside any retry or verification).
#
# Every network step is tried 3 times and then FAILS the script. Nothing is
# allowed to end in a quiet success.
#
# Usage: scripts/build-ollama.sh            (or: make build-ollama)
# Env:   OLLAMA_VERSION  tag to build            (default below)
#        CUDA_HOME       CUDA toolkit root       (default /usr/local/cuda-13.1)
#        CUDA_ARCH       CMAKE_CUDA_ARCHITECTURES (default 120-real)
#        OLLAMA_BUILD_ROOT  work dir             (default ~/.cache/devai/ollama-build)
#        JOBS            parallel compile jobs   (default: nproc)
#        FETCH_RETRY_DELAY seconds between attempts (default 5)
set -euo pipefail

OLLAMA_VERSION="${OLLAMA_VERSION:-v0.34.2}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.1}"
CUDA_ARCH="${CUDA_ARCH:-120-real}"
# Not under /var/cache/devai: a new top-level directory there is not
# volume-backed (CLAUDE.md mount-point convention).
BUILD_ROOT="${OLLAMA_BUILD_ROOT:-$HOME/.cache/devai/ollama-build}"
JOBS="${JOBS:-$(nproc)}"
FETCH_RETRY_DELAY="${FETCH_RETRY_DELAY:-5}"

OLLAMA_REPO="https://github.com/ollama/ollama.git"
LLAMA_CPP_REPO="https://github.com/ggml-org/llama.cpp.git"

die() { echo "error: $*" >&2; exit 1; }
say() { echo "==> $*"; }

# retry3 <label> <command...>: run the command up to 3 times. A failure is
# an ERROR: after the third one the script stops, non-zero, naming the step.
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

# clone_tag <repo> <tag> <dest>: shallow-clone one tag. A partial clone is
# never left where a later run could mistake it for a complete one.
clone_tag() {
    local repo="$1" tag="$2" dest="$3"
    rm -rf "$dest.partial"
    git clone --quiet --depth 1 --branch "$tag" "$repo" "$dest.partial" || {
        rm -rf "$dest.partial"; return 1; }
    mv "$dest.partial" "$dest"
}

# fetch_source <label> <repo> <tag> <dest>: fetch unless already present AT
# THAT TAG, then verify what is on disk really is that tag.
fetch_source() {
    local label="$1" repo="$2" tag="$3" dest="$4"
    if [ -d "$dest/.git" ]; then
        say "$label $tag already fetched"
    else
        say "fetching $label $tag"
        retry3 "fetching $label $tag" clone_tag "$repo" "$tag" "$dest"
    fi
    local have
    have="$(git -C "$dest" describe --tags --exact-match 2>/dev/null || true)"
    [ "$have" = "$tag" ] || die "$dest is at '${have:-unknown}', expected $label $tag"
}

preflight() {
    local t
    for t in git cmake make gcc g++ go; do
        command -v "$t" >/dev/null || die "'$t' not found on PATH"
    done
    [ -x "$CUDA_HOME/bin/nvcc" ] || die "no nvcc at $CUDA_HOME/bin/nvcc (set CUDA_HOME)"
    [ -f "$CUDA_HOME/include/cublas_v2.h" ] || die "no cuBLAS headers under $CUDA_HOME (libcublas-dev missing?)"
}

# cmake_build <preset> <build dir name> [extra -D args...]
cmake_build() {
    local preset="$1" bdir="$2"; shift 2
    local src="$BUILD_ROOT/ollama-$OLLAMA_VERSION"
    # Upstream applies its compat patch to the llama.cpp tree on every
    # configure, so each build gets its OWN pristine copy; sharing one would
    # patch it twice.
    local lcpp="$BUILD_ROOT/llama.cpp-$LLAMA_CPP_TAG.$bdir"
    rm -rf "$lcpp" "$src/build/$bdir"
    cp -a "$BUILD_ROOT/llama.cpp-$LLAMA_CPP_TAG" "$lcpp"
    say "cmake configure: $preset"
    # NOT the OLLAMA_LLAMA_CPP_SOURCE env var: that one makes upstream SKIP
    # the compat patch. The cache variable gets the tree patched.
    ( cd "$src" && cmake -S llama/server --preset "$preset" \
          -DFETCHCONTENT_SOURCE_DIR_LLAMA_CPP="$lcpp" "$@" )
    say "cmake build: $preset (-j$JOBS)"
    ( cd "$src" && cmake --build "build/$bdir" -- -j"$JOBS" )
    ( cd "$src" && cmake --install "build/$bdir" --component llama-server --strip )
}

main() {
    preflight
    mkdir -p "$BUILD_ROOT"
    local src="$BUILD_ROOT/ollama-$OLLAMA_VERSION"
    local dist="$BUILD_ROOT/dist"

    fetch_source "Ollama" "$OLLAMA_REPO" "$OLLAMA_VERSION" "$src"
    LLAMA_CPP_TAG="$(tr -d '[:space:]' < "$src/LLAMA_CPP_VERSION")"
    [ -n "$LLAMA_CPP_TAG" ] || die "empty LLAMA_CPP_VERSION in $src"
    fetch_source "llama.cpp" "$LLAMA_CPP_REPO" "$LLAMA_CPP_TAG" "$BUILD_ROOT/llama.cpp-$LLAMA_CPP_TAG"

    export PATH="$CUDA_HOME/bin:$PATH"
    rm -rf "$src/dist"
    cmake_build cpu llama-server-cpu
    cmake_build llama_cuda_v13_linux llama-server-cuda_v13 \
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
        -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc"

    say "go mod download"
    ( cd "$src" && retry3 "go mod download" go mod download )
    say "go build"
    local ver="${OLLAMA_VERSION#v}"
    ( cd "$src" && CGO_ENABLED=1 go build -trimpath -buildmode=pie \
          -ldflags "-w -s -X=github.com/ollama/ollama/version.Version=$ver -X=github.com/ollama/ollama/server.mode=release" \
          -o dist/bin/ollama . )

    # ── Verify before publishing the result ─────────────────────────────────
    [ -x "$src/dist/bin/ollama" ] || die "go build produced no dist/bin/ollama"
    [ -d "$src/dist/lib/ollama/cuda_v13" ] || die "no dist/lib/ollama/cuda_v13 -- the CUDA backend did not install"
    ls "$src/dist/lib/ollama/cuda_v13"/libggml-cuda.so >/dev/null 2>&1 \
        || die "no libggml-cuda.so in dist/lib/ollama/cuda_v13"
    local lib missing
    for lib in libcublas.so libcublasLt.so libcudart.so; do
        ls "$src/dist/lib/ollama/cuda_v13/$lib".* >/dev/null 2>&1 \
            || die "$lib was not bundled into dist/lib/ollama/cuda_v13"
    done
    # libcuda.so.1 is the driver; CDI injects it at run time. Anything ELSE
    # unresolved would only surface as a slow CPU fallback inside the container.
    missing="$(LD_LIBRARY_PATH="$src/dist/lib/ollama/cuda_v13:$src/dist/lib/ollama" \
        ldd "$src/dist/lib/ollama/cuda_v13/libggml-cuda.so" \
        | awk '/not found/ && $1 != "libcuda.so.1" {print $1}')"
    [ -z "$missing" ] || die "unresolved libraries in libggml-cuda.so: $missing"

    rm -rf "$dist"
    cp -a "$src/dist" "$dist"
    echo "$OLLAMA_VERSION" > "$dist/OLLAMA_VERSION"
    say "done: $dist"
    "$dist/bin/ollama" --version 2>&1 | sed 's/^/    /' || true
    du -sh "$dist" "$dist/bin/ollama" "$dist/lib/ollama/cuda_v13" | sed 's/^/    /'
}

# Sourcing this file (the tests do) defines the functions without running.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main "$@"
fi
