#!/usr/bin/env bash
# Compile vLLM FROM SOURCE on this host, with the HyperQwen patch series
# applied, for this machine's GPU only. Output: a wheelhouse that
# deploy/Dockerfile.vllm installs offline into debian:trixie-slim. No upstream
# vLLM image is used anywhere, not even as a build input (operator decision,
# same as devai-ollama -- see scripts/build-ollama.sh).
#
# What is built is a BUILD RECORD, deploy/vllm-builds/<version>.sh, chosen by
# VLLM_VERSION (default 0.30.0, platform-rebuild E1) and sourced below. It
# holds what used to be constants here:
#
#   VLLM_SDIST_SHA256  the PyPI sdist's sha256
#   VLLM_TORCH_PIN     torch for the build venv; the runtime set must pin the same
#   HYPERQWEN_COMMIT   a full commit of github.com/syv-ai/HyperQwen, or empty
#   HYPERQWEN_SKIP     patches of HyperQwen's series the record skips, as
#                      upstream does at that commit (an array, may be empty)
#   DEVAI_PATCHES      devai's own series directory, repository-relative, or empty
#   EXTERNALS          the kernel repositories, at the revisions that version's
#                      CMake files name
#
# A record without HyperQwen or without devai's series skips that series and
# the wheel check that looks for it (deploy/Dockerfile.vllm reads BUILD_INFO
# for its own HyperQwen check). Beside each record lie its runtime set,
# <version>.in, which this script writes (`--write-in`, run by
# `make vllm-lock`), and <version>.lock, its hash lock. The wheels are fetched
# from the lock with --require-hashes and the dist carries the lock as
# requirements.txt, so the image installs exactly the reviewed set.
#
# What is built, and why it is shaped this way:
#
#   * vLLM $VLLM_VERSION from the PyPI sdist, sha256-verified. HyperQwen
#     (github.com/syv-ai/HyperQwen) pins exactly one release per commit and
#     applies its patches with `--fuzz 0`, so the version is not a free choice.
#   * The HyperQwen series is applied to the SOURCE TREE before compiling, in
#     the order patches/series gives (a few patches carry context an earlier
#     one adds, so glob order is wrong), minus the record's skips. Their KVarN
#     KV-cache plugin is NOT installed -- 4/2-bit KV was ruled out here on
#     quality grounds (see docs/backends.md).
#   * devai's own patches (deploy/vllm-patches/series) are applied after
#     HyperQwen's, same rules. The first one stops the MTP drafter allocating
#     two vocabulary-sized layers vLLM deletes anyway -- the 0.5 GiB shortfall
#     that kept MTP from loading on 24 GiB.
#     Every patch of both series touches Python only (38 at c0c81bb, 46 at
#     7af097b, plus devai's). Compiling therefore produces the same kernels
#     the stock wheel ships, minus every GPU generation but ours.
#   * TORCH_CUDA_ARCH_LIST=12.0 (Blackwell): native code for this card, no
#     other architecture compiled.
#   * vLLM's CMake clones its external kernel repositories DURING CONFIGURE,
#     outside any retry. They are fetched here instead, each pinned to the
#     revision the source tree names, from the repository it names (a drifted
#     pin or repository is refused), and handed to CMake through the *_SRC_DIR
#     variables vLLM already honours.
#   * The Rust front end's crates are fetched up front for the same reason.
#
# Every network step is tried 3 times and then FAILS the script (operator
# rule: every failure is an error, and a download is repeated 3 times before
# bailing out). A failed fetch leaves no directory behind.
#
# Host prerequisites -- Debian packages, installed once by the operator:
#   cmake ninja-build gcc g++ git patch python3 python3-dev        (Debian main)
#   libdw-dev    (Debian main; elfutils/libdwfl.h, which 0.30.0's DeepGEMM
#                 fork includes; at run time it only dlopens libdw.so.1,
#                 optionally, for stack-trace symbols)
#   cuda-nvcc-13-1 cuda-cudart-dev-13-1 cuda-cccl-13-1 libcublas-dev-13-1
#   libcusparse-dev-13-1 libcusolver-dev-13-1 libcufft-dev-13-1
#   libcurand-dev-13-1 cuda-nvrtc-dev-13-1 libnvjitlink-dev-13-1
#   libcufile-dev-13-1 cuda-nvtx-13-1 cuda-profiler-api-13-1  (NVIDIA debian13)
# plus uv (creates the build venv; python3-venv is not needed) and a rustup
# toolchain matching the source tree's rust-toolchain.toml.
#
# Builds under ~/.cache/devai/vllm-build/<version>, one root per version, NOT
# /var/cache/devai (a new top-level directory there is not volume-backed).
# Each build deletes only its own dist/, so another version's wheelhouse --
# the rollback -- is never touched.
#
# The compile (about 20 minutes) is skipped when its inputs are unchanged:
# the newest wheel is kept under $BUILD_ROOT/compiled/<compile id>
# (compile_id hashes the build record, devai's patches, the arch list, the
# compilers and Python -- not the runtime lock), re-verified, and reused, so
# a lock-only change rebuilds the wheelhouse in minutes. VLLM_REBUILD=1
# compiles anyway.
#
# Usage: build-vllm.sh             compile and publish $BUILD_ROOT/dist
#        build-vllm.sh --write-in  write deploy/vllm-builds/$VLLM_VERSION.in only
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_VERSION="${VLLM_VERSION:-0.30.0}"
VLLM_RECORD_DIR="${VLLM_RECORD_DIR:-$REPO_ROOT/deploy/vllm-builds}"
HYPERQWEN_REPO="https://github.com/syv-ai/HyperQwen.git"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.1}"
CUDA_ARCH_LIST="${CUDA_ARCH_LIST:-12.0}"
BUILD_ROOT="${VLLM_BUILD_ROOT:-$HOME/.cache/devai/vllm-build/$VLLM_VERSION}"
# nvcc peaks at several GiB per translation unit on the attention kernels;
# 16 keeps a 64 GiB host out of swap where $(nproc) does not.
JOBS="${JOBS:-16}"
FETCH_RETRY_DELAY="${FETCH_RETRY_DELAY:-5}"
PYTHON="${PYTHON:-/usr/bin/python3}"
# Where host development headers outside the CUDA toolkit live (tests point
# it elsewhere).
VLLM_HOST_INCLUDE="${VLLM_HOST_INCLUDE:-/usr/include}"

die() { echo "error: $*" >&2; exit 1; }
say() { echo "==> $*"; }

# Source the record for $VLLM_VERSION. The record is authoritative: the
# environment cannot override what it pins. DEVAI_PATCH_DIR (an absolute
# directory) may still be set from outside, for the tests.
load_record() {
    local record="$VLLM_RECORD_DIR/$VLLM_VERSION.sh"
    [ -f "$record" ] || die "no build record for vLLM $VLLM_VERSION ($record)"
    VLLM_SDIST_SHA256="" VLLM_TORCH_PIN="" HYPERQWEN_COMMIT="" DEVAI_PATCHES=""
    HYPERQWEN_SKIP=() EXTERNALS=()
    # shellcheck source=/dev/null
    . "$record"
    [[ "$VLLM_SDIST_SHA256" =~ ^[0-9a-f]{64}$ ]] \
        || die "$record: VLLM_SDIST_SHA256 is not a sha256"
    [[ "$VLLM_TORCH_PIN" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] \
        || die "$record: VLLM_TORCH_PIN is not a version"
    [ -z "$HYPERQWEN_COMMIT" ] || [[ "$HYPERQWEN_COMMIT" =~ ^[0-9a-f]{40}$ ]] \
        || die "$record: HYPERQWEN_COMMIT must be a full 40-hex commit or empty"
    [ -n "$HYPERQWEN_COMMIT" ] || [ "${#HYPERQWEN_SKIP[@]}" -eq 0 ] \
        || die "$record: HYPERQWEN_SKIP without HYPERQWEN_COMMIT"
    [ "${#EXTERNALS[@]}" -gt 0 ] || die "$record: no EXTERNALS"
    DEVAI_PATCH_DIR="${DEVAI_PATCH_DIR:-${DEVAI_PATCHES:+$REPO_ROOT/$DEVAI_PATCHES}}"
}
load_record

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

# One attempt at downloading `url` to `dest`, through "$dest.part", so an
# interrupted download never looks like a finished one.
fetch_file() {
    local url="$1" dest="$2"
    curl -fsSL -o "$dest.part" "$url" && mv "$dest.part" "$dest"
}

fetch_sdist() {
    local tarball="$BUILD_ROOT/src/vllm-$VLLM_VERSION.tar.gz"
    mkdir -p "$BUILD_ROOT/src"
    if [ ! -f "$tarball" ]; then
        say "fetching vLLM $VLLM_VERSION sdist"
        # The file, not `pip download`: pip prepares the sdist's metadata,
        # which runs its setup.py, and 0.30.0's needs setuptools_rust in
        # the host Python. The sha256 below is the check either way.
        retry3 "fetching the vLLM sdist" fetch_file \
            "https://files.pythonhosted.org/packages/source/v/vllm/vllm-$VLLM_VERSION.tar.gz" \
            "$tarball" || { rm -f "$tarball.part"; return 1; }
    fi
    local have
    have="$(sha256sum "$tarball" | cut -d' ' -f1)"
    [ "$have" = "$VLLM_SDIST_SHA256" ] \
        || die "$tarball sha256 is $have, expected $VLLM_SDIST_SHA256"
}

# The record's sources: the sdist and, for a HyperQwen record, HyperQwen.
fetch_record_sources() {
    fetch_sdist
    if [ -n "$HYPERQWEN_COMMIT" ]; then
        fetch_source "HyperQwen" "$HYPERQWEN_REPO" "$HYPERQWEN_COMMIT" "$BUILD_ROOT/HyperQwen"
    fi
}

# Unpack a pristine source tree and print its path.
unpack_sdist() {
    local src="$BUILD_ROOT/src/vllm-$VLLM_VERSION"
    rm -rf "$src"
    tar -xzf "$BUILD_ROOT/src/vllm-$VLLM_VERSION.tar.gz" -C "$BUILD_ROOT/src"
    [ -d "$src/requirements" ] || die "$src has no requirements/ directory"
    echo "$src"
}

# The runtime set, deploy/vllm-builds/<version>.in, printed to stdout: the
# release's own (requirements/cuda.txt with the requirements files it
# includes) followed, for a HyperQwen record, by HyperQwen's
# docker/requirements.txt -- without vllm (built here), flashinfer-cubin
# (published only on flashinfer.ai's index since 0.6.14; FlashInfer
# JIT-compiles instead), the CUDA compiler packages (nvidia-cuda-nvcc,
# -crt, -cccl, nvidia-nvvm: HyperQwen pins pip's 13.0 set because its image
# has no toolkit; ours carries the host's CUDA 13.1, which FlashInfer
# compiles with -- M2, plan E8) and index lines (PyPI alone, M3). The release's torch
# pin must be the record's. Where both name a package, uv applies both
# specifiers, so a HyperQwen pin the release's specifier refuses fails
# `make vllm-lock`. Inline comments are dropped; markers are kept.
runtime_in() {
    local src="$1" hq_req=""
    [ -z "$HYPERQWEN_COMMIT" ] || hq_req="$BUILD_ROOT/HyperQwen/docker/requirements.txt"
    "$PYTHON" -I - "$src/requirements/cuda.txt" "$hq_req" "$VLLM_VERSION" \
        "$HYPERQWEN_COMMIT" "$VLLM_TORCH_PIN" <<'PY'
import re
import sys
from pathlib import Path

cuda_txt, hq_req, version, hq_commit, torch_pin = sys.argv[1:6]
DROP = {"vllm", "flashinfer-cubin",
        "nvidia-cuda-nvcc", "nvidia-cuda-crt", "nvidia-cuda-cccl", "nvidia-nvvm"}
# Extras that exist only to bring pip's CUDA compiler (nvcc, cccl) for a
# host without a toolkit: humming-kernels' cu12/cu13 add nvcc and cccl
# beside the runtime and nvrtc torch requires anyway. Humming picks the
# system toolkit over pip's when both are present (checked in the 0.30.0
# image: /usr/local/cuda, 13.1), so the image's CUDA_HOME serves it.
DROP_EXTRAS = {"humming-kernels": {"cu12", "cu13"}}
INDEX_OPTIONS = {"-i", "--index-url", "--extra-index-url", "-f", "--find-links"}
INCLUDE_OPTIONS = {"-r", "--requirement"}


def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def requirements(path, seen):
    if path in seen:
        sys.exit(f"error: {path} includes itself")
    seen = seen | {path}
    out = []
    for raw in path.read_text().splitlines():
        line = re.sub(r"(^|\s)#.*$", "", raw).strip()
        if not line:
            continue
        if line.startswith("-"):
            option = re.split(r"[\s=]", line, maxsplit=1)[0]
            arg = line[len(option):].lstrip(" \t=").strip()
            if option in INCLUDE_OPTIONS and arg:
                out += requirements((path.parent / arg).resolve(), seen)
            elif option in INDEX_OPTIONS:
                continue
            else:
                sys.exit(f"error: {path}: unexpected option line {raw!r}")
            continue
        m = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", line)
        if not m:
            sys.exit(f"error: {path}: cannot read {raw!r}")
        if norm(m.group(0)) in DROP:
            continue
        drop = DROP_EXTRAS.get(norm(m.group(0)))
        if drop:
            rest = line[m.end():]
            e = re.match(r"\s*\[([^\]]*)\]", rest)
            if e:
                keep = [x.strip() for x in e.group(1).split(",") if x.strip() and x.strip() not in drop]
                line = m.group(0) + (f"[{','.join(keep)}]" if keep else "") + rest[e.end():]
        out.append(line)
    return out


release = requirements(Path(cuda_txt).resolve(), frozenset())
torch = [ln for ln in release if norm(re.match(r"[A-Za-z0-9._-]+", ln).group(0)) == "torch"]
want = f"torch=={torch_pin}"
if not torch:
    release.append(want)
elif [re.sub(r"\s+", "", ln) for ln in torch] != [want]:
    sys.exit(f"error: vLLM {version} pins {torch}, the build record pins {want}")

print(f"# vLLM {version} runtime set, written by scripts/build-vllm.sh --write-in")
print(f"# (make vllm-lock VLLM_VERSION={version}) and hash-locked as {version}.lock.")
print("# Do not edit it: change the build record and run make vllm-lock.")
print("#")
print(f"# vllm-{version} sdist: requirements/cuda.txt and the files it includes,")
print("# without vllm, flashinfer-cubin, the CUDA compiler packages and index lines")
print("# (PyPI alone).")
print("\n".join(release))
if hq_req:
    hq = requirements(Path(hq_req).resolve(), frozenset())
    print("#")
    print(f"# HyperQwen {hq_commit}: docker/requirements.txt,")
    print("# without vllm, flashinfer-cubin, the CUDA compiler packages (the image carries")
    print("# the host's CUDA 13.1 toolkit, M2) and index lines.")
    print("\n".join(hq))
PY
}

# `--write-in`: (re)write the record's runtime set from its sources.
write_in() {
    local t
    for t in git tar sha256sum "$PYTHON"; do
        command -v "$t" >/dev/null || die "'$t' not found on PATH"
    done
    mkdir -p "$BUILD_ROOT"
    fetch_record_sources
    local src out="$VLLM_RECORD_DIR/$VLLM_VERSION.in"
    src="$(unpack_sdist)"
    runtime_in "$src" > "$out.tmp" || { rm -f "$out.tmp"; die "could not write $out"; }
    mv "$out.tmp" "$out"
    say "wrote $out"
}

# A build must use the runtime set the record's sources give: a stale .in
# (and so a stale lock) is refused, not built.
check_runtime_in() {
    local src="$1" in="$VLLM_RECORD_DIR/$VLLM_VERSION.in"
    [ -f "$in" ] || die "no runtime set $in -- run 'make vllm-lock VLLM_VERSION=$VLLM_VERSION'"
    [ -f "$VLLM_RECORD_DIR/$VLLM_VERSION.lock" ] \
        || die "no lock $VLLM_RECORD_DIR/$VLLM_VERSION.lock -- run 'make vllm-lock VLLM_VERSION=$VLLM_VERSION'"
    runtime_in "$src" | cmp -s - "$in" \
        || die "$in is not what the record's sources give -- run 'make vllm-lock VLLM_VERSION=$VLLM_VERSION' and review the lock's diff"
}

fetch_wheels() {
    # Runtime set: the record's hash lock, every file checked against it.
    # Build set: what pyproject's build-system asks for and the runtime set
    # does not already carry.
    local lock="$VLLM_RECORD_DIR/$VLLM_VERSION.lock"
    mkdir -p "$BUILD_ROOT/wheels" "$BUILD_ROOT/wheels-build"
    say "fetching runtime wheels ($VLLM_VERSION.lock, --require-hashes)"
    retry3 "fetching runtime wheels" "$PYTHON" -m pip download \
        -r "$lock" --require-hashes -d "$BUILD_ROOT/wheels" \
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
    [ -f "$VLLM_HOST_INCLUDE/elfutils/libdwfl.h" ] \
        || die "no elfutils/libdwfl.h under $VLLM_HOST_INCLUDE: install libdw-dev (sudo apt install libdw-dev)"
}

# Refuse to build against a pin or a repository the source tree no longer
# names: the record's table is only correct for the vLLM release it was
# written for (DeepGEMM, for one, moved to a vllm-project fork in 0.30.0).
check_pins() {
    local src="$1" row name repo ref file
    for row in "${EXTERNALS[@]}"; do
        IFS='|' read -r name _ repo ref file _ <<< "$row"
        grep -qF -- "$ref" "$src/$file" \
            || die "$name pin $ref is not named in $file -- update the build record for vLLM $VLLM_VERSION"
        grep -qF -- "$repo" "$src/$file" \
            || die "$name repository $repo is not named in $file -- update the build record for vLLM $VLLM_VERSION"
    done
}

series_names() {
    sed -e 's/#.*//' -e 's/^[[:space:]]*//;s/[[:space:]]*$//' -e '/^$/d' "$1"
}

# Apply one patch series to the vLLM package directory, minus the names given
# after the label. `--fuzz 0`: a patch that needs fuzz to apply is a patch
# written for different code.
apply_series() {
    local src="$1" dir="$2" label="$3"; shift 3
    local skips=" $* " name
    [ -f "$dir/series" ] || die "no series file in $dir"
    series_names "$dir/series" | while IFS= read -r name; do
        if [[ "$skips" == *" $name "* ]]; then
            echo "    skip $name (the build record for vLLM $VLLM_VERSION skips it)"
            continue
        fi
        echo "    $name"
        patch -p1 --fuzz 0 --no-backup-if-mismatch --quiet -d "$src/vllm" < "$dir/$name" \
            || die "$label/$name did not apply cleanly to vLLM $VLLM_VERSION"
        echo "$label/$name" >> "$src/PATCHES.applied"
    done
}

apply_patches() {
    local src="$1" skip
    : > "$src/PATCHES.applied"
    if [ -n "$HYPERQWEN_COMMIT" ]; then
        for skip in "${HYPERQWEN_SKIP[@]}"; do
            series_names "$BUILD_ROOT/HyperQwen/patches/series" | grep -qxF -- "$skip" \
                || die "the build record skips $skip, which HyperQwen's series does not list"
        done
        apply_series "$src" "$BUILD_ROOT/HyperQwen/patches" hyperqwen "${HYPERQWEN_SKIP[@]}"
        grep -q '^hyperqwen/' "$src/PATCHES.applied" \
            || die "no patch was applied -- empty patches/series?"
    fi
    if [ -n "$DEVAI_PATCH_DIR" ]; then
        apply_series "$src" "$DEVAI_PATCH_DIR" devai
    fi
}

# The patches the wheel must carry, one marker per series the record names.
verify_wheel() {
    local wheel="$1"
    "$PYTHON" -I - "$wheel" "${HYPERQWEN_COMMIT:+hyperqwen}" "${DEVAI_PATCH_DIR:+devai}" <<'PY'
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1]); names = z.namelist()
series = {a for a in sys.argv[2:] if a}
assert any(n.startswith("vllm/_C") and n.endswith(".so") for n in names), "no compiled vllm/_C*.so"
mtp = z.read("vllm/model_executor/models/qwen3_5_mtp.py").decode()
if "hyperqwen" in series:
    assert "draft_vocab_ids" in mtp, "HyperQwen qwen3_5-mtp-draft-vocab patch is not in the wheel"
if "devai" in series:
    assert "shares_target_vocab" in mtp, "devai qwen3_5-mtp-share-vocab-at-init patch is not in the wheel"
PY
}

# Bump when the compile steps change in a way compile_id's inputs do not show.
COMPILE_SCHEMA=1

# A content hash of every input the compiled wheel depends on: the build
# record (sdist sha256, torch pin, HyperQwen commit and skips, EXTERNALS),
# devai's patch files, the arch list, the compilers, Python and
# COMPILE_SCHEMA. NOT the runtime lock: only the wheelhouse and the image
# read it, so a lock-only change (plan E8) reuses the wheel instead of
# compiling for twenty minutes. Contents only, never paths, so a moved
# checkout gets the same id.
compile_id() {
    {
        echo "compile-schema=$COMPILE_SCHEMA"
        echo "vllm=$VLLM_VERSION arch=$CUDA_ARCH_LIST"
        "$CUDA_HOME/bin/nvcc" --version | tail -1
        gcc --version | head -1
        "$PYTHON" --version
        sha256sum < "$VLLM_RECORD_DIR/$VLLM_VERSION.sh"
        if [ -n "$DEVAI_PATCH_DIR" ]; then
            local f
            for f in $(cd "$DEVAI_PATCH_DIR" && ls | LC_ALL=C sort); do
                printf '%s ' "$f"
                sha256sum < "$DEVAI_PATCH_DIR/$f"
            done
        fi
    } | sha256sum | cut -c1-16
}

# The wheel kept for compile id $1, or nothing.
compiled_wheel() {
    local f
    for f in "$BUILD_ROOT/compiled/$1"/vllm-*.whl; do
        [ -f "$f" ] && { echo "$f"; return 0; }
    done
    return 0
}

# Keep the wheel just compiled ($2, from source tree $3) under its compile
# id $1, with the patch list it carries. Only the newest is kept (about 380
# MiB); the swap goes through a temporary directory.
keep_compiled() {
    local id="$1" wheel="$2" src="$3" dir="$BUILD_ROOT/compiled"
    rm -rf "$dir.tmp"
    mkdir -p "$dir.tmp/$id"
    cp "$wheel" "$src/PATCHES.applied" "$dir.tmp/$id/"
    rm -rf "$dir"
    mv "$dir.tmp" "$dir"
}

# Fetch the kernel repositories and Rust crates, apply the record's patch
# series and compile the wheel; sets WHEEL to its path.
compile_wheel() {
    local src="$1"
    local row name var repo ref sub
    for row in "${EXTERNALS[@]}"; do
        IFS='|' read -r name var repo ref _ sub <<< "$row"
        fetch_source "$name" "$repo" "$ref" "$BUILD_ROOT/ext/$name"
        [ -d "$BUILD_ROOT/ext/$name/$sub" ] || die "$name has no '$sub' directory"
        export "$var=$BUILD_ROOT/ext/$name${sub:+/$sub}"
    done

    say "fetching Rust crates"
    ( cd "$src/rust" && retry3 "cargo fetch" cargo fetch --locked --quiet )

    say "applying the record's patch series (--fuzz 0)"
    apply_patches "$src"

    say "creating the build venv"
    rm -rf "$BUILD_ROOT/venv"
    uv venv --quiet --python "$PYTHON" "$BUILD_ROOT/venv"
    # shellcheck disable=SC1091
    . "$BUILD_ROOT/venv/bin/activate"
    uv pip install --quiet --offline --no-index \
        --find-links "$BUILD_ROOT/wheels" --find-links "$BUILD_ROOT/wheels-build" \
        "torch==$VLLM_TORCH_PIN" numpy ninja packaging "setuptools>=77.0.3,<81.0.0" \
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

    WHEEL="$(ls "$src"/dist/vllm-*.whl 2>/dev/null | head -1)"
    [ -n "$WHEEL" ] || die "the build produced no wheel"
    verify_wheel "$WHEEL" || die "the wheel failed verification"
}

main() {
    case "${1:-}" in
        --write-in) write_in; return;;
        "") ;;
        *) die "usage: build-vllm.sh [--write-in]";;
    esac
    preflight
    mkdir -p "$BUILD_ROOT/ext"
    fetch_record_sources

    local src
    say "unpacking a pristine source tree"
    src="$(unpack_sdist)"
    check_pins "$src"
    check_runtime_in "$src"
    fetch_wheels

    local id wheel
    id="$(compile_id)"
    wheel="$(compiled_wheel "$id")"
    if [ -n "$wheel" ] && [ -z "${VLLM_REBUILD:-}" ] && verify_wheel "$wheel"; then
        say "reusing the wheel compiled from the same inputs ($id; VLLM_REBUILD=1 compiles anew)"
        cp "$(dirname "$wheel")/PATCHES.applied" "$src/PATCHES.applied"
    else
        compile_wheel "$src"
        wheel="$WHEEL"
        keep_compiled "$id" "$wheel" "$src"
    fi

    say "publishing the wheelhouse"
    local dist="$BUILD_ROOT/dist" devai_applied
    rm -rf "$dist"
    mkdir -p "$dist/wheels"
    cp "$BUILD_ROOT"/wheels/*.whl "$dist/wheels/"
    rm -f "$dist"/wheels/vllm-*.whl          # a PyPI build, if any; ours replaces it
    cp "$wheel" "$dist/wheels/"
    cp "$VLLM_RECORD_DIR/$VLLM_VERSION.lock" "$dist/requirements.txt"
    cp "$src/PATCHES.applied" "$dist/PATCHES.applied"
    devai_applied="$(sed -n 's|^devai/||p' "$src/PATCHES.applied" | paste -sd, -)"
    {
        echo "vllm=$VLLM_VERSION"
        echo "hyperqwen=${HYPERQWEN_COMMIT:-none}"
        echo "devai_patches=${devai_applied:-none}"
        echo "torch=$VLLM_TORCH_PIN"
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
