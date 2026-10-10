#!/usr/bin/env bash
# Build pipelock FROM SOURCE on the host and lay the binary out for
# deploy/Dockerfile.pipelock to copy into debian:trixie-slim
# (platform-rebuild Phase 2). Replaces ghcr.io/luckypipewrench/pipelock:latest.
#
# The version is the one that was already running (3.0.0, commit 9d2e37d), so
# the switch changes where the binary comes from and nothing else; a newer
# release is a separate, deliberate bump of PIPELOCK_VERSION and
# PIPELOCK_COMMIT together.
#
# Built as upstream builds its release (.goreleaser.yaml, read from the
# running binary with `go version -m`): CGO_ENABLED=0, -trimpath,
# -tags enterprise, the same -X version variables. Two differences:
#   1. The three public keys a release embeds (license, rule-bundle keyring,
#      self-update keyring) are left empty. devai configures no license, no
#      signed rule bundle (rules_dir is empty) and never self-updates the
#      binary, so none of them is read; an empty key makes each of those
#      features refuse rather than trust something.
#   2. The host's Go, not upstream's (go1.25.11 for 3.0.0); go.mod asks for
#      go 1.25.0, so any newer toolchain builds it.
# BuildDate is the commit's own date, so two builds of one commit agree.
#
# Every network step is tried 3 times and then FAILS the script.
#
# Usage: scripts/build-pipelock.sh            (or: make build-pipelock-dist)
# Env:   PIPELOCK_BUILD_ROOT  work dir (default ~/.cache/devai/pipelock-build)
#        FETCH_RETRY_DELAY    seconds between attempts (default 5)
set -euo pipefail

PIPELOCK_VERSION="v3.0.0"
PIPELOCK_COMMIT="9d2e37d296359c38e599525a0424cdc189a812e1"
# Not under /var/cache/devai: a new top-level directory there is a plain
# directory the operator creates (CLAUDE.md mount-point convention).
BUILD_ROOT="${PIPELOCK_BUILD_ROOT:-$HOME/.cache/devai/pipelock-build}"
FETCH_RETRY_DELAY="${FETCH_RETRY_DELAY:-5}"

PIPELOCK_REPO="https://github.com/luckyPipewrench/pipelock.git"
MODULE="github.com/luckyPipewrench/pipelock"

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
    git -c advice.detachedHead=false clone --quiet --depth 1 --branch "$tag" "$repo" "$dest.partial" || {
        rm -rf "$dest.partial"; return 1; }
    mv "$dest.partial" "$dest"
}

# fetch_source <dest>: fetch unless already present, then verify the tree is
# the pinned tag AND the pinned commit (a moved tag is refused).
fetch_source() {
    local dest="$1"
    if [ -d "$dest/.git" ]; then
        say "pipelock $PIPELOCK_VERSION already fetched"
    else
        say "fetching pipelock $PIPELOCK_VERSION"
        retry3 "fetching pipelock $PIPELOCK_VERSION" clone_tag "$PIPELOCK_REPO" "$PIPELOCK_VERSION" "$dest"
    fi
    local tag commit
    tag="$(git -C "$dest" describe --tags --exact-match 2>/dev/null || true)"
    [ "$tag" = "$PIPELOCK_VERSION" ] || die "$dest is at '${tag:-unknown}', expected $PIPELOCK_VERSION"
    commit="$(git -C "$dest" rev-parse HEAD)"
    [ "$commit" = "$PIPELOCK_COMMIT" ] \
        || die "tag $PIPELOCK_VERSION is commit $commit, expected $PIPELOCK_COMMIT -- the tag moved upstream"
}

main() {
    local t
    for t in git go; do
        command -v "$t" >/dev/null || die "'$t' not found on PATH"
    done
    mkdir -p "$BUILD_ROOT"
    local src="$BUILD_ROOT/pipelock-$PIPELOCK_VERSION"
    local dist="$BUILD_ROOT/dist"

    fetch_source "$src"

    say "go mod download"
    ( cd "$src" && retry3 "go mod download" go mod download )

    local ver="${PIPELOCK_VERSION#v}" date short gover
    date="$(TZ=UTC git -C "$src" log -1 --format=%cd --date=format-local:%Y-%m-%dT%H:%M:%SZ)"
    short="${PIPELOCK_COMMIT:0:7}"
    gover="$(go version | awk '{print $3}')"
    say "go build ($gover)"
    rm -rf "$src/out"
    ( cd "$src" && CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -tags enterprise \
          -ldflags "-s -w \
            -X $MODULE/internal/cliutil.Version=$ver \
            -X $MODULE/internal/cliutil.BuildDate=$date \
            -X $MODULE/internal/cliutil.GitCommit=$short \
            -X $MODULE/internal/cliutil.GoVersion=$gover \
            -X $MODULE/internal/proxy.Version=$ver" \
          -o out/pipelock ./cmd/pipelock )

    # ── Verify before publishing the result ─────────────────────────────────
    [ -x "$src/out/pipelock" ] || die "go build produced no out/pipelock"
    # Static: the image is a slim Debian, but the binary must not depend on it.
    if ldd "$src/out/pipelock" >/dev/null 2>&1; then
        die "out/pipelock is dynamically linked"
    fi
    local version
    # It prints to stderr.
    version="$("$src/out/pipelock" version 2>&1)"
    grep -q "^pipelock version $ver\$" <<<"$version" || die "binary reports: $version"
    grep -q "git commit: $short\$" <<<"$version" || die "binary reports: $version"

    rm -rf "$dist.partial"
    mkdir -p "$dist.partial"
    install -m 0755 "$src/out/pipelock" "$dist.partial/pipelock"
    echo "$PIPELOCK_VERSION $PIPELOCK_COMMIT" > "$dist.partial/PIPELOCK_VERSION"
    rm -rf "$dist"
    mv "$dist.partial" "$dist"
    say "done: $dist"
    sed 's/^/    /' <<<"$version"
}

# Sourcing this file (the tests do) defines the functions without running.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main "$@"
fi
