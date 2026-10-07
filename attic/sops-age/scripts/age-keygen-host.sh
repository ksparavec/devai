#!/usr/bin/env bash
# Generate a per-host age keypair for sops, store it under
# ~/.config/sops/age/keys.txt with mode 0600, and print the public key
# the operator should add to .sops.yaml.
#
# Idempotent: re-running on a host that already has a keypair prints
# the existing public key and exits 0 without touching the file.
#
# Usage:
#   bash scripts/age-keygen-host.sh
#
# Pre-conditions: `age` and `age-keygen` on PATH (provided by the
# fetch-cli sops/age block in the lab image; install on the host via
# the operator's package manager, or copy the binaries from
# /var/cache/devai/pip/bin/).
#
# Per docs/plans/sops-age-secrets.md decision 2: user-scoped key,
# mode 0600, no root required.

set -euo pipefail

# Allow overriding the key dir for tests; defaults match the sops
# convention.
KEY_DIR="${SOPS_AGE_KEY_DIR:-${HOME}/.config/sops/age}"
KEY_FILE="${KEY_DIR}/keys.txt"

extract_public_key() {
    # age-keygen output line: "# public key: age1..."
    grep -E '^# public key: ' "$1" | head -n1 | awk '{print $4}'
}

if [[ -f "${KEY_FILE}" ]]; then
    pub=$(extract_public_key "${KEY_FILE}" || true)
    if [[ -z "${pub}" ]]; then
        echo "ERROR: ${KEY_FILE} exists but contains no parseable public key." >&2
        echo "       Move it aside and re-run if you want a fresh key." >&2
        exit 1
    fi
    echo "age key already installed at ${KEY_FILE}"
    echo "public key: ${pub}"
    exit 0
fi

if ! command -v age-keygen >/dev/null 2>&1; then
    echo "ERROR: age-keygen not on PATH." >&2
    echo "       Install via apt/brew, copy from /var/cache/devai/pip/bin/age-keygen, or" >&2
    echo "       run 'make fetch-cli' to populate the cache." >&2
    exit 1
fi

umask 077
mkdir -p "${KEY_DIR}"
chmod 0700 "${KEY_DIR}"
age-keygen -o "${KEY_FILE}"
chmod 0600 "${KEY_FILE}"

pub=$(extract_public_key "${KEY_FILE}")
echo
echo "Generated new age keypair at ${KEY_FILE} (mode 0600)."
echo
echo "  public key: ${pub}"
echo
echo "Next step: add this line to .sops.yaml under the age: list, then"
echo "run 'sops updatekeys deploy/*.sops.env' on every encrypted file."
