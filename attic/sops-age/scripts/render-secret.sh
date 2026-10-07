#!/usr/bin/env bash
# Decrypt a single sops-encrypted .env file into a tmpfs-backed target
# under /run/devai/, with mode 0600. The decrypt lands in a temp file in
# the SAME (tmpfs) directory and is renamed into place only on success,
# so a failed decrypt never truncates a good rendered secret -- no
# plaintext ever reaches a persistent filesystem.
#
# Usage:
#   bash scripts/render-secret.sh <sops-input> <tmpfs-output>
#
# Example (from mcp-gateway Phase 2):
#   bash scripts/render-secret.sh \
#        deploy/mcp-secrets.sops.env \
#        /run/devai/mcp-secrets.env
#
# Pre-conditions:
#   - sops on PATH (provided by 'make fetch-cli' or the lab image)
#   - ~/.config/sops/age/keys.txt exists (run scripts/age-keygen-host.sh)
#   - /run/devai exists and is a tmpfs mount (run 'make secrets-tmpfs' once)
#
# Idempotent: writing the same plaintext twice is a no-op for the
# consumer -- but the file is rewritten unconditionally so the
# permission bits get re-asserted in case something weakened them.

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: render-secret.sh <sops-input> <tmpfs-output>" >&2
    exit 2
fi

src="$1"
dst="$2"

if [[ ! -f "${src}" ]]; then
    echo "ERROR: encrypted source not found: ${src}" >&2
    exit 1
fi

# Refuse to render to a non-tmpfs path unless the caller explicitly
# overrides via DEVAI_RENDER_ALLOW_NON_TMPFS=1. Plaintext on a regular
# filesystem defeats the purpose of the secret store.
dst_dir=$(dirname "${dst}")
if [[ "${DEVAI_RENDER_ALLOW_NON_TMPFS:-0}" != "1" ]]; then
    # The destination directory may not exist yet -- which is exactly the
    # case this gate exists for, since the mkdir -p below would otherwise
    # happily create it on persistent storage. Walk up to the nearest
    # existing ancestor and check THAT filesystem.
    probe_dir="${dst_dir}"
    while [[ ! -d "${probe_dir}" ]] && [[ "${probe_dir}" != "/" ]] \
          && [[ "${probe_dir}" != "." ]]; do
        probe_dir=$(dirname "${probe_dir}")
    done
    fs_type=$(stat -f -c '%T' "${probe_dir}" 2>/dev/null || echo "")
    case "${fs_type}" in
        tmpfs|ramfs)
            ;;
        "")
            echo "WARN: cannot determine filesystem type of ${probe_dir}; proceeding." >&2
            ;;
        *)
            echo "ERROR: ${probe_dir} is not a tmpfs (${fs_type}). Refusing to write" >&2
            echo "       plaintext to a persistent filesystem. Run 'make secrets-tmpfs'" >&2
            echo "       first, or set DEVAI_RENDER_ALLOW_NON_TMPFS=1 to override." >&2
            exit 1
            ;;
    esac
fi

if ! command -v sops >/dev/null 2>&1; then
    echo "ERROR: sops not on PATH." >&2
    echo "       Install via apt/brew, or copy from /var/cache/devai/pip/bin/sops." >&2
    exit 1
fi

umask 077
mkdir -p "${dst_dir}"

# Decrypt into a temp file in the SAME directory so the final mv is a
# same-filesystem rename (atomic). Writing straight to ${dst} truncates
# it before sops runs, so a failed decrypt -- wrong age key, corrupt
# input, unreachable KMS -- would replace a good rendered secret with a
# 0-byte file and consumers would silently start up unauthenticated.
tmp_dst=$(mktemp "${dst_dir}/.$(basename "${dst}").XXXXXX")
trap 'rm -f "${tmp_dst}"' EXIT
if ! sops --decrypt "${src}" > "${tmp_dst}"; then
    echo "ERROR: sops --decrypt failed for ${src}." >&2
    echo "       ${dst} left unchanged." >&2
    exit 1
fi
chmod 0600 "${tmp_dst}"
mv -f "${tmp_dst}" "${dst}"
trap - EXIT
echo "rendered ${src} -> ${dst} (mode 0600)"
