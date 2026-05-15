#!/usr/bin/env bash
# Decrypt a single sops-encrypted .env file into a tmpfs-backed target
# under /run/devai/, with mode 0600 and no plaintext intermediate.
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
if [[ -d "${dst_dir}" ]] && [[ "${DEVAI_RENDER_ALLOW_NON_TMPFS:-0}" != "1" ]]; then
    fs_type=$(stat -f -c '%T' "${dst_dir}" 2>/dev/null || echo "")
    case "${fs_type}" in
        tmpfs|ramfs)
            ;;
        "")
            echo "WARN: cannot determine filesystem type of ${dst_dir}; proceeding." >&2
            ;;
        *)
            echo "ERROR: ${dst_dir} is not a tmpfs (${fs_type}). Refusing to write" >&2
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
sops --decrypt "${src}" > "${dst}"
chmod 0600 "${dst}"
echo "rendered ${src} -> ${dst} (mode 0600)"
