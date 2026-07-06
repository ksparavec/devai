#!/usr/bin/env bash
# End-to-end smoke test for devai-backup (docs/backup-restore.md).
#
# Builds the real binary and drives it as a subprocess against temp
# directories standing in for deploy/, ~/.devai/, and ~/.config/sops/age/
# (via --repo-root/--home-dir overrides), so the real $HOME is never
# touched. Runs snapshot -> list -> verify -> delete originals ->
# restore --yes -> diff.
#
# No exit-77 skip -- the binary is always buildable locally, no external
# prerequisite (unlike the GPU-backed probe/bench tests).
#
# Usage: ./tests/test-backup-restore.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${REPO_ROOT}/devai-tools/bin/devai-backup"

WORKDIR="$(mktemp -d)"
trap 'rm -rf "${WORKDIR}"' EXIT

FAKE_REPO="${WORKDIR}/repo"
FAKE_HOME="${WORKDIR}/home"
BACKUP_DIR="${WORKDIR}/backups"

echo ">>> test-backup-restore: building devai-backup"
if [[ ! -x "${BIN}" ]]; then
    (cd "${REPO_ROOT}/devai-tools" && go build -o bin/devai-backup ./cmd/devai-backup)
fi

echo ">>> seeding fake repo + home state"
mkdir -p "${FAKE_REPO}/deploy" "${FAKE_HOME}/.devai/sessions" "${FAKE_HOME}/.config/sops/age"
echo '{"bench":"fixture"}' > "${FAKE_REPO}/deploy/.bench-cache.json"
echo 'vram: 24' > "${FAKE_HOME}/.devai/preferences.yaml"
echo '{"turn":1}' > "${FAKE_HOME}/.devai/sessions/claude.jsonl"
echo 'AGE-SECRET-KEY-1FIXTUREFIXTUREFIXTURE' > "${FAKE_HOME}/.config/sops/age/keys.txt"
chmod 600 "${FAKE_HOME}/.config/sops/age/keys.txt"

echo ">>> snapshot"
ARCHIVE="$("${BIN}" snapshot --repo-root "${FAKE_REPO}" --home-dir "${FAKE_HOME}" --dest "${BACKUP_DIR}")"
[[ -f "${ARCHIVE}" ]] || { echo "FAIL: snapshot did not produce ${ARCHIVE}" >&2; exit 1; }

echo ">>> list"
"${BIN}" list --dest "${BACKUP_DIR}" --home-dir "${FAKE_HOME}" | grep -q "${ARCHIVE}" \
    || { echo "FAIL: list did not report ${ARCHIVE}" >&2; exit 1; }

echo ">>> verify"
"${BIN}" verify --archive "${ARCHIVE}"

echo ">>> deleting originals"
rm -rf "${FAKE_REPO}" "${FAKE_HOME}"

echo ">>> restore --yes"
"${BIN}" restore --archive "${ARCHIVE}" --repo-root "${FAKE_REPO}" --home-dir "${FAKE_HOME}" --yes

echo ">>> diffing restored content"
[[ "$(cat "${FAKE_REPO}/deploy/.bench-cache.json")" == '{"bench":"fixture"}' ]] \
    || { echo "FAIL: bench cache content mismatch after restore" >&2; exit 1; }
[[ "$(cat "${FAKE_HOME}/.devai/preferences.yaml")" == 'vram: 24' ]] \
    || { echo "FAIL: preferences content mismatch after restore" >&2; exit 1; }
[[ "$(cat "${FAKE_HOME}/.devai/sessions/claude.jsonl")" == '{"turn":1}' ]] \
    || { echo "FAIL: session content mismatch after restore" >&2; exit 1; }
[[ "$(cat "${FAKE_HOME}/.config/sops/age/keys.txt")" == 'AGE-SECRET-KEY-1FIXTUREFIXTUREFIXTURE' ]] \
    || { echo "FAIL: age key content mismatch after restore" >&2; exit 1; }

MODE="$(stat -c '%a' "${FAKE_HOME}/.config/sops/age/keys.txt" 2>/dev/null || stat -f '%Lp' "${FAKE_HOME}/.config/sops/age/keys.txt")"
[[ "${MODE}" == "600" ]] || { echo "FAIL: age key mode = ${MODE}, want 600" >&2; exit 1; }

echo ">>> test-backup-restore OK"
exit 0
