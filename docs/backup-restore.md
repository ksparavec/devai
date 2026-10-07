# Backup and restore

devai accumulates host-local state that is expensive or impossible to
regenerate: probe/bench caches (real GPU time to rebuild) and the
`~/.devai/` preferences and agent session history. None of this is
backed up by git -- the caches are gitignored by design (host-specific,
VRAM-dependent). `devai-backup` (`devai-tools/cmd/devai-backup`)
snapshots and restores exactly this set.

## What gets backed up

| Source | Why | If missing |
|---|---|---|
| `deploy/.bench-cache.json` | bench harness results (schema v3); real GPU time to regenerate | skipped, snapshot continues |
| `deploy/.ollama-reasoning-cache.json` | Ollama probe cache | skipped, snapshot continues |
| `deploy/.vllm-reasoning-cache.json` | vLLM probe cache | skipped, snapshot continues |
| `deploy/.sglang-reasoning-cache.json` | SGLang probe cache | skipped, snapshot continues |
| `deploy/.model-status.json` | host-local model exclusion ledger | skipped, snapshot continues |
| `deploy/*.sops.env` | encrypted secrets files, if any exist (none do since the sops/age scaffold moved to the attic; the glob stays) | skipped, snapshot continues |
| `~/.devai/preferences.yaml` | last-used model/agent/vram/context | skipped, snapshot continues |
| `~/.devai/sessions/` | per-(agent,model) session history | skipped, snapshot continues |

Everything else under `~/.devai/` (the picker script, the probe-cache
symlinks `make install` stages there) is a symlink back into this
repo's `deploy/` tree, already covered by the `deploy/*` rows above --
`devai-backup` skips symlinks explicitly so they are never
double-counted.

**Explicitly excluded, on purpose:**

- `/var/cache/devai/{ollama,pip,registry}/` -- these are
  external-volume mount points (dedicated LVs), not ordinary
  directories (see CLAUDE.md's `/var/cache/devai/` convention), and
  hold re-downloadable model weights that would make an archive
  enormous for no recovery benefit.
- `~/.config/sops/age/keys.txt` -- no longer backed up since
  2026-10-07, when the sops/age scaffold moved to `attic/sops-age/`
  (platform-rebuild Phase 1, M12). A key already on the host is left
  where it is; keep an offline copy yourself if anything encrypted with
  it still matters.

## Archives made before 2026-10-07

Older archives carry the age key as `sops-age/keys.txt`. Restoring one
skips every `sops-age/...` member with a warning on stderr naming
`attic/sops-age`, writes nothing under `~/.config/sops`, and restores
the rest. The skip comes after the traversal check, so a hostile name
such as `sops-age/../x` is still rejected, not skipped.

Such an archive holds a private key: keep it on encrypted disk or
offline media, and never commit it or upload it to unencrypted storage.

## Commands

```bash
make backup-create [DEST=<dir>]              # snapshot -> ~/.devai/backups/<timestamp>.tar.gz
make backup-list   [DEST=<dir>]               # JSON: path, size, mtime, top-level dirs
make backup-verify ARCHIVE=<path>             # validate without extracting
make backup-restore ARCHIVE=<path> YES=1      # destructive; YES=1 required
```

Or call the binary directly (`devai-tools/bin/devai-backup`, built by
`make build-backup-tool`):

```bash
devai-backup snapshot [--dest DIR]
devai-backup list     [--dest DIR]
devai-backup verify   --archive PATH
devai-backup restore  --archive PATH --yes
```

`--dest` (or `$DEVAI_BACKUP_DIR`) defaults to `~/.devai/backups/`. The
archive filename is always auto-generated (`<UTC timestamp>.tar.gz`) so
`list`/`verify`/`restore` can rely on a predictable naming convention.

## Restore semantics

`restore` validates every entry in the archive **before writing
anything** -- a corrupt or hostile archive (path traversal, symlink
entries, absolute paths) is rejected up front and touches nothing.

For each entry whose target already exists on disk, `restore` renames
it aside to `<path>.before-restore-<timestamp>` (one shared timestamp
for the whole run) rather than deleting it. This means:

- A partial archive (e.g. one that only contains the bench cache) never
  destroys unrelated files -- only the paths actually present in the
  archive are touched.
- Restoring never silently clobbers state you didn't mean to
  overwrite; the pre-restore version is always recoverable from the
  `.before-restore-*` sibling until you clean it up yourself.

`restore` requires `--yes`; there is no interactive prompt.

## Recovery walkthrough

```bash
# 1. Find the archive to restore from.
make backup-list

# 2. Confirm it's intact before trusting it.
make backup-verify ARCHIVE=~/.devai/backups/20260615T030000Z.tar.gz

# 3. Restore (on a fresh host, or after losing local state).
make backup-restore ARCHIVE=~/.devai/backups/20260615T030000Z.tar.gz YES=1
```

See `docs/backends.md` for what the probe/bench caches actually contain.
