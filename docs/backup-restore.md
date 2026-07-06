# Backup and restore

devai accumulates host-local state that is expensive or impossible to
regenerate: probe/bench caches (real GPU time to rebuild), the
`~/.devai/` preferences and agent session history, and the sops+age
secret scaffold's private key. None of this is backed up by git --
the caches are gitignored by design (host-specific, VRAM-dependent),
and the age key must never be committed. `devai-backup`
(`devai-tools/cmd/devai-backup`) snapshots and restores exactly this
set.

## What gets backed up

| Source | Why | If missing |
|---|---|---|
| `deploy/.bench-cache.json` | bench harness results (schema v3); real GPU time to regenerate | skipped, snapshot continues |
| `deploy/.ollama-reasoning-cache.json` | Ollama probe cache | skipped, snapshot continues |
| `deploy/.vllm-reasoning-cache.json` | vLLM probe cache | skipped, snapshot continues |
| `deploy/.sglang-reasoning-cache.json` | SGLang probe cache | skipped, snapshot continues |
| `deploy/.model-status.json` | host-local model exclusion ledger | skipped, snapshot continues |
| `deploy/*.sops.env` | already git-tracked ciphertext; included anyway for a self-contained archive | skipped, snapshot continues |
| `~/.devai/preferences.yaml` | last-used model/agent/vram/context | skipped, snapshot continues |
| `~/.devai/sessions/` | per-(agent,model) session history | skipped, snapshot continues |
| `~/.config/sops/age/keys.txt` | **the** recovery-critical file -- see below | loud warning, snapshot continues |

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
- `/run/devai/` -- tmpfs, ephemeral by design; nothing there survives
  a reboot anyway.

## The age key: no backdoor

Per `docs/secrets.md`: **sops + age has no backdoor.** Losing
`~/.config/sops/age/keys.txt` means losing access to every secret
encrypted with its public key, permanently. `devai-backup` includes it
by default precisely because it is the single highest-value file in
this whole list -- opt out only if you manage that key's backup some
other way (`--exclude-age-key`).

Treat the resulting archive as secret-bearing:

- Store it on encrypted disk, in a password manager's file-attachment
  feature, or on offline media.
- **Never** commit it to git, and never upload it to unencrypted cloud
  storage.
- The `sops-age/keys.txt` entry inside the archive is restored with
  its original permission bits, and `devai-backup restore` re-asserts
  `0600` on it as defense-in-depth regardless.

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
devai-backup snapshot [--dest DIR] [--exclude-age-key]
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

- A partial archive (e.g. one that only contains the age key) never
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

# 4. Verify sops still decrypts against the restored key.
make secrets-render SOPS_FILE=deploy/mcp-secrets.sops.env DEST=/tmp/check.env
```

See `docs/secrets.md` for the sops+age scaffold itself, and
`docs/backends.md` for what the probe/bench caches actually contain.
