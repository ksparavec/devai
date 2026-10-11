# sops/age secret-store scaffold -- restore notes

Moved here on 2026-10-07 by platform-rebuild Phase 1, operator decision
M12: "The sops/age secrets scaffold moves to `attic/`. Its only remaining
consumer was the MCP gateway's two servers that needed secrets. The Slurm
plan can restore it if it wants its JWT key and MariaDB password
encrypted."

## State at freeze time

1. **No consumer.** The MCP gateway, its last consumer, was deleted the
   same day (M10); the cluster-mode and SkyPilot-fleet consumers were
   frozen on 2026-07-25 (`../cluster-mode/`). Its delivery path to the MCP
   gateway was never verified end to end.
2. **Nothing was ever encrypted with it.** No `deploy/*.sops.env` was ever
   committed; only `.example` templates were.
3. **`.sops.yaml` still carries the literal `age1xxx...` placeholder**, and
   its test (`TestSopsYaml.test_contains_placeholder_or_real_key`) passes on
   it (see `../cluster-mode/RESTORE.md`, defect 4).
4. **The binaries were never pinned.** `make fetch-cli` downloaded the
   latest sops and age releases from GitHub on every run; no image
   installed them, so the scaffold used whatever was on the host's PATH.

## What was moved here

```
sops-age/
  .sops.yaml
  scripts/age-keygen-host.sh, scripts/render-secret.sh
  deploy/setup-secrets-tmpfs.sh
  docs/secrets.md
  tests/python/test_sops_age_scaffold.py
  Makefile.frozen-targets   secrets-tmpfs, secrets-edit, secrets-render,
                            secrets-rotate, age-keygen-host, mcp-secrets-render
                            (whose consumer is gone), SOPS_FILE, and fetch-cli's
                            sops and age download blocks, verbatim
  frozen-branches.txt       the .gitignore block, and devai-backup's age-key
                            handling, verbatim
```

## What changed outside this directory

- **devai-backup no longer backs up `~/.config/sops/age/keys.txt`.** The
  manifest entry, the `sops-age` archive root, the missing-key warning, the
  0600 re-assert on restore and `--exclude-age-key` are gone. Restoring an
  archive made before 2026-10-07 skips its `sops-age/...` members with a
  warning naming this directory and restores the rest. Encrypted files
  matching `deploy/*.sops.env` are still backed up if any appear.
- **A key already on a host stays where it is.** Nothing deletes
  `~/.config/sops/age/keys.txt`, but no backup carries it any more: keep an
  offline copy if anything encrypted with it still matters.
- **`.gitleaks.toml` keeps its two `*.sops.env` path rules:** the scan
  covers the whole history, which still holds the deleted
  `deploy/mcp-secrets.sops.env.example`.

## How to thaw

1. Move the files back:

   ```
   git mv attic/sops-age/.sops.yaml .
   git mv attic/sops-age/scripts/age-keygen-host.sh attic/sops-age/scripts/render-secret.sh scripts/
   git mv attic/sops-age/deploy/setup-secrets-tmpfs.sh deploy/
   git mv attic/sops-age/docs/secrets.md docs/
   git mv attic/sops-age/tests/python/test_sops_age_scaffold.py tests/python/
   ```

2. Paste the Makefile targets back from `Makefile.frozen-targets`, and pin
   the sops and age versions in the fetch-cli blocks instead of fetching
   `latest`. Drop `mcp-secrets-render` unless a new consumer needs that
   shape.
3. Restore the `.gitignore` block from `frozen-branches.txt`;
   `TestGitignoreCovers` in the scaffold's test requires it.
4. Replace the `age1xxx...` placeholder in `.sops.yaml` with real
   recipients (`make age-keygen-host` prints one), and tighten the test so
   the placeholder fails.
5. Decide whether devai-backup should carry the age key again. If it
   should: put the code in `frozen-branches.txt` back, remove `sops-age`
   from `retiredArchiveRoots` in `devai-tools/internal/backup/manifest.go`
   (or the restore skip swallows it), and bring back the
   `root-depth-mismatch-escape` case in `malicious_test.go`.

The history before the move: `git log --follow -- attic/sops-age/docs/secrets.md`.
