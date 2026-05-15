# Secrets management with sops + age

devai's shared secret store uses [sops](https://github.com/getsops/sops)
+ [age](https://github.com/FiloSottile/age) for encryption-at-rest of
any credential the project needs at runtime: MCP gateway tokens
(GitHub PAT, HuggingFace token, Firecrawl key, Context7 key),
gpu-arbiter cluster bearer tokens, SkyPilot cloud credentials.

This file is the canonical reference for the **scaffold**. Each
consumer plan adds its own per-file rule and ships its own
`*.sops.env`; the underlying setup / rotation / recovery story lives
here so consumers do not duplicate it.

## Why sops + age

- Encrypted at rest, with **per-value** diffs in git so a teammate can
  see *what* changed without seeing the secret itself.
- Single binary on either side -- no service to run, no Docker Desktop
  dependency, no Vault to operate.
- Recipient list lives in `.sops.yaml` and grows by one line per host
  -- scales cleanly when cluster mode adds a second host.

Alternatives considered and rejected during the 2026-05-14 mcp-gateway
Phase 2 design review: plain `.env` 0600 (no encryption at rest),
podman `file`/`pass` drivers (per-runtime coupling), systemd-creds
(host-specific, harder to share across hosts), Vault / OpenBao (runs a
service, overkill for single-user lab).

Full rationale: `docs/plans/sops-age-secrets.md` "Confirmed decisions"
item 1.

## One-time setup

### 1. Install sops and age

The lab image ships both via `make fetch-cli`. On a host that runs the
compose stack outside the lab container:

```bash
make fetch-cli
sudo install /var/cache/devai/pip/bin/sops      /usr/local/bin/sops
sudo install /var/cache/devai/pip/bin/age       /usr/local/bin/age
sudo install /var/cache/devai/pip/bin/age-keygen /usr/local/bin/age-keygen
```

Distro packages also work (`apt install sops age` on Debian Trixie
backports, `brew install sops age` on macOS).

### 2. Generate this host's age keypair

```bash
bash scripts/age-keygen-host.sh
```

The script writes `~/.config/sops/age/keys.txt` with mode 0600 and
prints the public key. Idempotent -- re-running on a host that already
has a key prints the existing public key.

### 3. Add the public key to `.sops.yaml`

Edit `.sops.yaml` and replace the `age1xxx...` placeholder with the
real public key from step 2. For multi-host setups append additional
public keys to the `age:` list, one per line.

### 4. Mount the tmpfs render target

```bash
make secrets-tmpfs   # one-time; idempotent
```

Mounts `/run/devai` as a 4 MiB tmpfs with `nodev,nosuid,noexec`. Gone
on reboot; re-mount via the same command, or install
`deploy/systemd/run-devai.mount` for boot persistence.

## Editing a secrets file

```bash
make secrets-edit SOPS_FILE=deploy/mcp-secrets.sops.env
```

Opens the file in `$EDITOR` with values decrypted; saving re-encrypts
in place. No plaintext ever touches disk outside `/run/devai`.

## Rendering at startup

Each consumer plan's Makefile target wraps `scripts/render-secret.sh`:

```bash
bash scripts/render-secret.sh \
     deploy/mcp-secrets.sops.env \
     /run/devai/mcp-secrets.env
```

The render script refuses to write to a non-tmpfs path unless the
caller sets `DEVAI_RENDER_ALLOW_NON_TMPFS=1`, so a missing
`make secrets-tmpfs` step fails loudly instead of silently leaking
plaintext to a regular filesystem.

## Rotation

```bash
make secrets-rotate
```

Walks every `deploy/*.sops.env` file and runs `sops updatekeys`,
re-encrypting against the current `.sops.yaml` recipient list. Use
this after adding or removing a host's public key.

To rotate a *value* (e.g. revoked GitHub PAT):
1. `make secrets-edit SOPS_FILE=deploy/mcp-secrets.sops.env`
2. Replace the value, save.
3. Restart the consumer service so it re-reads the rendered tmpfs
   file.

## Recovery

**Lost age private key = lost access to every secret encrypted with
its public key.** sops + age has no backdoor.

Recommended posture:
- Keep an offline backup of `~/.config/sops/age/keys.txt` (e.g. on a
  printed paper or a dedicated USB stick in a safe).
- For multi-host setups, ensure at least two hosts can decrypt every
  secret -- losing one host's key doesn't lock you out as long as
  another host's key still appears in `.sops.yaml`.

If both backups and additional hosts are unavailable, the only
recovery is to rotate every secret out-of-band: regenerate API keys
upstream, re-encrypt against a new age key.

## Multi-host onboarding

The pattern is identical to "adding a teammate" today, since each
host's age key is per-user-per-host:

1. On the new host, run `bash scripts/age-keygen-host.sh`.
2. Copy the printed public key.
3. On any host with an existing checkout, append the new public key to
   `.sops.yaml` under `age:`.
4. Run `make secrets-rotate` to re-encrypt every file with the new
   recipient.
5. Commit `.sops.yaml` and the rotated `*.sops.env` files.
6. The new host now has decrypt access; pull the commit and run
   `make secrets-tmpfs && bash scripts/render-secret.sh ...` as
   needed.

## Paranoid mode (deferred)

Each of these wraps the age private key in additional protection:

- **systemd-creds**: encrypt `keys.txt` against the system's
  TPM-backed credential store; sops would unwrap on demand. Pinned to
  one boot-identity.
- **TPM seal**: lower-level analogue; encrypt against the TPM2 PCRs
  matching a known-good kernel/initramfs measurement. Survives suspend
  but not kernel updates.
- **YubiKey**: FIDO2-protected age plugin (`age-plugin-yubikey`) keeps
  the private key on hardware that can't be exfiltrated by malware.

Documented as a known gap, not built in v1. Per
`docs/plans/sops-age-secrets.md` item: "TPM / YubiKey / systemd-creds
wrapping of the age private key. Noted as a 'paranoid mode' follow-up
in the docs but not built here."

## Operator checklist before commit

- [ ] `.sops.yaml` does not still contain the literal
      `age1xxx...` placeholder string.
- [ ] No `*.env` (plaintext) files staged. The `.gitignore` blocks
      `*.env.plain`; the `!deploy/*.sops.env` exception keeps
      encrypted files trackable.
- [ ] `make secrets-rotate` completed cleanly after any change to the
      recipient list.
