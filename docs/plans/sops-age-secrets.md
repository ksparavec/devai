# sops/age Shared Secret Store

> **NON-FUNCTIONAL as of 2026-07-25.** The scaffold exists and its unit
> tests pass, but it has never performed a real encrypt or decrypt on
> this host: `.sops.yaml` still carries the literal `age1xxxx...`
> placeholder recipient, no age key has ever been generated
> (`~/.config/sops/age/keys.txt` is absent), `/run/devai` has never been
> mounted, and only `.example` files exist in `deploy/`.
>
> The test that should have caught the placeholder passes on it -- its
> regex `age1[0-9a-zA-Z]{30,}` matches the placeholder itself.
>
> Two of the three consumers this plan was built to serve (cluster mode,
> the SkyPilot fleet provisioner) are now frozen. The only live consumer
> is the MCP gateway's optional secret-bearing servers, and those are not
> required for the gateway to work.


_Establish the sops + age encrypted-at-rest secret-store scaffold once,
in one place, so MCP gateway Phase 2, gpu-arbiter cluster mode, and the
SkyPilot fleet provisioner all reuse the same pattern instead of each
re-implementing it._

## Status

**In Progress.** Scaffold shipped 2026-05-15 (one PR, all
deliverables landed). `scripts/age-keygen-host.sh`,
`scripts/render-secret.sh`, `deploy/setup-secrets-tmpfs.sh`,
`.sops.yaml` (placeholder rule), `Makefile` targets
(`secrets-tmpfs` / `secrets-edit` / `secrets-render` /
`secrets-rotate` / `age-keygen-host`), `fetch-cli` block for sops
+ age binaries, `docs/secrets.md`, `.gitignore` exception.
Stdlib-unittest coverage in
`tests/python/test_sops_age_scaffold.py` for the script gates,
.sops.yaml shape, idempotency, and the non-tmpfs refusal path.
Real binary fetch (`make fetch-cli`) and live encrypt/decrypt
round-trip require an actual age keypair on a host -- deferred to
the operator's first real `age-keygen-host` run.

## Dependencies

None.

## Enables / Unblocks

- [Plan: mcp-gateway](./mcp-gateway.md) Phase 2 -- Tier 2 servers
  (`github-official`, `firecrawl`, `hugging-face`, `context7`) need
  per-server secrets at tool-call time.
- [Plan: gpu-arbiter-cluster-mode](./gpu-arbiter-cluster-mode.md)
  Phase 1 -- worker/head bearer token (decision 8).
- [Plan: skypilot-fleet-provisioner](./skypilot-fleet-provisioner.md)
  Phase 1 -- cloud-account credentials for non-interactive providers
  (RunPod, Lambda) plus the gpu-arbiter <-> SkyPilot API server
  bearer token (decision 3).
- Any future credential-bearing integration (HuggingFace, GitHub,
  Firecrawl, etc.) inherits the same scaffold without re-design.

## Out of scope

- Per-user / multi-tenant secret scoping. devai is single-user-per-host
  today; one age key per host. Multi-tenant lifts the design later.
- TPM / YubiKey / systemd-creds wrapping of the age private key.
  Noted as a "paranoid mode" follow-up in the docs but not built
  here.
- Secret distribution across cluster hosts. When cluster mode lands
  the second host, that plan's own "secret distribution" deliverable
  can build on top of this scaffold (add the new host's public key
  to `.sops.yaml`, run `sops updatekeys`); no new infra here.
- Migration of any existing plaintext credentials. None exist in the
  repo today; net-new flow only.

## Confirmed decisions

Confirmed 2026-05-14 from the mcp-gateway Phase 2 discussion. Future
deviations require an explicit plan amendment.

1. **Tool choice: sops + age.** Reasons: encrypted at rest,
   git-friendly per-value diffs, one tool, no service, no Docker
   Desktop, scales to cluster mode by adding host public keys to
   the recipients list. Alternatives rejected during the 2026-05-14
   mcp-gateway Phase 2 discussion: plain `.env` 0600 (no encryption
   at rest), podman `file`/`pass` drivers (per-runtime coupling),
   systemd-creds (host-specific, harder to share across hosts), and
   Vault / OpenBao (runs a service, overkill for single-user lab).
2. **Age key custody: user-scoped `~/.config/sops/age/keys.txt`,
   mode 0600.** Single-user lab today; user owns the key; the
   render script reads it via bind-mount; no root required. When
   cluster mode lands, the second host generates its own age key
   and the operator adds its public key to `.sops.yaml`.
3. **Render target: tmpfs at `/run/devai`** with
   `nodev,nosuid,noexec`, size 4M, mode 0700. Idempotent setup
   script; optional systemd-mount unit for boot persistence. Each
   consumer (MCP gateway, cluster head, SkyPilot API server) reads
   a distinct `*.env` file under `/run/devai/`.
4. **Binary distribution: pre-fetched via `Makefile:fetch-cli`,
   same ETag/version-stamp pattern as the existing CLIs.** sops
   and age are not in Debian Trixie's default apt-cacher repos
   reliably; pre-fetching keeps image builds offline-safe.
5. **`.sops.yaml` at repo root with per-file regex rules.** One
   `creation_rules` entry per `deploy/*.sops.env` file. All consumer
   plans add their own entry; this plan ships the initial file with
   a single placeholder rule that consumer plans copy from.

## Context

Three approved plans declare a "soft dep on mcp-gateway Phase 2's
sops/age pattern" with the phrasing "whichever lands second inherits
the infrastructure." That phrasing hides a real cost: **whichever
lands first has to build the scaffold themselves**, even though it
is not the load-bearing work of that plan. Without an explicit
shared-infra plan, the scaffold either gets duplicated (three plans
each carry their own fetch-cli additions, tmpfs setup, render script,
`.sops.yaml`) or one of the three quietly grows past its scope to
host it.

This plan exists to factor the scaffold out so each consumer plan
can declare a clean hard dependency on it and concentrate on its
own surface (catalog YAML, bearer-token validation, cloud-credential
mount).

## Approach

Ship the four pieces of infrastructure that every consumer needs:

1. sops and age binaries available in the image build cache.
2. Tmpfs at `/run/devai` mounted before any service that consumes a
   rendered secret.
3. A render script that decrypts one `deploy/*.sops.env` file into
   `/run/devai/<name>.env` with `umask 077` + `chmod 0600`.
4. `.sops.yaml` at the repo root with an example rule consumer plans
   copy from.

Plus the operator workflow (`make secrets-edit`, `make secrets-render`,
`make secrets-rotate`) and the install-time helper that generates and
locates the age key.

The plan is single-phase: every deliverable is needed before any
consumer plan can use the scaffold, so there is no smaller shippable
intermediate.

---

## Implementation

### Deliverables

```
.sops.yaml                          (new at repo root: encryption rules
                                      + age recipients list; ships with
                                      one placeholder rule consumers
                                      copy from)
scripts/
  fetch-cli.sh                      (modify: add sops + age binaries to
                                      the CLI fetch list, pinned versions,
                                      checksum-verified, ETag-stamped at
                                      $(ETAG_DIR)/{sops,age}.version)
  age-keygen-host.sh                (new: one-shot install-time helper;
                                      generates ~/.config/sops/age/keys.txt
                                      mode 0600, prints public key, hints
                                      the user to add it to .sops.yaml)
  render-secret.sh                  (new: generic single-file render --
                                      `render-secret.sh <sops-input>
                                      <tmpfs-output>`; umask 077; chmod
                                      0600 after write; idempotent)
deploy/
  setup-secrets-tmpfs.sh            (new: idempotent tmpfs mount at
                                      /run/devai with nodev,nosuid,noexec,
                                      size 4M, mode 0700; optional
                                      systemd-mount unit shipped alongside
                                      for boot persistence)
Makefile                            (new targets: secrets-tmpfs,
                                      secrets-render, secrets-edit,
                                      secrets-rotate; document the
                                      cache-up dependency consumer plans
                                      add)
docs/
  secrets.md                        (new: setup, rotation, recovery,
                                      multi-host onboarding, paranoid-mode
                                      pointer; the canonical reference
                                      consumer plans link to)
.gitignore                          (add: /run/devai, *.env.plain;
                                      explicit !deploy/*.sops.env to make
                                      the encrypted files trackable)
```

### Detailed steps

1. **Pin sops and age versions** and add them to
   `scripts/fetch-cli.sh` following the existing per-binary pattern
   (curl -fsSL release URL into `$(CACHE_DIR)/bin/`, verify sha256
   against a pinned digest, write `$(ETAG_DIR)/sops.version` /
   `$(ETAG_DIR)/age.version` for change detection). Both binaries
   are single static Go binaries on Linux x86_64 -- straightforward
   fetch.

2. **`.sops.yaml` at repo root** with one placeholder rule and a
   comment explaining how consumer plans extend it:
   ```yaml
   # Each devai plan that ships an encrypted secrets file appends
   # a creation_rules entry here. The age: list is the union of
   # every host that should be able to decrypt any secret in this
   # repo. Add a host: run age-keygen-host.sh on that host, copy
   # the printed public key, append to age: below, run
   # `sops updatekeys deploy/*.sops.env`, commit.
   creation_rules:
     - path_regex: deploy/.*\.sops\.env$
       encrypted_regex: ^.*$
       age: >-
         age1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```
   The single `age1xxx...` placeholder is replaced with the real
   public key the first operator generates via
   `scripts/age-keygen-host.sh`. Subsequent hosts append their
   public keys to the same list.

3. **`scripts/age-keygen-host.sh`** -- one-shot helper:
   ```
   - If ~/.config/sops/age/keys.txt exists, print "already
     installed" + the public key, exit 0.
   - Otherwise: mkdir -p ~/.config/sops/age, chmod 0700 the dir,
     age-keygen -o ~/.config/sops/age/keys.txt, chmod 0600 the
     file, parse the public key out of the generated file, print
     it with a clear "add this line to .sops.yaml under age:"
     instruction.
   - Wire into `make install` so first-time operators get the key
     without reading docs.
   ```
   Multi-host / future cluster path is the same script run on the
   second host -- prints its public key, operator adds it to
   `.sops.yaml`, runs `sops updatekeys` on every encrypted file.

4. **`scripts/render-secret.sh`** -- generic single-file renderer:
   ```bash
   #!/bin/bash
   set -euo pipefail
   umask 077
   src="${1:?usage: render-secret.sh <sops-input> <tmpfs-output>}"
   dst="${2:?usage: render-secret.sh <sops-input> <tmpfs-output>}"
   sops --decrypt "$src" > "$dst"
   chmod 0600 "$dst"
   ```
   Consumer plans call this from their own Makefile target with
   their specific input/output pair (e.g., MCP gateway calls it
   with `deploy/mcp-secrets.sops.env` ->
   `/run/devai/mcp-secrets.env`).

5. **`deploy/setup-secrets-tmpfs.sh`** -- idempotent mount:
   ```
   - If /run/devai is already a tmpfs mount, exit 0.
   - mkdir -p /run/devai
   - mount -t tmpfs -o nodev,nosuid,noexec,size=4m,mode=0700 \
       tmpfs /run/devai
   - Optional: ship deploy/systemd/run-devai.mount alongside for
     boot persistence; not auto-installed, documented in
     docs/secrets.md.
   ```

6. **Makefile targets**:
   ```make
   secrets-tmpfs:    ; sudo bash deploy/setup-secrets-tmpfs.sh
   secrets-edit:     ; sops $(SOPS_FILE)         # consumer plans set SOPS_FILE
   secrets-render:   ; @echo "consumer plans override this; see docs/secrets.md"
   secrets-rotate:   ; for f in deploy/*.sops.env; do sops updatekeys "$$f"; done
   ```
   Consumer plans add a target like
   `mcp-secrets-render: secrets-tmpfs ; bash scripts/render-secret.sh deploy/mcp-secrets.sops.env /run/devai/mcp-secrets.env`
   and make their own `cache-up` depend on it.

7. **`docs/secrets.md`** -- canonical reference:
   - Why sops/age (one-paragraph summary; full rationale and
     rejected alternatives are in this plan's Confirmed
     decisions item 1).
   - Setup walkthrough: install sops/age (already in the image),
     run `age-keygen-host.sh`, add public key to `.sops.yaml`,
     `sops updatekeys` on any existing files.
   - Rotation: edit via `make secrets-edit SOPS_FILE=...`,
     re-render, restart consumer service.
   - Recovery: lost age key = lost secrets; offline backup pattern;
     accept data loss as last resort.
   - Multi-host onboarding: identical to "adding a team member"
     for now -- second host gets its own age key, public key
     appended to `.sops.yaml`, `sops updatekeys`, commit.
   - Paranoid-mode pointer: systemd-creds wrap, TPM seal, YubiKey
     hardware key. One paragraph each, with links; deferred work.

8. **`.gitignore`**: add `/run/devai`, `*.env.plain`. Explicit
   `!deploy/*.sops.env` to ensure encrypted files stay tracked
   even if a broader rule shadows them.

### Exit criteria

- `scripts/age-keygen-host.sh` on a clean host produces a working
  keypair and prints the public key.
- `sops --encrypt`/`--decrypt` round-trip works against a file
  matching the placeholder rule once a real public key is in
  `.sops.yaml`.
- `scripts/render-secret.sh` writes to `/run/devai/` with mode
  0600 and never leaves a plaintext copy anywhere else.
- `make secrets-rotate` re-keys every `deploy/*.sops.env` file
  without touching plaintext.
- `docs/secrets.md` covers setup / rotation / recovery /
  multi-host with concrete commands.
- Each consumer plan (mcp-gateway Phase 2, cluster-mode Phase 1,
  fleet-provisioner Phase 1) can declare this plan as a hard
  dependency and inherit the scaffold without copy-paste.

### Risks

| Risk                                                  | Mitigation                                                |
| ----------------------------------------------------- | --------------------------------------------------------- |
| sops or age binary release URL changes shape          | Pin to specific GitHub release tag + sha256; fetch-cli surfaces the failure on next refresh |
| Age private key file becomes the new single point of failure | Layered hardening documented (systemd-creds, TPM, YubiKey); offline backup pattern in docs/secrets.md |
| Lost age key locks out all secrets                    | Documented recovery: restore from offline backup OR rotate every secret manually |
| Tmpfs `/run/devai` gone on host reboot                | secrets-render is idempotent and runs from cache-up; documented |
| Non-standard $XDG_RUNTIME_DIR or non-systemd hosts    | systemd-mount unit shipped but not auto-installed; tmpfs setup script falls back to bare `mount` |
| `.sops.yaml` placeholder shipped to git with the dummy `age1xxx...` recipient | Pre-commit hook check (or CI lint) that rejects the literal placeholder string; documented in docs/secrets.md |
| Operator skips age-keygen-host.sh on a second host    | Render script fails loudly ("no matching key") -- operator runs the helper, no silent fallback |

## Migration / rollback story

- **Rollback**: revert this plan's commits. No consumer plan has
  shipped yet at the time this plan lands (it is a prerequisite),
  so rollback affects only the scaffold itself. Encrypted files
  remain unreadable but are not referenced by any running service.
- **Upgrade path**: net-new infrastructure. No existing devai
  install has secrets today; first-time operators run
  `age-keygen-host.sh` as part of `make install` and proceed.

## Estimated effort

| Phase    | Engineering effort                                | Wall-clock |
| -------- | ------------------------------------------------- | ---------- |
| Single   | ~1 PR, ~300 lines of bash + Makefile + docs       | 1-2 days   |

## References

- sops: github.com/getsops/sops (CNCF sandbox)
- age: github.com/FiloSottile/age
- Rationale and alternatives considered: Confirmed decisions
  item 1 above. Discussion provenance: 2026-05-14 mcp-gateway
  Phase 2 design review.
- Consumers:
  - [mcp-gateway](./mcp-gateway.md) Phase 2 (Tier 2 server secrets)
  - [gpu-arbiter-cluster-mode](./gpu-arbiter-cluster-mode.md)
    Phase 1 (worker bearer token)
  - [skypilot-fleet-provisioner](./skypilot-fleet-provisioner.md)
    Phase 1 (cloud creds + API server bearer token)
