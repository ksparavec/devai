# SkyPilot Agent Skill in Lab

_Pre-install the SkyPilot Agent Skill plugin and the SkyPilot CLI in the
devai lab container so users running CLI agents (Claude Code, Codex,
Cursor, GitHub Copilot) can launch GPU jobs through natural-language
instructions._

## Status

Design **approved 2026-05-14** -- all six open questions resolved
(see "Confirmed decisions" below). Can be implemented independently
of the SkyPilot Fleet Provisioner plan; not yet scheduled.

## Dependencies

None hard. This plan is additive to the existing lab container build and
does not require the head/worker split or any gpu-arbiter changes.

Soft dependency: user must have their own cloud credentials available
(AWS / GCP / Azure / RunPod / etc.) for any actual provisioning to work.
Documenting that prerequisite is part of this plan's deliverables.

## Enables / Unblocks

- Users in JupyterLab can ask a CLI agent ("Claude, spin up an A100 and
  fine-tune this model") and have it work end-to-end.
- Bridges devai's local inference focus with on-demand training workflows
  without forcing devai itself to handle training.
- Each user manages their own cloud accounts and budgets -- devai stays
  out of the billing path.
- Independent of cluster mode: a user can have this on a single-host devai
  install with no plans to ever run multiple workers.

## Out of scope

- System-managed clusters and routing. That is the sibling plan
  [skypilot-fleet-provisioner](./skypilot-fleet-provisioner.md).
- Pre-funded shared cloud accounts. Each user uses their own credentials.
- Pricing / budget UI in JupyterLab.
- Integration with Open WebUI (Open WebUI does not implement Anthropic's
  Agent Skills format; users would need an MCP-shaped surface instead --
  out of scope here).
- Custom devai-specific skills layered on top of the SkyPilot skill (e.g.,
  "spin up a worker registered with the head node"). Future work.

## Confirmed decisions

Confirmed 2026-05-14 before implementation. Future deviations
require an explicit plan amendment.

1. **Pre-install SkyPilot via the `fetch-cli` + COPY mechanism,
   same pattern as the other agent CLIs.** SkyPilot is a Python
   package (not a single binary like claude/codex/uv), so the
   pattern adapts as follows:
   - Extend `Makefile:fetch-cli` (around line 149) with a new
     SkyPilot block that runs `uv pip download skypilot[<extras>]
     --python-version 3.13 -d $(CACHE_DIR)/pip/wheels/skypilot`
     against `$(ETAG_DIR)/skypilot.version` for change detection
     (compare upstream PyPI metadata's `version` field to the
     last-cached value, same approach the Gemini CLI block already
     uses).
   - Pre-fetched wheel directory becomes one more cache artifact
     mounted into the image build (next to
     `/var/cache/bin/{claude,codex,ollama,uv,uvx,late,gemini,code-server}`).
   - `Dockerfile.lab` adds an offline-install step:
     `RUN uv pip install --system --offline
        --find-links /var/cache/wheels/skypilot 'skypilot[<extras>]'`
     -- mirrors the existing `cp /var/cache/bin/... /usr/local/bin/`
     pattern for binaries, but tailored to Python packages so the
     `sky` entry point and all cloud-extras land in `/usr/local/bin/`
     and `/usr/local/lib/python3.13/site-packages/`.
   - No network at image-build time; consistent with the rest of
     fetch-cli's offline-build guarantee. Image grows ~150-200 MB
     (mostly the broad cloud-extras set from decision 2).
2. **Cloud extras: broad set** --
   `skypilot[aws,gcp,azure,kubernetes,slurm,runpod,lambda]`.
   Binaries per extra are small; unused extras have zero runtime
   cost. Lets a user point at any cloud without rebuilding the
   image.
3. **Credentials: rely on `$HOME` mount.** devai already mounts
   the user's `$HOME` into the lab; `~/.aws/`, `~/.config/gcloud/`,
   `~/.config/sky/` flow through automatically. Document the
   expected files in `docs/skypilot.md`. No new credential surface.
4. **Plugin install: verify per-agent in pre-flight; ship
   per-agent docs if mechanisms differ.** Test the install on
   Claude Code, Codex, GitHub Copilot, and Cursor during
   pre-flight (~15 min added). Document per-agent install
   commands in `docs/skypilot.md` -- operator-facing reality, not
   aspirational uniformity.
5. **Version pinning: track latest.** The fetch-cli block (see
   decision 1) checks the upstream PyPI metadata each
   `make fetch-cli` invocation and re-downloads the wheel set
   when the version changes -- same ETag/version-stamp pattern
   the existing CLIs use (`$(ETAG_DIR)/skypilot.version`).
   `Dockerfile.lab` always installs whatever's currently in the
   cache. No version literal in the Dockerfile. Always-current
   wheels; accept the small risk that an upstream breaking
   change slips into the next lab rebuild. (Deviates from the
   original "recommend pin" -- chosen to keep the lab tracking
   SkyPilot's fast-moving CLI surface without manual quarterly
   bumps.)
6. **Pre-flight verification: yes, before implementation.**
   Install in scratch container, install plugin under Claude
   Code, ask Claude to `sky check`, watch output. Repeat under
   Codex / Copilot / Cursor to settle decision 4. ~15-30 min wall
   time.

## Context

In March 2026 SkyPilot shipped **Agent Skills**: a markdown-instruction-pack
plugin that teaches AI coding agents how to use the `sky` CLI for GPU
provisioning. Unlike MCP servers, Agent Skills are static instruction packs
loaded on demand (around 30-50 tokens until invoked), with no server
process. The skill format is Anthropic's open Agent Skills standard,
supported across Claude Code, Codex, GitHub Copilot, and Cursor.

devai already ships these CLI agents pre-installed in the lab container.
Adding the SkyPilot skill is a small Dockerfile change plus documentation.
The user benefits immediately: a single-host devai install becomes a
launchpad for cloud GPU jobs via natural-language conversation with the
agent already running in the lab.

The architectural separation from the sibling plan is clean:

- This plan: user asks Claude Code, Claude Code shells out to `sky` CLI,
  cloud VMs come up under the user's account, results return to the user.
  devai's gpu-arbiter is not involved at all.
- Sibling plan: gpu-arbiter (in head mode) calls SkyPilot API server,
  cloud VMs come up under a shared devai service account, workers
  register with head, requests get routed there. User is not involved.

## Approach

Two additions, mirroring the existing CLI fetch-and-bundle
pattern (`Makefile:fetch-cli` + `Dockerfile.lab` COPY/install):

1. Extend `Makefile:fetch-cli` to pre-fetch SkyPilot wheels (broad
   cloud-extras set) into `$(CACHE_DIR)/pip/wheels/skypilot/`,
   ETag/version-stamped at `$(ETAG_DIR)/skypilot.version`.
   `Dockerfile.lab` runs an offline `uv pip install --find-links`
   against that cache so `sky` lands in `/usr/local/bin/` with no
   network calls at image-build time. Same pattern as the existing
   binaries (claude / codex / ollama / uv / late / gemini /
   code-server), adapted for a Python package.
2. Install the SkyPilot Agent Skill plugin for each supported
   agent during image build, so the plugin is present out of the
   box.

Plus documentation explaining credential setup and a worked example.

---

## Phase 1 -- Install SkyPilot in the lab image

### Goal

`sky` CLI is available inside the lab container, `sky check` reports
enabled clouds when credentials are present.

### Deliverables

```
Makefile                            (modify: extend the `fetch-cli`
                                      target with a SkyPilot wheel-
                                      download block, ETag/version-
                                      stamped at $(ETAG_DIR)/
                                      skypilot.version)
deploy/
  Dockerfile.lab                    (modify: offline `uv pip install`
                                      from the mounted wheel cache;
                                      mirror the existing `cp /var/cache/
                                      bin/...` block for binaries)
docs/
  skypilot-user-guide.md            (new: credential setup, first launch,
                                      cost guidance)
scripts/
  sky-setup.sh                      (new: first-launch helper -- prints
                                      detected credentials, runs
                                      `sky check`, shows next step)
```

### Detailed steps

1. **Extend `Makefile:fetch-cli`** (around line 149) with a
   SkyPilot block following the existing pattern:
   ```makefile
   @LATEST=$$(curl -fsSL "https://pypi.org/pypi/skypilot/json" \
                | python3 -c "import sys,json; print(json.load(sys.stdin)['info']['version'])") \
       && CACHED=$$(cat $(ETAG_DIR)/skypilot.version 2>/dev/null || echo "none") \
       && if [ "$$LATEST" = "$$CACHED" ]; then echo "SkyPilot: up to date ($$CACHED)"; else \
           echo "Fetching SkyPilot $$LATEST wheels..." \
           && rm -rf $(CACHE_DIR)/pip/wheels/skypilot \
           && mkdir -p $(CACHE_DIR)/pip/wheels/skypilot \
           && uv pip download \
                  'skypilot[aws,gcp,azure,kubernetes,slurm,runpod,lambda]' \
                  --python-version 3.13 \
                  --dest $(CACHE_DIR)/pip/wheels/skypilot \
           && echo "$$LATEST" > $(ETAG_DIR)/skypilot.version \
           && echo "SkyPilot: updated to $$LATEST"; fi
   ```
   Output: `$(CACHE_DIR)/pip/wheels/skypilot/*.whl` (and any sdist
   fallbacks), plus `$(ETAG_DIR)/skypilot.version` for change
   detection. Same Gemini-CLI version-stamp pattern, adapted to
   PyPI's metadata JSON.

2. **Add offline-install step to Dockerfile.lab** (next to the
   existing `cp /var/cache/bin/{uv,uvx,claude,codex,ollama,late}
   /usr/local/bin/` line around `Dockerfile.lab:47`):
   ```dockerfile
   RUN uv pip install --system --offline \
         --find-links /var/cache/wheels/skypilot \
         'skypilot[aws,gcp,azure,kubernetes,slurm,runpod,lambda]'
   ```
   The `sky` entry point lands in `/usr/local/bin/`; cloud-extra
   packages land in `/usr/local/lib/python3.13/site-packages/`.
   The cache mount path mirrors how `/var/cache/bin/` is exposed
   to the build today.

3. **Smoke test during image build**: run `sky --version` and
   `sky check --help` as a build-time sanity check. Fail the
   build if either errors out (the wheel set is incomplete or
   the offline install missed a transitive dep).

4. **Credential discovery**: confirm that `$HOME/.aws`,
   `$HOME/.config/gcloud`, `$HOME/.config/sky` are mounted into
   the lab container under the user's UID. (Already devai's
   pattern for git/ssh.)

5. **First-launch helper**: `scripts/sky-setup.sh` (run by user
   inside the lab once) prints what credentials are detected,
   runs `sky check`, and shows the user the next step.

### Exit criteria

- `podman exec devai-lab-gpu sky --version` returns a version string.
- `podman exec devai-lab-gpu sky check` enumerates enabled clouds based
  on the user's mounted credentials.
- Image size increase under 250 MiB.

### Phase 1 risks

| Risk                                                | Mitigation                                            |
| --------------------------------------------------- | ----------------------------------------------------- |
| pip extras pull large cloud SDKs                    | Measure before/after; trim if image grows excessively |
| Credential paths differ on macOS / Windows hosts    | Test on Linux first; document host-OS expectations    |
| Some extras break on newer Python (3.13)            | Pin Python to 3.11 in lab base if needed              |

---

## Phase 2 -- Install the SkyPilot Agent Skill plugin

### Goal

Claude Code in the lab knows about SkyPilot tools out of the box. User
can say "spin up a 3090 on RunPod, run my script" and Claude does it.

### Deliverables

```
deploy/
  Dockerfile.lab                    (modify: install SkyPilot plugins for
                                      Claude Code / Codex / GitHub Copilot
                                      / Cursor as supported)
docs/
  skypilot-user-guide.md            (extend: agent usage section, examples)
```

### Detailed steps

1. **Verify plugin install mechanism** -- SkyPilot describes it as
   `claude plugin install skypilot-org/skypilot` or similar. Pin version
   and bake into image build. If different agents have different install
   commands, run each at image build.
2. **Install for each supported CLI agent that devai already ships**:
   Claude Code (definitely), Codex (verify), GitHub Copilot CLI
   (if devai ships it), Cursor (only if devai ships it; otherwise skip).
3. **Worked example in docs** -- a short script the user can read in
   `docs/skypilot-user-guide.md`:
   - "Claude, what GPUs are available right now across my clouds?"
   - "Claude, spin up a single 3090 on the cheapest cloud, run
     `train.py` from this directory, copy results back, then shut it
     down."
4. **Cost guidance** -- a clear paragraph on how easy it is to spend $20
   accidentally, pointers to `sky cost-report`, and instructions for
   setting per-user budgets.

### Exit criteria

- After `make build-gpu`, launching the lab and starting Claude Code, the
  SkyPilot tools are available without any user setup beyond credentials.
- A scripted scenario (lab + recorded transcript) completes a real
  spin-up / hello-world / shut-down cycle.
- Cost guidance is in the user guide and linked from the lab welcome
  banner.

### Phase 2 risks

| Risk                                                | Mitigation                                                     |
| --------------------------------------------------- | -------------------------------------------------------------- |
| Plugin format evolves; install command changes      | Pin version; tracking issue for upgrade                        |
| Per-agent install commands differ                   | Per-agent install in Dockerfile build, with comments           |
| User runs up large cloud bills inadvertently        | Cost guidance in docs; `sky cost-report` examples              |
| Plugin works for Claude Code but not for Codex etc. | Document support matrix; do not promise unsupported agents     |

---

## Combined risk register

| Risk                                              | Phase | Mitigation                                                |
| ------------------------------------------------- | ----- | --------------------------------------------------------- |
| Image bloat from cloud SDKs                       | 1     | Measure; trim extras if needed                            |
| Credential paths differ on non-Linux hosts        | 1     | Document Linux-host assumption                            |
| Plugin install command changes between versions   | 2     | Pin version; tracking issue                               |
| Cost runaway by user error                        | 2     | Documented cost guidance; sky cost-report examples        |
| Agent Skills format itself evolves                | 1-2   | Pin SkyPilot version; revisit on upgrades                 |

## Migration / rollback story

- **Phase 1 rollback**: remove the SkyPilot install lines from
  `Dockerfile.lab`; rebuild.
- **Phase 2 rollback**: remove the plugin install line from
  `Dockerfile.lab`; rebuild. Phase 1 install can stay.
- **Upgrade path**: existing lab installs gain SkyPilot on the next image
  rebuild. No state migration. Users without cloud credentials see the
  CLI present but `sky check` reports zero enabled clouds -- harmless.

## Estimated effort

| Phase    | Engineering effort                          | Wall-clock |
| -------- | ------------------------------------------- | ---------- |
| Phase 1  | ~1 PR, Dockerfile + fetch-cli + docs        | 1 day      |
| Phase 2  | ~1 PR, plugin install + worked example      | 1-2 days   |
| Total    | 1-2 PRs                                     | 2-3 days   |

## References

- SkyPilot Agent Skill announcement: blog.skypilot.co/agent-skill/
- SkyPilot v0.12 release: github.com/skypilot-org/skypilot/releases
- Anthropic Agent Skills standard: modelcontextprotocol.io documentation
  (Skills section)
- Comparison "MCP vs. Agent Skills": Analytics Vidhya, April 2026 --
  Skills are markdown packs loaded on demand, MCP is a client-server
  protocol. SkyPilot chose Skills.
- Related plans:
  - Sibling: [skypilot-fleet-provisioner](./skypilot-fleet-provisioner.md)
    (system-side SkyPilot integration; independent of this plan)
  - User-facing tool surface: [mcp-gateway](./mcp-gateway.md) (different
    mechanism for a different need)
