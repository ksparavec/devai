# Pi coding agent integration

_Bundle the pi coding agent (earendil-works/pi) into the lab image and wire it to the router as a first-class picker agent, alongside Claude Code / Codex / Aider / LATE / Open Interpreter._

## Status

Draft. Not yet scheduled for execution.

## Dependencies

- None. This is a lab-image + picker change, the same shape as the
  [skypilot-agent-skill](./skypilot-agent-skill.md) lab-image addition
  (bundle a CLI tool, wire it via the picker). It does not touch the Go
  router.

## Enables / Unblocks

- A token-efficient, minimal-prompt harness suited to the *smaller* local
  models devai serves. Pi ships a deliberately small system prompt and
  four tools (read / write / edit / bash); an 8B-class local model follows
  it more reliably and wastes fewer tokens than under a heavy frontier
  harness.
- AGENTS.md + skills synergy: this repo already uses AGENTS.md, and pi
  loads AGENTS.md and `/skill:name` skills natively.
- Pi's `models.json` can declare all three `router-<backend>` providers,
  so pi's in-session `/model` switcher demuxes across devai backends
  (subject to the GPU-mutex cold-start cost) -- making pi a natural client
  for the [router-fanout](./router-fanout.md) feature.
- Scriptable `-p` (print), `--mode json`, and `--mode rpc` surfaces for
  automation and possible bench-harness use.

## Out of scope

- Seeding pi's *cloud* providers (Anthropic / OpenAI / Google direct) or
  OAuth / subscription login. devai is local-model-only; we seed
  `apiKey: "local"` against the router. A user may add their own keys to
  `~/.pi/agent/models.json`, but provisioning that is not devai's job.
- Embedding pi's SDK / RPC mode into JupyterLab as a kernel. We add a
  launcher card that runs the picker, nothing deeper.
- Pi extensions / custom-provider plugins. `openai-completions` against
  the router covers all three backends; no extension is needed.
- Any router code change. Pi is an OpenAI-compatible client; the router
  already serves it.

## Open questions

1. Does pi accept `--model <id>` for a *custom* provider when that model
   is NOT pre-declared in the provider's `models[]` array?
   Recommendation: **assume no** -- the picker injects the chosen model
   entry into `~/.pi/agent/models.json` before launch. Backstop: the
   router already rejects any non-vetted model on vLLM/SGLang with HTTP
   404 (`main.go:1969-1985`), so this is UX, not a security control.
2. Bundle mechanism -- **RESOLVED: prebuilt release tarball.** Use the
   GitHub-release `pi-<os>-<arch>.tar.gz` (e.g. `pi-linux-x64.tar.gz`;
   v0.75.5 ships them) -- a Bun-compiled self-contained executable plus a
   few asset sidecars (WASM / themes). This is the **codex release-tarball
   pattern** (no Node runtime, no node_modules), simpler than pi's npm
   path (`dist/cli.js` + 17 runtime deps). Open sub-item: confirm the
   binary finds its sidecar assets relative to itself after extraction.
3. API surface per backend. Recommendation: **`openai-completions`
   everywhere** -- the router serves OpenAI-compat on all three ports, so
   one api type keeps the template uniform. (Pi also supports
   `anthropic-messages`, but there is no reason to mix.)

## Context

devai ships five picker agents today (`scripts/model-picker.py:155`):
Claude Code, Aider, Codex, LATE, Open Interpreter. Each is wired to the
router in `_build` (`model-picker.py:2255`) by setting env vars and/or
CLI flags so the agent talks to `http://devai-router:<port>`.

pi (`github.com/earendil-works/pi`, MIT, TypeScript) is a minimal terminal
coding harness. Its fit for devai is specific: a small system prompt and a
four-tool surface suit the smaller, weaker local models devai serves
better than Claude Code's heavy agentic harness. It configures custom
OpenAI-compatible providers via `~/.pi/agent/models.json` with exactly the
fields devai already uses for Codex's `router-<backend>` providers:
`baseUrl`, `api`, `apiKey`, `models[]`. It selects a model at launch with
`pi --provider <name> --model <pattern>` and also supports AGENTS.md,
skills, and non-interactive print/JSON/RPC modes.

Crucially, devai's existing model-name suffix grammar
(`<name>::<reasoning>::<mtp>@<ctx>`) rides through pi unchanged: pi sends
whatever `--model` id it is given as the OpenAI request's `model` field,
and the router parses the suffixes. So no pi-specific router logic is
needed -- this is a pure client-bundling-and-wiring task, structurally
identical to how Codex was added.

## Model-access policy (general)

This rule applies to every devai agent, not just pi:

- **Local GPU (the router's ollama / vllm / sglang ports): picker-vetted
  models only, no downloading.** Already enforced for vLLM/SGLang --
  `makeRequestHandler` rejects any `model` not loaded from the probe
  cache with HTTP 404 (`main.go:1969-1985`). So pi's `router-vllm` /
  `router-sglang` providers can only ever serve vetted models, regardless
  of what a client lists. The picker seeds the local providers' `models[]`
  with the vetted/fitting set so pi's `/model` switcher offers exactly
  those.
- **Gap to close (cross-cutting, NOT pi-specific).** The Ollama port has
  no such allowlist (`main.go:1973` skips it) and the router proxies
  unmatched paths through its catch-all (`main.go:1320`), leaving
  `/api/pull` / `/api/create` / `/api/push` / `/api/copy` / `/api/delete`
  reachable -- a download/mutation vector that violates the no-download
  rule. Fully enforcing the policy needs a small router guard that 403s
  those Ollama endpoints. Tracked here as a policy dependency, to be
  handled in `docs/router.md` + a tiny router change, not in this plan.
- **Cloud / remote routers and endpoints: unrestricted.** Agents may use
  any model they support. devai does not seed cloud providers (see Out of
  scope) but does not block a user from adding their own keys to
  `~/.pi/agent/models.json`.

## Approach

Bundle the pi CLI into the lab image via the existing `fetch-cli`
pre-fetch plus `Dockerfile.lab` copy (mirroring gemini/codex). Ship a
`config/pi/models.json` template declaring three `router-<backend>`
`openai-completions` providers (`apiKey: "local"`), seeded to
`~/.pi/agent/models.json` by the container entrypoint (mirroring the
`config/codex/config.toml` -> `~/.codex/config.toml` seed). Add pi to the
picker's `_AGENTS` registry and a `_build` branch that injects the chosen
`(model, ctx, reasoning)` as a model entry and launches
`pi --provider router-<backend> --model <name>`. Add a JupyterLab launcher
card and docs. No router change.

---

## Phase 1 -- bundle pi + wire it to the router

### Goal

Pi is installed in the lab image, seeded with the three router providers,
selectable in the picker, and a chat routed through pi reaches the chosen
backend end-to-end.

### Deliverables

```
Makefile                        modify -- fetch-cli: pre-fetch pi (mirror the codex release-tarball block)
deploy/Dockerfile.lab           modify -- install pi (copy + symlink); COPY config/pi/models.json -> /etc/devai/pi-models.json
config/pi/models.json           new    -- three router-<backend> openai-completions providers (apiKey: local)
entrypoint.sh                   modify -- seed ~/.pi/agent/models.json from /etc/devai/pi-models.json if absent (sibling of the codex seed)
scripts/model-picker.py         modify -- add ("pi",...) to _AGENTS (line 155); add the pi branch to _build (near line 2305)
scripts/pi-launcher.sh          new    -- (if model injection is cleaner in shell) ensure models.json carries the chosen model id, then exec pi
```

### Detailed steps

1. `config/pi/models.json` template:

   ```json
   {
     "providers": {
       "router-ollama": {
         "baseUrl": "http://devai-router:11434/v1",
         "api": "openai-completions",
         "apiKey": "local",
         "models": []
       },
       "router-vllm": {
         "baseUrl": "http://devai-router:11435/v1",
         "api": "openai-completions",
         "apiKey": "local",
         "models": []
       },
       "router-sglang": {
         "baseUrl": "http://devai-router:11436/v1",
         "api": "openai-completions",
         "apiKey": "local",
         "models": []
       }
     }
   }
   ```

   The header comment mirrors `config/codex/config.toml`: shipped at
   `/etc/devai/pi-models.json`, copied to `~/.pi/agent/models.json` on
   first launch only; delete the home copy to pick up a newer image.
2. `fetch-cli`: add a pi block mirroring **codex** (Makefile ~170-178) --
   download `pi-linux-<arch>.tar.gz` from the latest GitHub release,
   ETag-gated, extract under `$(CACHE_DIR)/pip/bin/pi/`. No Node, no
   node_modules.
3. `Dockerfile.lab`: copy the extracted pi bundle to `/usr/local/lib/pi/`
   (preserving the binary plus its asset sidecars) and symlink
   `/usr/local/lib/pi/pi` -> `/usr/local/bin/pi` (same shape as the codex
   binary copy, lines 47-49), plus the `~/.local/bin/pi` symlink (mirror
   lines 135-140). Add `COPY config/pi/models.json /etc/devai/pi-models.json`
   next to the codex COPY (line 144).
4. `entrypoint.sh`: add a seed step beside the existing codex one -- if
   `~/.pi/agent/models.json` is absent, `mkdir -p ~/.pi/agent` and copy
   `/etc/devai/pi-models.json` into it (first-launch-only semantics, same
   as codex).
5. `_AGENTS` (line 155): add
   `("pi", "Pi", "Minimal, token-efficient terminal coding harness")`.
6. `_build` pi branch: resolve `router-<backend>` from the chosen
   backend, ensure the chosen model id (the full suffixed
   `<name>[::reasoning][::mtp][@ctx]` string the picker already builds) is
   present under that provider's `models[]` (inject if open question 1 is
   "no"), then return
   `["pi", "--provider", f"router-{backend}", "--model", name]`. No env
   overrides needed -- `apiKey: "local"` is in the seeded models.json.
7. Verify the model id passes through to the router unchanged so the
   existing `@<ctx>` / `::<reasoning>` parsing applies (no pi-specific
   router code).

### Exit criteria

- `model-picker --agent pi`, pick a model on each backend, and a chat
  reaches the chosen backend through the router end-to-end (verified
  manually on a host with a GPU stack; this environment has none, so mark
  deferred-to-hardware).
- With `fetch-cli` not run, the image build degrades gracefully (pi
  absent, other agents unaffected) -- same posture as the SkyPilot wheel
  block (Dockerfile.lab ~94-105).
- `~/.pi/agent/models.json` is seeded on first launch and not clobbered on
  later launches.

### Phase 1 risks

| Risk                                                       | Mitigation                                                        |
| ---------------------------------------------------------- | ----------------------------------------------------------------- |
| pi rejects `--model` for an undeclared custom-provider model | Picker injects the model entry before launch (open question 1)    |
| Bun-compiled binary needs sidecar assets adjacent          | Extract the whole release tarball to one dir and symlink only the binary; verify asset resolution at runtime |
| Mid-session `/model` switch triggers a router cold start   | Document the GPU-mutex cost in `docs/`; it is inherent to devai, not a pi bug |
| Pi version drift changes models.json schema                | Template is a plain file; pin the pi version in fetch-cli and bump deliberately |

---

## Phase 2 -- surfaces + ergonomics

### Goal

Make pi a first-class surface (JupyterLab card, docs) and confirm the
AGENTS.md / skills passthrough.

### Deliverables

```
packages/jupyter-ai-launchers/src/index.ts  modify -- add a pi launcher card (id 'pi', command 'model-picker --agent pi') + logo svg
CLAUDE.md                                    modify -- add pi to the agent list / picker section
docs/ (router.md or a short note)            modify -- pi wiring + the /model cold-start caveat
```

### Detailed steps

1. Add a pi card to the launcher `agents` array (index.ts ~48-52) with an
   svg logo and `command: 'model-picker --agent pi'`. Rebuild the
   extension in-container per the CLAUDE.md "Building the JupyterLab
   extension" recipe (never on the host).
2. Document pi in CLAUDE.md's agent surface and add a short usage note
   (provider/model selection, the AGENTS.md pickup, the cold-start caveat
   on `/model`).
3. Confirm pi reads the repo's AGENTS.md automatically when launched from
   the mounted working directory (it walks parent dirs + cwd). No code
   change expected; document the behaviour.
4. (Optional) Note pi's `-p` / `--mode json` for scripted use against the
   router, for anyone wanting a non-interactive local agent.

### Exit criteria

- The JupyterLab launcher shows a Pi card that opens the picker scoped to
  pi.
- CLAUDE.md lists pi among the agents.

### Phase 2 risks

| Risk                                          | Mitigation                                              |
| --------------------------------------------- | ------------------------------------------------------- |
| Extension rebuild drift                       | Follow the in-container build recipe; commit the built labextension output as the repo already does |

---

## Combined risk register

| Risk                                            | Phase | Mitigation                                                        |
| ----------------------------------------------- | ----- | ----------------------------------------------------------------- |
| Undeclared-model rejection by pi                | 1     | Picker injects the model entry (open question 1)                  |
| Bundle/install mechanism mismatch               | 1     | Mirror gemini npm pattern; confirm package layout (open question 2) |
| Network during build if fetch-cli skipped       | 1     | Graceful-skip posture mirroring the SkyPilot wheel block          |
| GPU-mutex cold start on in-session model switch | 1-2   | Documented; inherent to single-GPU devai                          |

## Migration / rollback story

- Purely additive: a new agent in the lab image and picker. Existing
  agents and the router are untouched. If `fetch-cli` does not pull pi,
  the image still builds and the other agents still work.
- Rollback = revert the PR. No persistent state beyond the user's
  `~/.pi/agent/` (which the entrypoint only seeds when absent).

## Estimated effort

| Phase   | Engineering effort                                  | Wall-clock   |
| ------- | --------------------------------------------------- | ------------ |
| Phase 1 | 1 PR: fetch-cli + Dockerfile.lab + template + entrypoint + picker | 1-2 days     |
| Phase 2 | 1 PR: launcher card + docs                          | 0.5-1 day    |
| Total   | 1-2 PRs                                             | ~2-3 days    |

## References

- Pi: `github.com/earendil-works/pi`, site `pi.dev`, npm
  `@earendil-works/pi-coding-agent` (MIT, TypeScript); prebuilt binaries
  at GitHub releases (`pi-<os>-<arch>.tar.gz`, Bun-compiled, v0.75.5).
  Config reference:
  the package README + `docs/models.md` (models.json schema:
  `providers.<name>.{baseUrl, api, apiKey, models[]}`; api values
  `openai-completions` / `openai-responses` / `anthropic-messages` /
  `google-generative-ai`; only `id` required per model; apiKey accepts a
  literal, env var, or `!command`).
- [Plan: skypilot-agent-skill](./skypilot-agent-skill.md) -- prior art for
  bundling a CLI into the lab image via fetch-cli.
- [Plan: router-fanout](./router-fanout.md) -- pi's multi-provider
  models.json is a natural client for it.
- devai integration points: `scripts/model-picker.py:155` (`_AGENTS`),
  `:2255` (`_build`, esp. the Codex `router-<backend>` branch ~2291),
  `deploy/Dockerfile.lab` (gemini npm copy ~56-57, codex config COPY 144,
  `~/.local/bin` symlinks ~135-140), `config/codex/config.toml` (seed
  template pattern), `Makefile` `fetch-cli` (~152, gemini block ~217-227),
  `packages/jupyter-ai-launchers/src/index.ts` (~48-52, launcher cards).
