# MCP Gateway Integration Plan

## Status

Design **approved 2026-05-14** -- all six open questions resolved
(see "Confirmed decisions" below). Not yet scheduled for execution
pending the prerequisite plans listed under Dependencies.

**Amended 2026-05-14**: Phase 2's shared sops/age scaffold
(fetch-cli additions, tmpfs mount, render script, `.sops.yaml`,
age key custody helper, `docs/secrets.md`) extracted into the
new [sops-age-secrets](./sops-age-secrets.md) plan. This plan's
Phase 2 now consumes that scaffold and ships only the
MCP-specific surface on top of it. Dependencies section updated
accordingly.

## Dependencies

- [Plan: sops-age-secrets](./sops-age-secrets.md) -- Phase 2 reuses the
  shared sops/age scaffold for the four Tier 2 server secrets
  (`GITHUB_TOKEN`, `FIRECRAWL_API_KEY`, `HF_TOKEN`, `CONTEXT7_API_KEY`).
  Phase 1 has no dependency on it.

## Enables / Unblocks

- Agent tool access (filesystem, git, sqlite, web search, arXiv) across every
  CLI and Open WebUI session running in the lab container.
- A single secret store + audit trail for any future credential-bearing
  integration (HuggingFace, GitHub, Firecrawl, etc.).
- Foundation for future multi-agent orchestration work -- agents on the head
  node get a single tool surface to plug into.

## Out of scope

- Multi-tenancy (per-user secret scoping). devai is single-user-per-host today.
- Per-server TLS termination behind nginx-proxy. Add later if needed.
- Custom (non-catalog) MCP servers. Ship only community-vetted servers in v1.
- Hooking MCP server selection into the model picker / lab launcher UI.
- Distributed secret distribution for cluster mode. That belongs in the
  SkyPilot plan, not here.

## Confirmed decisions

Confirmed 2026-05-14 before implementation. Future deviations
require an explicit plan amendment.

1. **Server shortlist: ship all 10 Tier 1 servers as-is** --
   filesystem, git, sqlite-mcp-server, fetch, memory, time,
   sequentialthinking, duckduckgo, arxiv-mcp-server, wikipedia-mcp.
   All map to ML/AI developer workflows; none require secrets;
   combined image footprint is small.
2. **Port: 8088** (override of the proposed 11437). Moves MCP
   off the 1143X model-backend cluster to a generic HTTP-style
   port. Operator can still override via `MCP_PORT` env per the
   compose snippet. The 1143X sequence stays reserved for
   inference backends only (Ollama / vLLM / SGLang) and any future
   model-backend additions.
3. **Phase split: two PRs.** Phase 1 (no secrets, 10 Tier 1
   servers) ships first; small, no secret-mgmt code path. Phase 2
   (sops/age + 4 Tier 2 servers) lands once Phase 1 is shaken out.
   Each PR is independently testable and revertable.
4. **Age key custody: user-scoped `~/.config/sops/age/keys.txt`.**
   Single-user lab today; user owns the key; gateway reads via
   bind-mount; no root required. Escalate to host-wide when
   cluster mode lands.
5. **Compose profile: NO opt-in gate -- start by default.**
   `make cache-up` brings the gateway up alongside the existing
   stack. Couples baseline cluster up-time to the new component,
   but keeps the install/usage path uniform. (Note: deviates
   from the original "recommend yes -- gate" -- the lab wants the
   tool surface available without a separate opt-in step.)
6. **Pre-flight verification: yes, before Phase 1 starts.**
   `podman pull docker/mcp-gateway:v0.42.1` + minimal compose
   invocation on a clean Debian Trixie host; confirms Podman
   socket compatibility. ~10 minutes wall time.

## Context

The Docker MCP Gateway acts as a centralised proxy between MCP clients
(Claude Code, Gemini CLI, Codex CLI, Open WebUI) and the actual MCP server
processes (filesystem, github, sqlite, ...). Each agent's MCP config
collapses from N-server entries to one endpoint pointing at the gateway.
The gateway handles:

- A single curated server catalog (one YAML, not N agent configs).
- A single secret store (credentials never reach the agents).
- Container isolation per spawned server (cap-drop, no-new-privs, scoped fs).
- Transport translation between stdio and HTTP/SSE.
- One audit chokepoint for all tool calls.

devai gains the gateway as a peer service to `devai-router`. The gateway
runs on its own port, spawns per-call MCP server containers via the Podman
socket, and is opt-in via a compose profile until stable.

The gateway image (`docker/mcp-gateway`) is roughly 38 MiB compressed,
Apache-licensed, actively maintained, and runs without Docker Desktop.

## Server shortlist

Categorised by what devai's user (an ML/AI developer running CLIs and Open
WebUI in a lab container) actually needs.

### Tier 1 -- no secrets, ship in Phase 1

| Server                | What it does                          | Why for devai                                              |
| --------------------- | ------------------------------------- | ---------------------------------------------------------- |
| `filesystem`          | Read/write files                      | Agents touch the user's workspace                          |
| `git`                 | Local git ops (status, log, diff)     | Agents introspect repos                                    |
| `sqlite-mcp-server`   | Query SQLite databases                | Agents can query devai probe caches and bench results      |
| `fetch`               | HTTP GET URLs                         | Generic web fetch                                          |
| `memory`              | In-process scratch storage            | Working memory across tool calls                           |
| `time`                | Current date/time                     | Eliminates "what is today" hallucinations                  |
| `sequentialthinking`  | Chain-of-thought scaffolding          | Improves reasoning on complex tasks                        |
| `duckduckgo`          | Web search (no API key)               | Free baseline search                                       |
| `arxiv-mcp-server`    | Search/fetch arXiv papers             | Directly relevant to ML researchers                        |
| `wikipedia-mcp`       | Wikipedia lookups                     | Cheap factual grounding                                    |

### Tier 2 -- one secret each, ship in Phase 2

| Server            | Secret           | Value                                                           |
| ----------------- | ---------------- | --------------------------------------------------------------- |
| `github-official` | GitHub PAT       | Agents work with GitHub repos (issues, PRs, code search)        |
| `firecrawl`       | Firecrawl key    | Higher-quality web scraping than `fetch`                        |
| `hugging-face`    | HF token         | List / inspect / download models -- fits devai model management |
| `context7`        | API key (free)   | Up-to-date library docs (Anthropic-blessed pattern)             |

### Tier 3 -- opt-in, document but do not enable by default

`kubernetes`, `docker`, `dockerhub`, `playwright`, `mcp-code-interpreter`
(Python sandbox), `postgresql`, `redis`, `clickhouse`, `chroma`, `pinecone`.

### Explicitly NOT shipping

Productivity SaaS connectors (Notion / Slack / Linear / Asana), payments
(Stripe / PayPal / Razorpay), customer support tools, cloud-account
connectors (AWS / Azure / GCP IAM / billing). devai is a local lab, not a
corporate workflow surface.

---

## Phase 1 -- MCP Gateway, no secrets

Goal: `devai-mcp-gateway` container running alongside the existing stack,
exposing Tier 1 servers on port 8088. Zero touch to gpu-arbiter, router,
probes, or backends.

### Deliverables

```
deploy/
  docker-compose.yaml             (modify: + devai-mcp-gateway service)
  mcp-servers.yaml                (new: catalog with Tier 1 entries)
  mcp-gateway.env                 (new: non-secret config, PORT=8088)
docs/
  mcp.md                          (new: usage, security model, client configs)
scripts/
  mcp-health.sh                   (new: smoke test for the gateway)
Makefile                          (new targets: mcp-up, mcp-down, mcp-logs,
                                   mcp-test)
README.md                         (update: link to docs/mcp.md)
CLAUDE.md                         (update: add MCP gateway to architecture
                                   summary)
tests/
  test-mcp.sh                     (new: end-to-end test from a client)
```

### Detailed steps

1. **Add compose service** (`deploy/docker-compose.yaml`):

   ```yaml
   devai-mcp-gateway:
     image: docker/mcp-gateway:v0.42.1   # pinned, not :latest
     container_name: devai-mcp-gateway
     restart: unless-stopped
     networks: [devai-net]
     ports: ["${MCP_PORT:-8088}:8088"]
     profiles: [mcp]
     volumes:
       - ${XDG_RUNTIME_DIR}/podman/podman.sock:/var/run/docker.sock:Z
       - ./mcp-servers.yaml:/app/catalog.yaml:ro
       - mcp-state:/data
     command:
       - "--catalog=/app/catalog.yaml"
       - "--port=8088"
       - "--transport=streaming"
       - "--block-secrets"
   ```

2. **Curate `deploy/mcp-servers.yaml`** with the 10 Tier 1 servers. Use the
   Docker MCP catalog naming so the gateway pulls images directly from the
   `mcp/*` namespace.

3. **Podman socket**: devai uses Podman. The gateway expects a
   Docker-compatible socket. Verify
   `podman system service --time=0 unix://$XDG_RUNTIME_DIR/podman/podman.sock`
   is running, or add a systemd user unit to ensure it.

4. **Test from inside the lab container**:

   ```bash
   curl http://devai-mcp-gateway:8088/health
   curl -X POST http://devai-mcp-gateway:8088/tools/fetch \
        -d '{"url": "https://example.com"}'
   ```

5. **Document client configs** (`docs/mcp.md`):

   - Claude Code (`~/.config/claude/mcp.json`) -- one entry pointing at the
     gateway.
   - Gemini CLI -- equivalent.
   - Codex CLI -- equivalent.
   - Open WebUI -- tools UI configuration.

6. **Update Makefile**:

   ```make
   mcp-up:    ; podman-compose --profile mcp up -d devai-mcp-gateway
   mcp-down:  ; podman-compose stop devai-mcp-gateway && \
                podman-compose rm -f devai-mcp-gateway
   mcp-logs:  ; podman logs -f devai-mcp-gateway
   mcp-test:  ; bash tests/test-mcp.sh
   ```

   Use a compose profile so the gateway is opt-in for the first release.

7. **Smoke test (`tests/test-mcp.sh`)**: hit `/health`, list tools, call
   `fetch` against `example.com`, call `time`. Roughly 30 lines of bash.

### Phase 1 exit criteria

- `make cache-up && make mcp-up` brings the gateway live.
- Lab container can call all 10 Tier 1 tools.
- Claude Code in the lab container, configured to point at the gateway,
  can use `filesystem` to edit `~/workspace`.
- No changes to `gpu-arbiter/main.go`, no changes to the router, no changes
  to backend lifecycle, no changes to probe caches.

### Phase 1 risks

- Podman socket compatibility with the gateway's expected Docker socket.
  Mitigation: pre-flight smoke test on a clean Debian Trixie host; fallback
  to mount `/var/run/docker.sock` if a Docker daemon is present.
- Gateway-spawned MCP server containers running as root inside the per-call
  container. Mitigation: set `--security-opt=no-new-privileges` in the
  gateway's container spec; the gateway respects this for spawned containers.
- Server catalog format may change between gateway versions. Mitigation:
  pin to `v0.42.1`, document upgrade procedure in `docs/mcp.md`.

---

## Phase 2 -- Tier 2 secret-requiring servers

Goal: with the shared sops/age scaffold from
[sops-age-secrets](./sops-age-secrets.md) already in place,
encrypt the four Tier 2 server secrets, render them to tmpfs at
`cache-up` time, mount the rendered env file into the gateway,
flip on Tier 2 entries in `mcp-servers.yaml`. No new
secret-store infrastructure built here -- only the
MCP-specific surface on top of it.

### Rationale for sops + age

The full rationale (alternatives considered, why sops/age wins,
key custody model) lives in
[sops-age-secrets.md](./sops-age-secrets.md) and is not
duplicated here. Short version: encrypted at rest, git-friendly
per-value diffs, single tool, no service, no Docker Desktop.

`docker/mcp-gateway` only natively understands two secret
backends: `docker-desktop` (keychain, not usable for us) and a
`.env` file path. The shared scaffold's
`scripts/render-secret.sh` is what produces that `.env` file
securely, with no plaintext at rest on disk.

### Deliverables

```
deploy/
  mcp-secrets.sops.env            (new: encrypted, committed --
                                   the four Tier 2 secrets:
                                   GITHUB_TOKEN, FIRECRAWL_API_KEY,
                                   HF_TOKEN, CONTEXT7_API_KEY)
  mcp-servers.yaml                (modify: add Tier 2 entries with
                                   {secret: NAME} references)
  docker-compose.yaml             (modify: gateway gets
                                   /run/devai/mcp-secrets.env mount
                                   and --secrets flag)
docs/
  mcp.md                          (extend: secrets section,
                                   link to docs/secrets.md for
                                   the sops/age scaffold details)
Makefile                          (new target: mcp-secrets-render --
                                   thin wrapper around the shared
                                   scripts/render-secret.sh; cache-up
                                   depends on it)
.sops.yaml                        (modify: append the
                                   deploy/mcp-secrets.sops.env
                                   creation_rules entry)
```

Out of this plan's scope (owned by
[sops-age-secrets](./sops-age-secrets.md)):
`scripts/age-keygen-host.sh`, `scripts/render-secret.sh`,
`scripts/fetch-cli.sh` sops/age block, the initial `.sops.yaml`
file, `deploy/setup-secrets-tmpfs.sh`, the `/run/devai` tmpfs
mount, `.gitignore` entries for `/run/devai` / `*.env.plain`,
generic `secrets-edit` / `secrets-rotate` / `secrets-tmpfs`
Makefile targets, `docs/secrets.md`. All of those are
prerequisites this plan depends on, not deliverables it ships.

### Detailed steps

1. **Verify the shared scaffold is installed**: the operator
   has run `scripts/age-keygen-host.sh`, added their public key
   to `.sops.yaml`, and `/run/devai` is mounted (per
   sops-age-secrets exit criteria). This plan does not
   re-install any of that.

2. **Append the per-file rule to `.sops.yaml`** (the scaffold's
   placeholder regex `deploy/.*\.sops\.env$` already matches,
   so this step is only needed if the operator has chosen a
   tighter per-file rule).

3. **Initial secrets encryption** (one-time, by the user):

   ```bash
   umask 077
   cat > /tmp/mcp.env <<EOF
   GITHUB_TOKEN=ghp_xxx
   FIRECRAWL_API_KEY=fc-xxx
   HF_TOKEN=hf_xxx
   CONTEXT7_API_KEY=ctx_xxx
   EOF
   sops --encrypt /tmp/mcp.env > deploy/mcp-secrets.sops.env
   shred -u /tmp/mcp.env
   git add deploy/mcp-secrets.sops.env
   ```

4. **Wire into compose**:

   ```yaml
   devai-mcp-gateway:
     # ... existing keys ...
     volumes:
       - /run/devai/mcp-secrets.env:/secrets/.env:ro
     command:
       - "--catalog=/app/catalog.yaml"
       - "--secrets=/secrets/.env"
       - "--port=8088"
       - "--block-secrets"
   ```

5. **Update `mcp-servers.yaml`** to add Tier 2 entries with
   `{secret: GITHUB_TOKEN}` style references the gateway
   resolves from the loaded env at tool-call time.

6. **Makefile target**:

   ```make
   mcp-secrets-render: secrets-tmpfs
       bash scripts/render-secret.sh \
            deploy/mcp-secrets.sops.env \
            /run/devai/mcp-secrets.env

   cache-up: mcp-secrets-render
       $(MAKE) -f Makefile cache-up-real
   ```

   `secrets-tmpfs` and `scripts/render-secret.sh` come from the
   shared scaffold. This plan adds only the wrapper that knows
   the MCP-specific input/output pair.

7. **MCP-side docs**: `docs/mcp.md` gets a short "Secrets"
   section that lists the four Tier 2 secret names, links to
   `docs/secrets.md` for the underlying sops/age workflow, and
   shows how to add a new Tier 2 server by appending a key to
   `deploy/mcp-secrets.sops.env` and a `{secret: NAME}` entry
   in `deploy/mcp-servers.yaml`. Rotation, recovery, multi-host
   onboarding all live in the shared `docs/secrets.md`; not
   re-documented here.

### Phase 2 exit criteria

- `sops/age` scaffold from
  [sops-age-secrets](./sops-age-secrets.md) is installed and
  verified (this plan's prerequisite, not its exit criterion).
- `make cache-up` renders `deploy/mcp-secrets.sops.env` to
  `/run/devai/mcp-secrets.env` and brings the gateway up with
  Tier 2 servers active.
- `curl` against the gateway's `github` tool successfully
  lists repos.
- Plaintext secrets never written to non-tmpfs disk anywhere
  in the flow.
- `deploy/mcp-servers.yaml` references each of the four
  Tier 2 secrets via `{secret: NAME}` and the gateway resolves
  them at tool-call time.

### Phase 2 risks

- New gateway version breaks `--secrets` semantics or
  `{secret: NAME}` reference format. Mitigation: pin gateway
  version; integration test (`tests/test-mcp.sh`) covers
  Tier 2 round-trip before any version bump.
- Tier 2 server image tags drift independently of the gateway
  image. Mitigation: `mcp-servers.yaml` pins each Tier 2
  server's image tag; bump deliberately.
- `deploy/mcp-secrets.sops.env` shipped with stale secrets
  after a rotation in another consumer plan's secrets file.
  Mitigation: documented one-line rotate command in
  `docs/secrets.md`; no per-consumer rotation choreography.

---

## Combined risk register

| Risk                                                    | Phase | Mitigation                                                  |
| ------------------------------------------------------- | ----- | ----------------------------------------------------------- |
| Podman socket vs Docker socket compatibility            | 1     | Pre-flight smoke test; fallback path documented             |
| Gateway-spawned containers running as root              | 1     | `--security-opt=no-new-privileges` on the gateway container |
| Catalog schema drift on gateway version bumps           | 1     | Pin to v0.42.1; document upgrade procedure                  |
| New gateway version breaks `--secrets` semantics        | 2     | Pin version; integration test before bump                   |
| Tier 2 server image tags drift independently of gateway | 2     | Pin per-server image tags in mcp-servers.yaml               |

Secret-store risks (plaintext leakage on render, age key
custody, lost-key recovery) are tracked in
[sops-age-secrets](./sops-age-secrets.md) -- this plan
inherits its mitigations.

## Migration / rollback story

- **Phase 1 rollback**: comment out the `devai-mcp-gateway` service in
  compose; `podman stop devai-mcp-gateway`; no other surface is affected.
- **Phase 2 rollback**: revert to Phase 1 (no Tier 2 servers, no secrets
  mount). Encrypted secrets file remains in git; just unused.
- **Upgrade path**: existing devai installs gain MCP by running
  `make mcp-up`. Without that, nothing changes from the user's perspective.

## Estimated effort

| Phase    | Engineering effort                  | Wall-clock                |
| -------- | ----------------------------------- | ------------------------- |
| Phase 1  | ~1 PR, 200-400 lines of YAML/bash   | 1-2 days incl. testing    |
| Phase 2  | ~1 PR, 200-400 lines + doc          | 2-3 days incl. testing    |
| Total    | 2 PRs                               | ~1 week elapsed           |

---

## References

- Docker MCP Gateway image: `docker/mcp-gateway` on Docker Hub
- Docker MCP Gateway source: github.com/docker/mcp-gateway
- Docker MCP Registry (329 servers): github.com/docker/mcp-registry
- sops: github.com/getsops/sops (CNCF sandbox)
- age: github.com/FiloSottile/age
- Book reference: "Operational AI with Docker" -- Chapter 6 (MCP) and
  Chapter 9 (Docker Sandboxes) provide the conceptual framework. See
  `docs/graphify-out/GRAPH_REPORT.md` for the community map.
