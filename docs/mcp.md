# MCP Gateway

devai bundles the [Docker MCP Gateway](https://github.com/docker/mcp-gateway)
as a peer service to `devai-router`. The gateway acts as a single
HTTP endpoint that any MCP-aware agent (Claude Code, Gemini CLI,
Codex CLI, Open WebUI) can target. Per-call MCP server containers
are spawned by the gateway via the host's Podman socket; agents see
only the gateway, not the underlying server processes.

This is the operator reference. The architecture rationale and
phase split live in
[docs/plans/mcp-gateway.md](plans/mcp-gateway.md).

## Status snapshot

- **Phase 1 shipped**: 10 Tier 1 servers (no secrets), gateway
  reachable on port 8088 by default -- published on **127.0.0.1
  only**.
- **Phase 2 shipped**: 4 Tier 2 servers (`github-official`,
  `firecrawl`, `hugging-face`, `context7`) backed by the shared
  sops/age secret-store scaffold from
  [docs/secrets.md](secrets.md). Operators encrypt
  `deploy/mcp-secrets.sops.env`, run `make mcp-secrets-render`
  to populate `/run/devai/mcp-secrets.env`, then `make mcp-up`.
  When the secrets file is absent the gateway runs in Phase 1
  mode -- Tier 2 entries log "missing secret" and skip.

## Bring-up

```bash
make cache-up        # standard infra: router, ollama, vllm/sglang stubs
make mcp-up          # gateway in compose 'mcp' profile
make mcp-test        # smoke test: /health + tools/list
```

The gateway listens on `${MCP_PORT:-8088}` on the host -- bound to
`127.0.0.1` -- and on 8088 inside the container. Override `MCP_PORT`
in `.env` to relocate without touching `deploy/docker-compose.yaml`.

The loopback bind is deliberate: the gateway holds a read-write podman
socket (creating per-call server containers is its whole job), so
container-create through it is equivalent to host root, and it
requires no authentication. Reaching it from another machine means an
SSH tunnel or an authenticating reverse proxy -- not a wider bind. The
two documented client URLs are unaffected: `http://localhost:8088`
from the host, and `http://devai-mcp-gateway:8088` from inside
`devai-net`.

To stop:

```bash
make mcp-down
```

To watch logs:

```bash
make mcp-logs
```

## Tier 1 server catalog

Edit `deploy/mcp-servers.yaml` to add or remove servers. Each entry
pins an upstream image tag; bump deliberately and re-test after.

| Server                | Purpose                                       |
| --------------------- | --------------------------------------------- |
| `filesystem`          | Read/write files in the workspace             |
| `git`                 | Local git operations (status, log, diff)      |
| `sqlite`              | Query SQLite databases                        |
| `fetch`               | Generic HTTP GET                              |
| `memory`              | Per-session scratch storage                   |
| `time`                | Current date/time                             |
| `sequentialthinking`  | Multi-step reasoning scaffold                 |
| `duckduckgo`          | Free baseline web search                      |
| `arxiv`               | arXiv paper search/fetch                      |
| `wikipedia`           | Wikipedia article lookup                      |

Servers run in their own per-call containers spawned by the gateway,
with `no-new-privileges` enforced (set on the gateway in
`docker-compose.yaml`).

## First-party servers

`devai-model-status` is the first devai-authored MCP server (built
locally via `make build-mcp-modelstatus-image`, not pulled from a
registry) -- model catalog / probe-cache / bench-cache queries plus
live router status. See
[docs/mcp-model-status.md](mcp-model-status.md) for the tool
reference and the template it establishes for any future first-party
server.

## Client configurations

### Claude Code

`~/.config/claude/mcp.json`:

```json
{
  "mcpServers": {
    "devai-mcp": {
      "url": "http://devai-mcp-gateway:8088",
      "transport": "streaming"
    }
  }
}
```

When running Claude Code from the lab container, `devai-mcp-gateway`
resolves over `devai-net`. From the host, replace with
`http://localhost:8088`.

### Gemini CLI / Codex CLI

Both honour the same JSON shape (key may be `mcp` instead of
`mcpServers` depending on version). Point at the same URL.

### Open WebUI

Open WebUI's tool surface is configured in its admin UI. Add a tool
of type "MCP", URL `http://devai-mcp-gateway:8088`. The gateway's
streaming transport works against Open WebUI's HTTP-MCP shim.

## Security model

- Gateway runs in its own container with `no-new-privileges`.
- Spawned MCP server containers inherit the same flag.
- The gateway holds the Podman socket bind-mount; spawned containers
  do not.
- `--block-secrets` flag (set on the gateway command line) prevents
  secrets in environment variables from leaking through tool
  responses.
- Phase 1 ships zero secrets; Phase 2 mounts a tmpfs-backed
  `mcp-secrets.env` rendered from a sops-encrypted file.

## Phase 2 (Tier 2 servers, secrets)

Pre-requisites (per [docs/plans/mcp-gateway.md](plans/mcp-gateway.md)
Phase 2 + [docs/plans/sops-age-secrets.md](plans/sops-age-secrets.md)):

1. `make age-keygen-host` once on the host.
2. Add the printed public key to `.sops.yaml`.
3. `make secrets-tmpfs` once per boot.
4. Encrypt the four secrets:
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
5. `make mcp-secrets-render` (Phase 2 will add this target wrapping
   `scripts/render-secret.sh`).
6. Uncomment the Tier 2 entries in `deploy/mcp-servers.yaml`.
7. `make mcp-down && make mcp-up`.

Rotation, recovery, and multi-host onboarding live in
[docs/secrets.md](secrets.md).

## Troubleshooting

### Gateway exits immediately

Check `make mcp-logs` for catalog parse errors. The gateway pins
catalog schema v1; if `deploy/mcp-servers.yaml` adds a v2-only
field after a future bump, the gateway exits with a clear "unknown
field" error.

### Tools/list returns empty

Probably a Podman-socket permission issue. The gateway needs to be
able to spawn containers. Verify:

```bash
podman info --format '{{.Host.RemoteSocket.Path}}'
ls -la $(podman info --format '{{.Host.RemoteSocket.Path}}')
```

The path should be world-readable (mode 0660 or similar) and owned
by the user running compose.

### Tier 1 server times out on first call

Image pull happens on the first invocation per (server, host).
Subsequent calls reuse the cached image. Pre-pull the catalog with:

```bash
for img in $(yq '.servers[].image' deploy/mcp-servers.yaml); do
    podman pull "$img"
done
```

(`yq` is in the lab image; on the host install via apt or brew.)

## References

- Plan: [docs/plans/mcp-gateway.md](plans/mcp-gateway.md)
- Secrets scaffold: [docs/secrets.md](secrets.md)
- Upstream gateway: github.com/docker/mcp-gateway
- Server catalog: github.com/docker/mcp-registry
