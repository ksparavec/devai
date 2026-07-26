# MCP Gateway

Operator reference for `devai-mcp-gateway`, the
[Docker MCP Gateway](https://github.com/docker/mcp-gateway) devai runs
as a peer service to `devai-router`. It is a single HTTP endpoint that
any MCP-aware agent can target; it spawns one short-lived container per
MCP server through the host's rootless Podman socket, so agents see the
gateway and never the individual servers.

This is the source of truth for the gateway. See also
`docs/mcp-model-status.md` (the one MCP server this repo authors) and
`docs/pipelock.md` (the peer opt-in service whose compose conventions
this one mirrors). `docs/plans/mcp-gateway.md` holds the original
design; its Tier 1 / Tier 2 phase language predates the 2026-07-25
rebuild and is history, not the current shape.

## Status snapshot (2026-07-25)

Verified live on this host, by running it:

- `make mcp-up` starts the gateway; its log reports
  `Initialized in 13.419170893s` and `134 tools listed`.
- A real MCP handshake against `http://127.0.0.1:8088/mcp` returns
  **134 tools**.
- All three first-party tools enumerate, and an end-to-end `tools/call`
  of `list_fitting_models` returns 27 models.
- 13 of the 15 enabled servers start. `filesystem` and
  `arxiv-mcp-server` do not -- see [Known limitations](#known-limitations).

The gateway is **opt-in**: it sits behind the compose profile `mcp`, so
`make cache-up` does NOT start it and nothing else in the stack depends
on it.

## Architecture: whose catalog

Two catalogs, merged:

- **Docker's official MCP catalog** supplies every third-party server.
  The gateway loads it by default; upstream digest-pins and maintains
  the entries.
- **`deploy/mcp-catalog-devai.yaml`** adds exactly one entry,
  `devai-model-status` -- the only MCP server this repo builds.

They are merged with `--additional-catalog=`, NOT `--catalog=`. The
latter *replaces* the built-in catalog, which would take every
third-party server with it.

**This repo hand-maintains zero third-party server definitions, and
that is deliberate.** The previous `deploy/mcp-servers.yaml` declared
all 14 third-party servers itself and pinned each at `:0.7.0`. That tag
exists for none of them: 13 of the 14 `docker.io/mcp/*` repositories
publish only `latest`, and `mcp/hugging-face` is not a repository at all
-- upstream converted it into a `type: remote` streamable-HTTP endpoint.
Pinning them ourselves is not fixable in principle either: with upstream
publishing only `latest`, the only pin available to us is a digest, and
a digest taken against a moving `latest` goes stale with no signal.
Deferring to the official catalog removes both problems, and it is what
the gateway is built for.

Cataloguing and enabling are separate axes. A catalog entry only makes a
server *available*; the `--servers=` flag decides which ones are
actually started. Without `--servers`, the gateway enables nothing and
serves only its own `mcp-find` / `mcp-add` builtins -- the observed
symptom is `0 tools listed` against a perfectly valid catalog.

Placement matters on the pinned image: v0.43.3 rejects a catalog path
outside the gateway's own catalogs directory, so compose mounts the file
at `/root/.docker/mcp/catalogs/devai.yaml`. The older
`/app/catalog.yaml` mount does not work on this version.

### First-party catalog schema

`deploy/mcp-catalog-devai.yaml` follows the gateway's own schema
(`pkg/catalog/types.go`): top level is `name` / `displayName` /
`registry`, where `registry` is a **map keyed by server name**, and each
entry carries a required `type` of `server`, `remote`, or `poci`. The
old `apiVersion` / `schemaVersion` / `servers:`-as-a-list shape does not
error -- it parses into an EMPTY registry, and the gateway then starts
cleanly and serves zero tools. That silent-empty failure mode is why
`make mcp-test` now asserts a tool-count floor.

The `devai-model-status` entry points at
`localhost/devai-mcp-modelstatus:latest`, built by
`make build-mcp-modelstatus-image`. The gateway runs servers with
`--pull never`, so a locally built tag with no upstream registry is the
right shape here -- but the image must exist before the gateway starts.

## Bring-up

```bash
make build-mcp-modelstatus-image   # once; first-party server image
make mcp-up                        # start gateway (compose profile 'mcp')
make mcp-test                      # real handshake + tools/list + tools/call
make mcp-logs                      # follow the gateway log
make mcp-down                      # stop and remove it
```

The gateway listens on `${MCP_PORT:-8088}` on the host, bound to
`127.0.0.1`, and on 8088 inside the container. `MCP_PORT` is a compose
*substitution* variable: set it in `.env` or export it before
`make mcp-up`. Startup is not instant -- the gateway pulls and starts
every enabled server, then enumerates their tools, which took ~13s here.

`make mcp-health` (`scripts/mcp-health.sh`) is a liveness probe only and
should not be trusted as a functional check. Verified today: `/health`
answers HTTP 200 with an **empty body** regardless of whether any server
started, and the `/servers` enumeration the script then attempts answers
401 without a bearer token, which the script ignores. Use
`make mcp-test`, which performs the real protocol exchange.

## Talking to the gateway

Four things an operator needs to know before any client will work:

1. **The URL includes the `/mcp` path.** `http://127.0.0.1:8088` alone
   returns 401; the endpoint is `http://127.0.0.1:8088/mcp`.
2. **A bearer token is required.** The gateway mints one at startup and
   prints it exactly once to its own log:
   `Use Bearer token: Authorization: Bearer <token>`. Every request
   needs `Authorization: Bearer <token>`. Treat the token as invalid
   after any `make mcp-down && make mcp-up` and re-read it from the log.
3. **Responses are SSE-framed.** The transport is `streaming`
   (streamable HTTP), and even a single JSON-RPC reply comes back as
   `Content-Type: text/event-stream` with an `event: message` line
   followed by `data: {...}`. Clients that assume a bare JSON body will
   fail to parse it.
4. **The handshake is `initialize` -> `notifications/initialized` ->
   `tools/list`.** The `initialize` response carries an `Mcp-Session-Id`
   **header**; send it back on every subsequent request.

Worked example, run against the live gateway on 2026-07-25:

```bash
TOKEN=$(podman logs devai-mcp-gateway 2>&1 \
        | grep -oE 'Bearer [A-Za-z0-9]+' | tail -1 | awk '{print $2}')

# 1. initialize -- capture the response headers for the session id
curl -sS -D /tmp/h.txt -X POST http://127.0.0.1:8088/mcp \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{
       "protocolVersion":"2025-06-18","capabilities":{},
       "clientInfo":{"name":"devai-doc","version":"1"}}}'

SESSION=$(grep -i '^mcp-session-id:' /tmp/h.txt | tr -d '\r' | awk '{print $2}')

# 2. notifications/initialized -- no response body
curl -sS -X POST http://127.0.0.1:8088/mcp \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H "Mcp-Session-Id: $SESSION" \
  -d '{"jsonrpc":"2.0","method":"notifications/initialized"}'

# 3. tools/list -- strip the SSE 'data: ' prefix to get JSON
curl -sS -X POST http://127.0.0.1:8088/mcp \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H "Mcp-Session-Id: $SESSION" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
  | sed 's/^data: //' | grep -m1 '^{'
```

The `initialize` reply identifies the server as
`Docker AI MCP Gateway` version 2.0.1 (image tag `v0.43.3`).
`tests/test-mcp.sh` is the same sequence with assertions attached; read
it as the executable version of this section.

## Enabled servers

The `--servers=` list in `deploy/docker-compose.yaml` currently enables
15 servers. Names are **upstream catalog keys and are case-sensitive** --
note `SQLite`, not `sqlite`, and the `-mcp-server` / `-mcp` suffixes on
the arxiv and wikipedia entries.

| Server                | Source           | Notes                                            |
| --------------------- | ---------------- | ------------------------------------------------ |
| `filesystem`          | Docker catalog   | Does not start -- needs a host path (see below)   |
| `git`                 | Docker catalog   | Local git operations                             |
| `SQLite`              | Docker catalog   | Capital S, capital L -- exact upstream key        |
| `fetch`               | Docker catalog   | Generic HTTP fetch                               |
| `memory`              | Docker catalog   | Scratch storage                                  |
| `time`                | Docker catalog   | Date/time                                        |
| `sequentialthinking`  | Docker catalog   | Multi-step reasoning scaffold                    |
| `duckduckgo`          | Docker catalog   | Web search, no key                               |
| `arxiv-mcp-server`    | Docker catalog   | Does not start -- needs a papers volume           |
| `wikipedia-mcp`       | Docker catalog   | Article lookup                                   |
| `github-official`     | Docker catalog   | Enumerates; calls need a secret (unverified)     |
| `firecrawl`           | Docker catalog   | Enumerates; calls need a secret (unverified)     |
| `hugging-face`        | Docker catalog   | `type: remote`, connects anonymously             |
| `context7`            | Docker catalog   | `type: remote`, connects anonymously             |
| `devai-model-status`  | first-party      | See `docs/mcp-model-status.md`                   |

`github-official` resolves from upstream to
`ghcr.io/github/github-mcp-server`. The old repo-local definition
pointed at `mcp/github`, which upstream archived.

The 134 tools counted above are what the 13 working servers provide.

## Adding or removing a server

Enabling a third-party server is a one-line edit: add or remove its
upstream catalog key in the `--servers=` list in
`deploy/docker-compose.yaml`, then `make mcp-down && make mcp-up`. Do
not add a definition for it -- the official catalog already has one, and
a local copy reintroduces exactly the drift that broke this before.

Adding a **first-party** server is two edits: a new entry in
`deploy/mcp-catalog-devai.yaml` (map key, `type: server`, a
`localhost/...` image built by a Makefile target) plus its key in the
`--servers=` list. The image must be built before the gateway starts,
because the gateway runs servers with `--pull never`.

The operator's standing decision is that third-party servers are wanted
and first-party servers are not: `devai-model-status` exists because no
third-party server can know about this lab's probe and bench caches, and
it is grandfathered rather than a template to copy.

## Security model

- The gateway holds a **read-write Podman socket** bind-mount
  (`${XDG_RUNTIME_DIR}/podman/podman.sock`) and must keep it read-write:
  creating per-call server containers is its entire job. Under rootless
  Podman, container-create through that socket is equivalent to the
  **invoking user (uid 1000)**, not host root -- which still means read
  and write access to everything that user owns, including
  `/var/cache/devai/` and `~/.config/sops/age/`.
- The **`127.0.0.1` publish is the mitigation** for that, together with
  the bearer token. Reaching the gateway from another machine means an
  SSH tunnel or an authenticating reverse proxy in front, not a wider
  bind.
- `--block-secrets` is set on the gateway command line: it keeps values
  from the secrets file from being echoed back through tool responses.
- `no-new-privileges:true` is set on the gateway container.
- Spawned server containers are created by the gateway, not by compose;
  they do not receive the Podman socket.
- The service is dual-homed on `devai-net` and `devai-lab-egress`, so
  the egress-locked lab containers can resolve it. Both
  `devai-mcp-gateway` and `mcp-gateway` are registered aliases on both
  networks (verified via `podman inspect`) and both names are listed in
  the two NO_PROXY sets (`PIPELOCK_NO_PROXY` in the Makefile,
  `NO_PROXY_HOSTS` in `bin/devai-agent`) so agent traffic to the gateway
  does not get sent through pipelock.

## Tier 2 secrets -- unverified

Two enabled servers need credentials to do anything: `github-official`
and `firecrawl`. Both enumerate their tools without one; calls will
fail. `hugging-face` and `context7` are `type: remote` upstream and
connect anonymously, so they need nothing.

**The secrets path is unverified end-to-end.** Two known problems:

1. Upstream's secret names are `github.personal_access_token` and
   `firecrawl.api_key`. `deploy/mcp-secrets.sops.env.example` still
   carries the older `GITHUB_TOKEN` / `FIRECRAWL_API_KEY` /
   `HF_TOKEN` / `CONTEXT7_API_KEY` names, which do not match, and
   whether the gateway accepts dotted keys from the `--secrets` env file
   has not been tested here.
2. The sops/age scaffold itself is not functional on this host:
   `.sops.yaml` still carries the literal `age1xxxx...` placeholder
   recipient and no age key has been generated, so
   `make mcp-secrets-render` has never produced a real file. See
   `docs/secrets.md` and the status entry in `docs/plans/README.md`.

The intended shape, for when someone does verify it:
`make age-keygen-host` once, add the printed public key to `.sops.yaml`,
`make secrets-tmpfs` once per boot, encrypt the plaintext into
`deploy/mcp-secrets.sops.env`, `make mcp-secrets-render` to write
`/run/devai/mcp-secrets.env`, set `MCP_SECRETS_FILE` to that path so the
compose bind mounts it at `/secrets/.env`, then restart the gateway.
With `MCP_SECRETS_FILE` unset the bind source is `/dev/null` and the
gateway simply reads an empty secrets file.

## Client configuration

**Not yet verified against this gateway.** The endpoint facts below are
verified; the per-agent config files are not, and the previous version
of this document shipped four confidently-wrong snippets, so no
replacement snippets are given here until someone has actually connected
each agent.

What a client has to be told, in every case:

| Field      | Value                                                     |
| ---------- | --------------------------------------------------------- |
| URL        | `http://127.0.0.1:8088/mcp` from the host                  |
|            | `http://devai-mcp-gateway:8088/mcp` from a lab container   |
| Transport  | streamable HTTP (SSE-framed responses)                     |
| Auth       | `Authorization: Bearer <token from the gateway log>`       |

The in-lab URL follows from the dual-homing and the registered network
aliases; a full handshake from inside a lab container has NOT been run.

Claude Code, Gemini CLI, Codex CLI and Open WebUI each name these three
things differently (different file, different key, different field for
the auth header), and at least some of them need the header supplied
explicitly rather than in the URL. Confirm against the agent's own
current documentation, then verify with `tools/list` before trusting it.

## Troubleshooting

### tools/list returns zero tools

Two causes, both observed:

- **The catalog parsed empty.** A first-party catalog file in the wrong
  schema does not error -- the gateway starts and exposes nothing. Check
  that `deploy/mcp-catalog-devai.yaml` has top-level `registry:` as a
  map with a `type:` on each entry.
- **`--servers=` is missing or misspelled.** With no `--servers` the
  gateway enables nothing at all. Keys are case-sensitive; `sqlite` is
  silently not `SQLite`.

`make mcp-test` catches both: it fails (not skips) whenever a reachable
gateway is below the tool floor.

### A server does not start

The gateway logs `Can't start <name>: <reason>` during initialisation
and then carries on without it, so the gateway looks healthy while one
server's tools are simply absent. Grep for it:

```bash
podman logs devai-mcp-gateway 2>&1 | grep -i "can't start"
```

### A first-party server's tools are missing

The gateway runs servers with `--pull never` (it pulls upstream images
itself during startup). A `localhost/...` image that has not been built
therefore fails to start rather than being fetched. Run
`make build-mcp-modelstatus-image`, then restart the gateway.

### HTTP 401

Missing or stale bearer token, or the request went to `/` instead of
`/mcp`. The token changes when the container is recreated; re-read it
from the log.

### Requests from the lab hang or get intercepted

The lab is egress-locked behind pipelock. `devai-mcp-gateway` and
`mcp-gateway` are both in `NO_PROXY`, so a client that honours
`NO_PROXY` reaches the gateway directly. A client that ignores
`NO_PROXY` will send the request to pipelock instead; see
`docs/pipelock.md`.

## Known limitations

- **`filesystem` does not start.** Log:
  `Error accessing directory : Error: ENOENT: no such file or directory,
  stat ''`, followed by `Can't start filesystem: failed to connect`. It
  wants a host path supplied through gateway config that is not set yet.
- **`arxiv-mcp-server` does not start.** Log:
  `invalid docker volume ":/app/papers": source and target are
  required`. Same class of problem -- an unset host path.
- **`get_router_status` enumerates but returns unreachable.** The
  gateway spawns each server in its own container without attaching it
  to `devai-net`, so the spawned container cannot resolve
  `devai-router`. The catalog schema exposes no field for a custom
  network (only `disableNetwork` and `extraHosts`). Unresolved; needs an
  operator decision. The other two first-party tools,
  `list_fitting_models` and `get_model_bench`, read caches baked into
  the image and work fine.
- **Tier 2 secrets are unverified**, with mismatched key names in the
  example file -- see the section above.

## References

- Upstream gateway: github.com/docker/mcp-gateway
- Upstream server catalog: github.com/docker/mcp-registry
- First-party server: `docs/mcp-model-status.md`
- Secrets scaffold: `docs/secrets.md`
- Egress firewall: `docs/pipelock.md`
- Original plan (pre-rebuild): `docs/plans/mcp-gateway.md`
