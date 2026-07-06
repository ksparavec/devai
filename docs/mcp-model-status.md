# devai-model-status MCP server

The first devai-authored MCP server -- read-only queries over the
model catalog, the three probe caches, the bench cache, and live
router status. Registered with the Docker MCP Gateway
(`deploy/mcp-servers.yaml`), spawned per-call over stdio by the
gateway like any other server -- agents never invoke it directly.
Source: `devai-tools/cmd/devai-mcp-modelstatus` (Go, using the
official `github.com/modelcontextprotocol/go-sdk`). This doc is also
the template for any future first-party devai MCP server.

## Tools

### `list_fitting_models`

Input: `vram_gb` (int), `context` (int), `backend` (string, optional
-- `ollama`/`vllm`/`sglang`; omit for all three).

Joins `deploy/models.yaml` against the relevant probe cache(s). A
model is eligible only when the probe cache has an exact
`probes[vram_gb][context]` cell recorded as `fits: true` (vllm/sglang)
or `fully_on_gpu: true` (ollama) -- **no interpolation**. A gap means
"not probed at this cell", not "doesn't fit"; it never falls back to a
nearby context. This mirrors `scripts/model-picker.py`'s own filter
rule (see CLAUDE.md's "Filter" note under "Model picker").

Output: `{"models": [{"name", "backend", "family", "size", "purpose",
"repo"}, ...]}`. A probe-cache entry with no matching `models.yaml`
row (shouldn't normally happen, but the join is best-effort) still
appears, identified from the probe cache's own `repo`/`aliases`
fields rather than being silently dropped.

### `get_model_bench`

Input: `model` (catalog name, e.g. `Qwen3-8B-NVFP4` or
`qwen3.5:9b-q8_0`), `backend`, `context`.

Resolves the model name to the bench cache's base identifier
(`<repo>@<sha>` for vllm/sglang, straight off the matching
`models.yaml` row; the Ollama probe cache's digest for `ollama`, since
`models.yaml` carries no digest field), builds the bench-cache row key
`<base>::<backend>::<context>` per CLAUDE.md's documented convention,
and returns:

```json
{"tps": 98.3, "code_pct": 98.0, "reas_pct": 98.33, "total_pct": 97.67, "leak_pct": 0.0}
```

using the same formulas as the picker
(`scripts/model-picker.py:_picker_scores`): `reas_pct = 2/3 *
tools_use + 1/3 * gsm8k`, `total_pct = mean(gsm8k, humaneval,
tools_use)`, both as percentages.

On a miss (model/backend resolved, but no row at that exact context):

```json
{"message": "Bench: not available at ctx=32768 (have [131072]; run `make bench --ctx 32768` to populate)"}
```

-- the same message shape `model-picker.py`'s preview pane already
shows. On an unresolvable model/backend pair:
`{"message": "unknown model \"...\" for backend \"...\" (no matching catalog/probe-cache entry)"}`.

### `get_router_status`

No input. Tries the cluster-head control plane first
(`GET {DEVAI_ROUTER_CLUSTER_URL:-http://host.containers.internal:11444}/v1/cluster/status`,
short timeout); on success, relays the raw worker-status array under
`workers` and tags `mode: cluster-head`. On any failure to reach it
(the normal case for a single-mode host), falls back to probing
`/health` on the router's own three ports individually
(`{DEVAI_ROUTER_HOST:-host.containers.internal}:{11434,11435,11436}`,
each independently overridable via `DEVAI_ROUTER_{OLLAMA,VLLM,SGLANG}_PORT`)
and tags `mode: single`, returning each backend's `running` /
`current_model` / `active_reqs` (the router's actual `/health` handler
returns this JSON, not a bare "OK"). If nothing responds at all:
`{"mode": "unreachable", "error": "...", "backends": [...]}`. Never
returns a protocol error for an unreachable router -- an unreachable
status is itself a valid, informative result.

## Image and catalog registration

Built locally (`make build-mcp-modelstatus-image`, from
`deploy/Dockerfile.mcp-modelstatus`), tagged
`localhost/devai-mcp-modelstatus:latest` in the same host's Podman
image store the gateway already spawns containers from via the
mounted socket -- no registry push needed. Registered in
`deploy/mcp-servers.yaml` under a new "First-party (devai-authored)"
section, ahead of the Tier 2 entries.

**Config files are baked into the image at build time, not
bind-mounted.** None of the other 14 catalog entries in
`deploy/mcp-servers.yaml` demonstrate a per-server volume mount, and
the pinned gateway version's (`docker/mcp-gateway:v0.42.1`) exact
schema field for one isn't confirmed. Rather than guess at an
unverified catalog key, `Dockerfile.mcp-modelstatus` copies
`deploy/models.yaml` plus the four gitignored, host-generated
probe/bench cache files (falling back to an empty `{}` cache when one
doesn't exist yet, so the build never fails on a fresh checkout) into
`/etc/devai/` at build time. **Consequence: rebuild the image
(`make build-mcp-modelstatus-image`) after `make probe`/`make bench`
to pick up fresh data** -- `list_fitting_models`/`get_model_bench` see
whatever was baked in at the last build, not the live files.
`get_router_status` is unaffected -- it always talks to the live
router over HTTP.

If a future gateway version's schema is confirmed to support a
read-only per-server volume mount, switch to bind-mounting `deploy/`
instead and drop the build-time COPY step; note that migration here
when it happens.

## Verification

- `devai-tools/internal/modelcache/*_test.go`,
  `internal/routerclient/status_test.go`: table-driven unit tests
  (catalog/probe-cache/bench-cache parsing and joins, the three
  `get_router_status` code paths via `httptest.Server`).
- `cmd/devai-mcp-modelstatus/main_test.go`: drives all three tool
  handlers directly (bypassing stdio).
- A real stdio MCP-protocol round trip (`mcp.NewClient` +
  `mcp.CommandTransport` from the go-sdk, spawning the built binary)
  against the fixtures in `tests/fixtures/modelstatus/` verified
  `tools/list` and all three `tools/call` paths end-to-end during
  development, including the genuine `get_router_status` unreachable
  fallback path (real DNS/connection-refused errors surfaced per
  backend).
- `tests/test-mcp-modelstatus.sh`: end-to-end against a running
  gateway (`make mcp-up` with this catalog entry). Skips with exit 77
  when the gateway isn't reachable, matching `tests/test-mcp.sh`'s
  convention.
