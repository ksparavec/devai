# devai-model-status MCP server

The first devai-authored MCP server -- read-only queries over the
model catalog, the three probe caches, the bench cache, and live
router status. Registered with the Docker MCP Gateway
(`deploy/mcp-catalog-devai.yaml`), spawned per-call over stdio by the
gateway like any other server -- agents never invoke it directly.
Source: `devai-tools/cmd/devai-mcp-modelstatus` (Go, using the
official `github.com/modelcontextprotocol/go-sdk`). This doc is also
the template for any future first-party devai MCP server.

## Tools

### `list_fitting_models`

Input: `vram_gb` (int), `context` (int), `backend` (string, optional
-- `ollama`/`vllm`/`sglang`; omit for all three).

Joins `deploy/models.yaml` against the relevant probe cache(s). A
model is eligible when the probe cache has a cell **covering**
`(vram_gb, context)` recorded as `fits: true` (vllm/sglang) or
`fully_on_gpu: true` (ollama), not marked `serving_ok: false`, and --
for vllm/sglang -- backed by weights that are actually on disk.

Cell resolution reproduces `scripts/select-models.py` step for step,
and that file has **two** readers, not one, so the rule differs by
backend (`ProbeEntry.cellCovering` in
`devai-tools/internal/modelcache/probecache.go`):

1. **Clamp to the model's own ceiling first**, for every backend:
   `eff = min(context, max_context)` when the probe entry records a
   `max_context`. Asking a 65K-ceiling model to run at 256K just runs
   it at 65K -- the same rule `select-models.py` and the picker apply.
   `max_context` absent (`0`) means the entry has no clean cell, so
   there is nothing to clamp to and `eff` stays as requested.
2. **The exact cell at `eff` wins** when it is recorded. For **ollama**
   the lookup stops here: the Ollama cache records every probed tier
   separately, so a miss at `eff` genuinely means "that tier was never
   probed", and answering it from some other tier would invent a fit.
3. **vllm/sglang only**: on a miss at `eff`, the single binary-searched
   **winner cell at `max_context`** answers. Those caches keep exactly
   one winner per `(model, band)` -- the largest context that both fits
   and serves -- and by KV monotonicity it covers every smaller
   context. This is the single-cell invariant documented in
   [docs/backends.md](backends.md#probing----building-the-cache)
   ("Single-cell binary search (vLLM/SGLang)"); the two documents
   describe the same rule.

There is no upward extrapolation in either reader: a request that
resolves to no recorded cell means "not fitting", not "unknown".

`serving_ok` is honoured when present. A cell the LOAD probe recorded
as `serving_ok: false` does not fit; `serving_ok` absent keeps the
pre-load-probe, fit-only verdict.

**Weights-on-disk gate (vllm/sglang only).** A probe cache outlives the
weights it was measured from: cells stay behind after a model is
deleted, and the vLLM and SGLang stores are *separate volumes* (see the
`/var/cache/devai/` mount-point convention in `CLAUDE.md`), so a model
pulled for vLLM is invisible to SGLang. Each surviving row is therefore
also required to have `<store>/<name>/config.json`, where `<store>` is
`VLLM_MODELS_DIR` (default `/var/cache/devai/vllm`) or
`SGLANG_MODELS_DIR` (default `/var/cache/devai/sglang`) -- the same
test `model-picker.py` and `select-models.py`'s `sglang_weight_gaps`
use. Ollama is deliberately **not** gated: its models are a manifest
plus content-addressed blobs, not one directory per model, so there is
no equivalent path to stat.

When a store directory is not present at all, the gate cannot be
applied and the tool degrades to the un-gated, probe-cache-only verdict
rather than dropping every row -- and says so in a new top-level
`notes` array. An existing-but-empty store is **not** that case: it is
the real answer "this backend has no weights", and its rows are
correctly dropped.

Output: `{"models": [{"name", "backend", "family", "size", "purpose",
"repo"}, ...], "notes": ["..."]}`. `notes` is omitted when every
backend was gated normally. A probe-cache entry with no matching
`models.yaml` row (shouldn't normally happen, but the join is
best-effort) still appears, identified from the probe cache's own
`repo`/`aliases` fields rather than being silently dropped.

> **Known limitation in the published container.** The gateway-spawned
> server is distroless and mounts no weight volumes
> (`deploy/Dockerfile.mcp-modelstatus` bakes in only `models.yaml` and
> the four cache files), so in production the gate always degrades and
> `notes` names both HF backends on every call. The gate only bites
> when the binary runs on the host. Closing this needs either a
> read-only mount of the two store directories into the server, or a
> build-time weights manifest staged alongside the cache files.

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
{"tps": 98.3, "code_pct": 98.0, "code_plus_pct": 92.0, "leak_pct": 0.0}
```

`code_pct` is the `humaneval_subset_*` task (plain HumanEval) and
`code_plus_pct` the `humaneval_plus_subset_*` task (EvalPlus-hardened)
-- a bare `humaneval_` prefix matches both, which previously let
HumanEval+ be reported as HumanEval. The `reas_pct` / `total_pct`
composites are **gone**: the picker retired them as saturated, so
there was nothing left for them to be at parity with.

On a miss (model/backend resolved, but no row at that exact context):

```json
{"message": "Bench: not available at ctx=32768 (have [131072]; run `make bench --ctx 32768` to populate)"}
```

-- the same message shape `model-picker.py`'s preview pane already
shows. On an unresolvable model/backend pair:
`{"message": "unknown model \"...\" for backend \"...\" (no matching catalog/probe-cache entry)"}`.

### `get_router_status`

No input. Tries the cluster-head control plane first
(`GET {DEVAI_ROUTER_CLUSTER_URL:-http://devai-router:11444}/v1/cluster/status`,
short timeout); on success, relays the raw worker-status array under
`workers` and tags `mode: cluster-head`. On any failure to reach it
(the normal case for a single-mode host), falls back to probing
`/health` on the router's own three ports individually
(`{DEVAI_ROUTER_HOST:-devai-router}:{11434,11435,11436}`,
each independently overridable via `DEVAI_ROUTER_{OLLAMA,VLLM,SGLANG}_PORT`)
and tags `mode: single`, returning each backend's `running` /
`current_model` / `active_reqs` (the router's actual `/health` handler
returns this JSON, not a bare "OK"). If nothing responds at all:
`{"mode": "unreachable", "error": "...", "backends": [...]}`. Never
returns a protocol error for an unreachable router -- an unreachable
status is itself a valid, informative result.

Both defaults are the compose **service name** `devai-router`, not
`host.containers.internal`: the router publishes no host ports (it is
reachable only on `devai-net`), so the old host-gateway default could
never have answered and every call degraded to `mode: unreachable` by
construction. This requires the gateway-spawned server container to be
attached to `devai-net`; if it is not, name resolution fails and the
tool reports `unreachable` again.

`/v1/cluster/status` is bearer-authenticated, so the client sends
`Authorization: Bearer <token>` on that probe (and only on that probe
-- the per-backend `/health` handlers are unauthenticated and the
cluster token has no meaning on those ports). The token is read from
the first readable of `DEVAI_HEAD_TOKEN_FILE`,
`DEVAI_WORKER_TOKEN_FILE`, then `/run/devai/cluster-token`. When the
cluster probe does not answer, the reason is reported in the
`cluster_error` field of the result rather than being swallowed, so a
401 (stale or wrong token) is distinguishable from an unreachable head
before the tool falls back to the per-backend probes.

Caveat: the server runs in a gateway-spawned distroless container, and
its `deploy/mcp-catalog-devai.yaml` entry declares no volume mount, so
`/run/devai/cluster-token` is **not** visible inside it today. Against
a real authenticated head the client therefore sends no token and
`cluster_error` reports the 401. Point `DEVAI_HEAD_TOKEN_FILE` at a
path that is actually mounted into the container before expecting
`mode: cluster-head`.

## Image and catalog registration

Built locally (`make build-mcp-modelstatus-image`, from
`deploy/Dockerfile.mcp-modelstatus`), tagged
`localhost/devai-mcp-modelstatus:latest` in the same host's Podman
image store the gateway already spawns containers from via the
mounted socket -- no registry push needed. Registered in
`deploy/mcp-catalog-devai.yaml` under a new "First-party (devai-authored)"
section, ahead of the Tier 2 entries.

**Config files are baked into the image at build time, not
bind-mounted.** None of the other 14 catalog entries in
`deploy/mcp-catalog-devai.yaml` demonstrate a per-server volume mount, and
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
