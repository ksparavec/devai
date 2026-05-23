# Router fanout: per-query backend routing

_Route different queries within a single chat session to different backends/models -- via the request model field (agent model-slot env vars), explicit inline directives, and an @@all broadcast -- on one opt-in router surface, optimized for cluster mode._

## Status

Draft. Not yet scheduled for execution.

## Dependencies

- [Plan: gpu-arbiter-cluster-mode](./gpu-arbiter-cluster-mode.md) Phase 2
  (head mode + `routing_policy.go` + `cluster_proxy.go`) -- required for
  the concurrent-demux and broadcast phases. The cluster-first design
  centre lives on the head; without it, fanout degrades to serialized
  single-GPU switching.
- Single-host demux (Phase 1) has **no** hard dependency: it routes
  across the three local backends using the existing single-mode
  lifecycle. It is best-effort there (see the GPU-mutex constraint
  below).

## Enables / Unblocks

- Agent multi-model workflows mapped onto different devai targets without
  any agent-specific integration: Claude Code's foreground vs background
  slots (`ANTHROPIC_MODEL` vs `ANTHROPIC_SMALL_FAST_MODEL` /
  `ANTHROPIC_DEFAULT_HAIKU_MODEL`), Aider's `--model` / `--weak-model` /
  `--editor-model`, Open WebUI's per-message model picker. Each slot sets
  the request `model` field; the router maps it to a lane. This is the
  primary demux path and needs no `@@` prefix.
- Model-arena / ensemble evaluation via `@@all` -- send one prompt to N
  lanes and compare. Pairs naturally with the bench harness.
- Lifts the picker's one-model-per-session limitation: a session can
  reach multiple models/backends as the agent (or user) directs.

## Out of scope

- **Automatic intent classification.** Routing is explicit only (decided
  before writing this plan). The router never infers "this looks like
  code -> code lane" from message semantics. It only reads explicit
  signals: the `model` field and the `@@` directive. (A future opt-in
  classifier could be a separate plan; it is not this one.)
- **Router-held session state / memory.** The router stays per-request.
  "Stickiness" within a session, if added later, is derived by scanning
  the resent message history for the last directive -- not by storing
  server-side session state.
- **Response-synthesis quality / judging** beyond simple labeled
  aggregation. An optional judge lane is sketched but its prompt-
  engineering quality is not a goal of this plan.
- **Protocol translation (Anthropic <-> OpenAI)** and **tool execution /
  URL fetching in the router** -- separate concerns (the latter belongs
  to the MCP gateway, `docs/mcp.md`). Fanout dispatches a request to a
  backend; it does not transform the body's protocol or run tools.
- **Auto-discovering lanes from the probe caches.** Lanes are operator-
  declared in config. The router validates each lane's model against the
  caches but does not invent lanes.

## Open questions

1. Magic prefix character. `/` is intercepted client-side by many agents
   (Claude Code, Open WebUI); `::` is already devai's model-name reasoning
   suffix. Recommendation: **configurable, default `@@`**, matched only at
   the first non-whitespace position of the last user message.
2. Broadcast aggregation format. Recommendation: **`labeled`** (one
   response, lane-headed sections) as the cross-format default, since
   Anthropic `/v1/messages` has no multi-choice array; offer `choices`
   (OpenAI `n>1`) as an opt-in for OpenAI-only clients, and `judge` as the
   optional Phase 3 extra.
3. Single-mode `@@all` policy. On one GPU, broadcasting serializes into
   back-to-back cold starts. Recommendation: **refuse `@@all` in single
   mode** with a clear error unless every target lane is coexist-capable
   (e.g. all CPU-Ollama plus at most one GPU backend); always allowed in
   head mode.
4. Unknown `@@token`. Recommendation: **HTTP 400 listing valid lanes** --
   never silently forward a message with a stray `@@foo` to the default
   lane, and never leak the directive text to a model.
5. Wire surfaces in Phase 1. Recommendation: **OpenAI
   `/v1/chat/completions` + Anthropic `/v1/messages`, text content only.**
   Ollama-native `/api/chat` and multimodal content arrays are later
   additions.

## Context

devai today routes by PORT: port 11434 = Ollama, 11435 = vLLM, 11436 =
SGLang, and the router does not inspect message content (`docs/router.md`,
"What the router is NOT"). A client (wired by the picker) hits one port
for an entire session, so within a session everything goes to one
backend and one model (`<name>@<ctx>` bound at pick time).

The ask is to route *different queries within one session* to *different
backends*, generically (not Claude-Code-specific), using explicit inline
directives plus the agent's own model-slot env vars, with an `@@all`
broadcast option.

The hard constraint that shapes everything: on a single GPU the backends
are **mutually exclusive**. Only one of Ollama/vLLM/SGLang holds the GPU
at a time; switching drains, stops, and recreates the target -- a 60-300s
cold start (`docs/router.md`, "Backend switch"). So per-query fanout
across GPU-exclusive backends on one GPU is not concurrent: it serializes
with large cold-start penalties. It is genuinely useful only when (a) the
targets can coexist (CPU-Ollama alongside one GPU backend), or (b) you
are in **cluster mode**, where lanes live on different workers/GPUs and
serve in parallel. The design therefore centres on the cluster head
(decided: cluster-first), with single-host demux as a best-effort subset
that is honest about the cold-start cost.

A useful synergy: the cluster head *already* does per-request routing --
it parses `model` + `@<ctx>` + `::<reasoning>` (`parse_minimal.go`) and
scores workers (`routing_policy.go`) before proxying (`cluster_proxy.go`).
Fanout is largely a richer, user-steerable front-end to that machinery:
a lane abstraction, an inline override directive, and a broadcast fan.

## Approach

Add a shared, pure **lane resolver** -- `ResolveTargets(path, body) ->
([]Target, strippedBody, error)` -- that decides routing from explicit
signals only, in priority order: (1) an inline `@@<directive>` at the
start of the last user message (`@@all` -> broadcast; `@@<lane>` -> that
lane; `@@<model>[@ctx][::reasoning]` -> ad-hoc target reusing the existing
suffix grammar; unknown -> 400), stripping the directive from the
forwarded body; else (2) the request `model` field matched against
lane models / aliases (this is where agent env-var slots land); else (3)
the configured default lane. The resolver feeds a mode-specific
**dispatcher**: in single mode it drives one local backend through the
existing recreate/mutex lifecycle (broadcast serialized or refused); in
head mode each lane becomes worker-selection constraints fed to
`routing_policy.go` and proxied via `cluster_proxy.go` (broadcast = fan
to N workers concurrently, then aggregate). The whole feature is opt-in
(`DEVAI_FANOUT`, default off) on a dedicated surface; the existing
port==backend semantics are untouched for everyone who does not enable
it. Routing is deterministic and explicit -- the router parses directives
and the model field, never message semantics.

---

## Phase 1 -- lane resolver + single-host demux

### Goal

Per-query demux across the three local backends on one opt-in port, via
the model field (agent slots) and the inline `@@` directive. No broadcast
yet. Best-effort on a single GPU (honest about cold starts); fully
generic across clients.

### Deliverables

```
gpu-arbiter/fanout.go         new    -- LaneConfig load + pure ResolveTargets (model-field, @@ directive, default) + directive stripping
gpu-arbiter/fanout_test.go    new    -- table-driven resolver tests (OpenAI + Anthropic bodies, every resolution branch)
gpu-arbiter/main.go           modify -- load lane config at boot; when DEVAI_FANOUT_PORT set, start a fanout listener that runs the resolver then dispatches via the existing backend lifecycle
deploy/fanout-lanes.yaml      new    -- operator lane registry (prefix, default_lane, lanes, aliases), feature inert if absent
deploy/docker-compose.yaml    modify -- DEVAI_FANOUT / DEVAI_FANOUT_PORT / FANOUT_LANES env; mount the registry read-only
docs/fanout.md                new    -- operator + user guide (lanes, directives, agent env-var recipes, cold-start caveats)
docs/router.md                modify -- cross-link the fanout surface and note it does explicit (not semantic) content parsing
```

### Detailed steps

1. Lane registry shape (`deploy/fanout-lanes.yaml`):

   ```yaml
   prefix: "@@"            # inline directive marker (configurable)
   default_lane: chat
   lanes:
     chat:   { backend: ollama, model: "qwen3.5:9b-q8_0" }
     code:   { backend: vllm,   model: "gpt-oss-20b",    ctx: 65536 }
     reason: { backend: vllm,   model: "Qwen3-8B-NVFP4", reasoning: high }
   aliases:                 # model-field value -> lane (agent slot mapping)
     "claude-haiku-4-5-20251001": chat   # Claude Code background slot
     "weak":                      chat   # e.g. Aider --weak-model alias
   ```

2. Implement `ResolveTargets(path, body) ([]Target, []byte, error)` in
   `fanout.go`, pure and table-tested. A `Target` is
   `{backend, model, ctx, reasoning}`. Resolution order:
   a. Extract the last `user` message text (OpenAI `messages[-1].content`
      or Anthropic `messages[-1].content`; handle string and text-part
      array). If it starts (first non-whitespace) with `prefix`:
      - `@@all` (or configured broadcast keyword) -> reserved; in Phase 1
        return a 400 "broadcast not enabled until Phase 3".
      - `@@<lane>` -> that lane.
      - `@@<model>[@ctx][::reasoning][::mtp]` -> ad-hoc Target via the
        existing `parseCtxOverride` / `parseReasoningOverride` /
        `parseMTPOverride` chain.
      - unknown -> error (caller returns 400 listing valid lanes).
      Strip the directive line from the body and return the stripped body.
   b. Else match the request `model` against each lane's `model` and the
      `aliases` map. First match wins -> that lane.
   c. Else -> `default_lane`.
3. Start a fanout listener in `main.go` when `DEVAI_FANOUT_PORT` is set
   (default unset = feature off). The handler: read+cap body (reuse the
   32MB cap idiom), run `ResolveTargets`, rewrite `model` to the resolved
   target's name (carrying `@ctx` / `::reasoning` so the existing handler
   logic applies), and hand the resolved single Target to the existing
   `makeRequestHandler` machinery (`ensureBackendRunning`, mutex,
   reasoning policy, tool rewrites, proxy). No duplication of lifecycle
   logic -- fanout is a front-end that picks the backend, then delegates.
4. Validate each lane's model against the probe caches at boot; log and
   skip lanes whose model has no fitting probe cell (same eligibility
   rule the picker uses), so a typo'd lane fails loudly at startup, not
   mid-session.
5. Write `docs/fanout.md` with the agent env-var recipes, e.g.:
   - Claude Code: point `ANTHROPIC_BASE_URL` at the fanout port; set
     `ANTHROPIC_MODEL` to the `code` lane's model and
     `ANTHROPIC_DEFAULT_HAIKU_MODEL` to the `chat` lane's model -- the
     foreground turn lands on `code`, background bookkeeping on `chat`.
   - Aider: `--model` -> code lane, `--weak-model` -> chat lane.
   - Any client: type `@@reason explain this` to override per message.

### Exit criteria

- `ResolveTargets` returns the correct Target for: a model-field match, an
  alias match, each `@@<lane>` / `@@<model>@ctx` directive, the default
  fallback, and a 400 on unknown `@@token` -- all asserted in
  `fanout_test.go` over both OpenAI and Anthropic bodies.
- The directive is stripped from the forwarded body (the model never sees
  `@@...`), asserted by test.
- With `DEVAI_FANOUT_PORT` unset, the router is byte-identical to today
  (no new listener, existing ports unchanged).
- A real two-slot agent config (documented recipe) demuxes foreground vs
  background to two different lanes -- verified manually on a host with a
  stack (this environment has none; mark as deferred-to-hardware).

### Phase 1 risks

| Risk                                                    | Mitigation                                                                 |
| ------------------------------------------------------- | -------------------------------------------------------------------------- |
| Single-GPU cold-start thrash when two lanes are GPU-exclusive | Document loudly; recommend mapping the fast/weak lane to CPU-Ollama; this is why the design centre is cluster mode |
| Legit message starting with `@@` is misread as a directive | Match only at first non-whitespace; unknown token -> 400 (not silent); document an escape (`@@@@` -> literal `@@`) |
| Multimodal / non-text content arrays                    | Phase 1 handles text parts only; non-text last message with a `@@`-less body routes by model field / default |
| Lane model typo                                         | Boot-time validation against probe caches; skip + log invalid lanes        |

---

## Phase 2 -- cluster-mode demux (concurrent)

### Goal

Make demux first-class on the cluster head, where lanes serve
concurrently across workers -- the cluster-first payoff. Depends on
cluster-mode Phase 2 (head).

### Deliverables

```
gpu-arbiter/cluster_head.go   modify -- run ResolveTargets in the frontend handler before RouteDecision
gpu-arbiter/routing_policy.go modify -- accept lane constraints (backend/model/gpu-type) as routing inputs
gpu-arbiter/fanout.go         modify -- map a Target to worker-selection constraints
docs/fanout.md                modify -- cluster behaviour section
docs/cluster-mode.md          modify -- cross-link fanout as a steering layer over the 4-tier policy
```

### Detailed steps

1. In the head frontend handler (which today calls `ParseMinimal` then
   `RouteDecision`), call `ResolveTargets` first. A resolved Target
   becomes constraints fed into `routing_policy.go`: the lane pins the
   model (and optionally a `gpu_type` preference), and the existing 4-tier
   scorer picks the best matching worker. An `@@<model>` ad-hoc directive
   pins the model the same way a normal request would.
2. Because lanes live on different workers, two slots (foreground/code,
   background/chat) routed to two workers serve **concurrently** -- no GPU
   mutex between them. This is the behaviour single mode cannot offer.
3. Keep single-mode dispatch (Phase 1) unchanged; the resolver is shared,
   only the dispatcher differs by mode.

### Exit criteria

- On a head with two registered workers advertising the two lane models,
  alternating requests to the two lanes route to the two workers and do
  not serialize (verified in the cluster preflight harness or a head unit
  test with stub workers).
- `routing_policy.go` lane-constrained scoring is unit-tested
  (`routing_policy_test.go`).

### Phase 2 risks

| Risk                                          | Mitigation                                                  |
| --------------------------------------------- | ----------------------------------------------------------- |
| Lane model not present on any worker          | 4-tier policy already handles "right-model-too-small / different-model"; return its existing structured no-worker error |
| Coupling fanout tightly to head internals     | Resolver stays pure and mode-agnostic; only the thin dispatcher touches head code |

---

## Phase 3 -- broadcast (@@all)

### Goal

`@@all` (and `@@<lane1>,<lane2>`) sends one prompt to multiple lanes and
returns an aggregated response. Concurrent in head mode; refused or
serialized in single mode per the GPU-mutex policy.

### Deliverables

```
gpu-arbiter/fanout.go         modify -- ResolveTargets returns >1 Target for broadcast directives
gpu-arbiter/fanout_broadcast.go new  -- fan-out dispatch (concurrent in head, serialized/refused in single) + aggregation
gpu-arbiter/fanout_broadcast_test.go new -- aggregation golden tests (labeled + choices)
deploy/fanout-lanes.yaml      modify -- broadcast: { lanes, aggregate, judge_lane }
docs/fanout.md                modify -- broadcast section + single-mode caveat
```

### Detailed steps

1. Resolver: `@@all` -> the configured `broadcast.lanes`;
   `@@a,b` -> the named subset. Return a multi-Target result.
2. Dispatcher:
   - Head mode: issue the N upstream requests concurrently
     (`errgroup`-style), each through `cluster_proxy.go`, collect results.
   - Single mode: if all lanes are coexist-capable, serialize; otherwise
     refuse with a 400 (open question 3). Never silently take minutes.
   - Broadcast is **non-streaming first** (`stream:false` enforced;
     streamed broadcast deferred to Phase 4).
3. Aggregation (`aggregate` config):
   - `labeled` (default): one response whose content is lane-headed
     sections (`=== code (vllm/gpt-oss-20b) ===\n...`). Works for both
     OpenAI and Anthropic shapes.
   - `choices`: OpenAI-only; return N `choices[]`, one per lane.
   - `judge`: send the N answers to `judge_lane` for a synthesized pick
     (optional; quality not a goal here).
4. Tag each section/choice with its lane so the user knows the source.

### Exit criteria

- `@@all` in head mode returns all lane answers, labeled, with the
  upstream calls issued concurrently (asserted via stub-worker timing or
  call-count in a head test).
- Single-mode `@@all` either serializes (coexist-capable lanes) or returns
  the documented 400.
- Aggregation output matches golden fixtures for `labeled` and `choices`.

### Phase 3 risks

| Risk                                            | Mitigation                                                        |
| ----------------------------------------------- | ----------------------------------------------------------------- |
| One lane errors or times out mid-broadcast      | Per-lane error captured into its section; partial results returned, not a whole-request failure |
| Single-mode broadcast surprises with minutes-long latency | Refuse-by-default policy (open question 3); explicit operator override flag |
| Aggregation wire-format incorrectness           | Golden tests for both formats; non-streaming only in this phase   |

---

## Phase 4 -- polish (optional)

### Goal

Lower-priority refinements, each independently skippable.

### Candidate items

1. **Stickiness**: scan the resent message history for the most recent
   `@@<lane>` directive and apply it when the current message has none --
   stateless (history is resent by the client) yet sticky-feeling.
2. **Streamed broadcast**: interleave or sequentially stream lane outputs
   with lane separators over one SSE connection.
3. **Ollama-native + multimodal surfaces**: extend directive extraction to
   `/api/chat` and to text parts inside multimodal content arrays.
4. **Explicitly-deferred classifier hook**: a clearly opt-in, heuristic-
   first (keyword) auto-lane for prefix-less, default-bound requests --
   gated behind its own env, never overriding an explicit signal. Flagged
   here only so a future contributor sees it was considered and parked,
   consistent with the explicit-only decision.

### Exit criteria

- Each shipped item has its own tests; unshipped items stay documented as
  parked, not half-built.

---

## Combined risk register

| Risk                                                     | Phase | Mitigation                                                                 |
| -------------------------------------------------------- | ----- | -------------------------------------------------------------------------- |
| Per-query backend switching is cold-start-bound on one GPU | 1, 3 | Cluster-first design; single mode documented as best-effort; map fast lane to CPU-Ollama; refuse single-mode `@@all` |
| Content inspection departs from "route by port" tenet    | all   | Opt-in only; existing ports unchanged; parsing is explicit (directive + model field), never semantic -- same principle as the short-circuit plan |
| Directive false-trigger or leakage to the model          | 1     | First-token match, unknown -> 400, strip before forward, documented escape |
| Tight coupling to cluster head internals                 | 2-3   | Pure shared resolver; thin per-mode dispatcher                             |
| Security: fanout port reaches any backend/model          | all   | Internal to devai-net (same trust boundary as today); per-target model allowlist still enforced by the delegated handler |

## Migration / rollback story

- Off by default (`DEVAI_FANOUT` unset; registry absent = inert). Existing
  ports and the single-mode code path are byte-identical until an operator
  opts in. Rollback = unset the env / remove the registry / revert the PR.
- No persistent state or schema. The resolver is pure; the dispatcher
  reuses existing lifecycle code.
- Phase 1 ships and is useful without Phases 2-4. Phase 2+ require
  cluster-mode Phase 2 to be Done.

## Estimated effort

| Phase   | Engineering effort                              | Wall-clock        |
| ------- | ----------------------------------------------- | ----------------- |
| Phase 1 | 1-2 PRs, ~400 LoC Go + config + docs            | 3-5 days          |
| Phase 2 | 1 PR, ~200 LoC (head + policy wiring)           | 3-5 days (after cluster-mode P2) |
| Phase 3 | 1-2 PRs, ~300 LoC + aggregation + tests         | ~1 week           |
| Phase 4 | optional, per-item                              | 1-3 days each     |
| Total   | 3-5 PRs                                          | ~2-4 weeks (P1-3) |

## References

- [Plan: gpu-arbiter-cluster-mode](./gpu-arbiter-cluster-mode.md) -- the
  head routing this builds on (`parse_minimal.go`, `routing_policy.go`,
  `cluster_proxy.go`).
- [Plan: router-shortcircuit](./router-shortcircuit.md) -- shares the
  "router does explicit, not semantic, content parsing" principle.
- `docs/router.md` -- port layout, backend-switch / GPU-mutex cost, the
  `<name>@<ctx>::<reasoning>` suffix grammar the directive reuses.
- `scripts/model-picker.py:2255-2322` -- how agents are wired to the
  router today (the env-var slots the model-field path consumes).
- `deploy/fanout-lanes.yaml` (new) -- follows the config-registry pattern
  of `deploy/recovery-flags.json` / `deploy/vllm-plugins.json` /
  `deploy/mcp-servers.yaml`.
