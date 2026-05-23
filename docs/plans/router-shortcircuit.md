# Router request short-circuit

_Answer model-independent Claude Code probe requests directly in the router (without spinning up or occupying a GPU backend), and block Ollama download/mutation endpoints to enforce the local model-access policy._

## Status

Draft. Not yet scheduled for execution.

## Dependencies

- None. This is a single-mode router feature, independent of the
  sops-age scaffold, MCP gateway, and cluster mode. It touches only
  `gpu-arbiter/main.go`'s request path and adds one config file.

## Enables / Unblocks

- Relaxes (does not remove) the tier-collapse workaround in
  `scripts/model-picker.py:2267-2277`. Today all Claude tiers are pinned
  to one loaded model (`ANTHROPIC_SMALL_FAST_MODEL` /
  `ANTHROPIC_DEFAULT_HAIKU_MODEL`) specifically so Claude Code's
  background probes do not phantom-launch a second container and starve
  the foreground turn on the 600s health timeout. Once the *trivial*
  subset of those probes never reaches a backend at all, that pressure
  drops. (The non-trivial background calls -- compaction, summarization --
  still need a real model, so the pin stays useful; this only relaxes
  it.)
- Frees GPU cycles and avoids cold-start / drain churn caused by
  throwaway bookkeeping requests arriving mid-session.
- The Phase 1 fingerprint logger is reusable observability: it answers
  "which client is sending what to the router" for any future debugging,
  not just this feature.

## Out of scope

This section is load-bearing. The short-circuit is safe only because it
never replaces the model's *judgment*. The following are explicitly NOT
part of this plan, and a future contributor should not add them under
this banner:

- **Tool execution in the router** (running ripgrep, curl, jq, etc. and
  feeding the output back as a synthesized answer). The router sees a
  model-bound chat request, not the user's intent; deciding which tool
  to run is exactly the judgment the model exists to provide. Tool
  execution already happens locally one layer up -- the model emits a
  tool call and the agent harness runs it. Where local tools belong is
  the MCP gateway (`docs/mcp.md`: `fetch`, `filesystem`, `git`, `sqlite`,
  ...), invoked on the model's decision, not the router's guess.
- **URL fetching in the router.** Same reasoning. The `fetch` MCP server
  already does this when the model decides to.
- **Any request that requires semantic understanding**: summarization,
  context compaction, "is this a new topic", relevance judgments, tool
  selection. These look mechanical but are not. They fall through to the
  model untouched.
- **Exact-match response cache for real model outputs.** Caching a
  genuine model response keyed on request hash is a complementary idea,
  but it is a different mechanism (it caches model output; the first call
  still hits the model). This plan only covers answers that are produced
  with zero model involvement, ever. If a response cache is wanted, write
  it as a separate plan.
- **Protocol translation (Anthropic <-> OpenAI).** Unrelated; backends
  serve their own wire protocols and the router relies on that.

## Open questions

1. Detection patterns: config-driven JSON registry, or hardcoded in Go?
   Recommendation: **config-driven** (`deploy/shortcircuit-probes.json`,
   read at boot like the probe caches and `recovery-flags.json`). Claude
   Code's internal prompts change between releases; a config file lets an
   operator re-tune patterns without rebuilding the distroless router
   image.
2. Streaming probes: handle Anthropic SSE in Phase 2, or only
   non-streaming first? Recommendation: **handle both from the start.**
   Claude Code streams most `/v1/messages` calls; a non-streaming-only
   first cut may never fire for the common case. SSE synthesis is ~40
   lines of Go.
3. `count_tokens`: implement, or leave out? Recommendation: **defer to
   optional Phase 4**, and only build it if Phase 1 logs show Claude Code
   actually calls `/v1/messages/count_tokens` against the router AND a
   backend `/tokenize` endpoint is reachable to delegate to. A wrong
   token count drives wrong compaction decisions, so an approximation is
   worse than not answering.
4. Surface scope: `/v1/messages` only, or also `/v1/chat/completions`?
   Recommendation: **`/v1/messages` only.** The known probes are
   Claude-Code-specific and land on the Anthropic surface. Aider / Codex /
   interpreter on the OpenAI surface do not emit these.

## Context

A comparison with the free-claude-code project
(github.com/Alishahryar1/free-claude-code) surfaced a request-
optimization layer it ships: it intercepts a small set of throwaway
Claude Code probe requests and answers them locally without calling any
backend. Its detector (`api/detection.py`) and handlers
(`api/optimization_handlers.py`) cover five request types, three of which
are answered with a constant string and two with pure local string
parsing.

For free-claude-code the payoff is saving cloud quota and latency. For
devai the payoff is structural: every such probe currently contends for
the single, mutually-exclusive GPU, can trigger a cold start for a
backend whose answer will be discarded as a UI label, and in the worst
case hits the phantom-launch starvation that
`scripts/model-picker.py:2267-2277` already documents and works around.

The governing invariant, established before writing this plan, is the
formatter-vs-reasoner line: a request is safe to short-circuit only when
its correct answer is **independent of the conversation's semantic
content** -- a constant, or a pure syntactic function of the request
bytes. In those cases the model is being used as an expensive `printf`,
and removing it is lossless. The moment an answer needs judgment, the
request falls through to the model.

This environment (the box this plan was written on) has no GPU and no
running stack, so the patterns cannot be validated against live traffic
here. Rather than ship free-claude-code's heuristics on faith -- they may
be tuned to a specific Claude Code version -- this plan makes
observation its first phase: ship a request-fingerprint logger, run it
against a real session on a machine that has the stack, and only then
encode the validated patterns.

## Approach

Add an opt-in, fail-open short-circuit as "step 0" of the request
rewrite chain in `makeRequestHandler` (`gpu-arbiter/main.go:1890`),
inserted immediately after the malformed-JSON guard
(`main.go:1934`) and before any model-allowlist or backend-lifecycle
logic. It is gated behind `DEVAI_SHORTCIRCUIT` (default off), driven by a
config-file pattern registry, restricted to `/v1/messages`, and built so
that any uncertainty resolves by falling through to the model (a missed
probe is merely slower; a false match is silent corruption, so detection
must bias hard toward false negatives). Patterns are calibrated against
captured real traffic first, not assumed.

---

## Phase 1 -- request fingerprint instrumentation (observe-only)

### Goal

Give the router a way to record what requests actually arrive, so the
short-circuit patterns in later phases are grounded in evidence. No
behavior change: this phase never short-circuits anything. This is the
"empirical pass" productized so it can run wherever the live stack lives.

### Deliverables

```
gpu-arbiter/main.go            modify -- emit a structured fingerprint line per POST when DEVAI_REQUEST_FINGERPRINT=on
gpu-arbiter/fingerprint.go     new    -- pure fingerprint extraction (path, max_tokens, n_messages, has_system, system_sha256_12, has_tools, stream, body_bytes)
gpu-arbiter/fingerprint_test.go new   -- table-driven tests over captured/synthetic bodies
docs/router.md                 modify -- document the env var and the log-line shape
```

### Detailed steps

1. Add `fingerprintRequest(path string, body []byte) string` in a new
   `fingerprint.go`. It parses the body once and returns a single-line,
   greppable, content-free summary. Fields: `path`, `model` (raw, as
   sent), `max_tokens`, `n_messages`, `has_system`, `system_sha256_12`
   (first 12 hex of SHA-256 over the concatenated system text -- a stable
   bucket key that leaks no content), `has_tools`, `stream`, `body_bytes`.
2. In `makeRequestHandler`, after the body is read (`main.go:1913`) and
   JSON-validated (`main.go:1934`), if `DEVAI_REQUEST_FINGERPRINT=on`,
   `log.Printf("fingerprint: %s", fingerprintRequest(...))`. Default off.
3. Add an opt-in deep mode `DEVAI_REQUEST_FINGERPRINT_RAW=on` that
   additionally logs the first N bytes of the system prompt (truncated),
   for an operator doing a one-time pattern-discovery pass on their own
   local logs. Off by default; documented as content-revealing.
4. Document the operator procedure in `docs/router.md`: set the env,
   restart the router, run a real Claude Code session, then
   `make logs SERVICE=devai-router | grep '^fingerprint:'` and bucket by
   `(path, has_system, system_sha256_12, max_tokens, has_tools)`. The
   buckets that are high-frequency AND have a deterministic answer are the
   short-circuit candidates; capture one example body of each as a test
   fixture for Phase 2/3.

### Exit criteria

- With `DEVAI_REQUEST_FINGERPRINT=on`, every POST to the router emits
  exactly one `fingerprint:` line, and zero lines when the env is unset
  (verified by `make test-router`).
- The fingerprint line contains no message content (only a hash of the
  system text), verified by a unit test asserting the raw prompt text
  does not appear in the output.
- A short operator runbook exists in `docs/router.md`.

### Phase 1 risks

| Risk                                              | Mitigation                                                        |
| ------------------------------------------------- | ----------------------------------------------------------------- |
| Logging request shape leaks sensitive content     | Hash system text by default; raw mode is a separate explicit flag |
| Fingerprint parse adds latency to every request   | Single `json.Unmarshal` of already-read bytes; gated behind env   |

---

## Phase 2 -- constant-answer short-circuit

### Goal

Short-circuit the probes whose answer is a fixed string: quota check,
title generation, suggestion mode. Off by default. Calibrated against the
fixtures captured in Phase 1.

### Deliverables

```
deploy/shortcircuit-probes.json   new    -- pattern registry, "enabled": false default
gpu-arbiter/shortcircuit.go       new    -- registry load, detector, Anthropic JSON + SSE synthesizer
gpu-arbiter/shortcircuit_test.go  new    -- golden-response tests + match/no-match table over fixtures
gpu-arbiter/main.go               modify -- load registry at boot; call tryShortCircuit before backend logic
deploy/docker-compose.yaml        modify -- mount registry read-only; add DEVAI_SHORTCIRCUIT + SHORTCIRCUIT_REGISTRY env
docs/router.md                    modify -- document short-circuit as "step 0" of the rewrite chain
tests/fixtures/claude-code-probes/  new  -- captured request bodies (quota, title, suggestion)
```

### Detailed steps

1. Define the registry shape. Constant probes are fully data-driven:

   ```json
   {
     "enabled": false,
     "surface": "/v1/messages",
     "probes": [
       {
         "name": "quota_check",
         "match": { "max_tokens_eq": 1, "single_user_message": true,
                    "content_contains_any": ["quota"] },
         "respond": { "kind": "constant", "text": "Quota check passed." }
       },
       {
         "name": "title_generation",
         "match": { "system_contains_all": ["title"],
                    "system_contains_any": ["sentence-case title", "this session"] },
         "respond": { "kind": "constant", "text": "Conversation" }
       },
       {
         "name": "suggestion_mode",
         "match": { "user_content_contains_any": ["[SUGGESTION MODE:"] },
         "respond": { "kind": "constant", "text": "" }
       }
     ]
   }
   ```

   These mirror free-claude-code's `is_quota_check_request`,
   `is_title_generation_request`, `is_suggestion_mode_request` heuristics.
   Detection requires ALL declared markers to co-occur (high precision).
2. Implement `tryShortCircuit(path string, body []byte) (resp []byte,
   contentType string, matched bool)` in `shortcircuit.go`. Returns
   `matched=false` whenever the registry is disabled, the surface does
   not match, the body cannot be parsed, or no probe's full marker set is
   satisfied. Fail-open is the default for every error path.
3. Implement the synthesizer for both response modes:
   - Non-streaming Anthropic message object:
     `{"id":"msg_<uuid>","type":"message","role":"assistant",
       "model":<echoed>,"content":[{"type":"text","text":<answer>}],
       "stop_reason":"end_turn",
       "usage":{"input_tokens":<small>,"output_tokens":<small>}}`.
   - Streaming (`stream:true`): the Anthropic SSE sequence
     `message_start` -> `content_block_start` ->
     `content_block_delta`(text) -> `content_block_stop` ->
     `message_delta`(stop_reason) -> `message_stop`, flushed via
     `http.Flusher`, `Content-Type: text/event-stream`. (The router
     already flushes SSE on the proxy path; reuse the same flusher
     idiom.)
4. Load the registry at boot in `main.go` (next to the probe-cache loads)
   from `SHORTCIRCUIT_REGISTRY` (default
   `/etc/devai/shortcircuit-probes.json`). A missing file or
   `"enabled": false` means the feature is inert.
5. Insert the call in `makeRequestHandler` after `main.go:1934`:

   ```go
   if resp, ct, ok := a.tryShortCircuit(req.URL.Path, body); ok {
       w.Header().Set("Content-Type", ct)
       _, _ = w.Write(resp)
       log.Printf("shortcircuit: probe=%s path=%s model=%q bytes=%d",
           a.lastProbeName, req.URL.Path, parsed.Model, len(resp))
       return // GPU untouched: no allowlist, no recreate, no drain
   }
   ```

   Placed before the model-allowlist gate (`main.go:1973`) on purpose: a
   probe carrying Claude Code's hardcoded haiku id (unknown to the
   allowlist) gets answered instead of 404'd.
6. Mount the registry read-only into the router in
   `docs/docker-compose.yaml` and set `DEVAI_SHORTCIRCUIT` (the
   compose-level on/off; the registry's own `"enabled"` is the
   file-level switch -- both must be on).
7. Tests: golden-response assertions for each probe (exact JSON and exact
   SSE byte sequence), plus a no-match table proving a normal chat
   request, a summarization-style request, and a tool-call request all
   return `matched=false`.

### Exit criteria

- With the feature on and fixtures replayed through `tryShortCircuit`,
  each captured probe returns the correct canned response and a normal
  chat request falls through (`make test-router`).
- The synthesized Anthropic response (both stream and non-stream) is
  byte-validated against the golden fixtures.
- With the feature off (default), `tryShortCircuit` always returns
  `matched=false` and the request path is byte-identical to today.
- Every short-circuit hit logs a `shortcircuit:` line naming the matched
  probe, for audit.

### Phase 2 risks

| Risk                                                      | Mitigation                                                                 |
| --------------------------------------------------------- | -------------------------------------------------------------------------- |
| False positive: a real request matches a probe and gets a fabricated answer | Require ALL declared markers; bias to false negatives; off by default; audited via `shortcircuit:` log; calibrated against captured fixtures |
| Claude Code version drift changes a probe's wording       | Patterns in config (no rebuild); failure mode is graceful fall-through to the model; a regression test pins patterns to fixtures so a CI run flags when fixtures are regenerated against a newer Claude Code |
| Malformed synthesized wire format makes Claude Code error | Golden-response tests; shapes mirror free-claude-code's, which run against real Claude Code |
| Streamed probe not handled -> feature never fires         | Synthesizer handles both SSE and JSON from the start (open question 2)      |

---

## Phase 3 -- local-parse short-circuit

### Goal

Add the two probes whose answer is a pure syntactic function of the
request text rather than a constant: command-prefix detection and
filepath extraction. Same detection discipline; the only difference is
`respond.kind` dispatches to a Go function instead of returning a string.

### Deliverables

```
gpu-arbiter/shortcircuit.go       modify -- add "command_prefix" and "filepaths" responders + their markers
gpu-arbiter/shortcircuit_test.go  modify -- fixtures + golden outputs for both
deploy/shortcircuit-probes.json   modify -- two more probe entries
tests/fixtures/claude-code-probes/  modify -- captured prefix + filepath request bodies
```

### Detailed steps

1. Add registry entries:
   - `prefix_detection`: markers `body_contains_all: ["<policy_spec>",
     "Command:"]`; `respond.kind: "command_prefix"`.
   - `filepath_extraction`: markers `body_contains_all: ["Command:",
     "Output:"]` AND (`user_content_contains_any: ["filepaths"]` OR
     `system_contains_any: ["extract any file paths"]`);
     `respond.kind: "filepaths"`.
   These mirror free-claude-code's `is_prefix_detection_request` and
   `is_filepath_extraction_request`.
2. Implement the two responders as pure functions over the extracted
   `Command:` / `Output:` text:
   - `extractCommandPrefix(command string) string` -- the leading
     command token(s) Claude Code uses for permission matching.
   - `extractFilepaths(commandAndOutput string) string` -- regex
     file-path tokens out of the text.
   Port the exact logic from free-claude-code's
   `extract_command_prefix` / `extract_filepaths_from_command` and pin it
   with the captured fixtures.
3. Tests: golden outputs for representative commands; explicit no-match
   cases where `<policy_spec>` is absent.

### Exit criteria

- Both probes return outputs byte-identical to the captured fixtures.
- Removing a required marker from a fixture flips the result to
  fall-through.

### Phase 3 risks

| Risk                                              | Mitigation                                                       |
| ------------------------------------------------- | ---------------------------------------------------------------- |
| Prefix/filepath parser diverges from Claude Code's expectation | Port the upstream logic verbatim; pin with fixtures; fall through if extraction yields empty |

---

## Phase 4 -- count_tokens (optional)

### Goal

Answer `/v1/messages/count_tokens` in the router. Pure tokenization, zero
judgment -- the cleanest "model adds no information" case -- and the
backend (vLLM/SGLang) may not implement it at all, so this can convert a
current error into a correct answer. Optional because it depends on a
tokenizer.

### Detailed steps

1. Confirm via Phase 1 logs that Claude Code actually calls
   `/v1/messages/count_tokens` against the router. If it does not, stop --
   this phase has no purpose.
2. Prefer delegation over approximation: if the active backend exposes a
   `/tokenize` endpoint, proxy the count there (still no generation, no
   GPU contention beyond a cheap tokenize call). Only if no tokenizer is
   reachable, consider a model-specific tokenizer in the router -- and if
   that is not feasible, leave this unimplemented rather than return an
   approximate count.
3. Synthesize the Anthropic count_tokens response shape
   (`{"input_tokens": <n>}`).

### Exit criteria

- `/v1/messages/count_tokens` returns a count that matches the backend
  tokenizer for a set of fixtures, OR the phase is explicitly shelved
  with a one-line note saying why (no reachable tokenizer).

### Phase 4 risks

| Risk                                       | Mitigation                                                  |
| ------------------------------------------ | ----------------------------------------------------------- |
| Approximate token count drives wrong compaction | Delegate to a real tokenizer or do not implement; never approximate |

---

## Phase 5 -- Ollama mutation-endpoint guard (model-access policy)

### Goal

Enforce the devai model-access policy -- "local GPU = picker-vetted
models only, no downloading" (see
[pi-coding-agent](./pi-coding-agent.md) "Model-access policy") -- by
blocking Ollama download/mutation endpoints at the router's inbound
listener. It belongs in this plan because it is the same surface as the
short-circuit: an explicit, deterministic intervention in
`makeRequestHandler` / the backend mux, before the request reaches the
backend. vLLM/SGLang already enforce vetted-only via the model allowlist
(`main.go:1969-1985`); this closes the Ollama gap, where the allowlist is
skipped (`main.go:1973`) and unmatched paths proxy straight through (the
catch-all at `main.go:1320`).

Independently shippable, and arguably should ship first: a small
security/policy guard with no calibration step, unlike the probe
short-circuits.

### Deliverables

```
gpu-arbiter/main.go        modify -- register 403 handlers for Ollama mutation endpoints on the ollama listener, before the catch-all
gpu-arbiter/main_test.go   modify -- 403 on each guarded endpoint; chat/generate/tags/v1 still pass
docs/router.md             modify -- document the guard, the model-access policy, and why provisioning is unaffected
deploy/docker-compose.yaml modify -- DEVAI_OLLAMA_GUARD env (default on)
```

### Detailed steps

1. On the Ollama backend's mux (where `/v1/models`, `/api/tags`,
   `/health`, `/` are registered -- `main.go:1317-1320`), register
   explicit handlers returning HTTP 403 for the download/mutation
   endpoints, ONLY on the Ollama listener (vLLM/SGLang do not expose
   them):
   - `/api/pull`   -- download a model (the primary vector)
   - `/api/create` -- build/derive a model from a Modelfile
   - `/api/push`   -- upload to a registry (exfiltration)
   - `/api/copy`   -- mutate the local store
   - `/api/delete` -- destructive removal
   - `/api/blobs/` -- blob upload (path-prefix handler)
   The catch-all `/` keeps proxying `/api/chat`, `/api/generate`,
   embeddings, etc.
2. 403 body, actionable: `{"error":"endpoint disabled by devai policy:
   local models are picker-vetted only (no downloading). Provision on the
   host with 'make model-pull'."}`.
3. Gate behind `DEVAI_OLLAMA_GUARD` (default ON -- the policy is the
   default). An operator who genuinely needs `/api/delete` etc. disables
   it.
4. The guard is on the INBOUND listener only. The router's own outbound
   Ollama calls (warmup `/api/generate` at `main.go:1695`, `keep_alive`,
   `/api/ps`) go directly to `ollamaURL` and are unaffected.

### Exit criteria

- POST to `/api/pull`, `/api/create`, `/api/push`, `/api/copy`,
  `/api/delete` on the Ollama router port returns 403; `/api/chat`,
  `/api/generate`, `/api/tags`, `/v1/*` still work (`make test-router`).
- `make model-pull` still downloads -- the sanctioned path execs `ollama
  create` inside the devai-ollama container (`select-models.py:262`),
  bypassing the router entirely.
- With `DEVAI_OLLAMA_GUARD=off`, behaviour is byte-identical to today.

### Phase 5 risks

| Risk                                                          | Mitigation                                                                 |
| ------------------------------------------------------------- | -------------------------------------------------------------------------- |
| A devai workflow pulls via the router (`OLLAMA_HOST=devai-router` + `ollama pull`) and breaks | Verified the sanctioned pull path execs inside devai-ollama (`select-models.py:262`), bypassing the router; probers probe existing models and do not pull. Audit for router-routed pulls before enabling |
| Ollama adds a new download endpoint in a later version        | Guard is an explicit block-list; revisit on Ollama image bumps (same discipline as `deploy/backend-flags.yaml` pinning) |
| Over-blocking (`/api/delete`, `/api/copy` are mutations, not downloads) | Conservative reading of "vetted-only + no download"; per-endpoint granularity + the `DEVAI_OLLAMA_GUARD` toggle allow exempting them |

---

## Combined risk register

| Risk                                                | Phase | Mitigation                                                                 |
| --------------------------------------------------- | ----- | -------------------------------------------------------------------------- |
| Fabricating an answer to a real reasoning request   | 2-3   | Strict multi-marker detection, fail-open, off by default, per-hit audit log, calibrated against captured fixtures |
| Patterns rot across Claude Code versions            | 2-4   | Config-driven patterns; graceful fall-through; fixture-pinned regression tests |
| Privacy: request content in logs                    | 1     | Hash by default; raw is a separate explicit opt-in flag; logs are local    |
| Feature interacts badly with cluster head mode      | all   | Phases target `makeRequestHandler` (single/worker). In cluster mode a probe forwarded to a worker is still short-circuited at the worker, so correctness holds; short-circuiting at the head to save the head->worker hop is a noted optional follow-on, not part of this plan |
| Agent triggers a model download/mutation via the Ollama port | 5     | Router 403s `/api/pull` / `/api/create` / `/api/push` / `/api/copy` / `/api/delete`; provisioning bypasses the router (`select-models.py:262`) |

## Migration / rollback story

- Feature is off by default at two levels: `DEVAI_SHORTCIRCUIT` (compose)
  and `"enabled": false` (registry file). Existing installs see no
  behavior change until an operator opts in.
- No persistent state, no schema, no cache writes. Rollback is either
  flipping the env off or reverting the PR.
- Phase 1 (instrumentation) is independently revertible and has no
  dependency on Phases 2-4.
- Phase 5 (Ollama guard) defaults ON -- the policy is the default --
  unlike the probe short-circuit which defaults off. Disable per-install
  with `DEVAI_OLLAMA_GUARD=off`; rollback = revert the PR. It adds no
  state.

## Estimated effort

| Phase   | Engineering effort                          | Wall-clock   |
| ------- | ------------------------------------------- | ------------ |
| Phase 1 | 1 PR, ~150 LoC Go + tests                   | 0.5-1 day    |
| Phase 2 | 1 PR, ~250 LoC Go + registry + SSE + tests  | 1-2 days     |
| Phase 3 | 1 PR, ~150 LoC Go + fixtures                | ~1 day       |
| Phase 4 | 1 PR (optional), tokenizer-dependent        | 1-3 days or shelved |
| Phase 5 | 1 PR, ~60 LoC Go + tests (Ollama guard)     | ~0.5 day     |
| Total   | 4-5 PRs                                      | ~3-6 days (Phases 1-3 + 5) |

## References

- free-claude-code prior art: `api/detection.py` (the five detectors) and
  `api/optimization_handlers.py` (the five handlers and their canned
  responses) -- github.com/Alishahryar1/free-claude-code.
- `docs/router.md` -- the request rewrite chain this inserts as "step 0";
  `makeRequestHandler` integration point.
- `scripts/model-picker.py:2267-2277` -- the tier-collapse workaround this
  relaxes, and the phantom-launch failure that motivates it.
- `docs/mcp.md` -- where the out-of-scope "run local tools / fetch URLs"
  idea correctly lives.
- `deploy/recovery-flags.json`, `deploy/vllm-plugins.json` -- existing
  config-registry precedents the pattern registry follows.
- [pi-coding-agent](./pi-coding-agent.md) "Model-access policy" -- the
  general local=vetted-only / cloud=unrestricted rule Phase 5 enforces.
