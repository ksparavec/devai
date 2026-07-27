# Router-side Anthropic /v1/messages normalisation

_Make Anthropic-native agents (Claude Code) usable against vLLM and
SGLang by normalising the `/v1/messages` request body in the router._

## Status

Draft. Not scheduled.

Root cause is **verified** (see Context -- captured on the wire and
reproduced byte-for-byte on 2026-07-27). Whether the proposed fix is
**sufficient** is NOT verified -- see Open question 1, which must be
answered before any code is written.

## Dependencies

None. This is self-contained inside `gpu-arbiter/`.

## Enables / Unblocks

- Claude Code against every vLLM / SGLang row the picker offers. Today
  the picker advertises those rows and the default agent cannot use any
  of them, so the offer is misleading.
- Any other Anthropic-native agent pointed at the router's HF-backend
  ports (11435 vLLM, 11436 SGLang).
- Removes the need for the operator to remember "Claude Code means pick
  an Ollama row".

## Out of scope

- The OpenAI surface (`/v1/chat/completions`). Unaffected -- the models
  that agents reach over that path already work.
- Ollama's `/v1/messages`. Verified tolerant of the exact shape Claude
  Code sends (see Context). Do NOT touch a path that already works.
- Implementing the Anthropic beta features themselves
  (`context_management`, `output_config`, extended `thinking`). The goal
  is only that vLLM/SGLang stop rejecting the request; honouring those
  fields is the engine's business, not the router's.
- Hiding vLLM/SGLang rows from the picker when the chosen agent is
  Claude Code. That is a UX workaround for this bug, not a fix, and it
  would make the picker's row set depend on the agent chosen afterwards
  -- the picker chooses the model FIRST.
- Bumping vLLM to obtain a better Anthropic shim. Noted as an
  alternative under Approach, but it is an upstream dependency bump with
  its own probe/bench invalidation cost (`make probe-check` drift), not
  a fix we control.

## Open questions

1. **Is folding the stray `system` message sufficient, or does vLLM then
   reject a beta-only field?** The captured request also carries
   `context_management`, `output_config`, `thinking`, `metadata`, `tools`
   and the `?beta=true` query. If any of those is also rejected, fixing
   only the message array just surfaces the next 400 and the plan needs
   a field-filtering step. -- recommendation: answer this FIRST, by
   replay (method below). Do not write router code before it is known.
2. **Fold the stray message into the top-level `system`, or drop it?**
   -- recommendation: fold. The message carries real instructions;
   dropping them silently changes model behaviour, which is the harder
   class of bug to notice.
3. **Scope the rewrite to vllm + sglang, or apply it on every backend?**
   -- recommendation: vllm + sglang only. Ollama is verified tolerant,
   and a rewrite there is unnecessary risk.
4. **SGLang: does its Anthropic shim behave like vLLM's?** Untested.
   -- recommendation: test before assuming; SGLang may not expose
   `/v1/messages` at all, which changes the answer for that backend from
   "normalise" to "the picker should not offer claude+sglang".

### How to answer question 1

The GPU was in use when this plan was written, so the replay was not
run. It needs the model loaded, so it is GPU-exclusive.

1. Capture a real body. Stand a logging HTTP server on
   `devai-lab-egress`, point a lab container at it with
   `ANTHROPIC_BASE_URL=http://<server>:<port>` plus
   `ANTHROPIC_AUTH_TOKEN=local`, mount the repo as the work dir (so the
   real `CLAUDE.md` is in play), and run
   `claude -p "hi" --model <name>@<ctx>`. Save the raw POST body.
2. Replay it at the engine directly (`http://devai-vllm:11434/v1/messages`),
   three variants: as-is; with non-`user`/`assistant` messages folded
   into the top-level `system` list; and folded plus the beta/extra
   fields removed. Rewrite `model` to the BARE id -- the captured body
   carries `<name>@<ctx>` because that is what the picker hands the
   ROUTER, and a 404 model-not-found would otherwise be misread as a
   schema verdict. Set `stream:false` so the status is readable;
   schema validation happens before streaming either way.
3. The first variant that returns 200 defines the required scope of the
   rewrite.

## Context

### Symptom

`devai-agent` -> Claude Code -> `Ornith-1.0-9B-NVFP4 @ 256K via vLLM`,
first turn:

```
API Error: 400 1 validation error:
  {'type': 'literal_error', 'loc': ('body', 'messages', 1, 'role'),
   'msg': "Input should be 'user' or 'assistant'", 'input': 'system', ...}
  File ".../vllm/entrypoints/utils.py", line 48, in create_messages
```

### What was ruled out

- **The router does not touch this path.** `applyVLLMPolicy` (and
  `applySGLangPolicy`) early-return the body unchanged unless the path
  is exactly `/v1/chat/completions`, so `/v1/messages` proxies through
  untouched. There is no system-message injection anywhere in
  `gpu-arbiter/`.
- **vLLM is not mangling the top-level `system` field.** Probed
  directly against `devai-vllm:11434/v1/messages` while serving
  `Ornith-1.0-9B-NVFP4`:

  | Request shape                                    | Result |
  | ------------------------------------------------ | ------ |
  | no `system` field                                | 200    |
  | top-level `system` as a STRING                   | 200    |
  | top-level `system` as a BLOCK ARRAY              | 200    |
  | `role:"system"` message INSIDE `messages[]`      | 400    |

  Only the last one fails, and it reproduces the reported error
  byte-for-byte including the `('body','messages',1,'role')` locator.

- **This is not recent work.** The decision to point Claude Code at the
  HF backends' `/v1/messages` dates from 2026-05-01 (`d96dde3`), with
  the surrounding block older still. The four picker commits of
  2026-07-27 changed which rows are displayed and the menu/cursor
  behaviour; the two router commits of that day added startup adoption
  of an already-serving backend and a keepalive ordering test. None of
  them touches a request body or this path.

### Root cause

Captured off the wire by standing a logging server in place of the
router (Claude Code v2.1.220):

```
POST /v1/messages?beta=true  (191559 bytes)
  top-level keys: [context_management, max_tokens, messages, metadata,
                   model, output_config, stream, system, thinking, tools]
  top-level 'system': list, 3 blocks
  messages[] length: 2
    [0] role='user'    content=[text,text]
    [1] role='system'  content=[text]     <-- not user/assistant
```

Claude Code sends a `role:"system"` message **inside** `messages[]`, in
addition to a correct top-level `system`. vLLM's Anthropic-compat shim
implements the stricter schema (`user` | `assistant` only) and rejects
it. Ollama's shim accepts the same shape (verified, 200), which is why
Claude Code works against Ollama rows and fails against vLLM rows.

This is a client/server API-version mismatch, not a devai defect:
Claude Code emits a newer Anthropic beta wire format (note
`?beta=true`, `context_management`, `output_config`) that
`vllm/vllm-openai:v0.22.1` does not implement.

### Why it surfaced on 2026-07-27

Two contributing causes, which have NOT been separated:

- The picker was crashing at startup on any host with a recorded bench
  verdict (fixed separately, same day), so an agent launch was never
  reached. Fixing that exposed the next failure.
- Claude Code self-updates, and its wire format has moved since the
  May wiring was written. Which Claude Code version last worked here is
  **unknown** -- not investigated.

### Practical impact today

| Backend | Claude Code's shape       | Verified result           |
| ------- | ------------------------- | ------------------------- |
| Ollama  | system-role in messages[] | 200 -- works              |
| vLLM    | same                      | 400 -- fails every turn   |
| SGLang  | same                      | NOT TESTED                |

So Claude Code is usable only with Ollama rows, regardless of which
model is picked.

## Approach

Normalise the Anthropic request body in the router, which is already the
single choke point for every agent and the only component that knows
which backend a port maps to. On `/v1/messages` for vllm and sglang,
move every message whose role is not `user`/`assistant` into the
top-level `system` block list, preserving order, and leave the rest of
the body alone. Client-supplied fields keep winning, consistent with the
existing rewrite chain's contract.

The rejected alternative is bumping vLLM until its shim accepts the
newer schema: it is not under our control, it invalidates the probe and
bench caches for every HF row (`_meta` image-drift, `make probe-check`),
and it would have to be repeated every time Claude Code's wire format
moves again. A ~40-line normaliser in the router is cheaper and
testable offline.

## Implementation

Single phase. If Open question 1 shows beta fields are also rejected,
add a second deliverable for per-backend field filtering rather than
widening this one.

### Deliverables

```
gpu-arbiter/anthropic_compat.go       new    -- normaliseAnthropicMessages()
gpu-arbiter/anthropic_compat_test.go  new    -- table-driven, incl. captured shape
gpu-arbiter/main.go                   modify -- call it in the rewrite chain
docs/router.md                        modify -- document the new rewrite step
CLAUDE.md                             modify -- update the rewrite-chain summary
```

### Detailed steps

1. Answer Open question 1. If folding alone is not enough, revise this
   plan before continuing.
2. Add `normaliseAnthropicMessages(body []byte) []byte`: for each
   element of `messages`, if `role` is not `user`/`assistant`, append its
   content blocks (normalising a bare string to
   `{"type":"text","text":...}`) to the top-level `system` list and drop
   the message. No-op when there is nothing to move, so the warm path
   keeps its exact bytes.
3. Wire it into the rewrite chain for backends `vllm` and `sglang`, on
   `/v1/messages` only, alongside the existing reasoning-policy step.
   Log once per rewrite at info level, naming how many messages moved --
   a silent body rewrite is very hard to debug from the client side.
4. Tests: the captured two-message shape; a bare-string `system`; an
   absent `system`; multiple stray messages (order preserved); a body
   with nothing to move (assert byte-identical output); malformed JSON
   (assert passthrough, never a panic).
5. Update `docs/router.md` (source of truth for the router) and the
   rewrite-chain line in `CLAUDE.md`.

### Exit criteria

- `make test-router` green, including a case built from the real
  captured body shape.
- A live `devai-agent` run with Claude Code against a vLLM row completes
  a first turn and a follow-up turn, with tool use exercised at least
  once. Recorded here with the model and ctx used.
- The same run against an Ollama row still works -- proof the Ollama
  path was not disturbed.
- `docs/router.md` documents the step and its backend/path gating.

### Risks

| Risk | Mitigation |
| ---- | ---------- |
| Folding changes model behaviour vs the client's intent | Preserve order and content blocks verbatim; log the rewrite; never drop content |
| Beta fields are the real blocker, so the fold fixes nothing | Open question 1 is a hard gate before any code |
| Rewrite touches Ollama and breaks a working path | Gate on backend AND path; exit criteria include an Ollama regression run |
| Body rewrite on a 190 KB request adds latency | Rewrite is a single parse/serialise on a request already costing a multi-second prefill; measure only if it shows up |
| Claude Code's format moves again | Tests are built from a captured body; recapture is a documented 2-step procedure (above) |

## Migration / rollback story

- Rollback is reverting the PR. The normaliser is additive and gated on
  backend plus path; with it removed the behaviour is exactly today's.
- No cache, config or on-disk format changes, so no probe or bench
  re-run is needed and no operator action is required on upgrade.

## Estimated effort

| Phase | Engineering effort | Wall-clock |
| ----- | ------------------ | ---------- |
| Open question 1 (replay) | no code; GPU-exclusive | ~30 min |
| Implementation | 1 PR, ~40 LoC + ~150 LoC tests | ~half a day |
| Live verification | 2 agent sessions | ~30 min |

## References

- `docs/router.md` -- request rewrite chain; source of truth for the router.
- `docs/openai-api-and-streaming.md` -- the Anthropic `/v1/messages`
  variant as this repo documents it.
- `gpu-arbiter/main.go` -- `applyReasoningPolicy`, `applyVLLMPolicy`,
  `applySGLangPolicy` (the early-return that proves the path is untouched).
- Anthropic Messages API: `system` is a top-level parameter; message
  roles are `user` and `assistant` only.
- vLLM `vllm/entrypoints/utils.py` `create_messages` -- the validation
  site named in the 400.
