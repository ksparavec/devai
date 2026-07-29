package main

import (
	"encoding/json"
	"log"
	"strings"
)

// OpenAI Responses API (`/v1/responses`) support.
//
// Codex speaks ONLY this wire -- `wire_api = "chat"` was removed upstream,
// and `responses` is now the sole variant. Both pinned engines implement
// the endpoint (vLLM v0.22.1 verified live; SGLang v0.5.10 registers it at
// http_server.py:1563), but the router's reasoning rewrite was gated to
// `/v1/chat/completions` and `/v1/messages`, so every Codex request went
// through with no reasoning policy at all.
//
// The shapes differ from Chat Completions, and the differences were
// measured against the live engine rather than read from a spec -- because
// the wrong ones are ACCEPTED and silently ignored, which is the failure
// mode this repo keeps rediscovering:
//
//	reasoning: {"effort": "low"}   ->  200, reasoning_tokens 298 -> 37
//	reasoning_effort: "low"        ->  200, reasoning_tokens 282  (IGNORED)
//
// A 200 proves the field was accepted, not honoured. Widening the path
// gate and reusing applyVLLMPolicy would therefore have produced a fix
// that looked right and did nothing.
//
// Two more measured facts drive the code below:
//
//   - tools must be FLATTENED (`{"type":"function","name":...}`). The
//     nested Chat Completions shape is rejected with 400 and 25 validation
//     errors, which is why `toolNameAt` -- written for the nested shape --
//     silently finds no name here.
//   - tool_choice pinning is NOT available. A flat `{type,name}` returns
//     501 "Only 'auto' or 'none' tool_choice is supported in response API
//     with Harmony", and the nested shape the router emits returns 400
//     "Tool choice 'function' not found in 'tools' parameter".
//     Re-measured on SGLang v0.5.16: a named pin is still rejected
//     (400 literal_error, "Input should be 'auto', 'required' or 'none'"),
//     and even `"required"` splits by model -- accepted on
//     Qwen3.5-9B-NVFP4, rejected on gpt-oss-20b with 400 "Only 'auto'
//     tool_choice is supported in response API". So the promotion skip
//     below stays correct on both engines.
//
// SGLang function tools: FIXED, and this is a real capability change.
// Custom function tools used to be unsupported on its Responses surface
// entirely (`tools[].type` was Literal['web_search_preview',
// 'code_interpreter']), which meant Codex could not use SGLang at all.
// On v0.5.16 a FLAT function tool returns 200 and actually calls:
// Qwen3.5-9B-NVFP4 answered `{"city": "Paris"}` under
// tool_choice="required". The nested Chat shape is still rejected (400
// "Function tools must include a name"), same as vLLM.
//
// Tool STRIPPING needs no change: `maybeStripTools` is not path-gated and
// operates on the top-level `tools` / `tool_choice` keys, which the
// Responses API also uses.

// isResponsesPath reports whether this request is on the Responses API.
// Sub-paths (`/v1/responses/{id}/cancel`) are included: they carry no
// generation parameters, and the rewrites below are all no-ops on a body
// that has none.
func isResponsesPath(path string) bool {
	p := strings.TrimRight(path, "/")
	return p == "/v1/responses" || strings.HasPrefix(p, "/v1/responses/")
}

// responsesEffort maps a router policy to a Responses `reasoning.effort`.
// Returns "" when nothing should be injected.
//
// `auto` deliberately injects NOTHING. The model's own default is the
// right answer, and inventing "medium" would silently override what the
// checkpoint ships -- the same decision applySGLangPolicy makes.
func responsesEffort(policy string) string {
	switch policy {
	case "low", "medium", "high":
		return policy
	case "off":
		// NOTHING is emitted -- but the reason CHANGED with the SGLang
		// v0.5.16 bump, and the old reason was measured, so it is worth
		// recording precisely.
		//
		// It used to be schema-level on SGLang: ResponsesRequest.reasoning
		// .effort was Literal['low','medium','high'], so `none` was
		// invalid for every model. v0.5.16 widened that schema (upstream
		// #31784, "align reasoning_effort schema across chat, tokenize and
		// responses"). Re-measured on this fleet:
		//
		//   Qwen3.5-9B-NVFP4  effort="none"    -> 200
		//   gpt-oss-20b       effort="none"    -> 500
		//   gpt-oss-20b       effort="minimal" -> 500
		//   gpt-oss-20b       low/medium/high  -> 200 (290/430/764 chars
		//                     of reasoning; 1030 with no directive)
		//
		// So it is now MODEL-specific: the Harmony guard still rejects
		// `none`, everything else accepts it. vLLM v0.22.1 (not bumped)
		// still 400s: "reasoning_effort='none' is not supported by
		// Harmony."
		//
		// Emitting `none` per-model is therefore possible but not safe on
		// the evidence available: `disable_verified` is probed on
		// /v1/chat/completions, and this surface demonstrably disagrees
		// with that one (gpt-oss takes reasoning_effort="none" on chat --
		// 802 -> 14 chars -- and 500s on responses). Gating a 500 on a
		// verdict measured against a different endpoint is how this repo
		// got the tautological disable_verified bug. Until the probe
		// covers /v1/responses directly, not honouring the directive
		// loudly beats guessing.
		//
		// Rejected alternatives: emitting `low` would silently answer a
		// different question than the user asked; emitting
		// `chat_template_kwargs.enable_thinking=false` is accepted by
		// SGLang but only halves the trace (1104 -> 481 chars on
		// gpt-oss-20b), i.e. a partial disable presented as a disable.
		return ""
	}
	return ""
}

// applyResponsesPolicy injects the reasoning directive for a
// /v1/responses request, in the shape that engine actually honours.
//
// Client-supplied values win, consistent with the rest of the rewrite
// chain: an existing `reasoning` object is left completely alone.
func (a *arbiter) applyResponsesPolicy(
	backendName, modelName, policy string, body []byte,
) []byte {
	action := a.reasoningAction(backendName, modelName, policy)
	if action == reasoningNoop {
		return body
	}
	effort := ""
	switch action {
	case reasoningEnable:
		effort = responsesEffort(policy)
	case reasoningDisable:
		// Say so rather than failing silently: the user asked for
		// reasoning off and is not getting it.
		log.Printf("info: %s/%s responses reasoning DISABLE requested but "+
			"unsupported on /v1/responses (effort=\"none\" is rejected: "+
			"vLLM 400, SGLang 500 -- its schema allows only low|medium|high); "+
			"serving at the model default",
			backendName, modelName)
		return body
	}
	if effort == "" {
		return body
	}

	var doc map[string]json.RawMessage
	if json.Unmarshal(body, &doc) != nil {
		return body
	}
	if _, present := doc["reasoning"]; present {
		return body // client asked for something; do not override it
	}
	enc, err := json.Marshal(map[string]string{"effort": effort})
	if err != nil {
		return body
	}
	doc["reasoning"] = enc
	out, err := json.Marshal(doc)
	if err != nil {
		return body
	}
	log.Printf("info: %s/%s responses reasoning effort=%q (policy=%q)",
		backendName, modelName, effort, policy)
	return out
}
