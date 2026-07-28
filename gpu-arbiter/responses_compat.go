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
		// NOTHING is emitted. `none` is not a legal effort on this
		// surface, and the earlier "known limitation" turned out to be a
		// live defect when SGLang was actually tested:
		//
		//   vLLM    effort="none" -> 400 "reasoning_effort='none' is not
		//           supported by Harmony. Supported values are: high,
		//           medium, low."
		//   SGLang  effort="none" -> 500, and the cause is schema-level
		//           rather than model-specific:
		//           ResponsesRequest.reasoning.effort is
		//           Literal['low','medium','high'], so `none` is invalid
		//           for EVERY model, not just Harmony ones.
		//
		// The disable directive is also unreachable-or-useless in
		// practice today: 0 of 11 vLLM structured rows probe as
		// disable_verified, and the single SGLang row that does
		// (Qwen3.5-9B-NVFP4) returns 400 "input_ids should be a list of
		// lists for batch processing" on EVERY /v1/responses request,
		// minimal ones included, while /v1/chat/completions on the same
		// model returns 200.
		//
		// Rejected alternatives: emitting `low` would silently answer a
		// different question than the user asked; emitting
		// `chat_template_kwargs.enable_thinking=false` is accepted by
		// SGLang but only halves the trace (1104 -> 481 chars on
		// gpt-oss-20b), i.e. a partial disable presented as a disable.
		// Not honouring an unsupported directive, loudly, is the same
		// call this router makes for unverified tool parsers.
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
