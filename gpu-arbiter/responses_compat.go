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
		// Only reachable when the probe verified this model honours a
		// disable directive (see reasoningAction / modelDisableOK).
		//
		// KNOWN LIMITATION: that verification was performed on
		// /v1/chat/completions, and it does not transfer here. Harmony
		// models reject effort="none" with 400 ("Supported values are:
		// high, medium, low") -- but they also probe as
		// disable_verified=false for exactly that reason, so the gate
		// already keeps them out. A non-Harmony model that verified
		// disable on chat could still reject it here; that would surface
		// as a visible 400 rather than silent wrong behaviour.
		return "none"
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
		effort = responsesEffort("off")
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
