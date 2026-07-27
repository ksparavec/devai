package main

import (
	"bytes"
	"encoding/json"
	"log"
)

// Anthropic /v1/messages normalisation for the HF backends.
//
// Claude Code sends a `role:"system"` message INSIDE `messages[]`, in
// addition to a correct top-level `system`. The Anthropic Messages API
// defines message roles as `user` | `assistant` only, with `system` as a
// top-level parameter -- and vLLM's and SGLang's compat shims implement
// that stricter schema, so both reject the request:
//
//	400 1 validation error:
//	  {'type': 'literal_error', 'loc': ('body', 'messages', 1, 'role'),
//	   'msg': "Input should be 'user' or 'assistant'", 'input': 'system'}
//
// Ollama's shim accepts the same shape, which is why Claude Code works
// against Ollama rows and fails against every vLLM/SGLang row. That made
// the picker's offer misleading: it advertises those rows and the default
// agent could not complete a single turn against any of them.
//
// This is a client/server API-version mismatch, not a devai defect --
// Claude Code emits a newer Anthropic beta wire format (note the
// `?beta=true` query, `context_management`, `output_config`) that the
// pinned engine images do not implement. The router is the single choke
// point that knows which backend a port maps to, so it is where the
// shapes get reconciled.
//
// Scope was established by replay against the live engines, not by
// reading schemas. Against a REAL captured Claude Code body (183 KB, 25
// tools, all beta fields present) on vLLM:
//
//	as-is                       -> 400, the exact locator above
//	folded, beta fields KEPT    -> 200
//
// So folding the stray messages is sufficient and NO field filtering is
// needed -- the beta fields are not the blocker. SGLang was verified
// separately (Phase 0 of the sglang-backend-remediation plan): it does
// expose /v1/messages, rejects the same body with the same locator, and
// accepts it once folded.
//
// Ollama is deliberately NOT rewritten. It is verified tolerant of the
// exact shape Claude Code sends, and rewriting a path that already works
// is unnecessary risk.

// normaliseAnthropicMessages moves every message whose role is not
// `user` or `assistant` into the top-level `system` block list,
// preserving order, and leaves the rest of the body alone.
//
// It returns the original slice unchanged -- byte for byte -- when there
// is nothing to move, so the overwhelmingly common case keeps its exact
// bytes and no re-serialisation cost. The second return value is how
// many messages were folded, for the caller's log line: a silent body
// rewrite is very hard to debug from the client side.
//
// Malformed or unexpected JSON is passed through untouched rather than
// erroring. This runs on every /v1/messages request to two backends; it
// must never be the reason a request fails.
func normaliseAnthropicMessages(body []byte) ([]byte, int) {
	if len(body) == 0 {
		return body, 0
	}

	// UseNumber keeps integers exact. Without it every number in the
	// body -- including whatever sits inside 25 tool JSON-Schemas --
	// round-trips through float64 and can come out in exponent form.
	var root map[string]json.RawMessage
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.UseNumber()
	if err := dec.Decode(&root); err != nil {
		return body, 0
	}

	rawMessages, ok := root["messages"]
	if !ok {
		return body, 0
	}
	var messages []map[string]json.RawMessage
	if err := json.Unmarshal(rawMessages, &messages); err != nil {
		return body, 0
	}

	// Cheap pre-scan: decide whether anything needs moving before
	// allocating or re-serialising anything.
	if !hasStrayRole(messages) {
		return body, 0
	}

	systemBlocks, ok := anthropicSystemBlocks(root["system"])
	if !ok {
		return body, 0
	}

	kept := make([]map[string]json.RawMessage, 0, len(messages))
	moved := 0
	for _, m := range messages {
		if isConversationRole(m["role"]) {
			kept = append(kept, m)
			continue
		}
		blocks, ok := anthropicContentBlocks(m["content"])
		if !ok {
			// A stray message we cannot interpret. Leaving it in place
			// would just reproduce the 400 we exist to prevent, but
			// silently dropping it would change model behaviour -- so
			// bail out entirely and let the engine speak for itself.
			return body, 0
		}
		systemBlocks = append(systemBlocks, blocks...)
		moved++
	}
	if moved == 0 {
		return body, 0
	}

	newMessages, err := json.Marshal(kept)
	if err != nil {
		return body, 0
	}
	newSystem, err := json.Marshal(systemBlocks)
	if err != nil {
		return body, 0
	}
	root["messages"] = newMessages
	root["system"] = newSystem

	out, err := json.Marshal(root)
	if err != nil {
		return body, 0
	}
	return out, moved
}

// isConversationRole reports whether a raw role value is one of the two
// roles the Anthropic Messages schema permits inside messages[].
func isConversationRole(raw json.RawMessage) bool {
	var role string
	if err := json.Unmarshal(raw, &role); err != nil {
		return false
	}
	return role == "user" || role == "assistant"
}

func hasStrayRole(messages []map[string]json.RawMessage) bool {
	for _, m := range messages {
		if !isConversationRole(m["role"]) {
			return true
		}
	}
	return false
}

// anthropicSystemBlocks normalises the top-level `system` parameter to a
// block list. It is valid as a bare string, as a block array, or absent.
func anthropicSystemBlocks(raw json.RawMessage) ([]json.RawMessage, bool) {
	if len(raw) == 0 {
		return nil, true
	}
	var blocks []json.RawMessage
	if err := json.Unmarshal(raw, &blocks); err == nil {
		return blocks, true
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return []json.RawMessage{textBlock(s)}, true
	}
	return nil, false
}

// anthropicContentBlocks normalises a message's `content` to a block
// list. Like `system` it may be a bare string or a block array.
func anthropicContentBlocks(raw json.RawMessage) ([]json.RawMessage, bool) {
	if len(raw) == 0 {
		return nil, true
	}
	var blocks []json.RawMessage
	if err := json.Unmarshal(raw, &blocks); err == nil {
		return blocks, true
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return []json.RawMessage{textBlock(s)}, true
	}
	return nil, false
}

func textBlock(s string) json.RawMessage {
	b, err := json.Marshal(map[string]string{"type": "text", "text": s})
	if err != nil { // unreachable for a string map
		return json.RawMessage(`{"type":"text","text":""}`)
	}
	return b
}

// maybeNormaliseAnthropic applies the rewrite on the surfaces that need
// it: the Anthropic messages path, on vLLM and SGLang only.
func (a *arbiter) maybeNormaliseAnthropic(backendName, path string, body []byte) []byte {
	if backendName != "vllm" && backendName != "sglang" {
		return body
	}
	if path != "/v1/messages" {
		return body
	}
	out, moved := normaliseAnthropicMessages(body)
	if moved > 0 {
		log.Printf("anthropic-compat: folded %d non-user/assistant message(s) into the top-level system for %s (%s)",
			moved, backendName, path)
	}
	return out
}
