package main

import (
	"encoding/json"
	"log"
)

// Reasoning policy on the Anthropic /v1/messages surface of vLLM and
// SGLang.
//
// The chat-completions policy (applyVLLMPolicy / applySGLangPolicy) never
// ran here: both gate on the path being exactly /v1/chat/completions, so a
// /v1/messages body reached the engine's Anthropic shim as the client sent
// it. That was harmless while the shims ignored the Anthropic effort field
// and stopped being harmless on 2026-09-22:
//
//   - Claude Code 2.1.278 sends `output_config: {"effort": "high"}` and
//     `thinking: {"type": "adaptive"}` on every /v1/messages turn (captured
//     on the wire; "high" is its default effort level, a Claude knob).
//   - vLLM 0.28 (entrypoints/anthropic/serving.py, _handle_output_config)
//     and SGLang 0.5.16 (entrypoints/anthropic/serving.py) both copy
//     `output_config.effort` into the chat request's `reasoning_effort`.
//   - Qwen3.8's chat template validates that kwarg whenever thinking is
//     on and accepts only xhigh / medium / low, so every turn died with
//     `400 Unexpected reasoning effort high` -- for the picker's default
//     agent, on the backend built for that model.
//
// So the router owns `output_config.effort` on this path exactly as it
// owns `reasoning_effort` on chat, per engine rule. Under ENABLE the
// client's value is REPLACED: on vLLM by the policy's effort (auto ->
// medium, the value its chat path sends and the same override of the
// checkpoint's own default); on SGLang by an explicit low/medium/high
// only, while `auto` REMOVES the effort, because SGLang's chat and
// Responses paths both inject nothing under auto ("the model's own default
// is the right answer") and this surface follows the same rule. Either
// way Claude Code's `high` never reaches the template. This field, and
// SGLang's `thinking` object under DISABLE, are the two places on this
// path where a client-supplied value does not win: the Anthropic effort
// is a Claude-model setting emitted unconditionally, not a request aimed
// at this model, and honouring it is what broke.
//
// DISABLE removes the effort (neither shim's Literal admits "none") and
// then pulls the lever each engine actually reads:
//
//   - vLLM: `chat_template_kwargs.enable_thinking=false`. That dict is a
//     real top-level field on its AnthropicMessagesRequest and is handed to
//     the chat request unchanged; the `thinking` request field is ignored
//     by its shim (it reads thinking content blocks only).
//   - SGLang: `thinking: {"type": "disabled"}`, replacing the WHOLE object.
//     Its shim maps `thinking.type != "disabled"` -- adaptive included --
//     straight to apply_reasoning_enabled(true), so Claude Code's
//     `adaptive` would force reasoning back on; and its validator
//     (`AnthropicThinkingParam._validate_thinking_shape`, sglang 0.5.16)
//     rejects `disabled` combined with `display` or `budget_tokens`, so
//     merging into Claude Code's `{"type":"adaptive","display":"omitted"}`
//     would 400. Its AnthropicMessagesRequest has no `chat_template_kwargs`
//     field, so the vLLM shape would be silently discarded there (the
//     extra_body lesson again).
//
// Stock vLLM v0.22.1 (port 11435) has no `output_config` field on its
// AnthropicMessagesRequest at all -- pydantic ignores the key -- so there
// this rewrite is inert and harmless; only vllm-devai (0.28) and SGLang
// read it. A `structured` model whose disable is NOT probe-verified gets
// reasoningNoop under policy=off, exactly as on chat, and its body passes
// through with the client's effort intact.
//
// Only the top level and the touched sub-object are re-encoded; every
// other byte of the (up to 183 KB) Claude Code body is preserved.
func (a *arbiter) applyHFAnthropicMessagesPolicy(backendName, modelName, policy string, body []byte) []byte {
	engine := engineOf(backendName)
	switch a.reasoningAction(backendName, modelName, policy) {
	case reasoningEnable:
		var out []byte
		var prev json.RawMessage
		effort := "(removed: SGLang auto keeps the model default)"
		if engine == "vllm" || policy == "low" || policy == "medium" || policy == "high" {
			effort = openAIReasoningEffort(policy)
			out, prev = setChildJSONField(body, "output_config", "effort", effort, true)
		} else {
			out, prev = deleteChildJSONFieldReturningPrev(body, "output_config", "effort")
		}
		clientSent := "nothing"
		if prev != nil {
			clientSent = string(prev)
		}
		log.Printf("info: %s/%s reasoning ENABLE on /v1/messages (policy=%q, output_config.effort=%s, client sent %s)",
			backendName, modelName, policy, effort, clientSent)
		if engine == "vllm" {
			out, _ = setChildJSONField(out, "chat_template_kwargs", "enable_thinking", true, false)
		}
		return out
	case reasoningDisable:
		log.Printf("info: %s/%s reasoning DISABLE on /v1/messages (policy=%q)", backendName, modelName, policy)
		out := deleteChildJSONField(body, "output_config", "effort")
		if engine == "vllm" {
			out, _ = setChildJSONField(out, "chat_template_kwargs", "enable_thinking", false, false)
			return out
		}
		return setTopJSONField(out, "thinking", map[string]any{"type": "disabled"})
	default:
		return body
	}
}

// setChildJSONField sets body[parent][key] = value, creating `parent` when
// absent or null. With overwrite=false an existing leaf is left alone.
// Returns the (possibly unchanged) body and the leaf's previous raw value,
// nil when there was none. The body comes back unchanged when it, or
// `parent`, is not a JSON object: a client-typed non-object is never
// rewritten. Everything outside `parent` is copied byte for byte.
func setChildJSONField(body []byte, parent, key string, value any, overwrite bool) ([]byte, json.RawMessage) {
	var top map[string]json.RawMessage
	if json.Unmarshal(body, &top) != nil || top == nil {
		return body, nil
	}
	child := map[string]json.RawMessage{}
	if raw, ok := top[parent]; ok {
		if json.Unmarshal(raw, &child) != nil {
			return body, nil
		}
		if child == nil { // JSON null: treat as absent
			child = map[string]json.RawMessage{}
		}
	}
	prev, exists := child[key]
	if exists && !overwrite {
		return body, prev
	}
	enc, err := encodeJSON(value)
	if err != nil {
		return body, prev
	}
	child[key] = enc
	childEnc, err := encodeJSON(child)
	if err != nil {
		return body, prev
	}
	top[parent] = childEnc
	out, err := encodeJSON(top)
	if err != nil {
		return body, prev
	}
	return out, prev
}

// deleteChildJSONField removes body[parent][key]; an emptied `parent` goes
// with it. Unchanged when either is absent or `parent` is not an object.
func deleteChildJSONField(body []byte, parent, key string) []byte {
	out, _ := deleteChildJSONFieldReturningPrev(body, parent, key)
	return out
}

// deleteChildJSONFieldReturningPrev is deleteChildJSONField plus the raw
// value that was removed (nil when nothing was).
func deleteChildJSONFieldReturningPrev(body []byte, parent, key string) ([]byte, json.RawMessage) {
	var top map[string]json.RawMessage
	if json.Unmarshal(body, &top) != nil || top == nil {
		return body, nil
	}
	raw, ok := top[parent]
	if !ok {
		return body, nil
	}
	var child map[string]json.RawMessage
	if json.Unmarshal(raw, &child) != nil || child == nil {
		return body, nil
	}
	prev, ok := child[key]
	if !ok {
		return body, nil
	}
	delete(child, key)
	if len(child) == 0 {
		delete(top, parent)
	} else {
		enc, err := encodeJSON(child)
		if err != nil {
			return body, nil
		}
		top[parent] = enc
	}
	out, err := encodeJSON(top)
	if err != nil {
		return body, nil
	}
	return out, prev
}
