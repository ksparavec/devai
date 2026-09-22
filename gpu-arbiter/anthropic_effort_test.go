package main

import (
	"encoding/json"
	"testing"
)

// The /v1/messages shape Claude Code 2.1.278 sends, captured on the wire on
// 2026-09-22 (fields trimmed, values representative). `output_config.effort`
// is its default "high"; `thinking.type` is "adaptive"; the Read tool's
// schema carries MAX_SAFE_INTEGER, which any float64 round trip must keep.
const claudeCodeMessagesBody = `{"model":"qwen3.5:9b","max_tokens":32000,` +
	`"messages":[{"role":"user","content":"Reply with exactly: OK"}],` +
	`"system":[{"type":"text","text":"You are Claude Code"}],` +
	`"thinking":{"type":"adaptive","display":"omitted"},` +
	`"output_config":{"effort":"high"},` +
	`"metadata":{"user_id":"abc"},"stream":true,` +
	`"tools":[{"name":"Read","input_schema":{"type":"object","properties":{"limit":{"type":"integer","maximum":9007199254740991}}}}],` +
	`"context_management":{"edits":[]}}`

func hfTestArbiter() *arbiter {
	backends := []string{"vllm", "sglang", "vllm-devai"}
	return &arbiter{
		modelCapability: perBackend(backends, map[string]string{
			"qwen3.5:9b":      "structured",
			"some-vllm-model": "unknown",
			"inline-model":    "inline",
		}),
		modelDisableOK: perBackend(backends, map[string]bool{
			"qwen3.5:9b": true,
		}),
		defaultPolicy: "auto",
	}
}

func topLevel(t *testing.T, body []byte) map[string]json.RawMessage {
	t.Helper()
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("body not valid JSON: %v\n%s", err, body)
	}
	return raw
}

func childString(t *testing.T, body []byte, parent, key string) (string, bool) {
	t.Helper()
	raw := topLevel(t, body)
	p, ok := raw[parent]
	if !ok {
		return "", false
	}
	var child map[string]json.RawMessage
	if err := json.Unmarshal(p, &child); err != nil {
		t.Fatalf("%s is not an object: %s", parent, p)
	}
	v, ok := child[key]
	if !ok {
		return "", false
	}
	return string(v), true
}

func TestAnthropicEffort_ClaudeCodeHighBecomesPolicyEffort(t *testing.T) {
	a := hfTestArbiter()
	in := []byte(claudeCodeMessagesBody)
	out := a.applyReasoningPolicy("vllm", "/v1/messages", "qwen3.5:9b", "auto", in)

	if got, _ := childString(t, out, "output_config", "effort"); got != `"medium"` {
		t.Fatalf("auto must replace the client's effort with medium, got %s\n%s", got, out)
	}
	if got, _ := childString(t, out, "chat_template_kwargs", "enable_thinking"); got != "true" {
		t.Fatalf("vLLM enable must set chat_template_kwargs.enable_thinking=true, got %q\n%s", got, out)
	}
	// Everything the rewrite did not touch is byte-identical, including
	// the 2^53-1 in the tool schema and the adaptive thinking block.
	before, after := topLevel(t, in), topLevel(t, out)
	for k, v := range before {
		if k == "output_config" || k == "chat_template_kwargs" {
			continue
		}
		if string(after[k]) != string(v) {
			t.Fatalf("field %q changed: %s -> %s", k, v, after[k])
		}
	}
}

func TestAnthropicEffort_ExplicitPolicyLevels(t *testing.T) {
	a := hfTestArbiter()
	for _, tc := range []struct{ policy, want string }{
		{"low", `"low"`}, {"medium", `"medium"`}, {"high", `"high"`},
	} {
		out := a.applyReasoningPolicy("vllm", "/v1/messages", "qwen3.5:9b", tc.policy, []byte(claudeCodeMessagesBody))
		if got, _ := childString(t, out, "output_config", "effort"); got != tc.want {
			t.Fatalf("policy %s: effort %s, want %s", tc.policy, got, tc.want)
		}
	}
}

func TestAnthropicEffort_AbsentOutputConfigIsCreated(t *testing.T) {
	a := hfTestArbiter()
	in := []byte(`{"model":"qwen3.5:9b","messages":[]}`)
	out := a.applyReasoningPolicy("vllm", "/v1/messages", "qwen3.5:9b", "auto", in)
	if got, ok := childString(t, out, "output_config", "effort"); !ok || got != `"medium"` {
		t.Fatalf("expected output_config.effort=medium to be created, got %s", out)
	}
}

func TestAnthropicEffort_DisableOnVLLM(t *testing.T) {
	a := hfTestArbiter()
	out := a.applyReasoningPolicy("vllm", "/v1/messages", "qwen3.5:9b", "off", []byte(claudeCodeMessagesBody))
	if _, ok := childString(t, out, "output_config", "effort"); ok {
		t.Fatalf("disable must remove output_config.effort:\n%s", out)
	}
	if _, ok := topLevel(t, out)["output_config"]; ok {
		t.Fatalf("an emptied output_config must go too:\n%s", out)
	}
	if got, _ := childString(t, out, "chat_template_kwargs", "enable_thinking"); got != "false" {
		t.Fatalf("vLLM disable must set chat_template_kwargs.enable_thinking=false, got %q\n%s", got, out)
	}
	if got := string(topLevel(t, out)["thinking"]); got != `{"type":"adaptive","display":"omitted"}` {
		t.Fatalf("vLLM ignores the thinking field; it must be left alone, got %s", got)
	}
}

func TestAnthropicEffort_DisableOnSGLangSetsThinkingDisabled(t *testing.T) {
	a := hfTestArbiter()
	out := a.applyReasoningPolicy("sglang", "/v1/messages", "qwen3.5:9b", "off", []byte(claudeCodeMessagesBody))
	if _, ok := childString(t, out, "output_config", "effort"); ok {
		t.Fatalf("disable must remove output_config.effort:\n%s", out)
	}
	// The WHOLE object, not a merge: SGLang's AnthropicThinkingParam
	// validator (sglang 0.5.16, protocol.py) rejects `disabled` combined
	// with `display` or `budget_tokens`, and Claude Code sends
	// {"type":"adaptive","display":"omitted"}.
	if got := string(topLevel(t, out)["thinking"]); got != `{"type":"disabled"}` {
		t.Fatalf("SGLang treats adaptive as enabled; disable must replace the thinking object, got %s\n%s", got, out)
	}
	if _, ok := topLevel(t, out)["chat_template_kwargs"]; ok {
		t.Fatalf("SGLang's Anthropic request has no chat_template_kwargs field; sending one is a silent no-op:\n%s", out)
	}
}

func TestAnthropicEffort_SGLangAutoRemovesTheClientsEffort(t *testing.T) {
	// SGLang's chat and Responses paths inject nothing under auto (the
	// model's own default is the right answer); this surface follows the
	// same rule, and Claude Code's "high" still never reaches the template.
	a := hfTestArbiter()
	out := a.applyReasoningPolicy("sglang", "/v1/messages", "qwen3.5:9b", "auto", []byte(claudeCodeMessagesBody))
	if got, ok := childString(t, out, "output_config", "effort"); ok {
		t.Fatalf("SGLang auto must remove the client's effort, got %s\n%s", got, out)
	}
	if _, ok := topLevel(t, out)["chat_template_kwargs"]; ok {
		t.Fatalf("no chat_template_kwargs for SGLang:\n%s", out)
	}
	if got := string(topLevel(t, out)["thinking"]); got != `{"type":"adaptive","display":"omitted"}` {
		t.Fatalf("enable must leave thinking alone on SGLang, got %s", got)
	}
}

func TestAnthropicEffort_SGLangExplicitLevelsAreSet(t *testing.T) {
	a := hfTestArbiter()
	for _, tc := range []struct{ policy, want string }{
		{"low", `"low"`}, {"medium", `"medium"`}, {"high", `"high"`},
	} {
		out := a.applyReasoningPolicy("sglang", "/v1/messages", "qwen3.5:9b", tc.policy, []byte(claudeCodeMessagesBody))
		if got, _ := childString(t, out, "output_config", "effort"); got != tc.want {
			t.Fatalf("policy %s: effort %s, want %s", tc.policy, got, tc.want)
		}
		if _, ok := topLevel(t, out)["chat_template_kwargs"]; ok {
			t.Fatalf("no chat_template_kwargs for SGLang:\n%s", out)
		}
	}
}

func TestAnthropicEffort_SGLangInlineOffDisables(t *testing.T) {
	a := hfTestArbiter()
	out := a.applyReasoningPolicy("sglang", "/v1/messages", "inline-model", "off", []byte(claudeCodeMessagesBody))
	if got := string(topLevel(t, out)["thinking"]); got != `{"type":"disabled"}` {
		t.Fatalf("inline + off on SGLang must replace thinking, got %s", got)
	}
}

func TestAnthropicEffort_UnverifiedDisablePassesThrough(t *testing.T) {
	// structured + off + disable_verified=false is reasoningNoop, as on
	// chat. Here that means the client's effort reaches the template. It
	// cannot fire on today's caches (every such row is on stock vLLM
	// 0.22.1, which ignores output_config, or a non-Qwen3.8 template on
	// SGLang); pinned so the pass-through is deliberate, not accidental.
	a := &arbiter{
		modelCapability: perBackend([]string{"vllm"}, map[string]string{"unverified": "structured"}),
		modelDisableOK:  perBackend([]string{"vllm"}, map[string]bool{}),
		defaultPolicy:   "auto",
	}
	in := []byte(`{"model":"unverified","output_config":{"effort":"high"},"messages":[]}`)
	if out := a.applyReasoningPolicy("vllm", "/v1/messages", "unverified", "off", in); string(out) != string(in) {
		t.Fatalf("unverified disable must pass through untouched, got %s", out)
	}
}

func TestAnthropicEffort_ClientEnableThinkingFalseWinsUnderAuto(t *testing.T) {
	a := hfTestArbiter()
	in := []byte(`{"model":"qwen3.5:9b","chat_template_kwargs":{"enable_thinking":false},"output_config":{"effort":"high"},"messages":[]}`)
	out := a.applyReasoningPolicy("vllm", "/v1/messages", "qwen3.5:9b", "auto", in)
	if got, _ := childString(t, out, "chat_template_kwargs", "enable_thinking"); got != "false" {
		t.Fatalf("client's explicit enable_thinking must win on the lever half, got %s", got)
	}
	if got, _ := childString(t, out, "output_config", "effort"); got != `"medium"` {
		t.Fatalf("the effort half is ours regardless, got %s", got)
	}
}

func TestAnthropicEffort_ResponsesPathIsNotTouched(t *testing.T) {
	// Dispatch order: Responses first, then /v1/messages. A Responses body
	// carrying an Anthropic-shaped output_config must reach the Responses
	// policy, which has its own shape (reasoning.effort).
	a := hfTestArbiter()
	in := []byte(`{"model":"qwen3.5:9b","output_config":{"effort":"high"},"input":"hi"}`)
	out := a.applyReasoningPolicy("vllm", "/v1/responses", "qwen3.5:9b", "auto", in)
	if got, _ := childString(t, out, "output_config", "effort"); got != `"high"` {
		t.Fatalf("Responses path must not touch output_config, got %s", got)
	}
}

func TestAnthropicEffort_MalformedBodyPassesThroughOnDisable(t *testing.T) {
	a := hfTestArbiter()
	in := []byte(`{not json`)
	for _, backend := range []string{"vllm", "sglang"} {
		if out := a.applyReasoningPolicy(backend, "/v1/messages", "qwen3.5:9b", "off", in); string(out) != string(in) {
			t.Fatalf("%s: malformed body must pass through untouched on disable", backend)
		}
	}
}

func TestAnthropicEffort_UnknownCapabilityIsByteIdentical(t *testing.T) {
	a := hfTestArbiter()
	in := []byte(`{"model":"some-vllm-model","output_config":{"effort":"high"},"messages":[]}`)
	out := a.applyReasoningPolicy("vllm", "/v1/messages", "some-vllm-model", "auto", in)
	if string(out) != string(in) {
		t.Fatalf("unknown capability must pass through: %s", out)
	}
}

func TestAnthropicEffort_InlineOffDisables(t *testing.T) {
	a := hfTestArbiter()
	out := a.applyReasoningPolicy("vllm", "/v1/messages", "inline-model", "off", []byte(claudeCodeMessagesBody))
	if got, _ := childString(t, out, "chat_template_kwargs", "enable_thinking"); got != "false" {
		t.Fatalf("inline + off is the explicit opt-out; expected enable_thinking=false, got %q", got)
	}
}

func TestAnthropicEffort_VLLMDevaiReadsItsOwnCapability(t *testing.T) {
	// The derived checkpoints are probed on vllm-devai only. A lookup keyed
	// to "vllm" would read them as unknown and leave "high" in place.
	a := &arbiter{
		modelCapability: perBackend([]string{"vllm-devai"}, map[string]string{"Qwen3.8-27B-MTP-devai-NVFP4": "structured"}),
		modelDisableOK:  perBackend([]string{"vllm-devai"}, map[string]bool{"Qwen3.8-27B-MTP-devai-NVFP4": true}),
		defaultPolicy:   "auto",
	}
	out := a.applyReasoningPolicy("vllm-devai", "/v1/messages", "Qwen3.8-27B-MTP-devai-NVFP4", "auto", []byte(claudeCodeMessagesBody))
	if got, _ := childString(t, out, "output_config", "effort"); got != `"medium"` {
		t.Fatalf("vllm-devai must apply its own capability, got effort %s", got)
	}
	in := []byte(claudeCodeMessagesBody)
	if got := a.applyReasoningPolicy("vllm", "/v1/messages", "Qwen3.8-27B-MTP-devai-NVFP4", "auto", in); string(got) != string(in) {
		t.Fatalf("the same name on a backend that never probed it must be untouched")
	}
}

func TestAnthropicEffort_NonObjectOutputConfigIsRefused(t *testing.T) {
	a := hfTestArbiter()
	in := []byte(`{"model":"qwen3.5:9b","output_config":"nope","messages":[]}`)
	out := a.applyReasoningPolicy("vllm", "/v1/messages", "qwen3.5:9b", "auto", in)
	if got := string(topLevel(t, out)["output_config"]); got != `"nope"` {
		t.Fatalf("a client-typed non-object must never be rewritten, got %s", got)
	}
	if _, ok := topLevel(t, out)["reasoning_effort"]; ok {
		t.Fatalf("the effort must not leak to another field: %s", out)
	}
	out = a.applyReasoningPolicy("vllm", "/v1/messages", "qwen3.5:9b", "off", in)
	if got, _ := childString(t, out, "chat_template_kwargs", "enable_thinking"); got != "false" {
		t.Fatalf("disable still pulls the thinking lever even when effort could not be removed: %s", out)
	}
	if got := string(topLevel(t, out)["output_config"]); got != `"nope"` {
		t.Fatalf("output_config must survive untouched, got %s", got)
	}
}

func TestAnthropicEffort_OtherOutputConfigKeysSurvive(t *testing.T) {
	a := hfTestArbiter()
	in := []byte(`{"model":"qwen3.5:9b","output_config":{"effort":"high","format":{"type":"json_schema","schema":{"x":1}}},"messages":[]}`)
	out := a.applyReasoningPolicy("vllm", "/v1/messages", "qwen3.5:9b", "auto", in)
	if got, _ := childString(t, out, "output_config", "format"); got != `{"type":"json_schema","schema":{"x":1}}` {
		t.Fatalf("format must survive an effort rewrite, got %s", got)
	}
	out = a.applyReasoningPolicy("vllm", "/v1/messages", "qwen3.5:9b", "off", in)
	if got, _ := childString(t, out, "output_config", "format"); got != `{"type":"json_schema","schema":{"x":1}}` {
		t.Fatalf("format must survive an effort removal, got %s", got)
	}
	if _, ok := childString(t, out, "output_config", "effort"); ok {
		t.Fatalf("effort must be gone: %s", out)
	}
}

func TestAnthropicEffort_ClientEnableThinkingWins(t *testing.T) {
	// The effort is ours; the thinking kwarg still follows the chat-path
	// rule that an explicit client value is preserved.
	a := hfTestArbiter()
	in := []byte(`{"model":"qwen3.5:9b","chat_template_kwargs":{"enable_thinking":true},"messages":[]}`)
	out := a.applyReasoningPolicy("vllm", "/v1/messages", "qwen3.5:9b", "off", in)
	if got, _ := childString(t, out, "chat_template_kwargs", "enable_thinking"); got != "true" {
		t.Fatalf("client's explicit enable_thinking must win, got %s", got)
	}
}

func TestAnthropicEffort_MalformedBodyPassesThrough(t *testing.T) {
	a := hfTestArbiter()
	in := []byte(`{not json`)
	if out := a.applyReasoningPolicy("vllm", "/v1/messages", "qwen3.5:9b", "auto", in); string(out) != string(in) {
		t.Fatalf("malformed body must pass through untouched")
	}
}

func TestAnthropicEffort_ChatPathIsUnchanged(t *testing.T) {
	// The chat-completions surface keeps its own shape and its own
	// client-wins rule; this file must not have widened into it.
	a := hfTestArbiter()
	in := []byte(`{"model":"qwen3.5:9b","output_config":{"effort":"high"},"messages":[]}`)
	out := a.applyReasoningPolicy("vllm", "/v1/chat/completions", "qwen3.5:9b", "auto", in)
	if got, _ := childString(t, out, "output_config", "effort"); got != `"high"` {
		t.Fatalf("chat path must not touch output_config, got %s", got)
	}
	if got := string(topLevel(t, out)["reasoning_effort"]); got != `"medium"` {
		t.Fatalf("chat path keeps injecting reasoning_effort, got %s", got)
	}
}
