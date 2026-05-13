package main

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"
)

func newTestArbiter() *arbiter {
	disableTrue := true
	return &arbiter{
		modelCapability: map[string]string{
			"qwen3.5:9b":         "structured",
			"gemma4:e4b-it-bf16": "unsupported",
			"llama3.2:latest":    "error",
			"some-vllm-model":    "unknown",
		},
		modelDisableOK: map[string]bool{
			"qwen3.5:9b": disableTrue,
		},
		defaultPolicy: "auto",
	}
}

func bodyBoolField(t *testing.T, body []byte, key string) (set bool, val bool) {
	t.Helper()
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("body not valid JSON: %v", err)
	}
	r, ok := raw[key]
	if !ok {
		return false, false
	}
	var b bool
	if err := json.Unmarshal(r, &b); err != nil {
		t.Fatalf("%s not bool: %v", key, err)
	}
	return true, b
}

func bodyStringField(t *testing.T, body []byte, key string) (set bool, val string) {
	t.Helper()
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("body not valid JSON: %v", err)
	}
	r, ok := raw[key]
	if !ok {
		return false, ""
	}
	var s string
	if err := json.Unmarshal(r, &s); err != nil {
		t.Fatalf("%s not string: %v", key, err)
	}
	return true, s
}

func bodyObjectField(t *testing.T, body []byte, key string) (set bool, val map[string]any) {
	t.Helper()
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("body not valid JSON: %v", err)
	}
	r, ok := raw[key]
	if !ok {
		return false, nil
	}
	var m map[string]any
	if err := json.Unmarshal(r, &m); err != nil {
		t.Fatalf("%s not object: %v", key, err)
	}
	return true, m
}

func bodyHasThink(t *testing.T, body []byte) (set bool, val bool) {
	t.Helper()
	return bodyBoolField(t, body, "think")
}

func TestPolicy_AutoStructuredEnables(t *testing.T) {
	a := newTestArbiter()
	in := []byte(`{"model":"qwen3.5:9b","messages":[{"role":"user","content":"hi"}]}`)
	out := a.applyOllamaNativePolicy("qwen3.5:9b", "auto", in)
	set, val := bodyHasThink(t, out)
	if !set || !val {
		t.Fatalf("expected think:true, got set=%v val=%v body=%s", set, val, out)
	}
}

func TestPolicy_OffStructuredVerifiedDisables(t *testing.T) {
	a := newTestArbiter()
	in := []byte(`{"model":"qwen3.5:9b","messages":[]}`)
	out := a.applyOllamaNativePolicy("qwen3.5:9b", "off", in)
	set, val := bodyHasThink(t, out)
	if !set || val {
		t.Fatalf("expected think:false, got set=%v val=%v body=%s", set, val, out)
	}
}

func TestPolicy_UnsupportedNeverSets(t *testing.T) {
	a := newTestArbiter()
	in := []byte(`{"model":"gemma4:e4b-it-bf16","messages":[]}`)
	for _, p := range []string{"auto", "off", "low", "medium", "high"} {
		out := a.applyOllamaNativePolicy("gemma4:e4b-it-bf16", p, in)
		set, _ := bodyHasThink(t, out)
		if set {
			t.Fatalf("policy=%s: expected no think field for unsupported, got body=%s", p, out)
		}
	}
}

func TestPolicy_ErrorNeverSets(t *testing.T) {
	a := newTestArbiter()
	in := []byte(`{"model":"llama3.2:latest","messages":[]}`)
	out := a.applyOllamaNativePolicy("llama3.2:latest", "auto", in)
	set, _ := bodyHasThink(t, out)
	if set {
		t.Fatalf("expected no think for error capability")
	}
}

func TestPolicy_ClientThinkWins(t *testing.T) {
	a := newTestArbiter()
	in := []byte(`{"model":"qwen3.5:9b","think":false,"messages":[]}`)
	out := a.applyOllamaNativePolicy("qwen3.5:9b", "auto", in)
	set, val := bodyHasThink(t, out)
	// client supplied think:false; auto would set true; client wins.
	if !set || val {
		t.Fatalf("client think=false must survive, got set=%v val=%v body=%s", set, val, out)
	}
}

func TestPolicy_LowMediumHighAllEnable(t *testing.T) {
	a := newTestArbiter()
	for _, p := range []string{"low", "medium", "high"} {
		in := []byte(`{"model":"qwen3.5:9b","messages":[]}`)
		out := a.applyOllamaNativePolicy("qwen3.5:9b", p, in)
		set, val := bodyHasThink(t, out)
		if !set || !val {
			t.Fatalf("policy=%s: expected think:true (degenerate for ollama)", p)
		}
	}
}

func TestPolicy_PathSpecificOllamaNative(t *testing.T) {
	a := newTestArbiter()
	in := []byte(`{"model":"qwen3.5:9b","messages":[]}`)
	out := a.applyReasoningPolicy("ollama", "/api/chat", "qwen3.5:9b", "auto", in)
	set, val := bodyHasThink(t, out)
	if !set || !val {
		t.Fatalf("expected native path to set think:true, got body=%s", out)
	}
}

func TestPolicy_PathSpecificOllamaOpenAIChat(t *testing.T) {
	a := newTestArbiter()
	in := []byte(`{"model":"qwen3.5:9b","messages":[]}`)
	out := a.applyReasoningPolicy("ollama", "/v1/chat/completions", "qwen3.5:9b", "auto", in)
	setThink, _ := bodyHasThink(t, out)
	if setThink {
		t.Fatalf("OpenAI path must not set native think, got body=%s", out)
	}
	setEffort, effort := bodyStringField(t, out, "reasoning_effort")
	if !setEffort || effort != "medium" {
		t.Fatalf("expected reasoning_effort:medium, got set=%v effort=%q body=%s", setEffort, effort, out)
	}
}

func TestPolicy_OpenAIChatEffortLevels(t *testing.T) {
	a := newTestArbiter()
	for _, tc := range []struct {
		policy string
		want   string
	}{
		{"low", "low"},
		{"medium", "medium"},
		{"high", "high"},
	} {
		in := []byte(`{"model":"qwen3.5:9b","messages":[]}`)
		out := a.applyReasoningPolicy("ollama", "/v1/chat/completions", "qwen3.5:9b", tc.policy, in)
		_, effort := bodyStringField(t, out, "reasoning_effort")
		if effort != tc.want {
			t.Fatalf("policy=%s: expected effort=%q, got %q body=%s", tc.policy, tc.want, effort, out)
		}
	}
}

func TestPolicy_OpenAIChatOffStructuredVerifiedDisables(t *testing.T) {
	a := newTestArbiter()
	in := []byte(`{"model":"qwen3.5:9b","messages":[]}`)
	out := a.applyReasoningPolicy("ollama", "/v1/chat/completions", "qwen3.5:9b", "off", in)
	setEffort, effort := bodyStringField(t, out, "reasoning_effort")
	if !setEffort || effort != "none" {
		t.Fatalf("expected reasoning_effort:none, got set=%v effort=%q body=%s", setEffort, effort, out)
	}
}

func TestPolicy_ClientOpenAIReasoningEffortWins(t *testing.T) {
	a := newTestArbiter()
	in := []byte(`{"model":"qwen3.5:9b","reasoning_effort":"low","messages":[]}`)
	out := a.applyReasoningPolicy("ollama", "/v1/chat/completions", "qwen3.5:9b", "high", in)
	_, effort := bodyStringField(t, out, "reasoning_effort")
	if effort != "low" {
		t.Fatalf("client reasoning_effort must survive, got %q body=%s", effort, out)
	}
}

func TestPolicy_PathSpecificOllamaAnthropicMessages(t *testing.T) {
	a := newTestArbiter()
	in := []byte(`{"model":"qwen3.5:9b","messages":[]}`)
	out := a.applyReasoningPolicy("ollama", "/v1/messages", "qwen3.5:9b", "auto", in)
	setThink, _ := bodyHasThink(t, out)
	if setThink {
		t.Fatalf("Anthropic path must not set native think, got body=%s", out)
	}
	setThinking, thinking := bodyObjectField(t, out, "thinking")
	if !setThinking || thinking["type"] != "enabled" || thinking["budget_tokens"] != float64(2048) {
		t.Fatalf("expected Anthropic thinking object, got set=%v value=%v body=%s", setThinking, thinking, out)
	}
}

func TestPolicy_AnthropicMessagesOffIsNoop(t *testing.T) {
	a := newTestArbiter()
	in := []byte(`{"model":"qwen3.5:9b","messages":[]}`)
	out := a.applyReasoningPolicy("ollama", "/v1/messages", "qwen3.5:9b", "off", in)
	if string(out) != string(in) {
		t.Fatalf("Anthropic off_request is {}, expected unchanged body; got %s", out)
	}
}

func TestPolicy_UnknownOllamaPathIsNoOp(t *testing.T) {
	a := newTestArbiter()
	in := []byte(`{"model":"qwen3.5:9b","messages":[]}`)
	out := a.applyReasoningPolicy("ollama", "/api/embeddings", "qwen3.5:9b", "auto", in)
	if string(out) != string(in) {
		t.Fatalf("unknown ollama path should pass body through unchanged; got %s", out)
	}
}

func TestPolicy_VLLMUnknownCapabilityIsNoOp(t *testing.T) {
	// Phase 7+: vLLM is a no-op only when the model's capability is
	// not `structured` — same semantics as Ollama. `some-vllm-model`
	// has capability `unknown` in newTestArbiter().
	a := newTestArbiter()
	in := []byte(`{"model":"some-vllm-model","messages":[]}`)
	out := a.applyReasoningPolicy("vllm", "/v1/chat/completions", "some-vllm-model", "auto", in)
	if string(out) != string(in) {
		t.Fatalf("vllm path should pass body through unchanged for unknown; got %s", out)
	}
}

func TestPolicy_HeaderOverridesEnv(t *testing.T) {
	a := newTestArbiter() // defaultPolicy = auto
	req, _ := http.NewRequest("POST", "/", strings.NewReader(""))
	if got := a.requestPolicy(req); got != "auto" {
		t.Fatalf("expected auto by default, got %q", got)
	}
	req.Header.Set("X-DevAI-Reasoning", "off")
	if got := a.requestPolicy(req); got != "off" {
		t.Fatalf("expected off from header, got %q", got)
	}
	req.Header.Set("X-DevAI-Reasoning", "GARBAGE")
	if got := a.requestPolicy(req); got != "auto" {
		t.Fatalf("invalid header should fall back to default, got %q", got)
	}
}

func TestValidPolicy(t *testing.T) {
	for _, p := range []string{"auto", "off", "low", "medium", "high"} {
		if !validPolicy(p) {
			t.Errorf("%q should be valid", p)
		}
	}
	for _, p := range []string{"", "ON", "yes", "true"} {
		if validPolicy(p) {
			t.Errorf("%q should NOT be valid", p)
		}
	}
}

// ── Phase 7: vLLM / SGLang reasoning-policy rewrite ─────────────────────

// newTestArbiterHF builds an arbiter populated for vLLM/SGLang-style
// model names. Uses canonical HF repo basenames. `Qwen3.5-9B-NVFP4` has
// disable_verified=true; `Llama-3.1-8B-Instruct-NVFP4` is inline (no
// structured switch); `Untouched-Model` is unknown (the new noop case).
func newTestArbiterHF() *arbiter {
	return &arbiter{
		modelCapability: map[string]string{
			"Qwen3.5-9B-NVFP4":            "structured",
			"Qwen3-14B-NVFP4":             "structured",
			"Llama-3.1-8B-Instruct-NVFP4": "inline",
			"Untouched-Model":             "unknown",
		},
		modelDisableOK: map[string]bool{
			"Qwen3.5-9B-NVFP4": true,
			// Qwen3-14B-NVFP4 has structured but disable not verified.
		},
		// Qwen3.5-9B-NVFP4 has a probe-verified tool parser (used by the
		// strip-tools test); Qwen3-14B-NVFP4 doesn't. Keyed by backend.
		modelToolParser: map[string]map[string]string{
			"vllm": {"Qwen3.5-9B-NVFP4": "hermes"},
		},
		defaultPolicy: "auto",
	}
}

// readEnableThinking returns the value of
// extra_body.chat_template_kwargs.enable_thinking, or (false, false)
// when the path is missing.
func readEnableThinking(t *testing.T, body []byte) (set bool, val bool) {
	t.Helper()
	var root map[string]any
	if err := json.Unmarshal(body, &root); err != nil {
		t.Fatalf("body not valid JSON: %v", err)
	}
	eb, ok := root["extra_body"].(map[string]any)
	if !ok {
		return false, false
	}
	ctk, ok := eb["chat_template_kwargs"].(map[string]any)
	if !ok {
		return false, false
	}
	v, ok := ctk["enable_thinking"]
	if !ok {
		return false, false
	}
	b, ok := v.(bool)
	if !ok {
		t.Fatalf("enable_thinking not bool: %T %v", v, v)
	}
	return true, b
}

func TestPolicy_VLLMStructuredEnable(t *testing.T) {
	a := newTestArbiterHF()
	for _, p := range []string{"auto", "low", "medium", "high"} {
		in := []byte(`{"model":"Qwen3.5-9B-NVFP4","messages":[]}`)
		out := a.applyReasoningPolicy("vllm", "/v1/chat/completions", "Qwen3.5-9B-NVFP4", p, in)
		setT, valT := readEnableThinking(t, out)
		if !setT || !valT {
			t.Fatalf("policy=%s: expected enable_thinking=true, got set=%v val=%v body=%s", p, setT, valT, out)
		}
		_, effort := bodyStringField(t, out, "reasoning_effort")
		want := openAIReasoningEffort(p)
		if effort != want {
			t.Fatalf("policy=%s: expected reasoning_effort=%q, got %q body=%s", p, want, effort, out)
		}
	}
}

func TestPolicy_VLLMOffDisableVerified(t *testing.T) {
	a := newTestArbiterHF()
	in := []byte(`{"model":"Qwen3.5-9B-NVFP4","messages":[]}`)
	out := a.applyReasoningPolicy("vllm", "/v1/chat/completions", "Qwen3.5-9B-NVFP4", "off", in)
	setT, valT := readEnableThinking(t, out)
	if !setT || valT {
		t.Fatalf("expected enable_thinking=false, got set=%v val=%v body=%s", setT, valT, out)
	}
	_, effort := bodyStringField(t, out, "reasoning_effort")
	if effort != "none" {
		t.Fatalf("expected reasoning_effort=none, got %q body=%s", effort, out)
	}
}

func TestPolicy_VLLMOffWithoutDisableVerifiedNoop(t *testing.T) {
	a := newTestArbiterHF()
	// Qwen3-14B-NVFP4 is structured but disable not verified.
	in := []byte(`{"model":"Qwen3-14B-NVFP4","messages":[]}`)
	out := a.applyReasoningPolicy("vllm", "/v1/chat/completions", "Qwen3-14B-NVFP4", "off", in)
	if string(out) != string(in) {
		t.Fatalf("off without disable_verified must not modify body, got %s", out)
	}
}

func TestPolicy_VLLMInlineNonOffNoop(t *testing.T) {
	a := newTestArbiterHF()
	in := []byte(`{"model":"Llama-3.1-8B-Instruct-NVFP4","messages":[]}`)
	// Non-off policies: inline capability has no structured switch, so
	// we leave the body alone. Applies to auto/low/medium/high.
	for _, p := range []string{"auto", "low", "medium", "high"} {
		out := a.applyReasoningPolicy("vllm", "/v1/chat/completions", "Llama-3.1-8B-Instruct-NVFP4", p, in)
		if setT, _ := readEnableThinking(t, out); setT {
			t.Fatalf("policy=%s: inline must not set enable_thinking, got body=%s", p, out)
		}
		if setE, _ := bodyStringField(t, out, "reasoning_effort"); setE {
			t.Fatalf("policy=%s: inline must not set reasoning_effort, got body=%s", p, out)
		}
	}
}

func TestPolicy_VLLMInlineOffDisables(t *testing.T) {
	a := newTestArbiterHF()
	in := []byte(`{"model":"Llama-3.1-8B-Instruct-NVFP4","messages":[]}`)
	// Inline + off injects the disable shape. The policy is set
	// explicitly by the user (typically via the picker's `::nothink`
	// suffix), so we honour it without a modelDisableOK gate.
	out := a.applyReasoningPolicy("vllm", "/v1/chat/completions", "Llama-3.1-8B-Instruct-NVFP4", "off", in)
	setT, valT := readEnableThinking(t, out)
	if !setT || valT {
		t.Fatalf("inline+off: expected enable_thinking=false, got set=%v val=%v body=%s", setT, valT, out)
	}
	setE, valE := bodyStringField(t, out, "reasoning_effort")
	if !setE || valE != "none" {
		t.Fatalf("inline+off: expected reasoning_effort=none, got set=%v val=%q body=%s", setE, valE, out)
	}
}

func TestPolicy_VLLMClientExtraBodyWins(t *testing.T) {
	a := newTestArbiterHF()
	in := []byte(`{"model":"Qwen3.5-9B-NVFP4","messages":[],"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}`)
	out := a.applyReasoningPolicy("vllm", "/v1/chat/completions", "Qwen3.5-9B-NVFP4", "auto", in)
	setT, valT := readEnableThinking(t, out)
	if !setT || valT {
		t.Fatalf("client enable_thinking=false must survive, got set=%v val=%v body=%s", setT, valT, out)
	}
}

func TestPolicy_VLLMClientReasoningEffortWins(t *testing.T) {
	a := newTestArbiterHF()
	in := []byte(`{"model":"Qwen3.5-9B-NVFP4","messages":[],"reasoning_effort":"low"}`)
	out := a.applyReasoningPolicy("vllm", "/v1/chat/completions", "Qwen3.5-9B-NVFP4", "high", in)
	_, effort := bodyStringField(t, out, "reasoning_effort")
	if effort != "low" {
		t.Fatalf("client reasoning_effort=low must survive, got %q body=%s", effort, out)
	}
}

func TestPolicy_VLLMNonChatPathNoop(t *testing.T) {
	a := newTestArbiterHF()
	in := []byte(`{"model":"Qwen3.5-9B-NVFP4"}`)
	for _, p := range []string{"/v1/embeddings", "/health", "/v1/completions"} {
		out := a.applyReasoningPolicy("vllm", p, "Qwen3.5-9B-NVFP4", "auto", in)
		if string(out) != string(in) {
			t.Fatalf("path=%s: vllm non-chat path must noop, got %s", p, out)
		}
	}
}

// SGLang variants — same semantics, different surface fields.

func TestPolicy_SGLangStructuredEnable(t *testing.T) {
	a := newTestArbiterHF()
	in := []byte(`{"model":"Qwen3.5-9B-NVFP4","messages":[]}`)
	out := a.applyReasoningPolicy("sglang", "/v1/chat/completions", "Qwen3.5-9B-NVFP4", "auto", in)
	setSR, valSR := bodyBoolField(t, out, "separate_reasoning")
	if !setSR || !valSR {
		t.Fatalf("expected separate_reasoning=true, got set=%v val=%v body=%s", setSR, valSR, out)
	}
	setT, valT := readEnableThinking(t, out)
	if !setT || !valT {
		t.Fatalf("expected enable_thinking=true, got set=%v val=%v body=%s", setT, valT, out)
	}
}

func TestPolicy_SGLangOffDisableVerified(t *testing.T) {
	a := newTestArbiterHF()
	in := []byte(`{"model":"Qwen3.5-9B-NVFP4","messages":[]}`)
	out := a.applyReasoningPolicy("sglang", "/v1/chat/completions", "Qwen3.5-9B-NVFP4", "off", in)
	setSR, valSR := bodyBoolField(t, out, "separate_reasoning")
	if !setSR || valSR {
		t.Fatalf("expected separate_reasoning=false, got set=%v val=%v body=%s", setSR, valSR, out)
	}
	setT, valT := readEnableThinking(t, out)
	if !setT || valT {
		t.Fatalf("expected enable_thinking=false, got set=%v val=%v body=%s", setT, valT, out)
	}
}

func TestPolicy_SGLangOffWithoutDisableVerifiedNoop(t *testing.T) {
	a := newTestArbiterHF()
	in := []byte(`{"model":"Qwen3-14B-NVFP4","messages":[]}`)
	out := a.applyReasoningPolicy("sglang", "/v1/chat/completions", "Qwen3-14B-NVFP4", "off", in)
	if string(out) != string(in) {
		t.Fatalf("off without disable_verified must not modify body, got %s", out)
	}
}

func TestPolicy_SGLangInlineNonOffNoop(t *testing.T) {
	a := newTestArbiterHF()
	in := []byte(`{"model":"Llama-3.1-8B-Instruct-NVFP4","messages":[]}`)
	// Non-off policies: inline + auto/L/M/H must not inject anything.
	for _, p := range []string{"auto", "low", "medium", "high"} {
		out := a.applyReasoningPolicy("sglang", "/v1/chat/completions", "Llama-3.1-8B-Instruct-NVFP4", p, in)
		if setSR, _ := bodyBoolField(t, out, "separate_reasoning"); setSR {
			t.Fatalf("policy=%s: inline must not set separate_reasoning, got body=%s", p, out)
		}
		if setT, _ := readEnableThinking(t, out); setT {
			t.Fatalf("policy=%s: inline must not set enable_thinking, got body=%s", p, out)
		}
	}
}

func TestPolicy_SGLangInlineOffDisables(t *testing.T) {
	a := newTestArbiterHF()
	in := []byte(`{"model":"Llama-3.1-8B-Instruct-NVFP4","messages":[]}`)
	// Inline + off: explicit user opt-out injects the disable shape on
	// SGLang too. No modelDisableOK gate — the suffix IS the opt-in.
	out := a.applyReasoningPolicy("sglang", "/v1/chat/completions", "Llama-3.1-8B-Instruct-NVFP4", "off", in)
	setSR, valSR := bodyBoolField(t, out, "separate_reasoning")
	if !setSR || valSR {
		t.Fatalf("inline+off: expected separate_reasoning=false, got set=%v val=%v body=%s", setSR, valSR, out)
	}
	setT, valT := readEnableThinking(t, out)
	if !setT || valT {
		t.Fatalf("inline+off: expected enable_thinking=false, got set=%v val=%v body=%s", setT, valT, out)
	}
}

func TestPolicy_SGLangClientSeparateReasoningWins(t *testing.T) {
	a := newTestArbiterHF()
	in := []byte(`{"model":"Qwen3.5-9B-NVFP4","messages":[],"separate_reasoning":false}`)
	out := a.applyReasoningPolicy("sglang", "/v1/chat/completions", "Qwen3.5-9B-NVFP4", "auto", in)
	_, valSR := bodyBoolField(t, out, "separate_reasoning")
	if valSR {
		t.Fatalf("client separate_reasoning=false must survive, got body=%s", out)
	}
}

// ── setNestedJSONFieldIfAbsent ───────────────────────────────────────────

func TestSetNestedJSONFieldIfAbsent_CreatesPath(t *testing.T) {
	in := []byte(`{"model":"x"}`)
	out := setNestedJSONFieldIfAbsent(in,
		[]string{"extra_body", "chat_template_kwargs", "enable_thinking"}, true)
	setT, valT := readEnableThinking(t, out)
	if !setT || !valT {
		t.Fatalf("expected enable_thinking=true, got set=%v val=%v body=%s", setT, valT, out)
	}
}

func TestSetNestedJSONFieldIfAbsent_PreservesExistingLeaf(t *testing.T) {
	in := []byte(`{"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}`)
	out := setNestedJSONFieldIfAbsent(in,
		[]string{"extra_body", "chat_template_kwargs", "enable_thinking"}, true)
	_, valT := readEnableThinking(t, out)
	if valT {
		t.Fatalf("existing leaf must not be overwritten, got body=%s", out)
	}
}

func TestSetNestedJSONFieldIfAbsent_PreservesSiblings(t *testing.T) {
	in := []byte(`{"extra_body":{"cache_control":{"ttl":"1h"}}}`)
	out := setNestedJSONFieldIfAbsent(in,
		[]string{"extra_body", "chat_template_kwargs", "enable_thinking"}, true)
	var root map[string]any
	if err := json.Unmarshal(out, &root); err != nil {
		t.Fatalf("body not valid JSON: %v", err)
	}
	eb, ok := root["extra_body"].(map[string]any)
	if !ok {
		t.Fatalf("extra_body missing or not object: %v", root["extra_body"])
	}
	if cc, ok := eb["cache_control"].(map[string]any); !ok || cc["ttl"] != "1h" {
		t.Fatalf("client cache_control must survive, got %v", eb["cache_control"])
	}
	setT, valT := readEnableThinking(t, out)
	if !setT || !valT {
		t.Fatalf("expected enable_thinking=true alongside existing siblings, body=%s", out)
	}
}

func TestSetNestedJSONFieldIfAbsent_RefusesNonObjectIntermediate(t *testing.T) {
	// Client supplied extra_body as a string — refuse to rewrite.
	in := []byte(`{"extra_body":"not-an-object"}`)
	out := setNestedJSONFieldIfAbsent(in,
		[]string{"extra_body", "chat_template_kwargs", "enable_thinking"}, true)
	if string(out) != string(in) {
		t.Fatalf("non-object intermediate must produce passthrough, got %s", out)
	}
}

func TestSetNestedJSONFieldIfAbsent_InvalidJSONPassthrough(t *testing.T) {
	in := []byte(`not-json`)
	out := setNestedJSONFieldIfAbsent(in,
		[]string{"extra_body", "chat_template_kwargs", "enable_thinking"}, true)
	if string(out) != string(in) {
		t.Fatalf("invalid JSON must passthrough, got %s", out)
	}
}

// parseReasoningOverride parses the `::<token>` suffix the picker emits
// for inline-reasoning two-row picks. Cover the recognised tokens plus
// the negative cases (no suffix, unknown token).
func TestParseReasoningOverride_TokenMapping(t *testing.T) {
	cases := []struct {
		in       string
		wantName string
		wantPol  string
	}{
		{"qwen3:14b-q4_K_M", "qwen3:14b-q4_K_M", ""},
		{"qwen3:14b-q4_K_M::nothink", "qwen3:14b-q4_K_M", "off"},
		{"qwen3:14b-q4_K_M::think", "qwen3:14b-q4_K_M", "auto"},
		{"qwen3:14b-q4_K_M::off", "qwen3:14b-q4_K_M", "off"},
		{"qwen3:14b-q4_K_M::auto", "qwen3:14b-q4_K_M", "auto"},
		{"qwen3:14b-q4_K_M::low", "qwen3:14b-q4_K_M", "low"},
		{"qwen3:14b-q4_K_M::medium", "qwen3:14b-q4_K_M", "medium"},
		{"qwen3:14b-q4_K_M::high", "qwen3:14b-q4_K_M", "high"},
		// Unknown tokens leave the name untouched.
		{"qwen3:14b-q4_K_M::garbage", "qwen3:14b-q4_K_M::garbage", ""},
		// Embedded `::` in the name without a recognised token after
		// the LAST occurrence is also a noop.
		{"namespace::model", "namespace::model", ""},
	}
	for _, tc := range cases {
		gotName, gotPol := parseReasoningOverride(tc.in)
		if gotName != tc.wantName || gotPol != tc.wantPol {
			t.Errorf("parseReasoningOverride(%q) = (%q, %q), want (%q, %q)",
				tc.in, gotName, gotPol, tc.wantName, tc.wantPol)
		}
	}
}

// The picker convention is `<name>::<reasoning>@<ctx>`. The router
// strips them in the order ctx-first, reasoning-second so both end
// up in the right hands.
func TestParseSuffixes_OrderingMatchesPickerEmits(t *testing.T) {
	in := "Qwen3-14B-NVFP4::nothink@65536"
	ctxStripped, ctxOverride := parseCtxOverride(in)
	if ctxStripped != "Qwen3-14B-NVFP4::nothink" || ctxOverride != 65536 {
		t.Fatalf("after parseCtxOverride: got (%q, %d), want (%q, 65536)",
			ctxStripped, ctxOverride, "Qwen3-14B-NVFP4::nothink")
	}
	cleanName, reasoning := parseReasoningOverride(ctxStripped)
	if cleanName != "Qwen3-14B-NVFP4" || reasoning != "off" {
		t.Fatalf("after parseReasoningOverride: got (%q, %q), want (%q, %q)",
			cleanName, reasoning, "Qwen3-14B-NVFP4", "off")
	}
}

// parseMTPOverride parses the `::mtp` / `::nomtp` suffix the picker
// emits for catalog rows with an `mtp:` block. Cover the recognised
// tokens plus the negative cases (no suffix, unknown token) and the
// embedded `::` defensive case mirroring TestParseReasoningOverride.
func TestParseMTPOverride_TokenMapping(t *testing.T) {
	cases := []struct {
		in       string
		wantName string
		wantPol  string
	}{
		{"Gemma-4-26B-A4B-NVFP4", "Gemma-4-26B-A4B-NVFP4", ""},
		{"Gemma-4-26B-A4B-NVFP4::mtp", "Gemma-4-26B-A4B-NVFP4", "on"},
		{"Gemma-4-26B-A4B-NVFP4::nomtp", "Gemma-4-26B-A4B-NVFP4", "off"},
		{"Gemma-4-26B-A4B-NVFP4::MTP", "Gemma-4-26B-A4B-NVFP4", "on"},
		{"Gemma-4-26B-A4B-NVFP4::NoMTP", "Gemma-4-26B-A4B-NVFP4", "off"},
		// Unknown tokens leave the name untouched -- they must NOT be
		// consumed (parseReasoningOverride may still recognise them).
		{"Gemma-4-26B-A4B-NVFP4::nothink", "Gemma-4-26B-A4B-NVFP4::nothink", ""},
		{"Gemma-4-26B-A4B-NVFP4::garbage", "Gemma-4-26B-A4B-NVFP4::garbage", ""},
		// Embedded `::` in the name without a recognised token after
		// the LAST occurrence is also a noop.
		{"namespace::model", "namespace::model", ""},
	}
	for _, tc := range cases {
		gotName, gotPol := parseMTPOverride(tc.in)
		if gotName != tc.wantName || gotPol != tc.wantPol {
			t.Errorf("parseMTPOverride(%q) = (%q, %q), want (%q, %q)",
				tc.in, gotName, gotPol, tc.wantName, tc.wantPol)
		}
	}
}

// The picker's canonical emit order is `<name>::<reasoning>::<mtp>@<ctx>`.
// The router peels right-to-left: ctx first, mtp second, reasoning third.
// Exercise the full three-way chain end-to-end so Phase-1 parser additions
// don't break the existing reasoning/ctx flow.
func TestParseSuffixes_ThreeWayChainOrdering(t *testing.T) {
	in := "Gemma-4-26B-A4B-NVFP4::nothink::mtp@131072"
	// Step 1: strip @<ctx> from the right.
	ctxStripped, ctxOverride := parseCtxOverride(in)
	if ctxStripped != "Gemma-4-26B-A4B-NVFP4::nothink::mtp" || ctxOverride != 131072 {
		t.Fatalf("after parseCtxOverride: got (%q, %d), want (%q, 131072)",
			ctxStripped, ctxOverride, "Gemma-4-26B-A4B-NVFP4::nothink::mtp")
	}
	// Step 2: strip ::<mtp> from the right (now the LAST `::<token>`).
	mtpStripped, mtpOverride := parseMTPOverride(ctxStripped)
	if mtpStripped != "Gemma-4-26B-A4B-NVFP4::nothink" || mtpOverride != "on" {
		t.Fatalf("after parseMTPOverride: got (%q, %q), want (%q, %q)",
			mtpStripped, mtpOverride, "Gemma-4-26B-A4B-NVFP4::nothink", "on")
	}
	// Step 3: strip ::<reasoning>.
	cleanName, reasoning := parseReasoningOverride(mtpStripped)
	if cleanName != "Gemma-4-26B-A4B-NVFP4" || reasoning != "off" {
		t.Fatalf("after parseReasoningOverride: got (%q, %q), want (%q, %q)",
			cleanName, reasoning, "Gemma-4-26B-A4B-NVFP4", "off")
	}
}

// MTP-only path -- a model name without a reasoning override, just
// `::mtp@<ctx>`. Asserts that parseReasoningOverride doesn't accidentally
// consume the MTP token when it's the only `::` segment present.
func TestParseSuffixes_MTPOnlyPath(t *testing.T) {
	in := "Gemma-4-26B-A4B-NVFP4::mtp@65536"
	ctxStripped, ctxOverride := parseCtxOverride(in)
	if ctxStripped != "Gemma-4-26B-A4B-NVFP4::mtp" || ctxOverride != 65536 {
		t.Fatalf("after parseCtxOverride: got (%q, %d), want (%q, 65536)",
			ctxStripped, ctxOverride, "Gemma-4-26B-A4B-NVFP4::mtp")
	}
	mtpStripped, mtpOverride := parseMTPOverride(ctxStripped)
	if mtpStripped != "Gemma-4-26B-A4B-NVFP4" || mtpOverride != "on" {
		t.Fatalf("after parseMTPOverride: got (%q, %q), want (%q, %q)",
			mtpStripped, mtpOverride, "Gemma-4-26B-A4B-NVFP4", "on")
	}
	// parseReasoningOverride on the clean name must be a noop.
	cleanName, reasoning := parseReasoningOverride(mtpStripped)
	if cleanName != "Gemma-4-26B-A4B-NVFP4" || reasoning != "" {
		t.Fatalf("after parseReasoningOverride: got (%q, %q), want (%q, \"\")",
			cleanName, reasoning, "Gemma-4-26B-A4B-NVFP4")
	}
}

// maybeStripTools strips `tools` and `tool_choice` for vLLM/SGLang
// when the model has no probe-verified tool parser (engine launched
// without --enable-auto-tool-choice / --tool-call-parser).
func TestMaybeStripTools_VLLMUnverifiedDropsTools(t *testing.T) {
	a := newTestArbiterHF()
	in := []byte(`{"model":"Qwen3-14B-NVFP4","messages":[],"tools":[{"type":"function"}],"tool_choice":"auto"}`)
	out := a.maybeStripTools("vllm", "Qwen3-14B-NVFP4", in)
	var doc map[string]any
	if err := json.Unmarshal(out, &doc); err != nil {
		t.Fatalf("rewritten body not valid JSON: %v body=%s", err, out)
	}
	if _, ok := doc["tools"]; ok {
		t.Errorf("tools should have been stripped, body=%s", out)
	}
	if _, ok := doc["tool_choice"]; ok {
		t.Errorf("tool_choice should have been stripped, body=%s", out)
	}
}

func TestMaybeStripTools_VLLMVerifiedKeepsTools(t *testing.T) {
	a := newTestArbiterHF()
	in := []byte(`{"model":"Qwen3.5-9B-NVFP4","messages":[],"tools":[{"type":"function"}],"tool_choice":"auto"}`)
	out := a.maybeStripTools("vllm", "Qwen3.5-9B-NVFP4", in)
	if string(out) != string(in) {
		t.Fatalf("verified parser must preserve tools, got %s", out)
	}
}

func TestMaybeStripTools_OllamaPassthrough(t *testing.T) {
	a := newTestArbiterHF()
	in := []byte(`{"model":"some-ollama","messages":[],"tools":[{"type":"function"}]}`)
	out := a.maybeStripTools("ollama", "some-ollama", in)
	if string(out) != string(in) {
		t.Fatalf("ollama path must passthrough; tools negotiated per request, got %s", out)
	}
}

func TestMaybeStripTools_NoToolsNoop(t *testing.T) {
	a := newTestArbiterHF()
	in := []byte(`{"model":"Qwen3-14B-NVFP4","messages":[]}`)
	out := a.maybeStripTools("vllm", "Qwen3-14B-NVFP4", in)
	if string(out) != string(in) {
		t.Fatalf("body without tools must passthrough byte-for-byte, got %s", out)
	}
}

// Regression: openai/gpt-oss-20b is registered on both vLLM and SGLang with
// completely different parser names (vLLM: openai_gptoss/openai;
// SGLang: gpt-oss/gpt-oss). With a flat map[string]string the
// second-loaded backend's value would overwrite the first's, so launching
// vLLM would receive SGLang's "gpt-oss" tool parser and crash with
// `KeyError: invalid tool call parser: gpt-oss`. This test pins the
// backend-keyed lookup behavior.
func TestMaybeStripTools_BackendIsolation_NoCrossBackendOverwrite(t *testing.T) {
	a := &arbiter{
		modelToolParser: map[string]map[string]string{
			"vllm":   {"gpt-oss-20b": "openai"},
			"sglang": {"gpt-oss-20b": "gpt-oss"},
		},
	}
	body := []byte(`{"model":"gpt-oss-20b","messages":[],"tools":[{"type":"function"}],"tool_choice":"auto"}`)

	if out := a.maybeStripTools("vllm", "gpt-oss-20b", body); string(out) != string(body) {
		t.Fatalf("vllm: parser is verified for this backend, tools must survive; got %s", out)
	}
	if out := a.maybeStripTools("sglang", "gpt-oss-20b", body); string(out) != string(body) {
		t.Fatalf("sglang: parser is verified for this backend, tools must survive; got %s", out)
	}

	// Conversely: a model verified on vLLM only must have its tools
	// stripped when the request hits SGLang (otherwise the engine
	// rejects it because it was launched without --tool-call-parser).
	a2 := &arbiter{
		modelToolParser: map[string]map[string]string{
			"vllm": {"vllm-only-model": "hermes"},
		},
	}
	body2 := []byte(`{"model":"vllm-only-model","messages":[],"tools":[{"type":"function"}]}`)
	out := a2.maybeStripTools("sglang", "vllm-only-model", body2)
	var doc map[string]any
	if err := json.Unmarshal(out, &doc); err != nil {
		t.Fatalf("rewritten body invalid JSON: %v body=%s", err, out)
	}
	if _, ok := doc["tools"]; ok {
		t.Errorf("sglang request must NOT inherit vLLM's parser; tools should be stripped; body=%s", out)
	}
}
