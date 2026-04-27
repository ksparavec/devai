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

func TestPolicy_NonOllamaBackendIsNoOp(t *testing.T) {
	a := newTestArbiter()
	in := []byte(`{"model":"some-vllm-model","messages":[]}`)
	out := a.applyReasoningPolicy("vllm", "/v1/chat/completions", "some-vllm-model", "auto", in)
	if string(out) != string(in) {
		t.Fatalf("vllm path should pass body through unchanged; got %s", out)
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
