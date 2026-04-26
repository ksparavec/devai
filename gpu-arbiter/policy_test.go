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

func bodyHasThink(t *testing.T, body []byte) (set bool, val bool) {
	t.Helper()
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("body not valid JSON: %v", err)
	}
	r, ok := raw["think"]
	if !ok {
		return false, false
	}
	var b bool
	if err := json.Unmarshal(r, &b); err != nil {
		t.Fatalf("think not bool: %v", err)
	}
	return true, b
}

func TestPolicy_AutoStructuredEnables(t *testing.T) {
	a := newTestArbiter()
	in := []byte(`{"model":"qwen3.5:9b","messages":[{"role":"user","content":"hi"}]}`)
	out := a.applyOllamaPolicy("qwen3.5:9b", "auto", in)
	set, val := bodyHasThink(t, out)
	if !set || !val {
		t.Fatalf("expected think:true, got set=%v val=%v body=%s", set, val, out)
	}
}

func TestPolicy_OffStructuredVerifiedDisables(t *testing.T) {
	a := newTestArbiter()
	in := []byte(`{"model":"qwen3.5:9b","messages":[]}`)
	out := a.applyOllamaPolicy("qwen3.5:9b", "off", in)
	set, val := bodyHasThink(t, out)
	if !set || val {
		t.Fatalf("expected think:false, got set=%v val=%v body=%s", set, val, out)
	}
}

func TestPolicy_UnsupportedNeverSets(t *testing.T) {
	a := newTestArbiter()
	in := []byte(`{"model":"gemma4:e4b-it-bf16","messages":[]}`)
	for _, p := range []string{"auto", "off", "low", "medium", "high"} {
		out := a.applyOllamaPolicy("gemma4:e4b-it-bf16", p, in)
		set, _ := bodyHasThink(t, out)
		if set {
			t.Fatalf("policy=%s: expected no think field for unsupported, got body=%s", p, out)
		}
	}
}

func TestPolicy_ErrorNeverSets(t *testing.T) {
	a := newTestArbiter()
	in := []byte(`{"model":"llama3.2:latest","messages":[]}`)
	out := a.applyOllamaPolicy("llama3.2:latest", "auto", in)
	set, _ := bodyHasThink(t, out)
	if set {
		t.Fatalf("expected no think for error capability")
	}
}

func TestPolicy_ClientThinkWins(t *testing.T) {
	a := newTestArbiter()
	in := []byte(`{"model":"qwen3.5:9b","think":false,"messages":[]}`)
	out := a.applyOllamaPolicy("qwen3.5:9b", "auto", in)
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
		out := a.applyOllamaPolicy("qwen3.5:9b", p, in)
		set, val := bodyHasThink(t, out)
		if !set || !val {
			t.Fatalf("policy=%s: expected think:true (degenerate for ollama)", p)
		}
	}
}

func TestPolicy_NonOllamaBackendIsNoOp(t *testing.T) {
	a := newTestArbiter()
	in := []byte(`{"model":"some-vllm-model","messages":[]}`)
	out := a.applyReasoningPolicy("vllm", "some-vllm-model", "auto", in)
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
