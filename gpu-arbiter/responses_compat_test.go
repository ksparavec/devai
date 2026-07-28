package main

import (
	"encoding/json"
	"testing"
)

// Every expectation here was measured against the live vLLM v0.22.1
// engine serving gpt-oss-20b, not read from a spec -- because the wrong
// shapes are ACCEPTED with HTTP 200 and silently ignored.

func responsesArbiter() *arbiter {
	return &arbiter{
		modelCapability: map[string]map[string]string{
			"vllm":   {"structured-model": CapStructured, "inline-model": CapInline},
			"sglang": {"structured-model": CapStructured},
		},
		modelDisableOK: map[string]map[string]bool{
			"vllm": {"structured-model": true},
		},
		defaultPolicy: "auto",
	}
}

func reasoningEffortOf(t *testing.T, body []byte) (string, bool) {
	t.Helper()
	var doc map[string]json.RawMessage
	if err := json.Unmarshal(body, &doc); err != nil {
		t.Fatalf("body not valid JSON: %v", err)
	}
	raw, ok := doc["reasoning"]
	if !ok {
		return "", false
	}
	var r struct {
		Effort string `json:"effort"`
	}
	if err := json.Unmarshal(raw, &r); err != nil {
		t.Fatalf("reasoning not an object: %v", err)
	}
	return r.Effort, true
}

func TestIsResponsesPath(t *testing.T) {
	for _, p := range []string{"/v1/responses", "/v1/responses/",
		"/v1/responses/resp_abc/cancel"} {
		if !isResponsesPath(p) {
			t.Errorf("%q must be a responses path", p)
		}
	}
	for _, p := range []string{"/v1/chat/completions", "/v1/messages",
		"/api/chat", "/v1/models", "/v1/responsesX"} {
		if isResponsesPath(p) {
			t.Errorf("%q must NOT be a responses path", p)
		}
	}
}

func TestResponsesPolicy_ExplicitEffortIsInjected(t *testing.T) {
	a := responsesArbiter()
	in := []byte(`{"model":"structured-model","input":"hi"}`)
	for _, p := range []string{"low", "medium", "high"} {
		out := a.applyReasoningPolicy("vllm", "/v1/responses",
			"structured-model", p, in)
		got, ok := reasoningEffortOf(t, out)
		if !ok || got != p {
			t.Fatalf("policy=%s: want reasoning.effort=%s, got %q (set=%v)",
				p, p, got, ok)
		}
	}
}

// The Chat Completions shape must NOT be what we emit here: it returns
// 200 and is ignored (reasoning_tokens 282 vs a 37-token effect from the
// correct shape).
func TestResponsesPolicy_DoesNotEmitTheChatShape(t *testing.T) {
	a := responsesArbiter()
	in := []byte(`{"model":"structured-model","input":"hi"}`)
	out := a.applyReasoningPolicy("vllm", "/v1/responses",
		"structured-model", "low", in)
	var doc map[string]any
	if err := json.Unmarshal(out, &doc); err != nil {
		t.Fatal(err)
	}
	if _, bad := doc["reasoning_effort"]; bad {
		t.Fatal("reasoning_effort is silently ignored on /v1/responses; " +
			"emitting it is a fake fix")
	}
	if _, bad := doc["extra_body"]; bad {
		t.Fatal("extra_body has no meaning on /v1/responses")
	}
}

func TestResponsesPolicy_AutoInjectsNothing(t *testing.T) {
	// The model's own default is the right answer under `auto`;
	// inventing "medium" would override what the checkpoint ships.
	a := responsesArbiter()
	in := []byte(`{"model":"structured-model","input":"hi"}`)
	out := a.applyReasoningPolicy("vllm", "/v1/responses",
		"structured-model", "auto", in)
	if _, ok := reasoningEffortOf(t, out); ok {
		t.Fatalf("auto must not inject an effort, got %s", out)
	}
	if string(out) != string(in) {
		t.Fatalf("auto must leave the body byte-identical:\n got %s\nwant %s",
			out, in)
	}
}

func TestResponsesPolicy_OffEmitsNoneWhenVerified(t *testing.T) {
	a := responsesArbiter()
	in := []byte(`{"model":"structured-model","input":"hi"}`)
	out := a.applyReasoningPolicy("vllm", "/v1/responses",
		"structured-model", "off", in)
	got, ok := reasoningEffortOf(t, out)
	if !ok || got != "none" {
		t.Fatalf("verified disable must emit effort=none, got %q (set=%v)",
			got, ok)
	}
}

func TestResponsesPolicy_OffWithoutVerificationIsNoop(t *testing.T) {
	// Harmony models reject effort="none" with 400 and probe as
	// disable_verified=false; the gate is what keeps them out.
	a := responsesArbiter()
	a.modelDisableOK["vllm"]["structured-model"] = false
	in := []byte(`{"model":"structured-model","input":"hi"}`)
	out := a.applyReasoningPolicy("vllm", "/v1/responses",
		"structured-model", "off", in)
	if string(out) != string(in) {
		t.Fatalf("unverified disable must not inject anything, got %s", out)
	}
}

func TestResponsesPolicy_ClientReasoningWins(t *testing.T) {
	a := responsesArbiter()
	in := []byte(`{"model":"structured-model","input":"hi","reasoning":{"effort":"high"}}`)
	out := a.applyReasoningPolicy("vllm", "/v1/responses",
		"structured-model", "low", in)
	got, _ := reasoningEffortOf(t, out)
	if got != "high" {
		t.Fatalf("a client-supplied reasoning object must survive, got %q", got)
	}
}

func TestResponsesPolicy_AppliesToSGLangToo(t *testing.T) {
	// SGLang registers /v1/responses (http_server.py:1563). Not verified
	// live -- it was not loaded during this work -- but the shape is the
	// standard one and the alternative is leaving it with no policy.
	a := responsesArbiter()
	in := []byte(`{"model":"structured-model","input":"hi"}`)
	out := a.applyReasoningPolicy("sglang", "/v1/responses",
		"structured-model", "high", in)
	if got, ok := reasoningEffortOf(t, out); !ok || got != "high" {
		t.Fatalf("sglang responses: want high, got %q (set=%v)", got, ok)
	}
}

func TestResponsesPolicy_ChatPathIsUnaffected(t *testing.T) {
	// Regression: the Chat Completions surface must keep its own shape.
	a := responsesArbiter()
	in := []byte(`{"model":"structured-model","messages":[]}`)
	out := a.applyReasoningPolicy("vllm", "/v1/chat/completions",
		"structured-model", "low", in)
	var doc map[string]any
	if err := json.Unmarshal(out, &doc); err != nil {
		t.Fatal(err)
	}
	if _, ok := doc["reasoning_effort"]; !ok {
		t.Fatal("chat completions must still get reasoning_effort")
	}
	if _, bad := doc["reasoning"]; bad {
		t.Fatal("chat completions must NOT get the responses shape")
	}
}

func TestResponsesPolicy_MalformedBodyPassesThrough(t *testing.T) {
	a := responsesArbiter()
	in := []byte(`not json`)
	out := a.applyReasoningPolicy("vllm", "/v1/responses",
		"structured-model", "low", in)
	if string(out) != string(in) {
		t.Fatalf("malformed input must pass through, got %s", out)
	}
}
