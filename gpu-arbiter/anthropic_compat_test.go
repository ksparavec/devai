package main

import (
	"encoding/json"
	"strings"
	"testing"
)

// capturedClaudeCodeBody is the real wire shape, captured 2026-07-27 by
// standing a logging server in place of the router and running
// `claude -p "hi"` from the lab image (Claude Code v2.1.220). The real
// body is ~183 KB across 25 tools; the text is truncated and the
// device/session identifiers are replaced with obvious placeholders,
// but every FIELD and the message/system STRUCTURE are verbatim --
// which is the part the engines' schema validation acts on.
//
// The offending element is messages[1]: role "system", inside
// messages[], alongside a perfectly correct top-level system.
const capturedClaudeCodeBody = `{
  "model": "Gemma-4-26B-A4B-it-NVFP4@262144",
  "max_tokens": 32000,
  "stream": true,
  "system": [
    {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.220.04c"},
    {"type": "text", "text": "You are a Claude agent, built on Anthropic's Claude Agent SDK."},
    {"type": "text", "text": "You are an interactive agent that helps users with software engineering."}
  ],
  "messages": [
    {"role": "user", "content": [
      {"type": "text", "text": "<system-reminder>ambient context</system-reminder>"},
      {"type": "text", "text": "hi"}
    ]},
    {"role": "system", "content": [
      {"type": "text", "text": "Available agent types for the Agent tool:"}
    ]}
  ],
  "metadata": {"user_id": "{\"device_id\":\"REDACTED\",\"session_id\":\"REDACTED\"}"},
  "thinking": {"display": "omitted", "type": "adaptive"},
  "context_management": {"edits": [{"keep": "all", "type": "clear_thinking_20251015"}]},
  "output_config": {"effort": "high"},
  "tools": [
    {"name": "Agent", "description": "Launch a new agent",
     "input_schema": {"type": "object", "properties": {}, "required": []}}
  ]
}`

// decode is a small helper so assertions read as data, not as plumbing.
func decode(t *testing.T, b []byte) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("result is not valid JSON: %v", err)
	}
	return m
}

func systemTexts(t *testing.T, m map[string]any) []string {
	t.Helper()
	raw, ok := m["system"].([]any)
	if !ok {
		t.Fatalf("system is not a block array: %T", m["system"])
	}
	out := make([]string, 0, len(raw))
	for _, b := range raw {
		blk, ok := b.(map[string]any)
		if !ok {
			t.Fatalf("system block is not an object: %T", b)
		}
		out = append(out, blk["text"].(string))
	}
	return out
}

func roles(t *testing.T, m map[string]any) []string {
	t.Helper()
	raw, _ := m["messages"].([]any)
	out := make([]string, 0, len(raw))
	for _, x := range raw {
		out = append(out, x.(map[string]any)["role"].(string))
	}
	return out
}

// The headline case. Verified end-to-end against a live vLLM engine:
// this body as-is returns 400 with locator ('body','messages',1,'role');
// folded, with every beta field still present, it returns 200.
func TestNormaliseAnthropic_CapturedClaudeCodeBody(t *testing.T) {
	out, moved := normaliseAnthropicMessages([]byte(capturedClaudeCodeBody))
	if moved != 1 {
		t.Fatalf("want 1 message folded, got %d", moved)
	}
	m := decode(t, out)

	if got := roles(t, m); len(got) != 1 || got[0] != "user" {
		t.Fatalf("messages must retain only the user turn, got %v", got)
	}
	texts := systemTexts(t, m)
	if len(texts) != 4 {
		t.Fatalf("want 3 original + 1 folded system block, got %d: %v", len(texts), texts)
	}
	// Order preserved, and the folded content lands last.
	if !strings.HasPrefix(texts[0], "x-anthropic-billing-header") {
		t.Fatalf("original system order not preserved: %q", texts[0])
	}
	if texts[3] != "Available agent types for the Agent tool:" {
		t.Fatalf("folded block missing or reordered: %q", texts[3])
	}
	// Beta fields must survive untouched -- the replay proved they are
	// not the blocker, so stripping them would be an unrequested
	// behaviour change.
	for _, k := range []string{"context_management", "output_config", "thinking", "metadata", "tools", "max_tokens", "stream", "model"} {
		if _, ok := m[k]; !ok {
			t.Fatalf("field %q was dropped by the rewrite", k)
		}
	}
}

func TestNormaliseAnthropic_BareStringSystem(t *testing.T) {
	in := `{"system":"be brief","messages":[
		{"role":"user","content":[{"type":"text","text":"hi"}]},
		{"role":"system","content":[{"type":"text","text":"extra"}]}]}`
	out, moved := normaliseAnthropicMessages([]byte(in))
	if moved != 1 {
		t.Fatalf("want 1 moved, got %d", moved)
	}
	texts := systemTexts(t, decode(t, out))
	want := []string{"be brief", "extra"}
	if len(texts) != 2 || texts[0] != want[0] || texts[1] != want[1] {
		t.Fatalf("a bare-string system must be promoted to blocks then appended: %v", texts)
	}
}

func TestNormaliseAnthropic_AbsentSystem(t *testing.T) {
	in := `{"messages":[
		{"role":"user","content":[{"type":"text","text":"hi"}]},
		{"role":"system","content":[{"type":"text","text":"only"}]}]}`
	out, moved := normaliseAnthropicMessages([]byte(in))
	if moved != 1 {
		t.Fatalf("want 1 moved, got %d", moved)
	}
	texts := systemTexts(t, decode(t, out))
	if len(texts) != 1 || texts[0] != "only" {
		t.Fatalf("want system created with the folded block, got %v", texts)
	}
}

func TestNormaliseAnthropic_MultipleStrayMessagesPreserveOrder(t *testing.T) {
	in := `{"system":[{"type":"text","text":"s0"}],"messages":[
		{"role":"system","content":[{"type":"text","text":"a"}]},
		{"role":"user","content":[{"type":"text","text":"hi"}]},
		{"role":"developer","content":[{"type":"text","text":"b"}]},
		{"role":"assistant","content":[{"type":"text","text":"yes"}]},
		{"role":"tool","content":[{"type":"text","text":"c"}]}]}`
	out, moved := normaliseAnthropicMessages([]byte(in))
	if moved != 3 {
		t.Fatalf("want 3 moved, got %d", moved)
	}
	m := decode(t, out)
	if got := roles(t, m); len(got) != 2 || got[0] != "user" || got[1] != "assistant" {
		t.Fatalf("conversation turns must survive in order, got %v", got)
	}
	texts := systemTexts(t, m)
	want := []string{"s0", "a", "b", "c"}
	for i := range want {
		if i >= len(texts) || texts[i] != want[i] {
			t.Fatalf("stray order not preserved: want %v, got %v", want, texts)
		}
	}
}

// The warm path. Nothing to move must mean nothing touched -- not
// "re-serialised to something equivalent" -- so key order, spacing and
// number formatting are all guaranteed unchanged.
func TestNormaliseAnthropic_NothingToMoveIsByteIdentical(t *testing.T) {
	in := `{"model":"m","system":"s","messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]}`
	out, moved := normaliseAnthropicMessages([]byte(in))
	if moved != 0 {
		t.Fatalf("want 0 moved, got %d", moved)
	}
	if string(out) != in {
		t.Fatalf("body must be byte-identical when there is nothing to fold:\n got %s\nwant %s", out, in)
	}
}

func TestNormaliseAnthropic_MalformedJSONPassesThrough(t *testing.T) {
	for _, in := range []string{
		`not json at all`,
		`{"messages": "not an array"}`,
		`{"messages":[{"role":"system","content":42}]}`, // uninterpretable stray
		`{"system":42,"messages":[{"role":"system","content":[]}]}`,
		`{}`,
		``,
	} {
		out, moved := normaliseAnthropicMessages([]byte(in))
		if moved != 0 {
			t.Fatalf("input %q: want 0 moved, got %d", in, moved)
		}
		if string(out) != in {
			t.Fatalf("input %q must pass through unchanged, got %s", in, out)
		}
	}
}

// A stray message whose content is a bare string, not a block array.
func TestNormaliseAnthropic_StringContentStrayIsPromoted(t *testing.T) {
	in := `{"messages":[{"role":"user","content":"hi"},{"role":"system","content":"plain"}]}`
	out, moved := normaliseAnthropicMessages([]byte(in))
	if moved != 1 {
		t.Fatalf("want 1 moved, got %d", moved)
	}
	texts := systemTexts(t, decode(t, out))
	if len(texts) != 1 || texts[0] != "plain" {
		t.Fatalf("bare-string content must become a text block, got %v", texts)
	}
}

// Numbers must round-trip exactly. Without json.Number a large integer
// anywhere in the body -- including inside a tool's JSON-Schema -- comes
// back in exponent form and the engine sees a different request.
func TestNormaliseAnthropic_LargeNumbersSurviveExactly(t *testing.T) {
	in := `{"max_tokens":9007199254740993,"tools":[{"input_schema":{"maximum":12345678901234567890}}],` +
		`"messages":[{"role":"user","content":"hi"},{"role":"system","content":"x"}]}`
	out, moved := normaliseAnthropicMessages([]byte(in))
	if moved != 1 {
		t.Fatalf("want 1 moved, got %d", moved)
	}
	for _, want := range []string{"9007199254740993", "12345678901234567890"} {
		if !strings.Contains(string(out), want) {
			t.Fatalf("number %s was mangled: %s", want, out)
		}
	}
}

// --- Backend and path gating ---

func TestMaybeNormaliseAnthropic_Gating(t *testing.T) {
	a := &arbiter{}
	body := []byte(capturedClaudeCodeBody)

	tests := []struct {
		name     string
		backend  string
		path     string
		rewrites bool
	}{
		{"vllm messages", "vllm", "/v1/messages", true},
		{"sglang messages", "sglang", "/v1/messages", true},
		// Ollama's shim is verified tolerant of this exact shape.
		// Rewriting a path that already works is unnecessary risk.
		{"ollama messages untouched", "ollama", "/v1/messages", false},
		{"vllm chat completions untouched", "vllm", "/v1/chat/completions", false},
		{"vllm native ollama path untouched", "vllm", "/api/chat", false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			out := a.maybeNormaliseAnthropic(tc.backend, tc.path, body)
			changed := string(out) != string(body)
			if changed != tc.rewrites {
				t.Fatalf("%s/%s: rewrote=%v, want %v", tc.backend, tc.path, changed, tc.rewrites)
			}
		})
	}
}
