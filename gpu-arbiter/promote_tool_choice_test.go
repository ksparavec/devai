package main

import (
	"encoding/json"
	"strings"
	"testing"
)

// promoteFixture builds a minimal arbiter pre-populated with the given
// tool_mode for the (backend, modelName) tuple. modelToolParser is also
// set so maybeStripTools wouldn't fire — we want maybePromoteToolChoice
// to be the only rewrite path under test.
func promoteFixture(backend, model, mode string) *arbiter {
	return &arbiter{
		modelToolParser: map[string]map[string]string{
			backend: {model: "deepseek_string"},
		},
		modelToolMode: map[string]map[string]string{
			backend: {model: mode},
		},
	}
}

// requestBody packages a chat-completion request shape into raw JSON.
func requestBody(t *testing.T, tools []map[string]any, toolChoice any) []byte {
	t.Helper()
	doc := map[string]any{
		"model":    "DeepSeek-R1-Distill-Qwen-7B",
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	}
	if tools != nil {
		t := make([]any, len(tools))
		for i, x := range tools {
			t[i] = x
		}
		doc["tools"] = t
	}
	if toolChoice != nil {
		doc["tool_choice"] = toolChoice
	}
	out, err := json.Marshal(doc)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return out
}

func toolSpec(name string) map[string]any {
	return map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":       name,
			"parameters": map[string]any{"type": "object", "properties": map[string]any{}},
		},
	}
}

// --- Single-tool / forced mode ---

func TestPromote_SingleToolAutoOnForcedRewrites(t *testing.T) {
	a := promoteFixture("vllm", "DeepSeek-R1-Distill-Qwen-7B", "forced")
	body := requestBody(t, []map[string]any{toolSpec("get_time")}, "auto")
	out, perr := a.maybePromoteToolChoice("vllm", "DeepSeek-R1-Distill-Qwen-7B", body)
	if perr != nil {
		t.Fatalf("unexpected error: %v", perr)
	}
	var doc map[string]any
	if err := json.Unmarshal(out, &doc); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	tc, _ := doc["tool_choice"].(map[string]any)
	fn, _ := tc["function"].(map[string]any)
	if got, _ := fn["name"].(string); got != "get_time" {
		t.Errorf("tool_choice not pinned: %+v", doc["tool_choice"])
	}
	if got, _ := tc["type"].(string); got != "function" {
		t.Errorf("tool_choice type = %v, want function", tc["type"])
	}
}

func TestPromote_SingleToolNoChoiceOnForcedRewrites(t *testing.T) {
	// tool_choice absent is equivalent to "auto" per OpenAI semantics.
	// Same rewrite rule applies.
	a := promoteFixture("vllm", "M", "forced")
	body := requestBody(t, []map[string]any{toolSpec("search")}, nil)
	out, perr := a.maybePromoteToolChoice("vllm", "M", body)
	if perr != nil {
		t.Fatalf("unexpected error: %v", perr)
	}
	var doc map[string]any
	_ = json.Unmarshal(out, &doc)
	tc, _ := doc["tool_choice"].(map[string]any)
	fn, _ := tc["function"].(map[string]any)
	if got, _ := fn["name"].(string); got != "search" {
		t.Errorf("expected pinned to search, got %+v", doc["tool_choice"])
	}
}

// --- Multi-tool / forced mode ---

func TestPromote_MultiToolAutoOnForcedRejects(t *testing.T) {
	a := promoteFixture("vllm", "M", "forced")
	body := requestBody(t, []map[string]any{
		toolSpec("get_time"), toolSpec("search"), toolSpec("send_email"),
	}, "auto")
	out, perr := a.maybePromoteToolChoice("vllm", "M", body)
	if perr == nil {
		t.Fatal("expected promoteToolChoiceError, got nil")
	}
	if out != nil {
		t.Errorf("body should be nil on reject, got %d bytes", len(out))
	}
	if perr.HTTPStatus() != 400 {
		t.Errorf("status = %d, want 400", perr.HTTPStatus())
	}
	if perr.Model != "M" {
		t.Errorf("model = %q, want M", perr.Model)
	}
	expectedNames := []string{"get_time", "search", "send_email"}
	if len(perr.ToolNames) != len(expectedNames) {
		t.Errorf("tool names: %v", perr.ToolNames)
	}
	// Error JSON body must be parseable and OpenAI-shaped.
	var body2 map[string]any
	if err := json.Unmarshal(perr.JSON(), &body2); err != nil {
		t.Fatalf("error JSON unparseable: %v", err)
	}
	errObj, _ := body2["error"].(map[string]any)
	if code, _ := errObj["code"].(string); code != "tool_choice_pinning_required" {
		t.Errorf("code = %v", errObj["code"])
	}
	if param, _ := errObj["param"].(string); param != "tool_choice" {
		t.Errorf("param = %v", errObj["param"])
	}
	msg, _ := errObj["message"].(string)
	for _, name := range expectedNames {
		if !strings.Contains(msg, name) {
			t.Errorf("error message missing tool name %q: %s", name, msg)
		}
	}
}

func TestPromote_MultiToolNoChoiceOnForcedRejects(t *testing.T) {
	a := promoteFixture("vllm", "M", "forced")
	body := requestBody(t, []map[string]any{toolSpec("a"), toolSpec("b")}, nil)
	_, perr := a.maybePromoteToolChoice("vllm", "M", body)
	if perr == nil {
		t.Fatal("expected reject for multi-tool with no choice")
	}
}

// --- Pass-through cases ---

func TestPromote_AutoModeNeverRewrites(t *testing.T) {
	// Models the probe verified via tool_choice="auto" don't need help.
	// The probe round-trip succeeded with auto; pass through unchanged.
	a := promoteFixture("vllm", "M", "auto")
	body := requestBody(t, []map[string]any{toolSpec("get_time")}, "auto")
	out, perr := a.maybePromoteToolChoice("vllm", "M", body)
	if perr != nil {
		t.Fatalf("unexpected error: %v", perr)
	}
	if string(out) != string(body) {
		t.Errorf("auto-mode model should pass through unchanged")
	}
}

func TestPromote_RequiredPassesThrough(t *testing.T) {
	// tool_choice="required" is the agent saying "force some tool call".
	// Don't override the agent's explicit choice even if the model is
	// forced-only — best effort is the agent's call.
	a := promoteFixture("vllm", "M", "forced")
	body := requestBody(t, []map[string]any{
		toolSpec("get_time"), toolSpec("search"),
	}, "required")
	out, perr := a.maybePromoteToolChoice("vllm", "M", body)
	if perr != nil {
		t.Fatalf("unexpected error: %v", perr)
	}
	if string(out) != string(body) {
		t.Errorf("required tool_choice must pass through unchanged")
	}
}

func TestPromote_NonePassesThrough(t *testing.T) {
	// "none" is the agent explicitly disabling tools. Don't promote.
	a := promoteFixture("vllm", "M", "forced")
	body := requestBody(t, []map[string]any{toolSpec("get_time")}, "none")
	out, perr := a.maybePromoteToolChoice("vllm", "M", body)
	if perr != nil {
		t.Fatalf("unexpected error: %v", perr)
	}
	if string(out) != string(body) {
		t.Errorf(`tool_choice="none" must pass through unchanged`)
	}
}

func TestPromote_ExplicitFunctionPassesThrough(t *testing.T) {
	// The agent already pinned to a specific function. Don't second-guess.
	a := promoteFixture("vllm", "M", "forced")
	body := requestBody(t, []map[string]any{
		toolSpec("get_time"), toolSpec("search"),
	}, map[string]any{
		"type":     "function",
		"function": map[string]any{"name": "search"},
	})
	out, perr := a.maybePromoteToolChoice("vllm", "M", body)
	if perr != nil {
		t.Fatalf("unexpected error: %v", perr)
	}
	// JSON canonicalisation may reorder keys; round-trip both sides.
	var inDoc, outDoc map[string]any
	_ = json.Unmarshal(body, &inDoc)
	_ = json.Unmarshal(out, &outDoc)
	if !sameJSON(inDoc["tool_choice"], outDoc["tool_choice"]) {
		t.Errorf("explicit tool_choice mutated: %+v -> %+v",
			inDoc["tool_choice"], outDoc["tool_choice"])
	}
}

func TestPromote_NoToolsPassesThrough(t *testing.T) {
	// Plain chat without tools: nothing to promote.
	a := promoteFixture("vllm", "M", "forced")
	body := requestBody(t, nil, nil)
	out, perr := a.maybePromoteToolChoice("vllm", "M", body)
	if perr != nil {
		t.Fatalf("unexpected error: %v", perr)
	}
	if string(out) != string(body) {
		t.Errorf("plain chat must pass through unchanged")
	}
}

func TestPromote_OllamaBackendNeverRewrites(t *testing.T) {
	// Ollama negotiates tool support per request and doesn't need
	// router-side promotion. Even a "forced" entry on Ollama (which
	// shouldn't exist, but defensively) must be a no-op.
	a := &arbiter{
		modelToolMode: map[string]map[string]string{
			"ollama": {"M": "forced"},
		},
	}
	body := requestBody(t, []map[string]any{
		toolSpec("a"), toolSpec("b"),
	}, "auto")
	out, perr := a.maybePromoteToolChoice("ollama", "M", body)
	if perr != nil {
		t.Fatalf("unexpected error: %v", perr)
	}
	if string(out) != string(body) {
		t.Errorf("ollama backend must pass through unchanged")
	}
}

func TestPromote_UnknownModelPassesThrough(t *testing.T) {
	// Model with no entry in modelToolMode (e.g., the cache row's
	// tool_mode is empty because tool_parser wasn't verified) — no
	// promote, no reject. maybeStripTools handles this case downstream.
	a := promoteFixture("vllm", "OTHER-MODEL", "forced")
	body := requestBody(t, []map[string]any{toolSpec("a"), toolSpec("b")}, "auto")
	out, perr := a.maybePromoteToolChoice("vllm", "DeepSeek-R1-Distill-Qwen-7B", body)
	if perr != nil {
		t.Fatalf("unexpected error: %v", perr)
	}
	if string(out) != string(body) {
		t.Errorf("unknown model must pass through unchanged")
	}
}

func TestPromote_MalformedJSONPassesThrough(t *testing.T) {
	// Defensive posture mirrors maybeStripTools: a body the router
	// can't parse is forwarded as-is. The backend will reject it and
	// surface its own error. Don't double-fault by returning 400 on
	// data we don't understand.
	a := promoteFixture("vllm", "M", "forced")
	body := []byte(`{not valid json`)
	out, perr := a.maybePromoteToolChoice("vllm", "M", body)
	if perr != nil {
		t.Fatalf("unexpected error: %v", perr)
	}
	if string(out) != string(body) {
		t.Errorf("malformed body must pass through unchanged")
	}
}

// sameJSON compares two values via JSON canonicalisation. Suitable for
// shallow map[string]any compares where key order may differ.
func sameJSON(a, b any) bool {
	ab, errA := json.Marshal(a)
	bb, errB := json.Marshal(b)
	if errA != nil || errB != nil {
		return false
	}
	return string(ab) == string(bb)
}
