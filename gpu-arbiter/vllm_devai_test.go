package main

// A fourth backend, `vllm-devai` (2026-09-22): the home-built, HyperQwen-
// patched vLLM 0.28.0 image, on its own port (11437), container, and probe
// cache, so the stock vLLM 0.22.1 and the custom build can be addressed --
// and probed, benched and compared -- side by side. A per-request suffix
// was rejected: the probe cache stamps every row with the image it was
// measured on, and mixing two images in one cache is exactly what the
// drift check treats as invalid.
//
// The backend's NAME is what ports, containers, caches and lookup maps key
// on. Its ENGINE is vLLM, and every place the router's behaviour depends on
// the engine (reasoning-policy body shape, tool stripping, Anthropic
// normalisation, parser plugins, the legacy fp8 KV default, memory
// heuristic, recovery-flag scoping) must treat it as vLLM. These tests pin
// that every such branch goes through engineOf.

import (
	"bytes"
	"encoding/json"
	"testing"
)

func TestEngineOf(t *testing.T) {
	cases := map[string]string{
		"ollama": "ollama", "vllm": "vllm", "sglang": "sglang",
		"vllm-devai": "vllm",
	}
	for name, want := range cases {
		if got := engineOf(name); got != want {
			t.Errorf("engineOf(%q) = %q, want %q", name, got, want)
		}
	}
}

// The capability and disable_verified maps are keyed by BACKEND, and a
// derived checkpoint is probed on vllm-devai ONLY -- it has no entry under
// "vllm". The first version of this test copied the vllm map into the
// vllm-devai slot and so could not see that applyVLLMPolicy looked the
// model up under "vllm" regardless of the backend it was asked about.
// What that cost (2026-09-22, first real bench of the derived rows): the
// capability read back as unknown, no reasoning_effort was injected, and
// the Qwen3.8 chat template applied its OWN default -- `xhigh`, which
// prepends a 38-token "think carefully" system preamble. HumanEval fell
// from 97.6 % to 46 % on byte-identical weights, and `::nothink` was a
// silent no-op on port 11437.
func TestVLLMDevai_ReasoningPolicyReadsTheBackendsOwnCapability(t *testing.T) {
	a := &arbiter{
		modelCapability: map[string]map[string]string{
			"vllm-devai": {"m-devai": CapStructured},
		},
		modelDisableOK: map[string]map[string]bool{
			"vllm-devai": {"m-devai": true},
		},
	}
	in := []byte(`{"model":"m-devai","messages":[]}`)
	auto := a.applyReasoningPolicy("vllm-devai", "/v1/chat/completions", "m-devai", "auto", in)
	if got := jsonStringField(t, auto, "reasoning_effort"); got != "medium" {
		t.Fatalf("auto on a structured vllm-devai model must inject reasoning_effort=medium, got %q in %s", got, auto)
	}
	off := a.applyReasoningPolicy("vllm-devai", "/v1/chat/completions", "m-devai", "off", in)
	if got := jsonStringField(t, off, "reasoning_effort"); got != "none" {
		t.Fatalf("off on a disable-verified vllm-devai model must inject reasoning_effort=none, got %q in %s", got, off)
	}
	// And the same model name known ONLY to vllm-devai must not be
	// rewritten when asked about on stock vllm.
	if stock := a.applyReasoningPolicy("vllm", "/v1/chat/completions", "m-devai", "auto", in); !bytes.Equal(stock, in) {
		t.Fatalf("vllm has no capability for m-devai and must leave the body alone: %s", stock)
	}
}

func jsonStringField(t *testing.T, body []byte, key string) string {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(body, &m); err != nil {
		t.Fatalf("bad JSON %s: %v", body, err)
	}
	s, _ := m[key].(string)
	return s
}

func TestVLLMDevai_MemoryHeuristicMatchesVLLM(t *testing.T) {
	for _, size := range []float64{7.4, 17, 22} {
		if a, b := memFraction(size, 24, "vllm"), memFraction(size, 24, "vllm-devai"); a != b {
			t.Errorf("size %.1f: memFraction vllm=%v devai=%v", size, a, b)
		}
	}
}

func TestVLLMDevai_RecoveryEntriesScopeByEngineAndByName(t *testing.T) {
	r := &recoveryRegistry{Models: map[string]recoveryEntry{
		"vllm-scoped":  {Flags: []string{"--enforce-eager"}, Backends: backendsPtr("vllm")},
		"devai-scoped": {Flags: []string{"--mamba-ssm-cache-dtype", "float16"}, Backends: backendsPtr("vllm-devai")},
		"sglang-only":  {Flags: []string{"--x"}, Backends: backendsPtr("sglang")},
	}}
	// vLLM CLI flags are engine flags: an entry written for the stock vLLM
	// backend applies to the custom vLLM build as well (a flag the 0.28
	// image lacks fails the launch loudly, which is the right outcome).
	if _, ok := r.Lookup("vllm-devai", "vllm-scoped"); !ok {
		t.Error("an entry scoped to vllm must apply to vllm-devai (same engine)")
	}
	if _, ok := r.Lookup("vllm-devai", "devai-scoped"); !ok {
		t.Error("an entry scoped to vllm-devai must apply to vllm-devai")
	}
	// ...but not the other way round: a prepared checkpoint's flags name
	// the image that can serve it, and stock vLLM cannot.
	if _, ok := r.Lookup("vllm", "devai-scoped"); ok {
		t.Error("an entry scoped to vllm-devai must NOT apply to stock vllm")
	}
	if _, ok := r.Lookup("vllm-devai", "sglang-only"); ok {
		t.Error("an sglang-scoped entry must not apply to vllm-devai")
	}
}

func TestVLLMDevai_LegacyCellsDecodeToFP8LikeVLLM(t *testing.T) {
	// Unstamped cells in a vLLM-family cache were measured under fp8; the
	// custom build's prober writes the same cache shape.
	tool := "qwen3_xml"
	entry := func() *hfCacheEntry {
		return &hfCacheEntry{
			SchemaVersion: 2, Repo: "org/model", Aliases: []string{"model"},
			SizeGB: 10, MaxContext: 262144, Capability: CapStructured, ToolParser: &tool,
			Probes: map[string]map[string]hfCacheProbe{
				"24": {"32768": {Ctx: 32768, VramGB: 24, Fits: true, ActualVRAMGB: 20}},
			},
		}
	}
	for _, backend := range []string{"vllm", "vllm-devai"} {
		rows := synthesizeHFFromCache(map[string]*hfCacheEntry{"k": entry()}, backend, 24, 262144, nil)
		if len(rows) != 1 {
			t.Fatalf("%s: expected one serving row, got %d", backend, len(rows))
		}
		if kv := rows[0].KVByCtx[32768]; kv != "fp8" {
			t.Errorf("%s: unstamped cell must decode to fp8, got %q", backend, kv)
		}
	}
}

// Image-drift detection reads each HF backend's probe cache through a
// name-keyed map. vllm-devai was missing from it, so readProbedImageDigest
// got "" and the backend could never be reported stale -- the drift
// safety net built for exactly this kind of backend was silently off
// (found in review, 2026-09-22).
func TestVLLMDevai_DriftCheckReadsItsOwnProbeCache(t *testing.T) {
	m := probeCachePathByBackend("/v.json", "/s.json", "/d.json")
	want := map[string]string{"vllm": "/v.json", "sglang": "/s.json", "vllm-devai": "/d.json"}
	for name, path := range want {
		if m[name] != path {
			t.Errorf("probeCachePathByBackend[%q] = %q, want %q", name, m[name], path)
		}
	}
}
