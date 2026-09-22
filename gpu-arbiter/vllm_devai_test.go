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
