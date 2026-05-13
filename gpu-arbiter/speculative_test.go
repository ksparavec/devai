package main

import (
	"encoding/json"
	"strings"
	"testing"
)

// specEqual is a small helper but it sits on the recreate hot path,
// so cover the cases that matter: both nil, one nil, value-equal,
// each field differing.
func TestSpecEqual_AllPermutations(t *testing.T) {
	a := &configSpeculative{Method: "mtp", Drafter: "x", NumSpeculativeTokens: 4}
	a2 := &configSpeculative{Method: "mtp", Drafter: "x", NumSpeculativeTokens: 4}
	diffMethod := &configSpeculative{Method: "eagle3", Drafter: "x", NumSpeculativeTokens: 4}
	diffDrafter := &configSpeculative{Method: "mtp", Drafter: "y", NumSpeculativeTokens: 4}
	diffK := &configSpeculative{Method: "mtp", Drafter: "x", NumSpeculativeTokens: 3}
	cases := []struct {
		name string
		a, b *configSpeculative
		want bool
	}{
		{"both nil", nil, nil, true},
		{"left nil", nil, a, false},
		{"right nil", a, nil, false},
		{"value equal", a, a2, true},
		{"diff method", a, diffMethod, false},
		{"diff drafter", a, diffDrafter, false},
		{"diff k", a, diffK, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := specEqual(tc.a, tc.b); got != tc.want {
				t.Errorf("specEqual = %v, want %v", got, tc.want)
			}
		})
	}
}

// vllmSpeculativeJSON marshals the launch params into the JSON shape
// vLLM expects. External-drafter case must include the `model` field;
// built-in-MTP-head case must omit it. K floors to 1 on bad inputs.
func TestVLLMSpeculativeJSON_ExternalDrafter(t *testing.T) {
	got := vllmSpeculativeJSON(&configSpeculative{
		Method:               "mtp",
		Drafter:              "google/gemma-4-26B-A4B-it-assistant",
		NumSpeculativeTokens: 4,
	})
	var parsed map[string]any
	if err := json.Unmarshal([]byte(got), &parsed); err != nil {
		t.Fatalf("bad JSON: %v (raw=%q)", err, got)
	}
	if parsed["method"] != "mtp" {
		t.Errorf("method: got %v", parsed["method"])
	}
	if parsed["model"] != "/models/gemma-4-26B-A4B-it-assistant" {
		t.Errorf("model: got %v", parsed["model"])
	}
	// JSON numbers decode as float64; compare loose.
	if int(parsed["num_speculative_tokens"].(float64)) != 4 {
		t.Errorf("num_speculative_tokens: got %v", parsed["num_speculative_tokens"])
	}
}

func TestVLLMSpeculativeJSON_BuiltinHeadOmitsModel(t *testing.T) {
	got := vllmSpeculativeJSON(&configSpeculative{
		Method:               "qwen3_5_mtp",
		NumSpeculativeTokens: 3,
	})
	if strings.Contains(got, "\"model\"") {
		t.Errorf("built-in MTP head should not include `model` field; got %q", got)
	}
	if !strings.Contains(got, `"method":"qwen3_5_mtp"`) {
		t.Errorf("method missing: %q", got)
	}
	if !strings.Contains(got, `"num_speculative_tokens":3`) {
		t.Errorf("k missing: %q", got)
	}
}

func TestVLLMSpeculativeJSON_NilAndBadInputs(t *testing.T) {
	if got := vllmSpeculativeJSON(nil); got != "" {
		t.Errorf("nil should produce empty string; got %q", got)
	}
	if got := vllmSpeculativeJSON(&configSpeculative{}); got != "" {
		t.Errorf("empty method should produce empty string; got %q", got)
	}
	// K=0 floors to 1.
	got := vllmSpeculativeJSON(&configSpeculative{Method: "mtp", NumSpeculativeTokens: 0})
	if !strings.Contains(got, `"num_speculative_tokens":1`) {
		t.Errorf("k=0 should floor to 1; got %q", got)
	}
}

// sglangSpeculativeArgs maps catalog `method` to SGLang's
// --speculative-algorithm value. The picker-emitting common cases:
// `mtp` / `qwen3_5_mtp` / `deepseek_mtp` all collapse to NEXTN
// (SGLang's MTP-shaped alias of EAGLE).
func TestSGLangSpeculativeArgs_MethodMapping(t *testing.T) {
	cases := []struct {
		method   string
		wantAlgo string
	}{
		{"mtp", "NEXTN"},
		{"qwen3_5_mtp", "NEXTN"},
		{"deepseek_mtp", "NEXTN"},
		{"eagle", "EAGLE"},
		{"eagle3", "EAGLE3"},
		{"medusa", "MEDUSA"},
	}
	for _, tc := range cases {
		t.Run(tc.method, func(t *testing.T) {
			args := sglangSpeculativeArgs(&configSpeculative{
				Method: tc.method, NumSpeculativeTokens: 3,
			})
			// First two args must be ["--speculative-algorithm", <algo>].
			if len(args) < 2 || args[0] != "--speculative-algorithm" || args[1] != tc.wantAlgo {
				t.Errorf("method=%q args[:2] = %v, want [--speculative-algorithm %q]",
					tc.method, args[:2], tc.wantAlgo)
			}
		})
	}
}

func TestSGLangSpeculativeArgs_NilNoFlags(t *testing.T) {
	if got := sglangSpeculativeArgs(nil); len(got) != 0 {
		t.Errorf("nil should produce no args; got %v", got)
	}
}

func TestSGLangSpeculativeArgs_DrafterPath(t *testing.T) {
	args := sglangSpeculativeArgs(&configSpeculative{
		Method: "mtp", Drafter: "google/gemma-4-26B-A4B-it-assistant",
		NumSpeculativeTokens: 4,
	})
	found := false
	for i, a := range args {
		if a == "--speculative-draft-model-path" && i+1 < len(args) {
			if args[i+1] != "/models/gemma-4-26B-A4B-it-assistant" {
				t.Errorf("draft-model-path = %q, want /models/<basename>", args[i+1])
			}
			found = true
			break
		}
	}
	if !found {
		t.Errorf("--speculative-draft-model-path missing when Drafter set; args=%v", args)
	}
}

func TestSGLangSpeculativeArgs_BuiltinHeadOmitsDrafterPath(t *testing.T) {
	args := sglangSpeculativeArgs(&configSpeculative{
		Method: "qwen3_5_mtp", NumSpeculativeTokens: 3,
	})
	for _, a := range args {
		if a == "--speculative-draft-model-path" {
			t.Errorf("built-in MTP head must NOT emit --speculative-draft-model-path; args=%v", args)
		}
	}
}

// vllmEntrypoint must emit --speculative-config between the parser
// flags and the RecoveryFlags suffix when lc.Speculative is non-nil.
// Order matters: RecoveryFlags get appended last so an operator can
// last-flag-wins override the spec (vLLM resolves duplicate flags
// in left-to-right order).
func TestVLLMEntrypoint_EmitsSpeculativeConfig(t *testing.T) {
	lc := launchConfig{
		MemFraction: 0.9,
		MaxContext:  32768,
		ToolParser:  "hermes",
		Speculative: &configSpeculative{
			Method:               "mtp",
			Drafter:              "google/gemma-4-26B-A4B-it-assistant",
			NumSpeculativeTokens: 4,
		},
		RecoveryFlags: []string{"--enforce-eager"},
	}
	args := vllmEntrypoint("Gemma-4-26B-A4B-NVFP4", lc)
	specIdx := -1
	toolIdx := -1
	recoveryIdx := -1
	for i, a := range args {
		switch a {
		case "--speculative-config":
			specIdx = i
		case "--tool-call-parser":
			toolIdx = i
		case "--enforce-eager":
			recoveryIdx = i
		}
	}
	if specIdx < 0 {
		t.Fatalf("--speculative-config missing; args=%v", args)
	}
	if toolIdx < 0 {
		t.Fatalf("--tool-call-parser missing; args=%v", args)
	}
	if recoveryIdx < 0 {
		t.Fatalf("recovery flag missing; args=%v", args)
	}
	if !(toolIdx < specIdx && specIdx < recoveryIdx) {
		t.Errorf("ordering wrong: tool=%d, spec=%d, recovery=%d (want tool < spec < recovery); args=%v",
			toolIdx, specIdx, recoveryIdx, args)
	}
}

// When lc.Speculative is nil, vllmEntrypoint must NOT emit
// --speculative-config. Regression check for the gating logic.
func TestVLLMEntrypoint_NoSpecWhenNil(t *testing.T) {
	lc := launchConfig{
		MemFraction: 0.9,
		MaxContext:  32768,
	}
	args := vllmEntrypoint("Test-Model", lc)
	for _, a := range args {
		if a == "--speculative-config" {
			t.Errorf("--speculative-config should be absent when Speculative is nil; args=%v", args)
		}
	}
}

// sglangEntrypoint mirrors vllmEntrypoint: spec args emitted between
// parser flags and RecoveryFlags.
func TestSGLangEntrypoint_EmitsSpeculativeFlags(t *testing.T) {
	lc := launchConfig{
		MemFraction:  0.9,
		MaxContext:   32768,
		ToolParser:   "qwen",
		Speculative: &configSpeculative{
			Method:               "mtp",
			Drafter:              "google/gemma-4-26B-A4B-it-assistant",
			NumSpeculativeTokens: 4,
		},
		RecoveryFlags: []string{"--enforce-eager"},
	}
	args := sglangEntrypoint("Gemma-4-26B-A4B-NVFP4", lc)
	algoIdx := -1
	toolIdx := -1
	recoveryIdx := -1
	for i, a := range args {
		switch a {
		case "--speculative-algorithm":
			algoIdx = i
		case "--tool-call-parser":
			toolIdx = i
		case "--enforce-eager":
			recoveryIdx = i
		}
	}
	if algoIdx < 0 {
		t.Fatalf("--speculative-algorithm missing; args=%v", args)
	}
	if !(toolIdx < algoIdx && algoIdx < recoveryIdx) {
		t.Errorf("ordering: tool=%d, algo=%d, recovery=%d (want tool < algo < recovery); args=%v",
			toolIdx, algoIdx, recoveryIdx, args)
	}
}
