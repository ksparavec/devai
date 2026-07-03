package main

import "testing"

// TestPeelControlSuffixes locks in order-independent stripping of the
// picker/control-surface suffixes. The regression that motivated the helper:
// aiagent/litellm appends its own `::<reasoning>` AFTER the picker's `@<ctx>`
// (`<name>@<ctx>::nothink`), which the old strict ctx-last stripper left with
// `@<ctx>` glued to the name, so the vLLM/SGLang allowlist rejected it as an
// unknown model. Every ordering below must reduce to the same (clean, ctx, mtp,
// reasoning) tuple, and names with a legitimate `::` or `@` must survive.
func TestPeelControlSuffixes(t *testing.T) {
	tests := []struct {
		name          string
		in            string
		wantClean     string
		wantCtx       int
		wantMTP       string
		wantReasoning string
	}{
		{"bare name", "model", "model", 0, "", ""},
		{"ctx only", "model@131072", "model", 131072, "", ""},
		{"reasoning only", "model::nothink", "model", 0, "", "off"},
		{"canonical reasoning+ctx", "model::nothink@131072", "model", 131072, "", "off"},
		// The aiagent bug: reasoning appended after ctx.
		{"aiagent order reasoning-after-ctx", "model@131072::nothink", "model", 131072, "", "off"},
		{"canonical reasoning+mtp+ctx", "model::high::mtp@65536", "model", 65536, "on", "high"},
		{"reversed mtp+reasoning after ctx", "model@65536::mtp::high", "model", 65536, "on", "high"},
		// HF repo names carry a slash; the aiagent order must still strip.
		{"hf repo slash aiagent order", "nvidia/Qwen3-8B-NVFP4@32768::nothink", "nvidia/Qwen3-8B-NVFP4", 32768, "", "off"},
		{"think maps to auto after ctx", "model@40960::think", "model", 40960, "", "auto"},
		{"nomtp after ctx", "model@8192::nomtp", "model", 8192, "off", ""},
		// Unrecognised `::<token>` must be left intact (not a control suffix).
		{"legit double-colon preserved", "model::foo", "model::foo", 0, "", ""},
		{"legit double-colon preserved with ctx", "model::foo@131072", "model::foo", 131072, "", ""},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			clean, ctx, mtp, reasoning := peelControlSuffixes(tc.in)
			if clean != tc.wantClean {
				t.Errorf("clean = %q, want %q", clean, tc.wantClean)
			}
			if ctx != tc.wantCtx {
				t.Errorf("ctx = %d, want %d", ctx, tc.wantCtx)
			}
			if mtp != tc.wantMTP {
				t.Errorf("mtp = %q, want %q", mtp, tc.wantMTP)
			}
			if reasoning != tc.wantReasoning {
				t.Errorf("reasoning = %q, want %q", reasoning, tc.wantReasoning)
			}
		})
	}
}
