package main

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
)

// Flag-drift gate.
//
// deploy/backend-flags.yaml declares itself the pin for every launch flag
// the router and probers emit, and `make verify-backend-flags` asserts
// those flags exist on the pinned images. But nothing ever checked the
// other direction: that every flag actually EMITTED is pinned. So the
// file drifted out of step with the code, and the drift gate it exists to
// power was silently incomplete -- `--tp` was pinned but does not exist on
// the image (it resolved only by argparse prefix matching), while
// `--max-num-seqs`, `--max-running-requests` and `--served-model-name`
// were emitted or needed but never pinned at all.
//
// This test closes the loop: extract the flag literals from each
// entrypoint and require each to appear in the YAML. Adding a flag to an
// entrypoint without pinning it now fails here.

var flagLiteralRe = regexp.MustCompile(`"(--[a-z0-9][a-z0-9-]*)"`)

// entrypointFlags returns every `--flag` string literal in the named Go
// function. Parsing the source rather than the rendered argv is
// deliberate: a flag emitted only on a rare branch (a recovery override,
// an MTP toggle) still needs a pin, and would be missed by any matrix a
// person remembered to write.
func entrypointFlags(t *testing.T, src, funcName string) []string {
	t.Helper()
	start := strings.Index(src, "func "+funcName+"(")
	if start < 0 {
		t.Fatalf("function %s not found", funcName)
	}
	// Scan to the closing brace at column 0 -- gofmt guarantees it.
	rest := src[start:]
	end := strings.Index(rest, "\n}\n")
	if end < 0 {
		t.Fatalf("could not find end of %s", funcName)
	}
	body := rest[:end]

	seen := map[string]bool{}
	for _, m := range flagLiteralRe.FindAllStringSubmatch(body, -1) {
		seen[m[1]] = true
	}
	out := make([]string, 0, len(seen))
	for f := range seen {
		out = append(out, f)
	}
	sort.Strings(out)
	return out
}

func TestEmittedFlagsArePinned(t *testing.T) {
	srcBytes, err := os.ReadFile("main.go")
	if err != nil {
		t.Fatalf("read main.go: %v", err)
	}
	src := string(srcBytes)

	pinBytes, err := os.ReadFile(filepath.Join("..", "deploy", "backend-flags.yaml"))
	if err != nil {
		t.Fatalf("read backend-flags.yaml: %v", err)
	}
	pinned := string(pinBytes)

	// Helper functions called BY the entrypoints also emit flags; include
	// them so a flag hidden one call deep is not exempt.
	for _, fn := range []string{
		"vllmEntrypoint", "sglangEntrypoint",
		"vllmSpeculativeArgs", "sglangSpeculativeArgs",
	} {
		if !strings.Contains(src, "func "+fn+"(") {
			continue // optional helper
		}
		for _, flag := range entrypointFlags(t, src, fn) {
			if !strings.Contains(pinned, `"`+flag+`"`) {
				t.Errorf("%s emits %s but deploy/backend-flags.yaml does not pin it.\n"+
					"Add it there (and re-run `make verify-backend-flags`) so an "+
					"image bump that renames it fails the gate instead of breaking "+
					"launches silently.", fn, flag)
			}
		}
	}
}

// The inverse direction for the one flag that was actively wrong: --tp is
// not on the pinned image at all. Guarding the literal keeps a future
// edit from reintroducing the prefix-matching spelling.
func TestSGLangDoesNotEmitBareTP(t *testing.T) {
	srcBytes, err := os.ReadFile("main.go")
	if err != nil {
		t.Fatalf("read main.go: %v", err)
	}
	for _, flag := range entrypointFlags(t, string(srcBytes), "sglangEntrypoint") {
		if flag == "--tp" {
			t.Fatal("sglangEntrypoint emits `--tp`, which does not exist on the " +
				"pinned image (only --tp-size). It resolved by argparse prefix " +
				"matching, which is not a contract.")
		}
	}
}

// The probers must format the memory fraction exactly as the router does,
// or the fit a cell records was measured under a different allocation
// than serve time reproduces. The router rounded 0.8836 to 0.88, handing
// the engine ~0.09 GB less than the probe measured on a 24 GB card.
func TestMemFractionFormatMatchesProbers(t *testing.T) {
	srcBytes, err := os.ReadFile("main.go")
	if err != nil {
		t.Fatalf("read main.go: %v", err)
	}
	src := string(srcBytes)
	for _, want := range []string{
		`"--gpu-memory-utilization", fmt.Sprintf("%.4f", lc.MemFraction)`,
		`"--mem-fraction-static", fmt.Sprintf("%.4f", lc.MemFraction)`,
	} {
		if !strings.Contains(src, want) {
			t.Errorf("expected the 4-decimal memory fraction format matching the "+
				"probers' f\"{host_frac:.4f}\": %s", want)
		}
	}

	// And assert the Python side still agrees, so this cannot drift from
	// the other end either.
	for _, p := range []string{
		filepath.Join("..", "scripts", "probe-vllm-reasoning.py"),
		filepath.Join("..", "scripts", "probe-sglang-reasoning.py"),
	} {
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatalf("read %s: %v", p, err)
		}
		if !strings.Contains(string(b), `:.4f`) {
			t.Errorf("%s no longer formats the memory fraction with .4f", p)
		}
	}
}

// memFraction used to fall through a `default: // vllm` arm, and
// computeLaunchConfig calls it with cfg.Name for all three backends -- so
// Ollama silently drew vLLM's 2.0 GB reserve. Inert today (Ollama's
// entrypoint ignores MemFraction) but wrong, and nothing would have said
// so if it ever became load-bearing.
func TestMemFraction_PerBackendReserves(t *testing.T) {
	const total = 24.0
	const size = 15.0

	vllm := memFraction(size, total, "vllm")
	sglang := memFraction(size, total, "sglang")
	if !(sglang < vllm) {
		t.Fatalf("SGLang reserves more (RadixAttention tree + CUDA graphs), so its "+
			"fraction must be smaller: vllm=%.4f sglang=%.4f", vllm, sglang)
	}

	// Unknown model size takes the per-backend conservative default.
	if got := memFraction(0, total, "sglang"); got != 0.82 {
		t.Fatalf("sglang unknown-size default: want 0.82, got %.4f", got)
	}
	if got := memFraction(0, total, "vllm"); got != 0.88 {
		t.Fatalf("vllm unknown-size default: want 0.88, got %.4f", got)
	}

	// Ollama is recognised-and-ignored, not silently defaulted: it must
	// not warn, and it must stay within the documented clamp.
	oll := memFraction(size, total, "ollama")
	if oll < 0.40 || oll > 0.95 {
		t.Fatalf("ollama fraction escaped the [0.40,0.95] clamp: %.4f", oll)
	}
}
