package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// checkModelWeights is the "fail loudly at launch" half of the
// SGLANG_MODELS_DIR gap (source review HI-SW3): the probe cache can
// advertise a model whose weights were never downloaded, and without
// this the engine burns a full HEALTH_TIMEOUT_SECONDS cold start before
// dying with an opaque error.
func TestCheckModelWeights(t *testing.T) {
	store := t.TempDir()
	if err := os.MkdirAll(filepath.Join(store, "nvidia", "Qwen3-8B-NVFP4"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(store, "gpt-oss-20b"), 0o755); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name      string
		backend   string
		modelsDir string
		model     string
		wantErr   bool
	}{
		{"ollama is exempt (blob store, not dir-per-model)", "ollama", store, "does-not-exist", false},
		{"no models dir configured", "vllm", "", "anything", false},
		{"empty model name defers to the model-name guard", "vllm", store, "", false},
		{"store not mounted into the router degrades to a warning", "sglang", filepath.Join(store, "absent"), "whatever", false},
		{"weights present, flat name", "vllm", store, "gpt-oss-20b", false},
		{"weights present, HF repo form", "vllm", store, "nvidia/Qwen3-8B-NVFP4", false},
		{"weights missing", "sglang", store, "Qwen3-14B-NVFP4", true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := &arbiter{}
			err := a.checkModelWeights(
				backendConfig{Name: tc.backend, ModelsDir: tc.modelsDir}, tc.model,
			)
			if tc.wantErr && err == nil {
				t.Fatalf("expected an error, got nil")
			}
			if !tc.wantErr && err != nil {
				t.Fatalf("expected no error, got %v", err)
			}
			if !tc.wantErr {
				return
			}
			// The message has to tell the operator what to run; an
			// opaque failure here is barely better than the cold-start
			// timeout this replaces.
			for _, want := range []string{tc.model, tc.modelsDir, "make model-pull"} {
				if !strings.Contains(err.Error(), want) {
					t.Errorf("error %q does not mention %q", err, want)
				}
			}
		})
	}
}

// The degradation path must not fire once per request.
func TestCheckModelWeights_UnmountedStoreWarnsOnce(t *testing.T) {
	a := &arbiter{}
	cfg := backendConfig{Name: "sglang", ModelsDir: filepath.Join(t.TempDir(), "absent")}
	for i := 0; i < 3; i++ {
		if err := a.checkModelWeights(cfg, "some-model"); err != nil {
			t.Fatalf("unmounted store must not fail a launch: %v", err)
		}
	}
	if !a.weightWarned["sglang"] {
		t.Fatal("expected the sglang degradation to be recorded")
	}
	if len(a.weightWarned) != 1 {
		t.Fatalf("expected exactly one backend recorded, got %v", a.weightWarned)
	}
}
