package main

import (
	"os"
	"path/filepath"
	"testing"
)

// loadCatalogMTP indexes deploy/models.yaml's `mtp:` blocks by repo so
// synthesizeHFFromCache can attach them to the configModel rows it
// emits. Cover the happy path (well-formed entries make it in), the
// no-MTP path (rows without an `mtp:` block are silently skipped),
// invalid-method rejection, and the missing-file degradation case.
func TestLoadCatalogMTP_HappyPathExtractsByRepo(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "models.yaml")
	contents := `models:
  - name: "Some-NonMTP-Model"
    repo: "vendor/foo"
    size: "8 GB"
  - name: "Gemma-4-26B-A4B-NVFP4"
    repo: "nvidia/Gemma-4-26B-A4B-NVFP4"
    size: "17 GB"
    mtp:
      method: mtp
      drafter: google/gemma-4-26B-A4B-it-assistant
      num_speculative_tokens: 4
  - name: "Qwen-MTP-builtin"
    repo: "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP"
    mtp:
      method: qwen3_5_mtp
      num_speculative_tokens: 3
`
	if err := os.WriteFile(path, []byte(contents), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	r := loadCatalogMTP(path)
	if r == nil {
		t.Fatal("nil registry")
	}
	// External-drafter case.
	got, ok := r.Lookup("nvidia/Gemma-4-26B-A4B-NVFP4")
	if !ok {
		t.Fatalf("Gemma-4 entry not indexed")
	}
	if got.Method != "mtp" || got.Drafter != "google/gemma-4-26B-A4B-it-assistant" || got.NumSpeculativeTokens != 4 {
		t.Errorf("Gemma-4 mismatch: got %+v", got)
	}
	// Built-in MTP head (no drafter field).
	got, ok = r.Lookup("sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP")
	if !ok {
		t.Fatalf("Qwen entry not indexed")
	}
	if got.Method != "qwen3_5_mtp" || got.Drafter != "" || got.NumSpeculativeTokens != 3 {
		t.Errorf("Qwen mismatch: got %+v", got)
	}
	// Non-MTP row stays out of the registry entirely.
	if _, ok := r.Lookup("vendor/foo"); ok {
		t.Errorf("non-MTP row leaked into MTP registry")
	}
	// Unknown repo and empty repo are safe.
	if _, ok := r.Lookup(""); ok {
		t.Errorf("empty repo should miss")
	}
	if _, ok := r.Lookup("does/not-exist"); ok {
		t.Errorf("unknown repo should miss")
	}
}

func TestLoadCatalogMTP_MissingFileReturnsEmpty(t *testing.T) {
	r := loadCatalogMTP("/nonexistent/path/models.yaml")
	if r == nil {
		t.Fatal("nil registry on missing file")
	}
	if _, ok := r.Lookup("anything"); ok {
		t.Errorf("missing-file registry should always miss")
	}
}

func TestLoadCatalogMTP_InvalidEntriesSkipped(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "models.yaml")
	// One valid entry, three malformed ones. The valid one must still
	// land; the bad ones must not crash the loader.
	contents := `models:
  - name: "OK"
    repo: "good/one"
    mtp:
      method: mtp
      num_speculative_tokens: 4
  - name: "BadMissingMethod"
    repo: "bad/method"
    mtp:
      drafter: x
      num_speculative_tokens: 2
  - name: "BadZeroK"
    repo: "bad/zero-k"
    mtp:
      method: mtp
      num_speculative_tokens: 0
  - name: "NoRepo"
    mtp:
      method: mtp
      num_speculative_tokens: 2
`
	if err := os.WriteFile(path, []byte(contents), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	r := loadCatalogMTP(path)
	if _, ok := r.Lookup("good/one"); !ok {
		t.Errorf("good entry dropped")
	}
	for _, badRepo := range []string{"bad/method", "bad/zero-k", ""} {
		if _, ok := r.Lookup(badRepo); ok {
			t.Errorf("bad repo %q leaked into registry", badRepo)
		}
	}
}

func TestLoadCatalogMTP_NilReceiverSafe(t *testing.T) {
	var r *catalogMTPRegistry
	if _, ok := r.Lookup("anything"); ok {
		t.Errorf("nil receiver should miss")
	}
}
