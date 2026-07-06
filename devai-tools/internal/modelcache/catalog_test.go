package modelcache

import (
	"os"
	"path/filepath"
	"testing"
)

const sampleCatalog = `
models:
  - name: "qwen3.5:9b-q8_0"
    family: qwen3.5
    backend: [ollama]
    source: ollama
    size: "9.60 GB"
    purpose: "qwen3.5 - 9.6 GB"
    conversational: true

  - name: "Qwen3-8B-NVFP4"
    family: qwen3
    backend: [vllm, sglang]
    repo: "nvidia/Qwen3-8B-NVFP4"
    source: hf
    sha: "abc123def456"
    size: "5.2 GB"
    purpose: "qwen3 - 5.2 GB"
    conversational: true
`

func writeTempFile(t *testing.T, content string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "models.yaml")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLoadCatalog(t *testing.T) {
	path := writeTempFile(t, sampleCatalog)
	entries, err := LoadCatalog(path)
	if err != nil {
		t.Fatalf("LoadCatalog: %v", err)
	}
	if len(entries) != 2 {
		t.Fatalf("got %d entries, want 2", len(entries))
	}
	if entries[0].Name != "qwen3.5:9b-q8_0" || entries[0].Source != "ollama" {
		t.Errorf("entries[0] = %+v", entries[0])
	}
	if entries[1].Repo != "nvidia/Qwen3-8B-NVFP4" || entries[1].Sha != "abc123def456" {
		t.Errorf("entries[1] = %+v", entries[1])
	}
	if !hasBackend(entries[1].Backend, "vllm") || !hasBackend(entries[1].Backend, "sglang") {
		t.Errorf("entries[1].Backend = %v, want [vllm sglang]", entries[1].Backend)
	}
}

func TestLoadCatalogAgainstRepoFixture(t *testing.T) {
	path := repoFixturePath(t, "models.yaml")
	entries, err := LoadCatalog(path)
	if err != nil {
		t.Fatalf("LoadCatalog(repo fixture): %v", err)
	}
	if len(entries) == 0 {
		t.Fatal("expected at least one catalog entry from the repo fixture")
	}
}

// repoFixturePath resolves tests/fixtures/modelstatus/<name> relative to
// the repo root, so both the Go unit tests and tests/test-mcp-modelstatus.sh
// exercise the exact same hand-crafted fixtures.
func repoFixturePath(t *testing.T, name string) string {
	t.Helper()
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	// devai-tools/internal/modelcache -> repo root is 3 levels up.
	root := filepath.Join(wd, "..", "..", "..")
	path := filepath.Join(root, "tests", "fixtures", "modelstatus", name)
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("repo fixture %s not found: %v", path, err)
	}
	return path
}
