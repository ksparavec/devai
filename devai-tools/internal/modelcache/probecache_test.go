package modelcache

import (
	"os"
	"path/filepath"
	"testing"
)

func writeJSONFixture(t *testing.T, name, content string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

const sampleVLLMCache = `{
  "nvidia/Qwen3-8B-NVFP4@abc123def456": {
    "repo": "nvidia/Qwen3-8B-NVFP4",
    "sha": "abc123def456",
    "aliases": ["Qwen3-8B-NVFP4"],
    "probes": {
      "24": {
        "131072": {"ctx": 131072, "fits": true},
        "32768": {"ctx": 32768, "fits": false}
      }
    }
  }
}`

const sampleOllamaCache = `{
  "sha256:digestabc": {
    "aliases": ["qwen3.5:9b-q8_0"],
    "probes": {
      "24": {
        "131072": {"ctx": 131072, "fully_on_gpu": true}
      }
    }
  }
}`

func TestLoadProbeCacheMissingFileIsNil(t *testing.T) {
	cache, err := LoadProbeCache(filepath.Join(t.TempDir(), "does-not-exist.json"))
	if err != nil {
		t.Fatalf("LoadProbeCache on missing file: %v", err)
	}
	if cache != nil {
		t.Errorf("expected nil cache for missing file, got %+v", cache)
	}
}

func TestFitsAtExactCellOnly(t *testing.T) {
	path := writeJSONFixture(t, "vllm.json", sampleVLLMCache)
	cache, err := LoadProbeCache(path)
	if err != nil {
		t.Fatalf("LoadProbeCache: %v", err)
	}
	entry := cache["nvidia/Qwen3-8B-NVFP4@abc123def456"]

	if !entry.FitsAt(24, 131072, "vllm") {
		t.Error("expected fit at (24, 131072)")
	}
	if entry.FitsAt(24, 32768, "vllm") {
		t.Error("cell explicitly records fits=false at (24, 32768), should not fit")
	}
	if entry.FitsAt(16, 131072, "vllm") {
		t.Error("no cell at vram=16 -- no interpolation, should not fit")
	}
	if entry.FitsAt(24, 65536, "vllm") {
		t.Error("no cell at ctx=65536 -- no interpolation, should not fit")
	}
}

func TestListFittingJoinsCatalogAndFallsBack(t *testing.T) {
	catalog := []CatalogEntry{
		{Name: "Qwen3-8B-NVFP4", Family: "qwen3", Backend: []string{"vllm", "sglang"}, Repo: "nvidia/Qwen3-8B-NVFP4", Sha: "abc123def456", Size: "5.2 GB"},
		{Name: "qwen3.5:9b-q8_0", Family: "qwen3.5", Backend: []string{"ollama"}, Size: "9.6 GB"},
	}

	vllmPath := writeJSONFixture(t, "vllm.json", sampleVLLMCache)
	vllmCache, err := LoadProbeCache(vllmPath)
	if err != nil {
		t.Fatal(err)
	}
	ollamaPath := writeJSONFixture(t, "ollama.json", sampleOllamaCache)
	ollamaCache, err := LoadProbeCache(ollamaPath)
	if err != nil {
		t.Fatal(err)
	}

	caches := map[string]ProbeCache{"vllm": vllmCache, "ollama": ollamaCache}

	results := ListFitting(catalog, caches, 24, 131072, "")
	if len(results) != 2 {
		t.Fatalf("got %d results, want 2: %+v", len(results), results)
	}
	byBackend := map[string]FitResult{}
	for _, r := range results {
		byBackend[r.Backend] = r
	}
	if byBackend["vllm"].Name != "Qwen3-8B-NVFP4" || byBackend["vllm"].Family != "qwen3" {
		t.Errorf("vllm result = %+v, want catalog-enriched Qwen3-8B-NVFP4/qwen3", byBackend["vllm"])
	}
	if byBackend["ollama"].Name != "qwen3.5:9b-q8_0" {
		t.Errorf("ollama result = %+v", byBackend["ollama"])
	}

	scoped := ListFitting(catalog, caches, 24, 131072, "ollama")
	if len(scoped) != 1 || scoped[0].Backend != "ollama" {
		t.Errorf("backend-scoped ListFitting = %+v, want exactly one ollama result", scoped)
	}
}

func TestListFittingFallsBackWithoutCatalogMatch(t *testing.T) {
	// A probe entry whose repo/sha has no corresponding models.yaml row
	// should still surface, identified from the probe cache itself.
	uncataloged := `{
      "unknown/Repo@deadbeef": {
        "repo": "unknown/Repo",
        "sha": "deadbeef",
        "probes": {"24": {"131072": {"ctx": 131072, "fits": true}}}
      }
    }`
	path := writeJSONFixture(t, "vllm.json", uncataloged)
	cache, err := LoadProbeCache(path)
	if err != nil {
		t.Fatal(err)
	}
	results := ListFitting(nil, map[string]ProbeCache{"vllm": cache}, 24, 131072, "vllm")
	if len(results) != 1 || results[0].Repo != "unknown/Repo" {
		t.Fatalf("expected one fallback-identified result, got %+v", results)
	}
}

func TestResolveBenchBase(t *testing.T) {
	catalog := []CatalogEntry{
		{Name: "Qwen3-8B-NVFP4", Backend: []string{"vllm", "sglang"}, Repo: "nvidia/Qwen3-8B-NVFP4", Sha: "abc123def456"},
	}
	ollamaPath := writeJSONFixture(t, "ollama.json", sampleOllamaCache)
	ollamaCache, err := LoadProbeCache(ollamaPath)
	if err != nil {
		t.Fatal(err)
	}

	base, ok := ResolveBenchBase(catalog, nil, "vllm", "Qwen3-8B-NVFP4")
	if !ok || base != "nvidia/Qwen3-8B-NVFP4@abc123def456" {
		t.Errorf("ResolveBenchBase(vllm) = (%q, %v), want nvidia/Qwen3-8B-NVFP4@abc123def456, true", base, ok)
	}

	base, ok = ResolveBenchBase(nil, ollamaCache, "ollama", "qwen3.5:9b-q8_0")
	if !ok || base != "sha256:digestabc" {
		t.Errorf("ResolveBenchBase(ollama) = (%q, %v), want sha256:digestabc, true", base, ok)
	}

	if _, ok := ResolveBenchBase(catalog, ollamaCache, "vllm", "no-such-model"); ok {
		t.Error("expected ResolveBenchBase to fail for an unknown model")
	}
}
