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

func TestFitsAtExactCellWins(t *testing.T) {
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
		t.Error("exact cell records fits=false at (24, 32768) and must win over the larger cell")
	}
	if entry.FitsAt(16, 131072, "vllm") {
		t.Error("no cell at vram=16 -- bands never interpolate, should not fit")
	}
	if entry.FitsAt(24, 262144, "vllm") {
		t.Error("nothing recorded at or above ctx=262144 -- should not fit")
	}
}

// The vLLM/SGLang caches keep exactly ONE binary-searched winner cell per
// (model, band): the largest ctx that both fits and serves. By KV
// monotonicity it covers every smaller ctx, so a sub-winner query must
// resolve to it rather than miss. Regression for the exact-cell-only
// lookup that hid every single-cell model below its winner.
func TestFitsAtSingleWinnerCellCoversSmallerContexts(t *testing.T) {
	const singleCell = `{
      "openai/gpt-oss-20b@6cee5e81ee83": {
        "repo": "openai/gpt-oss-20b",
        "sha": "6cee5e81ee83",
        "max_context": 131072,
        "probes": {"24": {"131072": {"ctx": 131072, "fits": true, "serving_ok": true}}}
      }
    }`
	cache, err := LoadProbeCache(writeJSONFixture(t, "vllm.json", singleCell))
	if err != nil {
		t.Fatal(err)
	}
	entry := cache["openai/gpt-oss-20b@6cee5e81ee83"]

	for _, ctx := range []int{32768, 65536, 131072} {
		if !entry.FitsAt(24, ctx, "vllm") {
			t.Errorf("winner cell at 131072 must cover ctx=%d", ctx)
		}
	}
	// A request ABOVE max_context is not upward extrapolation -- it is the
	// design-ceiling clamp: hf_probe_at_context computes
	// eff = min(ctx, max_context) first, so 262144 resolves to the 131072
	// winner cell and the model fits. Verified against the real cache:
	// openai/gpt-oss-20b (max_context 131072) is one of the 16 vLLM rows
	// `make model-fit CONTEXT=262144` reports.
	if !entry.FitsAt(24, 262144, "vllm") {
		t.Error("ctx=262144 must clamp to max_context=131072 and fit, as select-models.py does")
	}
}

// Without max_context there is nothing to clamp to, so a sub-winner query
// misses -- exactly what hf_probe_at_context returns (None). Only entries
// with no clean cell lack max_context (refresh_top_level_from_cells in
// scripts/_probe_hf_common.py sets it from the largest clean
// actual_context), so in practice such entries never fit anywhere.
func TestFitsAtWithoutMaxContextDoesNotCoverSmallerContexts(t *testing.T) {
	const noMax = `{
      "some/model@aaa": {
        "repo": "some/model", "sha": "aaa",
        "probes": {"24": {"131072": {"ctx": 131072, "fits": true}}}
      }
    }`
	cache, err := LoadProbeCache(writeJSONFixture(t, "vllm.json", noMax))
	if err != nil {
		t.Fatal(err)
	}
	entry := cache["some/model@aaa"]
	if !entry.FitsAt(24, 131072, "vllm") {
		t.Error("the exact recorded cell must still answer")
	}
	if entry.FitsAt(24, 32768, "vllm") {
		t.Error("no max_context means no clamp and no fallback -- must miss, like hf_probe_at_context")
	}
}

// The design-ceiling clamp is what `make model-fit` applies, so a model
// whose ceiling sits below the request must still be reported. Regression
// for the exact case measured against the real caches: at ctx=262144 the
// Go join returned 4 vLLM rows where select-models.py returned 16.
func TestFitsAtClampsRequestToDesignCeiling(t *testing.T) {
	const belowRequest = `{
      "NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4@3b554c28e968": {
        "repo": "NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4",
        "sha": "3b554c28e968",
        "max_context": 65536,
        "probes": {"24": {"65536": {"ctx": 65536, "fits": true, "serving_ok": true}}}
      }
    }`
	cache, err := LoadProbeCache(writeJSONFixture(t, "vllm.json", belowRequest))
	if err != nil {
		t.Fatal(err)
	}
	entry := cache["NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4@3b554c28e968"]
	for _, ctx := range []int{32768, 65536, 131072, 262144} {
		if !entry.FitsAt(24, ctx, "vllm") {
			t.Errorf("ctx=%d: a 65536-ceiling model fits at every request -- it just runs at 65536", ctx)
		}
	}
}

// serving_ok is tri-state: absent keeps the pre-load-probe fit-only
// verdict, an explicit false means the LOAD probe proved the cell loads
// but cannot serve at that ctx.
func TestFitsAtHonoursServingOk(t *testing.T) {
	// max_context mirrors the winner cell, as the prober always writes it
	// for an entry with a clean cell -- without it the 32768 lookups below
	// would miss for cell-resolution reasons and stop testing serving_ok.
	const cells = `{
      "a/absent@aaa": {"repo": "a/absent", "sha": "aaa", "max_context": 131072,
        "probes": {"24": {"131072": {"ctx": 131072, "fits": true}}}},
      "b/false@bbb": {"repo": "b/false", "sha": "bbb", "max_context": 131072,
        "probes": {"24": {"131072": {"ctx": 131072, "fits": true, "serving_ok": false}}}},
      "c/true@ccc": {"repo": "c/true", "sha": "ccc", "max_context": 131072,
        "probes": {"24": {"131072": {"ctx": 131072, "fits": true, "serving_ok": true}}}}
    }`
	cache, err := LoadProbeCache(writeJSONFixture(t, "vllm.json", cells))
	if err != nil {
		t.Fatal(err)
	}
	if !cache["a/absent@aaa"].FitsAt(24, 32768, "vllm") {
		t.Error("serving_ok absent must keep the fit-only verdict (pre-load-probe data)")
	}
	if cache["b/false@bbb"].FitsAt(24, 32768, "vllm") {
		t.Error("serving_ok=false must veto the cell -- the LOAD probe proved it cannot serve")
	}
	if !cache["c/true@ccc"].FitsAt(24, 32768, "vllm") {
		t.Error("serving_ok=true must fit")
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

	// nil stores = "no weight store visible here", the degraded path this
	// server actually runs in (its container mounts no weight volumes).
	// The gate itself is covered by TestListFittingGatesOnWeightsOnDisk.
	results, _ := ListFitting(catalog, caches, 24, 131072, "", nil)
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

	scoped, _ := ListFitting(catalog, caches, 24, 131072, "ollama", nil)
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
	results, _ := ListFitting(nil, map[string]ProbeCache{"vllm": cache}, 24, 131072, "vllm", nil)
	if len(results) != 1 || results[0].Repo != "unknown/Repo" {
		t.Fatalf("expected one fallback-identified result, got %+v", results)
	}
}

// The Ollama cache records every probed tier separately and
// probe_at_context (scripts/select-models.py ~762) has NO max_context
// fallback -- only the exact tier answers. Routing Ollama through
// hf_probe_at_context's winner-cell fallback made the Go side answer
// unprobed contexts: measured against the real caches, vram=24/ctx=100000
// returned 2 rows where the Python rule returns 0.
func TestFitsAtOllamaIsExactCellOnly(t *testing.T) {
	const tiers = `{
      "sha256:deadbeef": {
        "aliases": ["gemma4:26b-a4b-it-q4_K_M"],
        "max_context": 131072,
        "probes": {"24": {
          "32768": {"ctx": 32768, "fully_on_gpu": true, "fits": true},
          "131072": {"ctx": 131072, "fully_on_gpu": true, "fits": true}
        }}
      }
    }`
	cache, err := LoadProbeCache(writeJSONFixture(t, "ollama.json", tiers))
	if err != nil {
		t.Fatal(err)
	}
	entry := cache["sha256:deadbeef"]

	for _, ctx := range []int{32768, 131072} {
		if !entry.FitsAt(24, ctx, "ollama") {
			t.Errorf("probed tier ctx=%d must fit", ctx)
		}
	}
	if entry.FitsAt(24, 100000, "ollama") {
		t.Error("ctx=100000 was never probed -- probe_at_context returns None, so this must not fit")
	}
	if entry.FitsAt(24, 65536, "ollama") {
		t.Error("ctx=65536 was never probed -- no winner-cell fallback for ollama")
	}
	// The design-ceiling clamp is shared by both readers, so a request
	// above max_context still resolves to the recorded 131072 tier.
	if !entry.FitsAt(24, 262144, "ollama") {
		t.Error("ctx=262144 must clamp to max_context=131072, which IS probed")
	}
	// Same cells read as vllm keep the winner-cell fallback (the cells
	// carry both fits and fully_on_gpu so only the lookup rule differs).
	if !entry.FitsAt(24, 100000, "vllm") {
		t.Error("hf_probe_at_context's max_context fallback must be unchanged for vllm")
	}
}

// vLLM/SGLang rows must be gated on the weights actually being in that
// backend's store, the way model-picker.py enumerates the store and
// select-models.py's sglang_weight_gaps checks it. Measured against the
// real host caches before this gate: 16 vLLM rows at vram=24/ctx=32768
// against 5 model directories on disk.
func TestListFittingGatesOnWeightsOnDisk(t *testing.T) {
	const twoModels = `{
      "nvidia/OnDisk-8B-NVFP4@aaa": {
        "repo": "nvidia/OnDisk-8B-NVFP4", "sha": "aaa", "max_context": 131072,
        "probes": {"24": {"131072": {"ctx": 131072, "fits": true}}}},
      "nvidia/Absent-8B-NVFP4@bbb": {
        "repo": "nvidia/Absent-8B-NVFP4", "sha": "bbb", "max_context": 131072,
        "probes": {"24": {"131072": {"ctx": 131072, "fits": true}}}}
    }`
	cache, err := LoadProbeCache(writeJSONFixture(t, "vllm.json", twoModels))
	if err != nil {
		t.Fatal(err)
	}
	catalog := []CatalogEntry{
		{Name: "OnDisk-8B-NVFP4", Backend: []string{"vllm"}, Repo: "nvidia/OnDisk-8B-NVFP4", Sha: "aaa"},
		{Name: "Absent-8B-NVFP4", Backend: []string{"vllm"}, Repo: "nvidia/Absent-8B-NVFP4", Sha: "bbb"},
	}
	caches := map[string]ProbeCache{"vllm": cache}

	store := t.TempDir()
	if err := os.MkdirAll(filepath.Join(store, "OnDisk-8B-NVFP4"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(store, "OnDisk-8B-NVFP4", "config.json"), []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}
	// A model directory with no config.json is not weights -- same test
	// the picker applies when enumerating the store.
	if err := os.MkdirAll(filepath.Join(store, "Absent-8B-NVFP4"), 0o755); err != nil {
		t.Fatal(err)
	}

	got, notes := ListFitting(catalog, caches, 24, 32768, "vllm", WeightStores{"vllm": store})
	if len(got) != 1 || got[0].Name != "OnDisk-8B-NVFP4" {
		t.Fatalf("gated ListFitting = %+v, want only OnDisk-8B-NVFP4", got)
	}
	if len(notes) != 0 {
		t.Errorf("a present store must produce no degradation note, got %v", notes)
	}

	// An existing-but-EMPTY store is a real answer ("no weights here"),
	// not a degradation -- it is precisely the advertised-but-absent state
	// select-models.py's sglang_weight_gaps banner exists for.
	empty := t.TempDir()
	got, notes = ListFitting(catalog, caches, 24, 32768, "vllm", WeightStores{"vllm": empty})
	if len(got) != 0 {
		t.Errorf("empty store must yield no rows, got %+v", got)
	}
	if len(notes) != 0 {
		t.Errorf("empty store is not a degradation, got notes %v", notes)
	}
}

// A store path that is absent entirely is the container case: gating there
// would drop every HF row, so degrade to the un-gated verdict and say so.
func TestListFittingDegradesWhenStoreAbsent(t *testing.T) {
	const oneModel = `{
      "nvidia/Absent-8B-NVFP4@bbb": {
        "repo": "nvidia/Absent-8B-NVFP4", "sha": "bbb", "max_context": 131072,
        "probes": {"24": {"131072": {"ctx": 131072, "fits": true}}}}
    }`
	cache, err := LoadProbeCache(writeJSONFixture(t, "vllm.json", oneModel))
	if err != nil {
		t.Fatal(err)
	}
	caches := map[string]ProbeCache{"vllm": cache}

	for name, stores := range map[string]WeightStores{
		"nil map":         nil,
		"missing backend": {"sglang": t.TempDir()},
		"absent path":     {"vllm": filepath.Join(t.TempDir(), "no-such-dir")},
	} {
		got, notes := ListFitting(nil, caches, 24, 32768, "vllm", stores)
		if len(got) != 1 {
			t.Errorf("%s: expected the un-gated row to survive, got %+v", name, got)
		}
		if len(notes) != 1 {
			t.Errorf("%s: expected exactly one degradation note, got %v", name, notes)
		}
	}
}

// Ollama models are not directory-backed, so the gate must never touch
// them -- a directory check would drop every Ollama row unconditionally.
func TestListFittingDoesNotGateOllama(t *testing.T) {
	cache, err := LoadProbeCache(writeJSONFixture(t, "ollama.json", sampleOllamaCache))
	if err != nil {
		t.Fatal(err)
	}
	got, notes := ListFitting(nil, map[string]ProbeCache{"ollama": cache}, 24, 131072, "ollama",
		WeightStores{"vllm": t.TempDir(), "sglang": t.TempDir()})
	if len(got) != 1 {
		t.Fatalf("ollama rows must not be weights-gated, got %+v", got)
	}
	if len(notes) != 0 {
		t.Errorf("ollama needs no store, so no note: %v", notes)
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
