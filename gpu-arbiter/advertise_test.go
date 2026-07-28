package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Advertisement vetting: what /v1/models and /api/tags may list.
//
// The load-bearing property is the SPLIT. `advertised` is vetted;
// `modelNames` (the serving allowlist) is not, because the bench harness
// drives every task through the router -- so requiring a bench row to be
// SERVED would mean a newly probed model could never earn its first one.

func writeJSON(t *testing.T, dir, name string, v any) string {
	t.Helper()
	p := filepath.Join(dir, name)
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if err := os.WriteFile(p, b, 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	return p
}

func TestLoadBenchedModels_IndexesByBackendAndKeepsLargestCtx(t *testing.T) {
	d := t.TempDir()
	p := writeJSON(t, d, "bench.json", map[string]any{
		"_meta":                   map[string]any{"schema_version": 3},
		"repo@sha::vllm::32768":   map[string]any{"model": "M", "backend": "vllm", "context": 32768},
		"repo@sha::vllm::262144":  map[string]any{"model": "M", "backend": "vllm", "context": 262144},
		"repo@sha::sglang::65536": map[string]any{"model": "M", "backend": "sglang", "context": 65536},
	})
	got := loadBenchedModels(p)
	if got["vllm"]["M"] != 262144 {
		t.Fatalf("want largest vllm ctx 262144, got %d", got["vllm"]["M"])
	}
	if got["sglang"]["M"] != 65536 {
		t.Fatalf("want sglang 65536, got %d", got["sglang"]["M"])
	}
	if _, ok := got["_meta"]; ok {
		t.Fatal("_meta must not be treated as a bench row")
	}
}

func TestLoadBenchedModels_MissingFileIsEmptyNotFatal(t *testing.T) {
	if got := loadBenchedModels("/nonexistent/bench.json"); len(got) != 0 {
		t.Fatalf("missing file must yield an empty index, got %v", got)
	}
	if got := loadBenchedModels(""); len(got) != 0 {
		t.Fatal("empty path must yield an empty index")
	}
}

func TestLoadBenchExclusions_OnlyBenchReasons(t *testing.T) {
	d := t.TempDir()
	p := writeJSON(t, d, "ledger.json", map[string]any{
		"models": map[string]any{
			"Dropped": map[string]any{"backends": map[string]any{
				"sglang": map[string]any{"status": "excluded", "reason": "bench_dropped",
					"judged_at": map[string]any{"ctx": 131072}},
			}},
			// A PROBE verdict must not gate advertisement -- it already
			// gates probing, and the model may be fine on another axis.
			"OomOnly": map[string]any{"backends": map[string]any{
				"vllm": map[string]any{"status": "excluded", "reason": "oom"},
			}},
		},
	})
	got := loadBenchExclusions(p)
	if _, ok := got["sglang"]["Dropped"]; !ok {
		t.Fatal("bench_dropped must be indexed")
	}
	if _, ok := got["vllm"]["OomOnly"]; ok {
		t.Fatal("a probe-side `oom` verdict must NOT gate advertisement")
	}
}

func TestBenchExcludedAt_JudgedCtxAndAbove(t *testing.T) {
	a := &arbiter{benchExclusions: map[string]map[string]benchVerdict{
		"sglang": {
			"AtCtx":   {reason: "bench_dropped", ctx: 131072},
			"Blanket": {reason: "bench_failed", ctx: 0},
		},
	}}
	// A verdict at 131072 says nothing about 32768: long-context failures
	// do not imply short-context ones.
	if ex, _ := a.benchExcludedAt("sglang", "AtCtx", 32768); ex {
		t.Fatal("a verdict at 131072 must not exclude 32768")
	}
	if ex, _ := a.benchExcludedAt("sglang", "AtCtx", 131072); !ex {
		t.Fatal("must exclude at the judged ctx")
	}
	if ex, _ := a.benchExcludedAt("sglang", "AtCtx", 262144); !ex {
		t.Fatal("must exclude above the judged ctx")
	}
	// No recorded ctx = we do not know where it was judged = everywhere.
	if ex, _ := a.benchExcludedAt("sglang", "Blanket", 1); !ex {
		t.Fatal("a ctx-less verdict must apply everywhere")
	}
	if ex, _ := a.benchExcludedAt("sglang", "Absent", 32768); ex {
		t.Fatal("an unrecorded model must not be excluded")
	}
}

// The central test: vetting narrows the LISTING and leaves the serving
// allowlist alone.
func TestAdvertisedNames_VetsWithoutTouchingModelNames(t *testing.T) {
	store := t.TempDir()
	for _, n := range []string{"Benched", "NoBench", "Dropped"} {
		if err := os.MkdirAll(filepath.Join(store, n), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	// "NoWeights" deliberately has no directory.

	bs := &backendState{
		config:     backendConfig{Name: "sglang", ModelsDir: store},
		modelNames: []string{"Benched", "NoBench", "Dropped", "NoWeights"},
	}
	a := &arbiter{
		modelContexts: map[string]map[string]int{
			"sglang": {"Benched": 131072, "NoBench": 131072, "Dropped": 131072,
				"NoWeights": 131072},
		},
		benchedModels: map[string]map[string]int{
			"sglang": {"Benched": 131072, "Dropped": 131072, "NoWeights": 131072},
		},
		benchExclusions: map[string]map[string]benchVerdict{
			"sglang": {"Dropped": {reason: "bench_dropped", ctx: 131072}},
		},
	}

	got := a.advertisedNames(bs)
	if len(got) != 1 || got[0] != "Benched" {
		t.Fatalf("only the fully-vetted model may be advertised, got %v", got)
	}
	// The serving allowlist is untouched -- this is what keeps benching
	// possible for the three withheld models.
	if len(bs.modelNames) != 4 {
		t.Fatalf("modelNames must NOT be narrowed: %v", bs.modelNames)
	}
}

func TestAdvertisedNames_OllamaSkipsTheWeightCheck(t *testing.T) {
	// Ollama has no directory-backed store; checkModelWeights returns nil
	// for it, so vetting must rest on the bench record alone.
	bs := &backendState{
		config:     backendConfig{Name: "ollama"},
		modelNames: []string{"m:tag", "unbenched:tag"},
	}
	a := &arbiter{
		modelContexts: map[string]map[string]int{"ollama": {"m:tag": 131072}},
		benchedModels: map[string]map[string]int{"ollama": {"m:tag": 131072}},
	}
	got := a.advertisedNames(bs)
	if len(got) != 1 || got[0] != "m:tag" {
		t.Fatalf("want only the benched ollama tag, got %v", got)
	}
}

func TestAdvertisedNames_EmptyBenchIndexAdvertisesNothing(t *testing.T) {
	// Safe direction for a display decision: with no bench data the
	// router lists nothing rather than everything. Serving is unaffected.
	bs := &backendState{
		config:     backendConfig{Name: "ollama"},
		modelNames: []string{"a", "b"},
	}
	a := &arbiter{
		modelContexts: map[string]map[string]int{},
		benchedModels: map[string]map[string]int{},
	}
	if got := a.advertisedNames(bs); len(got) != 0 {
		t.Fatalf("no bench data must advertise nothing, got %v", got)
	}
}

// The narrowing itself, at the handler level: a model that is serveable
// but not vetted must not appear in either listing.
func TestModelsHandler_HidesUnvettedModels(t *testing.T) {
	bs := &backendState{
		config:     backendConfig{Name: "vllm"},
		modelNames: []string{"vetted", "unvetted"}, // both SERVEABLE
		advertised: []string{"vetted"},             // only one ADVERTISED
	}
	a := testArbiter(bs)

	for _, tc := range []struct{ path, field string }{
		{"/v1/models", "data"},
		{"/api/tags", "models"},
	} {
		w := httptest.NewRecorder()
		var h http.HandlerFunc
		if tc.path == "/v1/models" {
			h = a.makeModelsHandler("vllm")
		} else {
			h = a.makeTagsHandler("vllm")
		}
		h(w, httptest.NewRequest("GET", tc.path, nil))
		body := w.Body.String()
		if !strings.Contains(body, "vetted") {
			t.Errorf("%s: vetted model missing: %s", tc.path, body)
		}
		if strings.Contains(body, "unvetted") {
			t.Errorf("%s: un-vetted model must not be advertised: %s",
				tc.path, body)
		}
	}

	// ...and it is still SERVEABLE by explicit name. This is the property
	// that keeps `make bench-*` able to earn a first bench row.
	found := false
	for _, n := range bs.modelNames {
		if n == "unvetted" {
			found = true
		}
	}
	if !found {
		t.Fatal("the serving allowlist must still contain the un-vetted model")
	}
}
