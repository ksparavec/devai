package modelcache

import (
	"testing"
)

const sampleBenchCache = `{
  "_meta": {"current_host_env_id": "abc"},
  "nvidia/Qwen3-8B-NVFP4@abc123def456::vllm::131072": {
    "context": 131072,
    "metrics": {"tps_sustained_p50": 98.3},
    "tasks": {
      "gsm8k_subset_100": {"score": 0.9, "ran_at": "2026-06-01T00:00:00Z"},
      "humaneval_subset_50": {"pass@1": 0.8, "ran_at": "2026-06-01T00:00:00Z"},
      "tools_use_20": {"score": 1.0, "ran_at": "2026-06-01T00:00:00Z"},
      "leak_probe": {"leak_rate": 0.05, "n_prompts": 40}
    }
  },
  "nvidia/Qwen3-8B-NVFP4@abc123def456::vllm::32768": {
    "context": 32768,
    "metrics": {"tps_sustained_p50": 110.0},
    "tasks": {}
  }
}`

func TestLoadBenchCacheSkipsMeta(t *testing.T) {
	path := writeJSONFixture(t, "bench.json", sampleBenchCache)
	cache, err := LoadBenchCache(path)
	if err != nil {
		t.Fatalf("LoadBenchCache: %v", err)
	}
	if _, ok := cache["_meta"]; ok {
		t.Error("_meta should not appear as a row")
	}
	if len(cache) != 2 {
		t.Fatalf("got %d rows, want 2", len(cache))
	}
}

func TestBenchKeyMatchesConvention(t *testing.T) {
	got := BenchKey("nvidia/Qwen3-8B-NVFP4@abc123def456", "vllm", 131072)
	want := "nvidia/Qwen3-8B-NVFP4@abc123def456::vllm::131072"
	if got != want {
		t.Errorf("BenchKey = %q, want %q", got, want)
	}
}

func TestComputeScoresMatchesPickerFormula(t *testing.T) {
	path := writeJSONFixture(t, "bench.json", sampleBenchCache)
	cache, err := LoadBenchCache(path)
	if err != nil {
		t.Fatal(err)
	}
	row := cache[BenchKey("nvidia/Qwen3-8B-NVFP4@abc123def456", "vllm", 131072)]
	scores := ComputeScores(row)

	if scores.TPS == nil || *scores.TPS != 98.3 {
		t.Errorf("TPS = %v, want 98.3", derefOrNil(scores.TPS))
	}
	if scores.Code == nil || *scores.Code != 0.8 {
		t.Errorf("Code = %v, want 0.8", derefOrNil(scores.Code))
	}
	// reas = 2/3*tools + 1/3*gsm8k = 2/3*1.0 + 1/3*0.9 = 0.9666...
	// Computed through float64 variables, not constant literals -- Go
	// constant-folds an all-literal expression at arbitrary precision and
	// rounds once, which differs in the last bit from the runtime
	// (round-per-operation) arithmetic ComputeScores actually performs.
	gsmVal, codeVal, toolsVal := 0.9, 0.8, 1.0
	wantReas := (2.0/3.0)*toolsVal + (1.0/3.0)*gsmVal
	if scores.Reas == nil || *scores.Reas != wantReas {
		t.Errorf("Reas = %v, want %v", derefOrNil(scores.Reas), wantReas)
	}
	// total = mean(gsm8k, code, tools) = mean(0.9, 0.8, 1.0)
	wantTotal := (gsmVal + codeVal + toolsVal) / 3.0
	if scores.Total == nil || *scores.Total != wantTotal {
		t.Errorf("Total = %v, want %v", derefOrNil(scores.Total), wantTotal)
	}
	if scores.Leak == nil || *scores.Leak != 0.05 {
		t.Errorf("Leak = %v, want 0.05", derefOrNil(scores.Leak))
	}
}

func TestComputeScoresEmptyTasksYieldsNils(t *testing.T) {
	path := writeJSONFixture(t, "bench.json", sampleBenchCache)
	cache, err := LoadBenchCache(path)
	if err != nil {
		t.Fatal(err)
	}
	row := cache[BenchKey("nvidia/Qwen3-8B-NVFP4@abc123def456", "vllm", 32768)]
	scores := ComputeScores(row)
	if scores.TPS == nil || *scores.TPS != 110.0 {
		t.Errorf("TPS = %v, want 110.0", derefOrNil(scores.TPS))
	}
	if scores.Code != nil || scores.Reas != nil || scores.Total != nil || scores.Leak != nil {
		t.Errorf("expected all quality scores nil for a row with no tasks, got %+v", scores)
	}
}

func TestOtherContexts(t *testing.T) {
	path := writeJSONFixture(t, "bench.json", sampleBenchCache)
	cache, err := LoadBenchCache(path)
	if err != nil {
		t.Fatal(err)
	}
	got := OtherContexts(cache, "nvidia/Qwen3-8B-NVFP4@abc123def456", "vllm")
	want := []int{32768, 131072}
	if len(got) != len(want) {
		t.Fatalf("OtherContexts = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("OtherContexts[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

// derefOrNil renders a *float64 as its value (or "nil") in test failure
// messages -- printing the pointer itself (%v on *float64) is useless noise.
func derefOrNil(f *float64) any {
	if f == nil {
		return nil
	}
	return *f
}

func TestLoadBenchCacheAgainstRepoFixture(t *testing.T) {
	path := repoFixturePath(t, "bench-cache.json")
	cache, err := LoadBenchCache(path)
	if err != nil {
		t.Fatalf("LoadBenchCache(repo fixture): %v", err)
	}
	if len(cache) == 0 {
		t.Fatal("expected at least one row from the repo fixture")
	}
}
