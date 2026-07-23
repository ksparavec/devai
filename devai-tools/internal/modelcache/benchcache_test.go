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
      "humaneval_plus_subset_50": {"pass@1": 0.7, "ran_at": "2026-06-02T00:00:00Z"},
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

func TestComputeScores(t *testing.T) {
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
	// The fixture's HumanEval+ row carries a LATER ran_at than plain
	// HumanEval, so a bare "humaneval_" prefix would let the max-ran_at
	// tiebreak return 0.7 here -- HumanEval+ reported as HumanEval.
	if scores.Code == nil || *scores.Code != 0.8 {
		t.Errorf("Code = %v, want 0.8 (humaneval_subset_50)", derefOrNil(scores.Code))
	}
	if scores.CodePlus == nil || *scores.CodePlus != 0.7 {
		t.Errorf("CodePlus = %v, want 0.7 (humaneval_plus_subset_50)", derefOrNil(scores.CodePlus))
	}
	if scores.Leak == nil || *scores.Leak != 0.05 {
		t.Errorf("Leak = %v, want 0.05", derefOrNil(scores.Leak))
	}
}

// A row benched before humaneval_plus existed must still report plain
// HumanEval, with CodePlus simply absent -- the two prefixes are
// independent lookups, not a fallback chain.
func TestComputeScoresHumanEvalWithoutPlus(t *testing.T) {
	const onlyPlain = `{
  "base::vllm::32768": {
    "context": 32768,
    "metrics": {},
    "tasks": {"humaneval_subset_50": {"pass@1": 0.62, "ran_at": "2026-05-01T00:00:00Z"}}
  }
}`
	cache, err := LoadBenchCache(writeJSONFixture(t, "bench.json", onlyPlain))
	if err != nil {
		t.Fatal(err)
	}
	scores := ComputeScores(cache["base::vllm::32768"])
	if scores.Code == nil || *scores.Code != 0.62 {
		t.Errorf("Code = %v, want 0.62", derefOrNil(scores.Code))
	}
	if scores.CodePlus != nil {
		t.Errorf("CodePlus = %v, want nil", derefOrNil(scores.CodePlus))
	}
}

// The mirror case: a row with ONLY humaneval_plus must not have that
// score leak into the plain HumanEval field.
func TestComputeScoresPlusDoesNotFillPlainHumanEval(t *testing.T) {
	const onlyPlus = `{
  "base::vllm::32768": {
    "context": 32768,
    "metrics": {},
    "tasks": {"humaneval_plus_subset_50": {"pass@1": 0.55, "ran_at": "2026-05-01T00:00:00Z"}}
  }
}`
	cache, err := LoadBenchCache(writeJSONFixture(t, "bench.json", onlyPlus))
	if err != nil {
		t.Fatal(err)
	}
	scores := ComputeScores(cache["base::vllm::32768"])
	if scores.Code != nil {
		t.Errorf("Code = %v, want nil (only humaneval_plus recorded)", derefOrNil(scores.Code))
	}
	if scores.CodePlus == nil || *scores.CodePlus != 0.55 {
		t.Errorf("CodePlus = %v, want 0.55", derefOrNil(scores.CodePlus))
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
	if scores.Code != nil || scores.CodePlus != nil || scores.Leak != nil {
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
