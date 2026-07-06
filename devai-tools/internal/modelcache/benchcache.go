package modelcache

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
)

// BenchRow is one deploy/.bench-cache.json row (schema v3). Tasks is keyed
// by e.g. "gsm8k_subset_100", "humaneval_subset_50", "tools_use_20", or the
// special "leak_probe" key; each value carries a "score" or "pass@1" field
// plus "ran_at" for picking the latest of several stale runs at different
// n. Metrics carries "tps_sustained_p50" among other measurements.
type BenchRow struct {
	Context int                       `json:"context"`
	Tasks   map[string]map[string]any `json:"tasks"`
	Metrics map[string]any            `json:"metrics"`
}

// BenchCache is a full bench cache file, top-level row key ->
// row. The real file also carries a "_meta" key holding host_env_history;
// LoadBenchCache drops it since it isn't a row.
type BenchCache map[string]BenchRow

// LoadBenchCache parses deploy/.bench-cache.json. A missing file is not an
// error -- callers treat a nil cache as "no bench data anywhere yet".
func LoadBenchCache(path string) (BenchCache, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}
	cache := make(BenchCache, len(raw))
	for key, msg := range raw {
		if key == "_meta" {
			continue
		}
		var row BenchRow
		if err := json.Unmarshal(msg, &row); err != nil {
			continue // tolerate a malformed row rather than failing the whole load
		}
		cache[key] = row
	}
	return cache, nil
}

// BenchKey builds the bench-cache row key exactly per CLAUDE.md's
// convention: "<repo>@<sha>::<backend>::<ctx>" (HF) or
// "<digest>::<backend>::<ctx>" (Ollama). base is the caller-supplied
// identifier (repo@sha or digest); this function only appends the
// backend/ctx suffix.
func BenchKey(base, backend string, ctx int) string {
	return fmt.Sprintf("%s::%s::%d", base, backend, ctx)
}

// Scores are the four picker-sort scores plus leak rate, computed exactly
// per scripts/model-picker.py's _picker_scores: tps = metrics.
// tps_sustained_p50; code = tasks.humaneval_*.pass@1; reas = 2/3*tools_use
// + 1/3*gsm8k; total = mean(gsm8k, code, tools_use). A nil field means the
// underlying task hasn't been recorded, not zero.
type Scores struct {
	TPS   *float64
	Code  *float64
	Reas  *float64
	Total *float64
	Leak  *float64
}

// ComputeScores derives Scores from a bench row.
func ComputeScores(row BenchRow) Scores {
	var s Scores
	if v, ok := row.Metrics["tps_sustained_p50"]; ok {
		if f, ok2 := toFloat(v); ok2 {
			s.TPS = &f
		}
	}
	s.Code = bestTaskScore(row.Tasks, "humaneval_", "pass@1")
	gsm := bestTaskScore(row.Tasks, "gsm8k_", "score")
	tools := bestTaskScore(row.Tasks, "tools_use", "score")
	if tools != nil && gsm != nil {
		r := (2.0/3.0)*(*tools) + (1.0/3.0)*(*gsm)
		s.Reas = &r
	}
	var parts []float64
	for _, p := range []*float64{gsm, s.Code, tools} {
		if p != nil {
			parts = append(parts, *p)
		}
	}
	if len(parts) > 0 {
		sum := 0.0
		for _, p := range parts {
			sum += p
		}
		t := sum / float64(len(parts))
		s.Total = &t
	}
	if leakProbe, ok := row.Tasks["leak_probe"]; ok {
		if v, ok2 := leakProbe["leak_rate"]; ok2 {
			if f, ok3 := toFloat(v); ok3 {
				s.Leak = &f
			}
		}
	}
	return s
}

// OtherContexts returns the sorted, deduplicated ctx values the cache has
// for base+backend at any ctx other than the one that just missed --
// feeds the "Bench: not available ... (have ...)" message.
func OtherContexts(cache BenchCache, base, backend string) []int {
	prefix := base + "::" + backend + "::"
	seen := map[int]bool{}
	var out []int
	for key, row := range cache {
		if !strings.HasPrefix(key, prefix) {
			continue
		}
		if !seen[row.Context] {
			seen[row.Context] = true
			out = append(out, row.Context)
		}
	}
	sort.Ints(out)
	return out
}

func bestTaskScore(tasks map[string]map[string]any, prefix, key string) *float64 {
	var best map[string]any
	for name, data := range tasks {
		if !strings.HasPrefix(name, prefix) {
			continue
		}
		if best == nil {
			best = data
			continue
		}
		curTS, _ := data["ran_at"].(string)
		bestTS, _ := best["ran_at"].(string)
		if curTS > bestTS {
			best = data
		}
	}
	if best == nil {
		return nil
	}
	v, ok := best[key]
	if !ok {
		return nil
	}
	f, ok := toFloat(v)
	if !ok {
		return nil
	}
	return &f
}

func toFloat(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case json.Number:
		f, err := n.Float64()
		return f, err == nil
	default:
		return 0, false
	}
}
