package main

import (
	"testing"
)

func mkW(id, model string, ctx, queue int) WorkerEntry {
	return WorkerEntry{
		WorkerID:    id,
		Name:        id,
		Backends:    []string{"vllm"},
		LoadedModel: model,
		LoadedCtx:   ctx,
		QueueDepth:  queue,
	}
}

func TestRouting_NoWorkersReturns503(t *testing.T) {
	p := &RoutingPolicy{}
	d := p.RouteDecision(nil, "Qwen3", 32768, "vllm")
	if d.WorkerID != "" {
		t.Errorf("expected empty worker_id, got %q", d.WorkerID)
	}
	if d.NoFitReason == "" {
		t.Errorf("expected NoFitReason set")
	}
}

func TestRouting_ExactMatchWins(t *testing.T) {
	workers := []WorkerEntry{
		mkW("w-other", "OtherModel", 32768, 0),
		mkW("w-exact", "Qwen3", 131072, 0),
		mkW("w-too-small", "Qwen3", 32768, 0),
	}
	p := &RoutingPolicy{}
	d := p.RouteDecision(workers, "Qwen3", 65536, "vllm")
	if d.WorkerID != "w-exact" {
		t.Errorf("got %q, want w-exact (loaded ctx >= requested)", d.WorkerID)
	}
	if d.Score != ScoreExactMatch {
		t.Errorf("score: got %d, want %d", d.Score, ScoreExactMatch)
	}
}

func TestRouting_RightModelTooSmallCtxBeatsDifferentModel(t *testing.T) {
	workers := []WorkerEntry{
		mkW("w-other", "OtherModel", 131072, 0),
		mkW("w-too-small", "Qwen3", 32768, 0),
	}
	p := &RoutingPolicy{}
	d := p.RouteDecision(workers, "Qwen3", 65536, "vllm")
	if d.WorkerID != "w-too-small" {
		t.Errorf("got %q, want w-too-small", d.WorkerID)
	}
	if d.Score != ScoreRightModelCtx {
		t.Errorf("score: got %d, want %d", d.Score, ScoreRightModelCtx)
	}
}

func TestRouting_IdleBeatsDifferentModel(t *testing.T) {
	workers := []WorkerEntry{
		mkW("w-loaded", "OtherModel", 131072, 0),
		mkW("w-idle", "", 0, 0),
	}
	p := &RoutingPolicy{}
	d := p.RouteDecision(workers, "Qwen3", 65536, "vllm")
	if d.WorkerID != "w-idle" {
		t.Errorf("got %q, want w-idle", d.WorkerID)
	}
	if d.Score != ScoreIdle {
		t.Errorf("score: got %d, want %d", d.Score, ScoreIdle)
	}
}

func TestRouting_DifferentModelLast(t *testing.T) {
	workers := []WorkerEntry{
		mkW("w-other", "OtherModel", 131072, 0),
	}
	p := &RoutingPolicy{}
	d := p.RouteDecision(workers, "Qwen3", 65536, "vllm")
	if d.WorkerID != "w-other" {
		t.Errorf("got %q, want w-other", d.WorkerID)
	}
	if d.Score != ScoreDifferentModel {
		t.Errorf("score: got %d, want %d", d.Score, ScoreDifferentModel)
	}
}

func TestRouting_BackendFilter(t *testing.T) {
	w := mkW("w-1", "Qwen3", 131072, 0)
	w.Backends = []string{"sglang"} // doesn't advertise vllm
	workers := []WorkerEntry{w}
	p := &RoutingPolicy{}
	d := p.RouteDecision(workers, "Qwen3", 32768, "vllm")
	if d.WorkerID != "" {
		t.Errorf("worker advertising sglang should NOT match vllm request")
	}
	d2 := p.RouteDecision(workers, "Qwen3", 32768, "sglang")
	if d2.WorkerID != "w-1" {
		t.Errorf("sglang request should pick w-1, got %q", d2.WorkerID)
	}
}

func TestRouting_QueueDepthThresholdSkips(t *testing.T) {
	workers := []WorkerEntry{
		mkW("w-busy", "Qwen3", 131072, 5),
		mkW("w-idle", "Qwen3", 131072, 0),
	}
	p := &RoutingPolicy{QueueDepthThreshold: 3}
	d := p.RouteDecision(workers, "Qwen3", 65536, "vllm")
	if d.WorkerID != "w-idle" {
		t.Errorf("got %q, want w-idle (busy worker should be skipped)", d.WorkerID)
	}
}

func TestRouting_AllOverloaded503(t *testing.T) {
	workers := []WorkerEntry{
		mkW("w-busy-1", "Qwen3", 131072, 5),
		mkW("w-busy-2", "Qwen3", 131072, 5),
	}
	p := &RoutingPolicy{QueueDepthThreshold: 3}
	d := p.RouteDecision(workers, "Qwen3", 65536, "vllm")
	if d.WorkerID != "" {
		t.Errorf("expected no fit when all overloaded, got %q", d.WorkerID)
	}
	if d.NoFitReason == "" {
		t.Errorf("expected NoFitReason populated")
	}
}

func TestRouting_TiesRoundRobin(t *testing.T) {
	// Two workers tied at exact-match score. Successive requests
	// should alternate via round-robin.
	workers := []WorkerEntry{
		mkW("w-a", "Qwen3", 131072, 0),
		mkW("w-b", "Qwen3", 131072, 0),
	}
	p := &RoutingPolicy{}
	picks := map[string]int{}
	for i := 0; i < 20; i++ {
		d := p.RouteDecision(workers, "Qwen3", 65536, "vllm")
		picks[d.WorkerID]++
	}
	if picks["w-a"] == 0 || picks["w-b"] == 0 {
		t.Errorf("round-robin did not exercise both: %v", picks)
	}
	// Should be roughly balanced; allow one off.
	if abs(picks["w-a"]-picks["w-b"]) > 1 {
		t.Errorf("round-robin imbalanced: %v", picks)
	}
}

func TestScoreWorker_DirectInvariants(t *testing.T) {
	tests := []struct {
		name  string
		w     WorkerEntry
		model string
		ctx   int
		want  int
	}{
		{"exact", mkW("x", "M", 131072, 0), "M", 65536, ScoreExactMatch},
		{"right model too small ctx", mkW("x", "M", 32768, 0), "M", 65536, ScoreRightModelCtx},
		{"idle", mkW("x", "", 0, 0), "M", 65536, ScoreIdle},
		{"different model", mkW("x", "Other", 131072, 0), "M", 65536, ScoreDifferentModel},
		{"exact ctx zero falls back to right-model", mkW("x", "M", 131072, 0), "M", 0, ScoreRightModelCtx},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := scoreWorker(tc.w, tc.model, tc.ctx); got != tc.want {
				t.Errorf("got %d, want %d", got, tc.want)
			}
		})
	}
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
