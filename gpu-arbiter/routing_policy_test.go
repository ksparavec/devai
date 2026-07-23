package main

import (
	"context"
	"strings"
	"testing"
	"time"
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

// --- B1: a worker on its way out must stop receiving new work ---

// The head kept routing to a worker that had already acked a drain or
// shutdown: the executor waits out the in-flight requests and (for
// shutdown) exits the process, so anything sent after the ack either
// lands mid-drain or dies with the process.
func TestRouting_SkipsDrainingWorker(t *testing.T) {
	healthy := mkW("w-ready", "Qwen3", 131072, 0)
	healthy.HealthStatus = HealthReady
	draining := mkW("w-draining", "Qwen3", 131072, 0)
	draining.HealthStatus = HealthDraining

	p := &RoutingPolicy{}
	// The draining worker is listed FIRST and is an exact match, so a
	// policy that ignores health status would pick it about half the
	// time (round-robin across the tied exact-match bucket).
	for i := 0; i < 8; i++ {
		d := p.RouteDecision([]WorkerEntry{draining, healthy}, "Qwen3", 65536, "vllm")
		if d.WorkerID != "w-ready" {
			t.Fatalf("iteration %d routed to %q, want w-ready", i, d.WorkerID)
		}
	}
}

func TestRouting_SkipsShuttingDownWorker(t *testing.T) {
	w := mkW("w-bye", "Qwen3", 131072, 0)
	w.HealthStatus = HealthShuttingDown
	p := &RoutingPolicy{}
	d := p.RouteDecision([]WorkerEntry{w}, "Qwen3", 65536, "vllm")
	if d.WorkerID != "" {
		t.Fatalf("routed to %q; a shutting-down worker must not get new work",
			d.WorkerID)
	}
	if !strings.Contains(d.NoFitReason, "draining or shutting down") {
		t.Errorf("NoFitReason = %q, want it to name the drain", d.NoFitReason)
	}
}

// Unknown / unset statuses must NOT remove a worker from the fleet --
// "registered" is what a worker carries before its first heartbeat, and
// "" is what an older worker build sends.
func TestRouting_UnknownHealthStatusStillRoutable(t *testing.T) {
	p := &RoutingPolicy{}
	for _, status := range []string{"", "registered", HealthReady, "weird"} {
		w := mkW("w-1", "Qwen3", 131072, 0)
		w.HealthStatus = status
		d := p.RouteDecision([]WorkerEntry{w}, "Qwen3", 65536, "vllm")
		if d.WorkerID != "w-1" {
			t.Errorf("health_status=%q: routed to %q, want w-1", status, d.WorkerID)
		}
	}
}

func TestWorkerAvailable(t *testing.T) {
	tests := []struct {
		status string
		want   bool
	}{
		{HealthReady, true},
		{"registered", true},
		{"", true},
		{HealthDraining, false},
		{HealthShuttingDown, false},
	}
	for _, tc := range tests {
		if got := workerAvailable(WorkerEntry{HealthStatus: tc.status}); got != tc.want {
			t.Errorf("workerAvailable(%q) = %v, want %v", tc.status, got, tc.want)
		}
	}
}

// --- Round 3 / S1-A: drain must not be a fleet-removing latch ---

// A drain is bounded (DRAIN_TIMEOUT) and the drain holds the arbiter
// mutex, so a request forwarded to a draining worker queues behind the
// drain and is then served. On a single-worker fleet that slow response
// beats a hard 503.
func TestRouting_FallsBackToDrainingWorkerRatherThan503(t *testing.T) {
	w := mkW("w-draining", "Qwen3", 131072, 0)
	w.HealthStatus = HealthDraining
	p := &RoutingPolicy{}
	d := p.RouteDecision([]WorkerEntry{w}, "Qwen3", 65536, "vllm")
	if d.WorkerID != "w-draining" {
		t.Fatalf("routed to %q (reason %q); a single-worker fleet must not 503 "+
			"for the duration of a bounded drain", d.WorkerID, d.NoFitReason)
	}
	if !d.Degraded {
		t.Error("fallback route not flagged Degraded")
	}
}

// The fallback must never admit a shutting-down worker: that process
// exits at the end of its drain, so the request would die with it.
func TestRouting_NoFallbackToShuttingDownWorker(t *testing.T) {
	w := mkW("w-bye", "Qwen3", 131072, 0)
	w.HealthStatus = HealthShuttingDown
	p := &RoutingPolicy{}
	d := p.RouteDecision([]WorkerEntry{w}, "Qwen3", 65536, "vllm")
	if d.WorkerID != "" {
		t.Fatalf("fell back to shutting-down worker %q", d.WorkerID)
	}
}

// Mixed fleet: a draining worker is only a fallback, never a preference.
func TestRouting_DrainingIsFallbackOnlyNotPreference(t *testing.T) {
	draining := mkW("w-draining", "Qwen3", 131072, 0)
	draining.HealthStatus = HealthDraining
	healthy := mkW("w-ready", "other-model", 131072, 0)
	healthy.HealthStatus = HealthReady

	p := &RoutingPolicy{}
	// The draining worker is the exact match; the healthy one is not.
	// Health still wins.
	for i := 0; i < 4; i++ {
		d := p.RouteDecision([]WorkerEntry{draining, healthy}, "Qwen3", 65536, "vllm")
		if d.WorkerID != "w-ready" {
			t.Fatalf("iteration %d routed to %q, want w-ready", i, d.WorkerID)
		}
		if d.Degraded {
			t.Fatal("healthy route incorrectly flagged Degraded")
		}
	}
}

// A drain that has COMPLETED puts the worker back at "ready", and the
// head must route to it again. This is the end-to-end shape of the
// one-way-latch bug: drain -> complete -> routable.
func TestRouting_DrainedWorkerIsRoutableAgainAfterDrainCompletes(t *testing.T) {
	a := arbiterWithBackends(map[string][]string{"vllm": {"Qwen3"}})
	a.drainTimeout = 100 * time.Millisecond
	state := &WorkerState{}
	e := newArbiterCommandExecutor(state, a)
	e.background = func(f func()) { f() }

	if err := e.Execute(context.Background(),
		Command{Type: CommandDrain, Backend: "vllm"}); err != nil {
		t.Fatalf("Execute: %v", err)
	}

	// Feed the worker's own reported status into the head's fleet view.
	w := mkW("w-1", "Qwen3", 131072, 0)
	w.HealthStatus = state.snapshot("w-1").HealthStatus
	p := &RoutingPolicy{}
	d := p.RouteDecision([]WorkerEntry{w}, "Qwen3", 65536, "vllm")
	if d.WorkerID != "w-1" {
		t.Fatalf("worker still unroutable after the drain completed "+
			"(health=%q, reason=%q)", w.HealthStatus, d.NoFitReason)
	}
	if d.Degraded {
		t.Errorf("health=%q routed as Degraded; the drain is over", w.HealthStatus)
	}
}
