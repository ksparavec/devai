//go:build devai_frozen_cluster

package main

import (
	"testing"
	"time"
)

func mkReq(name string) RegisterRequest {
	return RegisterRequest{
		Name:           name,
		Lifecycle:      LifecyclePersistent,
		GPUType:        "RTX4000",
		VRAMGB:         24,
		Backends:       []string{"vllm", "sglang"},
		ArbiterVersion: "v0.1",
		Endpoint:       "http://" + name + ":11444",
	}
}

func TestFleetState_RegisterAssignsID(t *testing.T) {
	f := NewFleetState()
	id := f.Register(mkReq("worker-a"), time.Now())
	if id == "" {
		t.Fatalf("expected non-empty worker_id")
	}
	if f.Count() != 1 {
		t.Errorf("expected 1 worker, got %d", f.Count())
	}
	got, ok := f.Get(id)
	if !ok {
		t.Fatalf("Get(%q) missing", id)
	}
	if got.Name != "worker-a" {
		t.Errorf("name: got %q", got.Name)
	}
}

func TestFleetState_RegisterReplacesByName(t *testing.T) {
	f := NewFleetState()
	id1 := f.Register(mkReq("worker-a"), time.Now())
	id2 := f.Register(mkReq("worker-a"), time.Now())
	if id1 == id2 {
		t.Errorf("re-registration should mint a new id")
	}
	if f.Count() != 1 {
		t.Errorf("expected 1 worker after replace, got %d", f.Count())
	}
	if _, ok := f.Get(id1); ok {
		t.Errorf("old id %q should be gone", id1)
	}
	if _, ok := f.Get(id2); !ok {
		t.Errorf("new id %q should be present", id2)
	}
}

func TestFleetState_HeartbeatUpdatesState(t *testing.T) {
	f := NewFleetState()
	id := f.Register(mkReq("w"), time.Now())
	res := f.Heartbeat(HeartbeatRequest{
		WorkerID:       id,
		LoadedModel:    "Qwen3-8B-NVFP4",
		LoadedCtx:      131072,
		QueueDepth:     2,
		UtilizationPct: 50,
		HealthStatus:   "ready",
		Counter:        1,
	}, time.Now())
	if res != HeartbeatAccepted {
		t.Fatalf("got %v, want HeartbeatAccepted", res)
	}
	w, _ := f.Get(id)
	if w.LoadedModel != "Qwen3-8B-NVFP4" {
		t.Errorf("loaded model: %q", w.LoadedModel)
	}
	if w.LoadedCtx != 131072 {
		t.Errorf("loaded ctx: %d", w.LoadedCtx)
	}
	if w.QueueDepth != 2 {
		t.Errorf("queue depth: %d", w.QueueDepth)
	}
}

func TestFleetState_HeartbeatRejectsStaleCounter(t *testing.T) {
	f := NewFleetState()
	id := f.Register(mkReq("w"), time.Now())
	_ = f.Heartbeat(HeartbeatRequest{WorkerID: id, Counter: 5, HealthStatus: "ready"}, time.Now())
	res := f.Heartbeat(HeartbeatRequest{WorkerID: id, Counter: 3, HealthStatus: "ready"}, time.Now())
	if res != HeartbeatStale {
		t.Errorf("got %v, want HeartbeatStale", res)
	}
	res2 := f.Heartbeat(HeartbeatRequest{WorkerID: id, Counter: 6, HealthStatus: "ready"}, time.Now())
	if res2 != HeartbeatAccepted {
		t.Errorf("got %v after monotonic resume, want HeartbeatAccepted", res2)
	}
}

func TestFleetState_HeartbeatUnknownWorker(t *testing.T) {
	f := NewFleetState()
	res := f.Heartbeat(HeartbeatRequest{WorkerID: "bogus", Counter: 1}, time.Now())
	if res != HeartbeatUnknownWorker {
		t.Errorf("got %v, want HeartbeatUnknownWorker", res)
	}
}

func TestFleetState_ExpireOlderThan(t *testing.T) {
	f := NewFleetState()
	t0 := time.Now()
	id1 := f.Register(mkReq("old"), t0.Add(-2*time.Minute))
	id2 := f.Register(mkReq("new"), t0)
	cutoff := t0.Add(-1 * time.Minute)
	n := f.ExpireOlderThan(cutoff)
	if n != 1 {
		t.Errorf("expected 1 expired, got %d", n)
	}
	if _, ok := f.Get(id1); ok {
		t.Errorf("old worker not removed")
	}
	if _, ok := f.Get(id2); !ok {
		t.Errorf("new worker erroneously removed")
	}
}

func TestFleetState_SnapshotIsSortedByName(t *testing.T) {
	f := NewFleetState()
	f.Register(mkReq("zeta"), time.Now())
	f.Register(mkReq("alpha"), time.Now())
	f.Register(mkReq("middle"), time.Now())
	snap := f.Snapshot()
	if len(snap) != 3 {
		t.Fatalf("snapshot len: %d", len(snap))
	}
	for i := 1; i < len(snap); i++ {
		if snap[i-1].Name >= snap[i].Name {
			t.Errorf("snapshot not sorted: %v", snap)
		}
	}
}

func TestFleetState_SnapshotDeepCopiesBackends(t *testing.T) {
	f := NewFleetState()
	id := f.Register(mkReq("w"), time.Now())
	snap := f.Snapshot()
	snap[0].Backends[0] = "MUTATED"
	w, _ := f.Get(id)
	if w.Backends[0] == "MUTATED" {
		t.Errorf("snapshot mutation leaked into FleetState")
	}
}