package main

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestPickCheapest_PicksLowestPerHour(t *testing.T) {
	prices := []CloudPricing{
		{Cloud: "runpod", GPUs: "3090:1", PerHour: 0.44, UseSpot: false},
		{Cloud: "runpod", GPUs: "3090:1", PerHour: 0.20, UseSpot: true},
		{Cloud: "lambda", GPUs: "A100:1", PerHour: 1.10, UseSpot: true},
	}
	p := NewSkyPilotPolicy(prices, "image", "http://head", 0)
	got, err := p.PickCheapest("3090:1")
	if err != nil {
		t.Fatalf("PickCheapest: %v", err)
	}
	if got.PerHour != 0.20 {
		t.Errorf("PerHour: got %v, want 0.20", got.PerHour)
	}
	if !got.UseSpot {
		t.Errorf("expected spot=true at $0.20 entry")
	}
}

func TestPickCheapest_NoMatchReturnsErr(t *testing.T) {
	p := NewSkyPilotPolicy(DefaultPricing, "image", "http://head", 0)
	_, err := p.PickCheapest("UnknownGPU:99")
	if !errors.Is(err, ErrNoCloudFits) {
		t.Errorf("got %v, want ErrNoCloudFits", err)
	}
}

func TestPickCheapest_BudgetExceededRefuses(t *testing.T) {
	prices := []CloudPricing{
		{Cloud: "runpod", GPUs: "H100:1", PerHour: 4.69},
	}
	p := NewSkyPilotPolicy(prices, "image", "http://head", 1.0)
	_, err := p.PickCheapest("H100:1")
	if !errors.Is(err, ErrBudgetExceeded) {
		t.Errorf("got %v, want ErrBudgetExceeded", err)
	}
}

func TestPickCheapest_BudgetSatisfiedAccepts(t *testing.T) {
	prices := []CloudPricing{
		{Cloud: "runpod", GPUs: "3090:1", PerHour: 0.44},
	}
	p := NewSkyPilotPolicy(prices, "image", "http://head", 1.0)
	if _, err := p.PickCheapest("3090:1"); err != nil {
		t.Errorf("PickCheapest: %v", err)
	}
}

func TestBuildLaunchRequest_FillsEnv(t *testing.T) {
	p := NewSkyPilotPolicy(
		[]CloudPricing{{Cloud: "runpod", GPUs: "3090:1", PerHour: 0.44, UseSpot: false}},
		"devai-worker-bootstrap",
		"http://head.lan:11444",
		0,
	)
	req, err := p.BuildLaunchRequest("c-1", "3090:1", "worker-c-1")
	if err != nil {
		t.Fatalf("BuildLaunchRequest: %v", err)
	}
	if req.ClusterName != "c-1" {
		t.Errorf("cluster: %q", req.ClusterName)
	}
	if req.Image != "devai-worker-bootstrap" {
		t.Errorf("image: %q", req.Image)
	}
	if req.Env["DEVAI_HEAD_URL"] != "http://head.lan:11444" {
		t.Errorf("DEVAI_HEAD_URL: %q", req.Env["DEVAI_HEAD_URL"])
	}
	if req.Env["DEVAI_LIFECYCLE"] != string(LifecycleEphemeral) {
		t.Errorf("DEVAI_LIFECYCLE: %q", req.Env["DEVAI_LIFECYCLE"])
	}
	if req.Env["DEVAI_WORKER_NAME"] != "worker-c-1" {
		t.Errorf("DEVAI_WORKER_NAME: %q", req.Env["DEVAI_WORKER_NAME"])
	}
	if req.Env["DEVAI_GPU_TYPE"] != "3090" {
		t.Errorf("DEVAI_GPU_TYPE: %q (expected 3090 prefix)", req.Env["DEVAI_GPU_TYPE"])
	}
}

func TestBuildLaunchRequest_RejectsMissingImage(t *testing.T) {
	p := NewSkyPilotPolicy(
		[]CloudPricing{{Cloud: "runpod", GPUs: "3090:1", PerHour: 0.44}},
		"", // no image
		"http://head", 0,
	)
	_, err := p.BuildLaunchRequest("c-1", "3090:1", "w-1")
	if err == nil {
		t.Errorf("expected error for empty WorkerImage")
	}
}

func TestBuildLaunchRequest_RejectsMissingHeadEndpoint(t *testing.T) {
	p := NewSkyPilotPolicy(
		[]CloudPricing{{Cloud: "runpod", GPUs: "3090:1", PerHour: 0.44}},
		"img", "", 0,
	)
	_, err := p.BuildLaunchRequest("c-1", "3090:1", "w-1")
	if err == nil {
		t.Errorf("expected error for empty HeadEndpoint")
	}
}

func TestDefaultVRAMForGPU(t *testing.T) {
	tests := []struct {
		gpus string
		want string
	}{
		{"3090:1", "24"},
		{"4090:1", "24"},
		{"A100:1", "40"},
		{"A100-80GB:1", "80"},
		{"H100:1", "80"},
		{"H200:1", "141"},
		{"UnknownGPU:1", "24"},
	}
	for _, tc := range tests {
		if got := defaultVRAMForGPU(tc.gpus); got != tc.want {
			t.Errorf("%s: got %s, want %s", tc.gpus, got, tc.want)
		}
	}
}

// --- IdleTeardownCoordinator tests ---

func TestTeardown_MarkAndSweep_GoneFromFleetTriggersDown(t *testing.T) {
	fleet := NewFleetState()
	// Worker registered, then disappears -- not in fleet snapshot.
	c := NewIdleTeardownCoordinator(fleet, nil) // nil sky client = no-op
	c.MarkForTeardown("cluster-a")
	n := c.SweepOnce(context.Background(), time.Now())
	// Without a configured sky client, tryDown returns nil (no-op) and
	// counts as a successful tear-down for accounting purposes.
	if n != 1 {
		t.Errorf("got %d teardowns, want 1", n)
	}
	if _, exists := c.Pending()["cluster-a"]; exists {
		t.Errorf("cluster-a should have been removed from pending")
	}
}

func TestTeardown_DeadlineNotPassedWaits(t *testing.T) {
	fleet := NewFleetState()
	// Register the worker so it's "still alive" in the snapshot.
	id := fleet.Register(mkReq("cluster-a"), time.Now())
	_ = id
	c := NewIdleTeardownCoordinator(fleet, nil)
	c.GraceSecs = 60
	c.MarkForTeardown("cluster-a")
	n := c.SweepOnce(context.Background(), time.Now())
	if n != 0 {
		t.Errorf("got %d teardowns, want 0 (deadline not passed, worker still alive)", n)
	}
	if _, exists := c.Pending()["cluster-a"]; !exists {
		t.Errorf("cluster-a should still be pending")
	}
}

func TestTeardown_DeadlinePassedTriggersDown(t *testing.T) {
	fleet := NewFleetState()
	_ = fleet.Register(mkReq("cluster-a"), time.Now())
	c := NewIdleTeardownCoordinator(fleet, nil)
	c.GraceSecs = 1
	c.MarkForTeardown("cluster-a")
	// Sweep with now = deadline + 5s.
	now := time.Now().Add(10 * time.Second)
	n := c.SweepOnce(context.Background(), now)
	if n != 1 {
		t.Errorf("got %d teardowns, want 1 (deadline passed)", n)
	}
}

func TestTeardown_MarkIsIdempotent(t *testing.T) {
	c := NewIdleTeardownCoordinator(NewFleetState(), nil)
	c.MarkForTeardown("c-1")
	d1 := c.Pending()["c-1"]
	c.MarkForTeardown("c-1")
	d2 := c.Pending()["c-1"]
	if !d1.Equal(d2) {
		t.Errorf("repeat MarkForTeardown changed deadline: %v -> %v", d1, d2)
	}
}

func TestTryDown_NilClientNoOp(t *testing.T) {
	if err := tryDown(context.Background(), nil, "any"); err != nil {
		t.Errorf("nil sky client should be no-op, got %v", err)
	}
}

func TestTryDown_UnconfiguredClientNoOp(t *testing.T) {
	c := NewSkyPilotClient("", nil)
	if err := tryDown(context.Background(), c, "any"); err != nil {
		t.Errorf("unconfigured sky client should be no-op, got %v", err)
	}
}
