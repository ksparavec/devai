//go:build devai_frozen_cluster

package main

import (
	"context"
	"errors"
	"net/http"
	"sync/atomic"
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
		{"L4:1", "24"},
		// L40S is a 48 GB card, not 24 -- advertising 24 hides half
		// the VRAM from the head's fit decisions.
		{"L40S:1", "48"},
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

// wid returns the worker_id FleetState assigned to `name`, which is the
// instance identity MarkForTeardown keys on.
func wid(t *testing.T, f *FleetState, name string) string {
	t.Helper()
	for _, w := range f.Snapshot() {
		if w.Name == name {
			return w.WorkerID
		}
	}
	t.Fatalf("no worker named %q in fleet", name)
	return ""
}

func TestTeardown_MarkAndSweep_GoneFromFleetTriggersDown(t *testing.T) {
	fleet := NewFleetState()
	// Worker registered, then disappears -- not in fleet snapshot.
	c := NewIdleTeardownCoordinator(fleet, nil) // nil sky client = no-op
	c.MarkForTeardown("cluster-a", "inst-1")
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
	_ = fleet.Register(mkReq("cluster-a"), time.Now())
	c := NewIdleTeardownCoordinator(fleet, nil)
	c.GraceSecs = 60
	c.MarkForTeardown("cluster-a", wid(t, fleet, "cluster-a"))
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
	c.MarkForTeardown("cluster-a", wid(t, fleet, "cluster-a"))
	// Sweep with now = deadline + 5s.
	now := time.Now().Add(10 * time.Second)
	n := c.SweepOnce(context.Background(), now)
	if n != 1 {
		t.Errorf("got %d teardowns, want 1 (deadline passed)", n)
	}
}

func TestTeardown_MarkIsIdempotentPerInstance(t *testing.T) {
	c := NewIdleTeardownCoordinator(NewFleetState(), nil)
	c.MarkForTeardown("c-1", "inst-1")
	d1 := c.Pending()["c-1"]
	c.MarkForTeardown("c-1", "inst-1")
	d2 := c.Pending()["c-1"]
	if !d1.Equal(d2) {
		t.Errorf("repeat MarkForTeardown changed deadline: %v -> %v", d1, d2)
	}
	if got := len(c.Entries()); got != 1 {
		t.Errorf("Entries() = %d, want 1", got)
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

// downFailsTransport makes every SkyPilot call fail, so tryDown
// returns an error and the coordinator must keep the cluster pending.
type downFailsTransport struct{ calls atomic.Int32 }

func (d *downFailsTransport) Do(*http.Request) (*http.Response, error) {
	d.calls.Add(1)
	return nil, errors.New("simulated sky api outage")
}

func failingSkyClient(t *testing.T) (*SkyPilotClient, *downFailsTransport) {
	t.Helper()
	tr := &downFailsTransport{}
	c := NewSkyPilotClient("http://sky.invalid", tokenStoreFor(t, "tok"))
	c.Client = tr
	return c, tr
}

func TestTeardown_FailedDownStaysPendingAndRetries(t *testing.T) {
	// A dropped pending entry means nothing ever revisits the cluster
	// -- the cloud VM keeps billing forever.
	sky, tr := failingSkyClient(t)
	c := NewIdleTeardownCoordinator(NewFleetState(), sky)
	c.GraceSecs = 1
	c.MarkForTeardown("cluster-a", "inst-1")

	now := time.Now().Add(time.Minute)
	if n := c.SweepOnce(context.Background(), now); n != 0 {
		t.Errorf("failed Down counted %d teardowns, want 0", n)
	}
	if _, exists := c.Pending()["cluster-a"]; !exists {
		t.Fatal("cluster-a dropped from pending despite a failed sky down -- " +
			"the cloud VM would bill forever")
	}
	// A later sweep (past the backoff window) must actually retry it.
	if n := c.SweepOnce(context.Background(), now.Add(time.Hour)); n != 0 {
		t.Errorf("second sweep counted %d teardowns, want 0", n)
	}
	if tr.calls.Load() != 2 {
		t.Errorf("sky down attempted %d times across 2 sweeps, want 2", tr.calls.Load())
	}
}

func TestTeardown_FailedDownDoesNotBlockOtherClusters(t *testing.T) {
	sky, _ := failingSkyClient(t)
	c := NewIdleTeardownCoordinator(NewFleetState(), sky)
	c.GraceSecs = 1
	c.MarkForTeardown("cluster-a", "inst-a")
	c.MarkForTeardown("cluster-b", "inst-b")

	now := time.Now().Add(time.Minute)
	_ = c.SweepOnce(context.Background(), now)
	pending := c.Pending()
	if len(pending) != 2 {
		t.Fatalf("pending = %v, want both clusters retained after failures", pending)
	}
}

// --- Round 3 / S1-B: a failing teardown must never abandon a VM ---

// Round 2 bounded the retry at 5 attempts x 10s ~= 50 seconds, then
// DELETED the entry. A SkyPilot API-server restart is longer than that,
// and the deleted entry means nothing ever tears the VM down again.
func TestTeardown_PermanentFailureIsRetriedForeverNeverDropped(t *testing.T) {
	sky, tr := failingSkyClient(t)
	c := NewIdleTeardownCoordinator(NewFleetState(), sky)
	c.GraceSecs = 1
	c.MarkForTeardown("cluster-a", "inst-1")

	base := time.Now().Add(time.Minute)
	// Sweep far into the future each time so the backoff window is
	// always open and every sweep is a real attempt.
	for i := 1; i <= 50; i++ {
		_ = c.SweepOnce(context.Background(), base.Add(time.Duration(i)*time.Hour))
		if _, exists := c.Pending()["cluster-a"]; !exists {
			t.Fatalf("entry dropped after %d failure(s); the cloud VM would "+
				"bill forever with nothing left to tear it down", i)
		}
	}
	if got := tr.calls.Load(); got != 50 {
		t.Errorf("sky down attempted %d times over 50 due sweeps, want 50", got)
	}
	if got := c.Failures("cluster-a"); got != 50 {
		t.Errorf("failures = %d, want 50", got)
	}
}

// The retry interval must GROW (that is what rate-limits the log), and
// must be capped so the entry is still revisited regularly.
func TestTeardown_BackoffGrowsAndCaps(t *testing.T) {
	sky, tr := failingSkyClient(t)
	c := NewIdleTeardownCoordinator(NewFleetState(), sky)
	c.GraceSecs = 0
	c.RetryBase = time.Second
	c.RetryCap = 8 * time.Second
	c.MarkForTeardown("cluster-a", "inst-1")

	now := time.Now().Add(time.Minute)
	want := []time.Duration{time.Second, 2 * time.Second, 4 * time.Second,
		8 * time.Second, 8 * time.Second}
	for i, w := range want {
		_ = c.SweepOnce(context.Background(), now)
		e := c.Entries()[0]
		if got := e.NextAttempt.Sub(now); got != w {
			t.Fatalf("attempt %d: next retry in %v, want %v", i+1, got, w)
		}
		// Not due yet: a sweep before nextAttempt must NOT attempt (and
		// therefore must not log).
		before := tr.calls.Load()
		_ = c.SweepOnce(context.Background(), now.Add(w/2))
		if tr.calls.Load() != before {
			t.Fatalf("attempt %d: swept inside the backoff window", i+1)
		}
		now = e.NextAttempt
	}
}

// The backoff window is the log rate limiter: across a long stretch of
// wall-clock the number of ATTEMPTS (and thus log lines) is bounded by
// elapsed/RetryCap, not by elapsed/SweepEvery.
func TestTeardown_AttemptsAreRateLimitedByBackoff(t *testing.T) {
	sky, tr := failingSkyClient(t)
	c := NewIdleTeardownCoordinator(NewFleetState(), sky)
	c.GraceSecs = 0
	c.RetryBase = 10 * time.Second
	c.RetryCap = 15 * time.Minute
	c.MarkForTeardown("cluster-a", "inst-1")

	start := time.Now()
	// One hour of 10s sweeps = 360 sweeps. Unbounded retry-every-sweep
	// would be 360 attempts and 360 log lines.
	for i := 0; i < 360; i++ {
		_ = c.SweepOnce(context.Background(), start.Add(time.Duration(i)*10*time.Second))
	}
	got := tr.calls.Load()
	if got > 12 {
		t.Fatalf("%d attempts in one hour; the backoff is not rate-limiting "+
			"(a sweep-rate retry would be 360)", got)
	}
	if got < 4 {
		t.Fatalf("%d attempts in one hour; the entry is barely being revisited", got)
	}
	if _, exists := c.Pending()["cluster-a"]; !exists {
		t.Fatal("entry dropped -- the VM would keep billing")
	}
}

// A zero-value coordinator must not read MaxAttempts==0 as "give up on
// the first failure".
func TestTeardown_ZeroValueCoordinatorIsSafe(t *testing.T) {
	sky, _ := failingSkyClient(t)
	c := &IdleTeardownCoordinator{Fleet: NewFleetState(), SkyClient: sky}
	c.MarkForTeardown("cluster-a", "inst-1") // must lazily init the map
	if n := c.SweepOnce(context.Background(), time.Now().Add(time.Hour)); n != 0 {
		t.Fatalf("got %d teardowns, want 0", n)
	}
	if _, exists := c.Pending()["cluster-a"]; !exists {
		t.Fatal("zero-value coordinator dropped a BILLING VM on the first failure")
	}
	e := c.Entries()[0]
	if e.NextAttempt.IsZero() {
		t.Fatal("no backoff scheduled")
	}
	// Zero MaxAge / MaxAttempts equivalents must resolve to the defaults.
	if got := c.retryBase(); got != DefaultTeardownRetryBase {
		t.Errorf("retryBase = %v, want %v", got, DefaultTeardownRetryBase)
	}
	if got := c.retryCap(); got != DefaultTeardownRetryCap {
		t.Errorf("retryCap = %v, want %v", got, DefaultTeardownRetryCap)
	}
	if got := c.sweepEvery(); got != HeartbeatInterval {
		t.Errorf("sweepEvery = %v, want %v", got, HeartbeatInterval)
	}
}

// A zero-value coordinator with a nil Fleet must not panic in SweepOnce.
func TestTeardown_NilFleetDoesNotPanic(t *testing.T) {
	c := &IdleTeardownCoordinator{}
	c.MarkForTeardown("cluster-a", "inst-1")
	if n := c.SweepOnce(context.Background(), time.Now().Add(time.Hour)); n != 1 {
		t.Fatalf("got %d teardowns, want 1 (nil sky client is a no-op success)", n)
	}
}

// The booby trap: a stuck entry is now retried FOREVER, so a later
// cluster reusing the name must get its own entry rather than inheriting
// the stuck one's long-elapsed deadline.
func TestTeardown_NameReuseGetsItsOwnEntry(t *testing.T) {
	sky, _ := failingSkyClient(t)
	fleet := NewFleetState()
	c := NewIdleTeardownCoordinator(fleet, sky)
	c.GraceSecs = 600
	c.MarkForTeardown("cluster-a", "inst-old")

	// The old entry fails and stays pending forever.
	_ = c.SweepOnce(context.Background(), time.Now().Add(time.Hour))
	if c.Failures("cluster-a") == 0 {
		t.Fatal("old entry never attempted")
	}

	// The name is reused by a new, live worker, which is then also
	// marked for teardown with its own (grace-fresh) deadline.
	_ = fleet.Register(mkReq("cluster-a"), time.Now())
	newID := wid(t, fleet, "cluster-a")
	c.MarkForTeardown("cluster-a", newID)

	entries := c.Entries()
	if len(entries) != 2 {
		t.Fatalf("Entries() = %d, want 2 (old stuck instance + new one)", len(entries))
	}
	// A sweep NOW must not tear down the new cluster: its own deadline
	// is 600s out and it is alive.
	if n := c.SweepOnce(context.Background(), time.Now()); n != 0 {
		t.Fatalf("%d teardown(s) on a freshly marked, still-alive cluster; the "+
			"new entry inherited a stale deadline", n)
	}
}

// And the old stuck entry must NOT tear down the live cluster that took
// over its name: `sky down <name>` would kill the new VM.
func TestTeardown_StuckEntryDoesNotKillTheClusterThatReusedItsName(t *testing.T) {
	sky, tr := failingSkyClient(t)
	fleet := NewFleetState()
	c := NewIdleTeardownCoordinator(fleet, sky)
	c.GraceSecs = 1
	c.MarkForTeardown("cluster-a", "inst-old")

	// Old entry fails once, stays pending.
	_ = c.SweepOnce(context.Background(), time.Now().Add(time.Minute))
	attemptsBefore := tr.calls.Load()

	// A NEW registration takes over the name.
	_ = fleet.Register(mkReq("cluster-a"), time.Now())

	_ = c.SweepOnce(context.Background(), time.Now().Add(24*time.Hour))
	if tr.calls.Load() != attemptsBefore {
		t.Fatal("attempted `sky down cluster-a` while a DIFFERENT live worker " +
			"held that name -- it would have killed the new cluster")
	}
	stuck := c.StuckEntries()
	if len(stuck) != 1 || !stuck[0].Conflicted {
		t.Fatalf("StuckEntries() = %+v, want one conflicted entry for the "+
			"operator to reconcile", stuck)
	}
}

// Repeatedly failing entries have to be visible to an operator.
func TestTeardown_StuckEntriesAreExposed(t *testing.T) {
	sky, _ := failingSkyClient(t)
	c := NewIdleTeardownCoordinator(NewFleetState(), sky)
	c.GraceSecs = 0
	c.StuckAfterFailures = 3
	c.MarkForTeardown("cluster-a", "inst-1")

	now := time.Now().Add(time.Minute)
	for i := 0; i < 2; i++ {
		_ = c.SweepOnce(context.Background(), now)
		now = c.Entries()[0].NextAttempt
	}
	if got := len(c.StuckEntries()); got != 0 {
		t.Fatalf("StuckEntries() = %d after 2 failures, want 0 (threshold 3)", got)
	}
	_ = c.SweepOnce(context.Background(), now)
	stuck := c.StuckEntries()
	if len(stuck) != 1 {
		t.Fatalf("StuckEntries() = %+v, want 1 after crossing the threshold", stuck)
	}
	if stuck[0].Cluster != "cluster-a" || stuck[0].Instance != "inst-1" {
		t.Errorf("stuck entry = %+v, want cluster-a/inst-1", stuck[0])
	}
	if stuck[0].LastError == "" {
		t.Error("stuck entry carries no last error for the operator")
	}
}

// A successful Down still retires the entry.
func TestTeardown_SuccessRetiresEntry(t *testing.T) {
	c := NewIdleTeardownCoordinator(NewFleetState(), nil)
	c.GraceSecs = 1
	c.MarkForTeardown("cluster-a", "inst-1")
	if n := c.SweepOnce(context.Background(), time.Now().Add(time.Minute)); n != 1 {
		t.Fatalf("got %d teardowns, want 1", n)
	}
	if _, exists := c.Pending()["cluster-a"]; exists {
		t.Error("successful teardown left the entry pending")
	}
	if got := c.Failures("cluster-a"); got != 0 {
		t.Errorf("Failures on a retired entry = %d, want 0", got)
	}
}

// An empty instance keeps the legacy name-only liveness semantics.
func TestTeardown_EmptyInstanceUsesNameOnlyLiveness(t *testing.T) {
	fleet := NewFleetState()
	_ = fleet.Register(mkReq("cluster-a"), time.Now())
	c := NewIdleTeardownCoordinator(fleet, nil)
	c.GraceSecs = 600
	c.MarkForTeardown("cluster-a", "")
	// Alive under that name and deadline not passed -> no teardown.
	if n := c.SweepOnce(context.Background(), time.Now()); n != 0 {
		t.Fatalf("got %d teardowns, want 0", n)
	}
	// Deadline passed -> torn down (no identity to distinguish).
	if n := c.SweepOnce(context.Background(), time.Now().Add(time.Hour)); n != 1 {
		t.Fatalf("got %d teardowns, want 1", n)
	}
}