// SkyPilot provisioning policy + idle-teardown coordinator.
//
// Per docs/plans/skypilot-fleet-provisioner.md Phase 2:
//   - When a request arrives at the head and no registered worker
//     fits, consult policy to pick the cheapest cloud + GPU type
//     that fits this (model, ctx, backend), call Launch, block (or
//     202) the request until the worker registers.
//   - Background loop: ephemeral workers idle for IdleMinutes get
//     a two-step graceful teardown -- send `shutdown` via
//     heartbeat, then sky down.

package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"sort"
	"strings"
	"sync"
	"time"
)

// CloudPricing models the per-GPU-class cost across clouds. The
// policy picks the cheapest entry for a given GPU class. Prices in
// USD per hour. Wired from a static map today; future work loads
// this from an upstream pricing feed.
type CloudPricing struct {
	Cloud   string
	GPUs    string // e.g. "3090:1", "A100:1"
	Region  string // optional region hint
	UseSpot bool
	PerHour float64 // USD/hr
}

// DefaultPricing is the project's static pricing table for the
// fleet provisioner. Operators override via NewSkyPilotPolicy's
// prices argument when they want to constrain cloud choice.
var DefaultPricing = []CloudPricing{
	// Spot rates first (lowest cost-per-hour). The provisioning
	// step iterates this list in the order returned and picks the
	// first match for the requested GPU class.
	{Cloud: "runpod", GPUs: "3090:1", UseSpot: true, PerHour: 0.20},
	{Cloud: "runpod", GPUs: "4090:1", UseSpot: true, PerHour: 0.34},
	{Cloud: "runpod", GPUs: "3090:1", UseSpot: false, PerHour: 0.44},
	{Cloud: "lambda", GPUs: "A100:1", UseSpot: true, PerHour: 1.10},
	{Cloud: "lambda", GPUs: "A100:1", UseSpot: false, PerHour: 1.29},
	{Cloud: "runpod", GPUs: "H100:1", UseSpot: true, PerHour: 1.99},
	{Cloud: "runpod", GPUs: "H100:1", UseSpot: false, PerHour: 4.69},
}

// SkyPilotPolicy maps a (model, ctx, backend) request to a
// LaunchRequest. Today it's a thin lookup -- future work plugs in
// real probe-cache fit data so the GPU class chosen actually fits
// the (model, ctx).
type SkyPilotPolicy struct {
	Pricing      []CloudPricing
	WorkerImage  string // worker-bootstrap image to launch
	HeadEndpoint string // DEVAI_HEAD_URL exported into the worker

	// MaxBudgetPerLaunchUSD: refuse any launch whose hourly rate
	// exceeds this. 0 disables.
	MaxBudgetPerLaunchUSD float64
}

// NewSkyPilotPolicy returns a policy with sane defaults. Operators
// inject their own pricing via the argument; nil uses DefaultPricing.
func NewSkyPilotPolicy(
	prices []CloudPricing,
	workerImage, headEndpoint string,
	maxBudgetPerLaunchUSD float64,
) *SkyPilotPolicy {
	if prices == nil {
		prices = DefaultPricing
	}
	return &SkyPilotPolicy{
		Pricing:               prices,
		WorkerImage:           workerImage,
		HeadEndpoint:          headEndpoint,
		MaxBudgetPerLaunchUSD: maxBudgetPerLaunchUSD,
	}
}

// PickCheapest returns the cheapest pricing entry that matches the
// requested GPU class. Empty string in `gpuClass` matches any GPU.
// Returns ErrNoCloudFits when no entry matches.
func (p *SkyPilotPolicy) PickCheapest(gpuClass string) (*CloudPricing, error) {
	matches := make([]CloudPricing, 0, len(p.Pricing))
	for _, e := range p.Pricing {
		if gpuClass == "" || e.GPUs == gpuClass {
			matches = append(matches, e)
		}
	}
	if len(matches) == 0 {
		return nil, ErrNoCloudFits
	}
	sort.Slice(matches, func(i, j int) bool {
		return matches[i].PerHour < matches[j].PerHour
	})
	cheapest := matches[0]
	if p.MaxBudgetPerLaunchUSD > 0 && cheapest.PerHour > p.MaxBudgetPerLaunchUSD {
		return nil, fmt.Errorf(
			"%w: cheapest match %s @ $%.2f/hr exceeds budget $%.2f/hr",
			ErrBudgetExceeded,
			cheapest.GPUs, cheapest.PerHour, p.MaxBudgetPerLaunchUSD,
		)
	}
	return &cheapest, nil
}

// ErrNoCloudFits surfaces "no cloud entry matches the requested GPU
// class." Callers (the head's frontend handler) translate this to
// 503 with a meaningful message.
var ErrNoCloudFits = errors.New("no cloud entry matches the requested GPU class")

// ErrBudgetExceeded surfaces a launch refusal when the cheapest
// matching cloud exceeds MaxBudgetPerLaunchUSD.
var ErrBudgetExceeded = errors.New("launch refused: budget exceeded")

// BuildLaunchRequest assembles a LaunchRequest for the chosen
// pricing entry. The worker-bootstrap image starts the arbiter in
// --mode=worker; cloud-init reads the env block.
func (p *SkyPilotPolicy) BuildLaunchRequest(
	clusterName, gpuClass, workerName string,
) (*LaunchRequest, error) {
	choice, err := p.PickCheapest(gpuClass)
	if err != nil {
		return nil, err
	}
	if p.WorkerImage == "" {
		return nil, errors.New("WorkerImage is required")
	}
	if p.HeadEndpoint == "" {
		return nil, errors.New("HeadEndpoint is required")
	}
	return &LaunchRequest{
		ClusterName: clusterName,
		Cloud:       choice.Cloud,
		GPUs:        choice.GPUs,
		Region:      choice.Region,
		UseSpot:     choice.UseSpot,
		Image:       p.WorkerImage,
		Env: map[string]string{
			"DEVAI_MODE":        "worker",
			"DEVAI_HEAD_URL":    p.HeadEndpoint,
			"DEVAI_WORKER_NAME": workerName,
			"DEVAI_LIFECYCLE":   string(LifecycleEphemeral),
			"DEVAI_GPU_TYPE":    strings.SplitN(choice.GPUs, ":", 2)[0],
			"GPU_MEMORY_GB":     defaultVRAMForGPU(choice.GPUs),
		},
	}, nil
}

// defaultVRAMForGPU is a static lookup for the project-supported
// GPU classes. Returns "24" as a safe fallback so the worker
// advertises *something* even on an unrecognised class.
func defaultVRAMForGPU(gpus string) string {
	prefix := strings.SplitN(gpus, ":", 2)[0]
	switch prefix {
	case "3090", "4090", "L4", "RTX4000":
		return "24"
	case "L40S":
		// L40S ships with 48 GB GDDR6 (NVIDIA L40S datasheet), not 24.
		return "48"
	case "A100":
		return "40"
	case "A100-80GB", "A100-80":
		return "80"
	case "H100":
		return "80"
	case "H200":
		return "141"
	}
	return "24"
}

// IdleTeardownCoordinator implements the two-step teardown per plan
// step 4: send `shutdown` to the worker via heartbeat, wait for the
// worker to drain or grace_seconds to elapse, then call sky down on
// the cloud VM.
//
// The actual shutdown command emission lives in ClusterHead.commandsFor
// (cluster_main Phase 2); this coordinator handles the cloud-side
// follow-through. A separate goroutine monitors the fleet for
// ephemeral workers that have stopped heartbeating and issues sky
// down.
type IdleTeardownCoordinator struct {
	Fleet      *FleetState
	SkyClient  *SkyPilotClient
	GraceSecs  int
	SweepEvery time.Duration

	// RetryBase / RetryCap bound the EXPONENTIAL BACKOFF between
	// `sky down` retries. A failing teardown is never abandoned, so
	// these only control how often it is retried (and therefore how
	// often it logs). Zero means "use the default" -- never "give up
	// immediately", so a zero-value coordinator is safe.
	RetryBase time.Duration
	RetryCap  time.Duration

	// StuckAfterFailures: consecutive failures after which an entry is
	// reported by Stuck() and in the periodic operator summary. Zero
	// uses DefaultStuckAfterFailures.
	StuckAfterFailures int

	// pending: (cluster_name, instance) -> teardown record. Workers are
	// added when the head's commandsFor sends a shutdown. The teardown
	// sweep checks whether the worker actually went away (heartbeat
	// expired), then calls Down (idempotent on the upstream side).
	//
	// The key carries the INSTANCE (the worker_id FleetState minted at
	// registration), not just the name: a failing entry is retried
	// forever, so a later cluster reusing the NAME must get its own
	// entry rather than inheriting the stuck one's long-elapsed
	// deadline and being torn down on its first sweep.
	mu           sync.Mutex
	pending      map[teardownKey]*pendingTeardown
	lastStuckLog time.Time
}

// teardownKey identifies one launch awaiting `sky down`.
type teardownKey struct {
	cluster  string
	instance string
}

// pendingTeardown is one cluster awaiting `sky down`.
type pendingTeardown struct {
	deadline    time.Time
	markedAt    time.Time
	failures    int
	backoff     time.Duration
	nextAttempt time.Time
	lastErr     string

	// conflicted: a DIFFERENT live worker currently holds this cluster
	// name, so `sky down <name>` would kill the NEW cluster. The entry
	// stays pending and untouched, and is surfaced to the operator.
	conflicted bool
}

// Teardown retry policy. A failing `sky down` is NEVER abandoned:
// dropping the entry orphans a BILLING cloud VM and nothing else ever
// revisits it. Instead the retry interval grows exponentially from
// DefaultTeardownRetryBase to DefaultTeardownRetryCap.
//
// That backoff doubles as the log rate limiter: an attempt only happens
// once its backoff window elapses, and only an attempt logs. So a
// SkyPilot API server that stays down costs one loud first line, a
// handful of doubling lines, then one line per RetryCap -- instead of
// one line per SweepEvery forever, which is what motivated the
// (VM-abandoning) attempt bound this replaces.
const (
	DefaultTeardownRetryBase = 10 * time.Second
	DefaultTeardownRetryCap  = 15 * time.Minute

	// DefaultStuckAfterFailures: consecutive failures after which an
	// entry counts as stuck for operator reporting.
	DefaultStuckAfterFailures = 3
)

// NewIdleTeardownCoordinator returns a coordinator with sane defaults.
func NewIdleTeardownCoordinator(
	fleet *FleetState, sky *SkyPilotClient,
) *IdleTeardownCoordinator {
	return &IdleTeardownCoordinator{
		Fleet:              fleet,
		SkyClient:          sky,
		GraceSecs:          30,
		SweepEvery:         HeartbeatInterval,
		RetryBase:          DefaultTeardownRetryBase,
		RetryCap:           DefaultTeardownRetryCap,
		StuckAfterFailures: DefaultStuckAfterFailures,
		pending:            make(map[teardownKey]*pendingTeardown),
	}
}

// retryBase / retryCap / stuckAfter / sweepEvery resolve the tunables,
// treating the zero value as "use the default". This is what makes a
// zero-value IdleTeardownCoordinator safe: with a bare `MaxAttempts int`
// the zero value meant "give up on the first failure", i.e. abandon a
// billing VM immediately.
func (c *IdleTeardownCoordinator) retryBase() time.Duration {
	if c.RetryBase > 0 {
		return c.RetryBase
	}
	return DefaultTeardownRetryBase
}

func (c *IdleTeardownCoordinator) retryCap() time.Duration {
	if c.RetryCap > 0 {
		return c.RetryCap
	}
	return DefaultTeardownRetryCap
}

func (c *IdleTeardownCoordinator) stuckAfter() int {
	if c.StuckAfterFailures > 0 {
		return c.StuckAfterFailures
	}
	return DefaultStuckAfterFailures
}

func (c *IdleTeardownCoordinator) sweepEvery() time.Duration {
	if c.SweepEvery > 0 {
		return c.SweepEvery
	}
	return HeartbeatInterval
}

// MarkForTeardown records that the head sent a shutdown command to the
// worker registration `instance` running as cluster `clusterName`.
// `instance` is the worker_id FleetState assigned at registration; pass
// "" only when no identity is available (name-only keying, the legacy
// behaviour). The next sweep checks whether the worker actually went
// away and, if so, calls Down on the SkyPilot side.
//
// Marking an already-pending (cluster, instance) is a no-op, so the
// deadline is stable across repeated shutdown commands. A DIFFERENT
// instance of the same name gets its OWN entry with a fresh deadline.
func (c *IdleTeardownCoordinator) MarkForTeardown(clusterName, instance string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.pending == nil {
		c.pending = make(map[teardownKey]*pendingTeardown)
	}
	k := teardownKey{cluster: clusterName, instance: instance}
	if _, exists := c.pending[k]; exists {
		return
	}
	now := time.Now()
	c.pending[k] = &pendingTeardown{
		deadline: now.Add(time.Duration(c.GraceSecs) * time.Second),
		markedAt: now,
	}
}

// TeardownEntry is the operator-facing view of one pending teardown.
type TeardownEntry struct {
	Cluster     string
	Instance    string
	Deadline    time.Time
	MarkedAt    time.Time
	Failures    int
	NextAttempt time.Time
	LastError   string
	Conflicted  bool
}

// Stuck reports whether this entry needs a human: either `sky down` has
// failed repeatedly, or a live worker has taken over the cluster name so
// the teardown cannot safely proceed.
func (e TeardownEntry) Stuck(after int) bool {
	return e.Conflicted || e.Failures >= after
}

// Entries returns a snapshot of every pending teardown, sorted by
// (cluster, instance). This is the accessor an operator surface (or a
// test) reads.
func (c *IdleTeardownCoordinator) Entries() []TeardownEntry {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.entriesLocked()
}

func (c *IdleTeardownCoordinator) entriesLocked() []TeardownEntry {
	out := make([]TeardownEntry, 0, len(c.pending))
	for k, v := range c.pending {
		out = append(out, TeardownEntry{
			Cluster:     k.cluster,
			Instance:    k.instance,
			Deadline:    v.deadline,
			MarkedAt:    v.markedAt,
			Failures:    v.failures,
			NextAttempt: v.nextAttempt,
			LastError:   v.lastErr,
			Conflicted:  v.conflicted,
		})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Cluster != out[j].Cluster {
			return out[i].Cluster < out[j].Cluster
		}
		return out[i].Instance < out[j].Instance
	})
	return out
}

// StuckEntries returns the pending teardowns an operator has to look
// at: repeatedly failing, or blocked by cluster-name reuse. These are
// exactly the entries that may correspond to a still-BILLING VM.
func (c *IdleTeardownCoordinator) StuckEntries() []TeardownEntry {
	after := c.stuckAfter()
	out := make([]TeardownEntry, 0)
	for _, e := range c.Entries() {
		if e.Stuck(after) {
			out = append(out, e)
		}
	}
	return out
}

// Pending returns a (clusterName -> deadline) snapshot. Convenience for
// operators and tests; when the same name has several instances pending
// (a reused name whose predecessor is stuck) the EARLIEST deadline
// wins. Use Entries for the per-instance view.
func (c *IdleTeardownCoordinator) Pending() map[string]time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make(map[string]time.Time, len(c.pending))
	for k, v := range c.pending {
		if prev, ok := out[k.cluster]; ok && prev.Before(v.deadline) {
			continue
		}
		out[k.cluster] = v.deadline
	}
	return out
}

// Failures reports the highest consecutive-failure count across the
// pending instances of `clusterName` (0 when none are pending).
func (c *IdleTeardownCoordinator) Failures(clusterName string) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	worst := 0
	for k, v := range c.pending {
		if k.cluster == clusterName && v.failures > worst {
			worst = v.failures
		}
	}
	return worst
}

// SweepOnce processes pending teardowns. For each entry whose deadline
// has passed OR whose worker has dropped out of the fleet, call Down
// and retire it on success.
//
// A failed Down NEVER retires the entry -- dropping it orphans a
// BILLING cloud VM and nothing else ever revisits it. Instead the entry
// backs off exponentially (retryBase doubling to retryCap) and is
// retried forever. Because only a due attempt logs, that backoff is
// also the log rate limiter.
//
// An entry whose cluster name has been taken over by a DIFFERENT live
// worker is skipped entirely: `sky down <name>` would kill the new
// cluster. It is flagged conflicted and surfaced via StuckEntries.
//
// Returns the number of cloud teardowns initiated. Test-facing.
func (c *IdleTeardownCoordinator) SweepOnce(ctx context.Context, now time.Time) int {
	type job struct {
		key      teardownKey
		deadline time.Time
	}
	c.mu.Lock()
	jobs := make([]job, 0, len(c.pending))
	for k, v := range c.pending {
		if now.Before(v.nextAttempt) {
			// Backoff window still open: not due, so no attempt and no
			// log line this sweep.
			continue
		}
		jobs = append(jobs, job{k, v.deadline})
	}
	c.mu.Unlock()
	sort.Slice(jobs, func(i, j int) bool {
		if jobs[i].key.cluster != jobs[j].key.cluster {
			return jobs[i].key.cluster < jobs[j].key.cluster
		}
		return jobs[i].key.instance < jobs[j].key.instance
	})

	// Fleet entries are keyed by Name on the head; the SkyPilot
	// cluster_name is the same string by convention (the launch sets
	// DEVAI_WORKER_NAME=cluster_name). Track WHICH registration holds
	// the name so a reused name is not mistaken for the instance we are
	// tearing down.
	aliveIDs := make(map[string]map[string]bool)
	if c.Fleet != nil {
		for _, w := range c.Fleet.Snapshot() {
			if aliveIDs[w.Name] == nil {
				aliveIDs[w.Name] = make(map[string]bool)
			}
			aliveIDs[w.Name][w.WorkerID] = true
		}
	}

	type outcome struct {
		key        teardownKey
		err        error
		conflicted bool
	}
	done := make([]teardownKey, 0, len(jobs))
	problems := make([]outcome, 0)
	n := 0
	for _, j := range jobs {
		ids, nameAlive := aliveIDs[j.key.cluster]
		// Identity-aware liveness. With no instance id we can only ask
		// "is anything running under this name?" (legacy behaviour).
		gone := !nameAlive
		conflicted := false
		if j.key.instance != "" {
			gone = !ids[j.key.instance]
			conflicted = nameAlive && !ids[j.key.instance]
		}
		if conflicted {
			// Someone else owns the name now. Tearing down by name would
			// kill the live cluster; leave it to the operator.
			problems = append(problems, outcome{key: j.key, conflicted: true})
			continue
		}
		if !now.After(j.deadline) && !gone {
			continue
		}
		if err := tryDown(ctx, c.SkyClient, j.key.cluster); err != nil {
			problems = append(problems, outcome{key: j.key, err: err})
			continue
		}
		n++
		done = append(done, j.key)
	}

	c.mu.Lock()
	for _, k := range done {
		delete(c.pending, k)
	}
	for _, p := range problems {
		e, ok := c.pending[p.key]
		if !ok {
			continue
		}
		if p.conflicted {
			c.noteConflictLocked(p.key, e, now)
			continue
		}
		c.noteFailureLocked(p.key, e, p.err, now)
	}
	c.logStuckSummaryLocked(now)
	c.mu.Unlock()
	return n
}

// noteFailureLocked records a failed `sky down`, grows the backoff and
// logs. c.mu must be held.
func (c *IdleTeardownCoordinator) noteFailureLocked(
	k teardownKey, e *pendingTeardown, err error, now time.Time,
) {
	e.failures++
	e.lastErr = err.Error()
	prev := e.backoff
	if e.backoff <= 0 {
		e.backoff = c.retryBase()
	} else {
		e.backoff *= 2
	}
	if capped := c.retryCap(); e.backoff > capped {
		e.backoff = capped
	}
	e.nextAttempt = now.Add(e.backoff)

	if e.failures == 1 {
		log.Printf("[teardown] ERROR: sky down %s (instance %q) failed: %v. "+
			"The cloud VM may still be BILLING; retrying forever with backoff "+
			"(next in %s, capped at %s)",
			k.cluster, k.instance, err, e.backoff, c.retryCap())
		return
	}
	if e.backoff != prev {
		log.Printf("[teardown] sky down %s still failing after %d attempt(s) "+
			"over %s: %v; backing off to %s",
			k.cluster, e.failures, now.Sub(e.markedAt).Round(time.Second),
			err, e.backoff)
		return
	}
	// At the cap: one line per retryCap, which is the rate limit.
	log.Printf("[teardown] sky down %s still failing after %d attempt(s) "+
		"over %s: %v; retrying in %s",
		k.cluster, e.failures, now.Sub(e.markedAt).Round(time.Second),
		err, e.backoff)
}

// noteConflictLocked flags an entry whose cluster name has been taken
// over by a different live registration. c.mu must be held.
func (c *IdleTeardownCoordinator) noteConflictLocked(
	k teardownKey, e *pendingTeardown, now time.Time,
) {
	e.nextAttempt = now.Add(c.retryCap())
	if e.conflicted {
		return
	}
	e.conflicted = true
	e.lastErr = "cluster name reused by a different live worker"
	log.Printf("[teardown] ERROR: cluster %s is now held by a DIFFERENT live "+
		"worker; refusing `sky down %s` for the old instance %q because it "+
		"would kill the live one. The old cloud VM may still be BILLING -- "+
		"reconcile it by hand", k.cluster, k.cluster, k.instance)
}

// logStuckSummaryLocked emits at most one operator summary per
// retryCap listing teardowns that need a human. c.mu must be held.
func (c *IdleTeardownCoordinator) logStuckSummaryLocked(now time.Time) {
	after := c.stuckAfter()
	names := make([]string, 0)
	for _, e := range c.entriesLocked() {
		if e.Stuck(after) {
			names = append(names, fmt.Sprintf("%s(instance=%q,failures=%d)",
				e.Cluster, e.Instance, e.Failures))
		}
	}
	if len(names) == 0 {
		return
	}
	if !c.lastStuckLog.IsZero() && now.Sub(c.lastStuckLog) < c.retryCap() {
		return
	}
	c.lastStuckLog = now
	log.Printf("[teardown] %d cluster(s) stuck awaiting teardown and possibly "+
		"still BILLING: %s", len(names), strings.Join(names, ", "))
}

// tryDown is split out so tests can fake the SkyPilot side without
// spinning a real client. Returns err only when both the client is
// configured AND the call failed; an unconfigured client is a
// silent no-op (head was started without SKYPILOT_API_ENDPOINT, so
// there's nothing to tear down on the cloud side).
func tryDown(ctx context.Context, sky *SkyPilotClient, clusterName string) error {
	if sky == nil || !sky.IsConfigured() {
		return nil
	}
	return sky.Down(ctx, clusterName)
}

// Run blocks: loop SweepOnce on SweepEvery until ctx is cancelled.
// A zero SweepEvery falls back to HeartbeatInterval rather than
// panicking in time.NewTicker.
func (c *IdleTeardownCoordinator) Run(ctx context.Context) {
	ticker := time.NewTicker(c.sweepEvery())
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			_ = c.SweepOnce(ctx, time.Now())
		}
	}
}
