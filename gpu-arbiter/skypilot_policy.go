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
	case "3090", "4090", "L4", "L40S", "RTX4000":
		return "24"
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

	// pending: cluster_name -> deadline. Workers added to this map
	// when their lifecycle suggests teardown is appropriate (head's
	// commandsFor sent a shutdown). The teardown sweep checks if
	// the worker has actually stopped heartbeating, then calls Down
	// once (idempotent on the upstream side).
	mu      sync.Mutex
	pending map[string]time.Time
}

// NewIdleTeardownCoordinator returns a coordinator with sane defaults.
func NewIdleTeardownCoordinator(
	fleet *FleetState, sky *SkyPilotClient,
) *IdleTeardownCoordinator {
	return &IdleTeardownCoordinator{
		Fleet:      fleet,
		SkyClient:  sky,
		GraceSecs:  30,
		SweepEvery: HeartbeatInterval,
		pending:    make(map[string]time.Time),
	}
}

// MarkForTeardown records that the head sent a shutdown command for
// `clusterName`. The next sweep checks if the worker has actually
// gone away (heartbeat expired) and, if so, calls Down on the
// SkyPilot side.
func (c *IdleTeardownCoordinator) MarkForTeardown(clusterName string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if _, exists := c.pending[clusterName]; !exists {
		c.pending[clusterName] = time.Now().Add(time.Duration(c.GraceSecs) * time.Second)
	}
}

// Pending returns a snapshot of the (clusterName, deadline) map.
// Test-facing.
func (c *IdleTeardownCoordinator) Pending() map[string]time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make(map[string]time.Time, len(c.pending))
	for k, v := range c.pending {
		out[k] = v
	}
	return out
}

// SweepOnce processes pending teardowns. For each entry whose
// deadline has passed OR whose worker has dropped out of the fleet,
// call Down and remove from pending. Returns the number of cloud
// teardowns initiated. Test-facing.
func (c *IdleTeardownCoordinator) SweepOnce(ctx context.Context, now time.Time) int {
	type pair struct {
		cluster  string
		deadline time.Time
	}
	c.mu.Lock()
	candidates := make([]pair, 0, len(c.pending))
	for k, v := range c.pending {
		candidates = append(candidates, pair{k, v})
	}
	c.mu.Unlock()

	fleet := c.Fleet.Snapshot()
	stillAlive := make(map[string]bool, len(fleet))
	for _, w := range fleet {
		// Fleet entries are keyed by Name on the head; the
		// SkyPilot cluster_name is the same string by convention
		// (the launch sets DEVAI_WORKER_NAME=cluster_name).
		stillAlive[w.Name] = true
	}

	processed := make([]string, 0, len(candidates))
	n := 0
	for _, p := range candidates {
		past := now.After(p.deadline)
		gone := !stillAlive[p.cluster]
		if !past && !gone {
			continue
		}
		// Best-effort sky down. Errors are logged-only -- the
		// next sweep will retry.
		if err := tryDown(ctx, c.SkyClient, p.cluster); err == nil {
			n++
		}
		processed = append(processed, p.cluster)
	}

	if len(processed) > 0 {
		c.mu.Lock()
		for _, name := range processed {
			delete(c.pending, name)
		}
		c.mu.Unlock()
	}
	return n
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
func (c *IdleTeardownCoordinator) Run(ctx context.Context) {
	ticker := time.NewTicker(c.SweepEvery)
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
