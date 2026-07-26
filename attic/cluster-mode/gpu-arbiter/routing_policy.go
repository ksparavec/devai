//go:build devai_frozen_cluster

// Head-side routing policy.
//
// Per docs/plans/gpu-arbiter-cluster-mode.md decision 4:
//
//   1. Exact match (right model AND loaded_ctx >= requested_ctx),
//      queue_depth < threshold -- zero cold-start cost.
//   2. Right model but loaded_ctx < requested_ctx -- recreate cost.
//   3. Idle (no model loaded) -- cold-load cost.
//   4. Different model loaded -- recreate cost.
//
// Tiebreak: round-robin among equally-scored candidates. Workers
// above queue_depth_threshold are skipped (overloaded).

package main

import (
	"sort"
	"strconv"
	"sync"
	"sync/atomic"
)

// RoutingDecision is what RouteDecision returns. WorkerID empty +
// NoFitReason populated means "no worker can serve this request"
// (head should 503 with the reason).
type RoutingDecision struct {
	WorkerID    string
	Score       int
	NoFitReason string

	// Degraded is set when the only candidates left were draining
	// workers, i.e. the request was routed to a worker that is on its
	// way to idle rather than 503'd. Callers may log it; it is not an
	// error.
	Degraded bool
}

// RoutingPolicy holds the tunable knobs (queue depth threshold,
// per-bucket round-robin counters) the routing layer reads on each
// request. Methods are safe for concurrent use.
type RoutingPolicy struct {
	// QueueDepthThreshold: workers with queue_depth >= this value
	// are skipped from candidate scoring. 0 disables the gate
	// (every worker always considered).
	QueueDepthThreshold int

	// rrCounters: per-bucket round-robin counter. Bucket key is
	// the score string (e.g. "100" / "50" / "30" / "10") so equal-
	// scoring workers are picked in round-robin order across
	// requests.
	rrCounters sync.Map // map[int]*atomic.Uint64
}

// Score constants. Values matter only relative to each other; named
// for readability against the plan's table.
const (
	ScoreExactMatch     = 100
	ScoreRightModelCtx  = 50
	ScoreIdle           = 30
	ScoreDifferentModel = 10
)

// RouteDecision picks the best worker for the (model, ctx) request.
// `backend` filters candidates to those advertising that backend
// (workers register their supported backends; head respects the
// advertised list).
func (p *RoutingPolicy) RouteDecision(
	workers []WorkerEntry, model string, ctx int, backend string,
) RoutingDecision {
	if len(workers) == 0 {
		return RoutingDecision{NoFitReason: "no workers registered"}
	}

	cands, skipped := p.eligible(workers, model, ctx, backend, false)

	// Degraded fallback: the healthy pass found nothing, but some
	// workers were skipped only because they are DRAINING. Prefer a
	// slow response over a 503.
	//
	// Why draining is safe to fall back to and shutting_down is not: a
	// drain waits out the requests already in flight while holding the
	// arbiter mutex, so a newly forwarded request parks on that mutex
	// and is served once the drain returns -- bounded by DRAIN_TIMEOUT
	// (30s default). A shutting-down worker exits the process at the
	// end of its drain, so the same request would die with it. Hence
	// the fallback admits HealthDraining and never HealthShuttingDown.
	degraded := false
	if len(cands) == 0 && skipped.draining > 0 {
		cands, _ = p.eligible(workers, model, ctx, backend, true)
		degraded = len(cands) > 0
	}

	if len(cands) == 0 {
		reason := "no worker advertises backend " + backend +
			" within the queue-depth threshold"
		if n := skipped.draining + skipped.shuttingDown; n > 0 {
			reason += " (" + strconv.Itoa(n) +
				" worker(s) skipped: draining or shutting down)"
		}
		return RoutingDecision{NoFitReason: reason}
	}

	// Top score wins; ties round-robined per bucket.
	sort.SliceStable(cands, func(i, j int) bool {
		return cands[i].score > cands[j].score
	})
	top := cands[0].score
	tied := []candidate{}
	for _, c := range cands {
		if c.score == top {
			tied = append(tied, c)
		}
	}
	pick := p.roundRobin(top, len(tied))
	chosen := tied[pick]
	return RoutingDecision{
		WorkerID: chosen.entry.WorkerID,
		Score:    chosen.score,
		Degraded: degraded,
	}
}

// candidate pairs a worker with its 4-tier score.
type candidate struct {
	entry WorkerEntry
	score int
}

// skipCounts records why workers were filtered out, for the no-fit
// message.
type skipCounts struct {
	draining     int
	shuttingDown int
}

// eligible applies the health / backend / queue-depth filters and
// scores whatever survives. With includeDraining set, workers whose
// last-reported status is HealthDraining are admitted anyway (the
// degraded fallback); HealthShuttingDown is excluded in both modes
// because that worker's process is about to exit.
func (p *RoutingPolicy) eligible(
	workers []WorkerEntry, model string, ctx int, backend string,
	includeDraining bool,
) ([]candidate, skipCounts) {
	cands := make([]candidate, 0, len(workers))
	var skipped skipCounts
	for _, w := range workers {
		if !workerAvailable(w) {
			if w.HealthStatus == HealthShuttingDown {
				// Terminal: the process exits at the end of its drain.
				skipped.shuttingDown++
				continue
			}
			skipped.draining++
			if !includeDraining {
				continue
			}
		}
		if !backendSupported(w, backend) {
			continue
		}
		if p.QueueDepthThreshold > 0 && w.QueueDepth >= p.QueueDepthThreshold {
			continue
		}
		cands = append(cands, candidate{entry: w, score: scoreWorker(w, model, ctx)})
	}
	return cands, skipped
}

// scoreWorker applies the 4-tier policy to one candidate.
func scoreWorker(w WorkerEntry, model string, ctx int) int {
	if w.LoadedModel == model && w.LoadedCtx >= ctx && ctx > 0 {
		return ScoreExactMatch
	}
	if w.LoadedModel == model {
		return ScoreRightModelCtx
	}
	if w.LoadedModel == "" {
		return ScoreIdle
	}
	return ScoreDifferentModel
}

// workerAvailable reports whether the worker's last-reported health
// status allows NEW work. Only the two "on the way out" states are
// excluded; every other value (including "registered" from a worker
// that has not heartbeated yet, and "" from an older worker build)
// counts as available, so an unknown status never silently removes a
// worker from the fleet.
func workerAvailable(w WorkerEntry) bool {
	switch w.HealthStatus {
	case HealthDraining, HealthShuttingDown:
		return false
	}
	return true
}

// backendSupported reports whether `backend` appears in the worker's
// advertised list. Empty backend = "any" (used by status endpoints).
func backendSupported(w WorkerEntry, backend string) bool {
	if backend == "" {
		return true
	}
	for _, b := range w.Backends {
		if b == backend {
			return true
		}
	}
	return false
}

// roundRobin returns a deterministic next index in [0, n) for the
// given bucket. Uses an atomic counter per bucket so concurrent
// callers don't race.
func (p *RoutingPolicy) roundRobin(bucket, n int) int {
	v, _ := p.rrCounters.LoadOrStore(bucket, &atomic.Uint64{})
	c := v.(*atomic.Uint64)
	if n == 0 {
		return 0
	}
	next := c.Add(1) - 1
	return int(next % uint64(n))
}