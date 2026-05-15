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

	type candidate struct {
		entry WorkerEntry
		score int
	}
	cands := make([]candidate, 0, len(workers))
	for _, w := range workers {
		if !backendSupported(w, backend) {
			continue
		}
		if p.QueueDepthThreshold > 0 && w.QueueDepth >= p.QueueDepthThreshold {
			continue
		}
		cands = append(cands, candidate{entry: w, score: scoreWorker(w, model, ctx)})
	}
	if len(cands) == 0 {
		return RoutingDecision{
			NoFitReason: "no worker advertises backend " + backend +
				" within the queue-depth threshold",
		}
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
	return RoutingDecision{WorkerID: chosen.entry.WorkerID, Score: chosen.score}
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
