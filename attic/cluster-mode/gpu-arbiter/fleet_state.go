//go:build devai_frozen_cluster

// Head-side in-memory fleet state.
//
// Per docs/plans/gpu-arbiter-cluster-mode.md Phase 2 + decision 9:
// re-derived from worker heartbeats on head restart; no named-volume
// persistence, no migration concerns. 10s heartbeat cadence gives
// fast recovery (workers re-register within 20s of head coming back).
//
// Mutex-protected, garbage-collected on heartbeat expiry.

package main

import (
	"sort"
	"sync"
	"time"
)

// WorkerEntry is the head's per-worker view, populated by registration
// and updated by heartbeats.
type WorkerEntry struct {
	WorkerID       string
	Name           string
	Lifecycle      LifecycleClass
	GPUType        string
	VRAMGB         int
	Backends       []string
	Endpoint       string
	ArbiterVersion string

	// Heartbeat-driven fields. Mutex must be held when reading or
	// writing.
	LoadedModel    string
	LoadedCtx      int
	QueueDepth     int
	UtilizationPct float64
	LastRequestAt  string
	HealthStatus   string
	LastHeartbeat  time.Time
	Counter        uint64
}

// FleetState owns the worker map. Methods are safe for concurrent use.
//
// Workers are registered with Register, updated via Heartbeat, and
// removed when their last heartbeat is older than HeartbeatTTL. The
// expiry sweep is driven by ExpireOlderThan; head's main loop calls
// it on a timer.
type FleetState struct {
	mu      sync.RWMutex
	workers map[string]*WorkerEntry

	// HeartbeatTTL controls how stale a worker's last heartbeat
	// can be before ExpireOlderThan removes it. Default 30s
	// (3 missed heartbeats at the canonical 10s cadence).
	HeartbeatTTL time.Duration

	// IDFactory mints opaque worker_ids. Tests inject deterministic
	// generators; production uses the default time+counter shape.
	IDFactory func(req RegisterRequest) string
}

// NewFleetState returns an initialised FleetState with sane defaults.
func NewFleetState() *FleetState {
	return &FleetState{
		workers:      make(map[string]*WorkerEntry),
		HeartbeatTTL: 30 * time.Second,
		IDFactory:    defaultWorkerID,
	}
}

// Register accepts a worker registration and returns the assigned
// worker_id. If a worker with the same Name already exists, the
// Register call REPLACES it (a worker re-registering after restart
// gets a fresh worker_id and the prior entry is garbage-collected).
func (f *FleetState) Register(req RegisterRequest, now time.Time) string {
	f.mu.Lock()
	defer f.mu.Unlock()

	// Drop any prior entry for the same Name.
	for id, w := range f.workers {
		if w.Name == req.Name {
			delete(f.workers, id)
		}
	}

	id := f.IDFactory(req)
	f.workers[id] = &WorkerEntry{
		WorkerID:       id,
		Name:           req.Name,
		Lifecycle:      req.Lifecycle,
		GPUType:        req.GPUType,
		VRAMGB:         req.VRAMGB,
		Backends:       append([]string(nil), req.Backends...),
		Endpoint:       req.Endpoint,
		ArbiterVersion: req.ArbiterVersion,
		LastHeartbeat:  now,
		HealthStatus:   "registered",
	}
	return id
}

// HeartbeatResult tells the caller what to do with the heartbeat.
// Stale = head saw an out-of-order counter and dropped the update.
// Unknown = the worker_id isn't registered (likely a stale worker
// pointing at a fresh head).
type HeartbeatResult int

const (
	HeartbeatAccepted HeartbeatResult = iota
	HeartbeatStale
	HeartbeatUnknownWorker
)

// Heartbeat applies a heartbeat to the named worker. Returns the
// outcome so the head can branch on it (e.g. tell an unknown
// worker to re-register instead of just heartbeating).
func (f *FleetState) Heartbeat(hb HeartbeatRequest, now time.Time) HeartbeatResult {
	f.mu.Lock()
	defer f.mu.Unlock()
	w, ok := f.workers[hb.WorkerID]
	if !ok {
		return HeartbeatUnknownWorker
	}
	if hb.Counter <= w.Counter {
		return HeartbeatStale
	}
	w.Counter = hb.Counter
	w.LoadedModel = hb.LoadedModel
	w.LoadedCtx = hb.LoadedCtx
	w.QueueDepth = hb.QueueDepth
	w.UtilizationPct = hb.UtilizationPct
	w.LastRequestAt = hb.LastRequestAt
	w.HealthStatus = hb.HealthStatus
	w.LastHeartbeat = now
	return HeartbeatAccepted
}

// ExpireOlderThan removes workers whose last heartbeat is older than
// `cutoff`. Returns the number of workers expired. Callers usually
// pass `time.Now().Add(-f.HeartbeatTTL)`.
func (f *FleetState) ExpireOlderThan(cutoff time.Time) int {
	f.mu.Lock()
	defer f.mu.Unlock()
	n := 0
	for id, w := range f.workers {
		if w.LastHeartbeat.Before(cutoff) {
			delete(f.workers, id)
			n++
		}
	}
	return n
}

// Snapshot returns a deep-copied list of all current workers, sorted
// by Name. Safe to expose over /v1/cluster/status without holding
// the mutex while serializing.
func (f *FleetState) Snapshot() []WorkerEntry {
	f.mu.RLock()
	defer f.mu.RUnlock()
	out := make([]WorkerEntry, 0, len(f.workers))
	for _, w := range f.workers {
		copied := *w
		copied.Backends = append([]string(nil), w.Backends...)
		out = append(out, copied)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// Get returns a copy of the named worker (not a live pointer) and
// whether it exists. Used by routing decisions.
func (f *FleetState) Get(workerID string) (WorkerEntry, bool) {
	f.mu.RLock()
	defer f.mu.RUnlock()
	w, ok := f.workers[workerID]
	if !ok {
		return WorkerEntry{}, false
	}
	copied := *w
	copied.Backends = append([]string(nil), w.Backends...)
	return copied, true
}

// Count returns the current number of registered workers. Used by
// the routing layer to short-circuit when the fleet is empty.
func (f *FleetState) Count() int {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return len(f.workers)
}

// defaultWorkerID returns a deterministic-but-unique id derived from
// the worker name and registration time. Production-acceptable; not
// cryptographically random.
func defaultWorkerID(req RegisterRequest) string {
	now := time.Now().UnixNano()
	return "wid-" + req.Name + "-" + uintToHex(uint64(now))
}

// uintToHex returns a lowercase hex string of u, no prefix. Local
// to avoid pulling fmt for a hot path.
func uintToHex(u uint64) string {
	const digits = "0123456789abcdef"
	if u == 0 {
		return "0"
	}
	var buf [16]byte
	i := len(buf)
	for u > 0 {
		i--
		buf[i] = digits[u&0xf]
		u >>= 4
	}
	return string(buf[i:])
}