//go:build devai_frozen_cluster

// Cluster worker mode: register with head, send heartbeats, execute
// commands. Per docs/plans/gpu-arbiter-cluster-mode.md Phase 1.
//
// Worker mode reuses the existing single-host scheduling code path
// for actual request handling (decision 2: full mutation chain stays
// on the worker). This file adds only the control-plane surface:
// register/heartbeat/inbound listener.

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// WorkerConfig is everything the worker needs to talk to a head.
// Built once at startup from env vars.
type WorkerConfig struct {
	HeadURL        string         // DEVAI_HEAD_URL
	WorkerName     string         // DEVAI_WORKER_NAME
	Lifecycle      LifecycleClass // DEVAI_LIFECYCLE
	Endpoint       string         // worker's own /v1/cluster/inbound URL
	GPUType        string         // probed from nvidia-smi
	VRAMGB         int            // GPU_MEMORY_GB
	Backends       []string       // ["ollama","vllm","sglang"]
	ArbiterVersion string         // git sha at build time
	Token          *TokenStore
	HTTPClient     HTTPDoer

	// Heartbeat cadence; defaults to HeartbeatInterval. Tests set
	// shorter values.
	HeartbeatInterval time.Duration

	// RegisterMaxBackoff caps the exponential backoff between
	// registration retries. Defaults to 60s.
	RegisterMaxBackoff time.Duration
}

// HTTPDoer is the small interface we need from net/http.Client so
// tests can inject a fake.
type HTTPDoer interface {
	Do(*http.Request) (*http.Response, error)
}

// WorkerState is the in-memory view the heartbeat loop sends to the
// head. CommandExecutor populates LoadedModel/LoadedCtx after a
// `serve` command finishes; the request handler increments
// QueueDepth.
type WorkerState struct {
	WorkerID       atomic.Pointer[string] // assigned by head at registration
	loadedModel    atomic.Pointer[string]
	loadedCtx      atomic.Int64
	queueDepth     atomic.Int64
	utilizationPct atomic.Uint64 // bits of float64
	lastRequestAt  atomic.Pointer[time.Time]
	counter        atomic.Uint64

	// healthStatus is read lock-free by snapshot() on the heartbeat
	// path; healthMu serialises the read-modify-write in
	// CompareAndSwapHealth so a drain finishing cannot clobber a
	// shutdown that arrived while it was running.
	healthMu     sync.Mutex
	healthStatus atomic.Pointer[string]
}

// SetLoadedModel atomically updates the (model, ctx) pair the worker
// is currently serving.
func (s *WorkerState) SetLoadedModel(model string, ctx int) {
	if model == "" {
		s.loadedModel.Store(nil)
		s.loadedCtx.Store(0)
		return
	}
	m := model
	s.loadedModel.Store(&m)
	s.loadedCtx.Store(int64(ctx))
}

// IncQueue / DecQueue track in-flight request count; the heartbeat
// reads these to inform head-side scoring.
func (s *WorkerState) IncQueue() { s.queueDepth.Add(1) }
func (s *WorkerState) DecQueue() { s.queueDepth.Add(-1) }

// SetUtilization stores a float64 GPU utilisation percentage in [0,100].
func (s *WorkerState) SetUtilization(pct float64) {
	if pct < 0 {
		pct = 0
	} else if pct > 100 {
		pct = 100
	}
	s.utilizationPct.Store(float64ToBits(pct))
}

// Health-status values carried in the heartbeat's health_status field.
// The head's routing policy treats HealthDraining / HealthShuttingDown
// as "do not send new work" (see workerAvailable); anything else --
// including the "registered" a fresh worker carries before its first
// heartbeat, and the empty string -- counts as available.
//
// The state machine is deliberately small:
//
//	ready  --drain-->      draining  --drain complete-->  ready
//	ready  --shutdown-->   shutting_down                  (terminal)
//	draining --shutdown--> shutting_down                  (terminal)
//
// draining is NOT terminal: a drain waits out in-flight requests
// (bounded by DRAIN_TIMEOUT) and the worker is perfectly serviceable
// afterwards. Leaving it latched at "draining" removes the worker from
// the head's routing pool forever -- on a single-worker fleet that is a
// permanent 503. shutting_down IS terminal: the process is about to
// exit, so nothing may transition out of it.
const (
	HealthReady        = "ready"
	HealthDraining     = "draining"
	HealthShuttingDown = "shutting_down"
)

// SetHealthStatus updates the string carried in the heartbeat.
// Recognised values: HealthReady, HealthDraining, HealthShuttingDown.
// Unconditional -- use CompareAndSwapHealth for transitions that must
// not overwrite a terminal state.
func (s *WorkerState) SetHealthStatus(status string) {
	s.healthMu.Lock()
	defer s.healthMu.Unlock()
	st := status
	s.healthStatus.Store(&st)
}

// HealthStatus returns the current health status, defaulting to
// HealthReady before anything has been stored.
func (s *WorkerState) HealthStatus() string {
	if hs := s.healthStatus.Load(); hs != nil {
		return *hs
	}
	return HealthReady
}

// SetHealthUnlessShuttingDown stores `status` unless the worker has
// already entered the terminal HealthShuttingDown state, and reports
// whether the store happened. Used when acknowledging a drain: a drain
// command that lands after a shutdown must not advertise the worker as
// merely draining when the process is already on its way out.
func (s *WorkerState) SetHealthUnlessShuttingDown(status string) bool {
	s.healthMu.Lock()
	defer s.healthMu.Unlock()
	if hs := s.healthStatus.Load(); hs != nil && *hs == HealthShuttingDown {
		return false
	}
	st := status
	s.healthStatus.Store(&st)
	return true
}

// CompareAndSwapHealth stores `to` only when the current status is
// `from`, and reports whether the swap happened. This is how a
// completed drain returns the worker to a routable state without
// resurrecting one that received a shutdown mid-drain: the CAS from
// HealthDraining fails once shutdown has moved the status to
// HealthShuttingDown, which is terminal.
func (s *WorkerState) CompareAndSwapHealth(from, to string) bool {
	s.healthMu.Lock()
	defer s.healthMu.Unlock()
	cur := HealthReady
	if hs := s.healthStatus.Load(); hs != nil {
		cur = *hs
	}
	if cur != from {
		return false
	}
	st := to
	s.healthStatus.Store(&st)
	return true
}

// MarkRequestAt stamps the wall-clock when a request finished.
func (s *WorkerState) MarkRequestAt(t time.Time) {
	tt := t
	s.lastRequestAt.Store(&tt)
}

// snapshot reads every atomic into a HeartbeatRequest body.
// Counter is incremented atomically so each call yields a unique
// monotonic value.
func (s *WorkerState) snapshot(workerID string) HeartbeatRequest {
	hb := HeartbeatRequest{
		WorkerID:   workerID,
		QueueDepth: int(s.queueDepth.Load()),
		Counter:    s.counter.Add(1),
	}
	if m := s.loadedModel.Load(); m != nil {
		hb.LoadedModel = *m
		hb.LoadedCtx = int(s.loadedCtx.Load())
	}
	hb.UtilizationPct = bitsToFloat64(s.utilizationPct.Load())
	if t := s.lastRequestAt.Load(); t != nil {
		hb.LastRequestAt = t.UTC().Format(time.RFC3339)
	}
	if hs := s.healthStatus.Load(); hs != nil {
		hb.HealthStatus = *hs
	} else {
		hb.HealthStatus = "ready"
	}
	return hb
}

// Worker drives the control-plane loop: register, heartbeat, command
// execution. Single-instance per arbiter process.
type Worker struct {
	Config   WorkerConfig
	State    *WorkerState
	Executor CommandExecutor
}

// CommandExecutor is the worker-side hook the heartbeat loop calls
// for each command in the heartbeat response. Implementations branch
// on Command.Type. Returning an error logs the failure but does not
// halt the loop -- subsequent commands still run, and the next
// heartbeat fires on schedule.
type CommandExecutor interface {
	Execute(ctx context.Context, cmd Command) error
}

// NewWorker builds a Worker with sensible default state. Caller
// MUST populate Executor before Start().
func NewWorker(cfg WorkerConfig) *Worker {
	if cfg.HeartbeatInterval == 0 {
		cfg.HeartbeatInterval = HeartbeatInterval
	}
	if cfg.RegisterMaxBackoff == 0 {
		cfg.RegisterMaxBackoff = 60 * time.Second
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = &http.Client{Timeout: 30 * time.Second}
	}
	w := &Worker{
		Config: cfg,
		State:  &WorkerState{},
	}
	w.State.SetHealthStatus("ready")
	return w
}

// Start runs the registration loop, then the heartbeat loop. Blocks
// until ctx is cancelled. Caller is responsible for the inbound
// HTTP listener (mounted into the existing single-host mux).
func (w *Worker) Start(ctx context.Context) error {
	id, err := w.registerWithBackoff(ctx)
	if err != nil {
		return fmt.Errorf("worker registration aborted: %w", err)
	}
	w.State.WorkerID.Store(&id)
	log.Printf("[worker] registered as %s (id=%s) with head=%s",
		w.Config.WorkerName, id, w.Config.HeadURL)
	return w.heartbeatLoop(ctx)
}

// registerWithBackoff retries POST /v1/cluster/register until ctx
// is cancelled or registration succeeds. Backoff doubles up to
// RegisterMaxBackoff with full jitter.
func (w *Worker) registerWithBackoff(ctx context.Context) (string, error) {
	body := RegisterRequest{
		Name:           w.Config.WorkerName,
		Lifecycle:      w.Config.Lifecycle,
		GPUType:        w.Config.GPUType,
		VRAMGB:         w.Config.VRAMGB,
		Backends:       w.Config.Backends,
		ArbiterVersion: w.Config.ArbiterVersion,
		Endpoint:       w.Config.Endpoint,
	}
	if err := body.Validate(); err != nil {
		return "", fmt.Errorf("registration body invalid: %w", err)
	}
	delay := time.Second
	for {
		id, err := w.registerOnce(ctx, body)
		if err == nil {
			return id, nil
		}
		log.Printf("[worker] register failed: %v; retrying in %v", err, delay)
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-time.After(jitter(delay)):
		}
		delay *= 2
		if delay > w.Config.RegisterMaxBackoff {
			delay = w.Config.RegisterMaxBackoff
		}
	}
}

// registerOnce posts a single registration attempt. Returns the
// assigned worker_id on success.
func (w *Worker) registerOnce(ctx context.Context, body RegisterRequest) (string, error) {
	tok, err := w.Config.Token.Read()
	if err != nil {
		return "", fmt.Errorf("read bearer token: %w", err)
	}
	buf, err := json.Marshal(body)
	if err != nil {
		return "", fmt.Errorf("marshal register body: %w", err)
	}
	req, err := http.NewRequestWithContext(
		ctx, http.MethodPost,
		w.Config.HeadURL+"/v1/cluster/register",
		bytes.NewReader(buf),
	)
	if err != nil {
		return "", fmt.Errorf("build register request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+tok)
	resp, err := w.Config.HTTPClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("post register: %w", err)
	}
	defer resp.Body.Close()
	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("register HTTP %d: %s", resp.StatusCode, string(respBody))
	}
	var rr RegisterResponse
	if err := json.Unmarshal(respBody, &rr); err != nil {
		return "", fmt.Errorf("decode register response: %w", err)
	}
	if rr.WorkerID == "" {
		return "", errors.New("head returned empty worker_id")
	}
	return rr.WorkerID, nil
}

// heartbeatLoop fires every HeartbeatInterval until ctx is done.
// Failed heartbeats log + retry on the next tick (no fancy
// backoff). Commands from the response are dispatched serially.
func (w *Worker) heartbeatLoop(ctx context.Context) error {
	ticker := time.NewTicker(w.Config.HeartbeatInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			w.tick(ctx)
		}
	}
}

func (w *Worker) tick(ctx context.Context) {
	idPtr := w.State.WorkerID.Load()
	if idPtr == nil {
		log.Printf("[worker] heartbeat: no worker_id assigned yet; skipping tick")
		return
	}
	resp, err := w.HeartbeatOnce(ctx, *idPtr)
	if err != nil {
		if errors.Is(err, ErrHeadUnknownWorker) {
			// The head restarted (fleet state is in-memory only, per
			// cluster-mode decision 9) and no longer knows this id.
			// Without re-registering here the worker would heartbeat
			// into a 410 forever and stay invisible to the fleet.
			w.reregister(ctx)
			return
		}
		log.Printf("[worker] heartbeat failed: %v", err)
		return
	}
	for _, cmd := range resp.Commands {
		if err := w.dispatchCommand(ctx, cmd); err != nil {
			log.Printf("[worker] command %s: %v", cmd.Type, err)
		}
	}
}

// ErrHeadUnknownWorker is what HeartbeatOnce returns when the head
// answers 410 Gone -- its contract for "I do not know this worker_id;
// re-register". See ClusterHead.handleHeartbeat.
var ErrHeadUnknownWorker = errors.New("head does not recognise this worker_id")

// reregister drops the stale worker_id and runs the registration
// backoff loop again. Called from tick when the head answers 410.
func (w *Worker) reregister(ctx context.Context) {
	log.Printf("[worker] head reported unknown worker_id; re-registering")
	w.State.WorkerID.Store(nil)
	id, err := w.registerWithBackoff(ctx)
	if err != nil {
		log.Printf("[worker] re-registration aborted: %v", err)
		return
	}
	w.State.WorkerID.Store(&id)
	log.Printf("[worker] re-registered as %s (id=%s) with head=%s",
		w.Config.WorkerName, id, w.Config.HeadURL)
}

// HeartbeatOnce runs one heartbeat round-trip. Exposed so tests can
// drive a single iteration without spinning up the loop goroutine.
func (w *Worker) HeartbeatOnce(ctx context.Context, workerID string) (*HeartbeatResponse, error) {
	tok, err := w.Config.Token.Read()
	if err != nil {
		return nil, fmt.Errorf("read bearer token: %w", err)
	}
	hb := w.State.snapshot(workerID)
	buf, _ := json.Marshal(hb)
	req, err := http.NewRequestWithContext(
		ctx, http.MethodPost,
		w.Config.HeadURL+"/v1/cluster/heartbeat",
		bytes.NewReader(buf),
	)
	if err != nil {
		return nil, fmt.Errorf("build heartbeat request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+tok)
	resp, err := w.Config.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("post heartbeat: %w", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode == http.StatusGone {
		return nil, fmt.Errorf("%w: %s", ErrHeadUnknownWorker, strings.TrimSpace(string(body)))
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("heartbeat HTTP %d: %s", resp.StatusCode, string(body))
	}
	var hr HeartbeatResponse
	if err := json.Unmarshal(body, &hr); err != nil {
		return nil, fmt.Errorf("decode heartbeat response: %w", err)
	}
	return &hr, nil
}

// dispatchCommand applies the lifecycle policy (persistent workers
// refuse shutdown) and delegates to the executor for the rest.
// Per cluster-mode decision 3.
func (w *Worker) dispatchCommand(ctx context.Context, cmd Command) error {
	switch cmd.Type {
	case CommandShutdown:
		if w.Config.Lifecycle == LifecyclePersistent {
			log.Printf("[worker] refusing shutdown command: lifecycle=persistent")
			return nil
		}
	case CommandDrain, CommandServe:
		// Honoured by every lifecycle.
	default:
		log.Printf("[worker] unknown command type %q; ignoring", cmd.Type)
		return nil
	}
	if w.Executor == nil {
		return fmt.Errorf("no command executor configured")
	}
	return w.Executor.Execute(ctx, cmd)
}

// jitter returns d +/- 30%, never below 100ms. Avoids thundering
// herd when N workers register against a head that just came back.
func jitter(d time.Duration) time.Duration {
	if d < 100*time.Millisecond {
		d = 100 * time.Millisecond
	}
	spread := float64(d) * 0.3
	delta := (rand.Float64() - 0.5) * 2 * spread
	out := time.Duration(float64(d) + delta)
	if out < 100*time.Millisecond {
		out = 100 * time.Millisecond
	}
	return out
}

// float64ToBits / bitsToFloat64: store float64 in atomic.Uint64.
func float64ToBits(f float64) uint64 {
	return math.Float64bits(f)
}
func bitsToFloat64(u uint64) float64 {
	return math.Float64frombits(u)
}