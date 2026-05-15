// Head-mode HTTP surface.
//
// Per docs/plans/gpu-arbiter-cluster-mode.md Phase 2:
// - Listens on the same OpenAI-compat ports (11434/5/6) with the
//   same handler shape as single mode, but dispatches to
//   cluster_proxy.Forward instead of the local recreate/serve path.
// - Maintains an in-memory FleetState; expires workers whose
//   heartbeat goes stale.
// - Exposes /v1/cluster/{register,heartbeat,status}.

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"
)

// ClusterHead owns the head-mode runtime.
type ClusterHead struct {
	Fleet   *FleetState
	Policy  *RoutingPolicy
	Token   *TokenStore
	Forward HeadForwarder

	// HeadListenPort is where the cluster control plane (register/
	// heartbeat/status) listens. Defaults to 11444.
	HeadListenPort int

	// FrontendPorts is the set of OpenAI-compat ports the head
	// mounts proxy handlers on. Defaults to {11434, 11435, 11436}
	// per the project's existing single-host shape.
	FrontendPorts map[string]int // backend name -> port

	// IdleSweepInterval drives FleetState.ExpireOlderThan ticks.
	// Defaults to HeartbeatInterval (10s) so a missed heartbeat is
	// reflected on the next tick.
	IdleSweepInterval time.Duration

	// IdleMinutes (DEVAI_IDLE_MINUTES per cluster-mode decision 14)
	// drives the head's optional shutdown-command issuer for
	// ephemeral workers. 0 disables.
	IdleMinutes int
}

// HeadForwarder is what the head calls to forward an OpenAI-compat
// request to a chosen worker. The ClusterProxy below implements it.
// Tests inject a fake.
type HeadForwarder interface {
	Forward(w http.ResponseWriter, r *http.Request, worker WorkerEntry, parsed MinimalRequest)
}

// NewClusterHead returns a head with sane defaults.
func NewClusterHead() *ClusterHead {
	tokenPath := env("DEVAI_HEAD_TOKEN_FILE", "/run/devai/cluster-token")
	idle := envInt("DEVAI_IDLE_MINUTES", 10)
	tokens := NewTokenStore(tokenPath, 30*time.Second)
	if _, err := tokens.Read(); err != nil {
		log.Fatalf("head mode: %v", err)
	}
	fleet := NewFleetState()
	policy := &RoutingPolicy{
		QueueDepthThreshold: envInt("DEVAI_QUEUE_DEPTH_THRESHOLD", 0),
	}
	return &ClusterHead{
		Fleet:             fleet,
		Policy:            policy,
		Token:             tokens,
		Forward:           NewClusterProxy(tokens),
		HeadListenPort:    envInt("DEVAI_HEAD_LISTEN_PORT", 11444),
		FrontendPorts:     defaultFrontendPorts(),
		IdleSweepInterval: HeartbeatInterval,
		IdleMinutes:       idle,
	}
}

func defaultFrontendPorts() map[string]int {
	return map[string]int{
		"ollama": envInt("OLLAMA_PORT", 11434),
		"vllm":   envInt("VLLM_PORT", 11435),
		"sglang": envInt("SGLANG_PORT", 11436),
	}
}

// Run blocks: starts the control-plane listener, the frontend proxy
// listeners, and the idle-sweep loop. Returns only on signal.
func (h *ClusterHead) Run() {
	ctx, cancel := signalContext()
	defer cancel()

	// Idle sweeper.
	go h.idleSweepLoop(ctx)

	// Control plane.
	go h.startControlPlane(ctx)

	// Frontend proxies (one per backend).
	for backend, port := range h.FrontendPorts {
		go h.startFrontend(ctx, backend, port)
	}

	log.Printf("[head] up: control-plane=:%d frontends=%v",
		h.HeadListenPort, h.FrontendPorts)
	<-ctx.Done()
	log.Println("[head] context cancelled; exiting")
}

// startControlPlane mounts /v1/cluster/{register,heartbeat,status}
// behind the bearer-token middleware.
func (h *ClusterHead) startControlPlane(ctx context.Context) {
	mux := http.NewServeMux()
	mux.Handle("/v1/cluster/register",
		h.Token.AuthMiddleware(http.HandlerFunc(h.handleRegister)))
	mux.Handle("/v1/cluster/heartbeat",
		h.Token.AuthMiddleware(http.HandlerFunc(h.handleHeartbeat)))
	mux.HandleFunc("/v1/cluster/status", h.handleStatus)
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("OK"))
	})
	srv := &http.Server{
		Addr:              fmt.Sprintf(":%d", h.HeadListenPort),
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}
	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutdownCtx)
	}()
	log.Printf("[head] control plane listening on %s", srv.Addr)
	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatalf("[head] control plane: %v", err)
	}
}

// startFrontend mounts the OpenAI-compat proxy on `port` for `backend`.
// All paths are forwarded; routing decision happens in the handler.
func (h *ClusterHead) startFrontend(ctx context.Context, backend string, port int) {
	mux := http.NewServeMux()
	mux.HandleFunc("/", h.makeFrontendHandler(backend))
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("OK"))
	})
	srv := &http.Server{
		Addr:              fmt.Sprintf(":%d", port),
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}
	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutdownCtx)
	}()
	log.Printf("[head] frontend %s listening on %s", backend, srv.Addr)
	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Printf("[head] frontend %s: %v", backend, err)
	}
}

// idleSweepLoop runs FleetState.ExpireOlderThan on a ticker.
func (h *ClusterHead) idleSweepLoop(ctx context.Context) {
	ticker := time.NewTicker(h.IdleSweepInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			n := h.Fleet.ExpireOlderThan(time.Now().Add(-h.Fleet.HeartbeatTTL))
			if n > 0 {
				log.Printf("[head] expired %d stale worker(s)", n)
			}
		}
	}
}

// handleRegister POST -> assigns worker_id, returns it.
func (h *ClusterHead) handleRegister(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "read body: "+err.Error(), http.StatusBadRequest)
		return
	}
	var req RegisterRequest
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, "decode: "+err.Error(), http.StatusBadRequest)
		return
	}
	if err := req.Validate(); err != nil {
		http.Error(w, "invalid registration: "+err.Error(), http.StatusBadRequest)
		return
	}
	id := h.Fleet.Register(req, time.Now())
	log.Printf("[head] register: %s -> %s (lifecycle=%s, gpu=%s)",
		req.Name, id, req.Lifecycle, req.GPUType)
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(RegisterResponse{WorkerID: id})
}

// handleHeartbeat POST -> updates state, returns commands list.
func (h *ClusterHead) handleHeartbeat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "read body: "+err.Error(), http.StatusBadRequest)
		return
	}
	var hb HeartbeatRequest
	if err := json.Unmarshal(body, &hb); err != nil {
		http.Error(w, "decode: "+err.Error(), http.StatusBadRequest)
		return
	}
	now := time.Now()
	res := h.Fleet.Heartbeat(hb, now)
	w.Header().Set("Content-Type", "application/json")
	switch res {
	case HeartbeatStale:
		// Don't error -- the worker's clock might just be racy.
		// Return empty commands; the next heartbeat (with a higher
		// counter) will be accepted.
		_ = json.NewEncoder(w).Encode(HeartbeatResponse{})
		return
	case HeartbeatUnknownWorker:
		// Tell the worker to re-register: 410 Gone is the closest
		// HTTP semantic. Worker logs and retries register on this.
		http.Error(w, "unknown worker_id; please re-register", http.StatusGone)
		return
	}
	cmds := h.commandsFor(hb.WorkerID, now)
	_ = json.NewEncoder(w).Encode(HeartbeatResponse{Commands: cmds})
}

// commandsFor returns the per-worker command list for this heartbeat.
// Today: optionally inject a `shutdown` command for ephemeral
// workers that have been idle for IdleMinutes. Phase 3 may add
// `serve` commands here once head-side request coalescing lands.
func (h *ClusterHead) commandsFor(workerID string, now time.Time) []Command {
	if h.IdleMinutes <= 0 {
		return nil
	}
	w, ok := h.Fleet.Get(workerID)
	if !ok || w.Lifecycle != LifecycleEphemeral {
		return nil
	}
	if w.LastRequestAt == "" {
		return nil
	}
	lastReq, err := time.Parse(time.RFC3339, w.LastRequestAt)
	if err != nil {
		return nil
	}
	if now.Sub(lastReq) < time.Duration(h.IdleMinutes)*time.Minute {
		return nil
	}
	return []Command{{Type: CommandShutdown, GraceSeconds: 30}}
}

// handleStatus GET -> JSON array of WorkerEntry. No auth (read-only;
// per cluster-mode plan v1; revisit if hostile networks become a
// concern).
func (h *ClusterHead) handleStatus(w http.ResponseWriter, _ *http.Request) {
	entries := h.Fleet.Snapshot()
	out := make([]StatusEntry, 0, len(entries))
	for _, e := range entries {
		out = append(out, StatusEntry{
			WorkerID:       e.WorkerID,
			Name:           e.Name,
			Lifecycle:      e.Lifecycle,
			GPUType:        e.GPUType,
			VRAMGB:         e.VRAMGB,
			Backends:       append([]string(nil), e.Backends...),
			Endpoint:       e.Endpoint,
			LoadedModel:    e.LoadedModel,
			LoadedCtx:      e.LoadedCtx,
			QueueDepth:     e.QueueDepth,
			UtilizationPct: e.UtilizationPct,
			LastHeartbeat:  e.LastHeartbeat.UTC().Format(time.RFC3339),
			HealthStatus:   e.HealthStatus,
			Counter:        e.Counter,
		})
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(out)
}

// makeFrontendHandler is the per-backend OpenAI-compat handler.
// Reads the body once (we need it both for routing and for forwarding),
// extracts the minimal model+ctx+reasoning hint, picks a worker, and
// hands off to the configured Forward implementation.
func (h *ClusterHead) makeFrontendHandler(backend string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "read body: "+err.Error(), http.StatusBadRequest)
			return
		}
		// Restore a fresh body for the proxy.
		r.Body = io.NopCloser(bytes.NewReader(body))

		parsed, err := ParseMinimal(body)
		if err != nil {
			http.Error(w,
				`{"error":{"type":"invalid_request_error","message":"`+
					escapeForJSON(err.Error())+`"}}`,
				http.StatusBadRequest)
			return
		}

		workers := h.Fleet.Snapshot()
		decision := h.Policy.RouteDecision(workers, parsed.Model, parsed.Context, backend)
		if decision.WorkerID == "" {
			h.respondNoFit(w, parsed, backend, decision.NoFitReason)
			return
		}
		entry, ok := h.Fleet.Get(decision.WorkerID)
		if !ok {
			// Worker expired between Snapshot and Get -- rare race;
			// 503 + retry-after is the right answer.
			w.Header().Set("Retry-After", "1")
			http.Error(w, "selected worker disappeared; retry", http.StatusServiceUnavailable)
			return
		}
		h.Forward.Forward(w, r, entry, parsed)
	}
}

// respondNoFit emits a structured JSON error so callers can see why
// no worker matched. Encodes 503 + Retry-After to nudge clients into
// retrying after a brief backoff (the operator may be in the middle
// of bringing a worker up).
func (h *ClusterHead) respondNoFit(
	w http.ResponseWriter, parsed MinimalRequest, backend, reason string,
) {
	w.Header().Set("Retry-After", "5")
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusServiceUnavailable)
	payload := map[string]interface{}{
		"error": map[string]interface{}{
			"type":    "no_worker_fit",
			"message": reason,
			"requested": map[string]interface{}{
				"model":   parsed.Model,
				"context": parsed.Context,
				"backend": backend,
			},
		},
	}
	_ = json.NewEncoder(w).Encode(payload)
}

// escapeForJSON does the bare minimum for a JSON string field: quote
// + backslash escaping. Avoids pulling encoding/json for a one-off
// error message.
func escapeForJSON(s string) string {
	r := strings.NewReplacer(`\`, `\\`, `"`, `\"`, "\n", `\n`, "\r", `\r`)
	return r.Replace(s)
}

// init wires runHeadMode (declared in cluster_main.go) to a real
// implementation by replacing the swappable factory.
func init() {
	newClusterHead = func() interface{ Run() } {
		return NewClusterHead()
	}
}

// _ is here to keep `sync` referenced when the proxy file (which
// also imports it) hasn't been compiled in yet (tests build single
// files).
var _ = sync.Mutex{}
