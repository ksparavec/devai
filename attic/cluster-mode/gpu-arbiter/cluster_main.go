//go:build devai_frozen_cluster

// Cluster-mode entrypoints. main() dispatches here when --mode is
// not "single". Head mode is stand-alone (no local GPU backends);
// worker mode builds the same single-host arbiter main() would and
// serves head-forwarded requests through it.

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync/atomic"
	"syscall"
	"time"
)

// runWorkerMode brings up the worker control plane: register with
// head, heartbeat every 10s, accept inbound forwarded requests on
// /v1/cluster/inbound. It also builds the ordinary single-host
// arbiter (buildArbiter) and dispatches every forwarded request
// through that scheduler's request handler, so the worker applies the
// full documented rewrite chain -- override parsing, reasoning policy,
// tool_choice promotion, tool stripping, ctx injection -- exactly as
// it would in single mode (cluster-mode decision 2).
//
// The worker does NOT mount the per-backend OpenAI-compat listeners:
// on a cluster host those ports are fronted by the head, and
// /v1/cluster/inbound is the documented head->worker surface.
func runWorkerMode() {
	headURL := strings.TrimRight(env("DEVAI_HEAD_URL", ""), "/")
	if headURL == "" {
		log.Fatal("worker mode: DEVAI_HEAD_URL is required")
	}

	tokenPath := env("DEVAI_WORKER_TOKEN_FILE", "/run/devai/cluster-token")
	tokens := NewTokenStore(tokenPath, 30*time.Second)
	if _, err := tokens.Read(); err != nil {
		log.Fatalf("worker mode: %v", err)
	}

	listenPort := envInt("DEVAI_WORKER_INBOUND_PORT", 11444)
	// The endpoint is what the HEAD dials, so it must be reachable
	// from another host: default to this host's name (the contract
	// deploy/worker-cloud-init.sh documents), never "localhost".
	endpoint := env("DEVAI_WORKER_ENDPOINT",
		fmt.Sprintf("http://%s:%d",
			env("DEVAI_WORKER_HOST", hostnameOrDefault("localhost")), listenPort))

	cfg := WorkerConfig{
		HeadURL:        headURL,
		WorkerName:     env("DEVAI_WORKER_NAME", hostnameOrDefault("devai-worker")),
		Lifecycle:      LifecycleClass(env("DEVAI_LIFECYCLE", string(LifecyclePersistent))),
		Endpoint:       endpoint,
		GPUType:        env("DEVAI_GPU_TYPE", "unknown"),
		VRAMGB:         envInt("GPU_MEMORY_GB", 24),
		Backends:       splitAndTrim(env("DEVAI_BACKENDS", "ollama,vllm,sglang"), ","),
		ArbiterVersion: env("DEVAI_ARBITER_VERSION", "dev"),
		Token:          tokens,
	}

	ctx, cancel := signalContext()
	defer cancel()

	worker := NewWorker(cfg)

	// The same scheduler single mode runs. Its idle watcher honours
	// IDLE_TIMEOUT here too, so a worker unloads on the same policy as
	// a standalone host.
	arb := buildArbiter()
	go arb.idleWatcher()

	// The executor drives that same scheduler: a head-issued drain
	// really drains the backend, and a shutdown drains before exiting.
	worker.Executor = newArbiterCommandExecutor(worker.State, arb)

	mux := http.NewServeMux()
	mux.Handle(
		"/v1/cluster/inbound",
		tokens.AuthMiddleware(http.HandlerFunc(makeInboundHandler(worker, arb))),
	)
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("OK"))
	})

	srv := &http.Server{
		Addr:              fmt.Sprintf(":%d", listenPort),
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}
	go func() {
		log.Printf("[worker] inbound listener on %s", srv.Addr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("[worker] inbound listener failed: %v", err)
		}
	}()

	if err := worker.Start(ctx); err != nil && err != context.Canceled {
		log.Fatalf("[worker] %v", err)
	}

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer shutdownCancel()
	_ = srv.Shutdown(shutdownCtx)
}

// runHeadMode is a stand-alone entrypoint placeholder. Phase 2 of
// cluster-mode wires in fleet state + routing + proxy; until that
// lands, runHeadMode reports the absence and exits non-zero so the
// operator sees a clear "not implemented yet" signal instead of a
// silent no-op.
func runHeadMode() {
	if hh := newClusterHead(); hh != nil {
		hh.Run()
		return
	}
	log.Fatal("head mode: cluster-mode Phase 2 head implementation not yet linked into this binary")
}

// arbiterCommandExecutor executes head-issued commands against the
// worker's REAL single-host scheduler. Lifecycle-policy gating (a
// persistent worker refuses shutdown) is already applied upstream in
// Worker.dispatchCommand; this type only carries out what survives it.
//
// Drain and shutdown both call arbiter.drainBackend, which waits for
// requests already proxied upstream to finish (bounded by
// DRAIN_TIMEOUT). Without that, a `shutdown` hard-exited the process
// mid-stream on every in-flight request.
type arbiterCommandExecutor struct {
	state *WorkerState
	arb   *arbiter

	// exit terminates the process after a shutdown drain. os.Exit in
	// production; tests substitute a recorder.
	exit func(int)

	// background runs the drain off the heartbeat goroutine. A drain
	// can block for DRAIN_TIMEOUT (default 30s) -- the same as the
	// head's HeartbeatTTL -- so doing it inline in Worker.tick would
	// stall heartbeats long enough for the head to expire the worker
	// exactly when it most needs the "draining" status to propagate.
	// Tests substitute a synchronous runner.
	background func(func())

	// inFlightDrains counts drains that have been started and not yet
	// finished. Only the LAST one to finish returns the worker to
	// HealthReady: with a per-drain reset, drain(vllm) completing would
	// advertise "ready" while a concurrently-issued drain(sglang) was
	// still waiting out its own in-flight requests.
	inFlightDrains atomic.Int64
}

func newArbiterCommandExecutor(state *WorkerState, a *arbiter) *arbiterCommandExecutor {
	return &arbiterCommandExecutor{
		state:      state,
		arb:        a,
		exit:       os.Exit,
		background: func(f func()) { go f() },
	}
}

func (e *arbiterCommandExecutor) Execute(_ context.Context, cmd Command) error {
	switch cmd.Type {
	case CommandDrain:
		// Status first: it rides the next heartbeat, and the head's
		// routing policy skips a draining worker (see workerAvailable).
		// That is what actually stops NEW work arriving -- the drain
		// below only waits out what is already in flight.
		//
		// A drain is NOT terminal. Count it in before flipping the
		// status so a second drain arriving mid-flight cannot see a
		// zero counter and let the first one declare "ready" early.
		e.inFlightDrains.Add(1)
		// Never downgrade a terminal shutdown to draining: once the
		// process is on its way out, "draining" would invite the head
		// to consider the worker recoverable.
		e.state.SetHealthUnlessShuttingDown(HealthDraining)
		log.Printf("[worker] drain backend=%s acknowledged; waiting for "+
			"in-flight upstream requests", cmd.Backend)
		e.background(func() {
			defer e.finishDrain(cmd.Backend)
			e.drain(cmd.Backend)
		})
	case CommandShutdown:
		e.state.SetHealthStatus(HealthShuttingDown)
		log.Printf("[worker] shutdown grace=%ds acknowledged; draining before exit",
			cmd.GraceSeconds)
		e.background(func() {
			// grace_seconds is the head's own budget before it calls
			// `sky down`; we do not sleep it out. We drain every
			// backend (each bounded by DRAIN_TIMEOUT) and exit as soon
			// as in-flight work is done, so a fast drain exits fast and
			// a slow one is not cut off at grace.
			e.drain("")
			log.Printf("[worker] drain complete; exiting")
			e.exit(0)
		})
	case CommandServe:
		log.Printf("[worker] serve req=%s model=%s ctx=%d",
			cmd.RequestID, cmd.TargetModel, cmd.TargetCtx)
		if cmd.TargetModel != "" {
			e.state.SetLoadedModel(cmd.TargetModel, cmd.TargetCtx)
		}
		e.state.MarkRequestAt(time.Now())
		// Deliberately does NOT touch health. Drain and shutdown own the
		// health lifecycle, and drain COMPLETION is the only path back to
		// HealthReady -- a serve flipping the status would both cut a
		// drain short (advertising "ready" while requests are still being
		// waited out) and resurrect a worker whose process is exiting.
	}
	return nil
}

// finishDrain runs after a background drain returns. It restores the
// worker to a routable state so a drain is a transient pause, not a
// one-way latch that removes the worker from the fleet forever -- but
// only when this was the LAST in-flight drain, and only via a CAS from
// HealthDraining, so a shutdown that arrived mid-drain stays terminal.
func (e *arbiterCommandExecutor) finishDrain(backend string) {
	if e.inFlightDrains.Add(-1) > 0 {
		log.Printf("[worker] drain backend=%s complete; %d drain(s) still "+
			"running, staying %s", backend, e.inFlightDrains.Load(), HealthDraining)
		return
	}
	if e.state.CompareAndSwapHealth(HealthDraining, HealthReady) {
		log.Printf("[worker] drain backend=%s complete; back to %s",
			backend, HealthReady)
		return
	}
	log.Printf("[worker] drain backend=%s complete; health=%s (not returning "+
		"to %s)", backend, e.state.HealthStatus(), HealthReady)
}

// drain waits out the requests already proxied upstream on `backend`
// (empty = every backend). a.mu is held across the wait, which is
// drainBackend's documented contract: it watches upstreamReqs, and a
// request still parked on the mutex cannot make progress -- so holding
// it also stops new requests from entering while we drain.
func (e *arbiterCommandExecutor) drain(backend string) {
	if e.arb == nil {
		return
	}
	e.arb.mu.Lock()
	defer e.arb.mu.Unlock()
	if backend != "" {
		bs, ok := e.arb.backends[backend]
		if !ok {
			log.Printf("[worker] drain: unknown backend %q; nothing to drain", backend)
			return
		}
		e.arb.drainBackend(bs)
		return
	}
	for _, bs := range e.arb.backends {
		e.arb.drainBackend(bs)
	}
}

// makeInboundHandler dispatches a head-forwarded request through the
// ordinary single-host request handler for the target backend, per
// cluster-mode decision 2: the whole rewrite chain stays on the
// worker, so a clustered request is mutated exactly like a local one.
//
// Backend selection: the head's frontend listeners are one-per-backend
// so only the head knows which one the client hit; it says so in
// HeaderBackend. When that header is absent (a caller talking to the
// inbound endpoint directly) we fall back to looking the model name up
// in the worker's registered models.
//
// Path: HeaderOriginalPath restores the client's original request path
// (e.g. /v1/chat/completions), which the request handler needs to
// decide whether options.num_ctx injection applies.
func makeInboundHandler(worker *Worker, a *arbiter) http.HandlerFunc {
	handlers := make(map[string]http.HandlerFunc, len(a.backends))
	for name := range a.backends {
		handlers[name] = a.makeRequestHandler(name)
	}
	return inboundHandler(worker, a, handlers)
}

// inboundHandler is makeInboundHandler's body with the per-backend
// handler map passed in, so tests can substitute stubs for the real
// (podman-driving) request handlers.
func inboundHandler(
	worker *Worker, a *arbiter, handlers map[string]http.HandlerFunc,
) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		worker.State.IncQueue()
		defer worker.State.DecQueue()

		backend := r.Header.Get(HeaderBackend)
		if backend == "" {
			body, err := io.ReadAll(
				http.MaxBytesReader(w, r.Body, ClusterMaxBodyBytes))
			if err != nil {
				writeInboundError(w, http.StatusRequestEntityTooLarge,
					"request_too_large",
					fmt.Sprintf("read body (limit %d bytes): %v",
						ClusterMaxBodyBytes, err))
				return
			}
			r.Body = io.NopCloser(bytes.NewReader(body))
			r.ContentLength = int64(len(body))
			if parsed, perr := ParseMinimal(body); perr == nil {
				backend = a.backendForModel(parsed.Model, parsed.Context)
			}
		}
		handler, ok := handlers[backend]
		if !ok {
			writeInboundError(w, http.StatusBadRequest, "invalid_request_error",
				fmt.Sprintf("cannot determine target backend (%s=%q): set the "+
					"header or send a model this worker serves",
					HeaderBackend, backend))
			return
		}

		if p := r.Header.Get(HeaderOriginalPath); strings.HasPrefix(p, "/") {
			r.URL.Path = p
		}

		handler(w, r)

		// Report what the backend is now holding so the head's routing
		// scoring (loaded_model / loaded_ctx) reflects reality on the
		// next heartbeat.
		if model, loadedCtx := a.currentLoaded(backend); model != "" {
			worker.State.SetLoadedModel(model, loadedCtx)
		}
		worker.State.MarkRequestAt(time.Now())
	}
}

// writeInboundError emits an OpenAI-shaped error envelope, matching
// what the single-host handler returns for its own rejections.
func writeInboundError(w http.ResponseWriter, status int, kind, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	payload := map[string]any{
		"error": map[string]any{"type": kind, "message": msg},
	}
	_ = json.NewEncoder(w).Encode(payload)
}

// backendPreferenceOrder is the deterministic tie-break used by
// backendForModel when probe evidence cannot separate two backends
// that both serve the same model name.
var backendPreferenceOrder = []string{"ollama", "vllm", "sglang"}

// backendForModel resolves a backend for `model` at `ctx`. This is the
// FALLBACK path only: the head always sets HeaderBackend (its frontend
// listeners are one-per-backend, so only it knows which port the client
// hit), and the project's invariant is "no message inspection -- port
// determines backend". Reaching here means someone POSTed to
// /v1/cluster/inbound without the header.
//
// Preference order, most-specific first:
//  1. a backend whose probe cache has a fitting cell covering `ctx`
//     (probed max ctx >= ctx) -- the only tier with real evidence that
//     the request can actually be served;
//  2. a backend with any fitting probe cell for the model (probed max
//     ctx > 0), highest ceiling first -- ctx was 0/unstated, or no
//     backend reaches it, so the best-probed one is the best guess;
//  3. bare catalog membership, in backendPreferenceOrder.
//
// Empty string = no backend serves this name.
func (a *arbiter) backendForModel(model string, ctx int) string {
	if model == "" {
		return ""
	}
	serving := make([]string, 0, len(backendPreferenceOrder))
	for _, name := range backendPreferenceOrder {
		bs, ok := a.backends[name]
		if !ok {
			continue
		}
		for _, n := range bs.modelNames {
			if n == model {
				serving = append(serving, name)
				break
			}
		}
	}
	if len(serving) == 0 {
		return ""
	}

	// Tier 1 + 2: rank by probe evidence. modelProbedMaxCtx is a nil
	// map on an unprobed arbiter; nil-map reads return 0, which lands
	// every candidate in tier 3.
	bestFitting, bestProbed := "", ""
	bestFittingCtx, bestProbedCtx := 0, 0
	for _, name := range serving {
		probed := a.modelProbedMaxCtx[name][model]
		if probed <= 0 {
			continue
		}
		if ctx > 0 && probed >= ctx && probed > bestFittingCtx {
			bestFitting, bestFittingCtx = name, probed
		}
		if probed > bestProbedCtx {
			bestProbed, bestProbedCtx = name, probed
		}
	}
	if bestFitting != "" {
		return bestFitting
	}
	if bestProbed != "" {
		return bestProbed
	}
	return serving[0]
}

// currentLoaded reports the (model, context) the backend is currently
// running, or ("", 0) when nothing is loaded / the backend is unknown.
func (a *arbiter) currentLoaded(backend string) (string, int) {
	a.mu.Lock()
	defer a.mu.Unlock()
	bs, ok := a.backends[backend]
	if !ok {
		return "", 0
	}
	return bs.currentModel, bs.currentContext
}

// signalContext returns a context that cancels on SIGINT/SIGTERM.
// Used by both worker and head main loops to coordinate shutdown.
func signalContext() (context.Context, context.CancelFunc) {
	ctx, cancel := context.WithCancel(context.Background())
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sig
		log.Println("[cluster] received signal; cancelling context")
		cancel()
	}()
	return ctx, cancel
}

// splitAndTrim returns sep-separated tokens with whitespace trimmed
// and empty strings dropped.
func splitAndTrim(s, sep string) []string {
	out := make([]string, 0)
	for _, p := range strings.Split(s, sep) {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

// hostnameOrDefault returns os.Hostname() or `def` on error.
func hostnameOrDefault(def string) string {
	if h, err := os.Hostname(); err == nil && h != "" {
		return h
	}
	return def
}

// newClusterHead is a hook the Phase 2 head implementation will
// replace by returning a non-nil *clusterHead. Until then it
// returns nil and runHeadMode logs the absence.
var newClusterHead = func() interface{ Run() } { return nil }