// Cluster-mode entrypoints. main() dispatches here when --mode is
// not "single". Both runners are stand-alone -- they do NOT call
// into the single-host scheduling code.

package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"
)

// runWorkerMode brings up the worker control plane: register with
// head, heartbeat every 10s, accept inbound forwarded requests on
// /v1/cluster/inbound. The single-host scheduling code path is NOT
// started -- worker-mode containers are typically minimal-bootstrap
// images that don't carry the backend containers locally; head-side
// dispatch is what brings the worker into a real serving state via
// the `serve` command (cluster-mode Phase 2).
//
// Phase 1 ships the protocol and the inbound endpoint; the actual
// `serve` command handler stays a thin pass-through until Phase 2
// wires in head-side routing.
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
	endpoint := env("DEVAI_WORKER_ENDPOINT",
		fmt.Sprintf("http://%s:%d", env("DEVAI_WORKER_HOST", "localhost"), listenPort))

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
	worker.Executor = &noopCommandExecutor{state: worker.State}

	mux := http.NewServeMux()
	mux.Handle(
		"/v1/cluster/inbound",
		tokens.AuthMiddleware(http.HandlerFunc(makeInboundHandler(worker))),
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

// noopCommandExecutor is the Phase 1 executor. It honours the
// lifecycle-policy gating (already done upstream in
// Worker.dispatchCommand) and updates worker state for `serve`
// commands but does not actually proxy a request body -- the
// response-path / body-fetch wiring lands in Phase 2.
type noopCommandExecutor struct {
	state *WorkerState
}

func (n *noopCommandExecutor) Execute(_ context.Context, cmd Command) error {
	switch cmd.Type {
	case CommandDrain:
		log.Printf("[worker] drain backend=%s acknowledged (Phase 1: no-op)", cmd.Backend)
		n.state.SetHealthStatus("draining")
	case CommandShutdown:
		log.Printf("[worker] shutdown grace=%ds acknowledged; exiting", cmd.GraceSeconds)
		n.state.SetHealthStatus("shutting_down")
		go func(grace int) {
			if grace > 0 {
				time.Sleep(time.Duration(grace) * time.Second)
			}
			os.Exit(0)
		}(cmd.GraceSeconds)
	case CommandServe:
		log.Printf("[worker] serve req=%s model=%s ctx=%d (Phase 1: state-only)",
			cmd.RequestID, cmd.TargetModel, cmd.TargetCtx)
		if cmd.TargetModel != "" {
			n.state.SetLoadedModel(cmd.TargetModel, cmd.TargetCtx)
		}
		n.state.MarkRequestAt(time.Now())
	}
	return nil
}

// makeInboundHandler returns a placeholder handler for forwarded
// requests. Phase 2 replaces this with the existing single-host
// makeRequestHandler dispatch chain (per cluster-mode decision 2).
func makeInboundHandler(worker *Worker) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		worker.State.IncQueue()
		defer worker.State.DecQueue()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(
			`{"error":{"type":"not_implemented","message":"cluster-mode Phase 1: ` +
				`inbound serve handler is a placeholder; Phase 2 wires routing"}}`,
		))
	}
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
