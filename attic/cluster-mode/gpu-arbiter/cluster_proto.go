//go:build devai_frozen_cluster

// Package main: cluster-mode protocol types.
//
// Shared between worker (POST /v1/cluster/{register,heartbeat}) and
// head (registration receiver, command issuer). Plain JSON over HTTP
// + bearer token; no gRPC, no NATS -- per docs/plans/gpu-arbiter-cluster-mode.md
// decision 6.
//
// All types here MUST stay backward-compatible across patch releases.
// Add fields as `omitempty` so older workers don't fail JSON-decode
// on heartbeats from newer heads (or vice versa).

package main

import (
	"fmt"
	"strings"
	"time"
)

// LifecycleClass declares whether a worker honours head-issued
// shutdown commands. Per cluster-mode decision 3.
type LifecycleClass string

const (
	// LifecycleEphemeral: head MAY issue shutdown after
	// DEVAI_IDLE_MINUTES of no requests; worker honours it. Default
	// for SkyPilot-launched cloud workers.
	LifecycleEphemeral LifecycleClass = "ephemeral"

	// LifecyclePersistent: head MUST NOT issue shutdown commands.
	// Worker stays up indefinitely so loaded models stay loaded.
	// Default for systemd-launched on-prem workers.
	LifecyclePersistent LifecycleClass = "persistent"
)

// IsValid returns true for the two recognised lifecycle classes.
func (lc LifecycleClass) IsValid() bool {
	return lc == LifecycleEphemeral || lc == LifecyclePersistent
}

// RegisterRequest is the body of POST /v1/cluster/register.
// Worker -> head once at startup; retried with exponential backoff
// until accepted.
type RegisterRequest struct {
	Name           string         `json:"name"`
	Lifecycle      LifecycleClass `json:"lifecycle"`
	GPUType        string         `json:"gpu_type"`
	VRAMGB         int            `json:"vram_gb"`
	Backends       []string       `json:"backends"`
	ArbiterVersion string         `json:"arbiter_version"`
	Endpoint       string         `json:"endpoint"`
}

// Validate ensures the registration body is internally consistent.
// Head SHOULD reject malformed registrations with HTTP 400.
func (r RegisterRequest) Validate() error {
	if strings.TrimSpace(r.Name) == "" {
		return fmt.Errorf("name is required")
	}
	if !r.Lifecycle.IsValid() {
		return fmt.Errorf(
			"lifecycle %q is not one of (%q, %q)",
			r.Lifecycle, LifecycleEphemeral, LifecyclePersistent,
		)
	}
	if strings.TrimSpace(r.Endpoint) == "" {
		return fmt.Errorf("endpoint is required")
	}
	if len(r.Backends) == 0 {
		return fmt.Errorf("at least one backend must be advertised")
	}
	if r.VRAMGB <= 0 {
		return fmt.Errorf("vram_gb must be positive (got %d)", r.VRAMGB)
	}
	return nil
}

// RegisterResponse is the body returned by head on a successful
// registration. WorkerID is the opaque string the worker MUST send
// in every subsequent heartbeat so head can route commands.
type RegisterResponse struct {
	WorkerID string `json:"worker_id"`
}

// HeartbeatRequest is the body of POST /v1/cluster/heartbeat. Sent
// every HeartbeatInterval (10s by default per cluster-mode decision
// 6). Counter is monotonically increasing per worker so head can
// drop stale messages.
type HeartbeatRequest struct {
	WorkerID       string  `json:"worker_id"`
	LoadedModel    string  `json:"loaded_model,omitempty"`
	LoadedCtx      int     `json:"loaded_ctx,omitempty"`
	QueueDepth     int     `json:"queue_depth"`
	UtilizationPct float64 `json:"utilization_pct"`
	LastRequestAt  string  `json:"last_request_at,omitempty"` // RFC3339
	HealthStatus   string  `json:"health_status"`
	Counter        uint64  `json:"counter"`
}

// HeartbeatInterval is the canonical 10-second poll cadence per
// cluster-mode decision 6.
const HeartbeatInterval = 10 * time.Second

// Head -> worker request headers on POST /v1/cluster/inbound. The
// head's frontend listeners are one-per-backend (11434/5/6), so the
// head is the only party that knows which backend the client asked
// for -- the forwarded body carries no backend field. The worker
// falls back to a model-name lookup when HeaderBackend is absent.
const (
	// HeaderBackend names the backend the head's frontend received
	// the request on: "ollama" | "vllm" | "sglang".
	HeaderBackend = "X-Devai-Backend"

	// HeaderOriginalPath carries the client's original request path
	// (e.g. /v1/chat/completions) so the worker can route it to the
	// right upstream surface.
	HeaderOriginalPath = "X-Devai-Original-Path"

	// HeaderWorkerID echoes the head's chosen worker_id; informational
	// (the worker logs it) -- the bearer token is what authenticates.
	HeaderWorkerID = "X-Devai-Worker-Id"
)

// ClusterMaxBodyBytes bounds every body the cluster control plane and
// the head frontends read into memory. Mirrors the single-host request
// handler's cap: without it any peer that reaches the port can stream
// an arbitrarily large body and exhaust RAM.
const ClusterMaxBodyBytes = 32 << 20

// CommandType enumerates the actions the head can request via a
// heartbeat response. Unknown types are logged and ignored on the
// worker side -- forward compatibility.
type CommandType string

const (
	// CommandDrain asks the worker to drain in-flight requests on a
	// specific backend (vllm/sglang) without exiting.
	CommandDrain CommandType = "drain"

	// CommandShutdown asks the worker to drain every backend and
	// exit the arbiter process. Persistent workers refuse this
	// command (logged + ignored).
	CommandShutdown CommandType = "shutdown"

	// CommandServe forwards a request body the head wants this
	// worker to handle. The body is fetched from BodyURL (a
	// head-internal URL) so the heartbeat response stays small.
	CommandServe CommandType = "serve"
)

// Command is the per-instruction payload returned in HeartbeatResponse.
// Fields are union-by-Type; consumers branch on Type and read only the
// fields relevant for that variant.
type Command struct {
	Type         CommandType `json:"type"`
	Backend      string      `json:"backend,omitempty"`       // drain
	GraceSeconds int         `json:"grace_seconds,omitempty"` // shutdown
	RequestID    string      `json:"request_id,omitempty"`    // serve
	TargetModel  string      `json:"target_model,omitempty"`  // serve
	TargetCtx    int         `json:"target_ctx,omitempty"`    // serve
	BodyURL      string      `json:"body_url,omitempty"`      // serve
	ResponsePath string      `json:"response_path,omitempty"` // serve
}

// HeartbeatResponse is what head returns to a worker's heartbeat
// POST. Empty Commands list = "nothing to do". Commands are executed
// in order; a worker that fails one command logs the failure and
// continues with the next.
type HeartbeatResponse struct {
	Commands []Command `json:"commands"`
}

// StatusEntry is one row in the GET /v1/cluster/status JSON array.
// Operator-facing only; not part of the worker protocol.
type StatusEntry struct {
	WorkerID       string         `json:"worker_id"`
	Name           string         `json:"name"`
	Lifecycle      LifecycleClass `json:"lifecycle"`
	GPUType        string         `json:"gpu_type"`
	VRAMGB         int            `json:"vram_gb"`
	Backends       []string       `json:"backends"`
	Endpoint       string         `json:"endpoint"`
	LoadedModel    string         `json:"loaded_model"`
	LoadedCtx      int            `json:"loaded_ctx"`
	QueueDepth     int            `json:"queue_depth"`
	UtilizationPct float64        `json:"utilization_pct"`
	LastHeartbeat  string         `json:"last_heartbeat"`
	HealthStatus   string         `json:"health_status"`
	Counter        uint64         `json:"counter"`
}