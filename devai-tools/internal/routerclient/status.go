// Package routerclient implements get_router_status: a best-effort probe
// of the running devai-router, tolerant of both cluster-head and
// single-host deployments.
package routerclient

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
)

// backendPortEnvVar / defaultPort mirror gpu-arbiter's own OLLAMA_PORT/
// VLLM_PORT/SGLANG_PORT defaults (main.go) -- these are the router's own
// published ports, not the upstream backend containers' internal ports.
// Each is independently overridable so a non-default port mapping (or a
// test double) doesn't require a matching change on the arbiter side.
var backendPortEnvVar = map[string]string{
	"ollama": "DEVAI_ROUTER_OLLAMA_PORT",
	"vllm":   "DEVAI_ROUTER_VLLM_PORT",
	"sglang": "DEVAI_ROUTER_SGLANG_PORT",
}

var defaultBackendPorts = []struct {
	Backend string
	Port    int
}{
	{"ollama", 11434},
	{"vllm", 11435},
	{"sglang", 11436},
}

// BackendHealth is one backend's /health result in single mode. The
// router's health handler (gpu-arbiter's makeHealthHandler) returns JSON
// with running/current_model/active_reqs, not a plain "OK" -- surfaced
// here rather than collapsed to a bare boolean.
type BackendHealth struct {
	Backend      string `json:"backend"`
	Port         int    `json:"port"`
	Reachable    bool   `json:"reachable"`
	Running      *bool  `json:"running,omitempty"`
	CurrentModel string `json:"current_model,omitempty"`
	ActiveReqs   *int64 `json:"active_reqs,omitempty"`
	Error        string `json:"error,omitempty"`
}

// Status is get_router_status's result. Mode is "cluster-head" (the
// /v1/cluster/status probe succeeded), "single" (cluster status
// unreachable, at least one backend health check succeeded), or
// "unreachable" (nothing responded).
type Status struct {
	Mode     string          `json:"mode"`
	Workers  json.RawMessage `json:"workers,omitempty"`
	Backends []BackendHealth `json:"backends,omitempty"`
	Error    string          `json:"error,omitempty"`
}

type healthBody struct {
	Running      bool   `json:"running"`
	CurrentModel string `json:"current_model"`
	ActiveReqs   int64  `json:"active_reqs"`
}

// GetStatus tries the cluster-head control-plane endpoint first, falling
// back to per-backend /health probes (the single-mode case) on any
// failure to reach it -- never returns an error itself, only a Status
// describing what was reachable.
func GetStatus(ctx context.Context, client *http.Client) Status {
	clusterURL := env("DEVAI_ROUTER_CLUSTER_URL", "http://host.containers.internal:11444")
	if body, err := getJSON(ctx, client, clusterURL+"/v1/cluster/status"); err == nil {
		return Status{Mode: "cluster-head", Workers: body}
	}

	host := env("DEVAI_ROUTER_HOST", "host.containers.internal")
	backends := make([]BackendHealth, 0, len(defaultBackendPorts))
	anyReachable := false
	for _, b := range defaultBackendPorts {
		port := envInt(backendPortEnvVar[b.Backend], b.Port)
		bh := probeBackendHealth(ctx, client, host, b.Backend, port)
		if bh.Reachable {
			anyReachable = true
		}
		backends = append(backends, bh)
	}

	if !anyReachable {
		return Status{
			Mode:     "unreachable",
			Backends: backends,
			Error:    "cluster status endpoint and all backend health checks failed",
		}
	}
	return Status{Mode: "single", Backends: backends}
}

func probeBackendHealth(ctx context.Context, client *http.Client, host, backend string, port int) BackendHealth {
	bh := BackendHealth{Backend: backend, Port: port}
	url := fmt.Sprintf("http://%s:%d/health", host, port)
	body, err := getJSON(ctx, client, url)
	if err != nil {
		bh.Error = err.Error()
		return bh
	}
	bh.Reachable = true
	var hb healthBody
	if err := json.Unmarshal(body, &hb); err == nil {
		bh.Running = &hb.Running
		bh.CurrentModel = hb.CurrentModel
		bh.ActiveReqs = &hb.ActiveReqs
	}
	return bh
}

func getJSON(ctx context.Context, client *http.Client, url string) (json.RawMessage, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%s: HTTP %d", url, resp.StatusCode)
	}
	return io.ReadAll(resp.Body)
}

func env(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func envInt(key string, fallback int) int {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return fallback
	}
	return n
}
