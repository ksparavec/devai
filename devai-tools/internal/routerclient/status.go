// Package routerclient implements get_router_status: a best-effort probe
// of the running devai-router.
//
// This probes the single-host serving path only. It previously tried a
// cluster-head control-plane endpoint first and fell back to per-backend
// health checks; cluster mode was frozen on 2026-07-25 (see
// attic/README.md), so that first probe could only ever fail. It cost a
// token-file read plus a doomed HTTP round trip on every single call.
// If cluster mode is thawed, restore the probe from
// `git log -S probeCluster -- devai-tools/internal/routerclient/`.
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

// defaultRouterHost is the router's compose service/container name on
// devai-net. The router service publishes NO host ports (see
// deploy/docker-compose.yaml -- only apt-cacher, the registry mirror, the
// webui proxy and the MCP gateway do), so host.containers.internal never
// answers on 11434/5/6 and probing it degraded every call to
// "unreachable" by construction. Reaching the router by service name over
// devai-net is how every other in-network consumer (open-webui, the lab)
// already addresses it.
const defaultRouterHost = "devai-router"

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

// Status is get_router_status's result. Mode is "single" (at least one
// backend health check succeeded) or "unreachable" (nothing responded).
//
// The "cluster-head" mode and the cluster_error field were removed when
// cluster mode was frozen -- see the package comment.
type Status struct {
	Mode     string          `json:"mode"`
	Backends []BackendHealth `json:"backends,omitempty"`
	Error    string          `json:"error,omitempty"`
}

type healthBody struct {
	Running      bool   `json:"running"`
	CurrentModel string `json:"current_model"`
	ActiveReqs   int64  `json:"active_reqs"`
}

// GetStatus probes each backend's /health through the router. It never
// returns an error itself, only a Status describing what was reachable.
func GetStatus(ctx context.Context, client *http.Client) Status {
	host := env("DEVAI_ROUTER_HOST", defaultRouterHost)
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
			Error: fmt.Sprintf(
				"no backend answered /health on %s (ports %d/%d/%d) -- is `make cache-up` running?",
				host, defaultBackendPorts[0].Port, defaultBackendPorts[1].Port, defaultBackendPorts[2].Port),
		}
	}
	return Status{Mode: "single", Backends: backends}
}

// httpStatusError is a non-2xx response, carrying the code so callers can
// tell an auth rejection from a transport failure.
type httpStatusError struct {
	URL  string
	Code int
}

func (e *httpStatusError) Error() string {
	return fmt.Sprintf("%s: HTTP %d", e.URL, e.Code)
}

func probeBackendHealth(ctx context.Context, client *http.Client, host, backend string, port int) BackendHealth {
	bh := BackendHealth{Backend: backend, Port: port}
	url := fmt.Sprintf("http://%s:%d/health", host, port)
	// The backend /health handlers are unauthenticated -- gpu-arbiter's
	// makeHealthHandler is mounted bare.
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

// getJSON GETs url. A non-200 comes back as *httpStatusError so the
// caller's error string names the status code rather than swallowing it.
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
		return nil, &httpStatusError{URL: url, Code: resp.StatusCode}
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
