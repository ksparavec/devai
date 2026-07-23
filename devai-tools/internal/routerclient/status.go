// Package routerclient implements get_router_status: a best-effort probe
// of the running devai-router, tolerant of both cluster-head and
// single-host deployments.
package routerclient

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
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
// webui proxy, the MCP gateway and the SkyPilot API server do), so
// host.containers.internal never answers on 11434/5/6 and probing it
// degraded every call to "unreachable" by construction. Reaching the
// router by service name over devai-net is how every other in-network
// consumer (open-webui, the lab) already addresses it.
const defaultRouterHost = "devai-router"

// defaultClusterTokenPath is where the sops/age scaffold renders the
// cluster bearer token (docs/secrets.md). The head wraps
// /v1/cluster/{register,heartbeat,status} in TokenStore.AuthMiddleware, so
// an unauthenticated status probe gets 401 -- and, because
// deploy/compose.head.yaml zeroes the local backends on a head, the
// fallback then reports "unreachable" and the cluster-head branch never
// fires. Same file the arbiter's own DEVAI_WORKER_TOKEN_FILE defaults to.
const defaultClusterTokenPath = "/run/devai/cluster-token"

// clusterTokenEnvVars are consulted in order for the token's location.
// DEVAI_HEAD_TOKEN_FILE first so a host that is both head and worker can
// point the two at different files; DEVAI_WORKER_TOKEN_FILE second because
// that is the variable already documented in docs/cluster-env.md.
var clusterTokenEnvVars = []string{"DEVAI_HEAD_TOKEN_FILE", "DEVAI_WORKER_TOKEN_FILE"}

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
//
// ClusterError carries why the cluster probe did not answer, whenever it
// did not. It exists so a misconfigured bearer token is diagnosable: a 401
// (token missing, unreadable, or wrong) is a very different operator
// problem from "this host is not a cluster head", yet both otherwise
// collapse into the same single/unreachable mode.
type Status struct {
	Mode         string          `json:"mode"`
	Workers      json.RawMessage `json:"workers,omitempty"`
	Backends     []BackendHealth `json:"backends,omitempty"`
	ClusterError string          `json:"cluster_error,omitempty"`
	Error        string          `json:"error,omitempty"`
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
	clusterURL := env("DEVAI_ROUTER_CLUSTER_URL", "http://"+defaultRouterHost+":11444")
	body, clusterErr := probeCluster(ctx, client, clusterURL)
	if clusterErr == "" {
		return Status{Mode: "cluster-head", Workers: body}
	}

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
			Mode:         "unreachable",
			Backends:     backends,
			ClusterError: clusterErr,
			Error:        "cluster status endpoint and all backend health checks failed: " + clusterErr,
		}
	}
	return Status{Mode: "single", Backends: backends, ClusterError: clusterErr}
}

// probeCluster GETs the head's /v1/cluster/status with a bearer token.
// Returns the body on success, or a human-readable reason string on
// failure (empty reason == success). A missing or unreadable token file is
// NOT fatal: the request still goes out unauthenticated, so a head running
// without auth (or a plain single-mode host, where the probe fails at the
// connection anyway) keeps working -- but the reason string then names the
// token problem so a 401 is traceable to it.
func probeCluster(ctx context.Context, client *http.Client, clusterURL string) (json.RawMessage, string) {
	token, tokenPath, tokenErr := readClusterToken()
	body, err := getJSON(ctx, client, clusterURL+"/v1/cluster/status", token)
	if err == nil {
		return body, ""
	}
	var httpErr *httpStatusError
	if errors.As(err, &httpErr) && (httpErr.Code == http.StatusUnauthorized || httpErr.Code == http.StatusForbidden) {
		if tokenErr != nil {
			return nil, fmt.Sprintf(
				"cluster status probe rejected (HTTP %d) and no bearer token was sent: %v "+
					"(set DEVAI_HEAD_TOKEN_FILE or DEVAI_WORKER_TOKEN_FILE, default %s)",
				httpErr.Code, tokenErr, defaultClusterTokenPath)
		}
		return nil, fmt.Sprintf(
			"cluster status probe rejected (HTTP %d) with the bearer token from %s -- "+
				"token is stale or does not match the head's; re-render it (make secrets-render)",
			httpErr.Code, tokenPath)
	}
	return nil, err.Error()
}

// readClusterToken loads the cluster bearer token, returning the token,
// the path it came from, and any read error. Every return path is
// non-fatal for the caller; an error simply means "no token to send".
func readClusterToken() (token, path string, err error) {
	path = defaultClusterTokenPath
	for _, key := range clusterTokenEnvVars {
		if v := os.Getenv(key); v != "" {
			path = v
			break
		}
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return "", path, err
	}
	token = strings.TrimSpace(string(data))
	if token == "" {
		return "", path, fmt.Errorf("token at %s is empty", path)
	}
	return token, path, nil
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
	// No bearer token here: the backend /health handlers are unauthenticated
	// (gpu-arbiter's makeHealthHandler is mounted bare), and the cluster
	// token has no meaning on those ports.
	body, err := getJSON(ctx, client, url, "")
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

// getJSON GETs url, sending "Authorization: Bearer <token>" when token is
// non-empty. A non-200 comes back as *httpStatusError so callers can
// branch on the code.
func getJSON(ctx context.Context, client *http.Client, url, token string) (json.RawMessage, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
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
