package routerclient

import (
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

func testClient() *http.Client {
	return &http.Client{Timeout: 2 * time.Second}
}

// closedPort starts and immediately closes a listener, returning a
// "host:port" pair that refuses connections fast -- used to simulate an
// unreachable service without a slow real-network timeout.
func closedPort(t *testing.T) string {
	t.Helper()
	srv := httptest.NewServer(http.NotFoundHandler())
	addr := srv.Listener.Addr().String()
	srv.Close()
	return addr
}

func TestGetStatusClusterModeSuccess(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/cluster/status" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`[{"worker_id":"w1","name":"worker-1","health_status":"ok"}]`))
	}))
	defer srv.Close()

	t.Setenv("DEVAI_ROUTER_CLUSTER_URL", srv.URL)
	status := GetStatus(context.Background(), testClient())

	if status.Mode != "cluster-head" {
		t.Fatalf("Mode = %q, want cluster-head", status.Mode)
	}
	if len(status.Workers) == 0 {
		t.Fatal("expected non-empty Workers payload")
	}
}

func TestGetStatusSingleModeFallback(t *testing.T) {
	// Cluster endpoint unreachable.
	t.Setenv("DEVAI_ROUTER_CLUSTER_URL", "http://"+closedPort(t))

	// One backend (vllm) healthy, the other two closed.
	vllm := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"ok","backend":"vllm","running":true,"current_model":"Qwen3-8B-NVFP4","active_reqs":2}`))
	}))
	defer vllm.Close()

	host, vllmPort := splitHostPort(t, vllm.URL)
	t.Setenv("DEVAI_ROUTER_HOST", host)
	t.Setenv("DEVAI_ROUTER_VLLM_PORT", vllmPort)
	t.Setenv("DEVAI_ROUTER_OLLAMA_PORT", strconv.Itoa(freeClosedTCPPort(t)))
	t.Setenv("DEVAI_ROUTER_SGLANG_PORT", strconv.Itoa(freeClosedTCPPort(t)))

	status := GetStatus(context.Background(), testClient())

	if status.Mode != "single" {
		t.Fatalf("Mode = %q, want single", status.Mode)
	}
	var vllmHealth *BackendHealth
	for i := range status.Backends {
		if status.Backends[i].Backend == "vllm" {
			vllmHealth = &status.Backends[i]
		}
	}
	if vllmHealth == nil || !vllmHealth.Reachable {
		t.Fatalf("expected vllm backend reachable, got %+v", status.Backends)
	}
	if vllmHealth.CurrentModel != "Qwen3-8B-NVFP4" {
		t.Errorf("CurrentModel = %q, want Qwen3-8B-NVFP4", vllmHealth.CurrentModel)
	}
	if vllmHealth.Running == nil || !*vllmHealth.Running {
		t.Errorf("Running = %v, want true", vllmHealth.Running)
	}
}

func TestGetStatusEverythingUnreachable(t *testing.T) {
	t.Setenv("DEVAI_ROUTER_CLUSTER_URL", "http://"+closedPort(t))
	t.Setenv("DEVAI_ROUTER_HOST", "127.0.0.1")
	t.Setenv("DEVAI_ROUTER_OLLAMA_PORT", strconv.Itoa(freeClosedTCPPort(t)))
	t.Setenv("DEVAI_ROUTER_VLLM_PORT", strconv.Itoa(freeClosedTCPPort(t)))
	t.Setenv("DEVAI_ROUTER_SGLANG_PORT", strconv.Itoa(freeClosedTCPPort(t)))

	status := GetStatus(context.Background(), testClient())

	if status.Mode != "unreachable" {
		t.Fatalf("Mode = %q, want unreachable", status.Mode)
	}
	if status.Error == "" {
		t.Error("expected a non-empty Error message")
	}
	for _, b := range status.Backends {
		if b.Reachable {
			t.Errorf("backend %s reported reachable, want none reachable", b.Backend)
		}
	}
}

// The router service publishes no host ports -- it is reachable only by
// service name on devai-net -- so the built-in defaults must address
// devai-router, not host.containers.internal. Asserted through the real
// URLs GetStatus builds (surfaced in each backend's error string) rather
// than by re-stating the constant.
func TestGetStatusDefaultsToComposeServiceName(t *testing.T) {
	// Empty value == unset for env(), so this restores the shipped defaults
	// even when the ambient environment has them set.
	t.Setenv("DEVAI_ROUTER_CLUSTER_URL", "")
	t.Setenv("DEVAI_ROUTER_HOST", "")
	t.Setenv("DEVAI_ROUTER_OLLAMA_PORT", "")
	t.Setenv("DEVAI_ROUTER_VLLM_PORT", "")
	t.Setenv("DEVAI_ROUTER_SGLANG_PORT", "")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	status := GetStatus(ctx, testClient())

	for _, b := range status.Backends {
		if b.Reachable {
			t.Skipf("a real devai-router answered on %s:%d -- nothing to assert", defaultRouterHost, b.Port)
		}
		if !strings.Contains(b.Error, defaultRouterHost) {
			t.Errorf("backend %s probed %q, want a URL against %q", b.Backend, b.Error, defaultRouterHost)
		}
	}
	wantPorts := []int{11434, 11435, 11436}
	if len(status.Backends) != len(wantPorts) {
		t.Fatalf("got %d backends, want %d", len(status.Backends), len(wantPorts))
	}
	for i, want := range wantPorts {
		if status.Backends[i].Port != want {
			t.Errorf("backend %s port = %d, want %d", status.Backends[i].Backend, status.Backends[i].Port, want)
		}
	}
}

// writeTokenFile drops a bearer token in a temp dir and returns its path.
func writeTokenFile(t *testing.T, token string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "cluster-token")
	if err := os.WriteFile(path, []byte(token+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// authHead is a stub cluster head that mirrors the real one: every
// /v1/cluster/* route sits behind TokenStore.AuthMiddleware, so an
// unauthenticated probe gets 401. seen captures the Authorization header
// each request arrived with.
func authHead(t *testing.T, wantToken string, seen *[]string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		*seen = append(*seen, r.Header.Get("Authorization"))
		if r.URL.Path != "/v1/cluster/status" {
			http.NotFound(w, r)
			return
		}
		if r.Header.Get("Authorization") != "Bearer "+wantToken {
			w.Header().Set("WWW-Authenticate", `Bearer realm="devai-cluster"`)
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`[{"worker_id":"w1","name":"worker-1","health_status":"ok"}]`))
	}))
	t.Cleanup(srv.Close)
	return srv
}

// The head wraps /v1/cluster/status in AuthMiddleware. Without a bearer
// token the probe gets 401 and -- since compose.head.yaml zeroes the local
// backends on a head -- the tool degrades all the way to "unreachable",
// making the cluster-head branch dead code. Regression for that.
func TestGetStatusClusterModeSendsBearerToken(t *testing.T) {
	var seen []string
	srv := authHead(t, "s3cr3t", &seen)

	t.Setenv("DEVAI_ROUTER_CLUSTER_URL", srv.URL)
	t.Setenv("DEVAI_HEAD_TOKEN_FILE", writeTokenFile(t, "s3cr3t"))

	status := GetStatus(context.Background(), testClient())

	if status.Mode != "cluster-head" {
		t.Fatalf("Mode = %q (cluster_error %q), want cluster-head", status.Mode, status.ClusterError)
	}
	if len(status.Workers) == 0 {
		t.Error("expected non-empty Workers payload")
	}
	if status.ClusterError != "" {
		t.Errorf("ClusterError = %q, want empty on success", status.ClusterError)
	}
	if len(seen) != 1 || seen[0] != "Bearer s3cr3t" {
		t.Errorf("Authorization headers seen = %q, want exactly [Bearer s3cr3t]", seen)
	}
}

// DEVAI_HEAD_TOKEN_FILE wins over DEVAI_WORKER_TOKEN_FILE so a host that is
// both head and worker can point the two at different files.
func TestGetStatusPrefersHeadTokenFile(t *testing.T) {
	var seen []string
	srv := authHead(t, "head-token", &seen)

	t.Setenv("DEVAI_ROUTER_CLUSTER_URL", srv.URL)
	t.Setenv("DEVAI_WORKER_TOKEN_FILE", writeTokenFile(t, "worker-token"))
	t.Setenv("DEVAI_HEAD_TOKEN_FILE", writeTokenFile(t, "head-token"))

	if status := GetStatus(context.Background(), testClient()); status.Mode != "cluster-head" {
		t.Fatalf("Mode = %q (cluster_error %q), want cluster-head", status.Mode, status.ClusterError)
	}
}

// DEVAI_WORKER_TOKEN_FILE is the variable docs/cluster-env.md already
// documents, so it must work on its own.
func TestGetStatusFallsBackToWorkerTokenFile(t *testing.T) {
	var seen []string
	srv := authHead(t, "worker-token", &seen)

	t.Setenv("DEVAI_ROUTER_CLUSTER_URL", srv.URL)
	t.Setenv("DEVAI_HEAD_TOKEN_FILE", "")
	t.Setenv("DEVAI_WORKER_TOKEN_FILE", writeTokenFile(t, "worker-token"))

	if status := GetStatus(context.Background(), testClient()); status.Mode != "cluster-head" {
		t.Fatalf("Mode = %q (cluster_error %q), want cluster-head", status.Mode, status.ClusterError)
	}
}

// A missing token file must degrade, not crash -- and the resulting 401
// must be reported distinctly from "nothing answered" so a misconfigured
// token is diagnosable rather than looking like a non-cluster host.
func TestGetStatusMissingTokenFileReports401Distinctly(t *testing.T) {
	var seen []string
	srv := authHead(t, "s3cr3t", &seen)

	t.Setenv("DEVAI_ROUTER_CLUSTER_URL", srv.URL)
	t.Setenv("DEVAI_HEAD_TOKEN_FILE", filepath.Join(t.TempDir(), "absent-token"))
	t.Setenv("DEVAI_ROUTER_HOST", "127.0.0.1")
	t.Setenv("DEVAI_ROUTER_OLLAMA_PORT", strconv.Itoa(freeClosedTCPPort(t)))
	t.Setenv("DEVAI_ROUTER_VLLM_PORT", strconv.Itoa(freeClosedTCPPort(t)))
	t.Setenv("DEVAI_ROUTER_SGLANG_PORT", strconv.Itoa(freeClosedTCPPort(t)))

	status := GetStatus(context.Background(), testClient())

	if status.Mode != "unreachable" {
		t.Fatalf("Mode = %q, want unreachable (head answered 401, no local backends)", status.Mode)
	}
	if !strings.Contains(status.ClusterError, "401") {
		t.Errorf("ClusterError = %q, want it to name the 401", status.ClusterError)
	}
	if !strings.Contains(status.ClusterError, "DEVAI_HEAD_TOKEN_FILE") {
		t.Errorf("ClusterError = %q, want it to name the env var to set", status.ClusterError)
	}
	if len(seen) != 1 || seen[0] != "" {
		t.Errorf("Authorization headers seen = %q, want one unauthenticated request", seen)
	}
}

// A present-but-wrong token is a different operator problem from a missing
// one: the message must point at the file that supplied the bad value.
func TestGetStatusWrongTokenNamesTheTokenFile(t *testing.T) {
	var seen []string
	srv := authHead(t, "s3cr3t", &seen)
	tokenPath := writeTokenFile(t, "stale-token")

	t.Setenv("DEVAI_ROUTER_CLUSTER_URL", srv.URL)
	t.Setenv("DEVAI_HEAD_TOKEN_FILE", tokenPath)
	t.Setenv("DEVAI_ROUTER_HOST", "127.0.0.1")
	t.Setenv("DEVAI_ROUTER_OLLAMA_PORT", strconv.Itoa(freeClosedTCPPort(t)))
	t.Setenv("DEVAI_ROUTER_VLLM_PORT", strconv.Itoa(freeClosedTCPPort(t)))
	t.Setenv("DEVAI_ROUTER_SGLANG_PORT", strconv.Itoa(freeClosedTCPPort(t)))

	status := GetStatus(context.Background(), testClient())

	if !strings.Contains(status.ClusterError, tokenPath) {
		t.Errorf("ClusterError = %q, want it to name the token file %q", status.ClusterError, tokenPath)
	}
	if len(seen) != 1 || seen[0] != "Bearer stale-token" {
		t.Errorf("Authorization headers seen = %q, want the stale token to have been sent", seen)
	}
}

// The backend /health handlers are unauthenticated and live on different
// ports; the cluster token must not leak to them.
func TestGetStatusDoesNotSendTokenToBackends(t *testing.T) {
	var backendAuth []string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		backendAuth = append(backendAuth, r.Header.Get("Authorization"))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"running":true,"current_model":"m","active_reqs":0}`))
	}))
	defer backend.Close()

	host, port := splitHostPort(t, backend.URL)
	t.Setenv("DEVAI_ROUTER_CLUSTER_URL", "http://"+closedPort(t))
	t.Setenv("DEVAI_HEAD_TOKEN_FILE", writeTokenFile(t, "s3cr3t"))
	t.Setenv("DEVAI_ROUTER_HOST", host)
	t.Setenv("DEVAI_ROUTER_OLLAMA_PORT", port)
	t.Setenv("DEVAI_ROUTER_VLLM_PORT", port)
	t.Setenv("DEVAI_ROUTER_SGLANG_PORT", port)

	if status := GetStatus(context.Background(), testClient()); status.Mode != "single" {
		t.Fatalf("Mode = %q, want single", status.Mode)
	}
	if len(backendAuth) != 3 {
		t.Fatalf("backend saw %d requests, want 3", len(backendAuth))
	}
	for i, got := range backendAuth {
		if got != "" {
			t.Errorf("backend request %d carried Authorization %q, want none", i, got)
		}
	}
}

func splitHostPort(t *testing.T, rawURL string) (host, port string) {
	t.Helper()
	u, err := url.Parse(rawURL)
	if err != nil {
		t.Fatal(err)
	}
	host = u.Hostname()
	port = u.Port()
	return host, port
}

// freeClosedTCPPort returns a port number that was briefly listened on and
// then closed, so a connection attempt fails fast (refused) rather than
// hanging on an unroutable address.
func freeClosedTCPPort(t *testing.T) int {
	t.Helper()
	srv := httptest.NewServer(http.NotFoundHandler())
	_, portStr, err := net.SplitHostPort(srv.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	srv.Close()
	port, err := strconv.Atoi(portStr)
	if err != nil {
		t.Fatal(err)
	}
	return port
}
