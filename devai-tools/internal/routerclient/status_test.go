package routerclient

import (
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"testing"
	"time"
)

func testClient() *http.Client {
	return &http.Client{Timeout: 2 * time.Second}
}

func TestGetStatusSingleMode(t *testing.T) {
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

// Cluster mode is frozen (attic/README.md), so "cluster-head" must no
// longer be reachable as a Mode however the environment is configured.
// Regression guard against a partial thaw reintroducing the doomed probe.
func TestGetStatusNeverReportsClusterMode(t *testing.T) {
	t.Setenv("DEVAI_ROUTER_HOST", "127.0.0.1")
	t.Setenv("DEVAI_ROUTER_OLLAMA_PORT", strconv.Itoa(freeClosedTCPPort(t)))
	t.Setenv("DEVAI_ROUTER_VLLM_PORT", strconv.Itoa(freeClosedTCPPort(t)))
	t.Setenv("DEVAI_ROUTER_SGLANG_PORT", strconv.Itoa(freeClosedTCPPort(t)))
	// Variables the removed cluster probe used to honour.
	t.Setenv("DEVAI_ROUTER_CLUSTER_URL", "http://127.0.0.1:11444")
	t.Setenv("DEVAI_HEAD_TOKEN_FILE", "/nonexistent/token")

	if status := GetStatus(context.Background(), testClient()); status.Mode == "cluster-head" {
		t.Fatalf("Mode = %q, want cluster mode to be unreachable while frozen", status.Mode)
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

// The backend /health handlers are unauthenticated. Nothing should send
// them an Authorization header -- there is no credential in this path to
// leak, and adding one later would be a change worth catching here.
func TestGetStatusSendsNoAuthorizationToBackends(t *testing.T) {
	var backendAuth []string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		backendAuth = append(backendAuth, r.Header.Get("Authorization"))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"running":true,"current_model":"m","active_reqs":0}`))
	}))
	defer backend.Close()

	host, port := splitHostPort(t, backend.URL)
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
