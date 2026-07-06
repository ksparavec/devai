package routerclient

import (
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
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
