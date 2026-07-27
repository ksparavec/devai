package main

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
)

// GPU exclusion runs entirely off in-memory state, which a restarted
// process does not have. Without adoption, a router restarted while a
// backend was serving believes the GPU is free and launches the next
// engine into an already-committed card -- observed live: devai-sglang
// held 22.3 of 24.5 GB, vLLM launched anyway and died with "Engine core
// initialization failed", a message that points at neither the real
// cause nor the stale sibling.
func newReconcileArbiter(t *testing.T, serving map[string]bool) *arbiter {
	t.Helper()
	a := &arbiter{
		backends:     map[string]*backendState{},
		healthClient: &http.Client{},
	}
	for name, up := range serving {
		srv := httptest.NewServer(http.HandlerFunc(
			func(w http.ResponseWriter, r *http.Request) {
				if !up {
					// A `sleep infinity` placeholder answers nothing.
					http.Error(w, "no engine", http.StatusServiceUnavailable)
					return
				}
				w.WriteHeader(http.StatusOK)
			}))
		t.Cleanup(srv.Close)
		u, err := url.Parse(srv.URL)
		if err != nil {
			t.Fatal(err)
		}
		a.backends[name] = &backendState{
			config: backendConfig{
				Name:          name,
				BackendURL:    u,
				HealthPath:    "/health",
				ContainerName: "devai-" + name,
			},
		}
	}
	return a
}

func TestReconcileAdoptsAServingBackend(t *testing.T) {
	a := newReconcileArbiter(t, map[string]bool{"sglang": true})
	bs := a.backends["sglang"]
	if bs.running || bs.containerLaunched {
		t.Fatal("fresh state should be false before reconciliation")
	}

	a.reconcileBackendState()

	if !bs.running || !bs.containerLaunched {
		t.Fatal("a serving backend must be adopted, or stopOtherBackends " +
			"will skip it and the next launch OOMs")
	}
}

func TestReconcileIgnoresAPlaceholder(t *testing.T) {
	// devai-vllm/devai-sglang exist as `sleep infinity` placeholders
	// whenever compose has run. Adopting one would make every first
	// request pay a pointless stop.
	a := newReconcileArbiter(t, map[string]bool{"vllm": false})
	a.reconcileBackendState()
	if bs := a.backends["vllm"]; bs.running || bs.containerLaunched {
		t.Fatal("a non-serving placeholder must not be adopted")
	}
}

func TestReconcileSkipsOllama(t *testing.T) {
	// Ollama's container is always up and always answers /health, so a
	// probe says nothing about whether a model is resident. `running`
	// for Ollama means "a model is loaded", which unloadOllama derives
	// from /api/ps at switch time.
	a := newReconcileArbiter(t, map[string]bool{"ollama": true})
	a.reconcileBackendState()
	if bs := a.backends["ollama"]; bs.running || bs.containerLaunched {
		t.Fatal("ollama must not be adopted from a /health probe")
	}
}

func TestReconcileLeavesModelUnknown(t *testing.T) {
	// Knowing a backend is live is not knowing what it loaded. Guessing
	// could serve a request from the wrong weights; an extra recreate is
	// the cheaper error.
	a := newReconcileArbiter(t, map[string]bool{"sglang": true})
	bs := a.backends["sglang"]
	bs.currentModel = "stale-from-a-previous-life"
	bs.currentContext = 131072

	a.reconcileBackendState()

	if bs.currentModel != "" || bs.currentContext != 0 {
		t.Fatalf("model/ctx must be cleared, got %q/%d",
			bs.currentModel, bs.currentContext)
	}
}

func TestReconcileAdoptsEachBackendIndependently(t *testing.T) {
	a := newReconcileArbiter(t, map[string]bool{"sglang": true, "vllm": false})
	a.reconcileBackendState()
	if !a.backends["sglang"].running {
		t.Fatal("serving sglang not adopted")
	}
	if a.backends["vllm"].running {
		t.Fatal("placeholder vllm wrongly adopted")
	}
}
