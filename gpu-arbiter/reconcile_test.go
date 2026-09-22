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

// newOllamaReconcileArbiter builds an arbiter whose ollama backend answers
// /api/ps with `ps` (or refuses the connection when ps is nil).
func newOllamaReconcileArbiter(t *testing.T, ps *string) *arbiter {
	t.Helper()
	a := &arbiter{
		backends:     map[string]*backendState{},
		healthClient: &http.Client{},
	}
	srv := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/ps" && ps != nil {
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(*ps))
				return
			}
			w.WriteHeader(http.StatusOK) // /health: always up
		}))
	t.Cleanup(srv.Close)
	u, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	if ps == nil {
		srv.Close() // connection refused
	}
	a.backends["ollama"] = &backendState{config: backendConfig{
		Name: "ollama", BackendURL: u, HealthPath: "/health", ContainerName: "devai-ollama"}}
	return a
}

func TestReconcileAdoptsAResidentOllamaModel(t *testing.T) {
	// Observed 2026-09-22: qwen3.6:35b-a3b-mtp-q4_K_M resident across a
	// `make cache-up` that recreated the router; the next vllm-devai
	// launch died on 0.3 GiB free because stopOtherBackends never unloads
	// a backend whose flags say idle.
	ps := `{"models":[{"name":"qwen3.6:35b-a3b-mtp-q4_K_M","size":22000000000,"context_length":131072}]}`
	a := newOllamaReconcileArbiter(t, &ps)
	a.reconcileBackendState()
	bs := a.backends["ollama"]
	if !bs.running || !bs.containerLaunched {
		t.Fatal("a resident ollama model must be adopted, or the next HF launch OOMs")
	}
	if bs.currentModel != "qwen3.6:35b-a3b-mtp-q4_K_M" || bs.currentContext != 131072 {
		t.Fatalf("adopted model/ctx = %q/%d, want the /api/ps entry", bs.currentModel, bs.currentContext)
	}
}

func TestReconcileLeavesOllamaIdleWhenNothingIsLoaded(t *testing.T) {
	// The container is always up and always answers /health; only
	// /api/ps says whether the GPU is held.
	ps := `{"models":[]}`
	a := newOllamaReconcileArbiter(t, &ps)
	a.reconcileBackendState()
	if bs := a.backends["ollama"]; bs.running || bs.containerLaunched || bs.currentModel != "" {
		t.Fatal("ollama must not be adopted from a /health probe alone")
	}
}

func TestReconcileUnreachableOllamaIsNotAdopted(t *testing.T) {
	a := newOllamaReconcileArbiter(t, nil)
	a.reconcileBackendState()
	if bs := a.backends["ollama"]; bs.running || bs.containerLaunched {
		t.Fatal("an unreachable ollama must be left idle (logged), not adopted")
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
