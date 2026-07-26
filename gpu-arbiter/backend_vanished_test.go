package main

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// The router used to tear down healthy backends. Observed 2026-07-25:
// gpt-oss-20b on SGLang was relaunched 9 times in 18 minutes, each
// teardown almost exactly 2s after the router logged "sglang ready",
// while the engine was answering /health 200 and serving 10 concurrent
// requests at 461 tok/s. Every in-flight request died with "proxy error:
// context canceled". The same signature accounts for ~100 recreates of a
// different model over an hour, which was initially -- and wrongly --
// attributed to that model crashing.
//
// Three independent causes, one test class each below.

func vanishTestBackend(t *testing.T, healthHandler http.HandlerFunc) (*arbiter, *backendState) {
	t.Helper()
	srv := httptest.NewServer(healthHandler)
	t.Cleanup(srv.Close)
	u, _ := url.Parse(srv.URL)

	bs := &backendState{
		config: backendConfig{
			Name: "sglang", BackendURL: u, HealthPath: "/health",
			ContainerName: "devai-sglang",
		},
		running: true,
	}
	a := &arbiter{
		backends:     map[string]*backendState{"sglang": bs},
		healthClient: &http.Client{Timeout: 2 * time.Second},
	}
	bs.recreateCond = sync.NewCond(&a.mu)
	return a, bs
}

// containerState stub: the real one talks to podman. `ok=false` models
// "podman API unreachable", which is the case that used to be
// misread as "container gone".
func (a *arbiter) stubContainerState(status string, ok bool) {
	a.containerStateStub = func(string) (string, int, bool) {
		return status, 0, ok
	}
}

// Cause 3: requests actively proxied upstream are proof of life, and
// tearing the backend down while it is answering is the specific harm.
func TestBackendVanished_InFlightRequestsMeanAlive(t *testing.T) {
	// /health deliberately fails: the in-flight check must win before it.
	a, bs := vanishTestBackend(t, func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "unavailable", http.StatusServiceUnavailable)
	})
	a.stubContainerState("running", true)
	atomic.StoreInt64(&bs.upstreamReqs, 3)

	if vanished, why := a.backendVanished(bs); vanished {
		t.Fatalf("backend with 3 upstream requests reported vanished (%s)", why)
	}
}

// Cause 1: containerState returning ok=false means "podman did not
// answer", NOT "the container is gone". containerState's own doc comment
// promises a podman blip never triggers a spurious recreate; the old
// caller broke that promise by collapsing unknown into false.
func TestBackendVanished_PodmanUnreachableIsNotGone(t *testing.T) {
	a, bs := vanishTestBackend(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	a.stubContainerState("", false) // podman blip

	if vanished, why := a.backendVanished(bs); vanished {
		t.Fatalf("podman blip reported as vanished (%s) -- containerState "+
			"documents this as unknown, not gone", why)
	}
}

// Cause 2: one slow /health from a loaded engine is not death. SGLang
// answers on a ~1s scheduler tick against a 2s client timeout, so a
// chunked prefill is enough to miss a single probe.
func TestBackendVanished_TransientHealthFailureIsRetried(t *testing.T) {
	var calls int64
	a, bs := vanishTestBackend(t, func(w http.ResponseWriter, r *http.Request) {
		if atomic.AddInt64(&calls, 1) == 1 {
			http.Error(w, "busy", http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
	})
	a.stubContainerState("running", true)

	if vanished, why := a.backendVanished(bs); vanished {
		t.Fatalf("one transient /health failure condemned the backend (%s)", why)
	}
	if got := atomic.LoadInt64(&calls); got < 2 {
		t.Errorf("/health probed %d time(s), expected a retry", got)
	}
}

// The guard must still catch genuinely dead backends, or it is useless.
func TestBackendVanished_StillDetectsRealFailures(t *testing.T) {
	t.Run("container exited", func(t *testing.T) {
		a, bs := vanishTestBackend(t, func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
		})
		a.stubContainerState("exited", true)

		vanished, why := a.backendVanished(bs)
		if !vanished {
			t.Fatal("exited container not detected as vanished")
		}
		if why == "" {
			t.Error("reason must be reported so the log names the condition")
		}
	})

	t.Run("placeholder up, nothing listening", func(t *testing.T) {
		// The `sleep infinity` placeholder: container running, no engine.
		a, bs := vanishTestBackend(t, func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, "no", http.StatusServiceUnavailable)
		})
		a.stubContainerState("running", true)

		vanished, why := a.backendVanished(bs)
		if !vanished {
			t.Fatal("placeholder container not detected as vanished")
		}
		if why == "" {
			t.Error("reason must be reported")
		}
	})
}
