package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

// The probe caches are the fit gate. For vLLM and SGLang the router
// enforces it directly via the modelNames allowlist; for Ollama it relies
// on upstream serving only locally pulled tags. That indirection only
// holds while the local tag set is controlled by the probe pipeline, so
// the router must refuse the endpoints that would change it.
//
// These tests drive the REAL route table (newBackendMux, the same
// function runSingleHost calls) rather than a hand-rolled copy, so a
// registration that regresses in production regresses here too.

func guardedMux(t *testing.T, backend string, upstreamHits *int64) *http.ServeMux {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt64(upstreamHits, 1)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"reached":"upstream"}`))
	}))
	t.Cleanup(server.Close)

	bs := testBackend(backend, server)
	bs.running = true
	return testArbiter(bs).newBackendMux(backend)
}

func TestMutationGuard_RefusesAndNeverReachesUpstream(t *testing.T) {
	for _, backend := range []string{"ollama", "vllm", "sglang"} {
		for _, path := range ollamaMutationPaths {
			t.Run(backend+path, func(t *testing.T) {
				var upstreamHits int64
				mux := guardedMux(t, backend, &upstreamHits)

				req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(`{"name":"unprobed:latest"}`))
				w := httptest.NewRecorder()
				mux.ServeHTTP(w, req)

				if w.Code != http.StatusForbidden {
					t.Errorf("status = %d, want 403", w.Code)
				}
				if got := atomic.LoadInt64(&upstreamHits); got != 0 {
					t.Errorf("upstream saw %d requests, want 0 -- the mutation was proxied through", got)
				}
			})
		}
	}
}

// The refusal has to tell the operator what to do instead, or it just
// looks like a broken router.
func TestMutationGuard_BodyNamesTheSanctionedPath(t *testing.T) {
	var upstreamHits int64
	mux := guardedMux(t, "ollama", &upstreamHits)

	req := httptest.NewRequest(http.MethodPost, "/api/pull", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	body := w.Body.String()
	for _, want := range []string{"/api/pull", "make model-pull", "probe"} {
		if !strings.Contains(body, want) {
			t.Errorf("body %q does not mention %q", body, want)
		}
	}
	if ct := w.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", ct)
	}
}

// GET is refused too. Ollama's delete takes DELETE and pull takes POST,
// but the guard is about the route, not the verb -- a client using the
// wrong method must not slip past into the catch-all proxy.
func TestMutationGuard_RefusesEveryMethod(t *testing.T) {
	for _, method := range []string{http.MethodGet, http.MethodPost, http.MethodDelete, http.MethodHead} {
		t.Run(method, func(t *testing.T) {
			var upstreamHits int64
			mux := guardedMux(t, "ollama", &upstreamHits)

			req := httptest.NewRequest(method, "/api/delete", nil)
			w := httptest.NewRecorder()
			mux.ServeHTTP(w, req)

			if w.Code != http.StatusForbidden {
				t.Errorf("%s status = %d, want 403", method, w.Code)
			}
			if got := atomic.LoadInt64(&upstreamHits); got != 0 {
				t.Errorf("%s reached upstream %d times, want 0", method, got)
			}
		})
	}
}

// matchedPattern reports which registered route a request resolves to,
// without invoking the handler. Used for the pass-through assertions:
// actually calling the catch-all would drive ensureBackendRunning into
// real container management, which is not what these tests are about.
func matchedPattern(t *testing.T, mux *http.ServeMux, method, path string) string {
	t.Helper()
	_, pattern := mux.Handler(httptest.NewRequest(method, path, nil))
	return pattern
}

// The guard must not become a general /api/* block: read and inference
// paths still have to reach their normal handlers. /api/tags has its own;
// /api/show, /api/chat and friends fall through to the catch-all.
func TestMutationGuard_LeavesNonMutatingPathsAlone(t *testing.T) {
	var upstreamHits int64
	mux := guardedMux(t, "ollama", &upstreamHits)

	cases := map[string]string{
		"/api/show":     "/",
		"/api/chat":     "/",
		"/api/generate": "/",
		"/api/embed":    "/",
		"/api/tags":     "/api/tags",
		"/v1/models":    "/v1/models",
		"/health":       "/health",
	}
	for path, want := range cases {
		if got := matchedPattern(t, mux, http.MethodPost, path); got != want {
			t.Errorf("%s routed to pattern %q, want %q", path, got, want)
		}
	}
}

// A near-miss path must not be caught by accident. Go's ServeMux treats a
// pattern without a trailing slash as an exact match, so /api/pullxyz and
// /api/pull/extra have to fall through to the catch-all.
func TestMutationGuard_ExactPathsOnly(t *testing.T) {
	var upstreamHits int64
	mux := guardedMux(t, "ollama", &upstreamHits)

	for _, path := range []string{"/api/pullxyz", "/api/pull/extra", "/api/deleted"} {
		if got := matchedPattern(t, mux, http.MethodPost, path); got != "/" {
			t.Errorf("%s routed to pattern %q, want the catch-all %q", path, got, "/")
		}
	}
}

// Complement to the above: every guarded path must resolve to its own
// exact pattern, not to the catch-all.
func TestMutationGuard_RegisteredOnEveryBackend(t *testing.T) {
	for _, backend := range []string{"ollama", "vllm", "sglang"} {
		var upstreamHits int64
		mux := guardedMux(t, backend, &upstreamHits)
		for _, path := range ollamaMutationPaths {
			if got := matchedPattern(t, mux, http.MethodPost, path); got != path {
				t.Errorf("%s: %s routed to pattern %q, want %q", backend, path, got, path)
			}
		}
	}
}
