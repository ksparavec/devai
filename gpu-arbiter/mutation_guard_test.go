package main

import (
	"encoding/json"
	"log"
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

// /api/blobs/ is the one subtree pattern: a blob route carries the digest
// in its path, and both the upload (POST) and the existence check (HEAD)
// that precede `ollama create` must stop at the router.
func TestMutationGuard_BlobRoutesRefused(t *testing.T) {
	for _, method := range []string{http.MethodPost, http.MethodHead} {
		t.Run(method, func(t *testing.T) {
			var upstreamHits int64
			mux := guardedMux(t, "ollama", &upstreamHits)

			req := httptest.NewRequest(method, "/api/blobs/sha256:"+strings.Repeat("ab", 32), strings.NewReader("GGUF"))
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
	var upstreamHits int64
	mux := guardedMux(t, "ollama", &upstreamHits)
	if got := matchedPattern(t, mux, http.MethodPost, "/api/blobsx"); got != "/" {
		t.Errorf("/api/blobsx routed to pattern %q, want the catch-all", got)
	}
}

// ServeMux matches on the escaped path, so an encoded slash turns a guarded
// route into one segment that misses its pattern and lands on the
// catch-all. Ollama decodes the path and runs the handler (verified
// against devai-ollama 0.34.2 in review: `DELETE /api%2Fdelete` reached
// the delete handler, `POST /api/blobs%2Fsha256:...` wrote a blob). The
// catch-all must refuse these on the decoded path.
func TestMutationGuard_EncodedSlashesRefused(t *testing.T) {
	digest := "sha256:" + strings.Repeat("ab", 32)
	cases := []struct{ method, target string }{
		{http.MethodDelete, "/api%2Fdelete"},
		{http.MethodPost, "/api%2fpull"},
		{http.MethodPost, "/api%2Fcreate"},
		{http.MethodPost, "/api%2Fcopy"},
		{http.MethodPost, "/api%2Fpush"},
		{http.MethodPost, "/api/blobs%2F" + digest},
		{http.MethodHead, "/api%2Fblobs%2F" + digest},
		{http.MethodPost, "/api%2F%2Fdelete"},
		{http.MethodPost, "/api%2Fdelete%2F"},
		{http.MethodPost, "/api%2Fx%2F..%2Fpull"},
	}
	for _, backend := range []string{"ollama", "vllm"} {
		for _, c := range cases {
			t.Run(backend+" "+c.method+" "+c.target, func(t *testing.T) {
				var upstreamHits int64
				mux := guardedMux(t, backend, &upstreamHits)
				req := httptest.NewRequest(c.method, c.target, strings.NewReader(`{"model":"m","name":"m"}`))
				w := httptest.NewRecorder()
				mux.ServeHTTP(w, req)
				if w.Code != http.StatusForbidden {
					t.Errorf("status = %d, want 403", w.Code)
				}
				if got := atomic.LoadInt64(&upstreamHits); got != 0 {
					t.Errorf("upstream saw %d requests, want 0", got)
				}
			})
		}
	}
}

func TestIsMutationPath(t *testing.T) {
	for p, want := range map[string]bool{
		"/api/delete": true, "/api/delete/": true, "/api//pull": true, "/api/blobs": true,
		"/api/blobs/sha256:ab": true, "/api/blobs/x/y": true,
		"/api/chat": false, "/api/pullxyz": false, "/api/blobsx": false, "/api/show": false,
		"/v1/chat/completions": false, "/api/tags": false,
	} {
		if got := isMutationPath(p); got != want {
			t.Errorf("isMutationPath(%q) = %v, want %v", p, got, want)
		}
	}
}

// The refusal echoes the client's path; it must stay valid JSON and one
// log line whatever the path holds (an encoded newline included).
func TestMutationGuard_ClientPathCannotBreakBodyOrLog(t *testing.T) {
	var upstreamHits int64
	mux := guardedMux(t, "ollama", &upstreamHits)

	var buf strings.Builder
	prev := log.Writer()
	log.SetOutput(&buf)
	t.Cleanup(func() { log.SetOutput(prev) })

	req := httptest.NewRequest(http.MethodPost, `/api/blobs/x%0A2026/01/01%2000:00:00%20[vllm]%20fingerprint:%20{%22forged%22:1}%22`, nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusForbidden {
		t.Fatalf("status = %d, want 403", w.Code)
	}
	if !json.Valid(w.Body.Bytes()) {
		t.Errorf("403 body is not valid JSON: %q", w.Body.String())
	}
	if n := strings.Count(strings.TrimRight(buf.String(), "\n"), "\n"); n != 0 {
		t.Errorf("the refusal logged %d extra line(s): %q", n, buf.String())
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
