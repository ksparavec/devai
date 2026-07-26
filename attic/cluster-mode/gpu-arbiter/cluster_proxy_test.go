//go:build devai_frozen_cluster

package main

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func proxyTokens(t *testing.T) *TokenStore {
	t.Helper()
	dir := t.TempDir()
	p := filepath.Join(dir, "tok")
	if err := os.WriteFile(p, []byte("the-token"), 0o600); err != nil {
		t.Fatalf("write token: %v", err)
	}
	return NewTokenStore(p, time.Hour)
}

func TestNewClusterProxy_BoundsResponseHeaderWait(t *testing.T) {
	p := NewClusterProxy(proxyTokens(t))
	if p.Client.Timeout != 0 {
		t.Errorf("overall Timeout = %v, want 0 -- streaming responses run "+
			"arbitrarily long", p.Client.Timeout)
	}
	tr, ok := p.Client.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("transport is %T, want *http.Transport", p.Client.Transport)
	}
	if tr.ResponseHeaderTimeout == 0 {
		t.Fatal("ResponseHeaderTimeout unset: a wedged worker would pin a head " +
			"goroutine and its connection forever")
	}
	// Must exceed the worker's worst-case cold start (HEALTH_TIMEOUT_SECONDS
	// defaults to 600s and is spent before the first response byte).
	if tr.ResponseHeaderTimeout <= 600*time.Second {
		t.Errorf("ResponseHeaderTimeout = %v, must exceed a 600s cold start",
			tr.ResponseHeaderTimeout)
	}
}

// B3: the old hardcoded 15m was justified by a bound the worker does
// not obey -- a request parked on recreateCond behind ANOTHER model's
// recreate waits one healthTimeout and then runs its own, which at the
// default 600s already exceeds 15m. And HEALTH_TIMEOUT_SECONDS is
// operator-tunable, so any value above ~450s broke the stated relation.
// The budget now derives from the same env the worker reads.
func TestClusterResponseHeaderBudget_DerivesFromWorkerBudget(t *testing.T) {
	tests := []struct {
		name          string
		health, drain string
		want          time.Duration
	}{
		{"defaults", "", "", 2*600*time.Second + 30*time.Second},
		{"tuned health", "1800", "", 2*1800*time.Second + 30*time.Second},
		{"tuned both", "900", "120", 2*900*time.Second + 120*time.Second},
		// envInt maps a non-positive value back to its default, so a
		// misconfigured env can never produce a zero/negative budget.
		{"garbage falls back", "0", "-5", 2*600*time.Second + 30*time.Second},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("HEALTH_TIMEOUT_SECONDS", tc.health)
			t.Setenv("DRAIN_TIMEOUT", tc.drain)
			if got := clusterResponseHeaderBudget(); got != tc.want {
				t.Errorf("budget = %v, want %v", got, tc.want)
			}
			// The invariant the old comment claimed but did not hold:
			// the head must outlast a request that waits out one
			// recreate and then performs its own.
			if got := clusterResponseHeaderBudget(); got <= 2*envHealth(t) {
				t.Errorf("budget %v does not cover two health timeouts", got)
			}
		})
	}
}

func envHealth(t *testing.T) time.Duration {
	t.Helper()
	return time.Duration(envInt("HEALTH_TIMEOUT_SECONDS", 600)) * time.Second
}

func TestNewClusterProxy_UsesDerivedBudget(t *testing.T) {
	t.Setenv("HEALTH_TIMEOUT_SECONDS", "1200")
	t.Setenv("DRAIN_TIMEOUT", "60")
	p := NewClusterProxy(proxyTokens(t))
	tr := p.Client.Transport.(*http.Transport)
	if want := 2*1200*time.Second + 60*time.Second; tr.ResponseHeaderTimeout != want {
		t.Errorf("ResponseHeaderTimeout = %v, want %v -- an operator who "+
			"raises HEALTH_TIMEOUT_SECONDS must not get 502s from the head",
			tr.ResponseHeaderTimeout, want)
	}
}

func TestClusterProxy_ForwardSetsBackendAndPathHeaders(t *testing.T) {
	seen := make(chan http.Header, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen <- r.Header.Clone()
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer srv.Close()

	p := NewClusterProxy(proxyTokens(t))
	worker := WorkerEntry{WorkerID: "wid-1", Endpoint: srv.URL}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		bytes.NewReader([]byte(`{"model":"m"}`)))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	p.Forward(rec, req, worker, MinimalRequest{Model: "m"}, "sglang")

	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	hdr := <-seen
	if got := hdr.Get(HeaderBackend); got != "sglang" {
		t.Errorf("%s = %q, want sglang -- the worker cannot infer the backend "+
			"from the body", HeaderBackend, got)
	}
	if got := hdr.Get(HeaderOriginalPath); got != "/v1/chat/completions" {
		t.Errorf("%s = %q, want /v1/chat/completions", HeaderOriginalPath, got)
	}
	if got := hdr.Get("Authorization"); got != "Bearer the-token" {
		t.Errorf("Authorization = %q", got)
	}
	if got := hdr.Get(HeaderWorkerID); got != "wid-1" {
		t.Errorf("%s = %q", HeaderWorkerID, got)
	}
}

func TestClusterProxy_ForwardCutsLooseWedgedWorker(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		<-release // never answers until the test lets go
	}))
	defer srv.Close()
	defer close(release)

	p := NewClusterProxy(proxyTokens(t))
	tr := p.Client.Transport.(*http.Transport)
	tr.ResponseHeaderTimeout = 150 * time.Millisecond

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		bytes.NewReader([]byte(`{"model":"m"}`)))
	rec := httptest.NewRecorder()

	done := make(chan struct{})
	go func() {
		defer close(done)
		p.Forward(rec, req, WorkerEntry{WorkerID: "wid-1", Endpoint: srv.URL},
			MinimalRequest{Model: "m"}, "vllm")
	}()

	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("Forward never returned: a worker that stops answering pins the head")
	}
	if rec.Code != http.StatusBadGateway {
		t.Errorf("status %d, want 502", rec.Code)
	}
}