//go:build devai_frozen_cluster

package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func tokenStoreFor(t *testing.T, value string) *TokenStore {
	t.Helper()
	dir := t.TempDir()
	p := filepath.Join(dir, "tok")
	if err := os.WriteFile(p, []byte(value), 0o600); err != nil {
		t.Fatalf("write tok: %v", err)
	}
	return NewTokenStore(p, time.Hour)
}

func TestSkyPilotClient_NotConfiguredEverywhere(t *testing.T) {
	c := NewSkyPilotClient("", nil)
	if c.IsConfigured() {
		t.Errorf("empty BaseURL should NOT be configured")
	}
	if _, err := c.Launch(context.Background(), LaunchRequest{}); err == nil {
		t.Errorf("Launch on unconfigured client should error")
	}
	if _, err := c.Status(context.Background()); err == nil {
		t.Errorf("Status on unconfigured client should error")
	}
	if err := c.Down(context.Background(), "x"); err == nil {
		t.Errorf("Down on unconfigured client should error")
	}
}

func TestSkyPilotClient_LaunchValidation(t *testing.T) {
	c := NewSkyPilotClient("http://example.invalid", tokenStoreFor(t, "tok"))
	if _, err := c.Launch(context.Background(), LaunchRequest{}); err == nil {
		t.Errorf("Launch should reject empty cluster_name")
	}
	if _, err := c.Launch(context.Background(),
		LaunchRequest{ClusterName: "x"}); err == nil {
		t.Errorf("Launch should reject empty image")
	}
	if err := c.Down(context.Background(), ""); err == nil {
		t.Errorf("Down should reject empty cluster_name")
	}
}

func TestSkyPilotClient_LaunchOK(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/launch" {
			t.Errorf("unexpected path %s", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Errorf("unexpected method %s", r.Method)
		}
		if r.Header.Get("Authorization") != "Bearer the-token" {
			t.Errorf("missing/bad bearer header: %s", r.Header.Get("Authorization"))
		}
		_ = json.NewEncoder(w).Encode(LaunchResponse{
			RequestID:   "req-1",
			ClusterName: "my-cluster",
		})
	}))
	defer srv.Close()

	c := NewSkyPilotClient(srv.URL, tokenStoreFor(t, "the-token"))
	resp, err := c.Launch(context.Background(), LaunchRequest{
		ClusterName: "my-cluster",
		Cloud:       "runpod",
		GPUs:        "3090:1",
		Image:       "devai-worker-bootstrap",
	})
	if err != nil {
		t.Fatalf("Launch: %v", err)
	}
	if resp.RequestID != "req-1" {
		t.Errorf("request_id: %q", resp.RequestID)
	}
}

func TestSkyPilotClient_LaunchNon2xxErrors(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "quota exceeded", http.StatusForbidden)
	}))
	defer srv.Close()

	c := NewSkyPilotClient(srv.URL, tokenStoreFor(t, "the-token"))
	_, err := c.Launch(context.Background(), LaunchRequest{
		ClusterName: "x", Image: "devai-worker-bootstrap",
	})
	if err == nil {
		t.Fatalf("expected error on 403")
	}
}

func TestSkyPilotClient_StatusReturnsList(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode([]SkyClusterStatus{
			{ClusterName: "c1", Status: "READY", Cloud: "runpod"},
			{ClusterName: "c2", Status: "INIT", Cloud: "lambda"},
		})
	}))
	defer srv.Close()
	c := NewSkyPilotClient(srv.URL, tokenStoreFor(t, "the-token"))
	out, err := c.Status(context.Background())
	if err != nil {
		t.Fatalf("Status: %v", err)
	}
	if len(out) != 2 {
		t.Fatalf("got %d entries, want 2", len(out))
	}
	if out[0].ClusterName != "c1" {
		t.Errorf("first entry: %q", out[0].ClusterName)
	}
}

func TestSkyPilotClient_DownIdempotentOn200(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/down" {
			t.Errorf("unexpected path %s", r.URL.Path)
		}
		var req DownRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		if req.ClusterName == "" {
			t.Errorf("empty cluster_name reached server")
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()
	c := NewSkyPilotClient(srv.URL, tokenStoreFor(t, "the-token"))
	if err := c.Down(context.Background(), "c1"); err != nil {
		t.Fatalf("Down: %v", err)
	}
	// Repeat -- upstream is idempotent; should not error.
	if err := c.Down(context.Background(), "c1"); err != nil {
		t.Fatalf("Down repeat: %v", err)
	}
}

// --- B2: Down's idempotency claim has to be true ---

// An already-gone cluster is a successful teardown. Reporting it as an
// error made the teardown coordinator retry a cluster that no longer
// exists until it hit its attempt bound, then log an ERROR about a VM
// that is not billing anything.
func TestSkyPilotClient_DownTreatsAlreadyGoneAsSuccess(t *testing.T) {
	for _, status := range []int{
		http.StatusOK, http.StatusAccepted,
		http.StatusNotFound, http.StatusGone,
	} {
		srv := httptest.NewServer(http.HandlerFunc(
			func(w http.ResponseWriter, _ *http.Request) {
				http.Error(w, "cluster my-cluster not found", status)
			}))
		c := NewSkyPilotClient(srv.URL, tokenStoreFor(t, "tok"))
		if err := c.Down(context.Background(), "my-cluster"); err != nil {
			t.Errorf("HTTP %d: Down returned %v, want nil", status, err)
		}
		srv.Close()
	}
}

func TestSkyPilotClient_DownRealFailureStillErrors(t *testing.T) {
	for _, status := range []int{
		http.StatusForbidden, http.StatusInternalServerError,
		http.StatusBadGateway,
	} {
		srv := httptest.NewServer(http.HandlerFunc(
			func(w http.ResponseWriter, _ *http.Request) {
				http.Error(w, "boom", status)
			}))
		if err := NewSkyPilotClient(srv.URL, tokenStoreFor(t, "tok")).
			Down(context.Background(), "my-cluster"); err == nil {
			t.Errorf("HTTP %d: Down returned nil, want an error the "+
				"coordinator can retry", status)
		}
		srv.Close()
	}
}