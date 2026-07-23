// SkyPilot API client for gpu-arbiter head mode.
//
// Per docs/plans/skypilot-fleet-provisioner.md Phase 2: minimal Go
// client wrapping the endpoints gpu-arbiter actually uses
// (POST /launch, GET /status, POST /down). Bearer-token auth from
// the sops-rendered tmpfs file.

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// SkyPilotClient is a small HTTP wrapper. Tests inject a fake
// HTTPDoer so the policy + teardown loops can run end-to-end without
// hitting a real cloud.
type SkyPilotClient struct {
	BaseURL string
	Token   *TokenStore
	Client  HTTPDoer
}

// NewSkyPilotClient returns a client pointing at SKYPILOT_API_ENDPOINT
// (or "" if unset -- callers MUST check IsConfigured before use).
func NewSkyPilotClient(baseURL string, tokens *TokenStore) *SkyPilotClient {
	return &SkyPilotClient{
		BaseURL: strings.TrimRight(baseURL, "/"),
		Token:   tokens,
		Client:  &http.Client{Timeout: 30 * time.Second},
	}
}

// IsConfigured reports whether the client has a base URL set. The
// fleet provisioner is opt-in -- a head with no SKYPILOT_API_ENDPOINT
// degrades to local-fleet-only routing per plan step 5.
func (c *SkyPilotClient) IsConfigured() bool {
	return c.BaseURL != ""
}

// LaunchRequest is the body POSTed to /api/v1/launch. We model only
// the fields gpu-arbiter actually sets; the upstream surface is
// larger but we keep the wire shape minimal.
type LaunchRequest struct {
	ClusterName string            `json:"cluster_name"`
	Cloud       string            `json:"cloud"`
	GPUs        string            `json:"gpus"` // e.g. "3090:1"
	Region      string            `json:"region,omitempty"`
	UseSpot     bool              `json:"use_spot,omitempty"`
	Image       string            `json:"image"` // worker-bootstrap image
	Env         map[string]string `json:"env,omitempty"`
	Run         string            `json:"run,omitempty"`
}

// LaunchResponse is what the API server returns. RequestID is the
// SkyPilot job handle; gpu-arbiter polls /status until the cluster
// reaches READY and the worker registers.
type LaunchResponse struct {
	RequestID   string `json:"request_id"`
	ClusterName string `json:"cluster_name"`
}

// StatusEntry mirrors one entry in the /api/v1/status response.
type SkyClusterStatus struct {
	ClusterName string `json:"cluster_name"`
	Status      string `json:"status"` // INIT/PENDING/READY/STOPPED/...
	Cloud       string `json:"cloud"`
	GPUs        string `json:"gpus"`
	IPAddress   string `json:"ip_address,omitempty"`
}

// DownRequest stops a cluster.
type DownRequest struct {
	ClusterName string `json:"cluster_name"`
}

// Launch provisions a SkyPilot cluster. Blocks for the upstream
// 10s connect/header timeout (the API server returns immediately;
// the cluster comes up async).
func (c *SkyPilotClient) Launch(ctx context.Context, req LaunchRequest) (*LaunchResponse, error) {
	if !c.IsConfigured() {
		return nil, errors.New("SkyPilotClient not configured (SKYPILOT_API_ENDPOINT empty)")
	}
	if req.ClusterName == "" {
		return nil, errors.New("cluster_name is required")
	}
	if req.Image == "" {
		return nil, errors.New("image is required")
	}
	body, _ := json.Marshal(req)
	resp, err := c.do(ctx, http.MethodPost, "/api/v1/launch", body)
	if err != nil {
		return nil, fmt.Errorf("launch %s: %w", req.ClusterName, err)
	}
	defer resp.Body.Close()
	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusAccepted {
		return nil, fmt.Errorf("launch %s: HTTP %d: %s",
			req.ClusterName, resp.StatusCode, string(respBody))
	}
	var lr LaunchResponse
	if err := json.Unmarshal(respBody, &lr); err != nil {
		return nil, fmt.Errorf("decode launch response: %w", err)
	}
	return &lr, nil
}

// Status returns the current cluster registry. Empty list = no
// clusters provisioned (fresh head install).
func (c *SkyPilotClient) Status(ctx context.Context) ([]SkyClusterStatus, error) {
	if !c.IsConfigured() {
		return nil, errors.New("SkyPilotClient not configured")
	}
	resp, err := c.do(ctx, http.MethodGet, "/api/v1/status", nil)
	if err != nil {
		return nil, fmt.Errorf("status: %w", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("status: HTTP %d: %s", resp.StatusCode, string(body))
	}
	var out []SkyClusterStatus
	if err := json.Unmarshal(body, &out); err != nil {
		return nil, fmt.Errorf("decode status response: %w", err)
	}
	return out, nil
}

// Down tears down a cluster. Idempotent: "the cluster is already gone"
// is success, not an error. Upstream signals that either with 200 (the
// record still exists but is already down) or with 404/410 (the record
// itself is gone) depending on how far the previous teardown got, so
// both are reported as success. Anything else is a real failure the
// teardown coordinator retries.
func (c *SkyPilotClient) Down(ctx context.Context, clusterName string) error {
	if !c.IsConfigured() {
		return errors.New("SkyPilotClient not configured")
	}
	if clusterName == "" {
		return errors.New("cluster_name is required")
	}
	body, _ := json.Marshal(DownRequest{ClusterName: clusterName})
	resp, err := c.do(ctx, http.MethodPost, "/api/v1/down", body)
	if err != nil {
		return fmt.Errorf("down %s: %w", clusterName, err)
	}
	defer resp.Body.Close()
	respBody, _ := io.ReadAll(resp.Body)
	switch resp.StatusCode {
	case http.StatusOK, http.StatusAccepted:
		return nil
	case http.StatusNotFound, http.StatusGone:
		// Already gone. Returning an error here would make the
		// coordinator retry a cluster that no longer exists until it
		// hit its attempt bound, and log an ERROR about a VM that is
		// not billing anything.
		return nil
	}
	return fmt.Errorf("down %s: HTTP %d: %s",
		clusterName, resp.StatusCode, string(respBody))
}

// do is the shared HTTP plumbing. Reads the bearer token on every
// call so a rotated token becomes effective immediately (TokenStore's
// own cache controls how often the file is actually re-read).
func (c *SkyPilotClient) do(
	ctx context.Context, method, path string, body []byte,
) (*http.Response, error) {
	tok := ""
	if c.Token != nil {
		t, err := c.Token.Read()
		if err != nil {
			return nil, fmt.Errorf("read token: %w", err)
		}
		tok = t
	}
	var bodyReader io.Reader
	if body != nil {
		bodyReader = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.BaseURL+path, bodyReader)
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	req.Header.Set("Accept", "application/json")
	if tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}
	return c.Client.Do(req)
}
