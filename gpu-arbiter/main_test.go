package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func testBackend(name string, server *httptest.Server) *backendState {
	u, _ := url.Parse(server.URL)
	return &backendState{
		config: backendConfig{
			Name:       name,
			BackendURL: u,
			HealthPath: "/health",
		},
		proxy: newProxy(u),
	}
}

func testArbiter(backends ...*backendState) *arbiter {
	a := &arbiter{
		backends:     make(map[string]*backendState),
		idleTimeout:  5 * time.Minute,
		drainTimeout: 5 * time.Second,
	}
	for _, bs := range backends {
		a.backends[bs.config.Name] = bs
	}
	return a
}

// --- TestMakeRequestHandler ---

func TestMakeRequestHandler_ProxiesToBackend(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]string{"backend": "ollama"})
	}))
	defer server.Close()

	bs := testBackend("ollama", server)
	bs.running = true
	a := testArbiter(bs)

	handler := a.makeRequestHandler("ollama")
	req := httptest.NewRequest("GET", "/v1/models", nil)
	w := httptest.NewRecorder()
	handler(w, req)

	var resp map[string]string
	json.NewDecoder(w.Body).Decode(&resp)
	if resp["backend"] != "ollama" {
		t.Errorf("expected backend=ollama, got %v", resp["backend"])
	}
}

func TestMakeRequestHandler_TracksActiveRequests(t *testing.T) {
	done := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-done
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	bs := testBackend("ollama", server)
	bs.running = true
	a := testArbiter(bs)

	handler := a.makeRequestHandler("ollama")

	go func() {
		req := httptest.NewRequest("GET", "/", nil)
		w := httptest.NewRecorder()
		handler(w, req)
	}()

	time.Sleep(50 * time.Millisecond)
	active := atomic.LoadInt64(&bs.activeReqs)
	if active != 1 {
		t.Errorf("expected 1 active request, got %d", active)
	}

	close(done)
	time.Sleep(50 * time.Millisecond)
	active = atomic.LoadInt64(&bs.activeReqs)
	if active != 0 {
		t.Errorf("expected 0 active requests after completion, got %d", active)
	}
}

func TestMakeRequestHandler_ExtractsModelForNonOllama(t *testing.T) {
	var receivedPath string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedPath = r.URL.Path
		json.NewEncoder(w).Encode(map[string]string{"ok": "true"})
	}))
	defer server.Close()

	bs := testBackend("vllm", server)
	bs.running = true
	bs.currentModel = "test-model"
	a := testArbiter(bs)

	handler := a.makeRequestHandler("vllm")
	body := `{"model":"test-model","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	handler(w, req)

	if receivedPath != "/v1/chat/completions" {
		t.Errorf("expected proxy to /v1/chat/completions, got %s", receivedPath)
	}
}

// --- TestMakeModelsHandler ---

func TestMakeModelsHandler_ReturnsConfiguredModels(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer server.Close()

	bs := testBackend("vllm", server)
	bs.modelNames = []string{"model-a", "model-b"}
	a := testArbiter(bs)

	handler := a.makeModelsHandler("vllm")
	req := httptest.NewRequest("GET", "/v1/models", nil)
	w := httptest.NewRecorder()
	handler(w, req)

	var resp struct {
		Data []struct {
			ID      string `json:"id"`
			OwnedBy string `json:"owned_by"`
		} `json:"data"`
	}
	json.NewDecoder(w.Body).Decode(&resp)

	if len(resp.Data) != 2 {
		t.Fatalf("expected 2 models, got %d", len(resp.Data))
	}
	if resp.Data[0].ID != "model-a" {
		t.Errorf("expected model-a, got %s", resp.Data[0].ID)
	}
	if resp.Data[0].OwnedBy != "vllm" {
		t.Errorf("expected owned_by=vllm, got %s", resp.Data[0].OwnedBy)
	}
}

func TestMakeModelsHandler_MergesLiveAndConfigured(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]string{{"id": "live-model", "object": "model", "owned_by": "backend"}},
		})
	}))
	defer server.Close()

	bs := testBackend("vllm", server)
	bs.running = true
	bs.modelNames = []string{"live-model", "config-only-model"}
	a := testArbiter(bs)

	handler := a.makeModelsHandler("vllm")
	req := httptest.NewRequest("GET", "/v1/models", nil)
	w := httptest.NewRecorder()
	handler(w, req)

	var resp struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	json.NewDecoder(w.Body).Decode(&resp)

	if len(resp.Data) != 2 {
		t.Fatalf("expected 2 models (no duplicates), got %d: %v", len(resp.Data), resp.Data)
	}
}

// --- TestMakeHealthHandler ---

func TestMakeHealthHandler(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer server.Close()

	bs := testBackend("vllm", server)
	bs.running = true
	bs.currentModel = "test-model"
	a := testArbiter(bs)

	handler := a.makeHealthHandler("vllm")
	req := httptest.NewRequest("GET", "/health", nil)
	w := httptest.NewRecorder()
	handler(w, req)

	var resp map[string]any
	json.NewDecoder(w.Body).Decode(&resp)

	if resp["status"] != "ok" {
		t.Errorf("expected status=ok, got %v", resp["status"])
	}
	if resp["backend"] != "vllm" {
		t.Errorf("expected backend=vllm, got %v", resp["backend"])
	}
	if resp["running"] != true {
		t.Errorf("expected running=true, got %v", resp["running"])
	}
	if resp["current_model"] != "test-model" {
		t.Errorf("expected current_model=test-model, got %v", resp["current_model"])
	}
}

// --- TestModelsForBackend ---

func TestModelsForBackend(t *testing.T) {
	models := []configModel{
		{Name: "qwen3.5:9b", Backend: []string{"ollama", "sglang"}},
		{Name: "Nemotron-NVFP4", Backend: []string{"vllm", "sglang"}},
		{Name: "llama3.2", Backend: []string{"ollama"}},
	}

	tests := []struct {
		backend string
		want    []string
	}{
		{"ollama", []string{"qwen3.5:9b", "llama3.2"}},
		{"vllm", []string{"Nemotron-NVFP4"}},
		{"sglang", []string{"qwen3.5:9b", "Nemotron-NVFP4"}},
		{"unknown", nil},
	}

	for _, tt := range tests {
		t.Run(tt.backend, func(t *testing.T) {
			got := modelsForBackend(models, tt.backend)
			if len(got) != len(tt.want) {
				t.Errorf("modelsForBackend(%q) = %v, want %v", tt.backend, got, tt.want)
				return
			}
			for i, name := range got {
				if name != tt.want[i] {
					t.Errorf("modelsForBackend(%q)[%d] = %q, want %q", tt.backend, i, name, tt.want[i])
				}
			}
		})
	}
}

// --- TestGPUExclusion ---

func TestEnsureBackendRunning_OllamaAlwaysSucceeds(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer server.Close()

	bs := testBackend("ollama", server)
	a := testArbiter(bs)

	a.mu.Lock()
	err := a.ensureBackendRunning(bs, "")
	a.mu.Unlock()

	if err != nil {
		t.Errorf("expected no error for ollama, got %v", err)
	}
	if !bs.running {
		t.Error("expected ollama to be running")
	}
}

func TestEnsureBackendRunning_RequiresModelForNonOllama(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer server.Close()

	bs := testBackend("vllm", server)
	bs.config.Image = "test-image"
	bs.config.ModelsDir = "/tmp"
	bs.config.Entrypoint = func(m string) []string { return []string{"echo", m} }
	a := testArbiter(bs)

	a.mu.Lock()
	err := a.ensureBackendRunning(bs, "")
	a.mu.Unlock()

	if err == nil {
		t.Error("expected error when model name is empty for vllm")
	}
}

// --- TestDrainBackend ---

func TestDrainBackend_ReturnsImmediatelyWhenNoActiveRequests(t *testing.T) {
	bs := &backendState{config: backendConfig{Name: "test"}}
	a := &arbiter{drainTimeout: 1 * time.Second}

	start := time.Now()
	a.drainBackend(bs)
	if time.Since(start) > 100*time.Millisecond {
		t.Error("drain should return immediately with no active requests")
	}
}

func TestDrainBackend_WaitsForActiveRequests(t *testing.T) {
	bs := &backendState{config: backendConfig{Name: "test"}}
	atomic.StoreInt64(&bs.activeReqs, 1)
	a := &arbiter{drainTimeout: 2 * time.Second}

	go func() {
		time.Sleep(200 * time.Millisecond)
		atomic.StoreInt64(&bs.activeReqs, 0)
	}()

	start := time.Now()
	a.drainBackend(bs)
	elapsed := time.Since(start)

	if elapsed < 150*time.Millisecond {
		t.Errorf("drain should wait for requests, elapsed %v", elapsed)
	}
	if elapsed > 1*time.Second {
		t.Errorf("drain should return once requests finish, elapsed %v", elapsed)
	}
}

func TestDrainBackend_TimesOut(t *testing.T) {
	bs := &backendState{config: backendConfig{Name: "test"}}
	atomic.StoreInt64(&bs.activeReqs, 1)
	a := &arbiter{drainTimeout: 200 * time.Millisecond}

	start := time.Now()
	a.drainBackend(bs)
	elapsed := time.Since(start)

	if elapsed < 150*time.Millisecond {
		t.Errorf("drain should wait for timeout, elapsed %v", elapsed)
	}
	if elapsed > 1*time.Second {
		t.Errorf("drain should not exceed timeout significantly, elapsed %v", elapsed)
	}
}

// --- TestSmartProxy ---

func TestSmartProxy_RewritesContextOverflow(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprint(w, `{"error":"maximum context length exceeded"}`)
	}))
	defer backend.Close()

	u, _ := url.Parse(backend.URL)
	proxy := newSmartProxy(u)

	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	w := httptest.NewRecorder()
	proxy.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", w.Code)
	}
}

func TestSmartProxy_PassesThroughOtherErrors(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprint(w, `{"error":"some other error"}`)
	}))
	defer backend.Close()

	u, _ := url.Parse(backend.URL)
	proxy := newSmartProxy(u)

	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	w := httptest.NewRecorder()
	proxy.ServeHTTP(w, req)

	if w.Code != http.StatusInternalServerError {
		t.Errorf("expected 500, got %d", w.Code)
	}
}

// --- TestEnvHelpers ---

func TestEnv(t *testing.T) {
	t.Setenv("TEST_VAR", "hello")
	if got := env("TEST_VAR", "default"); got != "hello" {
		t.Errorf("expected 'hello', got %q", got)
	}
	if got := env("NONEXISTENT_VAR", "default"); got != "default" {
		t.Errorf("expected 'default', got %q", got)
	}
}

func TestEnvInt(t *testing.T) {
	t.Setenv("TEST_INT", "42")
	if got := envInt("TEST_INT", 10); got != 42 {
		t.Errorf("expected 42, got %d", got)
	}
	if got := envInt("NONEXISTENT_INT", 10); got != 10 {
		t.Errorf("expected 10, got %d", got)
	}
}

// --- TestMakeTagsHandler ---

func TestMakeTagsHandler_ReturnsConfiguredModels(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer server.Close()

	bs := testBackend("vllm", server)
	bs.modelNames = []string{"model-a"}
	a := testArbiter(bs)

	handler := a.makeTagsHandler("vllm")
	req := httptest.NewRequest("GET", "/api/tags", nil)
	w := httptest.NewRecorder()
	handler(w, req)

	var resp struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	json.NewDecoder(w.Body).Decode(&resp)

	if len(resp.Models) != 1 || resp.Models[0].Name != "model-a" {
		t.Errorf("expected [model-a], got %v", resp.Models)
	}
}

// --- TestRequestHandler returns 503 when backend cannot start ---

func TestMakeRequestHandler_Returns503WhenBackendFails(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer server.Close()

	bs := testBackend("vllm", server)
	bs.config.Entrypoint = func(m string) []string { return []string{"echo", m} }
	bs.config.ContainerName = "test-vllm"

	a := testArbiter(bs)
	// Podman client pointing to nonexistent socket — all calls fail
	a.podmanClient = &http.Client{Timeout: 100 * time.Millisecond}

	handler := a.makeRequestHandler("vllm")
	body := `{"model":"test-model"}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()
	handler(w, req)

	if w.Code != http.StatusServiceUnavailable {
		t.Errorf("expected 503, got %d", w.Code)
	}
}
