package main

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// fakeForwarder records the worker chosen for each Forward call and
// returns a canned 200/JSON response so the head's frontend handler
// runs end-to-end without needing a real worker.
type fakeForwarder struct {
	choices  []WorkerEntry
	backends []string
}

func (f *fakeForwarder) Forward(
	w http.ResponseWriter, _ *http.Request, worker WorkerEntry,
	_ MinimalRequest, backend string,
) {
	f.choices = append(f.choices, worker)
	f.backends = append(f.backends, backend)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(`{"ok":true,"worker_id":"` + worker.WorkerID + `"}`))
}

func newHeadForTest(t *testing.T) (*ClusterHead, *fakeForwarder) {
	t.Helper()
	dir := t.TempDir()
	tokPath := filepath.Join(dir, "tok")
	if err := os.WriteFile(tokPath, []byte("the-token"), 0o600); err != nil {
		t.Fatalf("write token: %v", err)
	}
	tokens := NewTokenStore(tokPath, time.Hour)
	fake := &fakeForwarder{}
	h := &ClusterHead{
		Fleet:             NewFleetState(),
		Policy:            &RoutingPolicy{},
		Token:             tokens,
		Forward:           fake,
		HeadListenPort:    0,
		FrontendPorts:     map[string]int{"vllm": 0},
		IdleSweepInterval: time.Hour,
	}
	return h, fake
}

func TestHandleRegister_AssignsAndStores(t *testing.T) {
	h, _ := newHeadForTest(t)
	body, _ := json.Marshal(mkReq("worker-x"))
	req := httptest.NewRequest(http.MethodPost, "/v1/cluster/register", bytes.NewReader(body))
	req.Header.Set("Authorization", "Bearer the-token")
	w := httptest.NewRecorder()
	h.handleRegister(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("status: %d, body: %s", w.Code, w.Body.String())
	}
	var rr RegisterResponse
	if err := json.Unmarshal(w.Body.Bytes(), &rr); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if rr.WorkerID == "" {
		t.Errorf("worker_id empty")
	}
	if h.Fleet.Count() != 1 {
		t.Errorf("fleet count: %d", h.Fleet.Count())
	}
}

func TestHandleRegister_Rejects400OnBadBody(t *testing.T) {
	h, _ := newHeadForTest(t)
	req := httptest.NewRequest(http.MethodPost, "/v1/cluster/register",
		bytes.NewReader([]byte(`{"name":""}`)))
	w := httptest.NewRecorder()
	h.handleRegister(w, req)
	if w.Code != http.StatusBadRequest {
		t.Errorf("status: %d, want 400", w.Code)
	}
}

func TestHandleHeartbeat_Updates(t *testing.T) {
	h, _ := newHeadForTest(t)
	id := h.Fleet.Register(mkReq("w"), time.Now())

	hb := HeartbeatRequest{
		WorkerID: id, LoadedModel: "Qwen3", LoadedCtx: 65536,
		Counter: 1, HealthStatus: "ready",
	}
	body, _ := json.Marshal(hb)
	req := httptest.NewRequest(http.MethodPost, "/v1/cluster/heartbeat", bytes.NewReader(body))
	req.Header.Set("Authorization", "Bearer the-token")
	w := httptest.NewRecorder()
	h.handleHeartbeat(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("status: %d, body: %s", w.Code, w.Body.String())
	}
	got, _ := h.Fleet.Get(id)
	if got.LoadedModel != "Qwen3" {
		t.Errorf("loaded model: %q", got.LoadedModel)
	}
}

func TestHandleHeartbeat_UnknownWorker410(t *testing.T) {
	h, _ := newHeadForTest(t)
	body, _ := json.Marshal(HeartbeatRequest{WorkerID: "ghost", Counter: 1})
	req := httptest.NewRequest(http.MethodPost, "/v1/cluster/heartbeat", bytes.NewReader(body))
	w := httptest.NewRecorder()
	h.handleHeartbeat(w, req)
	if w.Code != http.StatusGone {
		t.Errorf("status: %d, want 410", w.Code)
	}
}

func TestHandleHeartbeat_StaleReturnsEmpty(t *testing.T) {
	h, _ := newHeadForTest(t)
	id := h.Fleet.Register(mkReq("w"), time.Now())
	_ = h.Fleet.Heartbeat(HeartbeatRequest{WorkerID: id, Counter: 5}, time.Now())

	body, _ := json.Marshal(HeartbeatRequest{WorkerID: id, Counter: 3})
	req := httptest.NewRequest(http.MethodPost, "/v1/cluster/heartbeat", bytes.NewReader(body))
	w := httptest.NewRecorder()
	h.handleHeartbeat(w, req)
	if w.Code != http.StatusOK {
		t.Errorf("stale heartbeat should still return 200, got %d", w.Code)
	}
	var hr HeartbeatResponse
	if err := json.Unmarshal(w.Body.Bytes(), &hr); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(hr.Commands) != 0 {
		t.Errorf("stale heartbeat should return no commands, got %v", hr.Commands)
	}
}

// Handler-level shape check. The mounted (authenticated) surface is
// covered by TestControlPlaneMux_StatusRequiresAuth.
func TestHandleStatus_ListsWorkers(t *testing.T) {
	h, _ := newHeadForTest(t)
	h.Fleet.Register(mkReq("worker-a"), time.Now())
	h.Fleet.Register(mkReq("worker-b"), time.Now())
	req := httptest.NewRequest(http.MethodGet, "/v1/cluster/status", nil)
	w := httptest.NewRecorder()
	h.handleStatus(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("status: %d", w.Code)
	}
	var arr []StatusEntry
	if err := json.Unmarshal(w.Body.Bytes(), &arr); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(arr) != 2 {
		t.Errorf("expected 2 entries, got %d", len(arr))
	}
}

func TestFrontendHandler_NoFleetReturns503(t *testing.T) {
	h, _ := newHeadForTest(t)
	body := []byte(`{"model":"Qwen3-8B-NVFP4","messages":[]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	w := httptest.NewRecorder()
	h.makeFrontendHandler("vllm")(w, req)
	if w.Code != http.StatusServiceUnavailable {
		t.Errorf("status: %d, want 503", w.Code)
	}
	if w.Header().Get("Retry-After") == "" {
		t.Errorf("missing Retry-After header")
	}
	// Body should mention the requested model + backend.
	if !bytes.Contains(w.Body.Bytes(), []byte("Qwen3-8B-NVFP4")) {
		t.Errorf("503 body missing model: %s", w.Body.String())
	}
	if !bytes.Contains(w.Body.Bytes(), []byte(`"backend":"vllm"`)) {
		t.Errorf("503 body missing backend: %s", w.Body.String())
	}
}

func TestFrontendHandler_InvalidBodyReturns400(t *testing.T) {
	h, _ := newHeadForTest(t)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		bytes.NewReader([]byte(`not json`)))
	w := httptest.NewRecorder()
	h.makeFrontendHandler("vllm")(w, req)
	if w.Code != http.StatusBadRequest {
		t.Errorf("status: %d, want 400", w.Code)
	}
}

func TestFrontendHandler_ForwardsToChosenWorker(t *testing.T) {
	h, fake := newHeadForTest(t)
	id := h.Fleet.Register(mkReq("w-1"), time.Now())
	_ = h.Fleet.Heartbeat(HeartbeatRequest{
		WorkerID: id, Counter: 1, LoadedModel: "Qwen3-8B-NVFP4",
		LoadedCtx: 131072, HealthStatus: "ready",
	}, time.Now())

	body := []byte(`{"model":"Qwen3-8B-NVFP4@65536","messages":[]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	w := httptest.NewRecorder()
	h.makeFrontendHandler("vllm")(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("status: %d, body: %s", w.Code, w.Body.String())
	}
	if len(fake.choices) != 1 {
		t.Fatalf("forwarder called %d times, want 1", len(fake.choices))
	}
	if fake.choices[0].Name != "w-1" {
		t.Errorf("chose %q, want w-1", fake.choices[0].Name)
	}
	out, _ := io.ReadAll(w.Body)
	if !bytes.Contains(out, []byte(id)) {
		t.Errorf("response body missing chosen worker_id: %s", string(out))
	}
}

func TestCommandsFor_IdleEphemeralGetsShutdown(t *testing.T) {
	h, _ := newHeadForTest(t)
	h.IdleMinutes = 1
	now := time.Now()
	id := h.Fleet.Register(mkReq("w"), now)
	// Patch lifecycle (mkReq returns persistent by default).
	w, _ := h.Fleet.Get(id)
	w.Lifecycle = LifecycleEphemeral
	h.Fleet.workers[id] = &w
	w.LastRequestAt = now.Add(-2 * time.Minute).UTC().Format(time.RFC3339)
	h.Fleet.workers[id] = &w

	cmds := h.commandsFor(id, now)
	if len(cmds) != 1 || cmds[0].Type != CommandShutdown {
		t.Errorf("expected single shutdown command, got %v", cmds)
	}
}

func TestCommandsFor_RecentRequestNoShutdown(t *testing.T) {
	h, _ := newHeadForTest(t)
	h.IdleMinutes = 10
	now := time.Now()
	id := h.Fleet.Register(mkReq("w"), now)
	w, _ := h.Fleet.Get(id)
	w.Lifecycle = LifecycleEphemeral
	w.LastRequestAt = now.Add(-1 * time.Second).UTC().Format(time.RFC3339)
	h.Fleet.workers[id] = &w

	cmds := h.commandsFor(id, now)
	if len(cmds) != 0 {
		t.Errorf("recent request should not trigger shutdown, got %v", cmds)
	}
}

func TestCommandsFor_PersistentNeverShutdown(t *testing.T) {
	h, _ := newHeadForTest(t)
	h.IdleMinutes = 1
	now := time.Now()
	id := h.Fleet.Register(mkReq("w"), now)
	w, _ := h.Fleet.Get(id)
	// Persistent + idle for ages should still get nothing.
	w.LastRequestAt = now.Add(-1 * time.Hour).UTC().Format(time.RFC3339)
	h.Fleet.workers[id] = &w

	cmds := h.commandsFor(id, now)
	if len(cmds) != 0 {
		t.Errorf("persistent worker should never get shutdown, got %v", cmds)
	}
}

// --- Control-plane surface: auth + body caps ---

func TestControlPlaneMux_StatusRequiresAuth(t *testing.T) {
	h, _ := newHeadForTest(t)
	h.Fleet.Register(mkReq("worker-a"), time.Now())
	mux := h.controlPlaneMux()

	// Unauthenticated: fleet topology (endpoints, GPU types, loaded
	// models) must not leak to anything that reaches the port.
	req := httptest.NewRequest(http.MethodGet, "/v1/cluster/status", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	if w.Code != http.StatusUnauthorized {
		t.Fatalf("unauthenticated status: %d, want 401 (body: %s)", w.Code, w.Body.String())
	}
	if bytes.Contains(w.Body.Bytes(), []byte("worker-a")) {
		t.Errorf("401 body leaked a worker name: %s", w.Body.String())
	}

	// Authenticated: unchanged behaviour.
	req2 := httptest.NewRequest(http.MethodGet, "/v1/cluster/status", nil)
	req2.Header.Set("Authorization", "Bearer the-token")
	w2 := httptest.NewRecorder()
	mux.ServeHTTP(w2, req2)
	if w2.Code != http.StatusOK {
		t.Fatalf("authenticated status: %d, want 200", w2.Code)
	}
	var arr []StatusEntry
	if err := json.Unmarshal(w2.Body.Bytes(), &arr); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(arr) != 1 {
		t.Errorf("expected 1 entry, got %d", len(arr))
	}
}

func TestControlPlaneMux_HealthStaysUnauthenticated(t *testing.T) {
	h, _ := newHeadForTest(t)
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()
	h.controlPlaneMux().ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("health: %d, want 200 (liveness probes carry no token)", w.Code)
	}
}

// oversizedBody returns a JSON body larger than ClusterMaxBodyBytes.
func oversizedBody() *bytes.Reader {
	pad := bytes.Repeat([]byte("a"), ClusterMaxBodyBytes+1024)
	return bytes.NewReader(append(append([]byte(`{"model":"`), pad...), []byte(`"}`)...))
}

func TestHandleRegister_RejectsOversizedBody(t *testing.T) {
	h, _ := newHeadForTest(t)
	req := httptest.NewRequest(http.MethodPost, "/v1/cluster/register", oversizedBody())
	w := httptest.NewRecorder()
	h.handleRegister(w, req)
	if w.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status: %d, want 413 -- an uncapped read lets one peer exhaust head RAM", w.Code)
	}
}

func TestHandleHeartbeat_RejectsOversizedBody(t *testing.T) {
	h, _ := newHeadForTest(t)
	req := httptest.NewRequest(http.MethodPost, "/v1/cluster/heartbeat", oversizedBody())
	w := httptest.NewRecorder()
	h.handleHeartbeat(w, req)
	if w.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status: %d, want 413", w.Code)
	}
}

func TestFrontendHandler_RejectsOversizedBody(t *testing.T) {
	h, fake := newHeadForTest(t)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", oversizedBody())
	w := httptest.NewRecorder()
	h.makeFrontendHandler("vllm")(w, req)
	if w.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status: %d, want 413", w.Code)
	}
	if len(fake.choices) != 0 {
		t.Errorf("oversized request reached the forwarder")
	}
}

func TestFrontendHandler_PassesBackendToForwarder(t *testing.T) {
	h, fake := newHeadForTest(t)
	id := h.Fleet.Register(mkReq("w-1"), time.Now())
	_ = h.Fleet.Heartbeat(HeartbeatRequest{
		WorkerID: id, Counter: 1, LoadedModel: "Qwen3-8B-NVFP4",
		LoadedCtx: 131072, HealthStatus: "ready",
	}, time.Now())

	body := []byte(`{"model":"Qwen3-8B-NVFP4","messages":[]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	w := httptest.NewRecorder()
	h.makeFrontendHandler("vllm")(w, req)

	if len(fake.backends) != 1 || fake.backends[0] != "vllm" {
		t.Fatalf("forwarder saw backends %v, want [vllm] -- the worker cannot "+
			"infer the backend from the body", fake.backends)
	}
}
