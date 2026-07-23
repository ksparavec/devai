package main

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"sync/atomic"
	"testing"
	"time"
)

// fakeExecutor records every command dispatched to it.
type fakeExecutor struct {
	mu       atomic.Pointer[fakeExecutorState]
	failType CommandType
}

type fakeExecutorState struct {
	cmds []Command
}

func (f *fakeExecutor) Execute(_ context.Context, cmd Command) error {
	for {
		old := f.mu.Load()
		var oldCmds []Command
		if old != nil {
			oldCmds = append(oldCmds, old.cmds...)
		}
		oldCmds = append(oldCmds, cmd)
		newState := &fakeExecutorState{cmds: oldCmds}
		if f.mu.CompareAndSwap(old, newState) {
			break
		}
	}
	if cmd.Type == f.failType {
		return errors.New("simulated failure")
	}
	return nil
}

func (f *fakeExecutor) commands() []Command {
	s := f.mu.Load()
	if s == nil {
		return nil
	}
	return s.cmds
}

func newWorkerWithToken(t *testing.T) (*Worker, *TokenStore, string) {
	t.Helper()
	dir := t.TempDir()
	tokPath := filepath.Join(dir, "tok")
	if err := os.WriteFile(tokPath, []byte("the-token"), 0o600); err != nil {
		t.Fatalf("write token: %v", err)
	}
	tokens := NewTokenStore(tokPath, time.Hour)
	w := NewWorker(WorkerConfig{
		HeadURL:    "http://example.invalid",
		WorkerName: "test-worker",
		Lifecycle:  LifecycleEphemeral,
		Endpoint:   "http://example.invalid:11444",
		GPUType:    "RTX4000",
		VRAMGB:     24,
		Backends:   []string{"vllm"},
		Token:      tokens,
	})
	return w, tokens, tokPath
}

func TestWorkerState_Snapshot(t *testing.T) {
	s := &WorkerState{}
	s.SetHealthStatus("ready")
	s.SetLoadedModel("Qwen3-8B-NVFP4", 131072)
	s.IncQueue()
	s.IncQueue()
	s.SetUtilization(42.5)
	s.MarkRequestAt(time.Date(2026, 5, 15, 10, 0, 0, 0, time.UTC))

	hb := s.snapshot("worker-id-1")
	if hb.WorkerID != "worker-id-1" {
		t.Errorf("worker id: got %q", hb.WorkerID)
	}
	if hb.LoadedModel != "Qwen3-8B-NVFP4" {
		t.Errorf("loaded model: got %q", hb.LoadedModel)
	}
	if hb.LoadedCtx != 131072 {
		t.Errorf("loaded ctx: got %d", hb.LoadedCtx)
	}
	if hb.QueueDepth != 2 {
		t.Errorf("queue depth: got %d, want 2", hb.QueueDepth)
	}
	if hb.UtilizationPct != 42.5 {
		t.Errorf("utilization: got %v, want 42.5", hb.UtilizationPct)
	}
	if hb.HealthStatus != "ready" {
		t.Errorf("health: got %q", hb.HealthStatus)
	}
	if hb.LastRequestAt == "" {
		t.Errorf("last request at not stamped")
	}

	// Counter increments per snapshot.
	hb2 := s.snapshot("worker-id-1")
	if hb2.Counter != hb.Counter+1 {
		t.Errorf("counter: got %d, want %d", hb2.Counter, hb.Counter+1)
	}
}

func TestWorkerState_UtilizationClamped(t *testing.T) {
	s := &WorkerState{}
	s.SetUtilization(-10)
	hb := s.snapshot("w")
	if hb.UtilizationPct != 0 {
		t.Errorf("negative not clamped to 0: %v", hb.UtilizationPct)
	}
	s.SetUtilization(150)
	hb2 := s.snapshot("w")
	if hb2.UtilizationPct != 100 {
		t.Errorf("over-100 not clamped: %v", hb2.UtilizationPct)
	}
}

func TestWorkerState_UnloadedClearsModel(t *testing.T) {
	s := &WorkerState{}
	s.SetLoadedModel("Qwen3", 32768)
	s.SetLoadedModel("", 0)
	hb := s.snapshot("w")
	if hb.LoadedModel != "" {
		t.Errorf("expected empty loaded_model after clear, got %q", hb.LoadedModel)
	}
	if hb.LoadedCtx != 0 {
		t.Errorf("expected zero ctx after clear, got %d", hb.LoadedCtx)
	}
}

// stubHead spins up an httptest.Server that:
//   - validates the bearer token
//   - on /v1/cluster/register returns the canned worker_id
//   - on /v1/cluster/heartbeat returns the canned commands list
//
// nReceivedRegister / nReceivedHeartbeat let tests assert call counts.
type stubHead struct {
	t                 *testing.T
	expectedToken     string
	registerWorkerID  string
	heartbeatCommands []Command
	nReceivedRegister atomic.Int32
	nReceivedHB       atomic.Int32
	failNHeartbeats   atomic.Int32 // how many heartbeats to reject with 500
}

func (s *stubHead) handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/cluster/register", func(w http.ResponseWriter, r *http.Request) {
		got := bearerFromHeader(r.Header.Get("Authorization"))
		if got != s.expectedToken {
			http.Error(w, "auth", http.StatusUnauthorized)
			return
		}
		s.nReceivedRegister.Add(1)
		_ = json.NewEncoder(w).Encode(RegisterResponse{
			WorkerID: s.registerWorkerID,
		})
	})
	mux.HandleFunc("/v1/cluster/heartbeat", func(w http.ResponseWriter, r *http.Request) {
		got := bearerFromHeader(r.Header.Get("Authorization"))
		if got != s.expectedToken {
			http.Error(w, "auth", http.StatusUnauthorized)
			return
		}
		s.nReceivedHB.Add(1)
		if remaining := s.failNHeartbeats.Add(-1); remaining >= 0 {
			http.Error(w, "stub-fail", http.StatusInternalServerError)
			return
		}
		_ = json.NewEncoder(w).Encode(HeartbeatResponse{
			Commands: s.heartbeatCommands,
		})
	})
	return mux
}

func TestWorker_RegisterOnceSucceeds(t *testing.T) {
	stub := &stubHead{
		t:                t,
		expectedToken:    "the-token",
		registerWorkerID: "wid-001",
	}
	srv := httptest.NewServer(stub.handler())
	defer srv.Close()

	w, _, _ := newWorkerWithToken(t)
	w.Config.HeadURL = srv.URL
	w.Config.HTTPClient = http.DefaultClient

	body := RegisterRequest{
		Name: w.Config.WorkerName, Lifecycle: w.Config.Lifecycle,
		Endpoint: w.Config.Endpoint, Backends: w.Config.Backends,
		VRAMGB: w.Config.VRAMGB,
	}
	id, err := w.registerOnce(context.Background(), body)
	if err != nil {
		t.Fatalf("registerOnce: %v", err)
	}
	if id != "wid-001" {
		t.Errorf("worker_id: got %q", id)
	}
}

func TestWorker_HeartbeatOnceDispatchesCommands(t *testing.T) {
	stub := &stubHead{
		t:                t,
		expectedToken:    "the-token",
		registerWorkerID: "wid-001",
		heartbeatCommands: []Command{
			{Type: CommandDrain, Backend: "vllm"},
			{Type: CommandServe, RequestID: "r1", TargetModel: "Qwen3-8B-NVFP4", TargetCtx: 32768},
		},
	}
	srv := httptest.NewServer(stub.handler())
	defer srv.Close()

	w, _, _ := newWorkerWithToken(t)
	w.Config.HeadURL = srv.URL
	w.Config.HTTPClient = http.DefaultClient
	exec := &fakeExecutor{}
	w.Executor = exec

	resp, err := w.HeartbeatOnce(context.Background(), "wid-001")
	if err != nil {
		t.Fatalf("HeartbeatOnce: %v", err)
	}
	if len(resp.Commands) != 2 {
		t.Fatalf("got %d commands, want 2", len(resp.Commands))
	}

	// dispatchCommand for each
	for _, cmd := range resp.Commands {
		if err := w.dispatchCommand(context.Background(), cmd); err != nil {
			t.Errorf("dispatch %s: %v", cmd.Type, err)
		}
	}
	cmds := exec.commands()
	if len(cmds) != 2 {
		t.Fatalf("executor saw %d commands, want 2", len(cmds))
	}
	if cmds[0].Type != CommandDrain {
		t.Errorf("first cmd: %q", cmds[0].Type)
	}
	if cmds[1].Type != CommandServe || cmds[1].TargetModel != "Qwen3-8B-NVFP4" {
		t.Errorf("second cmd unexpected: %+v", cmds[1])
	}
}

func TestWorker_PersistentRefusesShutdown(t *testing.T) {
	w, _, _ := newWorkerWithToken(t)
	w.Config.Lifecycle = LifecyclePersistent
	exec := &fakeExecutor{}
	w.Executor = exec

	err := w.dispatchCommand(context.Background(), Command{
		Type: CommandShutdown, GraceSeconds: 30,
	})
	if err != nil {
		t.Fatalf("persistent shutdown should be silently refused, got: %v", err)
	}
	if len(exec.commands()) != 0 {
		t.Errorf("executor was called for refused shutdown: %+v", exec.commands())
	}
}

func TestWorker_EphemeralAcceptsShutdown(t *testing.T) {
	w, _, _ := newWorkerWithToken(t)
	w.Config.Lifecycle = LifecycleEphemeral
	exec := &fakeExecutor{}
	w.Executor = exec

	err := w.dispatchCommand(context.Background(), Command{
		Type: CommandShutdown, GraceSeconds: 5,
	})
	if err != nil {
		t.Fatalf("ephemeral shutdown should be accepted: %v", err)
	}
	cmds := exec.commands()
	if len(cmds) != 1 || cmds[0].Type != CommandShutdown {
		t.Errorf("expected one shutdown cmd, got %+v", cmds)
	}
}

func TestWorker_UnknownCommandIgnored(t *testing.T) {
	w, _, _ := newWorkerWithToken(t)
	exec := &fakeExecutor{}
	w.Executor = exec

	err := w.dispatchCommand(context.Background(), Command{Type: "fubar"})
	if err != nil {
		t.Fatalf("unknown command should be silently ignored: %v", err)
	}
	if len(exec.commands()) != 0 {
		t.Errorf("executor was called for unknown command: %+v", exec.commands())
	}
}

func TestWorker_HeartbeatRetriesAfterFailure(t *testing.T) {
	stub := &stubHead{
		t:                t,
		expectedToken:    "the-token",
		registerWorkerID: "wid-001",
	}
	stub.failNHeartbeats.Store(1) // first heartbeat returns 500
	srv := httptest.NewServer(stub.handler())
	defer srv.Close()

	w, _, _ := newWorkerWithToken(t)
	w.Config.HeadURL = srv.URL
	w.Config.HTTPClient = http.DefaultClient

	_, err := w.HeartbeatOnce(context.Background(), "wid-001")
	if err == nil {
		t.Fatalf("expected first heartbeat to fail")
	}
	// Second heartbeat should succeed (failN counter exhausted).
	resp, err := w.HeartbeatOnce(context.Background(), "wid-001")
	if err != nil {
		t.Fatalf("second heartbeat: %v", err)
	}
	if resp == nil {
		t.Fatalf("nil response on success")
	}
}

func TestJitter_BoundsAndAlwaysPositive(t *testing.T) {
	// 30 % spread on 1s; result must lie in [0.7s, 1.3s].
	const base = time.Second
	for i := 0; i < 100; i++ {
		got := jitter(base)
		if got < 700*time.Millisecond || got > 1300*time.Millisecond {
			t.Fatalf("jitter out of bounds: %v", got)
		}
	}
	// Below floor: clamped to 100ms.
	got := jitter(0)
	if got < 100*time.Millisecond {
		t.Errorf("jitter below floor: %v", got)
	}
}

func TestFloat64Bits_RoundTrip(t *testing.T) {
	for _, v := range []float64{0, 0.5, -1.25, 42.7, 100.0} {
		if got := bitsToFloat64(float64ToBits(v)); got != v {
			t.Errorf("round trip lost %v -> %v", v, got)
		}
	}
}

// --- Re-registration after the head forgets the worker (410 Gone) ---

// goneThenOKHead answers the first `gone` heartbeats with 410 (the
// head's "I do not know this worker_id; re-register" contract) and
// mints a fresh worker_id on every registration.
type goneThenOKHead struct {
	remainingGone atomic.Int32
	nRegister     atomic.Int32
	nHeartbeat    atomic.Int32
}

func (s *goneThenOKHead) handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/cluster/register", func(w http.ResponseWriter, _ *http.Request) {
		n := s.nRegister.Add(1)
		_ = json.NewEncoder(w).Encode(RegisterResponse{
			WorkerID: "wid-" + strconv.Itoa(int(n)),
		})
	})
	mux.HandleFunc("/v1/cluster/heartbeat", func(w http.ResponseWriter, _ *http.Request) {
		s.nHeartbeat.Add(1)
		if s.remainingGone.Add(-1) >= 0 {
			http.Error(w, "unknown worker_id; please re-register", http.StatusGone)
			return
		}
		_ = json.NewEncoder(w).Encode(HeartbeatResponse{})
	})
	return mux
}

func TestHeartbeatOnce_410IsErrHeadUnknownWorker(t *testing.T) {
	stub := &goneThenOKHead{}
	stub.remainingGone.Store(1)
	srv := httptest.NewServer(stub.handler())
	defer srv.Close()

	w, _, _ := newWorkerWithToken(t)
	w.Config.HeadURL = srv.URL
	w.Config.HTTPClient = http.DefaultClient

	_, err := w.HeartbeatOnce(context.Background(), "stale-id")
	if !errors.Is(err, ErrHeadUnknownWorker) {
		t.Fatalf("410 produced %v, want ErrHeadUnknownWorker", err)
	}
}

func TestWorkerTick_ReregistersAfterHeadForgetsWorker(t *testing.T) {
	// The head restarted: its fleet map is in-memory only, so the
	// worker's id is gone. Without re-registration the worker
	// heartbeats into a 410 forever and never rejoins the fleet.
	stub := &goneThenOKHead{}
	stub.remainingGone.Store(1)
	srv := httptest.NewServer(stub.handler())
	defer srv.Close()

	w, _, _ := newWorkerWithToken(t)
	w.Config.HeadURL = srv.URL
	w.Config.HTTPClient = http.DefaultClient
	w.Executor = &fakeExecutor{}
	stale := "stale-id"
	w.State.WorkerID.Store(&stale)

	w.tick(context.Background())

	if stub.nRegister.Load() != 1 {
		t.Fatalf("register called %d times after 410, want 1", stub.nRegister.Load())
	}
	id := w.State.WorkerID.Load()
	if id == nil || *id != "wid-1" {
		t.Fatalf("worker_id after re-register = %v, want wid-1", id)
	}

	// The next tick must heartbeat normally under the new id.
	w.tick(context.Background())
	if stub.nHeartbeat.Load() != 2 {
		t.Errorf("heartbeats: got %d, want 2", stub.nHeartbeat.Load())
	}
	if stub.nRegister.Load() != 1 {
		t.Errorf("re-registered again on a healthy heartbeat (%d registers)",
			stub.nRegister.Load())
	}
}

func TestWorkerTick_OtherHeartbeatFailuresDoNotReregister(t *testing.T) {
	stub := &stubHead{t: t, expectedToken: "the-token", registerWorkerID: "wid-001"}
	stub.failNHeartbeats.Store(1)
	srv := httptest.NewServer(stub.handler())
	defer srv.Close()

	w, _, _ := newWorkerWithToken(t)
	w.Config.HeadURL = srv.URL
	w.Config.HTTPClient = http.DefaultClient
	w.Executor = &fakeExecutor{}
	id := "wid-001"
	w.State.WorkerID.Store(&id)

	w.tick(context.Background())

	if stub.nReceivedRegister.Load() != 0 {
		t.Fatalf("a 500 heartbeat must not trigger re-registration (%d registers)",
			stub.nReceivedRegister.Load())
	}
	if got := w.State.WorkerID.Load(); got == nil || *got != "wid-001" {
		t.Fatalf("worker_id changed on a transient failure: %v", got)
	}
}
