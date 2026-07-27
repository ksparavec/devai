package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// fakePodman serves canned libpod responses so tests can drive
// containerState / containerRecentLogs without a real podman socket.
type fakePodman struct {
	stateJSON string
	logs      string
}

func (f *fakePodman) RoundTrip(req *http.Request) (*http.Response, error) {
	mk := func(body string) *http.Response {
		return &http.Response{
			StatusCode: 200,
			Body:       io.NopCloser(strings.NewReader(body)),
			Header:     make(http.Header),
		}
	}
	switch {
	case strings.Contains(req.URL.Path, "/logs"):
		return mk(f.logs), nil
	case strings.HasSuffix(req.URL.Path, "/json"):
		return mk(f.stateJSON), nil
	default:
		return mk("{}"), nil
	}
}

func fakePodmanArbiter(stateJSON, logs string) *arbiter {
	return &arbiter{
		podmanClient: &http.Client{Transport: &fakePodman{stateJSON: stateJSON, logs: logs}},
	}
}

// --- Fail-fast: detectLaunchFailure ---

func TestDetectLaunchFailure_CrashedContainerReportsRootCause(t *testing.T) {
	logs := strings.Join([]string{
		"INFO loading model ...",
		`  File "gemma4.py", line 1554, in __init__`,
		"    self.lm_head = self.lm_head.tie_weights(...)",
		"NotImplementedError",
		"RuntimeError: Engine core initialization failed.",
	}, "\n")
	a := fakePodmanArbiter(`{"State":{"Status":"exited","ExitCode":1}}`, logs)

	err := a.detectLaunchFailure("devai-vllm")
	if err == nil {
		t.Fatal("want crash error, got nil")
	}
	if !strings.Contains(err.Error(), "engine crashed (exit 1)") {
		t.Fatalf("want 'engine crashed (exit 1)', got %q", err)
	}
	// last-match anchor surfaces the final exception line, not the traceback header
	if !strings.Contains(err.Error(), "Engine core initialization failed") {
		t.Fatalf("want root-cause line, got %q", err)
	}
}

func TestDetectLaunchFailure_ExitedNoSignatureStillFails(t *testing.T) {
	a := fakePodmanArbiter(`{"State":{"Status":"exited","ExitCode":137}}`, "just some benign shutdown noise\n")
	err := a.detectLaunchFailure("devai-vllm")
	if err == nil || !strings.Contains(err.Error(), "exit 137") {
		t.Fatalf("an exited container must fail-fast even without a signature, got %v", err)
	}
}

func TestDetectLaunchFailure_RunningWithOOMSignatureFails(t *testing.T) {
	logs := "INFO capturing graphs\ntorch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.37 GiB"
	a := fakePodmanArbiter(`{"State":{"Status":"running","ExitCode":0}}`, logs)
	err := a.detectLaunchFailure("devai-vllm")
	if err == nil || !strings.Contains(err.Error(), "fatal error") {
		t.Fatalf("a running container with an OOM signature must fail-fast, got %v", err)
	}
}

func TestDetectLaunchFailure_HealthySlowLoadReturnsNil(t *testing.T) {
	// A legitimate slow load: still running, no terminal signature. Must NOT
	// abort -- these lines all appear during a healthy NVFP4 cold start.
	logs := strings.Join([]string{
		"INFO Starting to load model /models/foo ...",
		"Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 42%",
		"INFO torch.compile took 7.78 s in total",
		"INFO init engine ... took 20.32 s",
	}, "\n")
	a := fakePodmanArbiter(`{"State":{"Status":"running","ExitCode":0}}`, logs)
	if err := a.detectLaunchFailure("devai-vllm"); err != nil {
		t.Fatalf("healthy slow load must not fail-fast, got %v", err)
	}
}

// --- demuxDockerStream ---

func TestDemuxDockerStream_RawPassthrough(t *testing.T) {
	raw := []byte("plain log line\nanother line\n")
	if got := demuxDockerStream(raw); got != string(raw) {
		t.Fatalf("raw passthrough failed: %q", got)
	}
}

func TestDemuxDockerStream_StripsFrameHeaders(t *testing.T) {
	frame := func(streamType byte, payload string) []byte {
		hdr := []byte{streamType, 0, 0, 0, 0, 0, 0, 0}
		binary.BigEndian.PutUint32(hdr[4:], uint32(len(payload)))
		return append(hdr, []byte(payload)...)
	}
	raw := append(frame(1, "hello\n"), frame(2, "boom\n")...)
	if got := demuxDockerStream(raw); got != "hello\nboom\n" {
		t.Fatalf("demux failed: %q", got)
	}
}

// --- lastErrorLine ---

func TestLastErrorLine_ReturnsLastMatch(t *testing.T) {
	logs := "RuntimeError: first\nsome noise\nValueError: last"
	if got := lastErrorLine(logs, failureAnchors); got != "ValueError: last" {
		t.Fatalf("want last match, got %q", got)
	}
	if got := lastErrorLine("no errors here at all", failureAnchors); got != "" {
		t.Fatalf("want empty for no match, got %q", got)
	}
}

// --- Concurrency admission cap (HTTP 429) ---

func TestMakeRequestHandler_ConcurrencyCap429(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	bs := testBackend("vllm", server)
	bs.running = true
	a := testArbiter(bs)
	a.maxConcurrent = 2
	atomic.StoreInt64(&bs.activeReqs, 2) // already at cap; this request makes it 3

	w := httptest.NewRecorder()
	a.makeRequestHandler("vllm")(w, httptest.NewRequest("GET", "/v1/models", nil))

	if w.Code != http.StatusTooManyRequests {
		t.Fatalf("want 429, got %d", w.Code)
	}
	if n := atomic.LoadInt64(&bs.activeReqs); n != 2 {
		t.Fatalf("activeReqs leaked after 429: want 2, got %d", n)
	}
}

func TestMakeRequestHandler_UnderCapProceeds(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]string{"ok": "1"})
	}))
	defer server.Close()

	bs := testBackend("ollama", server)
	bs.running = true
	a := testArbiter(bs)
	a.maxConcurrent = 5 // under cap

	w := httptest.NewRecorder()
	a.makeRequestHandler("ollama")(w, httptest.NewRequest("GET", "/v1/models", nil))

	if w.Code == http.StatusTooManyRequests {
		t.Fatal("under-cap request must not 429")
	}
}

// --- Never-unload (idleSweepOnce) ---

func TestIdleSweepOnce_KeepWarmNeverUnloads(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer server.Close()

	bs := testBackend("vllm", server)
	bs.running = true
	bs.lastRequest = time.Now().Add(-time.Hour) // very idle
	a := testArbiter(bs)
	a.idleTimeout = 0 // keep-warm

	a.idleSweepOnce()

	if !bs.running {
		t.Fatal("IDLE_TIMEOUT=0 must never auto-unload")
	}
}

func TestIdleSweepOnce_RecentRequestNotUnloaded(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer server.Close()

	bs := testBackend("vllm", server)
	bs.running = true
	bs.lastRequest = time.Now() // just used
	a := testArbiter(bs)
	a.idleTimeout = time.Hour

	a.idleSweepOnce()

	if !bs.running {
		t.Fatal("a recently-used backend must not be unloaded")
	}
}

func TestIdleSweepOnce_StaleBackendUnloaded(t *testing.T) {
	bs := &backendState{
		config:      backendConfig{Name: "vllm", ContainerName: "devai-vllm"},
		running:     true,
		lastRequest: time.Now().Add(-time.Hour),
	}
	a := &arbiter{
		backends:     map[string]*backendState{"vllm": bs},
		idleTimeout:  time.Minute,
		podmanClient: &http.Client{Transport: &fakePodman{stateJSON: "{}"}}, // containerStop -> 200
	}

	a.idleSweepOnce()

	if bs.running {
		t.Fatal("a stale backend past idleTimeout must be unloaded")
	}
}

// Regression for the narrowed fail-fast signature: a *running* container
// whose logs merely contain the generic "are not supported" tail (a benign
// startup warning) must NOT be aborted -- only the specific "Model
// architectures" anchor is terminal.
func TestDetectLaunchFailure_BenignArchWarningRunningReturnsNil(t *testing.T) {
	logs := "WARNING some optional features are not supported on this GPU; continuing\nINFO Starting to load model ..."
	a := fakePodmanArbiter(`{"State":{"Status":"running","ExitCode":0}}`, logs)
	if err := a.detectLaunchFailure("devai-vllm"); err != nil {
		t.Fatalf("benign 'are not supported' warning on a running container must not fail-fast, got %v", err)
	}
}

// --- Entrypoint --max-num-seqs / --max-running-requests ---

func TestVLLMEntrypoint_EmitsMaxNumSeqs(t *testing.T) {
	args := vllmEntrypoint("Qwen3.5-9B-NVFP4", launchConfig{MemFraction: 0.9, MaxContext: 32768, MaxNumSeqs: 32})
	if !sliceContains(args, "--max-num-seqs", "32") {
		t.Fatalf("--max-num-seqs 32 missing: %v", args)
	}
}

func TestVLLMEntrypoint_OmitsMaxNumSeqsWhenZero(t *testing.T) {
	args := vllmEntrypoint("Qwen3.5-9B-NVFP4", launchConfig{MemFraction: 0.9, MaxContext: 32768})
	for _, a := range args {
		if a == "--max-num-seqs" {
			t.Fatalf("--max-num-seqs must be omitted when MaxNumSeqs<=0: %v", args)
		}
	}
}

func TestVLLMEntrypoint_RecoveryMaxNumSeqsWinsLast(t *testing.T) {
	args := vllmEntrypoint("Qwen3.6-27B", launchConfig{
		MemFraction: 0.95, MaxContext: 32768, MaxNumSeqs: 32,
		RecoveryFlags: []string{"--max-num-seqs", "4"},
	})
	// Both the default (32) and the recovery override (4) appear; the
	// recovery value must be the LAST occurrence so vLLM's last-wins arg
	// parsing uses 4.
	last := -1
	for i := 0; i+1 < len(args); i++ {
		if args[i] == "--max-num-seqs" {
			last = i
		}
	}
	if last < 0 || args[last+1] != "4" {
		t.Fatalf("recovery --max-num-seqs 4 must be the last occurrence: %v", args)
	}
}

func TestSGLangEntrypoint_EmitsMaxRunningRequests(t *testing.T) {
	args := sglangEntrypoint("Qwen3.5-9B-NVFP4", launchConfig{MemFraction: 0.9, MaxContext: 32768, MaxNumSeqs: 32})
	if !sliceContains(args, "--max-running-requests", "32") {
		t.Fatalf("--max-running-requests 32 missing: %v", args)
	}
}

// --- Hard-error propagation: crash -> 400 (non-retryable), timeout -> 503 ---

func TestWriteLaunchError_CrashIs400NonRetryable(t *testing.T) {
	a := &arbiter{}
	w := httptest.NewRecorder()
	a.writeLaunchError(w, &launchFailure{crashed: true, msg: "vllm engine crashed (exit 1): boom"})

	if w.Code != http.StatusBadRequest {
		t.Fatalf("crash must be 400, got %d", w.Code)
	}
	if got := w.Header().Get("x-should-retry"); got != "false" {
		t.Fatalf("crash must set x-should-retry:false, got %q", got)
	}
	var body map[string]map[string]string
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatalf("bad body: %v", err)
	}
	if body["error"]["type"] != "invalid_request_error" {
		t.Fatalf("want invalid_request_error, got %v", body)
	}
}

func TestWriteLaunchError_WrappedCrashStillDetected(t *testing.T) {
	// waitForHealthy wraps the crash with the backend name via %w; errors.As
	// must still find the launchFailure so the status stays 400.
	a := &arbiter{}
	w := httptest.NewRecorder()
	a.writeLaunchError(w, fmt.Errorf("vllm %w", &launchFailure{crashed: true, msg: "engine crashed (exit 1): boom"}))
	if w.Code != http.StatusBadRequest {
		t.Fatalf("wrapped crash must still be 400, got %d", w.Code)
	}
}

func TestWriteLaunchError_TimeoutIs503Retryable(t *testing.T) {
	a := &arbiter{}
	w := httptest.NewRecorder()
	a.writeLaunchError(w, &launchFailure{crashed: false, msg: "vllm did not become ready within 10m0s"})
	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("timeout must be 503, got %d", w.Code)
	}
	if got := w.Header().Get("x-should-retry"); got != "" {
		t.Fatalf("timeout must not set x-should-retry, got %q", got)
	}
}

func TestWriteLaunchError_UnknownErrorIs503(t *testing.T) {
	a := &arbiter{}
	w := httptest.NewRecorder()
	a.writeLaunchError(w, fmt.Errorf("podman create failed: connection refused"))
	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("non-launchFailure error must default to 503, got %d", w.Code)
	}
}

// --- SGLang terminal signatures (Phase 1, sglang-backend-remediation) ---
//
// Every log fixture below is CAPTURED from /var/cache/devai/logs/
// devai-sglang.log, not invented. That matters: the whole defect class
// here is the router failing to recognise a real corpse, so a test built
// from a guessed banner would prove nothing.

// sglangSchedulerDeathLog is the actual shape of the Ornith-1.0-9B-NVFP4
// failure, verbatim from the log (180 occurrences). Note the `GET /health
// ... 200 OK` on the line BEFORE the assert: the HTTP server keeps
// answering health checks while the scheduler dies behind it, which is
// exactly why /health is not evidence a launch was good.
const sglangSchedulerDeathLog = `[2026-07-25 14:35:07] INFO:     10.89.0.23:55166 - "GET /health HTTP/1.1" 200 OK
/pytorch/aten/src/ATen/native/cuda/TensorCompare.cu:112: _assert_async_cuda_kernel: block: [0,0,0], thread: [0,0,0] Assertion ` + "`" + `probability tensor contains either ` + "`" + `inf` + "`" + `, ` + "`" + `nan` + "`" + `
[2026-07-25 14:35:08] Scheduler hit an exception: Traceback (most recent call last):
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 3616, in run_scheduler_process
    scheduler.run_event_loop()`

// sglangHealthyLaunchLog is a real successful SGLang cold start. The 503s
// are the normal pre-ready state (SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION
// makes /health run a real generation), so a signature set that fires on
// this would abort every healthy launch.
const sglangHealthyLaunchLog = `[2026-07-27 08:54:35] INFO:     Uvicorn running on http://0.0.0.0:11434 (Press CTRL+C to quit)
[2026-07-27 08:54:36] INFO:     127.0.0.1:58294 - "GET /model_info HTTP/1.1" 200 OK
[2026-07-27 08:54:36] INFO:     10.89.0.114:37694 - "GET /health HTTP/1.1" 503 Service Unavailable
[2026-07-27 08:54:42] INFO:     10.89.0.114:37694 - "GET /health HTTP/1.1" 503 Service Unavailable
[2026-07-27 08:54:43] Prefill batch, #new-seq: 1, #new-token: 6, #cached-token: 0, full token usage: 0.00
[2026-07-27 08:54:44] INFO:     127.0.0.1:58310 - "POST /generate HTTP/1.1" 200 OK
[2026-07-27 08:54:44] The server is fired up and ready to roll!`

func TestDetectLaunchFailure_SGLangSchedulerExceptionIsTerminal(t *testing.T) {
	// Container still reports "running" -- SGLang's HTTP server outlives
	// its scheduler, which is the entire problem.
	a := fakePodmanArbiter(`{"State":{"Status":"running","ExitCode":0}}`, sglangSchedulerDeathLog)
	err := a.detectLaunchFailure("devai-sglang")
	if err == nil {
		t.Fatal("a dead SGLang scheduler must be detected as terminal, got nil")
	}
	if !strings.Contains(err.Error(), "Scheduler hit an exception") {
		t.Fatalf("want the scheduler exception attributed, got %q", err)
	}
}

func TestDetectLaunchFailure_SGLangDeviceSideAssertIsTerminal(t *testing.T) {
	logs := "INFO loading weights\nRuntimeError: CUDA error: device-side assert triggered\n"
	a := fakePodmanArbiter(`{"State":{"Status":"running","ExitCode":0}}`, logs)
	if err := a.detectLaunchFailure("devai-sglang"); err == nil {
		t.Fatal("a device-side assert must be terminal, got nil")
	}
}

// The argparse rejection is lowercase (`error:`), which the `Error:`
// anchor did not match before lastErrorLine became case-insensitive. It
// was therefore filed under a generic container-exited tail with no root
// cause -- and, while the container was still running, burned the full
// 600s HEALTH_TIMEOUT_SECONDS first. String captured from the
// Qwen3-Coder-30B-A3B-Instruct-FP4 cell in .sglang-reasoning-cache.json.
func TestDetectLaunchFailure_SGLangArgparseRejectionIsAttributed(t *testing.T) {
	logs := "usage: sglang serve [-h] [--model-path MODEL_PATH]\nsglang serve: error: unrecognized arguments: --enforce-eager\n"
	a := fakePodmanArbiter(`{"State":{"Status":"running","ExitCode":0}}`, logs)
	err := a.detectLaunchFailure("devai-sglang")
	if err == nil {
		t.Fatal("an argparse rejection must fail fast, not wait out HEALTH_TIMEOUT")
	}
	if !strings.Contains(err.Error(), "unrecognized arguments: --enforce-eager") {
		t.Fatalf("want the rejected flag named in the error, got %q", err)
	}
}

func TestDetectLaunchFailure_SGLangHealthyLaunchReturnsNil(t *testing.T) {
	a := fakePodmanArbiter(`{"State":{"Status":"running","ExitCode":0}}`, sglangHealthyLaunchLog)
	if err := a.detectLaunchFailure("devai-sglang"); err != nil {
		t.Fatalf("a captured healthy SGLang launch must not fail-fast, got %v", err)
	}
}

func TestLastErrorLine_IsCaseInsensitive(t *testing.T) {
	logs := "sglang serve: error: unrecognized arguments: --enforce-eager"
	if got := lastErrorLine(logs, failureAnchors); got == "" {
		t.Fatal("lowercase 'error:' must match the Error: anchor")
	}
	// Returned text keeps the ORIGINAL casing, not the lowered copy.
	if got := lastErrorLine(logs, failureAnchors); got != logs {
		t.Fatalf("want the original line verbatim, got %q", got)
	}
}

// --- Circuit-breaker repayment (Phase 1 step 3) ---

// breakerSpent reads the budget under the breaker lock. Repayment now
// runs on the proxy path, so an unlocked read here would be a race.
func breakerSpent(bs *backendState) int {
	bs.breakerMu.Lock()
	defer bs.breakerMu.Unlock()
	return bs.failedLaunches
}

// testBackendCredited is testBackend plus the production credit wiring:
// the proxy repays the circuit breaker when the engine actually answers.
func testBackendCredited(name string, server *httptest.Server) *backendState {
	bs := testBackend(name, server)
	u, _ := url.Parse(server.URL)
	bs.proxy = newSmartProxy(u, false, bs.noteLaunchSucceeded)
	return bs
}

// This is the regression for the Ornith crash loop, and it is the test
// the three existing breaker tests cannot express: they drive the
// counters directly, so they cannot see WHERE repayment is triggered
// from. Against the old code -- repayment right after ensureBackendRunning
// returned -- this fails, because reaching the proxy at all was treated
// as proof the launch was good.
//
// Upstream answers 200 on /health and then drops the connection on the
// real request, which is what a dead SGLang scheduler looks like on the
// wire: 72 container recreates of one model in a single day.
func TestMakeRequestHandler_BreakerNotRepaidWhenEngineDiesServing(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
			return
		}
		hj, ok := w.(http.Hijacker)
		if !ok {
			t.Error("test server ResponseWriter must support hijacking")
			return
		}
		conn, _, err := hj.Hijack()
		if err == nil {
			conn.Close() // no response at all -- the engine is a corpse
		}
	}))
	defer server.Close()

	bs := testBackendCredited("sglang", server)
	bs.running = true
	bs.currentModel = "test-model"
	a := testArbiter(bs)

	bs.noteLaunchAttempt(launchKey("test-model", 0))
	if breakerSpent(bs) != 1 {
		t.Fatalf("setup: want budget spent 1, got %d", breakerSpent(bs))
	}

	w := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"test-model","messages":[]}`))
	a.makeRequestHandler("sglang")(w, req)

	if got := breakerSpent(bs); got != 1 {
		t.Fatalf("budget was repaid despite the engine never returning a response: spent=%d, want 1", got)
	}
}

// Positive control: a healthy engine must still clear the budget, or the
// breaker would refuse working models after maxFailedLaunches recreates.
// Credit lands on the response HEADERS, so a long generation counts too.
func TestMakeRequestHandler_BreakerRepaidOnRealResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"choices":[]}`))
	}))
	defer server.Close()

	bs := testBackendCredited("sglang", server)
	bs.running = true
	bs.currentModel = "test-model"
	a := testArbiter(bs)

	bs.noteLaunchAttempt(launchKey("test-model", 0))

	w := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"test-model","messages":[]}`))
	a.makeRequestHandler("sglang")(w, req)

	if got := breakerSpent(bs); got != 0 {
		t.Fatalf("a real upstream response must repay the budget: spent=%d, want 0", got)
	}
}

// A 5xx from the engine is NOT proof the launch was good. Without this
// an engine that boots and then 500s every request keeps its budget
// refilled forever -- the same unbounded loop in a different costume.
func TestMakeRequestHandler_BreakerNotRepaidOnUpstream5xx(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"error":"internal"}`))
	}))
	defer server.Close()

	bs := testBackendCredited("sglang", server)
	bs.running = true
	bs.currentModel = "test-model"
	a := testArbiter(bs)

	bs.noteLaunchAttempt(launchKey("test-model", 0))

	w := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"test-model","messages":[]}`))
	a.makeRequestHandler("sglang")(w, req)

	if got := breakerSpent(bs); got != 1 {
		t.Fatalf("an upstream 5xx must not repay the budget: spent=%d, want 1", got)
	}
}

// The 500 -> 400 "maximum context length" rewrite in newSmartProxy runs
// BEFORE the credit, so the breaker judges the status the client
// actually sees. An over-long request is a client error against a
// perfectly healthy engine and must not spend the launch budget.
func TestMakeRequestHandler_BreakerRepaidOnRewrittenContextLength400(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"error":"This model's maximum context length is 4096 tokens"}`))
	}))
	defer server.Close()

	bs := testBackendCredited("vllm", server)
	bs.running = true
	bs.currentModel = "test-model"
	a := testArbiter(bs)

	bs.noteLaunchAttempt(launchKey("test-model", 0))

	w := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"test-model","messages":[]}`))
	a.makeRequestHandler("vllm")(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("setup: want the 500->400 rewrite, got %d", w.Code)
	}
	if got := breakerSpent(bs); got != 0 {
		t.Fatalf("a rewritten 400 must repay the budget: spent=%d, want 0", got)
	}
}
