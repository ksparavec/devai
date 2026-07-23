package main

import (
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// podmanStub returns an http.Client shaped like the arbiter's real
// podmanClient (dials a fixed host regardless of the URL's "d" hostname)
// plus a counter of container-create calls -- the number of real
// recreates the lifecycle code decided to perform.
func podmanStub(t *testing.T) (*http.Client, *int64) {
	t.Helper()
	var creates int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, "/containers/create"):
			atomic.AddInt64(&creates, 1)
			w.WriteHeader(http.StatusCreated)
			_, _ = w.Write([]byte(`{"Id":"stub"}`))
		case strings.HasSuffix(r.URL.Path, "/json"):
			_, _ = w.Write([]byte(`{"State":{"Status":"running","ExitCode":0}}`))
		default:
			w.WriteHeader(http.StatusNoContent)
		}
	}))
	t.Cleanup(srv.Close)
	u, _ := url.Parse(srv.URL)
	client := &http.Client{
		Transport: &http.Transport{
			DialContext: func(_ context.Context, _, _ string) (net.Conn, error) {
				return net.Dial("tcp", u.Host)
			},
		},
		Timeout: 5 * time.Second,
	}
	return client, &creates
}

// --- F1: a model-less request must not release the GPU ---

// A bare `GET /` on an idle vLLM/SGLang port reaches ensureBackendRunning
// with modelName=="" (the POST-body block never runs). Before the fix,
// stopOtherBackends ran FIRST: Ollama was drained and every loaded model
// evicted with keep_alive=0, and only then did the call error out with
// 503. Any monitoring probe on the idle port cost a multi-minute cold
// reload of the warm model.
func TestEnsureBackendRunning_ModelLessRequestDoesNotEvictOtherBackend(t *testing.T) {
	var psHits int64
	ollamaSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/ps" {
			atomic.AddInt64(&psHits, 1)
			_, _ = w.Write([]byte(`{"models":[{"name":"warm-model"}]}`))
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer ollamaSrv.Close()
	vllmSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer vllmSrv.Close()

	ollamaBS := testBackend("ollama", ollamaSrv)
	ollamaBS.running = true
	ollamaBS.containerLaunched = true
	ollamaBS.currentModel = "warm-model"
	ollamaBS.currentContext = 32768
	vllmBS := testBackend("vllm", vllmSrv)

	a := testArbiter(ollamaBS, vllmBS)
	a.ollamaURL, _ = url.Parse(ollamaSrv.URL)

	a.mu.Lock()
	err := a.ensureBackendRunning(vllmBS, "", 0, false, nil)
	a.mu.Unlock()

	if err == nil {
		t.Fatal("expected an error for a model-less vllm request")
	}
	if !ollamaBS.running {
		t.Error("ollama must still hold the GPU after a model-less vllm request")
	}
	if ollamaBS.currentModel != "warm-model" {
		t.Errorf("ollama currentModel=%q, want the warm model to be untouched", ollamaBS.currentModel)
	}
	if n := atomic.LoadInt64(&psHits); n != 0 {
		t.Errorf("unloadOllama ran %d times; a model-less request must not touch ollama", n)
	}
}

// --- F2: unloadOllama must not swallow the unload outcome ---

// The keep_alive=0 POST used to discard both the transport error and the
// status code, so a refused unload was indistinguishable from a successful
// one and the next backend launched onto a card ollama still held. The
// unload loop must visit every listed model and return.
func TestUnloadOllama_ContinuesPastFailedUnload(t *testing.T) {
	var generateHits int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/ps":
			_, _ = w.Write([]byte(`{"models":[{"name":"a"},{"name":"b"}]}`))
		case "/api/generate":
			atomic.AddInt64(&generateHits, 1)
			w.WriteHeader(http.StatusInternalServerError)
		default:
			w.WriteHeader(http.StatusOK)
		}
	}))
	defer srv.Close()

	a := &arbiter{}
	a.ollamaURL, _ = url.Parse(srv.URL)
	a.unloadOllama()

	if n := atomic.LoadInt64(&generateHits); n != 2 {
		t.Errorf("unload attempted %d times, want one per listed model (2)", n)
	}
}

// --- F3: a launched-but-unhealthy container still holds the GPU ---

// waitForHealthy timing out (or detectLaunchFailure firing) left
// bs.running=false while the container was alive and still pulling
// weights onto the card. stopOtherBackends skipped it, and the next
// backend launched against a GPU that was not free.
func TestStopOtherBackends_StopsLaunchedButNeverHealthyBackend(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer srv.Close()

	stale := testBackend("vllm", srv)
	stale.config.ContainerName = "devai-vllm"
	stale.running = false          // health wait timed out
	stale.containerLaunched = true // ... but `podman start` had succeeded
	target := testBackend("sglang", srv)

	a := testArbiter(stale, target)
	client, _ := podmanStub(t)
	a.podmanClient = client

	a.mu.Lock()
	a.stopOtherBackends("sglang")
	a.mu.Unlock()

	if stale.containerLaunched {
		t.Error("a launched-but-unhealthy backend must be stopped and its launch flag cleared")
	}
}

// The idle sweeper carried the OLD `!bs.running` guard long after
// stopOtherBackends learned about containerLaunched, so the one state the
// flag exists for -- container alive, health wait timed out, running=false
// -- was the one state the sweeper could never reclaim. The card stayed
// occupied by a dead engine until an unrelated model switch came along.
func TestIdleSweepOnce_ReclaimsLaunchedButNeverHealthyBackend(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer srv.Close()

	bs := testBackend("vllm", srv)
	bs.config.ContainerName = "devai-vllm"
	bs.running = false          // health wait timed out
	bs.containerLaunched = true // ... but `podman start` had succeeded
	bs.currentModel = "test-model"
	bs.lastRequest = time.Now().Add(-time.Hour)

	a := testArbiter(bs)
	a.idleTimeout = time.Minute
	client, _ := podmanStub(t)
	a.podmanClient = client

	a.idleSweepOnce()

	if bs.containerLaunched {
		t.Error("idle sweeper must stop a launched-but-unhealthy backend and clear its launch flag")
	}
	if bs.currentModel != "" {
		t.Errorf("currentModel=%q, want it cleared after the idle stop", bs.currentModel)
	}
}

// ... but a backend that was never launched at all still has nothing to
// stop. A nil podmanClient would panic if the sweeper tried.
func TestIdleSweepOnce_SkipsBackendThatWasNeverLaunched(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer srv.Close()

	bs := testBackend("vllm", srv)
	bs.lastRequest = time.Now().Add(-time.Hour)
	a := testArbiter(bs)
	a.idleTimeout = time.Minute
	a.podmanClient = nil

	a.idleSweepOnce()
}

func TestStopOtherBackends_SkipsBackendThatWasNeverLaunched(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer srv.Close()

	idle := testBackend("vllm", srv)
	target := testBackend("sglang", srv)
	a := testArbiter(idle, target)
	// A nil podmanClient would panic if stopOtherBackends tried to stop
	// the idle backend -- which is exactly the assertion.
	a.podmanClient = nil

	a.mu.Lock()
	a.stopOtherBackends("sglang")
	a.mu.Unlock()
}

// --- F4: drain must ignore requests parked on the arbiter mutex ---

// drainBackend runs with a.mu held, so requests still blocked on that
// mutex can never finish while it waits. Counting them made every switch
// under load stall for the full DRAIN_TIMEOUT.
func TestDrainBackend_IgnoresRequestsBlockedOnArbiterMutex(t *testing.T) {
	bs := &backendState{config: backendConfig{Name: "test"}}
	atomic.StoreInt64(&bs.activeReqs, 8) // admitted, but all parked on a.mu
	atomic.StoreInt64(&bs.upstreamReqs, 0)
	a := &arbiter{drainTimeout: 2 * time.Second}

	start := time.Now()
	a.drainBackend(bs)
	if elapsed := time.Since(start); elapsed > 100*time.Millisecond {
		t.Errorf("drain waited %v for requests that cannot progress", elapsed)
	}
}

// --- F5/F6: per-backend context keying, and recording the LAUNCHED ctx ---

func TestResolveLaunchContext_IsPerBackend(t *testing.T) {
	a := &arbiter{
		modelSizes: map[string]map[string]float64{
			"vllm":   {"shared-model": 7.4},
			"sglang": {"shared-model": 7.4},
		},
		modelContexts: map[string]map[string]int{
			"vllm":   {"shared-model": 131072},
			"sglang": {"shared-model": 32768},
		},
		modelProbedMaxCtx: map[string]map[string]int{
			"vllm":   {"shared-model": 131072},
			"sglang": {"shared-model": 32768},
		},
		totalVRAMGB:   24.0,
		maxContextLen: 262144,
	}
	if got := a.resolveLaunchContext("vllm", "shared-model", 0); got != 131072 {
		t.Errorf("vllm bare ctx = %d, want 131072 (the vllm row)", got)
	}
	if got := a.resolveLaunchContext("sglang", "shared-model", 0); got != 32768 {
		t.Errorf("sglang bare ctx = %d, want 32768 (the sglang row)", got)
	}
}

func TestResolveLaunchContext_ClampsOverrideToProbedCeiling(t *testing.T) {
	a := &arbiter{
		modelSizes:        map[string]map[string]float64{"vllm": {"m": 7.4}},
		modelContexts:     map[string]map[string]int{"vllm": {"m": 131072}},
		modelProbedMaxCtx: map[string]map[string]int{"vllm": {"m": 32768}},
		totalVRAMGB:       24.0,
		maxContextLen:     262144,
	}
	if got := a.resolveLaunchContext("vllm", "m", 131072); got != 32768 {
		t.Errorf("resolveLaunchContext = %d, want the 32768 probe ceiling", got)
	}
}

// A bare `<name>` and a pinned `<name>@<ctx>` that resolve to the SAME
// effective context must not recreate each other. Before the fix
// currentContext held the REQUESTED value, so the two alternated forever.
func TestEnsureBackendRunning_NoRecreateWhenResolvedContextMatches(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer srv.Close()

	bs := testBackend("vllm", srv)
	bs.running = true
	bs.currentModel = "test-model"
	bs.currentContext = 32768 // what the previous launch actually settled on

	a := testArbiter(bs)
	a.modelContexts = map[string]map[string]int{"vllm": {"test-model": 131072}}
	a.modelProbedMaxCtx = map[string]map[string]int{"vllm": {"test-model": 32768}}
	client, creates := podmanStub(t)
	a.podmanClient = client

	// Bare name: desiredCtx is the declared 131072 cap, which clamps to
	// the 32768 probe ceiling -- i.e. exactly what is already loaded.
	a.mu.Lock()
	err := a.ensureBackendRunning(bs, "test-model", 131072, false, nil)
	a.mu.Unlock()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if n := atomic.LoadInt64(creates); n != 0 {
		t.Errorf("%d recreates; a bare name resolving to the loaded ctx must not recreate", n)
	}
}

// --- F7 + F10: setNumCtx, its path gate, and the client-value clamp ---

func TestSetNumCtx_InjectsWhenAbsent(t *testing.T) {
	out := setNumCtx([]byte(`{"model":"m"}`), 32768, false)
	if got, ok := clientNumCtx(out); !ok || got != 32768 {
		t.Errorf("setNumCtx produced %s", out)
	}
}

func TestSetNumCtx_SoftModeLeavesClientValue(t *testing.T) {
	out := setNumCtx([]byte(`{"model":"m","options":{"num_ctx":4096}}`), 32768, false)
	if got, _ := clientNumCtx(out); got != 4096 {
		t.Errorf("force=false must not overwrite a client value, got %s", out)
	}
}

func TestSetNumCtx_ForceOverwritesClientValue(t *testing.T) {
	out := setNumCtx([]byte(`{"model":"m","options":{"num_ctx":131072}}`), 32768, true)
	if got, _ := clientNumCtx(out); got != 32768 {
		t.Errorf("force=true must overwrite, got %s", out)
	}
}

func TestSetNumCtx_PreservesOtherOptions(t *testing.T) {
	out := setNumCtx([]byte(`{"model":"m","options":{"temperature":0.7}}`), 8192, true)
	var doc struct {
		Options struct {
			Temperature float64 `json:"temperature"`
			NumCtx      int     `json:"num_ctx"`
		} `json:"options"`
	}
	if err := json.Unmarshal(out, &doc); err != nil {
		t.Fatalf("bad JSON out: %v", err)
	}
	if doc.Options.Temperature != 0.7 || doc.Options.NumCtx != 8192 {
		t.Errorf("setNumCtx clobbered sibling options: %s", out)
	}
}

func TestSetNumCtx_NoopOnNonPositiveOrBadJSON(t *testing.T) {
	if out := setNumCtx([]byte(`{"model":"m"}`), 0, true); string(out) != `{"model":"m"}` {
		t.Errorf("num_ctx<=0 must be a no-op, got %s", out)
	}
	if out := setNumCtx([]byte(`not json`), 4096, true); string(out) != "not json" {
		t.Errorf("bad JSON must pass through unchanged, got %s", out)
	}
}

func TestClientNumCtx(t *testing.T) {
	cases := []struct {
		body string
		want int
		ok   bool
	}{
		{`{"options":{"num_ctx":4096}}`, 4096, true},
		{`{"options":{}}`, 0, false},
		{`{}`, 0, false},
		{`{"options":{"num_ctx":0}}`, 0, false},
		{`{"options":{"num_ctx":"big"}}`, 0, false},
		{`nope`, 0, false},
		// A2/A1: any NUMERIC shape must be recognised. A *int field
		// errored on these two, reported (0,false), and let the value
		// through the clamp at the makeRequestHandler call site.
		{`{"options":{"num_ctx":131072.0}}`, 131072, true},
		{`{"options":{"num_ctx":1.31072e5}}`, 131072, true},
		// Fractional values truncate toward zero; below 1 that is not a
		// usable override.
		{`{"options":{"num_ctx":4096.9}}`, 4096, true},
		{`{"options":{"num_ctx":0.5}}`, 0, false},
		{`{"options":{"num_ctx":-2.5}}`, 0, false},
		// Out of int range: saturate rather than rely on Go's
		// implementation-defined float->int conversion.
		{`{"options":{"num_ctx":1e30}}`, maxClientNumCtx, true},
	}
	for _, c := range cases {
		got, ok := clientNumCtx([]byte(c.body))
		if got != c.want || ok != c.ok {
			t.Errorf("clientNumCtx(%s) = (%d,%v), want (%d,%v)", c.body, got, ok, c.want, c.ok)
		}
	}
}

// ollamaCtxHandlerArbiter wires an already-running Ollama backend serving
// at ctx 32768, and returns the handler plus a pointer to the body the
// upstream actually received.
//
// The registered cap is deliberately EQUAL to the probed ceiling: that is
// the configuration in which the old clamp did nothing (it only fired when
// the ROUTER's own value exceeded the ceiling), so a client-supplied
// num_ctx sailed through unclamped.
func ollamaCtxHandlerArbiter(t *testing.T) (http.HandlerFunc, *[]byte) {
	t.Helper()
	var received []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		received = b
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(srv.Close)

	bs := testBackend("ollama", srv)
	bs.running = true
	bs.containerLaunched = true
	bs.currentModel = "test-model"
	bs.currentContext = 32768
	a := testArbiter(bs)
	a.ollamaURL, _ = url.Parse(srv.URL)
	a.modelProbedMaxCtx = map[string]map[string]int{"ollama": {"test-model": 32768}}
	a.modelContexts = map[string]map[string]int{"ollama": {"test-model": 32768}}
	return a.makeRequestHandler("ollama"), &received
}

// The docs mark this gate as must-not-change: options.num_ctx is
// meaningful ONLY on Ollama's native surfaces. Ollama discards it on
// /v1/chat/completions and /v1/messages.
func TestRequestHandler_NumCtxOnlyOnOllamaNativePaths(t *testing.T) {
	for _, tc := range []struct {
		path    string
		wantSet bool
	}{
		{"/api/chat", true},
		{"/api/generate", true},
		{"/v1/chat/completions", false},
		{"/v1/messages", false},
	} {
		handler, received := ollamaCtxHandlerArbiter(t)
		req := httptest.NewRequest("POST", tc.path,
			strings.NewReader(`{"model":"test-model"}`))
		handler(httptest.NewRecorder(), req)
		_, ok := clientNumCtx(*received)
		if ok != tc.wantSet {
			t.Errorf("%s: num_ctx injected=%v, want %v (body %s)",
				tc.path, ok, tc.wantSet, *received)
		}
	}
}

// F7: a client asking for more context than the probe proved fits must be
// clamped down. Previously the clamp guarded only the router's own value.
func TestRequestHandler_ClampsClientSuppliedNumCtx(t *testing.T) {
	handler, received := ollamaCtxHandlerArbiter(t)
	req := httptest.NewRequest("POST", "/api/chat",
		strings.NewReader(`{"model":"test-model","options":{"num_ctx":131072}}`))
	handler(httptest.NewRecorder(), req)
	got, ok := clientNumCtx(*received)
	if !ok || got != 32768 {
		t.Errorf("client num_ctx=131072 forwarded as %d (ok=%v); want clamp to 32768. body=%s",
			got, ok, *received)
	}
}

func TestRequestHandler_LeavesSmallerClientNumCtxAlone(t *testing.T) {
	handler, received := ollamaCtxHandlerArbiter(t)
	req := httptest.NewRequest("POST", "/api/chat",
		strings.NewReader(`{"model":"test-model","options":{"num_ctx":4096}}`))
	handler(httptest.NewRecorder(), req)
	if got, _ := clientNumCtx(*received); got != 4096 {
		t.Errorf("a client value below the ceiling must be left alone, got %d", got)
	}
}

// A1: the clamp must not be shape-dependent. `131072` was clamped while the
// numerically identical `131072.0` / `1.31072e5` sailed straight through to
// Ollama, which then loaded the runner at a context the probe never proved
// fits.
func TestRequestHandler_ClampsNonIntegerClientNumCtx(t *testing.T) {
	for _, literal := range []string{"131072.0", "1.31072e5"} {
		handler, received := ollamaCtxHandlerArbiter(t)
		req := httptest.NewRequest("POST", "/api/chat",
			strings.NewReader(`{"model":"test-model","options":{"num_ctx":`+literal+`}}`))
		handler(httptest.NewRecorder(), req)
		got, ok := clientNumCtx(*received)
		if !ok || got != 32768 {
			t.Errorf("num_ctx=%s forwarded as %d (ok=%v); want clamp to 32768. body=%s",
				literal, got, ok, *received)
		}
	}
}

// --- F8: MAX_CONCURRENT_REQUESTS=0 means unlimited ---

func TestEnvIntAllowZero(t *testing.T) {
	const key = "DEVAI_TEST_ALLOW_ZERO"
	cases := []struct {
		set  bool
		val  string
		want int
	}{
		{false, "", 32},
		{true, "0", 0},
		{true, "7", 7},
		{true, "-1", 32},
		{true, "abc", 32},
		{true, " 5 ", 5},
	}
	for _, c := range cases {
		os.Unsetenv(key)
		if c.set {
			os.Setenv(key, c.val)
		}
		if got := envIntAllowZero(key, 32); got != c.want {
			t.Errorf("envIntAllowZero(%q set=%v) = %d, want %d", c.val, c.set, got, c.want)
		}
	}
	os.Unsetenv(key)
}

// Zero must actually disable the limiter downstream, not admit nothing.
func TestMakeRequestHandler_ZeroMaxConcurrentIsUnlimited(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer srv.Close()

	bs := testBackend("ollama", srv)
	bs.running = true
	bs.currentModel = "test-model"
	bs.currentContext = 32768
	a := testArbiter(bs)
	a.ollamaURL, _ = url.Parse(srv.URL)
	a.maxConcurrent = 0
	atomic.StoreInt64(&bs.activeReqs, 500)

	w := httptest.NewRecorder()
	a.makeRequestHandler("ollama")(w, httptest.NewRequest("POST", "/api/chat",
		strings.NewReader(`{"model":"test-model"}`)))
	if w.Code == http.StatusTooManyRequests {
		t.Error("maxConcurrent=0 must not rate-limit")
	}
}

// The launch flag is derived from maxConcurrent, and 0 must mean "emit
// nothing" so the engine keeps its own default.
func TestEntrypoints_OmitBatchFlagWhenMaxNumSeqsIsZero(t *testing.T) {
	lc := launchConfig{MemFraction: 0.9, MaxContext: 32768, MaxNumSeqs: 0}
	for name, args := range map[string][]string{
		"vllm":   vllmEntrypoint("m", lc),
		"sglang": sglangEntrypoint("m", lc),
	} {
		joined := strings.Join(args, " ")
		if strings.Contains(joined, "--max-num-seqs") ||
			strings.Contains(joined, "--max-running-requests") {
			t.Errorf("%s emitted a batch cap for MaxNumSeqs=0: %s", name, joined)
		}
	}
}

// --- F9: mixed-KV Ollama tier policy ---

// ollamaTierArbiter builds a fully stubbed Ollama backend whose probed
// ceiling is 131072 and returns the arbiter plus the recreate counter.
func ollamaTierArbiter(t *testing.T) (*arbiter, *backendState, *int64) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/ps" {
			_, _ = w.Write([]byte(`{"models":[]}`))
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(srv.Close)

	bs := testBackend("ollama", srv)
	bs.config.ContainerName = "devai-ollama"
	bs.config.Image = "docker.io/ollama/ollama:latest"
	bs.config.ModelsDir = "/var/cache/devai/ollama"
	bs.config.Entrypoint = ollamaEntrypoint
	bs.config.DynamicEnv = ollamaDynamicEnv

	a := testArbiter(bs)
	a.healthTimeout = 5 * time.Second
	a.ollamaURL, _ = url.Parse(srv.URL)
	a.modelProbedMaxCtx = map[string]map[string]int{"ollama": {"mixed-kv": 131072}}
	a.modelSizes = map[string]map[string]float64{"ollama": {"mixed-kv": 7.4}}
	a.modelContexts = map[string]map[string]int{"ollama": {"mixed-kv": 131072}}
	client, creates := podmanStub(t)
	a.podmanClient = client
	return a, bs, creates
}

// An explicit pin is honoured exactly -- the bench harness relies on it to
// label rows -- so a different loaded tier IS recreated.
func TestEnsureOllamaRunning_ExplicitPinSwitchesTier(t *testing.T) {
	a, bs, creates := ollamaTierArbiter(t)
	bs.running = true
	bs.containerLaunched = true
	bs.currentModel = "mixed-kv"
	bs.currentContext = 131072

	a.mu.Lock()
	err := a.ensureOllamaRunning(bs, "mixed-kv", 32768, true)
	a.mu.Unlock()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if n := atomic.LoadInt64(creates); n != 1 {
		t.Errorf("pinned tier switch produced %d recreates, want 1", n)
	}
	if bs.currentContext != 32768 {
		t.Errorf("currentContext=%d, want the pinned 32768", bs.currentContext)
	}
}

// A bare name is served FROM the loaded tier -- no recreate, even though
// the default tier is larger.
func TestEnsureOllamaRunning_BareNameServesLoadedTier(t *testing.T) {
	a, bs, creates := ollamaTierArbiter(t)
	bs.running = true
	bs.containerLaunched = true
	bs.currentModel = "mixed-kv"
	bs.currentContext = 32768 // a previous `@32768` pin

	a.mu.Lock()
	err := a.ensureOllamaRunning(bs, "mixed-kv", 131072, false)
	a.mu.Unlock()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if n := atomic.LoadInt64(creates); n != 0 {
		t.Errorf("bare name produced %d recreates, want 0", n)
	}
	if bs.currentContext != 32768 {
		t.Errorf("currentContext=%d, want the loaded 32768 tier retained", bs.currentContext)
	}
}

// A bare name with nothing loaded picks the default tier.
func TestEnsureOllamaRunning_BareNameColdPicksDefaultTier(t *testing.T) {
	a, bs, creates := ollamaTierArbiter(t)

	a.mu.Lock()
	err := a.ensureOllamaRunning(bs, "mixed-kv", 131072, false)
	a.mu.Unlock()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if n := atomic.LoadInt64(creates); n != 1 {
		t.Errorf("cold start produced %d recreates, want 1", n)
	}
	if bs.currentContext != 131072 {
		t.Errorf("currentContext=%d, want the 131072 default tier", bs.currentContext)
	}
}

// The alternating-client sequence that used to recreate on EVERY request:
// one client pins @32768, another uses the bare name. Exactly one recreate
// must happen for the whole sequence.
func TestEnsureOllamaRunning_AlternatingPinnedAndBareRecreatesOnce(t *testing.T) {
	a, bs, creates := ollamaTierArbiter(t)

	seq := []struct {
		ctx    int
		pinned bool
	}{
		{32768, true},   // cold start on the pinned tier -> 1 recreate
		{131072, false}, // bare -> served from the loaded 32K tier
		{32768, true},   // pinned again, same tier -> no-op
		{131072, false}, // bare again -> still the loaded tier
	}
	for i, s := range seq {
		a.mu.Lock()
		err := a.ensureOllamaRunning(bs, "mixed-kv", s.ctx, s.pinned)
		a.mu.Unlock()
		if err != nil {
			t.Fatalf("step %d: unexpected error: %v", i, err)
		}
	}
	if n := atomic.LoadInt64(creates); n != 1 {
		t.Errorf("alternating pinned/bare produced %d recreates, want exactly 1", n)
	}
	if bs.currentContext != 32768 {
		t.Errorf("currentContext=%d, want the pinned 32768 tier held throughout", bs.currentContext)
	}
}

// --- A4: the probed flash_attention stamp must reach the launch env ---

// The prober records the OLLAMA_FLASH_ATTENTION it measured the cell under,
// but the router derived the setting from the KV dtype alone. A cell probed
// with flash attention ON under the DEFAULT f16 dtype was therefore SERVED
// with it off -- a different environment from the one the fit was measured
// in, which is the whole reason the stamp was added.
func TestOllamaDynamicEnv_HonoursProbedFlashAttention(t *testing.T) {
	for _, c := range []struct {
		name      string
		kv        string
		flash     *bool
		wantFlash string // "" = the key must be absent
		wantKV    string
	}{
		{"stamped on under default f16", "f16", boolPtr(true), "1", ""},
		{"stamped off under default f16", "f16", boolPtr(false), "", ""},
		// Absent stamp (pre-stamp cells): keep the historical
		// dtype-derived value, byte-identical to previous launches.
		{"absent under default f16", "f16", nil, "", ""},
		{"absent under quantized kv", "q8_0", nil, "1", "q8_0"},
		{"absent, no dtype recorded", "", nil, "", ""},
		// An explicit stamp still wins over the dtype-derived default --
		// serve time reproduces what was measured, not what we assume.
		{"stamped off under quantized kv", "q8_0", boolPtr(false), "", "q8_0"},
	} {
		env := ollamaDynamicEnv(launchConfig{
			MaxContext: 32768, KVCacheType: c.kv, FlashAttention: c.flash,
		})
		if got := env["OLLAMA_FLASH_ATTENTION"]; got != c.wantFlash {
			t.Errorf("%s: OLLAMA_FLASH_ATTENTION=%q, want %q", c.name, got, c.wantFlash)
		}
		if got := env["OLLAMA_KV_CACHE_TYPE"]; got != c.wantKV {
			t.Errorf("%s: OLLAMA_KV_CACHE_TYPE=%q, want %q", c.name, got, c.wantKV)
		}
	}
}

// The stamp and the dtype must come from the SAME probe cell, so
// resolveFlashAttention uses resolveKVCacheType's smallest-covering-tier
// rule. Tiers with no stamp are invisible to it (nil = fall back).
func TestResolveFlashAttention_UsesSmallestCoveringTier(t *testing.T) {
	m := map[int]*bool{32768: boolPtr(false), 131072: boolPtr(true)}
	if got := resolveFlashAttention(m, 8192); got == nil || *got {
		t.Errorf("ctx 8192 must resolve through the 32768 tier, got %v", got)
	}
	if got := resolveFlashAttention(m, 65536); got == nil || !*got {
		t.Errorf("ctx 65536 must resolve through the 131072 tier, got %v", got)
	}
	if got := resolveFlashAttention(m, 262144); got != nil {
		t.Errorf("no covering tier must resolve to nil, got %v", *got)
	}
	if got := resolveFlashAttention(map[int]*bool{32768: nil}, 8192); got != nil {
		t.Errorf("an unstamped tier must resolve to nil, got %v", *got)
	}
}

// End to end from the cache file: the stamp survives synthesizeFromCache,
// and an absent one stays absent.
func TestSynthesizeFromCache_CarriesFlashAttentionStamp(t *testing.T) {
	raw := []byte(`{
	  "digest-stamped": {
	    "schema_version": 3, "digest": "digest-stamped", "aliases": ["stamped:latest"],
	    "max_context": 32768, "capability": "none",
	    "probes": {"24": {"32768": {"ctx": 32768, "vram_gb": 24,
	      "actual_total_gb": 12.0, "fully_on_gpu": true,
	      "kv_cache_type": "f16", "flash_attention": true}}}
	  },
	  "digest-legacy": {
	    "schema_version": 3, "digest": "digest-legacy", "aliases": ["legacy:latest"],
	    "max_context": 32768, "capability": "none",
	    "probes": {"24": {"32768": {"ctx": 32768, "vram_gb": 24,
	      "actual_total_gb": 12.0, "fully_on_gpu": true}}}
	  }
	}`)
	var cache map[string]*cacheEntry
	if err := json.Unmarshal(raw, &cache); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	for _, m := range synthesizeFromCache(cache, 24, 262144) {
		got := resolveFlashAttention(m.FlashByCtx, 32768)
		switch m.Name {
		case "stamped:latest":
			if got == nil || !*got {
				t.Errorf("stamped row lost flash_attention=true, got %v", got)
			}
		case "legacy:latest":
			if got != nil {
				t.Errorf("pre-stamp row must stay unstamped, got %v", *got)
			}
		}
	}
}

// --- A6: the Ollama path must compare LAUNCHED against LAUNCHED ---

// ensureOllamaRunning compared bs.currentContext (a LAUNCHED ctx) against
// the REQUESTED probedCtx -- the same defect the vLLM/SGLang path fixed
// with resolveLaunchContext. Whenever the launch clamps below the request,
// ctxChanged stays true forever and every pinned request recreates the
// container. Reproduced here with MAX_CONTEXT_LEN below the pinned request,
// which is the clamp the launch applies and the comparison used to ignore.
func TestEnsureOllamaRunning_ComparesResolvedNotRequestedContext(t *testing.T) {
	a, bs, creates := ollamaTierArbiter(t)
	// The pinned 131072 clamps to the operator's 65536 MAX_CONTEXT_LEN.
	a.maxContextLen = 65536
	bs.running = true
	bs.containerLaunched = true
	bs.currentModel = "mixed-kv"
	bs.currentContext = 65536 // what the previous launch actually settled on

	for i := 0; i < 3; i++ {
		a.mu.Lock()
		err := a.ensureOllamaRunning(bs, "mixed-kv", 131072, true)
		a.mu.Unlock()
		if err != nil {
			t.Fatalf("request %d: unexpected error: %v", i, err)
		}
	}
	if n := atomic.LoadInt64(creates); n != 0 {
		t.Errorf("%d recreates; a pin resolving to the loaded ctx must not recreate", n)
	}
}

// --- F11: one malformed disable_verified must not empty the model list ---

func TestTristateBool_AcceptsBoolStringAndNull(t *testing.T) {
	cases := []struct {
		in   string
		want *bool
	}{
		{`true`, boolPtr(true)},
		{`false`, boolPtr(false)},
		{`null`, nil},
		{`"true"`, boolPtr(true)},
		{`"FALSE"`, boolPtr(false)},
		{`"error"`, nil},
		{`123`, nil},
		{`{"a":1}`, nil},
	}
	for _, c := range cases {
		var tb tristateBool
		if err := json.Unmarshal([]byte(c.in), &tb); err != nil {
			t.Fatalf("unmarshal %s returned an error: %v (it must never fail)", c.in, err)
		}
		got := tb.Value()
		switch {
		case c.want == nil && got != nil:
			t.Errorf("%s -> %v, want unknown", c.in, *got)
		case c.want != nil && (got == nil || *got != *c.want):
			t.Errorf("%s -> %v, want %v", c.in, got, *c.want)
		}
	}
}

func boolPtr(b bool) *bool { return &b }

// The whole file is unmarshalled in one call, so a string sentinel on ONE
// model used to abort the parse and leave cfg.Models EMPTY -- every Ollama
// model vanished from the router and the picker.
func TestOllamaCache_StringDisableVerifiedDoesNotKillTheFile(t *testing.T) {
	raw := []byte(`{
	  "digest-bad": {
	    "schema_version": 3, "digest": "digest-bad", "aliases": ["bad:latest"],
	    "max_context": 32768, "capability": "structured",
	    "disable_verified": "error",
	    "probes": {"24": {"32768": {"ctx": 32768, "vram_gb": 24,
	      "actual_total_gb": 12.0, "fully_on_gpu": true}}}
	  },
	  "digest-ok": {
	    "schema_version": 3, "digest": "digest-ok", "aliases": ["ok:latest"],
	    "max_context": 32768, "capability": "structured",
	    "disable_verified": true,
	    "probes": {"24": {"32768": {"ctx": 32768, "vram_gb": 24,
	      "actual_total_gb": 12.0, "fully_on_gpu": true}}}
	  }
	}`)
	var cache map[string]*cacheEntry
	if err := json.Unmarshal(raw, &cache); err != nil {
		t.Fatalf("one malformed field must not abort the parse: %v", err)
	}
	models := synthesizeFromCache(cache, 24, 262144)
	if len(models) != 2 {
		t.Fatalf("got %d models, want both rows to survive", len(models))
	}
	for _, m := range models {
		switch m.Name {
		case "bad:latest":
			if m.Reasoning.DisableVerified != nil {
				t.Errorf("bad row must degrade to unknown, got %v", *m.Reasoning.DisableVerified)
			}
		case "ok:latest":
			if m.Reasoning.DisableVerified == nil || !*m.Reasoning.DisableVerified {
				t.Errorf("good row lost its disable_verified=true")
			}
		default:
			t.Errorf("unexpected model %q", m.Name)
		}
	}
}

// --- F12: the recovery registry is backend-scoped ---

func backendsPtr(v ...string) *[]string { return &v }

func TestRecoveryRegistry_BackendScoping(t *testing.T) {
	r := &recoveryRegistry{Models: map[string]recoveryEntry{
		"vllm-only": {
			Flags:    []string{"--language-model-only"},
			Backends: backendsPtr("vllm"),
		},
		"everywhere": {
			Flags: []string{"--trust-remote-code"},
		},
	}}
	if _, ok := r.Lookup("vllm", "vllm-only"); !ok {
		t.Error("vllm-scoped entry must apply on vllm")
	}
	if _, ok := r.Lookup("sglang", "vllm-only"); ok {
		t.Error("vllm-scoped entry must NOT apply on sglang")
	}
	for _, b := range []string{"vllm", "sglang", "ollama"} {
		if _, ok := r.Lookup(b, "everywhere"); !ok {
			t.Errorf("an entry with no backends list must apply on %s", b)
		}
	}
	if _, ok := r.Lookup("vllm", "absent"); ok {
		t.Error("unknown model must not match")
	}
	var nilReg *recoveryRegistry
	if _, ok := nilReg.Lookup("vllm", "vllm-only"); ok {
		t.Error("nil registry must not match")
	}
}

func TestLoadRecoveryRegistry_ParsesBackendsKey(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/recovery-flags.json"
	if err := os.WriteFile(path, []byte(`{
	  "_comment": ["note"],
	  "models": {
	    "m": {"backends": ["vllm"], "engine_flags": ["--enforce-eager"]}
	  }
	}`), 0o600); err != nil {
		t.Fatal(err)
	}
	r := loadRecoveryRegistry(path)
	e, ok := r.Lookup("vllm", "m")
	if !ok || len(e.Flags) != 1 || e.Flags[0] != "--enforce-eager" {
		t.Fatalf("vllm lookup = (%+v, %v)", e, ok)
	}
	if _, ok := r.Lookup("sglang", "m"); ok {
		t.Error("sglang must be filtered out by the backends key")
	}
}

// --- A3: the `backends` key contract (C2), shared with the Python probers ---

// writeRecoveryRegistryFile drops a registry file into a temp dir and loads it.
func writeRecoveryRegistryFile(t *testing.T, body string) *recoveryRegistry {
	t.Helper()
	path := t.TempDir() + "/recovery-flags.json"
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return loadRecoveryRegistry(path)
}

// The four cases of contract C2, which the Go and Python readers must agree
// on exactly. Before the fix Go collapsed ABSENT and [] onto "applies to
// everything" (len(Backends)==0), while Python already read [] as "applies
// to nothing" -- so an operator disabling an entry with [] still got the
// flags applied at serve time but not at probe time.
func TestRecoveryRegistry_BackendsKeyContract(t *testing.T) {
	r := writeRecoveryRegistryFile(t, `{"models":{
	  "absent":  {"engine_flags": ["--a"]},
	  "null":    {"backends": null,     "engine_flags": ["--n"]},
	  "empty":   {"backends": [],       "engine_flags": ["--e"]},
	  "scoped":  {"backends": ["vllm"], "engine_flags": ["--s"]}
	}}`)
	for _, c := range []struct {
		model, backend string
		want           bool
	}{
		{"absent", "vllm", true}, {"absent", "sglang", true}, {"absent", "ollama", true},
		{"null", "vllm", true}, {"null", "sglang", true},
		{"empty", "vllm", false}, {"empty", "sglang", false}, {"empty", "ollama", false},
		{"scoped", "vllm", true}, {"scoped", "sglang", false},
	} {
		if _, ok := r.Lookup(c.backend, c.model); ok != c.want {
			t.Errorf("Lookup(%s, %s) applied=%v, want %v", c.backend, c.model, ok, c.want)
		}
	}
}

// A non-list `backends` value is a typo, not a scoping instruction: warn and
// treat the key as absent. Critically, it must NOT take the rest of the file
// down with it -- the whole-file decode used to fail on the first such value
// and return an EMPTY registry, silently stripping the OOM-rescue flags from
// every model in the file.
func TestLoadRecoveryRegistry_NonListBackendsKeepsWholeRegistry(t *testing.T) {
	r := writeRecoveryRegistryFile(t, `{"models":{
	  "typo":  {"backends": "vllm",     "engine_flags": ["--typo"]},
	  "good":  {"backends": ["vllm"],   "engine_flags": ["--good"]},
	  "plain": {"engine_flags": ["--plain"]}
	}}`)
	if len(r.Models) != 3 {
		t.Fatalf("registry has %d entries, want all 3 to survive one typo", len(r.Models))
	}
	// The typo entry keeps its flags and degrades to "applies everywhere".
	for _, backend := range []string{"vllm", "sglang"} {
		e, ok := r.Lookup(backend, "typo")
		if !ok {
			t.Errorf("a non-list backends value must be treated as absent (%s)", backend)
			continue
		}
		if len(e.Flags) != 1 || e.Flags[0] != "--typo" {
			t.Errorf("typo entry lost its flags: %+v", e.Flags)
		}
	}
	if _, ok := r.Lookup("sglang", "good"); ok {
		t.Error("the well-formed neighbour lost its backend scoping")
	}
}

// A structurally broken entry is dropped on its own, never taking siblings
// with it.
func TestLoadRecoveryRegistry_BrokenEntryDoesNotDropSiblings(t *testing.T) {
	r := writeRecoveryRegistryFile(t, `{"models":{
	  "broken": ["not", "an", "object"],
	  "good":   {"engine_flags": ["--good"]}
	}}`)
	if _, ok := r.Lookup("vllm", "broken"); ok {
		t.Error("a structurally broken entry must be dropped")
	}
	if _, ok := r.Lookup("vllm", "good"); !ok {
		t.Error("a broken sibling must not remove a valid entry")
	}
}
