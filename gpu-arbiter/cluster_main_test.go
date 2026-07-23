package main

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// newInboundWorker returns a Worker with just enough state for the
// inbound handler (no head, no HTTP client -- Start is never called).
func newInboundWorker() *Worker {
	w := &Worker{Config: WorkerConfig{WorkerName: "w-test"}, State: &WorkerState{}}
	w.State.SetHealthStatus("ready")
	return w
}

// arbiterWithBackends builds a bare arbiter carrying only the fields
// the inbound dispatch reads: the backend map with its model names and
// current (model, ctx).
func arbiterWithBackends(models map[string][]string) *arbiter {
	a := &arbiter{backends: make(map[string]*backendState, len(models))}
	for name, names := range models {
		a.backends[name] = &backendState{
			config:     backendConfig{Name: name},
			modelNames: names,
		}
	}
	return a
}

func inboundRequest(t *testing.T, body string, headers map[string]string) *http.Request {
	t.Helper()
	r := httptest.NewRequest(http.MethodPost, "/v1/cluster/inbound",
		bytes.NewReader([]byte(body)))
	for k, v := range headers {
		r.Header.Set(k, v)
	}
	return r
}

func TestInboundHandler_DispatchesBackendFromHeader(t *testing.T) {
	a := arbiterWithBackends(map[string][]string{"vllm": {"m"}, "sglang": {"m"}})
	var got []string
	handlers := map[string]http.HandlerFunc{
		"vllm":   func(w http.ResponseWriter, _ *http.Request) { got = append(got, "vllm") },
		"sglang": func(w http.ResponseWriter, _ *http.Request) { got = append(got, "sglang") },
	}
	h := inboundHandler(newInboundWorker(), a, handlers)

	rec := httptest.NewRecorder()
	h(rec, inboundRequest(t, `{"model":"m"}`, map[string]string{HeaderBackend: "sglang"}))

	if len(got) != 1 || got[0] != "sglang" {
		t.Fatalf("dispatched to %v, want [sglang] -- the header must win over the "+
			"model lookup (which prefers vllm here)", got)
	}
}

func TestInboundHandler_RestoresOriginalPath(t *testing.T) {
	a := arbiterWithBackends(map[string][]string{"ollama": {"m"}})
	var seen string
	handlers := map[string]http.HandlerFunc{
		"ollama": func(_ http.ResponseWriter, r *http.Request) { seen = r.URL.Path },
	}
	h := inboundHandler(newInboundWorker(), a, handlers)

	rec := httptest.NewRecorder()
	h(rec, inboundRequest(t, `{"model":"m"}`, map[string]string{
		HeaderBackend:      "ollama",
		HeaderOriginalPath: "/api/chat",
	}))

	if seen != "/api/chat" {
		t.Fatalf("handler saw path %q, want /api/chat -- without the rewrite the "+
			"worker would never apply Ollama-native num_ctx injection", seen)
	}
}

func TestInboundHandler_IgnoresNonAbsoluteOriginalPath(t *testing.T) {
	a := arbiterWithBackends(map[string][]string{"vllm": {"m"}})
	var seen string
	handlers := map[string]http.HandlerFunc{
		"vllm": func(_ http.ResponseWriter, r *http.Request) { seen = r.URL.Path },
	}
	h := inboundHandler(newInboundWorker(), a, handlers)

	rec := httptest.NewRecorder()
	h(rec, inboundRequest(t, `{"model":"m"}`, map[string]string{
		HeaderBackend:      "vllm",
		HeaderOriginalPath: "http://evil/x",
	}))

	if seen != "/v1/cluster/inbound" {
		t.Fatalf("handler saw path %q, want the untouched inbound path", seen)
	}
}

func TestInboundHandler_FallsBackToModelLookup(t *testing.T) {
	a := arbiterWithBackends(map[string][]string{
		"vllm":   {"Qwen3-8B-NVFP4"},
		"sglang": {"Llama-3.1-8B-Instruct-NVFP4"},
	})
	var got []string
	handlers := map[string]http.HandlerFunc{
		"vllm":   func(http.ResponseWriter, *http.Request) { got = append(got, "vllm") },
		"sglang": func(http.ResponseWriter, *http.Request) { got = append(got, "sglang") },
	}
	h := inboundHandler(newInboundWorker(), a, handlers)

	rec := httptest.NewRecorder()
	h(rec, inboundRequest(t, `{"model":"Llama-3.1-8B-Instruct-NVFP4@65536"}`, nil))

	if len(got) != 1 || got[0] != "sglang" {
		t.Fatalf("dispatched to %v, want [sglang] via the model-name fallback", got)
	}
}

func TestInboundHandler_UndeterminableBackendReturns400(t *testing.T) {
	a := arbiterWithBackends(map[string][]string{"vllm": {"known"}})
	called := false
	handlers := map[string]http.HandlerFunc{
		"vllm": func(http.ResponseWriter, *http.Request) { called = true },
	}
	h := inboundHandler(newInboundWorker(), a, handlers)

	rec := httptest.NewRecorder()
	h(rec, inboundRequest(t, `{"x":1}`, nil))

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status %d, want 400", rec.Code)
	}
	if called {
		t.Fatal("a request with no resolvable backend must not reach a handler")
	}
	if !strings.Contains(rec.Body.String(), "cannot determine target backend") {
		t.Fatalf("unhelpful error body: %s", rec.Body.String())
	}
}

func TestInboundHandler_TracksQueueDepthAndLoadedModel(t *testing.T) {
	a := arbiterWithBackends(map[string][]string{"vllm": {"m"}})
	// Pretend the request caused the backend to load this (model, ctx).
	a.backends["vllm"].currentModel = "Qwen3-8B-NVFP4"
	a.backends["vllm"].currentContext = 131072

	worker := newInboundWorker()
	var depthDuring int64
	handlers := map[string]http.HandlerFunc{
		"vllm": func(http.ResponseWriter, *http.Request) {
			depthDuring = worker.State.queueDepth.Load()
		},
	}
	h := inboundHandler(worker, a, handlers)

	rec := httptest.NewRecorder()
	h(rec, inboundRequest(t, `{"model":"m"}`, map[string]string{HeaderBackend: "vllm"}))

	if depthDuring != 1 {
		t.Errorf("queue depth during dispatch = %d, want 1", depthDuring)
	}
	if after := worker.State.queueDepth.Load(); after != 0 {
		t.Errorf("queue depth after dispatch = %d, want 0", after)
	}
	hb := worker.State.snapshot("wid")
	if hb.LoadedModel != "Qwen3-8B-NVFP4" || hb.LoadedCtx != 131072 {
		t.Errorf("heartbeat reports (%q, %d), want (Qwen3-8B-NVFP4, 131072) -- "+
			"the head scores routing off these fields",
			hb.LoadedModel, hb.LoadedCtx)
	}
	if hb.LastRequestAt == "" {
		t.Error("last_request_at not stamped; head's idle sweep needs it")
	}
}

// The regression this whole wiring exists for: cluster head mode used
// to answer every forwarded request with a hardcoded 503 placeholder.
// Go through the REAL makeRequestHandler (via makeInboundHandler) and
// assert we get the single-host handler's own 404-unknown-model
// rejection instead. An unknown model short-circuits before any podman
// call, so this needs no container runtime.
func TestMakeInboundHandler_ReachesRealRequestHandler(t *testing.T) {
	a := arbiterWithBackends(map[string][]string{"vllm": {"Qwen3-8B-NVFP4"}})
	h := makeInboundHandler(newInboundWorker(), a)

	rec := httptest.NewRecorder()
	h(rec, inboundRequest(t, `{"model":"no-such-model"}`,
		map[string]string{HeaderBackend: "vllm"}))

	if rec.Code == http.StatusServiceUnavailable &&
		strings.Contains(rec.Body.String(), "not_implemented") {
		t.Fatal("inbound is still the Phase 1 placeholder")
	}
	if rec.Code != http.StatusNotFound {
		t.Fatalf("status %d body %q, want 404 from the real request handler",
			rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "unknown model") {
		t.Fatalf("body %q, want the single-host allowlist rejection",
			rec.Body.String())
	}
}

func TestBackendForModel(t *testing.T) {
	a := arbiterWithBackends(map[string][]string{
		"ollama": {"qwen3.5:9b-q8_0", "shared"},
		"vllm":   {"Qwen3-8B-NVFP4", "shared"},
		"sglang": {"Llama-3.1-8B-Instruct-NVFP4"},
	})
	tests := []struct {
		name  string
		model string
		ctx   int
		want  string
	}{
		{"ollama-only", "qwen3.5:9b-q8_0", 0, "ollama"},
		{"vllm-only", "Qwen3-8B-NVFP4", 0, "vllm"},
		{"sglang-only", "Llama-3.1-8B-Instruct-NVFP4", 0, "sglang"},
		// Nothing probed -> tier 3, the declared preference order.
		{"shared-unprobed-uses-preference-order", "shared", 0, "ollama"},
		{"unknown", "nope", 0, ""},
		{"empty", "", 0, ""},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := a.backendForModel(tc.model, tc.ctx); got != tc.want {
				t.Fatalf("backendForModel(%q, %d) = %q, want %q",
					tc.model, tc.ctx, got, tc.want)
			}
		})
	}
}

// B4: a model probed on BOTH vLLM and SGLang used to resolve to vLLM
// unconditionally, because the fallback iterated a fixed name list and
// returned the first match. The probe cache is the only evidence
// available on this path, so it has to win over the name order.
func TestBackendForModel_PrefersBackendWithFittingProbeCell(t *testing.T) {
	const model = "Qwen3-8B-NVFP4"
	base := func() *arbiter {
		a := arbiterWithBackends(map[string][]string{
			"vllm":   {model},
			"sglang": {model},
		})
		a.modelProbedMaxCtx = map[string]map[string]int{
			"vllm":   {},
			"sglang": {},
		}
		return a
	}

	t.Run("only sglang covers the requested ctx", func(t *testing.T) {
		a := base()
		a.modelProbedMaxCtx["vllm"][model] = 32768
		a.modelProbedMaxCtx["sglang"][model] = 131072
		if got := a.backendForModel(model, 65536); got != "sglang" {
			t.Fatalf("got %q, want sglang -- vLLM's probed ceiling (32768) "+
				"cannot serve a 65536 request", got)
		}
	})

	t.Run("only vllm covers the requested ctx", func(t *testing.T) {
		a := base()
		a.modelProbedMaxCtx["vllm"][model] = 131072
		a.modelProbedMaxCtx["sglang"][model] = 32768
		if got := a.backendForModel(model, 65536); got != "vllm" {
			t.Fatalf("got %q, want vllm", got)
		}
	})

	t.Run("no ctx stated falls back to the best-probed backend", func(t *testing.T) {
		a := base()
		a.modelProbedMaxCtx["sglang"][model] = 131072
		if got := a.backendForModel(model, 0); got != "sglang" {
			t.Fatalf("got %q, want sglang -- it is the only backend with a "+
				"fitting probe cell", got)
		}
	})

	t.Run("neither reaches the requested ctx", func(t *testing.T) {
		a := base()
		a.modelProbedMaxCtx["vllm"][model] = 32768
		a.modelProbedMaxCtx["sglang"][model] = 65536
		if got := a.backendForModel(model, 262144); got != "sglang" {
			t.Fatalf("got %q, want sglang -- the highest probed ceiling is the "+
				"best available guess", got)
		}
	})

	t.Run("unprobed model keeps the declared preference order", func(t *testing.T) {
		a := base()
		if got := a.backendForModel(model, 65536); got != "vllm" {
			t.Fatalf("got %q, want vllm (backendPreferenceOrder)", got)
		}
	})

	t.Run("nil probe map does not panic", func(t *testing.T) {
		a := arbiterWithBackends(map[string][]string{"vllm": {model}})
		if got := a.backendForModel(model, 32768); got != "vllm" {
			t.Fatalf("got %q, want vllm", got)
		}
	})
}

// B1: the executor used to be a no-op that logged and, for shutdown,
// slept then os.Exit'd -- cutting live streams. It must now drive the
// real scheduler's drain.
func TestArbiterCommandExecutor_DrainDrainsNamedBackend(t *testing.T) {
	a := arbiterWithBackends(map[string][]string{"vllm": {"m"}, "ollama": {"m"}})
	a.drainTimeout = 2 * time.Second
	// One request "already proxied upstream" on vllm. drainBackend
	// must observe it; we release it from another goroutine.
	atomic.AddInt64(&a.backends["vllm"].upstreamReqs, 1)

	state := &WorkerState{}
	e := newArbiterCommandExecutor(state, a)
	e.background = func(f func()) { f() } // synchronous for the test

	// Observed from the releasing goroutine: while the drain is still
	// waiting, the heartbeat must advertise "draining" so the head stops
	// sending new work.
	midDrain := make(chan string, 1)
	go func() {
		time.Sleep(200 * time.Millisecond)
		midDrain <- state.snapshot("wid").HealthStatus
		atomic.AddInt64(&a.backends["vllm"].upstreamReqs, -1)
	}()

	start := time.Now()
	if err := e.Execute(context.Background(),
		Command{Type: CommandDrain, Backend: "vllm"}); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if elapsed := time.Since(start); elapsed < 150*time.Millisecond {
		t.Fatalf("drain returned in %v without waiting for the in-flight "+
			"upstream request -- it is still a no-op", elapsed)
	}
	if got := <-midDrain; got != HealthDraining {
		t.Errorf("health status mid-drain = %q, want %q", got, HealthDraining)
	}
	// ...and once the drain COMPLETES the worker must be routable again.
	// Latching at "draining" removes it from the fleet forever.
	if got := state.snapshot("wid").HealthStatus; got != HealthReady {
		t.Errorf("health status after drain = %q, want %q", got, HealthReady)
	}
	if left := atomic.LoadInt64(&a.backends["vllm"].upstreamReqs); left != 0 {
		t.Errorf("%d requests still in flight after drain", left)
	}
}

func TestArbiterCommandExecutor_DrainWithoutBackendDrainsAll(t *testing.T) {
	a := arbiterWithBackends(map[string][]string{"vllm": {"m"}, "sglang": {"m"}})
	a.drainTimeout = 500 * time.Millisecond
	for _, name := range []string{"vllm", "sglang"} {
		atomic.AddInt64(&a.backends[name].upstreamReqs, 1)
	}
	e := newArbiterCommandExecutor(&WorkerState{}, a)
	e.background = func(f func()) { f() }

	// Both drains time out (nothing releases the counters); the point
	// is that BOTH backends were waited on, i.e. ~2 x drainTimeout.
	start := time.Now()
	_ = e.Execute(context.Background(), Command{Type: CommandDrain})
	if elapsed := time.Since(start); elapsed < 900*time.Millisecond {
		t.Fatalf("empty-backend drain took %v; it did not wait on both "+
			"backends", elapsed)
	}
}

func TestArbiterCommandExecutor_ShutdownDrainsBeforeExit(t *testing.T) {
	a := arbiterWithBackends(map[string][]string{"vllm": {"m"}})
	a.drainTimeout = 2 * time.Second
	atomic.AddInt64(&a.backends["vllm"].upstreamReqs, 1)

	var drainedBeforeExit int64 = -1
	exited := make(chan int, 1)
	e := newArbiterCommandExecutor(&WorkerState{}, a)
	e.background = func(f func()) { f() }
	e.exit = func(code int) {
		drainedBeforeExit = atomic.LoadInt64(&a.backends["vllm"].upstreamReqs)
		exited <- code
	}

	go func() {
		time.Sleep(200 * time.Millisecond)
		atomic.AddInt64(&a.backends["vllm"].upstreamReqs, -1)
	}()
	_ = e.Execute(context.Background(),
		Command{Type: CommandShutdown, GraceSeconds: 30})

	select {
	case code := <-exited:
		if code != 0 {
			t.Errorf("exit code %d, want 0", code)
		}
	default:
		t.Fatal("shutdown never reached exit")
	}
	if drainedBeforeExit != 0 {
		t.Fatalf("%d request(s) still in flight at exit -- shutdown still "+
			"hard-exits mid-stream", drainedBeforeExit)
	}
}

func TestArbiterCommandExecutor_UnknownBackendDrainIsHarmless(t *testing.T) {
	a := arbiterWithBackends(map[string][]string{"vllm": {"m"}})
	e := newArbiterCommandExecutor(&WorkerState{}, a)
	e.background = func(f func()) { f() }
	if err := e.Execute(context.Background(),
		Command{Type: CommandDrain, Backend: "nope"}); err != nil {
		t.Fatalf("Execute: %v", err)
	}
}

func TestArbiterCommandExecutor_ServeUpdatesState(t *testing.T) {
	state := &WorkerState{}
	e := newArbiterCommandExecutor(state, arbiterWithBackends(nil))
	if err := e.Execute(context.Background(), Command{
		Type: CommandServe, RequestID: "r1",
		TargetModel: "Qwen3-8B-NVFP4", TargetCtx: 131072,
	}); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	hb := state.snapshot("wid")
	if hb.LoadedModel != "Qwen3-8B-NVFP4" || hb.LoadedCtx != 131072 {
		t.Errorf("state = (%q, %d)", hb.LoadedModel, hb.LoadedCtx)
	}
	if hb.LastRequestAt == "" {
		t.Error("last_request_at not stamped")
	}
}

func TestCurrentLoaded(t *testing.T) {
	a := arbiterWithBackends(map[string][]string{"vllm": {"m"}})
	if m, c := a.currentLoaded("vllm"); m != "" || c != 0 {
		t.Fatalf("cold backend reports (%q, %d), want empty", m, c)
	}
	a.backends["vllm"].currentModel = "m"
	a.backends["vllm"].currentContext = 32768
	if m, c := a.currentLoaded("vllm"); m != "m" || c != 32768 {
		t.Fatalf("got (%q, %d), want (m, 32768)", m, c)
	}
	if m, c := a.currentLoaded("nope"); m != "" || c != 0 {
		t.Fatalf("unknown backend reports (%q, %d), want empty", m, c)
	}
}

// makeInboundHandler builds one handler per backend up front; make
// sure concurrent dispatch through the shared map is safe to read.
func TestInboundHandler_ConcurrentDispatch(t *testing.T) {
	a := arbiterWithBackends(map[string][]string{"vllm": {"m"}})
	var mu sync.Mutex
	n := 0
	handlers := map[string]http.HandlerFunc{
		"vllm": func(http.ResponseWriter, *http.Request) {
			mu.Lock()
			n++
			mu.Unlock()
		},
	}
	h := inboundHandler(newInboundWorker(), a, handlers)

	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			rec := httptest.NewRecorder()
			r := httptest.NewRequest(http.MethodPost, "/v1/cluster/inbound",
				bytes.NewReader([]byte(`{"model":"m"}`)))
			r.Header.Set(HeaderBackend, "vllm")
			r.URL = &url.URL{Path: "/v1/cluster/inbound"}
			h(rec, r)
		}()
	}
	wg.Wait()
	if n != 16 {
		t.Fatalf("dispatched %d/16 requests", n)
	}
}

// --- Round 3 / S1-A: the drain health latch ---

// Drain is a transient pause, not a one-way latch. Leaving the status
// at "draining" forever removes the worker from the head's routing pool
// permanently (routing_policy.workerAvailable excludes it), which on a
// single-worker fleet is a permanent 503.
func TestArbiterCommandExecutor_DrainCompletionReturnsWorkerToReady(t *testing.T) {
	a := arbiterWithBackends(map[string][]string{"vllm": {"m"}})
	a.drainTimeout = 100 * time.Millisecond
	state := &WorkerState{}
	e := newArbiterCommandExecutor(state, a)
	e.background = func(f func()) { f() }

	if err := e.Execute(context.Background(),
		Command{Type: CommandDrain, Backend: "vllm"}); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if got := state.HealthStatus(); got != HealthReady {
		t.Fatalf("health after completed drain = %q, want %q -- the worker is "+
			"permanently removed from the fleet", got, HealthReady)
	}
	if got := state.snapshot("wid").HealthStatus; got != HealthReady {
		t.Errorf("heartbeat health = %q, want %q", got, HealthReady)
	}
}

// A shutdown arriving DURING a drain is terminal: the drain finishing
// must not swap the status back to ready and re-admit the worker to the
// fleet moments before its process exits.
func TestArbiterCommandExecutor_ShutdownDuringDrainStaysTerminal(t *testing.T) {
	a := arbiterWithBackends(map[string][]string{"vllm": {"m"}})
	a.drainTimeout = 2 * time.Second
	atomic.AddInt64(&a.backends["vllm"].upstreamReqs, 1)

	state := &WorkerState{}
	e := newArbiterCommandExecutor(state, a)
	e.exit = func(int) {}

	// Run the drain asynchronously so the shutdown can land mid-flight.
	drained := make(chan struct{})
	e.background = func(f func()) { go func() { f(); close(drained) }() }
	if err := e.Execute(context.Background(),
		Command{Type: CommandDrain, Backend: "vllm"}); err != nil {
		t.Fatalf("drain Execute: %v", err)
	}

	// Shutdown lands while the drain is still waiting.
	e.background = func(f func()) {} // don't run the shutdown drain/exit here
	if err := e.Execute(context.Background(),
		Command{Type: CommandShutdown, GraceSeconds: 30}); err != nil {
		t.Fatalf("shutdown Execute: %v", err)
	}
	if got := state.HealthStatus(); got != HealthShuttingDown {
		t.Fatalf("health after shutdown = %q, want %q", got, HealthShuttingDown)
	}

	atomic.AddInt64(&a.backends["vllm"].upstreamReqs, -1)
	<-drained

	if got := state.HealthStatus(); got != HealthShuttingDown {
		t.Fatalf("drain completion reset health to %q; a terminal shutdown "+
			"must not be swapped back", got)
	}
}

// A drain command that lands AFTER a shutdown must not downgrade the
// terminal status.
func TestArbiterCommandExecutor_DrainAfterShutdownKeepsTerminal(t *testing.T) {
	a := arbiterWithBackends(map[string][]string{"vllm": {"m"}})
	a.drainTimeout = 50 * time.Millisecond
	state := &WorkerState{}
	e := newArbiterCommandExecutor(state, a)
	e.exit = func(int) {}
	e.background = func(f func()) { f() }

	_ = e.Execute(context.Background(), Command{Type: CommandShutdown})
	_ = e.Execute(context.Background(), Command{Type: CommandDrain, Backend: "vllm"})
	if got := state.HealthStatus(); got != HealthShuttingDown {
		t.Fatalf("health = %q, want %q", got, HealthShuttingDown)
	}
}

// Two overlapping drains: only the LAST one to finish may declare the
// worker ready, otherwise the first completion advertises "ready" while
// the second is still waiting out its own in-flight requests.
func TestArbiterCommandExecutor_ConcurrentDrainsOnlyLastReturnsToReady(t *testing.T) {
	a := arbiterWithBackends(map[string][]string{"vllm": {"m"}, "sglang": {"m"}})
	a.drainTimeout = 2 * time.Second
	atomic.AddInt64(&a.backends["sglang"].upstreamReqs, 1)

	state := &WorkerState{}
	e := newArbiterCommandExecutor(state, a)

	slowDone := make(chan struct{})
	e.background = func(f func()) { go func() { f(); close(slowDone) }() }
	if err := e.Execute(context.Background(),
		Command{Type: CommandDrain, Backend: "sglang"}); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	// Wait until the slow drain owns the arbiter mutex.
	time.Sleep(100 * time.Millisecond)

	// The fast drain (nothing in flight on vllm) completes immediately.
	fastDone := make(chan struct{})
	e.background = func(f func()) { go func() { f(); close(fastDone) }() }
	if err := e.Execute(context.Background(),
		Command{Type: CommandDrain, Backend: "vllm"}); err != nil {
		t.Fatalf("Execute: %v", err)
	}

	atomic.AddInt64(&a.backends["sglang"].upstreamReqs, -1)
	<-slowDone
	<-fastDone
	if got := state.HealthStatus(); got != HealthReady {
		t.Fatalf("health after both drains = %q, want %q", got, HealthReady)
	}
}

// Serve is bookkeeping only: it must not resurrect a shutting-down
// worker nor cut a drain short.
func TestArbiterCommandExecutor_ServeDoesNotMutateHealth(t *testing.T) {
	for _, status := range []string{HealthDraining, HealthShuttingDown} {
		state := &WorkerState{}
		state.SetHealthStatus(status)
		e := newArbiterCommandExecutor(state, arbiterWithBackends(nil))
		if err := e.Execute(context.Background(), Command{
			Type: CommandServe, TargetModel: "m", TargetCtx: 32768,
		}); err != nil {
			t.Fatalf("Execute: %v", err)
		}
		if got := state.HealthStatus(); got != status {
			t.Errorf("serve changed health %q -> %q", status, got)
		}
	}
}

func TestWorkerState_HealthTransitions(t *testing.T) {
	s := &WorkerState{}
	if got := s.HealthStatus(); got != HealthReady {
		t.Errorf("zero value health = %q, want %q", got, HealthReady)
	}
	if s.CompareAndSwapHealth(HealthDraining, HealthReady) {
		t.Error("CAS from a status the worker is not in succeeded")
	}
	s.SetHealthStatus(HealthDraining)
	if !s.CompareAndSwapHealth(HealthDraining, HealthReady) {
		t.Error("CAS draining->ready failed")
	}
	if !s.SetHealthUnlessShuttingDown(HealthDraining) {
		t.Error("SetHealthUnlessShuttingDown refused on a non-terminal status")
	}
	s.SetHealthStatus(HealthShuttingDown)
	if s.SetHealthUnlessShuttingDown(HealthDraining) {
		t.Error("SetHealthUnlessShuttingDown downgraded a terminal status")
	}
	if s.CompareAndSwapHealth(HealthDraining, HealthReady) {
		t.Error("CAS escaped the terminal shutting_down state")
	}
}
