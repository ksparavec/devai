package main

import (
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// fakeTrainer is a job runner's HTTP side: /health answers with whatever
// the test set, everything else is recorded and answered 200.
type fakeTrainer struct {
	mu     sync.Mutex
	health string // JSON body; "" = answer 500
	down   bool   // refuse to answer at all (connection closed)
	seen   []string
	srv    *httptest.Server
}

func newFakeTrainer(t *testing.T, health string) *fakeTrainer {
	t.Helper()
	f := &fakeTrainer{health: health}
	f.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.mu.Lock()
		health, down := f.health, f.down
		f.seen = append(f.seen, r.Method+" "+r.URL.Path)
		f.mu.Unlock()
		if down {
			if hj, ok := w.(http.Hijacker); ok {
				if conn, _, err := hj.Hijack(); err == nil {
					_ = conn.Close()
				}
			}
			return
		}
		if r.URL.Path == "/health" {
			if health == "" {
				w.WriteHeader(http.StatusInternalServerError)
				return
			}
			_, _ = w.Write([]byte(health))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"fine_tuning.job","id":"ftjob-x","status":"cancelled"}`))
	}))
	t.Cleanup(f.srv.Close)
	return f
}

func (f *fakeTrainer) set(health string, down bool) {
	f.mu.Lock()
	f.health, f.down = health, down
	f.mu.Unlock()
}

func busyHealth(startedAt time.Time) string {
	return `{"status":"busy","job":"ftjob-abc","phase":"training","started_at":` +
		jsonInt(startedAt.Unix()) + `}`
}

func jsonInt(n int64) string {
	b, _ := json.Marshal(n)
	return string(b)
}

const idleHealth = `{"status":"ok","job":null,"phase":null,"started_at":null}`

// jobRunnerArbiter: a resident trainer and an idle vLLM, a podman stub that
// counts creates, and a container-state stub that says "running".
func jobRunnerArbiter(t *testing.T, trainer *fakeTrainer) (*arbiter, *backendState, *backendState, *int64) {
	t.Helper()
	tb := testBackend("laya-trainer", trainer.srv)
	tb.config.JobRunner = true
	tb.config.MaxHold = 2 * time.Hour
	tb.config.ContainerName = "devai-laya-trainer"
	tb.config.Entrypoint = layaTrainerEntrypoint
	tb.modelNames = []string{"laya-multilingual", "laya-english"}
	tb.running = true
	tb.containerLaunched = true

	vllmSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(vllmSrv.Close)
	vb := testBackend("vllm", vllmSrv)
	vb.config.ContainerName = "devai-vllm"
	vb.config.Entrypoint = vllmEntrypoint

	a := testArbiter(tb, vb)
	client, creates := podmanStub(t)
	a.podmanClient = client
	a.healthTimeout = 5 * time.Second
	a.containerStateStub = func(string) (string, int, bool) { return "running", 0, true }
	return a, tb, vb, creates
}

func ensure(a *arbiter, bs *backendState, model string) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.ensureBackendRunning(bs, model, 0, false, nil)
}

// --- the hold -------------------------------------------------------------

// The reason the hold exists: a request for another backend drains
// in-flight REQUESTS and then stops the container -- a training run is
// not a request, so without the hold the first teacher request kills it.
func TestJobRunner_BusyTrainerRefusesEviction(t *testing.T) {
	trainer := newFakeTrainer(t, busyHealth(time.Now()))
	a, tb, vb, creates := jobRunnerArbiter(t, trainer)

	err := ensure(a, vb, "test-model")

	var held *gpuHeldError
	if !errors.As(err, &held) {
		t.Fatalf("want a gpuHeldError, got %v", err)
	}
	if !tb.running || !tb.containerLaunched {
		t.Fatal("the busy trainer must not be stopped")
	}
	if *creates != 0 {
		t.Fatalf("nothing may be launched while the GPU is held (%d creates)", *creates)
	}
	if held.job != "ftjob-abc" || held.phase != "training" {
		t.Errorf("hold names job %q phase %q", held.job, held.phase)
	}
}

func TestJobRunner_HoldIsA503WithRetryAfter(t *testing.T) {
	trainer := newFakeTrainer(t, busyHealth(time.Now()))
	a, _, _, _ := jobRunnerArbiter(t, trainer)

	req := httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"test-model","messages":[]}`))
	w := httptest.NewRecorder()
	a.makeRequestHandler("vllm")(w, req)

	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("status %d, want 503", w.Code)
	}
	if got := w.Header().Get("Retry-After"); got != "30" {
		t.Errorf("Retry-After %q, want 30", got)
	}
	var body struct {
		Error struct {
			Type, Code, Message string
		} `json:"error"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body.Error.Code != "gpu_held_by_job" || !strings.Contains(body.Error.Message, "ftjob-abc") {
		t.Errorf("error body %+v", body.Error)
	}
}

// Ollama's model-less surfaces (/api/ps, /api/version, ...) also take the
// GPU when Ollama is not running. They must respect the hold too.
func TestJobRunner_OllamaModelLessRequestRespectsHold(t *testing.T) {
	trainer := newFakeTrainer(t, busyHealth(time.Now()))
	a, tb, _, _ := jobRunnerArbiter(t, trainer)
	ollama := testBackend("ollama", trainer.srv)
	ollama.recreateCond = sync.NewCond(&a.mu)
	a.backends["ollama"] = ollama

	a.mu.Lock()
	err := a.ensureOllamaRunning(ollama, "", 0, false)
	a.mu.Unlock()

	var held *gpuHeldError
	if !errors.As(err, &held) {
		t.Fatalf("want a gpuHeldError, got %v", err)
	}
	if ollama.running || !tb.running {
		t.Fatal("ollama must not take the GPU from a busy trainer")
	}
}

func TestJobRunner_HoldCapEvicts(t *testing.T) {
	trainer := newFakeTrainer(t, busyHealth(time.Now().Add(-3*time.Hour)))
	a, tb, vb, _ := jobRunnerArbiter(t, trainer)

	if err := ensure(a, vb, "test-model"); err != nil {
		t.Fatalf("past LAYA_MAX_HOLD_S the request must go through, got %v", err)
	}
	if tb.running || tb.containerLaunched {
		t.Fatal("a trainer past its hold cap must be evicted")
	}
}

func TestJobRunner_ZeroCapHoldsIndefinitely(t *testing.T) {
	trainer := newFakeTrainer(t, busyHealth(time.Now().Add(-100*time.Hour)))
	a, tb, vb, _ := jobRunnerArbiter(t, trainer)
	tb.config.MaxHold = 0

	var held *gpuHeldError
	if err := ensure(a, vb, "test-model"); !errors.As(err, &held) {
		t.Fatalf("MaxHold=0 means no cap; want a hold, got %v", err)
	}
}

func TestJobRunner_IdleTrainerIsEvictedNormally(t *testing.T) {
	trainer := newFakeTrainer(t, idleHealth)
	a, tb, vb, creates := jobRunnerArbiter(t, trainer)

	if err := ensure(a, vb, "test-model"); err != nil {
		t.Fatalf("an idle trainer must not hold the GPU, got %v", err)
	}
	if tb.running {
		t.Fatal("the idle trainer must be stopped for the switch")
	}
	if *creates != 1 {
		t.Fatalf("vllm must be launched once, got %d creates", *creates)
	}
}

// A controller busy packaging on every core can answer /health slowly; a
// single missed probe must not hand the GPU away mid-job.
func TestJobRunner_UnansweredHealthKeepsTheLastBusyAnswer(t *testing.T) {
	trainer := newFakeTrainer(t, busyHealth(time.Now()))
	a, tb, vb, _ := jobRunnerArbiter(t, trainer)
	var held *gpuHeldError
	if err := ensure(a, vb, "test-model"); !errors.As(err, &held) {
		t.Fatalf("first probe: want a hold, got %v", err)
	}

	trainer.set("", true)
	if err := ensure(a, vb, "test-model"); !errors.As(err, &held) {
		t.Fatalf("silent trainer seen busy before: want the hold kept, got %v", err)
	}
	if !tb.running {
		t.Fatal("the trainer must still be resident")
	}
}

func TestJobRunner_AGoneContainerHoldsNothing(t *testing.T) {
	trainer := newFakeTrainer(t, busyHealth(time.Now()))
	a, tb, vb, _ := jobRunnerArbiter(t, trainer)
	var held *gpuHeldError
	if err := ensure(a, vb, "test-model"); !errors.As(err, &held) {
		t.Fatalf("want a hold first, got %v", err)
	}

	trainer.set("", true)
	a.containerStateStub = func(string) (string, int, bool) { return "exited", 137, true }
	if err := ensure(a, vb, "test-model"); err != nil {
		t.Fatalf("a trainer whose container is gone cannot hold the GPU, got %v", err)
	}
	if tb.running {
		t.Fatal("the vanished trainer's state must be reset")
	}
}

// Nothing listening -- the compose `sleep infinity` placeholder, or a
// controller still starting -- runs no job and holds nothing.
func TestJobRunner_NothingListeningHoldsNothing(t *testing.T) {
	trainer := newFakeTrainer(t, busyHealth(time.Now()))
	a, tb, vb, _ := jobRunnerArbiter(t, trainer)
	tb.lastBusy = &jobRunnerHealth{Status: "busy", Job: "ftjob-old"} // stale memory
	trainer.srv.Close()                                              // connection refused
	if err := ensure(a, vb, "test-model"); err != nil {
		t.Fatalf("a runner nobody listens on holds nothing, got %v", err)
	}
}

// A runner whose container runs but whose /health is silent may be busy
// packaging; it was never seen busy (e.g. just adopted), yet it must still
// hold -- bounded by the cap, counted from when it went silent.
func TestJobRunner_SilentNeverSeenBusyHoldsBoundedByTheCap(t *testing.T) {
	trainer := newFakeTrainer(t, "")
	a, tb, vb, _ := jobRunnerArbiter(t, trainer)
	trainer.set("", true)

	var held *gpuHeldError
	if err := ensure(a, vb, "test-model"); !errors.As(err, &held) {
		t.Fatalf("a silent runner with a running container must hold, got %v", err)
	}
	if held.job != "" || !strings.Contains(held.Error(), "does not answer /health") {
		t.Errorf("hold %q", held.Error())
	}
	if d := time.Until(held.deadline); d < 119*time.Minute || d > 2*time.Hour {
		t.Errorf("deadline in %s, want MaxHold from now", d)
	}

	tb.silentSince = time.Now().Add(-3 * time.Hour)
	if err := ensure(a, vb, "test-model"); err != nil {
		t.Fatalf("silent past the cap: want eviction, got %v", err)
	}
	if tb.running {
		t.Fatal("the capped runner must be stopped")
	}
}

func TestJobRunner_APausedContainerStillHolds(t *testing.T) {
	trainer := newFakeTrainer(t, busyHealth(time.Now()))
	a, _, vb, _ := jobRunnerArbiter(t, trainer)
	trainer.set("", true)
	a.containerStateStub = func(string) (string, int, bool) { return "paused", 0, true }
	var held *gpuHeldError
	if err := ensure(a, vb, "test-model"); !errors.As(err, &held) {
		t.Fatalf("a paused container still holds its VRAM, got %v", err)
	}
}

// roundTripFunc lets a test make the trainer's /health fail in a chosen way.
type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func dnsFailingClient(timeout bool) *http.Client {
	return &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		return nil, &net.DNSError{Err: "server misbehaving", Name: "laya-trainer", IsTimeout: timeout}
	})}
}

// Re-review HIGH: a DNS failure used to count as "nothing listening", so
// one resolver hiccup during a job evicted it. It is silence; podman decides.
func TestJobRunner_ADNSHiccupDoesNotEvictABusyTrainer(t *testing.T) {
	trainer := newFakeTrainer(t, busyHealth(time.Now()))
	a, tb, vb, creates := jobRunnerArbiter(t, trainer)
	var held *gpuHeldError
	if err := ensure(a, vb, "test-model"); !errors.As(err, &held) {
		t.Fatalf("first probe: want a hold, got %v", err)
	}
	a.healthClient = dnsFailingClient(true)
	if err := ensure(a, vb, "test-model"); !errors.As(err, &held) {
		t.Fatalf("a DNS hiccup must not evict a busy trainer, got %v", err)
	}
	if !tb.running || *creates != 0 || held.job != "ftjob-abc" {
		t.Fatalf("running=%v creates=%d job=%q", tb.running, *creates, held.job)
	}
}

// Re-review LOW: libpod's 404 for a removed container reads as status "";
// that is gone, not "running and silent".
func TestJobRunner_ARemovedContainerHoldsNothing(t *testing.T) {
	trainer := newFakeTrainer(t, busyHealth(time.Now()))
	a, _, vb, _ := jobRunnerArbiter(t, trainer)
	a.healthClient = dnsFailingClient(false) // the name no longer resolves
	a.containerStateStub = func(string) (string, int, bool) { return "", 0, true }
	if err := ensure(a, vb, "test-model"); err != nil {
		t.Fatalf("a removed trainer holds nothing, got %v", err)
	}
}

// Re-review MEDIUM: compose can put the `sleep infinity` placeholder back
// under a router that still thinks the trainer runs. Nothing listens then,
// and the trainer must count as vanished (so the next job relaunches it)
// -- while a busy or merely slow controller never does.
func TestJobRunner_APlaceholderUnderARunningTrainerHasVanished(t *testing.T) {
	trainer := newFakeTrainer(t, busyHealth(time.Now()))
	a, tb, _, _ := jobRunnerArbiter(t, trainer)

	if gone, why := a.backendVanished(tb); gone {
		t.Fatalf("a busy controller vanished: %s", why)
	}
	trainer.set("", true)
	if gone, why := a.backendVanished(tb); gone {
		t.Fatalf("a silent controller vanished: %s", why)
	}
	trainer.srv.Close()
	if gone, why := a.backendVanished(tb); !gone || !strings.Contains(why, "nothing listens") {
		t.Fatalf("a placeholder must read as vanished, got %v %q", gone, why)
	}
}

// The review's CRITICAL: a request on the trainer's OWN port ran the engine
// liveness check, which condemned a silent (busy) trainer from /health
// alone. The next switch then launched vLLM onto its GPU, and a job POST
// recreated the trainer under its job.
func TestJobRunner_OwnPortRequestsNeverCondemnASilentTrainer(t *testing.T) {
	trainer := newFakeTrainer(t, busyHealth(time.Now()))
	a, tb, vb, creates := jobRunnerArbiter(t, trainer)
	trainer.set("", true) // mid-job, too busy to answer

	if err := ensure(a, tb, ""); err != nil { // a status read
		t.Fatalf("status read on a resident trainer: %v", err)
	}
	if err := ensure(a, tb, "laya-english"); err != nil { // a job POST
		t.Fatalf("job POST on a resident trainer: %v", err)
	}
	if !tb.running || !tb.containerLaunched || *creates != 0 {
		t.Fatalf("the trainer was condemned (running=%v launched=%v creates=%d)",
			tb.running, tb.containerLaunched, *creates)
	}
	var held *gpuHeldError
	if err := ensure(a, vb, "test-model"); !errors.As(err, &held) {
		t.Fatalf("and it must still hold the GPU, got %v", err)
	}
}

func TestJobRunner_ASilentTrainerAtBootIsAdopted(t *testing.T) {
	trainer := newFakeTrainer(t, "")
	a, tb, vb, creates := jobRunnerArbiter(t, trainer)
	tb.running, tb.containerLaunched = false, false
	trainer.set("", true)

	a.reconcileBackendState()
	if !tb.running {
		t.Fatal("a silent trainer whose container runs must be adopted")
	}
	var held *gpuHeldError
	if err := ensure(a, vb, "test-model"); !errors.As(err, &held) || *creates != 0 {
		t.Fatalf("want a hold and no launch, got %v, %d creates", err, *creates)
	}
}

func TestJobRunner_APlaceholderAtBootIsNotAdopted(t *testing.T) {
	trainer := newFakeTrainer(t, idleHealth)
	a, tb, _, _ := jobRunnerArbiter(t, trainer)
	tb.running, tb.containerLaunched = false, false
	trainer.srv.Close() // `sleep infinity`: nothing listens

	a.reconcileBackendState()
	if tb.running {
		t.Fatal("a placeholder must not be adopted")
	}
}

// A burst of refused requests costs one probe, not one each: the probe
// runs under the router's global mutex.
func TestJobRunner_TheHoldVerdictIsCachedBriefly(t *testing.T) {
	trainer := newFakeTrainer(t, busyHealth(time.Now()))
	a, _, vb, _ := jobRunnerArbiter(t, trainer)
	a.holdCacheTTL = time.Minute
	for i := 0; i < 5; i++ {
		var held *gpuHeldError
		if err := ensure(a, vb, "test-model"); !errors.As(err, &held) {
			t.Fatalf("request %d: want a hold, got %v", i, err)
		}
	}
	probes := 0
	for _, s := range trainer.seen {
		if s == "GET /health" {
			probes++
		}
	}
	if probes != 1 {
		t.Fatalf("%d /health probes for 5 refused requests, want 1", probes)
	}
}

func TestJobRunner_ANewContainerIsNotJudgedByAnOldJob(t *testing.T) {
	trainer := newFakeTrainer(t, idleHealth)
	a, tb, _, _ := jobRunnerArbiter(t, trainer)
	tb.running, tb.containerLaunched = false, false
	tb.lastBusy = &jobRunnerHealth{Status: "busy", Job: "ftjob-old", StartedAt: time.Now().Unix()}
	tb.busySeenAt = time.Now()
	if err := ensure(a, tb, "laya-multilingual"); err != nil {
		t.Fatal(err)
	}
	if tb.lastBusy != nil || !tb.busySeenAt.IsZero() {
		t.Fatal("a launch must forget the previous container's job")
	}
}

// A streaming request whose refusal outlasts the keepalive grace has a
// committed 200: the hold must still be recognisable in-band.
func TestJobRunner_StreamingRefusalCarriesTheCode(t *testing.T) {
	w := httptest.NewRecorder()
	writeSSELaunchError(w, "/v1/chat/completions",
		&gpuHeldError{holder: "laya-trainer", job: "ftjob-abc", phase: "training"})
	body := w.Body.String()
	if !strings.Contains(body, `"code":"gpu_held_by_job"`) || !strings.Contains(body, `"retry_after":30`) {
		t.Fatalf("in-band error %q", body)
	}
}

// A job submission can be in flight to the trainer at the moment another
// backend's request decides whether to evict it. Checking /health before
// that submission has landed would see "ok" and kill the job it was about
// to start; the check must come after the trainer's in-flight requests
// drained.
func TestJobRunner_HoldIsCheckedAfterInFlightRequestsDrain(t *testing.T) {
	trainer := newFakeTrainer(t, idleHealth)
	a, tb, vb, _ := jobRunnerArbiter(t, trainer)
	atomic.StoreInt64(&tb.upstreamReqs, 1) // a job submission is being proxied
	go func() {
		time.Sleep(300 * time.Millisecond)
		trainer.set(busyHealth(time.Now()), false) // ... and the trainer accepted it
		atomic.StoreInt64(&tb.upstreamReqs, 0)
	}()

	var held *gpuHeldError
	if err := ensure(a, vb, "test-model"); !errors.As(err, &held) {
		t.Fatalf("want the hold of the job that just started, got %v", err)
	}
	if !tb.running {
		t.Fatal("the trainer that just accepted a job must not be stopped")
	}
}

// A router restarted mid-job (make cache-up) adopts the serving trainer at
// boot; because the hold is read from the trainer's own /health, the
// adopted trainer is held with its original deadline.
func TestJobRunner_BootAdoptionHoldsTheGPU(t *testing.T) {
	trainer := newFakeTrainer(t, busyHealth(time.Now()))
	a, tb, vb, creates := jobRunnerArbiter(t, trainer)
	tb.running, tb.containerLaunched = false, false // a fresh router process

	a.reconcileBackendState()
	if !tb.running {
		t.Fatal("a serving trainer must be adopted at boot")
	}
	var held *gpuHeldError
	if err := ensure(a, vb, "test-model"); !errors.As(err, &held) {
		t.Fatalf("the adopted busy trainer must hold the GPU, got %v", err)
	}
	if *creates != 0 {
		t.Fatal("vLLM must not be launched onto the trainer's GPU")
	}
}

func TestJobRunner_IdleSweepSkipsABusyJobRunner(t *testing.T) {
	trainer := newFakeTrainer(t, busyHealth(time.Now()))
	a, tb, _, _ := jobRunnerArbiter(t, trainer)
	a.idleTimeout = time.Minute
	tb.lastRequest = time.Now().Add(-time.Hour) // no request for an hour: the job is the work

	a.idleSweepOnce()
	if !tb.running {
		t.Fatal("the idle sweep must not stop a busy job runner")
	}

	trainer.set(idleHealth, false)
	a.idleSweepOnce()
	if tb.running {
		t.Fatal("an idle job runner is swept like any backend")
	}
}

// --- model-agnostic lifecycle ------------------------------------------------

// The router tracks a model per backend and recreates on a change. For the
// trainer the model is a JOB setting: a job for laya-english while the
// trainer was launched for laya-multilingual must reach the controller
// (which answers 409 if a job runs), not recreate the container under it.
func TestJobRunner_ADifferentBaseNeverRecreates(t *testing.T) {
	trainer := newFakeTrainer(t, busyHealth(time.Now()))
	a, tb, _, creates := jobRunnerArbiter(t, trainer)
	if err := ensure(a, tb, "laya-english"); err != nil {
		t.Fatal(err)
	}
	if err := ensure(a, tb, "laya-multilingual"); err != nil {
		t.Fatal(err)
	}
	if *creates != 0 {
		t.Fatalf("a model name must never recreate a job runner (%d creates)", *creates)
	}
}

func TestJobRunner_LaunchIsModelAgnostic(t *testing.T) {
	trainer := newFakeTrainer(t, idleHealth)
	a, tb, _, creates := jobRunnerArbiter(t, trainer)
	tb.running, tb.containerLaunched = false, false

	if err := ensure(a, tb, "laya-multilingual"); err != nil {
		t.Fatal(err)
	}
	if *creates != 1 || !tb.running {
		t.Fatalf("want one launch, got %d creates, running=%v", *creates, tb.running)
	}
	if tb.currentModel != "" || tb.currentContext != 0 {
		t.Errorf("a job runner has no current model/context, got %q/%d", tb.currentModel, tb.currentContext)
	}
	if err := ensure(a, tb, "laya-english"); err != nil || *creates != 1 {
		t.Fatalf("the second base must reuse the container (err %v, %d creates)", err, *creates)
	}
}

// Status reads (GET job, events, list) carry no model. They must never
// launch the trainer: that would evict the teacher to answer a question
// the volume answers (runs/<job>/job.json).
func TestJobRunner_AModelLessRequestNeverLaunches(t *testing.T) {
	trainer := newFakeTrainer(t, idleHealth)
	a, tb, _, creates := jobRunnerArbiter(t, trainer)
	tb.running, tb.containerLaunched = false, false

	err := ensure(a, tb, "")
	if err == nil || !strings.Contains(err.Error(), "job.json") {
		t.Fatalf("want an error pointing at the volume, got %v", err)
	}
	if *creates != 0 {
		t.Fatal("a status read launched the trainer")
	}
}

// --- request surface ---------------------------------------------------------

// OpenAI's cancel call has no body. The router parses every POST body as
// JSON, so it answered 400 -- a stock OpenAI client could never cancel.
func TestJobRunner_EmptyBodyPostIsProxied(t *testing.T) {
	trainer := newFakeTrainer(t, busyHealth(time.Now()))
	a, _, _, _ := jobRunnerArbiter(t, trainer)

	req := httptest.NewRequest("POST", "/v1/fine_tuning/jobs/ftjob-x/cancel", nil)
	w := httptest.NewRecorder()
	a.makeRequestHandler("laya-trainer")(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status %d (%s), want 200 from the trainer", w.Code, w.Body.String())
	}
	found := false
	for _, s := range trainer.seen {
		found = found || s == "POST /v1/fine_tuning/jobs/ftjob-x/cancel"
	}
	if !found {
		t.Fatalf("the cancel did not reach the trainer: %v", trainer.seen)
	}
}

func TestJobRunner_EmptyBodyPostStillRefusedOnOtherBackends(t *testing.T) {
	trainer := newFakeTrainer(t, idleHealth)
	a, _, vb, _ := jobRunnerArbiter(t, trainer)
	vb.running = true

	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	w := httptest.NewRecorder()
	a.makeRequestHandler("vllm")(w, req)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status %d, want the existing 400 for an empty body on vllm", w.Code)
	}
}

// A typo'd base name must be refused before anything evicts the teacher.
func TestJobRunner_UnknownBaseIs404BeforeAnyLaunch(t *testing.T) {
	trainer := newFakeTrainer(t, idleHealth)
	a, tb, _, creates := jobRunnerArbiter(t, trainer)
	tb.running, tb.containerLaunched = false, false

	req := httptest.NewRequest("POST", "/v1/fine_tuning/jobs",
		strings.NewReader(`{"model":"laya-klingon","training_file":"ds-0123456789ab"}`))
	w := httptest.NewRecorder()
	a.makeRequestHandler("laya-trainer")(w, req)
	if w.Code != http.StatusNotFound || *creates != 0 {
		t.Fatalf("status %d, %d creates; want 404 and no launch", w.Code, *creates)
	}
}

func TestJobRunner_HealthShowsTheHold(t *testing.T) {
	started := time.Now().Add(-10 * time.Minute)
	trainer := newFakeTrainer(t, busyHealth(started))
	a, _, vb, _ := jobRunnerArbiter(t, trainer)
	vb.currentContext = 131072
	vb.currentSpec = &configSpeculative{Method: "qwen3_5_mtp", NumSpeculativeTokens: 3}

	w := httptest.NewRecorder()
	a.makeHealthHandler("laya-trainer")(w, httptest.NewRequest("GET", "/health", nil))
	var h struct {
		JobRunner struct {
			Status    string `json:"status"`
			Job       string `json:"job"`
			HoldUntil int64  `json:"hold_until"`
		} `json:"job_runner"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &h); err != nil {
		t.Fatal(err)
	}
	if h.JobRunner.Status != "busy" || h.JobRunner.Job != "ftjob-abc" {
		t.Errorf("job_runner %+v", h.JobRunner)
	}
	if want := started.Add(2 * time.Hour).Unix(); h.JobRunner.HoldUntil != want {
		t.Errorf("hold_until %d, want %d", h.JobRunner.HoldUntil, want)
	}

	w = httptest.NewRecorder()
	a.makeHealthHandler("vllm")(w, httptest.NewRequest("GET", "/health", nil))
	var v map[string]any
	_ = json.Unmarshal(w.Body.Bytes(), &v)
	if v["current_context"] != float64(131072) || v["current_spec"] != "qwen3_5_mtp/k=3" {
		t.Errorf("health current_context=%v current_spec=%v", v["current_context"], v["current_spec"])
	}
}

// --- configuration -------------------------------------------------------------

func TestLayaCatalogNames(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "laya-models.yaml")
	if err := os.WriteFile(p, []byte("schema_version: 1\nmodels:\n  - name: laya-multilingual\n  - name: laya-english\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	got := loadLayaModelNames(p)
	if strings.Join(got, ",") != "laya-multilingual,laya-english" {
		t.Fatalf("names %v", got)
	}
	if loadLayaModelNames(filepath.Join(dir, "missing.yaml")) != nil {
		t.Fatal("a missing catalog registers no names")
	}
}

func TestLayaCatalogNames_TheRealCatalog(t *testing.T) {
	got := loadLayaModelNames("../deploy/laya-models.yaml")
	if strings.Join(got, ",") != "laya-multilingual,laya-english" {
		t.Fatalf("deploy/laya-models.yaml names %v", got)
	}
}

func TestLayaTrainerBackendConfig(t *testing.T) {
	t.Setenv("LAYA_MAX_HOLD_S", "5400")
	bc := layaTrainerBackend("devai-net")
	if bc.Name != "laya-trainer" || bc.ListenPort != 11438 || !bc.JobRunner {
		t.Fatalf("config %+v", bc)
	}
	if bc.ContainerName != "devai-laya-trainer" || bc.Image != "localhost/devai-laya-trainer:latest" {
		t.Errorf("container %q image %q", bc.ContainerName, bc.Image)
	}
	if bc.BackendURL.String() != "http://laya-trainer:11434" || bc.HealthPath != "/health" {
		t.Errorf("url %s health %s", bc.BackendURL, bc.HealthPath)
	}
	if bc.ModelsDir != "/var/cache/devai/laya" || bc.MountDest != "/laya" || !bc.MountRW {
		t.Errorf("store %s -> %s rw=%v", bc.ModelsDir, bc.MountDest, bc.MountRW)
	}
	if bc.MaxHold != 90*time.Minute {
		t.Errorf("MaxHold %s, want LAYA_MAX_HOLD_S", bc.MaxHold)
	}
	args := strings.Join(bc.Entrypoint("laya-english", launchConfig{}), " ")
	if !strings.Contains(args, "laya_trainer.controller") || strings.Contains(args, "laya-english") {
		t.Errorf("entrypoint %q: must start the controller and ignore the model", args)
	}
}

// 900 s: about 6x the jobs measured in plan Phase 4 (123-140 s).
func TestLayaTrainerBackendConfig_DefaultHoldIsFifteenMinutes(t *testing.T) {
	t.Setenv("LAYA_MAX_HOLD_S", "") // empty = unset: the default applies
	if bc := layaTrainerBackend("devai-net"); bc.MaxHold != 15*time.Minute {
		t.Fatalf("default MaxHold %s", bc.MaxHold)
	}
}

// Re-review #2 LOW: a podman error (say a 500) decoded to status "" with
// ok=true, which the job-runner rules read as "gone": a silent busy trainer
// meeting a podman hiccup was evicted. Only a 404 means gone.
func TestContainerState_OnlyA404MeansGone(t *testing.T) {
	for _, tc := range []struct {
		code   int
		wantOK bool
	}{{http.StatusNotFound, true}, {http.StatusInternalServerError, false}} {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(tc.code)
			_, _ = w.Write([]byte(`{"cause":"x","message":"y","response":1}`))
		}))
		u := srv.URL
		a := &arbiter{podmanClient: &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			r.URL.Scheme, r.URL.Host = "http", strings.TrimPrefix(u, "http://")
			return http.DefaultTransport.RoundTrip(r)
		})}}
		status, _, ok := a.containerState("devai-laya-trainer")
		srv.Close()
		if ok != tc.wantOK || status != "" {
			t.Errorf("HTTP %d: status %q ok %v, want ok %v", tc.code, status, ok, tc.wantOK)
		}
	}
}
