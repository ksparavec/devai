package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
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
		proxy: newProxy(u, nil),
		// Default allowlist for tests: production code populates this
		// from modelsForBackend(cfg.Models, bc.Name) at startup. The
		// handler's allowlist check (HI4) refuses unknown vllm/sglang
		// model names with 404, so tests that hit the handler need at
		// least the names they POST in the test body. Individual tests
		// override modelNames where they need a different set.
		modelNames: []string{"test-model"},
		// recreateCond is bound to the arbiter's mutex in production code
		// (see backend construction in main()). testArbiter rebinds it
		// once the arbiter is constructed.
	}
}

func testArbiter(backends ...*backendState) *arbiter {
	a := &arbiter{
		backends:     make(map[string]*backendState),
		idleTimeout:  5 * time.Minute,
		drainTimeout: 5 * time.Second,
		modelSizes: map[string]map[string]float64{
			"ollama": {"test-model": 7.4},
			"vllm":   {"test-model": 7.4},
			"sglang": {"test-model": 7.4},
		},
		modelContexts: map[string]map[string]int{},
		totalVRAMGB:   24.0,
		maxContextLen: 131072,
		// Match production wiring so tests that exercise backendIsServing
		// don't NPE on a nil client.
		healthClient: &http.Client{Timeout: 2 * time.Second},
	}
	for _, bs := range backends {
		bs.recreateCond = sync.NewCond(&a.mu)
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
	// The handlers serve the VETTED subset now, not modelNames. Both are
	// vetted here so this test keeps testing what it was written for
	// (merge/dedup shape); TestModelsHandler_HidesUnvettedModels covers
	// the narrowing.
	bs.advertised = []string{"model-a", "model-b"}
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
	// Both vetted: this test is about merge/dedup between the live
	// passthrough and the configured list, not about narrowing.
	bs.advertised = []string{"live-model", "config-only-model"}
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

// --- TestSynthesizeHFFromCache ---

func TestSynthesizeHFFromCache_FilteringAndShape(t *testing.T) {
	parser := "qwen3_coder"
	cache := map[string]*hfCacheEntry{
		// Healthy model — fits at 24G/32K. Should produce a row.
		"nvidia/Llama-3.1-8B@bdb54e242984": {
			SchemaVersion: 2,
			Repo:          "nvidia/Llama-3.1-8B-Instruct-NVFP4",
			Sha:           "bdb54e242984",
			Aliases:       []string{"Llama-3.1-8B-Instruct-NVFP4"},
			ModelKind:     "dense",
			SizeGB:        5.61, // weight size; distinct from ActualVRAMGB (post-load)
			MaxContext:    131072,
			Capability:    CapInline,
			ToolParser:    &parser,
			Probes: map[string]map[string]hfCacheProbe{
				"24": {
					"32768": {Ctx: 32768, VramGB: 24, Fits: true,
						ActualVRAMGB: 22.49, ActualContext: 32768},
					"65536": {Ctx: 65536, VramGB: 24, Fits: false}, // skipped — not fits
				},
			},
		},
		// Custom-arch failure — must be dropped entirely.
		"talkie-lm/talkie@abc123": {
			SchemaVersion: 1,
			Repo:          "talkie-lm/talkie-1930-13b-it",
			Aliases:       []string{"talkie-1930"},
			Capability:    CapUnsupportedArch,
			Probes:        map[string]map[string]hfCacheProbe{},
		},
		// Infra failure — also dropped.
		"some/sglang-fp4@def456": {
			SchemaVersion: 1,
			Repo:          "some/sglang-fp4",
			Aliases:       []string{"sglang-fp4"},
			Capability:    CapError,
			Probes: map[string]map[string]hfCacheProbe{
				"24": {"32768": {Ctx: 32768, VramGB: 24, Fits: false}},
			},
		},
		// Probe at wrong VRAM band — dropped.
		"x/wrong-band@e": {
			SchemaVersion: 1,
			Repo:          "x/wrong-band",
			Aliases:       []string{"wrong-band"},
			Capability:    CapInline,
			Probes: map[string]map[string]hfCacheProbe{
				"16": {"32768": {Ctx: 32768, VramGB: 16, Fits: true}},
			},
		},
	}
	rows := synthesizeHFFromCache(cache, "vllm", 24, 131072, nil)
	if len(rows) != 1 {
		t.Fatalf("expected 1 emitted row, got %d: %+v", len(rows), rows)
	}
	r := rows[0]
	if r.Name != "Llama-3.1-8B-Instruct-NVFP4" {
		t.Errorf("Name=%q want Llama-3.1-8B-Instruct-NVFP4", r.Name)
	}
	if len(r.Backend) != 1 || r.Backend[0] != "vllm" {
		t.Errorf("Backend=%v want [vllm]", r.Backend)
	}
	if r.Context != 131072 {
		t.Errorf("Context=%d want 131072 (operator cap)", r.Context)
	}
	if r.ToolParser != "qwen3_coder" {
		t.Errorf("ToolParser=%q want qwen3_coder", r.ToolParser)
	}
	// Size MUST be the catalog weight size (5.61 GB), NOT the post-load
	// ActualVRAMGB (22.49 GB). containerRecreate consumes this for
	// memFraction; using the post-load total would clamp KV to a few
	// thousand tokens.
	if r.Size != "5.61 GB" {
		t.Errorf("Size=%q want \"5.61 GB\" (weight size, not post-load total)", r.Size)
	}
	if r.Reasoning == nil || r.Reasoning.Capability != CapInline {
		t.Errorf("Reasoning=%+v want capability=inline", r.Reasoning)
	}
}

// TestSynthesizeHFFromCache_SkipsMetaBlock locks in the Phase C invariant:
// the top-level `_meta` drift-stamp block must never become a serving row.
// Exercises the real decode path (JSON -> map[string]*hfCacheEntry -> synth)
// that loadHFCache uses, since `_meta` decodes into an aliasless zero entry.
func TestSynthesizeHFFromCache_SkipsMetaBlock(t *testing.T) {
	raw := `{
      "_meta": {
        "current_image_digest": "sha256:abc",
        "current_image_ref": "docker.io/vllm/vllm-openai:v0.22.1",
        "image_history": {"sha256:abc": {"image_ref": "docker.io/vllm/vllm-openai:v0.22.1", "first_seen": "2026-07-09T00:00:00+00:00"}}
      },
      "nvidia/Llama-3.1-8B@bdb54e242984": {
        "schema_version": 2,
        "repo": "nvidia/Llama-3.1-8B-Instruct-NVFP4",
        "sha": "bdb54e242984",
        "aliases": ["Llama-3.1-8B-Instruct-NVFP4"],
        "size_gb": 5.61,
        "max_context": 131072,
        "capability": "inline",
        "tool_parser": "qwen3_coder",
        "probes": {"24": {"32768": {"ctx": 32768, "vram_gb": 24, "fits": true, "actual_vram_gb": 22.49, "actual_context": 32768}}}
      }
    }`
	var cache map[string]*hfCacheEntry
	if err := json.Unmarshal([]byte(raw), &cache); err != nil {
		t.Fatalf("unmarshal failed: %v", err)
	}
	rows := synthesizeHFFromCache(cache, "vllm", 24, 131072, nil)
	if len(rows) != 1 {
		t.Fatalf("expected 1 row (model only, _meta skipped), got %d: %+v", len(rows), rows)
	}
	if rows[0].Name != "Llama-3.1-8B-Instruct-NVFP4" {
		t.Errorf("Name=%q want Llama-3.1-8B-Instruct-NVFP4", rows[0].Name)
	}
}

func TestSynthesizeHFFromCache_V2FieldsPropagated(t *testing.T) {
	// Phase 6: v2 cache entries carry reasoning_parser, tool_parser, and
	// disable_verified. The synthesizer must propagate all three onto
	// the configModel — the router's launch and policy paths read them
	// from there at containerRecreate / applyVLLMPolicy time.
	rp := "qwen3"
	tp := "hermes"
	dv := true
	cache := map[string]*hfCacheEntry{
		"nvidia/Qwen3-14B@deadbeef0001": {
			SchemaVersion:   2,
			Repo:            "nvidia/Qwen3-14B-NVFP4",
			Sha:             "deadbeef0001",
			Aliases:         []string{"Qwen3-14B-NVFP4"},
			ModelKind:       "dense",
			SizeGB:          7.4,
			MaxContext:      32768,
			Capability:      CapStructured,
			ReasoningParser: &rp,
			ToolParser:      &tp,
			DisableVerified: tristateBool{v: &dv},
			Probes: map[string]map[string]hfCacheProbe{
				"24": {
					"32768": {Ctx: 32768, VramGB: 24, Fits: true,
						ActualVRAMGB: 21.0, ActualContext: 32768},
				},
			},
		},
	}
	rows := synthesizeHFFromCache(cache, "vllm", 24, 131072, nil)
	if len(rows) != 1 {
		t.Fatalf("expected 1 row, got %d", len(rows))
	}
	r := rows[0]
	if r.ReasoningParser != "qwen3" {
		t.Errorf("ReasoningParser=%q want qwen3", r.ReasoningParser)
	}
	if r.ToolParser != "hermes" {
		t.Errorf("ToolParser=%q want hermes", r.ToolParser)
	}
	if r.Reasoning == nil ||
		r.Reasoning.DisableVerified == nil ||
		!*r.Reasoning.DisableVerified {
		t.Errorf("DisableVerified must propagate as true, got %+v", r.Reasoning)
	}
	if r.Reasoning == nil || r.Reasoning.Capability != CapStructured {
		t.Errorf("Capability=%v want structured", r.Reasoning)
	}
}

func TestSynthesizeHFFromCache_V1Rejected(t *testing.T) {
	// v1 cache entries lack DisableVerified. Serving them would mean the
	// router runs vLLM/SGLang with no --tool-call-parser flag and
	// maybeStripTools then silently drops every tools/tool_choice the
	// agent sends -- a corrupt-by-omission state with no log entry.
	// The synthesizer must REFUSE v1 entries; the operator re-probes
	// with `make probe-vllm` / `make probe-sglang` to upgrade.
	tp := "llama3_json"
	cache := map[string]*hfCacheEntry{
		"nvidia/llama@v1deadbeef": {
			SchemaVersion: 1,
			Repo:          "nvidia/Llama-3.1-8B-Instruct-NVFP4",
			Sha:           "v1deadbeef",
			Aliases:       []string{"Llama-3.1-8B-Instruct-NVFP4"},
			ModelKind:     "dense",
			SizeGB:        5.6,
			MaxContext:    131072,
			Capability:    CapInline,
			ToolParser:    &tp,
			Probes: map[string]map[string]hfCacheProbe{
				"24": {
					"32768": {Ctx: 32768, VramGB: 24, Fits: true,
						ActualVRAMGB: 19.0, ActualContext: 32768},
				},
			},
		},
	}
	rows := synthesizeHFFromCache(cache, "vllm", 24, 131072, nil)
	if len(rows) != 0 {
		t.Fatalf("v1 entry must be rejected, got %d rows", len(rows))
	}
}

func TestSynthesizeHFFromCache_FallbackWhenSizeGBMissing(t *testing.T) {
	// Pre-fix cache rows lack SizeGB. The fallback (half of ActualVRAMGB)
	// keeps launch math safe-ish until the operator re-probes. This test
	// pins that fallback so a future "remove the fallback" change
	// surfaces in CI as an intentional behavioural change.
	cache := map[string]*hfCacheEntry{
		"r/m@abc": {
			SchemaVersion: 2,
			Aliases:       []string{"m"},
			MaxContext:    65536,
			Capability:    CapInline,
			// SizeGB intentionally missing
			Probes: map[string]map[string]hfCacheProbe{
				"24": {"32768": {Ctx: 32768, VramGB: 24, Fits: true,
					ActualVRAMGB: 22.0, ActualContext: 32768}},
			},
		},
	}
	rows := synthesizeHFFromCache(cache, "vllm", 24, 65536, nil)
	if len(rows) != 1 {
		t.Fatalf("want 1 row, got %d", len(rows))
	}
	// Fallback: 0.5 × ActualVRAMGB = 11.0 GB
	if rows[0].Size != "11.00 GB" {
		t.Errorf("Size=%q want 11.00 GB (fallback for pre-fix cache)", rows[0].Size)
	}
}

func TestSynthesizeHFFromCache_OperatorCtxCapClamps(t *testing.T) {
	cache := map[string]*hfCacheEntry{
		"x/y@z": {
			SchemaVersion: 2,
			Aliases:       []string{"y"},
			MaxContext:    262144, // model declares big ceiling
			Capability:    CapInline,
			Probes: map[string]map[string]hfCacheProbe{
				"24": {
					"32768":  {Ctx: 32768, VramGB: 24, Fits: true},
					"65536":  {Ctx: 65536, VramGB: 24, Fits: true},
					"131072": {Ctx: 131072, VramGB: 24, Fits: true},
					"262144": {Ctx: 262144, VramGB: 24, Fits: true},
				},
			},
		},
	}
	rows := synthesizeHFFromCache(cache, "vllm", 24, 65536, nil) // operator caps at 64K
	if len(rows) != 1 {
		t.Fatalf("want 1 row, got %d", len(rows))
	}
	if rows[0].Context != 65536 {
		t.Errorf("Context=%d want 65536 (operator cap)", rows[0].Context)
	}
}

// TestSynthesizeHFFromCache_ServingOkGate verifies the serving-time LOAD
// probe gate: a cell that loaded (fits=true) but OOMed under a near-full
// context request (serving_ok=false) must NOT raise the per-name context
// cap. The cap settles on the largest tier that both fits AND serves.
// This is the DiffusionGemma regression: the fit probe said fits@256K,
// the load probe says serving_ok=false there, so the router must cap the
// model at the highest serving-verified tier instead of advertising 256K.
func TestSynthesizeHFFromCache_ServingOkGate(t *testing.T) {
	servingOK := true
	servingBad := false
	cache := map[string]*hfCacheEntry{
		"vendor/diffgemma@cafe": {
			SchemaVersion: 2,
			Repo:          "vendor/diffgemma",
			Aliases:       []string{"diffgemma"},
			MaxContext:    262144,
			Capability:    CapInline,
			Probes: map[string]map[string]hfCacheProbe{
				"24": {
					// Loads AND serves at 32K/64K.
					"32768": {Ctx: 32768, VramGB: 24, Fits: true, ServingOk: &servingOK},
					"65536": {Ctx: 65536, VramGB: 24, Fits: true, ServingOk: &servingOK},
					// Loads but OOMs under load at 128K/256K — must be excluded.
					"131072": {Ctx: 131072, VramGB: 24, Fits: true, ServingOk: &servingBad},
					"262144": {Ctx: 262144, VramGB: 24, Fits: true, ServingOk: &servingBad},
				},
			},
		},
	}
	rows := synthesizeHFFromCache(cache, "vllm", 24, 262144, nil)
	if len(rows) != 1 {
		t.Fatalf("want 1 row, got %d", len(rows))
	}
	// ProbedMaxCtx is the runtime request ceiling (applyProbeCeiling caps
	// each request to it). It must settle on the largest serving-verified
	// tier — 128K/256K fit but serving_ok=false, so 64K wins.
	if rows[0].ProbedMaxCtx != 65536 {
		t.Errorf("ProbedMaxCtx=%d want 65536 (largest serving-verified tier; "+
			"128K/256K fit but serving_ok=false)", rows[0].ProbedMaxCtx)
	}
}

// TestSynthesizeHFFromCache_NilServingOkIsLegacy verifies that a cell
// with no load-probe data (ServingOk == nil) gates on the fit verdict
// alone — byte-for-byte the pre-load-probe behaviour. A model probed for
// fit but never load-probed must still advertise its largest fitting ctx.
func TestSynthesizeHFFromCache_NilServingOkIsLegacy(t *testing.T) {
	cache := map[string]*hfCacheEntry{
		"x/y@z": {
			SchemaVersion: 2,
			Aliases:       []string{"y"},
			MaxContext:    131072,
			Capability:    CapInline,
			Probes: map[string]map[string]hfCacheProbe{
				"24": {
					"32768":  {Ctx: 32768, VramGB: 24, Fits: true},  // ServingOk nil
					"131072": {Ctx: 131072, VramGB: 24, Fits: true}, // ServingOk nil
				},
			},
		},
	}
	rows := synthesizeHFFromCache(cache, "vllm", 24, 262144, nil)
	if len(rows) != 1 {
		t.Fatalf("want 1 row, got %d", len(rows))
	}
	if rows[0].ProbedMaxCtx != 131072 {
		t.Errorf("ProbedMaxCtx=%d want 131072 (fit-only gate when load "+
			"probe never ran)", rows[0].ProbedMaxCtx)
	}
}

// TestSynthesizeHFFromCache_AllServingFailDropsRow verifies the drop
// path: a model whose ONLY fitting cell loads but OOMs under load
// (serving_ok=false everywhere) must not be advertised at all. bestCtx
// stays 0 and the row is dropped — handing an agent a model that
// crash-loops on every request is worse than hiding it. This is the
// pure DiffusionGemma case if even its smallest tier failed under load.
func TestSynthesizeHFFromCache_AllServingFailDropsRow(t *testing.T) {
	bad := false
	cache := map[string]*hfCacheEntry{
		"vendor/unservable@beef": {
			SchemaVersion: 2,
			Repo:          "vendor/unservable",
			Aliases:       []string{"unservable"},
			MaxContext:    32768,
			Capability:    CapInline,
			Probes: map[string]map[string]hfCacheProbe{
				"24": {
					"32768": {Ctx: 32768, VramGB: 24, Fits: true, ServingOk: &bad},
				},
			},
		},
	}
	rows := synthesizeHFFromCache(cache, "vllm", 24, 131072, nil)
	if len(rows) != 0 {
		t.Errorf("want 0 rows (model fits but never serves), got %d: %+v",
			len(rows), rows)
	}
}

// --- TestGPUExclusion ---

func TestEnsureBackendRunning_OllamaAlwaysSucceeds(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer server.Close()

	bs := testBackend("ollama", server)
	a := testArbiter(bs)

	a.mu.Lock()
	err := a.ensureBackendRunning(bs, "", 0, false, nil)
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
	bs.config.Entrypoint = func(m string, lc launchConfig) []string { return []string{"echo", m} }
	a := testArbiter(bs)

	a.mu.Lock()
	err := a.ensureBackendRunning(bs, "", 0, false, nil)
	a.mu.Unlock()

	if err == nil {
		t.Error("expected error when model name is empty for vllm")
	}
}

// --- TestEnsureBackendRunning_RecreateCoalescing ---
//
// Regression for the Claude Code split-model issue. Without
// recreate-coalescing, when two requests for the same model arrive
// while a recreate is in flight (lock released for the 50–60s
// `waitForHealthy` window), both observed `bs.running=false` and fired
// duplicate `podman rm` + `podman create` cycles — the second tearing
// down the first's half-built container. Verified by running this test
// against the pre-fix code: both goroutines proceeded into
// containerRecreate, podman errors leaked back through both, and
// `bs.running` was clobbered.

func TestEnsureBackendRunning_CoalescesConcurrentSameModelRecreates(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer server.Close()

	bs := testBackend("vllm", server)
	a := testArbiter(bs)

	// Simulate an in-flight recreate held open by an external goroutine.
	// (In production this state is set by the goroutine that won the
	// recreate race; here we set it directly so the orchestration is
	// deterministic.)
	a.mu.Lock()
	bs.recreating = true
	bs.pendingModel = "test-model"
	bs.pendingContext = 32768
	a.mu.Unlock()

	// Two concurrent requests for the same (model, ctx) the in-flight
	// recreate is targeting. Both must park on bs.recreateCond.
	var wg sync.WaitGroup
	var errA, errB error
	wg.Add(2)
	go func() {
		defer wg.Done()
		a.mu.Lock()
		errA = a.ensureBackendRunning(bs, "test-model", 32768, true, nil)
		a.mu.Unlock()
	}()
	go func() {
		defer wg.Done()
		a.mu.Lock()
		errB = a.ensureBackendRunning(bs, "test-model", 32768, true, nil)
		a.mu.Unlock()
	}()

	// Give both goroutines time to enter the cond.Wait().
	time.Sleep(50 * time.Millisecond)

	// Simulate the in-flight recreate completing successfully — exactly
	// the state the winning goroutine would commit before broadcasting.
	a.mu.Lock()
	bs.recreating = false
	bs.pendingModel = ""
	bs.pendingContext = 0
	bs.running = true
	bs.currentModel = "test-model"
	bs.currentContext = 32768
	bs.recreateCond.Broadcast()
	a.mu.Unlock()

	wg.Wait()

	if errA != nil {
		t.Errorf("goroutine A: unexpected error %v", errA)
	}
	if errB != nil {
		t.Errorf("goroutine B: unexpected error %v", errB)
	}
	// Critically, neither A nor B should have fired its own recreate —
	// they should have observed the in-flight recreate's result and
	// returned cleanly. State must match what the "completer" set.
	if !bs.running {
		t.Error("expected bs.running=true post-wakeup")
	}
	if bs.currentModel != "test-model" {
		t.Errorf("bs.currentModel=%q want test-model", bs.currentModel)
	}
	if bs.currentContext != 32768 {
		t.Errorf("bs.currentContext=%d want 32768", bs.currentContext)
	}
	if bs.recreating {
		t.Error("bs.recreating should be false after broadcast")
	}
}

// Two requests, recreate completes with FAILURE (e.g. health timeout).
// Waiters must wake up, see recreating=false + running=false, and be
// free to re-evaluate. This pins the defer-broadcast on error paths.
func TestEnsureBackendRunning_WaitersUnblockOnRecreateFailure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer server.Close()

	bs := testBackend("vllm", server)
	a := testArbiter(bs)

	a.mu.Lock()
	bs.recreating = true
	bs.pendingModel = "test-model"
	bs.pendingContext = 32768
	a.mu.Unlock()

	parked := make(chan struct{})
	done := make(chan struct{})

	go func() {
		a.mu.Lock()
		// Signal that we're about to enter the wait loop. (Locked here
		// so the test can't broadcast before this goroutine parks —
		// the orchestrator must contend with us for the lock.)
		close(parked)
		// We pass an empty model name here so that, post-wakeup, the
		// "model name required" check fires immediately and
		// ensureBackendRunning returns without calling the real
		// containerRecreate (which would touch podman).
		_ = a.ensureBackendRunning(bs, "", 0, false, nil)
		a.mu.Unlock()
		close(done)
	}()

	<-parked
	time.Sleep(50 * time.Millisecond) // ensure cond.Wait() has actually parked

	// Simulate the in-flight recreate FAILING — the winning goroutine's
	// defer would clear recreating + broadcast even on error, leaving
	// running=false.
	a.mu.Lock()
	bs.recreating = false
	bs.pendingModel = ""
	bs.pendingContext = 0
	bs.recreateCond.Broadcast()
	a.mu.Unlock()

	select {
	case <-done:
		// Waiter unblocked — fix is working.
	case <-time.After(2 * time.Second):
		t.Fatal("waiter never unblocked after recreate failure broadcast")
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
	// upstreamReqs, not activeReqs: drain only waits on requests already
	// proxied upstream (see drainBackend).
	atomic.StoreInt64(&bs.upstreamReqs, 1)
	a := &arbiter{drainTimeout: 2 * time.Second}

	go func() {
		time.Sleep(200 * time.Millisecond)
		atomic.StoreInt64(&bs.upstreamReqs, 0)
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
	atomic.StoreInt64(&bs.upstreamReqs, 1)
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
	proxy := newSmartProxy(u, false, nil)

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
	proxy := newSmartProxy(u, false, nil)

	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	w := httptest.NewRecorder()
	proxy.ServeHTTP(w, req)

	if w.Code != http.StatusInternalServerError {
		t.Errorf("expected 500, got %d", w.Code)
	}
}

// TestSmartProxy_ImageStaleSetsWarningHeader verifies the Phase C drift
// signal: when a backend's image has drifted from its probe baseline, every
// proxied response carries X-DevAI-Warning (advisory, non-blocking).
func TestSmartProxy_ImageStaleSetsWarningHeader(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, `{"ok":true}`)
	}))
	defer backend.Close()

	u, _ := url.Parse(backend.URL)

	// Not stale -> no header.
	fresh := newSmartProxy(u, false, nil)
	wf := httptest.NewRecorder()
	fresh.ServeHTTP(wf, httptest.NewRequest("POST", "/v1/chat/completions", nil))
	if got := wf.Header().Get("X-DevAI-Warning"); got != "" {
		t.Errorf("fresh backend set X-DevAI-Warning=%q, want empty", got)
	}

	// Stale -> header present.
	stale := newSmartProxy(u, true, nil)
	ws := httptest.NewRecorder()
	stale.ServeHTTP(ws, httptest.NewRequest("POST", "/v1/chat/completions", nil))
	if got := ws.Header().Get("X-DevAI-Warning"); got == "" {
		t.Error("stale backend did not set X-DevAI-Warning")
	}
	if ws.Code != http.StatusOK {
		t.Errorf("stale backend altered status: got %d, want 200", ws.Code)
	}
}

// TestNormalizeImageDigest covers the pure libpod-inspect -> bare-digest
// reducer that decides `probed != running`. It must byte-match the prober's
// image_digest_via_cli fallback ordering (.Digest first, then RepoDigests[0]
// tail) or every backend would flip to a false "stale" advisory.
func TestNormalizeImageDigest(t *testing.T) {
	tests := []struct {
		name        string
		digest      string
		repoDigests []string
		want        string
	}{
		{"digest populated wins", "sha256:aaa", []string{"repo@sha256:bbb"}, "sha256:aaa"},
		{"fallback to repodigest tail", "", []string{"docker.io/vllm/vllm-openai@sha256:bbb"}, "sha256:bbb"},
		{"repodigest without at-sign", "", []string{"sha256:ccc"}, "sha256:ccc"},
		{"neither present", "", nil, ""},
		{"empty repodigests slice", "", []string{}, ""},
		{"garbage digest, no sha", "not-a-digest", []string{"also-garbage"}, ""},
		{"first repodigest chosen over later", "", []string{"a@sha256:first", "b@sha256:second"}, "sha256:first"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := normalizeImageDigest(tt.digest, tt.repoDigests); got != tt.want {
				t.Errorf("normalizeImageDigest(%q, %v) = %q, want %q",
					tt.digest, tt.repoDigests, got, tt.want)
			}
		})
	}
}

// TestReadProbedImageDigest covers the _meta digest extractor: a stamped
// cache returns its digest, a pre-Phase-C cache (no _meta) and a missing file
// both fail open to "".
func TestReadProbedImageDigest(t *testing.T) {
	dir := t.TempDir()

	stamped := filepath.Join(dir, "stamped.json")
	os.WriteFile(stamped, []byte(`{"_meta":{"current_image_digest":"sha256:abc123"},`+
		`"repo@sha":{"schema_version":2}}`), 0o644)
	if got := readProbedImageDigest(stamped); got != "sha256:abc123" {
		t.Errorf("stamped cache: got %q, want sha256:abc123", got)
	}

	legacy := filepath.Join(dir, "legacy.json")
	os.WriteFile(legacy, []byte(`{"repo@sha":{"schema_version":2}}`), 0o644)
	if got := readProbedImageDigest(legacy); got != "" {
		t.Errorf("legacy cache: got %q, want empty", got)
	}

	if got := readProbedImageDigest(filepath.Join(dir, "nope.json")); got != "" {
		t.Errorf("missing file: got %q, want empty", got)
	}
	if got := readProbedImageDigest(""); got != "" {
		t.Errorf("empty path: got %q, want empty", got)
	}
}

// --- Entrypoint builders: parser flag emission ---

func sliceContains(haystack []string, needles ...string) bool {
	if len(needles) == 0 {
		return true
	}
	for i := 0; i <= len(haystack)-len(needles); i++ {
		match := true
		for j, n := range needles {
			if haystack[i+j] != n {
				match = false
				break
			}
		}
		if match {
			return true
		}
	}
	return false
}

func TestVLLMEntrypoint_OmitsParserFlagsWhenEmpty(t *testing.T) {
	args := vllmEntrypoint("Qwen3.5-9B-NVFP4", launchConfig{
		MemFraction: 0.9, MaxContext: 32768,
	})
	for _, flag := range []string{"--reasoning-parser", "--tool-call-parser", "--enable-auto-tool-choice"} {
		for _, a := range args {
			if a == flag {
				t.Fatalf("expected %s absent, got args=%v", flag, args)
			}
		}
	}
}

func TestVLLMEntrypoint_EmitsBothParserFlags(t *testing.T) {
	args := vllmEntrypoint("Qwen3.5-9B-NVFP4", launchConfig{
		MemFraction: 0.9, MaxContext: 32768,
		ReasoningParser: "qwen3", ToolParser: "hermes",
	})
	if !sliceContains(args, "--reasoning-parser", "qwen3") {
		t.Errorf("--reasoning-parser qwen3 missing: %v", args)
	}
	if !sliceContains(args, "--enable-auto-tool-choice") {
		t.Errorf("--enable-auto-tool-choice missing: %v", args)
	}
	if !sliceContains(args, "--tool-call-parser", "hermes") {
		t.Errorf("--tool-call-parser hermes missing: %v", args)
	}
}

func TestOllamaEntrypoint_ServeRegardlessOfModel(t *testing.T) {
	args := ollamaEntrypoint("qwen3.6:27b-q4_K_M", launchConfig{MaxContext: 65536})
	if len(args) != 2 || args[0] != "/bin/ollama" || args[1] != "serve" {
		t.Fatalf("ollamaEntrypoint = %v, want [/bin/ollama serve]", args)
	}
}

func TestBuildContainerSpec_OllamaBakesContextAndRWMount(t *testing.T) {
	cfg := backendConfig{
		Name:          "ollama",
		ContainerName: "devai-ollama",
		Image:         "docker.io/ollama/ollama:latest",
		ModelsDir:     "/var/cache/devai/ollama",
		MountDest:     "/root/.ollama",
		MountRW:       true,
		Network:       "devai-net",
		Entrypoint:    ollamaEntrypoint,
		EnvVars:       map[string]string{"OLLAMA_MAX_LOADED_MODELS": "1"},
		DynamicEnv: func(lc launchConfig) map[string]string {
			return map[string]string{"OLLAMA_CONTEXT_LENGTH": fmt.Sprintf("%d", lc.MaxContext)}
		},
	}
	spec := buildContainerSpec(cfg, "qwen3.6:27b-q4_K_M", launchConfig{MaxContext: 65536}, nil, nil)

	mounts, ok := spec["mounts"].([]map[string]any)
	if !ok || len(mounts) == 0 {
		t.Fatalf("spec mounts missing/wrong type: %#v", spec["mounts"])
	}
	if mounts[0]["destination"] != "/root/.ollama" {
		t.Errorf("mount destination = %v, want /root/.ollama", mounts[0]["destination"])
	}
	if opts, _ := mounts[0]["options"].([]string); len(opts) != 1 || opts[0] != "rw" {
		t.Errorf("mount options = %v, want [rw]", mounts[0]["options"])
	}

	envMap, ok := spec["env"].(map[string]string)
	if !ok {
		t.Fatalf("spec env missing/wrong type: %#v", spec["env"])
	}
	if envMap["OLLAMA_CONTEXT_LENGTH"] != "65536" {
		t.Errorf("OLLAMA_CONTEXT_LENGTH = %q, want 65536", envMap["OLLAMA_CONTEXT_LENGTH"])
	}
	if envMap["OLLAMA_MAX_LOADED_MODELS"] != "1" {
		t.Errorf("static env dropped: %v", envMap)
	}

	if ep, _ := spec["entrypoint"].([]string); len(ep) != 2 || ep[0] != "/bin/ollama" || ep[1] != "serve" {
		t.Errorf("entrypoint = %v, want [/bin/ollama serve]", spec["entrypoint"])
	}
}

func TestBuildContainerSpec_DefaultMountIsModelsReadOnly(t *testing.T) {
	cfg := backendConfig{
		Name:          "vllm",
		ContainerName: "devai-vllm",
		Image:         "img",
		ModelsDir:     "/var/cache/devai/vllm",
		Entrypoint:    func(string, launchConfig) []string { return []string{"x"} },
	}
	spec := buildContainerSpec(cfg, "m", launchConfig{MaxContext: 32768}, nil, nil)
	mounts := spec["mounts"].([]map[string]any)
	if mounts[0]["destination"] != "/models" {
		t.Errorf("default mount destination = %v, want /models", mounts[0]["destination"])
	}
	if opts, _ := mounts[0]["options"].([]string); len(opts) != 1 || opts[0] != "ro" {
		t.Errorf("default mount options = %v, want [ro]", mounts[0]["options"])
	}
}

func TestSGLangEntrypoint_OmitsParserFlagsWhenEmpty(t *testing.T) {
	args := sglangEntrypoint("Qwen3.5-9B-NVFP4", launchConfig{
		MemFraction: 0.85, MaxContext: 32768,
	})
	for _, flag := range []string{"--reasoning-parser", "--tool-call-parser"} {
		for _, a := range args {
			if a == flag {
				t.Fatalf("expected %s absent, got args=%v", flag, args)
			}
		}
	}
}

func TestSGLangEntrypoint_EmitsBothParserFlags(t *testing.T) {
	args := sglangEntrypoint("Qwen3.5-9B-NVFP4", launchConfig{
		MemFraction: 0.85, MaxContext: 32768,
		ReasoningParser: "qwen3", ToolParser: "qwen25",
	})
	if !sliceContains(args, "--reasoning-parser", "qwen3") {
		t.Errorf("--reasoning-parser qwen3 missing: %v", args)
	}
	if !sliceContains(args, "--tool-call-parser", "qwen25") {
		t.Errorf("--tool-call-parser qwen25 missing: %v", args)
	}
	// SGLang has no --enable-auto-tool-choice analogue — confirm we
	// don't accidentally emit it.
	for _, a := range args {
		if a == "--enable-auto-tool-choice" {
			t.Fatalf("SGLang must not emit --enable-auto-tool-choice: %v", args)
		}
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

// --- TestParseSizeGB ---

func TestParseSizeGB(t *testing.T) {
	tests := []struct {
		input string
		want  float64
	}{
		{"7.4 GB", 7.4},
		{"17 GB", 17.0},
		{"2.0 GB", 2.0},
		{"58 GB", 58.0},
		{"", 0},
		{"invalid", 0},
		{" 10 GB ", 10.0},
	}
	for _, tt := range tests {
		got := parseSizeGB(tt.input)
		if got != tt.want {
			t.Errorf("parseSizeGB(%q) = %v, want %v", tt.input, got, tt.want)
		}
	}
}

// --- TestMemFraction ---

func TestMemFraction(t *testing.T) {
	tests := []struct {
		name        string
		modelSizeGB float64
		totalVRAMGB float64
		backend     string
		wantMin     float64
		wantMax     float64
	}{
		// Small model on 24 GB — plenty of room
		{"small_vllm_24", 7.4, 24, "vllm", 0.90, 0.95},
		{"small_sglang_24", 7.4, 24, "sglang", 0.85, 0.90},
		// Medium model on 24 GB
		{"medium_vllm_24", 17, 24, "vllm", 0.90, 0.95},
		{"medium_sglang_24", 17, 24, "sglang", 0.85, 0.90},
		// Tight fit on 24 GB — fraction approaches max
		{"tight_vllm_24", 22, 24, "vllm", 0.80, 0.95},
		{"tight_sglang_24", 22, 24, "sglang", 0.80, 0.95},
		// Large GPU — clamped at 0.95
		{"small_vllm_80", 7.4, 80, "vllm", 0.95, 0.95},
		// Unknown model size — conservative defaults
		{"unknown_vllm", 0, 24, "vllm", 0.85, 0.92},
		{"unknown_sglang", 0, 24, "sglang", 0.78, 0.85},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := memFraction(tt.modelSizeGB, tt.totalVRAMGB, tt.backend)
			if got < tt.wantMin || got > tt.wantMax {
				t.Errorf("memFraction(%.1f, %.0f, %s) = %.2f, want [%.2f, %.2f]",
					tt.modelSizeGB, tt.totalVRAMGB, tt.backend, got, tt.wantMin, tt.wantMax)
			}
		})
	}
}

func TestMemFraction_SGLangLowerThanVLLM(t *testing.T) {
	// SGLang reserves more for RadixAttention, so its fraction should be lower
	vllm := memFraction(10, 24, "vllm")
	sglang := memFraction(10, 24, "sglang")
	if sglang >= vllm {
		t.Errorf("sglang fraction (%.2f) should be lower than vllm (%.2f)", sglang, vllm)
	}
}

// --- TestComputeLaunchConfig ---

func TestComputeLaunchConfig_ContextFitsInKVCache(t *testing.T) {
	// 7.4 GB model on 24 GB GPU — plenty of room for 128K context
	lc := computeLaunchConfig(7.4, 24, "vllm", 131072)
	if lc.MaxContext < 32768 {
		t.Errorf("expected at least 32K context for small model, got %d", lc.MaxContext)
	}
	if lc.MaxContext > 131072 {
		t.Errorf("context should not exceed declared max, got %d", lc.MaxContext)
	}
}

func TestComputeLaunchConfig_TightFitReducesContext(t *testing.T) {
	// 22 GB model on 24 GB GPU — very little KV cache room
	lc := computeLaunchConfig(22, 24, "vllm", 131072)
	if lc.MaxContext >= 131072 {
		t.Errorf("tight model should reduce context below 128K, got %d", lc.MaxContext)
	}
	if lc.MaxContext < 4096 {
		t.Errorf("context should be at least 4K, got %d", lc.MaxContext)
	}
}

func TestComputeLaunchConfig_ModelDeclaredContextRespected(t *testing.T) {
	// Model declares 32K max even though GPU has room for more
	lc := computeLaunchConfig(7.4, 24, "vllm", 32768)
	if lc.MaxContext > 32768 {
		t.Errorf("should respect declared max of 32K, got %d", lc.MaxContext)
	}
}

func TestComputeLaunchConfig_SGLangReservesMoreMemory(t *testing.T) {
	vllm := computeLaunchConfig(10, 24, "vllm", 131072)
	sglang := computeLaunchConfig(10, 24, "sglang", 131072)
	if sglang.MaxContext >= vllm.MaxContext {
		t.Errorf("sglang should have lower max context due to RadixAttention overhead: vllm=%d sglang=%d",
			vllm.MaxContext, sglang.MaxContext)
	}
}

// --- TestApplyProbeCeiling ---
//
// Regression tests for the gpt-oss-20b incident: fittableContext's
// heuristic table assigns 256 KB/token to 12–20 GB models, which
// collapses gpt-oss-20b's probe-verified 256K ceiling (22.30 GB measured
// at host_vram=24) to ~36K. The probe is the source of truth — the
// router must trust it.

func TestApplyProbeCeiling_TrustsProbeOverHeuristic(t *testing.T) {
	// gpt-oss-20b @ 24G: heuristic says 36864, probe says 262144 fits.
	// Request 256K → router must launch with 256K, not 36K.
	got := applyProbeCeiling(36864, 262144, 262144)
	if got != 262144 {
		t.Fatalf("probe-verified 256K must override heuristic 36K, got %d", got)
	}
}

func TestApplyProbeCeiling_RequestBelowProbeMax_UsesRequest(t *testing.T) {
	// User picks 64K even though probe verified 256K — honour the request.
	got := applyProbeCeiling(36864, 65536, 262144)
	if got != 65536 {
		t.Fatalf("requestedCtx 64K must win when ≤ probedMax, got %d", got)
	}
}

func TestApplyProbeCeiling_RequestAboveProbeMax_ClampsToProbe(t *testing.T) {
	// User picks 256K but probe only verified 128K — clamp to probe ceiling.
	got := applyProbeCeiling(36864, 262144, 131072)
	if got != 131072 {
		t.Fatalf("requestedCtx 256K must clamp to probedMax 128K, got %d", got)
	}
}

func TestApplyProbeCeiling_NoProbeData_FallsBackToHeuristic(t *testing.T) {
	// Pre-probe / YAML-only models keep the heuristic — there's no
	// authoritative ceiling to trust.
	got := applyProbeCeiling(36864, 262144, 0)
	if got != 36864 {
		t.Fatalf("probedMax=0 must keep heuristic, got %d", got)
	}
}

// --- resolveKVCacheType ---

func TestResolveKVCacheType(t *testing.T) {
	// qwen3.6:35b-a3b-mtp @ 24G: 32K/64K probed under f16 (one legacy
	// cell with no stamp), 128K only fits under q8_0.
	mixed := map[int]string{32768: "", 65536: "f16", 131072: "q8_0"}
	cases := []struct {
		name    string
		kvByCtx map[int]string
		ctx     int
		want    string
	}{
		{"covering tier returns its raw f16 stamp", mixed, 65536, "f16"},
		{"legacy unstamped 32K cell stays empty", mixed, 32768, ""},
		{"128K resolves to q8_0", mixed, 131072, "q8_0"},
		{"off-grid ctx picks smallest covering tier", mixed, 100000, "q8_0"},
		{"off-grid ctx under 64K covers to f16 tier", mixed, 40000, "f16"},
		{"no covering tier means empty", mixed, 262144, ""},
		{"nil map means empty", nil, 65536, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := resolveKVCacheType(tc.kvByCtx, tc.ctx); got != tc.want {
				t.Fatalf("resolveKVCacheType(%v, %d) = %q, want %q",
					tc.kvByCtx, tc.ctx, got, tc.want)
			}
		})
	}
}

func TestOllamaLaunchCtx(t *testing.T) {
	cases := []struct {
		name            string
		probed, desired int
		want            int
	}{
		{"no override launches at ceiling", 131072, 0, 131072},
		{"override below ceiling pins smaller tier", 131072, 65536, 65536},
		{"override equal to ceiling is a no-op", 131072, 131072, 131072},
		{"override above ceiling clamps to ceiling", 131072, 262144, 131072},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := ollamaLaunchCtx(tc.probed, tc.desired); got != tc.want {
				t.Fatalf("ollamaLaunchCtx(%d, %d) = %d, want %d",
					tc.probed, tc.desired, got, tc.want)
			}
		})
	}
}

func TestSynthesizeFromCache_KVByCtx(t *testing.T) {
	// Mixed-KV band: 64K f16-stamped, 128K q8_0, 256K q8_0 but NOT fully
	// on GPU. KVByCtx must contain only the fully-on-GPU tiers -- a
	// non-fitting cell's dtype must never influence launch env.
	tr := true
	cache := map[string]*cacheEntry{
		"digest1": {
			SchemaVersion:   3,
			Digest:          "digest1",
			Aliases:         []string{"mixed:latest"},
			MaxContext:      262144,
			Capability:      CapStructured,
			DisableVerified: tristateBool{v: &tr},
			Probes: map[string]map[string]cacheProbe{
				"24": {
					"65536":  {Ctx: 65536, VramGB: 24, ActualTotalGB: 20.7, FullyOnGPU: true, KVCacheType: "f16"},
					"131072": {Ctx: 131072, VramGB: 24, ActualTotalGB: 20.7, FullyOnGPU: true, KVCacheType: "q8_0"},
					"262144": {Ctx: 262144, VramGB: 24, FullyOnGPU: false, KVCacheType: "q8_0"},
				},
			},
		},
	}
	models := synthesizeFromCache(cache, 24, 262144)
	if len(models) != 1 {
		t.Fatalf("expected 1 synthesized model, got %d", len(models))
	}
	m := models[0]
	if m.ProbedMaxCtx != 131072 {
		t.Fatalf("ProbedMaxCtx = %d, want 131072 (non-fitting 256K excluded)", m.ProbedMaxCtx)
	}
	want := map[int]string{65536: "f16", 131072: "q8_0"}
	if len(m.KVByCtx) != len(want) {
		t.Fatalf("KVByCtx = %v, want %v", m.KVByCtx, want)
	}
	for ctx, kv := range want {
		if m.KVByCtx[ctx] != kv {
			t.Fatalf("KVByCtx[%d] = %q, want %q", ctx, m.KVByCtx[ctx], kv)
		}
	}
	if _, ok := m.KVByCtx[262144]; ok {
		t.Fatalf("non-fitting 256K cell leaked into KVByCtx: %v", m.KVByCtx)
	}
}

func TestBuildContainerSpec_RecoveryEnvWinsOverDynamicEnv(t *testing.T) {
	// Recovery env exists to override defaults for borderline checkpoints;
	// probe-derived DynamicEnv keys must not clobber it (same convention
	// as last-flag-wins for recovery CLI flags).
	cfg := backendConfig{
		Name:          "ollama",
		ContainerName: "devai-ollama",
		Image:         "docker.io/ollama/ollama:latest",
		ModelsDir:     "/var/cache/devai/ollama",
		MountDest:     "/root/.ollama",
		Network:       "devai-net",
		Entrypoint:    ollamaEntrypoint,
		DynamicEnv:    ollamaDynamicEnv,
	}
	lc := launchConfig{MaxContext: 131072, KVCacheType: "q8_0"}
	recovery := map[string]string{"OLLAMA_KV_CACHE_TYPE": "f16"}
	spec := buildContainerSpec(cfg, "mixed:latest", lc, nil, recovery)
	envMap, ok := spec["env"].(map[string]string)
	if !ok {
		t.Fatalf("spec env missing/wrong type: %#v", spec["env"])
	}
	if envMap["OLLAMA_KV_CACHE_TYPE"] != "f16" {
		t.Fatalf("recovery env must win over DynamicEnv, got %q", envMap["OLLAMA_KV_CACHE_TYPE"])
	}
	if envMap["OLLAMA_CONTEXT_LENGTH"] != "131072" {
		t.Fatalf("non-colliding DynamicEnv keys must still apply, got %v", envMap)
	}
}

func TestOllamaDynamicEnv_KVCacheType(t *testing.T) {
	// The ollama DynamicEnv must bake the resolved KV dtype (plus flash
	// attention, its prerequisite) into the recreated container — and
	// stay byte-identical to the legacy env when the dtype is default.
	got := ollamaDynamicEnv(launchConfig{MaxContext: 131072, KVCacheType: "q8_0"})
	if got["OLLAMA_KV_CACHE_TYPE"] != "q8_0" || got["OLLAMA_FLASH_ATTENTION"] != "1" {
		t.Fatalf("q8_0 launch must set KV type + flash attention, got %v", got)
	}
	got = ollamaDynamicEnv(launchConfig{MaxContext: 65536})
	if got["OLLAMA_CONTEXT_LENGTH"] != "65536" {
		t.Fatalf("context env must always be present, got %v", got)
	}
	if _, ok := got["OLLAMA_KV_CACHE_TYPE"]; ok {
		t.Fatalf("default-dtype launch must not set OLLAMA_KV_CACHE_TYPE: %v", got)
	}
	if _, ok := got["OLLAMA_FLASH_ATTENTION"]; ok {
		t.Fatalf("default-dtype launch must not set OLLAMA_FLASH_ATTENTION: %v", got)
	}
	// A raw "f16" stamp is the daemon default too — flagless.
	got = ollamaDynamicEnv(launchConfig{MaxContext: 65536, KVCacheType: "f16"})
	if _, ok := got["OLLAMA_KV_CACHE_TYPE"]; ok {
		t.Fatalf("f16-stamped launch must not set OLLAMA_KV_CACHE_TYPE: %v", got)
	}
}

func TestVllmEntrypoint_KVCacheDtype(t *testing.T) {
	// Legacy rows (no stamp) must keep the historical fp8 — every
	// pre-field fit cell was measured under it.
	args := vllmEntrypoint("m", launchConfig{MaxContext: 32768})
	if !hasFlagValue(args, "--kv-cache-dtype", "fp8") {
		t.Fatalf("unstamped launch must default to fp8: %v", args)
	}
	// A model re-probed under auto (unquantized KV) serves auto.
	args = vllmEntrypoint("m", launchConfig{MaxContext: 32768, KVCacheType: "auto"})
	if !hasFlagValue(args, "--kv-cache-dtype", "auto") {
		t.Fatalf("auto-stamped launch must serve auto: %v", args)
	}
}

func TestSglangEntrypoint_KVCacheDtype(t *testing.T) {
	// Legacy rows (no stamp) ran the engine default — no flag at all.
	args := sglangEntrypoint("m", launchConfig{MaxContext: 32768})
	for _, a := range args {
		if a == "--kv-cache-dtype" {
			t.Fatalf("unstamped sglang launch must not emit --kv-cache-dtype: %v", args)
		}
	}
	// An enforced-dtype stamp is reproduced.
	args = sglangEntrypoint("m", launchConfig{MaxContext: 32768, KVCacheType: "fp8_e5m2"})
	if !hasFlagValue(args, "--kv-cache-dtype", "fp8_e5m2") {
		t.Fatalf("stamped sglang launch must reproduce dtype: %v", args)
	}
}

// hasFlagValue reports whether args contains `flag` immediately followed
// by `value`.
func hasFlagValue(args []string, flag, value string) bool {
	for i := 0; i < len(args)-1; i++ {
		if args[i] == flag && args[i+1] == value {
			return true
		}
	}
	return false
}

func TestSynthesizeHFFromCache_KVByCtx(t *testing.T) {
	// vLLM legacy cells (no stamp) decode to fp8 — the dtype they were
	// factually measured under; stamped cells keep their stamp; sglang
	// legacy cells stay "" (engine default).
	tool := "qwen3_xml"
	entry := func() *hfCacheEntry {
		return &hfCacheEntry{
			SchemaVersion: 2,
			Repo:          "org/model",
			Aliases:       []string{"model"},
			SizeGB:        10,
			MaxContext:    262144,
			Capability:    CapStructured,
			ToolParser:    &tool,
			Probes: map[string]map[string]hfCacheProbe{
				"24": {
					"65536":  {Ctx: 65536, VramGB: 24, Fits: true, ActualVRAMGB: 20},
					"131072": {Ctx: 131072, VramGB: 24, Fits: true, ActualVRAMGB: 21, KVCacheType: "auto"},
				},
			},
		}
	}
	vllmRows := synthesizeHFFromCache(
		map[string]*hfCacheEntry{"k": entry()}, "vllm", 24, 262144, nil,
	)
	if len(vllmRows) != 1 {
		t.Fatalf("expected 1 vllm row, got %d", len(vllmRows))
	}
	if got := vllmRows[0].KVByCtx[65536]; got != "fp8" {
		t.Fatalf("legacy vllm cell must decode to fp8, got %q", got)
	}
	if got := vllmRows[0].KVByCtx[131072]; got != "auto" {
		t.Fatalf("stamped vllm cell must keep its stamp, got %q", got)
	}
	sglangRows := synthesizeHFFromCache(
		map[string]*hfCacheEntry{"k": entry()}, "sglang", 24, 262144, nil,
	)
	if got := sglangRows[0].KVByCtx[65536]; got != "" {
		t.Fatalf("legacy sglang cell must stay engine-default, got %q", got)
	}
}

// --- TestEnvFloat ---

func TestEnvFloat(t *testing.T) {
	t.Setenv("TEST_FLOAT", "48.5")
	if got := envFloat("TEST_FLOAT", 24.0); got != 48.5 {
		t.Errorf("expected 48.5, got %f", got)
	}
	if got := envFloat("NONEXISTENT_FLOAT", 24.0); got != 24.0 {
		t.Errorf("expected 24.0, got %f", got)
	}
	t.Setenv("TEST_FLOAT_BAD", "notanumber")
	if got := envFloat("TEST_FLOAT_BAD", 24.0); got != 24.0 {
		t.Errorf("expected fallback 24.0 for bad value, got %f", got)
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
	bs.advertised = []string{"model-a"}
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
	bs.config.Entrypoint = func(m string, lc launchConfig) []string { return []string{"echo", m} }
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

// TestBuildContainerSpecImageOverride locks the per-model image-override
// contract: launchConfig.RecoveryImage (sourced from a recovery-flags
// `image` field) wins over the backend default; empty preserves it. This
// is what routes DiffusionGemma to the vLLM "gemma" build while every
// other model stays on the global default.
func TestBuildContainerSpecImageOverride(t *testing.T) {
	cfg := backendConfig{
		Name:          "vllm",
		ContainerName: "devai-vllm",
		Image:         "docker.io/vllm/vllm-openai:latest-cu130-ubuntu2404",
		ModelsDir:     "/models",
		Network:       "devai-net",
		Entrypoint:    func(string, launchConfig) []string { return []string{"sleep"} },
	}
	tests := []struct {
		name          string
		recoveryImage string
		want          string
	}{
		{"empty override falls back to backend default", "", cfg.Image},
		{"override wins", "docker.io/vllm/vllm-openai:gemma-x86_64-cu130", "docker.io/vllm/vllm-openai:gemma-x86_64-cu130"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			spec := buildContainerSpec(cfg, "m", launchConfig{RecoveryImage: tt.recoveryImage}, nil, nil)
			if got := spec["image"]; got != tt.want {
				t.Errorf("spec[image] = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestBuildContainerSpecGPUDevice locks the GPU-vendor overlay contract
// (docs/gpu-vendors.md): DEVAI_GPU_DEVICE overrides the CDI device string,
// defaulting to NVIDIA's when unset so an nvidia-only host needs no config.
func TestBuildContainerSpecGPUDevice(t *testing.T) {
	cfg := backendConfig{
		Name:          "vllm",
		ContainerName: "devai-vllm",
		Image:         "docker.io/vllm/vllm-openai:latest-cu130-ubuntu2404",
		ModelsDir:     "/models",
		Network:       "devai-net",
		Entrypoint:    func(string, launchConfig) []string { return []string{"sleep"} },
	}

	t.Run("defaults to nvidia when unset", func(t *testing.T) {
		spec := buildContainerSpec(cfg, "m", launchConfig{}, nil, nil)
		devices, ok := spec["devices"].([]map[string]any)
		if !ok || len(devices) != 1 {
			t.Fatalf("spec[devices] = %#v, want a single-element []map[string]any", spec["devices"])
		}
		if got := devices[0]["path"]; got != "nvidia.com/gpu=all" {
			t.Errorf("devices[0][path] = %v, want nvidia.com/gpu=all", got)
		}
	})

	t.Run("DEVAI_GPU_DEVICE overrides", func(t *testing.T) {
		t.Setenv("DEVAI_GPU_DEVICE", "amd.com/gpu=all")
		spec := buildContainerSpec(cfg, "m", launchConfig{}, nil, nil)
		devices := spec["devices"].([]map[string]any)
		if got := devices[0]["path"]; got != "amd.com/gpu=all" {
			t.Errorf("devices[0][path] = %v, want amd.com/gpu=all", got)
		}
	})
}
