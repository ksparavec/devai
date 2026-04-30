// gpu-arbiter — GPU-aware multi-backend inference proxy.
//
// Exposes one port per backend. Each port is a reverse proxy that manages
// the backend's container lifecycle. Only one backend uses the GPU at a time;
// incoming requests trigger automatic switching with graceful drain.
//
// Ports:
//
//	:11434  Ollama (GGUF models)
//	:11435  vLLM   (NVFP4 models)
//	:11436  SGLang (NVFP4 + HuggingFace models)
//
// Environment variables:
//
//	OLLAMA_URL        Ollama backend URL (default http://devai-ollama:11434)
//	OLLAMA_PORT       Ollama listen port (default 11434)
//	VLLM_URL          vLLM backend URL (default http://devai-vllm:11434)
//	VLLM_PORT         vLLM listen port (default 11435)
//	VLLM_CONTAINER    vLLM container name (default devai-vllm)
//	VLLM_IMAGE        vLLM container image
//	VLLM_MODELS_DIR   host path to vLLM models
//	SGLANG_URL        SGLang backend URL (default http://devai-sglang:11434)
//	SGLANG_PORT       SGLang listen port (default 11436)
//	SGLANG_CONTAINER  SGLang container name (default devai-sglang)
//	SGLANG_IMAGE      SGLang container image
//	SGLANG_MODELS_DIR host path to SGLang models
//	IDLE_TIMEOUT      seconds before idle backend is stopped (default 300)
//	DRAIN_TIMEOUT     seconds to wait for in-flight requests before stopping (default 30)
//	PODMAN_SOCKET     path to Podman API socket (default /run/podman/podman.sock)
//	CONFIG_FILE       path to models.yaml (default /etc/devai/models.yaml)
//	NETWORK           Podman network name (default devai-net)
//	GPU_MEMORY_GB     total GPU VRAM in GB for memory fraction calc (default 24)
//	MAX_CONTEXT_LEN   default max context length in tokens (default 131072 = 128K)
//	DEVAI_REASONING   reasoning policy: auto|off|low|medium|high (default auto)
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/signal"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

func env(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func envInt(key string, fallback int) int {
	v := 0
	if s := os.Getenv(key); s != "" {
		fmt.Sscanf(s, "%d", &v)
		if v > 0 {
			return v
		}
	}
	return fallback
}

func envFloat(key string, fallback float64) float64 {
	if s := os.Getenv(key); s != "" {
		if v, err := strconv.ParseFloat(s, 64); err == nil && v > 0 {
			return v
		}
	}
	return fallback
}

// --- Configuration ---

type backendConfig struct {
	Name          string
	ListenPort    int
	BackendURL    *url.URL
	ContainerName string
	Image         string
	ModelsDir     string
	Network       string
	HealthPath    string
	Entrypoint    func(modelName string, lc launchConfig) []string
	EnvVars       map[string]string
}

type configModel struct {
	Name            string           `yaml:"name"`
	Aliases         []string         `yaml:"aliases,omitempty"`
	Digest          string           `yaml:"digest,omitempty"`
	Backend         []string         `yaml:"backend"`
	Repo            string           `yaml:"repo"`
	Size            string           `yaml:"size"`
	Context         int              `yaml:"context"`
	Purpose         string           `yaml:"purpose"`
	Reasoning       *configReasoning `yaml:"reasoning,omitempty"`
	ToolParser      string           `yaml:"tool_parser,omitempty"`      // populated from HF probe cache
	ReasoningParser string           `yaml:"reasoning_parser,omitempty"` // populated from HF probe cache
}

// configReasoning records what the runtime probe observed for this model.
// Capability values (per docs/ollama_models.md):
//
//	structured  – native API exposes a separate reasoning trace field
//	inline      – reasoning appears inline (e.g. <think> blocks) only
//	unsupported – no reasoning behavior observed
//	unknown     – not yet probed (e.g. vLLM/SGLang models pre-probe)
//	error       – probe failed (model load error, HTTP 4xx, etc.)
//
// DisableVerified is set only for `structured` capability and reports
// whether sending the protocol's "off" field actually suppresses reasoning.
type configReasoning struct {
	Capability      string `yaml:"capability"`
	DisableVerified *bool  `yaml:"disable_verified,omitempty"`
}

// launchConfig holds computed GPU parameters passed to backend entrypoints.
type launchConfig struct {
	MemFraction     float64
	MaxContext      int
	ToolParser      string // empty omits backend-specific tool flags
	ReasoningParser string // empty omits --reasoning-parser
}

type configFile struct {
	Defaults map[string]string `yaml:"defaults"`
	Models   []configModel     `yaml:"models"`
}

// cacheProbe is one (vram, ctx) cell from the probe driver's v3 cache.
type cacheProbe struct {
	Ctx           int     `json:"ctx"`
	VramGB        int     `json:"vram_gb"`
	ActualTotalGB float64 `json:"actual_total_gb"`
	FullyOnGPU    bool    `json:"fully_on_gpu"`
	Capability    string  `json:"capability"`
}

// cacheEntry mirrors the per-digest record in
// deploy/.ollama-reasoning-cache.json (schema v3).
type cacheEntry struct {
	SchemaVersion   int                              `json:"schema_version"`
	Digest          string                           `json:"digest"`
	Aliases         []string                         `json:"aliases"`
	MaxContext      int                              `json:"max_context"`
	Capability      string                           `json:"capability"`
	DisableVerified *bool                            `json:"disable_verified,omitempty"`
	Probes          map[string]map[string]cacheProbe `json:"probes"`
}

// synthesizeFromCache returns one configModel per cache entry that has a
// fully-on-GPU probe at the host VRAM band and at or below the operator's
// MAX_CONTEXT_LEN. Entries without a clean probe at the host band are
// dropped — they would not be servable on this GPU at this context cap.
//
// The synthesized rows feed the same downstream maps (modelSizes /
// modelContexts / modelCapability / modelDisableOK) that active-models.yaml
// used to feed.
func synthesizeFromCache(
	cache map[string]*cacheEntry, hostVRAMGB, operatorMaxCtx int,
) []configModel {
	hostKey := strconv.Itoa(hostVRAMGB)
	out := make([]configModel, 0, len(cache))
	for _, entry := range cache {
		if entry == nil || entry.SchemaVersion != 3 || len(entry.Aliases) == 0 {
			continue
		}
		band, ok := entry.Probes[hostKey]
		if !ok || len(band) == 0 {
			continue
		}
		bestCtx := 0
		var bestProbe cacheProbe
		for ctxStr, probe := range band {
			c, err := strconv.Atoi(ctxStr)
			if err != nil {
				continue
			}
			if operatorMaxCtx > 0 && c > operatorMaxCtx {
				continue
			}
			if !probe.FullyOnGPU {
				continue
			}
			if c >= bestCtx {
				bestCtx = c
				bestProbe = probe
			}
		}
		if bestCtx == 0 {
			continue
		}
		// Effective context cap: min(model_max, operator_max).
		effCtx := entry.MaxContext
		if operatorMaxCtx > 0 && (effCtx == 0 || effCtx > operatorMaxCtx) {
			effCtx = operatorMaxCtx
		}
		cap := entry.Capability
		if cap == "" {
			cap = "unknown"
		}
		// First alias is the placeholder Name; the rest are Aliases. The
		// router registers every name into the lookup maps regardless.
		canonical := entry.Aliases[0]
		var aliases []string
		if len(entry.Aliases) > 1 {
			aliases = append([]string(nil), entry.Aliases[1:]...)
		}
		out = append(out, configModel{
			Name:    canonical,
			Aliases: aliases,
			Digest:  entry.Digest,
			Backend: []string{"ollama"},
			Size:    fmt.Sprintf("%.2f GB", bestProbe.ActualTotalGB),
			Context: effCtx,
			Reasoning: &configReasoning{
				Capability:      cap,
				DisableVerified: entry.DisableVerified,
			},
		})
	}
	return out
}

// hfCacheProbe is one (vram, ctx) cell from a vLLM/SGLang probe cache.
// Mirrors the schema v1 record in deploy/.vllm-reasoning-cache.json and
// deploy/.sglang-reasoning-cache.json (see scripts/_probe_hf_common.py).
//
// `Fits` is the HF analog of Ollama's `FullyOnGPU`: the model + KV at
// the requested ctx loaded into the static pool, /v1/models reported
// the requested context, and a chat round-trip succeeded.
type hfCacheProbe struct {
	Ctx           int     `json:"ctx"`
	VramGB        int     `json:"vram_gb"`
	Fits          bool    `json:"fits"`
	ActualVRAMGB  float64 `json:"actual_vram_gb"`
	ActualContext int     `json:"actual_context"`
	Capability    string  `json:"capability"`
}

// hfCacheEntry mirrors the per-(repo, sha) record in the HF probe caches.
//
// Schema v2 added ReasoningParser, DisableVerified, and populated the
// previously-null ToolParser field. Pre-v2 caches read with ToolParser
// = nil and the new fields zero-valued; the synthesizer treats those
// as "no curated parsers" and emits a serving row with no parser flags
// — same behaviour as a model whose family declared no `parsers:` block.
type hfCacheEntry struct {
	SchemaVersion int      `json:"schema_version"`
	Repo          string   `json:"repo"`
	Sha           string   `json:"sha"`
	Aliases       []string `json:"aliases"`
	ModelKind     string   `json:"model_kind"`
	// SizeGB is the catalog-declared weight size on disk. Required for
	// memFraction launch math — without it, ActualVRAMGB (post-load,
	// weights + KV + CUDA graphs) would mistakenly be used as the
	// weight size and clamp --max-model-len to a few thousand tokens.
	SizeGB          float64                            `json:"size_gb,omitempty"`
	MaxContext      int                                `json:"max_context"`
	Capability      string                             `json:"capability"`
	ToolParser      *string                            `json:"tool_parser"`
	ReasoningParser *string                            `json:"reasoning_parser,omitempty"`
	DisableVerified *bool                              `json:"disable_verified,omitempty"`
	Probes          map[string]map[string]hfCacheProbe `json:"probes"`
}

// synthesizeHFFromCache returns one configModel per HF cache entry whose
// host-VRAM band has a fits=true probe at or below MAX_CONTEXT_LEN.
//
// `backendName` ("vllm" or "sglang") tags the row's Backend list so the
// downstream lookup (modelsForBackend) places it on the correct port.
//
// Entries with capability `unsupported_arch`, `error`, or no fitting
// probe at the host band are skipped — they won't serve. Anything else
// (inline, unsupported, unknown) is emitted; the picker decides what
// to surface.
func synthesizeHFFromCache(
	cache map[string]*hfCacheEntry,
	backendName string,
	hostVRAMGB, operatorMaxCtx int,
) []configModel {
	hostKey := strconv.Itoa(hostVRAMGB)
	out := make([]configModel, 0, len(cache))
	for _, entry := range cache {
		if entry == nil || len(entry.Aliases) == 0 {
			continue
		}
		// Terminal failure states never produce a serving row.
		if entry.Capability == "unsupported_arch" || entry.Capability == "error" {
			continue
		}
		band, ok := entry.Probes[hostKey]
		if !ok || len(band) == 0 {
			continue
		}
		bestCtx := 0
		var bestProbe hfCacheProbe
		for ctxStr, probe := range band {
			c, err := strconv.Atoi(ctxStr)
			if err != nil {
				continue
			}
			if operatorMaxCtx > 0 && c > operatorMaxCtx {
				continue
			}
			if !probe.Fits {
				continue
			}
			if c >= bestCtx {
				bestCtx = c
				bestProbe = probe
			}
		}
		if bestCtx == 0 {
			continue
		}
		// Effective context cap mirrors the Ollama path:
		// min(model_max, operator_max).
		effCtx := entry.MaxContext
		if operatorMaxCtx > 0 && (effCtx == 0 || effCtx > operatorMaxCtx) {
			effCtx = operatorMaxCtx
		}
		cap := entry.Capability
		if cap == "" {
			cap = "unknown"
		}
		canonical := entry.Aliases[0]
		var aliases []string
		if len(entry.Aliases) > 1 {
			aliases = append([]string(nil), entry.Aliases[1:]...)
		}
		toolParser := ""
		if entry.ToolParser != nil {
			toolParser = *entry.ToolParser
		}
		reasoningParser := ""
		if entry.ReasoningParser != nil {
			reasoningParser = *entry.ReasoningParser
		}
		// Prefer the catalog-declared weight size for the Size field —
		// it feeds memFraction at containerRecreate time, which needs
		// the WEIGHT footprint, not the post-load total. Older cache
		// entries (pre-fix) lack size_gb; fall back to ActualVRAMGB
		// minus a rough KV estimate so launch math doesn't collapse.
		// Operators should re-probe once to populate size_gb cleanly.
		sizeGB := entry.SizeGB
		if sizeGB <= 0 {
			// Fallback: assume the probe ran at ~half the model's
			// declared max context, so KV is ~half of total. This is
			// approximate but safer than passing the full footprint.
			sizeGB = bestProbe.ActualVRAMGB * 0.5
			if sizeGB <= 0 {
				sizeGB = bestProbe.ActualVRAMGB
			}
		}
		out = append(out, configModel{
			Name:            canonical,
			Aliases:         aliases,
			Backend:         []string{backendName},
			Repo:            entry.Repo,
			Size:            fmt.Sprintf("%.2f GB", sizeGB),
			Context:         effCtx,
			ToolParser:      toolParser,
			ReasoningParser: reasoningParser,
			Reasoning: &configReasoning{
				Capability:      cap,
				DisableVerified: entry.DisableVerified,
			},
		})
	}
	return out
}

// loadHFCache reads a vLLM/SGLang cache file from `path` and returns
// parsed entries plus a synthesized configModel slice for `backendName`.
// Missing files are reported informationally (probing hasn't run yet);
// parse failures emit a warning and skip without aborting.
func loadHFCache(
	path, backendName string, hostVRAMGB, operatorMaxCtx int,
) []configModel {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			log.Printf("probe cache: %s not present — no %s rows registered",
				path, backendName)
			return nil
		}
		log.Printf("warning: %s probe cache read failed: %v", backendName, err)
		return nil
	}
	var cache map[string]*hfCacheEntry
	if jerr := json.Unmarshal(data, &cache); jerr != nil {
		log.Printf("warning: %s probe cache parse failed: %v", backendName, jerr)
		return nil
	}
	rows := synthesizeHFFromCache(cache, backendName, hostVRAMGB, operatorMaxCtx)
	log.Printf("probe cache: %s loaded (%d entries → %d %s serving rows)",
		path, len(cache), len(rows), backendName)
	return rows
}

func modelsForBackend(models []configModel, backend string) []string {
	var names []string
	for _, m := range models {
		for _, b := range m.Backend {
			if b == backend {
				names = append(names, m.Name)
				break
			}
		}
	}
	return names
}

// --- Backend state ---

type backendState struct {
	config         backendConfig
	proxy          *httputil.ReverseProxy
	modelNames     []string
	running        bool
	currentModel   string
	currentContext int // baked --max-model-len / --context-length for vLLM/SGLang; 0 for Ollama
	lastRequest    time.Time
	activeReqs     int64
}

type arbiter struct {
	backends             map[string]*backendState
	mu                   sync.Mutex
	ollamaURL            *url.URL
	podmanClient         *http.Client
	idleTimeout          time.Duration
	drainTimeout         time.Duration
	healthTimeout        time.Duration      // configurable per HEALTH_TIMEOUT_SECONDS env (default 600s — vLLM/SGLang cold-start with NVFP4 weights + CUDA graph compilation can exceed 5 min on consumer GPUs)
	modelSizes           map[string]float64 // model name → weight size in GB
	modelContexts        map[string]int     // model name → declared max context (from models.yaml)
	modelCapability      map[string]string  // model name → reasoning.capability
	modelDisableOK       map[string]bool    // model name → disable_verified (only when present)
	modelToolParser      map[string]string  // model name → backend --tool-call-parser (empty omits the flag)
	modelReasoningParser map[string]string  // model name → backend --reasoning-parser (empty omits the flag)
	defaultPolicy        string             // DEVAI_REASONING env value: auto|off|low|medium|high
	totalVRAMGB          float64
	maxContextLen        int // global default from MAX_CONTEXT_LEN env (default 131072)
}

// --- Proxy factories ---

func noKeepAliveTransport() *http.Transport {
	return &http.Transport{
		DisableKeepAlives: true,
	}
}

func newProxy(target *url.URL) *httputil.ReverseProxy {
	p := httputil.NewSingleHostReverseProxy(target)
	p.FlushInterval = -1
	p.Transport = noKeepAliveTransport()
	return p
}

func newSmartProxy(target *url.URL) *httputil.ReverseProxy {
	p := httputil.NewSingleHostReverseProxy(target)
	p.FlushInterval = -1
	p.Transport = noKeepAliveTransport()
	p.ModifyResponse = func(resp *http.Response) error {
		if resp.StatusCode == http.StatusInternalServerError {
			body, err := io.ReadAll(resp.Body)
			resp.Body.Close()
			if err == nil && strings.Contains(string(body), "maximum context length") {
				resp.StatusCode = http.StatusBadRequest
				resp.Status = "400 Bad Request"
			}
			resp.Body = io.NopCloser(bytes.NewReader(body))
		}
		return nil
	}
	return p
}

// --- GPU memory calculation ---

// parseSizeGB extracts a float from strings like "7.4 GB", "17 GB", "2.0 GB".
func parseSizeGB(s string) float64 {
	s = strings.TrimSpace(s)
	s = strings.TrimSuffix(s, "GB")
	s = strings.TrimSpace(s)
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0
	}
	return v
}

// memFraction calculates the optimal GPU memory fraction for a backend.
//
// The fraction controls what share of total VRAM is allocated to the static
// memory pool (model weights + KV cache).  The remainder stays free for CUDA
// graph capture, activation tensors, and temporary buffers.
//
// Backend-specific reservations (kept outside the static pool):
//   - vLLM:   ~2 GB — CUDA graphs + activations
//   - SGLang: ~3 GB — RadixAttention tree + CUDA graphs + activations
func memFraction(modelSizeGB, totalVRAMGB float64, backend string) float64 {
	if modelSizeGB <= 0 {
		// Unknown model size — conservative defaults
		switch backend {
		case "sglang":
			return 0.82
		default:
			return 0.88
		}
	}

	// Base reservation for runtime memory outside the static pool
	var reserveGB float64
	switch backend {
	case "sglang":
		reserveGB = 3.0
	default: // vllm
		reserveGB = 2.0
	}

	// Tight fit: model barely leaves room for KV cache + runtime
	headroom := totalVRAMGB - modelSizeGB
	if headroom < reserveGB+2.0 {
		reserveGB = max(0.5, headroom*0.3)
	}

	frac := (totalVRAMGB - reserveGB) / totalVRAMGB

	// Clamp to [0.40, 0.95]
	return max(0.40, min(0.95, frac))
}

// maxContextLen estimates the maximum context length that fits in the
// available KV cache memory.  Both vLLM and SGLang store KV cache in BF16
// regardless of weight quantization.
//
// Rough per-token KV footprint (BF16, GQA):
//
//	2 (K+V) × layers × kv_heads × head_dim × 2 bytes
//
// We approximate this from the weight file size since we don't have the
// architecture config:
//
//	≤ 6 GB  →  ~100 KB/token  (small models, fewer layers)
//	≤ 12 GB →  ~160 KB/token  (9B class)
//	≤ 20 GB →  ~256 KB/token  (14-27B class)
//	> 20 GB →  ~400 KB/token  (35B+ class)
//
// fittableContext estimates the maximum context length that fits in the
// available KV cache memory.  Both vLLM and SGLang store KV cache in BF16
// regardless of weight quantization.
//
// Approximate per-token KV footprint (BF16, GQA):
//
//	2 (K+V) × layers × kv_heads × head_dim × 2 bytes
//
// Estimated from weight file size (architecture config unavailable):
//
//	≤ 6 GB  →  ~100 KB/token  (small models, fewer layers)
//	≤ 12 GB →  ~160 KB/token  (9B class)
//	≤ 20 GB →  ~256 KB/token  (14-27B class)
//	> 20 GB →  ~400 KB/token  (35B+ class)
func fittableContext(availableKVGB, modelSizeGB float64) int {
	var kvPerTokenKB float64
	switch {
	case modelSizeGB <= 6:
		kvPerTokenKB = 100
	case modelSizeGB <= 12:
		kvPerTokenKB = 160
	case modelSizeGB <= 20:
		kvPerTokenKB = 256
	default:
		kvPerTokenKB = 400
	}

	tokens := int(availableKVGB * 1024 * 1024 / kvPerTokenKB)
	tokens = (tokens / 4096) * 4096 // round down to 4K boundary

	if tokens < 4096 {
		return 4096
	}
	return tokens
}

// computeLaunchConfig builds a launchConfig for the given model and backend.
// desiredContext is the target context length (from models.yaml or
// MAX_CONTEXT_LEN env); the actual value is reduced when KV cache memory
// cannot support it.
func computeLaunchConfig(modelSizeGB, totalVRAMGB float64, backend string, desiredContext int) launchConfig {
	frac := memFraction(modelSizeGB, totalVRAMGB, backend)

	// KV cache budget = static pool minus weights
	availKV := frac*totalVRAMGB - modelSizeGB
	if availKV < 0 {
		availKV = 0
	}

	fits := fittableContext(availKV, modelSizeGB)
	ctx := desiredContext
	if ctx <= 0 {
		ctx = 131072
	}
	if fits < ctx {
		ctx = fits
	}

	return launchConfig{MemFraction: frac, MaxContext: ctx}
}

// --- Entrypoint builders ---

func vllmEntrypoint(modelName string, lc launchConfig) []string {
	args := []string{
		"python3", "-m", "vllm.entrypoints.openai.api_server",
		"--model", "/models/" + modelName,
		"--host", "0.0.0.0",
		"--port", "11434",
		"--tensor-parallel-size", "1",
		"--max-model-len", fmt.Sprintf("%d", lc.MaxContext),
		"--gpu-memory-utilization", fmt.Sprintf("%.2f", lc.MemFraction),
		"--enable-prefix-caching",
		"--trust-remote-code",
		"--served-model-name", modelName,
	}
	// Parser flags are per-model and read from the probe cache. Omit
	// when unverified so a non-matching parser doesn't crash the launch.
	// See deploy/backend-flags.yaml for the verified flag names.
	if lc.ReasoningParser != "" {
		args = append(args, "--reasoning-parser", lc.ReasoningParser)
	}
	if lc.ToolParser != "" {
		args = append(args, "--enable-auto-tool-choice", "--tool-call-parser", lc.ToolParser)
	}
	return args
}

func sglangEntrypoint(modelName string, lc launchConfig) []string {
	args := []string{
		"python3", "-m", "sglang.launch_server",
		"--model-path", "/models/" + modelName,
		"--host", "0.0.0.0",
		"--port", "11434",
		"--tp", "1",
		"--mem-fraction-static", fmt.Sprintf("%.2f", lc.MemFraction),
		"--context-length", fmt.Sprintf("%d", lc.MaxContext),
		"--trust-remote-code",
	}
	// SGLang flags verified against v0.5.10.post1-cu130 — see
	// deploy/backend-flags.yaml. SGLang accepts --tool-call-parser
	// without an --enable-auto-tool-choice analogue (unlike vLLM); the
	// flag is sufficient on its own to enable tool parsing.
	if lc.ReasoningParser != "" {
		args = append(args, "--reasoning-parser", lc.ReasoningParser)
	}
	if lc.ToolParser != "" {
		args = append(args, "--tool-call-parser", lc.ToolParser)
	}
	return args
}

// --- Main ---

func main() {
	ollamaURL, _ := url.Parse(env("OLLAMA_URL", "http://devai-ollama:11434"))
	vllmURL, _ := url.Parse(env("VLLM_URL", "http://devai-vllm:11434"))
	sglangURL, _ := url.Parse(env("SGLANG_URL", "http://devai-sglang:11434"))
	socketPath := env("PODMAN_SOCKET", "/run/podman/podman.sock")
	network := env("NETWORK", "devai-net")

	podmanClient := &http.Client{
		Transport: &http.Transport{
			DialContext: func(_ context.Context, _, _ string) (net.Conn, error) {
				return net.Dial("unix", socketPath)
			},
		},
		Timeout: 30 * time.Second,
	}

	// Load probe cache (schema v3). The cache is the single source of truth
	// for fit data after the resilient-splashing-peach refactor; the old
	// active-models.yaml is gone. We synthesize one configModel per cache
	// entry whose host-VRAM band has a fully-on-GPU probe at or below
	// MAX_CONTEXT_LEN, then feed downstream consumers (modelsForBackend,
	// the lookup maps) the same way they were fed by the YAML file.
	var cfg configFile
	cachePath := env("PROBE_CACHE", "/etc/devai/.ollama-reasoning-cache.json")
	hostVRAMGB := envInt("GPU_MEMORY_GB", 24)
	operatorMaxCtx := envInt("MAX_CONTEXT_LEN", 131072)
	if data, err := os.ReadFile(cachePath); err == nil {
		var cache map[string]*cacheEntry
		if jerr := json.Unmarshal(data, &cache); jerr == nil {
			cfg.Models = synthesizeFromCache(cache, hostVRAMGB, operatorMaxCtx)
			log.Printf("probe cache: %s loaded (%d v3 entries → %d serving rows)",
				cachePath, len(cache), len(cfg.Models))
		} else {
			log.Printf("warning: probe cache parse failed: %v", jerr)
		}
	} else if !os.IsNotExist(err) {
		log.Printf("warning: probe cache read failed: %v", err)
	} else {
		log.Printf("probe cache: %s not present — no models registered (run `make probe`)", cachePath)
	}

	// Append HF-backend rows synthesized from the per-backend probe caches.
	// Each cache is independent; missing files are not fatal — the
	// corresponding backend just exposes no models. Tool-parser values
	// flow through the configModel.ToolParser field into the
	// modelToolParser lookup, which the parameterized vllmEntrypoint
	// (Phase 0) reads when starting the backend.
	vllmCachePath := env("VLLM_PROBE_CACHE", "/etc/devai/.vllm-reasoning-cache.json")
	cfg.Models = append(cfg.Models,
		loadHFCache(vllmCachePath, "vllm", hostVRAMGB, operatorMaxCtx)...)
	sglangCachePath := env("SGLANG_PROBE_CACHE", "/etc/devai/.sglang-reasoning-cache.json")
	cfg.Models = append(cfg.Models,
		loadHFCache(sglangCachePath, "sglang", hostVRAMGB, operatorMaxCtx)...)

	backends := []backendConfig{
		{
			Name:          "ollama",
			ListenPort:    envInt("OLLAMA_PORT", 11434),
			BackendURL:    ollamaURL,
			ContainerName: env("OLLAMA_CONTAINER", "devai-ollama"),
			HealthPath:    "/",
		},
		{
			Name:          "vllm",
			ListenPort:    envInt("VLLM_PORT", 11435),
			BackendURL:    vllmURL,
			ContainerName: env("VLLM_CONTAINER", "devai-vllm"),
			Image:         env("VLLM_IMAGE", "docker.io/vllm/vllm-openai:latest-cu130-ubuntu2404"),
			ModelsDir:     env("VLLM_MODELS_DIR", "/var/cache/devai/ollama/models/vllm"),
			Network:       network,
			HealthPath:    "/health",
			Entrypoint:    vllmEntrypoint,
			EnvVars:       map[string]string{"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"},
		},
		{
			Name:          "sglang",
			ListenPort:    envInt("SGLANG_PORT", 11436),
			BackendURL:    sglangURL,
			ContainerName: env("SGLANG_CONTAINER", "devai-sglang"),
			Image:         env("SGLANG_IMAGE", "docker.io/lmsysorg/sglang:v0.5.10.post1-cu130"),
			ModelsDir:     env("SGLANG_MODELS_DIR", "/var/cache/devai/ollama/models/vllm"),
			Network:       network,
			HealthPath:    "/health",
			Entrypoint:    sglangEntrypoint,
		},
	}

	// Build model size, context, and reasoning capability lookups from catalog.
	// Reasoning capability comes from the runtime probe written by
	// scripts/probe-ollama-reasoning.py. The arbiter applies the policy
	// (DEVAI_REASONING env) per request using the protocol field for the
	// incoming Ollama API path.
	// Resolve canonical name + aliases (one digest = one set of weights, but
	// users may issue requests with any registered alias). Each lookup map
	// receives an entry per alias so a request for "qwen3.5:latest" finds
	// the same context cap and capability as the canonical "qwen3.5:9b-q8_0".
	modelSizes := make(map[string]float64)
	modelContexts := make(map[string]int)
	modelCapability := make(map[string]string)
	modelDisableOK := make(map[string]bool)
	modelToolParser := make(map[string]string)      // backend startup flag, populated from probe cache
	modelReasoningParser := make(map[string]string) // backend startup flag, populated from probe cache
	capCounts := make(map[string]int)
	for _, m := range cfg.Models {
		names := append([]string{m.Name}, m.Aliases...)
		sz := parseSizeGB(m.Size)
		cap := "unknown"
		if m.Reasoning != nil && m.Reasoning.Capability != "" {
			cap = m.Reasoning.Capability
		}
		disableOK := m.Reasoning != nil &&
			m.Reasoning.DisableVerified != nil &&
			*m.Reasoning.DisableVerified
		for _, name := range names {
			if name == "" {
				continue
			}
			if sz > 0 {
				modelSizes[name] = sz
			}
			if m.Context > 0 {
				modelContexts[name] = m.Context
			}
			modelCapability[name] = cap
			if disableOK {
				modelDisableOK[name] = true
			}
			if m.ToolParser != "" {
				modelToolParser[name] = m.ToolParser
			}
			if m.ReasoningParser != "" {
				modelReasoningParser[name] = m.ReasoningParser
			}
		}
		// Count capability once per canonical row, not once per alias —
		// otherwise a model with N aliases would dominate the histogram.
		capCounts[cap]++
	}
	policy := strings.ToLower(env("DEVAI_REASONING", "auto"))
	if !validPolicy(policy) {
		log.Printf("warning: invalid DEVAI_REASONING=%q; falling back to auto", policy)
		policy = "auto"
	}
	log.Printf("reasoning policy: %s; capability counts: %v", policy, capCounts)
	totalVRAMGB := envFloat("GPU_MEMORY_GB", 24.0)
	maxCtx := envInt("MAX_CONTEXT_LEN", 131072)

	a := &arbiter{
		backends:             make(map[string]*backendState),
		ollamaURL:            ollamaURL,
		podmanClient:         podmanClient,
		idleTimeout:          time.Duration(envInt("IDLE_TIMEOUT", 300)) * time.Second,
		drainTimeout:         time.Duration(envInt("DRAIN_TIMEOUT", 30)) * time.Second,
		healthTimeout:        time.Duration(envInt("HEALTH_TIMEOUT_SECONDS", 600)) * time.Second,
		modelSizes:           modelSizes,
		modelContexts:        modelContexts,
		modelCapability:      modelCapability,
		modelDisableOK:       modelDisableOK,
		modelToolParser:      modelToolParser,
		modelReasoningParser: modelReasoningParser,
		defaultPolicy:        policy,
		totalVRAMGB:          totalVRAMGB,
		maxContextLen:        maxCtx,
	}

	for _, bc := range backends {
		var proxy *httputil.ReverseProxy
		if bc.Name == "ollama" {
			proxy = newProxy(bc.BackendURL)
		} else {
			proxy = newSmartProxy(bc.BackendURL)
		}
		a.backends[bc.Name] = &backendState{
			config:     bc,
			proxy:      proxy,
			modelNames: modelsForBackend(cfg.Models, bc.Name),
		}
	}

	// Start idle watcher
	go a.idleWatcher()

	// Signal handler
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGTERM, syscall.SIGINT)
	go func() {
		<-sig
		log.Println("shutting down")
		os.Exit(0)
	}()

	// Start one listener per backend
	for name, bs := range a.backends {
		mux := http.NewServeMux()
		mux.HandleFunc("/v1/models", a.makeModelsHandler(name))
		mux.HandleFunc("/api/tags", a.makeTagsHandler(name))
		mux.HandleFunc("/health", a.makeHealthHandler(name))
		mux.HandleFunc("/", a.makeRequestHandler(name))

		addr := fmt.Sprintf(":%d", bs.config.ListenPort)
		log.Printf("  %s → %s (port %d, %d models)",
			name, bs.config.BackendURL, bs.config.ListenPort, len(bs.modelNames))
		go func(a string, m *http.ServeMux) {
			log.Fatal(http.ListenAndServe(a, m))
		}(addr, mux)
	}

	log.Printf("gpu-arbiter started: idle=%ds drain=%ds",
		int(a.idleTimeout.Seconds()), int(a.drainTimeout.Seconds()))

	select {} // block forever
}

// --- Podman container management ---

func (a *arbiter) containerStop(name string) error {
	url := fmt.Sprintf("http://d/v4.0.0/libpod/containers/%s/stop?timeout=10", name)
	resp, err := a.podmanClient.Post(url, "", nil)
	if err != nil {
		return fmt.Errorf("podman stop %s: %w", name, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotModified {
		return nil
	}
	if resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("podman stop %s: %s %s", name, resp.Status, body)
	}
	return nil
}

func (a *arbiter) containerRemove(name string) {
	delURL := fmt.Sprintf("http://d/v4.0.0/libpod/containers/%s?force=true", name)
	req, _ := http.NewRequest("DELETE", delURL, nil)
	if resp, err := a.podmanClient.Do(req); err == nil {
		resp.Body.Close()
	}
}

func (a *arbiter) containerIsRunning(name string) bool {
	if a.podmanClient == nil {
		return false
	}
	url := fmt.Sprintf("http://d/v4.0.0/libpod/containers/%s/json", name)
	resp, err := a.podmanClient.Get(url)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	var info struct {
		State struct {
			Status string `json:"Status"`
		} `json:"State"`
	}
	json.NewDecoder(resp.Body).Decode(&info)
	return info.State.Status == "running"
}

// backendIsServing checks whether the backend container is not just
// "running" (which is true for the `sleep infinity` placeholders set by
// docker-compose) but also actually has its inference server listening
// on the backend's health endpoint. Without this, a request that arrives
// when only the placeholder is up would proxy to a container that has
// no listener, returning 502 with no recovery.
//
// 2-second budget — a real backend that's been started should respond
// to /health in <100ms. Anything slower is treated as "not serving" and
// triggers a containerRecreate at the call site.
func (a *arbiter) backendIsServing(bs *backendState) bool {
	healthURL := bs.config.BackendURL.String() + bs.config.HealthPath
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get(healthURL)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	io.Copy(io.Discard, resp.Body)
	return resp.StatusCode == http.StatusOK
}

// containerRecreate launches the backend container with the given model.
// `desiredCtx > 0` overrides the catalog cap (used when a request carries a
// "<model>@<ctx>" picker override); 0 falls back to the catalog cap. The
// chosen context is always clamped against MAX_CONTEXT_LEN and against the
// memory-driven fittableContext inside computeLaunchConfig.
func (a *arbiter) containerRecreate(bs *backendState, modelName string, desiredCtx int) error {
	cfg := bs.config
	a.containerStop(cfg.ContainerName)
	a.containerRemove(cfg.ContainerName)

	modelSizeGB := a.modelSizes[modelName]
	declaredCtx := a.modelContexts[modelName]
	if declaredCtx == 0 {
		declaredCtx = a.maxContextLen
	}
	requestedCtx := declaredCtx
	if desiredCtx > 0 {
		requestedCtx = desiredCtx
		if a.maxContextLen > 0 && requestedCtx > a.maxContextLen {
			requestedCtx = a.maxContextLen
		}
	}
	lc := computeLaunchConfig(modelSizeGB, a.totalVRAMGB, cfg.Name, requestedCtx)
	if parser := a.modelToolParser[modelName]; parser != "" {
		lc.ToolParser = parser
	}
	if parser := a.modelReasoningParser[modelName]; parser != "" {
		lc.ReasoningParser = parser
	}
	log.Printf("  %s launch: model=%.1f GB, gpu=%.1f GB → fraction=%.2f, context=%dk, reasoning=%q tool=%q",
		cfg.Name, modelSizeGB, a.totalVRAMGB, lc.MemFraction, lc.MaxContext/1024,
		lc.ReasoningParser, lc.ToolParser)

	spec := map[string]any{
		"image":      cfg.Image,
		"name":       cfg.ContainerName,
		"entrypoint": cfg.Entrypoint(modelName, lc),
		"command":    []string{},
		"mounts": []map[string]any{{
			"destination": "/models",
			"source":      cfg.ModelsDir,
			"type":        "bind",
			"options":     []string{"ro"},
		}},
		"hostadd":      []string{"host.containers.internal:host-gateway"},
		"netns":        map[string]any{"nsmode": "bridge"},
		"Networks":     map[string]any{cfg.Network: map[string]any{}},
		"devices":      []map[string]any{{"path": "nvidia.com/gpu=all"}},
		"selinux_opts": []string{"disable"},
		"hostname":     cfg.Name,
	}
	if len(cfg.EnvVars) > 0 {
		spec["env"] = cfg.EnvVars
	}

	body, _ := json.Marshal(spec)
	resp, err := a.podmanClient.Post(
		"http://d/v4.0.0/libpod/containers/create",
		"application/json",
		bytes.NewReader(body),
	)
	if err != nil {
		return fmt.Errorf("podman create %s: %w", cfg.ContainerName, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("podman create %s: %s %s", cfg.ContainerName, resp.Status, respBody)
	}

	startURL := fmt.Sprintf("http://d/v4.0.0/libpod/containers/%s/start", cfg.ContainerName)
	resp2, err := a.podmanClient.Post(startURL, "", nil)
	if err != nil {
		return fmt.Errorf("podman start %s: %w", cfg.ContainerName, err)
	}
	defer resp2.Body.Close()
	if resp2.StatusCode >= 300 && resp2.StatusCode != http.StatusNotModified {
		respBody, _ := io.ReadAll(resp2.Body)
		return fmt.Errorf("podman start %s: %s %s", cfg.ContainerName, resp2.Status, respBody)
	}
	return nil
}

func (a *arbiter) waitForHealthy(bs *backendState, timeout time.Duration) error {
	healthURL := bs.config.BackendURL.String() + bs.config.HealthPath
	log.Printf("waiting for %s at %s (timeout %s)...", bs.config.Name, healthURL, timeout)
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		resp, err := http.Get(healthURL)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				log.Printf("%s ready", bs.config.Name)
				return nil
			}
		}
		time.Sleep(2 * time.Second)
	}
	return fmt.Errorf("%s did not become ready within %s", bs.config.Name, timeout)
}

// --- GPU exclusion and lifecycle ---

func (a *arbiter) unloadOllama() {
	resp, err := http.Get(a.ollamaURL.String() + "/api/ps")
	if err != nil {
		log.Printf("warning: cannot reach ollama: %v", err)
		return
	}
	defer resp.Body.Close()

	var ps struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	json.NewDecoder(resp.Body).Decode(&ps)

	for _, m := range ps.Models {
		log.Printf("unloading ollama model: %s", m.Name)
		b, _ := json.Marshal(map[string]any{"model": m.Name, "keep_alive": "0"})
		if resp, err := http.Post(a.ollamaURL.String()+"/api/generate", "application/json", bytes.NewReader(b)); err == nil {
			io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
		}
	}
	time.Sleep(2 * time.Second)
}

func (a *arbiter) drainBackend(bs *backendState) {
	if atomic.LoadInt64(&bs.activeReqs) == 0 {
		return
	}
	log.Printf("draining %s (%d active requests)...", bs.config.Name, atomic.LoadInt64(&bs.activeReqs))
	deadline := time.Now().Add(a.drainTimeout)
	for atomic.LoadInt64(&bs.activeReqs) > 0 && time.Now().Before(deadline) {
		time.Sleep(500 * time.Millisecond)
	}
	remaining := atomic.LoadInt64(&bs.activeReqs)
	if remaining > 0 {
		log.Printf("warning: %s drain timeout with %d requests still active", bs.config.Name, remaining)
	}
}

func (a *arbiter) stopOtherBackends(targetName string) {
	for name, bs := range a.backends {
		if name == targetName || !bs.running {
			continue
		}
		log.Printf("stopping %s (switching to %s)", name, targetName)
		a.drainBackend(bs)
		if name == "ollama" {
			a.unloadOllama()
		} else {
			if err := a.containerStop(bs.config.ContainerName); err != nil {
				log.Printf("warning: failed to stop %s: %v", name, err)
			}
		}
		bs.running = false
		bs.currentModel = ""
		bs.currentContext = 0
	}
}

// ensureBackendRunning makes sure the target backend is up with the given
// model and context. Called with the arbiter mutex held.
//
// `desiredCtx` is the per-request context cap resolved upstream (picker
// "@<int>" override or registered modelContexts cap). For Ollama it is
// passed only to the backend via the request body — Ollama doesn't bake
// max-ctx into the container. For vLLM and SGLang the context is baked
// into the entrypoint at startup, so a context change requires a full
// recreate even when the model is unchanged.
func (a *arbiter) ensureBackendRunning(bs *backendState, modelName string, desiredCtx int) error {
	if bs.config.Name == "ollama" {
		if !bs.running {
			a.stopOtherBackends("ollama")
			bs.running = true
		}
		return nil
	}

	// Verify the backend is actually serving. Two reasons a previously-
	// recreated workload may no longer be reachable:
	//   1. The container was stopped externally (operator, crash, etc.).
	//   2. Docker-compose's `cache-up` replaced our dynamic container
	//      with the `sleep infinity` placeholder while bs.running was
	//      still true. The placeholder responds to `containerIsRunning`
	//      with `running` but has no listener on the backend port.
	//
	// `backendIsServing` polls /health to distinguish these cases. If
	// it fails, the state is reset so the recreate path fires below.
	if bs.running && a.podmanClient != nil &&
		(!a.containerIsRunning(bs.config.ContainerName) || !a.backendIsServing(bs)) {
		log.Printf("%s not serving (container gone or placeholder up), resetting state",
			bs.config.Name)
		bs.running = false
		bs.currentModel = ""
		bs.currentContext = 0
	}

	// Recreate when the model OR the baked context cap changed. Context
	// only matters here for vLLM/SGLang because Ollama returned above.
	modelChanged := modelName != "" && bs.currentModel != modelName
	contextChanged := desiredCtx > 0 && bs.currentContext > 0 && bs.currentContext != desiredCtx
	needRecreate := !bs.running || modelChanged || contextChanged
	if !needRecreate {
		return nil
	}

	a.stopOtherBackends(bs.config.Name)

	if modelName == "" {
		return fmt.Errorf("model name required for %s", bs.config.Name)
	}

	if bs.currentModel != "" && bs.currentModel != modelName {
		log.Printf("switching %s model: %s → %s", bs.config.Name, bs.currentModel, modelName)
	} else if contextChanged {
		log.Printf("switching %s context (model %s): %d → %d",
			bs.config.Name, modelName, bs.currentContext, desiredCtx)
	}
	log.Printf("starting %s with model %s (ctx=%d)...", bs.config.Name, modelName, desiredCtx)
	if err := a.containerRecreate(bs, modelName, desiredCtx); err != nil {
		return fmt.Errorf("failed to start %s: %w", bs.config.Name, err)
	}

	// Release lock during health wait
	a.mu.Unlock()
	err := a.waitForHealthy(bs, a.healthTimeout)
	a.mu.Lock()

	if err != nil {
		return err
	}

	bs.running = true
	bs.currentModel = modelName
	// Record the context the launch config actually settled on (after
	// memory-driven clamping inside computeLaunchConfig). Without this the
	// modelChanged-only check could miss legitimate ctx-only switches when
	// the operator's MAX_CONTEXT_LEN clamps down a picker request.
	if desiredCtx > 0 {
		bs.currentContext = desiredCtx
	}
	return nil
}

// --- HTTP handlers ---

func (a *arbiter) makeRequestHandler(backendName string) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		bs := a.backends[backendName]
		atomic.AddInt64(&bs.activeReqs, 1)
		defer atomic.AddInt64(&bs.activeReqs, -1)

		// Read body for any POST so we can (a) extract the model name
		// for backend lifecycle decisions and (b) apply the reasoning
		// policy via the backend's native protocol field.
		var modelName string
		var numCtx int
		if req.Method == http.MethodPost && req.Body != nil {
			body, err := io.ReadAll(req.Body)
			if err != nil {
				http.Error(w, `{"error":"failed to read body"}`, http.StatusBadRequest)
				return
			}

			var parsed struct {
				Model string `json:"model"`
			}
			json.Unmarshal(body, &parsed)

			// Resolve the per-request num_ctx and strip any suffixes
			// from the model name. Suffix order (convention): the
			// picker emits `<name>::<reasoning>@<ctx>`. We strip in
			// the same order — @<ctx> first, ::<reasoning> second.
			// Override priority for num_ctx:
			//   1. Picker-supplied @<int>  → force-injected (user choice).
			//   2. Registered modelContexts cap (= min(model_max,
			//      MAX_CONTEXT_LEN) from the probe cache) → soft cap,
			//      only set when the client didn't supply num_ctx.
			//   3. None → request passes through unchanged.
			ctxStripped, ctxOverride := parseCtxOverride(parsed.Model)
			cleanName, reasoningOverride := parseReasoningOverride(ctxStripped)
			numCtx = ctxOverride
			force := ctxOverride > 0
			if numCtx == 0 {
				if cap, ok := a.modelContexts[cleanName]; ok && cap > 0 {
					numCtx = cap
				}
			}
			if cleanName != parsed.Model {
				body = setTopJSONField(body, "model", cleanName)
			}
			body = setNumCtx(body, numCtx, force)
			modelName = cleanName

			// Capability lookup must resolve picker-materialised
			// "<parent>-ctx<N>" derived tags back to the parent's entry.
			// The cache only registers the parent (the derived tag is
			// per-session and absent from /api/show metadata at probe
			// time). Without this strip, modelCapability[derived] = ""
			// and applyReasoningPolicy short-circuits to noop on every
			// derived-tag request.
			policyModel := stripCtxVariantSuffix(modelName)
			policy := a.requestPolicy(req)
			// Per-request `::<token>` suffix wins over both the
			// X-DevAI-Reasoning header and the env-var default. This
			// is what makes the picker's two-row split for inline
			// models work — each row dispatches the same agent CLI
			// but the model-name suffix forces a different policy.
			if reasoningOverride != "" {
				policy = reasoningOverride
			}
			body = a.applyReasoningPolicy(backendName, req.URL.Path, policyModel, policy, body)
			// Strip tools/tool_choice for backends that didn't probe a
			// working tool parser. vLLM rejects `tool_choice="auto"`
			// outright when launched without --enable-auto-tool-choice;
			// SGLang's default tool-call path also requires
			// --tool-call-parser. Agents like Claude Code always send
			// tools — without this rewrite, every chat would fail with
			// `BadRequestError: "auto" tool choice requires ...`. Cost:
			// agentic tool-call loops won't function for these models;
			// plain text chat works.
			body = a.maybeStripTools(backendName, policyModel, body)
			req.Body = io.NopCloser(bytes.NewReader(body))
			req.ContentLength = int64(len(body))
			req.Header.Set("Content-Length", strconv.Itoa(len(body)))
		}

		a.mu.Lock()
		bs.lastRequest = time.Now()
		if err := a.ensureBackendRunning(bs, modelName, numCtx); err != nil {
			a.mu.Unlock()
			log.Printf("error: %v", err)
			http.Error(w, fmt.Sprintf(`{"error":"%s"}`, err), http.StatusServiceUnavailable)
			return
		}
		a.mu.Unlock()

		bs.proxy.ServeHTTP(w, req)
	}
}

// validPolicy returns true for accepted DEVAI_REASONING values.
func validPolicy(p string) bool {
	switch p {
	case "auto", "off", "low", "medium", "high":
		return true
	}
	return false
}

// requestPolicy resolves the effective policy for a request: the
// X-DevAI-Reasoning header overrides the env-default if it parses to a
// valid value; otherwise the arbiter's defaultPolicy wins.
func (a *arbiter) requestPolicy(req *http.Request) string {
	if h := strings.ToLower(strings.TrimSpace(req.Header.Get("X-DevAI-Reasoning"))); h != "" {
		if validPolicy(h) {
			return h
		}
	}
	return a.defaultPolicy
}

// maybeStripTools removes `tools` and `tool_choice` from the request
// body when the upstream backend was launched without tool-call support
// for this model.
//
// vLLM and SGLang only enable tool calls when their respective
// --tool-call-parser flag is set at engine launch (vLLM additionally
// requires --enable-auto-tool-choice). The router omits those flags
// when the probe didn't verify a working parser for the model — see
// modelToolParser population. Agents like Claude Code unconditionally
// send a tool spec, so without this strip every request fails with
// `BadRequestError: "auto" tool choice requires ...`. Stripping is a
// graceful degradation: chat works, tool-calling functionality is
// silently absent for that model.
//
// Ollama is unaffected: its protocol negotiates tool support per
// request and tolerates `tools=[]` without launch-time flags.
//
// On JSON-decode failure the body is returned unchanged — same defensive
// behavior as the other rewrite helpers.
func (a *arbiter) maybeStripTools(backendName, modelName string, body []byte) []byte {
	if backendName != "vllm" && backendName != "sglang" {
		return body
	}
	if a.modelToolParser[modelName] != "" {
		return body
	}
	var doc map[string]any
	if err := json.Unmarshal(body, &doc); err != nil {
		return body
	}
	_, hadTools := doc["tools"]
	_, hadChoice := doc["tool_choice"]
	if !hadTools && !hadChoice {
		return body
	}
	delete(doc, "tools")
	delete(doc, "tool_choice")
	out, err := json.Marshal(doc)
	if err != nil {
		return body
	}
	return out
}

type reasoningAction int

const (
	reasoningNoop reasoningAction = iota
	reasoningEnable
	reasoningDisable
)

// applyReasoningPolicy rewrites the request body to carry the reasoning
// directive expected by the incoming protocol path. Per docs/ollama_models.md:
// prefer protocol fields over prompt-text tricks.
func (a *arbiter) applyReasoningPolicy(backendName, path, modelName, policy string, body []byte) []byte {
	switch backendName {
	case "ollama":
		switch strings.TrimRight(path, "/") {
		case "/api/chat", "/api/generate":
			return a.applyOllamaNativePolicy(modelName, policy, body)
		case "/v1/chat/completions":
			return a.applyOllamaOpenAIChatPolicy(modelName, policy, body)
		case "/v1/messages":
			return a.applyOllamaAnthropicMessagesPolicy(modelName, policy, body)
		}
	case "vllm":
		return a.applyVLLMPolicy(path, modelName, policy, body)
	case "sglang":
		return a.applySGLangPolicy(path, modelName, policy, body)
	}
	return body
}

// applyVLLMPolicy is the vLLM half of the reasoning router.
//
// Both vLLM and SGLang serve OpenAI-compatible /v1/chat/completions only,
// so the rewrite operates on that surface. Models classified `structured`
// got launched with `--reasoning-parser <X>` and emit `reasoning_content`
// when `enable_thinking` is true. `inline` and `unsupported` have no
// reliable structured switch, so we leave the body alone — same as
// Ollama's policy for non-structured capabilities.
//
// Enable shape:
//
//	extra_body.chat_template_kwargs.enable_thinking = true
//	reasoning_effort = "low" | "medium" | "high"
//
// Disable shape (only when disable_verified is true):
//
//	extra_body.chat_template_kwargs.enable_thinking = false
//	reasoning_effort = "none"
//
// Client-supplied fields always win.
func (a *arbiter) applyVLLMPolicy(path, modelName, policy string, body []byte) []byte {
	if strings.TrimRight(path, "/") != "/v1/chat/completions" {
		return body
	}
	switch a.reasoningAction(modelName, policy) {
	case reasoningEnable:
		body = setJSONFieldIfAbsent(
			body,
			[]string{"reasoning_effort", "reasoning"},
			"reasoning_effort",
			openAIReasoningEffort(policy),
		)
		return setNestedJSONFieldIfAbsent(
			body,
			[]string{"extra_body", "chat_template_kwargs", "enable_thinking"},
			true,
		)
	case reasoningDisable:
		body = setJSONFieldIfAbsent(
			body,
			[]string{"reasoning_effort", "reasoning"},
			"reasoning_effort",
			"none",
		)
		return setNestedJSONFieldIfAbsent(
			body,
			[]string{"extra_body", "chat_template_kwargs", "enable_thinking"},
			false,
		)
	default:
		return body
	}
}

// applySGLangPolicy mirrors applyVLLMPolicy. SGLang exposes a top-level
// `separate_reasoning` field on /v1/chat/completions plus the same
// `extra_body.chat_template_kwargs.enable_thinking` path for Qwen3-style
// templates. Setting both makes the disable directive robust regardless
// of which surface SGLang's runtime honours.
func (a *arbiter) applySGLangPolicy(path, modelName, policy string, body []byte) []byte {
	if strings.TrimRight(path, "/") != "/v1/chat/completions" {
		return body
	}
	switch a.reasoningAction(modelName, policy) {
	case reasoningEnable:
		body = setJSONFieldIfAbsent(
			body, []string{"separate_reasoning"}, "separate_reasoning", true,
		)
		return setNestedJSONFieldIfAbsent(
			body,
			[]string{"extra_body", "chat_template_kwargs", "enable_thinking"},
			true,
		)
	case reasoningDisable:
		body = setJSONFieldIfAbsent(
			body, []string{"separate_reasoning"}, "separate_reasoning", false,
		)
		return setNestedJSONFieldIfAbsent(
			body,
			[]string{"extra_body", "chat_template_kwargs", "enable_thinking"},
			false,
		)
	default:
		return body
	}
}

func (a *arbiter) reasoningAction(modelName, policy string) reasoningAction {
	switch a.modelCapability[modelName] {
	case "structured":
		switch policy {
		case "auto", "low", "medium", "high":
			return reasoningEnable
		case "off":
			// Disable only when the prober verified the model honours
			// `enable_thinking=false` / equivalent. Without that
			// confirmation the disable injection is a footgun.
			if a.modelDisableOK[modelName] {
				return reasoningDisable
			}
		}
	case "inline":
		// Inline models leak `<think>` blocks into content — there's
		// no parser that strips them. But the chat template typically
		// honours `enable_thinking=false`, so an EXPLICIT user opt-out
		// (policy=off, set via the picker's `::nothink` suffix or the
		// X-DevAI-Reasoning header) suppresses thinking. We don't
		// gate this on modelDisableOK because the suffix is itself an
		// explicit user opt-in to the disable path — they're saying
		// "I want this off, accept the consequences if it doesn't
		// work for this particular model".
		if policy == "off" {
			return reasoningDisable
		}
	}
	return reasoningNoop
}

// applyOllamaNativePolicy injects the native `think` field according to the
// (capability, policy) matrix. Ollama's `think` is boolean; effort levels
// (low/medium/high) all collapse to true.
//
//	capability  | policy   | think field
//	structured  | auto/L/M/H | true
//	structured  | off (disable_verified) | false
//	structured  | off (not verified) | (not set — avoid surprising client)
//	inline/unsupported/error | any | (not set — no reliable way to control)
//	unknown     | any      | (not set — let backend decide)
//
// Client-supplied `think` always wins; we never override it.
func (a *arbiter) applyOllamaNativePolicy(modelName, policy string, body []byte) []byte {
	switch a.reasoningAction(modelName, policy) {
	case reasoningEnable:
		return setJSONFieldIfAbsent(body, []string{"think"}, "think", true)
	case reasoningDisable:
		return setJSONFieldIfAbsent(body, []string{"think"}, "think", false)
	default:
		return body
	}
}

func (a *arbiter) applyOllamaOpenAIChatPolicy(modelName, policy string, body []byte) []byte {
	switch a.reasoningAction(modelName, policy) {
	case reasoningEnable:
		return setJSONFieldIfAbsent(
			body,
			[]string{"reasoning_effort", "reasoning"},
			"reasoning_effort",
			openAIReasoningEffort(policy),
		)
	case reasoningDisable:
		return setJSONFieldIfAbsent(
			body,
			[]string{"reasoning_effort", "reasoning"},
			"reasoning_effort",
			"none",
		)
	default:
		return body
	}
}

func (a *arbiter) applyOllamaAnthropicMessagesPolicy(modelName, policy string, body []byte) []byte {
	switch a.reasoningAction(modelName, policy) {
	case reasoningEnable:
		return setJSONFieldIfAbsent(
			body,
			[]string{"thinking"},
			"thinking",
			map[string]any{
				"type":          "enabled",
				"budget_tokens": anthropicThinkingBudget(policy),
			},
		)
	default:
		// docs/ollama_models.md defines off_request as {}; nothing to inject.
		return body
	}
}

func openAIReasoningEffort(policy string) string {
	switch policy {
	case "low", "high":
		return policy
	default:
		return "medium"
	}
}

func anthropicThinkingBudget(policy string) int {
	switch policy {
	case "low":
		return 1024
	case "high":
		return 4096
	default:
		return 2048
	}
}

// parseReasoningOverride extracts a "<name>::<token>" suffix carrying a
// per-request reasoning policy override. Recognised tokens:
//
//	nothink             → "off"
//	think               → "auto"
//	off|auto|low|medium|high → that policy verbatim
//
// The picker emits this suffix to expose explicit reasoning toggles for
// inline-reasoning models (where the global DEVAI_REASONING env var
// can't be toggled per-pick because each agent inherits one shared
// process environment). Returns the clean name and the override; an
// empty override means no recognised suffix was present.
//
// Suffix ordering convention: `<name>::<reasoning>@<ctx>`. parseCtxOverride
// runs first to strip the trailing @<int>, then this function strips the
// reasoning token from the result.
func parseReasoningOverride(name string) (clean string, override string) {
	const sep = "::"
	idx := strings.LastIndex(name, sep)
	if idx < 0 {
		return name, ""
	}
	token := strings.ToLower(strings.TrimSpace(name[idx+len(sep):]))
	switch token {
	case "nothink":
		return name[:idx], "off"
	case "think":
		return name[:idx], "auto"
	case "off", "auto", "low", "medium", "high":
		return name[:idx], token
	}
	return name, ""
}

// parseCtxOverride extracts a "<name>@<int>" suffix carrying an explicit
// num_ctx override. The picker appends this for every agent so the
// chosen tier is enforced regardless of which client (Claude Code,
// Aider, Codex, Open Interpreter, LATE, …) sends the request — no
// custom HTTP headers required, just the model name string. Returns
// the clean name and the override (0 when no suffix is present).
func parseCtxOverride(name string) (clean string, override int) {
	at := strings.LastIndex(name, "@")
	if at < 0 {
		return name, 0
	}
	n, err := strconv.Atoi(strings.TrimSpace(name[at+1:]))
	if err != nil || n <= 0 {
		return name, 0
	}
	return name[:at], n
}

// stripCtxVariantSuffix peels the "-ctx<int>" suffix the picker adds when
// it materialises a Modelfile-derived tag (PARAMETER num_ctx baked in).
// The suffix doesn't appear in the cache (only the parent does), so
// per-model lookups (capability, disable_verified, registered context
// cap) need to resolve to the parent. Returns the input unchanged when
// no suffix is present.
var ctxVariantSuffix = regexp.MustCompile(`-ctx\d+$`)

func stripCtxVariantSuffix(name string) string {
	loc := ctxVariantSuffix.FindStringIndex(name)
	if loc == nil {
		return name
	}
	return name[:loc[0]]
}

// setTopJSONField overwrites a top-level field unconditionally (no
// "if absent" gate). Used to rewrite the request's `model` field to
// the clean name before forwarding to the backend.
func setTopJSONField(body []byte, key string, value any) []byte {
	var raw map[string]json.RawMessage
	if json.Unmarshal(body, &raw) != nil {
		return body
	}
	v, err := encodeJSON(value)
	if err != nil {
		return body
	}
	raw[key] = v
	out, err := encodeJSON(raw)
	if err != nil {
		return body
	}
	return out
}

// setNumCtx injects/overrides options.num_ctx in any of the wire
// protocols Ollama accepts. `force=true` (picker @suffix override)
// replaces any existing value; `force=false` (registered modelContexts
// cap as a soft fallback) only sets when the client didn't.
//
// Ollama's /api/chat, /api/generate, /v1/chat/completions, and
// /v1/messages all honour `options.num_ctx` in the request body —
// the OpenAI- and Anthropic-compat layers pass extra fields through.
func setNumCtx(body []byte, numCtx int, force bool) []byte {
	if numCtx <= 0 {
		return body
	}
	var doc map[string]json.RawMessage
	if json.Unmarshal(body, &doc) != nil {
		return body
	}
	var opts map[string]json.RawMessage
	if rawOpts, ok := doc["options"]; ok {
		if json.Unmarshal(rawOpts, &opts) != nil {
			opts = map[string]json.RawMessage{}
		}
	} else {
		opts = map[string]json.RawMessage{}
	}
	if !force {
		if _, exists := opts["num_ctx"]; exists {
			return body
		}
	}
	v, err := encodeJSON(numCtx)
	if err != nil {
		return body
	}
	opts["num_ctx"] = v
	encOpts, err := encodeJSON(opts)
	if err != nil {
		return body
	}
	doc["options"] = encOpts
	out, err := encodeJSON(doc)
	if err != nil {
		return body
	}
	return out
}

func setJSONFieldIfAbsent(body []byte, existingKeys []string, setKey string, value any) []byte {
	var raw map[string]json.RawMessage
	if json.Unmarshal(body, &raw) != nil {
		return body
	}
	for _, key := range existingKeys {
		if _, exists := raw[key]; exists {
			return body
		}
	}
	v, err := encodeJSON(value)
	if err != nil {
		return body
	}
	raw[setKey] = v
	out, err := encodeJSON(raw)
	if err != nil {
		return body
	}
	return out
}

// setNestedJSONFieldIfAbsent walks `path` into the JSON body, creating
// intermediate object nodes as needed, and sets the leaf key to `value`
// — but only when no key on the path already has a non-object value
// (which would mean the client owns it). The leaf is only written when
// it doesn't exist; "client supplied wins" applies at the leaf.
//
// Returns the body unchanged on parse failure or path conflict.
//
// Example: path=["extra_body", "chat_template_kwargs", "enable_thinking"],
// value=true on body {"model":"x"}
//
//	→ {"model":"x","extra_body":{"chat_template_kwargs":{"enable_thinking":true}}}
//
// On body that already carries
// {"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}} the
// leaf exists, so the client's `false` is preserved.
func setNestedJSONFieldIfAbsent(body []byte, path []string, value any) []byte {
	if len(path) == 0 {
		return body
	}
	var root map[string]any
	if json.Unmarshal(body, &root) != nil {
		return body
	}
	if root == nil {
		root = make(map[string]any)
	}
	cur := root
	for i, key := range path[:len(path)-1] {
		next, ok := cur[key]
		if !ok || next == nil {
			child := make(map[string]any)
			cur[key] = child
			cur = child
			continue
		}
		obj, ok := next.(map[string]any)
		if !ok {
			// Intermediate exists but isn't an object — refuse to
			// rewrite the client's typed value.
			_ = i
			return body
		}
		cur = obj
	}
	leaf := path[len(path)-1]
	if _, exists := cur[leaf]; exists {
		return body
	}
	cur[leaf] = value
	out, err := encodeJSON(root)
	if err != nil {
		return body
	}
	return out
}

// encodeJSON marshals v without HTML-escaping (`<`, `>`, `&` stay literal)
// and strips json.Encoder's trailing newline. Used wherever we hand bytes
// back to the proxy or stash them as json.RawMessage.
func encodeJSON(v any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return bytes.TrimRight(buf.Bytes(), "\n"), nil
}

func (a *arbiter) makeModelsHandler(backendName string) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		bs := a.backends[backendName]

		type model struct {
			ID      string `json:"id"`
			Object  string `json:"object"`
			OwnedBy string `json:"owned_by"`
		}
		type modelList struct {
			Object string  `json:"object"`
			Data   []model `json:"data"`
		}

		var all []model

		// Try fetching live models from the backend if it's running
		a.mu.Lock()
		running := bs.running
		a.mu.Unlock()

		if running {
			if resp, err := http.Get(bs.config.BackendURL.String() + "/v1/models"); err == nil {
				var list modelList
				json.NewDecoder(resp.Body).Decode(&list)
				resp.Body.Close()
				all = append(all, list.Data...)
			}
		}

		// Always add configured models (they may not be loaded yet)
		seen := make(map[string]bool)
		for _, m := range all {
			seen[m.ID] = true
		}
		for _, name := range bs.modelNames {
			if !seen[name] {
				all = append(all, model{ID: name, Object: "model", OwnedBy: backendName})
			}
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(modelList{Object: "list", Data: all})
	}
}

func (a *arbiter) makeTagsHandler(backendName string) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		bs := a.backends[backendName]

		type ollamaModel struct {
			Name   string `json:"name"`
			Model  string `json:"model"`
			Size   int64  `json:"size"`
			Digest string `json:"digest"`
		}
		type tagList struct {
			Models []ollamaModel `json:"models"`
		}

		var all tagList

		// For Ollama, fetch live tags
		if backendName == "ollama" {
			a.mu.Lock()
			running := bs.running
			a.mu.Unlock()
			if running {
				if resp, err := http.Get(bs.config.BackendURL.String() + "/api/tags"); err == nil {
					json.NewDecoder(resp.Body).Decode(&all)
					resp.Body.Close()
				}
			}
		}

		// Add configured models
		seen := make(map[string]bool)
		for _, m := range all.Models {
			seen[m.Name] = true
		}
		for _, name := range bs.modelNames {
			if !seen[name] {
				all.Models = append(all.Models, ollamaModel{Name: name, Model: name})
			}
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(all)
	}
}

func (a *arbiter) makeHealthHandler(backendName string) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		bs := a.backends[backendName]
		a.mu.Lock()
		running := bs.running
		model := bs.currentModel
		active := atomic.LoadInt64(&bs.activeReqs)
		a.mu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"status":        "ok",
			"backend":       backendName,
			"running":       running,
			"current_model": model,
			"active_reqs":   active,
		})
	}
}

// --- Idle watcher ---

func (a *arbiter) idleWatcher() {
	for {
		time.Sleep(30 * time.Second)
		a.mu.Lock()
		for _, bs := range a.backends {
			if !bs.running || bs.lastRequest.IsZero() {
				continue
			}
			if time.Since(bs.lastRequest) <= a.idleTimeout {
				continue
			}
			if atomic.LoadInt64(&bs.activeReqs) > 0 {
				continue
			}
			log.Printf("%s idle for %s, stopping",
				bs.config.Name, time.Since(bs.lastRequest).Round(time.Second))
			if bs.config.Name == "ollama" {
				a.unloadOllama()
			} else {
				if err := a.containerStop(bs.config.ContainerName); err != nil {
					log.Printf("warning: failed to stop %s: %v", bs.config.Name, err)
				}
			}
			bs.running = false
			bs.currentModel = ""
		}
		a.mu.Unlock()
	}
}
