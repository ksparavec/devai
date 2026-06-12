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
//	MAX_CONTEXT_LEN   default max context length in tokens (default 262144 = 256K)
//	DEVAI_REASONING   reasoning policy: auto|off|low|medium|high (default auto)
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/signal"
	"path"
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
	// ToolMode is the verified tool-choice path: "auto" (model called
	// spontaneously) or "forced" (only forced-choice round-tripped).
	// Empty when no tool parser was verified or the field pre-dates v2.
	ToolMode string `yaml:"tool_mode,omitempty"`
	// ProbedMaxCtx is the highest context length the probe verified fits at
	// the host's VRAM band — i.e. the largest `fits=true` (HF) or
	// `fully_on_gpu=true` (Ollama) cell at hostKey in the cache. The router
	// trusts this over fittableContext's heuristic at launch time, so
	// MoE/GQA models like gpt-oss-20b aren't artificially clamped to 36K
	// when the probe verified 256K loads cleanly.
	ProbedMaxCtx int `yaml:"-"`
	// MTP is populated from the catalog metadata side-table (loaded
	// separately from the probe cache; see loadCatalogMTP). nil means
	// the catalog declares no `mtp:` block for this row; non-nil means
	// the model supports MTP and the picker may surface the `::mtp`
	// opt-in. The router emits speculative-decoding launch flags only
	// when both this is non-nil AND the per-request override resolves
	// to "on".
	MTP *configSpeculative `yaml:"-"`
}

// configReasoning records what the runtime probe observed for this model.
// Capability values, mirroring scripts/_capability.py (Python StrEnum).
// Wire format on disk and over JSON uses the lowercase strings; these
// constants prevent typo-driven mis-routing at the comparison sites
// below. Keep this block in sync with the Python module when adding
// or removing values.
//
//	structured       – native API exposes a separate reasoning trace field
//	inline           – reasoning appears inline (e.g. <think> blocks) only
//	unsupported      – probe attempted, no reasoning behaviour observed
//	none             – clean non-reasoning model (probe gave a clean answer)
//	unsupported_arch – TERMINAL: backend can't load this arch
//	error            – TERMINAL: probe failed (model load, HTTP 4xx, etc.)
//	unknown          – not yet probed (e.g. vLLM/SGLang pre-probe)
//
// DisableVerified is set only for `structured` capability and reports
// whether sending the protocol's "off" field actually suppresses reasoning.
const (
	CapStructured      = "structured"
	CapInline          = "inline"
	CapUnsupported     = "unsupported"
	CapNone            = "none"
	CapUnsupportedArch = "unsupported_arch"
	CapError           = "error"
	CapUnknown         = "unknown"
)

type configReasoning struct {
	Capability      string `yaml:"capability"`
	DisableVerified *bool  `yaml:"disable_verified,omitempty"`
}

// configSpeculative records the per-model multi-token-prediction (MTP) /
// speculative-decoding parameters loaded from the catalog's `mtp:` block.
// Mirrors the picker convention -- pointer-valued on configModel so a
// nil pointer cleanly distinguishes "model has no MTP available" from
// "model has MTP but it's currently off". Field semantics:
//
//	Method               vLLM/SGLang method name -- "mtp" / "qwen3_5_mtp" /
//	                     "deepseek_mtp" / "eagle" / "eagle3" / etc.
//	Drafter              HF repo path of the external drafter; empty for
//	                     built-in MTP heads (DeepSeek V3, Qwen3.6).
//	NumSpeculativeTokens K -- how many tokens the drafter proposes per round.
type configSpeculative struct {
	Method               string `yaml:"method"`
	Drafter              string `yaml:"drafter,omitempty"`
	NumSpeculativeTokens int    `yaml:"num_speculative_tokens"`
}

// launchConfig holds computed GPU parameters passed to backend entrypoints.
type launchConfig struct {
	MemFraction     float64
	MaxContext      int
	ToolParser      string // empty omits backend-specific tool flags
	ReasoningParser string // empty omits --reasoning-parser
	// Plugin paths are populated when ToolParser / ReasoningParser
	// resolve through the vllm-plugins registry. Each holds the
	// in-container absolute path to a plugin .py file. Empty values
	// keep the launch on the built-in path (no plugin flag emitted).
	// Only vLLM honours these — SGLang's plugin model is Python-import
	// based, not file-path based, so its entrypoint ignores the fields.
	ToolParserPlugin      string
	ReasoningParserPlugin string
	// RecoveryFlags carries per-model CLI args pre-resolved by
	// containerRecreate from the arbiter's recovery registry. Passing
	// them through launchConfig instead of having the entrypoints reach
	// up to a package global keeps the entrypoint functions pure (no
	// hidden global dependency) and lets a future signal-driven reload
	// of recovery-flags.json swap the registry atomically without
	// racing in-flight requests. Empty slice = no flags, which is the
	// path test fixtures already exercise.
	RecoveryFlags []string
	// Speculative carries the resolved multi-token-prediction launch
	// parameters for this request. nil = MTP off (entrypoints emit no
	// speculative-decoding flags); non-nil = the request body's model
	// name carried `::mtp` AND the catalog declares an `mtp:` block
	// for the resolved model name. Emitted as CLI args before
	// RecoveryFlags so a recovery-flags entry can still override the
	// flag value (vLLM/SGLang resolve duplicate flags last-wins). See
	// docs/multi-token-prediction.md Sec. 7.2.
	Speculative *configSpeculative
	// RecoveryImage optionally overrides the vLLM container image for this
	// model only (from the recovery registry's per-model `image` field).
	// Empty = use the global $VLLM_IMAGE default. Lets a single model that
	// needs a different engine build (e.g. DiffusionGemma on the vLLM
	// "gemma" image) run without changing the image for every other model.
	RecoveryImage string
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
		capability := entry.Capability
		if capability == "" {
			capability = CapUnknown
		}
		// First alias is the placeholder Name; the rest are Aliases. The
		// router registers every name into the lookup maps regardless.
		canonical := entry.Aliases[0]
		var aliases []string
		if len(entry.Aliases) > 1 {
			aliases = append([]string(nil), entry.Aliases[1:]...)
		}
		out = append(out, configModel{
			Name:         canonical,
			Aliases:      aliases,
			Digest:       entry.Digest,
			Backend:      []string{"ollama"},
			Size:         fmt.Sprintf("%.2f GB", bestProbe.ActualTotalGB),
			Context:      effCtx,
			ProbedMaxCtx: bestCtx,
			Reasoning: &configReasoning{
				Capability:      capability,
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
	SizeGB          float64 `json:"size_gb,omitempty"`
	MaxContext      int     `json:"max_context"`
	Capability      string  `json:"capability"`
	ToolParser      *string `json:"tool_parser"`
	ReasoningParser *string `json:"reasoning_parser,omitempty"`
	DisableVerified *bool   `json:"disable_verified,omitempty"`
	// ToolMode records HOW the tool parser was verified — `"auto"` if
	// the model spontaneously called the tool with tool_choice="auto",
	// `"forced"` if the call only round-tripped with explicit
	// tool_choice={function:{name:...}}. The router uses this to
	// promote tool_choice on incoming requests for `forced` models
	// (single-tool only) or fail multi-tool requests with an actionable
	// error. Nil for entries that pre-date the field or whose
	// tool_parser didn't verify.
	ToolMode *string                            `json:"tool_mode,omitempty"`
	Probes   map[string]map[string]hfCacheProbe `json:"probes"`
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
	mtpRegistry *catalogMTPRegistry,
) []configModel {
	hostKey := strconv.Itoa(hostVRAMGB)
	out := make([]configModel, 0, len(cache))
	for _, entry := range cache {
		if entry == nil || len(entry.Aliases) == 0 {
			continue
		}
		// Refuse pre-v2 entries. v1 entries lack DisableVerified and
		// have a nil ToolParser, which would otherwise pass through
		// synthesizeHFFromCache as "no curated parsers". The router
		// would then start the model with no --tool-call-parser flag,
		// at which point maybeStripTools silently drops every tools/
		// tool_choice the agent sends -- a corrupt-by-omission state
		// where tool calling appears broken with no log entry. Log
		// loudly and skip; operator re-probes the model to upgrade.
		if entry.SchemaVersion < 2 {
			repo := entry.Repo
			if repo == "" {
				repo = entry.Aliases[0]
			}
			log.Printf("warning: HF cache entry %s is schema_version=%d (< 2); skipping. Re-probe with `make probe-vllm` / `make probe-sglang` to upgrade.", repo, entry.SchemaVersion)
			continue
		}
		// Terminal failure states never produce a serving row.
		if entry.Capability == CapUnsupportedArch || entry.Capability == CapError {
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
		capability := entry.Capability
		if capability == "" {
			capability = CapUnknown
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
		toolMode := ""
		if entry.ToolMode != nil {
			toolMode = *entry.ToolMode
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
		// MTP metadata lives in the catalog side-table, not the probe
		// cache. Look up by repo so the configModel carries the
		// `Speculative` block when the catalog declares one. A nil
		// registry (test fixture or missing models.yaml) leaves MTP
		// unset and the model behaves as a normal non-MTP row.
		var mtpBlock *configSpeculative
		if mtpRegistry != nil {
			if e, ok := mtpRegistry.Lookup(entry.Repo); ok {
				mtpBlock = e
			}
		}
		out = append(out, configModel{
			Name:            canonical,
			Aliases:         aliases,
			Backend:         []string{backendName},
			Repo:            entry.Repo,
			Size:            fmt.Sprintf("%.2f GB", sizeGB),
			Context:         effCtx,
			ProbedMaxCtx:    bestCtx,
			ToolParser:      toolParser,
			ReasoningParser: reasoningParser,
			ToolMode:        toolMode,
			Reasoning: &configReasoning{
				Capability:      capability,
				DisableVerified: entry.DisableVerified,
			},
			MTP: mtpBlock,
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
	mtpRegistry *catalogMTPRegistry,
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
	rows := synthesizeHFFromCache(cache, backendName, hostVRAMGB, operatorMaxCtx, mtpRegistry)
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
	// currentSpec is the speculative-decoding configuration baked into
	// the running container, or nil when MTP is off. A toggle (nil <->
	// non-nil, or any field-level change) triggers a recreate the same
	// way currentModel and currentContext changes do. Reset to nil
	// whenever the container is observed gone.
	currentSpec *configSpeculative
	lastRequest time.Time
	activeReqs  int64
	// Recreate coalescing — without this, a second request that arrives
	// during the 50–60s cold-start `waitForHealthy` window sees
	// running=false, decides it needs its own recreate, and tears down
	// the in-flight one with a duplicate `podman rm` + `podman create`.
	// `recreating` is true while a recreate is in flight, `pendingModel`
	// and `pendingContext` describe its target, and `recreateCond` is the
	// sync.Cond (over arbiter.mu) that waiters block on until the
	// in-flight recreate either completes or fails.
	recreating     bool
	pendingModel   string
	pendingContext int
	recreateCond   *sync.Cond
}

type arbiter struct {
	backends        map[string]*backendState
	mu              sync.Mutex
	ollamaURL       *url.URL
	podmanClient    *http.Client
	idleTimeout     time.Duration
	drainTimeout    time.Duration
	healthTimeout   time.Duration      // configurable per HEALTH_TIMEOUT_SECONDS env (default 600s — vLLM/SGLang cold-start with NVFP4 weights + CUDA graph compilation can exceed 5 min on consumer GPUs)
	modelSizes      map[string]float64 // model name → weight size in GB
	modelContexts   map[string]int     // model name → declared max context (from models.yaml)
	modelCapability map[string]string  // model name → reasoning.capability
	modelDisableOK  map[string]bool    // model name → disable_verified (only when present)
	// Parser names are backend-specific: a model that runs on both vLLM and
	// SGLang (e.g. openai/gpt-oss-20b) typically uses different parser
	// names on each engine (vLLM: openai_gptoss/openai; SGLang: gpt-oss/
	// gpt-oss). Keying by (backend, modelName) prevents the second-loaded
	// backend from overwriting the first's flags and producing a startup
	// `KeyError: invalid tool call parser` on the wrong engine.
	modelToolParser      map[string]map[string]string // backend → model name → --tool-call-parser
	modelReasoningParser map[string]map[string]string // backend → model name → --reasoning-parser
	// modelToolMode keys (backend, modelName) → "auto" | "forced". Drives
	// maybePromoteToolChoice — `forced` models with tool_choice="auto"
	// either get promoted to a specific function (single tool) or
	// rejected with HTTP 400 (multi-tool). `auto` models pass through
	// since the probe verified spontaneous tool-calling works.
	modelToolMode map[string]map[string]string
	// modelProbedMaxCtx is the highest probe-verified context length per
	// (backend, model) at the host VRAM band. fittableContext's heuristic
	// is conservative for MoE/GQA models — when we have a fits=true cell
	// at hostKey, we trust it over the heuristic at launch time.
	modelProbedMaxCtx map[string]map[string]int // backend → model name → highest fits=true ctx
	// modelMTP holds the catalog-declared multi-token-prediction launch
	// params per (backend, model). Populated at startup from
	// configModel.MTP (which the catalog metadata side-table in
	// loadCatalogMTP wired in). A non-nil entry advertises MTP
	// *availability*; whether it actually gets emitted at launch is
	// gated by the per-request `::mtp` suffix (parseMTPOverride). nil
	// = catalog declares no MTP for this row -- the suffix is ignored.
	modelMTP      map[string]map[string]*configSpeculative
	defaultPolicy string // DEVAI_REASONING env value: auto|off|low|medium|high
	totalVRAMGB   float64
	maxContextLen int // global default from MAX_CONTEXT_LEN env (default 262144)
	// pluginRegistry resolves vLLM parser plugin names (loaded from
	// deploy/vllm-plugins.json). Always non-nil; entries map is empty
	// when the registry file is missing or has no plugins.
	pluginRegistry *vllmPluginRegistry
	// recoveryRegistry holds per-model recovery flags (CLI args and env
	// vars) from deploy/recovery-flags.json. Always non-nil; lookups
	// for unknown models return (zero, false). Held on the struct (vs.
	// a package-level var) so a future SIGHUP-style reload can swap
	// the pointer atomically without racing in-flight requests.
	recoveryRegistry *recoveryRegistry
	// healthClient is the short-timeout client used by backendIsServing
	// to probe /health on each request that finds bs.running=true. A
	// per-call &http.Client allocates a fresh transport (with its own
	// connection pool) every time -- hoisted here so successive probes
	// reuse the same idle connection.
	healthClient *http.Client
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

// applyProbeCeiling reconciles fittableContext's heuristic against the
// probe-verified ceiling at the host VRAM band.
//
// fittableContext is conservative — it assumes a fixed KV-bytes-per-token
// from a coarse model-size lookup table, which underestimates the
// achievable context for MoE/GQA models (e.g. gpt-oss-20b: heuristic
// 36K vs probe-verified 256K at 22.30 GB on a 24 GB host).
//
// When `probedMax > 0` we have an authoritative measurement from the
// probe runner: the engine actually loaded and served at that context
// at the host's VRAM band. The router uses min(requestedCtx, probedMax)
// in that case and ignores `heuristicCtx` entirely. When `probedMax == 0`
// (no probe data — legacy YAML rows or pre-probe entries) the heuristic
// stays as the only source of truth.
func applyProbeCeiling(heuristicCtx, requestedCtx, probedMax int) int {
	if probedMax <= 0 {
		return heuristicCtx
	}
	ctx := requestedCtx
	if ctx > probedMax {
		ctx = probedMax
	}
	return ctx
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
		ctx = 262144
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
		// FP8 KV cache halves KV memory vs the default fp16. On a 24 GiB
		// GPU this is what makes 128K+ contexts on 18 GiB NVFP4 weights
		// fit (KV at 128K drops from ~7 GiB to ~3.5 GiB). Must match
		// vllm_command_args in scripts/probe-vllm-reasoning.py so probe-
		// time fit data is consistent with serve-time launches.
		"--kv-cache-dtype", "fp8",
		"--gpu-memory-utilization", fmt.Sprintf("%.2f", lc.MemFraction),
		"--enable-prefix-caching",
		"--trust-remote-code",
		"--served-model-name", modelName,
	}
	// Parser flags are per-model and read from the probe cache. Omit
	// when unverified so a non-matching parser doesn't crash the launch.
	// See deploy/backend-flags.yaml for the verified flag names.
	//
	// Plugin flags MUST precede the parser-name flags: vLLM resolves
	// `--tool-call-parser <name>` against its parser registry at flag-
	// parse time, so the plugin file has to be loaded by then.
	if lc.ReasoningParserPlugin != "" {
		args = append(args, "--reasoning-parser-plugin", lc.ReasoningParserPlugin)
	}
	if lc.ReasoningParser != "" {
		args = append(args, "--reasoning-parser", lc.ReasoningParser)
	}
	if lc.ToolParserPlugin != "" {
		args = append(args, "--tool-parser-plugin", lc.ToolParserPlugin)
	}
	if lc.ToolParser != "" {
		args = append(args, "--enable-auto-tool-choice", "--tool-call-parser", lc.ToolParser)
	}
	// Multi-token-prediction launch flag. Emitted only when the catalog
	// declared MTP for this model AND the request carried `::mtp`. The
	// JSON shape is `{"method": "<m>", "model": "/models/<drafter>",
	// "num_speculative_tokens": <K>}` for external drafters, dropping the
	// `model` field for built-in MTP heads (DeepSeek V3 / Qwen3.6). The
	// drafter dir must already be mounted under VLLM_MODELS_DIR -- the
	// path used here is the in-container address. Emitted before
	// RecoveryFlags so operator overrides can still last-flag-wins
	// override the spec (e.g. force MTP off in production).
	if lc.Speculative != nil {
		if cfg := vllmSpeculativeJSON(lc.Speculative); cfg != "" {
			args = append(args, "--speculative-config", cfg)
		}
	}
	// Per-model recovery flags (e.g. --enforce-eager for checkpoints that
	// OOM at vLLM model-load time on 24G). Pre-resolved by containerRecreate
	// from a.recoveryRegistry. See deploy/recovery-flags.json.
	args = append(args, lc.RecoveryFlags...)
	return args
}

// vllmSpeculativeJSON marshals a configSpeculative into the JSON shape
// vLLM's --speculative-config expects. Compact (no whitespace) so the
// argv element stays single-token through podman's exec layer.
func vllmSpeculativeJSON(s *configSpeculative) string {
	if s == nil || s.Method == "" {
		return ""
	}
	type payload struct {
		Method               string `json:"method"`
		Model                string `json:"model,omitempty"`
		NumSpeculativeTokens int    `json:"num_speculative_tokens"`
	}
	p := payload{
		Method:               s.Method,
		NumSpeculativeTokens: s.NumSpeculativeTokens,
	}
	if s.Drafter != "" {
		// HF repo path → in-container path. Mirrors the convention used
		// for the target model (--model /models/<basename>).
		p.Model = "/models/" + path.Base(s.Drafter)
	}
	if p.NumSpeculativeTokens < 1 {
		p.NumSpeculativeTokens = 1
	}
	out, err := json.Marshal(p)
	if err != nil {
		// Marshal of a small typed struct effectively cannot fail; log
		// loud and degrade to MTP-off rather than corrupting the launch.
		log.Printf("warning: vllmSpeculativeJSON marshal failed: %v", err)
		return ""
	}
	return string(out)
}

// sglangSpeculativeArgs returns the SGLang --speculative-* flags for the
// given config. Empty slice when Speculative is nil. Method-to-algorithm
// mapping:
//
//	mtp        -> NEXTN  (the SGLang alias for EAGLE that mirrors the
//	                       drop-in MTP behaviour vLLM exposes)
//	eagle      -> EAGLE
//	eagle3     -> EAGLE3
//	other      -> the method verbatim (forward-compat for new methods)
//
// Empty Drafter -> built-in head path; SGLang's --speculative-draft-model-path
// is omitted (NEXTN can find the in-model head). External drafter -> path
// emitted as /models/<basename> mirroring vllmSpeculativeJSON.
func sglangSpeculativeArgs(s *configSpeculative) []string {
	if s == nil || s.Method == "" {
		return nil
	}
	algo := strings.ToUpper(s.Method)
	switch s.Method {
	case "mtp", "qwen3_5_mtp", "deepseek_mtp":
		algo = "NEXTN"
	case "eagle":
		algo = "EAGLE"
	case "eagle3":
		algo = "EAGLE3"
	}
	k := s.NumSpeculativeTokens
	if k < 1 {
		k = 1
	}
	args := []string{
		"--speculative-algorithm", algo,
		"--speculative-num-steps", fmt.Sprintf("%d", k),
		"--speculative-num-draft-tokens", fmt.Sprintf("%d", k+1),
		"--speculative-eagle-topk", "1",
	}
	if s.Drafter != "" {
		args = append(args, "--speculative-draft-model-path",
			"/models/"+path.Base(s.Drafter))
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
	// MTP launch flags. SGLang's NVFP4 path is broken upstream (per
	// scripts/model-families.yaml:60-72) so this branch rarely fires
	// in practice -- but emitting the flags keeps the entrypoint
	// forward-compatible for the day SGLang's NVFP4 loader is repaired.
	args = append(args, sglangSpeculativeArgs(lc.Speculative)...)
	// Per-model recovery flags (mirrors vllmEntrypoint). SGLang's NVFP4
	// loader path is currently broken upstream so this branch rarely
	// fires today, but the symmetry keeps the behaviour predictable.
	args = append(args, lc.RecoveryFlags...)
	return args
}

// specEqual returns true when two configSpeculative pointers carry the
// same launch parameters. Both nil = equal (MTP off in both). One nil
// = different (toggle). Both non-nil = field-by-field compare. Used by
// containerRecreate's specChanged check to decide whether a recreate
// is needed.
func specEqual(a, b *configSpeculative) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	return a.Method == b.Method &&
		a.Drafter == b.Drafter &&
		a.NumSpeculativeTokens == b.NumSpeculativeTokens
}

// specLabel produces a short human-readable description of an MTP spec
// for log lines. "off" for nil; otherwise "<method>/k=<N>". Drafter
// path is intentionally omitted -- the log already names the model.
func specLabel(s *configSpeculative) string {
	if s == nil {
		return "off"
	}
	return fmt.Sprintf("%s/k=%d", s.Method, s.NumSpeculativeTokens)
}

// --- Main ---

// arbiterMode selects single-host vs cluster behaviour. Default
// "single" preserves the pre-cluster-mode code path byte-for-byte.
// Per docs/plans/gpu-arbiter-cluster-mode.md decision 7 + 11.
var arbiterMode = flag.String(
	"mode", env("DEVAI_MODE", "single"),
	"arbiter mode: single|worker|head",
)

func main() {
	flag.Parse()
	switch *arbiterMode {
	case "single":
		// Fall through to the existing single-host code path.
	case "worker":
		runWorkerMode()
		return
	case "head":
		runHeadMode()
		return
	default:
		log.Fatalf("unknown --mode %q (want single|worker|head)", *arbiterMode)
	}

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
	// Parse GPU_MEMORY_GB once as float64 -- envInt's Sscanf("%d") silently
	// falls back to the default on values like "23.5", while envFloat
	// parses them correctly. Round to nearest int for the band-key
	// consumer (synthesizeFromCache filters on string(int) bands).
	hostVRAMFloat := envFloat("GPU_MEMORY_GB", 24.0)
	hostVRAMGB := int(math.Round(hostVRAMFloat))
	operatorMaxCtx := envInt("MAX_CONTEXT_LEN", 262144)
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

	// Catalog metadata side-table for MTP launch params. Loaded once at
	// startup; consulted from synthesizeHFFromCache so each HF row picks
	// up its MTP block (if any) without re-introducing models.yaml as a
	// primary source of fit truth. Missing file = empty registry = MTP
	// off for everyone; safe degradation.
	catalogPath := env("CATALOG_FILE", "/etc/devai/models.yaml")
	catalogMTP := loadCatalogMTP(catalogPath)

	// Append HF-backend rows synthesized from the per-backend probe caches.
	// Each cache is independent; missing files are not fatal — the
	// corresponding backend just exposes no models. Tool-parser values
	// flow through the configModel.ToolParser field into the
	// modelToolParser lookup, which the parameterized vllmEntrypoint
	// (Phase 0) reads when starting the backend.
	vllmCachePath := env("VLLM_PROBE_CACHE", "/etc/devai/.vllm-reasoning-cache.json")
	cfg.Models = append(cfg.Models,
		loadHFCache(vllmCachePath, "vllm", hostVRAMGB, operatorMaxCtx, catalogMTP)...)
	sglangCachePath := env("SGLANG_PROBE_CACHE", "/etc/devai/.sglang-reasoning-cache.json")
	cfg.Models = append(cfg.Models,
		loadHFCache(sglangCachePath, "sglang", hostVRAMGB, operatorMaxCtx, catalogMTP)...)

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
	modelToolParser := make(map[string]map[string]string)      // backend → model → --tool-call-parser
	modelReasoningParser := make(map[string]map[string]string) // backend → model → --reasoning-parser
	modelToolMode := make(map[string]map[string]string)        // backend → model → "auto" | "forced"
	modelProbedMaxCtx := make(map[string]map[string]int)       // backend → model → highest fits=true ctx
	modelMTP := make(map[string]map[string]*configSpeculative) // backend → model → catalog MTP block
	capCounts := make(map[string]int)
	for _, m := range cfg.Models {
		names := append([]string{m.Name}, m.Aliases...)
		sz := parseSizeGB(m.Size)
		capability := CapUnknown
		if m.Reasoning != nil && m.Reasoning.Capability != "" {
			capability = m.Reasoning.Capability
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
			modelCapability[name] = capability
			if disableOK {
				modelDisableOK[name] = true
			}
			// Parser maps and the probe-verified ctx ceiling are keyed by
			// backend so the same model name can carry different values on
			// vLLM vs SGLang without one backend overwriting the other.
			for _, backend := range m.Backend {
				if m.ToolParser != "" {
					if modelToolParser[backend] == nil {
						modelToolParser[backend] = make(map[string]string)
					}
					modelToolParser[backend][name] = m.ToolParser
				}
				if m.ReasoningParser != "" {
					if modelReasoningParser[backend] == nil {
						modelReasoningParser[backend] = make(map[string]string)
					}
					modelReasoningParser[backend][name] = m.ReasoningParser
				}
				if m.ToolMode != "" {
					if modelToolMode[backend] == nil {
						modelToolMode[backend] = make(map[string]string)
					}
					modelToolMode[backend][name] = m.ToolMode
				}
				if m.ProbedMaxCtx > 0 {
					if modelProbedMaxCtx[backend] == nil {
						modelProbedMaxCtx[backend] = make(map[string]int)
					}
					modelProbedMaxCtx[backend][name] = m.ProbedMaxCtx
				}
				if m.MTP != nil {
					if modelMTP[backend] == nil {
						modelMTP[backend] = make(map[string]*configSpeculative)
					}
					modelMTP[backend][name] = m.MTP
				}
			}
		}
		// Count capability once per canonical row, not once per alias —
		// otherwise a model with N aliases would dominate the histogram.
		capCounts[capability]++
	}
	policy := strings.ToLower(env("DEVAI_REASONING", "auto"))
	if !validPolicy(policy) {
		log.Printf("warning: invalid DEVAI_REASONING=%q; falling back to auto", policy)
		policy = "auto"
	}
	log.Printf("reasoning policy: %s; capability counts: %v", policy, capCounts)
	// Reuse the float parsed above (hostVRAMFloat) so the int band-key
	// consumer and the float memory-fraction consumer cannot disagree
	// when GPU_MEMORY_GB is e.g. "23.5".
	totalVRAMGB := hostVRAMFloat
	maxCtx := envInt("MAX_CONTEXT_LEN", 262144)

	pluginRegistry := loadVLLMPluginRegistry(
		env("VLLM_PLUGINS_REGISTRY", "/etc/devai/vllm-plugins.json"),
		env("VLLM_PLUGINS_HOST_DIR", ""),
	)
	recoveryRegistry := loadRecoveryRegistry(
		env("RECOVERY_FLAGS_REGISTRY", "/etc/devai/recovery-flags.json"),
	)

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
		modelToolMode:        modelToolMode,
		modelProbedMaxCtx:    modelProbedMaxCtx,
		modelMTP:             modelMTP,
		defaultPolicy:        policy,
		totalVRAMGB:          totalVRAMGB,
		maxContextLen:        maxCtx,
		pluginRegistry:       pluginRegistry,
		recoveryRegistry:     recoveryRegistry,
		healthClient:         &http.Client{Timeout: 2 * time.Second},
	}

	for _, bc := range backends {
		var proxy *httputil.ReverseProxy
		if bc.Name == "ollama" {
			proxy = newProxy(bc.BackendURL)
		} else {
			proxy = newSmartProxy(bc.BackendURL)
		}
		a.backends[bc.Name] = &backendState{
			config:       bc,
			proxy:        proxy,
			modelNames:   modelsForBackend(cfg.Models, bc.Name),
			recreateCond: sync.NewCond(&a.mu),
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
	reqURL := fmt.Sprintf("http://d/v4.0.0/libpod/containers/%s/stop?timeout=10", name)
	resp, err := a.podmanClient.Post(reqURL, "", nil)
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
	req, err := http.NewRequest("DELETE", delURL, nil)
	if err != nil {
		log.Printf("warning: containerRemove %s: build request: %v", name, err)
		return
	}
	resp, err := a.podmanClient.Do(req)
	if err != nil {
		log.Printf("warning: containerRemove %s: %v", name, err)
		return
	}
	defer resp.Body.Close()
	// 204/200 = removed, 404 = already gone (both fine). Anything else
	// means the container will still be there on the next create call,
	// which then fails with "name in use" -- log loudly so operators
	// can correlate.
	if resp.StatusCode >= 300 && resp.StatusCode != http.StatusNotFound {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		log.Printf("warning: containerRemove %s: HTTP %s: %s", name, resp.Status, body)
	}
}

func (a *arbiter) containerIsRunning(name string) bool {
	if a.podmanClient == nil {
		return false
	}
	reqURL := fmt.Sprintf("http://d/v4.0.0/libpod/containers/%s/json", name)
	resp, err := a.podmanClient.Get(reqURL)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	var info struct {
		State struct {
			Status string `json:"Status"`
		} `json:"State"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		// A malformed body would otherwise leave info.State.Status == ""
		// and we'd return false -- triggering an unnecessary recreate of
		// a backend that may be perfectly healthy. Treat decode failure
		// as "unknown" but log so operators can spot a podman API regression.
		log.Printf("warning: containerIsRunning %s: decode failed: %v", name, err)
		return false
	}
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
	resp, err := a.healthClient.Get(healthURL)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	io.Copy(io.Discard, resp.Body)
	return resp.StatusCode == http.StatusOK
}

// resolvePluginLaunch consults the vllm-plugins registry for the
// parser names already populated on lc, fills in lc.ToolParserPlugin /
// lc.ReasoningParserPlugin when matches are found, and returns the
// libpod mount spec to bind-mount the plugin directory into the
// recreated container. Returns (nil, nil) when no plugins are needed —
// every parser is built-in, the registry has no entries, or the
// backend doesn't support file-path plugins.
//
// A non-nil error is returned only when a plugin IS required (parser
// name matched a registry entry) but VLLM_PLUGINS_HOST_DIR is unset —
// otherwise the launch would proceed without the plugin file accessible
// in the container and crash with "parser not found" at startup.
func (a *arbiter) resolvePluginLaunch(
	backendName string, lc *launchConfig,
) (map[string]any, error) {
	// Only vLLM honours `--*-parser-plugin <path>`. SGLang registers
	// parsers via Python imports — file-path mounts wouldn't help.
	if backendName != "vllm" || a.pluginRegistry == nil {
		return nil, nil
	}
	var matched bool
	if entry, ok := a.pluginRegistry.Lookup(lc.ToolParser); ok {
		if entry.Kind != "tool" {
			return nil, fmt.Errorf(
				"vllm plugin %q registered under kind=%q but used as tool parser",
				lc.ToolParser, entry.Kind,
			)
		}
		lc.ToolParserPlugin = a.pluginRegistry.ContainerPath(entry.File)
		matched = true
	}
	if entry, ok := a.pluginRegistry.Lookup(lc.ReasoningParser); ok {
		if entry.Kind != "reasoning" {
			return nil, fmt.Errorf(
				"vllm plugin %q registered under kind=%q but used as reasoning parser",
				lc.ReasoningParser, entry.Kind,
			)
		}
		lc.ReasoningParserPlugin = a.pluginRegistry.ContainerPath(entry.File)
		matched = true
	}
	if !matched {
		return nil, nil
	}
	if strings.TrimSpace(a.pluginRegistry.HostDir) == "" {
		return nil, fmt.Errorf(
			"vllm plugin required (tool=%q reasoning=%q) but VLLM_PLUGINS_HOST_DIR is empty",
			lc.ToolParser, lc.ReasoningParser,
		)
	}
	return map[string]any{
		"destination": a.pluginRegistry.ContainerDir,
		"source":      a.pluginRegistry.HostDir,
		"type":        "bind",
		"options":     []string{"ro"},
	}, nil
}

// buildContainerSpec assembles the libpod container-create JSON spec.
// Extracted from containerRecreate so the lifecycle method stays focused
// on the launch sequence (stop, remove, compute launch config, post-
// create, post-start) and the spec layout has one home.
//
// `pluginVolume` is the optional plugin bind mount returned by
// resolvePluginLaunch (nil when the backend uses built-in parsers or
// SGLang's Python-import plugin model). `recoveryEnv` carries per-model
// env vars pre-resolved from the recovery registry.
func buildContainerSpec(
	cfg backendConfig,
	modelName string,
	lc launchConfig,
	pluginVolume map[string]any,
	recoveryEnv map[string]string,
) map[string]any {
	mounts := []map[string]any{{
		"destination": "/models",
		"source":      cfg.ModelsDir,
		"type":        "bind",
		"options":     []string{"ro"},
	}}
	if pluginVolume != nil {
		mounts = append(mounts, pluginVolume)
	}

	// Per-model image override (recovery registry `image` field) wins over
	// the backend's global default. Lets one model run on a different
	// engine build (e.g. DiffusionGemma on the vLLM "gemma" image) without
	// changing the image for every other model. The override travels with
	// the model name, so the existing model-change -> recreate path already
	// swaps it in; no separate image-change detection is needed.
	image := cfg.Image
	if lc.RecoveryImage != "" {
		image = lc.RecoveryImage
	}
	spec := map[string]any{
		"image":        image,
		"name":         cfg.ContainerName,
		"entrypoint":   cfg.Entrypoint(modelName, lc),
		"command":      []string{},
		"mounts":       mounts,
		"hostadd":      []string{"host.containers.internal:host-gateway"},
		"netns":        map[string]any{"nsmode": "bridge"},
		"Networks":     map[string]any{cfg.Network: map[string]any{}},
		"devices":      []map[string]any{{"path": "nvidia.com/gpu=all"}},
		"selinux_opts": []string{"disable"},
		"hostname":     cfg.Name,
	}
	// Merge per-backend env (cfg.EnvVars) with per-model recovery env.
	// Per-model entries win on key collision -- recovery env exists
	// precisely to override defaults for borderline checkpoints.
	envMap := make(map[string]string, len(cfg.EnvVars))
	for k, v := range cfg.EnvVars {
		envMap[k] = v
	}
	for k, v := range recoveryEnv {
		envMap[k] = v
	}
	if len(envMap) > 0 {
		spec["env"] = envMap
	}
	return spec
}

// containerRecreate launches the backend container with the given model.
// `desiredCtx > 0` overrides the catalog cap (used when a request carries a
// "<model>@<ctx>" picker override); 0 falls back to the catalog cap. The
// chosen context is always clamped against MAX_CONTEXT_LEN and against the
// memory-driven fittableContext inside computeLaunchConfig.
//
// `desiredSpec` carries the multi-token-prediction config to bake into the
// backend container. Nil = MTP off (no --speculative-config / no
// --speculative-* SGLang flags emitted). Non-nil = the request's `::mtp`
// suffix was on AND the catalog declares MTP for this model -- the
// entrypoint emits the appropriate launch flags. A spec change vs. the
// running container triggers a recreate the same way model/context
// changes do.
func (a *arbiter) containerRecreate(bs *backendState, modelName string, desiredCtx int, desiredSpec *configSpeculative) error {
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
	lc.MaxContext = applyProbeCeiling(
		lc.MaxContext, requestedCtx,
		a.modelProbedMaxCtx[cfg.Name][modelName],
	)
	if parser := a.modelToolParser[cfg.Name][modelName]; parser != "" {
		lc.ToolParser = parser
	}
	if parser := a.modelReasoningParser[cfg.Name][modelName]; parser != "" {
		lc.ReasoningParser = parser
	}
	// Pre-resolve recovery flags from the arbiter's registry once, then
	// hand both Flags (CLI args) and Env (env vars) to downstream code
	// via launchConfig and the spec envMap below. Single lookup keeps
	// the two halves synced if the registry ever swaps mid-call.
	var recoveryEnv map[string]string
	if rec, ok := a.recoveryRegistry.Lookup(modelName); ok {
		lc.RecoveryFlags = rec.Flags
		recoveryEnv = rec.Env
		lc.RecoveryImage = rec.Image
	}
	// Speculative-decoding config (MTP). Set by the request handler when
	// the model name carried `::mtp` AND the catalog declared an `mtp:`
	// block for it. Plumbed through launchConfig so the entrypoint
	// builders can emit --speculative-config / --speculative-* flags
	// before RecoveryFlags get appended (operator overrides win
	// last-flag-wins).
	lc.Speculative = desiredSpec
	// Resolve parser plugin paths and the plugin volume. Only vLLM
	// supports file-path plugins; SGLang's plugin model is Python-
	// import based, so its launchConfig plugin fields stay empty.
	pluginVolume, perr := a.resolvePluginLaunch(cfg.Name, &lc)
	if perr != nil {
		return perr
	}
	log.Printf("  %s launch: model=%.1f GB, gpu=%.1f GB → fraction=%.2f, context=%dk, reasoning=%q tool=%q tool_plugin=%q",
		cfg.Name, modelSizeGB, a.totalVRAMGB, lc.MemFraction, lc.MaxContext/1024,
		lc.ReasoningParser, lc.ToolParser, lc.ToolParserPlugin)

	spec := buildContainerSpec(cfg, modelName, lc, pluginVolume, recoveryEnv)

	body, err := json.Marshal(spec)
	if err != nil {
		// A marshal failure would otherwise POST a nil body to libpod,
		// which silently creates a container with no entrypoint -- the
		// next request then 502s with no obvious cause. Fail loud.
		return fmt.Errorf("marshal container spec for %s: %w", cfg.ContainerName, err)
	}
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
	// Per-probe timeout: the global http.DefaultClient has none, so a
	// half-open TCP connection or a hung backend could block a single
	// http.Get for the full 600s health window with no further attempts.
	// 5s per probe is well above any realistic /health turnaround and
	// keeps the polling loop progressing.
	client := &http.Client{Timeout: 5 * time.Second}
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		resp, err := client.Get(healthURL)
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
	if err := json.NewDecoder(resp.Body).Decode(&ps); err != nil {
		// A malformed /api/ps body would silently leave ps.Models empty,
		// the loop below would be a no-op, and stopOtherBackends would
		// then believe Ollama's GPU memory was released -- the vLLM/SGLang
		// container then starts on a GPU that still has Ollama's weights
		// resident, producing OOM or corrupted inference. Refuse to
		// proceed and log loudly so the operator sees the real cause.
		log.Printf("error: unloadOllama: cannot decode /api/ps: %v -- skipping unload (GPU may still be held by ollama)", err)
		return
	}

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
		if name == targetName {
			continue
		}
		// Wait out any in-flight recreate on this backend BEFORE checking
		// `running`. A recreate that started after we last released a.mu
		// (e.g. during another backend's waitForHealthy window) sets
		// recreating=true but only flips running=true after the health
		// wait completes. Without this wait we would observe running=false,
		// skip the backend, return, and let our caller fire its own
		// containerRecreate -- with the still-in-flight recreate competing
		// for the same GPU. Block on recreateCond (held over a.mu) until
		// the in-flight recreate exits its defer (success or failure).
		for bs.recreating {
			bs.recreateCond.Wait()
		}
		if !bs.running {
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
func (a *arbiter) ensureBackendRunning(bs *backendState, modelName string, desiredCtx int, desiredSpec *configSpeculative) error {
	if bs.config.Name == "ollama" {
		if !bs.running {
			a.stopOtherBackends("ollama")
			bs.running = true
		}
		return nil
	}

	// Recreate coalescing. If a recreate is already in flight on this
	// backend we MUST NOT fire our own — the second `podman rm` would
	// kill the half-built container from the first. Wait on the cond
	// until the in-flight recreate finishes (success or failure), then
	// re-evaluate from the top: maybe it produced exactly what we want
	// (same modelName + ctx) and we can return clean, or maybe it
	// targeted a different model and we now need to fire our own.
	for bs.recreating {
		bs.recreateCond.Wait()
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
		bs.currentSpec = nil
	}

	// Recreate when the model, the baked context cap, OR the
	// speculative-decoding config changed. The third trigger is what
	// makes the `::mtp` / `::nomtp` per-request suffix work -- toggling
	// MTP requires re-launching the backend container with (or without)
	// --speculative-config, which only takes effect at startup.
	modelChanged := modelName != "" && bs.currentModel != modelName
	contextChanged := desiredCtx > 0 && bs.currentContext > 0 && bs.currentContext != desiredCtx
	specChanged := !specEqual(bs.currentSpec, desiredSpec)
	needRecreate := !bs.running || modelChanged || contextChanged || specChanged
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
	} else if specChanged {
		log.Printf("switching %s MTP config (model %s): %s → %s",
			bs.config.Name, modelName, specLabel(bs.currentSpec), specLabel(desiredSpec))
	}
	specSummary := specLabel(desiredSpec)
	log.Printf("starting %s with model %s (ctx=%d, mtp=%s)...",
		bs.config.Name, modelName, desiredCtx, specSummary)

	// Mark the recreate in flight BEFORE releasing the lock. Concurrent
	// callers landing in the wait loop above will block on recreateCond
	// instead of duplicating the work. `defer` ensures the flag and
	// broadcast happen on every exit path (recreate failure, health
	// timeout, panic) — without that, a failure would leave waiters
	// stuck forever.
	bs.recreating = true
	bs.pendingModel = modelName
	bs.pendingContext = desiredCtx
	defer func() {
		bs.recreating = false
		bs.pendingModel = ""
		bs.pendingContext = 0
		bs.recreateCond.Broadcast()
	}()

	if err := a.containerRecreate(bs, modelName, desiredCtx, desiredSpec); err != nil {
		return fmt.Errorf("failed to start %s: %w", bs.config.Name, err)
	}

	// Release lock during health wait so concurrent /v1/* requests can
	// queue / waiters can park on the cond. The recreating flag prevents
	// them from kicking off duplicate recreates. The inner closure +
	// `defer a.mu.Lock()` guarantees we re-acquire BEFORE returning from
	// this scope, even on a panic out of waitForHealthy. Without that,
	// the outer recreate defer above would fire without the lock held
	// and race on bs.recreating / bs.pendingModel / bs.pendingContext.
	err := func() error {
		a.mu.Unlock()
		defer a.mu.Lock()
		return a.waitForHealthy(bs, a.healthTimeout)
	}()

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
	// Spec is recorded unconditionally (even when nil) so the next
	// request can compute specChanged correctly: a request that omits
	// `::mtp` after a previous `::mtp`-enabled launch is a recreate
	// trigger, not a no-op.
	bs.currentSpec = desiredSpec
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
		// mtpOverride must outlive the POST-body block: the
		// desiredSpec resolution below reads it for both POSTs and
		// GETs. A GET leaves it at "" (no override → MTP off).
		var mtpOverride string
		if req.Method == http.MethodPost && req.Body != nil {
			// Cap the request body before reading. Without this any peer
			// on devai-net (or any container in the compose network) can
			// stream an arbitrarily large body, exhaust RAM, and block
			// concurrent inference behind the arbiter mutex. 32 MB covers
			// any realistic chat-completion payload (huge multi-turn
			// histories with base64 image content are well below this).
			req.Body = http.MaxBytesReader(w, req.Body, 32<<20)
			body, err := io.ReadAll(req.Body)
			if err != nil {
				// MaxBytesReader's error string is informative ("http:
				// request body too large") but the HTTP status it has
				// already written depends on Go version. Be explicit.
				http.Error(w, `{"error":"failed to read body or body too large"}`, http.StatusRequestEntityTooLarge)
				return
			}

			var parsed struct {
				Model string `json:"model"`
			}
			if err := json.Unmarshal(body, &parsed); err != nil {
				// Malformed JSON would otherwise leave parsed.Model="",
				// flow through to ensureBackendRunning, and surface as
				// a confusing HTTP 503 ("model name required"). Fail
				// fast with 400 and log so operators can find which
				// client is misbehaving.
				log.Printf("warning: bad JSON body from %s: %v", req.RemoteAddr, err)
				http.Error(w, `{"error":"invalid JSON request body"}`, http.StatusBadRequest)
				return
			}

			// Resolve the per-request num_ctx and strip any suffixes
			// from the model name. Suffix order (convention): the
			// picker emits `<name>::<reasoning>::<mtp>@<ctx>`. We strip
			// right-to-left in the same order so each parser sees only
			// the suffix it owns:
			//   1. parseCtxOverride   strips trailing @<int>.
			//   2. parseMTPOverride   strips trailing ::mtp / ::nomtp.
			//   3. parseReasoningOverride strips trailing ::<reasoning>.
			// Each parser falls through (returns input unchanged) on an
			// unrecognised token, so a name that legitimately contains
			// `::` survives intact.
			//
			// Override priority for num_ctx:
			//   1. Picker-supplied @<int>  → force-injected (user choice).
			//   2. Registered modelContexts cap (= min(model_max,
			//      MAX_CONTEXT_LEN) from the probe cache) → soft cap,
			//      only set when the client didn't supply num_ctx.
			//   3. None → request passes through unchanged.
			ctxStripped, ctxOverride := parseCtxOverride(parsed.Model)
			var mtpStripped string
			mtpStripped, mtpOverride = parseMTPOverride(ctxStripped)
			cleanName, reasoningOverride := parseReasoningOverride(mtpStripped)
			// Defense-in-depth: refuse path-traversal segments before the
			// name flows into vllmEntrypoint/sglangEntrypoint where it is
			// concatenated as `--model /models/<name>`. The /models bind
			// mount is read-only so write-impact is bounded, but a name
			// containing `..` could still resolve a backend's model loader
			// to a file outside the intended directory. Slash is allowed
			// because legitimate HF repos look like `nvidia/Qwen3-8B-NVFP4`.
			if !isSafeModelName(cleanName) {
				http.Error(w, `{"error":"invalid model name"}`, http.StatusBadRequest)
				return
			}
			// Allowlist check for vLLM/SGLang: the name MUST be one the
			// router was started with (loaded from the probe cache). The
			// Ollama path forwards the name to upstream which has its own
			// allowlist via locally pulled tags, so we skip this gate there.
			if backendName != "ollama" && cleanName != "" {
				known := false
				for _, n := range bs.modelNames {
					if n == cleanName {
						known = true
						break
					}
				}
				if !known {
					http.Error(w, fmt.Sprintf(`{"error":"unknown model %q for %s"}`, cleanName, backendName), http.StatusNotFound)
					return
				}
			}
			numCtx = ctxOverride
			force := ctxOverride > 0
			if numCtx == 0 {
				if ctxCap, ok := a.modelContexts[cleanName]; ok && ctxCap > 0 {
					numCtx = ctxCap
				}
			}
			if cleanName != parsed.Model {
				body = setTopJSONField(body, "model", cleanName)
			}
			// Only inject options.num_ctx on Ollama's native API. Empirical
			// finding (2026-05-13): Ollama discards options.num_ctx on its
			// OpenAI-compat (/v1/chat/completions) and Anthropic-compat
			// (/v1/messages) surfaces -- the load always picks up the
			// global OLLAMA_CONTEXT_LENGTH. vLLM/SGLang don't parse
			// `options` at all (ctx is baked into the container via
			// --max-model-len / --context-length at recreate time, driven
			// by the @<ctx> picker suffix). So this rewrite is meaningful
			// only on /api/chat and /api/generate.
			if backendName == "ollama" &&
				(req.URL.Path == "/api/chat" || req.URL.Path == "/api/generate") {
				body = setNumCtx(body, numCtx, force)
			}
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
			// Reasoning + MTP + inline-reasoning guard for vllm#34650.
			// When the picker (or a client) opts into MTP via `::mtp`
			// on a model whose capability is `inline` while the
			// effective reasoning policy is anything but `off`, vLLM's
			// reasoning parser fails to detect the `</think>` close
			// token under speculative decoding and reasoning content
			// bleeds into the regular content stream. Refuse loud
			// rather than silently corrupting the stream. Operator
			// remedy: pick reasoning OFF (::nothink) or MTP OFF.
			if mtpOverride == "on" && policy != "off" &&
				a.modelCapability[policyModel] == CapInline {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusBadRequest)
				_, _ = w.Write([]byte(
					`{"error":"reasoning+MTP combo blocked by vllm#34650 for inline-reasoning models; ` +
						`pass ::nothink or omit ::mtp"}`))
				return
			}
			// Promote tool_choice for models the probe verified only
			// via forced choice (mode=forced). Single-tool requests
			// get auto/absent → {function:{name:...}}; multi-tool
			// requests are rejected with HTTP 400 instead of silently
			// running with auto on a model that won't call. Models
			// with mode=auto or no verified tool_parser pass through.
			promoted, perr := a.maybePromoteToolChoice(backendName, policyModel, body)
			if perr != nil {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(perr.HTTPStatus())
				_, _ = w.Write(perr.JSON())
				return
			}
			body = promoted
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

		// Resolve the speculative-decoding config for this request.
		// Three preconditions: (a) the catalog declared `mtp:` for this
		// (backend, model) -- looked up in a.modelMTP; (b) the request
		// name carried `::mtp` -- captured in mtpOverride; (c) the
		// override wasn't explicitly `nomtp`. When any condition fails,
		// desiredSpec stays nil and the entrypoint emits no
		// speculative-decoding flags. Ollama never participates --
		// SGLang/vLLM only.
		var desiredSpec *configSpeculative
		if backendName != "ollama" && mtpOverride == "on" && modelName != "" {
			if backendMTP, ok := a.modelMTP[backendName]; ok {
				if spec, ok := backendMTP[modelName]; ok && spec != nil {
					desiredSpec = spec
				}
			}
		}

		a.mu.Lock()
		bs.lastRequest = time.Now()
		if err := a.ensureBackendRunning(bs, modelName, numCtx, desiredSpec); err != nil {
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
// promoteToolChoiceError is the structured payload an HTTP handler should
// emit when maybePromoteToolChoice rejects a multi-tool auto request. The
// shape mirrors OpenAI's error envelope so SDKs surface it cleanly.
type promoteToolChoiceError struct {
	Model     string
	ToolNames []string
}

func (e *promoteToolChoiceError) Error() string {
	return fmt.Sprintf("tool_choice pinning required for %s with multiple tools", e.Model)
}

// HTTPStatus is the HTTP status code to return for this error.
func (e *promoteToolChoiceError) HTTPStatus() int { return http.StatusBadRequest }

// JSON returns the OpenAI-shaped error body.
func (e *promoteToolChoiceError) JSON() []byte {
	tools := strings.Join(e.ToolNames, ", ")
	msg := fmt.Sprintf(
		"Model %q requires tool_choice to be pinned to a specific function "+
			"when called with multiple tools. Set tool_choice to "+
			`{"type":"function","function":{"name":"<one of: %s>"}}`+
			", or route this request to a non-reasoning model "+
			"(e.g. Qwen3.5-9B-Q8, Llama-3.1-8B-Instruct) that handles "+
			"auto tool_choice reliably.",
		e.Model, tools,
	)
	doc := map[string]any{
		"error": map[string]any{
			"type":    "invalid_request_error",
			"code":    "tool_choice_pinning_required",
			"message": msg,
			"param":   "tool_choice",
		},
	}
	out, _ := json.Marshal(doc)
	return out
}

// maybePromoteToolChoice rewrites the request body to pin tool_choice to a
// specific function when the model's probe verified tool calls only via
// forced choice (tool_mode="forced") AND the request leaves tool_choice up
// to the model. Two cases:
//
//  1. Single tool in `tools`: rewrite tool_choice to that function name.
//     The agent gets a working tool call without any agent-side change.
//  2. Multiple tools: return *promoteToolChoiceError. The handler turns
//     it into HTTP 400. The router can't pick a tool for the agent, and
//     forced-only models won't pick one themselves with auto choice.
//
// All other shapes pass through:
//   - tool_choice already pinned to a function: agent took ownership.
//   - tool_choice="required": agent forced some call; model picks. Best
//     effort — for forced-only models this still often fails to elicit a
//     call within budget, but the agent made the choice.
//   - tool_choice="none": agent explicitly disabled tools.
//   - Models with tool_mode="auto": probe verified spontaneous calls work.
//   - Models without a verified tool_parser: handled by maybeStripTools.
//
// Returns (rewritten body, nil) on success, (nil, *promoteToolChoiceError)
// on the multi-tool reject path, or (original body, nil) when no rewrite
// applies. JSON-decode failure returns the body unchanged with no error —
// matches the defensive posture of the other rewrite helpers.
func (a *arbiter) maybePromoteToolChoice(
	backendName, modelName string, body []byte,
) ([]byte, *promoteToolChoiceError) {
	if backendName != "vllm" && backendName != "sglang" {
		return body, nil
	}
	if a.modelToolMode[backendName][modelName] != "forced" {
		return body, nil
	}
	var doc map[string]any
	if err := json.Unmarshal(body, &doc); err != nil {
		return body, nil
	}
	tools, _ := doc["tools"].([]any)
	if len(tools) == 0 {
		return body, nil
	}
	// Inspect tool_choice. Absent or "auto" → eligible for rewrite.
	// Anything else (a function spec map, "required", "none") → pass.
	choice, present := doc["tool_choice"]
	if present {
		if s, ok := choice.(string); !ok || (s != "" && s != "auto") {
			return body, nil
		}
	}
	if len(tools) == 1 {
		name := toolNameAt(tools, 0)
		if name == "" {
			return body, nil
		}
		doc["tool_choice"] = map[string]any{
			"type":     "function",
			"function": map[string]any{"name": name},
		}
		out, err := json.Marshal(doc)
		if err != nil {
			return body, nil
		}
		return out, nil
	}
	// Multi-tool reject. Collect names so the error message is
	// actionable — the agent author can pick one and pin client-side.
	names := make([]string, 0, len(tools))
	for i := range tools {
		if n := toolNameAt(tools, i); n != "" {
			names = append(names, n)
		}
	}
	return nil, &promoteToolChoiceError{Model: modelName, ToolNames: names}
}

// toolNameAt extracts the function name from `tools[i]`. OpenAI shape:
// {"type":"function","function":{"name":"...", ...}}. Returns "" on any
// type assertion failure — the caller treats that as "no usable name".
func toolNameAt(tools []any, i int) string {
	if i < 0 || i >= len(tools) {
		return ""
	}
	t, _ := tools[i].(map[string]any)
	fn, _ := t["function"].(map[string]any)
	name, _ := fn["name"].(string)
	return name
}

func (a *arbiter) maybeStripTools(backendName, modelName string, body []byte) []byte {
	if backendName != "vllm" && backendName != "sglang" {
		return body
	}
	if a.modelToolParser[backendName][modelName] != "" {
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
	// Tell the operator the strip happened. Without this, "tool calling
	// doesn't work for model X" reports have no trail -- the agent sent
	// tools, the router dropped them, and the upstream never saw them.
	log.Printf("info: stripped tools/tool_choice for %s/%s (no probe-verified tool_parser)",
		backendName, modelName)
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
		log.Printf("info: vllm/%s reasoning ENABLE (policy=%q, effort=%s)",
			modelName, policy, openAIReasoningEffort(policy))
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
		log.Printf("info: vllm/%s reasoning DISABLE (policy=%q)", modelName, policy)
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
		log.Printf("info: sglang/%s reasoning ENABLE (policy=%q)", modelName, policy)
		body = setJSONFieldIfAbsent(
			body, []string{"separate_reasoning"}, "separate_reasoning", true,
		)
		return setNestedJSONFieldIfAbsent(
			body,
			[]string{"extra_body", "chat_template_kwargs", "enable_thinking"},
			true,
		)
	case reasoningDisable:
		log.Printf("info: sglang/%s reasoning DISABLE (policy=%q)", modelName, policy)
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
	case CapStructured:
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
	case CapInline:
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

// parseMTPOverride extracts a "<name>::mtp" / "<name>::nomtp" suffix
// carrying a per-request multi-token-prediction toggle. Recognised
// tokens:
//
//	mtp   → "on"
//	nomtp → "off"
//
// The picker emits this suffix for catalog rows whose `mtp:` block
// declares an MTP variant. It is independent of the reasoning override
// (see parseReasoningOverride) and the context override (see
// parseCtxOverride); the picker's canonical emit order is
// `<name>::<reasoning>::<mtp>@<ctx>`, so the request handler strips
// `@<ctx>` first, then `::<mtp>` from the right, then `::<reasoning>`.
// Each parser is independent -- an unrecognised `::<token>` halts
// peeling at that step, so a name that legitimately contains `::` is
// left untouched.
//
// Empty override means no recognised suffix was present (passthrough).
// Wired into the request handler in Phase 5; placed here for Phase 1
// so the parser is testable in isolation.
func parseMTPOverride(name string) (clean string, override string) {
	const sep = "::"
	idx := strings.LastIndex(name, sep)
	if idx < 0 {
		return name, ""
	}
	token := strings.ToLower(strings.TrimSpace(name[idx+len(sep):]))
	switch token {
	case "mtp":
		return name[:idx], "on"
	case "nomtp":
		return name[:idx], "off"
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

// isSafeModelName rejects names that would let `--model /models/<name>`
// resolve outside the bind-mounted /models directory. Slash itself is
// allowed because HuggingFace repo names like `nvidia/Qwen3-8B-NVFP4`
// are the documented form for vLLM/SGLang. The empty name is treated
// as safe here -- empty triggers a "model name required" error later
// in ensureBackendRunning with a clearer message than "invalid name".
func isSafeModelName(name string) bool {
	if name == "" {
		return true
	}
	if strings.ContainsRune(name, 0) {
		return false
	}
	for _, seg := range strings.Split(name, "/") {
		if seg == ".." {
			return false
		}
	}
	return true
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

// setNumCtx injects/overrides options.num_ctx in an Ollama-native
// request body. `force=true` (picker @suffix override) replaces any
// existing value; `force=false` (registered modelContexts cap as a
// soft fallback) only sets when the client didn't.
//
// IMPORTANT: only call from /api/chat and /api/generate handlers.
// Ollama IGNORES options.num_ctx on its OpenAI-compat
// (/v1/chat/completions) and Anthropic-compat (/v1/messages)
// surfaces -- the load always picks up the global
// OLLAMA_CONTEXT_LENGTH there. Empirically verified 2026-05-13:
// a request with options.num_ctx=4096 to /v1/chat/completions
// loaded with context_length=131072 (= OLLAMA_CONTEXT_LENGTH);
// the same field on /api/chat loaded with context_length=4096.
// Call site at makeRequestHandler gates accordingly.
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
