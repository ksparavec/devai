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
//	IDLE_TIMEOUT      seconds before an idle backend is auto-unloaded; 0 = never (default 0, keep-warm)
//	DRAIN_TIMEOUT     seconds to wait for in-flight requests before stopping (default 30)
//	HEALTH_TIMEOUT_SECONDS  seconds to wait for a backend to become healthy after launch (default 600)
//	MAX_CONCURRENT_REQUESTS max in-flight requests per backend before HTTP 429; 0 = unlimited (default 32)
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
	"encoding/binary"
	"encoding/json"
	"errors"
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
	"path/filepath"
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

// envIntAllowZero is envInt for settings where 0 is a meaningful value
// rather than "unset". envInt maps 0 back to the fallback, which silently
// contradicts the documented `0 = unlimited` semantics of
// MAX_CONCURRENT_REQUESTS. Only a well-formed non-negative integer is
// accepted; anything else falls back.
func envIntAllowZero(key string, fallback int) int {
	s := os.Getenv(key)
	if s == "" {
		return fallback
	}
	v, err := strconv.Atoi(strings.TrimSpace(s))
	if err != nil || v < 0 {
		log.Printf("warning: invalid %s=%q; falling back to %d", key, s, fallback)
		return fallback
	}
	return v
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
	// MountDest is the in-container path for the ModelsDir bind mount.
	// Empty defaults to "/models" (vLLM/SGLang). Ollama uses
	// "/root/.ollama" and needs it read-write (MountRW).
	MountDest string
	MountRW   bool
	// DynamicEnv contributes per-recreate env vars computed from the
	// launch config (e.g. Ollama's OLLAMA_CONTEXT_LENGTH baked from the
	// probed ctx). Merged over EnvVars at buildContainerSpec time.
	DynamicEnv func(lc launchConfig) map[string]string
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
	// MTPProbedUnfit is true when the fit prober's MTP pass recorded
	// mtp_fits=false for this model (the draft lm_head OOMs at load) and
	// never true. The router refuses to emit --speculative-config for such
	// a model even when the catalog declares `mtp:`, serving baseline.
	// False when MTP fit was verified OR never probed (prior behaviour).
	MTPProbedUnfit bool `yaml:"-"`
	// KVByCtx maps each fully-on-GPU probed ctx tier at the host band to
	// the KV-cache dtype the cell was measured under ("" / "f16" = daemon
	// default). A tier probed with kv_cache_type=q8_0 only fits WITH
	// quantized KV, so the router must reproduce that dtype when serving
	// any ctx the tier covers (see resolveKVCacheType). Ollama only.
	KVByCtx map[int]string `yaml:"-"`
	// FlashByCtx is the KVByCtx sibling for OLLAMA_FLASH_ATTENTION: each
	// probed ctx tier -> the flash-attention setting the cell was measured
	// under (nil entry = the cell predates the stamp). Ollama only.
	FlashByCtx map[int]*bool `yaml:"-"`
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
	MemFraction float64
	MaxContext  int
	// MaxNumSeqs bounds the engine's concurrent-sequence batch (vLLM
	// --max-num-seqs / SGLang --max-running-requests). 0 omits the flag
	// (engine default). Set from the router's MAX_CONCURRENT_REQUESTS so
	// CUDA-graph capture only spans batch sizes we actually admit; a
	// per-model recovery flag still overrides it (appended last).
	MaxNumSeqs      int
	ToolParser      string // empty omits backend-specific tool flags
	ReasoningParser string // empty omits --reasoning-parser
	// KVCacheType is the KV dtype the probe measured the serving ctx's
	// covering tier under. Ollama's DynamicEnv bakes it into the
	// recreated container (plus OLLAMA_FLASH_ATTENTION=1, which KV
	// quantization requires); ""/"f16" emits nothing (daemon default).
	KVCacheType string
	// FlashAttention is the OLLAMA_FLASH_ATTENTION setting the probe cell
	// covering the serving ctx was measured under. nil = the cell predates
	// the stamp, and ollamaDynamicEnv derives the value from KVCacheType
	// instead (the behaviour those cells were probed under).
	FlashAttention *bool
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
	// KVCacheType is the KV dtype the daemon served this cell under
	// (stamped by the prober from OLLAMA_KV_CACHE_TYPE; "" on cells
	// probed before the field existed = f16).
	KVCacheType string `json:"kv_cache_type"`
	// FlashAttention is the OLLAMA_FLASH_ATTENTION setting the cell was
	// measured under. Pointer so ABSENT (cells written before the prober
	// stamped it) stays distinguishable from an explicit false: absent
	// falls back to the dtype-derived value in ollamaDynamicEnv (quantized
	// KV implies flash attention), which is exactly what those cells were
	// probed with. Without reading it, a cell probed with flash attention
	// ON under the default f16 dtype was SERVED with it off -- a different
	// environment from the one the fit was measured in.
	FlashAttention *bool `json:"flash_attention"`
}

// tristateBool decodes a probe cache's `disable_verified` field, whose
// on-disk shape is NOT strictly boolean: the probers can record a string
// sentinel (e.g. "error") for a model whose disable-probe never produced a
// verdict. A plain *bool field would make encoding/json abort the WHOLE
// file on the first such value -- and the router unmarshals each cache in
// one call, so one malformed field on one model emptied the entire model
// list for that backend. Accepted forms:
//
//	true / false  -> that value
//	null, absent  -> unknown
//	"true"/"false" (any case) -> that value
//	any other string, or any other JSON type -> unknown, logged once
//
// "unknown" degrades exactly like a missing field: modelDisableOK gets no
// entry for that model, so the reasoning policy declines to send the
// protocol's disable field. Bad data costs one model its disable
// optimisation instead of costing every model its existence.
type tristateBool struct {
	v *bool
}

// Value returns the decoded value, or nil when unknown.
func (t tristateBool) Value() *bool { return t.v }

func (t *tristateBool) UnmarshalJSON(data []byte) error {
	s := strings.TrimSpace(string(data))
	if s == "null" {
		t.v = nil
		return nil
	}
	var b bool
	if err := json.Unmarshal(data, &b); err == nil {
		t.v = &b
		return nil
	}
	var str string
	if err := json.Unmarshal(data, &str); err == nil {
		switch strings.ToLower(strings.TrimSpace(str)) {
		case "true":
			yes := true
			t.v = &yes
			return nil
		case "false":
			no := false
			t.v = &no
			return nil
		}
	}
	log.Printf("warning: probe cache disable_verified=%s is not a boolean; treating this model's disable-verified state as unknown (re-probe to fix)", s)
	t.v = nil
	return nil
}

// cacheEntry mirrors the per-digest record in
// deploy/.ollama-reasoning-cache.json (schema v3).
type cacheEntry struct {
	SchemaVersion   int                              `json:"schema_version"`
	Digest          string                           `json:"digest"`
	Aliases         []string                         `json:"aliases"`
	MaxContext      int                              `json:"max_context"`
	Capability      string                           `json:"capability"`
	DisableVerified tristateBool                     `json:"disable_verified,omitempty"`
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
		kvByCtx := make(map[int]string)
		flashByCtx := make(map[int]*bool)
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
			kvByCtx[c] = probe.KVCacheType
			if probe.FlashAttention != nil {
				flashByCtx[c] = probe.FlashAttention
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
			KVByCtx:      kvByCtx,
			FlashByCtx:   flashByCtx,
			Reasoning: &configReasoning{
				Capability:      capability,
				DisableVerified: entry.DisableVerified.Value(),
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
	// ServingOk is set by the serving-time LOAD probe (probe-vllm --load,
	// scripts/_probe_load.py): true when a near-full-context request ran
	// without OOMing the engine, false when the per-step transient
	// (softcap-logits + attention workspace) overflowed the VRAM the fit
	// probe left unmeasured. Nil when the load probe hasn't run for this
	// cell. The synthesizer treats a fits=true/serving_ok=false cell as
	// NOT serveable at that ctx, so the per-name context cap falls back to
	// the largest tier that both fits AND serves. A nil pointer preserves
	// pre-load-probe behaviour byte-for-byte (fit verdict alone gates).
	ServingOk *bool `json:"serving_ok,omitempty"`
	// MtpFits is set by the fit prober's separate MTP pass (probe-vllm on a
	// catalog row declaring `mtp:`, without --no-mtp): true when the model
	// loaded with --speculative-config, false when the draft lm_head OOMed
	// at model init. Nil when no MTP pass ran for this cell. The router
	// suppresses ::mtp for a model whose cells recorded mtp_fits=false so a
	// speculative launch that would 503 falls back to baseline instead.
	MtpFits *bool `json:"mtp_fits,omitempty"`
	// KVCacheType is the KV dtype the prober launched this cell with
	// (stamped from the pass's PROBE_KV_CACHE_TYPE). "" on legacy cells:
	// for vLLM that factually means fp8 (the prober always passed
	// --kv-cache-dtype fp8 before the field existed); for SGLang it
	// means the engine default (no flag). synthesizeHFFromCache encodes
	// that asymmetry.
	KVCacheType string `json:"kv_cache_type"`
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
	SizeGB          float64      `json:"size_gb,omitempty"`
	MaxContext      int          `json:"max_context"`
	Capability      string       `json:"capability"`
	ToolParser      *string      `json:"tool_parser"`
	ReasoningParser *string      `json:"reasoning_parser,omitempty"`
	DisableVerified tristateBool `json:"disable_verified,omitempty"`
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
		kvByCtx := make(map[int]string)
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
			// Serving-time gate: a cell that loaded (fits=true) but OOMed
			// under a near-full-context request (serving_ok=false) cannot
			// serve at that ctx. Exclude it so bestCtx settles on the
			// largest tier that both fits AND serves. Nil = load probe
			// hasn't run -> fall through on fit alone (legacy behaviour).
			if probe.ServingOk != nil && !*probe.ServingOk {
				continue
			}
			// Per-tier KV dtype. Legacy unstamped vLLM cells were all
			// measured under --kv-cache-dtype fp8 (the prober's
			// historical hardcode), so "" decodes to fp8 there; SGLang
			// legacy cells ran the engine default (no flag) and stay "".
			kv := probe.KVCacheType
			if kv == "" && backendName == "vllm" {
				kv = "fp8"
			}
			kvByCtx[c] = kv
			if c >= bestCtx {
				bestCtx = c
				bestProbe = probe
			}
		}
		if bestCtx == 0 {
			continue
		}
		// MTP fit verdict: the fit prober records mtp_fits per cell in a
		// separate --speculative-config pass. If any cell recorded false and
		// none true, MTP OOMs at load for this model on this card -> flag it
		// so the router won't emit --speculative-config even when the catalog
		// declares `mtp:`. Absent (un-probed) or any true leaves it false
		// (offer MTP; prior behaviour).
		var sawMTPTrue, sawMTPFalse bool
		for _, probe := range band {
			if probe.MtpFits == nil {
				continue
			}
			if *probe.MtpFits {
				sawMTPTrue = true
			} else {
				sawMTPFalse = true
			}
		}
		mtpProbedUnfit := sawMTPFalse && !sawMTPTrue
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
			KVByCtx:         kvByCtx,
			ToolParser:      toolParser,
			ReasoningParser: reasoningParser,
			ToolMode:        toolMode,
			Reasoning: &configReasoning{
				Capability:      capability,
				DisableVerified: entry.DisableVerified.Value(),
			},
			MTP:            mtpBlock,
			MTPProbedUnfit: mtpProbedUnfit,
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
		// Same blast radius as the Ollama cache: the whole file is
		// unmarshalled in one call, so a single malformed field takes
		// every row on this backend with it.
		log.Printf("ERROR: %s probe cache %s failed to parse: %v -- ZERO %s models registered; the router will 404 every %s request and the picker will list none. Fix the JSON or re-run `make probe-%s`, then restart devai-router.",
			backendName, path, jerr, backendName, backendName, backendName)
		return nil
	}
	rows := synthesizeHFFromCache(cache, backendName, hostVRAMGB, operatorMaxCtx, mtpRegistry)
	entryCount := len(cache)
	if _, ok := cache["_meta"]; ok {
		entryCount-- // _meta is a Phase C drift-stamp block, not a model entry
	}
	log.Printf("probe cache: %s loaded (%d entries → %d %s serving rows)",
		path, entryCount, len(rows), backendName)
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
	config     backendConfig
	proxy      *httputil.ReverseProxy
	modelNames []string
	running    bool
	// containerLaunched tracks CONTAINER LIVENESS, which is not the same
	// thing as `running` (= "healthy and serving"). `podman start`
	// succeeded but waitForHealthy then timed out (or detectLaunchFailure
	// reported a crash)? `running` was never set, yet the container is
	// alive and its engine is still pulling weights onto the GPU. Without
	// this flag stopOtherBackends would skip that backend and the next one
	// would launch against a card that is not free. Set right after
	// `podman start` returns, cleared only after a successful stop (or once
	// the container is observed gone).
	containerLaunched bool
	currentModel      string
	currentContext    int // baked --max-model-len / --context-length for vLLM/SGLang; 0 for Ollama
	// currentSpec is the speculative-decoding configuration baked into
	// the running container, or nil when MTP is off. A toggle (nil <->
	// non-nil, or any field-level change) triggers a recreate the same
	// way currentModel and currentContext changes do. Reset to nil
	// whenever the container is observed gone.
	currentSpec *configSpeculative
	lastRequest time.Time
	activeReqs  int64
	// upstreamReqs counts only requests that are PAST the arbiter mutex
	// and actually proxied to the backend. drainBackend must wait on this
	// and not on activeReqs: activeReqs also counts requests parked on
	// a.mu, and drainBackend itself runs with a.mu held -- so those can
	// never drop off and every switch under load would stall for the full
	// DRAIN_TIMEOUT. Incremented under a.mu (so a concurrent
	// stopOtherBackends cannot observe zero for a request already cleared
	// to proxy) and decremented after ServeHTTP returns.
	upstreamReqs int64
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
	// imageStale is set once at startup (Phase C) when the backend's running
	// image digest differs from the digest its probe cache was captured
	// against (_meta.current_image_digest). The fit/serving/parser data was
	// measured on a different image, so it may be unreliable. The router
	// still serves -- a genuine crash is failed hard by Phase A's
	// crash-detection -- but flags every response with X-DevAI-Warning and
	// surfaces the drift in /health. Immutable after startup. Ollama never
	// goes stale: the drift check looks the backend up in the
	// backend-name -> probe-cache-path map, which has entries only for
	// vllm and sglang, so readProbedImageDigest is called with "" and
	// returns no baseline to compare against.
	imageStale         bool
	probedImageDigest  string
	runningImageDigest string
}

type arbiter struct {
	backends      map[string]*backendState
	mu            sync.Mutex
	ollamaURL     *url.URL
	podmanClient  *http.Client
	idleTimeout   time.Duration
	drainTimeout  time.Duration
	healthTimeout time.Duration // configurable per HEALTH_TIMEOUT_SECONDS env (default 600s — vLLM/SGLang cold-start with NVFP4 weights + CUDA graph compilation can take minutes; waitForHealthy's crash-detection bails early on a dead engine, so this only bounds a genuinely-hung or slow-but-healthy load)
	// maxConcurrent bounds in-flight requests per backend; over it the
	// request handler returns HTTP 429. 0 = unlimited. Also drives the
	// engines' --max-num-seqs / --max-running-requests so CUDA-graph
	// capture covers exactly the batch sizes we admit.
	maxConcurrent int64
	// Size and declared-context are keyed by (backend, model name) for the
	// same reason every other lookup below is: a model probed on BOTH vLLM
	// and SGLang has an independent row per backend, and a name-only map
	// silently applied whichever row happened to be registered last (e.g.
	// SGLang's context cap used to size a vLLM launch).
	modelSizes    map[string]map[string]float64 // backend → model name → weight size in GB
	modelContexts map[string]map[string]int     // backend → model name → declared max context
	// Capability and disable-verified are backend-specific for the same
	// reason the parser maps below are: a model that runs on both vLLM and
	// SGLang can carry different reasoning classifications per engine.
	// openai/gpt-oss-20b is the canonical case — its Harmony format cannot
	// disable reasoning under vLLM (reasoning_effort="none" is rejected, so
	// the prober records disable_verified=false), but SGLang disables it via
	// separate_reasoning=false (disable_verified=true). A name-only map let
	// SGLang's `true` leak into the vLLM path and inject the invalid
	// reasoning_effort="none". Key by (backend, modelName) so each engine
	// honours its own probe verdict.
	modelCapability map[string]map[string]string // backend → model name → reasoning.capability
	modelDisableOK  map[string]map[string]bool   // backend → model name → disable_verified (only when present)
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
	// modelKVByCtx maps each fully-on-GPU probed ctx tier to the KV-cache
	// dtype it was measured under (""/"f16" = daemon default). A tier that
	// only fits with quantized KV (e.g. q8_0 at 128K on a model whose f16
	// KV spills there) must be served with that same dtype — the launch
	// path resolves the covering tier via resolveKVCacheType and bakes
	// OLLAMA_KV_CACHE_TYPE into the recreated container. Ollama only.
	modelKVByCtx map[string]map[string]map[int]string // backend → model → ctx tier → kv dtype
	// modelFlashByCtx is the modelKVByCtx sibling for flash attention:
	// the OLLAMA_FLASH_ATTENTION setting each probed tier was measured
	// under. A missing tier entry means the probe cell predates the stamp,
	// and ollamaDynamicEnv then derives the value from the KV dtype.
	modelFlashByCtx map[string]map[string]map[int]*bool // backend → model → ctx tier → flash attention
	// modelMTP holds the catalog-declared multi-token-prediction launch
	// params per (backend, model). Populated at startup from
	// configModel.MTP (which the catalog metadata side-table in
	// loadCatalogMTP wired in). A non-nil entry advertises MTP
	// *availability*; whether it actually gets emitted at launch is
	// gated by the per-request `::mtp` suffix (parseMTPOverride). nil
	// = catalog declares no MTP for this row -- the suffix is ignored.
	modelMTP map[string]map[string]*configSpeculative
	// modelMTPUnfit[backend][model] = true when the fit probe recorded
	// mtp_fits=false (the qwen3_5_mtp draft lm_head OOMs at load). The
	// ::mtp gate consults this and serves baseline rather than 503-looping
	// a speculative launch that cannot fit.
	modelMTPUnfit map[string]map[string]bool
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
	// weightWarned tracks which backends have already logged the
	// "models dir not visible" degradation in checkModelWeights, so an
	// unmounted store costs one log line rather than one per request.
	// Guarded by its own mutex: checkModelWeights runs before a.mu is
	// taken and must not reach for it.
	weightWarnMu sync.Mutex
	weightWarned map[string]bool
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

func newSmartProxy(target *url.URL, imageStale bool) *httputil.ReverseProxy {
	p := httputil.NewSingleHostReverseProxy(target)
	p.FlushInterval = -1
	p.Transport = noKeepAliveTransport()
	p.ModifyResponse = func(resp *http.Response) error {
		if imageStale {
			// Phase C: advisory drift warning. Non-blocking -- the response
			// is served as-is; the header just tells clients the probe data
			// behind this backend was captured on a different image.
			resp.Header.Set("X-DevAI-Warning",
				"backend image drifted from probe baseline; fit/serving data may "+
					"be unreliable -- re-run make probe-vllm / probe-sglang")
		}
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

// resolveKVCacheType returns the KV-cache dtype for serving `ctx`, from the
// per-tier map the probe cache recorded (kvByCtx: probed ctx tier → dtype
// the cell was measured under). The covering tier is the SMALLEST probed
// tier >= ctx: a fit at tier T validates every ctx <= T only under T's own
// KV dtype, and picking the smallest cover keeps low-ctx sessions on the
// highest-fidelity dtype that provably fits (e.g. 64K stays f16 even when
// a q8_0-only 128K tier exists above it). Returns the raw stamp — each
// backend's launch builder decides what its engine default means
// (ollamaDynamicEnv treats ""/"f16" as flagless; vllmEntrypoint treats
// "" as the legacy fp8). "" when no tier covers ctx or the map is empty.
func resolveKVCacheType(kvByCtx map[int]string, ctx int) string {
	best := 0
	kv := ""
	for tier, dtype := range kvByCtx {
		if tier >= ctx && (best == 0 || tier < best) {
			best = tier
			kv = dtype
		}
	}
	return kv
}

// resolveFlashAttention is resolveKVCacheType's sibling for the probed
// OLLAMA_FLASH_ATTENTION stamp: same smallest-covering-tier rule, so the
// dtype and the flash-attention setting always come from the SAME probe
// cell and serve time reproduces one coherent measured environment.
// Returns nil when no covering tier carries the stamp (pre-stamp cells),
// which tells ollamaDynamicEnv to fall back to the dtype-derived value.
func resolveFlashAttention(flashByCtx map[int]*bool, ctx int) *bool {
	best := 0
	var flash *bool
	for tier, on := range flashByCtx {
		if on == nil {
			continue
		}
		if tier >= ctx && (best == 0 || tier < best) {
			best = tier
			flash = on
		}
	}
	return flash
}

// ollamaLaunchCtx resolves the context an ollama container is launched at:
// the probed ceiling by default, clamped down by a per-request `@<ctx>`
// override (desired <= 0 = no override). The override is how mixed-KV
// models pin their full-quality f16 tier instead of the largest
// (quantized) one; it never raises the ceiling.
func ollamaLaunchCtx(probed, desired int) int {
	if desired > 0 && desired < probed {
		return desired
	}
	return probed
}

// ollamaDynamicEnv is the per-recreate env for the ollama backend: the
// probed context ceiling always, plus the KV dtype and the flash-attention
// setting the serving ctx's covering probe tier was measured under.
//
// Flash attention is a hard prerequisite for quantized KV in ollama, so a
// quantized dtype implies it -- but the converse does not hold: a cell can
// be probed with OLLAMA_FLASH_ATTENTION=1 under the DEFAULT f16 dtype, and
// deriving the setting from the dtype alone served that model without
// flash attention, i.e. in a different environment from the one its fit
// was measured in. lc.FlashAttention carries the probe's own stamp and
// wins whenever it is present; nil (pre-stamp cells) keeps the historical
// dtype-derived value.
func ollamaDynamicEnv(lc launchConfig) map[string]string {
	env := map[string]string{
		"OLLAMA_CONTEXT_LENGTH": fmt.Sprintf("%d", lc.MaxContext),
	}
	// ""/"f16" = daemon default: emit nothing so the container spec
	// stays byte-identical to pre-KV-field launches.
	quantizedKV := lc.KVCacheType != "" && lc.KVCacheType != "f16"
	if quantizedKV {
		env["OLLAMA_KV_CACHE_TYPE"] = lc.KVCacheType
	}
	flash := quantizedKV
	if lc.FlashAttention != nil {
		flash = *lc.FlashAttention
	}
	if flash {
		env["OLLAMA_FLASH_ATTENTION"] = "1"
	}
	return env
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

// ollamaEntrypoint runs `ollama serve`. Context is not a CLI flag for
// Ollama -- it is baked via the OLLAMA_CONTEXT_LENGTH env (see the
// backendConfig DynamicEnv), so the entrypoint is model-independent.
func ollamaEntrypoint(modelName string, lc launchConfig) []string {
	return []string{"/bin/ollama", "serve"}
}

func vllmEntrypoint(modelName string, lc launchConfig) []string {
	// Per-model KV dtype from the probe cache (resolveKVCacheType over
	// the covering tier's stamp). "" = legacy unstamped cells, which
	// were all measured under fp8 (the historical hardcode) — serving
	// must reproduce the measured dtype or fit data is invalid. A model
	// re-probed with PROBE_KV_CACHE_TYPE=auto serves unquantized KV.
	// Must match vllm_command_args in scripts/probe-vllm-reasoning.py.
	kvDtype := lc.KVCacheType
	if kvDtype == "" {
		kvDtype = "fp8"
	}
	args := []string{
		"python3", "-m", "vllm.entrypoints.openai.api_server",
		"--model", "/models/" + modelName,
		"--host", "0.0.0.0",
		"--port", "11434",
		"--tensor-parallel-size", "1",
		"--max-model-len", fmt.Sprintf("%d", lc.MaxContext),
		"--kv-cache-dtype", kvDtype,
		"--gpu-memory-utilization", fmt.Sprintf("%.2f", lc.MemFraction),
		"--enable-prefix-caching",
		"--trust-remote-code",
		"--served-model-name", modelName,
	}
	// Cap concurrent sequences at the router's admission limit so CUDA-
	// graph capture only spans batch sizes we serve. Before the parser /
	// recovery flags so a per-model --max-num-seqs still wins last.
	if lc.MaxNumSeqs > 0 {
		args = append(args, "--max-num-seqs", fmt.Sprintf("%d", lc.MaxNumSeqs))
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
		// SGLang v0.5.10 enables piecewise CUDA graph by default, which
		// torch.compiles the model forward. Dynamo then cannot trace
		// flashinfer's FP4 JIT-compile path (modelopt_quant.py:1482 ->
		// fp4_quantize -> a subprocess/threading.Lock call), so every NVFP4
		// load crashes with a Dynamo graph break during warmup_compile.
		// Disabling piecewise capture runs the FP4 quantize eagerly (it JIT-
		// compiles fine outside a compile context) -- the engine's own
		// documented workaround. Drop this when a future SGLang image can
		// trace the FP4 path. Pinned in deploy/backend-flags.yaml.
		"--disable-piecewise-cuda-graph",
	}
	// Per-model KV dtype from the probe cache. SGLang's legacy cells ran
	// the engine default (no flag), decoded as "" — emit nothing so those
	// launches stay byte-identical. A cell probed under an enforced dtype
	// (PROBE_KV_CACHE_TYPE=fp8_e5m2/...) carries the stamp and serving
	// reproduces it. Must match sglang_command_args in
	// scripts/probe-sglang-reasoning.py.
	if lc.KVCacheType != "" && lc.KVCacheType != "f16" && lc.KVCacheType != "auto" {
		args = append(args, "--kv-cache-dtype", lc.KVCacheType)
	}
	// SGLang's --max-running-requests is the analogue of vLLM's
	// --max-num-seqs (verified against v0.5.10.post1-cu130). Before the
	// recovery flags so a per-model override still wins last.
	if lc.MaxNumSeqs > 0 {
		args = append(args, "--max-running-requests", fmt.Sprintf("%d", lc.MaxNumSeqs))
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
	// MTP launch flags. NVFP4 loading itself is unblocked (see the
	// --disable-piecewise-cuda-graph note above), but the SGLang MTP /
	// --speculative-* path is not yet validated on this fleet, so the
	// prober never records a spec block for SGLang and this branch rarely
	// fires. Emitting the flags keeps the entrypoint forward-compatible.
	args = append(args, sglangSpeculativeArgs(lc.Speculative)...)
	// Per-model recovery flags (mirrors vllmEntrypoint). Appended last so a
	// per-model override wins over the base flags.
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

	runSingleHost(buildArbiter())
}

// buildArbiter constructs the single-host arbiter: probe caches,
// per-backend config, the lookup maps, and the image-drift check.
// Split out of main() (behaviour-identical) so cluster worker mode can
// build the very same scheduler and dispatch head-forwarded requests
// through it -- see makeInboundHandler in cluster_main.go.
func buildArbiter() *arbiter {
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
			// This is not a warning: cfg.Models stays EMPTY, so every
			// Ollama model disappears from the router AND from the picker
			// until the file parses again. Say so explicitly.
			log.Printf("ERROR: probe cache %s failed to parse: %v -- ZERO Ollama models registered; the router will 404 every Ollama request and the picker will list none. Fix the JSON or re-run `make probe`, then restart devai-router.",
				cachePath, jerr)
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
			Image:         env("OLLAMA_IMAGE", "docker.io/ollama/ollama:latest"),
			ModelsDir:     env("OLLAMA_DATA_DIR", "/var/cache/devai/ollama"),
			MountDest:     "/root/.ollama",
			MountRW:       true,
			Network:       network,
			HealthPath:    "/",
			Entrypoint:    ollamaEntrypoint,
			EnvVars: map[string]string{
				"OLLAMA_KEEP_ALIVE":        env("OLLAMA_KEEP_ALIVE", "300s"),
				"OLLAMA_MAX_LOADED_MODELS": "1",
				"OLLAMA_GPU_OVERHEAD":      env("OLLAMA_GPU_OVERHEAD", "0"),
			},
			// Bake the probed context into the container env. Unlike
			// options.num_ctx (which Ollama ignores on /v1), OLLAMA_CONTEXT_LENGTH
			// is honored on every request surface -- so recreating per model
			// with the probed ctx keeps /v1 clients from silently loading at
			// 256K and spilling to CPU.
			DynamicEnv: ollamaDynamicEnv,
		},
		{
			Name:          "vllm",
			ListenPort:    envInt("VLLM_PORT", 11435),
			BackendURL:    vllmURL,
			ContainerName: env("VLLM_CONTAINER", "devai-vllm"),
			Image:         env("VLLM_IMAGE", "docker.io/vllm/vllm-openai:v0.22.1-x86_64-cu129-ubuntu2404"),
			ModelsDir:     env("VLLM_MODELS_DIR", "/var/cache/devai/vllm"),
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
			ModelsDir:     env("SGLANG_MODELS_DIR", "/var/cache/devai/sglang"),
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
	modelSizes := make(map[string]map[string]float64)          // backend → model → weight size GB
	modelContexts := make(map[string]map[string]int)           // backend → model → declared max ctx
	modelCapability := make(map[string]map[string]string)      // backend → model → reasoning.capability
	modelDisableOK := make(map[string]map[string]bool)         // backend → model → disable_verified
	modelToolParser := make(map[string]map[string]string)      // backend → model → --tool-call-parser
	modelReasoningParser := make(map[string]map[string]string) // backend → model → --reasoning-parser
	modelToolMode := make(map[string]map[string]string)        // backend → model → "auto" | "forced"
	modelProbedMaxCtx := make(map[string]map[string]int)       // backend → model → highest fits=true ctx
	modelKVByCtx := make(map[string]map[string]map[int]string) // backend → model → ctx tier → kv dtype
	modelFlashByCtx := make(map[string]map[string]map[int]*bool)
	modelMTP := make(map[string]map[string]*configSpeculative) // backend → model → catalog MTP block
	modelMTPUnfit := make(map[string]map[string]bool)          // backend → model → probe recorded mtp_fits=false
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
			// Size, declared context, capability, disable-verified, parser
			// maps, and the probe-verified ctx ceiling are all keyed by
			// backend so the same model name can carry different values on
			// vLLM vs SGLang without one backend overwriting the other.
			for _, backend := range m.Backend {
				if sz > 0 {
					if modelSizes[backend] == nil {
						modelSizes[backend] = make(map[string]float64)
					}
					modelSizes[backend][name] = sz
				}
				if m.Context > 0 {
					if modelContexts[backend] == nil {
						modelContexts[backend] = make(map[string]int)
					}
					modelContexts[backend][name] = m.Context
				}
				if modelCapability[backend] == nil {
					modelCapability[backend] = make(map[string]string)
				}
				modelCapability[backend][name] = capability
				if disableOK {
					if modelDisableOK[backend] == nil {
						modelDisableOK[backend] = make(map[string]bool)
					}
					modelDisableOK[backend][name] = true
				}
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
				if len(m.KVByCtx) > 0 {
					if modelKVByCtx[backend] == nil {
						modelKVByCtx[backend] = make(map[string]map[int]string)
					}
					modelKVByCtx[backend][name] = m.KVByCtx
				}
				if len(m.FlashByCtx) > 0 {
					if modelFlashByCtx[backend] == nil {
						modelFlashByCtx[backend] = make(map[string]map[int]*bool)
					}
					modelFlashByCtx[backend][name] = m.FlashByCtx
				}
				if m.MTP != nil {
					if modelMTP[backend] == nil {
						modelMTP[backend] = make(map[string]*configSpeculative)
					}
					modelMTP[backend][name] = m.MTP
					if m.MTPProbedUnfit {
						if modelMTPUnfit[backend] == nil {
							modelMTPUnfit[backend] = make(map[string]bool)
						}
						modelMTPUnfit[backend][name] = true
					}
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
		idleTimeout:          time.Duration(envInt("IDLE_TIMEOUT", 0)) * time.Second,
		drainTimeout:         time.Duration(envInt("DRAIN_TIMEOUT", 30)) * time.Second,
		healthTimeout:        time.Duration(envInt("HEALTH_TIMEOUT_SECONDS", 600)) * time.Second,
		maxConcurrent:        int64(envIntAllowZero("MAX_CONCURRENT_REQUESTS", 32)),
		modelSizes:           modelSizes,
		modelContexts:        modelContexts,
		modelCapability:      modelCapability,
		modelDisableOK:       modelDisableOK,
		modelToolParser:      modelToolParser,
		modelReasoningParser: modelReasoningParser,
		modelToolMode:        modelToolMode,
		modelProbedMaxCtx:    modelProbedMaxCtx,
		modelKVByCtx:         modelKVByCtx,
		modelFlashByCtx:      modelFlashByCtx,
		modelMTP:             modelMTP,
		modelMTPUnfit:        modelMTPUnfit,
		defaultPolicy:        policy,
		totalVRAMGB:          totalVRAMGB,
		maxContextLen:        maxCtx,
		pluginRegistry:       pluginRegistry,
		recoveryRegistry:     recoveryRegistry,
		healthClient:         &http.Client{Timeout: 2 * time.Second},
	}

	// Image-drift detection (Phase C): compare each HF backend's running
	// image digest against the digest its probe cache was captured with. A
	// moved tag silently invalidates fit/serving/parser data; we serve
	// anyway but warn (loud log here + X-DevAI-Warning per response + a
	// /health flag). Ollama (empty Image, no _meta stamp) never goes stale.
	backendCachePath := map[string]string{
		"vllm":   vllmCachePath,
		"sglang": sglangCachePath,
	}
	for _, bc := range backends {
		probed := readProbedImageDigest(backendCachePath[bc.Name])
		running := a.imageDigestFromLibpod(bc.Image)
		stale := probed != "" && running != "" && probed != running
		if stale {
			log.Printf("WARNING: %s image drift -- probe cache captured on %s but running image %s is %s; serving with X-DevAI-Warning. Re-run `make probe-%s`.",
				bc.Name, probed, bc.Image, running, bc.Name)
		}
		var proxy *httputil.ReverseProxy
		if bc.Name == "ollama" {
			proxy = newProxy(bc.BackendURL)
		} else {
			proxy = newSmartProxy(bc.BackendURL, stale)
		}
		a.backends[bc.Name] = &backendState{
			config:             bc,
			proxy:              proxy,
			modelNames:         modelsForBackend(cfg.Models, bc.Name),
			recreateCond:       sync.NewCond(&a.mu),
			imageStale:         stale,
			probedImageDigest:  probed,
			runningImageDigest: running,
		}
	}

	return a
}

// runSingleHost starts the single-host serving path: idle watcher,
// signal handler, one listener per backend. Blocks forever.
func runSingleHost(a *arbiter) {
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

	idleDesc := fmt.Sprintf("%ds", int(a.idleTimeout.Seconds()))
	if a.idleTimeout == 0 {
		idleDesc = "never(keep-warm)"
	}
	log.Printf("gpu-arbiter started: idle=%s drain=%ds health=%ds max_concurrent=%d",
		idleDesc, int(a.drainTimeout.Seconds()), int(a.healthTimeout.Seconds()), a.maxConcurrent)

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
	status, _, ok := a.containerState(name)
	return ok && status == "running"
}

// containerState returns the container's libpod State.Status (e.g.
// "running", "exited", "created") and ExitCode. ok=false when the podman
// API is unreachable or the body can't be decoded -- callers treat that as
// "unknown", not "crashed", so a podman blip never triggers a spurious
// recreate or a false fail-fast.
func (a *arbiter) containerState(name string) (status string, exitCode int, ok bool) {
	if a.podmanClient == nil {
		return "", 0, false
	}
	reqURL := fmt.Sprintf("http://d/v4.0.0/libpod/containers/%s/json", name)
	resp, err := a.podmanClient.Get(reqURL)
	if err != nil {
		return "", 0, false
	}
	defer resp.Body.Close()
	var info struct {
		State struct {
			Status   string `json:"Status"`
			ExitCode int    `json:"ExitCode"`
		} `json:"State"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		log.Printf("warning: containerState %s: decode failed: %v", name, err)
		return "", 0, false
	}
	return info.State.Status, info.State.ExitCode, true
}

// imageDigestFromLibpod returns the manifest digest of a local image via the
// libpod image-inspect API, matching what scripts/_probe_core.image_digest_via_cli
// records at probe time (`podman image inspect --format {{.Digest}}`, falling
// back to the first RepoDigests entry). Returns "" on any error -- image-drift
// detection fails open, so a missing image or unreachable podman never
// produces a false "stale" verdict. The libpod route is `/images/{name:.*}/json`,
// whose greedy matcher accepts an unescaped registry/repo:tag ref.
func (a *arbiter) imageDigestFromLibpod(imageRef string) string {
	if a.podmanClient == nil || imageRef == "" {
		return ""
	}
	reqURL := fmt.Sprintf("http://d/v4.0.0/libpod/images/%s/json", imageRef)
	resp, err := a.podmanClient.Get(reqURL)
	if err != nil {
		return ""
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return ""
	}
	var info struct {
		Digest      string   `json:"Digest"`
		RepoDigests []string `json:"RepoDigests"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return ""
	}
	return normalizeImageDigest(info.Digest, info.RepoDigests)
}

// normalizeImageDigest reduces a libpod image-inspect result to a bare
// `sha256:...` manifest digest, byte-matching what the prober records at stamp
// time (scripts/_probe_core.image_digest_via_cli): prefer the top-level
// `.Digest`, else fall back to the FIRST `.RepoDigests` entry's `@sha256:...`
// tail. The index-0 choice is deliberate -- the Python stamper uses
// `{{index .RepoDigests 0}}`, so both sides must select the same entry for the
// drift comparison to be exact (an any-match here would diverge and could flag
// a spurious drift). Returns "" when neither field carries a sha256.
func normalizeImageDigest(digest string, repoDigests []string) string {
	if strings.Contains(digest, "sha256:") {
		return digest
	}
	if len(repoDigests) > 0 && strings.Contains(repoDigests[0], "sha256:") {
		rd := repoDigests[0]
		if i := strings.LastIndex(rd, "@"); i >= 0 {
			return rd[i+1:]
		}
		return rd
	}
	return ""
}

// readProbedImageDigest extracts _meta.current_image_digest from an HF probe
// cache (stamped by scripts/_probe_core.stamp_image_digest). Returns "" when
// the file is absent, unparseable, or predates Phase C (no _meta) -- callers
// treat "" as "no baseline to compare", so drift detection is skipped rather
// than firing a false warning.
func readProbedImageDigest(path string) string {
	if path == "" {
		return ""
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	var meta struct {
		Meta struct {
			CurrentImageDigest string `json:"current_image_digest"`
		} `json:"_meta"`
	}
	if err := json.Unmarshal(data, &meta); err != nil {
		return ""
	}
	return meta.Meta.CurrentImageDigest
}

// terminalLaunchErrors are log substrings that unambiguously mean the
// engine has crashed and will never answer /health. Conservative on
// purpose: these are checked even while the container still reports
// "running" (some crashes hang before exiting), so anything that could
// appear in a healthy-but-slow startup is excluded. Sourced from the probe
// failure classifier (scripts/_probe_hf_common.py).
var terminalLaunchErrors = []string{
	"Engine core initialization failed",
	"torch.cuda.OutOfMemoryError",
	"CUDA out of memory",
	"No available memory for the cache blocks",
	"quantization is not supported",
	"Model architectures", // "Model architectures ['X'] are not supported" -- specific anchor; the bare "are not supported" tail can hit benign startup warnings (see scripts/_probe_hf_common.py:722-732)
	"is not a supported model type",
}

// failureAnchors mark the root-cause line of a crash. Trusted only once the
// container has already EXITED (where any error line is by definition
// fatal), to pull a useful message out of a long traceback instead of the
// generic wrapper. Mirrors _FAILURE_ANCHORS in the probe classifier.
var failureAnchors = []string{
	"NotImplementedError", "ValueError", "RuntimeError", "AssertionError",
	"KeyError", "ImportError", "OSError", "Error:",
}

// containerRecentLogs fetches the last tailLines of a container's combined
// stdout+stderr via the libpod logs endpoint (follow=false, returns
// immediately). The stream is Docker-multiplexed for non-TTY containers;
// demuxDockerStream strips the frame headers. Returns "" on any error.
func (a *arbiter) containerRecentLogs(name string, tailLines int) string {
	if a.podmanClient == nil {
		return ""
	}
	reqURL := fmt.Sprintf(
		"http://d/v4.0.0/libpod/containers/%s/logs?stdout=true&stderr=true&follow=false&tail=%d",
		name, tailLines)
	resp, err := a.podmanClient.Get(reqURL)
	if err != nil {
		return ""
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 256<<10))
	if err != nil {
		return ""
	}
	return demuxDockerStream(raw)
}

// demuxDockerStream strips Docker/libpod multiplexed stream frame headers (1
// byte stream type, 3 zero bytes, uint32 big-endian payload size) and
// returns the concatenated payloads. Non-framed (raw TTY) data is returned
// unchanged.
func demuxDockerStream(raw []byte) string {
	framed := func(p []byte) bool {
		return len(p) >= 8 && p[0] <= 2 && p[1] == 0 && p[2] == 0 && p[3] == 0
	}
	if !framed(raw) {
		return string(raw)
	}
	var b strings.Builder
	for framed(raw) {
		size := int(binary.BigEndian.Uint32(raw[4:8]))
		raw = raw[8:]
		if size > len(raw) {
			size = len(raw)
		}
		b.Write(raw[:size])
		raw = raw[size:]
	}
	b.Write(raw) // trailing non-framed remainder, if any
	return b.String()
}

// lastErrorLine returns the last (trimmed) log line containing any of the
// substrings, or "". Last-match wins so a traceback yields its final
// exception line (the root cause) rather than the "Traceback" header.
func lastErrorLine(logs string, sigs []string) string {
	if logs == "" {
		return ""
	}
	lines := strings.Split(logs, "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		for _, sig := range sigs {
			if strings.Contains(lines[i], sig) {
				return strings.TrimSpace(lines[i])
			}
		}
	}
	return ""
}

// launchFailure marks a backend that failed to come up. crashed=true means
// the engine died (or logged a terminal error) -- a hard, permanent condition
// for the current image, surfaced to clients as a non-retryable 400 so they
// stop re-requesting a doomed reload. crashed=false is a plain health-wait
// timeout (a genuinely-slow or hung load) which stays a retryable 503.
type launchFailure struct {
	crashed bool
	msg     string
}

func (e *launchFailure) Error() string { return e.msg }

// detectLaunchFailure returns a non-nil *launchFailure the moment the backend
// container has demonstrably failed to start -- it exited/died, or its logs
// already carry a terminal error signature -- so waitForHealthy can bail in
// seconds instead of polling to the full HEALTH_TIMEOUT. Returns nil while
// the container is still legitimately starting.
func (a *arbiter) detectLaunchFailure(containerName string) error {
	status, exitCode, ok := a.containerState(containerName)
	crashed := ok && (status == "exited" || status == "dead" || status == "stopped")
	logs := a.containerRecentLogs(containerName, 200)

	if crashed {
		if line := lastErrorLine(logs, failureAnchors); line != "" {
			return &launchFailure{crashed: true, msg: fmt.Sprintf("engine crashed (exit %d): %s", exitCode, line)}
		}
		return &launchFailure{crashed: true, msg: fmt.Sprintf("engine container %s (exit %d) before becoming healthy", status, exitCode)}
	}
	// Still running (or state unknown): only a strong, unambiguous fatal
	// signature aborts the wait -- a healthy slow load never prints these.
	if line := lastErrorLine(logs, terminalLaunchErrors); line != "" {
		return &launchFailure{crashed: true, msg: fmt.Sprintf("engine reported a fatal error: %s", line)}
	}
	return nil
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
	mountDest := cfg.MountDest
	if mountDest == "" {
		mountDest = "/models"
	}
	mountOpts := []string{"ro"}
	if cfg.MountRW {
		mountOpts = []string{"rw"}
	}
	mounts := []map[string]any{{
		"destination": mountDest,
		"source":      cfg.ModelsDir,
		"type":        "bind",
		"options":     mountOpts,
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
		"devices":      []map[string]any{{"path": env("DEVAI_GPU_DEVICE", "nvidia.com/gpu=all")}},
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
	// Per-recreate dynamic env (e.g. Ollama's OLLAMA_CONTEXT_LENGTH baked
	// from the probed ctx). Applied last so it can override statics --
	// but never a recovery-env key: recovery entries exist precisely to
	// let an operator override defaults for borderline checkpoints, and
	// that must hold for probe-derived keys (OLLAMA_KV_CACHE_TYPE /
	// OLLAMA_FLASH_ATTENTION) the same way it does for CLI flags
	// (last-flag-wins).
	if cfg.DynamicEnv != nil {
		for k, v := range cfg.DynamicEnv(lc) {
			if _, ok := recoveryEnv[k]; ok {
				continue
			}
			envMap[k] = v
		}
	}
	if len(envMap) > 0 {
		spec["env"] = envMap
	}
	return spec
}

// requestedContext resolves the context a launch ASKS for, before the
// memory heuristic and the probe ceiling clamp it: the per-request
// `@<ctx>` override when present, otherwise the registered per-backend
// declared cap, otherwise MAX_CONTEXT_LEN. An override is itself capped
// at MAX_CONTEXT_LEN.
func (a *arbiter) requestedContext(backendName, modelName string, desiredCtx int) int {
	declaredCtx := a.modelContexts[backendName][modelName]
	if declaredCtx == 0 {
		declaredCtx = a.maxContextLen
	}
	if desiredCtx <= 0 {
		return declaredCtx
	}
	if a.maxContextLen > 0 && desiredCtx > a.maxContextLen {
		return a.maxContextLen
	}
	return desiredCtx
}

// resolveLaunchContext returns the context a launch of `modelName` on
// `backendName` would actually settle on for the given per-request
// `desiredCtx` (0 = no override). Pure function of the (immutable after
// startup) arbiter lookup maps, and it applies exactly the same clamps as
// containerRecreate does.
//
// ensureBackendRunning compares this against bs.currentContext -- which now
// holds a LAUNCHED context, not a requested one. Comparing a requested ctx
// against a launched one is what made a bare `<name>` and a pinned
// `<name>@<ctx>` recreate each other on every alternating request.
func (a *arbiter) resolveLaunchContext(backendName, modelName string, desiredCtx int) int {
	requestedCtx := a.requestedContext(backendName, modelName, desiredCtx)
	lc := computeLaunchConfig(
		a.modelSizes[backendName][modelName], a.totalVRAMGB, backendName, requestedCtx)
	return applyProbeCeiling(
		lc.MaxContext, requestedCtx, a.modelProbedMaxCtx[backendName][modelName])
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
//
// Returns the context length actually baked into the launch (lc.MaxContext,
// i.e. AFTER the MAX_CONTEXT_LEN clamp, the memory heuristic, and the probe
// ceiling), so callers record what was launched rather than what was asked
// for. Zero on error.
func (a *arbiter) containerRecreate(bs *backendState, modelName string, desiredCtx int, desiredSpec *configSpeculative) (int, error) {
	cfg := bs.config
	a.containerStop(cfg.ContainerName)
	a.containerRemove(cfg.ContainerName)

	modelSizeGB := a.modelSizes[cfg.Name][modelName]
	requestedCtx := a.requestedContext(cfg.Name, modelName, desiredCtx)
	lc := computeLaunchConfig(modelSizeGB, a.totalVRAMGB, cfg.Name, requestedCtx)
	lc.MaxContext = applyProbeCeiling(
		lc.MaxContext, requestedCtx,
		a.modelProbedMaxCtx[cfg.Name][modelName],
	)
	// Reproduce the KV-cache dtype the probe measured the serving ctx
	// under (q8_0-only tiers OOM at f16 and vice-versa would waste
	// quality; see resolveKVCacheType).
	lc.KVCacheType = resolveKVCacheType(
		a.modelKVByCtx[cfg.Name][modelName], lc.MaxContext,
	)
	// Same covering tier, so the dtype and the flash-attention setting
	// always describe one probe cell (see resolveFlashAttention).
	lc.FlashAttention = resolveFlashAttention(
		a.modelFlashByCtx[cfg.Name][modelName], lc.MaxContext,
	)
	// Bound the engine's concurrent-sequence batch to the router's
	// admission cap so CUDA-graph capture only covers batch sizes we
	// actually serve. Emitted before RecoveryFlags, so a per-model
	// --max-num-seqs (VRAM rescue) still wins last.
	lc.MaxNumSeqs = int(a.maxConcurrent)
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
	if rec, ok := a.recoveryRegistry.Lookup(cfg.Name, modelName); ok {
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
		return 0, perr
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
		return 0, fmt.Errorf("marshal container spec for %s: %w", cfg.ContainerName, err)
	}
	resp, err := a.podmanClient.Post(
		"http://d/v4.0.0/libpod/containers/create",
		"application/json",
		bytes.NewReader(body),
	)
	if err != nil {
		return 0, fmt.Errorf("podman create %s: %w", cfg.ContainerName, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		respBody, _ := io.ReadAll(resp.Body)
		return 0, fmt.Errorf("podman create %s: %s %s", cfg.ContainerName, resp.Status, respBody)
	}

	startURL := fmt.Sprintf("http://d/v4.0.0/libpod/containers/%s/start", cfg.ContainerName)
	resp2, err := a.podmanClient.Post(startURL, "", nil)
	if err != nil {
		return 0, fmt.Errorf("podman start %s: %w", cfg.ContainerName, err)
	}
	defer resp2.Body.Close()
	if resp2.StatusCode >= 300 && resp2.StatusCode != http.StatusNotModified {
		respBody, _ := io.ReadAll(resp2.Body)
		return 0, fmt.Errorf("podman start %s: %s %s", cfg.ContainerName, resp2.Status, respBody)
	}
	return lc.MaxContext, nil
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
		// Fail fast: a crashed engine will never answer /health, so don't
		// burn the full timeout waiting for a corpse. Only the vLLM/SGLang
		// backends are crash-checked: detectLaunchFailure keys off their
		// container-exit / engine-log signatures. Ollama now also reaches
		// waitForHealthy (recreate-per-model), but its misfit signal is the
		// warm-load 500 in warmLoadOllama, not an engine crash log -- so it
		// stays exempt from detectLaunchFailure here.
		if bs.config.Name != "ollama" {
			if failErr := a.detectLaunchFailure(bs.config.ContainerName); failErr != nil {
				return fmt.Errorf("%s %w", bs.config.Name, failErr)
			}
		}
		time.Sleep(2 * time.Second)
	}
	return &launchFailure{crashed: false, msg: fmt.Sprintf("%s did not become ready within %s", bs.config.Name, timeout)}
}

// writeLaunchError translates a backend-launch failure into an HTTP response.
// A crashed engine (launchFailure.crashed) is a hard, permanent condition for
// the current image, so it returns 400 invalid_request_error with
// x-should-retry:false -- OpenAI/litellm clients treat that as non-retryable
// and stop hammering the router with a doomed reload. A plain health-wait
// timeout (or any other error) stays 503, which clients may legitimately retry.
func (a *arbiter) writeLaunchError(w http.ResponseWriter, err error) {
	log.Printf("error: %v", err)
	status := http.StatusServiceUnavailable
	errType := "server_error"
	var lf *launchFailure
	if errors.As(err, &lf) && lf.crashed {
		status = http.StatusBadRequest
		errType = "invalid_request_error"
		w.Header().Set("x-should-retry", "false")
	}
	body, _ := json.Marshal(map[string]any{
		"error": map[string]any{"type": errType, "message": err.Error()},
	})
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	w.Write(body)
}

// --- GPU exclusion and lifecycle ---

// unloadOllamaTimeout bounds each HTTP call unloadOllama makes to the
// Ollama daemon. unloadOllama runs from stopOtherBackends with the
// arbiter mutex HELD, so an unbounded call against a daemon that accepts
// TCP but never answers would wedge every listener on all three ports
// with no log line at all. 60s is far above a real unload (which only
// has to free VRAM) while still guaranteeing the lock is released.
const unloadOllamaTimeout = 60 * time.Second

func (a *arbiter) unloadOllama() {
	client := &http.Client{Timeout: unloadOllamaTimeout}
	resp, err := client.Get(a.ollamaURL.String() + "/api/ps")
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
		// Neither the transport error nor a non-2xx status may be
		// discarded: both mean the model was probably NOT evicted, and
		// the caller is about to hand the GPU to another backend. Log
		// loudly so the operator sees a cause instead of an unexplained
		// OOM on the next launch.
		unloadResp, err := client.Post(
			a.ollamaURL.String()+"/api/generate", "application/json", bytes.NewReader(b))
		if err != nil {
			log.Printf("error: unloadOllama: keep_alive=0 for %s failed: %v -- GPU may still be held by ollama", m.Name, err)
			continue
		}
		io.Copy(io.Discard, unloadResp.Body)
		unloadResp.Body.Close()
		if unloadResp.StatusCode >= 300 {
			log.Printf("error: unloadOllama: keep_alive=0 for %s returned %s -- GPU may still be held by ollama", m.Name, unloadResp.Status)
		}
	}
	time.Sleep(2 * time.Second)
}

// checkModelWeights fails fast when a directory-backed backend is asked
// for a model whose weights are not on disk.
//
// Ollama is exempt: its store is a blob/manifest tree, not one directory
// per catalog name, so a path test there would be wrong.
//
// The check is deliberately skipped when ModelsDir itself is not visible
// from inside the router container. The store is bind-mounted read-only
// by deploy/docker-compose.yaml, but an older compose (or a bare `go run`
// on a dev box) will not have it, and degrading to the previous
// behaviour is far better than refusing every model. That degradation is
// logged once per backend so it is diagnosable rather than silent.
func (a *arbiter) checkModelWeights(cfg backendConfig, modelName string) error {
	if cfg.Name == "ollama" || cfg.ModelsDir == "" || modelName == "" {
		return nil
	}
	if _, err := os.Stat(cfg.ModelsDir); err != nil {
		a.weightCheckOnce(cfg.Name)
		return nil
	}
	// Defense-in-depth, and the local barrier a path-injection scanner
	// needs to see: confirm modelName cannot escape ModelsDir before it is
	// joined into a filesystem path. isSafeModelName already rejects `..`
	// segments upstream, but filepath.IsLocal makes the containment
	// guarantee local to this function -- it rejects absolute paths and any
	// upward traversal while still allowing the HF repo form
	// `nvidia/Qwen3-8B-NVFP4` (a relative, non-escaping path).
	if !filepath.IsLocal(modelName) {
		return fmt.Errorf(
			"%s model %q is not a valid on-disk name", cfg.Name, modelName)
	}
	// path (not path/filepath): these are POSIX container paths, and
	// modelName may legitimately carry a `/` (the HF repo form). The name
	// is already allowlisted and `..`-checked by isSafeModelName upstream.
	dir := path.Join(cfg.ModelsDir, modelName)
	if _, err := os.Stat(dir); err == nil {
		return nil
	}
	return fmt.Errorf(
		"%s model %q has no weights on disk at %s -- the probe cache "+
			"advertises it but the store was never populated; run "+
			"`make model-pull NAME=%s` (see docs/backends.md)",
		cfg.Name, modelName, dir, modelName,
	)
}

// weightCheckOnce logs the "store not mounted" degradation a single time
// per backend, so an unmounted store is visible in the log without
// spamming a line per request.
func (a *arbiter) weightCheckOnce(backend string) {
	a.weightWarnMu.Lock()
	defer a.weightWarnMu.Unlock()
	if a.weightWarned == nil {
		a.weightWarned = map[string]bool{}
	}
	if a.weightWarned[backend] {
		return
	}
	a.weightWarned[backend] = true
	log.Printf("warning: %s models dir not visible to the router; "+
		"skipping the weights-on-disk check (mount it read-only to enable)", backend)
}

// drainBackend waits for requests already proxied upstream to finish.
// It deliberately watches upstreamReqs, not activeReqs: this runs with
// a.mu held, so any request still parked on that mutex cannot make
// progress while we wait and would keep the count permanently non-zero.
func (a *arbiter) drainBackend(bs *backendState) {
	if atomic.LoadInt64(&bs.upstreamReqs) == 0 {
		return
	}
	log.Printf("draining %s (%d requests in flight upstream)...", bs.config.Name, atomic.LoadInt64(&bs.upstreamReqs))
	deadline := time.Now().Add(a.drainTimeout)
	for atomic.LoadInt64(&bs.upstreamReqs) > 0 && time.Now().Before(deadline) {
		time.Sleep(500 * time.Millisecond)
	}
	remaining := atomic.LoadInt64(&bs.upstreamReqs)
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
		// `running` alone is not enough: a launch whose health wait timed
		// out (or whose engine crashed) left the container alive and still
		// holding the GPU while `running` stayed false. containerLaunched
		// covers exactly that window -- see backendState.containerLaunched.
		if !bs.running && !bs.containerLaunched {
			continue
		}
		log.Printf("stopping %s (switching to %s)", name, targetName)
		a.drainBackend(bs)
		stopped := true
		if name == "ollama" {
			a.unloadOllama()
		} else {
			if err := a.containerStop(bs.config.ContainerName); err != nil {
				log.Printf("warning: failed to stop %s: %v", name, err)
				stopped = false
			}
		}
		bs.running = false
		bs.currentModel = ""
		bs.currentContext = 0
		if stopped {
			bs.containerLaunched = false
		}
	}
}

// ensureBackendRunning makes sure the target backend is up with the given
// model and context. Called with the arbiter mutex held.
//
// `desiredCtx` is the per-request context cap resolved upstream (picker
// "@<int>" override or registered modelContexts cap). For vLLM/SGLang the
// context is baked into the entrypoint at startup so a context change
// requires a full recreate even when the model is unchanged. For Ollama,
// ensureOllamaRunning launches at min(desiredCtx, probed ceiling) -- a
// `@<ctx>` override below the ceiling pins a smaller tier, which is how
// mixed-KV models select their full-quality f16 tier instead of the
// largest (quantized) one (see resolveKVCacheType).
//
// `ctxPinned` says whether the request name carried an explicit `@<ctx>`
// suffix. Only Ollama's mixed-KV tier policy consults it -- see
// ensureOllamaRunning.
func (a *arbiter) ensureBackendRunning(bs *backendState, modelName string, desiredCtx int, ctxPinned bool, desiredSpec *configSpeculative) error {
	if bs.config.Name == "ollama" {
		return a.ensureOllamaRunning(bs, modelName, desiredCtx, ctxPinned)
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
		bs.containerLaunched = false
		bs.currentModel = ""
		bs.currentContext = 0
		bs.currentSpec = nil
	}

	// Recreate when the model, the baked context cap, OR the
	// speculative-decoding config changed. The third trigger is what
	// makes the `::mtp` / `::nomtp` per-request suffix work -- toggling
	// MTP requires re-launching the backend container with (or without)
	// --speculative-config, which only takes effect at startup.
	//
	// The context test compares LAUNCHED against LAUNCHED: bs.currentContext
	// holds the ctx the running container was actually built with, so the
	// candidate must be run through the same clamps before comparing.
	modelChanged := modelName != "" && bs.currentModel != modelName
	resolvedCtx := 0
	if modelName != "" {
		resolvedCtx = a.resolveLaunchContext(bs.config.Name, modelName, desiredCtx)
	}
	contextChanged := resolvedCtx > 0 && bs.currentContext > 0 && bs.currentContext != resolvedCtx
	specChanged := !specEqual(bs.currentSpec, desiredSpec)
	needRecreate := !bs.running || modelChanged || contextChanged || specChanged
	if !needRecreate {
		return nil
	}

	// Bail BEFORE touching the GPU. A vLLM/SGLang container can never be
	// launched without a model, so releasing the card first buys nothing --
	// and stopOtherBackends would drain Ollama and evict its resident model
	// (keep_alive=0) only for this call to error out and serve 503. A bare
	// `GET /` health/monitoring probe on an idle backend port lands here
	// (the mux catch-all never runs the POST-body block, so modelName is
	// ""), so that eviction was reachable from any liveness checker.
	if modelName == "" {
		return fmt.Errorf("model name required for %s", bs.config.Name)
	}

	// Same reasoning: fail before the GPU is touched. A model the probe
	// cache advertises but whose weights are not on disk cannot launch --
	// the engine would burn a full HEALTH_TIMEOUT_SECONDS cold start and
	// then die with an opaque "repo not found". This is the live
	// SGLANG_MODELS_DIR gap (nothing populates that store), so the check
	// pays for itself there, but it is backend-agnostic.
	if err := a.checkModelWeights(bs.config, modelName); err != nil {
		return err
	}

	a.stopOtherBackends(bs.config.Name)

	if bs.currentModel != "" && bs.currentModel != modelName {
		log.Printf("switching %s model: %s → %s", bs.config.Name, bs.currentModel, modelName)
	} else if contextChanged {
		log.Printf("switching %s context (model %s): %d → %d",
			bs.config.Name, modelName, bs.currentContext, resolvedCtx)
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

	launchedCtx, err := a.containerRecreate(bs, modelName, desiredCtx, desiredSpec)
	if err != nil {
		return fmt.Errorf("failed to start %s: %w", bs.config.Name, err)
	}
	// `podman start` succeeded: the container is alive and consuming the
	// GPU from here on, whatever the health wait below decides. Recorded
	// before that wait so a health timeout still leaves stopOtherBackends
	// able to reclaim the card.
	bs.containerLaunched = true

	// Release lock during health wait so concurrent /v1/* requests can
	// queue / waiters can park on the cond. The recreating flag prevents
	// them from kicking off duplicate recreates. The inner closure +
	// `defer a.mu.Lock()` guarantees we re-acquire BEFORE returning from
	// this scope, even on a panic out of waitForHealthy. Without that,
	// the outer recreate defer above would fire without the lock held
	// and race on bs.recreating / bs.pendingModel / bs.pendingContext.
	err = func() error {
		a.mu.Unlock()
		defer a.mu.Lock()
		return a.waitForHealthy(bs, a.healthTimeout)
	}()

	if err != nil {
		return err
	}

	bs.running = true
	bs.currentModel = modelName
	// Record the context the launch config actually settled on (after the
	// MAX_CONTEXT_LEN clamp, the memory heuristic in computeLaunchConfig,
	// and the probe ceiling) -- NOT the requested one. The contextChanged
	// test above resolves its candidate through the same clamps, so the
	// two are directly comparable and a bare `<name>` no longer recreates
	// a container that a pinned `<name>@<ctx>` just launched at the very
	// same effective context.
	if launchedCtx > 0 {
		bs.currentContext = launchedCtx
	}
	// Spec is recorded unconditionally (even when nil) so the next
	// request can compute specChanged correctly: a request that omits
	// `::mtp` after a previous `::mtp`-enabled launch is a recreate
	// trigger, not a no-op.
	bs.currentSpec = desiredSpec
	return nil
}

// ensureOllamaRunning brings the Ollama backend up with `modelName` loaded
// at min(desiredCtx, probed ceiling), recreating the container when the
// model or context changes -- mirroring the vLLM/SGLang recreate-per-model
// lifecycle. The context is baked into OLLAMA_CONTEXT_LENGTH at launch
// (honored on /v1, unlike options.num_ctx), and a warm-load with num_gpu
// forced to full GPU makes a misfit fail hard instead of silently spilling
// to CPU. desiredCtx <= 0 means "no override" (launch at the ceiling).
// Called with the arbiter mutex held.
//
// Mixed-KV tier policy (`ctxPinned`):
//
//   - `<name>@<ctx>` PINS that tier. A different loaded tier is recreated
//     onto the pinned one -- an explicit pin is always honoured exactly,
//     which the bench harness relies on to label its rows.
//   - a BARE `<name>` while some tier is already loaded is served FROM THE
//     LOADED TIER, with no recreate. Without this, a pinning client and a
//     bare client alternating on the same model recreated the container on
//     every single request.
//   - a BARE `<name>` with nothing loaded picks the default tier (the
//     probed ceiling, clamped by the registered cap).
func (a *arbiter) ensureOllamaRunning(bs *backendState, modelName string, desiredCtx int, ctxPinned bool) error {
	// Model-less surfaces (/api/tags, /api/ps, /v1/models): just make sure
	// the GPU is ours; there is nothing to load or recreate.
	if modelName == "" {
		if !bs.running {
			a.stopOtherBackends("ollama")
			bs.running = true
		}
		return nil
	}

	// Resolve the probed context: the single fits=true ctx the Ollama probe
	// verified fully-on-GPU. Fall back to the catalog cap, then MAX_CONTEXT_LEN.
	// This is the hard ceiling -- Ollama is never loaded above it. A
	// per-request `@<ctx>` override below the ceiling pins a smaller tier
	// (mixed-KV models use this to stay on their f16 tier).
	probedCtx := a.modelProbedMaxCtx["ollama"][modelName]
	if probedCtx <= 0 {
		probedCtx = a.modelContexts["ollama"][modelName]
	}
	if probedCtx <= 0 {
		probedCtx = a.maxContextLen
	}
	probedCtx = ollamaLaunchCtx(probedCtx, desiredCtx)

	// Coalesce with any in-flight recreate on this backend.
	for bs.recreating {
		bs.recreateCond.Wait()
	}

	// Drop stale state if the container vanished or a placeholder replaced it.
	if bs.running && a.podmanClient != nil &&
		(!a.containerIsRunning(bs.config.ContainerName) || !a.backendIsServing(bs)) {
		log.Printf("ollama not serving (container gone or placeholder up), resetting state")
		bs.running = false
		bs.containerLaunched = false
		bs.currentModel = ""
		bs.currentContext = 0
	}

	// resolvedCtx is what a launch at probedCtx would ACTUALLY settle on --
	// same clamp chain containerRecreate runs (requestedContext ->
	// computeLaunchConfig -> applyProbeCeiling). bs.currentContext holds a
	// LAUNCHED context, so comparing it against the requested probedCtx
	// would be the same requested-vs-launched mismatch the vLLM/SGLang path
	// fixed with resolveLaunchContext: whenever the two diverge, ctxChanged
	// stays true forever and every pinned request recreates the container.
	resolvedCtx := a.resolveLaunchContext("ollama", modelName, probedCtx)

	modelChanged := bs.currentModel != modelName
	// Only an explicit `@<ctx>` pin may force a tier switch; a bare name is
	// served from whatever tier is resident (see the policy note above).
	ctxChanged := ctxPinned && bs.currentContext != resolvedCtx
	if bs.running && !modelChanged && !ctxChanged {
		return nil
	}

	a.stopOtherBackends("ollama")
	if bs.currentModel != "" && modelChanged {
		log.Printf("switching ollama model: %s → %s", bs.currentModel, modelName)
	}
	log.Printf("starting ollama with model %s (ctx=%d)...", modelName, probedCtx)

	bs.recreating = true
	bs.pendingModel = modelName
	bs.pendingContext = probedCtx
	defer func() {
		bs.recreating = false
		bs.pendingModel = ""
		bs.pendingContext = 0
		bs.recreateCond.Broadcast()
	}()

	// launchedCtx is what actually went into OLLAMA_CONTEXT_LENGTH (it can
	// be below probedCtx when the memory heuristic clamps). Warm-load and
	// the recorded currentContext both use it so the container env, the
	// warm-load num_ctx, and the router's idea of the serving tier are one
	// number.
	launchedCtx, err := a.containerRecreate(bs, modelName, probedCtx, nil)
	if err != nil {
		return fmt.Errorf("failed to start ollama: %w", err)
	}
	// `podman start` succeeded -- the container is alive from here on, even
	// if the health wait or the warm-load below fails. See
	// backendState.containerLaunched.
	bs.containerLaunched = true

	// Release the lock for the network-bound health wait + warm-load so
	// concurrent requests park on recreateCond instead of the mutex.
	if err := func() error {
		a.mu.Unlock()
		defer a.mu.Lock()
		if err := a.waitForHealthy(bs, a.healthTimeout); err != nil {
			return err
		}
		return a.warmLoadOllama(modelName, launchedCtx)
	}(); err != nil {
		return err
	}

	bs.running = true
	bs.currentModel = modelName
	bs.currentContext = launchedCtx
	return nil
}

// warmLoadOllama preloads `modelName` at `numCtx` with num_gpu forced high so
// Ollama either loads the model 100% on the GPU or errors. This is the "no
// silent CPU spill" enforcement: at the probed ctx the model fits fully, so
// this succeeds and leaves the model GPU-resident (keep_alive holds it so the
// following real request reuses the same runner); if VRAM regressed or the
// probe is stale it fails hard here -- surfaced as a non-retryable 400 by
// writeLaunchError -- instead of serving at single-digit tok/s off system RAM.
func (a *arbiter) warmLoadOllama(modelName string, numCtx int) error {
	body, _ := json.Marshal(map[string]any{
		"model":    modelName,
		"messages": []map[string]string{{"role": "user", "content": "ok"}},
		"stream":   false,
		// Integer -1 = keep resident indefinitely. Must be a JSON number:
		// Ollama parses a *string* keep_alive as a Go duration, so "-1"
		// fails with `missing unit in duration "-1"`.
		"keep_alive": -1,
		"options": map[string]any{
			"num_ctx":     numCtx,
			"num_gpu":     999,
			"num_predict": 1,
		},
	})
	client := &http.Client{Timeout: a.healthTimeout}
	resp, err := client.Post(a.ollamaURL.String()+"/api/chat", "application/json", bytes.NewReader(body))
	if err != nil {
		return &launchFailure{crashed: true, msg: fmt.Sprintf("ollama warm-load %s failed: %v", modelName, err)}
	}
	defer resp.Body.Close()
	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		excerpt := string(respBody)
		if len(excerpt) > 300 {
			excerpt = excerpt[:300]
		}
		return &launchFailure{crashed: true, msg: fmt.Sprintf(
			"ollama model %s does not fit fully on GPU at ctx=%d (num_gpu forced full): %s",
			modelName, numCtx, excerpt)}
	}
	return nil
}

// --- HTTP handlers ---

func (a *arbiter) makeRequestHandler(backendName string) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		bs := a.backends[backendName]
		// Admission control: bound in-flight requests per backend. Increment
		// first (single source of truth, race-free) and always defer the
		// decrement, so an over-cap request still accounts correctly on the
		// way out. maxConcurrent==0 disables the cap.
		inflight := atomic.AddInt64(&bs.activeReqs, 1)
		defer atomic.AddInt64(&bs.activeReqs, -1)
		if a.maxConcurrent > 0 && inflight > a.maxConcurrent {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusTooManyRequests)
			fmt.Fprintf(w, `{"error":{"type":"rate_limit_error","code":"max_concurrent_requests","message":"router at capacity: more than MAX_CONCURRENT_REQUESTS=%d requests in flight for backend %q; retry shortly"}}`, a.maxConcurrent, backendName)
			return
		}

		// Read body for any POST so we can (a) extract the model name
		// for backend lifecycle decisions and (b) apply the reasoning
		// policy via the backend's native protocol field.
		var modelName string
		var numCtx int
		// ctxPinned records whether the request name carried an explicit
		// `@<ctx>` suffix. A pin is honoured exactly (it may force a tier
		// switch); a bare name never does -- see ensureOllamaRunning.
		var ctxPinned bool
		// body / ollamaNativeCtx outlive the POST-body block: the
		// Ollama-native options.num_ctx injection is deferred until AFTER
		// the lifecycle decision so it can use the context the container
		// is actually serving at.
		var body []byte
		var ollamaNativeCtx bool
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
			var err error
			body, err = io.ReadAll(req.Body)
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

			// Override priority for num_ctx:
			//   1. Picker-supplied @<int>  → force-injected (user choice).
			//   2. Registered modelContexts cap (= min(model_max,
			//      MAX_CONTEXT_LEN) from the probe cache) → soft cap,
			//      only set when the client didn't supply num_ctx.
			//   3. None → request passes through unchanged.
			// Peel the picker/control-surface suffixes off the model name
			// in whatever order the client appended them. Canonical emit
			// order is `<name>::<reasoning>::<mtp>@<ctx>`, but aiagent/
			// litellm appends its own `::<reasoning>` AFTER the picker's
			// `@<ctx>`, so a strict ctx-last strip would leave `@<ctx>` in
			// the name and the vLLM/SGLang allowlist would reject it. See
			// peelControlSuffixes.
			var cleanName string
			var ctxOverride int
			var reasoningOverride string
			cleanName, ctxOverride, mtpOverride, reasoningOverride = peelControlSuffixes(parsed.Model)
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
			ctxPinned = ctxOverride > 0
			if numCtx == 0 {
				if ctxCap, ok := a.modelContexts[backendName][cleanName]; ok && ctxCap > 0 {
					numCtx = ctxCap
				}
			}
			// Ollama is launched per-model at its probed context (baked into
			// OLLAMA_CONTEXT_LENGTH and warm-loaded 100% on GPU). Cap the
			// launch request at that ceiling so an /api/chat request can't
			// force a larger-context reload that spills to CPU.
			if backendName == "ollama" {
				if pc := a.modelProbedMaxCtx["ollama"][cleanName]; pc > 0 && (numCtx == 0 || numCtx > pc) {
					numCtx = pc
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
			// only on /api/chat and /api/generate. The injection itself is
			// deferred until after ensureBackendRunning -- see below.
			ollamaNativeCtx = backendName == "ollama" &&
				(req.URL.Path == "/api/chat" || req.URL.Path == "/api/generate")
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
				a.modelCapability[backendName][policyModel] == CapInline {
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
			// Suppress MTP when the fit probe recorded mtp_fits=false: the
			// draft lm_head OOMs at model load, so --speculative-config would
			// 503 the request. Serve baseline (no speculative decoding).
			if desiredSpec != nil {
				if unfit, ok := a.modelMTPUnfit[backendName]; ok && unfit[modelName] {
					log.Printf("warning: MTP requested for %s/%s but the fit probe recorded mtp_fits=false (draft head OOMs); serving baseline", backendName, modelName)
					desiredSpec = nil
				}
			}
		}

		a.mu.Lock()
		bs.lastRequest = time.Now()
		if err := a.ensureBackendRunning(bs, modelName, numCtx, ctxPinned, desiredSpec); err != nil {
			a.mu.Unlock()
			a.writeLaunchError(w, err)
			return
		}
		servedCtx := bs.currentContext
		// Count as in-flight-upstream while still holding a.mu, so a
		// stopOtherBackends that acquires the mutex right after we drop it
		// sees this request in drainBackend instead of pulling the backend
		// out from under it.
		atomic.AddInt64(&bs.upstreamReqs, 1)
		a.mu.Unlock()
		defer atomic.AddInt64(&bs.upstreamReqs, -1)

		// Ollama-native options.num_ctx, injected only now that the
		// lifecycle decision is made: servedCtx is the context the running
		// container was actually launched at, which for a BARE `<name>` may
		// be a smaller pinned tier that a previous `<name>@<ctx>` request
		// selected. Injecting the probed ceiling instead would make Ollama
		// reload the runner at a context the pinned tier deliberately
		// avoided (mixed-KV models) and spill to CPU.
		if ollamaNativeCtx && servedCtx > 0 {
			// force=true replaces any client-supplied value; that is right
			// for an explicit `@<ctx>` pin, and also for a client value
			// ABOVE the serving ceiling (which the probe never proved
			// fits). A client value at or below the ceiling is left alone.
			force := ctxPinned
			if cn, ok := clientNumCtx(body); ok && cn > servedCtx {
				force = true
			}
			body = setNumCtx(body, servedCtx, force)
			req.Body = io.NopCloser(bytes.NewReader(body))
			req.ContentLength = int64(len(body))
			req.Header.Set("Content-Length", strconv.Itoa(len(body)))
		}

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
	switch a.reasoningAction("vllm", modelName, policy) {
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
	switch a.reasoningAction("sglang", modelName, policy) {
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

func (a *arbiter) reasoningAction(backend, modelName, policy string) reasoningAction {
	switch a.modelCapability[backend][modelName] {
	case CapStructured:
		switch policy {
		case "auto", "low", "medium", "high":
			return reasoningEnable
		case "off":
			// Disable only when the prober verified the model honours
			// `enable_thinking=false` / equivalent on THIS backend. Without
			// that confirmation the disable injection is a footgun — e.g.
			// gpt-oss under vLLM 400s on the reasoning_effort="none" shape,
			// so its vLLM disable_verified is false even though SGLang's is
			// true. Backend-keying keeps the two verdicts separate.
			if a.modelDisableOK[backend][modelName] {
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
	switch a.reasoningAction("ollama", modelName, policy) {
	case reasoningEnable:
		return setJSONFieldIfAbsent(body, []string{"think"}, "think", true)
	case reasoningDisable:
		return setJSONFieldIfAbsent(body, []string{"think"}, "think", false)
	default:
		return body
	}
}

func (a *arbiter) applyOllamaOpenAIChatPolicy(modelName, policy string, body []byte) []byte {
	switch a.reasoningAction("ollama", modelName, policy) {
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
	switch a.reasoningAction("ollama", modelName, policy) {
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

// peelControlSuffixes strips every recognised picker/control-surface suffix
// from a model name, in whatever order the client appended them. The picker's
// canonical emit order is `<name>::<reasoning>::<mtp>@<ctx>` (ctx last), and the
// original handler stripped strictly in that order. But not every client
// respects it: aiagent/litellm carries its own `default_reasoning` and appends
// `::<reasoning>` AFTER the `@<ctx>` the picker already baked into the tag,
// producing `<name>@<ctx>::<reasoning>`. Under a strict ctx-last strip,
// parseCtxOverride's Atoi("<ctx>::<reasoning>") fails, so `@<ctx>` stays in the
// name and the vLLM/SGLang allowlist rejects it as an unknown model.
//
// Instead, peel whichever of the three recognised suffixes is currently
// trailing and loop until none remain, so the order the client used doesn't
// matter. Each sub-parser strips only a token it recognises (integer ctx / mtp
// keyword / reasoning keyword) and otherwise returns its input unchanged, so a
// name that legitimately contains `::` or `@` survives untouched. Every peel
// shortens the name, so the loop always terminates. When the same suffix class
// appears more than once (malformed input), the innermost -- last-peeled --
// value wins, matching the strict parser's single-strip behaviour on the
// canonical order.
func peelControlSuffixes(name string) (clean string, ctx int, mtp, reasoning string) {
	clean = name
	for {
		if s, ov := parseCtxOverride(clean); ov > 0 {
			clean, ctx = s, ov
			continue
		}
		if s, ov := parseMTPOverride(clean); ov != "" {
			clean, mtp = s, ov
			continue
		}
		if s, ov := parseReasoningOverride(clean); ov != "" {
			clean, reasoning = s, ov
			continue
		}
		break
	}
	return clean, ctx, mtp, reasoning
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

// maxClientNumCtx bounds the value clientNumCtx reports. A float literal
// larger than this (1e30) has no int representation, and Go's float->int
// conversion is implementation-defined out of range; the caller only ever
// compares the value against the serving ceiling and clamps down, so
// saturating here is behaviourally identical and well-defined.
const maxClientNumCtx = 1 << 31

// clientNumCtx reads the client-supplied options.num_ctx out of an
// Ollama-native request body. Returns (0, false) when the body is not a
// JSON object, has no options block, or the value is not a positive
// number. The caller uses it to decide whether a client value needs
// clamping down to the context the backend is actually serving at.
//
// The value is decoded as json.Number rather than *int so EVERY JSON
// numeric shape is recognised: `131072`, `131072.0` and `1.31072e5` all
// report 131072. A *int field errors on the two float forms, which made
// clientNumCtx return (0,false) and let an oversized context sail past the
// clamp at the makeRequestHandler call site. Fractional values truncate
// toward zero (0.5 -> 0 -> not a valid override).
func clientNumCtx(body []byte) (int, bool) {
	var doc struct {
		Options struct {
			NumCtx json.Number `json:"num_ctx"`
		} `json:"options"`
	}
	// A non-numeric num_ctx (string, object, bool) fails this decode and
	// is reported as absent -- the router then leaves the body alone
	// rather than guessing, and Ollama itself rejects the request.
	if json.Unmarshal(body, &doc) != nil {
		return 0, false
	}
	if doc.Options.NumCtx.String() == "" {
		return 0, false
	}
	if n, err := doc.Options.NumCtx.Int64(); err == nil {
		if n <= 0 {
			return 0, false
		}
		if n > maxClientNumCtx {
			return maxClientNumCtx, true
		}
		return int(n), true
	}
	f, err := doc.Options.NumCtx.Float64()
	if err != nil || f < 1 {
		return 0, false
	}
	if f >= maxClientNumCtx {
		return maxClientNumCtx, true
	}
	return int(f), true
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
			// imageStale/probed/running digests are set once in main() before
			// any listener goroutine starts, so this lock-free read is safe.
			// If these ever become runtime-mutable, move them under a.mu above.
			"image_stale":          bs.imageStale,
			"probed_image_digest":  bs.probedImageDigest,
			"running_image_digest": bs.runningImageDigest,
		})
	}
}

// --- Idle watcher ---

func (a *arbiter) idleWatcher() {
	for {
		time.Sleep(30 * time.Second)
		a.idleSweepOnce()
	}
}

// idleSweepOnce stops any backend idle longer than idleTimeout. A zero
// idleTimeout means keep-warm (never auto-unload): the model stays resident
// until a different model is requested. Extracted from idleWatcher so the
// policy is unit-testable without the 30s loop.
func (a *arbiter) idleSweepOnce() {
	if a.idleTimeout == 0 {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	for _, bs := range a.backends {
		// Mirror stopOtherBackends' guard: `running` alone misses a
		// backend whose health wait timed out AFTER `podman start`
		// succeeded -- the container is alive and holding the GPU while
		// running stayed false. That is precisely the state
		// containerLaunched exists to describe, so the sweeper must be
		// able to reclaim it too. lastRequest still gates: a backend that
		// has never served has no idle clock to compare against.
		if (!bs.running && !bs.containerLaunched) || bs.lastRequest.IsZero() {
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
		stopped := true
		if bs.config.Name == "ollama" {
			a.unloadOllama()
		} else {
			if err := a.containerStop(bs.config.ContainerName); err != nil {
				log.Printf("warning: failed to stop %s: %v", bs.config.Name, err)
				stopped = false
			}
		}
		bs.running = false
		if stopped {
			bs.containerLaunched = false
		}
		bs.currentModel = ""
	}
}
