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
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"gopkg.in/yaml.v3"
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
	Name      string           `yaml:"name"`
	Aliases   []string         `yaml:"aliases,omitempty"`
	Digest    string           `yaml:"digest,omitempty"`
	Backend   []string         `yaml:"backend"`
	Repo      string           `yaml:"repo"`
	Size      string           `yaml:"size"`
	Context   int              `yaml:"context"`
	Purpose   string           `yaml:"purpose"`
	Reasoning *configReasoning `yaml:"reasoning,omitempty"`
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
	MemFraction float64
	MaxContext  int
}

type configFile struct {
	Defaults map[string]string `yaml:"defaults"`
	Models   []configModel     `yaml:"models"`
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
	config       backendConfig
	proxy        *httputil.ReverseProxy
	modelNames   []string
	running      bool
	currentModel string
	lastRequest  time.Time
	activeReqs   int64
}

type arbiter struct {
	backends        map[string]*backendState
	mu              sync.Mutex
	ollamaURL       *url.URL
	podmanClient    *http.Client
	idleTimeout     time.Duration
	drainTimeout    time.Duration
	modelSizes      map[string]float64 // model name → weight size in GB
	modelContexts   map[string]int     // model name → declared max context (from models.yaml)
	modelCapability map[string]string  // model name → reasoning.capability
	modelDisableOK  map[string]bool    // model name → disable_verified (only when present)
	defaultPolicy   string             // DEVAI_REASONING env value: auto|off|low|medium|high
	totalVRAMGB     float64
	maxContextLen   int // global default from MAX_CONTEXT_LEN env (default 131072)
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
	return []string{
		"python3", "-m", "vllm.entrypoints.openai.api_server",
		"--model", "/models/" + modelName,
		"--host", "0.0.0.0",
		"--port", "11434",
		"--tensor-parallel-size", "1",
		"--max-model-len", fmt.Sprintf("%d", lc.MaxContext),
		"--gpu-memory-utilization", fmt.Sprintf("%.2f", lc.MemFraction),
		"--enable-prefix-caching",
		"--enable-auto-tool-choice",
		"--tool-call-parser", "qwen3_coder",
		"--trust-remote-code",
		"--served-model-name", modelName,
	}
}

func sglangEntrypoint(modelName string, lc launchConfig) []string {
	return []string{
		"python3", "-m", "sglang.launch_server",
		"--model-path", "/models/" + modelName,
		"--host", "0.0.0.0",
		"--port", "11434",
		"--tp", "1",
		"--mem-fraction-static", fmt.Sprintf("%.2f", lc.MemFraction),
		"--context-length", fmt.Sprintf("%d", lc.MaxContext),
		"--trust-remote-code",
	}
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

	// Load model catalog
	var cfg configFile
	configPath := env("CONFIG_FILE", "/etc/devai/models.yaml")
	if data, err := os.ReadFile(configPath); err == nil {
		yaml.Unmarshal(data, &cfg)
	}

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
			Image:         env("SGLANG_IMAGE", "docker.io/lmsysorg/sglang:latest"),
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
		backends:        make(map[string]*backendState),
		ollamaURL:       ollamaURL,
		podmanClient:    podmanClient,
		idleTimeout:     time.Duration(envInt("IDLE_TIMEOUT", 300)) * time.Second,
		drainTimeout:    time.Duration(envInt("DRAIN_TIMEOUT", 30)) * time.Second,
		modelSizes:      modelSizes,
		modelContexts:   modelContexts,
		modelCapability: modelCapability,
		modelDisableOK:  modelDisableOK,
		defaultPolicy:   policy,
		totalVRAMGB:     totalVRAMGB,
		maxContextLen:   maxCtx,
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

func (a *arbiter) containerRecreate(bs *backendState, modelName string) error {
	cfg := bs.config
	a.containerStop(cfg.ContainerName)
	a.containerRemove(cfg.ContainerName)

	modelSizeGB := a.modelSizes[modelName]
	declaredCtx := a.modelContexts[modelName]
	if declaredCtx == 0 {
		declaredCtx = a.maxContextLen
	}
	lc := computeLaunchConfig(modelSizeGB, a.totalVRAMGB, cfg.Name, declaredCtx)
	log.Printf("  %s launch: model=%.1f GB, gpu=%.1f GB → fraction=%.2f, context=%dk",
		cfg.Name, modelSizeGB, a.totalVRAMGB, lc.MemFraction, lc.MaxContext/1024)

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
	}
}

// ensureBackendRunning makes sure the target backend is up with the given model.
// Called with the arbiter mutex held.
func (a *arbiter) ensureBackendRunning(bs *backendState, modelName string) error {
	if bs.config.Name == "ollama" {
		if !bs.running {
			a.stopOtherBackends("ollama")
			bs.running = true
		}
		return nil
	}

	// Verify container is actually running (may have been stopped externally)
	if bs.running && a.podmanClient != nil && !a.containerIsRunning(bs.config.ContainerName) {
		log.Printf("%s container not running (externally stopped), resetting state", bs.config.Name)
		bs.running = false
		bs.currentModel = ""
	}

	needRecreate := !bs.running || (modelName != "" && bs.currentModel != modelName)
	if !needRecreate {
		return nil
	}

	a.stopOtherBackends(bs.config.Name)

	if modelName == "" {
		return fmt.Errorf("model name required for %s", bs.config.Name)
	}

	if bs.currentModel != "" && bs.currentModel != modelName {
		log.Printf("switching %s model: %s → %s", bs.config.Name, bs.currentModel, modelName)
	}
	log.Printf("starting %s with model %s...", bs.config.Name, modelName)
	if err := a.containerRecreate(bs, modelName); err != nil {
		return fmt.Errorf("failed to start %s: %w", bs.config.Name, err)
	}

	// Release lock during health wait
	a.mu.Unlock()
	err := a.waitForHealthy(bs, 300*time.Second)
	a.mu.Lock()

	if err != nil {
		return err
	}

	bs.running = true
	bs.currentModel = modelName
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
			modelName = parsed.Model

			policy := a.requestPolicy(req)
			body = a.applyReasoningPolicy(backendName, req.URL.Path, modelName, policy, body)
			req.Body = io.NopCloser(bytes.NewReader(body))
			req.ContentLength = int64(len(body))
			req.Header.Set("Content-Length", strconv.Itoa(len(body)))
		}

		a.mu.Lock()
		bs.lastRequest = time.Now()
		if err := a.ensureBackendRunning(bs, modelName); err != nil {
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
	if backendName != "ollama" {
		// vLLM/SGLang are dormant and need separate probe/policy recipes.
		return body
	}

	switch strings.TrimRight(path, "/") {
	case "/api/chat", "/api/generate":
		return a.applyOllamaNativePolicy(modelName, policy, body)
	case "/v1/chat/completions":
		return a.applyOllamaOpenAIChatPolicy(modelName, policy, body)
	case "/v1/messages":
		return a.applyOllamaAnthropicMessagesPolicy(modelName, policy, body)
	default:
		return body
	}
}

func (a *arbiter) reasoningAction(modelName, policy string) reasoningAction {
	if a.modelCapability[modelName] != "structured" {
		return reasoningNoop
	}
	switch policy {
	case "auto", "low", "medium", "high":
		return reasoningEnable
	case "off":
		if a.modelDisableOK[modelName] {
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
