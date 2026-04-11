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
	Entrypoint    func(modelName string) []string
	EnvVars       map[string]string
}

type configModel struct {
	Name    string   `yaml:"name"`
	Backend []string `yaml:"backend"`
	Repo    string   `yaml:"repo"`
	Size    string   `yaml:"size"`
	Purpose string   `yaml:"purpose"`
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
	backends     map[string]*backendState
	mu           sync.Mutex
	ollamaURL    *url.URL
	podmanClient *http.Client
	idleTimeout  time.Duration
	drainTimeout time.Duration
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

// --- Entrypoint builders ---

func vllmEntrypoint(modelName string) []string {
	return []string{
		"python3", "-m", "vllm.entrypoints.openai.api_server",
		"--model", "/models/" + modelName,
		"--host", "0.0.0.0",
		"--port", "11434",
		"--tensor-parallel-size", "1",
		"--max-model-len", "65536",
		"--gpu-memory-utilization", "0.95",
		"--enable-prefix-caching",
		"--enable-auto-tool-choice",
		"--tool-call-parser", "qwen3_coder",
		"--trust-remote-code",
		"--served-model-name", modelName,
	}
}

func sglangEntrypoint(modelName string) []string {
	return []string{
		"python3", "-m", "sglang.launch_server",
		"--model-path", "/models/" + modelName,
		"--host", "0.0.0.0",
		"--port", "11434",
		"--tp", "1",
		"--mem-fraction-static", "0.95",
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

	a := &arbiter{
		backends:     make(map[string]*backendState),
		ollamaURL:    ollamaURL,
		podmanClient: podmanClient,
		idleTimeout:  time.Duration(envInt("IDLE_TIMEOUT", 300)) * time.Second,
		drainTimeout: time.Duration(envInt("DRAIN_TIMEOUT", 30)) * time.Second,
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

	spec := map[string]any{
		"image":      cfg.Image,
		"name":       cfg.ContainerName,
		"entrypoint": cfg.Entrypoint(modelName),
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
	if bs.running && !a.containerIsRunning(bs.config.ContainerName) {
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

		// For non-Ollama backends, extract model from request body
		var modelName string
		if backendName != "ollama" && req.Method == http.MethodPost && req.Body != nil {
			body, err := io.ReadAll(req.Body)
			if err != nil {
				http.Error(w, `{"error":"failed to read body"}`, http.StatusBadRequest)
				return
			}
			req.Body = io.NopCloser(bytes.NewReader(body))

			var parsed struct {
				Model string `json:"model"`
			}
			json.Unmarshal(body, &parsed)
			modelName = parsed.Model
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

func (a *arbiter) makeModelsHandler(backendName string) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		bs := a.backends[backendName]

		type model struct {
			ID       string `json:"id"`
			Object   string `json:"object"`
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
