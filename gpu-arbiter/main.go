// gpu-arbiter — LLM inference router with automatic GPU exclusion.
//
// Routes requests to Ollama (GGUF) or vLLM (NVFP4) based on model name.
// Manages vLLM container lifecycle via the Podman API socket.
// Ensures only one backend uses the GPU at a time.
//
// Environment variables:
//
//	LISTEN_ADDR       listen address (default :11434)
//	OLLAMA_URL        Ollama backend URL (default http://devai-ollama:11434)
//	VLLM_URL          vLLM backend URL (default http://devai-vllm:11434)
//	VLLM_CONTAINER    vLLM container name (default devai-vllm)
//	VLLM_PATTERNS     comma-separated model name patterns for vLLM routing (default NVFP4,nvfp4)
//	IDLE_TIMEOUT      seconds before idle vLLM is stopped (default 300)
//	PODMAN_SOCKET     path to Podman API socket (default /run/podman/podman.sock)
package main

import (
	"bufio"
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
	"syscall"
	"time"

	"gopkg.in/yaml.v3"
)

type configFile struct {
	Models struct {
		VLLM []struct {
			Name string `yaml:"name"`
		} `yaml:"vllm"`
	} `yaml:"models"`
}

func env(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

type router struct {
	ollamaURL      *url.URL
	vllmURL        *url.URL
	ollamaProxy    *httputil.ReverseProxy
	vllmProxy      *httputil.ReverseProxy
	patterns       []string
	idleTimeout    time.Duration
	vllmContainer  string
	vllmImage      string
	vllmModelsDir  string
	vllmModelNames []string
	vllmNetwork    string
	podmanClient   *http.Client

	mu              sync.Mutex
	activeBackend   string // "ollama" or "vllm"
	vllmRunning     bool
	vllmCurrentModel string
	lastVLLMReq     time.Time
}

func main() {
	ollamaURL, _ := url.Parse(env("OLLAMA_URL", "http://devai-ollama:11434"))
	vllmURL, _ := url.Parse(env("VLLM_URL", "http://devai-vllm:11434"))
	patterns := strings.Split(env("VLLM_PATTERNS", "NVFP4,nvfp4"), ",")
	socketPath := env("PODMAN_SOCKET", "/run/podman/podman.sock")

	idleSec := 300
	if v := os.Getenv("IDLE_TIMEOUT"); v != "" {
		fmt.Sscanf(v, "%d", &idleSec)
	}

	// HTTP client that talks to Podman via Unix socket
	podmanClient := &http.Client{
		Transport: &http.Transport{
			DialContext: func(_ context.Context, _, _ string) (net.Conn, error) {
				return net.Dial("unix", socketPath)
			},
		},
		Timeout: 30 * time.Second,
	}

	// Load vLLM model names from models.yaml
	var vllmModelNames []string
	configPath := env("CONFIG_FILE", "/etc/devai/models.yaml")
	if data, err := os.ReadFile(configPath); err == nil {
		var cfg configFile
		if err := yaml.Unmarshal(data, &cfg); err == nil {
			for _, m := range cfg.Models.VLLM {
				vllmModelNames = append(vllmModelNames, m.Name)
			}
		}
	}

	r := &router{
		ollamaURL:      ollamaURL,
		vllmURL:        vllmURL,
		ollamaProxy:    newProxy(ollamaURL),
		vllmProxy:      newProxy(vllmURL),
		patterns:       patterns,
		idleTimeout:    time.Duration(idleSec) * time.Second,
		vllmContainer:  env("VLLM_CONTAINER", "devai-vllm"),
		vllmImage:      env("VLLM_IMAGE", "docker.io/vllm/vllm-openai:latest-cu130-ubuntu2404"),
		vllmModelsDir:  env("VLLM_MODELS_DIR", "/var/cache/devai/ollama/models/vllm"),
		vllmNetwork:    env("VLLM_NETWORK", "devai-net"),
		vllmModelNames: vllmModelNames,
		podmanClient:   podmanClient,
		activeBackend:  "ollama",
	}

	go r.idleWatcher()

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGTERM, syscall.SIGINT)
	go func() {
		<-sig
		log.Println("shutting down")
		os.Exit(0)
	}()

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/models", r.handleModels)
	mux.HandleFunc("/api/tags", r.handleTags)
	mux.HandleFunc("/health", r.handleHealth)
	mux.HandleFunc("/", r.handleRequest)

	addr := env("LISTEN_ADDR", ":11434")
	log.Printf("gpu-arbiter %s → ollama=%s vllm=%s patterns=%v idle=%ds",
		addr, ollamaURL, vllmURL, patterns, idleSec)
	log.Fatal(http.ListenAndServe(addr, mux))
}

func newProxy(target *url.URL) *httputil.ReverseProxy {
	p := httputil.NewSingleHostReverseProxy(target)
	p.FlushInterval = -1
	return p
}

func (r *router) isVLLMModel(model string) bool {
	lower := strings.ToLower(model)
	for _, p := range r.patterns {
		if strings.Contains(lower, strings.ToLower(p)) {
			return true
		}
	}
	return false
}

// --- Podman container management ---

func (r *router) containerRecreate(modelName string) error {
	// Stop and remove existing container
	r.containerStop()
	delURL := fmt.Sprintf("http://d/v4.0.0/libpod/containers/%s?force=true", r.vllmContainer)
	req, _ := http.NewRequest("DELETE", delURL, nil)
	if resp, err := r.podmanClient.Do(req); err == nil {
		resp.Body.Close()
	}

	// Create new container with the requested model
	modelPath := "/models/" + modelName
	spec := map[string]any{
		"image": r.vllmImage,
		"name":  r.vllmContainer,
		"entrypoint": []string{
			"python3", "-m", "vllm.entrypoints.openai.api_server",
			"--model", modelPath,
			"--host", "0.0.0.0",
			"--port", "11434",
			"--tensor-parallel-size", "1",
			"--max-model-len", "16384",
			"--gpu-memory-utilization", "0.95",
			"--trust-remote-code",
			"--served-model-name", modelName,
		},
		"command": []string{},
		"mounts": []map[string]any{{
			"destination": "/models",
			"source":      r.vllmModelsDir,
			"type":        "bind",
			"options":     []string{"ro"},
		}},
		"hostadd":      []string{"host.containers.internal:host-gateway"},
		"netns":        map[string]any{"nsmode": "bridge"},
		"Networks":     map[string]any{r.vllmNetwork: map[string]any{}},
		"devices":      []map[string]any{{"path": "nvidia.com/gpu=all"}},
		"selinux_opts": []string{"disable"},
		"hostname":     "vllm",
	}

	body, _ := json.Marshal(spec)
	createURL := "http://d/v4.0.0/libpod/containers/create"
	resp, err := r.podmanClient.Post(createURL, "application/json", bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("podman create: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("podman create: %s %s", resp.Status, respBody)
	}

	// Start the container
	startURL := fmt.Sprintf("http://d/v4.0.0/libpod/containers/%s/start", r.vllmContainer)
	resp2, err := r.podmanClient.Post(startURL, "", nil)
	if err != nil {
		return fmt.Errorf("podman start: %w", err)
	}
	defer resp2.Body.Close()
	if resp2.StatusCode >= 300 && resp2.StatusCode != http.StatusNotModified {
		respBody, _ := io.ReadAll(resp2.Body)
		return fmt.Errorf("podman start: %s %s", resp2.Status, respBody)
	}

	return nil
}

func (r *router) containerStop() error {
	url := fmt.Sprintf("http://d/v4.0.0/libpod/containers/%s/stop?timeout=10", r.vllmContainer)
	resp, err := r.podmanClient.Post(url, "", nil)
	if err != nil {
		return fmt.Errorf("podman stop: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotModified {
		return nil // already stopped
	}
	if resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("podman stop: %s %s", resp.Status, body)
	}
	return nil
}

func (r *router) containerIsRunning() bool {
	url := fmt.Sprintf("http://d/v4.0.0/libpod/containers/%s/json", r.vllmContainer)
	resp, err := r.podmanClient.Get(url)
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

func (r *router) waitForVLLM(timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	healthURL := r.vllmURL.String() + "/health"
	log.Printf("waiting for vLLM at %s (timeout %s)...", healthURL, timeout)
	for time.Now().Before(deadline) {
		resp, err := http.Get(healthURL)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				log.Println("vLLM ready")
				return nil
			}
		}
		time.Sleep(2 * time.Second)
	}
	return fmt.Errorf("vLLM did not become ready within %s", timeout)
}

// --- Request handlers ---

func (r *router) handleHealth(w http.ResponseWriter, req *http.Request) {
	r.mu.Lock()
	active := r.activeBackend
	vllmUp := r.vllmRunning
	model := r.vllmCurrentModel
	r.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"status":         "ok",
		"active_backend": active,
		"vllm_running":   vllmUp,
		"vllm_model":     model,
	})
}

func (r *router) handleTags(w http.ResponseWriter, req *http.Request) {
	// Merge Ollama /api/tags with vLLM models for Open WebUI compatibility
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

	// Fetch Ollama tags
	if resp, err := http.Get(r.ollamaURL.String() + "/api/tags"); err == nil {
		json.NewDecoder(resp.Body).Decode(&all)
		resp.Body.Close()
	}

	// Add vLLM models from config.yaml (always visible regardless of vLLM state)
	for _, name := range r.vllmModelNames {
		all.Models = append(all.Models, ollamaModel{
			Name:  name,
			Model: name,
		})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(all)
}

func (r *router) handleModels(w http.ResponseWriter, req *http.Request) {
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

	if resp, err := http.Get(r.ollamaURL.String() + "/v1/models"); err == nil {
		var list modelList
		json.NewDecoder(resp.Body).Decode(&list)
		resp.Body.Close()
		all = append(all, list.Data...)
	}

	if resp, err := http.Get(r.vllmURL.String() + "/v1/models"); err == nil {
		var list modelList
		json.NewDecoder(resp.Body).Decode(&list)
		resp.Body.Close()
		all = append(all, list.Data...)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(modelList{Object: "list", Data: all})
}

func (r *router) handleRequest(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodPost || req.Body == nil {
		r.ollamaProxy.ServeHTTP(w, req)
		return
	}

	body, err := io.ReadAll(io.LimitReader(req.Body, 64*1024))
	if err != nil {
		http.Error(w, `{"error":"failed to read body"}`, http.StatusBadRequest)
		return
	}
	req.Body = io.NopCloser(bytes.NewReader(body))

	var parsed struct {
		Model string `json:"model"`
	}
	json.Unmarshal(body, &parsed)

	if r.isVLLMModel(parsed.Model) {
		r.routeToVLLM(w, req, body)
	} else {
		r.routeToOllama(w, req, body)
	}
}

func (r *router) routeToOllama(w http.ResponseWriter, req *http.Request, body []byte) {
	r.mu.Lock()
	if r.activeBackend == "vllm" {
		log.Println("switching to ollama — stopping vLLM container")
		if err := r.containerStop(); err != nil {
			log.Printf("warning: failed to stop vLLM: %v", err)
		}
		r.vllmRunning = false
		r.vllmCurrentModel = ""
		r.activeBackend = "ollama"
	}
	r.mu.Unlock()

	req.Body = io.NopCloser(bytes.NewReader(body))
	r.ollamaProxy.ServeHTTP(w, req)
}

func (r *router) routeToVLLM(w http.ResponseWriter, req *http.Request, body []byte) {
	// Extract model name
	var parsed struct {
		Model string `json:"model"`
	}
	json.Unmarshal(body, &parsed)

	r.mu.Lock()
	needRecreate := !r.vllmRunning || r.vllmCurrentModel != parsed.Model

	if r.activeBackend != "vllm" {
		log.Println("switching to vllm — unloading ollama models")
		r.unloadOllama()
		r.activeBackend = "vllm"
	}

	if needRecreate {
		if r.vllmCurrentModel != "" && r.vllmCurrentModel != parsed.Model {
			log.Printf("switching vLLM model: %s → %s", r.vllmCurrentModel, parsed.Model)
		}
		log.Printf("starting vLLM container with model %s...", parsed.Model)
		if err := r.containerRecreate(parsed.Model); err != nil {
			r.mu.Unlock()
			log.Printf("error: failed to start vLLM: %v", err)
			http.Error(w, fmt.Sprintf(`{"error":"failed to start vLLM: %s"}`, err), http.StatusServiceUnavailable)
			return
		}
		r.mu.Unlock()

		if err := r.waitForVLLM(180 * time.Second); err != nil {
			log.Printf("error: %v", err)
			http.Error(w, fmt.Sprintf(`{"error":"%s"}`, err), http.StatusServiceUnavailable)
			return
		}

		r.mu.Lock()
		r.vllmRunning = true
		r.vllmCurrentModel = parsed.Model
	}
	r.lastVLLMReq = time.Now()
	r.mu.Unlock()

	// Translate Ollama API paths to OpenAI API for vLLM
	if req.URL.Path == "/api/chat" || req.URL.Path == "/api/generate" {
		r.translateOllamaToOpenAI(w, req, body)
		return
	}

	req.Body = io.NopCloser(bytes.NewReader(body))
	r.vllmProxy.ServeHTTP(w, req)
}

func (r *router) translateOllamaToOpenAI(w http.ResponseWriter, req *http.Request, body []byte) {
	// Parse Ollama request format
	var ollamaReq struct {
		Model    string           `json:"model"`
		Messages []map[string]any `json:"messages"`
		Prompt   string           `json:"prompt"`
		Stream   *bool            `json:"stream"`
	}
	json.Unmarshal(body, &ollamaReq)

	// Build OpenAI request
	messages := ollamaReq.Messages
	if len(messages) == 0 && ollamaReq.Prompt != "" {
		messages = []map[string]any{{"role": "user", "content": ollamaReq.Prompt}}
	}

	stream := true
	if ollamaReq.Stream != nil {
		stream = *ollamaReq.Stream
	}

	openaiReq := map[string]any{
		"model":    ollamaReq.Model,
		"messages": messages,
		"stream":   stream,
	}
	openaiBody, _ := json.Marshal(openaiReq)

	// Forward to vLLM's OpenAI endpoint
	vllmURL := r.vllmURL.String() + "/v1/chat/completions"
	proxyReq, _ := http.NewRequestWithContext(req.Context(), "POST", vllmURL, bytes.NewReader(openaiBody))
	proxyReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 300 * time.Second}
	resp, err := client.Do(proxyReq)
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"%s"}`, err), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	if stream {
		// Streaming: translate OpenAI SSE to Ollama streaming format
		w.Header().Set("Content-Type", "application/x-ndjson")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)

		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				// Send final Ollama done message
				done, _ := json.Marshal(map[string]any{
					"model":              ollamaReq.Model,
					"done":               true,
					"done_reason":        "stop",
					"message":            map[string]string{"role": "assistant", "content": ""},
				})
				w.Write(done)
				w.Write([]byte("\n"))
				if flusher != nil {
					flusher.Flush()
				}
				break
			}
			var chunk struct {
				Choices []struct {
					Delta struct {
						Content string `json:"content"`
					} `json:"delta"`
				} `json:"choices"`
			}
			if err := json.Unmarshal([]byte(data), &chunk); err != nil || len(chunk.Choices) == 0 {
				continue
			}
			ollamaChunk, _ := json.Marshal(map[string]any{
				"model":   ollamaReq.Model,
				"done":    false,
				"message": map[string]string{"role": "assistant", "content": chunk.Choices[0].Delta.Content},
			})
			w.Write(ollamaChunk)
			w.Write([]byte("\n"))
			if flusher != nil {
				flusher.Flush()
			}
		}
	} else {
		// Non-streaming: translate OpenAI response to Ollama format
		var openaiResp struct {
			Model   string `json:"model"`
			Choices []struct {
				Message struct {
					Role    string `json:"role"`
					Content string `json:"content"`
				} `json:"message"`
				FinishReason string `json:"finish_reason"`
			} `json:"choices"`
		}
		respBody, _ := io.ReadAll(resp.Body)
		if err := json.Unmarshal(respBody, &openaiResp); err != nil || len(openaiResp.Choices) == 0 {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(resp.StatusCode)
			w.Write(respBody)
			return
		}

		ollamaResp, _ := json.Marshal(map[string]any{
			"model":       openaiResp.Model,
			"done":        true,
			"done_reason": openaiResp.Choices[0].FinishReason,
			"message": map[string]string{
				"role":    openaiResp.Choices[0].Message.Role,
				"content": openaiResp.Choices[0].Message.Content,
			},
		})
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write(ollamaResp)
	}
}

func (r *router) unloadOllama() {
	resp, err := http.Get(r.ollamaURL.String() + "/api/ps")
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
		if resp, err := http.Post(r.ollamaURL.String()+"/api/generate", "application/json", bytes.NewReader(b)); err == nil {
			io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
		}
	}
	time.Sleep(2 * time.Second)
}

func (r *router) idleWatcher() {
	for {
		time.Sleep(30 * time.Second)
		r.mu.Lock()
		if r.vllmRunning && !r.lastVLLMReq.IsZero() {
			if time.Since(r.lastVLLMReq) > r.idleTimeout {
				log.Printf("vllm idle for %s, stopping container", time.Since(r.lastVLLMReq).Round(time.Second))
				if err := r.containerStop(); err != nil {
					log.Printf("warning: failed to stop vLLM: %v", err)
				}
				r.vllmRunning = false
				r.vllmCurrentModel = ""
				r.activeBackend = "ollama"
			}
		}
		r.mu.Unlock()
	}
}
