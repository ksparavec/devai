package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/url"
)

// ollamaLoadedModel is one entry of Ollama's /api/ps.
type ollamaLoadedModel struct {
	Name          string `json:"name"`
	ContextLength int    `json:"context_length"`
}

// ollamaLoadedModelsAt reads /api/ps. An error means "unknown", never
// "nothing loaded": no caller may read it as a free GPU.
func ollamaLoadedModelsAt(client *http.Client, base *url.URL) ([]ollamaLoadedModel, error) {
	resp, err := client.Get(base.String() + "/api/ps")
	if err != nil {
		return nil, fmt.Errorf("cannot reach ollama: %w", err)
	}
	defer resp.Body.Close()
	var ps struct {
		Models []ollamaLoadedModel `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&ps); err != nil {
		// A malformed body would silently read as "no models", and the
		// caller would then hand the GPU to another backend with Ollama's
		// weights still resident.
		return nil, fmt.Errorf("cannot decode /api/ps: %w", err)
	}
	return ps.Models, nil
}

// reconcileOllamaState is the Ollama half of reconcileBackendState. A
// /health probe says nothing here (the container is always up), but
// /api/ps says exactly which model is resident, so Ollama is adopted
// WITH its model rather than merely as "live".
//
// Without this, a router restarted while Ollama held a warm model believed
// the card was free. stopOtherBackends only touches a backend whose flags
// say it is running, so unloadOllama -- the one place that does read
// /api/ps at switch time -- was never reached, and the next HF launch died
// with "Free memory on device cuda:0 (0.3/23.43 GiB) on startup is less
// than desired GPU memory utilization" (observed 2026-09-22:
// qwen3.6:35b-a3b-mtp-q4_K_M resident across the `make cache-up` that
// recreated the router).
//
// Adopting the model name is safe here, unlike for the HF backends: /api/ps
// is authoritative, and a name mismatch merely costs a reload.
func (a *arbiter) reconcileOllamaState(bs *backendState) {
	client := a.healthClient
	if client == nil {
		client = &http.Client{Timeout: unloadOllamaTimeout}
	}
	models, err := ollamaLoadedModelsAt(client, bs.config.BackendURL)
	if err != nil {
		log.Printf("warning: reconcile: %v; if a model is resident, the next non-ollama launch will fail on a full GPU", err)
		return
	}
	if len(models) == 0 {
		return
	}
	m := models[0]
	bs.running = true
	bs.containerLaunched = true
	bs.currentModel = m.Name
	bs.currentContext = m.ContextLength
	log.Printf("adopted resident ollama model %s (ctx=%d, %d loaded) from /api/ps; it is unloaded on a switch",
		m.Name, m.ContextLength, len(models))
}
