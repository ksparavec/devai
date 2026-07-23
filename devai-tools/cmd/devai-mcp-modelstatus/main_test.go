package main

import (
	"context"
	"net/http"
	"testing"
	"time"

	"github.com/sparavec/devai-tools/internal/modelcache"
)

func testServer() *server {
	catalog := []modelcache.CatalogEntry{
		{Name: "Qwen3-8B-NVFP4", Family: "qwen3", Backend: []string{"vllm"}, Repo: "nvidia/Qwen3-8B-NVFP4", Sha: "abc123def456", Size: "5.2 GB"},
	}
	vllmCache := modelcache.ProbeCache{
		"nvidia/Qwen3-8B-NVFP4@abc123def456": {
			Repo: "nvidia/Qwen3-8B-NVFP4",
			Sha:  "abc123def456",
			Probes: map[string]map[string]modelcache.ProbeCell{
				"24": {"131072": {Fits: true}},
			},
		},
	}
	benchCache := modelcache.BenchCache{
		"nvidia/Qwen3-8B-NVFP4@abc123def456::vllm::131072": {
			Context: 131072,
			Metrics: map[string]any{"tps_sustained_p50": 98.3},
			Tasks: map[string]map[string]any{
				"gsm8k_subset_100":    {"score": 0.9},
				"humaneval_subset_50": {"pass@1": 0.8},
				"tools_use_20":        {"score": 1.0},
			},
		},
	}
	return &server{
		catalog:     catalog,
		probeCaches: map[string]modelcache.ProbeCache{"vllm": vllmCache},
		benchCache:  benchCache,
		httpClient:  &http.Client{Timeout: time.Second},
		// weightStores deliberately nil: the deployed container mounts no
		// weight volumes, so this is the degraded path (see
		// TestListFittingModelsHandlerReportsMissingStore).
	}
}

func TestListFittingModelsHandler(t *testing.T) {
	srv := testServer()
	_, out, err := srv.listFittingModels(context.Background(), nil, listFittingModelsInput{VRAMGB: 24, Context: 131072})
	if err != nil {
		t.Fatalf("listFittingModels: %v", err)
	}
	if len(out.Models) != 1 || out.Models[0].Name != "Qwen3-8B-NVFP4" {
		t.Fatalf("Models = %+v, want one Qwen3-8B-NVFP4 row", out.Models)
	}
}

// When the weight store cannot be seen the rows are probe-cache-only, and
// the tool must say so rather than presenting them as verified-servable.
func TestListFittingModelsHandlerReportsMissingStore(t *testing.T) {
	srv := testServer()
	_, out, err := srv.listFittingModels(context.Background(), nil, listFittingModelsInput{VRAMGB: 24, Context: 131072})
	if err != nil {
		t.Fatalf("listFittingModels: %v", err)
	}
	if len(out.Models) != 1 {
		t.Fatalf("Models = %+v, want the un-gated row to survive the degradation", out.Models)
	}
	if len(out.Notes) != 1 {
		t.Fatalf("Notes = %v, want one weights-check-skipped note", out.Notes)
	}
}

// A visible store filters the list down to models whose weights are
// actually on disk, and reports no note.
func TestListFittingModelsHandlerGatesOnStore(t *testing.T) {
	srv := testServer()
	srv.weightStores = modelcache.WeightStores{"vllm": t.TempDir()}
	_, out, err := srv.listFittingModels(context.Background(), nil, listFittingModelsInput{VRAMGB: 24, Context: 131072})
	if err != nil {
		t.Fatalf("listFittingModels: %v", err)
	}
	if len(out.Models) != 0 {
		t.Fatalf("Models = %+v, want none -- the store has no weights for it", out.Models)
	}
	if len(out.Notes) != 0 {
		t.Fatalf("Notes = %v, want none -- an empty store is an answer, not a degradation", out.Notes)
	}
}

func TestListFittingModelsHandlerNoMatch(t *testing.T) {
	srv := testServer()
	_, out, err := srv.listFittingModels(context.Background(), nil, listFittingModelsInput{VRAMGB: 8, Context: 131072})
	if err != nil {
		t.Fatalf("listFittingModels: %v", err)
	}
	if len(out.Models) != 0 {
		t.Fatalf("Models = %+v, want empty (not nil, not an error) for a VRAM band with no fitting cell", out.Models)
	}
}

func TestGetModelBenchHandlerHit(t *testing.T) {
	srv := testServer()
	_, out, err := srv.getModelBench(context.Background(), nil, getModelBenchInput{
		Model: "Qwen3-8B-NVFP4", Backend: "vllm", Context: 131072,
	})
	if err != nil {
		t.Fatalf("getModelBench: %v", err)
	}
	if out.TPS == nil || *out.TPS != 98.3 {
		t.Errorf("TPS = %v, want 98.3", out.TPS)
	}
	if out.Message != "" {
		t.Errorf("Message = %q, want empty on a hit", out.Message)
	}
}

func TestGetModelBenchHandlerUnknownModel(t *testing.T) {
	srv := testServer()
	_, out, err := srv.getModelBench(context.Background(), nil, getModelBenchInput{
		Model: "no-such-model", Backend: "vllm", Context: 131072,
	})
	if err != nil {
		t.Fatalf("getModelBench: %v", err)
	}
	if out.Message == "" {
		t.Error("expected a not-found message for an unknown model")
	}
}

func TestGetModelBenchHandlerMissingCtx(t *testing.T) {
	srv := testServer()
	_, out, err := srv.getModelBench(context.Background(), nil, getModelBenchInput{
		Model: "Qwen3-8B-NVFP4", Backend: "vllm", Context: 32768,
	})
	if err != nil {
		t.Fatalf("getModelBench: %v", err)
	}
	if out.TPS != nil {
		t.Errorf("TPS = %v, want nil when the ctx wasn't benched", out.TPS)
	}
	want := "Bench: not available at ctx=32768 (have [131072]; run `make bench --ctx 32768` to populate)"
	if out.Message != want {
		t.Errorf("Message = %q, want %q", out.Message, want)
	}
}

func TestGetRouterStatusHandlerUnreachable(t *testing.T) {
	t.Setenv("DEVAI_ROUTER_CLUSTER_URL", "http://127.0.0.1:1")
	t.Setenv("DEVAI_ROUTER_HOST", "127.0.0.1")
	t.Setenv("DEVAI_ROUTER_OLLAMA_PORT", "1")
	t.Setenv("DEVAI_ROUTER_VLLM_PORT", "1")
	t.Setenv("DEVAI_ROUTER_SGLANG_PORT", "1")

	srv := testServer()
	_, status, err := srv.getRouterStatus(context.Background(), nil, getRouterStatusInput{})
	if err != nil {
		t.Fatalf("getRouterStatus: %v", err)
	}
	if status.Mode != "unreachable" {
		t.Errorf("Mode = %q, want unreachable", status.Mode)
	}
}
