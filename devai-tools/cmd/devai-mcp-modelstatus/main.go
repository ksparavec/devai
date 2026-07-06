// Command devai-mcp-modelstatus is a stdio MCP server exposing read-only
// query tools over devai's model catalog, probe caches, bench cache, and
// live router status. Registered with the Docker MCP Gateway (see
// deploy/mcp-servers.yaml) rather than invoked directly by agents.
// See docs/mcp-model-status.md.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/sparavec/devai-tools/internal/modelcache"
	"github.com/sparavec/devai-tools/internal/routerclient"
)

type server struct {
	catalog     []modelcache.CatalogEntry
	probeCaches map[string]modelcache.ProbeCache // keyed by backend: ollama, vllm, sglang
	benchCache  modelcache.BenchCache
	httpClient  *http.Client
}

// ── list_fitting_models ──────────────────────────────────────────────────

type listFittingModelsInput struct {
	VRAMGB  int    `json:"vram_gb" jsonschema:"advertised VRAM budget in GB"`
	Context int    `json:"context" jsonschema:"requested context length in tokens"`
	Backend string `json:"backend,omitempty" jsonschema:"optional: restrict to one backend (ollama, vllm, or sglang); omit for all three"`
}

type listFittingModelsOutput struct {
	Models []modelcache.FitResult `json:"models"`
}

func (s *server) listFittingModels(_ context.Context, _ *mcp.CallToolRequest, in listFittingModelsInput) (*mcp.CallToolResult, listFittingModelsOutput, error) {
	models := modelcache.ListFitting(s.catalog, s.probeCaches, in.VRAMGB, in.Context, in.Backend)
	if models == nil {
		models = []modelcache.FitResult{}
	}
	return nil, listFittingModelsOutput{Models: models}, nil
}

// ── get_model_bench ──────────────────────────────────────────────────────

type getModelBenchInput struct {
	Model   string `json:"model" jsonschema:"catalog model name, e.g. Qwen3-8B-NVFP4 or qwen3.5:9b-q8_0"`
	Backend string `json:"backend" jsonschema:"ollama, vllm, or sglang"`
	Context int    `json:"context" jsonschema:"context length in tokens the model would be served at"`
}

type getModelBenchOutput struct {
	TPS      *float64 `json:"tps,omitempty"`
	CodePct  *float64 `json:"code_pct,omitempty"`
	ReasPct  *float64 `json:"reas_pct,omitempty"`
	TotalPct *float64 `json:"total_pct,omitempty"`
	LeakPct  *float64 `json:"leak_pct,omitempty"`
	Message  string   `json:"message,omitempty"`
}

func (s *server) getModelBench(_ context.Context, _ *mcp.CallToolRequest, in getModelBenchInput) (*mcp.CallToolResult, getModelBenchOutput, error) {
	base, ok := modelcache.ResolveBenchBase(s.catalog, s.probeCaches["ollama"], in.Backend, in.Model)
	if !ok {
		return nil, getModelBenchOutput{
			Message: fmt.Sprintf("unknown model %q for backend %q (no matching catalog/probe-cache entry)", in.Model, in.Backend),
		}, nil
	}

	key := modelcache.BenchKey(base, in.Backend, in.Context)
	row, found := s.benchCache[key]
	if !found {
		have := modelcache.OtherContexts(s.benchCache, base, in.Backend)
		msg := fmt.Sprintf("Bench: not available at ctx=%d ", in.Context)
		if len(have) > 0 {
			msg += fmt.Sprintf("(have %v; run `make bench --ctx %d` to populate)", have, in.Context)
		} else {
			msg += fmt.Sprintf("(have none; run `make bench --ctx %d` to populate)", in.Context)
		}
		return nil, getModelBenchOutput{Message: msg}, nil
	}

	scores := modelcache.ComputeScores(row)
	return nil, getModelBenchOutput{
		TPS:      scores.TPS,
		CodePct:  asPercent(scores.Code),
		ReasPct:  asPercent(scores.Reas),
		TotalPct: asPercent(scores.Total),
		LeakPct:  asPercent(scores.Leak),
	}, nil
}

func asPercent(v *float64) *float64 {
	if v == nil {
		return nil
	}
	p := *v * 100
	return &p
}

// ── get_router_status ────────────────────────────────────────────────────

type getRouterStatusInput struct{}

func (s *server) getRouterStatus(ctx context.Context, _ *mcp.CallToolRequest, _ getRouterStatusInput) (*mcp.CallToolResult, routerclient.Status, error) {
	return nil, routerclient.GetStatus(ctx, s.httpClient), nil
}

// ── main ─────────────────────────────────────────────────────────────────

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func main() {
	modelsYAML := flag.String("models-yaml", envOr("DEVAI_MODELS_YAML", "/etc/devai/models.yaml"), "path to models.yaml")
	ollamaCachePath := flag.String("ollama-cache", envOr("DEVAI_OLLAMA_PROBE_CACHE", "/etc/devai/.ollama-reasoning-cache.json"), "path to the Ollama probe cache")
	vllmCachePath := flag.String("vllm-cache", envOr("DEVAI_VLLM_PROBE_CACHE", "/etc/devai/.vllm-reasoning-cache.json"), "path to the vLLM probe cache")
	sglangCachePath := flag.String("sglang-cache", envOr("DEVAI_SGLANG_PROBE_CACHE", "/etc/devai/.sglang-reasoning-cache.json"), "path to the SGLang probe cache")
	benchCachePath := flag.String("bench-cache", envOr("DEVAI_BENCH_CACHE", "/etc/devai/.bench-cache.json"), "path to the bench cache")
	flag.Parse()

	catalog, err := modelcache.LoadCatalog(*modelsYAML)
	if err != nil {
		log.Fatalf("load catalog %s: %v", *modelsYAML, err)
	}

	probeCaches := map[string]modelcache.ProbeCache{}
	for backend, path := range map[string]string{
		"ollama": *ollamaCachePath,
		"vllm":   *vllmCachePath,
		"sglang": *sglangCachePath,
	} {
		cache, err := modelcache.LoadProbeCache(path)
		if err != nil {
			log.Fatalf("load %s probe cache %s: %v", backend, path, err)
		}
		probeCaches[backend] = cache
	}

	benchCache, err := modelcache.LoadBenchCache(*benchCachePath)
	if err != nil {
		log.Fatalf("load bench cache %s: %v", *benchCachePath, err)
	}

	srv := &server{
		catalog:     catalog,
		probeCaches: probeCaches,
		benchCache:  benchCache,
		httpClient:  &http.Client{Timeout: 5 * time.Second},
	}

	mcpServer := mcp.NewServer(&mcp.Implementation{Name: "devai-model-status", Version: "0.1.0"}, nil)

	mcp.AddTool(mcpServer, &mcp.Tool{
		Name:        "list_fitting_models",
		Description: "List catalog models that fit at a given VRAM budget and context length, per devai's probe caches. No interpolation: a model is eligible only when the relevant probe cache has an exact (vram_gb, context) cell recorded as fitting.",
	}, srv.listFittingModels)

	mcp.AddTool(mcpServer, &mcp.Tool{
		Name:        "get_model_bench",
		Description: "Look up bench scores (tps, code_pct, reas_pct, total_pct, leak_pct) for one model/backend/context. Returns a not-available message (naming any other benched contexts) when the exact triple hasn't been benched.",
	}, srv.getModelBench)

	mcp.AddTool(mcpServer, &mcp.Tool{
		Name:        "get_router_status",
		Description: "Live status of the running devai-router: cluster-head worker list, or single-mode per-backend health (running/current_model/active_reqs), or an unreachable report if nothing responds.",
	}, srv.getRouterStatus)

	if err := mcpServer.Run(context.Background(), &mcp.StdioTransport{}); err != nil {
		log.Fatalf("server failed: %v", err)
	}
}
