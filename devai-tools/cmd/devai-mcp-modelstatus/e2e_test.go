package main

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// TestStdioProtocolEndToEnd builds the real binary and drives it as a real
// MCP client over stdio (mcp.CommandTransport), the same wire protocol the
// Docker MCP Gateway speaks to it -- not just calling the Go handler
// functions directly (that's main_test.go's job). Uses the hand-crafted
// fixtures under tests/fixtures/modelstatus/ so this exercises the exact
// files a real deployment would read.
func TestStdioProtocolEndToEnd(t *testing.T) {
	binPath := buildBinary(t)
	fixtures := repoFixturesDir(t)

	cmd := exec.Command(binPath,
		"--models-yaml", filepath.Join(fixtures, "models.yaml"),
		"--ollama-cache", filepath.Join(fixtures, "ollama-reasoning-cache.json"),
		"--vllm-cache", filepath.Join(fixtures, "vllm-reasoning-cache.json"),
		"--sglang-cache", filepath.Join(fixtures, "does-not-exist.json"),
		"--bench-cache", filepath.Join(fixtures, "bench-cache.json"),
	)
	// list_fitting_models gates vLLM/SGLang rows on the weights being in
	// the backend's store, so the child must not inherit this host's real
	// /var/cache/devai/vllm (the fixture model is not in it, and on a host
	// that has the directory at all the vLLM row would vanish). Point it
	// at a fixture store holding just that model.
	cmd.Env = append(os.Environ(),
		"VLLM_MODELS_DIR="+fixtureStore(t, "Qwen3-8B-NVFP4"),
		"SGLANG_MODELS_DIR="+filepath.Join(t.TempDir(), "no-sglang-store"),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	transport := &mcp.CommandTransport{Command: cmd}
	client := mcp.NewClient(&mcp.Implementation{Name: "e2e-test-client", Version: "0.0.1"}, nil)
	cs, err := client.Connect(ctx, transport, nil)
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	defer func() { _ = cs.Close() }()

	wantTools := map[string]bool{"list_fitting_models": false, "get_model_bench": false, "get_router_status": false}
	for tool, err := range cs.Tools(ctx, nil) {
		if err != nil {
			t.Fatalf("tools/list: %v", err)
		}
		if _, ok := wantTools[tool.Name]; ok {
			wantTools[tool.Name] = true
		}
	}
	for name, found := range wantTools {
		if !found {
			t.Errorf("tools/list did not include %q", name)
		}
	}

	res, err := cs.CallTool(ctx, &mcp.CallToolParams{
		Name:      "list_fitting_models",
		Arguments: map[string]any{"vram_gb": 24, "context": 131072},
	})
	if err != nil {
		t.Fatalf("tools/call list_fitting_models: %v", err)
	}
	if res.IsError {
		t.Fatalf("list_fitting_models returned a tool error: %+v", res.Content)
	}
	var listOut struct {
		Models []struct {
			Name    string `json:"name"`
			Backend string `json:"backend"`
		} `json:"models"`
	}
	if err := unmarshalStructured(res.StructuredContent, &listOut); err != nil {
		t.Fatalf("unmarshal list_fitting_models result: %v", err)
	}
	if len(listOut.Models) != 2 {
		t.Fatalf("list_fitting_models = %+v, want 2 models (ollama qwen3.5 + vllm Qwen3-8B-NVFP4)", listOut.Models)
	}

	res, err = cs.CallTool(ctx, &mcp.CallToolParams{
		Name:      "get_model_bench",
		Arguments: map[string]any{"model": "Qwen3-8B-NVFP4", "backend": "vllm", "context": 131072},
	})
	if err != nil {
		t.Fatalf("tools/call get_model_bench: %v", err)
	}
	var benchOut struct {
		TPS *float64 `json:"tps"`
	}
	if err := unmarshalStructured(res.StructuredContent, &benchOut); err != nil {
		t.Fatalf("unmarshal get_model_bench result: %v", err)
	}
	if benchOut.TPS == nil || *benchOut.TPS != 98.3 {
		t.Errorf("get_model_bench tps = %v, want 98.3", benchOut.TPS)
	}

	res, err = cs.CallTool(ctx, &mcp.CallToolParams{Name: "get_router_status", Arguments: map[string]any{}})
	if err != nil {
		t.Fatalf("tools/call get_router_status: %v", err)
	}
	var statusOut struct {
		Mode string `json:"mode"`
	}
	if err := unmarshalStructured(res.StructuredContent, &statusOut); err != nil {
		t.Fatalf("unmarshal get_router_status result: %v", err)
	}
	if statusOut.Mode == "" {
		t.Error("get_router_status: empty Mode")
	}
}

func unmarshalStructured(v any, out any) error {
	data, err := json.Marshal(v)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, out)
}

func buildBinary(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	binPath := filepath.Join(dir, "devai-mcp-modelstatus")
	cmd := exec.Command("go", "build", "-o", binPath, ".")
	cmd.Dir = mustGetwd(t)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("go build: %v\n%s", err, out)
	}
	return binPath
}

func mustGetwd(t *testing.T) string {
	t.Helper()
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	return wd
}

// fixtureStore builds a throwaway VLLM_MODELS_DIR holding one model
// directory per name, each with the config.json that marks weights as
// present -- the same shape model-picker.py enumerates the real store with.
func fixtureStore(t *testing.T, names ...string) string {
	t.Helper()
	dir := t.TempDir()
	for _, name := range names {
		if err := os.MkdirAll(filepath.Join(dir, name), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, name, "config.json"), []byte("{}"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

// repoFixturesDir resolves tests/fixtures/modelstatus relative to the repo
// root (devai-tools/cmd/devai-mcp-modelstatus is 3 levels down from it).
func repoFixturesDir(t *testing.T) string {
	t.Helper()
	wd := mustGetwd(t)
	root := filepath.Join(wd, "..", "..", "..")
	dir := filepath.Join(root, "tests", "fixtures", "modelstatus")
	if _, err := os.Stat(dir); err != nil {
		t.Fatalf("fixtures dir %s not found: %v", dir, err)
	}
	return dir
}
