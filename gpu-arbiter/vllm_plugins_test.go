package main

import (
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// indexOf returns the position of needle in haystack, or -1 if absent.
// Used to assert flag ordering inside the entrypoint arg slice.
func indexOf(haystack []string, needle string) int {
	for i, v := range haystack {
		if v == needle {
			return i
		}
	}
	return -1
}

// writeRegistry writes a small registry JSON to a tempfile and returns
// its path. Tests use it to exercise the loader without depending on
// the checked-in deploy/vllm-plugins.json.
func writeRegistry(t *testing.T, body string) string {
	t.Helper()
	dir := t.TempDir()
	p := filepath.Join(dir, "vllm-plugins.json")
	if err := os.WriteFile(p, []byte(body), 0o600); err != nil {
		t.Fatalf("write registry: %v", err)
	}
	return p
}

func TestLoadVLLMPluginRegistry_MissingFileYieldsEmpty(t *testing.T) {
	r := loadVLLMPluginRegistry("/no/such/file.json", "/host/plugins")
	if r == nil {
		t.Fatal("expected non-nil registry on missing file")
	}
	if len(r.Plugins) != 0 {
		t.Errorf("expected zero entries, got %d", len(r.Plugins))
	}
	if r.ContainerDir != defaultPluginContainerDir {
		t.Errorf("container_dir default = %q, want %q",
			r.ContainerDir, defaultPluginContainerDir)
	}
}

func TestLoadVLLMPluginRegistry_ParsesEntries(t *testing.T) {
	p := writeRegistry(t, `{
		"container_dir": "/etc/devai/vllm-plugins",
		"plugins": {
			"deepseek_string": {"kind": "tool", "file": "deepseek_string_tool_parser.py"}
		}
	}`)
	r := loadVLLMPluginRegistry(p, "/host/plugins")
	e, ok := r.Lookup("deepseek_string")
	if !ok {
		t.Fatal("deepseek_string not found in registry")
	}
	if e.Kind != "tool" || e.File != "deepseek_string_tool_parser.py" {
		t.Errorf("entry mismatch: %+v", e)
	}
	if r.HostDir != "/host/plugins" {
		t.Errorf("host dir = %q, want /host/plugins", r.HostDir)
	}
	got := r.ContainerPath(e.File)
	want := "/etc/devai/vllm-plugins/deepseek_string_tool_parser.py"
	if got != want {
		t.Errorf("container path = %q, want %q", got, want)
	}
}

func TestLoadVLLMPluginRegistry_RejectsInvalidKind(t *testing.T) {
	p := writeRegistry(t, `{
		"plugins": {
			"bogus": {"kind": "weird", "file": "x.py"},
			"good":  {"kind": "tool", "file": "good.py"}
		}
	}`)
	r := loadVLLMPluginRegistry(p, "")
	if _, ok := r.Lookup("bogus"); ok {
		t.Error("invalid-kind entry should be filtered out")
	}
	if _, ok := r.Lookup("good"); !ok {
		t.Error("valid entry should remain")
	}
}

func TestLoadVLLMPluginRegistry_TolerantToCommentField(t *testing.T) {
	// The checked-in deploy/vllm-plugins.json carries an `_comment` field
	// for human readers; the loader must ignore it without error.
	p := writeRegistry(t, `{
		"_comment": ["this is a comment"],
		"plugins": {"x": {"kind": "tool", "file": "x.py"}}
	}`)
	r := loadVLLMPluginRegistry(p, "")
	if _, ok := r.Lookup("x"); !ok {
		t.Error("expected x to load despite _comment field")
	}
}

func TestVLLMPluginRegistry_LookupNilReceiver(t *testing.T) {
	var r *vllmPluginRegistry
	if _, ok := r.Lookup("anything"); ok {
		t.Error("nil receiver must return ok=false")
	}
	if got := r.ContainerPath("x.py"); got != "" {
		t.Errorf("nil ContainerPath = %q, want empty", got)
	}
}

// --- Entrypoint plugin-flag wiring ---

func TestVLLMEntrypoint_EmitsToolParserPluginBeforeToolCallParser(t *testing.T) {
	// vLLM resolves --tool-call-parser <name> at flag-parse time, so the
	// plugin file must be loaded by then. Asserting the order is the
	// whole point of the wiring — without it the launch fails with
	// "Unknown tool parser: deepseek_string".
	args := vllmEntrypoint("DeepSeek-R1-Distill-Qwen-7B", launchConfig{
		MemFraction:      0.90,
		MaxContext:       32768,
		ToolParser:       "deepseek_string",
		ToolParserPlugin: "/etc/devai/vllm-plugins/deepseek_string_tool_parser.py",
	})
	iPlug := indexOf(args, "--tool-parser-plugin")
	iParser := indexOf(args, "--tool-call-parser")
	if iPlug < 0 {
		t.Fatalf("--tool-parser-plugin missing: %v", args)
	}
	if iParser < 0 {
		t.Fatalf("--tool-call-parser missing: %v", args)
	}
	if iPlug >= iParser {
		t.Fatalf("plugin flag must precede parser flag (got plug=%d parser=%d): %v",
			iPlug, iParser, args)
	}
	if !sliceContains(args, "--tool-parser-plugin",
		"/etc/devai/vllm-plugins/deepseek_string_tool_parser.py") {
		t.Errorf("plugin path missing from args: %v", args)
	}
	if !sliceContains(args, "--tool-call-parser", "deepseek_string") {
		t.Errorf("parser name flag missing: %v", args)
	}
}

func TestVLLMEntrypoint_EmitsReasoningParserPluginBeforeReasoningParser(t *testing.T) {
	args := vllmEntrypoint("future-model", launchConfig{
		MemFraction:           0.90,
		MaxContext:            32768,
		ReasoningParser:       "future_reasoning",
		ReasoningParserPlugin: "/etc/devai/vllm-plugins/future_reasoning_parser.py",
	})
	iPlug := indexOf(args, "--reasoning-parser-plugin")
	iParser := indexOf(args, "--reasoning-parser")
	if iPlug < 0 || iParser < 0 {
		t.Fatalf("plugin or parser flag missing: %v", args)
	}
	if iPlug >= iParser {
		t.Fatalf("plugin flag must precede parser flag (got plug=%d parser=%d): %v",
			iPlug, iParser, args)
	}
}

func TestVLLMEntrypoint_BuiltinParserUnchanged(t *testing.T) {
	// Built-in parsers (no plugin path) must not gain plugin flags. This
	// is the existing fast-path; regression-guard it so the plugin
	// changes don't accidentally always emit --tool-parser-plugin.
	args := vllmEntrypoint("Qwen3.5-9B-NVFP4", launchConfig{
		MemFraction:     0.90,
		MaxContext:      32768,
		ReasoningParser: "qwen3",
		ToolParser:      "hermes",
	})
	for _, flag := range []string{"--tool-parser-plugin", "--reasoning-parser-plugin"} {
		if indexOf(args, flag) >= 0 {
			t.Fatalf("built-in path must not emit %s: %v", flag, args)
		}
	}
}

func TestSGLangEntrypoint_IgnoresPluginPaths(t *testing.T) {
	// SGLang's plugin model is Python-import based; file-path plugin
	// flags don't apply. Even when the launch config carries a plugin
	// path (e.g. when a future SGLang plugin re-uses the same field), the
	// SGLang launch must NOT emit --*-parser-plugin — those flags don't
	// exist in SGLang and would crash the launch.
	args := sglangEntrypoint("DeepSeek-R1-Distill-Qwen-7B", launchConfig{
		MemFraction:      0.85,
		MaxContext:       32768,
		ToolParser:       "deepseek-r1",
		ToolParserPlugin: "/etc/devai/vllm-plugins/deepseek_string_tool_parser.py",
	})
	for _, a := range args {
		if a == "--tool-parser-plugin" || a == "--reasoning-parser-plugin" {
			t.Fatalf("SGLang must not emit plugin flags: %v", args)
		}
	}
	if !sliceContains(args, "--tool-call-parser", "deepseek-r1") {
		t.Errorf("SGLang parser flag missing: %v", args)
	}
}

// --- arbiter.resolvePluginLaunch ---

func newPluginArbiter(reg *vllmPluginRegistry) *arbiter {
	return &arbiter{
		backends:       map[string]*backendState{},
		pluginRegistry: reg,
		mu:             sync.Mutex{},
	}
}

func TestResolvePluginLaunch_NoBackendMatchReturnsNil(t *testing.T) {
	r := loadVLLMPluginRegistry(writeRegistry(t, `{
		"plugins": {"deepseek_string": {"kind": "tool", "file": "x.py"}}
	}`), "/host")
	a := newPluginArbiter(r)

	// SGLang is not the vLLM backend → never resolves plugins, even
	// when the parser name happens to match a registry entry.
	lc := &launchConfig{ToolParser: "deepseek_string"}
	mount, err := a.resolvePluginLaunch("sglang", lc)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if mount != nil {
		t.Errorf("expected nil mount for sglang, got %+v", mount)
	}
	if lc.ToolParserPlugin != "" {
		t.Errorf("ToolParserPlugin must stay empty for sglang, got %q",
			lc.ToolParserPlugin)
	}
}

func TestResolvePluginLaunch_BuiltinToolParserReturnsNil(t *testing.T) {
	r := loadVLLMPluginRegistry(writeRegistry(t, `{
		"plugins": {"deepseek_string": {"kind": "tool", "file": "x.py"}}
	}`), "/host")
	a := newPluginArbiter(r)

	lc := &launchConfig{ToolParser: "hermes"}
	mount, err := a.resolvePluginLaunch("vllm", lc)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if mount != nil {
		t.Errorf("built-in parser must not allocate a mount: %+v", mount)
	}
	if lc.ToolParserPlugin != "" {
		t.Errorf("ToolParserPlugin must stay empty: %q", lc.ToolParserPlugin)
	}
}

func TestResolvePluginLaunch_PluginToolParserAddsMountAndPath(t *testing.T) {
	r := loadVLLMPluginRegistry(writeRegistry(t, `{
		"container_dir": "/etc/devai/vllm-plugins",
		"plugins": {"deepseek_string": {"kind": "tool", "file": "deepseek_string_tool_parser.py"}}
	}`), "/host/plugins")
	a := newPluginArbiter(r)

	lc := &launchConfig{ToolParser: "deepseek_string"}
	mount, err := a.resolvePluginLaunch("vllm", lc)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if mount == nil {
		t.Fatal("expected mount, got nil")
	}
	wantPath := "/etc/devai/vllm-plugins/deepseek_string_tool_parser.py"
	if lc.ToolParserPlugin != wantPath {
		t.Errorf("ToolParserPlugin = %q, want %q", lc.ToolParserPlugin, wantPath)
	}
	if mount["source"] != "/host/plugins" {
		t.Errorf("mount source = %v, want /host/plugins", mount["source"])
	}
	if mount["destination"] != "/etc/devai/vllm-plugins" {
		t.Errorf("mount destination = %v, want /etc/devai/vllm-plugins",
			mount["destination"])
	}
}

func TestResolvePluginLaunch_MissingHostDirIsActionable(t *testing.T) {
	// When the registry resolves a plugin but VLLM_PLUGINS_HOST_DIR is
	// empty, the launch can't proceed — bind-mount source would be
	// blank. The router must fail loudly so the operator knows to set
	// the env, not silently launch with a missing plugin file and
	// crash later in vLLM with "Unknown tool parser".
	r := loadVLLMPluginRegistry(writeRegistry(t, `{
		"plugins": {"deepseek_string": {"kind": "tool", "file": "x.py"}}
	}`), "")
	a := newPluginArbiter(r)

	lc := &launchConfig{ToolParser: "deepseek_string"}
	_, err := a.resolvePluginLaunch("vllm", lc)
	if err == nil {
		t.Fatal("expected error when host dir is empty")
	}
	if !strings.Contains(err.Error(), "VLLM_PLUGINS_HOST_DIR") {
		t.Errorf("error must mention VLLM_PLUGINS_HOST_DIR: %v", err)
	}
}

func TestResolvePluginLaunch_KindMismatchIsActionable(t *testing.T) {
	// A registry entry under kind="reasoning" used as a tool parser is
	// almost certainly a config error — fail loudly rather than wire the
	// wrong flag.
	r := loadVLLMPluginRegistry(writeRegistry(t, `{
		"plugins": {"weird": {"kind": "reasoning", "file": "weird.py"}}
	}`), "/host")
	a := newPluginArbiter(r)

	lc := &launchConfig{ToolParser: "weird"}
	_, err := a.resolvePluginLaunch("vllm", lc)
	if err == nil {
		t.Fatal("expected error on kind mismatch")
	}
	if !strings.Contains(err.Error(), "kind=") {
		t.Errorf("error must mention kind: %v", err)
	}
}
