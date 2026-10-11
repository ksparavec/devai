package main

// The machine-made recovery file (plan I25) and the re-read at each launch.
//
// deploy/recovery-flags.generated.json holds groups a program added
// (scripts/recipe-apply.py; Phase 18's auto-fix later). For a launch on
// backend B the router appends the curated entry's flags when it applies to
// B, then every generated group whose `backends` names B or B's engine, in
// file order. Both files are re-read at each launch when either changed, so
// a file replaced with a new inode (the writers' rename, git checkout's
// unlink and create) is seen without a restart. A file that fails to parse
// keeps what was loaded, and so does a curated file that disappears.

import (
	"bytes"
	"encoding/json"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
)

// writeByRename writes body to dir/name the way the writers do: a temporary
// file in the same directory, then a rename over the target.
func writeByRename(t *testing.T, dir, name, body string) {
	t.Helper()
	tmp, err := os.CreateTemp(dir, "."+name+".tmp-*")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := tmp.WriteString(body); err != nil {
		t.Fatal(err)
	}
	if err := tmp.Close(); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(tmp.Name(), filepath.Join(dir, name)); err != nil {
		t.Fatal(err)
	}
}

const curatedFixture = `{"models":{
  "M": {"backends": ["vllm"], "engine_flags": ["--language-model-only"], "engine_env": {"A": "curated"}},
  "Everywhere": {"engine_flags": ["--trust-remote-code"]}
}}`

// newFixtureFiles returns a recoveryFiles over a temp dir holding the
// curated fixture and, when generated != "", a generated file.
func newFixtureFiles(t *testing.T, generated string) (*recoveryFiles, string) {
	t.Helper()
	dir := t.TempDir()
	writeByRename(t, dir, "recovery-flags.json", curatedFixture)
	if generated != "" {
		writeByRename(t, dir, "recovery-flags.generated.json", generated)
	}
	cur := filepath.Join(dir, "recovery-flags.json")
	return newRecoveryFiles(cur, generatedRecoveryPath(cur)), dir
}

func lookupFlags(t *testing.T, f *recoveryFiles, backend, model string) []string {
	t.Helper()
	e, _ := f.Lookup(backend, model)
	return e.Flags
}

func TestGeneratedRecoveryPath_IsBesideTheCuratedFile(t *testing.T) {
	if got := generatedRecoveryPath("/etc/devai/deploy/recovery-flags.json"); got != "/etc/devai/deploy/recovery-flags.generated.json" {
		t.Errorf("generatedRecoveryPath = %q", got)
	}
}

func TestGeneratedGroup_AppliesOnlyToItsBackends(t *testing.T) {
	f, _ := newFixtureFiles(t, `{"version": 1, "models": {
	  "G": [
	    {"backends": ["vllm"], "engine_flags": ["--enforce-eager"], "source": "recipe", "reason": "r", "added": "d"},
	    {"backends": ["vllm-devai"], "engine_flags": ["--max-num-batched-tokens", "2048"], "source": "recipe"},
	    {"backends": ["sglang"], "engine_flags": ["--disable-radix-cache"], "source": "auto-fix"}
	  ]
	}}`)
	cases := []struct {
		backend string
		want    []string
	}{
		// A ["vllm"] group reaches a vllm-devai launch (same engine) ...
		{"vllm-devai", []string{"--enforce-eager", "--max-num-batched-tokens", "2048"}},
		// ... but a ["vllm-devai"] group does not reach stock vllm.
		{"vllm", []string{"--enforce-eager"}},
		{"sglang", []string{"--disable-radix-cache"}},
		{"ollama", nil},
	}
	for _, c := range cases {
		e, ok := f.Lookup(c.backend, "G")
		if ok != (c.want != nil) || !reflect.DeepEqual(e.Flags, c.want) {
			t.Errorf("Lookup(%s, G) = (%v, %v), want (%v, %v)", c.backend, e.Flags, ok, c.want, c.want != nil)
		}
	}
}

func TestGeneratedGroups_FollowTheCuratedEntryInFileOrder(t *testing.T) {
	f, _ := newFixtureFiles(t, `{"version": 1, "models": {
	  "M": [
	    {"backends": ["vllm"], "engine_flags": ["--max-num-seqs", "8"], "engine_env": {"A": "first", "B": "1"}},
	    {"backends": ["vllm"], "engine_flags": ["--max-num-seqs", "4"], "engine_env": {"A": "second"}}
	  ]
	}}`)
	e, ok := f.Lookup("vllm", "M")
	if !ok {
		t.Fatal("no match")
	}
	want := []string{"--language-model-only", "--max-num-seqs", "8", "--max-num-seqs", "4"}
	if !reflect.DeepEqual(e.Flags, want) {
		t.Errorf("flags = %v, want %v (curated first, then groups in file order; the last flag wins)", e.Flags, want)
	}
	if !reflect.DeepEqual(e.Env, map[string]string{"A": "second", "B": "1"}) {
		t.Errorf("env = %v: later groups must win over earlier ones and over the curated entry", e.Env)
	}
	// The curated entry still does not reach SGLang, and neither do the
	// vllm-scoped groups.
	if _, ok := f.Lookup("sglang", "M"); ok {
		t.Error("nothing for M is scoped to sglang")
	}
}

func TestGeneratedGroup_RemovingItRevertsExactlyItsFlags(t *testing.T) {
	two := `{"version": 1, "models": {"M": [
	  {"backends": ["vllm"], "engine_flags": ["--enforce-eager"], "source": "recipe"},
	  {"backends": ["vllm"], "engine_flags": ["--max-num-batched-tokens", "4096"], "engine_env": {"X": "1"}, "source": "auto-fix"}
	]}}`
	f, dir := newFixtureFiles(t, two)
	before := lookupFlags(t, f, "vllm", "M")
	if !reflect.DeepEqual(before, []string{"--language-model-only", "--enforce-eager", "--max-num-batched-tokens", "4096"}) {
		t.Fatalf("before = %v", before)
	}
	writeByRename(t, dir, "recovery-flags.generated.json", `{"version": 1, "models": {"M": [
	  {"backends": ["vllm"], "engine_flags": ["--enforce-eager"], "source": "recipe"}
	]}}`)
	e, _ := f.Lookup("vllm", "M")
	if !reflect.DeepEqual(e.Flags, []string{"--language-model-only", "--enforce-eager"}) {
		t.Errorf("after removing the second group: flags = %v", e.Flags)
	}
	if _, ok := e.Env["X"]; ok || e.Env["A"] != "curated" {
		t.Errorf("after removing the second group: env = %v", e.Env)
	}
	// Removing the whole file reverts every generated flag.
	if err := os.Remove(filepath.Join(dir, "recovery-flags.generated.json")); err != nil {
		t.Fatal(err)
	}
	if got := lookupFlags(t, f, "vllm", "M"); !reflect.DeepEqual(got, []string{"--language-model-only"}) {
		t.Errorf("after removing the file: flags = %v, want the curated entry alone", got)
	}
}

func TestRecoveryFiles_RenameReplacedFileIsSeenAtTheNextLaunch(t *testing.T) {
	f, dir := newFixtureFiles(t, `{"version": 1, "models": {}}`)
	if _, ok := f.Lookup("vllm", "New"); ok {
		t.Fatal("New has no entry yet")
	}
	writeByRename(t, dir, "recovery-flags.generated.json",
		`{"version": 1, "models": {"New": [{"backends": ["vllm"], "engine_flags": ["--enforce-eager"]}]}}`)
	if got := lookupFlags(t, f, "vllm", "New"); !reflect.DeepEqual(got, []string{"--enforce-eager"}) {
		t.Errorf("generated file replaced by rename not picked up: %v", got)
	}
	// The curated file is re-read the same way.
	writeByRename(t, dir, "recovery-flags.json", `{"models": {"New": {"engine_flags": ["--curated"]}}}`)
	if got := lookupFlags(t, f, "vllm", "New"); !reflect.DeepEqual(got, []string{"--curated", "--enforce-eager"}) {
		t.Errorf("curated file replaced by rename not picked up: %v", got)
	}
}

func TestRecoveryFiles_MissingGeneratedFileReadsAsEmpty(t *testing.T) {
	f, _ := newFixtureFiles(t, "")
	e, ok := f.Lookup("vllm", "M")
	if !ok || !reflect.DeepEqual(e.Flags, []string{"--language-model-only"}) {
		t.Errorf("curated entry alone expected, got (%v, %v)", e.Flags, ok)
	}
	if got := lookupFlags(t, f, "sglang", "Everywhere"); !reflect.DeepEqual(got, []string{"--trust-remote-code"}) {
		t.Errorf("unscoped curated entry: %v", got)
	}
}

func TestRecoveryFiles_MalformedFileKeepsThePreviousRegistry(t *testing.T) {
	good := `{"version": 1, "models": {"M": [{"backends": ["vllm"], "engine_flags": ["--enforce-eager"]}]}}`
	f, dir := newFixtureFiles(t, good)
	want := []string{"--language-model-only", "--enforce-eager"}
	if got := lookupFlags(t, f, "vllm", "M"); !reflect.DeepEqual(got, want) {
		t.Fatalf("before: %v", got)
	}
	for _, bad := range []string{
		`{"version": 1, "models": {`,   // truncated
		`{"version": 2, "models": {}}`, // a format this reader does not know
		`{"models": {}}`,               // no version
	} {
		writeByRename(t, dir, "recovery-flags.generated.json", bad)
		if got := lookupFlags(t, f, "vllm", "M"); !reflect.DeepEqual(got, want) {
			t.Errorf("generated %q: flags = %v, want the previous registry %v", bad, got, want)
		}
	}
	writeByRename(t, dir, "recovery-flags.json", `{"models": `)
	if got := lookupFlags(t, f, "vllm", "M"); !reflect.DeepEqual(got, want) {
		t.Errorf("truncated curated file: flags = %v, want the previous registry %v", got, want)
	}
	// A good file afterwards is picked up again.
	writeByRename(t, dir, "recovery-flags.generated.json", `{"version": 1, "models": {}}`)
	writeByRename(t, dir, "recovery-flags.json", curatedFixture)
	if got := lookupFlags(t, f, "vllm", "M"); !reflect.DeepEqual(got, []string{"--language-model-only"}) {
		t.Errorf("recovery after the files were fixed: %v", got)
	}
}

func TestGeneratedGroup_WithoutBackendsIsSkipped(t *testing.T) {
	// `backends` is required: there is no "every backend" group. A group
	// without one (or with a non-list) is dropped on its own; its siblings
	// and the other models stay.
	f, _ := newFixtureFiles(t, `{"version": 1, "models": {
	  "M": [
	    {"engine_flags": ["--absent"]},
	    {"backends": "vllm", "engine_flags": ["--not-a-list"]},
	    {"backends": ["vllm"], "engine_flags": ["--kept"], "image": "x/y:z"}
	  ],
	  "Broken": "not a list",
	  "Other": [{"backends": ["vllm"], "engine_flags": ["--other"]}]
	}}`)
	if got := lookupFlags(t, f, "vllm", "M"); !reflect.DeepEqual(got, []string{"--language-model-only", "--kept"}) {
		t.Errorf("M: %v", got)
	}
	if e, _ := f.Lookup("vllm", "M"); e.Image != "" {
		t.Errorf("a generated group may not set an image, got %q", e.Image)
	}
	if got := lookupFlags(t, f, "vllm", "Other"); !reflect.DeepEqual(got, []string{"--other"}) {
		t.Errorf("Other: %v", got)
	}
}

// The argv golden corpus renders launchConfig values directly, so it cannot
// move; what must hold is that an EMPTY generated file adds nothing to any
// curated lookup -- the repository ships the file empty.
func TestRecoveryFiles_EmptyGeneratedFileAddsNothing(t *testing.T) {
	curated := filepath.Join("..", "deploy", "recovery-flags.json")
	alone := loadRecoveryRegistry(curated)
	dir := t.TempDir()
	writeByRename(t, dir, "recovery-flags.generated.json", `{"version": 1, "models": {}}`)
	f := newRecoveryFiles(curated, filepath.Join(dir, "recovery-flags.generated.json"))
	if len(alone.Models) == 0 {
		t.Fatal("the repository's curated file has entries; none were read")
	}
	for model := range alone.Models {
		for _, backend := range []string{"ollama", "vllm", "vllm-devai", "sglang"} {
			a, aok := alone.Lookup(backend, model)
			b, bok := f.Lookup(backend, model)
			if aok != bok || !reflect.DeepEqual(a.Flags, b.Flags) || !reflect.DeepEqual(a.Env, b.Env) || a.Image != b.Image {
				t.Errorf("%s on %s: curated alone (%v %v %v) != with an empty generated file (%v %v %v)",
					model, backend, a.Flags, a.Env, aok, b.Flags, b.Env, bok)
			}
		}
	}
}

// The file the repository ships must parse, with I25's shape.
func TestRecoveryFiles_ShippedGeneratedFileParses(t *testing.T) {
	if _, err := readGeneratedRecovery(filepath.Join("..", "deploy", "recovery-flags.generated.json")); err != nil {
		t.Fatalf("deploy/recovery-flags.generated.json: %v", err)
	}
}

func TestRecoveryFiles_NilReaderMatchesNothing(t *testing.T) {
	var f *recoveryFiles
	if _, ok := f.Lookup("vllm", "M"); ok {
		t.Error("nil reader must not match")
	}
}

// Launches on different backends may look up concurrently while a writer
// replaces the file.
func TestRecoveryFiles_ConcurrentLookupsWhileReplacing(t *testing.T) {
	f, dir := newFixtureFiles(t, `{"version": 1, "models": {}}`)
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 50; j++ {
				f.Lookup("vllm", "M")
			}
		}()
	}
	for j := 0; j < 20; j++ {
		writeByRename(t, dir, "recovery-flags.generated.json",
			`{"version": 1, "models": {"M": [{"backends": ["vllm"], "engine_flags": ["--enforce-eager"]}]}}`)
	}
	wg.Wait()
	if got := lookupFlags(t, f, "vllm", "M"); !reflect.DeepEqual(got, []string{"--language-model-only", "--enforce-eager"}) {
		t.Errorf("final: %v", got)
	}
}

// Both readers -- this one and scripts/_probe_hf_common.py -- must make the
// same call on every malformed input, or a probe and a serve-time launch get
// different flags. tests/python/test_recovery_flags_generated.py loads the
// same files and asserts the same expected.json.
type fixtureLookup struct {
	Flags []string          `json:"flags"`
	Env   map[string]string `json:"env"`
	Image string            `json:"image"`
}

type recoveryFixture struct {
	Lookups       map[string]fixtureLookup `json:"lookups"`
	KeepsPrevious map[string][]string      `json:"keeps_previous"`
}

func fixtureDir() string { return filepath.Join("..", "tests", "fixtures", "recovery-flags") }

func loadRecoveryFixture(t *testing.T) recoveryFixture {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(fixtureDir(), "expected.json"))
	if err != nil {
		t.Fatal(err)
	}
	var fx recoveryFixture
	if err := json.Unmarshal(data, &fx); err != nil {
		t.Fatal(err)
	}
	return fx
}

func fixtureLookupOf(f *recoveryFiles, key string) fixtureLookup {
	backend, model, _ := strings.Cut(key, "/")
	e, _ := f.Lookup(backend, model)
	got := fixtureLookup{Flags: e.Flags, Env: e.Env, Image: e.Image}
	if got.Flags == nil {
		got.Flags = []string{}
	}
	if got.Env == nil {
		got.Env = map[string]string{}
	}
	return got
}

func TestRecoveryFiles_SharedFixtureMatchesThePythonReader(t *testing.T) {
	fx := loadRecoveryFixture(t)
	f := newRecoveryFiles(filepath.Join(fixtureDir(), "curated.json"), filepath.Join(fixtureDir(), "generated.json"))
	for key, want := range fx.Lookups {
		if got := fixtureLookupOf(f, key); !reflect.DeepEqual(got, want) {
			t.Errorf("%s: got %+v, want %+v", key, got, want)
		}
	}
}

// Every file in broken/ fails as a whole: replacing either recovery file with
// it (a new inode) leaves every lookup as it was, and a good file afterwards
// is read again.
func TestRecoveryFiles_SharedFixtureBrokenFilesKeepThePreviousRegistry(t *testing.T) {
	fx := loadRecoveryFixture(t)
	read := func(name string) string {
		data, err := os.ReadFile(filepath.Join(fixtureDir(), name))
		if err != nil {
			t.Fatal(err)
		}
		return string(data)
	}
	dir := t.TempDir()
	writeByRename(t, dir, "recovery-flags.json", read("curated.json"))
	writeByRename(t, dir, "recovery-flags.generated.json", read("generated.json"))
	f := newRecoveryFiles(filepath.Join(dir, "recovery-flags.json"), filepath.Join(dir, "recovery-flags.generated.json"))
	check := func(label string) {
		for key, want := range fx.Lookups {
			if got := fixtureLookupOf(f, key); !reflect.DeepEqual(got, want) {
				t.Errorf("%s: %s: got %+v, want the previous %+v", label, key, got, want)
			}
		}
	}
	targets := map[string]string{"curated": "recovery-flags.json", "generated": "recovery-flags.generated.json"}
	for which, files := range fx.KeepsPrevious {
		if len(files) == 0 {
			t.Fatalf("no broken %s files listed", which)
		}
		for _, name := range files {
			writeByRename(t, dir, targets[which], read(filepath.Join("broken", name)))
			check(name)
		}
		writeByRename(t, dir, targets[which], read(which+".json"))
		check("restored " + which)
	}
}

// git checkout replaces a file by unlink and create, so a launch can see the
// curated file missing: it keeps the previous entries until the file is
// back. A removed generated file reverts what it added (I25).
func TestRecoveryFiles_ADisappearedCuratedFileKeepsItsEntries(t *testing.T) {
	f, dir := newFixtureFiles(t, `{"version": 1, "models": {"M": [{"backends": ["vllm"], "engine_flags": ["--enforce-eager"]}]}}`)
	if err := os.Remove(filepath.Join(dir, "recovery-flags.json")); err != nil {
		t.Fatal(err)
	}
	if got := lookupFlags(t, f, "vllm", "M"); !reflect.DeepEqual(got, []string{"--language-model-only", "--enforce-eager"}) {
		t.Errorf("curated file gone: flags = %v, want the previous curated entry kept", got)
	}
	writeByRename(t, dir, "recovery-flags.json", `{"models": {"M": {"engine_flags": ["--back"]}}}`)
	if got := lookupFlags(t, f, "vllm", "M"); !reflect.DeepEqual(got, []string{"--back", "--enforce-eager"}) {
		t.Errorf("curated file back: flags = %v", got)
	}
	if err := os.Remove(filepath.Join(dir, "recovery-flags.generated.json")); err != nil {
		t.Fatal(err)
	}
	if got := lookupFlags(t, f, "vllm", "M"); !reflect.DeepEqual(got, []string{"--back"}) {
		t.Errorf("generated file removed: flags = %v, want its groups reverted", got)
	}
}

func TestRecoveryFiles_ACuratedFileMissingAtBootIsEmpty(t *testing.T) {
	dir := t.TempDir()
	f := newRecoveryFiles(filepath.Join(dir, "recovery-flags.json"), filepath.Join(dir, "recovery-flags.generated.json"))
	if _, ok := f.Lookup("vllm", "M"); ok {
		t.Error("nothing to match with both files missing")
	}
	writeByRename(t, dir, "recovery-flags.json", `{"models": {"M": {"engine_flags": ["--late"]}}}`)
	if got := lookupFlags(t, f, "vllm", "M"); !reflect.DeepEqual(got, []string{"--late"}) {
		t.Errorf("a curated file that appears later is read: %v", got)
	}
}

// The merged result never hands out the registry's own slices.
func TestRecoveryRegistry_LookupDoesNotAliasTheRegistry(t *testing.T) {
	f, _ := newFixtureFiles(t, `{"version": 1, "models": {"M": [{"backends": ["vllm"], "engine_flags": ["--enforce-eager"], "engine_env": {"X": "1"}}]}}`)
	e, _ := f.Lookup("vllm", "M")
	if e.Backends != nil {
		t.Errorf("merged result carries Backends %v", *e.Backends)
	}
	e.Flags[0] = "--mutated"
	e.Env["A"] = "mutated"
	again, _ := f.Lookup("vllm", "M")
	if !reflect.DeepEqual(again.Flags, []string{"--language-model-only", "--enforce-eager"}) ||
		again.Env["A"] != "curated" {
		t.Errorf("a caller's edit reached the registry: %v %v", again.Flags, again.Env)
	}
}

// containerRecreate hands Lookup's result to the builders unchanged: the
// generated flags end the argv, and generated env, like curated env, wins
// over the backend's DynamicEnv.
func TestRecoveryFiles_GeneratedFlagsAndEnvReachTheContainerSpec(t *testing.T) {
	f, _ := newFixtureFiles(t, `{"version": 1, "models": {
	  "M": [{"backends": ["vllm"], "engine_flags": ["--enforce-eager"], "engine_env": {"VLLM_X": "1"}}],
	  "mixed:latest": [{"backends": ["ollama"], "engine_env": {"OLLAMA_KV_CACHE_TYPE": "f16"}}]
	}}`)
	rec, ok := f.Lookup("vllm", "M")
	if !ok {
		t.Fatal("no match")
	}
	lc := launchConfig{MemFraction: 0.9, MaxContext: 32768, RecoveryFlags: rec.Flags}
	cfg := backendConfig{Name: "vllm", ContainerName: "devai-vllm", Image: "img",
		ModelsDir: "/m", MountDest: "/models", Network: "devai-net", Entrypoint: vllmEntrypoint}
	spec := buildContainerSpec(cfg, "M", lc, nil, rec.Env)
	argv := spec["entrypoint"].([]string)
	if tail := argv[len(argv)-2:]; !reflect.DeepEqual(tail, []string{"--language-model-only", "--enforce-eager"}) {
		t.Errorf("argv must end with the recovery flags, curated then generated: %v", argv)
	}
	if env, _ := spec["env"].(map[string]string); env["VLLM_X"] != "1" || env["A"] != "curated" {
		t.Errorf("recovery env missing from the spec: %v", spec["env"])
	}

	orec, _ := f.Lookup("ollama", "mixed:latest")
	ocfg := backendConfig{Name: "ollama", ContainerName: "devai-ollama", Image: "img",
		ModelsDir: "/o", MountDest: "/root/.ollama", Network: "devai-net",
		Entrypoint: ollamaEntrypoint, DynamicEnv: ollamaDynamicEnv}
	ospec := buildContainerSpec(ocfg, "mixed:latest", launchConfig{MaxContext: 131072, KVCacheType: "q8_0"}, nil, orec.Env)
	if env, _ := ospec["env"].(map[string]string); env["OLLAMA_KV_CACHE_TYPE"] != "f16" {
		t.Errorf("generated env must win over DynamicEnv: %v", ospec["env"])
	}
}

// containerRecreate end to end, with the repository's real curated file and
// the shipped (empty) generated file: for every curated model on each
// backend it applies to, the create body the router sends carries the
// entry's image, ends its argv with the entry's flags and holds the entry's
// env. Expectations are decoded straight from the raw JSON (map[string]any),
// not through readCuratedRecovery, so a reader bug cannot hide itself.
func recreateHarness(t *testing.T, generatedPath string) (*arbiter, map[string]*backendState, func() [][]byte) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	t.Cleanup(srv.Close)
	mk := func(name, image string, entry func(string, launchConfig) []string) *backendState {
		bs := testBackend(name, srv)
		bs.config.ContainerName = "devai-" + name
		bs.config.Image = image
		bs.config.ModelsDir = "/var/cache/devai/vllm"
		bs.config.Network = "devai-net"
		bs.config.Entrypoint = entry
		if name == "ollama" {
			bs.config.MountDest, bs.config.DynamicEnv = "/root/.ollama", ollamaDynamicEnv
		}
		return bs
	}
	backends := map[string]*backendState{
		"ollama":     mk("ollama", "img/ollama", ollamaEntrypoint),
		"vllm":       mk("vllm", "img/vllm-stock", vllmEntrypoint),
		"vllm-devai": mk("vllm-devai", "img/vllm-devai", vllmEntrypoint),
		"sglang":     mk("sglang", "img/sglang", sglangEntrypoint),
	}
	a := testArbiter(backends["ollama"], backends["vllm"], backends["vllm-devai"], backends["sglang"])
	client, _, recorded := podmanStubRecording(t)
	a.podmanClient = client
	a.recoveryFiles = newRecoveryFiles(filepath.Join("..", "deploy", "recovery-flags.json"), generatedPath)
	return a, backends, recorded
}

type createBody struct {
	Image      string            `json:"image"`
	Entrypoint []string          `json:"entrypoint"`
	Env        map[string]string `json:"env"`
}

func lastCreate(t *testing.T, recorded func() [][]byte) createBody {
	t.Helper()
	bodies := recorded()
	if len(bodies) == 0 {
		t.Fatal("no container was created")
	}
	var cb createBody
	if err := json.Unmarshal(bodies[len(bodies)-1], &cb); err != nil {
		t.Fatal(err)
	}
	return cb
}

func TestContainerRecreate_RealCuratedFileReachesTheCreateBody(t *testing.T) {
	dir := t.TempDir()
	shipped, err := os.ReadFile(filepath.Join("..", "deploy", "recovery-flags.generated.json"))
	if err != nil {
		t.Fatal(err)
	}
	writeByRename(t, dir, "recovery-flags.generated.json", string(shipped))
	a, backends, recorded := recreateHarness(t, filepath.Join(dir, "recovery-flags.generated.json"))

	raw, err := os.ReadFile(filepath.Join("..", "deploy", "recovery-flags.json"))
	if err != nil {
		t.Fatal(err)
	}
	var doc map[string]any
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	models, _ := doc["models"].(map[string]any)
	if len(models) == 0 {
		t.Fatal("the repository's curated file has no entries")
	}
	applies := func(entry map[string]any, backend string) bool {
		list, present := entry["backends"]
		if !present || list == nil {
			return true
		}
		names, ok := list.([]any)
		if !ok {
			return true // C2: a non-list is treated as absent
		}
		for _, n := range names {
			if n == backend || (backend == "vllm-devai" && n == "vllm") {
				return true
			}
		}
		return false
	}
	checked := 0
	for model, v := range models {
		entry, _ := v.(map[string]any)
		for _, backend := range []string{"vllm", "vllm-devai", "sglang"} {
			bs := backends[backend]
			a.mu.Lock()
			_, err := a.containerRecreate(bs, model, 32768, nil)
			a.mu.Unlock()
			if err != nil {
				t.Fatalf("%s on %s: %v", model, backend, err)
			}
			cb := lastCreate(t, recorded)
			if !applies(entry, backend) {
				if cb.Image != bs.config.Image {
					t.Errorf("%s on %s (not scoped here): image %q", model, backend, cb.Image)
				}
				envs, _ := entry["engine_env"].(map[string]any)
				for k := range envs {
					if _, leaked := cb.Env[k]; leaked {
						t.Errorf("%s on %s (not scoped here): env %s leaked", model, backend, k)
					}
				}
				continue
			}
			checked++
			wantImage := bs.config.Image
			if img, _ := entry["image"].(string); img != "" {
				wantImage = img
			}
			if cb.Image != wantImage {
				t.Errorf("%s on %s: image %q, want %q", model, backend, cb.Image, wantImage)
			}
			var wantFlags []string
			flags, _ := entry["engine_flags"].([]any)
			for _, f := range flags {
				wantFlags = append(wantFlags, f.(string))
			}
			if n := len(wantFlags); len(cb.Entrypoint) < n || !reflect.DeepEqual(cb.Entrypoint[len(cb.Entrypoint)-n:], wantFlags) {
				t.Errorf("%s on %s: argv %v does not end with %v", model, backend, cb.Entrypoint, wantFlags)
			}
			if envs, ok := entry["engine_env"].(map[string]any); ok {
				for k, v := range envs {
					if cb.Env[k] != v.(string) {
						t.Errorf("%s on %s: env %s=%q, want %q", model, backend, k, cb.Env[k], v)
					}
				}
			}
		}
	}
	if checked < len(models) {
		t.Errorf("only %d (model, backend) launches carried a curated entry for %d models", checked, len(models))
	}
}

func TestContainerRecreate_AGeneratedGroupFollowsTheCuratedFlags(t *testing.T) {
	dir := t.TempDir()
	writeByRename(t, dir, "recovery-flags.generated.json", `{"version": 1, "models": {
	  "NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4": [{"backends": ["vllm-devai"], "engine_flags": ["--async-scheduling"],
	    "engine_env": {"VLLM_FLASHINFER_MOE_BACKEND": "latency"}, "source": "recipe"}]}}`)
	a, backends, recorded := recreateHarness(t, filepath.Join(dir, "recovery-flags.generated.json"))
	var logBuf bytes.Buffer
	log.SetOutput(&logBuf)
	defer log.SetOutput(os.Stderr)
	a.mu.Lock()
	_, err := a.containerRecreate(backends["vllm-devai"], "NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4", 32768, nil)
	a.mu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	cb := lastCreate(t, recorded)
	tail := cb.Entrypoint[len(cb.Entrypoint)-5:]
	if want := []string{"--max-num-seqs", "8", "--trust-remote-code", "--enforce-eager", "--async-scheduling"}; !reflect.DeepEqual(tail, want) {
		t.Errorf("argv tail %v, want curated flags then the group's %v", tail, want)
	}
	if cb.Env["VLLM_FLASHINFER_MOE_BACKEND"] != "latency" || cb.Env["VLLM_USE_FLASHINFER_MOE_FP4"] != "1" {
		t.Errorf("env %v: the group's value must win, curated keys must stay", cb.Env)
	}
	if !strings.Contains(logBuf.String(), `--async-scheduling`) ||
		!strings.Contains(logBuf.String(), "curated entry: true, generated groups: 1") {
		t.Errorf("the launch log must name the flags and their sources:\n%s", logBuf.String())
	}
	// On stock vllm the vllm-devai group does not apply.
	a.mu.Lock()
	_, _ = a.containerRecreate(backends["vllm"], "NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4", 32768, nil)
	a.mu.Unlock()
	if cb := lastCreate(t, recorded); cb.Entrypoint[len(cb.Entrypoint)-1] != "--enforce-eager" {
		t.Errorf("stock vllm got the vllm-devai group: %v", cb.Entrypoint)
	}
}
