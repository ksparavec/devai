package main

// Per-model recovery flags + env registry — Go side.
//
// Reads two files, the same two scripts/_probe_hf_common.py reads, so that
// probe launches and router-driven recreates apply the same flags/env to a
// borderline checkpoint — without it, a probe would OOM while the router's
// serve-time launch would succeed (or vice versa), and the probe cache would
// never get a fits=true cell for the model:
//
//   - deploy/recovery-flags.json -- hand-curated, one entry per model
//     (contract C2 below). Never written by a program.
//   - deploy/recovery-flags.generated.json -- machine-made (plan I25):
//     `{"version": 1, "models": {"<name>": [group, ...]}}`, each group
//     `{backends, engine_flags, engine_env, source, reason, added}` with
//     `backends` REQUIRED and no `image`. Written only by
//     scripts/recipe-apply.py today (Phase 18's auto-fix later), whole file
//     to a temporary name and renamed. A missing file reads as empty.
//
// For a launch on backend B the flags are the curated entry's (when it
// applies to B), then every generated group whose `backends` names B or
// B's engine, in file order; env maps merge in the same order. The engine
// still takes the LAST occurrence of a repeated flag, so a later group wins.
//
// Both files are re-read at each launch when either one's device, inode,
// mtime or size changed (recoveryFiles). A file that fails to parse keeps
// the previously loaded content and logs the error; so does a curated file
// that disappears after a good load (git checkout replaces a file by unlink
// and create, so a launch can land between the two). The router mounts the
// repository's deploy/ DIRECTORY (docker-compose.yaml), not single files: a
// single-file bind mount pins the old inode and would go on serving the old
// content after the file is replaced with a new inode.
//
// Decoding rules, the same in scripts/_probe_hf_common.py and pinned by the
// shared fixture in tests/fixtures/recovery-flags/ that both suites load:
//
//   - keys are exact and case-sensitive (objects are decoded into
//     map[string]json.RawMessage, never into tagged structs, whose key
//     matching is case-insensitive);
//   - a top level that is not a JSON object (`null` included), or a
//     `models` that is neither absent/null nor an object, fails the file;
//   - an entry or group with a malformed field is skipped WHOLE, never
//     applied in part: `engine_flags` must be absent, null or a list of
//     strings (no null or non-string element), `engine_env` absent, null
//     or an object of strings, a curated `image` absent, null or a string;
//   - a curated `backends` follows contract C2 (absent/null: everywhere;
//     list: those backends; non-list: warn, treat as absent) except that a
//     list with a non-string element skips the entry; a generated group's
//     `backends` must be a list of strings, else the group is skipped;
//   - a generated `image` is ignored with a warning; `source`, `reason` and
//     `added` are informational and never read.
//
// Names absent from both files pass through unchanged: no extra flags, no
// extra env.

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"syscall"
)

// recoveryGeneratedVersion is the only `version` the generated file may carry.
const recoveryGeneratedVersion = 1

// recoveryEntry is the per-model flag set. Any field may be empty.
type recoveryEntry struct {
	Flags []string          `json:"engine_flags"`
	Env   map[string]string `json:"engine_env"`
	// Backends optionally restricts the entry to a subset of backends.
	// It is a POINTER so absent and empty stay distinguishable -- they
	// mean opposite things (see the contract below).
	//
	// Cross-unit contract C2, implemented identically here and in
	// scripts/_probe_hf_common.py so probe launches and serve-time
	// recreates agree on which entry applies:
	//
	//	key ABSENT           -> applies to ALL backends (backward
	//	                        compatible with pre-`backends` entries)
	//	key present, []       -> applies to NO backend (an operator
	//	                        writing [] means "disable this entry")
	//	key present, list     -> applies only to the named backends
	//	key present, non-list -> warn naming the model, treat as ABSENT
	//
	// Without the filter, vLLM-only rescue flags (--language-model-only,
	// --quantization modelopt, --max-num-seqs, VLLM_* env) were appended
	// verbatim to SGLang launches, where those flags do not exist.
	Backends *[]string `json:"backends"`
	// Image optionally overrides the vLLM container image for this model
	// only (falls back to $VLLM_IMAGE when empty). Needed when a model
	// requires a different engine build than the global default -- e.g.
	// DiffusionGemma needs the vLLM "gemma" bring-up image, which in turn
	// regresses Qwen NVFP4 loading, so it cannot be the global default.
	// Curated file only: a generated group has no image (plan I25, M8).
	Image string `json:"image"`
	// Set on a Lookup result only: whether the curated entry contributed,
	// and how many generated groups did (the launch log names both).
	FromCurated     bool `json:"-"`
	GeneratedGroups int  `json:"-"`
}

// recoveryGroup is one machine-made group of deploy/recovery-flags.generated.json
// (plan I25). Unlike a curated entry, `backends` is required: there is no
// "every backend" group.
type recoveryGroup struct {
	Backends []string
	Flags    []string
	Env      map[string]string
}

// recoveryRegistry resolves canonical model names to their recovery
// flags/env. An empty registry means "no per-model overrides" (files
// missing or empty).
type recoveryRegistry struct {
	// Models holds the hand-curated entries (deploy/recovery-flags.json).
	Models map[string]recoveryEntry `json:"models"`
	// Generated holds the machine-made groups per model, in file order
	// (deploy/recovery-flags.generated.json).
	Generated map[string][]recoveryGroup `json:"-"`
}

// ── strict decoding helpers ─────────────────────────────────────────────────

func isJSONNull(raw json.RawMessage) bool {
	t := bytes.TrimSpace(raw)
	return len(t) == 0 || string(t) == "null"
}

// rawObject decodes data as a JSON object with exact keys; false for
// anything else (`null`, a list, a scalar, invalid JSON).
func rawObject(data []byte) (map[string]json.RawMessage, bool) {
	t := bytes.TrimSpace(data)
	if len(t) == 0 || t[0] != '{' {
		return nil, false
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(t, &m); err != nil || m == nil {
		return nil, false
	}
	return m, true
}

// jsonString decodes one JSON string; false for null and every non-string.
func jsonString(raw json.RawMessage) (string, bool) {
	t := bytes.TrimSpace(raw)
	if len(t) == 0 || t[0] != '"' {
		return "", false
	}
	var s string
	if err := json.Unmarshal(t, &s); err != nil {
		return "", false
	}
	return s, true
}

// decodeStringList: absent or null -> (nil, true); a list whose every
// element is a string -> the list; anything else -> false.
func decodeStringList(raw json.RawMessage) ([]string, bool) {
	if isJSONNull(raw) {
		return nil, true
	}
	t := bytes.TrimSpace(raw)
	if t[0] != '[' {
		return nil, false
	}
	var items []json.RawMessage
	if err := json.Unmarshal(t, &items); err != nil {
		return nil, false
	}
	out := make([]string, 0, len(items))
	for _, it := range items {
		s, ok := jsonString(it)
		if !ok {
			return nil, false
		}
		out = append(out, s)
	}
	return out, true
}

// decodeStringMap: absent or null -> (nil, true); an object whose every
// value is a string -> the map; anything else -> false.
func decodeStringMap(raw json.RawMessage) (map[string]string, bool) {
	if isJSONNull(raw) {
		return nil, true
	}
	fields, ok := rawObject(raw)
	if !ok {
		return nil, false
	}
	out := make(map[string]string, len(fields))
	for k, v := range fields {
		s, ok := jsonString(v)
		if !ok {
			return nil, false
		}
		out[k] = s
	}
	return out, true
}

// decodeOptionalString: absent or null -> ("", true); a string -> it.
func decodeOptionalString(raw json.RawMessage) (string, bool) {
	if isJSONNull(raw) {
		return "", true
	}
	return jsonString(raw)
}

// modelsObject returns the `models` map of a top-level object: absent or
// null is empty, anything but an object is an error.
func modelsObject(top map[string]json.RawMessage) (map[string]json.RawMessage, bool) {
	raw := top["models"]
	if isJSONNull(raw) {
		return map[string]json.RawMessage{}, true
	}
	return rawObject(raw)
}

// ── curated file ────────────────────────────────────────────────────────────

// loadRecoveryRegistry reads the curated JSON registry alone. A missing
// file logs a warning and returns an empty (non-nil) registry — every model
// then launches without recovery flags, matching pre-registry behaviour.
// Parse errors log a warning and likewise return empty. The router itself
// goes through recoveryFiles, which also reads the generated file and
// re-reads both on change; this one-file, one-shot reader is what the tests
// of the curated file's own contract (C2) use.
func loadRecoveryRegistry(path string) *recoveryRegistry {
	models, err := readCuratedRecovery(path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			log.Printf("warning: recovery registry: %s not present — no per-model overrides", path)
		} else {
			log.Printf("warning: %v", err)
		}
		models = map[string]recoveryEntry{}
	}
	return &recoveryRegistry{Models: models}
}

// readCuratedRecovery reads deploy/recovery-flags.json. A missing file is
// an error wrapping fs.ErrNotExist (the caller decides what it means); so
// is an unreadable file and a top level that is not an object.
func readCuratedRecovery(path string) (map[string]recoveryEntry, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("recovery registry %s read failed: %w", path, err)
	}
	// The registry tolerates an `_comment` field at the top level (used
	// for human-readable notes); only `models` is consumed.
	//
	// Entries are decoded ONE AT A TIME. Decoding the whole file in a
	// single call made any malformed entry abort the parse and return an
	// EMPTY registry -- silently stripping the OOM-rescue flags from every
	// model at once, which is exactly the failure the registry exists to
	// prevent. A malformed entry is skipped on its own.
	top, ok := rawObject(data)
	if !ok {
		return nil, fmt.Errorf("recovery registry %s parse failed: not a JSON object", path)
	}
	raw, ok := modelsObject(top)
	if !ok {
		return nil, fmt.Errorf("recovery registry %s parse failed: `models` is not an object", path)
	}
	models := map[string]recoveryEntry{}
	for name, rawEntry := range raw {
		entry, ok := decodeRecoveryEntry(name, rawEntry)
		if !ok {
			continue
		}
		models[name] = entry
	}
	log.Printf("recovery registry: %s loaded (%d entries)", path, len(models))
	return models, nil
}

// decodeRecoveryEntry decodes one curated entry. A field of the wrong type
// skips the whole entry, logged with the model's name; dropping it never
// affects the other entries. A `backends` value that is not a JSON list is
// logged and treated as ABSENT -- the entry keeps its flags/env and applies
// everywhere, per contract C2.
func decodeRecoveryEntry(name string, data []byte) (recoveryEntry, bool) {
	fields, ok := rawObject(data)
	if !ok {
		log.Printf("warning: recovery registry: entry %q skipped (not an object)", name)
		return recoveryEntry{}, false
	}
	flags, ok := decodeStringList(fields["engine_flags"])
	if !ok {
		log.Printf("warning: recovery registry: entry %q skipped (`engine_flags` must be a list of strings)", name)
		return recoveryEntry{}, false
	}
	env, ok := decodeStringMap(fields["engine_env"])
	if !ok {
		log.Printf("warning: recovery registry: entry %q skipped (`engine_env` must be an object of strings)", name)
		return recoveryEntry{}, false
	}
	image, ok := decodeOptionalString(fields["image"])
	if !ok {
		log.Printf("warning: recovery registry: entry %q skipped (`image` must be a string)", name)
		return recoveryEntry{}, false
	}
	e := recoveryEntry{Flags: flags, Env: env, Image: image}
	raw := fields["backends"]
	// Absent, or an explicit null, both mean "no restriction".
	if isJSONNull(raw) {
		return e, true
	}
	if bytes.TrimSpace(raw)[0] != '[' {
		log.Printf("warning: recovery registry: entry %q has a non-list `backends` value (%s); "+
			"treating it as absent (entry applies to every backend)", name, raw)
		return e, true
	}
	backends, ok := decodeStringList(raw)
	if !ok {
		log.Printf("warning: recovery registry: entry %q skipped (`backends` must list strings)", name)
		return recoveryEntry{}, false
	}
	e.Backends = &backends
	return e, true
}

// ── generated file ──────────────────────────────────────────────────────────

// readGeneratedRecovery reads deploy/recovery-flags.generated.json. A
// missing file reads as empty (removing it reverts what it added, I25). A
// file that is unreadable, not a JSON object, of another `version` (it must
// be the integer 1), or whose `models` is not an object is an error (the
// caller keeps what it had). Within a file that parses, a malformed model
// or group is skipped on its own, with a warning.
func readGeneratedRecovery(path string) (map[string][]recoveryGroup, error) {
	out := map[string][]recoveryGroup{}
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			log.Printf("recovery registry: %s not present — no machine-made flags", path)
			return out, nil
		}
		return nil, fmt.Errorf("generated recovery flags %s read failed: %v", path, err)
	}
	top, ok := rawObject(data)
	if !ok {
		return nil, fmt.Errorf("generated recovery flags %s parse failed: not a JSON object", path)
	}
	var version int
	if raw := top["version"]; isJSONNull(raw) || json.Unmarshal(raw, &version) != nil ||
		version != recoveryGeneratedVersion {
		return nil, fmt.Errorf("generated recovery flags %s: `version` must be the integer %d",
			path, recoveryGeneratedVersion)
	}
	models, ok := modelsObject(top)
	if !ok {
		return nil, fmt.Errorf("generated recovery flags %s parse failed: `models` is not an object", path)
	}
	groups := 0
	for name, rawGroups := range models {
		var list []json.RawMessage
		if t := bytes.TrimSpace(rawGroups); len(t) == 0 || t[0] != '[' || json.Unmarshal(t, &list) != nil {
			log.Printf("warning: generated recovery flags: model %q skipped (not a list of groups)", name)
			continue
		}
		for i, rg := range list {
			g, ok := decodeRecoveryGroup(name, i, rg)
			if !ok {
				continue
			}
			out[name] = append(out[name], g)
			groups++
		}
	}
	log.Printf("recovery registry: %s loaded (%d groups for %d models)", path, groups, len(out))
	return out, nil
}

// decodeRecoveryGroup decodes one generated group under the rules in the
// file header: `backends` a list of strings (required), `engine_flags` and
// `engine_env` well-formed when present, else the whole group is skipped.
func decodeRecoveryGroup(name string, idx int, data []byte) (recoveryGroup, bool) {
	fields, ok := rawObject(data)
	if !ok {
		log.Printf("warning: generated recovery flags: %s group %d skipped (not an object)", name, idx)
		return recoveryGroup{}, false
	}
	backends, ok := decodeStringList(fields["backends"])
	if !ok || isJSONNull(fields["backends"]) {
		log.Printf("warning: generated recovery flags: %s group %d skipped (`backends` must be a list of strings)", name, idx)
		return recoveryGroup{}, false
	}
	flags, ok := decodeStringList(fields["engine_flags"])
	if !ok {
		log.Printf("warning: generated recovery flags: %s group %d skipped (`engine_flags` must be a list of strings)", name, idx)
		return recoveryGroup{}, false
	}
	env, ok := decodeStringMap(fields["engine_env"])
	if !ok {
		log.Printf("warning: generated recovery flags: %s group %d skipped (`engine_env` must be an object of strings)", name, idx)
		return recoveryGroup{}, false
	}
	if _, has := fields["image"]; has {
		log.Printf("warning: generated recovery flags: %s group %d has an `image`, which a generated group "+
			"may not carry; ignored", name, idx)
	}
	return recoveryGroup{Backends: backends, Flags: flags, Env: env}, true
}

// ── lookup ──────────────────────────────────────────────────────────────────

// backendListMatches is the shared scoping rule: a list names a backend by
// its own name or by its engine.
//
// vLLM CLI flags written for the stock `vllm` backend are engine flags and
// apply to the custom vLLM build too (`vllm-devai`). The reverse does not
// hold -- an entry scoped to `vllm-devai` names the only image that can
// serve that checkpoint.
func backendListMatches(backends []string, backendName string) bool {
	for _, b := range backends {
		if b == backendName || b == engineOf(backendName) {
			return true
		}
	}
	return false
}

// appliesTo reports whether the entry is valid for `backendName`, per
// contract C2: an ABSENT `backends` key (nil pointer) applies everywhere,
// while a PRESENT but empty list applies nowhere.
func (e recoveryEntry) appliesTo(backendName string) bool {
	if e.Backends == nil {
		return true
	}
	return backendListMatches(*e.Backends, backendName)
}

// mergeRecoveryEnv returns base with add's keys on top, as a new map; nil
// when both are empty.
func mergeRecoveryEnv(base, add map[string]string) map[string]string {
	if len(add) == 0 {
		return base
	}
	out := make(map[string]string, len(base)+len(add))
	for k, v := range base {
		out[k] = v
	}
	for k, v := range add {
		out[k] = v
	}
	return out
}

// Lookup returns the merged entry for (backendName, modelName) plus whether
// anything matched: the curated entry's flags and env when it applies, then
// every generated group scoped to the backend, in file order. A nil
// receiver and an empty modelName both yield (_, false), as does a model
// whose entry and groups all exclude backendName. The returned slices and
// maps are fresh copies; the registry's own are never handed out. The
// result's Backends is nil: it would describe the curated entry alone.
func (r *recoveryRegistry) Lookup(backendName, modelName string) (recoveryEntry, bool) {
	if r == nil || modelName == "" {
		return recoveryEntry{}, false
	}
	var out recoveryEntry
	if e, ok := r.Models[modelName]; ok {
		if e.appliesTo(backendName) {
			out.Flags = append(out.Flags, e.Flags...)
			out.Env = mergeRecoveryEnv(nil, e.Env)
			out.Image = e.Image
			out.FromCurated = true
		} else {
			log.Printf("recovery registry: entry for %s is scoped to %v -- not applied to %s",
				modelName, *e.Backends, backendName)
		}
	}
	for _, g := range r.Generated[modelName] {
		if !backendListMatches(g.Backends, backendName) {
			continue
		}
		out.Flags = append(out.Flags, g.Flags...)
		out.Env = mergeRecoveryEnv(out.Env, g.Env)
		out.GeneratedGroups++
	}
	if !out.FromCurated && out.GeneratedGroups == 0 {
		return recoveryEntry{}, false
	}
	return out, true
}

// describe renders a Lookup result for the launch log: the flags, the env
// as sorted KEY=VALUE pairs, the image, and where they came from.
func (e recoveryEntry) describe() string {
	env := make([]string, 0, len(e.Env))
	for k, v := range e.Env {
		env = append(env, k+"="+v)
	}
	sort.Strings(env)
	return fmt.Sprintf("flags=%q env=%q image=%q (curated entry: %v, generated groups: %d)",
		e.Flags, env, e.Image, e.FromCurated, e.GeneratedGroups)
}

// ── re-read on change ───────────────────────────────────────────────────────

// fileSig identifies one version of a file: device, inode, mtime and size.
// A new inode (a writer's rename, git checkout's unlink and create) or an
// in-place write changes it.
type fileSig struct {
	missing bool
	statErr bool
	dev     uint64
	ino     uint64
	mtimeNs int64
	size    int64
}

func statFileSig(path string) fileSig {
	fi, err := os.Stat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return fileSig{missing: true}
		}
		return fileSig{statErr: true}
	}
	sig := fileSig{mtimeNs: fi.ModTime().UnixNano(), size: fi.Size()}
	if st, ok := fi.Sys().(*syscall.Stat_t); ok {
		sig.dev = uint64(st.Dev)
		sig.ino = st.Ino
	}
	return sig
}

// recoveryFiles is the router's view of both recovery files. Lookup
// re-stats both at each launch and reloads a file whose signature changed.
// A reload that fails keeps the previously loaded content and logs why
// (the signature is not advanced, so the next launch tries again). The two
// files differ in one case only: a curated file that disappears after a
// good load keeps its previous content (it is never meant to be absent;
// git checkout replaces it by unlink and create), while a removed
// generated file reverts what it added (I25).
type recoveryFiles struct {
	curatedPath   string
	generatedPath string

	mu           sync.Mutex
	curatedSig   fileSig
	generatedSig fileSig
	loaded       [2]bool
	curated      map[string]recoveryEntry
	generated    map[string][]recoveryGroup
}

// generatedRecoveryPath is the default location of the machine-made file:
// beside the curated one.
func generatedRecoveryPath(curatedPath string) string {
	return filepath.Join(filepath.Dir(curatedPath), "recovery-flags.generated.json")
}

// newRecoveryFiles loads both files once and returns the reader.
func newRecoveryFiles(curatedPath, generatedPath string) *recoveryFiles {
	f := &recoveryFiles{
		curatedPath:   curatedPath,
		generatedPath: generatedPath,
		curated:       map[string]recoveryEntry{},
		generated:     map[string][]recoveryGroup{},
	}
	f.mu.Lock()
	f.refreshLocked()
	f.mu.Unlock()
	return f
}

// refreshLocked reloads whichever file changed since the last good load.
// Caller holds f.mu. The maps are replaced, never mutated, so a registry
// returned by current() stays valid after a later reload.
func (f *recoveryFiles) refreshLocked() {
	if sig := statFileSig(f.curatedPath); !f.loaded[0] || sig != f.curatedSig {
		models, err := readCuratedRecovery(f.curatedPath)
		switch {
		case err == nil:
			f.curated, f.curatedSig, f.loaded[0] = models, sig, true
		case errors.Is(err, fs.ErrNotExist) && !f.loaded[0]:
			log.Printf("warning: recovery registry: %s not present — no per-model overrides", f.curatedPath)
			f.curatedSig, f.loaded[0] = fileSig{missing: true}, true
		case errors.Is(err, fs.ErrNotExist):
			log.Printf("warning: recovery registry: %s disappeared; keeping the previously loaded "+
				"curated entries (%d)", f.curatedPath, len(f.curated))
		default:
			log.Printf("warning: %v; keeping the previously loaded curated entries (%d)", err, len(f.curated))
		}
	}
	if sig := statFileSig(f.generatedPath); !f.loaded[1] || sig != f.generatedSig {
		groups, err := readGeneratedRecovery(f.generatedPath)
		if err != nil {
			log.Printf("warning: %v; keeping the previously loaded generated groups (%d models)", err, len(f.generated))
		} else {
			f.generated, f.generatedSig, f.loaded[1] = groups, sig, true
		}
	}
}

// current returns the registry as of now, re-reading a changed file first.
func (f *recoveryFiles) current() *recoveryRegistry {
	if f == nil {
		return nil
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	f.refreshLocked()
	return &recoveryRegistry{Models: f.curated, Generated: f.generated}
}

// Lookup is recoveryRegistry.Lookup on the current content of both files.
// A nil receiver matches nothing.
func (f *recoveryFiles) Lookup(backendName, modelName string) (recoveryEntry, bool) {
	if f == nil {
		return recoveryEntry{}, false
	}
	return f.current().Lookup(backendName, modelName)
}
