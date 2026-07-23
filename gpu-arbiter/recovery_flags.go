package main

// Per-model recovery flags + env registry — Go side.
//
// Reads deploy/recovery-flags.json (the same file consumed by
// scripts/_probe_hf_common.py). Single source of truth so that probe
// launches and router-driven recreates apply the same flags/env to a
// borderline checkpoint — without it, a probe would OOM while the
// router's serve-time launch would succeed (or vice versa), and the
// probe cache would never get a fits=true cell for the model.
//
// Names absent from the registry pass through unchanged: no extra
// flags, no extra env. This file's behaviour for unmatched names is
// identical to pre-recovery-registry builds.

import (
	"encoding/json"
	"log"
	"os"
)

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
	Image string `json:"image"`
}

// recoveryRegistry resolves canonical model names to their recovery
// flags/env. Always non-nil; an empty Models map means "no per-model
// overrides" (file missing or empty).
type recoveryRegistry struct {
	Models map[string]recoveryEntry `json:"models"`
}

// loadRecoveryRegistry reads the JSON registry. A missing file logs a
// notice and returns an empty (non-nil) registry — every model then
// launches without recovery flags, matching pre-registry behaviour.
// Parse errors log a warning and likewise return empty.
func loadRecoveryRegistry(path string) *recoveryRegistry {
	r := &recoveryRegistry{Models: map[string]recoveryEntry{}}
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			log.Printf("recovery registry: %s not present — no per-model overrides", path)
		} else {
			log.Printf("warning: recovery registry %s read failed: %v", path, err)
		}
		return r
	}
	// The registry tolerates an `_comment` field at the top level (used
	// for human-readable notes); only `models` is consumed.
	//
	// Entries are held as RawMessage and decoded ONE AT A TIME. Decoding
	// the whole file in a single call made any malformed entry (e.g. a
	// `"backends": "vllm"` typo) abort the parse and return an EMPTY
	// registry -- silently stripping the OOM-rescue flags from every
	// model at once, which is exactly the failure the registry exists to
	// prevent. Per-entry decoding keeps every entry that parses.
	var raw struct {
		Models map[string]json.RawMessage `json:"models"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		log.Printf("warning: recovery registry %s parse failed: %v", path, err)
		return r
	}
	for name, rawEntry := range raw.Models {
		entry, ok := decodeRecoveryEntry(name, rawEntry)
		if !ok {
			continue
		}
		r.Models[name] = entry
	}
	log.Printf("recovery registry: %s loaded (%d entries)", path, len(r.Models))
	return r
}

// decodeRecoveryEntry decodes one registry entry leniently. A `backends`
// value that is not a JSON list is logged (naming the model, so the typo
// is findable) and treated as ABSENT -- the entry keeps its flags/env and
// applies everywhere, per contract C2. Only a structurally broken entry is
// dropped, and dropping it never affects the other entries.
func decodeRecoveryEntry(name string, data []byte) (recoveryEntry, bool) {
	// Shadow type: `backends` stays raw so a bad value cannot fail the
	// decode of the surrounding flags/env/image fields.
	var shadow struct {
		Flags    []string          `json:"engine_flags"`
		Env      map[string]string `json:"engine_env"`
		Backends json.RawMessage   `json:"backends"`
		Image    string            `json:"image"`
	}
	if err := json.Unmarshal(data, &shadow); err != nil {
		log.Printf("warning: recovery registry: entry %q skipped (parse failed: %v)", name, err)
		return recoveryEntry{}, false
	}
	e := recoveryEntry{Flags: shadow.Flags, Env: shadow.Env, Image: shadow.Image}
	// Absent, or an explicit null, both mean "no restriction".
	if len(shadow.Backends) == 0 || string(shadow.Backends) == "null" {
		return e, true
	}
	var backends []string
	if err := json.Unmarshal(shadow.Backends, &backends); err != nil {
		log.Printf("warning: recovery registry: entry %q has a non-list `backends` value (%s); "+
			"treating it as absent (entry applies to every backend)", name, shadow.Backends)
		return e, true
	}
	e.Backends = &backends
	return e, true
}

// appliesTo reports whether the entry is valid for `backendName`, per
// contract C2: an ABSENT `backends` key (nil pointer) applies everywhere,
// while a PRESENT but empty list applies nowhere.
func (e recoveryEntry) appliesTo(backendName string) bool {
	if e.Backends == nil {
		return true
	}
	for _, b := range *e.Backends {
		if b == backendName {
			return true
		}
	}
	return false
}

// Lookup returns the entry for (backendName, modelName) plus whether it
// matched. A nil receiver and an empty modelName both yield (_, false), as
// does an entry whose `backends` list excludes backendName.
func (r *recoveryRegistry) Lookup(backendName, modelName string) (recoveryEntry, bool) {
	if r == nil || modelName == "" {
		return recoveryEntry{}, false
	}
	e, ok := r.Models[modelName]
	if !ok {
		return recoveryEntry{}, false
	}
	if !e.appliesTo(backendName) {
		log.Printf("recovery registry: entry for %s is scoped to %v -- not applied to %s",
			modelName, *e.Backends, backendName)
		return recoveryEntry{}, false
	}
	return e, true
}
