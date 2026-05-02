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

// recoveryEntry is the per-model flag set. Either field may be empty.
type recoveryEntry struct {
	Flags []string          `json:"engine_flags"`
	Env   map[string]string `json:"engine_env"`
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
	var raw struct {
		Models map[string]recoveryEntry `json:"models"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		log.Printf("warning: recovery registry %s parse failed: %v", path, err)
		return r
	}
	for name, entry := range raw.Models {
		r.Models[name] = entry
	}
	log.Printf("recovery registry: %s loaded (%d entries)", path, len(r.Models))
	return r
}

// Lookup returns the entry for modelName plus whether it matched.
// A nil receiver and an empty modelName both yield (_, false).
func (r *recoveryRegistry) Lookup(modelName string) (recoveryEntry, bool) {
	if r == nil || modelName == "" {
		return recoveryEntry{}, false
	}
	e, ok := r.Models[modelName]
	return e, ok
}
