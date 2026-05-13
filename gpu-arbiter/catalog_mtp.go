package main

// Catalog metadata side-table for multi-token-prediction launch params.
//
// The probe cache (.vllm-reasoning-cache.json) is the source of truth for
// "does this model fit at this (vram, ctx) cell?" -- it does NOT carry the
// catalog's MTP block. That block lives in deploy/models.yaml, which the
// rest of the router doesn't parse (see main.go:868-895 for the cache-only
// load path).
//
// This file fills the gap: a small side-table loader, called once at
// startup, that walks deploy/models.yaml and indexes any rows carrying an
// `mtp:` block by their `repo:` field. The result is consulted from
// synthesizeHFFromCache so each emitted configModel can pick up its MTP
// metadata without re-introducing models.yaml as a primary source of fit
// truth. Same pattern as recovery_flags.go.
//
// Missing file = empty registry (no MTP for anyone). Parse errors emit a
// warning and likewise return empty -- the router degrades gracefully to
// non-MTP behaviour rather than refusing to boot.

import (
	"log"
	"os"

	"gopkg.in/yaml.v3"
)

// catalogMTPRegistry maps a model's HuggingFace repo (e.g.
// "nvidia/Gemma-4-26B-A4B-NVFP4") to its declared MTP launch params.
// Always non-nil; lookups for absent repos return (nil, false).
type catalogMTPRegistry struct {
	byRepo map[string]*configSpeculative
}

// catalogYAML mirrors deploy/models.yaml just deeply enough to extract
// each model's repo and its (optional) mtp block. All other fields are
// intentionally ignored -- yaml.v3 with KnownFields(false) lets us pluck
// the two keys we care about without re-defining every catalog column.
type catalogYAML struct {
	Models []catalogYAMLModel `yaml:"models"`
}

type catalogYAMLModel struct {
	Repo string             `yaml:"repo"`
	MTP  *configSpeculative `yaml:"mtp,omitempty"`
}

// loadCatalogMTP reads deploy/models.yaml and indexes its `mtp:` blocks
// by repo. Missing or malformed files yield an empty (non-nil) registry
// -- callers can always invoke Lookup safely.
func loadCatalogMTP(path string) *catalogMTPRegistry {
	r := &catalogMTPRegistry{byRepo: map[string]*configSpeculative{}}
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			log.Printf("catalog MTP: %s not present — no MTP rows registered", path)
		} else {
			log.Printf("warning: catalog MTP %s read failed: %v", path, err)
		}
		return r
	}
	var doc catalogYAML
	if err := yaml.Unmarshal(data, &doc); err != nil {
		log.Printf("warning: catalog MTP %s parse failed: %v", path, err)
		return r
	}
	for _, m := range doc.Models {
		if m.Repo == "" || m.MTP == nil {
			continue
		}
		// Defensive: drop entries with a method that didn't validate.
		// generate-catalog.py's _normalize_mtp filters these before
		// writing, but the router stays paranoid in case operators edit
		// models.yaml by hand.
		if m.MTP.Method == "" || m.MTP.NumSpeculativeTokens < 1 {
			log.Printf("warning: catalog MTP for %q skipped: invalid method=%q num_speculative_tokens=%d",
				m.Repo, m.MTP.Method, m.MTP.NumSpeculativeTokens)
			continue
		}
		r.byRepo[m.Repo] = m.MTP
	}
	log.Printf("catalog MTP: %s loaded (%d MTP-enabled entries)", path, len(r.byRepo))
	return r
}

// Lookup returns the MTP block for repo plus whether it matched. A nil
// receiver and an empty repo both yield (nil, false).
func (r *catalogMTPRegistry) Lookup(repo string) (*configSpeculative, bool) {
	if r == nil || repo == "" {
		return nil, false
	}
	e, ok := r.byRepo[repo]
	return e, ok
}
