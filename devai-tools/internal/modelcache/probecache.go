package modelcache

import (
	"encoding/json"
	"os"
	"sort"
	"strconv"
)

// ProbeCell is one (vram_gb, ctx) cell in a probe cache entry's nested
// probes map. Only the eligibility fields are modeled -- fits (vLLM/
// SGLang) and fully_on_gpu (Ollama); an absent field decodes to Go's zero
// value (false), which is the correct "not eligible" reading for a field
// the real prober never wrote.
type ProbeCell struct {
	Fits       bool `json:"fits"`
	FullyOnGpu bool `json:"fully_on_gpu"`
}

// ProbeEntry is one top-level row of a probe cache, keyed by digest
// (Ollama) or "<repo>@<sha>" (vLLM/SGLang). Probes is nested
// probes[vram_gb][ctx] per CLAUDE.md's documented cache shape -- there is
// no interpolation; a missing cell means "not probed".
type ProbeEntry struct {
	Repo    string                          `json:"repo"`
	Sha     string                          `json:"sha"`
	Aliases []string                        `json:"aliases"`
	Probes  map[string]map[string]ProbeCell `json:"probes"`
}

// ProbeCache is a full probe cache file: top-level key -> entry.
type ProbeCache map[string]ProbeEntry

// LoadProbeCache parses one backend's probe cache
// (deploy/.{ollama,vllm,sglang}-reasoning-cache.json). A missing file is
// not an error -- callers treat a nil cache as "this backend has no probe
// data", matching the picker's degradation ("absent = no rows shown").
func LoadProbeCache(path string) (ProbeCache, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var cache ProbeCache
	if err := json.Unmarshal(data, &cache); err != nil {
		return nil, err
	}
	return cache, nil
}

// FitsAt reports whether entry has an eligible cell at exactly (vramGB,
// ctx) for the given backend: fits==true for vllm/sglang, fully_on_gpu==
// true for ollama.
func (e ProbeEntry) FitsAt(vramGB, ctx int, backend string) bool {
	band, ok := e.Probes[strconv.Itoa(vramGB)]
	if !ok {
		return false
	}
	cell, ok := band[strconv.Itoa(ctx)]
	if !ok {
		return false
	}
	if backend == "ollama" {
		return cell.FullyOnGpu
	}
	return cell.Fits
}

// FitResult is one row returned by ListFitting: a model that fits at the
// requested (vram_gb, context) on the given backend, enriched with
// deploy/models.yaml metadata when a matching catalog row is found.
type FitResult struct {
	Name    string `json:"name"`
	Backend string `json:"backend"`
	Family  string `json:"family,omitempty"`
	Size    string `json:"size,omitempty"`
	Purpose string `json:"purpose,omitempty"`
	Repo    string `json:"repo,omitempty"`
}

// ListFitting joins the given probe caches (keyed by backend name) against
// catalog for the requested (vramGB, ctx), optionally scoped to one
// backend. The probe cache is the source of truth for eligibility;
// catalog only supplies display metadata; a probe entry with no catalog
// match is still returned, identified from the probe cache's own
// repo/aliases fields.
func ListFitting(catalog []CatalogEntry, caches map[string]ProbeCache, vramGB, ctx int, backendFilter string) []FitResult {
	backends := []string{"ollama", "vllm", "sglang"}
	if backendFilter != "" {
		backends = []string{backendFilter}
	}

	var out []FitResult
	for _, backend := range backends {
		cache := caches[backend]
		for _, entry := range cache {
			if !entry.FitsAt(vramGB, ctx, backend) {
				continue
			}
			out = append(out, buildFitResult(catalog, backend, entry))
		}
	}

	sort.Slice(out, func(i, j int) bool {
		if out[i].Backend != out[j].Backend {
			return out[i].Backend < out[j].Backend
		}
		return out[i].Name < out[j].Name
	})
	return out
}

func buildFitResult(catalog []CatalogEntry, backend string, entry ProbeEntry) FitResult {
	if cat := findCatalogMatch(catalog, backend, entry); cat != nil {
		return FitResult{
			Name:    cat.Name,
			Backend: backend,
			Family:  cat.Family,
			Size:    cat.Size,
			Purpose: cat.Purpose,
			Repo:    cat.Repo,
		}
	}
	name := entry.Repo
	if backend == "ollama" && len(entry.Aliases) > 0 {
		name = entry.Aliases[0]
	}
	return FitResult{Name: name, Backend: backend, Repo: entry.Repo}
}

// ResolveBenchBase resolves a catalog model name + backend into the bench
// cache's base identifier: "<repo>@<sha>" for vllm/sglang (read straight
// off the matching models.yaml row), or the probe cache's own digest key
// for ollama (models.yaml carries no digest field -- the Ollama probe
// cache is the only place it lives, keyed by digest with the catalog name
// in its aliases list).
func ResolveBenchBase(catalog []CatalogEntry, ollamaProbeCache ProbeCache, backend, name string) (string, bool) {
	if backend == "ollama" {
		for key, entry := range ollamaProbeCache {
			if containsString(entry.Aliases, name) {
				return key, true
			}
		}
		return "", false
	}
	for i := range catalog {
		m := &catalog[i]
		if m.Name == name && hasBackend(m.Backend, backend) && m.Repo != "" && m.Sha != "" {
			return m.Repo + "@" + m.Sha, true
		}
	}
	return "", false
}

func findCatalogMatch(catalog []CatalogEntry, backend string, entry ProbeEntry) *CatalogEntry {
	for i := range catalog {
		m := &catalog[i]
		if !hasBackend(m.Backend, backend) {
			continue
		}
		if backend == "ollama" {
			if containsString(entry.Aliases, m.Name) {
				return m
			}
			continue
		}
		if entry.Repo != "" && m.Repo == entry.Repo && m.Sha == entry.Sha {
			return m
		}
	}
	return nil
}
