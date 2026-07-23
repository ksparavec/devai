package modelcache

import (
	"encoding/json"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strconv"
)

// ProbeCell is one (vram_gb, ctx) cell in a probe cache entry's nested
// probes map. Only the eligibility fields are modeled -- fits (vLLM/
// SGLang) and fully_on_gpu (Ollama); an absent field decodes to Go's zero
// value (false), which is the correct "not eligible" reading for a field
// the real prober never wrote.
//
// ServingOk is a *bool because absent and false mean different things:
// the serving-time LOAD probe (`make probe-load-vllm`) writes false when
// a cell that loaded (fits=true) then OOMed under a near-full-context
// request, while absent means the load probe never ran for that cell.
// Absent must keep the pre-load-probe verdict -- same rule as
// scripts/select-models.py's `serving_ok is not False` gate and the
// picker's _vram_from_hf_probe.
type ProbeCell struct {
	Fits       bool  `json:"fits"`
	FullyOnGpu bool  `json:"fully_on_gpu"`
	ServingOk  *bool `json:"serving_ok"`
}

// ProbeEntry is one top-level row of a probe cache, keyed by digest
// (Ollama) or "<repo>@<sha>" (vLLM/SGLang). Probes is nested
// probes[vram_gb][ctx] per CLAUDE.md's documented cache shape. There is
// no upward interpolation: a request above every recorded cell means
// "not probed there". Downward is different -- see FitsAt.
//
// MaxContext is the model's design ceiling as the prober measured it (the
// largest clean actual_context, capped at position_limit -- see
// refresh_top_level_from_cells in scripts/_probe_hf_common.py). It is what
// select-models.py clamps a request against before looking a cell up;
// without it the Go side could not reproduce that clamp and disagreed with
// `make model-fit` for every request above a model's own ceiling. Absent
// (0) means the entry has no clean cell, so there is nothing to clamp to.
type ProbeEntry struct {
	Repo       string                          `json:"repo"`
	Sha        string                          `json:"sha"`
	Aliases    []string                        `json:"aliases"`
	MaxContext int                             `json:"max_context"`
	Probes     map[string]map[string]ProbeCell `json:"probes"`
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

// FitsAt reports whether entry has an eligible cell covering (vramGB,
// ctx) for the given backend: fits==true for vllm/sglang, fully_on_gpu==
// true for ollama, and in both cases serving_ok not explicitly false.
//
// Cell resolution mirrors scripts/select-models.py step for step, and that
// file has TWO readers, not one: hf_probe_at_context for vllm/sglang and
// probe_at_context for ollama. They differ, so cellCovering takes the
// backend -- see its comment.
func (e ProbeEntry) FitsAt(vramGB, ctx int, backend string) bool {
	cell, ok := e.cellCovering(vramGB, ctx, backend)
	if !ok {
		return false
	}
	if cell.ServingOk != nil && !*cell.ServingOk {
		return false
	}
	if backend == "ollama" {
		return cell.FullyOnGpu
	}
	return cell.Fits
}

// cellCovering returns the cell that answers a (vramGB, ctx) query,
// reproducing whichever of scripts/select-models.py's two readers owns the
// backend.
//
// Shared by both readers:
//
//  1. Clamp the request to the model's design ceiling:
//     eff = min(ctx, MaxContext) when MaxContext is recorded. Asking a
//     65K-ceiling model to run at 256K just runs it at 65K -- the same
//     rule select-models.py and the picker apply. Without this clamp the
//     Go side dropped every model whose ceiling sat below the requested
//     context (measured: vllm 4 rows vs make model-fit's 16 at 256K).
//  2. Exact cell at eff wins when recorded.
//
// Step 3 is vllm/sglang ONLY (hf_probe_at_context, ~line 726): the single
// binary-searched winner cell at MaxContext answers, because those caches
// keep exactly one winner per (model, band) -- the largest ctx that both
// fits and serves -- and by KV monotonicity it covers every smaller ctx.
//
// Ollama stops after step 2, matching probe_at_context (~line 762), which
// has no such fallback: the Ollama cache records every probed tier
// separately, so a miss at eff genuinely means "that tier was never
// probed". Routing Ollama through the HF fallback was a measured
// divergence -- at vram=24/ctx=100000 it returned 2 rows where the Python
// rule returns 0, by silently answering an unprobed 100K request with the
// 131K winner cell.
func (e ProbeEntry) cellCovering(vramGB, ctx int, backend string) (ProbeCell, bool) {
	band, ok := e.Probes[strconv.Itoa(vramGB)]
	if !ok {
		return ProbeCell{}, false
	}
	eff := ctx
	if e.MaxContext > 0 && e.MaxContext < eff {
		eff = e.MaxContext
	}
	if cell, ok := band[strconv.Itoa(eff)]; ok {
		return cell, true
	}
	if backend == "ollama" {
		return ProbeCell{}, false
	}
	if e.MaxContext > 0 {
		if cell, ok := band[strconv.Itoa(e.MaxContext)]; ok {
			return cell, true
		}
	}
	return ProbeCell{}, false
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

// WeightStores maps a backend name to the directory its safetensors
// weights live in. vLLM and SGLang read from SEPARATE volumes (see
// CLAUDE.md's /var/cache/devai mount-point convention), so a model pulled
// for one is invisible to the other. Ollama has no entry: its blobs are
// content-addressed under the Ollama tree, not one directory per model, so
// there is no per-model directory to test (see storeGate).
type WeightStores map[string]string

// DefaultWeightStores reads the same env vars with the same defaults as
// the rest of the tree (Makefile, docker-compose.yaml, select-models.py,
// model-picker.py, gpu-arbiter/main.go).
func DefaultWeightStores() WeightStores {
	return WeightStores{
		"vllm":   envOrDefault("VLLM_MODELS_DIR", "/var/cache/devai/vllm"),
		"sglang": envOrDefault("SGLANG_MODELS_DIR", "/var/cache/devai/sglang"),
	}
}

func envOrDefault(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// storeGate returns the weight-store directory to gate backend's rows on,
// plus a note when the gate could not be applied.
//
// Two deliberate non-gates:
//
//   - Ollama. Its models are not directory-backed -- a pulled tag is a
//     manifest plus content-addressed blobs, and the probe cache is keyed
//     by digest, so "is this model on disk" has no <dir>/<name>/config.json
//     analogue. The probe cache is only ever written for a digest the
//     prober actually loaded, so cache presence already implies weights.
//   - A store directory that is absent ENTIRELY. The published server runs
//     in a distroless container that mounts no weight volumes at all
//     (deploy/Dockerfile.mcp-modelstatus bakes only the cache files), so
//     gating there would drop every HF row. Degrade to the un-gated
//     verdict and say so instead.
//
// An existing-but-empty store is NOT the absent case: it is the real
// answer "this backend has no weights", which is exactly the state
// select-models.py's sglang_weight_gaps warns about.
func storeGate(stores WeightStores, backend string) (dir string, note string) {
	if backend == "ollama" {
		return "", ""
	}
	dir, ok := stores[backend]
	if !ok || dir == "" {
		return "", fmt.Sprintf(
			"%s: weights-on-disk check skipped (no store directory configured); "+
				"rows are probe-cache-only and may not be servable", backend)
	}
	if fi, err := os.Stat(dir); err != nil || !fi.IsDir() {
		return "", fmt.Sprintf(
			"%s: weights-on-disk check skipped (store %s is not present in this "+
				"environment); rows are probe-cache-only and may not be servable",
			backend, dir)
	}
	return dir, ""
}

// hasWeights reports whether dir holds a model directory for name with a
// config.json in it -- the same test model-picker.py enumerates the store
// with and select-models.py's sglang_weight_gaps uses.
//
// name is the catalog name, which is also the directory basename the
// download path writes to. A probe entry with no catalog row falls back to
// its "<org>/<repo>" identifier, so only the last segment can ever be a
// directory name; path.Base also neutralises any traversal a hand-edited
// cache could smuggle in.
func hasWeights(dir, name string) bool {
	base := path.Base(name)
	if base == "." || base == ".." || base == "/" {
		return false
	}
	fi, err := os.Stat(filepath.Join(dir, base, "config.json"))
	return err == nil && fi.Mode().IsRegular()
}

// ListFitting joins the given probe caches (keyed by backend name) against
// catalog for the requested (vramGB, ctx), optionally scoped to one
// backend. The probe cache is the source of truth for eligibility;
// catalog only supplies display metadata; a probe entry with no catalog
// match is still returned, identified from the probe cache's own
// repo/aliases fields.
//
// vLLM/SGLang rows are additionally gated on the weights being on disk,
// because a probe cache outlives the weights it was measured from: a cell
// probed against the vLLM store stays in the SGLang cache after the model
// was only ever pulled for vLLM, and both caches keep rows for models
// since deleted. Without the gate this returned models nothing can serve
// (measured on the host: 16 vLLM rows at vram=24/ctx=32768 against 5
// models actually on disk). Passing a nil/empty stores map, or running
// somewhere the store is not mounted, degrades to the un-gated verdict and
// returns a note saying so -- see storeGate.
//
// The second return value is that set of notes; it is nil when every
// backend was gated normally.
func ListFitting(catalog []CatalogEntry, caches map[string]ProbeCache, vramGB, ctx int, backendFilter string, stores WeightStores) ([]FitResult, []string) {
	backends := []string{"ollama", "vllm", "sglang"}
	if backendFilter != "" {
		backends = []string{backendFilter}
	}

	var out []FitResult
	var notes []string
	for _, backend := range backends {
		cache := caches[backend]
		gateDir, note := storeGate(stores, backend)
		if note != "" && len(cache) > 0 {
			notes = append(notes, note)
		}
		for _, entry := range cache {
			if !entry.FitsAt(vramGB, ctx, backend) {
				continue
			}
			row := buildFitResult(catalog, backend, entry)
			if gateDir != "" && !hasWeights(gateDir, row.Name) {
				continue
			}
			out = append(out, row)
		}
	}

	sort.Slice(out, func(i, j int) bool {
		if out[i].Backend != out[j].Backend {
			return out[i].Backend < out[j].Backend
		}
		return out[i].Name < out[j].Name
	})
	return out, notes
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
