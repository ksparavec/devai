package main

import (
	"encoding/json"
	"log"
	"os"
	"strings"
)

// Advertisement vetting.
//
// The router has always answered /v1/models and /api/tags from
// `bs.modelNames`, which is exactly "has a fitting probe cell" -- nothing
// about whether the model was benched, whether its weights are on disk
// for THAT backend, or whether a bench session already dropped it. On
// this fleet that meant 25 advertised rows of which 8 were actually
// serveable-and-benched: 11 vLLM and 6 SGLang rows pointed at weights
// that are not in that backend's store, so a client picking one gets a
// launch failure, and 6 more per backend had never been benched at all.
//
// The fix separates two questions that were previously one field:
//
//	ADVERTISED  what /v1/models and /api/tags list, and therefore what a
//	            human or an agent picks from. Fully vetted.
//	SERVEABLE   what the request handler will accept when a caller names
//	            it explicitly. Unchanged (probe cache).
//
// They must NOT be the same set, because the bench harness is itself a
// router client: it drives every scored task through router_url/v1
// (bench_runner.py). If the serving allowlist required a bench row, a
// newly probed model could never earn its first one -- the router would
// refuse it, so it would never be benched, so the router would keep
// refusing it. Gating only the advertisement breaks that loop while
// still satisfying "never advertise anything un-vetted": the bench names
// its target explicitly and never reads the listing.

// benchedModels indexes the bench cache as backend -> model -> largest
// benched ctx. Absent file or malformed JSON yields an empty index,
// which -- because vetting requires a POSITIVE bench record -- means
// nothing is advertised rather than everything. That is the safe
// direction for a display decision, and it is loud: the caller logs the
// count at startup.
func loadBenchedModels(path string) map[string]map[string]int {
	out := map[string]map[string]int{}
	if path == "" {
		return out
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return out
	}
	var rows map[string]json.RawMessage
	if json.Unmarshal(raw, &rows) != nil {
		return out
	}
	for key, val := range rows {
		if strings.HasPrefix(key, "_") { // _meta and friends are not rows
			continue
		}
		var row struct {
			Model   string `json:"model"`
			Backend string `json:"backend"`
			Context int    `json:"context"`
		}
		if json.Unmarshal(val, &row) != nil {
			continue
		}
		if row.Model == "" || row.Backend == "" {
			continue
		}
		if out[row.Backend] == nil {
			out[row.Backend] = map[string]int{}
		}
		if row.Context > out[row.Backend][row.Model] {
			out[row.Backend][row.Model] = row.Context
		}
	}
	return out
}

// benchVerdict is a recorded bench exclusion for one (model, backend).
type benchVerdict struct {
	reason string
	ctx    int // 0 = no recorded ctx, i.e. applies everywhere
}

// loadBenchExclusions reads deploy/.model-status.json and returns
// backend -> model -> verdict for BENCH reasons only.
//
// Deliberately narrower than the Python `is_bench_excluded`: it applies
// the ctx rule ("judged ctx and above", and a verdict with no ctx applies
// everywhere) but not the sha rule, because the router has no sha for an
// advertised name. Erring toward hiding is correct here -- an
// over-hidden model is still serveable by explicit name, while an
// over-advertised one is the bug being fixed.
func loadBenchExclusions(path string) map[string]map[string]benchVerdict {
	out := map[string]map[string]benchVerdict{}
	if path == "" {
		return out
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return out
	}
	var doc struct {
		Models map[string]struct {
			Backends map[string]struct {
				Status   string `json:"status"`
				Reason   string `json:"reason"`
				JudgedAt struct {
					Ctx *int `json:"ctx"`
				} `json:"judged_at"`
			} `json:"backends"`
		} `json:"models"`
	}
	if json.Unmarshal(raw, &doc) != nil {
		return out
	}
	for name, m := range doc.Models {
		for backend, e := range m.Backends {
			if !strings.HasPrefix(e.Reason, "bench_") {
				continue // probe/operator verdicts are not advertisement gates
			}
			if e.Status != "excluded" {
				continue
			}
			ctx := 0
			if e.JudgedAt.Ctx != nil {
				ctx = *e.JudgedAt.Ctx
			}
			if out[backend] == nil {
				out[backend] = map[string]benchVerdict{}
			}
			out[backend][name] = benchVerdict{reason: e.Reason, ctx: ctx}
		}
	}
	return out
}

// benchExcludedAt reports whether a bench verdict disqualifies this
// (model, backend) at `ctx`. A verdict recorded at ctx N applies at N and
// ABOVE -- long-context failures do not imply short-context ones -- and a
// verdict with no recorded ctx applies everywhere, which is the only safe
// reading of "we do not know where this was judged".
func (a *arbiter) benchExcludedAt(backend, model string, ctx int) (bool, string) {
	v, ok := a.benchExclusions[backend][model]
	if !ok {
		return false, ""
	}
	if v.ctx == 0 || ctx >= v.ctx {
		return true, v.reason
	}
	return false, ""
}

// advertisedNames returns the vetted subset of bs.modelNames.
//
// A model is advertised only when all four hold:
//
//  1. it has a fitting probe cell (it is in bs.modelNames at all);
//  2. its weights are present in THIS backend's store -- each engine
//     bind-mounts only its own, so a model present only in the peer store
//     cannot be launched here;
//  3. it has a bench row on this backend;
//  4. no bench verdict disqualifies it at the ctx we would serve.
//
// Computed once at startup, like modelNames itself: every input is a
// file the router reads at boot and never re-reads.
func (a *arbiter) advertisedNames(bs *backendState) []string {
	out := make([]string, 0, len(bs.modelNames))
	for _, name := range bs.modelNames {
		if !a.isAdvertisable(bs, name) {
			continue
		}
		out = append(out, name)
	}
	return out
}

func (a *arbiter) isAdvertisable(bs *backendState, name string) bool {
	backend := bs.config.Name
	if err := a.checkModelWeights(bs.config, name); err != nil {
		return false
	}
	ctx := a.modelContexts[backend][name]
	if excluded, _ := a.benchExcludedAt(backend, name, ctx); excluded {
		return false
	}
	return a.benchedModels[backend][name] > 0
}

// logAdvertisementGap reports, once per backend at startup, how many
// probed models are being withheld and why. Silence here would make a
// shrunken model list look like a probe-cache problem.
func (a *arbiter) logAdvertisementGap(bs *backendState, advertised []string) {
	total := len(bs.modelNames)
	shown := len(advertised)
	if total == shown {
		return
	}
	var noWeights, noBench, dropped int
	for _, name := range bs.modelNames {
		switch {
		case a.checkModelWeights(bs.config, name) != nil:
			noWeights++
		default:
			ctx := a.modelContexts[bs.config.Name][name]
			if excluded, _ := a.benchExcludedAt(bs.config.Name, name, ctx); excluded {
				dropped++
			} else if a.benchedModels[bs.config.Name][name] == 0 {
				noBench++
			}
		}
	}
	log.Printf("  %s: advertising %d of %d probed model(s) -- withheld: "+
		"%d no weights in this backend's store, %d never benched, "+
		"%d bench-dropped. Withheld models are still SERVEABLE by explicit "+
		"name (this is what lets `make bench-%s` earn a first bench row); "+
		"they are only hidden from /v1/models and /api/tags.",
		bs.config.Name, shown, total, noWeights, noBench, dropped, bs.config.Name)
}
