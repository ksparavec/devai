package main

import "testing"

// The fit prober records mtp_fits per cell in a separate --speculative-config
// pass. synthesizeHFFromCache must surface a per-model MTPProbedUnfit verdict:
// true iff some cell recorded mtp_fits=false and none recorded true, so the
// router can refuse to emit --speculative-config for a model whose draft head
// OOMs at load (Qwen3.6-27B-MTP-pi-tune-NVFP4 on the 24G card).
func TestSynthesizeHFFromCache_MTPProbedUnfit(t *testing.T) {
	bt, bf := true, false
	mk := func(mtp *bool) map[string]*hfCacheEntry {
		return map[string]*hfCacheEntry{
			"repo@sha": {
				SchemaVersion: 2,
				Repo:          "repo@sha",
				Aliases:       []string{"M"},
				Capability:    CapStructured,
				MaxContext:    32768,
				Probes: map[string]map[string]hfCacheProbe{
					"24": {"32768": {Ctx: 32768, VramGB: 24, Fits: true, ActualVRAMGB: 21.2, MtpFits: mtp}},
				},
			},
		}
	}
	for _, tc := range []struct {
		name string
		mtp  *bool
		want bool
	}{
		{"mtp_fits=false -> unfit", &bf, true},
		{"mtp_fits=true -> fit", &bt, false},
		{"mtp_fits absent -> not unfit (un-probed)", nil, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := synthesizeHFFromCache(mk(tc.mtp), "vllm", 24, 0, nil)
			if len(got) != 1 {
				t.Fatalf("want 1 model, got %d", len(got))
			}
			if got[0].MTPProbedUnfit != tc.want {
				t.Fatalf("MTPProbedUnfit = %v, want %v", got[0].MTPProbedUnfit, tc.want)
			}
		})
	}
}

// A single cell recording mtp_fits=true must win over another recording false
// (MTP is context-independent, but if any tier verified it fits, don't suppress).
func TestSynthesizeHFFromCache_MTPUnfitAnyTrueWins(t *testing.T) {
	bt, bf := true, false
	cache := map[string]*hfCacheEntry{
		"repo@sha": {
			SchemaVersion: 2, Repo: "repo@sha", Aliases: []string{"M"},
			Capability: CapStructured, MaxContext: 65536,
			Probes: map[string]map[string]hfCacheProbe{
				"24": {
					"32768": {Ctx: 32768, VramGB: 24, Fits: true, ActualVRAMGB: 21.0, MtpFits: &bf},
					"65536": {Ctx: 65536, VramGB: 24, Fits: true, ActualVRAMGB: 22.0, MtpFits: &bt},
				},
			},
		},
	}
	got := synthesizeHFFromCache(cache, "vllm", 24, 0, nil)
	if len(got) != 1 {
		t.Fatalf("want 1 model, got %d", len(got))
	}
	if got[0].MTPProbedUnfit {
		t.Fatalf("any-true-wins: MTPProbedUnfit should be false, got true")
	}
}
