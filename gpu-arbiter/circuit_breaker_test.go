package main

import (
	"strings"
	"testing"
)

// The router used to relaunch a dying model forever. Ornith-1.0-9B-NVFP4
// at 256K on SGLang was recreated 10 times in a single gsm8k run -- 3 of
// those teardowns logged `container exited`, i.e. the engine genuinely
// terminating, with the router's separate spurious-recreate bug already
// fixed. Each cycle costs a full cold start, so one client asking for a
// model that cannot sustain its advertised context pins the GPU with no
// progress and no error ever reaching the caller.
//
// This is distinct from detectLaunchFailure, which catches an engine that
// dies DURING launch. This catches one that launches cleanly, passes
// /health, and then dies serving.

func TestCircuitBreaker_TripsAfterBudget(t *testing.T) {
	bs := &backendState{}
	key := launchKey("Ornith-1.0-9B-NVFP4", 262144)

	for i := 1; i < maxFailedLaunches; i++ {
		bs.noteLaunchAttempt(key)
		if bs.launchBudgetExhausted(key) {
			t.Fatalf("tripped early after %d attempt(s); budget is %d",
				i, maxFailedLaunches)
		}
	}
	bs.noteLaunchAttempt(key)
	if !bs.launchBudgetExhausted(key) {
		t.Fatalf("did not trip after %d attempts", maxFailedLaunches)
	}
}

// A completed request is the ONLY proof a launch was good -- reaching
// /health is not, because the failure mode is an engine that becomes
// healthy and then dies.
func TestCircuitBreaker_CompletedRequestRepaysBudget(t *testing.T) {
	bs := &backendState{}
	key := launchKey("gpt-oss-20b", 131072)

	for i := 0; i < maxFailedLaunches; i++ {
		bs.noteLaunchAttempt(key)
	}
	if !bs.launchBudgetExhausted(key) {
		t.Fatal("precondition: expected the budget to be exhausted")
	}

	bs.noteLaunchSucceeded()
	if bs.launchBudgetExhausted(key) {
		t.Error("a completed request must clear the budget")
	}
	if bs.failedLaunches != 0 {
		t.Errorf("failedLaunches = %d after success, want 0", bs.failedLaunches)
	}
}

// The reset is deliberately UNKEYED. The attempt is charged against the
// RESOLVED ctx while the request is served at the LAUNCHED ctx; if those
// ever drift, a keyed reset would never match, the budget would never
// clear, and the breaker would eventually refuse a model that works.
// This test pins that: success at a different key must still clear.
func TestCircuitBreaker_ResetIsUnkeyed(t *testing.T) {
	bs := &backendState{}
	charged := launchKey("Model-A", 262144)
	for i := 0; i < maxFailedLaunches; i++ {
		bs.noteLaunchAttempt(charged)
	}

	// Served at a different resolved context than the one charged.
	bs.noteLaunchSucceeded()

	if bs.launchBudgetExhausted(charged) {
		t.Error("budget survived a completed request because the reset was " +
			"keyed -- a working model would eventually be refused")
	}
}

// A different model must not inherit the previous one's failures.
func TestCircuitBreaker_BudgetIsPerModelAndContext(t *testing.T) {
	bs := &backendState{}
	bad := launchKey("Broken-9B", 262144)
	for i := 0; i < maxFailedLaunches; i++ {
		bs.noteLaunchAttempt(bad)
	}
	if !bs.launchBudgetExhausted(bad) {
		t.Fatal("precondition: expected exhausted for the bad model")
	}

	other := launchKey("Working-9B", 131072)
	if bs.launchBudgetExhausted(other) {
		t.Error("a different model inherited the failure budget")
	}

	// Same model at a smaller ctx is a genuinely different proposition --
	// the whole point of the refusal message is "try a smaller @<ctx>".
	smaller := launchKey("Broken-9B", 131072)
	if bs.launchBudgetExhausted(smaller) {
		t.Error("a smaller ctx inherited the failure budget; the remedy the " +
			"error message suggests would be unusable")
	}
}

func TestCircuitBreaker_DisabledByZero(t *testing.T) {
	orig := maxFailedLaunches
	maxFailedLaunches = 0
	t.Cleanup(func() { maxFailedLaunches = orig })

	bs := &backendState{}
	key := launchKey("Anything", 32768)
	for i := 0; i < 50; i++ {
		bs.noteLaunchAttempt(key)
	}
	if bs.launchBudgetExhausted(key) {
		t.Error("DEVAI_MAX_FAILED_LAUNCHES=0 must disable the breaker")
	}
}

// The refusal is the only thing the caller sees, so it has to be
// actionable: name the model, the context, and the way out.
func TestCircuitBreaker_RefusalNamesModelAndRemedy(t *testing.T) {
	srv := &backendState{
		config:          backendConfig{Name: "sglang"},
		failedLaunches:  3,
		failedLaunchKey: launchKey("Ornith-1.0-9B-NVFP4", 262144),
	}
	// Mirror the message ensureBackendRunning builds.
	msg := refusalMessage(srv, "Ornith-1.0-9B-NVFP4", 262144)
	for _, want := range []string{"Ornith-1.0-9B-NVFP4", "262144", "probe-load", "@<ctx>"} {
		if !strings.Contains(msg, want) {
			t.Errorf("refusal message does not mention %q: %s", want, msg)
		}
	}
}
