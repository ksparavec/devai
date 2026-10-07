package main

import (
	"reflect"
	"strings"
	"testing"
)

// Prefix-cache retention (plan 4v, adopted 2026-10-07): every launch on the
// 0.30.0 backend carries --prefix-cache-retention-interval None, emitted by
// the builder before the recovery flags (so a recovery flag still wins
// last); stock vLLM 0.22.1 has no such flag, so a `vllm` launch must not.
func TestEngineArgDefaults_RetentionOnTheHomeBuiltVLLMOnly(t *testing.T) {
	want := []string{"--prefix-cache-retention-interval", "None"}
	if got := engineArgDefaults("vllm-devai"); !reflect.DeepEqual(got, want) {
		t.Errorf("vllm-devai engine args = %q, want %q", got, want)
	}
	for _, b := range []string{"vllm", "sglang", "ollama", "laya-trainer"} {
		if got := engineArgDefaults(b); len(got) != 0 {
			t.Errorf("%s engine args = %q, want none", b, got)
		}
	}
}

func TestVLLMEntrypoint_EngineArgsPrecedeRecoveryFlags(t *testing.T) {
	args := vllmEntrypoint("m", launchConfig{
		MemFraction: 0.9, MaxContext: 32768,
		EngineArgs:    engineArgDefaults("vllm-devai"),
		RecoveryFlags: []string{"--prefix-cache-retention-interval", "0"},
	})
	n := len(args)
	tail := args[n-4:]
	want := []string{"--prefix-cache-retention-interval", "None", "--prefix-cache-retention-interval", "0"}
	if !reflect.DeepEqual(tail, want) {
		t.Errorf("argv tail = %q, want %q (engine args, then recovery flags)", tail, want)
	}
	if got := vllmEntrypoint("m", launchConfig{MemFraction: 0.9, MaxContext: 32768}); strings.Contains(strings.Join(got, " "), "retention") {
		t.Errorf("a launch without EngineArgs emitted the retention flag: %q", got)
	}
}
