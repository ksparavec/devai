//go:build devai_frozen_cluster

package main

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestLifecycleClass_IsValid(t *testing.T) {
	if !LifecycleEphemeral.IsValid() {
		t.Errorf("ephemeral should be valid")
	}
	if !LifecyclePersistent.IsValid() {
		t.Errorf("persistent should be valid")
	}
	if LifecycleClass("garbage").IsValid() {
		t.Errorf("garbage should not be valid")
	}
	if LifecycleClass("").IsValid() {
		t.Errorf("empty should not be valid")
	}
}

func TestRegisterRequest_Validate(t *testing.T) {
	good := RegisterRequest{
		Name:           "worker-a",
		Lifecycle:      LifecycleEphemeral,
		Endpoint:       "http://10.0.0.1:11444",
		Backends:       []string{"vllm"},
		VRAMGB:         24,
		ArbiterVersion: "abc123",
	}
	if err := good.Validate(); err != nil {
		t.Fatalf("good: %v", err)
	}

	tests := []struct {
		name string
		mut  func(*RegisterRequest)
		want string
	}{
		{"empty name", func(r *RegisterRequest) { r.Name = "  " }, "name"},
		{"bad lifecycle", func(r *RegisterRequest) { r.Lifecycle = "fake" }, "lifecycle"},
		{"empty endpoint", func(r *RegisterRequest) { r.Endpoint = "" }, "endpoint"},
		{"no backends", func(r *RegisterRequest) { r.Backends = nil }, "backend"},
		{"zero vram", func(r *RegisterRequest) { r.VRAMGB = 0 }, "vram"},
		{"negative vram", func(r *RegisterRequest) { r.VRAMGB = -1 }, "vram"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			r := good
			tc.mut(&r)
			err := r.Validate()
			if err == nil {
				t.Fatalf("expected error mentioning %q", tc.want)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error %q does not mention %q", err.Error(), tc.want)
			}
		})
	}
}

func TestHeartbeatRequest_RoundTripJSON(t *testing.T) {
	hb := HeartbeatRequest{
		WorkerID:       "w-123",
		LoadedModel:    "Qwen3-8B-NVFP4",
		LoadedCtx:      131072,
		QueueDepth:     2,
		UtilizationPct: 42.5,
		LastRequestAt:  "2026-05-15T10:00:00Z",
		HealthStatus:   "ready",
		Counter:        99,
	}
	data, err := json.Marshal(hb)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var back HeartbeatRequest
	if err := json.Unmarshal(data, &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if back != hb {
		t.Fatalf("round-trip mismatch:\n got %+v\nwant %+v", back, hb)
	}
}

func TestHeartbeatRequest_OmitsEmptyOptionalFields(t *testing.T) {
	// loaded_model / loaded_ctx / last_request_at are omitempty so a
	// fresh worker (no model loaded yet) doesn't send "" or 0 that
	// the head might mis-interpret as "ctx=0".
	hb := HeartbeatRequest{
		WorkerID:     "w-fresh",
		QueueDepth:   0,
		HealthStatus: "ready",
		Counter:      1,
	}
	data, err := json.Marshal(hb)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	str := string(data)
	if strings.Contains(str, "loaded_model") {
		t.Errorf("loaded_model should be omitted: %s", str)
	}
	if strings.Contains(str, "loaded_ctx") {
		t.Errorf("loaded_ctx should be omitted: %s", str)
	}
	if strings.Contains(str, "last_request_at") {
		t.Errorf("last_request_at should be omitted: %s", str)
	}
}

func TestCommandTypeConstants(t *testing.T) {
	// Lock the wire-format strings; changing them is a breaking
	// change for already-deployed workers.
	if CommandDrain != "drain" {
		t.Errorf("drain const drifted: %q", CommandDrain)
	}
	if CommandShutdown != "shutdown" {
		t.Errorf("shutdown const drifted: %q", CommandShutdown)
	}
	if CommandServe != "serve" {
		t.Errorf("serve const drifted: %q", CommandServe)
	}
}