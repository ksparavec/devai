package main

import (
	"testing"
)

// Head-side minimal parser: cover every emit shape from
// scripts/model-picker.py and the agent-tests router transcripts.
func TestParseModelAndSuffixes(t *testing.T) {
	tests := []struct {
		name    string
		in      string
		want    MinimalRequest
		wantErr bool
	}{
		{
			name: "bare HF name",
			in:   "Qwen3-8B-NVFP4",
			want: MinimalRequest{Model: "Qwen3-8B-NVFP4"},
		},
		{
			name: "HF name with @ctx suffix",
			in:   "Qwen3-8B-NVFP4@131072",
			want: MinimalRequest{Model: "Qwen3-8B-NVFP4", Context: 131072},
		},
		{
			name: "Ollama tag with reasoning override",
			in:   "qwen3.5:9b-q8_0::nothink",
			want: MinimalRequest{Model: "qwen3.5:9b-q8_0", Reasoning: "nothink"},
		},
		{
			name: "ctx and reasoning together",
			in:   "Qwen3-8B-NVFP4::nothink@65536",
			want: MinimalRequest{
				Model:     "Qwen3-8B-NVFP4",
				Context:   65536,
				Reasoning: "nothink",
			},
		},
		{
			name: "HF name with sha @ but no ctx (sha not numeric)",
			in:   "openai/gpt-oss-20b@deadbeef",
			want: MinimalRequest{Model: "openai/gpt-oss-20b@deadbeef"},
		},
		{
			name: "HF name with sha @ followed by @ctx",
			in:   "openai/gpt-oss-20b@deadbeef@262144",
			want: MinimalRequest{
				Model:   "openai/gpt-oss-20b@deadbeef",
				Context: 262144,
			},
		},
		{
			name: "negative ctx leaves model intact",
			in:   "model@-5",
			want: MinimalRequest{Model: "model@-5"},
		},
		{
			name: "non-int ctx leaves model intact",
			in:   "model@abc",
			want: MinimalRequest{Model: "model@abc"},
		},
		{
			name:    "all-suffix garbage produces empty",
			in:      "::reason",
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseModelAndSuffixes(tc.in)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error, got %+v", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Fatalf("got %+v, want %+v", got, tc.want)
			}
		})
	}
}

func TestParseMinimal_FromBody(t *testing.T) {
	tests := []struct {
		name    string
		body    string
		want    MinimalRequest
		wantErr bool
	}{
		{
			name: "openai-compat shape",
			body: `{"model":"Qwen3-8B-NVFP4@131072","messages":[]}`,
			want: MinimalRequest{Model: "Qwen3-8B-NVFP4", Context: 131072},
		},
		{
			name: "anthropic-compat shape",
			body: `{"model":"qwen3.5:9b::nothink","max_tokens":50,"messages":[]}`,
			want: MinimalRequest{Model: "qwen3.5:9b", Reasoning: "nothink"},
		},
		{
			name:    "missing model field",
			body:    `{"messages":[]}`,
			wantErr: true,
		},
		{
			name:    "model not a string",
			body:    `{"model":123}`,
			wantErr: true,
		},
		{
			name:    "empty model field",
			body:    `{"model":""}`,
			wantErr: true,
		},
		{
			name:    "malformed JSON",
			body:    `{not json`,
			wantErr: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ParseMinimal([]byte(tc.body))
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error, got %+v", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Fatalf("got %+v, want %+v", got, tc.want)
			}
		})
	}
}
