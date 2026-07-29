package main

import "testing"

// Claude Code filters discovered models with /^(claude|anthropic)/i, so
// every id this lab serves is dropped unless aliased. These pin the alias
// round-trip and, more importantly, the cases where the router must NOT
// rewrite a name.

func TestIsClaudeCodeUA(t *testing.T) {
	cases := []struct {
		ua   string
		want bool
	}{
		// Captured from the real binary on the wire.
		{"claude-code/2.1.220", true},
		// Version must not be pinned -- upstream bumps it constantly.
		{"claude-code/9.9.9", true},
		{"CLAUDE-CODE/2.1.220", true},
		{"  claude-code/2.1.220  ", true},
		// Everything else keeps byte-identical /v1/models output.
		{"", false},
		{"curl/8.5.0", false},
		{"python-httpx/0.27", false},
		{"OpenAI/Python 1.0", false},
		// Substring, not prefix: must not match.
		{"mytool (claude-code/2.1.220)", false},
		// Near-misses.
		{"claude-cli/1.0", false},
		{"claude/2.1", false},
	}
	for _, c := range cases {
		if got := isClaudeCodeUA(c.ua); got != c.want {
			t.Errorf("isClaudeCodeUA(%q) = %v, want %v", c.ua, got, c.want)
		}
	}
}

func TestClaudeAliasFor(t *testing.T) {
	cases := []struct{ in, want string }{
		{"Qwen3.5-9B-NVFP4", "claude-Qwen3.5-9B-NVFP4"},
		{"gpt-oss-20b", "claude-gpt-oss-20b"},
		{"qwen3.6:35b-a3b-mtp-q4_K_M", "claude-qwen3.6:35b-a3b-mtp-q4_K_M"},
		// Already passes the filter -> unchanged, no claude-claude-.
		{"claude-3-5-sonnet", "claude-3-5-sonnet"},
		{"anthropic.claude-x", "anthropic.claude-x"},
		{"Claude-Whatever", "Claude-Whatever"},
	}
	for _, c := range cases {
		if got := claudeAliasFor(c.in); got != c.want {
			t.Errorf("claudeAliasFor(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestResolveClaudeAlias(t *testing.T) {
	known := []string{"Qwen3.5-9B-NVFP4", "gpt-oss-20b", "claude-native-model"}
	cases := []struct {
		name, in, want string
	}{
		{"alias resolves to the real model",
			"claude-Qwen3.5-9B-NVFP4", "Qwen3.5-9B-NVFP4"},
		{"unprefixed name is untouched",
			"Qwen3.5-9B-NVFP4", "Qwen3.5-9B-NVFP4"},
		// The safety property: a real model whose id genuinely starts with
		// claude- must never be rewritten out of existence.
		{"real claude-prefixed model wins over stripping",
			"claude-native-model", "claude-native-model"},
		// An unknown name passes through so the allowlist reports the name
		// the caller actually sent, not a silently rewritten one.
		{"unknown alias passes through unchanged",
			"claude-not-a-real-model", "claude-not-a-real-model"},
		{"empty", "", ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := resolveClaudeAlias(c.in, known); got != c.want {
				t.Errorf("resolveClaudeAlias(%q) = %q, want %q", c.in, got, c.want)
			}
		})
	}
}

func TestResolveClaudeAliasIsCaseInsensitiveOnPrefixOnly(t *testing.T) {
	known := []string{"Qwen3.5-9B-NVFP4"}
	// Prefix match is case-insensitive, but the REMAINDER must match the
	// real id exactly -- model dirs are case-sensitive on disk and the
	// backend is launched with `--model /models/<name>`.
	if got := resolveClaudeAlias("CLAUDE-Qwen3.5-9B-NVFP4", known); got != "Qwen3.5-9B-NVFP4" {
		t.Errorf("got %q, want the real id", got)
	}
	if got := resolveClaudeAlias("claude-qwen3.5-9b-nvfp4", known); got != "claude-qwen3.5-9b-nvfp4" {
		t.Errorf("wrong-case remainder must NOT resolve, got %q", got)
	}
}
