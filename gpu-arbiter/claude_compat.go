package main

import "strings"

// Claude Code model-list compatibility.
//
// Claude Code can populate its model picker from a local endpoint, but only
// under an opt-in env var AND only for ids matching a hard filter. Both
// facts were read out of the shipped binary and then confirmed on the wire,
// because guessing here is how you build a feature that looks right and
// lists nothing. On claude-code/2.1.278 (2026-09-22):
//
//	CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1 -> GET <base>/v1/models
//	                                                ?limit=1000
//
// under the first-party provider, whenever ANTHROPIC_BASE_URL is set to a
// non-Anthropic host and a token is present. On 2.1.220 the request was
// additionally gated on CLAUDE_CODE_USE_GATEWAY=1, which the picker set;
// from 2.1.265 that variable selects the enterprise Cloud-gateway sign-in
// instead, and 2.1.278 refuses a plain-http URL under it unless the host
// is literally localhost / 127.0.0.1 / [::1], so the picker no longer sets
// it. Discovery was re-verified without it: this router logged the GET
// from the claude-code UA. (The /v1/messages POSTs arrive as
// `claude-cli/<version>`; only the discovery GET is matched here.)
//
// Then, in both the discovery and bootstrap paths:
//
//	.filter((m) => /^(claude|anthropic)/i.test(m.id))
//
// Every id this lab serves (`Qwen3.5-9B-NVFP4`, `gpt-oss-20b`, ...) fails
// that test, so without an alias Claude Code discovers a list of zero and
// silently shows nothing.
//
// The alias is gated on User-Agent rather than applied globally. That was
// an explicit operator choice: every other client keeps byte-identical
// /v1/models output, at the cost of depending on an undocumented UA string
// that upstream may change. If Claude Code ever stops sending
// `claude-code/<version>`, the failure mode is the status quo ante -- an
// empty picker -- not a broken session, because serving never depends on
// the alias (see resolveClaudeAlias).
const claudeAliasPrefix = "claude-"

// claudeCodeUAPrefix is what claude-code/2.1.220 sends. Matched as a
// PREFIX so the version can move without touching this.
const claudeCodeUAPrefix = "claude-code/"

// isClaudeCodeUA reports whether this request came from Claude Code.
func isClaudeCodeUA(ua string) bool {
	return strings.HasPrefix(
		strings.ToLower(strings.TrimSpace(ua)), claudeCodeUAPrefix)
}

// claudeAliasFor returns the id Claude Code will accept for a model.
//
// Prefixing an id that ALREADY passes the filter would produce
// `claude-claude-x`, so those are returned unchanged -- the filter is
// satisfied either way.
func claudeAliasFor(name string) string {
	lower := strings.ToLower(name)
	if strings.HasPrefix(lower, "claude") || strings.HasPrefix(lower, "anthropic") {
		return name
	}
	return claudeAliasPrefix + name
}

// resolveClaudeAlias maps an incoming model name back to a real one.
//
// The prefix is stripped ONLY when stripping names a model this backend
// actually has and the unstripped form does not. A model genuinely called
// `claude-something` therefore keeps working, and an unknown name is
// returned untouched so the allowlist rejects it with its own error rather
// than a confusingly rewritten one.
func resolveClaudeAlias(name string, known []string) string {
	if !strings.HasPrefix(strings.ToLower(name), claudeAliasPrefix) {
		return name
	}
	stripped := name[len(claudeAliasPrefix):]
	strippedKnown := false
	for _, n := range known {
		if n == name {
			return name // a real model owns this exact id; never rewrite it
		}
		if n == stripped {
			strippedKnown = true
		}
	}
	if strippedKnown {
		return stripped
	}
	return name
}
