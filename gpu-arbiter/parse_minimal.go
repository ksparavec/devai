// Head-side minimal request parsing.
//
// The head extracts only what it needs to make a routing decision:
// the model name plus the optional `@<ctx>` and `::<reasoning>`
// suffixes the picker emits. The full mutation chain
// (parseReasoningOverride, parseCtxOverride, maybeStripTools,
// setNumCtx, reasoning-policy injection, parser-plugin selection)
// continues to live on the worker per cluster-mode decision 2.
//
// The shape we accept matches both OpenAI-compat
// (/v1/chat/completions) and Anthropic-compat (/v1/messages) bodies
// because both carry `model` at the top level. Ollama's /api/chat
// and /api/generate have the same convention. We do NOT touch the
// request body otherwise -- workers reject the modified shape.

package main

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
)

// MinimalRequest captures the routing-relevant fields of an incoming
// request body. Reasoning is "" when no suffix was present.
type MinimalRequest struct {
	Model     string
	Context   int    // 0 when no @<ctx> suffix
	Reasoning string // "" when no ::<reasoning> suffix
}

// ParseMinimal reads `body` (already-decoded request bytes) and
// returns a MinimalRequest. Returns an error when the body isn't
// JSON or doesn't carry a string `model` field. Order of suffix
// stripping matches the worker-side parser: `@<ctx>` first, then
// `::<reasoning>`.
//
// Examples:
//
//	"Qwen3-8B-NVFP4"                 -> model=Qwen3-8B-NVFP4
//	"Qwen3-8B-NVFP4@131072"           -> model=..., ctx=131072
//	"qwen3.5:9b::nothink"             -> model=qwen3.5:9b, reasoning=nothink
//	"Qwen3-8B-NVFP4::nothink@65536"   -> model=..., ctx=65536, reasoning=nothink
func ParseMinimal(body []byte) (MinimalRequest, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return MinimalRequest{}, fmt.Errorf("decode request body: %w", err)
	}
	modelRaw, ok := raw["model"]
	if !ok {
		return MinimalRequest{}, fmt.Errorf("request body missing required 'model' field")
	}
	var model string
	if err := json.Unmarshal(modelRaw, &model); err != nil {
		return MinimalRequest{}, fmt.Errorf("'model' field is not a JSON string: %w", err)
	}
	model = strings.TrimSpace(model)
	if model == "" {
		return MinimalRequest{}, fmt.Errorf("'model' field is empty")
	}
	return parseModelAndSuffixes(model)
}

// parseModelAndSuffixes is the core of ParseMinimal exposed for
// tests and the worker-side reuse path. Strips `@<ctx>` first,
// then `::<reasoning>`.
func parseModelAndSuffixes(name string) (MinimalRequest, error) {
	r := MinimalRequest{Model: name}

	// `@<ctx>` is the trailing token after the LAST `@`. The model
	// name itself can contain `@` (HF repos always do:
	// "openai/gpt-oss-20b@deadbeef" carries an @-prefixed sha that
	// is part of the canonical name -- but the picker's emitted
	// `@<ctx>` is always after that, e.g.
	// "Qwen3-8B-NVFP4@131072"). We split on the last `@`, check the
	// tail is a positive integer; otherwise leave it alone.
	if at := strings.LastIndex(r.Model, "@"); at != -1 {
		tail := r.Model[at+1:]
		if n, err := strconv.Atoi(tail); err == nil && n > 0 {
			r.Context = n
			r.Model = r.Model[:at]
		}
	}

	// `::<reasoning>` is a single suffix; strip from the LAST `::`
	// so a name like "qwen3.5:9b" with its single colon stays
	// intact.
	if cc := strings.LastIndex(r.Model, "::"); cc != -1 {
		reasoning := strings.TrimSpace(r.Model[cc+2:])
		if reasoning != "" {
			r.Reasoning = reasoning
			r.Model = r.Model[:cc]
		}
	}

	r.Model = strings.TrimSpace(r.Model)
	if r.Model == "" {
		return MinimalRequest{}, fmt.Errorf(
			"model name became empty after stripping suffixes from %q",
			name,
		)
	}
	return r, nil
}
