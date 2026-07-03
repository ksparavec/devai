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
// JSON or doesn't carry a string `model` field. Suffix stripping is
// order-independent (see parseModelAndSuffixes), matching the
// worker-side peelControlSuffixes.
//
// Examples:
//
//	"Qwen3-8B-NVFP4"                 -> model=Qwen3-8B-NVFP4
//	"Qwen3-8B-NVFP4@131072"           -> model=..., ctx=131072
//	"qwen3.5:9b::nothink"             -> model=qwen3.5:9b, reasoning=nothink
//	"Qwen3-8B-NVFP4::nothink@65536"   -> model=..., ctx=65536, reasoning=nothink
//	"Qwen3-8B-NVFP4@65536::nothink"   -> model=..., ctx=65536, reasoning=nothink (aiagent order)
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
// tests and the worker-side reuse path. It peels the `@<ctx>` and
// `::<reasoning>` suffixes in whatever order the client appended
// them, mirroring the worker-side peelControlSuffixes: the picker's
// canonical order is `<name>::<reasoning>@<ctx>` (ctx last), but
// aiagent/litellm appends its own `::<reasoning>` AFTER the `@<ctx>`
// (`<name>@<ctx>::<reasoning>`). A strict ctx-first strip would fail
// the Atoi on `<ctx>::<reasoning>`, leaving `@<ctx>` glued to the
// name (ctx=0) so the head routes on the wrong model/ctx. Looping
// until neither suffix strips reaches the same (model, ctx,
// reasoning) either way.
func parseModelAndSuffixes(name string) (MinimalRequest, error) {
	r := MinimalRequest{Model: name}

	// Peel whichever suffix is currently trailing, looping until
	// neither strips. `@<ctx>` is the token after the LAST `@` and
	// must be a positive integer -- the model name itself can contain
	// `@` (HF repos carry an @-prefixed sha, "openai/gpt-oss-20b@
	// deadbeef"), so a non-numeric tail is left alone. `::<reasoning>`
	// is the token after the LAST `::`, so a name like "qwen3.5:9b"
	// with its single colon stays intact. Each peel shortens the
	// name, so the loop always terminates.
	for {
		if at := strings.LastIndex(r.Model, "@"); at != -1 {
			if n, err := strconv.Atoi(r.Model[at+1:]); err == nil && n > 0 {
				r.Context = n
				r.Model = r.Model[:at]
				continue
			}
		}
		if cc := strings.LastIndex(r.Model, "::"); cc != -1 {
			if reasoning := strings.TrimSpace(r.Model[cc+2:]); reasoning != "" {
				r.Reasoning = reasoning
				r.Model = r.Model[:cc]
				continue
			}
		}
		break
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
