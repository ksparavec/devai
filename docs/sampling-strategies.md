# Sampling strategies -- how the next token is actually chosen

This page covers what happens *after* the model emits its
151 936-element logit vector and *before* a token reaches the user.
That step -- picking one token out of all possible vocabulary tokens
based on the logit distribution -- is **sampling**, and it is where
every "temperature", "top-p", "top-k" knob you have seen in API docs
actually lives.

If you already know what a logit is and the difference between greedy
and stochastic decoding, you can jump to Sec. 6 (sane defaults).
Otherwise, this is intentionally a short, self-contained doc -- pick
it up in 15 minutes, never wonder again.

This doc assumes the bigger picture from
[`attention-and-the-transformer.md`](attention-and-the-transformer.md)
Sec. 10: the model's final layer projects each position's hidden vector
through `lm_head` to produce a vector of vocabulary scores called
**logits**. Sampling turns one such vector into one token ID.

---

## 1. From logits to a token

Recall that the model emits, for each position, a vector of length
`vocab_size` (151 936 for Qwen3) of unnormalised real numbers -- the
**logits**. They can be negative or positive, of any magnitude. They
are not probabilities.

To get a probability distribution over the vocabulary, we apply the
**softmax** function:

```
   p[i] = exp(logits[i]) / sum_j exp(logits[j])
```

This produces a vector of non-negative numbers summing to 1 -- a
probability for each possible next token. From that distribution,
sampling picks one token.

Throughout this doc we will use a tiny worked example. Suppose the
vocabulary has only 5 tokens: `{cat, dog, fish, bird, fox}`, and the
model produces logits:

```
   logits = [3.0, 2.5, 0.5, 0.3, -1.0]
   #         cat  dog  fish bird fox

   exp(logits) = [20.09, 12.18, 1.65, 1.35, 0.37]
   sum = 35.64
   probabilities = [0.564, 0.342, 0.046, 0.038, 0.010]
```

So absent any sampling controls, the model thinks "cat" has 56 %
chance, "dog" 34 %, "fish" 5 %, "bird" 4 %, "fox" 1 %. Sampling is
how we turn this distribution into a single chosen token. The
sections below show how each strategy changes this picture.

---

## 2. Greedy / argmax -- the deterministic baseline

The simplest strategy: always pick the most likely token.

```
   chosen = argmax(probabilities) = "cat"
```

In API terms this is **`temperature: 0`** (or, on some servers, also
`top_k: 1`). Properties:

- **Deterministic**: same prompt -> same output, every time.
- **Repetitive**: long greedy generations tend to fall into loops
  ("the the the the ..."), because once a token gets a slight edge
  it dominates downstream probabilities too.
- **Best for**: deterministic eval (see
  [`bench-results.md`](bench-results.md) -- GSM8K and HumanEval are
  scored under temperature 0), exact-answer tasks, regression tests,
  any case where reproducibility matters more than variety.

---

## 3. Temperature -- flatten or sharpen the distribution

Temperature `T` divides the logits before softmax:

```
   p[i] = exp(logits[i] / T) / sum_j exp(logits[j] / T)
```

The effect:

- `T = 0` -> equivalent to greedy (argmax wins).
- `T < 1` -> sharpens the distribution; the most likely tokens get
  *more* probability, less likely ones get less.
- `T = 1` -> no change; sample directly from the model's raw
  distribution.
- `T > 1` -> flattens the distribution; less likely tokens get *more*
  probability.
- `T -> inf` -> uniform random over the vocabulary.

Same logits, three temperatures:

| Token | logit | `T = 0.5` | `T = 1.0` | `T = 2.0` |
|---|---:|---:|---:|---:|
| cat   |  3.0 | 0.79 | 0.564 | 0.391 |
| dog   |  2.5 | 0.18 | 0.342 | 0.305 |
| fish  |  0.5 | 0.003 | 0.046 | 0.112 |
| bird  |  0.3 | 0.002 | 0.038 | 0.101 |
| fox   | -1.0 | 0.0001 | 0.010 | 0.052 |

At `T = 0.5`, "cat" is nearly inevitable. At `T = 2`, "fox" gets a
real 5 % chance even though its logit was sharply negative. Most
models default to `T = 1.0`; many APIs default to `T = 0.7`-`1.0`;
chat product UIs commonly default to `T = 0.7`-`0.8`.

**Rule of thumb**: lower temperature = more focused/conservative
output; higher temperature = more diverse/creative output. For code
or math, use `T <= 0.3`. For brainstorming, `T = 0.9-1.2`. Above 1.5
output usually becomes incoherent.

---

## 4. Top-k -- truncate to the k most likely tokens

Top-k zeros out all but the `k` highest-probability tokens before
sampling. With our example and `k = 2`:

```
   keep top 2: ["cat", "dog"]
   re-normalise probabilities: [0.564 / (0.564+0.342),
                                0.342 / (0.564+0.342)] = [0.622, 0.378]
   sample from this restricted distribution
```

"fish", "bird", "fox" have probability 0 of being chosen.

Properties:

- **Cuts off the long tail.** The model assigns small but non-zero
  probability to thousands of plausible tokens; top-k says "I don't
  care about anything past rank k".
- **Depends on `k` being well-tuned to the task**: too low and the
  output becomes repetitive (collapses to greedy when `k = 1`); too
  high and it does nothing useful (when `k > effective vocab size at
  this position`).
- **Common values**: `k = 40` (a popular default), `k = 50`, `k = 100`.
- **`k = -1` or `k = 0`**: disable top-k entirely (use the full
  distribution).

---

## 5. Top-p (nucleus sampling) -- adaptive truncation

Top-p, a.k.a. nucleus sampling (Holtzman *et al.* 2019,
*The Curious Case of Neural Text Degeneration*,
[arXiv:1904.09751](https://arxiv.org/abs/1904.09751)), instead keeps
the **smallest set of tokens whose cumulative probability exceeds
`p`**.

With our example and `p = 0.9`:

```
   sort by probability descending:
      cat:  0.564
      dog:  0.342  -> cumulative 0.906  (just exceeds 0.9 -- stop)
      fish: 0.046
      bird: 0.038
      fox:  0.010

   nucleus = ["cat", "dog"]
   re-normalise within nucleus, sample
```

Same restricted distribution as top-k=2 in this case. But for a
different logit distribution where the top few tokens are *less*
peaked, the nucleus would expand:

```
   suppose probabilities = [0.3, 0.25, 0.2, 0.15, 0.1]
   p = 0.9 -> keep [0.3, 0.25, 0.2, 0.15] (cumulative 0.9), drop the last
   nucleus = top 4 tokens
```

Properties:

- **Adaptive**: when the model is confident (peaked distribution),
  top-p selects few tokens; when the model is uncertain (flat
  distribution), it selects many. This is the key advantage over
  fixed top-k.
- **Common values**: `p = 0.9`, `p = 0.95`. `p = 1.0` disables it
  (always keep everything). `p = 0` is degenerate -- don't use it.
- **Often combined with temperature**: temperature first, then top-p
  filter, then sample.

---

## 6. Min-p -- a newer, simpler filter

Min-p (Khandelwal *et al.* 2023, popularised on HuggingFace
discussions) keeps tokens whose probability is at least
`min_p x max_probability`. With our example and `min_p = 0.1`:

```
   max_probability = 0.564 (cat)
   threshold = 0.1 x 0.564 = 0.0564
   keep tokens with p >= 0.0564:
      cat (0.564), dog (0.342)  -> both kept
   fish (0.046), bird (0.038), fox (0.010) -> dropped
```

Properties:

- **Distribution-shape-aware**: like top-p, it adapts. Unlike top-p,
  it scales relative to the top token's probability rather than
  cumulative mass.
- **Robust at high temperatures**: when temperature flattens the
  distribution and top-p's nucleus would explode, min-p still keeps
  only tokens that are reasonably probable relative to the leader.
- **Common values**: `min_p = 0.05` to `0.1`. `min_p = 0` disables.
- **Newer than top-p**, available in vLLM (and SGLang via the
  `extra_body` extension), not in older OpenAI-only servers.

---

## 7. Repetition / frequency / presence penalties

These three penalties discourage the model from emitting tokens it
has already used. They subtract from the logits *before* softmax.

### 7.1 Repetition penalty (HuggingFace style)

```
   for each token t already in the output:
       if logits[t] > 0: logits[t] /= penalty
       else:             logits[t] *= penalty
```

Multiplicative; values > 1 push down the previous tokens' logits.
Common values: `1.05` to `1.2`. `1.0` disables.

### 7.2 Frequency penalty (OpenAI style)

```
   for each token t in the output, count its occurrences c:
       logits[t] -= frequency_penalty x c
```

Subtractive, scales linearly with how often `t` already appeared.
Common values: `0.0` to `2.0`.

### 7.3 Presence penalty (OpenAI style)

```
   for each token t in the output (regardless of count):
       logits[t] -= presence_penalty
```

Same as frequency penalty but doesn't scale with count -- every
token that has appeared *at all* gets the same fixed penalty.
Common values: `0.0` to `2.0`.

### 7.4 When to use them

- **None** for code, structured output, or anything where exact
  repetition is required (function names, JSON keys, etc.).
- **Mild** (`0.1` - `0.3` for OpenAI-style, `1.05` - `1.1` for
  HF-style) for chat to suppress the obvious "the the the" loops.
- **Aggressive** (`> 0.5` OpenAI-style, `> 1.2` HF-style) only if
  you observe loops; over-applied, these damage coherence
  (the model starts avoiding common words like "the").

---

## 8. Putting them together -- the typical pipeline

A real inference call applies these in a fixed order:

```
   1. compute logits from the model
   2. apply repetition / frequency / presence penalties
   3. divide by temperature
   4. compute softmax -> probabilities
   5. apply top-k filter
   6. apply top-p filter
   7. apply min-p filter
   8. re-normalise the surviving probabilities
   9. sample one token
```

Steps 5-7 are filters and order between them is mostly defensive -- 
the intersection is the same. Steps 2 and 3 modify the distribution
shape; they do interact (penalties applied at low temperature have
more relative impact).

---

## 9. Sane defaults for common tasks

These are starting points, not law. Tune to your model and task.

### Deterministic eval (GSM8K, HumanEval, scored benchmarks)

```json
{ "temperature": 0 }
```

That's it. Greedy decoding for reproducibility. The bench in this
project runs this way.

### Code generation (interactive)

```json
{ "temperature": 0.2, "top_p": 0.95 }
```

Low temperature for syntactic correctness; modest top-p to allow
some choice in variable names or comment phrasing.

### General chat / Q&A

```json
{ "temperature": 0.7, "top_p": 0.9 }
```

The widely used "OpenAI default-ish" recipe.

### Creative writing / brainstorming

```json
{ "temperature": 1.0, "top_p": 0.95, "presence_penalty": 0.3 }
```

Higher temperature for variety; presence penalty to keep the model
from circling back to the same imagery.

### Reasoning models (`<think>` blocks)

```json
{ "temperature": 0.6, "top_p": 0.95 }
```

DeepSeek-R1's recommended settings. Reasoning models are sensitive
to over-deterministic decoding (collapses the reasoning trace);
they're also sensitive to over-randomness (loses chain-of-thought
coherence). Stay close to these unless the model card says
otherwise.

### Tool calling

```json
{ "temperature": 0, "tool_choice": "auto" }
```

Tool-call arguments are JSON; you want them syntactically perfect.
Greedy is safest. Some servers (vLLM with guided generation)
constrain output to schema regardless of temperature; greedy is
still the simpler choice.

---

## 10. Practical implications

- **Same prompt, same seed, same model, same temperature, same
  config -> same output.** Use `seed` for reproducibility under
  non-greedy sampling. (Greedy doesn't need a seed.)
- **`temperature: 0` makes every other sampling parameter a
  no-op.** No need to set top-p, top-k, etc. when you're greedy.
- **Lowering temperature is *not* the same as raising top-p.**
  Temperature reshapes the entire distribution; top-p truncates the
  tail. They compose; they don't substitute.
- **Repetition loops usually mean "raise temperature slightly,
  then add modest penalty"**, not "raise penalty alone".
- **The default sampling settings on a server (e.g. vLLM, Ollama,
  SGLang) are not always the same as OpenAI's defaults.** vLLM
  defaults to `T = 1.0, top_p = 1.0` (no truncation), which can
  feel "too random" to users coming from `T = 0.7` chat UIs.
  Always set what you want explicitly.
- **Reasoning models lose quality at the extremes.** Don't set
  `temperature: 0` on a thinking model -- the reasoning trace
  collapses to one path and you lose the value of having a
  reasoning step at all.

---

## 11. References

### Foundational

- Holtzman, A. *et al.* (2019). *The Curious Case of Neural Text
  Degeneration.*
  [arXiv:1904.09751](https://arxiv.org/abs/1904.09751). Introduced
  top-p / nucleus sampling and the diagnosis of "neural text
  degeneration" under greedy/beam decoding.
- Fan, A. *et al.* (2018). *Hierarchical Neural Story Generation.*
  [arXiv:1805.04833](https://arxiv.org/abs/1805.04833). Original
  top-k sampling for neural text generation.
- Su *et al.* (2022). *Contrastive Search Is What You Need For
  Neural Text Generation.*
  [arXiv:2210.14140](https://arxiv.org/abs/2210.14140). Background
  on the failure modes of greedy/beam, motivating min-p and similar
  newer filters.

### Engine-specific docs

- OpenAI sampling parameters reference:
  <https://platform.openai.com/docs/api-reference/chat/create>.
  The `temperature`, `top_p`, `frequency_penalty`,
  `presence_penalty`, `seed` field semantics that everyone copies.
- vLLM sampling parameters:
  <https://docs.vllm.ai/en/latest/dev/sampling_params.html>.
  Includes vLLM-specific `top_k`, `min_p`, `repetition_penalty`,
  and the order of operations.
- Ollama options reference:
  <https://github.com/ollama/ollama/blob/main/docs/modelfile.md#parameter>.
  How to set the same knobs on Ollama models, including in
  Modelfiles.

### Recommended sampling settings per model

- DeepSeek-R1 model card: recommends `T = 0.5-0.7, top_p = 0.95`
  for reasoning prompts.
- Qwen3 technical report: recommends `T = 0.6, top_p = 0.95` for
  general chat, `T = 0` for evaluation.

### Project-internal cross-links

- [`attention-and-the-transformer.md`](attention-and-the-transformer.md)
  Sec. 10 -- where logits come from.
- [`openai-api-and-streaming.md`](openai-api-and-streaming.md) Sec. 2.2 -- 
  how to set these parameters via the API.
- [`reasoning-tool-calling-chat-templates.md`](reasoning-tool-calling-chat-templates.md)
  Sec. 3 -- why reasoning models prefer different defaults.
- [`bench-results.md`](bench-results.md) -- runs at temperature 0 for
  scored evals; see "Methodology" section.
