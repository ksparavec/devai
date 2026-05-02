# Attention and the transformer -- what an LLM actually does

This page is the conceptual foundation that the other beginner docs in
this folder ([`nvfp4-number-formats.md`](nvfp4-number-formats.md),
[`nvfp4-coldstart.md`](nvfp4-coldstart.md),
[`llm-tokens-and-speed.md`](llm-tokens-and-speed.md),
[`bench-results.md`](bench-results.md)) implicitly assume. It walks
through *what actually happens* between the moment a tokeniser turns
your prompt into integer IDs and the moment the model emits the next
token.

The reference architecture is the **dense decoder-only transformer**
used by Qwen3-8B-NVFP4, Llama-3.1-8B, gpt-oss-20b, and almost every
other open-weights chat model in this project's catalog.

If you already know what attention is and you can sketch a transformer
block on a whiteboard, you can skip to Sec. 6 (KV cache mechanics) -- that
is where this doc connects back to the bandwidth math in
`llm-tokens-and-speed.md`.

---

## 1. The 30 000-foot view

A modern LLM is, mechanically, a function that takes a sequence of
**integer token IDs** and produces a probability distribution over the
next integer token ID:

```
   input:  [9707, 1879, 11, 25, 374, ...]               <- N tokens
   model:  ---------------------------------------->
   output: probability over the 151 936 possible next tokens
```

Internally, the model:

1. **Looks up** each token ID in an embedding table to get a vector of
   numbers (a dense representation of "what this token means").
2. **Runs that sequence of vectors through L identical transformer
   blocks** (36 blocks for Qwen3-8B). Each block updates each vector
   in light of every other vector in the sequence.
3. **Projects the final vector for the last position** through an
   output matrix (the "lm_head") that turns it back into a vector of
   151 936 logits -- one score per possible next token.
4. **Picks one token** from those logits using a sampling strategy
   (greedy / temperature / top-p -- see
   [`sampling-strategies.md`](sampling-strategies.md)).

Steps 2 and 3 are the entire workload. Step 1 is a memory lookup
(cheap). Step 4 is arithmetic on a 151 936-element vector (also
cheap). Almost every interesting thing in an LLM happens *inside* the
36 transformer blocks of step 2.

---

## 2. Embeddings -- turning token IDs into vectors

The vocabulary has `vocab_size` entries (151 936 for Qwen3). The model
holds an embedding matrix `E` of shape `[vocab_size, hidden_size]` -- 
for Qwen3-8B that is `[151 936, 4 096]`. Each row is a learned
vector for one vocabulary token.

Looking up token ID `t` is just indexing:

```
   embedding[t] = E[t]                  # shape [4096]
```

For an N-token prompt, you stack N rows:

```
   x = E[token_ids]                     # shape [N, 4096]
```

That `[N, 4096]` matrix is the model's internal representation of the
sequence. Every transformer block reads in such a matrix and writes
out a same-shaped matrix that has been "updated" -- the per-token
4096-dim vectors carry richer information at each successive layer.

The embedding table is also where the **lm_head** lives at the *output*
end. In tied-embedding models like Qwen3, the same matrix `E` is used
for both the input lookup and the final projection back to vocab
logits. That tying is why the project's
[`nvfp4-coldstart.md`](nvfp4-coldstart.md) Sec. 2 budget shows
embeddings + lm_head as a single 1.2 GB BF16 entry rather than two
separate ones.

---

## 3. The transformer block -- anatomy

Every transformer block in a Qwen3-style decoder is identical in shape
and computes:

```
   x  = x + Attention(RMSNorm(x))      # attention sub-block
   x  = x + MLP(RMSNorm(x))            # feed-forward sub-block
```

The key features:

- **Pre-norm**: RMSNorm runs *before* attention and MLP, not after.
  More stable training; standard since GPT-2's "pre-norm" variant.
- **Residual stream**: each sub-block adds its output back to the
  input. The "residual stream" is the running 4096-dim vector that
  every block reads and adds to.
- **Two sub-blocks per layer**: attention (mixes information *across*
  positions) and MLP (transforms information *within* a position
  independently).
- **L identical layers**: 36 for Qwen3-8B. Each has its own learned
  parameters -- they are not weight-shared.

The interesting structural questions are:

- *How does Attention let one position look at another?*
- *How does the MLP transform a single position's vector?*
- *How does the model encode where each token is in the sequence?*

Sections 4-6 answer these.

---

## 4. Attention from scratch

Self-attention is the mechanism that lets a transformer block mix
information *across positions* -- it is the only place where the
representation at one token position depends on the representations at
other token positions.

### 4.1 Q, K, V -- three projections of the same thing

For each input vector `x_i` at position `i`, the attention sub-block
computes three projections via three learned weight matrices `W_Q`,
`W_K`, `W_V`:

```
   q_i = x_i @ W_Q          # query  -- "what am I looking for?"
   k_i = x_i @ W_K          # key    -- "what do I represent?"
   v_i = x_i @ W_V          # value  -- "what should I contribute?"
```

The mnemonic: each token *broadcasts* a key (a label for itself), it
*emits* a value (the content it would contribute to attending tokens),
and it *casts* a query (a description of what it wants from elsewhere).

### 4.2 Dot-product attention

The model decides "how much should position `i` attend to position
`j`?" by taking the dot product `q_i . k_j` of `i`'s query with `j`'s
key. Big positive dot product -> strong match -> `i` should pay
attention to `j`. The full attention output for position `i` is:

```
   scores_i = [q_i . k_0,  q_i . k_1,  ..., q_i . k_i] / sqrt(d_head)
   weights_i = softmax(scores_i)        # turns scores into a probability distribution
   out_i    = sum_j weights_i[j] * v_j
```

Three things to notice:

- **Causal masking**: in a *decoder* model, position `i` can only
  attend to positions `0..i` (itself and earlier). That is the
  `[q_i . k_0, ..., q_i . k_i]` truncation. Future tokens are masked
  out *because* during training the model predicts each position's
  next token from only what it has seen so far.
- **Scaled by `sqrt(d_head)`**: prevents the softmax saturating when
  `d_head` is large. Standard "scaled dot-product attention".
- **The output is a weighted sum of values**, not keys. The weights
  come from `Q . K`, but the *content* that flows out is `V`. Q and K
  are the addressing scheme; V is the payload.

### 4.3 Worked example -- three tokens

Imagine a tiny model with `d_model = 4` and three input vectors:

```
   x_0 = [1, 0, 1, 0]      ("the")
   x_1 = [0, 1, 0, 1]      ("cat")
   x_2 = [1, 1, 0, 0]      ("sat")
```

Pretend the model has learned `W_Q = W_K = W_V = identity` (so
`q_i = k_i = v_i = x_i`). Compute attention for position 2 ("sat"):

```
   q_2 = [1, 1, 0, 0]
   scores = [q_2 . x_0,  q_2 . x_1,  q_2 . x_2]
          = [1.1+1.0+0.1+0.0,  1.0+1.1+0.0+0.1,  1.1+1.1+0.0+0.0]
          = [1, 1, 2]
   scores / sqrt(4) = [0.5, 0.5, 1.0]
   softmax = [0.21, 0.21, 0.58]
   out_2 = 0.21.x_0 + 0.21.x_1 + 0.58.x_2
         = [0.21+0.58, 0.21+0.58, 0.21+0, 0+0.21]
         = [0.79, 0.79, 0.21, 0.21]
```

The model ended up taking 58 % of "sat" itself, 21 % of "the", and
21 % of "cat", and produced a new vector for position 2 that is a
blend of all three values, weighted by how well each preceding
token's *key* matched "sat"'s *query*.

In a real model the W matrices are not identity, the dot products
mean something semantically (one head might learn "find the previous
verb", another "find the noun being modified"), and there are many
heads in parallel -- but the math is exactly this.

---

## 5. Multi-head, GQA, and why Qwen3-8B has "8 KV heads"

A single attention head has limited capacity -- it picks one
"interpretation" of what relevance means. Real transformers use
**multi-head attention**: split the 4096-dim vector into many
smaller-dim heads, run independent attention on each, concatenate the
results.

For Qwen3-8B (`config.json` excerpt):

```json
"hidden_size": 4096,
"num_attention_heads": 32,
"num_key_value_heads": 8,
"head_dim": 128
```

Read this as:

- `head_dim = 128` -- each head operates on 128-dim sub-vectors.
- `num_attention_heads = 32` -- 32 query heads, each with its own
  projection. Total Q dimensionality = 32 x 128 = 4096 (matches
  hidden_size).
- `num_key_value_heads = 8` -- only **8** independent K and V heads.
  The 32 Q heads are partitioned into 8 groups of 4; each group
  shares one (K, V) head.

The last bullet is **Grouped-Query Attention (GQA)**, introduced by
Ainslie *et al.* (2023, *GQA: Training Generalized Multi-Query
Transformer Models from Multi-Head Checkpoints*,
[arXiv:2305.13245](https://arxiv.org/abs/2305.13245)). It is a
performance hack with a real impact:

- **Multi-Head Attention (MHA)**: one K and one V per Q head. KV
  cache scales as `num_layers x num_attention_heads x head_dim`.
- **Grouped-Query Attention (GQA)**: one K and one V per *group* of
  Q heads. KV cache scales as `num_layers x num_kv_heads x head_dim`,
  i.e. **4x smaller** when `num_attention_heads = 4 x num_kv_heads`.
- **Multi-Query Attention (MQA)**: extreme case, one K/V shared
  across all Q heads. Smallest KV cache, biggest quality hit.

For Qwen3-8B, GQA shrinks the per-token KV from `36 x 32 x 128 x 1 B`
(MHA equivalent) to `36 x 8 x 128 x 1 B` -- exactly the **72 KB/token**
quoted in [`nvfp4-coldstart.md`](nvfp4-coldstart.md) Sec. 2 and
[`llm-tokens-and-speed.md`](llm-tokens-and-speed.md) Sec. 7. **GQA is the
single biggest reason a 128 K context is feasible on a 24 GB card.**

---

## 6. KV cache mechanics -- and the bandwidth tie-in

This is the section that links every other doc in the project
together. The mechanism is straightforward once Sec. 4 and Sec. 5 are clear.

### 6.1 What the KV cache stores

Recall from Sec. 4: attention at position `i` needs `k_0..k_i` and
`v_0..v_i` to compute the output. Each of those K and V vectors was
itself computed from `x_0..x_i` via the projections `x . W_K` and
`x . W_V`.

**Naive approach**: every time you generate a new token, recompute
all the K and V vectors for the entire sequence so far. Cost is
quadratic in sequence length. Untenable for any non-trivial chat.

**KV cache approach**: after computing `k_i` and `v_i` for each new
token, *save them*. The next token only computes its own
`q_{i+1}, k_{i+1}, v_{i+1}` and reads the cached `k_0..k_i, v_0..v_i`
from memory. Per-token compute is now constant in sequence length.

The cost shifts from compute to memory: **the KV cache is the data
that must be read from VRAM at every decode step**. This is why
[`llm-tokens-and-speed.md`](llm-tokens-and-speed.md) Sec. 7 includes
"KV cache up to current position" in the per-token byte budget.

### 6.2 Why this lines up with the bandwidth math

For Qwen3-8B-NVFP4 at decode step *N*:

- Read all model weights: ~5.1 GB (NVFP4 + BF16 embed)
- Read entire KV cache so far: `N x 72 KB` bytes (FP8 KV)

At low context (N small), KV is tiny and decode is bandwidth-bound on
*weights*. At high context, KV starts to dominate. At N = 70 000
tokens, the KV cache (~5 GB) is roughly equal to the weights -- and
beyond that, it dominates the per-token byte budget.

This is why
[`bench-results.md`](bench-results.md)'s 98.3 tok/s for
Qwen3-8B-NVFP4 is measured on *short* prompts -- the bench keeps KV
small, so the bandwidth ceiling is `640 GB/s / 5.1 GB ~ 125 tok/s`.
At 128 K context, the same model would slow to roughly
`640 / (5.1 + 9.4) ~ 44 tok/s`.

### 6.3 Prefill vs decode in light of KV cache

The distinction in
[`llm-tokens-and-speed.md`](llm-tokens-and-speed.md) Sec. 5 now makes
deeper sense:

- **Prefill**: runs the model once on all N prompt tokens in
  parallel. Computes `k_i, v_i` for every i, *populates the KV cache
  in one pass*. Compute-bound because there are N independent (and
  big) matrix multiplications happening on the same weights.
- **Decode**: runs the model once per generated token. Only
  computes one new `(q, k, v)` triple per layer per step. The big
  per-step cost is *reading the cache and the weights*, not
  computing matmuls. Memory-bandwidth-bound.

Prefill at 1000 tokens does roughly the same FLOPs as 1000 decode
steps but spends them in *one* well-batched forward pass instead of
*1000* tiny ones -- that is why TTFT for short prompts can be a few
milliseconds while sustained decode of 100 tokens takes a full
second.

---

## 7. Positional encoding -- RoPE, and why context extension is hard

The attention sub-block, as described in Sec. 4, treats the sequence as
a **set** -- there is nothing in `q . k` that knows position 5 came
before position 47. The model has to be told.

The classical approach (GPT-2, BERT) was **learned absolute
positional embeddings**: a separate `[max_position, hidden]` matrix
indexed by absolute position. Maximum context was baked into the
embedding table.

Modern models instead use **Rotary Position Embeddings (RoPE)**, from
Su *et al.* (2021, *RoFormer: Enhanced Transformer with Rotary
Position Embedding*,
[arXiv:2104.09864](https://arxiv.org/abs/2104.09864)). RoPE rotates
each `(q, k)` pair in 2D subspaces by an angle that depends on the
absolute position. The math is short:

```
   for each pair of dimensions (2j, 2j+1):
       theta_{j} = 10000 ^ (-2j / d_head)
       rotate (q[2j], q[2j+1]) by angle (i x theta_j) at position i
       rotate (k[2j], k[2j+1]) by angle (i x theta_j) at position i
```

The crucial property: the dot product `q_i . k_j` after rotation only
depends on the *difference* `i - j`, not the absolute positions. So
the model effectively learns *relative* position information through
learned content + a fixed rotation.

### 7.1 Why this matters operationally

Qwen3-8B's `config.json` declares `max_position_embeddings: 40960` -- 
the natively trained context. But the project uses it at
**131 072** ctx successfully (see
[`bench-results.md`](bench-results.md)). How?

**RoPE scaling techniques** -- YaRN, Dynamic NTK, Longrope, etc. -- 
modify the `theta` schedule above so the rotation angles slow down at
long positions. The model can then attend over longer ranges than it
was trained on, with some quality degradation that depends on the
technique. vLLM and SGLang apply these via `--rope-scaling` flags
when you pass a `--max-model-len` larger than
`max_position_embeddings`.

This is also why
[`nvfp4-coldstart.md`](nvfp4-coldstart.md)'s "context-cap change
triggers full container recreate" rule applies: a different
`--max-model-len` recomputes RoPE tables and re-allocates the KV
pool.

---

## 8. The MLP / FFN sub-block -- per-position transformation

After attention has mixed information across positions, the MLP
sub-block transforms the result *within each position independently*.
For Qwen3-8B and most modern transformers, the MLP is:

```
   y = down_proj( silu(gate_proj(x)) * up_proj(x) )
```

This is the **SwiGLU** activation (Shazeer 2020,
[arXiv:2002.05202](https://arxiv.org/abs/2002.05202)) -- three linear
layers and an element-wise gate, instead of the classic two-layer
ReLU MLP.

The shapes for Qwen3-8B (`config.json`):

- `gate_proj`: `[hidden=4096, intermediate=12288]` -- expand
- `up_proj`:   `[hidden=4096, intermediate=12288]` -- expand
- `down_proj`: `[intermediate=12288, hidden=4096]` -- project back

The MLP holds **roughly 3x the parameters of the attention
sub-block** in each layer (3 x 4096 x 12288 ~ 150 M vs attention's
~50 M). This is why "where do the parameters of an 8 B model live"
answers as "two-thirds in MLPs, one-third in attention, plus
embeddings". And why
[`nvfp4-number-formats.md`](nvfp4-number-formats.md) cares about
quantising Linear layers -- they *are* the model.

---

## 9. RMSNorm -- keeping activations stable

Modern transformers use **RMSNorm** (Zhang & Sennrich 2019,
[arXiv:1910.07467](https://arxiv.org/abs/1910.07467)) instead of
classic LayerNorm:

```
   rmsnorm(x) = x / sqrt(mean(x^2) + eps) * gain
```

Compared to LayerNorm, RMSNorm:

- Drops the mean-subtraction (assumes activations are roughly
  zero-centred already);
- Drops the bias term (just a learned per-channel `gain`);
- Is roughly 25 % cheaper to compute.

It is functionally a regulariser: each layer's input is normalised so
the residual stream cannot grow unboundedly through 36 layers. The
`gain` parameter lets each channel keep some of its magnitude.

You will not interact with RMSNorm directly, but it shows up in
quantisation discussions because RMSNorm parameters typically stay in
BF16 (each is a 4096-dim vector, negligible bytes).

---

## 10. Putting one full layer together

For each layer `l` in `0..35`, given input `x` of shape `[N, 4096]`:

```
   # Attention sub-block
   h    = rmsnorm_attn(x)                  # [N, 4096]
   q, k, v = h @ W_Q, h @ W_K, h @ W_V     # split into heads, apply RoPE
   # update KV cache: append k, v to layer l's cache
   attn = scaled_dot_product_attention(q, K_cache, V_cache, causal=True)
   x    = x + attn @ W_O                   # residual add

   # MLP sub-block
   h    = rmsnorm_mlp(x)
   x    = x + W_down @ (silu(W_gate @ h) * (W_up @ h))   # SwiGLU + residual
```

Stack 36 of these layers. Then:

```
   x_final     = rmsnorm_final(x)          # final norm
   logits_last = x_final[N-1] @ E^T        # tied embedding -> vocab logits
                                           # shape [vocab_size = 151 936]
```

Sample a token from `logits_last` (per
[`sampling-strategies.md`](sampling-strategies.md)), append it to the
sequence, and run the loop again -- this is **decode**.

---

## 11. What every weight in Qwen3-8B-NVFP4 is, and why

Putting it all together for the reference model -- every parameter on
disk lives in one of these buckets:

| Bucket | Shape (per layer x layers) | Approx params (8B model) | Format on disk | Quantised? |
|---|---|---|---|---|
| Embedding / lm_head (tied) | `[151 936, 4 096]` x 1 | 622 M | BF16 | no -- listed in `quantization_config.ignore` |
| Attn `W_Q`           | `[4 096, 4 096]` x 36 | 605 M | NVFP4 | **yes** |
| Attn `W_K`           | `[4 096, 1 024]` x 36 | 151 M | NVFP4 | **yes** |
| Attn `W_V`           | `[4 096, 1 024]` x 36 | 151 M | NVFP4 | **yes** |
| Attn `W_O`           | `[4 096, 4 096]` x 36 | 605 M | NVFP4 | **yes** |
| MLP `gate_proj`      | `[4 096, 12 288]` x 36 | 1.81 B | NVFP4 | **yes** |
| MLP `up_proj`        | `[4 096, 12 288]` x 36 | 1.81 B | NVFP4 | **yes** |
| MLP `down_proj`      | `[12 288, 4 096]` x 36 | 1.81 B | NVFP4 | **yes** |
| RMSNorm gains        | `[4 096]` x 72 | 0.3 M | BF16 | no |
| **Total**            | | **~7.6 B params** | mixed | mostly NVFP4 |

(Note `W_K` and `W_V` are 4x narrower than `W_Q` and `W_O` because of
GQA: the model still produces 32 query heads' worth of `Q`, but only
8 KV heads' worth of `K` and `V`. That asymmetry, multiplied across
36 layers, accounts for the ~1 GB the model "saves" by being GQA
instead of MHA.)

When [`nvfp4-coldstart.md`](nvfp4-coldstart.md) Sec. 2 says "~3.9 GB of
NVFP4 transformer weights", it is summing rows 2-8 above x 0.5625
B/param ~ 6.95 B x 0.5625 ~ 3.9 GB. When it says "~1.2 GB of BF16
embeddings", it is row 1 x 2 B/param ~ 1.24 GB. The whole budget is
recoverable from `config.json` plus the format rules in
[`nvfp4-number-formats.md`](nvfp4-number-formats.md).

---

## 12. Why this all matters operationally

- **Every "decode is bandwidth-bound" claim flows from Sec. 6.** The KV
  cache is the unavoidable per-token read; weights are the
  unavoidable per-step read. Quantisation shrinks one (NVFP4),
  GQA + FP8-KV shrink the other.
- **Every "context length is expensive" claim flows from Sec. 4 and Sec. 6.**
  Attention is `O(N)` per generated token *because of caching*, but
  reads `O(N)` bytes per layer -- at long N, KV reads dominate the
  bandwidth budget.
- **Every "different model, different parser" claim from
  [`bench-results.md`](bench-results.md) traces back to chat
  templates and tool-call formats** -- covered in
  [`reasoning-tool-calling-chat-templates.md`](reasoning-tool-calling-chat-templates.md),
  not in the architecture itself. The model's *internal* shape is
  the same; the *interface* differs.
- **Every "quantisation hurts X but not Y" claim is about which
  buckets in Sec. 11 you touch.** Quantising MLPs (the bulk) saves a
  lot. Quantising attention `W_O` is risky (small layer, big quality
  effect). Quantising `lm_head` is forbidden by most NVFP4
  checkpoints. The `quantization_config.ignore` field encodes this.

---

## 13. References

### Foundational papers

- Vaswani, A. *et al.* (2017). *Attention Is All You Need.* NeurIPS.
  [arXiv:1706.03762](https://arxiv.org/abs/1706.03762). The
  transformer paper.
- Su, J. *et al.* (2021). *RoFormer: Enhanced Transformer with
  Rotary Position Embedding.*
  [arXiv:2104.09864](https://arxiv.org/abs/2104.09864). RoPE.
- Ainslie, J. *et al.* (2023). *GQA: Training Generalized
  Multi-Query Transformer Models from Multi-Head Checkpoints.*
  [arXiv:2305.13245](https://arxiv.org/abs/2305.13245). GQA.
- Shazeer, N. (2020). *GLU Variants Improve Transformer.*
  [arXiv:2002.05202](https://arxiv.org/abs/2002.05202). SwiGLU.
- Zhang, B., Sennrich, R. (2019). *Root Mean Square Layer
  Normalization.*
  [arXiv:1910.07467](https://arxiv.org/abs/1910.07467). RMSNorm.

### Pedagogical resources

- *The Illustrated Transformer* (Jay Alammar, 2018):
  <http://jalammar.github.io/illustrated-transformer/>. Still the
  best visual walkthrough of multi-head attention.
- Karpathy, A. (2023). *Let's build GPT: from scratch, in code,
  spelled out.* YouTube. Builds nanoGPT step by step in 2 hours.
- Brendan Bycroft's interactive transformer visualisation:
  <https://bbycroft.net/llm>. Lets you watch a real LLM compute one
  token, layer by layer.

### Architecture-specific

- Qwen3 technical report (2024): the architecture excerpt that fixes
  GQA ratios, RoPE base, vocab size etc. Available on the Qwen3 HF
  model card.
- HuggingFace `transformers` source for `Qwen3Model`:
  <https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3>.
  Authoritative for what the math actually is in code.

### Project-internal cross-links

- [`nvfp4-number-formats.md`](nvfp4-number-formats.md) -- what each
  weight bucket from Sec. 11 is stored as.
- [`nvfp4-coldstart.md`](nvfp4-coldstart.md) -- how those weights
  populate VRAM and KV cache during cold start.
- [`llm-tokens-and-speed.md`](llm-tokens-and-speed.md) -- bandwidth
  math that Sec. 6 connects to.
- [`bench-results.md`](bench-results.md) -- measured TPS / TTFT for
  the reference model.
- [`reasoning-tool-calling-chat-templates.md`](reasoning-tool-calling-chat-templates.md)
  -- what the *interface* layer above this architecture looks like.
