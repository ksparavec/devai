# Mixture of Experts (MoE) -- what `gpt-oss-20b` actually is

This page covers the architecture variation that breaks every "8 B
model fits in 8 x 0.5625 GB" assumption from the rest of this docs
folder. **MoE models** have a *total* parameter count and an *active*
parameter count that can differ by 5-10x, and that asymmetry is the
whole reason the architecture exists.

The reference example is `openai/gpt-oss-20b`, which appears in
[`bench-results.md`](bench-results.md) as the project's coding
specialist (HumanEval 0.98). Most of what is true for it generalises
to other MoE families -- Mixtral, DeepSeek-V3, Qwen3-MoE, etc.

If [`attention-and-the-transformer.md`](attention-and-the-transformer.md)
is your foundation, this doc swaps out the **MLP sub-block** from Sec. 8
of that doc with an MoE variant. Everything else (attention, KV
cache, RoPE, RMSNorm) stays identical.

---

## 1. Dense vs MoE -- the one structural difference

Recall from
[`attention-and-the-transformer.md`](attention-and-the-transformer.md)
Sec. 8 that the MLP sub-block in a dense transformer is:

```
   y = down_proj( silu(gate_proj(x)) * up_proj(x) )       # SwiGLU MLP
```

Three Linear layers, applied *to every token at every layer*. For
Qwen3-8B these MLPs hold ~5.4 B of the model's 7.6 B transformer
parameters -- most of the model.

A **Mixture of Experts** MLP replaces this single MLP with **N
parallel MLPs ("experts")** plus a small **router** (a tiny
classifier) that decides which experts each token should use:

```
   gate_logits = router(x)                  # [N] scores over experts
   chosen = top_k_indices(gate_logits)      # pick K experts (K << N)
   weights = softmax(gate_logits[chosen])   # normalise over selected
   y = sum over k in chosen:
         weights[k] * expert_k(x)           # SwiGLU MLP per expert
```

Every layer has its own set of `N` experts and its own router. A
typical config: `N = 64` experts per layer, `K = 4` activated per
token (top-4 routing).

The crucial property: **each token only flows through K of N
experts**. The other N-K experts sit idle for that token at that
layer.

---

## 2. Total params vs active params -- the headline difference

For `gpt-oss-20b` and similar MoE models, the model card lists two
numbers:

- **Total parameters** -- count of every weight on disk. This is what
  `ls -lh` of the safetensors files reports. ~21 B for `gpt-oss-20b`.
- **Active parameters per token** -- count of weights actually used in
  the forward pass for a single token. ~3.6 B for `gpt-oss-20b` (top-4
  routing across 32 experts in the MLPs).

The ratio (~6x for `gpt-oss-20b`) is the **sparsity factor**.

What this means in practice:

| Metric | Dense 8 B | MoE 21 B "active 3.6 B" |
|---|---|---|
| Total weights on disk | ~16 GB BF16 / ~5 GB NVFP4 | ~42 GB BF16 / ~12 GB NVFP4 |
| **VRAM required to host** | total params | **total params** (every expert must be loadable) |
| FLOPs per token | proportional to total params | proportional to *active* params |
| Memory bandwidth per decode token | proportional to *active* MLP params + all attn + all KV | similar -- decode reads only loaded experts |
| Quality | dense baseline | matches a dense model 2-4x the active-param count |

So `gpt-oss-20b`:

- Needs **VRAM for all 21 B weights** (you must store every expert
  even if you only use 4 per token).
- Computes only **~3.6 B params of work** per token (4 of 32
  experts active in MLP, plus full attention and embeddings).
- **Quality of a much-smaller dense model**? No -- quality of roughly
  a *dense 12-14 B* model on most benchmarks. The sparsity gives you
  more capacity than active-param count would suggest, but less than
  total-param count.

The bench in [`bench-results.md`](bench-results.md) ranks
`gpt-oss-20b` as the project's **coding specialist** (HumanEval
0.98) -- beating denser models like Qwen3-14B-NVFP4 (0.92) on
HumanEval despite a smaller active-param count, because the experts
specialise.

---

## 3. The router, and what "specialisation" means

The MoE router is the model's own learned scheduler. During training,
the router learns to send different *kinds* of tokens to different
experts. Common patterns observed in interpretability research:

- One expert specialises on syntactic tokens (punctuation, brackets).
- One expert specialises on numbers.
- One expert handles non-English text.
- Several experts cover code-specific token patterns.
- Some experts are "general purpose" backups.

The specialisation is *emergent* -- the model discovers the
partitioning from data; nothing in the architecture says "expert 12
shall handle Python". The training objective is an auxiliary
**load-balancing loss** that penalises the router for sending all
tokens to the same expert (would collapse to a single dense MLP).

The mechanism:

```
   def moe_forward(x):                       # x: [batch, seq, hidden]
       gate_logits = router_linear(x)         # [batch, seq, num_experts]
       weights, indices = top_k(gate_logits, k=4)  # pick 4 best per token
       weights = softmax(weights, dim=-1)     # normalise the 4 chosen weights
       y = zeros_like(x)
       for k in range(4):
           expert_idx = indices[:,:,k]
           expert_input = x                   # entire token vector
           expert_output = expert_lookup(expert_idx)(expert_input)
           y += weights[:,:,k] * expert_output
       return y
```

The actual implementation is a fused kernel that gathers the right
expert weights and dispatches in one shot, but the math is exactly
the above.

---

## 4. Why MoE is harder to serve

The MoE architecture is a serving-engineer's nuisance. The
challenges:

### 4.1 Imbalanced expert load -> idle compute

Even with a load-balancing training loss, real workloads have
non-uniform token distributions. A code-only request might activate
the same 4-6 "coding experts" 90 % of the time and leave the other
26 sitting cold. The cold experts still take VRAM but contribute
nothing.

### 4.2 Per-token routing decisions defeat batching

In dense models, every token at every layer goes through the same
weights. A batch of 100 tokens does one big matmul. In MoE, those
100 tokens each pick their top-4 experts independently, so the
batched op becomes 32 *small* matmuls (one per expert, with whatever
subset of tokens routed there). The GPU does less work per
synchronisation point. Modern engines (vLLM, SGLang, DeepSpeed-MoE)
ship specialised kernels (`grouped_gemm`, expert-parallel sharding)
to mitigate this.

### 4.3 Expert weights bloat VRAM

For `gpt-oss-20b` at NVFP4, total weights are ~12 GB -- vs ~5 GB for
a dense 8 B model. The KV-cache and elastic-pool budget ([`nvfp4-
coldstart.md`](nvfp4-coldstart.md) Sec. 2) is what is left after weights,
so MoE models leave much less room for context and concurrent
sequences.

`gpt-oss-20b` lands at peak 22.37 GB on the 24 GB RTX PRO 4000
Blackwell ([`bench-results.md`](bench-results.md) "KV-pressure
observations") -- comfortably fitting at 262 K context, but with
only ~10 GB of paged-KV pool vs ~13 GB for the dense Qwen3-8B-NVFP4.

### 4.4 Cold-start time scales with total weights, not active

`gpt-oss-20b` has the project's slowest measured cold start among
the agentic-tier models (57.9 s vs 45.6 s for Qwen3-8B-NVFP4 -- see
[`bench-results.md`](bench-results.md) "Cold-start signal" table).
All experts must be loaded into VRAM before serving begins, even
the ones that won't fire on the first request.

### 4.5 Different router behaviour can break in subtle ways

If the load-balancing loss was too weak during training, or if
routing collapses under low temperature, the model can degenerate
into "always pick expert 0" -- equivalent to a dense model with one
sixteenth the capacity. Some MoE models behave noticeably worse at
`temperature: 0` for this reason. Test before assuming dense
sampling defaults transfer.

---

## 5. Why MoE is worth it

For all the serving annoyance:

- **Quality per FLOP** is much better than dense. A 21 B MoE with
  3.6 B active params has FLOPs comparable to a dense 4 B model but
  quality competitive with a dense 13-14 B. The decode bandwidth
  cost (which scales with active params + KV reads) is also lower
  per token than a dense 13 B would be.
- **Adding capacity is cheap to compute, expensive to serve.**
  Want more knowledge? Add more experts. Active FLOPs stay the
  same; VRAM grows. This is the lever DeepSeek-V3 (671 B total /
  37 B active) and Mixtral-8x22B use to scale.
- **Specialisation matches multi-domain workloads.** Coders,
  multilinguals, math problems all activate different experts. A
  single MoE model serves them all without paying full dense cost
  for any one.

For a single-card serving setup like this project's -- one user, one
agent at a time -- the MoE win is more nuanced. You **pay full VRAM
for total params** but only get **active-params bandwidth savings**.
This is why the bench ranks `gpt-oss-20b` as the *coding specialist*
(highest HumanEval) but not the *production agentic default* (that
title goes to dense Qwen3-8B-NVFP4 for its better tok/s and lower
cold-start cost).

---

## 6. Decode bandwidth math for MoE

The bandwidth derivation in
[`llm-tokens-and-speed.md`](llm-tokens-and-speed.md) Sec. 7 generalises
to MoE with one caveat: the per-token byte budget includes only the
*active* MLP weights at each layer.

For `gpt-oss-20b` at decode:

- Total weights: ~12 GB on device (NVFP4, all experts loaded).
- Active per token: ~3.6 B params x 0.5625 B/param ~ 2 GB
  (attention + 4-of-32 expert MLPs + embeddings).
- Plus KV cache reads (similar to a dense model with the same
  attention structure).

Theoretical decode ceiling at near-zero context:

```
   ceiling = 640 GB/s / 2 GB ~ 320 tok/s
```

**Measured** (`bench-results.md`, post-fix): `38.4 tok/s` (still
flagged with a `+` as the bench hadn't yet been re-run with the
corrected TPS counter -- likely the real number is higher).

The gap between 38 and 320 is large, suggesting either:

- The `+` qualifier is doing real work -- we don't have a
  parser-corrected measurement yet.
- MoE serving overhead (per-expert kernel dispatches, routing
  compute, gather/scatter costs) is non-trivial on this card and
  workload.
- Both.

This is one of the open follow-ups in
[`bench-results.md`](bench-results.md) -- re-run `gpt-oss-20b` with
the post-fix TPS counter to get a definitive number.

---

## 7. How to spot an MoE model in the wild

Indicators you are looking at an MoE checkpoint:

- **`config.json`**:
  - `"model_type": "mixtral"` / `"qwen3_moe"` / `"deepseek_v3"` etc.
  - `"num_experts"` / `"n_routed_experts"` field present.
  - `"num_experts_per_tok"` / `"num_active_experts"` field present.
  - `"router_aux_loss_coef"` field (load-balancing loss weight).
- **Model card text**: phrases like "X total parameters, Y active
  per token", or "MoE with K experts".
- **File sizes**: total safetensors size dramatically larger than
  the active-param count would predict.

For `gpt-oss-20b` specifically (`config.json` excerpt -- public on
the HuggingFace model card):

```json
"model_type": "gpt_oss",
"num_experts": 32,
"num_experts_per_tok": 4,
"hidden_size": 4096,
"num_attention_heads": 64,
"num_key_value_heads": 8,   <- GQA, like Qwen3
"num_hidden_layers": 32
```

So 32 experts, top-4 routing, 32 layers, GQA with 8 KV heads.

---

## 8. Practical implications

- **Pick MoE for quality-per-active-FLOP, not for VRAM efficiency.**
  An MoE model is bigger than its active-param dense equivalent,
  not smaller.
- **Don't expect to fit a 200 B-total MoE on a 24 GB card.** Even
  if active params are only 20 B, the total params must all sit in
  VRAM. The project deliberately scopes its MoE catalog to models
  that fit (`gpt-oss-20b` at 22.37 GB peak, the largest entry).
- **Cold-start time grows with total weights**, including unused
  experts. Budget accordingly; the router's
  `HEALTH_TIMEOUT_SECONDS = 600` is comfortable for this scale.
- **Routing depends on temperature in subtle ways.** If MoE quality
  feels degraded at `temperature: 0`, raise to `0.3-0.5` and
  retest -- that often unblocks specialised experts.
- **Streaming behaviour is identical.** From an
  [`openai-api-and-streaming.md`](openai-api-and-streaming.md)
  perspective, you can't tell an MoE response from a dense one
  without inspecting `model` and looking up its spec.

---

## 9. References

### Foundational papers

- Shazeer, N. *et al.* (2017). *Outrageously Large Neural Networks:
  The Sparsely-Gated Mixture-of-Experts Layer.*
  [arXiv:1701.06538](https://arxiv.org/abs/1701.06538). The
  original sparsely-gated MoE for transformers.
- Fedus, W. *et al.* (2021). *Switch Transformers: Scaling to
  Trillion Parameter Models with Simple and Efficient Sparsity.*
  [arXiv:2101.03961](https://arxiv.org/abs/2101.03961). Switch
  Transformer; introduced top-1 routing.
- Lepikhin, D. *et al.* (2020). *GShard: Scaling Giant Models with
  Conditional Computation and Automatic Sharding.*
  [arXiv:2006.16668](https://arxiv.org/abs/2006.16668). MoE on
  TPUs; shard-by-expert.
- DeepSeek-V3 technical report (2024). Detailed MoE design at the
  frontier scale. [arXiv:2412.19437](https://arxiv.org/abs/2412.19437).

### Engine-side

- vLLM MoE kernels and expert parallelism:
  <https://docs.vllm.ai/en/latest/serving/distributed_serving.html>.
- DeepSpeed-MoE (Microsoft):
  <https://github.com/microsoft/DeepSpeed-MLIR>.
- HuggingFace MoE models docs page:
  <https://huggingface.co/docs/transformers/main/en/model_doc/mixtral>
  and similar for `qwen3_moe`, `deepseek_v3`, `gpt_oss`.

### Specific model cards

- `openai/gpt-oss-20b` model card on HuggingFace -- architecture and
  expert count.
- Mixtral 8x7B / 8x22B model cards (the MoE family that popularised
  the "XxY" naming convention).

### Project-internal

- [`attention-and-the-transformer.md`](attention-and-the-transformer.md)
  Sec. 8 -- what the dense MLP looks like; MoE swaps just this sub-block.
- [`nvfp4-coldstart.md`](nvfp4-coldstart.md) Sec. 2 -- VRAM stack;
  MoE pushes the "weights" bar much higher relative to the elastic
  KV pool.
- [`llm-tokens-and-speed.md`](llm-tokens-and-speed.md) Sec. 7 -- decode
  bandwidth math; MoE substitutes "active params" for "total
  params" in the per-token byte budget.
- [`bench-results.md`](bench-results.md) -- `gpt-oss-20b` row, KV-
  pressure column, the `+`-qualified TPS pending re-run.
- `scripts/model-families.yaml` -- `gpt-oss` family entry, including
  the harmony reasoning + tool parsers.
