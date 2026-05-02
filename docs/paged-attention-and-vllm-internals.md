# PagedAttention and vLLM internals -- why the elastic KV pool exists

This page explains the trick that makes vLLM (and SGLang and every
other modern inference engine that copied them) actually faster and
more memory-efficient than the naive approach. The trick is called
**PagedAttention**, and it is the reason the project's other docs can
talk about an *"elastic KV pool"* that fills the remaining VRAM after
weights and overhead.

If you have read
[`attention-and-the-transformer.md`](attention-and-the-transformer.md)
Sec. 6 (KV cache mechanics) and
[`nvfp4-coldstart.md`](nvfp4-coldstart.md) Sec. 2 (VRAM budget), you
already know *what* the KV cache is and *that* the elastic pool fills
remaining VRAM. This doc explains *why* that pool can be "elastic" at
all, and what magic vLLM does to make it work.

---

## 1. The problem PagedAttention solves

Recall from
[`attention-and-the-transformer.md`](attention-and-the-transformer.md)
Sec. 6 that during decode, every layer of the model maintains a per-
sequence KV cache: one (K, V) pair per token in the sequence so far.

The **naive approach** to storing this in VRAM:

```
   for each running sequence s:
       allocate two contiguous tensors of shape
       [num_layers, num_kv_heads, max_seq_len, head_dim]
       -- one for K, one for V
```

For Qwen3-8B with FP8 KV at 128 K context, that is `36 x 8 x 131072 x
128 x 1 B = 4.5 GB` for K plus another 4.5 GB for V -- **9 GB
contiguous per sequence**. The runtime has no idea how long the
sequence will actually become at the time it allocates.

This is the source of two real problems:

### 1.1 Internal fragmentation

You allocate 9 GB but generate only 1000 tokens. Then 8.93 GB of that
allocation sits unused for the entire lifetime of the request. On a
24 GB card, you can host *one* sequence with up to ~14 GB headroom
total -- meaning *zero* concurrent sequences if you reserve the worst
case.

### 1.2 External fragmentation

Sequence A allocates 9 GB, runs, finishes, freed. Sequence B
allocates 9 GB. So far so good. But if A and B don't fit
back-to-back in the freed slot, you fragment the heap and lose
serving capacity even with bytes still nominally available.

The original 2023 vLLM paper (Kwon *et al.*, *Efficient Memory
Management for Large Language Model Serving with PagedAttention*,
SOSP 2023, [arXiv:2309.06180](https://arxiv.org/abs/2309.06180))
benchmarked this and found 60-80 % of allocated KV memory was wasted
in production serving with the naive scheme.

---

## 2. The insight -- borrow from operating systems

This is exactly the problem virtual memory solved for operating
systems in the 1960s. The OS doesn't allocate contiguous physical
RAM for each process; it allocates a **page table** that maps each
process's virtual addresses to scattered physical pages of fixed
size (typically 4 KB).

PagedAttention does the same for KV cache:

- **Physical KV pool**: VRAM is partitioned at startup into fixed-
  size **blocks**, each holding `block_size` tokens' worth of K and
  V values for all layers and heads. (Default `block_size = 16` in
  vLLM.) The number of blocks is determined by total VRAM minus
  weights minus overhead.
- **Per-sequence block table**: each running sequence holds a small
  list of block indices it owns, in logical order. The K and V for
  tokens 0-15 live in block #41, for tokens 16-31 live in block #87,
  etc.
- **Allocation grows on demand**: when a sequence's last block fills
  up (16 more tokens generated), the runtime grabs one more free
  block from the pool and appends its index to the sequence's block
  table.

---

## 3. What this changes -- five wins, one cost

### 3.1 Win: near-zero internal fragmentation

A sequence that generates 1000 tokens uses `ceil(1000 / 16) = 63`
blocks. The 64th block (started but only 8/16 tokens used) is the
*only* over-allocation. Total waste: <= `block_size - 1` tokens per
sequence ~ 0.0001 % of a long-context request.

### 3.2 Win: zero external fragmentation

Blocks are all the same size, drawn from a single free pool. Free
becomes "return block index to pool"; allocate becomes "pop one off
the pool". No coalescing needed; no neighbour-merging; no failure
under fragmentation.

### 3.3 Win: prefix sharing

Two sequences that begin with the same prompt (very common -- every
agent has the same system prompt) can **share the early blocks of
the KV cache by reference**. Same physical block, two logical block
tables pointing at it. Each sequence gets its own writable copy
only when their content diverges (copy-on-write). vLLM exposes this
as `--enable-prefix-caching` (on by default in modern versions).

For an agent with a 5 KB system prompt, this means:

- First request: prefill computes ~1.2 K KV blocks and writes them.
- Every subsequent request with the same system prompt: prefill
  *skips* those blocks entirely, reusing them by reference. TTFT on
  a long-prompt-warm-cache becomes near-instant.

### 3.4 Win: continuous batching

Older inference engines used "static batching" -- wait for N requests
to arrive, batch them, run them all to completion, return all
results. Slow requests blocked fast ones; the GPU sat idle waiting
to gather a full batch.

PagedAttention enables **continuous batching**: the scheduler can
add a new request to a partially-running batch every step (because
KV blocks are independent), and finished requests free their blocks
mid-batch (because no contiguous allocation needs to be preserved).
The GPU stays maximally busy. vLLM and SGLang both run continuous
batching by default.

### 3.5 Win: dynamic admission control

Because the runtime knows exactly how many free blocks remain at any
moment, it can decide *not* to admit a new request that wouldn't
fit, or to **preempt** an existing low-priority request by swapping
its blocks out to host RAM (or just dropping them -- the request can
re-prefill). vLLM exposes this as `vllm:num_preemptions_total` in
its Prometheus metrics.

### 3.6 Cost: a small per-attention-call overhead

PagedAttention requires the attention kernel to dereference the
block table for each token, instead of reading from a contiguous
tensor. vLLM ships custom CUDA kernels (`paged_attention_v1`,
`paged_attention_v2`) that handle this gather pattern with minimal
slowdown -- single-digit-percent vs the contiguous version.
Worth it.

---

## 4. How this maps to the project's "elastic pool" language

[`nvfp4-coldstart.md`](nvfp4-coldstart.md) Sec. 2 calls the top of the
VRAM stack the *"Free / KV elastic pool"* and notes that *"vLLM and
SGLang both grow the paged-KV pool until total VRAM hits
`--gpu-memory-utilization`"*. Here is what that means precisely:

- At container start, the runtime computes
  `free_vram = (gpu_memory_utilization x physical_VRAM) - weights -
  cuda_graphs - workspace - overhead`.
- It then divides `free_vram` by the per-block size (`block_size x
  num_layers x num_kv_heads x head_dim x dtype_bytes x 2` for K+V)
  and rounds down. That is the **fixed number of paged-KV blocks**
  the runtime has to play with for the entire lifetime of the
  container.
- "Elastic" refers to how those blocks are *distributed across
  running sequences*. The pool itself is fixed-size; the per-
  sequence allocation is what stretches and shrinks.

For Qwen3-8B-NVFP4 on the 24 GB RTX PRO 4000 Blackwell, using the
project's defaults:

```
   physical_VRAM      = 24 GB
   gpu_memory_util    = 0.94
   budget             = ~22.6 GB
   weights (NVFP4 + BF16 embed)    = ~5.1 GB
   cuda graphs + workspace + overhead = ~4.0 GB
   -> free_vram        = ~13.5 GB

   block_size         = 16 tokens
   per-block KV bytes = 16 x 36 x 8 x 128 x 1 x 2  = 1 179 648 B ~ 1.13 MB
   -> number of blocks ~ 13.5 GB / 1.13 MB ~ 11 950

   capacity at one sequence: 11 950 blocks x 16 tokens ~ 191 K tokens
                              (more than the 131 K context cap)
   capacity at 8 concurrent 16K-context sequences:
                              8 x 1 024 = 8 192 blocks needed -> fits
```

That is the math the bench harness's `peak_vram_gb = 22.53`
indirectly confirms -- the runtime grew the paged pool until VRAM
hit the budget cap, then stopped.

---

## 5. The PagedAttention kernel -- what it actually does on the GPU

The standard attention kernel (FlashAttention, the cuDNN MHA, etc.)
expects K and V to be contiguous in memory. PagedAttention's kernel
takes an additional input -- the **block table** -- and gathers K, V
fragments from scattered physical blocks.

Pseudocode for one decode step:

```
   inputs:
     q:           [num_heads, head_dim]               # query for the new token
     block_table: [num_blocks_for_this_seq]           # logical -> physical mapping
     k_cache, v_cache: [total_blocks, num_kv_heads, block_size, head_dim]

   output: [num_heads, head_dim]   # attention output for the new token

   loop over each logical block in this sequence:
       physical_idx = block_table[logical_idx]
       k_block = k_cache[physical_idx]   # gather
       v_block = v_cache[physical_idx]
       scores  = q @ k_block.transpose() / sqrt(head_dim)
       scores  = causal_mask(scores, position_offset_of_this_block)
       weights = softmax_partial(scores)         # online softmax
       output += weights @ v_block
   normalise output by accumulated softmax denominator
```

The tricks that make this fast:

- **Online softmax** -- accumulate the weighted sum and the softmax
  denominator together in a streaming fashion, never materialising
  the full attention matrix.
- **Block-level parallelism** -- multiple GPU threads handle
  different logical blocks in parallel and their partial results
  combine via the online softmax.
- **Vectorised gather** -- physical block indices are loaded as
  vector loads to amortise the indirection cost.

The implementation lives in vLLM's `csrc/attention/` directory; the
algorithmic essence comes from FlashAttention 2 (Dao 2023,
[arXiv:2307.08691](https://arxiv.org/abs/2307.08691)) extended with
the block-table indirection.

---

## 6. SGLang's twist -- RadixAttention

SGLang (the project's port-11436 backend) builds on PagedAttention
but adds **RadixAttention** (Zheng *et al.* 2024,
*SGLang: Efficient Execution of Structured Language Model
Programs*, [arXiv:2312.07104](https://arxiv.org/abs/2312.07104)).

The key idea: prefix sharing in vLLM is per-block (16 tokens at a
time). RadixAttention organises the cached prefixes in a **radix
tree** indexed by token sequence. Two requests can share a common
prefix **even if its length is not block-aligned** -- the tree
matches at the longest common token-level prefix, and the per-block
representation is reused below that.

Practical effect: in workloads with many sub-string-overlapping
prompts (say, an agent that hits the same template with slightly
different parameters), SGLang's KV cache hit rate is materially
higher than vLLM's, translating to faster TTFT for repetitive
patterns. For the project's typical workload (single user, varied
prompts), the two engines perform comparably.

---

## 7. Practical implications

- **The "elastic pool" in this project's docs is a fixed pool of
  fixed-size blocks.** "Elastic" = dynamic allocation across
  sequences, not dynamic resizing of the pool.
- **Number of blocks = derived from VRAM budget at container
  start.** Changing `--gpu-memory-utilization` or `--max-model-len`
  changes the block count, which is one reason the router has to
  recreate containers when those change (see
  [`nvfp4-coldstart.md`](nvfp4-coldstart.md) Sec. 1).
- **Prefix caching is huge for agents.** If your agent reuses a
  big system prompt, vLLM's `--enable-prefix-caching` (default on)
  saves the prefill cost on every reuse. The router doesn't need
  to do anything special -- it just forwards requests and the
  engine deduplicates underneath.
- **Continuous batching makes single-stream and multi-stream
  performance look quite similar at low concurrency.** The benefit
  shows up under load -- the GPU stays at high utilisation
  regardless of mid-batch arrivals or completions.
- **Preemption is rare in practice on this project's setup.** The
  router enforces GPU mutual exclusion (one backend at a time), so
  vLLM only sees the requests routed to it; with batch sizes well
  below the block budget, preemption doesn't trigger.
- **Increasing `block_size` reduces metadata overhead but
  increases minimum-allocation waste.** Default 16 is a good
  balance. Don't change unless you know why.

---

## 8. Diagnosing PagedAttention behaviour

vLLM exposes Prometheus metrics for the paged pool. Useful ones:

- `vllm:gpu_cache_usage_perc` -- fraction of paged blocks currently
  in use. If this hits 100 % under load, you are bandwidth-bound on
  KV reads and admission control will start preempting.
- `vllm:num_preemptions_total` -- cumulative number of preempted
  requests. Should be 0 under normal load.
- `vllm:prompt_tokens_total`, `vllm:generation_tokens_total` -- 
  cumulative tokens processed. Useful for sanity-checking the
  TPS counts the bench produces.

Scrape these from `http://devai-router:11435/metrics` (or the
upstream vLLM container's `/metrics` if you bypass the router).
[`bench-results.md`](bench-results.md)'s "Followup work" item #7
proposes folding these into the bench cache so KV pressure is
visible per-model.

---

## 9. References

### Foundational papers

- Kwon, W. *et al.* (2023). *Efficient Memory Management for Large
  Language Model Serving with PagedAttention.* SOSP 2023.
  [arXiv:2309.06180](https://arxiv.org/abs/2309.06180). The vLLM
  paper.
- Dao, T. (2023). *FlashAttention-2: Faster Attention with Better
  Parallelism and Work Partitioning.*
  [arXiv:2307.08691](https://arxiv.org/abs/2307.08691). The
  attention kernel PagedAttention extends.
- Dao, T. *et al.* (2022). *FlashAttention: Fast and Memory-
  Efficient Exact Attention with IO-Awareness.*
  [arXiv:2205.14135](https://arxiv.org/abs/2205.14135). Original
  online-softmax-on-GPU paper.
- Zheng, L. *et al.* (2024). *SGLang: Efficient Execution of
  Structured Language Model Programs.*
  [arXiv:2312.07104](https://arxiv.org/abs/2312.07104). RadixAttention.

### Engine documentation

- vLLM PagedAttention design doc:
  <https://docs.vllm.ai/en/latest/design/v1/prefix_caching.html>.
- vLLM scheduler & continuous batching docs:
  <https://docs.vllm.ai/en/latest/design/v1/torch_compile.html>.
- SGLang RadixAttention overview:
  <https://docs.sglang.ai/backend/native_api.html>.

### Project-internal

- [`attention-and-the-transformer.md`](attention-and-the-transformer.md)
  Sec. 6 -- what the KV cache is and why decode reads it every step.
- [`nvfp4-coldstart.md`](nvfp4-coldstart.md) Sec. 2 -- the elastic-KV-
  pool entry in the VRAM stack.
- [`backends.md`](backends.md) -- vLLM vs SGLang container
  lifecycle.
- [`router.md`](router.md) -- how `--gpu-memory-utilization` and
  `--max-model-len` get computed and passed to vLLM.
- [`bench-results.md`](bench-results.md) -- peak/mean VRAM that
  reflects the paged pool's saturation.
