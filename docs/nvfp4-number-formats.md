# Number formats in modern LLMs -- a beginner's guide

This page explains the number formats that show up in
[`nvfp4-coldstart.md`](nvfp4-coldstart.md) and most other modern LLM
serving docs: **FP32, FP16, BF16, FP8 (E4M3 / E5M2), FP4, MXFP4, and
NVFP4**. It covers how a regular IEEE-754 single-precision number gets
squeezed into these much smaller bit layouts, why model weights and the
KV cache typically use *different* formats, and what the practical
trade-offs are.

If you already know what an exponent bias is and you can decode
`0x3F800000` as 1.0 in your head, you can skip to Sec. 4.

---

## 1. What a floating-point number actually is

A floating-point number stores three pieces of information in a fixed
budget of bits:

```
   +----+--------------+--------------------------+
   | s  |   exponent   |         mantissa         |
   +----+--------------+--------------------------+
```

- **sign** (`s`) -- 1 bit. `0` = positive, `1` = negative.
- **exponent** -- a few bits encoding a power of two. Stored "biased"
  (we add a constant so the field is always non-negative). The
  exponent decides how *big* or *small* the number is.
- **mantissa** (a.k.a. *significand*) -- the remaining bits encoding
  the fractional part. The mantissa decides how *precise* the number
  is within the range chosen by the exponent.

The value is reconstructed as roughly:

```
    value = (-1)^s x 1.<mantissa> x 2^(exponent - bias)
```

The implicit `1.` in front of the mantissa is the trick that gives you
one extra bit of precision for free in the "normal" range. Numbers very
close to zero are encoded as **denormals** without that implicit `1.` -- 
a detail that doesn't matter for most ML work but is worth knowing
exists.

### Range vs precision: the central trade-off

Given a fixed bit budget, you choose how to split it between exponent
and mantissa:

- **More exponent bits** -> wider range (you can represent both very
  big and very small numbers). Useful for gradients in training and
  for activations with large outliers.
- **More mantissa bits** -> finer precision within whatever range the
  exponent gives you. Useful for weights where small differences
  between, say, 0.0123 and 0.0124 matter.

This single trade-off is what distinguishes every format below.

---

## 2. The reference: IEEE-754 single precision (FP32)

FP32 is what you get from `torch.float32` and from a plain `float` in
NumPy. 32 bits split as **1 sign + 8 exponent + 23 mantissa**, with an
exponent bias of 127.

```
   +-+--------+-----------------------+
   |1|   8    |          23           |   = 32 bits
   +-+--------+-----------------------+
```

- **Range**: ~1.18 x 10^-^3^8 ... ~3.4 x 10^3^8
- **Precision**: about 7 decimal digits
- **Cost**: 4 bytes per number

Worked example -- encoding pi (3.141592653589793...) in FP32:

```
   sign     exponent (biased)   mantissa
    0       1000 0000           100 1001 0000 1111 1101 1011

   value = +1 x 1.10010010000111111011011 (binary)
                        x 2^(128 - 127)
         = 1.5707963705... x 2
         = 3.1415927410...
```

So FP32 represents pi as **3.1415927410**, off from true pi by about
4 x 10^-^8. That's the floor for everything else in this document -- every
smaller format starts here and loses something.

---

## 3. Half-precision: FP16 vs BF16 (16 bits, two flavours)

Cutting from 32 to 16 bits requires throwing away either range or
precision. The two common choices:

| Format | Sign | Exp | Mantissa | Bias | Range | Precision (decimal digits) |
|---|---|---|---|---|---|---|
| **FP16** ("half") | 1 | 5  | 10 | 15  | ~6 x 10^-^5 ... ~6.5 x 10^4 | ~3.3 |
| **BF16** ("bfloat16") | 1 | 8  | 7  | 127 | same range as FP32 (~10^-^3^8 ... ~10^3^8) | ~2.4 |

```
   FP16:  +-+-----+----------+
          |1|  5  |    10    |
          +-+-----+----------+

   BF16:  +-+--------+-------+
          |1|   8    |   7   |
          +-+--------+-------+
```

**FP16** keeps more mantissa bits (more precision) but a much smaller
range -- it overflows around 65 504, which is a real problem for
activations and gradients in training.

**BF16** keeps the *same exponent range as FP32* by using 8 exponent
bits, sacrificing mantissa precision. This is why modern training
defaults to BF16: range matters more than the third decimal digit when
you're multiplying matrices.

Worked example -- encoding pi in both:

| Format | Bits | Stored value | Error vs pi |
|---|---|---|---|
| FP32 | `0 10000000 10010010000111111011011` | 3.14159274... | ~4 x 10^-^8 |
| FP16 | `0 10000 1001001000` | 3.140625 | ~0.0010 |
| BF16 | `0 10000000 1001001` | 3.140625 | ~0.0010 |

Both 16-bit formats land on 3.140625 here (a coincidence -- for other
numbers FP16's extra mantissa bits give noticeably better precision).
Activations and weights stored in BF16 take 2 bytes instead of 4 -- half
the memory footprint and half the memory bandwidth.

---

## 4. FP8: two flavours optimised for different tensors

FP8 cuts the bit budget in half again. Only 8 bits, and again you have
to choose how to split exponent vs mantissa. The `OCP FP8` standard
defines two:

| Format | Sign | Exp | Mantissa | Bias | Max representable | Used for |
|---|---|---|---|---|---|---|
| **E4M3** | 1 | 4 | 3 | 7  | +/-448 | activations, weights |
| **E5M2** | 1 | 5 | 2 | 15 | +/-57 344 (with quirks for inf/NaN) | gradients |

```
   E4M3:  +-+----+-----+
          |1| 4  |  3  |
          +-+----+-----+

   E5M2:  +-+-----+----+
          |1|  5  | 2  |
          +-+-----+----+
```

**E4M3** is the "weights and activations" flavour -- narrower range, but
3 mantissa bits give visibly better precision. **E5M2** mirrors FP16's
exponent range, useful for gradients and other quantities with large
dynamic range.

Worked example -- encoding **0.1** in FP8 E4M3:

```
   target:  0.1 = 1.6 x 2^-4

   exponent: -4 + bias 7 = 3 -> bits 0011
   mantissa: closest 3-bit value to 1.6 ->
             1.5  (binary 100, |error|=0.1)
             1.625 (binary 101, |error|=0.025)  <- winner

   stored bits: 0 0011 101
   decoded:     1.625 x 2^-4 = 0.1015625
   error:       ~0.0016  (~1.6 % of the original value)
```

A single FP8 value can represent at most a few hundred distinct values
across its entire range. This is fine for an *individual* tensor whose
values cluster in a narrow band, but useless for a tensor whose values
span many orders of magnitude -- that's where **scaling** enters.

### Per-tensor scales

In practice, no real tensor's values fit neatly inside FP8's tiny
range. The trick: **store a single FP32 scale alongside the tensor**.

```
   stored_value_FP8 = round( real_value / scale ) into FP8
   real_value       = stored_value_FP8 x scale
```

Pick the scale to put the tensor's largest absolute value near the top
of FP8's range. Now every value in the tensor is encoded as an FP8
fraction of the scale. The cost is one FP32 number per tensor (a few
bytes -- negligible vs the tensor itself).

This works passably when all values in the tensor have similar
magnitudes. When they don't -- large outliers, or a wide spread -- a
single scale forces the small values to round to zero. Hence
**per-channel** and **per-block** scaling.

---

## 5. FP4 -- fewer bits than there are useful values

FP4 is 4 bits total. Practically all real implementations use **E2M1**:
1 sign + 2 exponent + 1 mantissa.

That's **16 distinct values** in the entire format. Here's the full
list of representable values in standard E2M1 (sign x significand,
including subnormals):

```
    +/-0      +/-0.5   +/-1.0   +/-1.5
    +/-2.0   +/-3.0   +/-4.0   +/-6.0
```

You cannot meaningfully store a typical model weight in FP4 directly -- 
the values jump from 4 to 6 with nothing in between. FP4 is *only*
useful when paired with an aggressive scaling scheme that makes those
16 values mean something different in every group.

This is exactly what NVFP4 and MXFP4 do.

---

## 6. NVFP4 vs MXFP4 -- block-scaled FP4

Both formats encode the bulk of the tensor as FP4 values, but attach
fine-grained scales so that each *group of 16 elements* gets its own
local dynamic range.

| Format | Element | Block size | Block scale type | Per-tensor scale |
|---|---|---|---|---|
| **MXFP4** (Microscaling) | E2M1 (4-bit) | 32 | E8M0 (8-bit power-of-two only) | none |
| **NVFP4** (NVIDIA) | E2M1 (4-bit) | 16 | E4M3 (8-bit FP8) | FP32 |

```
   NVFP4 storage layout (per tensor):

     +-------------------------------------------------+
     |  one FP32 per-tensor scale  (4 bytes, ~no cost) |
     +-------------------------------------------------+
     |  block 0: 16 x FP4 values  +  1 x FP8 scale     |  <- 16x4 + 8 = 72 bits
     |  block 1: 16 x FP4 values  +  1 x FP8 scale     |     ~ 4.5 bits per value
     |  ...                                              |
     +-------------------------------------------------+
```

Effective storage: **4 bits + 8 bits / 16 = 4.5 bits per parameter ~
0.5625 bytes per parameter**. That's the magic constant that makes a
70 B model fit on a 96 GB GPU.

### Why two scales?

- The **per-block FP8 scale** is the workhorse. It adapts to the local
  dynamic range of every 16-element group, so a tensor with both very
  small and very large weights doesn't lose its small values to
  rounding.
- The **per-tensor FP32 scale** acts as an outer "calibration" knob.
  If the per-block FP8 scales themselves can't span the tensor's
  global range (FP8 caps at +/-448), the FP32 scale shifts the whole
  tensor into FP8's sweet spot first.

### MXFP4 vs NVFP4 -- what's actually different

- **MXFP4** uses **E8M0** block scales: 8 bits, but only powers of two
  (no mantissa). Smaller, simpler, hardware-cheap, but quantisation
  steps are coarser within a block.
- **NVFP4** uses **E4M3** block scales: full FP8 with 3 mantissa bits,
  so the block scale itself can be a non-power-of-two. Slightly more
  expensive, noticeably better quality on real models. NVIDIA's
  Blackwell tensor cores include native NVFP4 GEMM kernels.

For a more concrete sense of NVFP4's effect, here is the same
0.1-target value encoded in NVFP4 with a hypothetical block scale:

```
   target:        0.1
   block scale:   0.0625   (chosen so the block's max value lands in FP4 range)
   FP4 needed:    0.1 / 0.0625 = 1.6
   nearest FP4:   1.5      (closest of the 16 representable values)
   reconstructed: 1.5 x 0.0625 = 0.09375
   error:         ~0.00625  (~6 %)
```

The block scale is chosen by the *quantisation algorithm* (e.g.
`modelopt`, `llm-compressor`) when the model is converted, not at
runtime. It minimises total reconstruction error across all 16 values
in the block, not for any single value.

---

## 7. The full menu, side by side

| Format | Bits | Sign | Exp | Mant | Block scale | Per-tensor scale | Effective bytes/param | Typical use |
|---|---|---|---|---|---|---|---|---|
| FP32   | 32 | 1 | 8 | 23 | -- | -- | 4.0   | reference / scales / accumulators |
| TF32   | 19 | 1 | 8 | 10 | -- | -- | n/a   | NVIDIA tensor-core matmul intermediates |
| BF16   | 16 | 1 | 8 |  7 | -- | -- | 2.0   | training default, inference embeddings |
| FP16   | 16 | 1 | 5 | 10 | -- | -- | 2.0   | inference, classic KV cache |
| FP8 E4M3 | 8 | 1 | 4 | 3 | -- | per-tensor (FP32) | 1.0   | inference activations & KV cache |
| FP8 E5M2 | 8 | 1 | 5 | 2 | -- | per-tensor (FP32) | 1.0   | training gradients |
| INT8   | 8 | -- | -- | -- | per-channel (FP32) | per-tensor | 1.0 | older quant schemes (LLM.int8(), GPTQ) |
| INT4   | 4 | -- | -- | -- | per-group (FP16) | per-tensor | ~0.5 | GPTQ-int4, AWQ-int4 |
| MXFP4  | 4 | 1 | 2 | 1 | E8M0 / 32-elem | -- | ~0.5  | OCP-standard FP4, AMD MI3xx |
| **NVFP4** | 4 | 1 | 2 | 1 | E4M3 / 16-elem | FP32 | **~0.5625** | **Blackwell-native FP4** |

(TF32 is included for completeness; it's not a storage format, it's
what NVIDIA's tensor cores feed FP32 multiplies into.)

---

## 8. Why weights and KV cache get *different* formats

A real LLM checkpoint has at least three distinct kinds of tensors:

1. **Model weights** (transformer linear layers, attention projections,
   MLP layers). Loaded once. Read billions of times. Static for the
   lifetime of the model.
2. **Embeddings + lm_head** (vocabulary tables). Loaded once, read once
   per token. Small in proportion but very sensitive to precision loss
   because they sit at the input and output of the network.
3. **KV cache** (per-layer key/value tensors for every token in every
   sequence currently being decoded). Created at runtime, grows with
   context length, dominates VRAM at long context.

These have completely different cost structures, and modern checkpoints
quantise them differently to match.

### Weights: aggressive quantisation pays off massively

- **Static at conversion time** -> can afford expensive group-wise
  calibration (run a calibration dataset, find the best scale per
  16-element block, tune outliers individually). The quality cost of
  going from BF16 -> NVFP4 is small if you put real engineering into
  the conversion.
- **Cost is dominated by one-time storage and recurring memory
  bandwidth.** Smaller weights means more weights fit in VRAM, fewer
  bytes per matmul travel from HBM/GDDR to the SMs, and the matmul
  itself runs on Blackwell tensor cores that natively consume FP4
  operands.
- **Errors in individual weights are spread across many activations**
  (each weight participates in many matmuls). Random per-weight noise
  averages out; structured per-channel noise does not, which is why
  per-channel or per-group scales matter.

-> Weights happily go to **NVFP4** (or INT4, GPTQ, AWQ).

### Embeddings + lm_head: usually left alone

- Tiny in count compared to transformer layers (vocab x hidden, vs all
  the matmul weights).
- Quantising them *typically* hurts output quality: the embedding
  table sits at the input edge of the network where small differences
  matter, and the `lm_head` projection determines the logits over
  every vocabulary token.
- The savings from quantising them are small in absolute bytes
  (~1-2 GB on an 8 B model).

-> Embeddings + lm_head stay **BF16** in NVIDIA's `*-NVFP4`
checkpoints. The `quantization_config.ignore: [lm_head]` field in
`config.json` makes this explicit.

### KV cache: compromise between savings and online cost

- **Generated at runtime, per token, per layer.** Quantisation has to
  happen on the GPU during decode -- no offline calibration is
  possible. Whatever scheme you pick must be cheap to compute
  per-tensor on the fly.
- **Errors in KV compound across the entire context.** Every later
  token attends back to every earlier token's K and V; precision loss
  in one token's K bleeds into attention scores for every subsequent
  decode step. This rules out the most aggressive formats.
- **Memory footprint scales linearly with context length.** At long
  context the KV cache often dominates VRAM (see
  [`nvfp4-coldstart.md`](nvfp4-coldstart.md) Sec. 2 for concrete
  numbers).

The two practical KV cache choices:

| KV format | Bytes/element | Pros | Cons |
|---|---|---|---|
| **FP16**    | 2 | classic, no quality loss | doubles long-context VRAM |
| **FP8 E4M3** | 1 | halves long-context VRAM | small per-token quality cost; needs tensor-core support |

INT4/NVFP4 KV exists in research papers but is rarely shipped because
the per-token quantise/dequantise cost and the compounding-error
problem outweigh the savings.

-> Most modern NVFP4 checkpoints (including the reference
`nvidia/Qwen3-8B-NVFP4`) declare **FP8 KV** in
`quantization_config.kv_cache_scheme`. Older ones still use FP16.

### Putting it together

A typical `nvidia/*-NVFP4` checkpoint stores:

```
   transformer linear weights  ->  NVFP4   (4.5 bits / param effective)
   embeddings + lm_head        ->  BF16    (2 bytes / param)
   KV cache (runtime)          ->  FP8     (1 byte / element)
   per-tensor accumulators     ->  FP32    (4 bytes; bookkeeping only)
```

This mix is why a Qwen3-8B-NVFP4 with **FP8 KV** fits at 128 K context
on a 24 GB card while a hypothetical *FP16-KV* version of the same
model would not -- the KV cache alone would be ~19 GB instead of
~9.5 GB at that context, before counting weights, graphs, or workspace.

---

## 9. Cheat sheet for reading checkpoint metadata

When you `cat config.json` on an NVFP4 checkpoint, the relevant fields
are:

```json
"torch_dtype": "bfloat16",
"quantization_config": {
  "quant_algo": "NVFP4",
  "kv_cache_scheme": { "num_bits": 8, "type": "float" },
  "config_groups": {
    "group_0": {
      "weights":            { "num_bits": 4, "type": "float", "group_size": 16 },
      "input_activations":  { "num_bits": 4, "type": "float", "group_size": 16 },
      "targets": ["Linear"]
    }
  },
  "ignore": ["lm_head"],
  "producer": { "name": "modelopt", "version": "0.35.0" }
}
```

Translation:

- `torch_dtype: bfloat16` -> unquantised tensors (embeddings, lm_head,
  norms) are BF16.
- `quant_algo: NVFP4` -> block-scaled FP4 with FP8 block scales.
- `weights: {num_bits: 4, group_size: 16}` -> NVFP4 with 16-element
  blocks (the standard).
- `input_activations: {num_bits: 4, ...}` -> activations are also
  quantised to NVFP4 going *into* each Linear (not the same as KV
  cache; this is the input to the matmul, computed on the fly).
- `kv_cache_scheme: {num_bits: 8, type: float}` -> FP8 KV cache
  (typically E4M3 in practice).
- `ignore: [lm_head]` -> output projection stays BF16.

Compare this to a checkpoint without the `kv_cache_scheme` field -- that
one will use FP16 KV by default and need roughly twice the VRAM at
long context.

---

## 10. Further reading

- IEEE-754 floating-point standard -- the canonical reference for FP32
  / FP16 layouts.
- [Open Compute Project FP8 specification](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf-1)
  -- defines E4M3 and E5M2.
- [OCP Microscaling Formats specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
  -- defines MXFP4, MXFP6, MXFP8.
- NVIDIA Transformer Engine docs -- NVFP4 layout, block-scale handling
  on Blackwell tensor cores.
- HuggingFace model cards under
  `https://huggingface.co/nvidia/*-NVFP4` -- concrete `config.json`
  examples for various model classes.
- Project-internal: [`nvfp4-coldstart.md`](nvfp4-coldstart.md) puts
  the formats in this guide to work -- it shows how the Qwen3-8B-NVFP4
  reference model's `(NVFP4 weights + BF16 embed + FP8 KV)` mix
  produces a measured 22.53 GB peak VRAM on the project's RTX PRO
  4000 Blackwell card.
