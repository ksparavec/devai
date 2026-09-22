# HyperQwen preparation scripts (vendored, unmodified)

Everything under this directory is copied verbatim from

    https://github.com/syv-ai/HyperQwen
    commit c0c81bbbbf91f11b54af7b95f49bd6d1570c2ba0

and is the work of its authors. It is redistributed here under its own
licence, Apache License 2.0 (see `LICENSE` next to this file; upstream's
README states "Apache-2.0, same as the model"). Upstream ships no NOTICE
file and the scripts carry no per-file headers, so this README is the
attribution. Nothing in this directory was edited: `MANIFEST.sha256` holds
the checksums of every file as copied, and tests/python/test_hyperqwen_vendor.py
fails if any of them drifts from it.

## What is here and why

`prepare/` is the checkpoint preparation that lets Qwen3.8-27B serve with
speculative decoding and a usable context on one 24 GB card. devai runs
these scripts, never edits them, from `scripts/prepare-checkpoint.py`
(`make model-prepare NAME=<row>`):

| file | what it does |
|---|---|
| `quant_lm_head.py` | `lm_head` bf16 -> int8, group 128, symmetric; in place, keeps `.bak` |
| `quant_embed.py` | `embed_tokens` likewise, keeps `.bak_embed` |
| `quant_mtp.py` | the `mtp.*` draft module likewise (default int8, `mtp.fc` included), keeps `.bak-mtp` |
| `quant_heads_stream.py` | all three of the above in one streaming pass, for single-shard checkpoints the three cannot read whole; renames originals to `.bak-orig` |
| `build_draft_vocab.py` | slices a 40,960-row draft head out of the int8 `lm_head` and writes the id map; needs the `qwen3_5-mtp-draft-vocab` vLLM patch, which devai's image carries |
| `draft_vocab_ids.json` | the shipped id list, counted over 5.4M tokens of the model's own outputs (97.5% coverage per upstream) |

The vLLM patch series these scripts pair with is NOT vendored: scripts/build-vllm.sh
fetches the same pinned commit at build time and applies it to the source tree.

## Upstream's own documentation

`prepare/README.md` and `docs/optimizations.md` in the upstream repository
explain each step with measurements. Read them there; they are not copied
because they describe upstream's Docker and venv layouts, not devai's.

## Updating

Re-copy from a newer upstream commit, regenerate `MANIFEST.sha256`, and
change the commit above. The scripts assume a compressed-tensors
`pack-quantized` int4 body; on an NVFP4 body they need
scripts/mixed_quant_groups.py afterwards (see that script's docstring).
