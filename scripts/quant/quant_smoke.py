# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "llmcompressor",
#   "datasets",
#   "huggingface_hub",
# ]
# ///
"""Reusable NVFP4 quantize + reload + validate driver.

One job does the whole loop on a single GPU:
  1. QUANTIZE a bf16 model to NVFP4 with llm-compressor (compressed-tensors output).
  2. RELOAD the NVFP4 checkpoint on the SAME GPU (vLLM by default; transformers fallback).
  3. GENERATE a few deterministic prompts and assert the output is non-degenerate.
  4. (optional) PUSH the checkpoint to a HF repo.

Designed to run via `hf jobs uv run --image vllm/vllm-openai:latest ...` so vLLM +
a Blackwell-capable torch come from the image and only llm-compressor is added on
top (see the Makefile quant-* targets). Also runnable locally with `uv run`.

Same driver for the whole ladder -- only --model / --recipe / --out change:
  tiny (~1.5B)  -> plumbing smoke (~$0.25)
  Ornith-9B     -> dense qwen3_5 arch + a 24G-servable NVFP4 byproduct
  Ornith-35B    -> the real MoE+vision run (pass --recipe recipe-ornith-nvfp4.yaml)

UNVERIFIED: this is the FIRST end-to-end exercise of the llm-compressor NVFP4 API,
the uv/image dependency resolution, and Blackwell FP4 reload. The point of the tiny
rung is to shake these out for ~$0.25 before the multi-hour 35B run. The exact
llm-compressor call shapes (scheme name, oneshot kwargs) may need adjustment on the
first run -- treat a first-run failure as data, not a surprise.
"""
from __future__ import annotations

import argparse
import gc
import os
import re
import sys


def log(*a: object) -> None:
    print(*a, flush=True)


PROMPTS = [
    "The capital of France is",
    "List three primary colors:",
    "Q: What is 2 + 2? A:",
    "def add(a, b):\n    return",
]


def _is_degenerate(text: str) -> bool:
    """Lenient 'is it garbage' check -- tiny models degrade under 4-bit, so we
    only reject empty / no-real-words / single-token-repetition output, and print
    everything for a human eyeball."""
    t = (text or "").strip()
    if len(t) < 2:
        return True
    if not re.search(r"[A-Za-z]", t):
        return True
    toks = t.split()
    if len(toks) >= 6 and len(set(toks)) == 1:
        return True
    return False


# Built-in ignore-lists so the driver is self-contained on HF Jobs (no recipe
# file needs to travel with the uploaded script). `ornith` is derived from the
# canonical llm-compressor Qwen3.5-VL example and fits both Ornith models (dense
# 9B + MoE 35B, both qwen3_5 multimodal): it NVFP4's the text linears and keeps
# lm_head, embeddings, MoE router/shared-expert gates, the linear-attention
# layers, and the whole VISION TOWER in bf16 -- so vision stays a working
# feature AND the saved config keeps the multimodal wrapper that vLLM requires
# (vLLM has no text-only Qwen3_5ForCausalLM loader).
_PRESET_IGNORE = {
    "dense": ["lm_head"],
    "ornith": ["lm_head", "re:.*embed_tokens", r"re:.*mlp\.gate$",
               "re:.*shared_expert_gate", "re:.*visual.*", "re:.*vision.*",
               "re:.*linear_attn.*"],
}


def quantize(model_id: str, save_dir: str, recipe_path: str | None, preset: str,
             moe_all_experts: bool, scheme: str, dataset: str,
             calib_samples: int, max_seq: int) -> None:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from llmcompressor import oneshot

    cfg = AutoConfig.from_pretrained(model_id)
    is_multimodal = hasattr(cfg, "vision_config") or any(
        "ForConditionalGeneration" in a
        for a in (getattr(cfg, "architectures", None) or []))
    log(f"[quant] load {model_id}  (multimodal={is_multimodal})")
    processor = None
    if is_multimodal:
        # Load the FULL wrapper (e.g. Qwen3_5ForConditionalGeneration) so the
        # saved config + vision weights are what vLLM expects. AutoModelForCausalLM
        # would strip the wrapper down to a text-only Qwen3_5ForCausalLM that vLLM
        # cannot load. The `ornith` recipe ignores the vision tower -> it stays
        # bf16 and vision remains a working feature.
        from transformers import AutoModelForImageTextToText, AutoProcessor
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto")
        # Save the processor alongside the checkpoint too: vLLM refuses to load a
        # multimodal wrapper without preprocessor_config.json / processor_config.json
        # / video_preprocessor_config.json, and model.save_pretrained does NOT emit
        # them. Without this the NVFP4 output is text-servable only after a manual
        # file copy from the source repo.
        processor = AutoProcessor.from_pretrained(model_id)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype="auto", device_map="cuda")
    tok = AutoTokenizer.from_pretrained(model_id)

    if recipe_path:
        recipe: object = recipe_path  # llm-compressor accepts a recipe YAML path
        log(f"[quant] recipe file: {recipe_path}")
    else:
        from llmcompressor.modifiers.quantization import QuantizationModifier
        ignore = _PRESET_IGNORE[preset]
        recipe = QuantizationModifier(targets="Linear", scheme=scheme, ignore=ignore)
        log(f"[quant] preset={preset} scheme={scheme} ignore={ignore}")

    oneshot_kwargs = dict(model=model, tokenizer=tok, dataset=dataset, recipe=recipe,
                          num_calibration_samples=calib_samples, max_seq_length=max_seq)
    if moe_all_experts:
        oneshot_kwargs["moe_calibrate_all_experts"] = True
    log(f"[quant] oneshot: dataset={dataset} samples={calib_samples} seq={max_seq} "
        f"moe_all_experts={moe_all_experts}")
    oneshot(**oneshot_kwargs)

    model.save_pretrained(save_dir, save_compressed=True)
    tok.save_pretrained(save_dir)
    if processor is not None:
        processor.save_pretrained(save_dir)  # image/video preprocessor configs vLLM needs
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log(f"[quant] QUANTIZE_OK -> {save_dir}")


def validate_vllm(save_dir: str, max_model_len: int) -> list[tuple[str, str]]:
    from vllm import LLM, SamplingParams
    llm = LLM(model=save_dir, max_model_len=max_model_len,
              gpu_memory_utilization=0.55, enforce_eager=True, trust_remote_code=True)
    outs = llm.generate(PROMPTS, SamplingParams(temperature=0.0, max_tokens=48))
    return [(o.prompt, o.outputs[0].text) for o in outs]


def validate_transformers(save_dir: str) -> list[tuple[str, str]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        save_dir, device_map="cuda", torch_dtype="auto")
    tok = AutoTokenizer.from_pretrained(save_dir)
    res: list[tuple[str, str]] = []
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=48, do_sample=False)
        text = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
        res.append((p, text))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--recipe", default=None,
                    help="path to an llm-compressor recipe YAML (overrides --recipe-preset)")
    ap.add_argument("--recipe-preset", choices=sorted(_PRESET_IGNORE), default="dense",
                    help="built-in recipe: dense (ignore lm_head) or ornith "
                         "(also keeps MoE gates, vision tower, linear-attn in bf16)")
    ap.add_argument("--moe-all-experts", action="store_true",
                    help="pass moe_calibrate_all_experts=True to oneshot (MoE models)")
    ap.add_argument("--scheme", default="NVFP4")
    ap.add_argument("--dataset", default="open_platypus")
    ap.add_argument("--calib-samples", type=int, default=128)
    ap.add_argument("--max-seq", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--save-dir", default=os.environ.get("SAVE_DIR", "/tmp/nvfp4-out"))
    ap.add_argument("--serve", choices=["vllm", "transformers", "none"], default="vllm")
    ap.add_argument("--out", default=None,
                    help="optional HF repo id to push the NVFP4 checkpoint to")
    args = ap.parse_args()

    quantize(args.model, args.save_dir, args.recipe, args.recipe_preset,
             args.moe_all_experts, args.scheme, args.dataset,
             args.calib_samples, args.max_seq)

    # In-run validation is BEST-EFFORT: the checkpoint is already saved above, so
    # a validation that cannot run (transformers decompresses NVFP4 -> bf16 and
    # OOMs for models that only fit *because* they're quantized; and it cannot
    # load a multimodal wrapper via AutoModelForCausalLM) must NOT fail the run.
    results = None
    if args.serve == "none":
        log("[serve] validation skipped (--serve none) -- checkpoint saved.")
    else:
        log(f"[serve] in-run validation via {args.serve} (best-effort)")
        try:
            results = (validate_vllm(args.save_dir, args.max_model_len)
                       if args.serve == "vllm" else validate_transformers(args.save_dir))
        except Exception as exc:  # noqa: BLE001
            log(f"[serve] in-run validation could not run: {type(exc).__name__}: {exc}")
            log("[serve] the checkpoint IS saved -- validate serving natively on vLLM: "
                f"point a catalog row at {args.save_dir}, then `make probe-vllm`.")

    if results:
        bad = 0
        for prompt, text in results:
            log("-" * 64)
            log("PROMPT :", prompt.replace("\n", "\\n"))
            log("OUTPUT :", repr(text.strip()[:240]))
            if _is_degenerate(text):
                bad += 1
                log("  -> DEGENERATE")
        log("=" * 64)
        if bad:
            log(f"VALIDATE_FAIL: {bad}/{len(results)} outputs degenerate")
            return 1
        log("VALIDATE_OK: all outputs non-degenerate (eyeball the quality above)")

    if args.out:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.out, repo_type="model", private=True, exist_ok=True)
        log(f"[push] uploading {args.save_dir} -> {args.out}")
        api.upload_folder(folder_path=args.save_dir, repo_id=args.out, repo_type="model")
        log(f"PUSHED -> {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
