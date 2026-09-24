"""Loading a laya checkpoint and fine-tuning it (single GPU).

The loop is laya's own fine-tuning recipe (notebooks/laya_finetune_typed_
decisions_2xT4_kaggle.ipynb, cell 8, laya 0.3.20): soft-target cross-entropy
against the teacher's distribution plus a policy-gradient term scored with
laya's `proper_reward`. Changed for this host:

- one GPU, no DDP;
- bf16 autocast and no GradScaler (the notebook's fp16 + scaler was for T4s);
- `memory_mode`: "lean" freezes the vocabulary embedding and checkpoints
  activations, "fast" trains everything without checkpointing;
- fp32 weights are saved (the notebook saved `.half()`), a checkpoint after
  every epoch;
- a non-finite loss stops the job (exit 6), an out-of-memory error too (exit 5).
"""

from __future__ import annotations

import json
import math
import random
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .contract import ExitCode, JobError, copy_dir_files, write_json_atomic

LR_ENCODER = 2.5e-5
LR_HEAD = 1.0e-4
WEIGHT_DECAY = 0.01
GROUP_SIZE = 4          # noisy samples per item for the policy-gradient baseline
SIGMA_START, SIGMA_END = 0.4, 0.1
W_SPH, W_RPS = 0.75, 1.0
GRAD_CLIP = 1.0
ETA_MIN = 1e-6


def load_model(ckpt_dir: Path, cfg: dict) -> torch.nn.Module:
    """A laya DecisionModel with the checkpoint's weights, fp32, on CPU.

    Same steps as laya.agent.Agent.__init__: build from encoder/config.json
    with no random init, verify the weights fit, load strictly.
    """
    from laya.agent import _verify_compatibility
    from laya.common import build_model
    from safetensors.torch import load_file
    try:
        from transformers.initialization import no_init_weights
    except ImportError:  # transformers 4.x
        from transformers.modeling_utils import no_init_weights

    ckpt_dir = Path(ckpt_dir)
    with no_init_weights():
        model = build_model(cfg, encoder_dir=str(ckpt_dir / "encoder"), pretrained=False)
    weights = load_file(str(ckpt_dir / "model.safetensors"))
    _verify_compatibility(model, cfg, weights, str(ckpt_dir))
    model.load_state_dict(weights, strict=True)
    try:
        model.encoder.config.reference_compile = False
    except AttributeError:
        pass
    return model.float()


def save_checkpoint(model: torch.nn.Module, out_dir: Path, *, base_dir: Path, cfg: dict,
                    meta: dict) -> None:
    """A laya-loadable checkpoint: fp32 weights, encoder config, tokenizer, config."""
    from safetensors.torch import save_file

    out_dir = Path(out_dir)
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "encoder").mkdir(parents=True)
    sd = {k: v.detach().float().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(sd, str(tmp / "model.safetensors"))
    shutil.copyfile(Path(base_dir) / "encoder" / "config.json", tmp / "encoder" / "config.json")
    copy_dir_files(Path(base_dir) / "tokenizer", tmp / "tokenizer")
    write_json_atomic(tmp / "rl_agent_config.json", cfg)
    write_json_atomic(tmp / "checkpoint_meta.json", meta)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)


def collate(items: list[dict], pad_id: int) -> dict:
    from laya.common import collate_items
    return collate_items([items], pad_id)


def rlcd_loss(logits: torch.Tensor, mask: torch.Tensor, target: torch.Tensor,
              qtype: torch.Tensor, sigma: float) -> tuple[torch.Tensor, torch.Tensor]:
    """The notebook's loss: policy gradient on noisy logits + soft cross-entropy."""
    from laya.common import proper_reward

    logits = logits.float()
    k = mask.sum(-1, keepdim=True).float()
    eps = torch.randn((GROUP_SIZE,) + logits.shape, device=logits.device) * sigma * mask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
    with torch.no_grad():
        r = proper_reward(q, target.unsqueeze(0), qtype, mask, w_sph=W_SPH, w_rps=W_RPS)
        adv = r - r.mean(0, keepdim=True)
        adv = adv / (adv.std() + 1e-6)
    logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
    loss_rl = -(adv * logp).mean()
    loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
    return loss_rl + loss_ce, r.mean()


@dataclass
class TrainResult:
    epochs: list[dict] = field(default_factory=list)
    trained_tokens: int = 0
    updates: int = 0
    seconds: float = 0.0
    peak_vram_gib: float | None = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


def configure_memory_mode(model: torch.nn.Module, mode: str) -> None:
    if mode == "lean":
        model.encoder.get_input_embeddings().weight.requires_grad_(False)
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = True


def train(model: torch.nn.Module, items: list[dict], hp: dict, *, device: torch.device,
          seed: int, pad_id: int,
          on_epoch: Callable[[dict], None] = lambda _e: None,
          on_checkpoint: Callable[[int], None] = lambda _e: None) -> TrainResult:
    """Fine-tune `model` in place on `items` (the train split)."""
    if not items:
        raise JobError(ExitCode.DATASET, "the train split has no items")
    seed_everything(seed)
    configure_memory_mode(model, hp["memory_mode"])
    try:
        model.to(device).train()
    except torch.cuda.OutOfMemoryError as e:
        raise JobError(ExitCode.GPU, f"out of GPU memory placing the model: {e}") from None
    mult = hp["learning_rate_multiplier"]
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in named if n.startswith("encoder.")], "lr": LR_ENCODER * mult},
        {"params": [p for n, p in named if not n.startswith("encoder.")], "lr": LR_HEAD * mult},
    ], weight_decay=WEIGHT_DECAY)
    micro = min(hp["micro_batch_size"], hp["batch_size"])
    accum = math.ceil(hp["batch_size"] / micro)
    n_epochs = hp["n_epochs"]
    chunks = math.ceil(len(items) / micro)
    total_updates = math.ceil(chunks / accum) * n_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_updates), eta_min=ETA_MIN)
    use_amp = device.type == "cuda"
    params = [p for _, p in named]
    rng = random.Random(seed)
    result = TrainResult()
    tokens_per_epoch = sum(len(it["ids"]) for it in items)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.monotonic()
    try:
        for epoch in range(n_epochs):
            order = list(items)
            rng.shuffle(order)
            sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * (epoch / max(1, n_epochs - 1))
            loss_sum = reward_sum = 0.0
            optimizer.zero_grad(set_to_none=True)
            for step in range(chunks):
                batch = collate(order[step * micro:(step + 1) * micro], pad_id)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    logits, _act = model(batch["input_ids"].to(device),
                                         batch["attention_mask"].to(device),
                                         batch["marker_pos"].to(device),
                                         batch["marker_mask"].to(device),
                                         batch["qtype"].to(device))
                loss, reward = rlcd_loss(logits, batch["marker_mask"].to(device),
                                         batch["target"].to(device), batch["qtype"].to(device), sigma)
                if not torch.isfinite(loss):
                    raise JobError(ExitCode.DIVERGED,
                                   f"non-finite loss at epoch {epoch + 1}, step {step + 1}")
                (loss / accum).backward()
                loss_sum += float(loss.detach())
                reward_sum += float(reward)
                if (step + 1) % accum == 0 or step + 1 == chunks:
                    torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    result.updates += 1
            result.trained_tokens += tokens_per_epoch
            ep = {"epoch": epoch + 1, "n_epochs": n_epochs, "loss": loss_sum / chunks,
                  "reward": reward_sum / chunks, "lr": scheduler.get_last_lr()[0],
                  "seconds": round(time.monotonic() - t0, 1)}
            result.epochs.append(ep)
            on_epoch(ep)
            on_checkpoint(epoch + 1)
    except torch.cuda.OutOfMemoryError as e:
        raise JobError(ExitCode.GPU, f"out of GPU memory ({hp['memory_mode']} mode): {e}") from None
    result.seconds = round(time.monotonic() - t0, 1)
    if device.type == "cuda":
        result.peak_vram_gib = round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 2)
    model.head_checkpointing = False
    if hp["memory_mode"] == "lean":
        model.encoder.gradient_checkpointing_disable()
    model.eval()
    return result


def checkpoint_cfg(cfg: dict, manifest: dict) -> dict:
    """The run's rl_agent_config.json: the base config, with the dataset's
    token limits (the ONLY source of max_len/head_max_len)."""
    out = json.loads(json.dumps(cfg))
    out["max_len"] = manifest["max_len"]
    out["head_max_len"] = manifest["head_max_len"]
    return out
