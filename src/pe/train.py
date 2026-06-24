"""Train the parallel drafter on cached target features.

The target is never resident here — only its (frozen) embedding and LM head are,
via :func:`pe.target.load_target_heads`. Per-depth cross-entropy is accumulated
with optional sequence partitioning and gradient accumulation across examples,
so long-context training fits a small GPU.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from .config import DrafterConfig, TargetConfig, TrainConfig
from .drafter import ParallelDrafter
from .features import FeatureDataset
from .partition import mtp_backward
from .target import load_heads_from_dump, load_target_heads

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def build_optimizer(params, lr: float, weight_decay: float, use_8bit: bool):
    if use_8bit:
        try:
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=weight_decay)
        except Exception as exc:  # noqa: BLE001
            print(f"8-bit Adam unavailable ({exc}); falling back to AdamW")
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def lr_at(step: int, total: int, peak: float, warmup_ratio: float) -> float:
    warmup = max(1, int(total * warmup_ratio))
    if step < warmup:
        return peak * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * max(0.0, 1.0 - progress)


def train(tcfg: TargetConfig, dcfg: DrafterConfig, tr: TrainConfig, dtype: str = "float32"):
    torch.manual_seed(tr.seed)
    param_dtype = _DTYPES[dtype]

    data = FeatureDataset(tr.feature_cache_dir)
    if data.heads_dump.exists():
        # Lean path: rebuild shared embed/LM head from the dump (no model, no
        # quantization library) — essential for the 7B int4 target.
        heads = load_heads_from_dump(data.heads_dump, tcfg, data.feature_layers)
    else:
        heads = load_target_heads(tcfg)
    if data.feature_dim != heads.feature_dim:
        raise ValueError(
            f"cached feature_dim {data.feature_dim} != target feature_dim {heads.feature_dim}; "
            "regenerate the feature cache for this target."
        )

    drafter = ParallelDrafter.from_target(heads, dcfg).to(tcfg.device, param_dtype)
    drafter.grad_checkpoint = tr.grad_checkpoint
    drafter.train()

    opt = build_optimizer(drafter.trainable_parameters(), tr.lr, tr.weight_decay, tr.use_8bit_adam)
    total_updates = max(1, math.ceil(len(data) * tr.epochs / tr.grad_accum))

    Path(tr.out_dir).mkdir(parents=True, exist_ok=True)
    update, micro, running = 0, 0, 0.0
    opt.zero_grad(set_to_none=True)

    for epoch in range(tr.epochs):
        for input_ids, features in data:
            input_ids = input_ids[: tr.max_seq_len]
            features = features[: tr.max_seq_len]
            if input_ids.shape[0] < 2:
                continue

            running += mtp_backward(
                drafter, input_ids, features, tr.num_segments, loss_scale=1.0 / tr.grad_accum
            )
            micro += 1
            if micro % tr.grad_accum != 0:
                continue

            for g in opt.param_groups:
                g["lr"] = lr_at(update, total_updates, tr.lr, tr.warmup_ratio)
            torch.nn.utils.clip_grad_norm_(drafter.trainable_parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            update += 1
            if update % tr.log_every == 0:
                avg = running / (tr.grad_accum * tr.log_every)
                print(f"epoch {epoch} update {update}/{total_updates} loss {avg:.4f}")
                running = 0.0

        ckpt = Path(tr.out_dir) / "drafter.pt"
        drafter.save_checkpoint(ckpt, target_name=tcfg.model_name)
        print(f"saved {ckpt} after epoch {epoch}")

    return drafter


def _parse():
    d, t = DrafterConfig(), TrainConfig()
    p = argparse.ArgumentParser(description="Train the parallel drafter on cached features.")
    p.add_argument("--target", default=TargetConfig().model_name)
    p.add_argument("--device", default=TargetConfig().device)
    p.add_argument("--dtype", default="float32", choices=list(_DTYPES))
    p.add_argument("--cache-dir", default=str(t.feature_cache_dir))
    p.add_argument("--out-dir", default=str(t.out_dir))
    p.add_argument("--num-layers", type=int, default=d.num_layers)
    p.add_argument("--max-depth", type=int, default=d.max_depth, help="K_train")
    p.add_argument("--max-seq-len", type=int, default=t.max_seq_len)
    p.add_argument("--epochs", type=int, default=t.epochs)
    p.add_argument("--lr", type=float, default=t.lr)
    p.add_argument("--grad-accum", type=int, default=t.grad_accum)
    p.add_argument("--num-segments", type=int, default=t.num_segments)
    p.add_argument("--no-8bit-adam", action="store_true")
    p.add_argument("--no-grad-checkpoint", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    a = _parse()
    tcfg = TargetConfig(model_name=a.target, device=a.device, dtype=a.dtype)
    dcfg = DrafterConfig(num_layers=a.num_layers, max_depth=a.max_depth)
    tr = TrainConfig(
        feature_cache_dir=Path(a.cache_dir),
        out_dir=Path(a.out_dir),
        max_seq_len=a.max_seq_len,
        epochs=a.epochs,
        lr=a.lr,
        grad_accum=a.grad_accum,
        num_segments=a.num_segments,
        use_8bit_adam=not a.no_8bit_adam,
        grad_checkpoint=not a.no_grad_checkpoint,
    )
    train(tcfg, dcfg, tr, dtype=a.dtype)
