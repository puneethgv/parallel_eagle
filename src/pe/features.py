"""Offline feature extraction.

Runs the frozen target once over the training data and writes, per example, the
token ids and the fused hidden states to disk shards. Training then reads these
shards and never loads the target — which is what frees the GPU budget for the
drafter and its optimizer state.

Features are stored in fp16 to halve disk/IO; they are upcast as needed at train
time. A small ``manifest.json`` records shard files and the feature dimension.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import FeatureConfig, TargetConfig
from .target import TargetModel, dump_target_heads


def _example_to_ids(example: dict, tokenizer, max_len: int) -> torch.Tensor | None:
    """Best-effort conversion of a dataset row to a single token sequence."""
    ids = None
    has_template = getattr(tokenizer, "chat_template", None)
    if "messages" in example and has_template:
        res = tokenizer.apply_chat_template(
            example["messages"], tokenize=True, add_generation_prompt=False
        )
        ids = res["input_ids"] if hasattr(res, "keys") else res
    elif "instruction" in example and has_template:
        # Format instruction datasets (e.g. Alpaca) with the chat template so the
        # drafter trains on the same distribution it will decode in.
        user = example["instruction"]
        if example.get("input"):
            user += "\n" + example["input"]
        msgs = [{"role": "user", "content": user}]
        if example.get("output"):
            msgs.append({"role": "assistant", "content": example["output"]})
        res = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
        ids = res["input_ids"] if hasattr(res, "keys") else res
    elif "text" in example:
        ids = tokenizer(example["text"]).input_ids
    elif "prompt" in example:
        text = example["prompt"] + example.get("completion", "")
        ids = tokenizer(text).input_ids
    if not ids or len(ids) < 8:
        return None
    return torch.tensor(ids[:max_len], dtype=torch.long)


def extract_features(fcfg: FeatureConfig, tcfg: TargetConfig) -> Path:
    from datasets import load_dataset

    target = TargetModel(tcfg)
    tokenizer = target.tokenizer
    if tokenizer is None:
        raise RuntimeError("A tokenizer is required for feature extraction.")

    ds = load_dataset(fcfg.dataset, split=fcfg.split, streaming=True)
    out_dir = Path(fcfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dump the (frozen) embedding + LM-head once so training never reloads the
    # target — and never needs the quantization library.
    dump_target_heads(target.model, out_dir / "heads.pt")

    shards: list[str] = []
    buf: list[dict] = []
    n_done = 0

    def flush():
        nonlocal buf
        if not buf:
            return
        name = f"shard_{len(shards):05d}.pt"
        torch.save(buf, out_dir / name)
        shards.append(name)
        buf = []

    for example in ds:
        if n_done >= fcfg.max_examples:
            break
        ids = _example_to_ids(example, tokenizer, fcfg.max_seq_len)
        if ids is None:
            continue
        ids_dev = ids.to(tcfg.device).unsqueeze(0)
        out = target.forward(ids_dev)
        buf.append(
            {
                "input_ids": ids.cpu(),
                "features": out.fused[0].to(torch.float16).cpu(),
            }
        )
        n_done += 1
        if len(buf) >= fcfg.shard_size:
            flush()
        if n_done % 100 == 0:
            print(f"extracted {n_done}/{fcfg.max_examples}")
    flush()

    manifest = {
        "shards": shards,
        "num_examples": n_done,
        "feature_dim": target.feature_dim,
        "feature_layers": list(target.feature_layers),
        "target": tcfg.model_name,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {n_done} examples across {len(shards)} shards to {out_dir}")
    return out_dir / "manifest.json"


class FeatureDataset:
    """Iterates cached ``(input_ids, features)`` examples across shards."""

    def __init__(self, cache_dir: str | Path):
        self.dir = Path(cache_dir)
        manifest = self.dir / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(
                f"No feature cache at {self.dir} (missing manifest.json). "
                "Run `python -m pe.features` first."
            )
        self.manifest = json.loads(manifest.read_text())
        self.feature_dim = self.manifest["feature_dim"]
        self.feature_layers = tuple(self.manifest.get("feature_layers", ()))
        self.heads_dump = self.dir / "heads.pt"

    def __len__(self) -> int:
        return self.manifest["num_examples"]

    def __iter__(self):
        for shard in self.manifest["shards"]:
            for ex in torch.load(self.dir / shard, weights_only=False):
                yield ex["input_ids"], ex["features"]


def _parse_args() -> tuple[FeatureConfig, TargetConfig]:
    p = argparse.ArgumentParser(description="Cache frozen-target features for drafter training.")
    p.add_argument("--target", default=TargetConfig().model_name)
    p.add_argument("--dataset", default=FeatureConfig().dataset)
    p.add_argument("--split", default=FeatureConfig().split)
    p.add_argument("--max-examples", type=int, default=FeatureConfig().max_examples)
    p.add_argument("--max-seq-len", type=int, default=FeatureConfig().max_seq_len)
    p.add_argument("--shard-size", type=int, default=FeatureConfig().shard_size)
    p.add_argument("--out-dir", default=str(FeatureConfig().out_dir))
    p.add_argument("--device", default=TargetConfig().device)
    a = p.parse_args()
    fcfg = FeatureConfig(
        out_dir=Path(a.out_dir),
        dataset=a.dataset,
        split=a.split,
        max_examples=a.max_examples,
        max_seq_len=a.max_seq_len,
        shard_size=a.shard_size,
    )
    tcfg = TargetConfig(model_name=a.target, device=a.device)
    return fcfg, tcfg


if __name__ == "__main__":
    fcfg, tcfg = _parse_args()
    extract_features(fcfg, tcfg)
