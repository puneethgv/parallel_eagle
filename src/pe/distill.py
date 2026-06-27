"""Self-distillation feature generation.

The drafter is rewarded at decode time for predicting the target's *argmax*, but
features cached over human-written responses (``pe.features``) carry the human
token as the label — a different objective. This module closes that gap: for each
prompt it runs the target's **own** greedy generation, then caches fused features
over ``[prompt | target response]``. The next-token label in the response region
is therefore the target's argmax by construction, so per-depth cross-entropy
directly optimizes acceptance length.

Output is the same shard layout as :mod:`pe.features` (so :class:`FeatureDataset`
and :mod:`pe.train` read it unchanged), with one addition: each example records
``prompt_len`` so training can mask the prompt region — where the label is the
template/user text, not a target argmax — out of the loss.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import DistillConfig, TargetConfig
from .decode.baselines import vanilla_generate_cached
from .target import TargetModel, dump_target_heads


def _example_to_prompt_ids(example: dict, tokenizer, max_len: int) -> torch.Tensor | None:
    """Convert a dataset row to the *prompt only*, with a generation prompt appended.

    Mirrors :func:`pe.features._example_to_ids` but keeps only the user turn(s) and
    sets ``add_generation_prompt=True`` so the target continues as the assistant.
    """
    ids = None
    has_template = getattr(tokenizer, "chat_template", None)
    if "messages" in example and has_template:
        msgs = [m for m in example["messages"] if m.get("role") != "assistant"]
        if not msgs:
            return None
        res = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        ids = res["input_ids"] if hasattr(res, "keys") else res
    elif "instruction" in example and has_template:
        user = example["instruction"]
        if example.get("input"):
            user += "\n" + example["input"]
        msgs = [{"role": "user", "content": user}]
        res = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        ids = res["input_ids"] if hasattr(res, "keys") else res
    elif "prompt" in example:
        ids = tokenizer(example["prompt"]).input_ids
    if not ids or len(ids) < 4:
        return None
    return torch.tensor(ids[:max_len], dtype=torch.long)


@torch.no_grad()
def _featurize(target: TargetModel, full: list[int], prompt_len: int):
    """Run the frozen target once over the unpadded ``[prompt | response]`` and
    return ``(full_ids, features_fp16_cpu, prompt_len, labels)``.

    Unpadded so the cached features carry the exact positions/RoPE the drafter sees
    at inference. ``labels[p]`` is the target's argmax *at* position ``p`` (its
    teacher-forced next-token prediction) from the **same** forward as the features —
    so features and labels are mutually consistent. Training derives the supervision
    from these labels (not the generated token), which makes the objective exactly
    "predict the target's argmax" regardless of how the sequence was produced (and
    so robust to batched left-padding / cached-vs-full numerical ties)."""
    ids_dev = torch.tensor(full, device=target.device).unsqueeze(0)
    out = target.forward(ids_dev)
    feats = out.fused[0].to(torch.float16).cpu()
    labels = out.logits[0].argmax(-1).to(torch.long).cpu()
    return torch.tensor(full, dtype=torch.long), feats, prompt_len, labels


@torch.no_grad()
def distill_example(
    target: TargetModel,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None,
) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor] | None:
    """Generate the target's greedy continuation of ``prompt_ids`` and featurize the
    full ``[prompt | response]`` sequence.

    Returns ``(full_ids, features_fp16_cpu, prompt_len, labels)`` or ``None`` if the
    target produced no new tokens (e.g. immediate EOS), which carries no training
    signal.
    """
    prompt = [int(t) for t in prompt_ids.tolist()]
    res = vanilla_generate_cached(target, prompt, max_new_tokens, eos_token_id=eos_token_id)
    if not res.output_ids:
        return None
    return _featurize(target, prompt + res.output_ids, len(prompt))


@torch.no_grad()
def distill_batch(
    target: TargetModel,
    prompts: list[torch.Tensor],
    max_new_tokens: int,
    eos_token_id: int | None,
) -> list[tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]]:
    """Batched greedy generation for a list of prompts, then per-example unpadded
    featurization.

    Only the autoregressive generation (the expensive part — ``max_new_tokens``
    forwards) is batched, via the target's HF ``.generate`` with left padding;
    feature extraction stays per-example and unpadded so positions are inference-
    exact. Left-padding can flip the occasional near-tie token versus unpadded
    decoding, but training supervises on the unpadded teacher-forced argmax (see
    :func:`_featurize`), so the per-example labels are exact regardless. Examples
    with no generated tokens are dropped.
    """
    tok = target.tokenizer
    dev = target.device
    pad_id = getattr(tok, "pad_token_id", None)
    if pad_id is None:
        pad_id = eos_token_id if eos_token_id is not None else 0

    maxlen = max(p.shape[0] for p in prompts)
    input_ids = torch.full((len(prompts), maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((len(prompts), maxlen), dtype=torch.long)
    for i, p in enumerate(prompts):  # left-pad so all real tokens are right-aligned
        input_ids[i, maxlen - p.shape[0] :] = p
        attn[i, maxlen - p.shape[0] :] = 1

    gen = target.model.generate(
        input_ids=input_ids.to(dev),
        attention_mask=attn.to(dev),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=pad_id,
        eos_token_id=eos_token_id,
    )

    out: list[tuple[torch.Tensor, torch.Tensor, int]] = []
    for i, p in enumerate(prompts):
        resp: list[int] = []
        for t in gen[i, maxlen:].tolist():  # generated tokens follow the padded prompt block
            if eos_token_id is not None and t == eos_token_id:
                break
            resp.append(int(t))
        if not resp:
            continue
        out.append(_featurize(target, [int(x) for x in p.tolist()] + resp, p.shape[0]))
    return out


def build_distill_dataset(dcfg: DistillConfig, tcfg: TargetConfig, batch_size: int = 1) -> Path:
    from datasets import load_dataset

    target = TargetModel(tcfg)
    tokenizer = target.tokenizer
    if tokenizer is None:
        raise RuntimeError("A tokenizer is required for self-distillation.")
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    ds = load_dataset(dcfg.dataset, split=dcfg.split, streaming=True)
    out_dir = Path(dcfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
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

    def record(results):
        nonlocal n_done
        for full_ids, feats, prompt_len, labels in results:
            buf.append(
                {
                    "input_ids": full_ids,
                    "features": feats,
                    "prompt_len": prompt_len,
                    "labels": labels,
                }
            )
            n_done += 1
            if len(buf) >= dcfg.shard_size:
                flush()
            if n_done % 50 == 0:
                print(f"distilled {n_done}/{dcfg.max_examples}")

    pending: list[torch.Tensor] = []
    for example in ds:
        if n_done + len(pending) >= dcfg.max_examples:
            break
        prompt_ids = _example_to_prompt_ids(example, tokenizer, dcfg.max_prompt_len)
        if prompt_ids is None:
            continue
        if batch_size <= 1:
            out = distill_example(target, prompt_ids, dcfg.max_new_tokens, eos_token_id)
            record([out] if out is not None else [])
            continue
        pending.append(prompt_ids)
        if len(pending) >= batch_size:
            record(distill_batch(target, pending, dcfg.max_new_tokens, eos_token_id))
            pending = []
    if pending:  # final partial batch
        record(distill_batch(target, pending, dcfg.max_new_tokens, eos_token_id))
    flush()

    manifest = {
        "shards": shards,
        "num_examples": n_done,
        "feature_dim": target.feature_dim,
        "feature_layers": list(target.feature_layers),
        "target": tcfg.model_name,
        "self_distilled": True,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {n_done} self-distilled examples across {len(shards)} shards to {out_dir}")
    return out_dir / "manifest.json"


def _parse_args() -> tuple[DistillConfig, TargetConfig, int]:
    d, t = DistillConfig(), TargetConfig()
    p = argparse.ArgumentParser(description="Generate self-distilled drafter training data.")
    p.add_argument("--target", default=t.model_name)
    p.add_argument("--dataset", default=d.dataset)
    p.add_argument("--split", default=d.split)
    p.add_argument("--max-examples", type=int, default=d.max_examples)
    p.add_argument("--max-prompt-len", type=int, default=d.max_prompt_len)
    p.add_argument("--max-new-tokens", type=int, default=d.max_new_tokens)
    p.add_argument("--shard-size", type=int, default=d.shard_size)
    p.add_argument("--batch-size", type=int, default=1, help="batch greedy generation (>1)")
    p.add_argument("--out-dir", default=str(d.out_dir))
    p.add_argument("--device", default=t.device)
    a = p.parse_args()
    dcfg = DistillConfig(
        out_dir=Path(a.out_dir),
        dataset=a.dataset,
        split=a.split,
        max_examples=a.max_examples,
        max_prompt_len=a.max_prompt_len,
        max_new_tokens=a.max_new_tokens,
        shard_size=a.shard_size,
    )
    tcfg = TargetConfig(model_name=a.target, device=a.device)
    return dcfg, tcfg, a.batch_size


if __name__ == "__main__":
    dcfg, tcfg, batch_size = _parse_args()
    build_distill_dataset(dcfg, tcfg, batch_size=batch_size)
