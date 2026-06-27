"""Diagnose the acceptance ceiling: per-depth top-1 accuracy of the parallel drafter.

Acceptance length in the decode loop is bounded by how often each speculated depth's
predicted token equals the target's own greedy next token. The drafter predicts all
K depths in a single forward from each anchor (parallel drafting), so a teacher-forced
per-depth top-1 match is exactly the per-depth draft the live loop emits. Measuring it
on the target's *own* greedy continuations separates the two possible causes of low
acceptance:

  * weak drafter         -> low depth-0 top-1 (it can't even predict 1 token ahead);
  * inefficient loop      -> high depth-0 top-1 but realized acceptance far below it.

It prints per-depth top-1, the implied chain-acceptance ceiling (1 + sum of the
cumulative products), and the realized acceptance from the shipped decode loop.
"""

from __future__ import annotations

import argparse

import torch

from pe.config import TargetConfig
from pe.decode.baselines import vanilla_generate_cached
from pe.drafter import load_drafter
from pe.masks import allow_from_slots, to_additive_bias
from pe.partition import _segment_slots
from pe.serve import generate_speculative_cached
from pe.target import TargetModel

PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number.",
    "Explain why the sky appears blue in two sentences.",
    "Solve: a train travels 60 km in 1.5 hours. What is its average speed?",
    "Write a haiku about autumn leaves.",
    "Reverse the string 'speculative' and explain your steps.",
    "What is the time complexity of binary search and why?",
    "Give three tips for writing clear commit messages.",
    "Translate 'good morning, how are you?' into French.",
    "Implement bubble sort in Python with a short comment.",
    "Summarize what a transformer attention layer does.",
]


def _prompt_ids(target: TargetModel, text: str) -> list[int]:
    tok = target.tokenizer
    if getattr(tok, "chat_template", None):
        res = tok.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True
        )
        return list(res["input_ids"] if hasattr(res, "keys") else res)
    return list(tok(text).input_ids)


@torch.no_grad()
def _slot_correct(target, drafter, full_ids, prompt_len):
    """Per-depth (correct, total) counts over the response region of one sequence."""
    dev = drafter.device
    ids = torch.tensor(full_ids, device=dev)
    n = ids.shape[0]
    out = target.forward(ids.unsqueeze(0))
    features = out.fused[0]
    teacher_labels = out.logits[0].argmax(-1)  # argmax at p predicts token p+1

    k = drafter.max_depth
    anchor, depth, n_real = _segment_slots(0, n, k, dev)
    real_x = drafter.real_input(ids[:n_real], features[:n_real])
    x = real_x if k == 1 else torch.cat([real_x, drafter.mask_input(anchor.shape[0] - n_real)])
    pos = (anchor + depth).unsqueeze(0)
    bias = to_additive_bias(allow_from_slots(anchor, depth), drafter.param_dtype)
    hidden = drafter(x.unsqueeze(0), pos, bias)[0]
    pred = drafter.lm_head(hidden).argmax(-1)

    tgt = anchor + 1 + depth
    valid = (tgt < n) & (tgt >= prompt_len)
    # invalid slots are dropped by `valid`; clamp the gather index so they don't OOB
    correct = pred == teacher_labels[(tgt - 1).clamp(min=0, max=n - 1)]
    per_depth = {}
    for d in range(k):
        sel = valid & (depth == d)
        per_depth[d] = (int((correct & sel).sum()), int(sel.sum()))
    return per_depth


def run(args):
    tcfg = TargetConfig(model_name=args.target, device=args.device, dtype=args.dtype)
    target = TargetModel(tcfg)
    drafter = load_drafter(args.ckpt, target).to(target.device, target.dtype).eval()
    eos = target.tokenizer.eos_token_id
    k = drafter.max_depth

    totals = {d: [0, 0] for d in range(k)}
    realized, realized_tps, van_tps = [], [], []
    for text in PROMPTS:
        p = _prompt_ids(target, text)
        ref = vanilla_generate_cached(target, p, args.max_new_tokens, eos)
        full = list(p) + list(ref.output_ids)
        for d, (c, t) in _slot_correct(target, drafter, full, len(p)).items():
            totals[d][0] += c
            totals[d][1] += t
        spec = generate_speculative_cached(
            target, drafter, p, k=args.k, mode="tree",
            max_new_tokens=args.max_new_tokens, tree_top_k=args.tree_top_k,
            tree_max_nodes=args.tree_max_nodes, eos_token_id=eos,
        )
        realized.append(spec.acceptance_length)
        realized_tps.append(spec.tokens_per_second)
        van_tps.append(ref.tokens_per_second)

    print("\n=== per-depth top-1 accuracy (teacher-forced, response region) ===", flush=True)
    print(f"{'depth':>5}  {'top1':>7}  {'cumprod':>8}  {'n_slots':>8}")
    cum, ceiling = 1.0, 1.0  # depth-0 is the always-committed next token
    for d in range(k):
        c, t = totals[d]
        acc = c / t if t else 0.0
        cum *= acc
        ceiling += cum
        print(f"{d:>5}  {acc:>7.3f}  {cum:>8.3f}  {t:>8}")

    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    print("\n=== acceptance ===", flush=True)
    print(f"implied chain ceiling (1 + sum cumprod): {ceiling:.3f}")
    print(f"realized (tree decode loop):             {mean(realized):.3f}")
    print(f"vanilla tok/s {mean(van_tps):.2f}   tree tok/s {mean(realized_tps):.2f}   "
          f"speedup {mean(realized_tps)/mean(van_tps):.3f}")


def _parse():
    p = argparse.ArgumentParser(description="Diagnose drafter per-depth acceptance.")
    p.add_argument("--target", default="Qwen/Qwen2.5-14B-Instruct")
    p.add_argument("--ckpt", default="checkpoints/drafter.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--k", type=int, default=7)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--tree-top-k", type=int, default=3)
    p.add_argument("--tree-max-nodes", type=int, default=24)
    return p.parse_args()


if __name__ == "__main__":
    torch.manual_seed(0)
    run(_parse())
