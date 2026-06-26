"""Benchmark decode strategies: acceptance length, throughput, call efficiency.

For each speculation depth K, every strategy is run on the same held-out prompts
and compared against plain greedy decoding (the wall-clock denominator and the
losslessness reference). Results are written to ``results/benchmark.csv``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean

import torch

from pe.config import TargetConfig
from pe.decode.baselines import vanilla_generate, vanilla_generate_cached
from pe.drafter import load_drafter
from pe.serve import generate_speculative, generate_speculative_cached
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

SPECULATIVE = {"sequential": "sequential", "chain": "chain", "tree": "tree"}


def build_prompts(target: TargetModel) -> list[list[int]]:
    tok = target.tokenizer
    out = []
    for p in PROMPTS:
        if getattr(tok, "chat_template", None):
            res = tok.apply_chat_template(
                [{"role": "user", "content": p}], add_generation_prompt=True, tokenize=True
            )
            ids = res["input_ids"] if hasattr(res, "keys") else res
        else:
            ids = tok(p).input_ids
        out.append(list(ids))
    return out


def run(args) -> Path:
    tcfg = TargetConfig(model_name=args.target, device=args.device, dtype=args.dtype)
    target = TargetModel(tcfg)
    drafter = load_drafter(args.ckpt, target).to(target.device, target.dtype).eval()
    eos = target.tokenizer.eos_token_id
    prompts = build_prompts(target)

    # The cached one-forward loop is what ships; benchmark it (and its matching
    # KV-cached vanilla denominator) by default. --recompute selects the prefix-
    # recompute loop instead (no KV cache) for the lossless-equivalence reference.
    vgen = vanilla_generate if args.recompute else vanilla_generate_cached
    sgen = generate_speculative if args.recompute else generate_speculative_cached

    # warm up CUDA kernels so timings are representative
    _ = vgen(target, prompts[0], 8, eos)
    _ = sgen(target, drafter, prompts[0], k=args.k_values[0], mode="tree",
             max_new_tokens=8, tree_top_k=args.tree_top_k,
             tree_max_nodes=args.tree_max_nodes, eos_token_id=eos)

    # vanilla is independent of K; compute once per prompt
    refs = [vgen(target, p, args.max_new_tokens, eos) for p in prompts]
    van_tps = mean(r.tokens_per_second for r in refs)

    rows = [
        {
            "method": "vanilla",
            "k": "-",
            "acceptance_length": 1.0,
            "tokens_per_second": round(van_tps, 2),
            "speedup_vs_vanilla": 1.0,
            "target_calls_per_token": round(mean(r.target_calls_per_token for r in refs), 3),
            "drafter_calls_per_token": 0.0,
            "lossless_match_rate": 1.0,
        }
    ]

    for k in args.k_values:
        for name, mode in SPECULATIVE.items():
            al, tps, tcpt, dcpt, match = [], [], [], [], []
            for p, ref in zip(prompts, refs, strict=True):
                res = sgen(
                    target, drafter, p, k=k, mode=mode, max_new_tokens=args.max_new_tokens,
                    tree_top_k=args.tree_top_k, tree_max_nodes=args.tree_max_nodes,
                    eos_token_id=eos,
                )
                al.append(res.acceptance_length)
                tps.append(res.tokens_per_second)
                tcpt.append(res.target_calls_per_token)
                dcpt.append(res.drafter_calls / max(1, res.num_generated))
                match.append(1.0 if res.output_ids == ref.output_ids else 0.0)
            rows.append(
                {
                    "method": name,
                    "k": k,
                    "acceptance_length": round(mean(al), 3),
                    "tokens_per_second": round(mean(tps), 2),
                    "speedup_vs_vanilla": round(mean(tps) / van_tps, 3),
                    "target_calls_per_token": round(mean(tcpt), 3),
                    "drafter_calls_per_token": round(mean(dcpt), 3),
                    "lossless_match_rate": round(mean(match), 3),
                }
            )
            print(f"k={k:<2} {name:<11} AL={rows[-1]['acceptance_length']:<6} "
                  f"speedup={rows[-1]['speedup_vs_vanilla']:<6} "
                  f"lossless={rows[-1]['lossless_match_rate']}")

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "benchmark.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {csv_path}")
    return csv_path


def _parse():
    p = argparse.ArgumentParser(description="Benchmark decode strategies.")
    p.add_argument("--target", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--ckpt", default="checkpoints/drafter.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--k-values", type=int, nargs="+", default=[3, 5])
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--tree-top-k", type=int, default=3)
    p.add_argument("--tree-max-nodes", type=int, default=24)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--recompute", action="store_true",
                   help="use the prefix-recompute loop instead of the shipped KV-cache loop")
    return p.parse_args()


if __name__ == "__main__":
    torch.manual_seed(0)
    run(_parse())
