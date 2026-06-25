"""Side-by-side demo: naive autoregressive vs. parallel tree speculative decoding.

Runs both on one prompt, prints the (identical) outputs, and a metrics table so
you can sense where speculation helps. With ``--stream`` it prints tokens as they
commit, so naive decoding dribbles token-by-token while the tree emits bursts
(multi-token commits shown in bold).

Note: with the small, lightly-trained example drafter, acceptance is low, so the
wall-clock loop (which recomputes the prefix each step — no KV cache) is not yet
faster than naive. The honest wins here are: identical output (lossless), fewer
target forward passes per token, and visibly chunked emission.
"""

from __future__ import annotations

import argparse

from pe.config import TargetConfig
from pe.decode.baselines import vanilla_generate, vanilla_generate_cached
from pe.drafter import load_drafter
from pe.serve import generate_speculative, generate_speculative_cached
from pe.target import TargetModel

BOLD, RESET = "\033[1m", "\033[0m"


def build_prompt(target: TargetModel, text: str) -> list[int]:
    tok = target.tokenizer
    if getattr(tok, "chat_template", None):
        res = tok.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True
        )
        return list(res["input_ids"] if hasattr(res, "keys") else res)
    return list(tok(text).input_ids)


def streamer(target: TargetModel):
    def cb(toks: list[int]):
        txt = target.tokenizer.decode(toks, skip_special_tokens=True)
        print(f"{BOLD}{txt}{RESET}" if len(toks) > 1 else txt, end="", flush=True)

    return cb


def run(args):
    tcfg = TargetConfig(model_name=args.target, device=args.device, dtype=args.dtype)
    target = TargetModel(tcfg)
    drafter = load_drafter(args.ckpt, target).to(target.device, target.dtype).eval()
    eos = target.tokenizer.eos_token_id
    pid = build_prompt(target, args.prompt)

    vgen = vanilla_generate if args.no_cache else vanilla_generate_cached
    sgen = generate_speculative if args.no_cache else generate_speculative_cached
    spec = dict(
        k=args.k, mode="tree", max_new_tokens=args.max_new_tokens,
        tree_top_k=args.tree_top_k, tree_max_nodes=args.tree_max_nodes, eos_token_id=eos,
    )
    # warm up CUDA kernels
    vgen(target, pid, 4, eos)
    sgen(target, drafter, pid, **{**spec, "max_new_tokens": 4})

    print(f"\nPrompt: {args.prompt}\n" + "=" * 72)

    if args.stream:
        print(f"\n{BOLD}[naive autoregressive]{RESET}")
        v = vgen(target, pid, args.max_new_tokens, eos, on_commit=streamer(target))
        print(f"\n{BOLD}[parallel tree]{RESET}  (bold = multi-token burst)")
        t = sgen(target, drafter, pid, on_commit=streamer(target), **spec)
        print()
    else:
        v = vgen(target, pid, args.max_new_tokens, eos)
        t = sgen(target, drafter, pid, **spec)
        print("\nnaive output :", target.tokenizer.decode(v.output_ids, skip_special_tokens=True))
        print("\ntree  output :", target.tokenizer.decode(t.output_ids, skip_special_tokens=True))

    print("\n" + "=" * 72)
    print(f"identical output (lossless): {v.output_ids == t.output_ids}\n")
    print(f"{'metric':<26}{'naive':>14}{'parallel tree':>16}")
    print("-" * 56)
    rows = [
        ("wall-clock (s)", f"{v.seconds:.2f}", f"{t.seconds:.2f}"),
        ("tokens / sec", f"{v.tokens_per_second:.1f}", f"{t.tokens_per_second:.1f}"),
        ("target forward passes", v.target_calls, t.target_calls),
        ("target calls / token", f"{v.target_calls_per_token:.2f}", f"{t.target_calls_per_token:.2f}"),
        ("acceptance length", f"{v.acceptance_length:.2f}", f"{t.acceptance_length:.2f}"),
    ]
    for name, a, b in rows:
        print(f"{name:<26}{str(a):>14}{str(b):>16}")

    call_cut = (1 - t.target_calls / max(1, v.target_calls)) * 100
    print(
        f"\ntree uses {call_cut:.0f}% fewer target forward passes; "
        f"wall-clock {v.seconds / max(1e-9, t.seconds):.2f}x "
        f"({'faster' if t.seconds < v.seconds else 'slower'})"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Side-by-side naive vs tree speculative decoding.")
    p.add_argument("--prompt", default="Write a Python function that checks if a number is prime.")
    p.add_argument("--target", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--ckpt", default="checkpoints/drafter.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--tree-top-k", type=int, default=4)
    p.add_argument("--tree-max-nodes", type=int, default=24)
    p.add_argument("--stream", action="store_true")
    p.add_argument("--no-cache", action="store_true", help="use the recompute loop instead of KV cache")
    run(p.parse_args())
