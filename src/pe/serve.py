"""Speculative generation loop wiring the drafter and the frozen target.

After an initial prefill, each iteration is a **single** target forward:
  1. the drafter proposes candidates from the last token whose target feature is
     known — a chain (top-1 per depth), the same chain via sequential passes, or a
     dynamic tree;
  2. the target scores ``[sequence | candidates]`` in one pass; the longest
     target-consistent prefix/path is accepted, plus one bonus token.

The verification pass also yields the target's hidden states for the accepted
positions, which are carried forward as the next step's drafter features — so no
extra target pass is needed. The freshly produced bonus token has no feature yet;
it is re-fed (as part of the prefix) on the next iteration, which is why the
drafter is asked for ``pending + k`` depths and the first ``pending`` are skipped.

Greedy acceptance makes the output identical to plain decoding (lossless).
"""

from __future__ import annotations

import time

import torch

from .decode.baselines import vanilla_generate  # noqa: F401 (re-export)
from .decode.chain import propose_chain, propose_chain_sequential
from .decode.result import GenResult
from .decode.tree import propose_tree
from .decode.verify import accept_chain_greedy, accept_tree_greedy
from .drafter import ParallelDrafter
from .masks import to_additive_bias, tree_allow_mask, tree_position_ids
from .target import TargetModel

__all__ = ["generate_speculative", "vanilla_generate"]

_CHAIN_MODES = {"chain", "sequential"}


@torch.no_grad()
def generate_speculative(
    target: TargetModel,
    drafter: ParallelDrafter,
    prompt_ids: list[int],
    *,
    k: int,
    mode: str = "tree",
    max_new_tokens: int = 256,
    tree_top_k: int = 3,
    tree_max_nodes: int = 24,
    eos_token_id: int | None = None,
) -> GenResult:
    if mode not in _CHAIN_MODES and mode != "tree":
        raise ValueError(f"unknown mode {mode!r}")
    dev = target.device
    seq = list(prompt_ids)
    base = len(seq)

    # prefill: features for every prompt token
    feats = target.forward(torch.tensor(seq, device=dev).unsqueeze(0)).fused[0]
    featured_len = len(seq)
    target_calls, drafter_calls, accepted, steps = 1, 0, 0, 0
    stop = False
    t0 = time.perf_counter()

    while len(seq) - base < max_new_tokens and not stop:
        g = len(seq) - featured_len  # confirmed-but-unfeatured tail (bonus); 0 on first step
        prefix_len = len(seq)
        ctx = torch.tensor(seq[:featured_len], device=dev)

        if mode in _CHAIN_MODES:
            if mode == "chain":
                fresh = propose_chain(drafter, ctx, feats, k, pending=g)
                drafter_calls += 1
            else:
                fresh = propose_chain_sequential(drafter, ctx, feats, k, pending=g)
                drafter_calls += k + g
            vout = target.forward(torch.tensor(seq + fresh, device=dev).unsqueeze(0))
            target_calls += 1
            committed, bonus = accept_chain_greedy(vout.logits[0], prefix_len, fresh)
            num_acc = len(committed)
            feats = vout.fused[0][: prefix_len + num_acc]
        else:
            node_tokens, parents = propose_tree(
                drafter, ctx, feats, k, tree_top_k, tree_max_nodes, pending=g
            )
            drafter_calls += 1
            bias = to_additive_bias(tree_allow_mask(parents, prefix_len, device=dev), target.dtype)
            pos = tree_position_ids(parents, prefix_len, device=dev).unsqueeze(0)
            vout = target.forward(
                torch.tensor(seq + node_tokens, device=dev).unsqueeze(0),
                attention_mask=bias,
                position_ids=pos,
            )
            target_calls += 1
            path, bonus = accept_tree_greedy(vout.logits[0], prefix_len, node_tokens, parents)
            committed = [node_tokens[t] for t in path]
            num_acc = len(committed)
            idx = list(range(prefix_len)) + [prefix_len + t for t in path]
            feats = vout.fused[0][idx]

        accepted += num_acc
        steps += 1
        seq.extend(committed)
        seq.append(bonus)
        featured_len = prefix_len + num_acc
        if eos_token_id is not None and (bonus == eos_token_id or eos_token_id in committed):
            stop = True

    if dev.type == "cuda":
        torch.cuda.synchronize()
    return GenResult(
        output_ids=seq[base : base + max_new_tokens],
        steps=steps,
        target_calls=target_calls,
        drafter_calls=drafter_calls,
        accepted_tokens=accepted,
        num_generated=min(len(seq) - base, max_new_tokens),
        seconds=time.perf_counter() - t0,
    )
