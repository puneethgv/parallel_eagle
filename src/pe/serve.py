"""Speculative generation loop wiring the drafter and the frozen target.

One iteration:
  1. run the target over the current sequence to refresh fused features
     (and the immediate next-token distribution);
  2. propose candidates with the drafter — a chain (top-1 per depth), the same
     chain produced by sequential passes, or a dynamic tree;
  3. run the target once over ``[sequence | candidates]`` and accept the longest
     target-consistent prefix/path, plus one bonus token.

The two target passes per iteration keep the loop simple and obviously lossless
without KV-cache bookkeeping; the reported acceptance length and target-calls-
per-token are independent of that implementation choice.
"""

from __future__ import annotations

import time

import torch

from .decode.baselines import vanilla_generate  # re-exported for convenience
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
    tree_max_nodes: int = 48,
    eos_token_id: int | None = None,
) -> GenResult:
    if mode not in _CHAIN_MODES and mode != "tree":
        raise ValueError(f"unknown mode {mode!r}")
    dev = target.device
    seq = list(prompt_ids)
    base = len(seq)
    steps = target_calls = drafter_calls = accepted = 0
    stop = False
    t0 = time.perf_counter()

    while len(seq) - base < max_new_tokens and not stop:
        prefix_len = len(seq)
        feats = target.forward(torch.tensor(seq, device=dev).unsqueeze(0)).fused[0]
        target_calls += 1
        ctx = torch.tensor(seq, device=dev)

        if mode in _CHAIN_MODES:
            if mode == "chain":
                drafts = propose_chain(drafter, ctx, feats, k)
                drafter_calls += 1
            else:
                drafts = propose_chain_sequential(drafter, ctx, feats, k)
                drafter_calls += k
            vout = target.forward(torch.tensor(seq + drafts, device=dev).unsqueeze(0))
            target_calls += 1
            acc, bonus = accept_chain_greedy(vout.logits[0], prefix_len, drafts)
        else:
            node_tokens, parents = propose_tree(drafter, ctx, feats, k, tree_top_k, tree_max_nodes)
            drafter_calls += 1
            bias = to_additive_bias(tree_allow_mask(parents, prefix_len, device=dev), target.dtype)
            pos = tree_position_ids(parents, prefix_len, device=dev).unsqueeze(0)
            vout = target.forward(
                torch.tensor(seq + node_tokens, device=dev).unsqueeze(0),
                attention_mask=bias,
                position_ids=pos,
            )
            target_calls += 1
            acc, bonus = accept_tree_greedy(vout.logits[0], prefix_len, node_tokens, parents)

        accepted += len(acc)
        steps += 1
        for tok in [*acc, bonus]:
            seq.append(tok)
            if eos_token_id is not None and tok == eos_token_id:
                stop = True
                break
            if len(seq) - base >= max_new_tokens:
                break

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
