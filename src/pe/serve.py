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
from collections.abc import Callable

import torch

from .decode.baselines import vanilla_generate  # noqa: F401 (re-export)
from .decode.chain import propose_chain, propose_chain_sequential
from .decode.result import GenResult
from .decode.tree import propose_tree
from .decode.verify import (
    accept_chain_cached,
    accept_chain_greedy,
    accept_tree_cached,
    accept_tree_greedy,
)
from .drafter import ParallelDrafter
from .masks import to_additive_bias, tree_allow_mask, tree_position_ids
from .target import TargetModel

__all__ = ["generate_speculative", "generate_speculative_cached", "vanilla_generate"]

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
    on_commit: Callable[[list[int]], None] | None = None,
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
        if on_commit is not None:
            on_commit([*committed, bonus])
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


@torch.no_grad()
def generate_speculative_cached(
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
    on_commit: Callable[[list[int]], None] | None = None,
) -> GenResult:
    """One-forward KV-cache speculative decoding.

    A persistent cache holds the confirmed prefix. Each step usually does a *single*
    target forward: the drafter proposes the easy next token at depth 0 (re-rooting
    on the most recent bonus), the target verifies the candidate tree over the cache,
    and the accepted path's KV is kept by index-selecting the cache (rejected branches
    dropped) — so no separate commit pass is needed. The bonus is emitted but left
    "pending" (re-confirmed at depth 0 next step); on the rare step where the drafter
    misses it, a single 1-token forward featurizes it. Net target forwards per token
    approach ``1 / acceptance_length``. Lossless by construction.
    """
    if mode not in _CHAIN_MODES and mode != "tree":
        raise ValueError(f"unknown mode {mode!r}")
    dev = target.device
    seq = list(prompt_ids)
    base = len(seq)

    cache = target.make_cache()
    pos = torch.arange(len(seq), device=dev).unsqueeze(0)
    logits, fused, cache = target.forward_cached(
        torch.tensor(seq, device=dev).unsqueeze(0), cache, pos
    )
    feats = fused[0]  # features for the cache_len featurized tokens
    cache_len = len(seq)
    root_logit = logits[0, -1]
    target_calls, drafter_calls, accepted, steps = 1, 0, 0, 0

    # Seed the first pending bonus so the loop invariant holds (cache_len == len(seq) - 1).
    bonus = int(root_logit.argmax())
    seq.append(bonus)
    if on_commit is not None:
        on_commit([bonus])
    stop = eos_token_id is not None and bonus == eos_token_id
    t0 = time.perf_counter()

    while len(seq) - base < max_new_tokens and not stop:
        ctx = torch.tensor(seq[:cache_len], device=dev)

        if mode in _CHAIN_MODES:
            if mode == "chain":
                drafts = propose_chain(drafter, ctx, feats, k)
                drafter_calls += 1
            else:
                drafts = propose_chain_sequential(drafter, ctx, feats, k)
                drafter_calls += k
            npos = torch.arange(cache_len, cache_len + len(drafts), device=dev).unsqueeze(0)
            vlogits, vfused, cache = target.forward_cached(
                torch.tensor(drafts, device=dev).unsqueeze(0), cache, npos
            )
            target_calls += 1
            acc_tokens, new_bonus = accept_chain_cached(root_logit, vlogits[0], drafts)
            a = len(acc_tokens)
            keep_extra = list(range(cache_len, cache_len + a))
            path_feats = vfused[0][:a]
            new_root = root_logit if a == 0 else vlogits[0][a - 1]
        else:
            node_tokens, parents = propose_tree(drafter, ctx, feats, k, tree_top_k, tree_max_nodes)
            drafter_calls += 1
            allow = tree_allow_mask(parents, cache_len, device=dev)[cache_len:, :]
            bias = to_additive_bias(allow, target.dtype)
            npos = tree_position_ids(parents, cache_len, device=dev)[cache_len:].unsqueeze(0)
            vlogits, vfused, cache = target.forward_cached(
                torch.tensor(node_tokens, device=dev).unsqueeze(0), cache, npos, attention_mask=bias
            )
            target_calls += 1
            path, new_bonus = accept_tree_cached(root_logit, vlogits[0], node_tokens, parents)
            acc_tokens = [node_tokens[t] for t in path]
            keep_extra = [cache_len + t for t in path]
            path_feats = vfused[0][path] if path else vfused[0][:0]
            new_root = root_logit if not path else vlogits[0][path[-1]]

        if acc_tokens:
            # acc_tokens[0] re-confirms the pending bonus (already emitted); keep the
            # accepted path's KV + features, drop the rest, and re-root.
            keep = torch.tensor(list(range(cache_len)) + keep_extra, device=dev)
            target.gather_cache(cache, keep)
            feats = torch.cat([feats, path_feats], dim=0)
            cache_len += len(acc_tokens)
            root_logit = new_root
            emit = [*acc_tokens[1:], int(new_bonus)]
            accepted += len(acc_tokens) - 1
        else:
            # Drafter missed the easy re-prediction; featurize the pending bonus with
            # a single 1-token forward.
            pending = seq[cache_len]
            cache.crop(cache_len)
            bpos = torch.tensor([[cache_len]], device=dev)
            blogits, bfused, cache = target.forward_cached(
                torch.tensor([[pending]], device=dev), cache, bpos
            )
            target_calls += 1
            feats = torch.cat([feats, bfused[0]], dim=0)
            cache_len += 1
            root_logit = blogits[0, -1]
            emit = [int(root_logit.argmax())]

        steps += 1
        for tok in emit:
            seq.append(tok)
            if on_commit is not None:
                on_commit([tok])
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
