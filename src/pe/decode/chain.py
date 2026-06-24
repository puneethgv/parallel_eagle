"""Parallel chain drafting: one drafter pass, top-1 per depth."""

from __future__ import annotations

import torch

from ..drafter import ParallelDrafter


def propose_chain(
    drafter: ParallelDrafter, context_ids: torch.Tensor, context_feats: torch.Tensor, k: int
) -> list[int]:
    logits = drafter.draft_logits(context_ids, context_feats, k)  # (k, V)
    return [int(t) for t in logits.argmax(-1)]


def propose_chain_sequential(
    drafter: ParallelDrafter, context_ids: torch.Tensor, context_feats: torch.Tensor, k: int
) -> list[int]:
    """Same chain, produced by ``k`` sequential drafter passes (baseline)."""
    logits = drafter.draft_logits_sequential(context_ids, context_feats, k)
    return [int(t) for t in logits.argmax(-1)]
