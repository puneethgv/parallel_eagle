"""Parallel chain drafting: one drafter pass, top-1 per depth.

``pending`` is the number of leading prediction depths that correspond to
already-confirmed tokens (the previous step's bonus, whose feature the target has
not produced yet); those are skipped so only fresh tokens are returned.
"""

from __future__ import annotations

import torch

from ..drafter import ParallelDrafter


def propose_chain(
    drafter: ParallelDrafter,
    context_ids: torch.Tensor,
    context_feats: torch.Tensor,
    k: int,
    pending: int = 0,
) -> list[int]:
    logits = drafter.draft_logits(context_ids, context_feats, k + pending)
    return [int(t) for t in logits.argmax(-1)[pending:]]


def propose_chain_sequential(
    drafter: ParallelDrafter,
    context_ids: torch.Tensor,
    context_feats: torch.Tensor,
    k: int,
    pending: int = 0,
) -> list[int]:
    """Same chain, produced by sequential drafter passes (baseline)."""
    logits = drafter.draft_logits_sequential(context_ids, context_feats, k + pending)
    return [int(t) for t in logits.argmax(-1)[pending:]]


def propose_chain_sampling(
    drafter: ParallelDrafter,
    context_ids: torch.Tensor,
    context_feats: torch.Tensor,
    k: int,
    temperature: float,
    generator: torch.Generator | None = None,
) -> tuple[list[int], torch.Tensor]:
    """Sample a chain of ``k`` tokens from the drafter's per-depth distributions.

    Returns the sampled tokens and the full per-depth distributions ``q`` (needed
    for the speculative-sampling residual)."""
    logits = drafter.draft_logits(context_ids, context_feats, k)
    q = torch.softmax(logits.float() / temperature, dim=-1)  # (k, V)
    tokens = torch.multinomial(q, 1, generator=generator).squeeze(-1)
    return [int(t) for t in tokens], q
