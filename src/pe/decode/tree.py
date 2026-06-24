"""Parallel dynamic tree drafting.

From a single drafter pass we have a distribution at each of the ``k`` prediction
depths. Rather than committing to the top-1 chain, we keep a beam of the
highest joint-probability continuations: at each depth the top candidates are
attached to every surviving parent and the beam is re-pruned by cumulative
log-probability. This allocates branching toward the depths/contexts where the
drafter is uncertain (competitive candidates) and produces a compact tree that
the target verifies in one pass. Total nodes are bounded by ``k * beam_width``.
"""

from __future__ import annotations

import torch

from ..drafter import ParallelDrafter


def propose_tree(
    drafter: ParallelDrafter,
    context_ids: torch.Tensor,
    context_feats: torch.Tensor,
    k: int,
    top_k: int,
    max_nodes: int,
    pending: int = 0,
) -> tuple[list[int], list[int]]:
    """Return ``(node_tokens, parents)`` in topological order (parent before child).

    ``parents[t] == -1`` means the node hangs off the confirmed prefix. ``pending``
    leading depths (already-confirmed tokens) are skipped before building the tree.
    """
    logits = drafter.draft_logits(context_ids, context_feats, k + pending)
    logprobs = torch.log_softmax(logits.float(), dim=-1)[pending:]  # (k, V)

    beam_width = max(1, min(top_k, max_nodes // max(1, k)))
    beam: list[tuple[int, float]] = [(-1, 0.0)]  # (parent node index, cumulative logprob)
    node_tokens: list[int] = []
    parents: list[int] = []

    for d in range(k):
        topv, topi = logprobs[d].topk(beam_width)
        topv, topi = topv.tolist(), topi.tolist()
        cand = [
            (par, tok, cum + lp) for (par, cum) in beam for lp, tok in zip(topv, topi, strict=True)
        ]
        cand.sort(key=lambda c: c[2], reverse=True)
        cand = cand[:beam_width]

        beam = []
        for par, tok, cum in cand:
            idx = len(node_tokens)
            node_tokens.append(tok)
            parents.append(par)
            beam.append((idx, cum))

    return node_tokens, parents
