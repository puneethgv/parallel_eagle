"""Lossless verification: accept the longest target-consistent draft prefix/path.

Greedy acceptance compares each draft token against the target's argmax at the
appropriate position; the bonus token is the target's argmax at the first
mismatch. Because every accepted token *is* the target's greedy choice and the
bonus is too, the emitted sequence is identical to plain greedy decoding,
regardless of how the drafts were proposed.
"""

from __future__ import annotations

from collections import defaultdict

import torch


def accept_chain_greedy(
    logits: torch.Tensor, prefix_len: int, draft_tokens: list[int]
) -> tuple[list[int], int]:
    """``logits`` covers ``[prefix | draft_tokens]`` (shape ``(P + K, V)``).

    Returns the accepted draft tokens and the bonus token (target's correction).
    """
    p, k = prefix_len, len(draft_tokens)
    truths = logits[p - 1 : p - 1 + k].argmax(-1)
    a = 0
    for m in range(k):
        if int(draft_tokens[m]) != int(truths[m]):
            break
        a += 1
    bonus = int(logits[p - 1 + a].argmax())
    return [int(t) for t in draft_tokens[:a]], bonus


def accept_tree_greedy(
    logits: torch.Tensor, prefix_len: int, node_tokens: list[int], parents: list[int]
) -> tuple[list[int], int]:
    """``logits`` covers ``[prefix | tree nodes]``. Follow the target's argmax down
    the tree as far as a matching child exists.

    Returns the accepted path as a list of **node indices** (so the caller can both
    read the tokens and gather those nodes' hidden states) and the bonus token.
    """
    p = prefix_len
    children: dict[int, list[int]] = defaultdict(list)
    for t, par in enumerate(parents):
        children[par].append(t)

    path: list[int] = []
    cur_pos, cur_parent = p - 1, -1
    while True:
        truth = int(logits[cur_pos].argmax())
        nxt = next((t for t in children.get(cur_parent, []) if int(node_tokens[t]) == truth), None)
        if nxt is None:
            return path, truth
        path.append(nxt)
        cur_parent, cur_pos = nxt, p + nxt


def accept_chain_cached(
    root_logit: torch.Tensor, node_logits: torch.Tensor, draft_tokens: list[int]
) -> tuple[list[int], int]:
    """Cached variant: ``root_logit`` (V,) predicts the token after the last
    confirmed token; ``node_logits`` (K, V) predicts the token after each draft."""
    a = 0
    for m in range(len(draft_tokens)):
        truth = int((root_logit if m == 0 else node_logits[m - 1]).argmax())
        if int(draft_tokens[m]) != truth:
            break
        a += 1
    bonus = int((root_logit if a == 0 else node_logits[a - 1]).argmax())
    return [int(t) for t in draft_tokens[:a]], bonus


def accept_chain_sampling(
    target_dists: torch.Tensor,
    draft_tokens: list[int],
    q_dists: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[list[int], int]:
    """Lossless speculative-sampling acceptance for a chain.

    ``target_dists`` (K+1, V) are the target's distributions at each draft position
    plus the trailing position; ``q_dists`` (K, V) are the drafter's. Accept draft
    ``i`` with probability ``min(1, p_i / q_i)``; on rejection sample the bonus from
    the residual ``norm(relu(p - q))``; if all accept, sample from the trailing
    target distribution. The emitted token stream is distributed exactly as samples
    from the target."""
    accepted: list[int] = []
    for i, tok in enumerate(draft_tokens):
        p = target_dists[i, tok]
        q = q_dists[i, tok]
        u = torch.rand((), generator=generator, device=target_dists.device)
        if u < torch.clamp(p / q, max=1.0):
            accepted.append(int(tok))
        else:
            residual = torch.clamp(target_dists[i] - q_dists[i], min=0)
            residual = residual / residual.sum()
            return accepted, int(torch.multinomial(residual, 1, generator=generator))
    bonus = int(torch.multinomial(target_dists[len(draft_tokens)], 1, generator=generator))
    return accepted, bonus


def accept_tree_cached(
    root_logit: torch.Tensor,
    node_logits: torch.Tensor,
    node_tokens: list[int],
    parents: list[int],
) -> tuple[list[int], int]:
    """Cached tree variant: walk the target's argmax down the tree. ``root_logit``
    is the prediction after the last confirmed token; ``node_logits[t]`` is the
    prediction after tree node ``t``. Returns accepted node indices + bonus."""
    children: dict[int, list[int]] = defaultdict(list)
    for t, par in enumerate(parents):
        children[par].append(t)

    path: list[int] = []
    cur_logit, cur_parent = root_logit, -1
    while True:
        truth = int(cur_logit.argmax())
        nxt = next((t for t in children.get(cur_parent, []) if int(node_tokens[t]) == truth), None)
        if nxt is None:
            return path, truth
        path.append(nxt)
        cur_parent, cur_logit = nxt, node_logits[nxt]
