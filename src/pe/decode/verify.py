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
    """``logits`` covers ``[prefix | draft_tokens]`` (shape ``(P + K, V)``)."""
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
    """``logits`` covers ``[prefix | tree nodes]``; follow the target's argmax down
    the tree as far as a matching child exists."""
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
            return [int(node_tokens[t]) for t in path], truth
        path.append(nxt)
        cur_parent, cur_pos = nxt, p + nxt
