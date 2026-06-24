"""Attention masks for parallel multi-token training and tree verification.

Two structures live here:

1. **Parallel-MTP training mask.** Each real position (anchor) ``i`` is expanded
   into ``K`` prediction slots: depth ``d`` predicts the token ``d + 1`` steps
   ahead. Slots are laid out *anchor-major* (flat index ``i * K + d``) so that the
   mask for a length-``n`` sequence is exactly the top-left ``(nK, nK)`` block of
   the mask for any longer sequence. :class:`AmortizedMtpMask` exploits this:
   build once at the maximum length, then slice (a constant-time view) per batch.

   Attendance rule for query slot ``(i, d)`` over key slot ``(j, e)``:
     - real key (``e == 0``): attend iff ``j <= i`` (causal over the real stream);
     - prediction key (``e >= 1``): attend iff ``j == i`` and ``e <= d`` (same
       anchor, lower-or-equal depth — i.e. the chain that precedes this slot).

2. **Tree-verification mask.** A flattened candidate tree appended after a
   confirmed prefix; every node attends to the whole prefix and to its own
   ancestor chain, so the target can score all candidate continuations in one pass.
"""

from __future__ import annotations

import torch

NEG_INF_FILL = "min"  # use the dtype's most-negative value as the additive mask fill


def to_additive_bias(allow: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert a boolean attend-mask to a 4D additive bias ``(1, 1, S, S)``."""
    bias = torch.zeros(allow.shape, dtype=dtype, device=allow.device)
    bias.masked_fill_(~allow, torch.finfo(dtype).min)
    return bias.unsqueeze(0).unsqueeze(0)


def mtp_slot_anchor_depth(num_anchors: int, k: int, device="cpu"):
    """Return per-slot anchor and depth index tensors, anchor-major."""
    anchor = torch.arange(num_anchors, device=device).repeat_interleave(k)
    depth = torch.arange(k, device=device).repeat(num_anchors)
    return anchor, depth


def allow_from_slots(
    anchor_q: torch.Tensor,
    depth_q: torch.Tensor,
    anchor_k: torch.Tensor | None = None,
    depth_k: torch.Tensor | None = None,
) -> torch.Tensor:
    """Attend-mask for arbitrary query/key slot sets given their anchors/depths.

    The single rule that defines the whole scheme: a query slot ``(i, d)`` attends
    to a real key (depth 0) iff ``j <= i``, and to a prediction key (depth >= 1)
    iff ``j == i`` and ``e <= d``. Works for any subset of slots, which is what
    lets the sequence-partitioning code instantiate only a segment's slots.
    """
    if anchor_k is None:
        anchor_k, depth_k = anchor_q, depth_q
    qi, qd = anchor_q[:, None], depth_q[:, None]
    kj, ke = anchor_k[None, :], depth_k[None, :]
    real_key = ke == 0
    return torch.where(real_key, kj <= qi, (kj == qi) & (ke <= qd))


def mtp_allow_mask(num_anchors: int, k: int, device="cpu") -> torch.Tensor:
    """Boolean ``(nK, nK)`` attend-mask for the full parallel-MTP layout."""
    anchor, depth = mtp_slot_anchor_depth(num_anchors, k, device=device)
    return allow_from_slots(anchor, depth)


def mtp_position_ids(num_anchors: int, k: int, device="cpu") -> torch.Tensor:
    """Rotary position id per slot: ``pos(i, d) = i + d``."""
    anchor, depth = mtp_slot_anchor_depth(num_anchors, k, device=device)
    return anchor + depth


def mtp_labels(input_ids: torch.Tensor, k: int) -> torch.Tensor:
    """Target token per slot: slot ``(i, d)`` predicts ``input_ids[i + 1 + d]``.

    Out-of-range targets are set to ``-100`` (ignored by cross-entropy).
    """
    n = input_ids.shape[0]
    anchor, depth = mtp_slot_anchor_depth(n, k, device=input_ids.device)
    tgt = anchor + 1 + depth
    valid = tgt < n
    labels = input_ids[tgt.clamp(max=n - 1)]
    labels = labels.masked_fill(~valid, -100)
    return labels


class AmortizedMtpMask:
    """Precompute the MTP attend-mask once at max length; slice per batch.

    The slice is a top-left submatrix view (no allocation), which is the whole
    point: per-step mask construction is otherwise the dominant data-loading cost
    on long sequences.
    """

    def __init__(self, max_anchors: int, k: int, device="cpu"):
        self.max_anchors = max_anchors
        self.k = k
        self._allow = mtp_allow_mask(max_anchors, k, device=device)

    def allow(self, num_anchors: int) -> torch.Tensor:
        if num_anchors > self.max_anchors:
            raise ValueError(f"{num_anchors} anchors exceeds max {self.max_anchors}")
        s = num_anchors * self.k
        return self._allow[:s, :s]

    def bias(self, num_anchors: int, dtype: torch.dtype) -> torch.Tensor:
        return to_additive_bias(self.allow(num_anchors), dtype)


# --------------------------------------------------------------------------- #
# Tree verification masks
# --------------------------------------------------------------------------- #


def tree_depths(parents: list[int]) -> list[int]:
    """Depth (distance from the prefix) of each tree node; ``parents[t] == -1``
    means the node hangs directly off the prefix (depth 0)."""
    depths = [0] * len(parents)
    for t, p in enumerate(parents):
        depths[t] = 0 if p < 0 else depths[p] + 1
    return depths


def tree_allow_mask(parents: list[int], prefix_len: int, device="cpu") -> torch.Tensor:
    """Boolean attend-mask over ``[prefix | tree nodes]``.

    Prefix is causal; every node attends to the entire prefix and to its own
    ancestor chain (including itself). ``parents`` must be topologically ordered
    (a parent appears before its children).
    """
    t = len(parents)
    n = prefix_len + t
    allow = torch.zeros(n, n, dtype=torch.bool, device=device)
    if prefix_len:
        allow[:prefix_len, :prefix_len] = torch.tril(
            torch.ones(prefix_len, prefix_len, dtype=torch.bool, device=device)
        )
        allow[prefix_len:, :prefix_len] = True
    for node in range(t):
        a = node
        while a != -1:
            allow[prefix_len + node, prefix_len + a] = True
            a = parents[a]
    return allow


def tree_position_ids(parents: list[int], prefix_len: int, device="cpu") -> torch.Tensor:
    """Position id per token: prefix is ``0..P-1``; a node at tree-depth ``dpt``
    sits at absolute position ``P + dpt`` (siblings share a position)."""
    depths = tree_depths(parents)
    prefix = torch.arange(prefix_len, device=device)
    nodes = torch.tensor([prefix_len + d for d in depths], dtype=torch.long, device=device)
    return torch.cat([prefix, nodes]) if prefix_len else nodes
