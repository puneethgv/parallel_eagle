"""Sequence partitioning for within-sequence gradient accumulation.

Training the parallel drafter expands each length-``n`` sequence into ``n * K``
prediction slots, so the attention cost grows with ``(nK)^2`` and dominates the
memory budget on long contexts. This module splits one sequence into ``S``
segments and accumulates gradients across them, instantiating the ``K - 1``
prediction slots only for the *current* segment's anchors (the real depth-0
stream is always materialized as the shared causal context).

Because every depth of an anchor stays in the same segment, each slot's full
attendance set is contained in its segment's sub-sequence — so the accumulated
gradient is **exactly** the full-sequence gradient, not an approximation. The
per-pass sequence length drops from ``nK`` toward ``~2n``, which is the memory win.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .drafter import ParallelDrafter
from .masks import allow_from_slots, to_additive_bias


def _segment_bounds(n: int, num_segments: int) -> list[tuple[int, int]]:
    num_segments = max(1, min(num_segments, n))
    edges = [round(n * s / num_segments) for s in range(num_segments + 1)]
    return [(edges[i], edges[i + 1]) for i in range(num_segments) if edges[i + 1] > edges[i]]


def _total_valid(n: int, k: int) -> int:
    # number of (anchor, depth) slots whose target token exists in-sequence
    return sum(1 for i in range(n) for d in range(k) if i + 1 + d < n)


def _total_valid_after(n: int, k: int, prompt_len: int = 0) -> int:
    # like _total_valid, but only slots whose target token is in the response region
    return sum(1 for i in range(n) for d in range(k) if prompt_len <= i + 1 + d < n)


def _segment_slots(a0: int, a1: int, k: int, device):
    """Anchors/depths for one segment: real prefix 0..a1-1 plus segment MTP slots."""
    anchor_real = torch.arange(a1, device=device)
    depth_real = torch.zeros(a1, dtype=torch.long, device=device)
    if k > 1:
        seg = torch.arange(a0, a1, device=device)
        anchor_mtp = seg.repeat_interleave(k - 1)
        depth_mtp = torch.arange(1, k, device=device).repeat(a1 - a0)
    else:
        anchor_mtp = torch.empty(0, dtype=torch.long, device=device)
        depth_mtp = torch.empty(0, dtype=torch.long, device=device)
    anchor = torch.cat([anchor_real, anchor_mtp])
    depth = torch.cat([depth_real, depth_mtp])
    n_real = a1
    return anchor, depth, n_real


def mtp_backward(
    drafter: ParallelDrafter,
    input_ids: torch.Tensor,
    features: torch.Tensor,
    num_segments: int = 1,
    loss_scale: float = 1.0,
    prompt_len: int = 0,
    teacher_labels: torch.Tensor | None = None,
) -> float:
    """Compute the per-depth cross-entropy and accumulate gradients.

    Returns the mean loss (for logging). Gradients are left on the drafter's
    parameters; the caller is responsible for the optimizer step. ``prompt_len``
    masks slots whose label falls in the prompt region (``tgt < prompt_len``) out
    of the loss — for self-distilled data, only response-region labels equal the
    target's argmax and carry the acceptance-optimizing signal.

    ``teacher_labels`` (self-distilled caches): ``teacher_labels[p]`` is the
    target's argmax *at* position ``p``, so the supervision for predicting the
    token at position ``tgt`` is ``teacher_labels[tgt - 1]`` — the exact "predict
    the target's argmax" objective, from the same forward as the features. When
    ``None`` the next sequence token ``input_ids[tgt]`` is used (human-text caches).
    """
    dev = drafter.device
    input_ids = input_ids.to(dev)
    features = features.to(dev)
    if teacher_labels is not None:
        teacher_labels = teacher_labels.to(dev)
    n = input_ids.shape[0]
    k = drafter.max_depth
    total_valid = max(1, _total_valid_after(n, k, prompt_len))

    loss_sum = 0.0
    for a0, a1 in _segment_bounds(n, num_segments):
        anchor, depth, n_real = _segment_slots(a0, a1, k, dev)

        real_x = drafter.real_input(input_ids[:n_real], features[:n_real])
        x = real_x if k == 1 else torch.cat([real_x, drafter.mask_input(anchor.shape[0] - n_real)])
        pos = (anchor + depth).unsqueeze(0)
        bias = to_additive_bias(allow_from_slots(anchor, depth), drafter.param_dtype)

        hidden = drafter(x.unsqueeze(0), pos, bias)[0]

        tgt = anchor + 1 + depth
        is_loss_slot = (anchor >= a0) & (tgt < n) & (tgt >= prompt_len)
        labels = torch.full_like(anchor, -100)
        slot_tgt = tgt[is_loss_slot]
        if teacher_labels is not None:
            labels[is_loss_slot] = teacher_labels[slot_tgt - 1]
        else:
            labels[is_loss_slot] = input_ids[slot_tgt]

        valid = labels != -100
        if not bool(valid.any()):
            continue
        logits = drafter.lm_head(hidden[valid]).float()
        seg_loss = F.cross_entropy(logits, labels[valid], reduction="sum")
        (seg_loss * (loss_scale / total_valid)).backward()
        loss_sum += float(seg_loss.detach())

    return loss_sum / total_valid


@torch.no_grad()
def mtp_eval_loss(
    drafter: ParallelDrafter, input_ids: torch.Tensor, features: torch.Tensor
) -> float:
    """Full-sequence mean loss without gradients (for validation / tests)."""
    dev = drafter.device
    x, pos, bias, labels = drafter.build_training_packed(input_ids.to(dev), features.to(dev))
    hidden = drafter(x, pos, bias)[0]
    valid = labels != -100
    logits = drafter.lm_head(hidden[valid]).float()
    return float(F.cross_entropy(logits, labels[valid].to(dev)))
