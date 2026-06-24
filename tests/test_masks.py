"""Mask correctness: parallel-MTP layout, amortized slicing, and tree masks."""

import torch

from pe.masks import (
    AmortizedMtpMask,
    mtp_allow_mask,
    mtp_labels,
    mtp_position_ids,
    tree_allow_mask,
    tree_depths,
    tree_position_ids,
)


def test_mtp_allow_small_exact():
    # n=2 anchors, K=2 depths -> slots ordered (0,0),(0,1),(1,0),(1,1)
    allow = mtp_allow_mask(2, 2)
    expected = torch.tensor(
        [
            [1, 0, 0, 0],  # (0,0) real: only itself (causal, no later anchors)
            [1, 1, 0, 0],  # (0,1) mtp: real anchor0 + self
            [1, 0, 1, 0],  # (1,0) real: anchors 0 and 1
            [1, 0, 1, 1],  # (1,1) mtp: real anchors 0,1 + self (same-anchor depth<=1)
        ],
        dtype=torch.bool,
    )
    assert torch.equal(allow, expected)


def test_amortized_equals_naive_and_is_topleft():
    k = 4
    amort = AmortizedMtpMask(max_anchors=8, k=k)
    for n in (1, 3, 5, 8):
        naive = mtp_allow_mask(n, k)
        assert torch.equal(amort.allow(n), naive)
    # the n=3 mask is literally the top-left block of the n=8 mask
    full = mtp_allow_mask(8, k)
    assert torch.equal(mtp_allow_mask(3, k), full[: 3 * k, : 3 * k])


def test_mtp_positions_and_labels():
    k = 3
    pos = mtp_position_ids(4, k)
    # slot (i,d) -> i + d
    assert pos.tolist() == [0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5]

    ids = torch.tensor([10, 11, 12, 13])
    labels = mtp_labels(ids, k)
    # anchor 0: predicts ids[1],ids[2],ids[3]; anchor 3: all out of range -> -100
    assert labels[:3].tolist() == [11, 12, 13]
    assert labels[-3:].tolist() == [-100, -100, -100]


def test_tree_mask_and_positions():
    # prefix length 2; tree: node0 off prefix, node1 child of node0, node2 off prefix
    parents = [-1, 0, -1]
    assert tree_depths(parents) == [0, 1, 0]

    allow = tree_allow_mask(parents, prefix_len=2)
    assert allow.shape == (5, 5)
    # prefix is causal
    assert allow[0, 0] and not allow[0, 1]
    assert allow[1, 0] and allow[1, 1]
    # every node sees the whole prefix
    assert allow[2:, :2].all()
    # node1 (index 3) attends its ancestors {node1,node0} but not node2
    assert allow[3, 2] and allow[3, 3] and not allow[3, 4]
    # node0 (index 2) attends only itself among tree nodes
    assert allow[2, 2] and not allow[2, 3] and not allow[2, 4]

    pos = tree_position_ids(parents, prefix_len=2)
    assert pos.tolist() == [0, 1, 2, 3, 2]  # prefix 0,1 ; depths 0,1,0 -> 2,3,2
