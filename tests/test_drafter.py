"""Drafter shapes plus the exact full-vs-partitioned gradient equivalence."""

import torch

from pe.config import DrafterConfig
from pe.drafter import ParallelDrafter
from pe.partition import mtp_backward, mtp_eval_loss


def _build(tiny_target, k=4, layers=2):
    torch.manual_seed(1)
    dcfg = DrafterConfig(num_layers=layers, max_depth=k)
    return ParallelDrafter.from_target(tiny_target, dcfg)


def test_training_pack_and_draft_shapes(tiny_target):
    d = _build(tiny_target, k=4)
    n = 6
    ids = torch.randint(0, tiny_target.vocab_size, (n,))
    feats = torch.randn(n, tiny_target.feature_dim)

    x, pos, bias, labels = d.build_training_packed(ids, feats)
    assert x.shape == (1, n * 4, tiny_target.hidden_size)
    assert pos.shape == (1, n * 4)
    assert bias.shape == (1, 1, n * 4, n * 4)
    assert labels.shape == (n * 4,)

    hidden = d(x, pos, bias)
    assert hidden.shape == (1, n * 4, tiny_target.hidden_size)

    logits = d.draft_logits(ids[:5], feats[:5], k=4)
    assert logits.shape == (4, tiny_target.vocab_size)


def test_partition_matches_full_gradient(tiny_target):
    d = _build(tiny_target, k=4)
    d.train()
    ids = torch.randint(0, tiny_target.vocab_size, (9,))
    feats = torch.randn(9, tiny_target.feature_dim)

    def grads_for(segments):
        d.zero_grad(set_to_none=True)
        loss = mtp_backward(d, ids, feats, num_segments=segments)
        return loss, {
            n: p.grad.detach().clone() for n, p in d.named_parameters() if p.grad is not None
        }

    l1, g1 = grads_for(1)
    l3, g3 = grads_for(3)

    assert abs(l1 - l3) < 1e-4
    assert g1.keys() == g3.keys()
    for name in g1:
        assert torch.allclose(g1[name], g3[name], atol=1e-5, rtol=1e-4), name


def test_eval_loss_runs(tiny_target):
    d = _build(tiny_target, k=3)
    ids = torch.randint(0, tiny_target.vocab_size, (7,))
    feats = torch.randn(7, tiny_target.feature_dim)
    assert mtp_eval_loss(d, ids, feats) > 0
