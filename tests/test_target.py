"""Target wrapper: fused-feature shape and robust feature-layer resolution."""

import torch

from pe.target import TargetModel


def test_feature_dim_and_forward(tiny_target):
    t = tiny_target
    assert t.num_feature_layers == 3
    assert t.feature_dim == 3 * t.hidden_size

    ids = torch.randint(0, t.vocab_size, (1, 12))
    out = t.forward(ids)
    assert out.logits.shape == (1, 12, t.vocab_size)
    assert out.fused.shape == (1, 12, t.feature_dim)


def test_feature_layers_distinct_on_shallow_model(tiny_target):
    # The 4-layer model can't honor the (2, 8, 15) defaults verbatim; the wrapper
    # must still pick three distinct, in-range hidden-state indices.
    layers = tiny_target.feature_layers
    assert len(set(layers)) == 3
    assert all(1 <= idx <= tiny_target.config.num_hidden_layers for idx in layers)
