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
    # Depth-relative auto-selection still yields three distinct, in-range layers
    # even for a shallow 4-layer model.
    layers = tiny_target.feature_layers
    assert len(set(layers)) == 3
    assert all(1 <= idx <= tiny_target.config.num_hidden_layers for idx in layers)


def test_auto_feature_layers_include_near_final():
    # For a deep model, auto-selection must include a near-output layer.
    layers = TargetModel._resolve_feature_layers(None, 24)
    assert len(set(layers)) == 3
    assert max(layers) >= 20
