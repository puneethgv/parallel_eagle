"""Edge cases and clear errors."""

import pytest
import torch

from pe.config import DrafterConfig
from pe.decode.baselines import vanilla_generate
from pe.drafter import ParallelDrafter, load_drafter
from pe.features import FeatureDataset
from pe.serve import generate_speculative_cached


def test_load_drafter_missing_checkpoint(tiny_target):
    with pytest.raises(FileNotFoundError, match="pe.train"):
        load_drafter("/no/such/drafter.pt", tiny_target)


def test_feature_dataset_missing_cache(tmp_path):
    with pytest.raises(FileNotFoundError, match="pe.features"):
        FeatureDataset(tmp_path)


def test_single_depth_tree_decodes_losslessly(tiny_target):
    # k=1 is a degenerate one-level tree; it must still be lossless.
    torch.manual_seed(0)
    drafter = ParallelDrafter.from_target(tiny_target, DrafterConfig(num_layers=2, max_depth=2)).eval()
    prompt = [1, 2, 3, 4]
    ref = vanilla_generate(tiny_target, prompt, 10).output_ids
    res = generate_speculative_cached(
        tiny_target, drafter, prompt, k=1, mode="tree", max_new_tokens=10, tree_top_k=2, tree_max_nodes=4
    )
    assert res.output_ids == ref
