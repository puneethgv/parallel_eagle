"""Losslessness: every speculative strategy reproduces vanilla greedy exactly.

The drafter here is untrained (random), so acceptance is low — but correctness is
independent of draft quality, so the emitted tokens must still match plain greedy
decoding for all of chain, sequential, and tree drafting.
"""

import torch

from pe.config import DrafterConfig
from pe.decode.baselines import vanilla_generate
from pe.drafter import ParallelDrafter
from pe.serve import generate_speculative, generate_speculative_cached

PROMPT = [1, 2, 3, 4, 5, 6]
N = 24


def _drafter(tiny_target, k=5):
    torch.manual_seed(0)
    d = ParallelDrafter.from_target(tiny_target, DrafterConfig(num_layers=2, max_depth=k))
    return d.eval()


def test_chain_matches_vanilla(tiny_target):
    d = _drafter(tiny_target)
    ref = vanilla_generate(tiny_target, PROMPT, N).output_ids
    res = generate_speculative(tiny_target, d, PROMPT, k=5, mode="chain", max_new_tokens=N)
    assert res.output_ids == ref
    assert res.num_generated == N


def test_sequential_matches_vanilla(tiny_target):
    d = _drafter(tiny_target)
    ref = vanilla_generate(tiny_target, PROMPT, N).output_ids
    res = generate_speculative(tiny_target, d, PROMPT, k=5, mode="sequential", max_new_tokens=N)
    assert res.output_ids == ref


def test_tree_matches_vanilla(tiny_target):
    d = _drafter(tiny_target)
    ref = vanilla_generate(tiny_target, PROMPT, N).output_ids
    res = generate_speculative(
        tiny_target, d, PROMPT, k=5, mode="tree", max_new_tokens=N, tree_top_k=3, tree_max_nodes=15
    )
    assert res.output_ids == ref


def test_cached_chain_matches_vanilla(tiny_target):
    d = _drafter(tiny_target)
    ref = vanilla_generate(tiny_target, PROMPT, N).output_ids
    res = generate_speculative_cached(tiny_target, d, PROMPT, k=5, mode="chain", max_new_tokens=N)
    assert res.output_ids == ref


def test_cached_tree_matches_vanilla(tiny_target):
    d = _drafter(tiny_target)
    ref = vanilla_generate(tiny_target, PROMPT, N).output_ids
    res = generate_speculative_cached(
        tiny_target, d, PROMPT, k=5, mode="tree", max_new_tokens=N, tree_top_k=3, tree_max_nodes=15
    )
    assert res.output_ids == ref


def test_tree_accepts_at_least_as_many_as_chain(tiny_target):
    # Not a correctness requirement, but the tree should never commit fewer
    # tokens per step than the chain on the same model/prompt.
    d = _drafter(tiny_target)
    chain = generate_speculative(tiny_target, d, PROMPT, k=5, mode="chain", max_new_tokens=N)
    tree = generate_speculative(
        tiny_target, d, PROMPT, k=5, mode="tree", max_new_tokens=N, tree_top_k=3, tree_max_nodes=15
    )
    assert tree.acceptance_length >= chain.acceptance_length - 1e-9
