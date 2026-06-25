"""Speculative sampling is lossless: the emitted distribution matches the target.

Even with an untrained (random) drafter, the rejection-sampling rule must make the
first emitted token distributed exactly like a direct sample from the target.
"""

from collections import Counter

import torch

from pe.config import DrafterConfig
from pe.drafter import ParallelDrafter
from pe.serve import generate_speculative_sampling


def test_first_token_matches_target_distribution(tiny_target):
    torch.manual_seed(0)
    drafter = ParallelDrafter.from_target(tiny_target, DrafterConfig(num_layers=2, max_depth=4)).eval()
    prompt = [1, 2, 3, 4, 5]
    temp = 1.0

    tgt_logit = tiny_target.forward(torch.tensor(prompt).unsqueeze(0)).logits[0, -1]
    p = torch.softmax(tgt_logit / temp, dim=-1)

    n = 500
    counts: Counter[int] = Counter()
    for s in range(n):
        res = generate_speculative_sampling(
            tiny_target, drafter, prompt, k=4, max_new_tokens=1, temperature=temp, seed=s
        )
        counts[res.output_ids[0]] += 1

    emp = torch.zeros_like(p)
    for tok, cnt in counts.items():
        emp[tok] = cnt / n
    tv = 0.5 * (emp - p).abs().sum().item()
    assert tv < 0.2, f"total-variation distance {tv:.3f} too high — sampling is not lossless"
