"""Self-distillation produces labels equal to the target's argmax.

The whole point of self-distillation is that, over the generated (response)
region, the next-token label *is* the target's greedy choice — exactly what
decode-time acceptance measures. These tests pin that property and the
prompt-region loss masking, on the tiny CPU target (no tokenizer / download).
"""

import torch

from pe.config import DrafterConfig
from pe.distill import distill_batch, distill_example
from pe.drafter import ParallelDrafter
from pe.partition import _total_valid, _total_valid_after, mtp_backward


def test_distill_labels_equal_target_argmax(tiny_target):
    prompt = torch.tensor([1, 2, 3, 4, 5, 6])
    out = distill_example(tiny_target, prompt, max_new_tokens=12, eos_token_id=None)
    assert out is not None
    full_ids, feats, prompt_len, labels = out

    assert prompt_len == len(prompt)
    assert full_ids[:prompt_len].tolist() == prompt.tolist()
    assert feats.shape == (full_ids.shape[0], tiny_target.feature_dim)
    assert labels.shape == (full_ids.shape[0],)
    assert full_ids.shape[0] > prompt_len  # some response was generated

    # The teacher label at position j-1 must equal the target's argmax given the
    # preceding context — the supervision is "predict the target's argmax".
    for j in range(prompt_len, full_ids.shape[0]):
        logits = tiny_target.forward(full_ids[:j].unsqueeze(0)).logits[0, -1]
        assert int(logits.argmax()) == int(labels[j - 1]), f"label {j - 1} != target argmax"
    # Single-sequence greedy is self-consistent: the response IS the argmax stream.
    assert full_ids[prompt_len:].tolist() == labels[prompt_len - 1 : -1].tolist()


def test_batched_labels_equal_target_argmax(tiny_target):
    # Different-length prompts force left padding, so this also guards position handling.
    prompts = [torch.tensor([1, 2, 3, 4, 5, 6]), torch.tensor([7, 8, 9, 10, 11, 12, 13, 14, 15])]
    results = distill_batch(tiny_target, prompts, max_new_tokens=10, eos_token_id=None)
    assert len(results) == len(prompts)

    for (full_ids, feats, prompt_len, labels), prompt in zip(results, prompts, strict=True):
        assert prompt_len == len(prompt)
        assert full_ids[:prompt_len].tolist() == prompt.tolist()
        assert feats.shape == (full_ids.shape[0], tiny_target.feature_dim)
        assert labels.shape == (full_ids.shape[0],)
        assert full_ids.shape[0] > prompt_len
        # Labels are the unpadded teacher-forced argmax even though generation was
        # batched + left-padded — this is what keeps batched training exact.
        for j in range(prompt_len, full_ids.shape[0]):
            logits = tiny_target.forward(full_ids[:j].unsqueeze(0)).logits[0, -1]
            assert int(logits.argmax()) == int(labels[j - 1]), f"batched label {j - 1} wrong"


def test_prompt_masking_counts_only_response_slots():
    n, k, prompt_len = 20, 4, 7
    full = _total_valid(n, k)
    masked = _total_valid_after(n, k, prompt_len)
    assert masked < full
    # No slot whose label lands in the prompt region should be counted.
    expected = sum(1 for i in range(n) for d in range(k) if prompt_len <= i + 1 + d < n)
    assert masked == expected


def test_mtp_backward_accepts_prompt_len(tiny_target):
    torch.manual_seed(0)
    drafter = ParallelDrafter.from_target(
        tiny_target, DrafterConfig(num_layers=2, max_depth=4)
    ).train()
    ids = torch.randint(0, tiny_target.vocab_size, (24,))
    feats = torch.randn(24, tiny_target.feature_dim)

    loss = mtp_backward(drafter, ids, feats, num_segments=1, prompt_len=8)
    assert loss > 0
    assert any(p.grad is not None for p in drafter.trainable_parameters())


def test_mtp_backward_teacher_labels_equivalent_when_self_consistent(tiny_target):
    # When teacher_labels are the next-token stream (greedy self-consistency), the
    # teacher-label path must give exactly the same loss as the next-token path.
    torch.manual_seed(0)
    drafter = ParallelDrafter.from_target(
        tiny_target, DrafterConfig(num_layers=2, max_depth=4)
    ).eval()
    ids = torch.randint(0, tiny_target.vocab_size, (20,))
    feats = torch.randn(20, tiny_target.feature_dim)
    teacher = torch.empty_like(ids)
    teacher[:-1] = ids[1:]  # labels[p] == next token  => teacher_labels[tgt-1] == ids[tgt]
    teacher[-1] = ids[-1]

    drafter.zero_grad(set_to_none=True)
    base = mtp_backward(drafter, ids, feats, num_segments=1)
    drafter.zero_grad(set_to_none=True)
    with_labels = mtp_backward(drafter, ids, feats, num_segments=1, teacher_labels=teacher)
    assert abs(base - with_labels) < 1e-5
