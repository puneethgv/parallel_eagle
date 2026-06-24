"""Non-speculative baseline: plain greedy autoregressive decoding."""

from __future__ import annotations

import time

import torch

from ..target import TargetModel
from .result import GenResult


@torch.no_grad()
def vanilla_generate(
    target: TargetModel,
    prompt_ids: list[int],
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> GenResult:
    seq = list(prompt_ids)
    base = len(seq)
    t0 = time.perf_counter()
    for _ in range(max_new_tokens):
        ids = torch.tensor(seq, device=target.device).unsqueeze(0)
        out = target.forward(ids)
        nxt = int(out.logits[0, -1].argmax())
        seq.append(nxt)
        if eos_token_id is not None and nxt == eos_token_id:
            break
    if target.device.type == "cuda":
        torch.cuda.synchronize()
    n = len(seq) - base
    return GenResult(
        output_ids=seq[base:],
        steps=n,
        target_calls=n,
        drafter_calls=0,
        accepted_tokens=0,
        num_generated=n,
        seconds=time.perf_counter() - t0,
    )
