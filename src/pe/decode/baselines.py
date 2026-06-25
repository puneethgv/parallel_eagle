"""Non-speculative baseline: plain greedy autoregressive decoding."""

from __future__ import annotations

import time
from collections.abc import Callable

import torch

from ..target import TargetModel
from .result import GenResult


@torch.no_grad()
def vanilla_generate(
    target: TargetModel,
    prompt_ids: list[int],
    max_new_tokens: int,
    eos_token_id: int | None = None,
    on_commit: Callable[[list[int]], None] | None = None,
) -> GenResult:
    seq = list(prompt_ids)
    base = len(seq)
    t0 = time.perf_counter()
    for _ in range(max_new_tokens):
        ids = torch.tensor(seq, device=target.device).unsqueeze(0)
        out = target.forward(ids)
        nxt = int(out.logits[0, -1].argmax())
        seq.append(nxt)
        if on_commit is not None:
            on_commit([nxt])
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


@torch.no_grad()
def vanilla_generate_cached(
    target: TargetModel,
    prompt_ids: list[int],
    max_new_tokens: int,
    eos_token_id: int | None = None,
    on_commit: Callable[[list[int]], None] | None = None,
) -> GenResult:
    """Standard greedy decoding with a KV cache — the fair speed baseline (one
    incremental target forward per token)."""
    dev = target.device
    seq = list(prompt_ids)
    base = len(seq)
    cache = target.make_cache()
    pos = torch.arange(len(seq), device=dev).unsqueeze(0)
    logits, _, cache = target.forward_cached(
        torch.tensor(seq, device=dev).unsqueeze(0), cache, pos
    )
    nxt = int(logits[0, -1].argmax())
    target_calls = 1
    t0 = time.perf_counter()
    while len(seq) - base < max_new_tokens:
        seq.append(nxt)
        if on_commit is not None:
            on_commit([nxt])
        if eos_token_id is not None and nxt == eos_token_id:
            break
        if len(seq) - base >= max_new_tokens:
            break
        npos = torch.tensor([[len(seq) - 1]], device=dev)
        logits, _, cache = target.forward_cached(
            torch.tensor([[nxt]], device=dev), cache, npos
        )
        target_calls += 1
        nxt = int(logits[0, -1].argmax())
    if dev.type == "cuda":
        torch.cuda.synchronize()
    n = len(seq) - base
    return GenResult(
        output_ids=seq[base:],
        steps=n,
        target_calls=target_calls,
        drafter_calls=0,
        accepted_tokens=0,
        num_generated=n,
        seconds=time.perf_counter() - t0,
    )
