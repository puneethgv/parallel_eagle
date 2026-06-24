"""Minimal, from-scratch transformer building blocks (Llama-style).

Just enough to build the drafter: RMSNorm, rotary position embeddings,
grouped-query attention that accepts an explicit additive bias and position ids,
and a SwiGLU feed-forward. Keeping these in-repo (rather than importing model
internals) makes the drafter's behavior fully explicit and version-stable.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def build_rope_cache(
    position_ids: torch.Tensor, head_dim: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(cos, sin)`` of shape ``(B, S, head_dim)`` for the given positions."""
    device = position_ids.device
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    angles = position_ids.float()[..., None] * inv_freq[None, None, :]  # (B, S, hd/2)
    emb = torch.cat([angles, angles], dim=-1)  # (B, S, hd)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, n_heads, S, hd); cos/sin: (B, S, hd) -> broadcast over heads
    cos = cos.unsqueeze(1).to(x.dtype)
    sin = sin.unsqueeze(1).to(x.dtype)
    return x * cos + _rotate_half(x) * sin


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, kvh, s, hd = x.shape
    return x[:, :, None, :, :].expand(b, kvh, n_rep, s, hd).reshape(b, kvh * n_rep, s, hd)


class Attention(nn.Module):
    def __init__(self, hidden: int, n_heads: int, n_kv_heads: int, theta: float):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = hidden // n_heads
        self.theta = theta
        self.q_proj = nn.Linear(hidden, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, hidden, bias=False)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k = repeat_kv(k, self.n_heads // self.n_kv_heads)
        v = repeat_kv(v, self.n_heads // self.n_kv_heads)

        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim**0.5)
        scores = scores + bias  # (B,1,S,S) or (1,1,S,S) additive
        attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        out = torch.matmul(attn, v)  # (B, n_heads, S, hd)
        out = out.transpose(1, 2).reshape(b, s, -1)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(
        self,
        hidden: int,
        n_heads: int,
        n_kv_heads: int,
        intermediate: int,
        theta: float,
        eps: float,
    ):
        super().__init__()
        self.input_norm = RMSNorm(hidden, eps)
        self.attn = Attention(hidden, n_heads, n_kv_heads, theta)
        self.post_norm = RMSNorm(hidden, eps)
        self.mlp = SwiGLU(hidden, intermediate)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attn(self.input_norm(x), cos, sin, bias)
        x = x + self.mlp(self.post_norm(x))
        return x
