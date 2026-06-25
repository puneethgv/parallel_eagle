"""The feature-conditioned parallel multi-token drafter.

The drafter consumes the target's fused hidden states and proposes ``K`` future
tokens in a single forward pass. Its input at a position is
``in_proj(concat(token_embedding, feat_proj(target_features)))``. The ``K - 1``
"future" prediction slots have no real token or feature yet, so they are filled
with two learnable vectors — a shared hidden state ``h_shared`` and a
``mask_emb`` standing in for the unknown token. Positional structure is supplied
entirely by rotary attention (slot ``(i, d)`` sits at position ``i + d``), so no
depth-specific encoding is needed.

The token embedding and LM head are **shared with the frozen target** and held
outside the module's parameter set (so they are neither trained nor copied); the
mask representation is its own small parameter instead of an unfrozen vocabulary
row.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .config import DrafterConfig
from .masks import (
    mtp_allow_mask,
    mtp_labels,
    mtp_position_ids,
    to_additive_bias,
)
from .nn import DecoderLayer, RMSNorm, build_rope_cache


class ParallelDrafter(nn.Module):
    def __init__(
        self,
        *,
        hidden: int,
        n_heads: int,
        n_kv_heads: int,
        intermediate: int,
        vocab: int,
        feature_dim: int,
        rope_theta: float,
        eps: float,
        num_layers: int,
        max_depth: int,
        embed_module: nn.Module,
        lm_head_module: nn.Module,
        final_norm_module: nn.Module,
    ):
        super().__init__()
        self.hidden = hidden
        self.head_dim = hidden // n_heads
        self.rope_theta = rope_theta
        self.max_depth = max_depth
        self.vocab = vocab

        self.feature_dim = feature_dim
        self.num_feature_layers = feature_dim // hidden
        # Target hidden states carry "massive activations" (a few dims with huge
        # magnitude) and differ in scale across layers; normalize each fused layer
        # block before projection so training is stable.
        self.feat_norm = RMSNorm(hidden, eps)
        self.feat_proj = nn.Linear(feature_dim, hidden, bias=False)
        self.in_proj = nn.Linear(2 * hidden, hidden, bias=False)
        self.mask_emb = nn.Parameter(torch.zeros(hidden))
        self.h_shared = nn.Parameter(torch.zeros(hidden))
        self.layers = nn.ModuleList(
            DecoderLayer(hidden, n_heads, n_kv_heads, intermediate, rope_theta, eps)
            for _ in range(num_layers)
        )
        nn.init.normal_(self.mask_emb, std=0.02)
        nn.init.normal_(self.h_shared, std=0.02)

        # Shared + frozen; stashed in tuples so they are not registered as
        # submodules (kept out of the optimizer and the drafter checkpoint). The
        # final norm is shared too, so the drafter outputs in the LM head's scale.
        self._embed = (embed_module,)
        self._lm_head = (lm_head_module,)
        self._final_norm = (final_norm_module,)
        self.grad_checkpoint = False

    # ------------------------------------------------------------------ #
    # Factory
    # ------------------------------------------------------------------ #
    @classmethod
    def from_target(cls, target, dcfg: DrafterConfig) -> ParallelDrafter:
        c = target.config
        model = cls(
            hidden=target.hidden_size,
            n_heads=c.num_attention_heads,
            n_kv_heads=getattr(c, "num_key_value_heads", c.num_attention_heads),
            intermediate=c.intermediate_size,
            vocab=target.vocab_size,
            feature_dim=target.feature_dim,
            rope_theta=getattr(c, "rope_theta", 10000.0),
            eps=getattr(c, "rms_norm_eps", 1e-5),
            num_layers=dcfg.num_layers,
            max_depth=dcfg.max_depth,
            embed_module=target.get_input_embeddings(),
            lm_head_module=target.get_lm_head(),
            final_norm_module=target.get_final_norm(),
        )
        return model

    # ------------------------------------------------------------------ #
    # Input construction
    # ------------------------------------------------------------------ #
    @property
    def device(self) -> torch.device:
        return self.in_proj.weight.device

    @property
    def param_dtype(self) -> torch.dtype:
        return self.in_proj.weight.dtype

    def _embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self._embed[0](token_ids)

    def lm_head(self, hidden: torch.Tensor) -> torch.Tensor:
        return self._lm_head[0](hidden)

    def real_input(self, token_ids: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Input vector for real positions: ``in_proj([emb(token); proj(norm(feat))])``."""
        emb = self._embed_tokens(token_ids).to(self.param_dtype)
        f = features.to(self.param_dtype)
        f = self.feat_norm(f.unflatten(-1, (self.num_feature_layers, self.hidden)))
        feat = self.feat_proj(f.flatten(-2))
        return self.in_proj(torch.cat([emb, feat], dim=-1))

    def mask_input(self, n: int) -> torch.Tensor:
        """Input vector (repeated ``n`` times) for prediction slots."""
        vec = self.in_proj(torch.cat([self.mask_emb, self.h_shared], dim=-1))
        return vec.unsqueeze(0).expand(n, -1)

    def token_shared_input(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Input for a known token whose target feature is unavailable: the real
        token embedding combined with the shared hidden state. Used by the
        sequential-drafting baseline for positions the target has not yet seen."""
        emb = self._embed_tokens(token_ids).to(self.param_dtype)
        shared = self.h_shared.unsqueeze(0).expand(emb.shape[0], -1)
        return self.in_proj(torch.cat([emb, shared], dim=-1))

    # ------------------------------------------------------------------ #
    # Core forward over an arbitrary packed sequence
    # ------------------------------------------------------------------ #
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        """``x``: (B, S, H); ``position_ids``: (B, S); ``bias``: (B|1, 1, S, S).
        Returns the final hidden states (B, S, H)."""
        cos, sin = build_rope_cache(position_ids, self.head_dim, self.rope_theta)
        for layer in self.layers:
            if self.grad_checkpoint and self.training:
                x = checkpoint(layer, x, cos, sin, bias, use_reentrant=False)
            else:
                x = layer(x, cos, sin, bias)
        return self._final_norm[0](x)

    # ------------------------------------------------------------------ #
    # Training: pack one example into the parallel-MTP layout
    # ------------------------------------------------------------------ #
    def build_training_packed(self, input_ids: torch.Tensor, features: torch.Tensor):
        """Return ``(x, position_ids, bias, labels)`` for one example.

        ``input_ids``: (n,), ``features``: (n, feature_dim). Layout is anchor-major
        with ``K = max_depth`` depths per anchor.
        """
        k = self.max_depth
        n = input_ids.shape[0]
        dev = self.device
        s = n * k

        real_x = self.real_input(input_ids, features)  # (n, H)
        mask_x = self.mask_input(n * (k - 1)) if k > 1 else None

        x = torch.empty(s, self.hidden, dtype=self.param_dtype, device=dev)
        x[0::k] = real_x
        if k > 1:
            # slots i*k + d for d = 1..k-1 are all mask slots
            mask_idx = torch.cat([torch.arange(d, s, k, device=dev) for d in range(1, k)])
            x[mask_idx] = mask_x

        position_ids = mtp_position_ids(n, k, device=dev).unsqueeze(0)
        bias = to_additive_bias(mtp_allow_mask(n, k, device=dev), self.param_dtype)
        labels = mtp_labels(input_ids, k)
        return x.unsqueeze(0), position_ids, bias, labels

    # ------------------------------------------------------------------ #
    # Inference: draft K tokens given a confirmed context
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def draft_logits(
        self, context_ids: torch.Tensor, context_feats: torch.Tensor, k: int
    ) -> torch.Tensor:
        """Logits for the ``k`` tokens following ``context_ids`` (shape ``(k, V)``).

        The minimal packed sequence is ``[real depth-0 slots for the whole context]
        + [mask slots for depths 1..k-1 anchored at the last context position]`` —
        exactly the attendance pattern used in training, evaluated only where the
        next ``k`` predictions live.
        """
        dev = self.device
        p = context_ids.shape[0]
        m = p + (k - 1)

        real_x = self.real_input(context_ids, context_feats)  # (p, H)
        x = torch.empty(m, self.hidden, dtype=self.param_dtype, device=dev)
        x[:p] = real_x
        if k > 1:
            x[p:] = self.mask_input(k - 1)

        # positions: 0..p-1 for the real stream, then (p-1)+d for d=1..k-1
        pos = torch.arange(m, device=dev)
        pos[p:] = (p - 1) + torch.arange(1, k, device=dev)

        allow = torch.zeros(m, m, dtype=torch.bool, device=dev)
        allow[:p, :p] = torch.tril(torch.ones(p, p, dtype=torch.bool, device=dev))
        if k > 1:
            allow[p:, :p] = True  # mask slots see the whole real context
            tri = torch.tril(torch.ones(k - 1, k - 1, dtype=torch.bool, device=dev))
            allow[p:, p:] = tri  # depth d attends depths <= d (incl. self)
        bias = to_additive_bias(allow, self.param_dtype)

        hidden = self.forward(x.unsqueeze(0), pos.unsqueeze(0), bias)[0]  # (m, H)
        logits = self.lm_head(hidden)  # (m, V)

        # depth 0 prediction is at the last real slot; depth d at mask slot p+d-1
        idx = torch.empty(k, dtype=torch.long, device=dev)
        idx[0] = p - 1
        if k > 1:
            idx[1:] = torch.arange(p, m, device=dev)
        return logits[idx]

    @torch.no_grad()
    def draft_logits_sequential(
        self, context_ids: torch.Tensor, context_feats: torch.Tensor, k: int
    ) -> torch.Tensor:
        """Greedy autoregressive drafting: ``k`` separate forward passes.

        Each step appends the previously drafted token (embedding + shared hidden
        state, since the target has not produced its feature) and predicts the
        next one. This is the sequential counterpart to :meth:`draft_logits` and
        exists to measure the latency cost that parallel drafting removes.
        """
        dev = self.device
        real_x = self.real_input(context_ids, context_feats)  # (P, H)
        drafted: list[int] = []
        out_logits = []
        for _ in range(k):
            d = len(drafted)
            if d == 0:
                x = real_x
            else:
                tok = torch.tensor(drafted, device=dev)
                x = torch.cat([real_x, self.token_shared_input(tok)], dim=0)
            s = x.shape[0]
            pos = torch.arange(s, device=dev).unsqueeze(0)
            bias = to_additive_bias(
                torch.tril(torch.ones(s, s, dtype=torch.bool, device=dev)), self.param_dtype
            )
            hidden = self.forward(x.unsqueeze(0), pos, bias)[0]
            logits = self.lm_head(hidden[-1])
            out_logits.append(logits)
            drafted.append(int(logits.argmax()))
        return torch.stack(out_logits)  # (k, V)

    # ------------------------------------------------------------------ #
    # Checkpoint IO (drafter-only state; target weights reload separately)
    # ------------------------------------------------------------------ #
    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def save_checkpoint(self, path, *, target_name: str) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "num_layers": len(self.layers),
                "max_depth": self.max_depth,
                "target_name": target_name,
            },
            path,
        )


def load_drafter(path, target) -> ParallelDrafter:
    """Rebuild a drafter from a checkpoint, re-sharing ``target``'s embed/LM head."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    dcfg = DrafterConfig(num_layers=ckpt["num_layers"], max_depth=ckpt["max_depth"])
    model = ParallelDrafter.from_target(target, dcfg)
    model.load_state_dict(ckpt["state_dict"])
    return model
