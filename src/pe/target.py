"""Frozen target-model wrapper.

The target is never trained. This wrapper exposes exactly what the rest of the
system needs from it:

- **fused hidden states** — the concatenation of an early, a middle, and a late
  decoder layer's outputs, which become the drafter's per-position context;
- a **masked verification forward** that accepts an arbitrary additive attention
  bias and explicit position ids, so a flattened candidate *tree* can be scored
  in a single pass;
- the (frozen, weight-tied) token embedding and LM head that the drafter borrows.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .config import TargetConfig


def _is_quantized(model_name: str) -> bool:
    try:
        cfg = AutoConfig.from_pretrained(model_name)
    except Exception:  # noqa: BLE001 - local/test models with no hub config
        return False
    return getattr(cfg, "quantization_config", None) is not None


def _load_target_model(cfg: TargetConfig, dtype):
    """Load a target, taking the device-map path for pre-quantized checkpoints."""
    if _is_quantized(cfg.model_name):
        device_map = cfg.device_map or {"": cfg.device}
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name, device_map=device_map)
        return model, True
    return AutoModelForCausalLM.from_pretrained(cfg.model_name, dtype=dtype), False


def materialize_fp16_weight(module: torch.nn.Module) -> torch.Tensor:
    """Return a plain fp16 weight for a linear/embedding, dequantizing 4-bit if needed."""
    w = module.weight
    if w.__class__.__name__ == "Params4bit":
        import bitsandbytes.functional as bnbf

        return bnbf.dequantize_4bit(w.data, w.quant_state).to(torch.float16).cpu()
    return w.data.to(torch.float16).cpu()

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass
class TargetOutput:
    logits: torch.Tensor  # (B, S, V)
    fused: torch.Tensor  # (B, S, num_feature_layers * H)


class TargetModel:
    """Thin, frozen wrapper around a causal LM."""

    def __init__(self, cfg: TargetConfig, model=None, tokenizer=None):
        self.cfg = cfg
        self.dtype = _DTYPES[cfg.dtype]
        quantized = False
        if model is None:
            model, quantized = _load_target_model(cfg, self.dtype)
        # A 4-bit model is already placed by its device map and must not be moved.
        self.model = (model if quantized else model.to(cfg.device)).eval()
        if quantized:
            self.dtype = self.model.get_input_embeddings().weight.dtype
        for p in self.model.parameters():
            p.requires_grad_(False)

        if tokenizer is None and model is not None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
            except Exception:  # noqa: BLE001 - tests pass a model with no hub entry
                tokenizer = None
        self.tokenizer = tokenizer

        hidden_layers = self.model.config.num_hidden_layers
        self.feature_layers = self._resolve_feature_layers(cfg.feature_layers, hidden_layers)

    @staticmethod
    def _resolve_feature_layers(
        layers: tuple[int, ...] | None, num_layers: int
    ) -> tuple[int, ...]:
        """Map requested indices onto valid hidden-state slots.

        ``output_hidden_states`` returns ``num_layers + 1`` tensors (index 0 is the
        embedding output). ``None`` picks depth-relative layers (≈ quarter / middle /
        near-final) so the fusion includes a near-output representation regardless of
        model depth. Explicit (possibly negative / out-of-range) indices are clamped,
        and tiny test models still get three distinct layers.
        """
        n = num_layers  # last valid hidden-state index
        if layers is None:
            layers = tuple(round(n * f) for f in (0.25, 0.55, 0.95))
        out = []
        for layer in layers:
            idx = layer if layer >= 0 else n + 1 + layer
            idx = max(1, min(idx, n))
            out.append(idx)
        # Keep them distinct and sorted when the model is too shallow for the defaults.
        if len(set(out)) < len(out):
            out = sorted({max(1, min(round(n * frac), n)) for frac in (0.25, 0.55, 0.95)})
            while len(out) < len(layers):
                out.append(out[-1])
        return tuple(out)

    @property
    def config(self):
        return self.model.config

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.model.config.vocab_size

    @property
    def num_feature_layers(self) -> int:
        return len(self.feature_layers)

    @property
    def feature_dim(self) -> int:
        return self.hidden_size * self.num_feature_layers

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.get_input_embeddings()

    def get_lm_head(self) -> torch.nn.Module:
        return self.model.get_output_embeddings()

    def _fuse(self, hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return torch.cat([hidden_states[i] for i in self.feature_layers], dim=-1)

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> TargetOutput:
        """Run the target and return logits plus fused hidden states.

        ``attention_mask`` may be a standard 2D padding mask or a 4D additive bias
        (``(B, 1, q, kv)``) for tree verification; both are passed through to the
        underlying model unchanged.
        """
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=False,
        )
        return TargetOutput(logits=out.logits, fused=self._fuse(out.hidden_states))


@dataclass
class TargetHeads:
    """Just the frozen embedding + LM head (and dims) the drafter needs.

    Used at train time so the full target never has to occupy the GPU: only the
    two large matrices the drafter shares are kept resident.
    """

    config: object
    hidden_size: int
    vocab_size: int
    feature_dim: int
    num_feature_layers: int
    feature_layers: tuple[int, ...]
    embed: torch.nn.Module
    head: torch.nn.Module

    def get_input_embeddings(self):
        return self.embed

    def get_lm_head(self):
        return self.head


def _heads_from_weights(
    config, embed_w: torch.Tensor, head_w: torch.Tensor, feature_layers, device, dtype
) -> TargetHeads:
    vocab, hidden = embed_w.shape
    embed = torch.nn.Embedding(vocab, hidden)
    embed.weight.data = embed_w.to(dtype)
    head = torch.nn.Linear(hidden, head_w.shape[0], bias=False)
    head.weight.data = head_w.to(dtype)
    for m in (embed, head):
        m.to(device)
        for p in m.parameters():
            p.requires_grad_(False)
    return TargetHeads(
        config=config,
        hidden_size=config.hidden_size,
        vocab_size=config.vocab_size,
        feature_dim=config.hidden_size * len(feature_layers),
        num_feature_layers=len(feature_layers),
        feature_layers=tuple(feature_layers),
        embed=embed,
        head=head,
    )


def dump_target_heads(model, path) -> None:
    """Save fp16 embedding + LM-head weights (dequantizing 4-bit) for lean training."""
    torch.save(
        {
            "embed": materialize_fp16_weight(model.get_input_embeddings()),
            "head": materialize_fp16_weight(model.get_output_embeddings()),
        },
        path,
    )


def load_heads_from_dump(
    dump_path, cfg: TargetConfig, feature_layers, feature_dim_check: int | None = None
) -> TargetHeads:
    """Build shared heads from a dumped weight file + the target config (no model load)."""
    config = AutoConfig.from_pretrained(cfg.model_name)
    fl = TargetModel._resolve_feature_layers(feature_layers, config.num_hidden_layers)
    w = torch.load(dump_path, map_location="cpu", weights_only=False)
    return _heads_from_weights(config, w["embed"], w["head"], fl, cfg.device, _DTYPES[cfg.dtype])


def load_target_heads(cfg: TargetConfig) -> TargetHeads:
    """Load only the embedding and LM head onto the device; free the rest."""
    import gc

    model, quantized = _load_target_model(cfg, _DTYPES[cfg.dtype])
    config = model.config
    feature_layers = TargetModel._resolve_feature_layers(
        cfg.feature_layers, config.num_hidden_layers
    )
    dtype = model.get_input_embeddings().weight.dtype if quantized else _DTYPES[cfg.dtype]
    heads = _heads_from_weights(
        config,
        materialize_fp16_weight(model.get_input_embeddings()),
        materialize_fp16_weight(model.get_output_embeddings()),
        feature_layers,
        cfg.device,
        dtype,
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return heads
