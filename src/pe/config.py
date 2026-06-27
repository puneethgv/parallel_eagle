"""Configuration dataclasses shared across training, decoding, and benchmarking.

Architectural dimensions that must match the target model (hidden size, head
counts, vocabulary, RoPE base) are intentionally left as ``None`` here and filled
in from the loaded target's own config at build time, so the drafter can never
drift out of sync with the model it serves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_TARGET = "meta-llama/Llama-3.2-1B-Instruct"


@dataclass
class TargetConfig:
    """How to load the frozen target and which hidden states to expose."""

    model_name: str = DEFAULT_TARGET
    # Decoder layers whose hidden states are fused into the drafter input. ``None``
    # selects depth-relative layers (≈ a quarter, middle, and near-final layer), so
    # the fusion always includes a representation close to what the LM head
    # consumes. An explicit tuple of indices overrides this.
    feature_layers: tuple[int, ...] | None = None
    dtype: str = "bfloat16"
    device: str = "cuda"
    # Pre-quantized checkpoints (4-bit) are detected automatically and loaded with
    # a device map; override the placement here if needed. ``dtype`` is ignored for
    # quantized targets (their compute dtype comes from the checkpoint).
    device_map: object | None = None


@dataclass
class DrafterConfig:
    """Drafter architecture knobs.

    ``num_layers`` and ``max_depth`` are the two levers that matter most: depth
    buys per-step accuracy, ``max_depth`` (K_train) is the number of parallel
    prediction positions the drafter is trained to emit in one pass.
    """

    num_layers: int = 2
    max_depth: int = 8  # K_train
    dropout: float = 0.0
    rms_norm_eps: float = 1e-5

    # Filled from the target model config at build time:
    hidden_size: int | None = None
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    intermediate_size: int | None = None
    vocab_size: int | None = None
    rope_theta: float | None = None
    num_feature_layers: int = 3


@dataclass
class TrainConfig:
    """Drafter training on cached features."""

    feature_cache_dir: Path = Path("features_cache")
    out_dir: Path = Path("checkpoints")
    max_seq_len: int = 2048
    batch_size: int = 1
    grad_accum: int = 8
    num_segments: int = 1  # sequence-partitioning segments S (1 = disabled)
    lr: float = 1e-4
    warmup_ratio: float = 0.0025
    weight_decay: float = 0.0
    epochs: int = 1
    seed: int = 0
    use_8bit_adam: bool = True
    grad_checkpoint: bool = True
    log_every: int = 20
    # Periodic mid-epoch checkpointing for preemption-resilience. 0 = save only at
    # epoch boundaries; >0 = also save every N optimizer updates (and on resume,
    # fast-forward into the in-progress epoch). Essential on preemptible cloud GPUs
    # where an epoch can exceed the mean time-between-preemptions.
    save_every: int = 0


@dataclass
class FeatureConfig:
    """Offline feature-extraction settings."""

    out_dir: Path = Path("features_cache")
    dataset: str = "HuggingFaceH4/ultrachat_200k"
    split: str = "train_sft"
    max_examples: int = 2000
    max_seq_len: int = 2048
    shard_size: int = 256  # examples per shard


@dataclass
class DistillConfig:
    """Self-distillation feature generation.

    Instead of caching features over human-written responses, run the target's
    *own* greedy generation on each prompt and cache features over
    ``[prompt | target response]``. The training label at each response position
    then equals the target's argmax — exactly what decode-time acceptance
    measures — so training directly optimizes acceptance length.
    """

    out_dir: Path = Path("distill_cache")
    dataset: str = "tatsu-lab/alpaca"
    split: str = "train"
    max_examples: int = 2000
    max_prompt_len: int = 512  # truncate the user prompt to this many tokens
    max_new_tokens: int = 256  # cap on the target's generated response length
    shard_size: int = 256


@dataclass
class DecodeConfig:
    """Inference / speculation settings shared by the decode strategies."""

    k_infer: int = 5
    temperature: float = 0.0
    max_new_tokens: int = 256
    seed: int = 0

    # Tree drafting:
    tree_top_k: int = 3  # candidate branches considered per prediction depth
    tree_max_nodes: int = 48  # cap on total drafted nodes verified per step
    tree_min_prob: float = 0.0  # prune paths whose joint prob falls below this

    # Independent-draft baseline only:
    draft_model_name: str | None = None


@dataclass
class BenchConfig:
    """Benchmark sweep settings."""

    target: TargetConfig = field(default_factory=TargetConfig)
    drafter_ckpt: Path = Path("checkpoints/drafter.pt")
    out_dir: Path = Path("bench_out")
    results_dir: Path = Path("results")
    k_values: tuple[int, ...] = (3, 5, 7)
    num_prompts: int = 20
    repeats: int = 3
    max_new_tokens: int = 256
