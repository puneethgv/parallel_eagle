"""Shared test fixtures: a tiny randomly-initialized causal LM on CPU.

Keeping the model this small (hidden 32, vocab 64) lets correctness tests —
including the lossless-decoding tests — run in CI without a GPU or a model
download.
"""

import pytest
import torch


@pytest.fixture(scope="session")
def tiny_target():
    from transformers import LlamaConfig, LlamaForCausalLM

    from pe.config import TargetConfig
    from pe.target import TargetModel

    torch.manual_seed(0)
    config = LlamaConfig(
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        vocab_size=64,
        max_position_embeddings=256,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
    )
    model = LlamaForCausalLM(config)
    model.eval()
    tcfg = TargetConfig(model_name="tiny", device="cpu", dtype="float32")
    return TargetModel(tcfg, model=model, tokenizer=None)
