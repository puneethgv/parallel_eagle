"""Mid-epoch checkpoint/resume for preemption-resilient training.

On preemptible cloud GPUs an epoch can outlast the time between preemptions, so
training must checkpoint *within* an epoch and a restarted process must continue
from there — not from epoch 0. These tests drive the real ``pe.train.train`` loop
on the toy CPU model: they simulate a preemption (raise inside the save hook),
then resume and assert the run continues from the saved position and finishes.
"""

import json

import pytest
import torch

from pe.config import DrafterConfig, TargetConfig, TrainConfig
from pe.drafter import load_drafter


def _build_cache(cache_dir, tiny_target, n=16, seq_len=20, prompt_len=4):
    """Write a tiny self-distilled feature cache (no heads dump — see test)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    buf = []
    for _ in range(n):
        ids = torch.randint(0, tiny_target.vocab_size, (seq_len,))
        out = tiny_target.forward(ids.unsqueeze(0))
        buf.append(
            {
                "input_ids": ids,
                "features": out.fused[0].to(torch.float16),
                "prompt_len": prompt_len,
                "labels": out.logits[0].argmax(-1).to(torch.long),
            }
        )
    torch.save(buf, cache_dir / "shard_00000.pt")
    (cache_dir / "manifest.json").write_text(
        json.dumps(
            {
                "shards": ["shard_00000.pt"],
                "num_examples": n,
                "feature_dim": buf[0]["features"].shape[-1],
                "feature_layers": list(tiny_target.feature_layers),
                "self_distilled": True,
            }
        )
    )


def _train_cfg(cache_dir, out_dir):
    return TrainConfig(
        feature_cache_dir=cache_dir,
        out_dir=out_dir,
        epochs=2,
        grad_accum=2,
        max_seq_len=64,
        use_8bit_adam=False,
        grad_checkpoint=False,
        save_every=2,
        log_every=100,
    )


def test_resume_continues_from_checkpoint(tiny_target, tmp_path, monkeypatch, capsys):
    import pe.train as train_mod

    # No heads dump in the cache -> train() falls back to load_target_heads, which we
    # point at the toy model (its real config can't be fetched from the Hub).
    monkeypatch.setattr(train_mod, "load_target_heads", lambda tcfg: tiny_target)

    cache, out_dir = tmp_path / "cache", tmp_path / "ck"
    _build_cache(cache, tiny_target)
    tcfg = TargetConfig(model_name="tiny", device="cpu", dtype="float32")
    dcfg = DrafterConfig(num_layers=2, max_depth=4)
    tr = _train_cfg(cache, out_dir)

    # Phase 1: train until the 2nd checkpoint, then simulate a worker preemption by
    # raising inside the save hook (which fires right after state is persisted).
    class Preempted(Exception):
        pass

    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        if calls["n"] >= 2:
            raise Preempted

    with pytest.raises(Preempted):
        train_mod.train(tcfg, dcfg, tr, dtype="float32", on_save=boom)

    state = torch.load(out_dir / "train_state.pt", weights_only=False)
    assert (out_dir / "drafter.pt").exists()
    assert 0 < state["update"] < 16  # interrupted partway, not finished
    assert state["epoch"] < tr.epochs

    # Phase 2: a fresh train() call must resume (not restart) and run to completion.
    resumed_update = state["update"]
    capsys.readouterr()
    train_mod.train(tcfg, dcfg, tr, dtype="float32", on_save=lambda: None)
    out = capsys.readouterr().out
    assert f"resuming: epoch {state['epoch']}" in out  # resume path, not a cold start
    assert f"update {resumed_update}/16" in out  # continued from the saved update count

    final = torch.load(out_dir / "train_state.pt", weights_only=False)
    assert final["epoch"] == tr.epochs  # marks training complete
    # The resulting checkpoint is well-formed and reloadable.
    drafter = load_drafter(out_dir / "drafter.pt", tiny_target)
    assert len(drafter.layers) == dcfg.num_layers


def test_completed_run_is_a_noop_on_reentry(tiny_target, tmp_path, monkeypatch, capsys):
    import pe.train as train_mod

    monkeypatch.setattr(train_mod, "load_target_heads", lambda tcfg: tiny_target)
    cache, out_dir = tmp_path / "cache", tmp_path / "ck"
    _build_cache(cache, tiny_target)
    tcfg = TargetConfig(model_name="tiny", device="cpu", dtype="float32")
    dcfg = DrafterConfig(num_layers=2, max_depth=4)
    tr = _train_cfg(cache, out_dir)

    train_mod.train(tcfg, dcfg, tr, dtype="float32")
    assert torch.load(out_dir / "train_state.pt", weights_only=False)["epoch"] == tr.epochs

    # Re-entering a finished run resumes at epoch == epochs, so it trains nothing.
    capsys.readouterr()
    train_mod.train(tcfg, dcfg, tr, dtype="float32")
    out = capsys.readouterr().out
    assert "resuming: epoch 2" in out
    assert "after epoch" not in out  # no epoch completed -> the training loop never ran


def test_mismatched_config_starts_fresh(tiny_target, tmp_path, monkeypatch, capsys):
    import pe.train as train_mod

    monkeypatch.setattr(train_mod, "load_target_heads", lambda tcfg: tiny_target)
    cache, out_dir = tmp_path / "cache", tmp_path / "ck"
    _build_cache(cache, tiny_target)
    tcfg = TargetConfig(model_name="tiny", device="cpu", dtype="float32")
    tr = _train_cfg(cache, out_dir)

    # Train a 2-layer drafter to completion (writes train_state.pt).
    train_mod.train(tcfg, DrafterConfig(num_layers=2, max_depth=4), tr, dtype="float32")

    # A different shape (3 layers) must NOT resume the 2-layer state.
    capsys.readouterr()
    train_mod.train(tcfg, DrafterConfig(num_layers=3, max_depth=4), tr, dtype="float32")
    out = capsys.readouterr().out
    assert "different run — starting fresh" in out
    assert len(load_drafter(out_dir / "drafter.pt", tiny_target).layers) == 3
