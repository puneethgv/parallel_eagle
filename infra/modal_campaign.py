"""Run the self-distillation campaign on a Modal cloud A100.

Modal is serverless: each function below runs in a container on an A100-80GB, and
results persist to a Modal Volume (so distill -> train -> bench share state). The
target (Qwen2.5-14B-Instruct) is public, so no HF token is needed.

Usage (after `pip install modal` and `modal setup`):
    modal run infra/modal_campaign.py::smoke      # ~50 examples, validates plumbing (cheap)
    modal run infra/modal_campaign.py::main        # full run: distill -> train -> bench
    modal run infra/modal_campaign.py::gif         # record the side-by-side GIF
    modal volume get pe-data results/benchmark.csv .   # pull results down

Cost: A100-80GB is ~$2.5/hr billed per second; the full run is ~30-60 min (~$2-4).
"""

import os

import modal

# GPU + target are env-overridable so the same app runs the A100/14B headline or a
# card-free fallback (e.g. PE_GPU=L4 PE_TARGET=Qwen/Qwen2.5-7B-Instruct).
GPU = os.environ.get("PE_GPU", "A100-80GB")

REPO = "/root/parallel_eagle"
HF_DIR = "/root/.cache/huggingface"
DATA = "/data"

# Image: the repo + its pinned deps. torch's PyPI wheel bundles CUDA, so no CUDA
# base image is needed. hf_transfer speeds the 14B download.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl")
    .pip_install("hf_transfer")
    .add_local_file("pyproject.toml", f"{REPO}/pyproject.toml", copy=True)
    .add_local_file("README.md", f"{REPO}/README.md", copy=True)
    .add_local_file("LICENSE", f"{REPO}/LICENSE", copy=True)
    .add_local_dir("src", f"{REPO}/src", copy=True)
    .add_local_dir("bench", f"{REPO}/bench", copy=True)
    .add_local_file("scripts/record_demo.sh", f"{REPO}/scripts/record_demo.sh", copy=True)
    .run_commands(f"pip install -e '{REPO}[train]'")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": HF_DIR})
)

app = modal.App("parallel-eagle-campaign", image=image)
pe_data = modal.Volume.from_name("pe-data", create_if_missing=True)      # caches + ckpts + results
hf_cache = modal.Volume.from_name("pe-hf", create_if_missing=True)        # downloaded model weights
VOLS = {DATA: pe_data, HF_DIR: hf_cache}

TARGET = os.environ.get("PE_TARGET", "Qwen/Qwen2.5-14B-Instruct")
_TAG = TARGET.split("/")[-1].replace(".", "").replace("-Instruct", "").lower()
CACHE = f"{DATA}/distill_cache_{_TAG}"
CKPT_DIR = f"{DATA}/checkpoints_{_TAG}"


@app.function(gpu=GPU, volumes=VOLS, timeout=60 * 90)
def distill(target: str = TARGET, examples: int = 4000, batch: int = 24, max_new: int = 256):
    from pathlib import Path

    from pe.config import DistillConfig, TargetConfig
    from pe.distill import build_distill_dataset

    dcfg = DistillConfig(
        out_dir=Path(CACHE), dataset="tatsu-lab/alpaca", split="train",
        max_examples=examples, max_new_tokens=max_new, max_prompt_len=256,
    )
    tcfg = TargetConfig(model_name=target, device="cuda", dtype="bfloat16")
    build_distill_dataset(dcfg, tcfg, batch_size=batch)
    hf_cache.commit()
    pe_data.commit()
    print(f"distill done -> {CACHE}")


@app.function(gpu=GPU, volumes=VOLS, timeout=60 * 90)
def train(target: str = TARGET, layers: int = 2, depth: int = 7, epochs: int = 3):
    from pathlib import Path

    from pe.config import DrafterConfig, TargetConfig, TrainConfig
    from pe.train import train as train_fn

    tcfg = TargetConfig(model_name=target, device="cuda", dtype="bfloat16")
    dcfg = DrafterConfig(num_layers=layers, max_depth=depth)
    tr = TrainConfig(
        feature_cache_dir=Path(CACHE), out_dir=Path(CKPT_DIR),
        epochs=epochs, max_seq_len=512, use_8bit_adam=False,
    )
    train_fn(tcfg, dcfg, tr, dtype="bfloat16")
    pe_data.commit()
    print(f"train done -> {CKPT_DIR}/drafter.pt")


@app.function(gpu=GPU, volumes=VOLS, timeout=60 * 45)
def bench(target: str = TARGET):
    import subprocess

    # inherit stdout (no capture) so each per-(k, mode) row streams live to the logs
    subprocess.run(
        ["python", "-u", f"{REPO}/bench/run_bench.py", "--target", target, "--dtype", "bfloat16",
         "--ckpt", f"{CKPT_DIR}/drafter.pt", "--k-values", "5", "7",
         "--max-new-tokens", "256", "--results-dir", f"{DATA}/results"],
        check=True,
    )
    pe_data.commit()
    print("=== benchmark.csv ===")
    print(open(f"{DATA}/results/benchmark.csv").read())


@app.function(gpu=GPU, volumes=VOLS, timeout=60 * 45)
def diagnose(target: str = TARGET, k: int = 7, max_new: int = 128):
    """Per-depth top-1 accuracy of the trained drafter vs the target's own greedy
    argmax — pinpoints whether low acceptance is a weak drafter or an inefficient
    loop. Reads the checkpoint from the volume; no training."""
    import subprocess

    subprocess.run(
        ["python", "-u", f"{REPO}/bench/diagnose_acceptance.py", "--target", target,
         "--dtype", "bfloat16", "--ckpt", f"{CKPT_DIR}/drafter.pt",
         "--k", str(k), "--max-new-tokens", str(max_new)],
        check=True,
    )


@app.function(gpu=GPU, volumes=VOLS, timeout=60 * 30)
def gif(target: str = TARGET, prompt: str = "Explain how a CPU executes an instruction, step by step."):
    import subprocess

    # agg (cast -> gif) prebuilt binary; asciinema via pip.
    subprocess.run("pip install asciinema", shell=True, check=True)
    subprocess.run(
        "curl -sL https://github.com/asciinema/agg/releases/download/v1.5.0/"
        "agg-x86_64-unknown-linux-gnu -o /usr/local/bin/agg && chmod +x /usr/local/bin/agg",
        shell=True, check=True,
    )
    env = {"TARGET": target, "CKPT": f"{CKPT_DIR}/drafter.pt", "DTYPE": "bfloat16",
           "DEVICE": "cuda", "PROMPT": prompt, "OUT_DIR": f"{DATA}/results", "NAME": "demo"}
    subprocess.run(["bash", f"{REPO}/scripts/record_demo.sh"], env={**os.environ, **env}, check=True)
    pe_data.commit()
    print(f"gif -> {DATA}/results/demo.gif")


@app.function(gpu=GPU, volumes=VOLS, timeout=60 * 150)
def campaign(examples: int = 4000, batch: int = 24, layers: int = 2, depth: int = 7, epochs: int = 3,
             target: str = TARGET):
    """Whole pipeline (distill -> train -> bench) in ONE container, so `modal run
    --detach` keeps the entire run alive after the local process disconnects."""
    import subprocess
    from pathlib import Path

    from pe.config import DistillConfig, DrafterConfig, TargetConfig, TrainConfig
    from pe.distill import build_distill_dataset
    from pe.train import train as train_fn

    tcfg = TargetConfig(model_name=target, device="cuda", dtype="bfloat16")

    print(f"=== [1/3] self-distill {examples} examples from {target} (batch {batch}) ===", flush=True)
    dcfg = DistillConfig(out_dir=Path(CACHE), dataset="tatsu-lab/alpaca", split="train",
                         max_examples=examples, max_new_tokens=256, max_prompt_len=256)
    build_distill_dataset(dcfg, tcfg, batch_size=batch)
    pe_data.commit()
    hf_cache.commit()

    print(f"=== [2/3] train drafter ({layers}L depth {depth}, {epochs} epochs) ===", flush=True)
    tr = TrainConfig(feature_cache_dir=Path(CACHE), out_dir=Path(CKPT_DIR),
                     epochs=epochs, max_seq_len=512, use_8bit_adam=False)
    train_fn(tcfg, DrafterConfig(num_layers=layers, max_depth=depth), tr, dtype="bfloat16")
    pe_data.commit()

    print("=== [3/3] benchmark cached one-forward loop (tree vs vanilla) ===", flush=True)
    subprocess.run(
        ["python", "-u", f"{REPO}/bench/run_bench.py", "--target", target, "--dtype", "bfloat16",
         "--ckpt", f"{CKPT_DIR}/drafter.pt", "--k-values", "5", "7",
         "--max-new-tokens", "256", "--results-dir", f"{DATA}/results"],
        check=True,
    )
    pe_data.commit()
    print("=== RESULTS: benchmark.csv ===", flush=True)
    print(open(f"{DATA}/results/benchmark.csv").read(), flush=True)


@app.function(gpu=GPU, volumes=VOLS, timeout=60 * 240, retries=10)
def finish(layers: int = 4, depth: int = 7, epochs: int = 6, target: str = TARGET):
    """Resume from phase 2: train + bench on an already-distilled cache (skips the
    expensive distill).

    Preemption-resilient: training checkpoints mid-epoch (``save_every``) and commits
    each checkpoint to the volume, so a preempted container's ``retries`` restart
    *resumes* from the last checkpoint instead of restarting from epoch 0. This is
    essential here because one epoch (~12 min) can exceed the time between preemptions,
    so without mid-epoch resume the run could loop forever. retries=10 gives plenty of
    resume segments; each restart only repeats the work since the last commit (~4 min).
    timeout=4h is a generous cap (per-second billing makes an early finish free)."""
    import subprocess
    from pathlib import Path

    from pe.config import DrafterConfig, TargetConfig, TrainConfig
    from pe.train import train as train_fn

    tcfg = TargetConfig(model_name=target, device="cuda", dtype="bfloat16")
    print(f"=== [2/3] train drafter ({layers}L depth {depth}, {epochs} epochs) on cached data ===",
          flush=True)
    # grad_checkpoint off: the 80GB A100 has memory to spare, and checkpointing
    # recomputes the forward in backward (~1.5x slower). num_segments=1 likewise.
    # save_every=250 (~4 min) bounds work lost to a preemption; on_save commits the
    # checkpoint to the volume so the next container can resume it.
    tr = TrainConfig(feature_cache_dir=Path(CACHE), out_dir=Path(CKPT_DIR),
                     epochs=epochs, max_seq_len=512, use_8bit_adam=False,
                     grad_checkpoint=False, save_every=250)
    train_fn(tcfg, DrafterConfig(num_layers=layers, max_depth=depth), tr,
             dtype="bfloat16", on_save=pe_data.commit)
    pe_data.commit()

    print("=== [3/3] benchmark cached one-forward loop (tree vs vanilla) ===", flush=True)
    subprocess.run(
        ["python", "-u", f"{REPO}/bench/run_bench.py", "--target", target, "--dtype", "bfloat16",
         "--ckpt", f"{CKPT_DIR}/drafter.pt", "--k-values", "5", "7",
         "--max-new-tokens", "256", "--results-dir", f"{DATA}/results"],
        check=True,
    )
    pe_data.commit()
    print("=== RESULTS: benchmark.csv ===", flush=True)
    print(open(f"{DATA}/results/benchmark.csv").read(), flush=True)


@app.local_entrypoint()
def resume(layers: int = 4, depth: int = 7, epochs: int = 6):
    """Detach-safe resume: train + bench on the existing distill cache."""
    finish.remote(layers=layers, depth=depth, epochs=epochs)


@app.local_entrypoint()
def diag(k: int = 7, max_new: int = 128):
    """Diagnose the acceptance ceiling on the trained checkpoint (cheap, read-only)."""
    diagnose.remote(k=k, max_new=max_new)


@app.local_entrypoint()
def run_all(examples: int = 4000, batch: int = 24, layers: int = 2, depth: int = 7, epochs: int = 3):
    """Detach-safe full campaign: the single `campaign` container survives disconnect."""
    campaign.remote(examples=examples, batch=batch, layers=layers, depth=depth, epochs=epochs)


@app.local_entrypoint()
def smoke():
    """Cheap end-to-end validation: tiny data, 1 epoch, then benchmark."""
    distill.remote(examples=60, batch=12, max_new=128)
    train.remote(layers=2, depth=7, epochs=1)
    bench.remote()


@app.local_entrypoint()
def main(examples: int = 4000, batch: int = 24, layers: int = 2, depth: int = 7, epochs: int = 3):
    """Full campaign: self-distill -> train -> benchmark."""
    distill.remote(examples=examples, batch=batch)
    train.remote(layers=layers, depth=depth, epochs=epochs)
    bench.remote()
