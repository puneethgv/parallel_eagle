.PHONY: install quickstart features distill train bench demo record-demo test lint format

# Override TARGET / CKPT / PROMPT on the command line, e.g.
#   make demo TARGET=Qwen/Qwen2.5-14B-Instruct CKPT=checkpoints_14b/drafter.pt
TARGET ?= Qwen/Qwen2.5-14B-Instruct
CKPT   ?= checkpoints_14b/drafter.pt
PROMPT ?= Explain how a CPU executes an instruction, step by step.

install:
	pip install -e ".[train,dev]"

# Full pipeline on a toy model in seconds, CPU, no downloads (smoke check).
quickstart:
	python scripts/quickstart.py

# Offline: run the frozen target over training data, cache fused hidden states.
features:
	python -m pe.features

# Self-distillation: cache features over the target's OWN greedy generations, so
# the training label equals the target's argmax (directly optimizes acceptance).
distill:
	python -m pe.distill

# Train the parallel drafter on cached features.
train:
	python -m pe.train

# Benchmark all decode strategies; writes CSV + plots.
bench:
	python bench/run_bench.py

# Live side-by-side: naive (token-by-token) vs parallel tree (multi-token bursts).
demo:
	python bench/demo.py --stream --target "$(TARGET)" --ckpt "$(CKPT)" --prompt "$(PROMPT)"

# Record the side-by-side stream to docs/demo.gif (needs asciinema + agg).
record-demo:
	TARGET="$(TARGET)" CKPT="$(CKPT)" PROMPT="$(PROMPT)" scripts/record_demo.sh

test:
	pytest -q

lint:
	ruff check .

format:
	ruff format .
