#!/usr/bin/env bash
# The A100 self-distillation campaign, end to end: generate the target's own
# greedy data, train the drafter on it (prompt-masked, teacher-argmax labels),
# benchmark the shipped cached loop, and record the side-by-side GIF.
#
# Run on the A100 VM (see scripts/azure_a100_setup.sh) after `pip install -e .`
# and `hf auth login`. Tune EXAMPLES / BATCH / LAYERS via env.
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET="${TARGET:-Qwen/Qwen2.5-14B-Instruct}"
DTYPE="${DTYPE:-bfloat16}"
DATASET="${DATASET:-tatsu-lab/alpaca}"
EXAMPLES="${EXAMPLES:-5000}"
BATCH="${BATCH:-24}"
MAX_NEW="${MAX_NEW:-256}"
LAYERS="${LAYERS:-2}"
DEPTH="${DEPTH:-7}"
EPOCHS="${EPOCHS:-2}"
CACHE="${CACHE:-distill_cache_14b}"
CKPT_DIR="${CKPT_DIR:-checkpoints_14b}"

echo "== 1/4 self-distill: $EXAMPLES examples from $TARGET (batched x$BATCH) =="
python -m pe.distill --target "$TARGET" --dataset "$DATASET" \
  --max-examples "$EXAMPLES" --batch-size "$BATCH" --max-new-tokens "$MAX_NEW" \
  --out-dir "$CACHE"

echo "== 2/4 train drafter: ${LAYERS}L depth ${DEPTH}, ${EPOCHS} epochs =="
python -m pe.train --target "$TARGET" --dtype "$DTYPE" \
  --cache-dir "$CACHE" --out-dir "$CKPT_DIR" \
  --num-layers "$LAYERS" --max-depth "$DEPTH" --epochs "$EPOCHS"

echo "== 3/4 benchmark the cached one-forward loop (tree vs vanilla; cached is default) =="
python bench/run_bench.py --target "$TARGET" --dtype "$DTYPE" \
  --ckpt "$CKPT_DIR/drafter.pt" --k-values 5 7 --max-new-tokens 256
python bench/plot.py || true

echo "== 4/4 record side-by-side GIF (needs asciinema + agg) =="
TARGET="$TARGET" CKPT="$CKPT_DIR/drafter.pt" DTYPE="$DTYPE" \
  scripts/record_demo.sh || echo "  (skip GIF if asciinema/agg unavailable)"

echo "done. results/benchmark.csv + docs/demo.gif ready. Remember: az vm deallocate."
