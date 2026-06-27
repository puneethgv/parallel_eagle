#!/usr/bin/env bash
# Record the side-by-side naive-vs-tree streaming demo to a GIF for the README.
#
# Naive decoding dribbles one token at a time; the parallel-tree loop commits
# accepted tokens in bursts and finishes sooner. This captures that visibly.
#
# Requires asciinema (record) + agg (cast -> gif):
#   pipx install asciinema            # or: apt install asciinema
#   cargo install --git https://github.com/asciinema/agg
#
# Usage:
#   scripts/record_demo.sh                          # uses the env defaults below
#   TARGET=Qwen/Qwen2.5-14B-Instruct CKPT=checkpoints_14b/drafter.pt \
#   PROMPT="Explain how a CPU works." scripts/record_demo.sh
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET="${TARGET:-Qwen/Qwen2.5-14B-Instruct}"
CKPT="${CKPT:-checkpoints_14b/drafter.pt}"
PROMPT="${PROMPT:-Explain how a CPU executes an instruction, step by step.}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
MAX_NEW="${MAX_NEW:-160}"
OUT_DIR="${OUT_DIR:-docs}"
NAME="${NAME:-demo}"

for tool in asciinema agg; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "error: '$tool' not found. Install it first (see header of this script)." >&2
    exit 1
  fi
done

mkdir -p "$OUT_DIR"
CAST="$OUT_DIR/$NAME.cast"
GIF="$OUT_DIR/$NAME.gif"

CMD="python bench/demo.py --stream --target $TARGET --ckpt $CKPT --device $DEVICE \
  --dtype $DTYPE --max-new-tokens $MAX_NEW --prompt \"$PROMPT\""

echo "recording: $CMD"
asciinema rec --overwrite --command "$CMD" "$CAST"
agg --theme monokai "$CAST" "$GIF"
echo "wrote $GIF"
