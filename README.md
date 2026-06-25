# parallel_eagle

[![CI](https://github.com/puneethgv/parallel_eagle/actions/workflows/ci.yml/badge.svg)](https://github.com/puneethgv/parallel_eagle/actions/workflows/ci.yml)

A from-scratch **speculative decoder** whose draft model proposes many future
tokens in a **single forward pass** and verifies a **branching tree** of candidates
against the target in one pass — accelerating autoregressive generation while
producing output that is provably identical to plain decoding.

Built to train and run on a single 8 GB consumer GPU, against targets up to a
**4-bit-quantized 7B** model. It implements the whole pipeline in PyTorch: a
frozen-target feature extractor, a feature-conditioned parallel drafter, a
memory-scalable training recipe, **KV-cache decoding** (one target forward per
step) with **lossless greedy and sampling** verification, and a benchmark harness
that measures acceptance length, call efficiency, and wall-clock.

> **Status:** the algorithm, training recipe, int4 + KV-cache serving, and lossless
> verification are complete and tested. See *Results* for measured acceptance and
> the honest wall-clock picture, and *Engineering highlights* for the debugging
> story behind the drafter.

## The idea in one paragraph

Autoregressive decoding emits one token per target forward pass and is
memory-bandwidth bound. *Speculative decoding* fixes this: a cheap draft model
proposes several next tokens that the target verifies in a single pass; correct
tokens are accepted "for free." Here the drafter is **feature-conditioned** — it
consumes the target's own intermediate hidden states rather than raw tokens — so a
tiny network can draft accurately. Crucially it predicts all `K` tokens in **one
parallel pass** instead of `K` sequential passes: the unknown future positions are
filled with a single learnable *shared hidden state* and a learnable *mask-token
embedding*, and positional structure is left to rotary attention. On top of that,
drafting produces a **dynamic tree** of candidates (not a single chain), so one
early mistake no longer throws away the whole draft.

## How it works

**Feature-conditioned drafter.** For each position the drafter input is
`in_proj(concat(token_embedding, feat_proj(target_features)))`, where
`target_features` is a fusion of an early, a middle, and a late target layer. The
token embedding and LM head are **shared with the frozen target** (no copy, no
gradient). The mask representation that stands in for "future, unknown" positions
is a dedicated learnable vector pair (`mask_emb`, `h_shared`) rather than an
unfrozen vocabulary row — same effect, far less memory.

**Parallel multi-token prediction.** Predicting the `d`-th future token from a
real position is a slot at rotary position `pos + d` filled with the shared
hidden state / mask embedding. A single attention pass over the real stream plus
these slots yields `K` token distributions at once. Depth is recoverable from
position via rotary attention, so no depth-specific encoding is added.

**Dynamic tree drafting.** From the one drafter pass we have a distribution at each
depth. Instead of committing to the top-1 chain, a beam keeps the highest
joint-probability continuations — at each depth the top candidates are attached to
every surviving parent and the beam is re-pruned. This concentrates branching where
the drafter is uncertain and yields a compact tree the target verifies in one pass
via a custom tree-attention mask.

**Memory-scalable training.** Training expands each length-`n` sequence into `n·K`
prediction slots, so attention cost grows with `(nK)²`. Two techniques keep this
tractable:
- *Amortized mask construction*: the cross-depth causal mask is position-invariant,
  so it is built once at the maximum length and sliced (a constant-time view) per
  batch.
- *Sequence partitioning*: one sequence is split into `S` segments with gradients
  accumulated across them, instantiating the prediction slots only for the current
  segment. Because every depth of an anchor stays in the same segment, the
  accumulated gradient is **exactly** the full-sequence gradient (verified in
  tests), while the per-pass sequence length shrinks from `nK` toward `~2n`.

**Lossless.** Greedy acceptance only ever commits the target's own argmax tokens,
so the output is identical to plain greedy decoding regardless of draft quality
(verified token-for-token in the test suite).

## Repository layout

```
src/pe/
  config.py     # configuration dataclasses
  target.py     # frozen target: fused hidden states + masked verification forward
  features.py   # offline feature extraction to disk shards
  nn.py         # from-scratch transformer blocks (RMSNorm, RoPE, GQA, SwiGLU)
  drafter.py    # parallel multi-token drafter
  masks.py      # amortized training mask + tree attention mask
  partition.py  # sequence partitioning for within-sequence gradient accumulation
  train.py      # drafter training loop
  decode/
    verify.py     # lossless acceptance (chain + tree)
    baselines.py  # vanilla autoregressive decoding
    chain.py      # parallel chain + sequential chain drafting
    tree.py       # parallel dynamic tree drafting
  serve.py      # single-pass speculative generation loop
bench/          # benchmark sweep, memory-scaling, plotting
tests/          # CPU correctness tests (losslessness, mask + gradient equivalence)
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # add CUDA torch per your platform; ".[train]" adds 8-bit Adam
```

## Quickstart

The defaults target `Qwen/Qwen2.5-0.5B-Instruct` (open, fits 8 GB). Any causal LM
with hidden states + an LM head works via `--target`.

```bash
# 1) cache the frozen target's fused hidden states over training data
python -m pe.features --target Qwen/Qwen2.5-0.5B-Instruct \
    --dataset tatsu-lab/alpaca --split train --max-examples 3000 --max-seq-len 384

# 2) train the parallel drafter (sequence partitioning keeps long contexts in 8 GB)
python -m pe.train --target Qwen/Qwen2.5-0.5B-Instruct --dtype bfloat16 \
    --num-layers 3 --max-depth 6 --max-seq-len 384 --epochs 6 \
    --num-segments 6 --no-8bit-adam

# 3) benchmark every strategy against vanilla greedy
python bench/run_bench.py --target Qwen/Qwen2.5-0.5B-Instruct --k-values 3 5
python bench/plot.py

# memory-scaling demonstration of sequence partitioning
python bench/mem_scaling.py --segments 1 2 3 6
```

## Results

Target `Qwen2.5-0.5B-Instruct`, a 3-layer drafter (`K_train=6`) trained on a single
8 GB RTX 3070, greedy decoding, fp32, 10 held-out prompts, `K=5`
(reproduce with `bench/run_bench.py`):

| Strategy | Acceptance length | Target calls / token | Drafter calls / token | Lossless |
|---|---|---|---|---|
| vanilla autoregressive | 1.000 | 1.000 | 0.00 | ✓ |
| sequential chain | 1.023 | 1.006 | 5.84 | ✓ |
| parallel chain | 1.029 | 1.001 | 0.97 | ✓ |
| **parallel dynamic tree** | **1.151** | **0.901** | 0.87 | ✓ |

![benchmark](results/benchmark.png)

Takeaways:
- **Dynamic tree drafting lifts acceptance +12% over the chain** (1.151 vs 1.029) at
  effectively the same drafter cost, and brings **target forward passes per token
  below 1.0** (0.901) — a net reduction in target compute, losslessly.
- **Parallel drafting is ~6× more call-efficient than sequential** (0.87 vs 5.84
  drafter calls per token) — the cost that one-pass drafting removes.
- Losslessness is exact: every strategy reproduces vanilla greedy token-for-token.

**Memory scaling.** Training one 512-token example through the drafter, varying the
number of sequence-partitioning segments `S` (`bench/mem_scaling.py`):

| Segments `S` | Peak training memory |
|---|---|
| 1 | out of memory |
| 2 | 5.1 GB |
| 3 | 3.5 GB |
| 6 | 2.0 GB |

The loss is identical across `S` (the partitioned gradient equals the
full-sequence gradient exactly), yet the context that **OOMs at `S=1` trains
comfortably at `S=6`** — which is how long-context training fits an 8 GB card.

### Scaling up: int4 7B target + KV-cache decoding

The same code runs against a **pre-quantized 4-bit 7B target** (`--target
unsloth/mistral-7b-instruct-v0.3-bnb-4bit`, ~4 GB, fits 8 GB) with a **KV-cache**
generation loop (`generate_speculative_cached`): the cache holds the confirmed
prefix, so each step does incremental target forwards instead of recomputing the
prefix, and a fair cached vanilla baseline (`vanilla_generate_cached`) is the
comparison. Training stays light via the offline head/feature dump and sequence
partitioning (a 4096-hidden drafter trains in ~5 GB with 8-bit Adam).

Findings on the 7B-int4 target (greedy, lossless):
- **Drafter learning is gated by feature scale.** Late target layers have "massive
  activations" (|x| up to ~220); without normalizing the fused features the drafter
  collapses to a unigram predictor. RMS-normalizing each fused block **and sharing
  the target's final norm** (so the drafter's output is in the LM head's scale)
  breaks the loss plateau (5.4 → ~4.0) and the drafter learns real predictions.
- **Train/inference distribution must match.** Trained on raw instruction text, the
  drafter is in-distribution for that format: **dynamic-tree acceptance ≈ 1.39**
  (chain ≈ 1.14) on held-out instruction prompts — but drops on chat-template
  prompts it never saw. Format-matched training is the lever here.
- **The remaining wall-clock gap is decode-loop efficiency, not the cache.** The
  loop currently does *two* target forwards per step (verify the tree, then commit
  the accepted path) and recomputes the small drafter each step, so it is still
  slower than the (very fast, KV-cached) 7B vanilla baseline at this acceptance.

**Decode loop.** Serving uses a **one-target-forward-per-step** KV-cache loop
(`generate_speculative_cached`): the candidate tree is verified over the cached
prefix, the accepted path's KV is kept by index-selecting the cache (rejected
branches dropped, no recompute), and the loop re-roots on the most recent bonus so
the next step's depth-0 draft is the easy next token. This drives
`target_calls/token → ~1/acceptance` — so the win now scales directly with the
drafter's acceptance, which is the remaining lever (more/format-matched training,
or a deeper drafter). `temperature > 0` uses the same cache with a lossless
rejection-sampling rule (`generate_speculative_sampling`).

## Engineering highlights

The interesting part of this project was debugging *why* a feature-conditioned
drafter wouldn't learn on a real 7B target — a sequence of concrete, measurable
fixes:

- **Massive activations.** Late-layer hidden states have a few outlier dimensions
  with magnitude ~200× the rest. Fed raw into the projection, they made the drafter
  collapse to a unigram predictor (0% next-token match). Fix: **RMS-normalize each
  fused layer block** before projection.
- **LM-head input scale.** The drafter shares the target's frozen LM head, which
  expects input in the target's *final-norm* scale. Giving the drafter its own
  fresh norm left it mis-scaled. Fix: **share the target's final norm**. Together
  with the above, the loss broke its plateau (5.4 → ~4.0) and the drafter started
  predicting.
- **Train/inference distribution mismatch.** Trained on raw instruction text but
  served with a chat template, the drafter was out-of-distribution at generation
  time (acceptance ~1.1). Fix: **format training data with the chat template**;
  in-distribution tree acceptance rose to ~1.4.
- **Two forwards per step.** The first cached loop committed the accepted path with
  a second target forward. Fix: the **one-forward re-rooted loop** above
  (cache index-select + pending bonus), so `target_calls/token → ~1/acceptance`.
- **4-bit "massive activation" interplay, memory, masks.** Along the way: int4
  loading via a pre-quantized checkpoint, an offline head/feature dump so training
  never holds the target, exact-gradient sequence partitioning to fit long-context
  training in 8 GB, and amortized + tree attention masks verified against naive
  rebuilds.

## Tests

```bash
pytest          # losslessness, mask correctness, exact full-vs-partitioned gradients
ruff check .
```

## License

MIT — see [LICENSE](LICENSE).
