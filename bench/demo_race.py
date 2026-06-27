"""Side-by-side 'race': naive autoregressive vs. parallel-tree speculative decoding.

Two phases so the recorded GIF has no model-load preamble:

  emit   - load the target + drafter, run both decoders on one prompt, and record
           each committed chunk with its wall-clock timestamp to a small JSON file.
  replay - read that JSON (no model, instant start) and render the two streams in
           two live columns at their real recorded pacing: naive dribbles one token
           at a time, the tree commits accepted tokens in bursts and finishes first.

  python bench/demo_race.py emit   --target ... --ckpt ... --out race.json
  python bench/demo_race.py replay race.json            # this is what asciinema records

Output is identical between the two streams (lossless); the replay shows it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

GREEN, DIM, BOLD, YEL, RESET = "\033[92m", "\033[2m", "\033[1m", "\033[93m", "\033[0m"
WHITE, CYAN = "\033[97m", "\033[96m"
NV_BG = TR_BG = "\033[48;5;238m"  # medium-gray column panels (dark gutter splits them)


# --------------------------------------------------------------------------- #
# emit: run both decoders, record timestamped text chunks
# --------------------------------------------------------------------------- #
def _build_prompt(target, text: str) -> list[int]:
    tok = target.tokenizer
    if getattr(tok, "chat_template", None):
        res = tok.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True
        )
        return list(res["input_ids"] if hasattr(res, "keys") else res)
    return list(tok(text).input_ids)


def _load(args):
    from pe.config import TargetConfig
    from pe.drafter import load_drafter
    from pe.target import TargetModel

    target = TargetModel(TargetConfig(model_name=args.target, device=args.device, dtype=args.dtype))
    drafter = load_drafter(args.ckpt, target).to(target.device, target.dtype).eval()
    return target, drafter


def _run_pair(target, drafter, prompt, args):
    """Run naive + tree once on one prompt; return the events/stats dict."""
    from pe.decode.baselines import vanilla_generate_cached
    from pe.serve import generate_speculative_cached

    eos = target.tokenizer.eos_token_id
    pid = _build_prompt(target, prompt)
    spec = dict(k=args.k, mode="tree", tree_top_k=args.tree_top_k,
                tree_max_nodes=args.tree_max_nodes, eos_token_id=eos)
    vanilla_generate_cached(target, pid, 4, eos)  # warm kernels
    generate_speculative_cached(target, drafter, pid, max_new_tokens=4, **spec)

    def recorder():
        events, state, t0 = [], [], []

        def cb(toks: list[int]):
            if not t0:
                t0.append(time.perf_counter())
            now = time.perf_counter() - t0[0]
            prev = target.tokenizer.decode(state, skip_special_tokens=True)
            state.extend(toks)
            events.append([round(now, 4),
                           target.tokenizer.decode(state, skip_special_tokens=True)[len(prev):],
                           len(toks)])

        return events, cb

    nv_events, nv_cb = recorder()
    nv = vanilla_generate_cached(target, pid, args.max_new, eos, on_commit=nv_cb)
    tr_events, tr_cb = recorder()
    tr = generate_speculative_cached(target, drafter, pid, max_new_tokens=args.max_new,
                                     on_commit=tr_cb, **spec)
    return {
        "prompt": prompt,
        "naive": {"events": nv_events, "wall": round(nv.seconds, 3),
                  "tps": round(nv.tokens_per_second, 1), "target_calls": nv.target_calls},
        "tree": {"events": tr_events, "wall": round(tr.seconds, 3),
                 "tps": round(tr.tokens_per_second, 1), "target_calls": tr.target_calls,
                 "acceptance": round(tr.acceptance_length, 2)},
        "lossless": nv.output_ids == tr.output_ids,
    }


def _speedup(out):
    return out["tree"]["tps"] / max(1e-9, out["naive"]["tps"])


def emit(args):
    target, drafter = _load(args)
    out = _run_pair(target, drafter, args.prompt, args)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print("\n=== SPEEDUP (single prompt) ===", flush=True)
    print(f"naive : {out['naive']['tps']:5.1f} tok/s  ({out['naive']['wall']:.2f}s)")
    print(f"tree  : {out['tree']['tps']:5.1f} tok/s  ({out['tree']['wall']:.2f}s)   "
          f"speedup {_speedup(out):.2f}x   acceptance {out['tree']['acceptance']}   "
          f"lossless={out['lossless']}")


def probe(args):
    """Load the model once, run every candidate prompt, and keep the one whose tree
    decode wins by the most — so the demo GIF shows a prompt where speculation helps."""
    target, drafter = _load(args)
    best = None
    print("\n=== PROBE (per-prompt tree speedup) ===", flush=True)
    for prompt in args.prompts:
        out = _run_pair(target, drafter, prompt, args)
        s = _speedup(out)
        print(f"  {s:4.2f}x  accept {out['tree']['acceptance']:.2f}  | {prompt[:54]}", flush=True)
        if best is None or s > _speedup(best):
            best = out
    with open(args.out, "w") as f:
        json.dump(best, f)
    print(f"BEST: {_speedup(best):.2f}x (accept {best['tree']['acceptance']}) "
          f"-> {args.out}\n  prompt: {best['prompt']}", flush=True)


# --------------------------------------------------------------------------- #
# replay: render the recorded streams side by side at real pacing
# --------------------------------------------------------------------------- #
def _wrap(text: str, width: int) -> list[str]:
    out, line = [], ""
    for word in text.split(" "):
        while len(word) > width:  # hard-break overlong tokens
            if line:
                out.append(line)
                line = ""
            out.append(word[:width])
            word = word[width:]
        if not line:
            line = word
        elif len(line) + 1 + len(word) <= width:
            line += " " + word
        else:
            out.append(line)
            line = word
    out.append(line)
    return out


def _timeline(side):
    acc, tl = "", []
    for t, delta, n in side["events"]:
        acc += delta
        tl.append((t, acc, n))
    return tl


def _state_at(tl, t):
    acc, last_n, last_t = "", 1, -9.0
    for et, a, n in tl:
        if et <= t:
            acc, last_n, last_t = a, n, et
        else:
            break
    return acc, last_n, last_t


def replay(args):
    data = json.load(open(args.events))
    nv_tl, tr_tl = _timeline(data["naive"]), _timeline(data["tree"])
    total = max(nv_tl[-1][0] if nv_tl else 0, tr_tl[-1][0] if tr_tl else 0)

    W, H, fps = 34, 9, 20
    sep = f"   {BOLD}{CYAN}┃{RESET}   "  # wide bright gutter so the columns can't blur
    bar = "═" * (W * 2 + 7)

    def column(text, color, bg):
        # wrap to W-2 so a one-space inner margin on each side keeps every rendered
        # line exactly W cells wide — the separator then stays perfectly vertical.
        lines = _wrap(text, W - 2)[-H:]
        lines += [""] * (H - len(lines))
        return [f"{bg}{color} {ln:<{W - 1}}{RESET}" for ln in lines]

    def frame(ct):
        nv_txt, _, _ = _state_at(nv_tl, ct)
        tr_txt, tr_n, tr_lt = _state_at(tr_tl, ct)
        burst = ct - tr_lt < 0.25 and tr_n > 1
        out = [f"\033[2J\033[H{BOLD} parallel_eagle{RESET}{DIM} — naive vs. speculative "
               f"decoding (lossless){RESET}",
               f"{DIM} prompt: {data['prompt'][:W * 2 - 6]}{RESET}", bar,
               f"{NV_BG}{BOLD}{WHITE}{' naive (1 token / step)':<{W}}{RESET}{sep}"
               f"{TR_BG}{BOLD}{GREEN}{' parallel tree (bursts)':<{W}}{RESET}"]
        ncol, tcol = column(nv_txt, WHITE, NV_BG), column(tr_txt, GREEN, TR_BG)
        for a, b in zip(ncol, tcol, strict=True):
            out.append(f"{a}{sep}{b}")
        nv_done = ct >= (nv_tl[-1][0] if nv_tl else 0)
        tr_done = ct >= (tr_tl[-1][0] if tr_tl else 0)
        tstat = f"burst +{tr_n}" if burst else ("done" if tr_done else "...")
        nleft = f" naive    {'done' if nv_done else '...'}"
        tright = f" tree     {tstat}"
        out.append(bar)
        out.append(f"{NV_BG}{BOLD}{WHITE}{nleft:<{W}}{RESET}{sep}"
                   f"{TR_BG}{BOLD}{GREEN}{tright:<{W}}{RESET}")
        return "\n".join(out)

    sys.stdout.write("\033[2J")
    nframes = int(total * fps / args.speed)  # covers the full sim time at any speed
    for f in range(nframes + 1):
        ct = (f / fps) * args.speed
        sys.stdout.write(frame(ct))
        sys.stdout.flush()
        time.sleep(1.0 / fps)
    # hold the finished frame briefly
    for _ in range(int(1.5 * fps)):
        sys.stdout.write(frame(total))
        sys.stdout.flush()
        time.sleep(1.0 / fps)


def replay_one(args):
    """Render ONE stream (naive or tree) full-width, scrolling — recorded on its own
    so two of these GIFs can be composited side by side. Both run for the same total
    duration (the slower stream's), so the two recordings stay frame-aligned."""
    data = json.load(open(args.events))
    which = args.which
    tl = _timeline(data[which])
    end = tl[-1][0] if tl else 0
    total = max(_timeline(data["naive"])[-1][0], _timeline(data["tree"])[-1][0])

    W, H, fps = 40, 10, 20  # H small so the header label never scrolls off the short pty
    color = WHITE if which == "naive" else GREEN
    label = ("naive  ·  one token per step" if which == "naive"
             else "parallel tree  ·  multi-token bursts")
    rule = "─" * (W + 2)

    def frame(ct):
        txt, n, lt = _state_at(tl, ct)
        burst = which == "tree" and ct - lt < 0.3 and n > 1
        done = ct >= end
        lines = _wrap(txt, W)[-H:]
        lines += [""] * (H - len(lines))
        body = "\n".join(f" {color}{ln}{RESET}" for ln in lines)
        stat = (f"{YEL}⚡ burst{RESET}" if burst
                else (f"{color}✓ done{RESET}" if done else f"{DIM}generating…{RESET}"))
        return (f"\033[2J\033[H {BOLD}{color}{label}{RESET}\n {DIM}{rule}{RESET}\n"
                f"{body}\n {DIM}{rule}{RESET}\n {stat}")

    sys.stdout.write("\033[2J")
    for f in range(int(total * fps / args.speed) + 1):
        sys.stdout.write(frame((f / fps) * args.speed))
        sys.stdout.flush()
        time.sleep(1.0 / fps)
    for _ in range(int(1.2 * fps)):
        sys.stdout.write(frame(total))
        sys.stdout.flush()
        time.sleep(1.0 / fps)


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Side-by-side naive vs tree decoding demo.")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("emit", help="run both decoders and record timestamped chunks")
    e.add_argument("--target", required=True)
    e.add_argument("--ckpt", required=True)
    e.add_argument("--device", default="cuda")
    e.add_argument("--dtype", default="bfloat16")
    e.add_argument("--k", type=int, default=5)
    e.add_argument("--tree-top-k", type=int, default=3)
    e.add_argument("--tree-max-nodes", type=int, default=24)
    e.add_argument("--max-new", type=int, default=110)
    e.add_argument("--prompt", default="Explain how a CPU executes an instruction, step by step.")
    e.add_argument("--out", default="race.json")

    r = sub.add_parser("replay", help="render a recorded race side by side")
    r.add_argument("events")
    r.add_argument("--speed", type=float, default=1.0)

    o = sub.add_parser("one", help="render a single stream (for side-by-side compositing)")
    o.add_argument("events")
    o.add_argument("which", choices=["naive", "tree"])
    o.add_argument("--speed", type=float, default=1.0)

    pr = sub.add_parser("probe", help="try many prompts, keep the best-speedup one")
    pr.add_argument("--target", required=True)
    pr.add_argument("--ckpt", required=True)
    pr.add_argument("--device", default="cuda")
    pr.add_argument("--dtype", default="bfloat16")
    pr.add_argument("--k", type=int, default=5)
    pr.add_argument("--tree-top-k", type=int, default=3)
    pr.add_argument("--tree-max-nodes", type=int, default=24)
    pr.add_argument("--max-new", type=int, default=120)
    pr.add_argument("--prompts", nargs="+", required=True)
    pr.add_argument("--out", default="race.json")

    args = p.parse_args()
    {"emit": emit, "replay": replay, "one": replay_one, "probe": probe}[args.cmd](args)


if __name__ == "__main__":
    main()
