"""Composite two single-stream terminal GIFs into one side-by-side GIF.

Both inputs are sampled on a common clock (``--fps``) over the longer of the two
durations, so the panels stay time-aligned: the faster stream simply finishes
first and then holds. Each frame is cropped to ``--cols`` terminal columns to drop
the recorder's empty right margin, then the two are placed side by side with a gap
and a thin divider.

    python bench/sidebyside.py naive.gif tree.gif out.gif
"""

from __future__ import annotations

import argparse

from PIL import Image, ImageDraw


def _frames(path: str):
    im = Image.open(path)
    out, t = [], 0.0
    for i in range(im.n_frames):
        im.seek(i)
        out.append((t, im.convert("RGB").copy()))
        t += im.info.get("duration", 50) / 1000.0
    return out, t


def _at(frames, t):
    img = frames[0][1]
    for ft, fi in frames:
        if ft <= t:
            img = fi
        else:
            break
    return img


def main():
    p = argparse.ArgumentParser(description="Stitch two terminal GIFs side by side.")
    p.add_argument("left")
    p.add_argument("right")
    p.add_argument("out")
    p.add_argument("--cols", type=int, default=44, help="terminal columns to keep per side")
    p.add_argument("--termcols", type=int, default=80, help="columns the GIFs were recorded at")
    p.add_argument("--gap", type=int, default=26)
    p.add_argument("--fps", type=int, default=20)
    a = p.parse_args()

    lf, lt = _frames(a.left)
    rf, rt = _frames(a.right)
    total = max(lt, rt)
    px_per_col = lf[0][1].width / a.termcols
    cw = int(px_per_col * a.cols)
    h = max(lf[0][1].height, rf[0][1].height)

    out = []
    for k in range(int(total * a.fps) + 1):
        t = k / a.fps
        left = _at(lf, t).crop((0, 0, cw, h))
        right = _at(rf, t).crop((0, 0, cw, h))
        canvas = Image.new("RGB", (cw * 2 + a.gap, h), (13, 13, 13))
        canvas.paste(left, (0, 0))
        canvas.paste(right, (cw + a.gap, 0))
        d = ImageDraw.Draw(canvas)
        x = cw + a.gap // 2
        d.line([(x, 6), (x, h - 6)], fill=(90, 90, 90), width=2)
        out.append(canvas)

    out[0].save(a.out, save_all=True, append_images=out[1:],
                duration=int(1000 / a.fps), loop=0, optimize=True)
    print(f"wrote {a.out}: {len(out)} frames, {cw * 2 + a.gap}x{h}")


if __name__ == "__main__":
    main()
