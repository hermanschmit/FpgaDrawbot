"""
Inspect and shrink drawbot .bin files.

A .bin is float32 (x, y) pairs (Segments.binWrite: (col, row) of the source
image, scaled to roughly [-1, 1]) followed by NaN padding. Generators often
emit points far closer together than the pen can resolve, so this tool works
in physical units: give it the drawn size of the longest side (--size_mm)
and the pen line width (--pen_mm).

- stats (always): point count, path length, step-length distribution in mm,
  and how many steps are below the pen width / machine grid.
- shrink (--out): optional snap to a machine grid (--grid_mm), drop
  consecutive duplicates, then Ramer-Douglas-Peucker simplification with
  max deviation --tol_mm. Output keeps the same format.
- svg (--svg): true-scale vector render of the (shrunk, if --out is given)
  path in mm at pen width -- zoom in freely to see exactly what the file
  tells the pen to do.
- compare (--png): renders original and shrunk at real pen width side by
  side plus a difference panel, and prints how many pixels changed. If the
  difference is invisible at pen width, the dropped points were wasted
  precision.
"""
import argparse

import numpy as np

NAN_PAD = 20


def bin_read(fn):
    a = np.fromfile(fn, dtype=np.float32)
    nan = np.flatnonzero(np.isnan(a))
    n = nan[0] if len(nan) else len(a)
    if len(nan) and not np.all(np.isnan(a[n:])):
        print("Warning: NaNs inside the point data; reading only up to the first one")
    return a[:n - n % 2].reshape(-1, 2).astype(np.float64)


def bin_write(fn, pts):
    out = np.concatenate((np.asarray(pts, dtype=np.float32).ravel(),
                          np.full(NAN_PAD, np.nan, dtype=np.float32)))
    out.tofile(fn)


def dedupe(pts):
    keep = np.concatenate(([True], np.any(np.diff(pts, axis=0) != 0, axis=1)))
    return pts[keep]


def snap(pts, grid):
    return dedupe(np.round(pts / grid) * grid)


def rdp(pts, tol):
    """Ramer-Douglas-Peucker, iterative. Every dropped point lies within tol
    of the kept polyline."""
    n = len(pts)
    if n < 3:
        return pts.copy()
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        a, b = pts[i], pts[j]
        seg = pts[i + 1:j] - a
        # Distance to the segment, not the infinite line: a path that doubles
        # back on itself (A -> B -> point between A and B) must keep B.
        ab = b - a
        L2 = ab @ ab
        t = np.clip(seg @ ab / L2, 0.0, 1.0) if L2 > 0 else np.zeros(len(seg))
        d = np.hypot(*(seg - t[:, None] * ab).T)
        k = np.argmax(d)
        if d[k] > tol:
            m = i + 1 + k
            keep[m] = True
            stack.append((i, m))
            stack.append((m, j))
    return pts[keep]


def stats(pts, mm, pen_mm, grid_mm, label):
    steps = np.hypot(*np.diff(pts, axis=0).T) * mm
    ext = (pts.max(axis=0) - pts.min(axis=0)) * mm
    print("%s: %d points (%.1f KB), path %.2f m, drawing %.0f x %.0f mm"
          % (label, len(pts), (2 * len(pts) + NAN_PAD) * 4 / 1024.0,
             steps.sum() / 1000.0, ext[0], ext[1]))
    if len(steps):
        p = np.percentile(steps, [1, 10, 50, 90, 99])
        print("  step mm  p1 %.3f  p10 %.3f  median %.3f  p90 %.3f  p99 %.3f  max %.3f"
              % (tuple(p) + (steps.max(),)))
        for name, t in (("pen/4", pen_mm / 4), ("pen width", pen_mm), ("grid", grid_mm)):
            if t > 0:
                print("  steps shorter than %s (%.3f mm): %.1f%%"
                      % (name, t, 100.0 * np.mean(steps < t)))


def render(pts, mm, pen_mm, dpi, lo, hi):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    size_in = (hi - lo) * mm / 25.4 + 2 * pen_mm / 25.4
    fig = plt.figure(figsize=tuple(size_in), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    pad = pen_mm / mm
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(hi[1] + pad, lo[1] - pad)  # image rows grow downward
    ax.plot(pts[:, 0], pts[:, 1], 'k-', lw=pen_mm / 25.4 * 72, solid_capstyle='round',
            solid_joinstyle='round')
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, 0].astype(float) / 255.0
    plt.close(fig)
    return img


def compare(orig, new, mm, pen_mm, dpi, fn):
    import imageio.v2 as imageio
    lo, hi = orig.min(axis=0), orig.max(axis=0)
    a = render(orig, mm, pen_mm, dpi, lo, hi)
    b = render(new, mm, pen_mm, dpi, lo, hi)
    diff = np.abs(a - b)
    inked = np.mean(a < 0.5)
    changed = np.mean(diff > 0.5)
    print("Render @ %d dpi: ink coverage %.1f%%, pixels changed %.3f%% (%.2f%% of inked area)"
          % (dpi, 100 * inked, 100 * changed, 100 * changed / max(inked, 1e-9)))
    # Difference panel: grey = original ink, red = lost, blue = added.
    d = np.dstack([np.where(a < 0.5, 0.8, 1.0)] * 3)
    d[(a < 0.5) & (b >= 0.5)] = (1, 0, 0)
    d[(b < 0.5) & (a >= 0.5)] = (0, 0, 1)
    gap = np.ones((a.shape[0], 10, 3))
    out = np.hstack([np.dstack([a] * 3), gap, np.dstack([b] * 3), gap, d])
    imageio.imwrite(fn, (out * 255).astype(np.uint8))


def svg_write(pts, mm, pen_mm, fn):
    """True-scale vector render of a .bin path: units are mm, stroke is the
    pen width, and the whole path is one polyline (as the pen draws it)."""
    lo = pts.min(axis=0)
    p = (pts - lo) * mm + pen_mm
    w, h = p.max(axis=0) + pen_mm
    with open(fn, 'w') as f:
        f.write('<svg xmlns="http://www.w3.org/2000/svg" width="%.2fmm" height="%.2fmm" '
                'viewBox="0 0 %.3f %.3f">\n' % (w, h, w, h))
        f.write('<rect width="100%" height="100%" fill="white"/>\n')
        f.write('<polyline fill="none" stroke="black" stroke-width="%g" '
                'stroke-linejoin="round" stroke-linecap="round" points="' % pen_mm)
        for i in range(0, len(p), 4096):
            f.write(' '.join('%.3f,%.3f' % (x, y) for x, y in p[i:i + 4096]))
            f.write('\n')
        f.write('"/>\n</svg>\n')


def main(argv=None):
    ap = argparse.ArgumentParser(description='Inspect / shrink drawbot .bin files')
    ap.add_argument('input_bin_fn')
    ap.add_argument('--size_mm', type=float, default=342.0,
                    help='Physical size of the longest side of the drawing (the bot draws the [-1, 1] range as 342 mm)')
    ap.add_argument('--pen_mm', type=float, default=0.5, help='Pen line width')
    ap.add_argument('--grid_mm', type=float, default=0.0,
                    help='Snap to this machine resolution before simplifying (0 = off)')
    ap.add_argument('--tol_mm', type=float, default=0.05,
                    help='RDP max deviation. Default 0.05 mm: on the calibration '
                         'drawing, +-0.043 mm jitter looked straight and a '
                         '0.1 mm bump was invisible')
    ap.add_argument('--out', default=None, help='Write shrunk .bin here')
    ap.add_argument('--png', default=None,
                    help='Write original | shrunk | difference render here')
    ap.add_argument('--svg', default=None,
                    help='Write a true-scale SVG (mm, pen-width stroke) of the result here')
    ap.add_argument('--dpi', type=int, default=300, help='Render resolution for --png')
    args = ap.parse_args(argv)

    pts = bin_read(args.input_bin_fn)
    mm = args.size_mm / np.max(pts.max(axis=0) - pts.min(axis=0))  # mm per bin unit
    stats(pts, mm, args.pen_mm, args.grid_mm, "original")
    if args.out is None and args.png is None:
        if args.svg:
            svg_write(pts, mm, args.pen_mm, args.svg)
        return

    tol = args.tol_mm
    new = snap(pts, args.grid_mm / mm) if args.grid_mm > 0 else dedupe(pts)
    new = rdp(new, tol / mm)
    stats(new, mm, args.pen_mm, args.grid_mm, "shrunk (tol %.3f mm)" % tol)
    print("  %.1f%% of points kept" % (100.0 * len(new) / len(pts)))
    if args.out:
        bin_write(args.out, new)
    if args.svg:
        svg_write(new, mm, args.pen_mm, args.svg)
    if args.png:
        compare(pts, new, mm, args.pen_mm, args.dpi, args.png)


if __name__ == "__main__":
    main()
