"""
Calibration test pattern for the drawbot: one continuous path (no pen lifts)
made of graded features, used to find what the pen/machine can actually
resolve so generators and BinTool can drop invisible detail.

The pattern is laid out in nominal millimetres on a SIZE x SIZE frame. The
.bin is normalized to [-1, 1], so the real size is whatever the bot draws:
measure the frame and scale every nominal value below by measured/SIZE.

  Frame  square with ticks every 10 (long ticks every 50) on top/left edges:
         physical scale, x/y aspect.
  A      groups of 9 parallel lines at decreasing spacing: where lines merge
         ~= pen width; smallest clean spacing = minimum useful line pitch.
  B      wave grid, wavelength (columns) x amplitude (rows): finest wave
         that still reads as a wave -> tone-wave limits (e.g. Guilloche).
  C1     straight line with rectangular bumps of decreasing height: smallest
         visible deviation -> simplification tolerance (BinTool --tol_mm).
  C2     straight line made of dense points (0.5 apart) jittered by +-d:
         the kind of sub-visible noise generators emit; where it looks
         straight, those points are wasted.
  D      circles of decreasing diameter: smallest drawable feature,
         roundness (an oval hints at x/y backlash or scale mismatch).
  E      zigzags with apex angle 90/60/30/15 deg: corner overshoot/rounding.
  F      back-and-forth combs (vertical and horizontal lines, 2 apart):
         uneven alternating spacing = backlash in that axis.
  G      star of diameters through one centre: repeatability.

Straight connector lines between panels are part of the path.

Optional --speed_prefix P also writes P_sparse.bin, P_0p5mm.bin and
P_0p05mm.bin: the same nested squares, with only corner points, a point
every 0.5, and a point every 0.05 (nominal mm). Timing each shows the
per-point overhead, i.e. how much faster a shrunk file draws.
"""
import argparse

import numpy as np

import Segments

SIZE = 200.0

SPACINGS = [2.0, 1.4, 1.0, 0.7, 0.5, 0.35, 0.25]
WAVELENGTHS = [4.0, 2.0, 1.0, 0.5, 0.25]
AMPLITUDES = [2.0, 1.0, 0.5, 0.25, 0.12]
BUMPS = [1.0, 0.5, 0.25, 0.12, 0.06, 0.03]
JITTERS = [0.2, 0.1, 0.05, 0.025, 0.012]
DIAMETERS = [10.0, 5.0, 3.0, 2.0, 1.0, 0.5]
ANGLES = [90, 60, 30, 15]


class _Path:
    def __init__(self, start):
        self.parts = [np.array([start], dtype=float)]

    def to(self, *pts):
        self.parts.append(np.array(pts, dtype=float))

    def extend(self, arr):
        self.parts.append(np.asarray(arr, dtype=float))

    @property
    def pos(self):
        return self.parts[-1][-1]

    def array(self):
        pts = np.vstack(self.parts)
        keep = np.concatenate(([True], np.any(np.diff(pts, axis=0) != 0, axis=1)))
        return pts[keep]


def build_pattern():
    """Returns (points as (x, y) mm with y down, labels as (x, y, text, size))."""
    labels = []

    def label(x, y, text, size=2.0):
        labels.append((x, y, text, size))

    c = SIZE / 2
    p = _Path((c, c))  # start at the centre, where the pen starts
    p.to((0, c))

    # Frame, clockwise from the left edge midpoint; ticks are drawn as
    # retraced spurs so the path stays continuous.
    def tick_len(v):
        return 5.0 if v % 50 == 0 else 3.0
    for y in range(int(c) - 10, 0, -10):
        p.to((0, y), (tick_len(y), y), (0, y))
    p.to((0, 0))
    for x in range(10, int(SIZE), 10):
        p.to((x, 0), (x, tick_len(x)), (x, 0))
    p.to((SIZE, 0), (SIZE, SIZE), (0, SIZE))
    for y in range(int(SIZE) - 10, int(c), -10):
        p.to((0, y), (tick_len(y), y), (0, y))
    p.to((0, c))
    label(SIZE + 2, 4, "Frame %g x %g, ticks every 10 (long every 50)" % (SIZE, SIZE), 2.5)

    # A: line spacing.
    p.to((0, 12))  # along the already-drawn frame
    label(10, 9, "A  line spacing (9 lines per group)", 2.5)
    for g, s in enumerate(SPACINGS):
        x0, x1 = 10 + 26 * g, 32 + 26 * g
        for i in range(9):
            y = 12 + i * s
            p.to(*([(x0, y), (x1, y)] if i % 2 == 0 else [(x1, y), (x0, y)]))
        label(x0, 32, "%g" % s)

    # B: wave grid. Rows alternate direction; a cell is a whole number of
    # wavelengths so it starts and ends on the row's centre line.
    label(10, 39, "B  waves: columns = wavelength, rows = amplitude", 2.5)
    for ci, lam in enumerate(WAVELENGTHS):
        label(13 + 36 * ci + 12, 94, "wl %g" % lam)
    for r, amp in enumerate(AMPLITUDES):
        y = 45 + 10 * r
        label(190.5, y + 0.7, "a %g" % amp, 1.6)
        row = []
        for ci, lam in enumerate(WAVELENGTHS):
            x0 = 13 + 36 * ci
            xs = np.linspace(x0, x0 + 30, int(round(30 / lam * 16)) + 1)
            row.append(np.column_stack((xs, y + amp * np.sin(2 * np.pi * (xs - x0) / lam))))
        row = np.vstack(row)
        if r % 2 == 0:
            row = row[::-1]
        p.extend(row)

    # C1: bumps.
    label(10, 106, "C1  bump height     C2 (below)  point jitter +-d, points 0.5 apart", 2.5)
    p.to((10, 112))
    for k, h in enumerate(BUMPS):
        x = 20 + 25 * k
        p.to((x, 112), (x, 112 - h), (x + 2, 112 - h), (x + 2, 112))
        label(x - 1, 116, "%g" % h)
    p.to((190, 112), (190, 120))

    # C2: jitter, right to left.
    for k, d in enumerate(JITTERS):
        xa = 188 - 34 * k
        xs = np.arange(xa, xa - 30 - 1e-9, -0.5)
        ys = 120 + d * (-1.0) ** np.arange(len(xs))
        ys[0] = ys[-1] = 120
        p.to((xa, 120))
        p.extend(np.column_stack((xs, ys)))
        label(xa - 20, 124, "d %g" % d)
    p.to((10, 120))

    # D: circles sitting on a baseline at y = 140.
    label(10, 128, "D  circle diameter", 2.5)
    p.to((10, 140))
    for d, cx in zip(DIAMETERS, [20, 38, 52, 63, 72, 79]):
        r = d / 2
        t = np.linspace(0, 2 * np.pi, 65)
        p.to((cx, 140))
        p.extend(np.column_stack((cx + r * np.sin(t), 140 - r + r * np.cos(t))))
        label(cx - 1.5, 144, "%g" % d)

    # E: zigzags, two teeth per apex angle, 8 tall.
    label(105, 128, "E  apex angle (deg)", 2.5)
    x = 105.0
    p.to((x, 140))
    for a in ANGLES:
        w = 2 * 8 * np.tan(np.radians(a) / 2)
        p.to((x + w / 2, 132), (x + w, 140), (x + 1.5 * w, 132), (x + 2 * w, 140))
        label(x, 144, "%d" % a)
        x += 2 * w + 3
        p.to((x, 140))

    # F: combs. Vertical lines 2 apart drawn alternately down/up, then
    # horizontal lines drawn alternately left/right.
    label(100, 176, "F  backlash combs (lines 2 apart)", 2.5)
    for i in range(31):
        xv = 190 - 2 * i
        p.to(*([(xv, 150), (xv, 170)] if i % 2 == 0 else [(xv, 170), (xv, 150)]))
    for j in range(11):
        yh = 170 - 2 * j
        p.to(*([(120, yh), (100, yh)] if j % 2 == 0 else [(100, yh), (120, yh)]))

    # G: star of 12 diameters through one centre, alternating direction.
    label(30, 186, "G  repeatability star", 2.5)
    cx, cy, r = 50.0, 168.0, 12.0
    for k in range(12):
        u = np.array([np.cos(np.radians(15 * k)), np.sin(np.radians(15 * k))])
        a, b = (cx + r * u[0], cy + r * u[1]), (cx - r * u[0], cy - r * u[1])
        p.to(*([a, b] if k % 2 == 0 else [b, a]))

    return p.array(), labels


def speed_squares(step):
    """Nested squares (side 40..200, centred), resampled to a point every
    `step` nominal mm (0 = corners only)."""
    corners = [(SIZE / 2, SIZE / 2)]  # start where the pen starts
    for side in range(40, int(SIZE) + 1, 40):
        lo, hi = (SIZE - side) / 2, (SIZE + side) / 2
        corners += [(lo, lo), (hi, lo), (hi, hi), (lo, hi), (lo, lo)]
    corners = np.array(corners, dtype=float)
    if step <= 0:
        return corners
    out = [corners[:1]]
    for a, b in zip(corners[:-1], corners[1:]):
        n = max(1, int(np.ceil(np.hypot(*(b - a)) / step)))
        out.append(a + (b - a) * np.linspace(0, 1, n + 1)[1:, None])
    return np.vstack(out)


def to_segments(pts):
    """(x, y) mm -> Segments in the repo's (row, col) convention."""
    s = Segments.Segments()
    s.append(np.ascontiguousarray(pts[:, ::-1]))
    return s


def write_bin(pts, fn):
    s = to_segments(pts)
    s.scaleBin()
    s.binWrite(fn)


def legend_write(pts, labels, fn, questions):
    w, h = SIZE + 110, SIZE + 12
    with open(fn, 'w') as f:
        f.write('<svg xmlns="http://www.w3.org/2000/svg" width="%gmm" height="%gmm" '
                'viewBox="-5 -5 %g %g">\n' % (w, h, w, h))
        f.write('<rect x="-5" y="-5" width="100%" height="100%" fill="white"/>\n')
        f.write('<polyline fill="none" stroke="black" stroke-width="0.15" '
                'stroke-linejoin="round" points="')
        f.write(' '.join('%.3f,%.3f' % (x, y) for x, y in pts))
        f.write('"/>\n')
        f.write('<circle cx="%g" cy="%g" r="1.2" fill="none" stroke="green" stroke-width="0.3"/>\n'
                % (SIZE / 2, SIZE / 2))
        f.write('<text x="%g" y="%g" font-size="2" fill="green" font-family="sans-serif">start</text>\n'
                % (SIZE / 2 + 2, SIZE / 2 - 1.5))
        for x, y, text, size in labels:
            f.write('<text x="%g" y="%g" font-size="%g" fill="#1060c0" '
                    'font-family="sans-serif">%s</text>\n' % (x, y, size, text))
        y = 12
        f.write('<text x="%g" y="%g" font-size="3" font-weight="bold" '
                'font-family="sans-serif">Report back</text>\n' % (SIZE + 4, y))
        for q in questions:
            y += 6
            f.write('<text x="%g" y="%g" font-size="2.3" font-family="sans-serif">%s</text>\n'
                    % (SIZE + 4, y, q))
        f.write('</svg>\n')


QUESTIONS = [
    "1. Frame width and height as drawn (mm)",
    "2. Pen tip / line width (mm)",
    "3. A: smallest spacing with 9 separate lines",
    "4. B: per amplitude row, smallest wavelength still a wave",
    "5. C1: smallest visible bump",
    "6. C2: largest jitter that looks like a plain line",
    "7. D: smallest circle that is a ring; any ovals?",
    "8. E: overshoot or rounding at sharp apexes?",
    "9. F: alternating uneven spacing (which comb)?",
    "10. G: do all lines meet at one point?",
    "11. Total drawing time; speed files: time each",
    "12. A photo or scan, if easy",
]


def main(ofile_name1, bin_fn="bfile.bin", legend_fn=None, speed_prefix=None):
    pts, labels = build_pattern()
    print("Test pattern: %d points, path %.1f nominal mm"
          % (len(pts), np.hypot(*np.diff(pts, axis=0).T).sum()))
    to_segments(pts).savePreview(ofile_name1, scale=4)
    write_bin(pts, bin_fn)
    if legend_fn is not None:
        legend_write(pts, labels, legend_fn, QUESTIONS)
    if speed_prefix is not None:
        for name, step in (("sparse", 0), ("0p5mm", 0.5), ("0p05mm", 0.05)):
            sp = speed_squares(step)
            write_bin(sp, "%s_%s.bin" % (speed_prefix, name))
            print("Speed test %s: %d points" % (name, len(sp)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Drawbot calibration test pattern')
    parser.add_argument('output_image_fn', help='Output preview (.svg for vector, else raster)')
    parser.add_argument('output_bin_fn', nargs='?', default='bfile.bin', help='Output bin file')
    parser.add_argument('--legend', default=None,
                        help='Optional labeled SVG legend with the report-back checklist')
    parser.add_argument('--speed_prefix', default=None,
                        help='Also write PREFIX_sparse/_0p5mm/_0p05mm.bin timing files')
    args = parser.parse_args()
    main(args.output_image_fn, args.output_bin_fn, legend_fn=args.legend,
         speed_prefix=args.speed_prefix)
