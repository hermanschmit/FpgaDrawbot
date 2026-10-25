"""
Space-filling curve with a tone wave on every segment -- a square cousin of
Guilloche.py.

The curve is a generalized Hilbert curve ("gilbert", after Jakub Cervený's
gilbert2d) on a grid of cells matching the image's aspect ratio, so it fills
any rectangle, not just 2^n squares. Every grid step is one segment of the
same length, and each segment gets the same kind of perturbation along its
normal, set by the darkness under the segment's midpoint:

- k whole wave periods per segment, k = 1 + round(freq_mod * darkness), so
  the wave is zero at both ends and the curve's corners stay crisp;
- amplitude grows with darkness, bounded by a diamond envelope.

The envelope keeps neighboring segments' waves apart. Each grid step owns
the square rotated 45 degrees that has the step as its diagonal; those
diamonds tile the plane, so curves kept inside their own diamond can't cross
or overlap other segments' curves (consecutive steps share only the diamond
tip at the common cell center). The diamond is inset by half the pen width
so the ink of neighbors doesn't merge either. In segment coordinates (t
along the step in [0, 1], offset in cells) that is the tent

    |offset| <= E(t) = max(0, 0.5 - pen_inset - |t - 0.5|)

with pen_inset = sqrt(2) * (pen_mm / 2) / cell_mm. Because the diamond is
convex and every sampled vertex lies inside it, the straight segments
between samples do too. 'scale' multiplies the wave by E(t) (waves taper
toward the ends); 'clip' clips a full-size wave to +-E(t) (flat-topped,
more ink).

Segments under white paper are drawn straight and collinear runs collapse to
their end points, so light areas cost almost nothing.

Wave shapes: 'sine' (ppw points per period), 'triangle' (sine sampled at
quarter periods: 4 points per period) and 'square' (4 points per period).
Point count drives drawing time on this bot (it accelerates from rest on
every segment), so 'triangle'/'square' are much faster to draw than a finely
sampled sine.

Coordinates follow the repo convention: points are (row, col).
"""
import numpy as np
from scipy import ndimage

import Segments
from Guilloche import tone_map


def _sgn(v):
    return (v > 0) - (v < 0)


def gilbert2d(width, height):
    """Cells (x, y) of a generalized Hilbert curve covering width x height,
    starting at (0, 0). Each step moves to an adjacent cell, except for a
    single diagonal step when the long side is odd and the short side even."""
    out = []

    def gen(x, y, ax, ay, bx, by):
        w, h = abs(ax + ay), abs(bx + by)
        dax, day, dbx, dby = _sgn(ax), _sgn(ay), _sgn(bx), _sgn(by)
        if h == 1:
            for _ in range(w):
                out.append((x, y))
                x, y = x + dax, y + day
            return
        if w == 1:
            for _ in range(h):
                out.append((x, y))
                x, y = x + dbx, y + dby
            return
        ax2, ay2 = int(ax / 2), int(ay / 2)
        bx2, by2 = int(bx / 2), int(by / 2)
        w2, h2 = abs(ax2 + ay2), abs(bx2 + by2)
        if 2 * w > 3 * h:
            if w2 % 2 and w > 2:
                ax2, ay2 = ax2 + dax, ay2 + day
            gen(x, y, ax2, ay2, bx, by)
            gen(x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by)
        else:
            if h2 % 2 and h > 2:
                bx2, by2 = bx2 + dbx, by2 + dby
            gen(x, y, bx2, by2, ax2, ay2)
            gen(x + bx2, y + by2, ax, ay, bx - bx2, by - by2)
            gen(x + (ax - dax) + (bx2 - dbx), y + (ay - day) + (by2 - dby),
                -bx2, -by2, -(ax - ax2), -(ay - ay2))

    if width >= height:
        gen(0, 0, width, 0, 0, height)
    else:
        gen(0, 0, 0, height, width, 0)
    return np.array(out, dtype=float)


def _template(shape, k, ppw):
    """(t, s) samples of k periods over t in (0, 1], s in [-1, 1], ending at s=0."""
    if shape == 'square':
        t, s = [], []
        for p in range(k):
            t += [p / k, (p + 0.5) / k, (p + 0.5) / k, (p + 1) / k]
            s += [1.0, 1.0, -1.0, -1.0]
        t.append(1.0)
        s.append(0.0)
        return np.column_stack((t, s))
    n = k * (4 if shape == 'triangle' else ppw)
    t = np.arange(1, n + 1) / n
    s = np.sin(2 * np.pi * k * t)
    s[np.abs(s) < 1e-12] = 0.0
    return np.column_stack((t, s))


class HilbertWave:
    CELL_PX = 8  # output units per grid cell (sets the raster preview size)

    def __init__(self, im, cells=96, amplitude=1.0, freq_mod=2.0, wave='sine', ppw=8,
                 envelope='scale', pen_mm=0.5, size_mm=342.0,
                 gamma=0.7, white_cut=0.1, min_amp=0.03):
        """
        im: grayscale float image, 0 = black, 255 = white.
        cells: grid cells along the image's long side.
        amplitude: wave amplitude at full black as a fraction of the
            envelope's peak. 'scale' caps it at 1; with 'clip', values above
            1 push more of the wave flat against the envelope.
        freq_mod: periods per segment run from 1 (white) to 1 + freq_mod
            (black), rounded to whole periods.
        wave: 'sine', 'triangle' or 'square'.
        ppw: points per period for 'sine'.
        envelope: 'scale' or 'clip' (see module docstring).
        pen_mm, size_mm: pen width and drawn size of the long side, used to
            inset the envelope by half a pen width.
        gamma, white_cut: tone mapping, as in Guilloche.tone_map().
        min_amp: segments whose amplitude (fraction of full) is below this
            are drawn straight.
        """
        if wave not in ('sine', 'triangle', 'square'):
            raise ValueError("wave must be 'sine', 'triangle' or 'square'")
        if envelope not in ('scale', 'clip'):
            raise ValueError("envelope must be 'scale' or 'clip'")
        im = np.asarray(im, dtype=float)
        h0, w0 = im.shape
        s = cells / max(h0, w0)
        # Even dimensions: gilbert2d needs a diagonal step only when the long
        # side is odd and the short side even.
        nx, ny = (max(2, 2 * int(round(v * s / 2))) for v in (w0, h0))

        # Darkness sampled 4x per cell and smoothed over about half a cell.
        sub = 4
        dark = ndimage.gaussian_filter(tone_map(im, (ny * sub, nx * sub), white_cut, gamma), 2.0)

        cells_xy = gilbert2d(nx, ny)
        p0, p1 = cells_xy[:-1], cells_xy[1:]
        mid = 0.5 * (p0 + p1)
        d = ndimage.map_coordinates(dark, [mid[:, 1] * sub + 1.5, mid[:, 0] * sub + 1.5],
                                    order=1, mode='nearest')
        k = 1 + np.rint(freq_mod * d).astype(int)
        frac = (min(amplitude, 1.0) if envelope == 'scale' else amplitude) * d
        frac[d < min_amp] = 0.0

        cell_mm = size_mm / max(nx, ny)
        self.env_peak = 0.5 - np.sqrt(2) * 0.5 * pen_mm / cell_mm
        if self.env_peak <= 0:
            raise ValueError("cells of %.2f mm are too small for a %.2f mm pen" % (cell_mm, pen_mm))

        def env(t):
            return np.maximum(0.0, self.env_peak - np.abs(t - 0.5))

        templates = {}
        self.pieces = []  # (start, end, wave points) per grid step, for checking
        parts = [p0[:1]]
        for a, b, kk, f in zip(p0, p1, k, frac):
            if f == 0.0:
                parts.append(b[None, :])
                continue
            tmpl = templates.get(kk)
            if tmpl is None:
                tmpl = templates[kk] = _template(wave, kk, ppw)
            t = tmpl[:, 0]
            if envelope == 'scale':
                off = f * tmpl[:, 1] * env(t)
            else:
                e = env(t)
                off = np.clip(f * self.env_peak * tmpl[:, 1], -e, e)
            v = b - a
            n = np.array([-v[1], v[0]]) / np.hypot(*v)
            piece = a + np.outer(t, v) + np.outer(off, n)
            self.pieces.append((a, b, piece))
            parts.append(piece)
        pts = np.vstack(parts)
        pts = pts[np.concatenate(([True], np.any(np.diff(pts, axis=0) != 0, axis=1)))]

        # Collapse straight runs (white stretches) to their end points.
        e = np.diff(pts, axis=0)
        cross = e[:-1, 0] * e[1:, 1] - e[:-1, 1] * e[1:, 0]
        dot = np.einsum('ij,ij->i', e[:-1], e[1:])
        keep = np.ones(len(pts), dtype=bool)
        keep[1:-1] = ~((np.abs(cross) < 1e-9) & (dot > 0))
        pts = pts[keep]

        # Cell centers sit at (c + 0.5); the envelope keeps waves within half
        # a cell of the curve, so everything stays on canvas.
        pts = (pts + 0.5) * self.CELL_PX
        self.segments = Segments.Segments()
        self.segments.append(np.ascontiguousarray(pts[:, ::-1]))  # (x, y) -> (row, col)
        print("HilbertWave: %d x %d cells, %d points" % (nx, ny, len(pts)))

    def binWrite(self, fn):
        self.segments.binWrite(fn)
