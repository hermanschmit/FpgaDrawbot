from unittest import TestCase

import numpy as np

import HilbertWave


class TestGilbertCoversGrid(TestCase):
    def runTest(self):
        for w in range(1, 33):
            for h in range(1, 33):
                c = HilbertWave.gilbert2d(w, h)
                self.assertEqual(len(c), w * h)
                self.assertEqual(len({tuple(p) for p in c}), w * h)
                self.assertTrue(np.all(c.min(axis=0) >= 0))
                self.assertTrue(np.all(c.max(axis=0) == [w - 1, h - 1]), (w, h))
                np.testing.assert_array_equal(c[0], [0, 0])
                if not (max(w, h) % 2 == 1 and min(w, h) % 2 == 0):
                    steps = np.abs(np.diff(c, axis=0)).sum(axis=1)
                    self.assertTrue(np.all(steps == 1), (w, h))


class TestHilbertWave(TestCase):
    def runTest(self):
        # Portrait image, left half black: output must stay portrait, on
        # canvas, and put far more points on the dark side.
        im = np.full((90, 60), 255.0)
        im[:, :30] = 0.0
        for wave in ('sine', 'triangle', 'square'):
            hw = HilbertWave.HilbertWave(im, cells=20, wave=wave)
            self.assertEqual(len(hw.segments.segmentList), 1)
            pts = hw.segments.segmentList[0]  # (row, col)
            self.assertTrue(np.all(pts >= 0))
            ext = pts.max(axis=0) - pts.min(axis=0)
            self.assertGreater(ext[0], ext[1])
            mid_col = 0.5 * (pts[:, 1].min() + pts[:, 1].max())
            self.assertGreater(np.sum(pts[:, 1] < mid_col), 3 * np.sum(pts[:, 1] > mid_col))

        # Blank image: just the bare curve, straight runs collapsed.
        hw = HilbertWave.HilbertWave(np.full((40, 40), 255.0), cells=8)
        pts = hw.segments.segmentList[0]
        self.assertLess(len(pts), 64)
        steps = np.diff(pts, axis=0)
        self.assertTrue(np.all((steps[:, 0] == 0) | (steps[:, 1] == 0)))


def _seg_point_dist(p, a, b):
    """Distance from points p (N,2) to segments a->b (M,2): (N, M)."""
    ab = b - a
    L2 = np.maximum(np.einsum('ij,ij->i', ab, ab), 1e-18)
    ap = p[:, None, :] - a[None, :, :]
    t = np.clip(np.einsum('nmj,mj->nm', ap, ab) / L2, 0, 1)
    return np.linalg.norm(ap - t[..., None] * ab[None], axis=2)


class TestHilbertWaveEnvelope(TestCase):
    def runTest(self):
        # Solid black so every step carries a full wave. size_mm=20 over 10
        # cells -> 2 mm cells, so a 0.5 mm pen is a quarter cell.
        im = np.zeros((50, 50))
        pen_cells = 0.25
        for wave in ('sine', 'triangle', 'square'):
            for envelope, amp in (('scale', 1.0), ('clip', 3.0)):
                hw = HilbertWave.HilbertWave(im, cells=10, wave=wave, envelope=envelope,
                                             amplitude=amp, pen_mm=0.5, size_mm=20.0)
                self.assertAlmostEqual(hw.env_peak, 0.5 - np.sqrt(2) * pen_cells / 2)
                polys = []
                for a, b, piece in hw.pieces:
                    # Every vertex inside the inset diamond of its own step.
                    v = b - a
                    n = np.array([-v[1], v[0]])
                    t = (piece - a) @ v
                    off = (piece - a) @ n
                    self.assertTrue(np.all(np.abs(off) + np.abs(t - 0.5)
                                           <= hw.env_peak + 1e-9), (wave, envelope))
                    polys.append((tuple(a), tuple(b), np.vstack((a, piece))))
                # Brute force: steps that don't share an end point stay at
                # least a pen width apart (so they neither cross nor touch).
                worst = np.inf
                for i in range(len(polys)):
                    ai, bi, pi = polys[i]
                    for j in range(i + 1, len(polys)):
                        aj, bj, pj = polys[j]
                        if {ai, bi} & {aj, bj}:
                            continue
                        if np.abs(pi.mean(0) - pj.mean(0)).max() > 2.5:
                            continue
                        d = min(_seg_point_dist(pi, pj[:-1], pj[1:]).min(),
                                _seg_point_dist(pj, pi[:-1], pi[1:]).min())
                        worst = min(worst, d)
                self.assertGreaterEqual(worst, pen_cells - 1e-9, (wave, envelope, worst))
