from unittest import TestCase

import numpy as np

import Guilloche


def _path_length(pts):
    return np.hypot(*np.diff(pts, axis=0).T).sum()


class TestGuillocheSingleStrokeInBounds(TestCase):
    def runTest(self):
        im = np.tile(np.linspace(0, 255, 120), (100, 1))
        for kw in ({}, {'rosette_amp': 0.8, 'relief': 0.5}, {'shape': 'circle'}):
            g = Guilloche.Guilloche(im, canvas_size=120, turns=15, **kw)
            self.assertEqual(len(g.segments.segmentList), 1)
            pts = g.segments.segmentList[0]
            self.assertTrue(np.all(np.isfinite(pts)))
            # Must stay on canvas (segment2grad indexes with these directly).
            self.assertGreaterEqual(pts[:, 0].min(), 0)
            self.assertGreaterEqual(pts[:, 1].min(), 0)
            self.assertLessEqual(pts[:, 0].max(), 99)
            self.assertLessEqual(pts[:, 1].max(), 119)


class TestGuillocheDarkUsesMoreInk(TestCase):
    def runTest(self):
        # Left half black, right half white: the wave should put far more
        # path length on the left.
        im = np.full((100, 100), 255.0)
        im[:, :50] = 0.0
        g = Guilloche.Guilloche(im, canvas_size=100, turns=15)
        pts = g.segments.segmentList[0]
        mid = 0.5 * (pts[1:] + pts[:-1])
        steps = np.hypot(*np.diff(pts, axis=0).T)
        left = steps[mid[:, 1] < 40].sum()
        right = steps[mid[:, 1] > 60].sum()
        self.assertGreater(left, 1.5 * right)


class TestGuillocheWhiteIsPlainSpiral(TestCase):
    def runTest(self):
        # A blank image yields a bare spiral with only coarse sampling.
        g = Guilloche.Guilloche(np.full((80, 80), 255.0), canvas_size=80, turns=10)
        pts = g.segments.segmentList[0]
        r = np.hypot(pts[:, 0] - 39.5, pts[:, 1] - 39.5)
        self.assertTrue(np.all(np.diff(r[5:]) > -1e-6))  # radius never shrinks
        self.assertLess(len(pts), 2 * _path_length(pts))
