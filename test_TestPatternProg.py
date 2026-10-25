import os
import tempfile
from unittest import TestCase

import numpy as np

import BinTool
import TestPatternProg as T


class TestPatternGeometry(TestCase):
    def runTest(self):
        pts, labels = T.build_pattern()
        np.testing.assert_array_equal(pts[0], [T.SIZE / 2, T.SIZE / 2])
        # The frame defines the bounds, so nominal mm maps exactly to [-1, 1].
        np.testing.assert_array_equal(pts.min(axis=0), [0, 0])
        np.testing.assert_array_equal(pts.max(axis=0), [T.SIZE, T.SIZE])
        self.assertFalse(np.any(np.all(np.diff(pts, axis=0) == 0, axis=1)))
        self.assertTrue(labels)


class TestPatternSpeedSquares(TestCase):
    def runTest(self):
        sparse = T.speed_squares(0)
        for step in (0.5, 0.05):
            dense = T.speed_squares(step)
            self.assertLessEqual(np.hypot(*np.diff(dense, axis=0).T).max(), step + 1e-9)
            # Same geometry: every sparse corner is on the dense path, and
            # simplifying the dense path recovers the sparse one.
            np.testing.assert_allclose(BinTool.rdp(dense, 1e-6), sparse, atol=1e-9)
        np.testing.assert_array_equal(sparse[0], [T.SIZE / 2, T.SIZE / 2])


class TestPatternBinStartsAtCentre(TestCase):
    def runTest(self):
        fd, fn = tempfile.mkstemp(suffix='.bin')
        os.close(fd)
        try:
            T.write_bin(T.build_pattern()[0], fn)
            pts = BinTool.bin_read(fn)
            np.testing.assert_allclose(pts[0], [0, 0], atol=1e-6)
            np.testing.assert_allclose(pts.min(axis=0), [-1, -1], atol=1e-6)
            np.testing.assert_allclose(pts.max(axis=0), [1, 1], atol=1e-6)
            # bin is (x, y): the first move goes left to the frame.
            np.testing.assert_allclose(pts[1], [-1, 0], atol=1e-6)
        finally:
            os.remove(fn)
