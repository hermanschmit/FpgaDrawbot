import os
import tempfile
from unittest import TestCase

import numpy as np

import BinTool
import Segments


class TestBinToolRoundTrip(TestCase):
    def runTest(self):
        s = Segments.Segments()
        s.append(np.array([[0.0, 0.0], [3.0, 1.0], [5.0, 4.0]]))
        s.scaleBin()
        fd, fn = tempfile.mkstemp(suffix='.bin')
        os.close(fd)
        try:
            s.binWrite(fn)
            pts = BinTool.bin_read(fn)
            # bin is (col, row), i.e. swapped relative to segmentList
            np.testing.assert_allclose(pts, np.array(s.segmentList[0])[:, ::-1], atol=1e-6)
            BinTool.bin_write(fn, pts)
            self.assertEqual(os.path.getsize(fn), os.path.getsize(fn))
            np.testing.assert_allclose(BinTool.bin_read(fn), pts, atol=1e-6)
            self.assertEqual(os.path.getsize(fn), (2 * len(pts) + BinTool.NAN_PAD) * 4)
        finally:
            os.remove(fn)


class TestBinToolRdp(TestCase):
    def runTest(self):
        # Collinear points collapse to the endpoints.
        line = np.column_stack((np.linspace(0, 10, 50), np.zeros(50)))
        np.testing.assert_array_equal(BinTool.rdp(line, 0.01), line[[0, -1]])

        # A back-and-forth retrace (A -> B -> A' between them) keeps B.
        retrace = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 0.0]])
        np.testing.assert_array_equal(BinTool.rdp(retrace, 0.01), retrace)

        # Every dropped point is within tol of the kept polyline.
        rng = np.random.default_rng(0)
        walk = np.cumsum(rng.normal(size=(500, 2)), axis=0)
        tol = 0.5
        kept = BinTool.rdp(walk, tol)
        self.assertLess(len(kept), len(walk))
        for p in walk:
            a, b = kept[:-1], kept[1:]
            ab = b - a
            t = np.clip(np.einsum('ij,ij->i', p - a, ab) / np.maximum(np.einsum('ij,ij->i', ab, ab), 1e-12), 0, 1)
            d = np.hypot(*(p - a - t[:, None] * ab).T).min()
            self.assertLessEqual(d, tol + 1e-9)


class TestBinToolSnapDedupe(TestCase):
    def runTest(self):
        pts = np.array([[0.0, 0.0], [0.0, 0.0], [0.04, 0.01], [1.0, 1.0]])
        np.testing.assert_array_equal(BinTool.dedupe(pts), pts[[0, 2, 3]])
        np.testing.assert_allclose(BinTool.snap(pts, 0.1), [[0, 0], [1, 1]])
