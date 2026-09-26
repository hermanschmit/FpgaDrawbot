from unittest import TestCase

import numpy as np

import Segments
from Skeleton import Skeleton

class Skeleton_identity40(TestCase):
    def setUp(self):
        im = np.eye(40, 40, 0)
        for k in range(1, 10):
            im += np.eye(40, 40, k)
        im = 100. * im
        self.SkeletonInst = Skeleton(im)

class Skeleton1(Skeleton_identity40):
    def runTest(self):
        #self.assertEqual(len(self.SkeletonInst.segments.segmentList),2)
        pass


class Skeleton_identityInv40(TestCase):
    def setUp(self):
        im = np.eye(40, 40, 0)
        for k in range(1, 10):
            im += np.eye(40, 40, k)
        im = -100. * im
        im = 100. + im
        self.SkeletonInst = Skeleton(im)

class Skeleton2(Skeleton_identityInv40):
    def runTest(self):
        self.assertEqual(len(self.SkeletonInst.segments.segmentList),1)


class Skeleton_2parallel10x10(TestCase):
    def setUp(self):
        im = np.zeros((10, 10))
        for k in range(4, 6):
            im += np.eye(10, 10, k)
        imT = np.transpose(im)
        im = im + imT
        im = -100. * im
        im = 100. + im
        self.SkeletonInst = Skeleton(im)

class Skeleton3(Skeleton_2parallel10x10):
    def runTest(self):
        self.assertEqual(len(self.SkeletonInst.segments.segmentList),2)


class Skeleton_X10x10(TestCase):
    def setUp(self):
        im = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                       [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                       [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        im = -100. * im
        im = 100. + im
        self.SkeletonInst = Skeleton(im)

class Skeleton4(Skeleton_X10x10):
    def runTest(self):
        self.assertEqual(len(self.SkeletonInst.segments.segmentList),3)

class Skeleton_Box10x10(TestCase):
    def setUp(self):
        im = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                       [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                       [0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
                       [0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
                       [0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
                       [0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
                       [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                       [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        im = -100. * im
        im = 100. + im
        self.SkeletonInst = Skeleton(im)

class Skeleton5(Skeleton_Box10x10):
    def runTest(self):
        self.assertEqual(len(self.SkeletonInst.segments.segmentList),1)


''' trace_skeleton (junction-pairing) testing

These call trace_skeleton() directly on a hand-built skeleton pixel array
instead of going through the full Skeleton() pipeline (image -> Otsu
threshold -> skimage.skeletonize -> trace_skeleton). That pipeline can
distort a small hand-designed pattern before it ever reaches the tracing
logic -- a plain 5x5 single-pixel X crossing gets reshaped by skeletonize
into a different pattern entirely -- so building the skeleton array
directly is the only way to get exact control over the junction pixel
pattern under test.
'''

def _make_traced_skeleton(skel_array):
    s = Skeleton.__new__(Skeleton)
    s.skeleton = np.asarray(skel_array, dtype=np.uint8)
    s.segments = Segments.Segments()
    s.count_neighbors()
    s.trace_skeleton()
    return s


class TraceSkeletonCleanCrossing(TestCase):
    # A true single-pixel crossing: exactly one degree-4 pixel, the minimal
    # case the junction-pairing logic was built for. Should trace as two
    # long continuous corner-to-corner strokes and nothing else.
    def setUp(self):
        self.skel = np.array([[1, 0, 0, 0, 1],
                              [0, 1, 0, 1, 0],
                              [0, 0, 1, 0, 0],
                              [0, 1, 0, 1, 0],
                              [1, 0, 0, 0, 1]])

class TraceSkeletonCleanCrossingTest0(TraceSkeletonCleanCrossing):
    def runTest(self):
        s = _make_traced_skeleton(self.skel)
        self.assertEqual(len(s.segments.segmentList), 2)
        lengths = sorted(len(seg) for seg in s.segments.segmentList)
        self.assertEqual(lengths, [5, 5])
        endpoints = sorted(tuple(sorted((tuple(seg[0]), tuple(seg[-1]))))
                           for seg in s.segments.segmentList)
        expected = sorted([tuple(sorted(((0, 0), (4, 4)))),
                           tuple(sorted(((0, 4), (4, 0))))])
        self.assertEqual(endpoints, expected)


class TraceSkeletonThickCrossing(TestCase):
    '''
    A crossing that is two pixels wide at the center (a 2x2 block of four
    degree-4 pixels) instead of a single shared pixel -- the actual shape
    skimage.skeletonize produces from the Skeleton4/X10x10 fixture above.
    Discovered while reviewing test coverage: the junction-pairing logic
    still produces the two intended long diagonal strokes, but also leaves
    behind a small residual closed loop cycling through the four center
    pixels as a third fragment. This is a known, minor limitation -- not a
    crash, not lost data, just one extra short spurious segment -- and this
    test exists to pin down and document that current behavior rather than
    let it be silently re-discovered (or silently changed) later.
    '''
    def setUp(self):
        self.skel = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                              [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                              [1, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

class TraceSkeletonThickCrossingTest0(TraceSkeletonThickCrossing):
    def runTest(self):
        s = _make_traced_skeleton(self.skel)
        self.assertEqual(len(s.segments.segmentList), 3)
        lengths = sorted(len(seg) for seg in s.segments.segmentList)
        self.assertEqual(lengths, [5, 10, 10])
        long_segments = [seg for seg in s.segments.segmentList if len(seg) == 10]
        endpoints = sorted(tuple(sorted((tuple(seg[0]), tuple(seg[-1]))))
                           for seg in long_segments)
        expected = sorted([tuple(sorted(((0, 0), (9, 9)))),
                           tuple(sorted(((9, 0), (0, 9))))])
        self.assertEqual(endpoints, expected)
        # the residual artifact is a closed loop: it starts and ends at the
        # same pixel.
        loop_segment = [seg for seg in s.segments.segmentList if len(seg) == 5][0]
        self.assertEqual(tuple(loop_segment[0]), tuple(loop_segment[-1]))
