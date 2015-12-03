from unittest import TestCase

from Segments import Segments
import numpy

__author__ = 'herman'


class SegmentsInst0(TestCase):
    def setUp(self):
        segList1 = [(0, 1), (1, 1), (2, 1)]
        segList2 = [(2, 5), (2, 4), (2, 3)]
        self.seg = Segments()
        self.seg.append(segList1)
        self.seg.append(segList2)


class SegmentsInst0Test0(SegmentsInst0):
    def runTest(self):
        self.assertEqual(self.seg.xmax, 2)
        self.assertEqual(self.seg.xmin, 0)
        self.assertEqual(self.seg.ymax, 5)
        self.assertEqual(len(self.seg.segmentList), 2)
        self.seg.addInitialStartPt()
        self.assertEqual(len(self.seg.segmentList), 3)
        self.seg.simplify()
        self.assertEqual(len(self.seg.segmentList), 3)
        self.seg.concatSegments()
        self.assertEqual(len(self.seg.segmentList),1)
        self.assertEqual(len(self.seg.segmentList[0]), 5)
        self.assertEqual(len(self.seg.segmentList[0]), self.seg.numpts)
        self.seg.simplify()
        self.assertEqual(len(self.seg.segmentList[0]), self.seg.numpts)

        self.assertEqual(numpy.amax(self.seg.segmentList), 5.0)
        self.assertEqual(numpy.amin(self.seg.segmentList), 0.0)
        self.seg.scale(0.4)
        self.assertEqual(numpy.amax(self.seg.segmentList), 2.0)
        self.assertEqual(self.seg.xmax, 0.8)
        self.assertEqual(self.seg.ymax, 2.0)
        self.seg.offset((-0.5 * self.seg.xmax, -0.5 * self.seg.ymax))
        self.assertEqual(self.seg.ymax, 1.0)
        self.assertEqual(self.seg.ymin, -0.6)

class SegmentsInst0Test1(SegmentsInst0):
    def runTest(self):
        self.seg.segment2grad()
        i = numpy.nonzero(self.seg.grad)
        self.assertEqual(len(i[0]),7)
        self.seg.segment2grad(interior=True) # there is no interior, given the original segments, results the same
        i = numpy.nonzero(self.seg.grad)
        self.assertEqual(len(i[0]),7)

class SegmentsInst0Test2(SegmentsInst0):
    def runTest(self):
        self.seg.simplify()
        self.seg.segment2grad()
        i = numpy.nonzero(self.seg.grad)
        self.assertEqual(len(i[0]),6) # skip because redundant pts removed
        self.seg.segment2grad(interior=True)
        i = numpy.nonzero(self.seg.grad)
        self.assertEqual(len(i[0]),7) # simplify should have no impact if interior is drawn

class SegmentsInst1(TestCase):
    def setUp(self):
        segList1 = [(0, 1), (2, 1)]
        segList2 = [(0, 3), (2, 3)]
        self.seg = Segments()
        self.seg.append(segList1)
        self.seg.append(segList2)

class SegmentsInst1Test1(SegmentsInst1):
    def runTest(self):
        self.seg.segment2grad()
        i = numpy.nonzero(self.seg.grad)
        self.assertEqual(len(i[0]),5)
        self.seg.segment2grad(interior=True)
        i = numpy.nonzero(self.seg.grad)
        self.assertEqual(len(i[0]),7)

class SegmentsInst2(TestCase):
    def setUp(self):
        segList1 = [(0, 1), (1, 1), (2, 1)]
        segList2 = [(2, 3), (2, 4), (2, 5)] # same as Inst0, except sequence is reversed
        self.seg = Segments()
        self.seg.append(segList1)
        self.seg.append(segList2)

class SegmentsInst2Test0(SegmentsInst2):
    def runTest(self):
        self.seg.simplify()
        self.seg.segment2grad()
        i = numpy.nonzero(self.seg.grad)
        self.assertEqual(len(i[0]),5) # skip because redundant pts removed
        self.seg.segment2grad(interior=True)
        i = numpy.nonzero(self.seg.grad)
        self.assertEqual(len(i[0]),7) # simplify should have no impact if interior is drawn
