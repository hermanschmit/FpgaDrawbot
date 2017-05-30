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
        self.seg.svgwrite("test.svg")
        self.seg.segment2grad(scale=2)
        self.seg.segment2grad(interior=True,scale=1)

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
        self.seg.scaleBin()
        self.assertTrue(self.seg.xmax==1.0 or self.seg.ymax==1.0)
        self.assertTrue(self.seg.ymin==-1.0 or self.seg.xmin==-1.0)

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
        self.seg.scaleBin()
        self.assertTrue(self.seg.xmax==1.0 or self.seg.ymax==1.0)
        self.assertTrue(self.seg.ymin==-1.0 or self.seg.xmin==-1.0)

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
        self.seg.scaleBin()
        self.assertTrue(self.seg.xmax==1.0 or self.seg.ymax==1.0)
        self.assertTrue(self.seg.ymin==-1.0 or self.seg.xmin==-1.0)

class SegmentsInst2Test1(SegmentsInst2):
    def runTest(self):
        self.seg.offset((3.0,-0.5))
        seg2 = Segments()
        for s in self.seg.segmentList:
            seg2.append(s)
        self.assertEqual(self.seg.xmax,seg2.xmax)
        self.assertEqual(self.seg.ymax,seg2.ymax)
        self.assertEqual(self.seg.xmin,seg2.xmin)
        self.assertEqual(self.seg.ymin,seg2.ymin)
        l = self.seg.binList()
        self.assertTrue(max(l)==self.seg.xmax or max(l)==self.seg.ymax)
        self.assertTrue(min(l)==self.seg.xmin or min(l)==self.seg.ymin)


class SegmentsInst2Test2(SegmentsInst2):
    def runTest(self):
        self.seg.scale(0.15)
        seg2 = Segments()
        for s in self.seg.segmentList:
            seg2.append(s)
        self.assertEqual(self.seg.xmax,seg2.xmax)
        self.assertEqual(self.seg.ymax,seg2.ymax)
        self.assertEqual(self.seg.xmin,seg2.xmin)
        self.assertEqual(self.seg.ymin,seg2.ymin)

class SegmentsInst2Test3(SegmentsInst2):
    def runTest(self):
        self.seg.scale(3.2)
        seg2 = Segments()
        for s in self.seg.segmentList:
            seg2.append(s)
        self.assertEqual(self.seg.xmax,seg2.xmax)
        self.assertEqual(self.seg.ymax,seg2.ymax)
        self.assertEqual(self.seg.xmin,seg2.xmin)
        self.assertEqual(self.seg.ymin,seg2.ymin)

class SegmentsInst2Test4(SegmentsInst2):
    def runTest(self):
        self.seg.scaleBin()
        seg2 = Segments()
        for s in self.seg.segmentList:
            seg2.append(s)
        self.assertEqual(self.seg.xmax,seg2.xmax)
        self.assertEqual(self.seg.ymax,seg2.ymax)
        self.assertEqual(self.seg.xmin,seg2.xmin)
        self.assertEqual(self.seg.ymin,seg2.ymin)

class SegmentsInst2Test5(SegmentsInst2):
    def runTest(self):
        self.seg.simplify()
        seg2 = Segments()
        for s in self.seg.segmentList:
            seg2.append(s)
        self.assertEqual(self.seg.xmax,seg2.xmax)
        self.assertEqual(self.seg.ymax,seg2.ymax)
        self.assertEqual(self.seg.xmin,seg2.xmin)
        self.assertEqual(self.seg.ymin,seg2.ymin)

class SegmentsInst2Test6(SegmentsInst2):
    def runTest(self):
        self.seg.simplify()
        self.seg.offset((0.2,-0.4))
        seg2 = Segments()
        for s in self.seg.segmentList:
            seg2.append(s)
        self.assertEqual(self.seg.xmax,seg2.xmax)
        self.assertEqual(self.seg.ymax,seg2.ymax)
        self.assertEqual(self.seg.xmin,seg2.xmin)
        self.assertEqual(self.seg.ymin,seg2.ymin)

class SegmentsInst2Test7(SegmentsInst2):
    def runTest(self):
        self.seg.simplify()
        self.seg.scale(0.2)
        seg2 = Segments()
        for s in self.seg.segmentList:
            seg2.append(s)
        self.assertEqual(self.seg.xmax,seg2.xmax)
        self.assertEqual(self.seg.ymax,seg2.ymax)
        self.assertEqual(self.seg.xmin,seg2.xmin)
        self.assertEqual(self.seg.ymin,seg2.ymin)

class SegmentsInst2Test8(SegmentsInst2):
    def runTest(self):
        self.seg.simplify()
        self.seg.scaleBin()
        seg2 = Segments()
        for s in self.seg.segmentList:
            seg2.append(s)
        self.assertEqual(self.seg.xmax,seg2.xmax)
        self.assertEqual(self.seg.ymax,seg2.ymax)
        self.assertEqual(self.seg.xmin,seg2.xmin)
        self.assertEqual(self.seg.ymin,seg2.ymin)

class SegmentsInst2Test9(SegmentsInst2):
    def runTest(self):
        self.seg.concatSegments()
        self.seg.scaleBin()
        seg2 = Segments()
        for s in self.seg.segmentList:
            seg2.append(s)
        self.assertEqual(self.seg.xmax,seg2.xmax)
        self.assertEqual(self.seg.ymax,seg2.ymax)
        self.assertEqual(self.seg.xmin,seg2.xmin)
        self.assertEqual(self.seg.ymin,seg2.ymin)
        l = self.seg.binList()
        self.assertLessEqual(max(l),1.0)
        self.assertGreaterEqual(min(l),-1.0)

