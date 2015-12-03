from unittest import TestCase

import numpy as np

from Canny import Canny

__author__ = 'herman'


class Canny_identity10_1(TestCase):
    '''
    This case should have no edges due to the blur
    '''
    def setUp(self):
        im = np.identity(10)
        im = 1. * im
        self.CannyInst = Canny(im)


class TestCannyI10_1(Canny_identity10_1):
    def runTest(self):
        self.assertEqual(self.CannyInst.grad.shape, (8, 8))
        self.assertAlmostEqual(np.amax(self.CannyInst.grad), 0.0)  # blur removes any edge from identity matrix
        self.assertAlmostEqual(np.amin(self.CannyInst.grad), 0.0)  # blur
        self.assertEqual(self.CannyInst.segments.numpts, 0)

class Canny_identity10_100(TestCase):
    def setUp(self):
        im = np.eye(10, 10, 0)
        for k in xrange(1, 10):
            im += np.eye(10, 10, k)
        im = 100. * im
        self.CannyInst = Canny(im, sigma=0.0001)


class TestCannyI10_100(Canny_identity10_100):
    def runTest(self):
        self.assertEqual(self.CannyInst.grad.shape, (8, 8))
        self.assertAlmostEqual(np.amax(self.CannyInst.grad), 0.0)
        self.assertAlmostEqual(np.amin(self.CannyInst.grad), -1.0)
        self.assertEqual(self.CannyInst.segments.numpts, 18) # 18 points
        self.assertGreater(len(self.CannyInst.segments.segmentList),1)

class Canny_identity10_100_2(TestCase):
    def setUp(self):
        im = np.eye(10, 10, 0)
        for k in xrange(1, 10):
            im += np.eye(10, 10, -1 * k)
        im = 100. * im
        self.CannyInst = Canny(im, sigma=0.0001)


class TestCannyI10_100_2(Canny_identity10_100_2):
    def runTest(self):
        self.assertEqual(self.CannyInst.grad.shape, (8, 8))
        self.assertAlmostEqual(np.amax(self.CannyInst.grad), 0.0)
        self.assertAlmostEqual(np.amin(self.CannyInst.grad), -1.0)

        self.assertEqual(self.CannyInst.segments.numpts, 18)
        self.assertEqual(len(self.CannyInst.segments.segmentList), 1) # should have one segment

class Canny_eye20x20_mst(TestCase):
    def setUp(self):
        im = np.eye(20, 20, 0)
        for k in xrange(1, 20):
            im += np.eye(20, 20, -1 * k)

        for k in xrange(18, 20):
            im += np.eye(20, 20, k)
        im = 100. * im
        self.CannyInst = Canny(im, sigma=0.0001)


class TestCanny_eye_mst(Canny_eye20x20_mst):
    def runTest(self):
        self.assertEqual(self.CannyInst.grad.shape, (18, 18))
        self.assertEqual(self.CannyInst.segments.numpts, 41) # must have at least two segments
        self.assertEqual(len(self.CannyInst.segments.segmentList),2)
