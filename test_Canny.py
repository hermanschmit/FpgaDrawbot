from unittest import TestCase
from Canny import Canny
import numpy as np

__author__ = 'herman'

class Canny_identity10_1(TestCase):
    def setUp(self):
        im = np.identity(10)
        im = 1. * im
        self.CannyInst = Canny(im)

class TestCannyI10_1(Canny_identity10_1):
    def runTest(self):
        self.assertEqual(self.CannyInst.grad.shape, (10,10))
        self.assertEqual(len(self.CannyInst.segmentList),0)

class Canny_identity10_100(TestCase):
    def setUp(self):
        im = np.eye(10,10,0)
        for k in xrange(1,10):
            im += np.eye(10,10,k)
        im = 100. * im
        self.CannyInst = Canny(im,sigma=0.0001)

class TestCannyI10_100(Canny_identity10_100):
    def runTest(self):
        self.assertEqual(self.CannyInst.grad.shape, (10,10))
        self.assertEqual(len(self.CannyInst.segmentList),1)

class Canny_identity10_100_2(TestCase):
    def setUp(self):
        im = np.eye(10,10,0)
        for k in xrange(1,10):
            im += np.eye(10,10,-1*k)
        im = 100. * im
        self.CannyInst = Canny(im,sigma=0.0001)

class TestCannyI10_100_2(Canny_identity10_100_2):
    def runTest(self):
        self.assertEqual(self.CannyInst.grad.shape, (10,10))
        self.assertEqual(len(self.CannyInst.segmentList),2)

class Canny_eye20x20_mst(TestCase):
    def setUp(self):
        im = np.eye(20,20,0)
        for k in xrange(1,20):
            im += np.eye(20,20,-1*k)

        for k in xrange(18,20):
            im += np.eye(20,20,k)
        im = 100. * im
        print im
        self.CannyInst = Canny(im,sigma=0.0001)
        print self.CannyInst.grad

class TestCanny_eye_mst(Canny_eye20x20_mst):
    def runTest(self):
        self.assertEqual(self.CannyInst.grad.shape, (20,20))
        self.assertEqual(len(self.CannyInst.segmentList),3)
        self.CannyInst.euclidMstPrune(False,2)
        self.assertEqual(len(self.CannyInst.segmentList),2)
