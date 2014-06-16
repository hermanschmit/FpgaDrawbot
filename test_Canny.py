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
