from unittest import TestCase

import numpy as np

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
