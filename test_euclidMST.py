from unittest import TestCase
from EuclidMST import EuclidMST
import numpy as np

__author__ = 'herman'

class EuclidMST_instZ(TestCase):
    def setUp(self):
        segList = np.array([[[0, 0], [3, 0]], [[2, 1], [5, 1]]])
        self.euclidmst = EuclidMST(segList)


class TestEuclidMSTZ(EuclidMST_instZ):
    def runTest(self):
        self.assertEqual(self.euclidmst.size,4)
        d = self.euclidmst.spnTree.sum()
        d /= 2 # spnTree is undirected, so sum is 2x expected
        self.assertAlmostEqual(d,2)

class EuclidMST_instY(TestCase):
    def setUp(self):
        segList = np.array([[[0, 3], [1, 3]], [[3, 3], [4, 3]], [[2, 0], [2, 2]]])
        self.euclidmst = EuclidMST(segList)


class TestEuclidMSTY(EuclidMST_instY):
    def runTest(self):
        self.assertEqual(self.euclidmst.size,6)
        d = self.euclidmst.spnTree.sum()
        d /= 2 # spnTree is undirected, so sum is 2x expected
        self.assertAlmostEqual(d,4)

class TestEuclidMST_dfo(EuclidMST_instY):
    def runTest(self):
        tree = self.euclidmst.dfo_nonrec(0)
        self.euclidmst.treetrav_nonrec(tree)
        self.assertEqual(len(self.euclidmst.nodeTrav),11)


class EuclidMST_instW(TestCase):
    def setUp(self):
        segList = np.array([[[1, 3]], [[3, 3], [4, 3]], [[2, 0], [2, 2]]])
        self.euclidmst = EuclidMST(segList)

class TestEuclidMST_pt(EuclidMST_instW):
    def runTest(self):
        self.assertEqual(self.euclidmst.size,5)
        d = self.euclidmst.spnTree.sum()
        d /= 2
        self.assertAlmostEqual(d,4)
        tree = self.euclidmst.dfo_nonrec(0)
        self.euclidmst.treetrav_nonrec(tree)
        self.assertAlmostEqual(len(self.euclidmst.nodeTrav),9)

class EuclidMST_instV(TestCase):
    def setUp(self):
        segList = np.array([[[1, 3]], [[3, 3]], [[2, 0]]])
        self.euclidmst = EuclidMST(segList)

class TestEuclidMST_pts(EuclidMST_instV):
    def runTest(self):
        self.assertEqual(self.euclidmst.size,3)
        d = self.euclidmst.spnTree.sum()
        d /= 2
        self.assertAlmostEqual(d,14)
        tree = self.euclidmst.dfo_nonrec(0)
        self.euclidmst.treetrav_nonrec(tree)
        self.assertAlmostEqual(len(self.euclidmst.nodeTrav),5)

class EuclidMST_instU(TestCase):
    def setUp(self):
        segList = np.array([[[3, 3], [4, 3]], [[2, 0], [2, 2]], [[1, 3]]])
        self.euclidmst = EuclidMST(segList)

class TestEuclidMST_ptsU(EuclidMST_instU):
    def runTest(self):
        self.assertEqual(self.euclidmst.size,5)
        d = self.euclidmst.spnTree.sum()
        d /= 2
        self.assertAlmostEqual(d,4)
        tree = self.euclidmst.dfo_nonrec(0)
        self.euclidmst.treetrav_nonrec(tree)
        self.assertAlmostEqual(len(self.euclidmst.nodeTrav), 9)

class EuclidMST_instDEEP(TestCase):
    def setUp(self):
        l = []
        for x in xrange(0,10000):
            l.append([[x,0],[x,1]])
        segList = np.array(l)
        self.euclidmst = EuclidMST(segList)

class TestEuclidMST_deep(EuclidMST_instDEEP):
    def runTest(self):
        # Shouldn't crash
        self.euclidmst.segmentOrdering()
