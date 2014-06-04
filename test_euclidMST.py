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
        segList = np.array([[[0, 3], [1, 3]], [[3, 3], [4, 3]], [[2,0],[2,2]]])
        self.euclidmst = EuclidMST(segList)


class TestEuclidMSTY(EuclidMST_instY):
    def runTest(self):
        self.assertEqual(self.euclidmst.size,6)
        d = self.euclidmst.spnTree.sum()
        d /= 2 # spnTree is undirected, so sum is 2x expected
        self.assertAlmostEqual(d,4)
