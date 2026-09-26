from unittest import TestCase

import numpy as np
from scipy.sparse import lil_matrix

from EuclidMST import EuclidMST

__author__ = 'herman'


class EuclidMST_instZ(TestCase):
    def setUp(self):
        segList = np.array([[[0, 0], [3, 0]], [[2, 1], [5, 1]]])
        self.euclidmst = EuclidMST(segList)


class TestEuclidMSTZ(EuclidMST_instZ):
    def runTest(self):
        self.assertEqual(self.euclidmst.size, 4)
        d = self.euclidmst.spnTree.sum()
        d /= 2  # spnTree is undirected, so sum is 2x expected
        self.assertAlmostEqual(d, 2)


class EuclidMST_instY(TestCase):
    def setUp(self):
        segList = np.array([[[0, 3], [1, 3]], [[3, 3], [4, 3]], [[2, 0], [2, 2]]])
        self.euclidmst = EuclidMST(segList)


class TestEuclidMSTY(EuclidMST_instY):
    def runTest(self):
        self.assertEqual(self.euclidmst.size, 6)
        d = self.euclidmst.spnTree.sum()
        d /= 2  # spnTree is undirected, so sum is 2x expected
        self.assertAlmostEqual(d, 4)


class TestEuclidMST_dfo(EuclidMST_instY):
    def runTest(self):
        tree = self.euclidmst.dfo_nonrec(0)
        self.euclidmst.treetrav_nonrec(tree)
        self.assertEqual(len(self.euclidmst.nodeTrav), 11)


class EuclidMST_instW(TestCase):
    def setUp(self):
        segList = [[[1, 3]], [[3, 3], [4, 3]], [[2, 0], [2, 2]]]
        self.euclidmst = EuclidMST(segList)


class TestEuclidMST_pt(EuclidMST_instW):
    def runTest(self):
        self.assertEqual(self.euclidmst.size, 5)
        d = self.euclidmst.spnTree.sum()
        d /= 2
        self.assertAlmostEqual(d, 4)
        tree = self.euclidmst.dfo_nonrec(0)
        self.euclidmst.treetrav_nonrec(tree)
        self.assertAlmostEqual(len(self.euclidmst.nodeTrav), 9)


class EuclidMST_instV(TestCase):
    def setUp(self):
        segList = np.array([[[1, 3]], [[3, 3]], [[2, 0]]])
        self.euclidmst = EuclidMST(segList)


class TestEuclidMST_pts(EuclidMST_instV):
    def runTest(self):
        self.assertEqual(self.euclidmst.size, 3)
        d = self.euclidmst.spnTree.sum()
        d /= 2
        self.assertAlmostEqual(d, 14)
        tree = self.euclidmst.dfo_nonrec(0)
        self.euclidmst.treetrav_nonrec(tree)
        self.assertAlmostEqual(len(self.euclidmst.nodeTrav), 5)


class EuclidMST_instU(TestCase):
    def setUp(self):
        segList = [[[3, 3], [4, 3]], [[2, 0], [2, 2]], [[1, 3]]]
        self.euclidmst = EuclidMST(segList)


class TestEuclidMST_ptsU(EuclidMST_instU):
    def runTest(self):
        self.assertEqual(self.euclidmst.size, 5)
        d = self.euclidmst.spnTree.sum()
        d /= 2
        self.assertAlmostEqual(d, 4)
        tree = self.euclidmst.dfo_nonrec(0)
        self.euclidmst.treetrav_nonrec(tree)
        self.assertAlmostEqual(len(self.euclidmst.nodeTrav), 9)

class EuclidMST_inst100(TestCase):
    def setUp(self):
        l = []
        for x in range(0, 100):
            l.append([[x, 0], [x, 1]])
        segList = np.array(l)
        self.euclidmst = EuclidMST(segList)


class TestEuclidMST_100(EuclidMST_inst100):
    def runTest(self):
        # Shouldn't crash
        self.euclidmst.segmentOrdering()


''' Regression tests for two fixed bugs, both only reachable when the graph
built from segmentList is not fully connected -- none of the fixtures above
exercise that, since a handful of well-separated points always triangulates
into a single connected component. '''


class EuclidMST_disconnectedGraph(TestCase):
    '''
    dfo_nonrec's childCount/tree arrays used to be sized to the number of
    nodes actually reached from the start node (narray.shape) instead of
    the graph's real size (self.size). That's only wrong when some node is
    unreachable from node 0 -- here node 1 has no edges at all, so a DFS
    from node 0 only reaches {0, 2}, sizing those arrays to 2 while still
    indexing them by real node id (2), which raised IndexError. Built
    directly on a hand-made disconnected sparse graph, bypassing Delaunay
    entirely, so this targets the array-sizing bug in isolation regardless
    of how a disconnected graph might arise in practice (see
    EuclidMST_duplicatePoints below for that).
    '''
    def setUp(self):
        # A valid 3-point EuclidMST just to get a properly-constructed
        # instance; its own spnTree/distMatrix are replaced below.
        segList = np.array([[[0, 3], [1, 3]], [[3, 3], [4, 3]], [[2, 0], [2, 2]]])
        self.euclidmst = EuclidMST(segList)
        self.euclidmst.size = 3
        graph = lil_matrix((3, 3))
        graph[0, 2] = 1.0  # node 0 -- node 2 connected; node 1 isolated
        graph[2, 0] = 1.0
        self.euclidmst.spnTree = graph.tocsr()
        self.euclidmst.distMatrix = graph.tocsr()


class TestEuclidMST_disconnectedGraph(EuclidMST_disconnectedGraph):
    def runTest(self):
        tree = self.euclidmst.dfo_nonrec(0)  # used to raise IndexError
        self.euclidmst.treetrav_nonrec(tree)
        self.assertEqual(set(self.euclidmst.nodeTrav), {0, 2})


class EuclidMST_duplicatePoints(TestCase):
    '''
    Qhull can leave a point untriangulated -- and so with zero graph edges
    -- when the input has exact duplicate coordinates. A 2+-point segment's
    endpoints get a tiny fallback edge to each other regardless of
    triangulation (see idxl/lidx in __init__), which happens to mask this
    for most duplicate-endpoint cases; a single-point segment gets no such
    fallback. Three single-point segments at the same coordinate reliably
    reproduces it: Qhull keeps one of the three as a real vertex and drops
    the other two entirely, disconnecting them, which used to crash
    dfo_nonrec via the bug above. Fixed by nudging duplicate points apart
    by a sub-pixel epsilon before triangulating.
    '''
    def setUp(self):
        shared = [5, 5]
        self.segList = [
            [shared],
            [shared],
            [shared],
            [[0, 0], [0, 10]],
            [[10, 0], [10, 10]],
            [[0, 10], [10, 0]],
        ]
        self.euclidmst = EuclidMST(self.segList)


class TestEuclidMST_duplicatePoints(EuclidMST_duplicatePoints):
    def runTest(self):
        tree = self.euclidmst.dfo_nonrec(0)  # used to raise IndexError
        self.euclidmst.treetrav_nonrec(tree)
        # every point -- including the two dropped duplicates -- must still
        # be reachable, not just non-crashing.
        self.assertEqual(set(self.euclidmst.nodeTrav), set(range(self.euclidmst.size)))
        self.euclidmst.segmentOrdering()  # shouldn't crash either


'''
class EuclidMST_instDEEP(TestCase):
    def setUp(self):
        l = []
        for x in xrange(0, 10000):
            l.append([[x, 0], [x, 1]])
        segList = np.array(l)
        self.euclidmst = EuclidMST(segList)


class TestEuclidMST_deep(EuclidMST_instDEEP):
    def runTest(self):
        # Shouldn't crash
        self.euclidmst.segmentOrdering()
'''