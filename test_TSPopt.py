from unittest import TestCase
import TSPopt
import numpy

''' TSP testing '''

class TSP0(TestCase):
    def setUp(self):
        self.segList1 = [(0., 1.), (0., 2.), (0., 3.), (0., 4.), (0., 5.)]


class TSP0Test0(TSP0):
    def runTest(self):
        delta, seglist2 = TSPopt.threeOptLoop(self.segList1)
        self.assertEqual(delta,0.)
        self.assertEqual(numpy.amax(seglist2),5.0)
        self.assertEqual(numpy.amin(seglist2),0.0)

class TSP1(TestCase):
    # In this testcase, 0,4 is last, so it cannot be optimized fully.
    def setUp(self):
        self.segList1 = [(0., 1.), (0., 2.), (0., 6.), (0., 5.), (0., 3.), (0., 4.)]

class TSP1Test0(TSP1):
    def runTest(self):
        delta, seglist2 = TSPopt.threeOptLoop(self.segList1)
        self.assertEqual(delta,-2.0)
        self.assertEqual(numpy.amax(seglist2),6.0)
        self.assertEqual(numpy.amin(seglist2),0.0)

class TSP2(TestCase):
    def setUp(self):
        self.segList1 = [(0., 1.), (0., 2.), (0., 6.), (0., 5.), (0., 3.), (0., 4.), (0., 7.)]

class TSP2Test0(TSP2):
    def runTest(self):
        delta, seglist2 = TSPopt.threeOptLoop(self.segList1)
        self.assertEqual(delta,-6.0)
        self.assertEqual(numpy.amax(seglist2),7.0)
        self.assertEqual(numpy.amin(seglist2),0.0)

''' distABtoP Testing '''

class Dist0(TestCase):
    def setUp(self):
        self.segList1 = [(0., 1.), (0., 2.), (0., 3.), (0., 4.), (0., 5.)]

class Dist0Test0(Dist0):
    def runTest(self):
        d1,(x,y) = TSPopt.distABtoP(self.segList1[0],
                                self.segList1[1],
                                self.segList1[2])
        self.assertEqual(d1, 1)
        self.assertEqual(x,0.)
        self.assertEqual(y,2.)
        d2,(x,y) = TSPopt.distABtoP(self.segList1[0],
                                self.segList1[1],
                                self.segList1[3])
        self.assertEqual(d2, 2)
        self.assertEqual(x,0.)
        self.assertEqual(y,2.)
        d3,(x,y) = TSPopt.distABtoP(self.segList1[0],
                                self.segList1[1],
                                (1., 1.))
        self.assertEqual(d3, 1)
        self.assertEqual(x,0.)
        self.assertEqual(y,1.)
        d4,(x,y) = TSPopt.distABtoP(self.segList1[0],
                                self.segList1[2],
                                (1., 1.))
        self.assertEqual(d4, 1)
        self.assertEqual(x,0.)
        self.assertEqual(y,1.)
        d5,(x,y) = TSPopt.distABtoP(self.segList1[0],
                                self.segList1[2],
                                (-5., 2.))
        self.assertEqual(d5, 5)
        self.assertEqual(x,0.)
        self.assertEqual(y,2.)
        d6,(x,y) = TSPopt.distABtoP(self.segList1[0],
                                self.segList1[2],
                                (4., 6.))
        self.assertEqual(d6, 5)
        self.assertEqual(x,0.)
        self.assertEqual(y,3.)

''' threeOptLocal testing

threeOptLocal had zero test coverage before these were added -- every
existing test above exercises threeOptLoop or distABtoP instead. That gap
is what let a change to threeOptLocal (batching its per-point kdtree
queries) ship numerically correct but ~2.6x slower, undetected, and
separately let two pre-existing edge-case bugs go unnoticed: neither showed
up until it was actually run across a range of sizes/nn it had never been
exercised at before.
'''


def _path_length(points):
    points = numpy.asarray(points, dtype=float)
    diffs = points[1:] - points[:-1]
    return float(numpy.sum(numpy.hypot(diffs[:, 0], diffs[:, 1])))


class ThreeOptLocalLine5(TestCase):
    # Same fixture as TSP0: already sorted, nothing to improve.
    def setUp(self):
        self.segList1 = numpy.array([(0., 1.), (0., 2.), (0., 3.), (0., 4.), (0., 5.)])

class ThreeOptLocalLine5Test0(ThreeOptLocalLine5):
    def runTest(self):
        delta, result = TSPopt.threeOptLocal(self.segList1.copy(), nn=4)
        self.assertEqual(delta, 0.)
        self.assertEqual(numpy.amax(result), 5.0)
        self.assertEqual(numpy.amin(result), 0.0)


class ThreeOptLocalScrambled6(TestCase):
    # Same fixture as TSP1, where threeOptLoop's exhaustive search finds
    # -2.0. threeOptLocal's k-nearest-neighbor-limited search is a weaker,
    # faster local search, not a full 3-opt -- with nn small relative to n
    # it does not find that improvement here, and should leave the path
    # untouched rather than doing something worse.
    def setUp(self):
        self.segList1 = numpy.array([(0., 1.), (0., 2.), (0., 6.), (0., 5.), (0., 3.), (0., 4.)])

class ThreeOptLocalScrambled6Test0(ThreeOptLocalScrambled6):
    def runTest(self):
        delta, result = TSPopt.threeOptLocal(self.segList1.copy(), nn=3)
        self.assertEqual(delta, 0.)
        numpy.testing.assert_array_equal(result, self.segList1)


class ThreeOptLocalScrambled7(TestCase):
    # Same fixture as TSP2, where threeOptLoop's exhaustive search finds
    # -6.0 (fully sorted). With nn large enough relative to n, threeOptLocal
    # finds that same optimum; with nn too small, it finds nothing.
    def setUp(self):
        self.segList1 = numpy.array([(0., 1.), (0., 2.), (0., 6.), (0., 5.),
                                     (0., 3.), (0., 4.), (0., 7.)])

class ThreeOptLocalScrambled7Test0(ThreeOptLocalScrambled7):
    def runTest(self):
        delta, result = TSPopt.threeOptLocal(self.segList1.copy(), nn=3)
        self.assertEqual(delta, 0.)

class ThreeOptLocalScrambled7Test1(ThreeOptLocalScrambled7):
    def runTest(self):
        delta, result = TSPopt.threeOptLocal(self.segList1.copy(), nn=5)
        self.assertEqual(delta, -6.0)
        numpy.testing.assert_array_equal(result, numpy.array([[0., float(i)] for i in range(1, 8)]))


class ThreeOptLocalInvariants(TestCase):
    '''
    Property-based check across a range of sizes/nn/seeds instead of one
    fixed example: whatever threeOptLocal does internally, it must always
    (a) return the same multiset of points, just possibly reordered, (b)
    never claim an improvement it didn't actually make -- the returned
    delta must match the real path-length change -- and (c) never make the
    path longer. This is the kind of coverage that would catch a change
    that silently altered results. It would NOT have caught the ~2.6x
    slowdown from the batched-query change: that was numerically identical
    to the original, so only a timed benchmark (not a correctness test)
    would catch a regression of that specific kind.
    '''
    def runTest(self):
        rng = numpy.random.default_rng(12345)
        for n, nn in [(5, 3), (10, 5), (50, 5), (200, 5), (200, 40), (500, 10), (2000, 5)]:
            with self.subTest(n=n, nn=nn):
                seg0 = rng.uniform(0, 500, size=(n, 2))
                before = _path_length(seg0)
                delta, result = TSPopt.threeOptLocal(seg0.copy(), nn=nn)
                after = _path_length(result)
                self.assertLessEqual(delta, 0.)
                self.assertAlmostEqual(before + delta, after, places=6)
                self.assertEqual(sorted(map(tuple, seg0.tolist())),
                                 sorted(map(tuple, numpy.asarray(result).tolist())))


class ThreeOptLocalAllCandidatesTooClose(TestCase):
    '''
    Regression test for a fixed bug: when every neighbor candidate for a
    given point failed the `c <= a+1` filter, the inner loop that sets
    `delta` never ran, and the subsequent `if delta < 0` read a
    stale-or-unbound `delta` -- crashing with UnboundLocalError the first
    time it happened at all, and silently mis-breaking the search on a
    stale value from an unrelated earlier point otherwise. Found while
    investigating a performance change to threeOptLocal; not a contrived
    corner case -- this reproduces on a plain evenly-spaced line. Fixed by
    resetting delta=0 for every candidate c.
    '''
    def runTest(self):
        seg0 = numpy.array([[0., float(i)] for i in range(5)])
        delta, result = TSPopt.threeOptLocal(seg0.copy(), nn=5)
        self.assertEqual(delta, 0.)
        numpy.testing.assert_array_equal(result, seg0)


class ThreeOptLocalNNExceedsCount(TestCase):
    '''
    Regression test for a fixed bug: when nn was >= the number of points,
    scipy padded missing neighbors with the out-of-bounds sentinel index n,
    which survived threeOptLocal's filtering often enough to be used as a
    real array index into seg0, raising IndexError. Fixed by clamping nn to
    at most len(seg0) before querying.
    '''
    def runTest(self):
        seg0 = numpy.array([[0., float(i)] for i in range(4)])
        delta, result = TSPopt.threeOptLocal(seg0.copy(), nn=5)
        self.assertEqual(delta, 0.)
        numpy.testing.assert_array_equal(result, seg0)


