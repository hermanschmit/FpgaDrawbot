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


