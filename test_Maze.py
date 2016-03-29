from unittest import TestCase

from Maze import Maze
import numpy as np
import sys

__author__ = 'herman'

class Maze_identity40(TestCase):
    def setUp(self):
        im = np.eye(40, 40, 0)
        for k in range(1, 10):
            im += np.eye(40, 40, k)
        im = 100. * im
        self.MazeInst = Maze(im)

class Maze0(Maze_identity40):
    def runTest(self):
        self.MazeInst.optimize_loop1(800)
        pass

class Maze_identity100(TestCase):
    def setUp(self):
        im = np.eye(100, 100, 0)
        for k in range(1, 10):
            im += np.eye(100, 100, k)
        im = 100. * im
        self.MazeInst = Maze(im)

class Maze1b(Maze_identity100):
    def runTest(self):
        self.MazeInst.optimize_loop1(3000)
        pass

class Maze_seg1(TestCase):
    def setUp(self):
        #placekeeper
        im = np.eye(100, 100, 0)
        self.MazeInst = Maze(im)
        self.MazeInst.R0 = 6
        # Because of the fact that neighboring segments are not computed, This should be all zeros
        segList = [[ -100.,  0.],
                   [  0., 0.],
                   [  2., 0.],
                   [ 50., 0.],
                   [ 54., 0.],
                   [100., 0.],
                   [106., 0.],
                   [150., 0.],
                   [158., 0.]
                   ]
        self.MazeInst.seg.segmentList[0] = np.array(segList)

class Maze1(Maze_seg1):
    def runTest(self):
        a_r = self.MazeInst.attract_repel_serial()
        self.assertEqual(a_r[0,0], 0.)
        self.assertEqual(a_r[0,1], 0.)
        self.assertEqual(a_r[1,0], 0.)
        self.assertEqual(a_r[2,0], 0.)
        self.assertEqual(a_r[3,0], 0.)
        self.assertEqual(a_r[4,0], 0.)
        self.assertEqual(a_r[5,0], 0.)
        self.assertEqual(a_r[6,0], 0.)
        self.assertEqual(a_r[7,0], 0.)
        self.assertEqual(self.MazeInst.minDist,sys.float_info.max)

class Maze1p(Maze_seg1):
    def runTest(self):
        a_r = self.MazeInst.attract_repel_parallel()
        self.assertEqual(a_r[0,0], 0.)
        self.assertEqual(a_r[0,1], 0.)
        self.assertEqual(a_r[1,0], 0.)
        self.assertEqual(a_r[2,0], 0.)
        self.assertEqual(a_r[3,0], 0.)
        self.assertEqual(a_r[4,0], 0.)
        self.assertEqual(a_r[5,0], 0.)
        self.assertEqual(a_r[6,0], 0.)
        self.assertEqual(a_r[7,0], 0.)
        self.assertEqual(self.MazeInst.minDist,sys.float_info.max)

class Maze_seg2(TestCase):
    def setUp(self):
        #placekeeper
        # Because of the fact that neighboring segments are not computed, This should be all zeros
        im = np.eye(100, 100, 0)
        self.MazeInst = Maze(im)
        self.MazeInst.R0 = 6
        segList = [[  0., 0.],
                   [  2., 0.],
                   [ 50., 0.],
                   [ 54., 0.],
                   [100., 0.],
                   [106., 0.],
                   [150., 0.],
                   [158., 0.]
                   ]
        self.MazeInst.seg.segmentList[0] = np.array(segList)

class Maze2(Maze_seg2):
    def runTest(self):
        a_r = self.MazeInst.attract_repel_serial()
        self.assertEqual(a_r[0,0], 0.)
        self.assertEqual(a_r[0,1], 0.)
        self.assertEqual(a_r[1,0], 0.)
        self.assertEqual(a_r[2,0], 0.)
        self.assertEqual(a_r[3,0], 0.)
        self.assertEqual(a_r[4,0], 0.)
        self.assertEqual(a_r[5,0], 0.)
        self.assertEqual(a_r[6,0], 0.)
        self.assertEqual(a_r[7,0], 0.)
        self.assertEqual(self.MazeInst.minDist,sys.float_info.max)

class Maze2p(Maze_seg2):
    def runTest(self):
        a_r = self.MazeInst.attract_repel_parallel()
        self.assertEqual(a_r[0,0], 0.)
        self.assertEqual(a_r[0,1], 0.)
        self.assertEqual(a_r[1,0], 0.)
        self.assertEqual(a_r[2,0], 0.)
        self.assertEqual(a_r[3,0], 0.)
        self.assertEqual(a_r[4,0], 0.)
        self.assertEqual(a_r[5,0], 0.)
        self.assertEqual(a_r[6,0], 0.)
        self.assertEqual(a_r[7,0], 0.)
        self.assertEqual(self.MazeInst.minDist,sys.float_info.max)


class Maze_seg3(TestCase):
    def setUp(self):
        # This has dummy points so, we should get forces
        im = np.eye(100, 100, 0)
        self.MazeInst = Maze(im)
        self.MazeInst.R0 = 6
        segList = [[-99.,0.],
                   [  0., 0.],
                   [  1., 0.],
                   [  2., 0.],
                   [ 50., 0.],
                   [ 52., 0.],
                   [ 54., 0.],
                   [100., 0.],
                   [103., 0.],
                   [106., 0.],
                   [150., 0.],
                   [154., 0.],
                   [158., 0.],
                   [200., 0.]
                   ]
        self.MazeInst.seg.segmentList[0] = np.array(segList)

class Maze2(Maze_seg3):
    def runTest(self):
        a_r = self.MazeInst.attract_repel_serial()
        self.assertEqual(a_r[1,0]+a_r[3,0], 0.)
        self.assertLess(a_r[1,0],0.)
        self.assertEqual(a_r[4,0]+a_r[6,0], 0.)
        self.assertLess(a_r[4,0],0.)
        self.assertEqual(a_r[10,0]+a_r[12,0], 0.)
        self.assertGreater(a_r[10,0],0)
        self.assertEqual(self.MazeInst.minDist,2.)

class Maze2p(Maze_seg3):
    def runTest(self):
        a_r = self.MazeInst.attract_repel_parallel()
        self.assertEqual(a_r[1,0]+a_r[3,0], 0.)
        self.assertLess(a_r[1,0],0.)
        self.assertEqual(a_r[4,0]+a_r[6,0], 0.)
        self.assertLess(a_r[4,0],0.)
        self.assertEqual(a_r[10,0]+a_r[12,0], 0.)
        self.assertGreater(a_r[10,0],0)
        self.assertEqual(self.MazeInst.minDist,2.)


