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
        self.MazeInst.optimize_loop2(5)
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
        self.MazeInst.optimize_loop2(5)
        pass

class Maze_seg1(TestCase):
    def setUp(self):
        #placekeeper
        im = np.eye(200, 200, 0)
        self.MazeInst = Maze(im)
        self.MazeInst.R0 = 1.
        self.MazeInst.R1_R0 = 2.5
        # with pixel value 0, r0 -> 9, r1 -> 22.5
        # Because of the fact that neighboring segments are not computed, This should be all zeros
        segList = [[  0., 0.],
                   [  2., 0.],
                   [ 30., 0.],
                   [ 34., 0.],
                   [ 60., 0.],
                   [ 66., 0.],
                   [100., 0.],
                   [102., 0.],
                   [109., 0.], # should be zero because r0 -> 9.0
                   [199., 0.]
                   ]
        self.MazeInst.maze_path = np.array(segList)

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
#        self.assertEqual(self.MazeInst.minDist, 9.0)

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
#        self.assertEqual(self.MazeInst.minDist, 9.0)

