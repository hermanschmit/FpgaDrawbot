from unittest import TestCase

from Maze import Maze
import numpy as np

__author__ = 'herman'

class Maze_identity10_100(TestCase):
    def setUp(self):
        im = np.eye(10, 10, 0)
        for k in xrange(1, 10):
            im += np.eye(10, 10, k)
        im = 100. * im
        self.MazeInst = Maze(im)

class Maze0(Maze_identity10_100):
    def runTest(self):
        pass
