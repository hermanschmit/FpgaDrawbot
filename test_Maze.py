from unittest import TestCase

import numpy as np

import AttractRepel
import Maze

'''
Maze.py had no test coverage at all before these were added, despite being
CLAUDE.md's own "most elaborate" module and the target of a full rewrite of
its performance-critical inner loop (attract_repel/boundary_slow/faring, now
backed by AttractRepel's compiled kernels) plus the removal of its
ProcessPoolExecutor-based parallelism. That work was verified only by
throwaway comparison scripts during the investigation; these make it
permanent. The underlying math itself is covered by test_AttractRepel.py --
what these check is that Maze wires its own state (imin, R0, R1_R0, Fa,
R0_B, Fo, boundary_seg, maze_path) into that math correctly, plus that the
optimize loop runs without blowing up.
'''


def _small_maze(seed=3, size=30, levels=3):
    rng = np.random.default_rng(seed)
    im = rng.uniform(0, 255, size=(size, size))
    return Maze.Maze(im, levels=levels, init_shape=Maze.Maze.INIT_DIAG)


class MazeConstruction(TestCase):
    def runTest(self):
        m = _small_maze()
        # INIT_DIAG always produces a fixed 21-point diagonal path.
        self.assertEqual(np.asarray(m.maze_path).shape, (21, 2))
        self.assertEqual(m.imin.shape, (30, 30))
        # quantized to at most `levels` distinct values
        self.assertLessEqual(len(np.unique(m.imin)), 3)


class MazeFaringStraightLine(TestCase):
    # Every interior point already sits exactly at the average of its
    # neighbors, so the fairing force pulling it toward that average must
    # be exactly zero everywhere.
    def runTest(self):
        m = _small_maze()
        m.maze_path = np.array([[float(i), float(i)] for i in range(10)])
        fare = m.faring()
        np.testing.assert_array_equal(fare, np.zeros((10, 2)))


class MazeFaringKink(TestCase):
    def runTest(self):
        m = _small_maze()
        m.maze_path = np.array([[0., 0.], [1., 0.], [2., 0.], [2., 5.], [2., 10.]])
        fare = m.faring()
        # endpoints never move
        np.testing.assert_array_equal(fare[0], [0., 0.])
        np.testing.assert_array_equal(fare[-1], [0., 0.])
        # interior points on the straight runs (index 1, 3) are unaffected
        np.testing.assert_array_equal(fare[1], [0., 0.])
        np.testing.assert_array_equal(fare[3], [0., 0.])
        # the kink at index 2: Ff * (0.5*((1,0)+(2,5)) - (2,0))
        expected = m.Ff * np.array([-0.5, 2.5])
        np.testing.assert_allclose(fare[2], expected)


class MazeAttractRepelWiring(TestCase):
    # attract_repel() should be exactly what you get from calling
    # AttractRepel's own pipeline with this Maze instance's own state --
    # this checks the wiring, not the math (see test_AttractRepel.py).
    def runTest(self):
        m = _small_maze()
        rng = np.random.default_rng(9)
        m.maze_path = rng.uniform(0, 29, size=(15, 2))

        out = m.attract_repel()

        maze_path = np.asarray(m.maze_path, dtype=np.float64)
        r0_arr, r1_arr = AttractRepel.r0_r1_vectorized(maze_path, m.imin, m.R0, m.R1_R0)
        offsets, flat_j = AttractRepel.build_candidate_csr(maze_path, m.kdtree, r1_arr)
        direct = AttractRepel.attract_repel_kernel(maze_path, r0_arr, r1_arr, offsets, flat_j, m.Fa)

        np.testing.assert_array_equal(out, direct)


class MazeBoundaryWiring(TestCase):
    def runTest(self):
        m = _small_maze()
        rng = np.random.default_rng(9)
        m.maze_path = rng.uniform(0, 29, size=(15, 2))
        m.minDist = float('inf')

        out = m.boundary_slow()

        maze_path = np.asarray(m.maze_path, dtype=np.float64)
        boundary_seg = np.asarray(m.boundary_seg, dtype=np.float64)
        direct_out, direct_mindist = AttractRepel.boundary_kernel(maze_path, boundary_seg, m.R0_B, m.Fo)

        np.testing.assert_array_equal(out, direct_out)
        self.assertEqual(m.minDist, float(direct_mindist.min()))


class MazeOptimizeLoopSmoke(TestCase):
    # Shouldn't crash, and shouldn't produce a degenerate path -- exercises
    # the full per-iteration pipeline (attract_repel/faring/boundary_slow/
    # resampling) together, unlike the wiring tests above which call each
    # piece in isolation.
    def runTest(self):
        m = _small_maze()
        m.optimize_loop2(loop_bound=3, img_dump=-1, equil=2, tsp=100)
        path = np.asarray(m.maze_path, dtype=float)
        self.assertFalse(np.isnan(path).any())
        self.assertFalse(np.isinf(path).any())
        self.assertGreater(len(path), 2)
