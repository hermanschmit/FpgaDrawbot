from unittest import TestCase

import numpy as np
from scipy import spatial

import AttractRepel

'''
AttractRepel had no test coverage before these were added, despite housing
the compiled hot-path force kernels (attract_repel_kernel, boundary_kernel)
that a full rewrite of Maze.py's performance-critical inner loop went
through this same investigation, verified only by throwaway comparison
scripts at the time. These make that verification permanent.
'''

''' _seg_dist testing '''

class SegDist(TestCase):
    def runTest(self):
        # Perpendicular: closest point is the foot of the perpendicular.
        dist, x, y = AttractRepel._seg_dist(0., 0., 0., 10., 5., 5.)
        self.assertEqual(dist, 5.)
        self.assertEqual((x, y), (0., 5.))

        # Beyond the b end: clamps to b.
        dist, x, y = AttractRepel._seg_dist(0., 1., 0., 2., 0., 4.)
        self.assertEqual(dist, 2.)
        self.assertEqual((x, y), (0., 2.))

        # Beyond the a end: clamps to a.
        dist, x, y = AttractRepel._seg_dist(0., 1., 0., 2., 0., -3.)
        self.assertEqual(dist, 4.)
        self.assertEqual((x, y), (0., 1.))

        # On the segment: distance floors to 1e-10 instead of 0, so a
        # Lennard-Jones-style 1/dist force never divides by zero.
        dist, x, y = AttractRepel._seg_dist(0., 0., 10., 0., 5., 0.)
        self.assertEqual(dist, 1e-10)
        self.assertEqual((x, y), (5., 0.))


''' r0_r1_vectorized testing '''

class R0R1Vectorized(TestCase):
    def runTest(self):
        im = np.array([[0., 110.], [220., 0.]])
        maze_path = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        R0, R1_R0 = 2.0, 2.5
        r0, r1 = AttractRepel.r0_r1_vectorized(maze_path, im, R0, R1_R0)
        # r0 = R0 * 256/(256-pixel_val) at each point's own (rounded) pixel
        expected_r0 = R0 * 256. / (256. - np.array([0., 110., 220., 0.]))
        np.testing.assert_allclose(r0, expected_r0)
        np.testing.assert_allclose(r1, R1_R0 * expected_r0)


''' build_candidate_csr testing '''

class BuildCandidateCSR(TestCase):
    # 5 points on a line 0..4; radius 1.5 reaches only immediate neighbors.
    def setUp(self):
        self.pts = np.array([[0., 0.], [0., 1.], [0., 2.], [0., 3.], [0., 4.]])
        self.kdtree = spatial.cKDTree(self.pts)
        self.r1 = np.full(5, 1.5)

    def runTest(self):
        offsets, flat_j = AttractRepel.build_candidate_csr(self.pts, self.kdtree, self.r1)

        def candidates(i):
            return sorted(flat_j[offsets[i]:offsets[i + 1]].tolist())

        # point i's raw neighbors are {i-1,i,i+1} (clipped to range); the
        # candidate set unions that with each neighbor's predecessor, then
        # drops indices < 0 or == the last point (no segment starts there).
        self.assertEqual(candidates(0), [0, 1])
        self.assertEqual(candidates(1), [0, 1, 2])
        self.assertEqual(candidates(2), [0, 1, 2, 3])
        self.assertEqual(candidates(3), [1, 2, 3])
        self.assertEqual(candidates(4), [2, 3])


''' attract_repel_kernel / boundary_kernel testing

Verified against independent, straightforward (non-compiled) reference
implementations of the same documented formulas, rather than against any
prior version of the compiled code -- so this doesn't just pin down
"whatever the kernel currently does" but actually checks it against the
underlying math. The Lennard-Jones-style 1/dist**12 term makes a few
points/segments coming close together produce very large but still
correctly-matching force values; excluding the top 10% by magnitude gives
a tight comparison on the rest without that being flaky.
'''

def _ref_seg_dist(a, b, p):
    seg = b - a
    seglen_sqrd = np.dot(seg, seg)
    u = 0.0 if seglen_sqrd == 0 else np.dot(p - a, seg) / seglen_sqrd
    u = min(1.0, max(0.0, u))
    closest = a + u * seg
    dist = max(np.linalg.norm(closest - p), 1e-10)
    return dist, closest


def _ref_attract_repel(maze_path, r0_arr, r1_arr, offsets, flat_j, Fa):
    n = len(maze_path)
    out = np.zeros((n, 2))
    for i in range(n):
        for k in range(offsets[i], offsets[i + 1]):
            j = flat_j[k]
            if i - 2 <= j < i + 2:
                continue
            dist, closest = _ref_seg_dist(maze_path[j], maze_path[j + 1], maze_path[i])
            if dist < r1_arr[i]:
                ratio = r0_arr[i] / dist
                force = (ratio ** 12 - ratio ** 6) * Fa / dist
                out[i] += (maze_path[i] - closest) * force
    return out


def _ref_boundary(maze_path, boundary_seg, R0_B, Fo):
    n = len(maze_path)
    R1 = 2.0 * R0_B
    out = np.zeros((n, 2))
    min_dist = np.empty(n)
    for i in range(n):
        local_min = float('inf')
        for j in range(len(boundary_seg) - 1):
            dist, closest = _ref_seg_dist(boundary_seg[j], boundary_seg[j + 1], maze_path[i])
            local_min = min(local_min, dist)
            if dist < R1:
                ratio = R0_B / dist
                denom = max(dist, 0.00001)
                force = (ratio ** 12) * Fo / denom
                out[i] += (maze_path[i] - closest) * force
        min_dist[i] = local_min
    return out, min_dist


class KernelFixture(TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        n = 40
        self.maze_path = rng.uniform(0, 50, size=(n, 2))
        im = rng.uniform(0, 200, size=(60, 60))
        self.R0, self.R1_R0, self.Fa = 2.0, 2.5, 1.0
        self.kdtree = spatial.cKDTree(self.maze_path)
        self.r0_arr, self.r1_arr = AttractRepel.r0_r1_vectorized(self.maze_path, im, self.R0, self.R1_R0)
        self.offsets, self.flat_j = AttractRepel.build_candidate_csr(self.maze_path, self.kdtree, self.r1_arr)
        self.boundary_seg = np.array([[0., 0.], [0., 50.], [50., 50.], [50., 0.], [0., 0.]])
        self.R0_B, self.Fo = 10.0, 1.0


class AttractRepelKernelTest0(KernelFixture):
    def runTest(self):
        kernel_out = AttractRepel.attract_repel_kernel(
            self.maze_path, self.r0_arr, self.r1_arr, self.offsets, self.flat_j, self.Fa)
        ref_out = _ref_attract_repel(
            self.maze_path, self.r0_arr, self.r1_arr, self.offsets, self.flat_j, self.Fa)
        self.assertTrue(np.allclose(kernel_out, ref_out))
        mag = np.abs(kernel_out).max(axis=1)
        mask = mag < np.percentile(mag, 90)
        self.assertTrue(np.allclose(kernel_out[mask], ref_out[mask], atol=1e-6, rtol=1e-6))


class BoundaryKernelTest0(KernelFixture):
    def runTest(self):
        kernel_out, kernel_mindist = AttractRepel.boundary_kernel(
            self.maze_path, self.boundary_seg, self.R0_B, self.Fo)
        ref_out, ref_mindist = _ref_boundary(
            self.maze_path, self.boundary_seg, self.R0_B, self.Fo)
        self.assertTrue(np.allclose(kernel_out, ref_out))
        self.assertTrue(np.allclose(kernel_mindist, ref_mindist))
        mag = np.abs(kernel_out).max(axis=1)
        mask = mag < np.percentile(mag, 90)
        self.assertTrue(np.allclose(kernel_out[mask], ref_out[mask], atol=1e-6, rtol=1e-6))
