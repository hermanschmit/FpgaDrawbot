import Segments
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
from numba import jit
import concurrent.futures as cf
from scipy import spatial
import Quantization
import AttractRepel
from functools import partial
import timeit


@jit
def _ptlen_local(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

@jit
def _ptlen2_local(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

@jit
def _repulse(r):
    force = (r ** 12)
    return force

class Maze:
    K0 = 0.1  # [0.1;0.3]
    K1 = 0.15  # [1.5*K0; 2.5*K0]
    D = 10  # dimensional adjustment?
    KMIN = 0.25
    KMAX = 0.7
    Ff = 0.1  # [0.005; 0.3]
    Fb = 0.1  # [0; 0.2]
    Fa = 1.  # [0; 10]
    Fo = 1.

    R0 = 2.
    R1_R0 = 2.5
    R0_B = 10.

    PROCESSORS = 4
    CHUNK = 2000

    '''
    Notes
    I'm going to first debug this with the intent of balancing the forces, without regard to the
    image.
    R0 (zero crossing) will be 6 pixels
    StdDev of Brownian will be 1 pixel
    Faring force will be zero to start
    Resampling on <2 or >8 distance

    Remove the delta and D term
    '''

    def delta(self, i):
        d = self.maze_path[i]
        v = self.imin[d[0], d[1]]
        assert v >= 0
        assert v < 256
        return float(v + 1) / 256.

    def brownian(self):
        mean = [0., 0.]
        cov = [[1., 0.], [0., 1.]]
        size = len(self.maze_path)
        x, y = np.random.multivariate_normal(mean, cov, size).T
        z = list(zip(x, y))
        brown = []
        for i, zi in enumerate(z):
            n = np.array(zi)
            # n = np.multiply(n, self.Fb*self.delta(i)*self.D)
            n = np.multiply(n, self.Fb)
            brown.append(n)
        assert len(brown) == len(self.maze_path)
        return np.array(brown)

    def faring(self):
        null = (0., 0.)
        fare = [null]  # initial element
        for i in range(1,
                       len(self.maze_path) - 1):
            pim1 = np.array(self.maze_path[i - 1])
            pi = np.array(self.maze_path[i])
            pip1 = np.array(self.maze_path[i + 1])
            f = self.Ff * (np.multiply((pim1 + pip1), 0.5) - pi)
            # f = self.Ff * (((pim1*self.delta(i-1) + pip1*self.delta(i+1))/(self.delta(i-1)+self.delta(i+1))) - pi)
            fare.append(f)
        fare.append(null)  # final element
        assert len(fare) == len(self.maze_path)
        return np.array(fare)

    @staticmethod
    @jit
    def density(pixel_val):
        x = 256/(256-pixel_val)
        #x = 1. + math.log(pixel_val + 1, 2.)
        return x

    def R0_val(self, i_pt):
        i_pt0 = max(min(round(i_pt[0]), self.imin.shape[0] - 1), 0)
        i_pt1 = max(min(round(i_pt[1]), self.imin.shape[1] - 1), 0)
        r0 = self.R0 * self.density(self.imin[i_pt0][i_pt1])
        return r0, self.R1_R0 * r0

    def attract_repel_serial(self):
        """
        This is the brute force version
        Returns:
        attract repel vector
        """
        self.kdtree = spatial.cKDTree(self.maze_path)
        returnList = []
        for x in range(0, len(self.maze_path)):
            fi_l = AttractRepel.attract_repel_segment(x, im=self.imin, maze_path=self.maze_path,
                                                      kdtree=self.kdtree, R0=self.R0, R1_R0=self.R1_R0,
                                                      Fa=self.Fa, chunk=1)
            returnList.extend(fi_l)
        return np.array(returnList)

    def attract_repel_parallel(self):
        """
        This is the parallel version, attempt 1
        Returns:
        attract repel vector
        """
        self.kdtree = spatial.cKDTree(self.maze_path)

        mapfunc = partial(AttractRepel.attract_repel_segment, im=self.imin, maze_path=self.maze_path,
                          kdtree=self.kdtree, R0=self.R0, R1_R0=self.R1_R0, Fa=self.Fa, chunk=self.CHUNK)

        with cf.ProcessPoolExecutor(self.PROCESSORS) as pool:
            x = pool.map(mapfunc, range(0, len(self.maze_path), self.CHUNK))
        returnList = []
        for fi_l in x:
            returnList.extend(fi_l)
        return np.array(returnList)

    def boundary_slow(self):
        """
        This is the brute force version
        Returns:
        attract repel vector
        """
        returnList = []
        R1 = 2.0 * self.R0_B
        pt_00 = (self.xmin - 0.5 * self.R0_B, self.ymin - 0.5 * self.R0_B)
        pt_01 = (self.xmin - 0.5 * self.R0_B, self.ymax + 0.5 * self.R0_B)
        pt_11 = (self.xmax + 0.5 * self.R0_B, self.ymax + 0.5 * self.R0_B)
        pt_10 = (self.xmax + 0.5 * self.R0_B, self.ymin - 0.5 * self.R0_B)
        boundary_seg = [pt_00, pt_01, pt_11, pt_10, pt_00]

        for i in range(0,
                       len(self.maze_path)):
            fi = np.array([0., 0.])
            # delta_pi = self.delta(i)
            pi = np.array(self.maze_path[i])
            for j in range(0,
                           len(boundary_seg) - 1):
                j_pt = boundary_seg[j]
                jp1_pt = boundary_seg[j + 1]
                pi2xij, xij = self.seg.distABtoP(j_pt, jp1_pt, pi)
                self.minDist = min(self.minDist, pi2xij)
                if pi2xij < R1:
                    fij = (pi - xij) / max(0.00001, pi2xij)
                    fij *= _repulse(self.R0_B / pi2xij) * self.Fo
                    fi += fij
            returnList.append(fi)
        assert len(returnList) == len(self.maze_path)
        return np.array(returnList)

    def resampling(self):
        tmp3 = []
        ptA = self.maze_path[0]
        tmp3.append(ptA)
        r0_a, _ = self.R0_val(ptA)
        skip = False
        for ptB in self.maze_path[1:]:
            skip = False
            r0_b, _ = self.R0_val(ptB)
            d = _ptlen_local(ptA, ptB)
            r0_ab = (r0_a + r0_b) / 2

            if d > self.KMAX * r0_ab:
                ptAB = np.multiply(np.add(ptA, ptB), 0.5)
                tmp3.append(ptAB)
                tmp3.append(ptB)
            elif d < self.KMIN * r0_ab:
                skip = True
            else:
                tmp3.append(ptB)
            ptA = ptB
            r0_a = r0_b

        # if the last value was skipped, reattach it.
        if skip:
            tmp3.append(ptB)
        self.maze_path = tmp3

    def optimize_loop1(self, loop_bound):
        # Main optimize loop
        # keep running until stopping criteria met
        loop_count = 0
        while True:

            # BIG
            self.minDist = sys.float_info.max
            # compute force on each node
            brownian = self.brownian()
            if len(self.maze_path) < self.CHUNK:
                attract_repel = self.attract_repel_serial()
            else:
                attract_repel = self.attract_repel_parallel()
            fairing = self.faring()
            boundary = self.boundary_slow()

            # move each node
            netforce = np.add(boundary, np.add(fairing, attract_repel))
            # netforce = np.add(boundary,np.add(fairing,np.add(brownian, attract_repel)))
            # netforce = np.add(fairing,np.add(brownian, attract_repel))
            deltaforce = [np.hypot(a[0], a[1]) for a in netforce]
            maxdelta = np.max(deltaforce)
            assert self.minDist / maxdelta > 0.0
            if maxdelta > 1. * self.minDist:
                netforce = np.multiply(netforce, 1. * self.minDist / maxdelta)
            netmove = np.multiply(netforce, 1.)

            # don't scale noise by size maxdelta

            netmove = np.add(netmove, brownian)

            tmp2 = np.add(self.maze_path, netmove)
            tmp2[0] = self.maze_path[0]
            tmp2[-1] = self.maze_path[-1]
            self.maze_path = tmp2

            # resampling
            self.resampling()
            test1 = self.maze_path[-1]
            test2 = np.array((float(self.xmax), float(self.ymax)))
            assert np.array_equal(test1, test2)
            # stopping criteria
            if loop_count > loop_bound:
                break
            loop_count += 1
            if loop_count % 100 == 0:
                print(str(loop_count) + " " + str(len(self.maze_path)))
            if loop_count % 200 == 1:
                plt_x = [a[0] for a in self.maze_path]
                plt_y = [a[1] for a in self.maze_path]
                plt.imshow(np.transpose(self.imin), cmap=cm.Greys)
                plt.plot(plt_x, plt_y, '.-')
                plt.savefig("fig" + str(loop_count) + ".png")
                plt.clf()

        plt_x = [a[0] for a in self.maze_path]
        plt_y = [a[1] for a in self.maze_path]
        plt.plot(plt_x, plt_y, '.-')
        plt.savefig("fig" + str(loop_count) + ".png")

    def optimize_loop2(self, loop_bound):
        # Main optimize loop
        # keep running until stopping criteria met
        loop_count = 0
        start_time = timeit.default_timer()
        while True:

            # compute force on each node
            brownian = self.brownian()
            if len(self.maze_path) < self.CHUNK:
                attract_repel = self.attract_repel_serial()
            else:
                attract_repel = self.attract_repel_parallel()
            fairing = self.faring()
            boundary = self.boundary_slow()

            # move each node
            netforce = np.add(boundary, np.add(fairing, attract_repel))
            deltaforce = [np.hypot(a[0], a[1]) for a in netforce]

            n_neighbor_d2, _ = self.kdtree.query(self.maze_path, 2)
            n_neighbor_d = [x[1] for x in n_neighbor_d2]

            ceil_force = list()
            for nf, nn_d, df in zip(netforce, n_neighbor_d, deltaforce):
                if df > nn_d / 2:
                    ceil_force.append(np.multiply(nf, nn_d / (4. * df)))
                else:
                    ceil_force.append(nf)

            ceil_force = np.array(ceil_force)

            # don't scale noise by size maxdelta

            netmove = np.add(ceil_force, brownian)

            tmp2 = np.add(self.maze_path, netmove)
            tmp2[0] = self.maze_path[0]
            tmp2[-1] = self.maze_path[-1]
            self.maze_path = tmp2

            # resampling
            self.resampling()
            test1 = self.maze_path[-1]
            test2 = np.array((float(self.xmax), float(self.ymax)))
            assert np.array_equal(test1, test2)
            # stopping criteria
            if loop_count > loop_bound:
                break
            loop_count += 1
            if loop_count % 100 == 0:
                elapsed = timeit.default_timer() - start_time
                start_time = timeit.default_timer()
                print(str(loop_count) + " " + str(len(self.maze_path)) + " " + str(elapsed))
            if loop_count % 100 == 0:
                plt_x = [a[0] for a in self.maze_path]
                plt_y = [a[1] for a in self.maze_path]
                plt.imshow(np.transpose(self.imin), cmap=cm.gray)
                plt.plot(plt_x, plt_y, '-')
                plt.savefig("fig" + str(loop_count) + ".png")
                plt.clf()

        plt_x = [a[0] for a in self.maze_path]
        plt_y = [a[1] for a in self.maze_path]
        plt.plot(plt_x, plt_y, '.-')
        plt.savefig("fig" + str(loop_count) + ".png")

    def maze_to_segments(self):
        self.segments = Segments.Segments()
        self.segments.append(self.maze_path)

    def __init__(self, image_matrix, white=1, levels=4):
        """
        :param image_matrix:
        """

        self.imin = image_matrix
        self.xmin = 0
        self.ymin = 0
        self.xmax = self.imin.shape[0] - 1
        self.ymax = self.imin.shape[1] - 1

        # whiten
        self.imin /= white
        self.imin += 255 - (255 // white)

        # quantize
        self.centroids = Quantization.measCentroid(self.imin, levels)
        print(self.centroids)
        nq = np.array([[x * 255 / (levels - 1)] for x in range(0, levels)])
        self.imin = Quantization.quantMatrix(self.imin, nq, self.centroids)

        # Initial segment
        self.maze_path = [(0., 0.)]
        segListEnd = tuple([x - 1 for x in self.imin.shape])
        self.maze_path.append(segListEnd)
        self.seg = Segments.Segments()
        self.minDist = sys.float_info.max

        # self.seg.scale(1.0) # fix the types. Hygiene
