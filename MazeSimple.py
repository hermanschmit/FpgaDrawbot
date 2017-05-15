import concurrent.futures as cf
import math
import sys
import timeit

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from numba import jit

import AttractRepel
import Hilbert
import Quantization
import Segments
import TSPopt


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


class MazeSimple:
    K0 = 0.1  # [0.1;0.3]
    K1 = 0.15  # [1.5*K0; 2.5*K0]
    D = 10  # dimensional adjustment?
    KMIN = 0.25
    KMAX = 0.7
    Ff = 0.1  # [0.005; 0.3]
    Fb = 0.03  # [0; 0.2]
    Fa = 1.  # [0; 10]
    Fo = 1.

    R0 = 2.
    R1_R0 = 2.5
    R0_B = 10.
    TAKEN_SAMPLE_SIZE = 20
    CHUNK = 4000
    PROCESSORS = 4

    def delta(self, i):
        d = self.maze_path[i]
        v = self.imin[d[0], d[1]]
        assert v >= 0
        assert v < 256
        return float(v + 1) / 256.

    @jit(cache=True)
    def brownian(self):
        mean = [0., 0.]
        cov = [[1., 0.], [0., 1.]]
        size = len(self.maze_path)
        x, y = np.random.multivariate_normal(mean, cov, size).T
        z = list(zip(x, y))
        brownA = np.empty([size, 2])
        for i, zi in enumerate(z):
            n = np.array(zi)
            n = np.multiply(n, self.Fb)
            brownA[i] = n
        return brownA

    @staticmethod
    @jit
    def density(pixel_val):
        x = 256 / (256 - pixel_val)
        # x = 1. + math.log(pixel_val + 1, 2.)
        return x

    @jit
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
        fi_l = AttractRepel.attract_repel_global(im=self.imin, maze_path=self.maze_path,
                                                 R0=self.R0, R1_R0=self.R1_R0,
                                                 Fa=self.Fa)
        return np.array(fi_l)

    @jit
    def boundary_slow(self):
        """
        This is the brute force version
        Returns:
        boundary vector
        """
        returnA = np.empty([len(self.maze_path), 2])
        R1 = 2.0 * self.R0_B

        for i in range(0,
                       len(self.maze_path)):
            fi = np.array([0., 0.])
            pi = np.array(self.maze_path[i])
            for j in range(0,
                           len(self.boundary_seg) - 1):
                j_pt = self.boundary_seg[j]
                jp1_pt = self.boundary_seg[j + 1]
                pi2xij, xij = TSPopt.distABtoP(j_pt, jp1_pt, pi)
                self.minDist = min(self.minDist, pi2xij)
                if pi2xij < R1:
                    fij = (pi - xij) / max(0.00001, pi2xij)
                    fij *= _repulse(self.R0_B / pi2xij) * self.Fo
                    fi += fij
            returnA[i] = fi

        return returnA

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
        self.lenList.append(len(self.maze_path))

    def optimize_loop2(self, loop_bound=1000, img_dump=100, equil=1.025, tsp=10):
        # Main optimize loop
        # keep running until stopping criteria met
        loop_count = 0
        start_time = timeit.default_timer()
        while True:

            # compute force on each node
            brownian = self.brownian()
            attract_repel = self.attract_repel_serial()

            boundary = self.boundary_slow()

            # move each node
            netforce = np.add(boundary, attract_repel)
            deltaforce = np.array([np.hypot(a[0], a[1]) for a in netforce])
            maxdelta = deltaforce.max()
            if maxdelta > 0.1:
                ceil_force = np.multiply(netforce,0.1 / maxdelta)
                #for nf, df in zip(netforce, deltaforce):
                #    ceil_force.append(np.multiply(nf, 0.1 / (maxdelta)))
                #ceil_force = np.array(ceil_force)
            else:
                ceil_force = np.array(netforce)

            netmove = np.add(ceil_force, brownian)

            tmp2 = np.add(self.maze_path, netmove)
            tmp3 = [[min(self.bndry_xmax - 1, max(self.xmin + 1, x)),
                     min(self.bndry_ymax - 1, max(self.ymin + 1, y))] for x, y in tmp2]
            tmp3[0] = self.maze_path[0]
            tmp3[-1] = self.maze_path[-1]

            self.maze_path = np.array(tmp3)

            # resampling
            self.resampling()

            # stopping criteria
            if loop_count > loop_bound:
                break

            if loop_count % img_dump == 0:
                self.plotMazeImage("img/fig" + str(loop_count).zfill(5) + ".png")
                elapsed = timeit.default_timer() - start_time
                start_time = timeit.default_timer()
                print(str(loop_count) + " " + str(len(self.maze_path)) + " " + str(elapsed))

            loop_count += 1

        self.plotMazeImage("figLast.png", points=True)



    def plotMazeImage(self, name, points=False,superimpose=False):
        plt_x = [a[0] for a in self.maze_path]
        plt_y = [a[1] for a in self.maze_path]
        if superimpose:
            plt.imshow(np.transpose(self.imin), cmap=cm.gray)

        if points:
            plt.plot(plt_x, plt_y, '.-')
        else:
            plt.plot(plt_x, plt_y, '-', linewidth=0.3)

        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig(name,dpi=600)
        plt.clf()

    def maze_to_segments(self):
        self.segments = Segments.Segments()
        self.segments.append(self.maze_path)


    def __init__(self, image_matrix, white=1, levels=4, init_shape=3):
        """
        :param image_matrix:
        """

        self.dnCount = 0
        self.dnSum = 0.
        self.upCount = 0
        self.upSum = 0.

        self.lenList = list()

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
        levels = min(levels, len(self.centroids))
        levels = max(2, levels)

        nq = np.array([[x * 255 / (levels - 1)] for x in range(0, levels)])
        self.imin = Quantization.quantMatrix(self.imin, nq, self.centroids)
        plt.imshow(self.imin, cmap=cm.gray)
        plt.savefig("figStartOrig.png")
        plt.clf()

        # self.R0_B = self.density(nq[-1][0])

        # Initial segment
        if init_shape == 1:

            moore = []
            m = []
            n = 1 << 7
            for i in range(0, n ** 2):
                x, y = Hilbert.d2xy(n, i, True)
                m.append((x, y))
                moore.append(((self.imin.shape[0] * x) / (n - 1),
                              (self.imin.shape[1] * y) / (n - 1)))
            '''
            Rotate the moore graph to start in the middle
            '''

            m2q = len(moore) // 4
            moore2 = moore[m2q:]
            moore2.extend(moore[:m2q])

            '''
            Add the first and last point to return to start
            '''
            ptAlpha = np.multiply(np.array(self.imin.shape), 0.5)
            moore2.append(tuple(ptAlpha))
            moore2.insert(0, tuple(ptAlpha))

            moore3 = [(0.95 * x + 0.025 * self.imin.shape[0], 0.95 * y + 0.025 * self.imin.shape[1]) for x, y in moore2]
            self.maze_path = np.array(moore3)

            self.plotMazeImage("figStart0.png")
            self.maze_path = TSPopt.simplify(self.maze_path)
            for i in range(10):
                self.resampling()
            self.plotMazeImage("figStart1.png")
            self.maze_path = TSPopt.simplify(self.maze_path)

            while True:
                delta, seg1 = TSPopt.threeOptLocal(self.maze_path, 40)
                self.maze_path = seg1
                if delta == 0.:
                    break

            self.plotMazeImage("figStart2.png")
            for i in range(10):
                self.resampling()
            '''
            Have to add a brownian to thois because when you do the resample, you could end up with points
            on the same line, which will lead to a divb0 issue.
            '''

            brownian = self.brownian()
            self.maze_path = np.add(self.maze_path, brownian)
            self.plotMazeImage("figStart3.png")

        elif init_shape == 2:
            import LSystem

            gosper = LSystem.LSystem(axiom='B',
                                     rules=[('A', 'A-B--B+A++AA+B-'),
                                            ('B', '+A-BB--B-A++A+B')],
                                     angle=60.0)

            gosper.iterate(5)
            self.maze_path = np.array(gosper.segment(initialpt=[200.0, 600.0], d=4.0))
            self.plotMazeImage("figStartGosper0.png")

        elif init_shape == 3:
            import LSystem

            fass2 = LSystem.LSystem(axiom="FX",
                                    rules=[('X','Y-LFL-FRF-LFLFL-FRFR+F'),
                                           ('Y','X+RFR+FLF+RFRFR+FLFL-F'),
                                           ('L','LF+RFR+FL-F-LFLFL-FRFR+'),
                                           ('R','-LFLF+RFRFR+F+RF-LFL-FR')],
                                    angle = 90)
            fass2.iterate(5)
            path1=np.array(fass2.segment(initialpt=[0.0,0.0], d=1.0))
            dim = path1.max() - path1.min()
            path2 = list()
            path1min = path1.min()
            for pt in path1:
                path2.append(((self.imin.shape[0] * (pt[0]-path1min)) / (dim - 1),
                              (self.imin.shape[1] * (pt[1]-path1min)) / (dim - 1)))
            path3 = [(0.95 * x + 0.025 * self.imin.shape[0], 0.95 * y + 0.025 * self.imin.shape[1]) for x, y in path2]
            self.maze_path = path3
            self.plotMazeImage("figFass2_0.png")
            self.maze_path = TSPopt.simplify(self.maze_path)
            for i in range(10):
                self.resampling()
            self.plotMazeImage("figFass2_1.png")
            self.maze_path = TSPopt.simplify(self.maze_path)

            while True:
                delta, seg1 = TSPopt.threeOptLocal(self.maze_path, 40)
                self.maze_path = seg1
                if delta == 0.:
                    break

            self.plotMazeImage("figFass2_2.png")
            for i in range(10):
                self.resampling()

        else:
            self.maze_path = [(0., 0.)]
            segListEnd = tuple([x - 1 for x in self.imin.shape])
            self.maze_path.append(segListEnd)
            self.maze_path = np.array(self.maze_path)
        self.seg = Segments.Segments()

        factor = 0.5
        delta = 0.0
        self.bndry_xmax = self.xmax + factor * self.R0_B - delta
        self.bndry_ymax = self.ymax + factor * self.R0_B - delta
        self.bndry_xmin = self.xmin - factor * self.R0_B + delta
        self.bndry_ymin = self.ymin - factor * self.R0_B + delta
        pt_00 = (self.bndry_xmin, self.bndry_ymin)
        pt_01 = (self.bndry_xmin, self.bndry_ymax)
        pt_11 = (self.bndry_xmax, self.bndry_ymax)
        pt_10 = (self.bndry_xmax, self.bndry_ymin)
        self.boundary_seg = [pt_00, pt_01, pt_11, pt_10, pt_00]

        self.minDist = sys.float_info.max

        # self.seg.scale(1.0) # fix the types. Hygiene
