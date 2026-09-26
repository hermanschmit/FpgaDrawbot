import math
import os
import statistics
import sys
import timeit
import uuid

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from scipy import spatial, stats, ndimage
from skimage.morphology import skeletonize

import AttractRepel
import Hilbert
import Quantization
import Segments
import TSPopt
import Skeleton


def _save_debug_image(name, dpi=None):
    """
    Best-effort debug output: two concurrent runs writing plain, fixed or
    loop-count-only filenames (figStartOrig.png, img/fig00199.png, ...) can
    momentarily collide on the same path and hit an OS-level write error --
    each Maze/Skeleton instance tags its own filenames to avoid that (see
    Maze._run_tag), but this still guards against any other transient
    failure (antivirus, cloud-sync locking, ...) so a lost preview frame
    doesn't take down an otherwise-successful multi-minute run.
    """
    try:
        plt.savefig(name, dpi=dpi)
    except OSError as e:
        print("Warning: failed to save debug image %r (%s); continuing." % (name, e))
    finally:
        plt.clf()


def _write_debug_svg(segments, name):
    """Same best-effort reasoning as _save_debug_image, for the periodic
    TSP debug .svg dumps in optimize_loop2."""
    try:
        segments.svgwrite(name)
    except OSError as e:
        print("Warning: failed to write debug svg %r (%s); continuing." % (name, e))


@jit
def _ptlen_local(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


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
    TAKEN_SAMPLE_SIZE = 20

    INIT_MOORE = 1
    INIT_FASS = 3
    INIT_DIAG = 4
    INIT_SKEL = 5



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
        brownA = np.empty([size, 2])
        for i, zi in enumerate(z):
            n = np.array(zi)
            n = np.multiply(n, self.Fb)
            brownA[i] = n
        return brownA

    def faring(self):
        p = np.asarray(self.maze_path, dtype=np.float64)
        fare = np.zeros_like(p)
        if len(p) > 2:
            fare[1:-1] = self.Ff * (0.5 * (p[:-2] + p[2:]) - p[1:-1])
        return fare

    @staticmethod
    def density(pixel_val):
        x = 256 / (256 - pixel_val)
        # x = 1. + math.log(pixel_val + 1, 2.)
        return x

    def R0_val(self, i_pt):
        i_pt0 = max(min(round(i_pt[0]), self.imin.shape[0] - 1), 0)
        i_pt1 = max(min(round(i_pt[1]), self.imin.shape[1] - 1), 0)
        r0 = self.R0 * self.density(self.imin[int(i_pt0)][int(i_pt1)])
        return r0, self.R1_R0 * r0

    def attract_repel(self):
        """
        Attract/repel force vector, computed as one compiled, multi-threaded
        pass instead of a Python loop calling small jitted leaf functions per
        point/neighbor pair (that dispatch overhead, not the math itself, was
        the actual bottleneck -- see AttractRepel.attract_repel_kernel).
        """
        self.kdtree = spatial.cKDTree(self.maze_path)
        maze_path = np.asarray(self.maze_path, dtype=np.float64)
        r0_arr, r1_arr = AttractRepel.r0_r1_vectorized(maze_path, self.imin, self.R0, self.R1_R0)
        offsets, flat_j = AttractRepel.build_candidate_csr(maze_path, self.kdtree, r1_arr)
        return AttractRepel.attract_repel_kernel(maze_path, r0_arr, r1_arr, offsets, flat_j, self.Fa)

    def boundary_slow(self):
        """
        Repulsive force from the bounding rectangle, computed as one
        compiled, multi-threaded pass (see AttractRepel.boundary_kernel).
        """
        maze_path = np.asarray(self.maze_path, dtype=np.float64)
        boundary_seg = np.asarray(self.boundary_seg, dtype=np.float64)
        out, min_dist = AttractRepel.boundary_kernel(maze_path, boundary_seg, self.R0_B, self.Fo)
        self.minDist = min(self.minDist, float(min_dist.min()))
        return out

    def resampling(self):
        # Insert/skip/keep decisions only ever compare each point against
        # its ORIGINAL predecessor in self.maze_path (never against the
        # filtered/growing tmp3), so every decision is independent and the
        # expensive per-point work -- R0_val's pixel lookup and the
        # consecutive-point distance -- can be computed for the whole path
        # at once instead of one point at a time in a Python loop calling a
        # jitted leaf function per point.
        maze_path = np.asarray(self.maze_path, dtype=np.float64)
        n = len(maze_path)
        r0_arr, _ = AttractRepel.r0_r1_vectorized(maze_path, self.imin, self.R0, self.R1_R0)
        diffs = maze_path[1:] - maze_path[:-1]
        d = np.hypot(diffs[:, 0], diffs[:, 1])
        r0_ab = (r0_arr[:-1] + r0_arr[1:]) / 2.0

        insert_mask = d > self.KMAX * r0_ab
        skip_mask = d < self.KMIN * r0_ab
        # The last point is never actually dropped, no matter what its own
        # distance test says -- matches the original's "reattach if the
        # final point was skipped" special case.
        skip_mask[-1] = False
        midpoints = 0.5 * (maze_path[:-1] + maze_path[1:])

        tmp3 = [maze_path[0]]
        for i in range(n - 1):
            if insert_mask[i]:
                tmp3.append(midpoints[i])
                tmp3.append(maze_path[i + 1])
            elif skip_mask[i]:
                continue
            else:
                tmp3.append(maze_path[i + 1])
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
            attract_repel = self.attract_repel()
            fairing = self.faring()
            boundary = self.boundary_slow()

            # move each node
            netforce = boundary + fairing + attract_repel
            deltaforce = np.hypot(netforce[:, 0], netforce[:, 1])

            n_neighbor_d2, _ = self.kdtree.query(self.maze_path, 2)
            n_neighbor_d = n_neighbor_d2[:, 1]

            # Cap any move that would overshoot halfway to the nearest other
            # node. Only points that actually get capped need the division,
            # but np.where evaluates both branches for every element, so
            # some elements harmlessly divide by a deltaforce of 0 here
            # (the result is discarded by np.where for those, same as the
            # original's if/else skipped the division for them entirely).
            with np.errstate(divide='ignore', invalid='ignore'):
                scale = np.where(deltaforce > n_neighbor_d / 2,
                                 n_neighbor_d / (4. * deltaforce), 1.0)
            ceil_force = netforce * scale[:, np.newaxis]

            netmove = ceil_force + brownian

            maze_path = np.asarray(self.maze_path, dtype=np.float64)
            tmp2 = maze_path + netmove
            tmp3 = np.clip(tmp2,
                          [self.xmin + 1, self.ymin + 1],
                          [self.bndry_xmax - 1, self.bndry_ymax - 1])
            tmp3[0] = maze_path[0]
            tmp3[-1] = maze_path[-1]

            self.maze_path = tmp3

            # resampling
            self.resampling()
            # stopping criteria
            if loop_count > loop_bound:
                break

            if self.stopping(equil):
                break

            if loop_count % tsp == 0:
                self.maze_to_segments()
                _write_debug_svg(self.segments, "svg/TSP_" + str(loop_count) + "_" + self._run_tag + "a.svg")
                while True:
                    delta, seg1 = TSPopt.threeOptLocal(self.maze_path, 30)
                    self.maze_path = seg1
                    print("TSP: " + str(delta) + " loop: " + str(loop_count))
                    if delta == 0.:
                        break
                self.maze_to_segments()
                _write_debug_svg(self.segments, "svg/TSP_" + str(loop_count) + "_" + self._run_tag + "b.svg")

            if img_dump > 0 and loop_count % img_dump == 0:
                self.plotMazeImage("img/fig" + str(loop_count).zfill(5) + "_" + self._run_tag + ".png")
                elapsed = timeit.default_timer() - start_time
                start_time = timeit.default_timer()
                print(str(loop_count) + " " + str(len(self.maze_path)) + " " + str(elapsed))

            loop_count += 1

        self.plotMazeImage("figLast_%s.png" % self._run_tag, points=True)

    def stopping(self, equil_ratio):
        if len(self.lenList) > 40:
            stddev = statistics.stdev(self.lenList[-40:])
            mean = statistics.mean(self.lenList[-40:])
            slope, _, rval, _, _ = stats.linregress(range(40), self.lenList[-40:])
            # print("slope: "+str(slope))
            # print("r2: "+str(rval**2))
            # print("stddev/mean: " + str(stddev/mean))
            if stddev / mean < 0.0025 and slope < 2.:
                return True
            return False
        return False

    def equilibrium(self, equil_ratio, delta):
        if self.upCount + self.dnCount < self.TAKEN_SAMPLE_SIZE:
            if delta < 0.:
                self.dnCount += 1
                self.dnSum += delta
            elif delta > 0.:
                self.upCount += 1
                self.upSum += delta
            return False
        else:
            if self.dnCount == 0 or self.upCount == 0:
                print("No Equil Check")
                print(self.upCount, self.upSum, self.dnCount, self.dnSum)
                self.upCount = 0
                self.dnCount = 0
                self.upSum = 0.
                self.dnSum = 0.
                return False
            dnAvg = -1. * self.dnSum / self.dnCount
            upAvg = self.upSum / self.upCount
            self.upCount = 0
            self.dnCount = 0
            self.upSum = 0.
            self.dnSum = 0.
            print("equil: " + str(upAvg) + " " + str(dnAvg))
            if dnAvg < upAvg and upAvg / dnAvg < equil_ratio or \
                                    dnAvg >= upAvg and dnAvg / upAvg < equil_ratio:
                return True
            else:
                print("Equil Fail")
                return False

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
        _save_debug_image(name, dpi=600)

    def maze_to_segments(self):
        self.segments = Segments.Segments()
        self.segments.append(self.maze_path)

    def mazeSegmentOptimize(self):
        while True:
            delta, self.maze_path = TSPopt.threeOptLocal(self.maze_path, 40)
            if delta == 0:
                break

    def __init__(self, image_matrix, white=1, levels=4, init_shape=INIT_SKEL, maxQuant = 220):
        """
        :param image_matrix:
        """

        self.dnCount = 0
        self.dnSum = 0.
        self.upCount = 0
        self.upSum = 0.

        self.lenList = list()

        # Included in every debug-output filename below so two concurrent
        # runs (e.g. on different images) never fight over the same path --
        # see _save_debug_image.
        self._run_tag = "%d_%s" % (os.getpid(), uuid.uuid4().hex[:6])

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
        print("Centroids: ")
        print(self.centroids)
        levels = min(levels, len(self.centroids))
        levels = max(2, levels)
        nq = np.array([[x * maxQuant / (levels - 1)] for x in range(0, levels)])
        print(nq)
        self.imin = Quantization.quantMatrix(self.imin, nq, self.centroids)
        plt.imshow(self.imin, cmap=cm.gray)
        _save_debug_image("figStartOrig_%s.png" % self._run_tag)

        # self.R0_B = self.density(nq[-1][0])

        # Initial segment
        if init_shape == self.INIT_MOORE:

            moore = []
            m = []
            n = 1 << 7
            for i in range(0, n ** 2):
                x, y = Hilbert.d2xy(n, i, True)
                m.append((x, y))
                moore.append(((self.imin.shape[0] * x) / (n - 1),
                              (self.imin.shape[1] * y) / (n - 1)))
            '''
            Ordinarily, the moore curve starts in the middle of one
            edge.
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

            self.maze_path = TSPopt.simplify(self.maze_path)
            for i in range(10):
                self.resampling()
            self.maze_path = TSPopt.simplify(self.maze_path)

            while True:
                delta, seg1 = TSPopt.threeOptLocal(self.maze_path, 40)
                self.maze_path = seg1
                if delta == 0.:
                    break

            for i in range(10):
                self.resampling()
            '''
            Have to add a brownian to thois because when you do the resample, you could end up with points
            on the same line, which will lead to a divb0 issue.
            '''

            brownian = self.brownian()
            self.maze_path = np.add(self.maze_path, brownian)
            self.plotMazeImage("figStartMoore_%s.png" % self._run_tag,superimpose=True)

        elif init_shape == self.INIT_FASS:
            """ FASS is for Filling, self-Avoiding, Simple, and self-Similar.
                This is one instance of a FASS system. This one starts in the
                center, which is why it is advantageous for us.
            """
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
            self.plotMazeImage("figStartFass0_%s.png" % self._run_tag,superimpose=True)
            self.maze_path = TSPopt.simplify(self.maze_path)
            for _ in range(10):
                self.resampling()
            self.maze_path = TSPopt.simplify(self.maze_path)

            while True:
                delta, seg1 = TSPopt.threeOptLocal(self.maze_path, 40)
                self.maze_path = seg1
                if delta == 0.:
                    break

            for i in range(10):
                self.resampling()

            self.plotMazeImage("figStartFass_%s.png" % self._run_tag,superimpose=True)

        elif init_shape == self.INIT_DIAG:
            # simple diagonal
            segListEnd = tuple([x - 1 for x in self.imin.shape])
            self.maze_path = list()
            for i in range(20):
                self.maze_path.append((int(i*segListEnd[0]/20),
                                       int(i*segListEnd[1]/20)))
            self.maze_path.append(segListEnd)
            self.maze_path = np.array(self.maze_path)

        elif init_shape == self.INIT_SKEL: # use skeleton to cover most of dark image (>128)
            b = np.array([[0.], [128.]])
            q = np.array([[0.], [1.]])
            blacks = Quantization.quantMatrix(self.imin, q, b)
            skeleton = Skeleton.Skeleton(blacks)

            skeleton.segments.addInitialStartPt()
            skeleton.euclidMstOrder()
            skeleton.segments.concatSegments()

            oneD = [item for sublist in skeleton.segments.segmentList for item in sublist]
            self.maze_path = np.reshape(oneD, (-1, 2))
            brownian = self.brownian()
            self.maze_path = np.add(self.maze_path, brownian)
            self.maze_path = TSPopt.simplify(self.maze_path)

            size = 60
            while True:
                delta, seg1 = TSPopt.threeOptLocal(self.maze_path, size)
                self.maze_path = seg1
                if delta == 0.:
                    break
                size = max(5,size-5)
            self.plotMazeImage("figStartSkeleton_%s.png" % self._run_tag,superimpose=True)


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
