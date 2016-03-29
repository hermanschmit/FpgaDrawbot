import Segments
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
from numba import jit
import concurrent.futures as cf
from scipy import spatial

@jit
def _ptlen_local(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

@jit
def _ptlen2_local(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2

@jit
def _LennardJones(r):
    force = (r ** 12 - r ** 6)
    return force

@jit
def _LennardJones2(R0,i_pt,pi2xij,xij,Fa):
    fij = (i_pt-xij)/ pi2xij
    fij *= _LennardJones(R0/pi2xij) * Fa
    return fij

@jit
def _repulse(r):
    force = (r ** 12)
    return force


class Maze:
    K0 = 0.1  # [0.1;0.3]
    K1 = 0.15  # [1.5*K0; 2.5*K0]
    D = 10  # dimensional adjustment?
    KMIN = 1.5
    KMAX = 3.5
    Ff = 0.1  # [0.005; 0.3]
    Fb = 0.05  # [0; 0.2]
    Fa = 1.  # [0; 10]
    Fo = 1.
    NMIN = 1  # [1,2]
    SIGMA_LJ = 1.  # Is this K0 * D == R0

    R0 = 6.
    R0_B = 8.

    PROCESSORS = 3
    CHUNK = 200

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
        d = self.seg.segmentList[0][i]
        v = self.imin[d[0], d[1]]
        assert v >= 0
        assert v < 256
        return float(v + 1) / 256.

    def brownian(self):
        mean = [0., 0.]
        cov = [[1., 0.], [0., 1.]]
        size = len(self.seg.segmentList[0])
        x, y = np.random.multivariate_normal(mean, cov, size).T
        z = list(zip(x, y))
        brown = []
        for i, zi in enumerate(z):
            n = np.array(zi)
            # n = np.multiply(n, self.Fb*self.delta(i)*self.D)
            n = np.multiply(n, self.Fb)
            brown.append(n)
        assert len(brown) == len(self.seg.segmentList[0])
        return np.array(brown)

    def faring(self):
        null = (0., 0.)
        fare = [null]  # initial element
        for i in range(1,
                        len(self.seg.segmentList[0]) - 1):
            pim1 = np.array(self.seg.segmentList[0][i - 1])
            pi = np.array(self.seg.segmentList[0][i])
            pip1 = np.array(self.seg.segmentList[0][i + 1])
            f = self.Ff * (np.multiply((pim1 + pip1), 0.5) - pi)
            # f = self.Ff * (((pim1*self.delta(i-1) + pip1*self.delta(i+1))/(self.delta(i-1)+self.delta(i+1))) - pi)
            fare.append(f)
        fare.append(null)  # final element
        assert len(fare) == len(self.seg.segmentList[0])
        return np.array(fare)

    def attract_repel_segment(self, s, chunk=self.CHUNK):
        R1 = 2.5 * self.R0
        local_min = sys.float_info.max
        fi_l = []
        for i in range(s, s + chunk):
            if i >= len(self.seg.segmentList[0]):
                continue
            fi = np.array([0., 0.])
            i_pt = self.seg.segmentList[0][i]
            neighbors = self.kdtree.query_ball_point(i_pt, R1)
            n_set = set(neighbors)
            for x in neighbors:
                n_set.add(x - 1)

            for j in n_set:
                if j < 0 or j == len(self.seg.segmentList[0]) - 1:
                    continue
                if j < i - 2 or j >= i + 2:
                    j_pt = self.seg.segmentList[0][j]
                    jp1_pt = self.seg.segmentList[0][j + 1]
                    pi2xij, xij = self.seg.distABtoP(j_pt, jp1_pt, i_pt)
                    local_min = min(local_min, pi2xij)
                    if pi2xij < R1:
                        fij = _LennardJones2(self.R0, i_pt, pi2xij, xij, self.Fa)
                        fi += fij
            fi_l.append(fi)

        return fi_l, local_min

    def attract_repel_serial(self):
        """
        This is the brute force version
        Returns:
        attract repel vector
        """
        self.kdtree = spatial.cKDTree(self.seg.segmentList[0])
        returnList = []
        for x in range(0,len(self.seg.segmentList[0])):
            fi_l, lm = self.attract_repel_segment(x,1)
            self.minDist = min(self.minDist,lm)
            returnList.extend(fi_l)
        return np.array(returnList)

    def attract_repel_parallel(self):
        """
        This is the parallel version, attempt 1
        Returns:
        attract repel vector
        """
        self.kdtree = spatial.cKDTree(self.seg.segmentList[0])

        with cf.ProcessPoolExecutor(self.PROCESSORS) as pool:
            x = pool.map(self.attract_repel_segment,range(0,len(self.seg.segmentList[0]),self.CHUNK))
        returnList = []
        for fi_l, lm in x:
            returnList.extend(fi_l)
            self.minDist=min(self.minDist,lm)
        return np.array(returnList)


    def boundary_slow(self):
        """
        This is the brute force version
        Returns:
        attract repel vector
        """
        returnList = []
        R1 = 3.0 * self.R0_B
        pt_00 = (self.seg.xmin - 0.5 * self.R0_B, self.seg.ymin - 0.5 * self.R0_B)
        pt_01 = (self.seg.xmin - 0.5 * self.R0_B, self.seg.ymax + 0.5 * self.R0_B)
        pt_11 = (self.seg.xmax + 0.5 * self.R0_B, self.seg.ymax + 0.5 * self.R0_B)
        pt_10 = (self.seg.xmax + 0.5 * self.R0_B, self.seg.ymin - 0.5 * self.R0_B)
        boundary_seg = [pt_00, pt_01, pt_11, pt_10, pt_00]

        for i in range(0,
                        len(self.seg.segmentList[0])):
            fi = np.array([0., 0.])
            # delta_pi = self.delta(i)
            pi = np.array(self.seg.segmentList[0][i])
            for j in range(0,
                            len(boundary_seg) - 1):
                j_pt = boundary_seg[j]
                jp1_pt = boundary_seg[j + 1]
                pi2xij, xij = self.seg.distABtoP(j_pt, jp1_pt, pi)
                self.minDist = min(self.minDist,pi2xij)
                if pi2xij < R1:
                    fij = (pi - xij)/ max(0.00001, pi2xij)
                    fij *= _repulse(self.R0_B / pi2xij) * self.Fo
                    fi += fij
            returnList.append(fi)
        assert len(returnList) == len(self.seg.segmentList[0])
        return np.array(returnList)

    def resampling(self):
        tmp3 = []
        ptA = self.seg.segmentList[0][0]
        tmp3.append(ptA)
        for ptB in self.seg.segmentList[0][1:]:
            d = _ptlen_local(ptA, ptB)
            if d > self.KMAX:
                ptAB = np.multiply(np.add(ptA, ptB), 0.5)
                tmp3.append(ptAB)
                tmp3.append(ptB)
            elif d < self.KMIN:
                pass
            else:
                tmp3.append(ptB)
            ptA = ptB

        self.seg.segmentList[0] = tmp3


    def optimize_loop1(self,loop_bound):
        # Main optimize loop
        # keep running until stopping criteria met
        loop_count = 0
        while True:

            # BIG
            self.minDist = sys.float_info.max
            # compute force on each node
            brownian = self.brownian()
            if len(self.seg.segmentList[0]) < 200*self.PROCESSORS:
                attract_repel = self.attract_repel_serial()
            else:
                attract_repel = self.attract_repel_parallel()
            fairing = self.faring()
            boundary = self.boundary_slow()

            # move each node
            netforce = np.add(boundary,np.add(fairing,attract_repel))
            #netforce = np.add(boundary,np.add(fairing,np.add(brownian, attract_repel)))
            #netforce = np.add(fairing,np.add(brownian, attract_repel))
            deltaforce = [np.hypot(a[0], a[1]) for a in netforce]
            maxdelta = np.max(deltaforce)
            assert self.minDist / maxdelta > 0.0
            if maxdelta > 1.*self.minDist:
                netforce = np.multiply(netforce, 1. * self.minDist / maxdelta)
            netmove = np.multiply(netforce, 1.)

            # don't scale noise by size maxdelta

            netmove = np.add(netmove,brownian)

            tmp1 = np.add(self.seg.segmentList[0], netmove)
            tmp2 = tmp1[:]
            tmp2[0] = self.seg.segmentList[0][0]
            tmp2[-1] = self.seg.segmentList[0][-1]
            self.seg.segmentList[0] = tmp2

            # resampling
            self.resampling()

            # stopping criteria
            if loop_count > loop_bound:
                break
            loop_count += 1
            if loop_count % 100 == 0:
                print(str(loop_count)+" "+str(len(self.seg.segmentList[0])))
            if loop_count % 200 == 1:
                plt_x = [a[0] for a in self.seg.segmentList[0]]
                plt_y = [a[1] for a in self.seg.segmentList[0]]
                plt.plot(plt_x, plt_y, '.-')
                plt.savefig("fig"+str(loop_count)+".png")

        plt_x = [a[0] for a in self.seg.segmentList[0]]
        plt_y = [a[1] for a in self.seg.segmentList[0]]
        plt.plot(plt_x, plt_y, '.-')
        plt.savefig("fig"+str(loop_count)+".png")

    def __init__(self, image_matrix, white=1):
        """
        :param image_matrix:
        """

        self.imin = image_matrix
        # whiten
        self.imin /= white
        self.imin += 255 - (255 // white)

        # TODO: do we want to quantize the image?

        # Initial segment
        segList1 = [(0., 0.)]
        segListEnd = tuple([x - 1 for x in self.imin.shape])
        segList1.append(segListEnd)
        self.seg = Segments.Segments()
        self.seg.append(segList1)
        self.minDist = sys.float_info.max

        # self.seg.scale(1.0) # fix the types. Hygiene

