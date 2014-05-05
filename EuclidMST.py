import scipy
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import depth_first_order

import numpy
import math


def hyp(ax,ay,bx,by):
    return math.hypot(float(ax - bx), float(ay - by))

class EuclidMST:

    def minDist(self,i):
        x0 = self.distMatrix[i[0],:]
        x1 = self.distMatrix[i[1],:]
        m = 1e10
        d0 = x0.nonzero()
        d1 = x1.nonzero()
        for k in d0[1]:
            m = min(m,x0[0,k])
        for k in d1[1]:
            m = min(m,x1[0,k])
        return m

    def treetrav(self,tree):
        self.nodeTrav.append(tree[0])
        for st in tree[2]:
            self.treetrav(st)
            self.nodeTrav.append(tree[0])

    def dfo(self,node,parent):
        children = self.spnTree[node, :]
        child_idx = children.nonzero()
        sum = 0
        l = []
        for c in child_idx[1]:
            if c == node or c == parent:
                continue
            (c_ret, total, child_l) = self.dfo(c,node)
            l.append((c_ret, total, child_l))
            sum += total + children[0, c]
        ls = sorted(l, key=lambda dist: dist[1])
        return (node, sum, ls)


    def __init__(self,segmentList):

        pl = []
        idxl = []
        lidx = []
        i = 0
        j = 0
        for s in segmentList:
            pl.append(s[0])
            lidx.append((j, True))
            i += 1
            if len(s) > 1:
                pl.append(s[-1])
                idxl.append([i-1, i])
                lidx.append((j, False))
                i += 1
            j += 1

        points = numpy.array(pl,float)
        self.tri = scipy.spatial.Delaunay(points)
        self.size = len(self.tri.points)
        self.distMatrix = csr_matrix((self.size, self.size),
                                     dtype=float )
        for smplx in self.tri.simplices:
            p0 = self.tri.points[smplx[0]]
            p1 = self.tri.points[smplx[1]]
            p2 = self.tri.points[smplx[2]]

            eA = hyp(p0[0],p0[1],p1[0],p1[1])
            eB = hyp(p1[0],p1[1],p2[0],p2[1])
            eC = hyp(p2[0],p2[1],p0[0],p0[1])

            self.distMatrix[smplx[0], smplx[1]] = eA
            self.distMatrix[smplx[1], smplx[0]] = eA
            self.distMatrix[smplx[1], smplx[2]] = eB
            self.distMatrix[smplx[2], smplx[1]] = eB
            self.distMatrix[smplx[2], smplx[0]] = eC
            self.distMatrix[smplx[0], smplx[2]] = eC

        # add the segment list with small delta
        # make sure no points from triangulation connecting real points
        # they will be restored
        for i in idxl:
            self.distMatrix[i[0], i[1]] = 1e-10
            self.distMatrix[i[1], i[0]] = 1e-10

        # TODO use triangulation to remove lonely segments
        # This attempts to remove points based on distance in the triangulation.
        # Incomplete because it doesn't remove the points, and that seems scary.
        # for i in idxl:
        #     self.distMatrix[i[0], i[1]] = 0
        #     self.distMatrix[i[1], i[0]] = 0
        #
        # survivors = [False] * len(segmentList)
        # survivors[0] = True
        #
        # for i in idxl:
        #     m1 = self.minDist(i)
        #     s1 = segmentList[lidx[i[0]][0]]
        #     limit = 100 * max(hyp(*(s1[0] + s1[-1])),len(s1))
        #     if m1 < limit:
        #         self.distMatrix[i[0], i[1]] = 1e-10
        #         self.distMatrix[i[1], i[0]] = 1e-10
        #     else:
        #         pass
        #         # TODO remove points!


        self.spnTree = minimum_spanning_tree(self.distMatrix)
        (aidx, bidx) = self.spnTree.nonzero()
        # make undirected
        for (a,b) in zip(aidx,bidx):
            self.spnTree[bidx, aidx] = self.spnTree[aidx, bidx]

        traversal = self.dfo(0,None)
        self.nodeTrav = []
        self.treetrav(traversal)

        self.newSegmentTree = []

        # TODO This does not have to be complete traversal. Reduce to cover all vertices
        for z in xrange(0,len(self.nodeTrav)-1):
            s0 = lidx[self.nodeTrav[z]]
            s1 = lidx[self.nodeTrav[z+1]]
            if s0[0] == s1[0]:
                if s0[1]:
                    self.newSegmentTree.append(segmentList[s0[0]])
                else:
                    self.newSegmentTree.append(numpy.flipud(segmentList[s0[0]]))


        # import matplotlib.pyplot as plt
        #
        # for (a,b) in zip(aidx,bidx):
        #     linex = [points[a,0],points[b,0]]
        #     liney = [points[a,1],points[b,1]]
        #     plt.plot(liney, linex, 'o-')
        # ax = plt.gca()
        # ax.invert_yaxis()
        # plt.show()
        #
        # print self.spnTree.nonzero()
        # print lidx











