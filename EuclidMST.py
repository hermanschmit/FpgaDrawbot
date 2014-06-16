import scipy
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import depth_first_order

import numpy
import math


def hyp(ax,ay,bx,by):
    # This is sloppy, but faster
    return float(ax-bx)**2 + float(ay - by)**2
    #return math.hypot(float(ax - bx), float(ay - by))

class EuclidMST:

    def minDist(self,i):
        x0 = self.distMatrix.getrow(i)
        return x0.data.min()
        # x0 = self.distMatrix[i,:]
        # m = 1e10
        # d0 = x0.nonzero()
        # for k in d0[1]:
        #     m = min(m,x0[0,k])
        # return m

    def treetrav(self,tree):
        self.nodeTrav.append(tree[0])
        for st in tree[2]:
            self.treetrav(st)
            self.nodeTrav.append(tree[0])

    def dfo(self,node,parent):
        children = self.spnTree[node, :]
        child_idx = children.nonzero()
        sumtotal = 0
        l = []
        for c in child_idx[1]:
            if c == node or c == parent:
                continue
            (c_ret, total, child_l) = self.dfo(c,node)
            l.append((c_ret, total, child_l))
            sumtotal += total + children[0, c]
        ls = sorted(l, key=lambda dist: dist[1])
        return (node, sumtotal, ls)

    def __init__(self,segmentList):
        self.segmentList = segmentList
        pl = []
        self.idxl = []
        self.lidx = []
        i = 0
        j = 0
        for s in segmentList:
            pl.append(s[0])
            self.lidx.append((j, True))
            i += 1
            if len(s) > 1:
                pl.append(s[-1])
                self.idxl.append([i-1, i])
                self.lidx.append((j, False))
                i += 1
            j += 1

        points = numpy.array(pl,float)
        self.tri = scipy.spatial.Delaunay(points)
        print "Delaunay Done"
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
        for i in self.idxl:
            self.distMatrix[i[0], i[1]] = 1e-10
            self.distMatrix[i[1], i[0]] = 1e-10



        print "start MST"

        self.spnTree = minimum_spanning_tree(self.distMatrix)
        print "done MST"
        (aidx, bidx) = self.spnTree.nonzero()
        # make undirected
        for (a,b) in zip(aidx,bidx):
            self.spnTree[b, a] = self.spnTree[a, b]


    def lonelySegmentRemoval(self, firstPreserved = True):
        # This attempts to remove points based on distance in the triangulation.
        # Incomplete because it doesn't remove the points, and that seems scary.

        survivors = [False] * len(self.segmentList)
        survivors[0] = firstPreserved

        for i in self.idxl:
             self.distMatrix[i[0], i[1]] = 0
             self.distMatrix[i[1], i[0]] = 0

        if firstPreserved:
            lidx2 = self.lidx[1:]
            idx = 1
        else:
            lidx2 = self.lidx
            idx = 0

        for i in lidx2:
            m1 = self.minDist(idx)
            s1 = self.segmentList[i[0]]
            limit = 40 * max(hyp(*(s1[0] + s1[-1])),len(s1))
            if m1 < limit:
                 survivors[i[0]] = True
            idx += 1
        self.newSegmentTree = []
        for (seg,surv) in zip(self.segmentList,survivors):
            if surv:
                self.newSegmentTree.append(seg)



    def segmentOrdering(self):
        traversal = self.dfo(0,None)
        print "Done dfo"
        self.nodeTrav = []
        self.treetrav(traversal)
        print "treetrav done"

        covered = [False] * len(self.segmentList)
        covered[0] = True
        uncovered_count = len(self.segmentList)-1

        self.newSegmentTree = []

        for z in xrange(0,len(self.nodeTrav)-1):
            s0 = self.lidx[self.nodeTrav[z]]
            s1 = self.lidx[self.nodeTrav[z+1]]
            if s0[0] == s1[0]:
                if s0[1]:
                    self.newSegmentTree.append(self.segmentList[s0[0]])
                else:
                    self.newSegmentTree.append(numpy.flipud(self.segmentList[s0[0]]))
                if not covered[s0[0]]:
                    uncovered_count -= 1
                    covered[s0[0]] = True
            if uncovered_count <= 0:
                break














