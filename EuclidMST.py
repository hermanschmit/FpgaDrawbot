import scipy
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix, lil_matrix
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
        x0rd = x0.data
        x0rd.sort()
        for d in x0rd:
            if d > 1e-10: return d

    def treetrav_nonrec(self,tree):
        stack = []
        stack.append(0)
        self.nodeTrav = []
        while len(stack) > 0:
            self.nodeTrav.append(stack[-1])
            if len(tree[stack[-1]]) > 0:
                (curr, d) = tree[stack[-1]].pop()
                stack.append(curr)
            else:
                stack.pop()

    def dfo_nonrec(self, node):
        (narray, pred) = scipy.sparse.csgraph.depth_first_order(self.spnTree,node, False, True)
        childCount = numpy.zeros(narray.shape)
        tree = []
        for i in narray[::-1]:
            tree.append([])
            if i == node: continue
            childCount[pred[i]] += (self.distMatrix[pred[i],i] + childCount[i])
        for i in xrange(0,len(narray)):
            if pred[i] == -9999: continue
            tree[pred[i]].append((i, childCount[i]))
        for l in tree:
            l.sort(key=lambda x: x[1], reverse=True)
        return tree


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
        lilmatrix = lil_matrix((self.size, self.size), dtype=float)

        for smplx in self.tri.simplices:
            p0 = self.tri.points[smplx[0]]
            p1 = self.tri.points[smplx[1]]
            p2 = self.tri.points[smplx[2]]

            eA = hyp(p0[0],p0[1],p1[0],p1[1])
            eB = hyp(p1[0],p1[1],p2[0],p2[1])
            eC = hyp(p2[0],p2[1],p0[0],p0[1])

            lilmatrix[smplx[0], smplx[1]] = eA
            lilmatrix[smplx[1], smplx[0]] = eA
            lilmatrix[smplx[1], smplx[2]] = eB
            lilmatrix[smplx[2], smplx[1]] = eB
            lilmatrix[smplx[2], smplx[0]] = eC
            lilmatrix[smplx[0], smplx[2]] = eC

        # add the segment list with small delta
        # make sure no points from triangulation connecting real points
        # they will be restored
        for i in self.idxl:
            lilmatrix[i[0], i[1]] = 1e-10
            lilmatrix[i[1], i[0]] = 1e-10

        self.distMatrix = lilmatrix.tocsr()

        print "start MST"

        self.spnTree = minimum_spanning_tree(self.distMatrix)
        print "done MST"
        (aidx, bidx) = self.spnTree.nonzero()
        # make undirected
        for (a,b) in zip(aidx,bidx):
            self.spnTree[b, a] = self.spnTree[a, b]


    def lonelySegmentRemoval(self, firstPreserved = True, factor = 40.):
        # This attempts to remove points based on distance in the triangulation.
        # Incomplete because it doesn't remove the points, and that seems scary.

        survivors = [False] * len(self.segmentList)
        survivors[0] = firstPreserved

        #for i in self.idxl:
        #     self.distMatrix[i[0], i[1]] = 0
        #     self.distMatrix[i[1], i[0]] = 0

        if firstPreserved:
            lidx2 = self.lidx[1:]
            idx = 1
        else:
            lidx2 = self.lidx[:]
            idx = 0

        for i in lidx2:
            m1 = self.minDist(idx)
            s1 = self.segmentList[i[0]]
            limit = factor * max(hyp(*(s1[0] + s1[-1])),len(s1))
            if m1 < limit:
                 survivors[i[0]] = True
            idx += 1
        self.newSegmentTree = []
        for (seg,surv) in zip(self.segmentList,survivors):
            if surv:
                self.newSegmentTree.append(seg)



    def segmentOrdering(self):
        #traversal = self.dfo(0,None)
        tree = self.dfo_nonrec(0)
        print "Done dfo"
        self.nodeTrav = []
        #self.treetrav(traversal)
        self.treetrav_nonrec(tree)
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














