__author__ = 'herman'

import math
import numpy
import sys


def _hyp(ax, ay, bx, by):
    xdiff = ax - bx
    ydiff = ay - by
    return math.hypot(xdiff, ydiff)

def _is_on(a, b, c, tol=1e-5):
    "Return true iff point c intersects the line segment from a to b."
    # (or the degenerate case that all 3 points are coincident)
    return (_collinear(a, b, c, tol)
            and (_within(a[0], c[0], b[0]) if a[0] != b[0] else
                 _within(a[1], c[1], b[1])))

def _collinear(a, b, c, tol=1e-5):
    "Return true iff a, b, and c all lie on the same line."
    return abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) < tol

def _within(p, q, r):
    "Return true iff q is between p and r (inclusive)."
    return p <= q <= r or r <= q <= p

def _simplifysegment(s):
    if len(s) < 3:
        return s
    new_s = []
    p0 = s[0]
    p1 = s[1]
    new_s.append(p0)
    for p2 in s[2:]:
        if _is_on(p0, p2, p1):
            p1 = p2
        else:
            new_s.append(p1)
            p0 = p1
            p1 = p2
    new_s.append(p2)
    return numpy.array(new_s)

class Segments:

    def __init__(self, max_depth=2**20):
        self.segmentList = []
        self.xmax = 0
        self.ymax = 0
        self.xmin = sys.maxsize
        self.ymin = sys.maxsize
        self.max_depth = max_depth
        self.numpts = 0

    def addInitialStartPt(self):
        self.segmentList.insert(0, [[(self.xmax + self.xmin) // 2, (self.ymax + self.ymax) // 2]])

    def append(self, segment):
        self.segmentList.append(segment)
        for pt in segment:
            self.xmax = max(pt[0], self.xmax)
            self.ymax = max(pt[1], self.ymax)
            self.xmin = min(pt[0], self.xmin)
            self.ymin = min(pt[1], self.ymin)
            self.numpts += 1

    def simplify(self):
        segmentSimp = []
        numpts = 0
        for s in self.segmentList:
            ns = _simplifysegment(s)
            segmentSimp.append(ns)
            numpts += len(ns)
        if numpts-1 >= self.max_depth:
            print "Number of points exceeds limit: "+repr(numpts)
            raise ValueError
        self.numpts = numpts
        self.segmentList = numpy.array(segmentSimp)

    def cArrayWrite(self, fname):
        f = open(fname, 'w')
        f.write("float diag["+repr(self.max_depth)+"][2] = {\n")
        i = 0
        for s in self.segmentSimp:
            for p in s:
                # x, y = self.pixelscale(p)
                f.write("       {"+repr(p[1])+", "+repr(p[0])+"},\n")
                i += 1
        for j in xrange(i, self.max_depth-1):
            f.write("       {NAN, NAN},\n")
        f.write("       {NAN, NAN}\n")
        f.write(" };\n")
        f.close()

    def binWrite(self, fname):
        from array import array
        output_file = open(fname, 'wb')
        l = [float('NaN')] * (2 * self.numpts + 20)
        i = 0
        for s in self.segmentList:
            for p in s:
                #x, y = self.pixelscale(p)
                l[i] = p[1]
                l[i+1] = p[0]
                i += 2

        float_array = array('f', l)
        float_array.tofile(output_file)
        output_file.close()

    def concatSegments(self):
        sL = []
        s_prev = self.segmentList[0]
        for s_new in self.segmentList[1:]:
            s_prev = numpy.append(s_prev, s_new, axis=0)
        self.numpts = len(s_prev)
        sL.append(s_prev)
        self.segmentList = sL

    def flipY(self):
        for i, seg in enumerate(self.segmentList):
            self.segmentList[i] = [[self.xmax-p[0]+self.xmin, p[1]] for p in seg]

    def flipX(self):
        for i, seg in enumerate(self.segmentList):
            self.segmentList[i] = [[p[0], self.ymax-p[1]+self.ymin] for p in seg]

    def scale(self, ratio):
        for i, seg in enumerate(self.segmentList):
            self.segmentList[i] = [[ratio*p[0],ratio*p[1]] for p in seg]
        self.xmax *= ratio
        self.ymax *= ratio
        self.xmin *= ratio
        self.ymin *= ratio

    def offset(self,(dx,dy)):
        for i, seg in enumerate(self.segmentList):
            self.segmentList[i] = [[dx+p[0],dy*p[1]] for p in seg]
        self.xmax += dx
        self.ymax += dy
        self.xmin += dx
        self.ymin += dy

    def scaleBin(self):
        self.offset((-self.xmin,-self.ymin))
        scale = 2.0/max(self.xmax,self.ymax)
        self.scale(scale)
        self.offset((-self.xmax/2,-self.ymax/2))

    def bresenhamFillIn(self, p0, p1, scale=1):
        """
        Bresenham's line algorithm
        """
        dx = scale*abs(p1[0] - p0[0])
        dy = scale*abs(p1[1] - p0[1])
        x = scale*p0[0]
        y = scale*p0[1]
        sx = -1 if p0[0] > p1[0] else 1
        sy = -1 if p0[1] > p1[1] else 1
        if dx > dy:
            err = dx / 2.0
            while x != scale*p1[0]:
                self.grad[x, y] = -1
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != scale*p1[1]:
                self.grad[x, y] = -1
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        self.grad[x, y] = -1

    def segment2grad(self, interior=False, scale=1, maxsegments=2**20):
        self.grad = numpy.zeros((scale*(self.xmax+1), scale*(self.ymax+1)), dtype=numpy.int)
        for s in self.segmentList:
            for p in s:
                self.grad[scale*p[0], scale*p[1]] = -1
        s0 = self.segmentList[0]
        segcount = 0
        for s1 in self.segmentList[1:]:
            self.bresenhamFillIn(s0[-1], s1[0], scale)
            s0 = s1
            segcount += 1
            if segcount == maxsegments:
                return

        if interior:
            for s0 in self.segmentList:
                p0 = s0[0]
                for p1 in s0[1:]:
                    self.bresenhamFillIn(p0, p1, scale)
                    p0 = p1
                    segcount += 1
                    if segcount == maxsegments:
                        return


def main_tsp(ifile_tsp, ifile_sol, bin_fn="bfile.bin"):
    s = Segments()
    f1 = open(ifile_tsp, 'r')
    tsp = {}
    for line in f1:
        col = line.split()
        if col[0] == "DIMENSION:":
            dim = int(col[1])
        elif col[0] == "NAME:" or col[0] == "COMMENT:" or \
            col[0] == "TYPE:" or col[0] == "EDGE_WEIGHT_TYPE:" or \
            col[0] == "NODE_COORD_SECTION" or col[0] == "EOF":
            pass
        else:
            assert len(col) == 3
            tsp[int(col[0])] = (int(col[1]), int(col[2]))
    f1.close()
    f2 = open(ifile_sol,'r')
    seg = []
    for line in f2:
        col = line.split()
        if col[0] == "DIMENSION:":
            assert dim == int(col[1])
        elif col[0] == "NAME:" or col[0] == "COMMENT:" or \
            col[0] == "TYPE:" or \
            col[0] == "TOUR_SECTION" or col[0] == "EOF":
            pass
        else:
            assert len(col) == 1
            pt = tsp[int(col[0])]
            x = int(pt[0])
            y = int(pt[1])
            # seg.append([pt[0],pt[1]])
            seg.append([y, x])

    s.append(seg)
    f2.close()

    s.flipY()
    # s.scale(0.5)

    s.binWrite(bin_fn)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main_tsp(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 3:
        main_tsp(sys.argv[1], sys.argv[2])
    else:
        print "Error: unknown usage"