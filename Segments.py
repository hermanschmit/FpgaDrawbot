__author__ = 'herman'

import math
import numpy
import sys


def _hyp(ax, ay, bx, by):
    xdiff = ax - bx
    ydiff = ay - by
    return math.hypot(xdiff, ydiff)

def _colinear(p0, p1, p2):
    x1, y1 = p1[0] - p0[0], p1[1] - p0[1]
    x2, y2 = p2[0] - p0[0], p2[1] - p0[1]
    return abs(x1 * y2 - x2 * y1) < 1e-5

def _simplifysegment(s):
    if len(s) < 3:
        return s
    new_s = []
    p0 = s[0]
    p1 = s[1]
    new_s.append(p0)
    for p2 in s[2:]:
        if _colinear(p0, p1, p2):
            p1 = p2
        else:
            new_s.append(p1)
            p0 = p1
            p1 = p2
    new_s.append(p2)
    return new_s

class Segments:

    def __init__(self, max_depth=2**20):
        self.segmentList = []
        self.xmax = 0
        self.ymax = 0
        self.xmin = sys.maxsize
        self.ymin = sys.maxsize
        self.max_depth = max_depth

    def addInitialStartPt(self, shape):
        if shape[0] < self.xmax:
            print "Uhoh"
        else:
            self.xmax = shape[0]

        if shape[1] < self.ymax:
            print "Uhoh"
        else:
            self.ymax = shape[1]

        self.segmentList.insert(0, [[self.xmax // 2, self.ymax // 2]])

    def append(self, segment):
        self.segmentList.append(segment)
        # update xmax, ymax
        for pt in segment:
            self.xmax = max(pt[0], self.xmax)
            self.ymax = max(pt[1], self.ymax)
            self.xmin = min(pt[0], self.xmin)
            self.ymin = min(pt[1], self.ymin)

    def simplifyScale(self):
        numpts = 0
        self.segmentSimp = []
        for s in self.segmentList:
            ns = _simplifysegment(s)
            nss = [self.pixelscale(p) for p in ns]
            self.segmentSimp.append(nss)
            numpts += len(ns)
        self.numpts = numpts
        if numpts-1 >= self.max_depth:
            print "Number of points exceeds limit: "+repr(numpts)
            raise ValueError

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
        for s in self.segmentSimp:
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
        sL.append(s_prev)
        self.segmentList = sL

    def pixelscale(self, pt):
        maxXY = max(self.xmax, self.ymax)
        px = 2.0*pt[0]/maxXY - 1.0
        py = 2.0*pt[1]/maxXY - 1.0
        return px, py

    def flipY(self):
        for i, seg in enumerate(self.segmentList):
            self.segmentList[i] = [[self.xmax-p[0], p[1]] for p in seg]

    def flipX(self):
        for i, seg in enumerate(self.segmentList):
            self.segmentList[i] = [[p[0], self.ymax-p[1]] for p in seg]

    def scale_point(self, pt, ratio):
        x = int((pt[0] - (self.xmax-self.xmin)//2)*ratio)+(self.xmax-self.xmin)//2
        y = int((pt[1] - (self.ymax-self.ymin)//2)*ratio)+(self.ymax-self.ymin)//2
        return [x, y]

    def scale(self, ratio):
        for i, seg in enumerate(self.segmentList):
            self.segmentList[i] = [self.scale_point(p, ratio) for p in seg]

    def pruneLonelySegments(self, ratio=100.):
        newSegmentList = list()
        newSegmentList.append(self.segmentList[0])

        if len(self.segmentList) < 3:
            return
        s0 = self.segmentList[0]
        s1 = self.segmentList[1]
        for i in xrange(2, len(self.segmentList)):
            s2 = self.segmentList[i]
            drawnA = _hyp(*(s0[-1] + s1[0]))
            drawnB = _hyp(*(s1[-1] + s2[0]))
            segment = max(_hyp(*(s1[0] + s1[-1])), len(s1))
            limit = segment * ratio
            if drawnA < limit or drawnB < limit:
                newSegmentList.append(s1)
                s0 = s1
                s1 = s2
            else:
                s1 = s2

        # add the last?
        last_2 = self.segmentList[-2]
        last = self.segmentList[-1]
        drawn = _hyp(*(last_2[-1] + last[0]))
        limit = ratio * _hyp(*(last[0] + last[-1]))
        if drawn < limit:
            newSegmentList.append(self.segmentList[-1])
        self.segmentList = newSegmentList[:]

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