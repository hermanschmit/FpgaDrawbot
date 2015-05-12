__author__ = 'herman'

import math
import numpy
import sys


def _hyp(ax, ay, bx, by):
    xdiff = ax - bx
    ydiff = ay - by
    return math.hypot(xdiff,ydiff)

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


    def __init__(self,max_depth=2**20):
        self.segmentList = []
        self.xmax = 0
        self.ymax = 0
        self.xmin = sys.maxsize
        self.ymin = sys.maxsize
        self.max_depth = max_depth

    def addInitialStartPt(self,shape):
        if shape[0] < self.xmax:
            print "Uhoh"
        else:
            self.xmax = shape[0]

        if shape[1] < self.ymax:
            print "Uhoh"
        else:
            self.ymax = shape[1]

        self.segmentList.insert(0, [[self.xmax // 2, self.ymax // 2]])


    def append(self,segment):
        self.segmentList.append(segment)
        # update xmax, ymax
        for pt in segment:
            self.xmax = max(pt[0],self.xmax)
            self.ymax = max(pt[1],self.ymax)
            self.xmin = min(pt[0],self.xmin)
            self.ymin = min(pt[1],self.ymin)

    def cArrayWrite(self, fname):
        f = open(fname,'w')
        numpts = 0
        segmentList_simp = []
        for s in self.segmentList:
            ns = _simplifysegment(s)
            segmentList_simp.append(ns)
            numpts += len(ns)
        if numpts-1 >= self.max_depth:
            print "Number of points exceeds limit: "+repr(numpts)
            raise ValueError
        f.write("float diag["+repr(self.max_depth)+"][2] = {\n")
        m = max(self.xmax//2, self.ymax//2)
        i = 0
        for s in segmentList_simp:
            for p in s:
                x, y = self.pixelscale(p)
                f.write("       {"+repr(y)+", "+repr(x)+"},\n")
                i += 1
        for j in xrange(i, self.max_depth-1):
            f.write("       {NAN, NAN},\n")
        f.write("       {NAN, NAN}\n")
        f.write(" };\n")
        f.close()

    def binWrite(self, fname):
        numpts = 0
        segmentList_simp = []
        for s in self.segmentList:
            ns = _simplifysegment(s)
            segmentList_simp.append(ns)
            numpts += len(ns)
        if numpts-1 >= self.max_depth:
            print "Number of points exceeds limit: "+repr(numpts)
            raise ValueError
        m = max(self.xmax//2,self.ymax//2)
        i = 0
        from array import array
        output_file = open(fname, 'wb')
        l = [float('NaN')] * (2 * self.max_depth)

        for s in segmentList_simp:
            for p in s:
                x, y = self.pixelscale(p)
                l[i] = y
                l[i+1] = x
                i += 2

        float_array = array('f', l)
        float_array.tofile(output_file)
        output_file.close()

    def concatSegments(self):
        sL = []
        s_prev = self.segmentList[0]
        for s_new in self.segmentList[1:]:
            s_prev = numpy.append(s_prev,s_new,axis=0)
        sL.append(s_prev)
        self.segmentList = sL

    def pixelscale(self,pt):
        maxXY = max(self.xmax,self.ymax)//2
        px = float(pt[0]-self.xmax//2) / maxXY
        py = 1.0 * float(pt[1]-self.ymax//2) / maxXY
        return px, py

    def flipY(self):
        for i, seg in enumerate(self.segmentList):
            self.segmentList[i] = [[self.xmax-p[0], p[1]] for p in seg]

    def flipX(self):
        for i, seg in enumerate(self.segmentList):
            self.segmentList[i] = [[p[0], self.ymax-p[1]] for p in seg]

    def scale_point(self,pt,ratio):
        x = int((pt[0] - (self.xmax-self.xmin)//2)*ratio)+(self.xmax-self.xmin)//2
        y = int((pt[1] - (self.ymax-self.ymin)//2)*ratio)+(self.ymax-self.ymin)//2
        return [x,y]

    def scale(self,ratio):
        for i, seg in enumerate(self.segmentList):
            self.segmentList[i] = [self.scale_point(p,ratio) for p in seg]

    def pruneLonelySegments(self,ratio=100.):
        newSegmentList = list()
        newSegmentList.append(self.segmentList[0])

        if len(self.segmentList) < 3: return
        s0 = self.segmentList[0]
        s1 = self.segmentList[1]
        for i in xrange(2, len(self.segmentList)):
            s2 = self.segmentList[i]
            drawnA = _hyp(*(s0[-1] + s1[0]))
            drawnB = _hyp(*(s1[-1] + s2[0]))
            segment = max(_hyp(*(s1[0] + s1[-1])),len(s1))
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



def main_tsp(ifile_tsp, ifile_sol, bin_fn="bfile.bin"):
    s = Segments()
    f1 = open(ifile_tsp,'r')
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
            tsp[int(col[0])] = (int(col[1]),int(col[2]))
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
            seg.append([y,x])

    s.append(seg)
    f2.close()

    s.flipY()
    s.scale(0.5)

    s.binWrite(bin_fn)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main_tsp(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 3:
        main_tsp(sys.argv[1], sys.argv[2])
    else:
        print "Error: unknown usage"