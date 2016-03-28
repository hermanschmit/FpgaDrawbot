__author__ = 'herman'
from scipy.cluster.vq import kmeans, vq
from scipy import ndimage
from scipy import misc
import numpy as np
from numpy import reshape, uint8, ndarray
import math
import random


def botTransform(coords, m, offset, s):
    x = coords[0]
    y = coords[1]
    f = math.sqrt((m+x)*(m+x) + m + y*y)
    g = math.sqrt((m-s+x)*(m-s+x) + m + y*y)
    return f-offset, g-offset


def botTransformReverse(coords, m, offset, s):
    f = coords[0]+offset
    g = coords[1]+offset
    x = (f*f-g*g+s*s-2*m*s)/(2*s)
    mx2 = (m+x)*(m+x)
    if mx2+m > f*f:
        return float('nan'), float('nan')
    y = math.sqrt(f*f - mx2 - m)
    return x, y


def colinear(p0, p1, p2):
    x1, y1 = p1[0] - p0[0], p1[1] - p0[1]
    x2, y2 = p2[0] - p0[0], p2[1] - p0[1]
    return abs(x1 * y2 - x2 * y1) < 1e-5


def neighbor(p0, p1):
    x, y = abs(p1[0]-p0[0]), abs(p1[1]-p1[0])
    return x <= 1 and y <= 1


def simplifysegments(s):
    if len(s) < 3:
        return s
    new_s = []
    p0 = s[0]
    p1 = s[1]
    new_s.append(p0)
    for p2 in s[2:]:
        if colinear(p0, p1, p2):
            p1 = p2
        else:
            new_s.append(p1)
            p0 = p1
            p1 = p2
    new_s.append(p2)
    return new_s


class Sketchy:

    MAXDELTA = 512
    BASELINE = 1024
    KEEPWHITE = -50

    def score(self, pt):
        if self.drawn_mat[pt] == 0 and self.quant_idx[pt] == 0:
            return self.KEEPWHITE
        else:
            return self.quant_idx[pt] - self.drawn_mat[pt]

    def bresenhamScore(self, p0, p1, score=0, commit=False):
        """
        Bresenham's line algorithm
        """
        dx = abs(p1[0] - p0[0])
        dy = abs(p1[1] - p0[1])
        x, y = p0[0], p0[1]
        sx = -1 if p0[0] > p1[0] else 1
        sy = -1 if p0[1] > p1[1] else 1
        if dx > dy:
            err = dx / 2.0
            while x != p1[0]:
                score += self.score((x, y))
                if commit:
                    if self.drawn_mat[x, y] < self.levels-1:
                        self.drawn_mat[x, y] += 1
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != p1[1]:
                score += self.score((x, y))
                if commit:
                    if self.drawn_mat[x, y] < self.levels-1:
                        self.drawn_mat[x, y] += 1
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        if commit:
            if self.drawn_mat[x, y] < self.levels-1:
                self.drawn_mat[x, y] += 1
            self.segment.append((x, y))
        score += self.score((x, y))
        return score

    def measCentroid(self, mat, levels):
        pixel = reshape(mat, (mat.shape[0]*mat.shape[1], 1))
        centroids, _ = kmeans(pixel, levels)
        print(centroids)
        self.centroids = np.sort(centroids, axis=0)
        print((self.centroids))

    def quantMatrix(self, mat):
        pixel = reshape(mat, (mat.shape[0]*mat.shape[1], 1))
        qnt, _ = vq(pixel, self.centroids)
        self.quant_idx = reshape(qnt, (mat.shape[0], mat.shape[1]))
        self.quant_mat = self.centroids[self.quant_idx, 0]

    def untransform(self):
        rt = ndimage.interpolation.geometric_transform(self.drawn_mat,
                                                       botTransform,
                                                       output_shape=self.imat.shape,
                                                       extra_arguments=(self.m, self.m, self.BASELINE))
        self.drawn_mat = rt[:]

    def __init__(self, image_matrix, levels, scale=False, transform=False):
        self.moveEval = 200

        self.measCentroid(image_matrix, levels)
        self.levels = levels

        if scale:
            # resize the image
            alpha = self.MAXDELTA / (self.BASELINE * math.sqrt(2.0))
            width = image_matrix.shape[1]
            sc = self.BASELINE * alpha / width
            self.m = self.BASELINE*(1-alpha)/2
            self.imat = misc.imresize(image_matrix, sc)

            if transform:
                self.rot_mat = misc.imrotate(self.imat, 90)
                self.target_mat = ndimage.interpolation.geometric_transform(self.rot_mat,
                                                                            botTransformReverse,
                                                                            output_shape=(512, 512),
                                                                            extra_arguments=(self.m,
                                                                                             self.m,
                                                                                             self.BASELINE))
            else:
                self.target_mat = self.imat[:]
        else:
            self.target_mat = image_matrix[:]

        self.pen = tuple([z/2 for z in self.target_mat.shape])
        (self.x,self.y) = self.target_mat.shape

        self.segment = []
        self.segmentList = [self.segment]
        self.segment.append(self.pen)

        self.quantMatrix(self.target_mat)

        self.drawn_mat = ndarray(shape=self.target_mat.shape, dtype=uint8)
        self.drawn_mat.fill(0)

    def draw_line(self):
        best = self.pen
        for delta in (40,80,120,160):
            best_val = float("-inf")
            for e in range(self.moveEval):

                newpt = (self.pen[0] + random.randint(-delta, delta), self.pen[1] + random.randint(-delta, delta))
                newpt = (max(0, min(self.target_mat.shape[0]-1, newpt[0])),
                         max(0, min(self.target_mat.shape[1]-1, newpt[1])))
                s = self.bresenhamScore(self.pen, newpt)
                if s > best_val:
                    best = newpt
                    best_val = s
            if best_val >= 0.0:
                break
        self.bresenhamScore(self.pen, best, commit=True)
        self.pen = best
        # Evaluate Move

        # Draw line in drawn_mat

        # Evaluate Moves

    def pixelscale(self, pt, maxXY):
        px = float(pt[0]-self.x//2) / maxXY
        # py = -1.0 * float(pt[1]-self.y//2) / maxXY
        py = 1.0 * float(pt[1]-self.y//2) / maxXY
        return px, py

    def cArrayWrite(self, fname, depth=2**20):
        f = open(fname, 'w')
        numpts = 0
        segmentList_simp = []
        for s in self.segmentList:
            ns = simplifysegments(s)
            segmentList_simp.append(ns)
            numpts += len(ns)
        if numpts-1 >= depth:
            print(("Number of points exceeds limit: "+repr(numpts)))
            raise ValueError
        f.write("float diag["+repr(depth)+"][2] = {\n")
        m = max(self.x//2, self.y//2)
        i = 0
        for s in segmentList_simp:
            for p in s:
                x, y = self.pixelscale(p, m)
                f.write("       {"+repr(y)+", "+repr(x)+"},\n")
                i += 1
        for j in range(i, depth-1):
            f.write("       {NAN, NAN},\n")
        f.write("       {NAN, NAN}\n")
        f.write(" };\n")
        f.close()

    def binWrite(self, fname, depth=2**20):
        numpts = 0
        segmentList_simp = []
        for s in self.segmentList:
            ns = simplifysegments(s)
            segmentList_simp.append(ns)
            numpts += len(ns)
        if numpts-1 >= depth:
            print(("Number of points exceeds limit: "+repr(numpts)))
            raise ValueError
        m = max(self.x//2, self.y//2)
        i = 0
        from array import array
        output_file = open(fname, 'wb')
        l = [float('NaN')] * (2 * depth)

        for s in segmentList_simp:
            for p in s:
                x, y = self.pixelscale(p, m)
                l[i] = y
                l[i+1] = x
                i += 2

        float_array = array('f', l)
        float_array.tofile(output_file)
        output_file.close()
