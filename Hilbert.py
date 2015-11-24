"""
Hilbert based imaging
"""

import Queue as Q

import numpy
from scipy import *
from scipy.cluster.vq import kmeans, vq

import Segments


def xy2d(n, x, y, moore=False):
    d = 0
    s = n // 2
    while s > 0:
        rx = (x & s) > 0
        ry = (y & s) > 0
        d += s * s * ((3 * rx) ^ ry)
        if moore and s * 2 >= n:
            x, y = rot_mooreI(s, x, y, rx, ry)
        else:
            x, y = rot(s, x, y, rx, ry)
        s //= 2
    return d


def rot(n, x, y, rx, ry):
    """
    rotate/flip a quadrant appropriately
    """
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        return y, x
    return x, y


def rot_moore(n, x, y, rx, ry):
    """
    rotate/flip a quadrant appropriately

    """
    if rx == 0:
        return n - 1 - y, x
    else:
        return y, n - 1 - x


def rot_mooreI(n, x, y, rx, ry):
    if rx == 0:
        return y, n - 1 - x
    else:
        return n - 1 - y, x


def ptlen(a, b):
    return hypot(a[0] - b[0], a[1] - b[1])


def almost_equal(a, b):
    return abs(a - b) < 0.00001


class Hilbert:
    """
        Create instances of this class to apply the Canny edge
        detection algorithm to an image.

        input: imagename(string),sigma for gaussian blur
        optional args: thresHigh,thresLow

        output: numpy ndarray.


    """

    def segment2grad(self, interior=False, scale=1, maxsegments=2 ** 20):
        self.segments.segment2grad(interior, scale, maxsegments)

    def stipple(self, stride=1):
        # probably want to smooth imin
        self.stipple_im = self.imin[:]

        A, B, G, S = [7.0 / 16.0, 3.0 / 16.0, 5.0 / 16.0, 1.0 / 16.0]
        (xdim, ydim) = self.imin.shape

        for y in xrange(0, ydim - 2 - stride, stride):
            for x in xrange(0, xdim - 2 - stride, stride):
                oldpixel = self.stipple_im[x, y]
                if oldpixel > 128:
                    newpixel = 255
                else:
                    newpixel = 0

                self.stipple_im[x, y] = float(newpixel)
                quant_error = float(oldpixel - newpixel)
                if x < xdim - 2 - 1:
                    self.stipple_im[x + stride, y] += (A * quant_error)
                if (x > 0) and (y < ydim - 2 - 1):
                    self.stipple_im[x - stride, y + stride] += (B * quant_error)
                if y < ydim - 2 - 1:
                    self.stipple_im[x, y + stride] += (G * quant_error)
                if (x < xdim - 2 - 1) and (y < ydim - 2 - 1):
                    self.stipple_im[x + stride, y + stride] += (S * quant_error)

    @staticmethod
    def power2(x):
        y = 2 ** ceil(log2(x))
        return y

    def hilbertSequence(self):
        q = Q.PriorityQueue()

        (xL, yL) = self.power2(shape(self.stipple_im))
        dim = max(xL, yL)
        scaleX, scaleY = tuple([dim / z for z in shape(self.stipple_im)])

        xS, yS = where(self.stipple_im == 0)

        for x, y in zip(xS, yS):
            xS2 = int(x * scaleX)
            yS2 = int(y * scaleY)
            d = xy2d(int(dim), xS2, yS2, moore=True)
            q.put((d, (x, y)))
        # Cut out first quarter of the image to append to end
        seg1 = []
        seg234 = []
        limit = int(dim ** 2) / 4
        while not q.empty():
            p = q.get()
            pt = p[1]
            if p[0] <= limit:
                seg1.append([pt[0], pt[1]])
            else:
                seg234.append([pt[0], pt[1]])
        seg234.extend(seg1)
        return seg234

    def twoOpt(self, seg, maxdelta=100):
        totald = 0
        for a in xrange(len(seg) - 4):
            a_pt = seg[a]
            b_pt = seg[a + 1]
            ab_len = ptlen(a_pt, b_pt)
            for c in xrange(a + 2, min(a + maxdelta, len(seg) - 2)):
                c_pt = seg[c]
                d_pt = seg[c + 1]
                cd_len = ptlen(c_pt, d_pt)
                delta = ptlen(a_pt, c_pt) + ptlen(b_pt, d_pt) - ab_len - cd_len
                if delta < -0.01:
                    aseg = seg[:a + 1]
                    bcseg = seg[a + 1:c + 1]
                    bcseg.reverse()
                    dseg = seg[c + 1:]
                    seg = aseg
                    seg.extend(bcseg)
                    seg.extend(dseg)

                    a_pt = seg[a]
                    b_pt = seg[a + 1]
                    ab_len = ptlen(a_pt, b_pt)
                    totald += delta
        return totald, seg

    def threeOpt(self, seg, maxdelta=10):
        totald = 0
        for a in xrange(len(seg) - 6):
            a_pt = seg[a]
            b_pt = seg[a + 1]
            ab_len = ptlen(a_pt, b_pt)
            for c in xrange(a + 2, min(a + maxdelta, len(seg) - 4)):
                c_pt = seg[c]
                d_pt = seg[c + 1]
                cd_len = ptlen(c_pt, d_pt)
                for e in xrange(c + 2, min(c + maxdelta, len(seg) - 2)):
                    e_pt = seg[e]
                    f_pt = seg[e + 1]
                    ef_len = ptlen(e_pt, f_pt)
                    orig = ab_len + cd_len + ef_len
                    ac_len = ptlen(a_pt, c_pt)
                    ad_len = ptlen(a_pt, d_pt)
                    ae_len = ptlen(a_pt, e_pt)
                    bd_len = ptlen(b_pt, d_pt)
                    be_len = ptlen(b_pt, e_pt)
                    bf_len = ptlen(b_pt, f_pt)
                    ce_len = ptlen(c_pt, e_pt)
                    cf_len = ptlen(c_pt, f_pt)
                    df_len = ptlen(d_pt, f_pt)

                    acbedf = ac_len + be_len + df_len
                    adebcf = ad_len + be_len + cf_len
                    adecbf = ad_len + ce_len + bf_len
                    aedbcf = ae_len + bd_len + cf_len

                    new = min(acbedf, adebcf, adecbf, aedbcf)
                    if new - orig < -0.01:
                        aseg = seg[:a + 1]
                        bcseg = seg[a + 1:c + 1]
                        deseg = seg[c + 1:e + 1]
                        fseg = seg[e + 1:]
                        if acbedf == new:
                            bcseg.reverse()
                            deseg.reverse()
                            seg = aseg
                            seg.extend(bcseg)
                            seg.extend(deseg)
                            seg.extend(fseg)
                        elif adebcf == new:
                            seg = aseg
                            seg.extend(deseg)
                            seg.extend(bcseg)
                            seg.extend(fseg)
                        elif adecbf == new:
                            seg = aseg
                            seg.extend(deseg)
                            bcseg.reverse()
                            seg.extend(bcseg)
                            seg.extend(fseg)
                        else:
                            seg = aseg
                            deseg.reverse()
                            seg.extend(deseg)
                            seg.extend(bcseg)
                            seg.extend(fseg)

                        a_pt = seg[a]
                        b_pt = seg[a + 1]
                        ab_len = ptlen(a_pt, b_pt)
                        c_pt = seg[c]
                        d_pt = seg[c + 1]
                        cd_len = ptlen(c_pt, d_pt)

                        totald += (new - orig)
        return totald, seg

    def totalLength(self, seg):
        total = 0.
        for a in xrange(len(seg) - 1):
            total += ptlen(seg[a], seg[a + 1])
        return total

    def measCentroid(self, mat, levels):
        pixel = reshape(mat, (mat.shape[0] * mat.shape[1], 1))
        centroids, _ = kmeans(pixel, levels)
        self.centroids = sort(centroids, axis=0)
        print self.centroids

    def quantMatrix(self, mat, augmentWhite=True):
        pixel = reshape(mat, (mat.shape[0] * mat.shape[1], 1))
        qnt, _ = vq(pixel, self.centroids)
        self.quant_idx = reshape(qnt, (mat.shape[0], mat.shape[1]))
        self.imin = self.centroids[self.quant_idx, 0]
        if augmentWhite:
            x, y = where(self.imin == self.centroids[-1])
            self.imin[x, y] = 255

    def __init__(self, image_matrix, white=1, levels=4):
        """

        :param image_matrix:
        :param white:
        :param levels:
        """

        self.imin = image_matrix

        # whiten
        self.imin /= white
        self.imin += 255 - (255 // white)

        # quantize
        self.measCentroid(self.imin, levels)
        self.quantMatrix(self.imin)

        # stipple
        self.stipple()
        self.grad = zeros(shape(self.imin), dtype=numpy.int)

        seg = self.hilbertSequence()

        d = self.totalLength(seg)

        for s in xrange(4, 20, 2):
            while True:
                delta, seg = self.twoOpt(seg, maxdelta=s)
                print delta
                d2 = self.totalLength(seg)
                assert almost_equal(delta, d2 - d)
                d = d2
                if delta == 0:
                    break

        while True:
            delta, seg = self.threeOpt(seg)
            print delta
            d2 = self.totalLength(seg)
            assert almost_equal(delta, d2 - d)
            d = d2
            if delta == 0:
                break

        while True:
            delta, seg = self.twoOpt(seg, maxdelta=40)
            print delta
            if delta == 0:
                break

        self.segments = Segments.Segments()
        self.segments.append(seg)

    def renderGrad(self):
        """
        Convert grad == -1 to pixels
        """
        x, y = where(self.grad == -1)
        self.grad[:, :] = 255
        self.grad[x, y] = 0

    def cArrayWrite(self, fname):
        self.segments.simplifyScale()
        self.segments.cArrayWrite(fname)

    def binWrite(self, fname):
        self.segments.simplifyScale()
        self.segments.binWrite(fname)

    def concatSegments(self):
        self.segments.concatSegments()

# End of module Canny
