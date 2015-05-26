"""
Hilbert based imaging
"""

from scipy import *
from scipy.misc import *
from scipy.signal import convolve2d as conv
import scipy.ndimage as ndi
from scipy.ndimage import (gaussian_filter, generate_binary_structure, binary_erosion, label)
import numpy
import Segments
import Queue as Q

def xy2d(n, x, y):
    d = 0
    s = n//2
    while s > 0:
        rx = (x & s) > 0
        ry = (y & s) > 0
        d += s * s * ((3 * rx) ^ ry)
        rot(s, x, y, rx, ry)
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


class Hilbert:
    """
        Create instances of this class to apply the Canny edge
        detection algorithm to an image.

        input: imagename(string),sigma for gaussian blur
        optional args: thresHigh,thresLow

        output: numpy ndarray.


    """


    def segment2grad(self,interior=False):
        self.segments.segment2grad(interior)

    def stipple(self,stride=1):
        # probably want to smooth imin
        self.stipple_im = self.imin[:]

        A,B,G,S = [7.0/16.0, 3.0/16.0, 5.0/16.0, 1.0/16.0]
        (xdim,ydim) = self.imin.shape

        for y in xrange(0,ydim-2-stride,stride):
            for x in xrange(0,xdim-2-stride,stride):
                oldpixel = self.stipple_im[x,y]
                if oldpixel > 128:
                    newpixel = 255
                else:
                    newpixel = 0

                self.stipple_im[x,y] = float(newpixel)
                quant_error = float(oldpixel - newpixel)
                if x < xdim-2 - 1:
                    self.stipple_im[x+stride, y] += (A * quant_error)
                if (x > 0) and (y < ydim-2 - 1):
                    self.stipple_im[x-stride, y+stride] += (B * quant_error)
                if y < ydim-2 - 1:
                    self.stipple_im[x, y+stride] += (G * quant_error)
                if (x < xdim-2 - 1) and (y < ydim-2 - 1):
                    self.stipple_im[x+stride, y+stride] += (S * quant_error)

    def __init__(self, image_matrix):
        """

        :param imname:
        :param sigma:
        :param thresHigh:
        :param thresLow:
        """
        self.imin = image_matrix
        self.imin /= 4
        self.imin += 255 - (255//4)
        self.stipple()
        self.grad = zeros(shape(self.imin), dtype=numpy.int)

        q = Q.PriorityQueue()
        xS, yS = where(self.stipple_im == 0)
        for x,y in zip(xS,yS):
            d = xy2d(1024,x,y)
            q.put((d,(x,y)))
        seg = []
        while not q.empty():
            p = q.get()
            pt = p[1]
            seg.append([pt[0], pt[1]])

        self.segments = Segments.Segments()
        self.segments.append(seg)


    def renderGrad(self):
        """
        Convert grad == -1 to pixels
        """
        x,y = where(self.grad == -1)
        self.grad[:, :] = 255
        self.grad[x, y] = 0


    def bresenhamFillIn(self,p0,p1):
        """
        Bresenham's line algorithm
        """
        dx = abs(p1[0] - p0[0])
        dy = abs(p1[1] - p0[1])
        x, y = p0[0],p0[1]
        sx = -1 if p0[0] > p1[0] else 1
        sy = -1 if p0[1] > p1[1] else 1
        if dx > dy:
            err = dx / 2.0
            while x != p1[0]:
                self.grad[x, y] = -1
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != p1[1]:
                self.grad[x, y] = -1
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        self.grad[x, y] = -1


    def cArrayWrite(self, fname):
        self.segments.cArrayWrite(fname)

    def binWrite(self, fname):
        self.segments.binWrite(fname)

    def concatSegments(self):
        self.segments.concatSegments()


# End of module Canny

