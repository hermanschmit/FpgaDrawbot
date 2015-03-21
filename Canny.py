"""
Module for Canny edge detection
Requirements: 1.scipy.(numpy is also mandatory, but it is assumed to be
                      installed with scipy)
              2. Python Image Library(only for viewing the final image.)
Author: Vishwanath
contact: vishwa.hyd@gmail.com
"""

from scipy import *
from scipy.misc import *
from scipy.signal import convolve2d as conv
import scipy.ndimage as ndi
from scipy.ndimage import (gaussian_filter, generate_binary_structure, binary_erosion, label)
import numpy
import EuclidMST


def hyp(ax, ay, bx, by):
    xdiff = ax - bx
    ydiff = ay - by
    return math.hypot(xdiff,ydiff)


def colinear(p0, p1, p2):
    x1, y1 = p1[0] - p0[0], p1[1] - p0[1]
    x2, y2 = p2[0] - p0[0], p2[1] - p0[1]
    return abs(x1 * y2 - x2 * y1) < 1e-5


def neighbor(p0,p1):
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

def smooth_with_function_and_mask(image, function, mask):

    bleed_over = function(mask.astype(float))
    masked_image = numpy.zeros(image.shape, image.dtype)
    masked_image[mask] = image[mask]
    smoothed_image = function(masked_image)
    output_image = smoothed_image / (bleed_over + numpy.finfo(float).eps)
    return output_image


class Canny:
    """
        Create instances of this class to apply the Canny edge
        detection algorithm to an image.

        input: imagename(string),sigma for gaussian blur
        optional args: thresHigh,thresLow

        output: numpy ndarray.

        P.S: use canny.grad to access the image array

        Note:
        1. Large images take a lot of time to process, Not yet optimised
        2. thresHigh will decide the number of edges to be detected. It
           does not affect the length of the edges being detected
        3. thresLow will decide the lenght of hte edges, will not affect
           the number of edges that will be detected.

    """

    def euclidMstPrune(self,firstPreserved=True,factor=40):
        emst = EuclidMST.EuclidMST(self.segmentList)
        emst.lonelySegmentRemoval(firstPreserved,factor)
        self.segmentList = emst.newSegmentTree

    def euclidMstOrder(self):
        emst2 = EuclidMST.EuclidMST(self.segmentList)
        emst2.segmentOrdering()
        self.segmentList = emst2.newSegmentTree

    def addInitialStartPt(self):
        self.x, self.y = self.grad.shape
        self.segmentList.insert(0, [[self.x // 2, self.y // 2]])

    def segment2grad(self,interior=False):
        self.grad[:, :] = 0
        for s in self.segmentList:
            for p in s:
                self.grad[p[0], p[1]] = -1
        s0 = self.segmentList[0]
        for s1 in self.segmentList[1:]:
            self.bresenhamFillIn(s0[-1], s1[0])
            s0 = s1
        if interior:
            for s0 in self.segmentList:
                p0 = s0[0]
                for p1 in s0[1:]:
                    self.bresenhamFillIn(p0,p1)
                    p0 = p1

    def __init__(self, image_matrix, sigma = 1.8, thresHigh = 40, thresLow = 6, thresHighLimit=2**18):
        """

        :param imname:
        :param sigma:
        :param thresHigh:
        :param thresLow:
        """
        self.imin = image_matrix
        self.thresHigh = thresHigh
        self.thresLow = thresLow


        mask = numpy.ones(self.imin.shape, dtype=bool)
        fsmooth = lambda x: gaussian_filter(x, sigma, mode='constant')
        imout = smooth_with_function_and_mask(self.imin, fsmooth, mask)

        grady = ndi.prewitt(imout, axis=1, mode='constant') * -1.0
        gradx = ndi.prewitt(imout, axis=0, mode='constant')

        grad = numpy.hypot(gradx, grady)

        # Net gradient is the square root of sum of square of the horizontal
        # and vertical gradients

        # grad = numpy.hypot(gradx, grady)
        theta = numpy.arctan2(grady, gradx)
        theta = 180 + (180/pi)*theta
        # Only significant magnitudes are considered. All others are removed
        x, y = where(grad < 10)
        theta[x, y] = 0
        grad[x, y] = 0

        # The angles are quantized. This is the first step in non-maximum
        # supression. Since, any pixel will have only 4 approach directions.
        x0, y0 = where(((theta<22.5)+(theta>=157.5)*(theta<202.5)
                       +(theta>=337.5)) == True)
        x45, y45 = where( ((theta>=22.5)*(theta<67.5)
                          +(theta>=202.5)*(theta<247.5)) == True)
        x90, y90 = where( ((theta>=67.5)*(theta<112.5)
                          +(theta>=247.5)*(theta<292.5)) == True)
        x135, y135 = where( ((theta>=112.5)*(theta<157.5)
                            +(theta>=292.5)*(theta<337.5)) == True)

        #self.theta = theta
        # Image.fromarray(self.theta).convert('L').save('Angle map.jpg')
        theta[x0, y0] = 0
        theta[x45, y45] = 45
        theta[x90, y90] = 90
        theta[x135, y135] = 135

        self.grad = grad[1:-1, 1:-1]
        self.theta = theta[1:-1, 1:-1]

        x, y = self.grad.shape
        grad2 = self.grad.copy()

        for i in range(x):
            for j in range(y):

                if self.theta[i, j] == 0:
                    test = self.nms_check(grad2, i, j, 1, 0, -1, 0)
                    if not test:
                        self.grad[i, j] = 0

                elif self.theta[i, j] == 45:
                    test = self.nms_check(grad2, i, j, 1, -1, -1, 1)
                    if not test:
                        self.grad[i, j] = 0

                elif self.theta[i, j] == 90:
                    test = self.nms_check(grad2, i, j, 0, 1, 0, -1)
                    if not test:
                        self.grad[i, j] = 0
                elif self.theta[i, j] == 135:
                    test = self.nms_check(grad2, i, j, 1, 1, -1, -1)
                    if not test:
                        self.grad[i, j] = 0

        #print self.grad

        init_point = self.initPt(thresHighLimit)
        # Hysteresis tracking. Since we know that significant edges are
        # continuous contours, we will exploit the same.
        # thresHigh is used to track the starting point of edges and
        # thresLow is used to track the whole edge till end of the edge.

        self.segmentList = []
        segment = [init_point]

        while init_point != -1:
            # print 'next segment at',init_point
            self.grad[init_point[0], init_point[1]] = -1
            p2 = init_point
            p1 = init_point
            p0 = init_point
            p0 = self.nextNbd(p0, p1, p2)

            while p0 != -1:
                segment.append(p0)
                p2 = p1
                p1 = p0
                self.grad[p0[0], p0[1]] = -1
                p0 = self.nextNbd(p0, p1, p2)

            if len(segment) >= 2:
                self.segmentList.append(segment)

            init_point = self.nextPt(self.grad)
            segment = [init_point]

    def renderGrad(self):
        """
        Convert grad == -1 to pixels
        """
        x,y = where(self.grad == -1)
        self.grad[:, :] = 0
        self.grad[x, y] = 255

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


    def createFilter(self,rawfilter):
        """
            This method is used to create an NxN matrix to be used as a filter,
            given a N*N list
        """
        order = pow(len(rawfilter), 0.5)
        order = int(order)
        filt_array = array(rawfilter)
        outfilter = filt_array.reshape((order, order))
        return outfilter

    def gaussFilter(self,sigma,window = 3):
        """
            This method is used to create a gaussian kernel to be used
            for the blurring purpose. inputs are sigma and the window size
        """
        kernel = zeros((window,window))
        c0 = window // 2

        for x in range(window):
            for y in range(window):
                r = math.hypot((x-c0), (y-c0))
                val = (1.0/2*pi*sigma*sigma)*exp(-(r*r)/(2*sigma*sigma))
                kernel[x, y] = val
        return kernel / kernel.sum()

    def nms_check(self, grad, i, j, x1, y1, x2, y2):
        """
            Method for non maximum supression check. A gradient point is an
            edge only if the gradient magnitude and the slope agree

            for example, consider a horizontal edge. if the angle of gradient
            is 0 degress, it is an edge point only if the value of gradient
            at that point is greater than its top and bottom neighbours.
        """
        try:
            if (i+x1 < 0) or (i+x2 < 0) or (j+x1 < 0) or (j+x2 < 0): return True
            if (grad[i,j] > grad[i+x1,j+y1]) and (grad[i,j] > grad[i+x2,j+y2]):
                return True
            else:
                return False
        except IndexError:
            return True

    def initPt(self, limit=200000):
        done = False
        while not done:
            X,Y = where(self.grad > self.thresHigh)
            if len(X) > limit:
                self.thresHigh *= 1.25
                self.thresLow *= 1.25
                print "Degrade threshold", self.thresHigh, len(X)
            else:
                done = True
                XY = zip(X,Y)
        self.threshPts = sorted(XY, key=lambda x: x[1])
        if len(self.threshPts) == 0: return -1
        iP = self.threshPts.pop(0)
        return [iP[0],iP[1]]

    def nextPt(self,im):
        if len(self.threshPts) == 0: return -1
        iP = self.threshPts.pop(0)
        while im[iP[0],iP[1]] == -1:
            if len(self.threshPts) == 0: return -1
            iP = self.threshPts.pop(0)
        return [iP[0],iP[1]]

    def stop(self,im,thres):
        """
            This method is used to find the starting point of an edge.
        """
        X,Y = where(im > thres)
        try:
            idx = Y.argmin()
        except:
            return -1
        x = X[idx]
        y = Y[idx]
        return [x,y]

    def nextNbd(self, p0, p1, p2):
        """
            This method is used to return the next point on the edge.
        """
        X,Y = self.grad.shape
        for (i,j) in [(-1,0),(0,-1),(1,0),(0,1),(1,1),(1,-1),(-1,1),(-1,-1)]:
            x = p0[0]+i
            y = p0[1]+j

            if (x<0) or (y<0) or (x>=X) or (y>=Y):
                continue
            if ([x,y] == p1) or ([x,y] == p2):
                continue
            if self.grad[x,y] > self.thresLow:
                return [x,y]
        return -1

    def pruneLonelySegments(self,ratio=100.):
        newSegmentList = list()
        newSegmentList.append(self.segmentList[0])

        if len(self.segmentList) < 3: return
        s0 = self.segmentList[0]
        s1 = self.segmentList[1]
        for i in xrange(2, len(self.segmentList)):
            s2 = self.segmentList[i]
            drawnA = hyp(*(s0[-1] + s1[0]))
            drawnB = hyp(*(s1[-1] + s2[0]))
            segment = max(hyp(*(s1[0] + s1[-1])),len(s1))
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
        drawn = hyp(*(last_2[-1] + last[0]))
        limit = ratio * hyp(*(last[0] + last[-1]))
        if drawn < limit:
            newSegmentList.append(self.segmentList[-1])
        self.segmentList = newSegmentList[:]


    def pixelscale(self,pt,maxXY):
        px = float(pt[0]-self.x//2) / maxXY
        # py = -1.0 * float(pt[1]-self.y//2) / maxXY
        py = 1.0 * float(pt[1]-self.y//2) / maxXY
        return px, py

    def cArrayWrite(self, fname, depth = 2**20):
        f = open(fname,'w')
        numpts = 0
        segmentList_simp = []
        for s in self.segmentList:
            ns = simplifysegments(s)
            segmentList_simp.append(ns)
            numpts += len(ns)
        if numpts-1 >= depth:
            print "Number of points exceeds limit: "+repr(numpts)
            raise ValueError
        f.write("float diag["+repr(depth)+"][2] = {\n")
        m = max(self.x//2, self.y//2)
        i = 0
        for s in segmentList_simp:
            for p in s:
                x, y = self.pixelscale(p,m)
                f.write("       {"+repr(y)+", "+repr(x)+"},\n")
                i += 1
        for j in xrange(i, depth-1):
            f.write("       {NAN, NAN},\n")
        f.write("       {NAN, NAN}\n")
        f.write(" };\n")
        f.close()

    def binWrite(self, fname, depth = 2**20):
        numpts = 0
        segmentList_simp = []
        for s in self.segmentList:
            ns = simplifysegments(s)
            segmentList_simp.append(ns)
            numpts += len(ns)
        if numpts-1 >= depth:
            print "Number of points exceeds limit: "+repr(numpts)
            raise ValueError
        m = max(self.x//2,self.y//2)
        i = 0
        from array import array
        output_file = open(fname, 'wb')
        l = [float('NaN')] * (2 * depth)

        for s in segmentList_simp:
            for p in s:
                x, y = self.pixelscale(p,m)
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
            # if neighbor(s_prev[-1],s_new[0]):
            #     s_prev = numpy.append(s_prev,s_new,axis=0)
            # else:
            #     sL.append(s_prev)
            #     s_prev = s_new
        sL.append(s_prev)
        self.segmentList = sL


# End of module Canny

