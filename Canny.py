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
import Segments


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
        emst = EuclidMST.EuclidMST(self.segments.segmentList)  # TODO fix
        emst.lonelySegmentRemoval(firstPreserved,factor)
        self.segments.segmentList = emst.newSegmentTree # TODO fix

    def euclidMstOrder(self):
        emst2 = EuclidMST.EuclidMST(self.segments.segmentList)  # TODO fix
        emst2.segmentOrdering()
        self.segments.segmentList = emst2.newSegmentTree # TODO fix

    def addInitialStartPt(self):
        self.segments.addInitialStartPt(self.grad.shape)

    def segment2grad(self,interior=False):
        self.segments.segment2grad(interior)

    def stipple(self,stride=4):
        # probably want to smooth imin
        im = self.imin[:]

        A,B,G,S = [7.0/16.0, 3.0/16.0, 5.0/16.0, 1.0/16.0]
        (xdim,ydim) = self.imin.shape

        for y in xrange(0,ydim-2-stride,stride):
            for x in xrange(0,xdim-2-stride,stride):
                oldpixel = im[x,y]
                if oldpixel > 128:
                    newpixel = 255
                else:
                    newpixel = 0

                im[x,y] = float(newpixel)
                quant_error = float(oldpixel - newpixel)
                if x < xdim-2 - 1:
                    im[x+stride, y] += (A * quant_error)
                if (x > 0) and (y < ydim-2 - 1):
                    im[x-stride, y+stride] += (B * quant_error)
                if y < ydim-2 - 1:
                    im[x, y+stride] += (G * quant_error)
                if (x < xdim-2 - 1) and (y < ydim-2 - 1):
                    im[x+stride, y+stride] += (S * quant_error)

        self.stippleSegmentList = []
        for xi in xrange(0,xdim-2-stride,stride):
            for yi in xrange(0,ydim-2-stride,stride):
                if im[xi,yi] < 128 and self.grad[xi,yi] != -1:
                    self.stippleSegmentList.append([[xi,yi]])
                    # self.segmentList.append([[xi,yi]])



    def __init__(self, image_matrix, sigma = 1.0, thresHigh = 40, thresLow = 6, thresHighLimit=2**18):
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

        init_point = self.initPt(thresHighLimit)
        # Hysteresis tracking. Since we know that significant edges are
        # continuous contours, we will exploit the same.
        # thresHigh is used to track the starting point of edges and
        # thresLow is used to track the whole edge till end of the edge.

        self.segments = Segments.Segments()
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
                self.segments.append(segment)

            init_point = self.nextPt(self.grad)
            segment = [init_point]

        self.stippleSegmentList = []

    def renderGrad(self):
        """
        Convert grad == -1 to pixels
        """
        x,y = where(self.grad == -1)
        self.segments.grad[:, :] = 255
        self.segments.grad[x, y] = 0




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

    def cArrayWrite(self, fname):
        self.segments.cArrayWrite(fname)

    def binWrite(self, fname):
        self.segments.binWrite(fname)

    def concatSegments(self):
        self.segments.concatSegments()


# End of module Canny

