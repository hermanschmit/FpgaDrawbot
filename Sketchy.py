__author__ = 'herman'
from scipy.cluster.vq import kmeans, vq
from scipy import ndimage
from scipy import misc
import numpy as np
from numpy import reshape, uint8, flipud, zeros, ndarray, sort
import math
import random


def botTransform(x,y,s=1000):
    #f = math.sqrt(x*x + y*y)
    #g = math.sqrt((s-x)*(s-x) + y*y)
    return round(botTransformRaw(x,y,s))

def botTransformRaw(coords):
    s = 512
    x=coords[0]
    y=coords[1]
    f = math.sqrt(x*x + y*y)
    g = math.sqrt((s-x)*(s-x) + y*y)
    return f, g

def botTransformReverse(coords, m, offset, s):
    f = coords[0]+offset
    g = coords[1]+offset
    x = (f*f-g*g+s*s-2*m*s)/(2*s)
    mx2 = (m+x)*(m+x)
    if mx2+m > f*f:
        return float('nan'),float('nan')
    y = math.sqrt(f*f - mx2 - m)
    return x, y
    #return x-m,y-m


class Sketchy:

    MAXDELTA = 512
    BASELINE = 1024
    KEEPWHITE = -50

    def score(self, pt):
        if self.drawn_mat[pt] == 0 and self.quant_idx[pt] == 0:
            return self.KEEPWHITE
        else:
            return self.quant_idx[pt] - self.drawn_mat[pt]

    def bresenhamScore(self,p0,p1,score=0,commit=False):
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
                score += self.score((x,y))
                if commit:
                    if self.drawn_mat[x, y] < self.levels-1 :
                        self.drawn_mat[x, y] += 1
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != p1[1]:
                score += self.score((x,y))
                if commit:
                    if self.drawn_mat[x, y] < self.levels-1 :
                        self.drawn_mat[x, y] += 1
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        if commit:
            if self.drawn_mat[x, y] < self.levels-1 :
                self.drawn_mat[x, y] += 1
        score += self.score((x,y))
        return score


    def measCentroid(self,mat,levels):
        pixel = reshape(mat, (mat.shape[0]*mat.shape[1],1))
        centroids,_ = kmeans(pixel,levels)
        print centroids
        self.centroids = np.sort(centroids,axis=0)
        print self.centroids

    def quantMatrix(self,mat):
        pixel = reshape(mat, (mat.shape[0]*mat.shape[1],1))
        qnt,_ = vq(pixel,self.centroids)
        self.quant_idx = reshape(qnt,(mat.shape[0],mat.shape[1]))
        self.quant_mat = self.centroids[self.quant_idx,0]

    def __init__(self, image_matrix, levels, transform=False ):
        self.moveEval = 200


        self.measCentroid(image_matrix,levels)
        self.levels = levels

        if transform:
            # resize the image
            alpha = self.MAXDELTA / (self.BASELINE * math.sqrt(2.0))
            width = image_matrix.shape[1]
            scale = self.BASELINE * alpha / width
            m = self.BASELINE*(1-alpha)/2
            self.imat = misc.imresize(image_matrix, scale)
            print alpha, width, scale, m

            self.rot_mat = misc.imrotate(self.imat,90)


            self.target_mat = ndimage.interpolation.geometric_transform(self.rot_mat,
                                                                     botTransformReverse,
                                                                     output_shape=(512,512),
                                                                     extra_arguments=(m,
                                                                                      m,
                                                                                      self.BASELINE))
        else:
            self.target_mat = image_matrix[:]

        self.pen = tuple([z/2 for z in self.target_mat.shape])

        self.quantMatrix(self.target_mat)

        self.drawn_mat = ndarray(shape=self.target_mat.shape,dtype=uint8)
        self.drawn_mat.fill(0)

        #self.bot_mat = ndimage.filters.gaussian_filter(self.bot_mat, 2.5, mode='nearest')

    def draw_line(self):
        best = self.pen
        best_val = float("-inf")
        for eval in xrange(self.moveEval):
            newpt = (random.randint(0,511),
                     random.randint(0,511))
            s = self.bresenhamScore(self.pen,newpt)
            if s > best_val:
                best = newpt
                best_val = s
        self.bresenhamScore(self.pen,best,commit=True)
        self.pen = best
        # Evaluate Move

        # Draw line in drawn_mat

        # Evaluate Moves
