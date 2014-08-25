__author__ = 'herman'
from scipy.cluster.vq import kmeans, vq
from scipy import ndimage
from numpy import reshape, uint8, flipud, zeros
import math


def botTransform(x,y,s=1000):
    #f = math.sqrt(x*x + y*y)
    #g = math.sqrt((s-x)*(s-x) + y*y)
    return round(botTransformRaw(x,y,s))

def botTransformRaw(x,y,s=1000):
    f = math.sqrt(x*x + y*y)
    g = math.sqrt((s-x)*(s-x) + y*y)
    return f, g

class Sketchy:

    def quantMatrix(self,levels):
        pixel = reshape(self.imat, (self.imat.shape[0]*self.imat.shape[1],1))
        self.centroids,_ = kmeans(pixel,levels)
        qnt,_ = vq(pixel,self.centroids)
        self.quant_idx = reshape(qnt,(self.imat.shape[0],self.imat.shape[1]))
        self.quant_mat = self.centroids[self.quant_idx,0]

    def __init__(self, image_matrix,levels):
        self.imat = image_matrix
        self.quantMatrix(levels)

        if False:
            (x,y) = self.imat.shape
            self.bot_mat = zeros((3*x,3*y), dtype=int)
            for i in xrange(x):
                for j in xrange(y):
                    (f, g) = botTransform(i, j)
                    #self.bot_mat[f, g] = self.quant_mat[i, j]
                    self.bot_mat[f,g] = self.imat[i,j]
        xl = []
        yl = []
        (x,y) = self.imat.shape
        for i in xrange(x):
            for j in xrange(y):
                (f, g) = botTransformRaw(i, j)
                xl.append(f)
                yl.append(g)

        self.bot_mat = ndimage.map_coordinates(self.imat, [xl, yl], order=1)
        self.bot_mat = reshape(a,)

        #self.bot_mat = ndimage.filters.gaussian_filter(self.bot_mat, 2.5, mode='nearest')
