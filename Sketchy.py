__author__ = 'herman'
from scipy.cluster.vq import kmeans,vq
from numpy import reshape,uint8,flipud,zeros
import math


def botTransform(x,y,s=1000):
    f = math.sqrt(x*x + y*y)
    g = math.sqrt((s-x)*(s-x) + y*y)
    return round(f), round(g)

class Sketchy:
    def __init__(self, image_matrix,levels):
        self.imat = image_matrix
        pixel = reshape(self.imat, (self.imat.shape[0]*self.imat.shape[1],1))
        centroids,_ = kmeans(pixel,levels)
        qnt,_ = vq(pixel,centroids)
        self.quant_idx = reshape(qnt,(self.imat.shape[0],self.imat.shape[1]))
        self.quant_mat = centroids[self.quant_idx,0]

        (x,y) = self.quant_idx.shape
        self.bot_mat = zeros((3*x,3*y), dtype=int)
        for i in xrange(x):
            for j in xrange(y):
                (f, g) = botTransform(i, j)
                self.bot_mat[f, g] = self.quant_mat[i, j]
