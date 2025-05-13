from scipy import *
from scipy.cluster.vq import kmeans, vq
import numpy as np

def measCentroid(mat, levels):
    pixel = np.reshape(mat, (mat.shape[0] * mat.shape[1], 1))
    centroids, _ = kmeans(pixel, levels)
    return np.sort(centroids, axis=0)


def quantMatrix(mat, newQuant, centroids, augmentWhite=True):
    pixel = np.reshape(mat, (mat.shape[0] * mat.shape[1], 1))
    qnt, _ = vq(pixel, centroids)
    quant_idx = np.reshape(qnt, (mat.shape[0], mat.shape[1]))
    imin = newQuant[quant_idx, 0]
    if augmentWhite:
        x, y = np.where(imin == centroids[-1])
        imin[x, y] = 255
    return imin
