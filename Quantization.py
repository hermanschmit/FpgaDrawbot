from scipy import *
from scipy.cluster.vq import kmeans, vq


def measCentroid(mat, levels):
    pixel = reshape(mat, (mat.shape[0] * mat.shape[1], 1))
    centroids, _ = kmeans(pixel, levels)
    return sort(centroids, axis=0)


def quantMatrix(mat, newQuant, centroids, augmentWhite=True):
    pixel = reshape(mat, (mat.shape[0] * mat.shape[1], 1))
    qnt, _ = vq(pixel, centroids)
    quant_idx = reshape(qnt, (mat.shape[0], mat.shape[1]))
    imin = newQuant[quant_idx, 0]
    if augmentWhite:
        x, y = where(imin == centroids[-1])
        imin[x, y] = 255
    return imin
