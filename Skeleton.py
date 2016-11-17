from skimage.morphology import skeletonize
from skimage import draw
import numpy as np
import Segments
import Quantization
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.ndimage as ndi
import EuclidMST

class Skeleton:

    def count_neighbors(self):
        k = np.array([[1,1,1],
                     [1,0,1],
                      [1,1,1]])

        m = ndi.filters.convolve(self.skeleton,k,mode='constant',cval=0)
        nc = np.multiply(m,self.skeleton)
        self.neighbor_count = nc

    def euclidMstOrder(self):
        emst2 = EuclidMST.EuclidMST(self.segments.segmentList)  # TODO fix
        emst2.segmentOrdering()
        self.segments.segmentList = emst2.newSegmentTree  # TODO fix



    def __init__(self, image_matrix, thresHigh=40, thresLow=6, thresHighLimit=2 ** 18):
        self.imin = image_matrix
        self.thresHigh = thresHigh
        self.thresLow = thresLow

        self.centroids = Quantization.measCentroid(self.imin, 2)

        nq = np.array([[x * 255] for x in range(0, 2)])
        self.imin = Quantization.quantMatrix(self.imin, nq, self.centroids)
        plt.imshow(self.imin, cmap=cm.gray)
        plt.savefig("figStartOrig.png")
        plt.clf()

        self.ibin = np.zeros(self.imin.shape, dtype=np.uint8)
        x0,y0 = np.where(self.imin == 0)
        self.ibin[x0,y0] = 1

        skel = skeletonize(self.ibin)
        self.skeleton = skel.astype(np.uint8)
        plt.imshow(self.skeleton, cmap=cm.gray)
        plt.savefig("figStartSkel.png")
        plt.clf()

        self.count_neighbors()
        self.segments = Segments.Segments()

        startx, starty = np.where(self.neighbor_count == 1)
        X,Y = self.skeleton.shape
        for (x,y) in zip(startx, starty):
            if self.skeleton[x,y] == 1:
                segment = [[x,y]]
                self.skeleton[x,y] = 0

                while True:

                    # search for neighbor
                    # decrement neighbor counts
                    adjacent_counter = self.neighbor_count[x,y]
                    done = adjacent_counter > 1 or adjacent_counter == 0
                    self.neighbor_count[x,y] = 0

                    for (i, j) in [(-1, 0), (0, -1), (1, 0), (0, 1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                        x_n = x + i
                        y_n = y + j

                        if (x_n < 0) or (y_n < 0) or (x_n >= X) or (y_n >= Y):
                            continue
                        if self.skeleton[x_n,y_n] == 1:
                            self.neighbor_count[x_n,y_n] -= 1
                            x_save, y_save = x_n,y_n

                    if not done:
                        segment.append([x_save,y_save])
                        x,y = x_save,y_save
                        self.skeleton[x, y] = 0
                    else:
                        break
                if len(segment) > 1:
                    self.segments.append(segment)


        pass






