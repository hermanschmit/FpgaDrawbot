#from skimage.morphology import skeletonize
from skimage import draw
import numpy as np
import Segments
import Quantization
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.ndimage as ndi
import EuclidMST

def smooth_with_function_and_mask(image, function, mask):
    bleed_over = function(mask.astype(float))
    masked_image = np.zeros(image.shape, image.dtype)
    masked_image[mask] = image[mask]
    smoothed_image = function(masked_image)
    output_image = smoothed_image / (bleed_over + np.finfo(float).eps)
    return output_image

def skeletonize(image):
    """Return the skeleton of a binary image.

    Thinning is used to reduce each connected component in a binary image
    to a single-pixel wide skeleton.

    Parameters
    ----------
    image : numpy.ndarray
        A binary image containing the objects to be skeletonized. '1'
        represents foreground, and '0' represents background. It
        also accepts arrays of boolean values where True is foreground.

    Returns
    -------
    skeleton : ndarray
        A matrix containing the thinned image.

    See also
    --------
    medial_axis

    Notes
    -----
    The algorithm [1] works by making successive passes of the image,
    removing pixels on object borders. This continues until no
    more pixels can be removed.  The image is correlated with a
    mask that assigns each pixel a number in the range [0...255]
    corresponding to each possible pattern of its 8 neighbouring
    pixels. A look up table is then used to assign the pixels a
    value of 0, 1, 2 or 3, which are selectively removed during
    the iterations.

    Note that this algorithm will give different results than a
    medial axis transform, which is also often referred to as
    "skeletonization".

    References
    ----------
    .. [1] A fast parallel algorithm for thinning digital patterns,
       T. Y. ZHANG and C. Y. SUEN, Communications of the ACM,
       March 1984, Volume 27, Number 3


    Examples
    --------
    >>> X, Y = np.ogrid[0:9, 0:9]
    >>> ellipse = (1./3 * (X - 4)**2 + (Y - 4)**2 < 3**2).astype(np.uint8)
    >>> ellipse
    array([[0, 0, 0, 1, 1, 1, 0, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 1, 1, 1, 0, 0, 0]], dtype=uint8)
    >>> skel = skeletonize(ellipse)
    >>> skel
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)

    """
    # look up table - there is one entry for each of the 2^8=256 possible
    # combinations of 8 binary neighbours. 1's, 2's and 3's are candidates
    # for removal at each iteration of the algorithm.
    lut = [0, 0, 0, 1, 0, 0, 1, 3, 0, 0, 3, 1, 1, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 3, 0, 3, 3,
           0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 2, 2,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 2, 0,
           0, 1, 3, 1, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
           3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           2, 3, 1, 3, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           2, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 1, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0]

    # convert to unsigned int (this should work for boolean values)
    skeleton = np.array(image).astype(np.uint8)

    # check some properties of the input image:
    #  - 2D
    #  - binary image with only 0's and 1's
    if skeleton.ndim != 2:
        raise ValueError('Skeletonize requires a 2D array')
    if not np.all(np.in1d(skeleton.flat, (0, 1))):
        raise ValueError('Image contains values other than 0 and 1')

    # create the mask that will assign a unique value based on the
    #  arrangement of neighbouring pixels
    mask = np.array([[1, 2, 4],
                     [128, 0, 8],
                     [64, 32, 16]], np.uint8)

    pixelRemoved = True
    while pixelRemoved:
        pixelRemoved = False;

        # assign each pixel a unique value based on its foreground neighbours
        neighbours = ndi.correlate(skeleton, mask, mode='nearest')

        # ignore background
        neighbours *= skeleton

        # use LUT to categorize each foreground pixel as a 0, 1, 2 or 3
        codes = np.take(lut, neighbours)

        # pass 1 - remove the 1's and 3's
        code_mask = (codes == 1)
        if np.any(code_mask):
            pixelRemoved = True
            skeleton[code_mask] = 0
        code_mask = (codes == 3)
        if np.any(code_mask):
            pixelRemoved = True
            skeleton[code_mask] = 0

        # pass 2 - remove the 2's and 3's
        neighbours = ndi.correlate(skeleton, mask, mode='nearest')
        neighbours *= skeleton
        codes = np.take(lut, neighbours)
        code_mask = (codes == 2)
        if np.any(code_mask):
            pixelRemoved = True
            skeleton[code_mask] = 0
        code_mask = (codes == 3)
        if np.any(code_mask):
            pixelRemoved = True
            skeleton[code_mask] = 0

    return skeleton


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



    def __init__(self, image_matrix, sigma=0.5):

        if sigma >= 0.:
            mask = np.ones(image_matrix.shape, dtype=bool)
            fsmooth = lambda x: ndi.gaussian_filter(x, sigma, mode='constant')
            self.imin = smooth_with_function_and_mask(image_matrix, fsmooth, mask)
        else:
            self.imin = image_matrix

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

        self.segments = Segments.Segments()


        for neighbor_limit in range(1,5):
            while True:
                self.count_neighbors()
                for internal_limit in range(1,neighbor_limit+1):
                    startx, starty = np.where(self.neighbor_count == internal_limit)
                    if len(startx) > 0:
                        break
                if len(startx) == 0:
                    break
                for (x,y) in zip(startx, starty):
                    if self.skeleton[x,y] == 1:
                        self.follow_line(x,y)
        print(np.sum(self.skeleton))
        print(np.sum(self.neighbor_count))
        print(np.max(self.neighbor_count))
        assert (np.sum(self.neighbor_count) == 0)

    def follow_line(self,x,y):
        X,Y = self.skeleton.shape
        segment = [[x, y]]
        self.skeleton[x, y] = 0
        while True:
            # search for neighbor
            # decrement neighbor counts
            adjacent_counter = self.neighbor_count[x, y]
            done = adjacent_counter > 1 or adjacent_counter == 0
            self.neighbor_count[x, y] = 0

            for (i, j) in [(-1, 0), (0, -1), (1, 0), (0, 1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                x_n = x + i
                y_n = y + j

                if (x_n < 0) or (y_n < 0) or (x_n >= X) or (y_n >= Y):
                    continue
                if self.skeleton[x_n, y_n] == 1:
                    self.neighbor_count[x_n, y_n] -= 1
                    x_save, y_save = x_n, y_n

            if not done:
                segment.append([x_save, y_save])
                x, y = x_save, y_save
                self.skeleton[x, y] = 0
            else:
                break
        if len(segment) > 1:
            self.segments.append(segment)





