import sys
from time import time

import imageio

import Canny
import Segments


def main(ifile_name, ofile_name1, bin_fn="bfile.bin"):
    im = imageio.imread(ifile_name, as_gray=True)
    t1 = time()
    canny = Canny.Canny(im,sigma=1.0)
    print("Canny Done:", time() - t1)
    canny.addInitialStartPt()
    # canny.euclidMstPrune(True,40)
    print(len(canny.segments.segmentList))
    canny.euclidMstOrder()
    print(len(canny.segments.segmentList))
    canny.concatSegments()
    print(len(canny.segments.segmentList))
    canny.segments.simplify()
    canny.segments.segment2grad(interior=True)
    canny.segments.renderGrad()
    im = canny.segments.grad
    imageio.imsave(ofile_name1, im)
    canny.segments.svgwrite("test.svg")
    segNew = Segments.Segments()
    segNew.svgread("test.svg")
    segNew.svgwrite("test2.svg")

    canny.segments.scaleBin()
    canny.binWrite(bin_fn)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Error: unknown usage")
