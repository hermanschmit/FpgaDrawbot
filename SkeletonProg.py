import sys

import imageio.v2 as imageio
import numpy as np

import Skeleton

def main(ifile_name, ofile_name1, bin_fn="bfile.bin", svg_file=None):
    im = imageio.imread(ifile_name, mode='F')
    skeleton = Skeleton.Skeleton(im)
    skeleton.segments.addInitialStartPt()
    skeleton.euclidMstOrder()
    skeleton.segments.concatSegments()
    skeleton.segments.chaikin_smoothing(2, limit=1e-4)
    skeleton.segments.simplify()
    skeleton.segments.segment2grad(interior=True)
    skeleton.segments.renderGrad()
    im = skeleton.segments.grad.astype(np.uint8)
    imageio.imwrite(ofile_name1, im)
    skeleton.segments.scaleBin()
    skeleton.segments.binWrite(bin_fn)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Error: unknown usage")

