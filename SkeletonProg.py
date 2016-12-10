import sys

from scipy import misc

import Skeleton

def main(ifile_name, ofile_name1, bin_fn="bfile.bin", svg_file=None):
    im = misc.imread(ifile_name, flatten=True)
    skeleton = Skeleton.Skeleton(im)
    skeleton.segments.addInitialStartPt()
    skeleton.euclidMstOrder()
    skeleton.segments.concatSegments()
    #skeleton.segments.simplify()
    skeleton.segments.segment2grad(interior=True)
    skeleton.segments.renderGrad()
    im = skeleton.segments.grad
    misc.imsave(ofile_name1, im)
    skeleton.segments.scaleBin()
    skeleton.segments.binWrite(bin_fn)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Error: unknown usage")

