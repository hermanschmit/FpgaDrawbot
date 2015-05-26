import Hilbert
import sys
from scipy import misc
from time import time



def main(ifile_name, ofile_name1, bin_fn="bfile.bin"):
    im = misc.imread(ifile_name, flatten=True)
    h = Hilbert.Hilbert(im)
    h.segment2grad(interior=True)
    h.binWrite(bin_fn)
    h.renderGrad()
    im = h.segments.grad
    misc.imsave(ofile_name1, im)

if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print "Error: unknown usage"