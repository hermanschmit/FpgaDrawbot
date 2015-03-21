import Canny
import sys
from scipy import misc
from time import time


def main(ifile_name, ofile_name1, bin_fn="bfile.bin"):
    im = misc.imread(ifile_name, flatten=True)
    t1 = time()
    canny = Canny.Canny(im)
    print "Canny Done:", time()-t1
    canny.x = 500
    canny.y = 500

    canny.segmentList = [[250,250],[50,50],[450,50],[450,450],[50,450]]

    canny.binWrite(bin_fn)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print "Error: unknown usage"