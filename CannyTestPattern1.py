import sys
from time import time

from scipy import misc

import Canny


def main(ifile_name, ofile_name1, bin_fn="bfile.bin"):
    im = misc.imread(ifile_name, flatten=True)
    t1 = time()
    canny = Canny.Canny(im)
    print("Canny Done:", time() - t1)
    canny.x = 500
    canny.y = 500
    canny.segmentList = []
    for x in range(0, 500, 20):
        canny.segmentList.append([[x, 0], [x, 500], [x + 10, 500], [x + 10, 0]])
    for y in range(0, 500, 20):
        canny.segmentList.append([[500, y], [0, y], [0, y + 10], [500, y + 10]])

    print(canny.segmentList)
    canny.binWrite(bin_fn)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Error: unknown usage")
