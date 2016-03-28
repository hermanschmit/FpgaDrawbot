__author__ = 'herman'
import Sketchy
import sys
from scipy.misc import *

def main(ifile_name, ofile_name1, ofile_name2, bin_fn="bfile.bin", levels=4,scale=True,transform=True):
    sketch = Sketchy.Sketchy(imread(ifile_name, flatten=True),levels,scale,transform)

    for x in range(1000):
        sketch.draw_line()
        print(x)


#    om = sketch.quant_mat
#    imsave(ofile_name1, om)
    imsave(ofile_name1, sketch.quant_idx)
    if transform:
        sketch.untransform()
    imsave(ofile_name2, sketch.drawn_mat * 20)
    sketch.binWrite(bin_fn)

    print(sketch.quant_idx)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 5:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print("Error: unknown usage")