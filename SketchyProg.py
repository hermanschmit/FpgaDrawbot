__author__ = 'herman'
import Sketchy
import sys
from scipy.misc import *

def main(ifile_name, ofile_name1, ofile_name2, levels=4):
    sketch = Sketchy.Sketchy(imread(ifile_name, flatten=True),levels)
    print sketch.quant_idx * 20
    for x in xrange(5000):
        sketch.draw_line()
        print x
#    om = sketch.quant_mat
#    imsave(ofile_name1, om)
    imsave(ofile_name1, sketch.quant_idx)
    imsave(ofile_name2, sketch.drawn_mat * 20)
    print sketch.quant_idx


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print "Error: unknown usage"