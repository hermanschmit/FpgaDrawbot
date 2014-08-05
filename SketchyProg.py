__author__ = 'herman'
import Sketchy
import sys
from scipy.misc import *

def main(ifile_name, ofile_name1, levels=2):
    sketch = Sketchy.Sketchy(imread(ifile_name, flatten=True),levels)
    om = sketch.quant_mat
#    imsave(ofile_name1, om)
    imsave(ofile_name1, sketch.bot_mat)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print "Error: unknown usage"