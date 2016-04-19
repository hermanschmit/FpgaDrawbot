__author__ = 'herman'
import sys

from scipy import misc

import Maze


def main(ifile_name, ofile_name1, bin_fn="bfile.bin"):
    im = misc.imread(ifile_name, flatten=True)
    m = Maze.Maze(im)
    m.optimize_loop2(1000)
    m.mazeSegmentOptimize()
    m.maze_to_segments()
    m.segments.segment2grad(interior=True, scale=2)
    m.segments.renderGrad()
    im = m.segments.grad
    misc.imsave(ofile_name1, im)
    m.segments.scaleBin()
    m.segments.binWrite(bin_fn)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Error: unknown usage")
