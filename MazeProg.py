__author__ = 'herman'
import sys

from scipy import misc

import Maze


def main(ifile_name, ofile_name1, bin_fn="bfile.bin", svg_file=None):
    im = misc.imread(ifile_name, flatten=True)
    m = Maze.Maze(im,levels=5)
    #m.optimize_loop2(500,10,1.025,10)
    m.optimize_loop2(150,1,2,10)
    m.Fb = 0.0
    m.optimize_loop2(20,1000,2,10)

    m.mazeSegmentOptimize()
    m.maze_to_segments()
    m.segments.segment2grad(interior=True, scale=2)
    m.segments.renderGrad()
    im = m.segments.grad
    misc.imsave(ofile_name1, im)
    if svg_file!=None:
        m.segments.svgwrite(svg_file)
    m.segments.scaleBin()
    m.segments.binWrite(bin_fn)


if __name__ == "__main__":
    if len(sys.argv) == 5:
        main(sys.argv[1], sys.argv[2], sys.argv[3],sys.argv[4])
    elif len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Error: unknown usage")
