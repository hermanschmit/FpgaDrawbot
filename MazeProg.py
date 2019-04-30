__author__ = 'herman'
import sys
import argparse
import imageio

import Maze


def main(ifile_name, ofile_name1, bin_fn="bfile.bin", svg_file=None,
         quant_levels=5,
         init_shape=Maze.Maze.INIT_DIAG):
    # im = misc.imread(ifile_name, flatten=True)
    im = imageio.imread(ifile_name,as_gray=True)
    m = Maze.Maze(im,levels=quant_levels,init_shape=init_shape)
    m.optimize_loop2(1000,1,2,10)
    m.Fb = 0.0
    m.optimize_loop2(20,-1,2,10)

    m.mazeSegmentOptimize()
    m.maze_to_segments()
    m.segments.segment2grad(interior=True, scale=2)
    m.segments.renderGrad()
    im = m.segments.grad
    #misc.imsave(ofile_name1, im)
    imageio.imwrite(ofile_name1, im)
    if svg_file!=None:
        m.segments.svgwrite(svg_file)
    m.segments.scaleBin()
    m.segments.binWrite(bin_fn)
    m.segments.openScadArrayWrite("maze.scad")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Maze Program')
    parser.add_argument('--quant_levels', action='store', dest='quant_levels', type=int, default=5)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--moore',action='store_true')
    group.add_argument('--skeleton',action='store_true')
    group.add_argument('--fass',action='store_true')
    group.add_argument('--diag',action='store_true')
    parser.add_argument('input_image_fn', action="store", help='Input Image')
    parser.add_argument('output_image_fn',action="store", help='Output Image')
    parser.add_argument('output_bin_fn',action='store',help='Output Bin File')
    parser.add_argument('output_svg_fn',action='store', nargs='?', default=None, help='SVG File')
    args = parser.parse_args()

    if args.moore:
        shape = Maze.Maze.INIT_MOORE
    elif args.skeleton:
        shape = Maze.Maze.INIT_SKEL
    elif args.fass:
        shape = Maze.Maze.INIT_FASS
    elif args.diag:
        shape = Maze.Maze.INIT_DIAG
    else:
        shape = Maze.Maze.INIT_SKEL


    main(args.input_image_fn,
         args.output_image_fn,
         args.output_bin_fn,
         args.output_svg_fn,
         args.quant_levels,
         init_shape=shape)
