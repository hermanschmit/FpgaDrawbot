__author__ = 'herman'
import argparse

import imageio.v2 as imageio
import numpy as np

import StringArt


def main(ifile_name, ofile_name1, bin_fn="bfile.bin", model_fn=None,
         n_pins=200, canvas_size=600, min_pin_gap=4, pen_alpha=0.12,
         max_lines=2500, min_score=0.0, removal_interval=250,
         removal_threshold=0.0):
    im = imageio.imread(ifile_name, mode='F')
    sa = StringArt.StringArt(im, n_pins=n_pins, canvas_size=canvas_size,
                             min_pin_gap=min_pin_gap, pen_alpha=pen_alpha,
                             max_lines=max_lines, min_score=min_score,
                             removal_interval=removal_interval,
                             removal_threshold=removal_threshold)

    # model_fn (optional): the optimizer's own internal model (anti-aliased,
    # clamped at full black) -- this is the meaningful visual reference for
    # judging string-art quality and for tuning pen_alpha: if it looks right
    # but the real pen comes out lighter, lower pen_alpha and regenerate.
    if model_fn is not None:
        imageio.imwrite(model_fn, (np.clip(sa.rendered_canvas, 0, 1) * 255).astype(np.uint8))

    sa.segments.concatSegments()

    # ofile_name1: the actual polyline geometry that will be drawn and that
    # also drives the .bin output (vector if it ends in .svg), rendered the
    # same binary touched/untouched way every other *Prog.py in this repo
    # previews its strokes. For a dense string-art result a raster preview
    # will characteristically look mostly solid black --
    # Segments.renderGrad() has no density/count accumulation, so once
    # overlapping chords cover most of the canvas nearly every pixel reads
    # as "touched". That's expected, not a bug; --model_png above is where
    # to actually judge the result.
    sa.segments.savePreview(ofile_name1)

    sa.segments.scaleBin()
    sa.segments.binWrite(bin_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='String-art-style continuous line drawing')
    parser.add_argument('input_image_fn', help='Input image')
    parser.add_argument('output_image_fn', help='Output preview PNG (actual pen strokes)')
    parser.add_argument('output_bin_fn', nargs='?', default='bfile.bin', help='Output bin file')
    parser.add_argument('--model_png', default=None,
                        help="Optional output PNG of the optimizer's own internal model "
                             "(anti-aliased, the useful reference for tuning pen_alpha). "
                             "Skipped if not given.")
    parser.add_argument('--n_pins', type=int, default=200, help='Number of pins around the circle')
    parser.add_argument('--canvas_size', type=int, default=600, help='Internal optimization resolution')
    parser.add_argument('--min_pin_gap', type=int, default=4,
                        help='Skip chords between pins closer together than this')
    parser.add_argument('--pen_alpha', type=float, default=0.12,
                        help='Darkness contributed by one pass, in (0,1]. '
                             'Lower this if your pen is lighter than the '
                             'string the algorithm assumes -- more passes '
                             'will then be used to reach full black. Also '
                             'controls optimization granularity: a value as '
                             'high as 1.0 lets a single chord fully saturate '
                             'every pixel it touches, which starves the '
                             'greedy search of room to build up fine tonal '
                             'detail regardless of actual pen darkness.')
    parser.add_argument('--max_lines', type=int, default=2500, help='Cap on number of chords drawn')
    parser.add_argument('--min_score', type=float, default=0.0,
                        help='Stop once no remaining chord would help by more than this')
    parser.add_argument('--removal_interval', type=int, default=250,
                        help='Sweep out over-saturated chords every N additions (0 disables)')
    parser.add_argument('--removal_threshold', type=float, default=0.0,
                        help='An active chord scoring below this gets removed in a sweep')
    args = parser.parse_args()

    main(args.input_image_fn, args.output_image_fn, args.output_bin_fn,
         model_fn=args.model_png, n_pins=args.n_pins, canvas_size=args.canvas_size,
         min_pin_gap=args.min_pin_gap, pen_alpha=args.pen_alpha,
         max_lines=args.max_lines, min_score=args.min_score,
         removal_interval=args.removal_interval,
         removal_threshold=args.removal_threshold)
