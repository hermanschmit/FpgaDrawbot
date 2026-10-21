__author__ = 'herman'
import argparse

import imageio.v2 as imageio
import numpy as np

import PointArt


def main(ifile_name, ofile_name1, bin_fn="bfile.bin", model_fn=None,
         layout='grid', canvas_size=600,
         grid_rows=30, grid_cols=30, n_rings=14, points_per_ring=28,
         min_edge_len=8, max_edge_len=120, pen_alpha=0.12, length_penalty=0.0,
         max_lines=3000, min_score=0.0, removal_interval=250,
         removal_threshold=0.0):
    im = imageio.imread(ifile_name, mode='F')
    pa = PointArt.PointArt(im, layout=layout, canvas_size=canvas_size,
                           grid_rows=grid_rows, grid_cols=grid_cols,
                           n_rings=n_rings, points_per_ring=points_per_ring,
                           min_edge_len=min_edge_len, max_edge_len=max_edge_len,
                           pen_alpha=pen_alpha, length_penalty=length_penalty,
                           max_lines=max_lines,
                           min_score=min_score, removal_interval=removal_interval,
                           removal_threshold=removal_threshold)

    if not pa.selected:
        # pa.segments.segmentList is empty with zero lines selected, and
        # concatSegments() below indexes segmentList[0] unconditionally --
        # it would crash rather than write anything. Most likely cause:
        # length_penalty (or min_score) set high enough that no candidate's
        # score clears min_score on the very first pick. Lower
        # length_penalty, or lower min_score to let weaker-but-still-short
        # candidates through.
        raise RuntimeError(
            "PointArt selected 0 lines -- nothing to write. Try a lower "
            "--length_penalty and/or a lower --min_score.")

    # model_fn (optional): the optimizer's own internal model (anti-aliased,
    # clamped at full black) -- the meaningful visual reference for judging
    # quality and for tuning pen_alpha, same as StringArtProg.py's --model_png.
    if model_fn is not None:
        imageio.imwrite(model_fn, (np.clip(pa.rendered_canvas, 0, 1) * 255).astype(np.uint8))

    pa.segments.concatSegments()

    # ofile_name1: the actual polyline geometry that will be drawn and that
    # also drives the .bin output -- see StringArtProg.py's comment on the
    # same rendering convention for why a dense result looks mostly black.
    pa.segments.savePreview(ofile_name1)

    pa.segments.scaleBin()
    pa.segments.binWrite(bin_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Line-art rendering over a free point field (grid or concentric rings), '
                     'instead of chords on a fixed circle of pins (see StringArt.py)')
    parser.add_argument('input_image_fn', help='Input image')
    parser.add_argument('output_image_fn', help='Output preview PNG (actual pen strokes)')
    parser.add_argument('output_bin_fn', nargs='?', default='bfile.bin', help='Output bin file')
    parser.add_argument('--model_png', default=None,
                        help="Optional output PNG of the optimizer's own internal model "
                             "(anti-aliased, the useful reference for tuning pen_alpha). "
                             "Skipped if not given.")
    parser.add_argument('--layout', choices=['grid', 'rings'], default='grid',
                        help='Point field shape: rectangular lattice or concentric rings')
    parser.add_argument('--canvas_size', type=int, default=600, help='Internal optimization resolution')
    parser.add_argument('--grid_rows', type=int, default=30, help='Point rows (layout=grid)')
    parser.add_argument('--grid_cols', type=int, default=30, help='Point columns (layout=grid)')
    parser.add_argument('--n_rings', type=int, default=14, help='Number of rings (layout=rings)')
    parser.add_argument('--points_per_ring', type=int, default=28,
                        help='Points per ring (layout=rings)')
    parser.add_argument('--min_edge_len', type=float, default=8,
                        help='Drop candidate lines shorter than this')
    parser.add_argument('--max_edge_len', type=float, default=120,
                        help='Drop candidate lines longer than this -- the main '
                             'performance/quality knob: candidate count grows roughly '
                             'with the square of this relative to point spacing')
    parser.add_argument('--pen_alpha', type=float, default=0.12,
                        help='Darkness contributed by one pass, in (0,1]. '
                             'Lower this if your pen is lighter than the algorithm '
                             'assumes -- more passes will then be used to reach full '
                             'black. Also controls optimization granularity -- see '
                             'StringArtProg.py --pen_alpha help for why 1.0 is a poor '
                             'default regardless of pen darkness.')
    parser.add_argument('--length_penalty', type=float, default=0.0,
                        help='Stylistic bias toward shorter lines: subtracted from '
                             'each candidate\'s score, scaled by its pixel length, '
                             'before ranking (and added back for removal-sweep '
                             'decisions, so long lines already drawn are also more '
                             'likely to get swept out). 0 (default) disables it. No '
                             'canvas-independent good value -- start at 0 and double '
                             'it a few times, watching the printed "selected N / M" '
                             'count. Too high and selection collapses to 0 (this '
                             'script then raises instead of writing a corrupt file).')
    parser.add_argument('--max_lines', type=int, default=3000, help='Cap on number of lines drawn')
    parser.add_argument('--min_score', type=float, default=0.0,
                        help='Stop once no remaining line would help by more than this')
    parser.add_argument('--removal_interval', type=int, default=250,
                        help='Sweep out over-saturated lines every N additions (0 disables)')
    parser.add_argument('--removal_threshold', type=float, default=0.0,
                        help='An active line scoring below this gets removed in a sweep')
    args = parser.parse_args()

    main(args.input_image_fn, args.output_image_fn, args.output_bin_fn,
         model_fn=args.model_png, layout=args.layout, canvas_size=args.canvas_size,
         grid_rows=args.grid_rows, grid_cols=args.grid_cols, n_rings=args.n_rings,
         points_per_ring=args.points_per_ring, min_edge_len=args.min_edge_len,
         max_edge_len=args.max_edge_len, pen_alpha=args.pen_alpha,
         length_penalty=args.length_penalty,
         max_lines=args.max_lines, min_score=args.min_score,
         removal_interval=args.removal_interval,
         removal_threshold=args.removal_threshold)
