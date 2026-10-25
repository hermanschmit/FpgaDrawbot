import argparse

import imageio.v2 as imageio

import HilbertWave


def main(ifile_name, ofile_name1, bin_fn="bfile.bin", svg_fn=None, **kwargs):
    im = imageio.imread(ifile_name, mode='F')
    hw = HilbertWave.HilbertWave(im, **kwargs)
    if svg_fn is not None:
        hw.segments.svgwrite(svg_fn)
    hw.segments.savePreview(ofile_name1)
    hw.segments.scaleBin()
    hw.binWrite(bin_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Space-filling curve with a darkness-driven wave on every segment')
    parser.add_argument('input_image_fn', help='Input image')
    parser.add_argument('output_image_fn', help='Output preview (.svg for vector, else raster)')
    parser.add_argument('output_bin_fn', nargs='?', default='bfile.bin', help='Output bin file')
    parser.add_argument('--svg', default=None, help='Optional output SVG')
    parser.add_argument('--cells', type=int, default=96, help='Grid cells along the long side')
    parser.add_argument('--amplitude', type=float, default=1.0,
                        help='Wave amplitude at full black, as a fraction of the '
                             'no-overlap envelope (clip mode allows >1)')
    parser.add_argument('--envelope', choices=['scale', 'clip'], default='scale',
                        help='scale: waves taper to fit the envelope; clip: '
                             'full waves cut flat at the envelope')
    parser.add_argument('--pen_mm', type=float, default=0.5,
                        help='Pen width; neighbors are kept at least this far apart')
    parser.add_argument('--size_mm', type=float, default=342.0,
                        help='Drawn size of the long side')
    parser.add_argument('--freq_mod', type=float, default=2.0,
                        help='Periods per segment go from 1 (white) to 1 + this (black)')
    parser.add_argument('--wave', choices=['sine', 'triangle', 'square'], default='sine')
    parser.add_argument('--ppw', type=int, default=8, help='Points per period for sine')
    parser.add_argument('--gamma', type=float, default=0.7,
                        help='Darkness gamma (<1 darkens midtones, >1 lightens)')
    parser.add_argument('--white_cut', type=float, default=0.1,
                        help='Darkness (after auto-levels) below this is drawn straight')
    args = parser.parse_args()

    main(args.input_image_fn, args.output_image_fn, args.output_bin_fn, svg_fn=args.svg,
         cells=args.cells, amplitude=args.amplitude, freq_mod=args.freq_mod, wave=args.wave,
         ppw=args.ppw, envelope=args.envelope, pen_mm=args.pen_mm, size_mm=args.size_mm,
         gamma=args.gamma, white_cut=args.white_cut)
