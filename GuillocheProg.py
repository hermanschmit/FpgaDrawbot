import argparse

import imageio.v2 as imageio

import Guilloche


def main(ifile_name, ofile_name1, bin_fn="bfile.bin", svg_fn=None, preview_scale=2, **kwargs):
    im = imageio.imread(ifile_name, mode='F')
    g = Guilloche.Guilloche(im, **kwargs)
    if svg_fn is not None:
        g.segments.svgwrite(svg_fn)
    g.segments.savePreview(ofile_name1, scale=preview_scale)
    g.segments.scaleBin()
    g.binWrite(bin_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Guilloche-style single spiral line drawing')
    parser.add_argument('input_image_fn', help='Input image')
    parser.add_argument('output_image_fn', help='Output preview PNG')
    parser.add_argument('output_bin_fn', nargs='?', default='bfile.bin', help='Output bin file')
    parser.add_argument('--svg', default=None, help='Optional output SVG')
    parser.add_argument('--preview_scale', type=int, default=2, help='Preview PNG upscale factor')
    parser.add_argument('--canvas_size', type=int, default=800, help='Long image side in pixels')
    parser.add_argument('--turns', type=int, default=80, help='Spiral turns from center to edge')
    parser.add_argument('--shape', choices=['ellipse', 'circle'], default='ellipse')
    parser.add_argument('--wavelength', type=float, default=1.0,
                        help='Tone-wave wavelength at white, in pitches')
    parser.add_argument('--amplitude', type=float, default=0.9,
                        help='Tone-wave amplitude at full black, in pitches')
    parser.add_argument('--freq_mod', type=float, default=1.0,
                        help='Extra wave frequency at full black (1 = doubled)')
    parser.add_argument('--rosette_amp', type=float, default=0.0,
                        help='Decorative rosette amplitude in pitches (>0.5 makes turns cross)')
    parser.add_argument('--rosette_lobes', type=int, default=24, help='Rosette lobes per turn')
    parser.add_argument('--rosette_twist', type=float, default=0.5,
                        help='Rosette phase shift per turn, in lobes')
    parser.add_argument('--relief', type=float, default=0.0,
                        help='Radial displacement at full white, in pitches')
    parser.add_argument('--gamma', type=float, default=0.7,
                        help='Darkness gamma (<1 darkens midtones, >1 lightens)')
    parser.add_argument('--white_cut', type=float, default=0.1,
                        help='Darkness (after auto-levels) below this is drawn as a plain line')
    args = parser.parse_args()

    main(args.input_image_fn, args.output_image_fn, args.output_bin_fn,
         svg_fn=args.svg, preview_scale=args.preview_scale,
         canvas_size=args.canvas_size, turns=args.turns, shape=args.shape,
         wavelength=args.wavelength, amplitude=args.amplitude, freq_mod=args.freq_mod,
         rosette_amp=args.rosette_amp, rosette_lobes=args.rosette_lobes,
         rosette_twist=args.rosette_twist, relief=args.relief, gamma=args.gamma,
         white_cut=args.white_cut)
