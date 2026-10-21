import sys

import imageio.v2 as imageio
import Hilbert


def main(ifile_name, ofile_name1, bin_fn="bfile.bin"):
    im = imageio.imread(ifile_name, mode='F')
    # im = ndimage.interpolation.zoom(im,0.5)
    h = Hilbert.Hilbert(im)
    h.segments.savePreview(ofile_name1, scale=2)
    h.segments.scaleBin()
    h.binWrite(bin_fn)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Error: unknown usage")
