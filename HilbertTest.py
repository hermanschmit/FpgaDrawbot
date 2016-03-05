__author__ = 'herman'


import matplotlib
matplotlib.use('Agg')
from optparse import OptionParser
import Segments

def initOptions(parser):
    parser.add_option('-n', '--level', dest='level', default=8,
                      type='int',
                      help=('determines the length of one side of the square by '
                            '2^LEVEL. There is a restriction that LEVEL <= 8. Using large level '
                            'values can take a long time or create enormous / resource '
                            'intensive plots. default=%default'))
    parser.add_option('-m', '--moore', action="store_true", dest="moore", default=False,
                      help=('chooses moore curve rather than Hilbert'))

def checkOptions(options, args, parser):
    options.max = 0
    if options.level < 1:
        parser.error('--level must by greater than 0')
    if len(args) > 1:
        parser.error('Only one filename output argument')

########################################
# These functions refactored from those available at
# wikipedia for Hilbert curves http://en.wikipedia.org/wiki/Hilbert_curve


def d2xy(n, d, moore=False):
    """
    take a d value in [0, n**2 - 1] and map it to
    an x, y value (e.g. c, r).
    """
    assert(d <= n**2 - 1)
    t = d
    x = y = 0
    s = 1
    while s < n:
        rx = 1 & (t / 2)
        ry = 1 & (t ^ rx)
        if moore and s*2 >= n:
            x, y = rot_moore(s, x, y, rx, ry)
        else:
            x, y = rot(s, x, y, rx, ry)
        # if not moore or s*2 < n:
        x += s * rx
        y += s * ry
        t /= 4
        s *= 2
    return x, y


def xy2d(n, x, y, moore=False):
    d = 0
    s = n//2
    while s > 0:
        rx = (x & s) > 0
        ry = (y & s) > 0
        d += s * s * ((3 * rx) ^ ry)
        if moore and s*2 >= n:
            x, y = rot_mooreI(s, x, y, rx, ry)
        else:
            x, y = rot(s, x, y, rx, ry)
        s //= 2
    return d


def rot(n, x, y, rx, ry):
    """
    rotate/flip a quadrant appropriately
    """
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        return y, x
    return x, y


def rot_moore(n, x, y, rx, ry):
    """
    rotate/flip a quadrant appropriately

    """
    if rx == 0:
        return n-1-y,x
    else:
        return y,n-1-x


def rot_mooreI(n, x, y, rx, ry):
    if rx == 0:
        return y,n-1-x
    else:
        return n-1-y,x
#
########################################

def hilbert(args,options):
    seg = Segments.Segments()
    s = []
    n = (1<<options.level)
    for d in xrange(n**2):
        x,y = d2xy(n, d, options.moore)
        s.append([x,y])
    seg.append(s)
    return seg


def main():
    usage = ('usage: %prog --level=LEVEL\n\n')
    parser = OptionParser(usage=usage)
    initOptions(parser)
    options, args = parser.parse_args()
    checkOptions(options, args, parser)


    seg = hilbert(args, options)
    seg.scaleBin()
    seg.binWrite(args[0])

if __name__ == '__main__':
    main()