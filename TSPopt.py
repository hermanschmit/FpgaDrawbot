import math
from numba import jit
import numpy
from scipy import spatial


def _is_on(a, b, c, tol=1e-5):
    "Return true iff point c intersects the line segment from a to b."
    # (or the degenerate case that all 3 points are coincident)
    return (_collinear(a, b, c, tol)
            and (_within(a[0], c[0], b[0]) if a[0] != b[0] else
                 _within(a[1], c[1], b[1])))


def _collinear(a, b, c, tol=1e-5):
    "Return true iff a, b, and c all lie on the same line."
    return abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) < tol


def _within(p, q, r):
    "Return true iff q is between p and r (inclusive)."
    return p <= q <= r or r <= q <= p


def simplify(s):
    print("Start len: "+str(len(s)))
    if len(s) < 3:
        return s
    new_s = []
    p0 = s[0]
    p1 = s[1]
    new_s.append(p0)
    for p2 in s[2:]:
        if _is_on(p0, p2, p1):
            p1 = p2
        else:
            new_s.append(p1)
            p0 = p1
            p1 = p2
    new_s.append(p2)
    print("End len: "+str(len(new_s)))

    return numpy.array(new_s)

@jit
def ptlen(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

@jit
def _ptlen_local(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

@jit
def distanceCombinations(a_pt, b_pt, c_pt, d_pt, e_pt, f_pt):
    """
    Partitioned for JIT
    """
    ab_len = _ptlen_local(a_pt, b_pt)
    cd_len = _ptlen_local(c_pt, d_pt)
    ef_len = _ptlen_local(e_pt, f_pt)
    ac_len = _ptlen_local(a_pt, c_pt)
    ad_len = _ptlen_local(a_pt, d_pt)
    ae_len = _ptlen_local(a_pt, e_pt)
    bd_len = _ptlen_local(b_pt, d_pt)
    be_len = _ptlen_local(b_pt, e_pt)
    bf_len = _ptlen_local(b_pt, f_pt)
    ce_len = _ptlen_local(c_pt, e_pt)
    cf_len = _ptlen_local(c_pt, f_pt)
    df_len = _ptlen_local(d_pt, f_pt)

    abcdef = ab_len + cd_len + ef_len
    abcedf = ab_len + ce_len + df_len  # 2-opt
    acbdef = ac_len + bd_len + ef_len  # 2-opt
    acbedf = ac_len + be_len + df_len
    adebcf = ad_len + be_len + cf_len
    adecbf = ad_len + ce_len + bf_len
    aedbcf = ae_len + bd_len + cf_len

    return abcdef, abcedf, acbdef, acbedf, adebcf, adecbf, aedbcf

@jit
def threeOpt(seg0, a, c, e, twoOpt=False):
    a_pt = seg0[a]
    b_pt = seg0[a + 1]
    c_pt = seg0[c]
    d_pt = seg0[c + 1]
    e_pt = seg0[e]
    f_pt = seg0[e + 1]

    (orig, abcedf, acbdef, acbedf, adebcf, adecbf, aedbcf) = \
        distanceCombinations(a_pt, b_pt, c_pt, d_pt, e_pt, f_pt)

    if twoOpt:
        new = min(abcedf, acbdef)
    else:
        new = min(abcedf, acbdef, acbedf, adebcf, adecbf, aedbcf)
    if new - orig < -0.01:
        aseg = seg0[:a + 1]
        bcseg = seg0[a + 1:c + 1]
        deseg = seg0[c + 1:e + 1]
        fseg = seg0[e + 1:]
        if abcedf == new:
            seg0 = numpy.concatenate((aseg,
                                      bcseg,
                                      numpy.flipud(deseg),
                                      fseg))
        elif acbdef == new:
            seg0 = numpy.concatenate((aseg,
                                      numpy.flipud(bcseg),
                                      deseg,
                                      fseg))
        elif acbedf == new:
            seg0 = numpy.concatenate((aseg,
                                      numpy.flipud(bcseg),
                                      numpy.flipud(deseg),
                                      fseg))
        elif adebcf == new:
            seg0 = numpy.concatenate((aseg,
                                      deseg,
                                      bcseg,
                                      fseg))
        elif adecbf == new:
            seg0 = numpy.concatenate((aseg,
                                      deseg,
                                      numpy.flipud(bcseg),
                                      fseg))
        elif aedbcf == new:
            seg0 = numpy.concatenate((aseg,
                                      numpy.flipud(deseg),
                                      bcseg,
                                      fseg))
        else:
            assert(False)
        return new - orig, seg0
    else:
        return 0, seg0

@jit
def threeOptLoop(seg0, maxdelta=10):
    totald = 0
    for a in range(len(seg0) - 3):
        for c in range(a + 1, min(a + maxdelta, len(seg0) - 2)):
            for e in range(c + 1, min(c + maxdelta, len(seg0) - 1)):
                delta, seg0 = threeOpt(seg0, a, c, e)
                totald += delta
    return totald, seg0

#@jit
def threeOptLocal(seg0, nn=5, twoOpt=False):
    totald = 0
    kdtree = spatial.cKDTree(seg0)
    for a in range(len(seg0) - 1):
        _, n_neighbor_i = kdtree.query(seg0[a], nn)
        n_neighbor_i.sort()
        nn_list = list(n_neighbor_i)
        while len(nn_list) > 2:
            c = nn_list.pop(0)
            if c <= a + 1:
                continue
            for e in nn_list:
                if c + 1 == e:
                    continue
                if e + 1 == len(seg0):
                    continue
                delta,seg0 = threeOpt(seg0, a, c, e, twoOpt=twoOpt)
                totald += delta
                if delta < 0:
                    kdtree = spatial.cKDTree(seg0)
                    break
            if delta < 0:
                break

    return totald, seg0

@jit
def distABtoP(a_pt, b_pt, p_pt):

    seg_x = b_pt[0] - a_pt[0]
    seg_y = b_pt[1] - a_pt[1]

    seglen_sqrd = seg_x * seg_x + seg_y * seg_y

    u = ((p_pt[0] - a_pt[0]) * seg_x + (p_pt[1] - a_pt[1]) * seg_y) / float(seglen_sqrd)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x = a_pt[0] + u * seg_x
    y = a_pt[1] + u * seg_y

    dx = x - p_pt[0]
    dy = y - p_pt[1]

    dist = math.sqrt(dx * dx + dy * dy)

    return dist, (x, y)
