import math
from numba import jit
import numpy
from scipy import spatial

import Segments


def simplify(s):
    print("Start len: "+str(len(s)))
    new_s = Segments.simplify_segment(s)
    print("End len: "+str(len(new_s)))
    return new_s

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

def threeOptLoop(seg0, maxdelta=10):
    totald = 0
    for a in range(len(seg0) - 3):
        for c in range(a + 1, min(a + maxdelta, len(seg0) - 2)):
            for e in range(c + 1, min(c + maxdelta, len(seg0) - 1)):
                delta, seg0 = threeOpt(seg0, a, c, e)
                totald += delta
    return totald, seg0

def _threeOptLocalMove(seg0, position_to_id, a, c, e, twoOpt=False):
    """
    Same move/acceptance decision as threeOpt (kept as a separate function
    rather than reusing threeOpt directly so threeOpt/threeOptLoop are not
    touched at all by this), but MUTATES seg0 and position_to_id IN PLACE
    over just the affected [a+1:e+1] range, rather than rebuilding the
    entire array via numpy.concatenate as threeOpt does. That full-array
    rebuild costs O(n) per accepted move regardless of how small e-a is,
    which dominates threeOptLocal's total time at large n; the untouched
    prefix/suffix never need to be copied at all. This is only safe because
    threeOptLocal's KD-tree is built on a separate, frozen coordinate copy
    (see identity_coords below) rather than on seg0 itself, so mutating
    seg0 can't invalidate it the way it would have before that change.
    The caller is expected to already own seg0/position_to_id (i.e. not
    share them with something that expects them to stay unchanged).
    """
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

    if new - orig >= -0.01:
        return 0

    def reorder_inplace(arr):
        # bc/de must be copied before writing back: they're views into the
        # very range arr[a+1:e+1] we're about to overwrite in place.
        bc = arr[a + 1:c + 1].copy()
        de = arr[c + 1:e + 1].copy()
        if abcedf == new:
            arr[a + 1:e + 1] = numpy.concatenate((bc, numpy.flipud(de)))
        elif acbdef == new:
            arr[a + 1:e + 1] = numpy.concatenate((numpy.flipud(bc), de))
        elif acbedf == new:
            arr[a + 1:e + 1] = numpy.concatenate((numpy.flipud(bc), numpy.flipud(de)))
        elif adebcf == new:
            arr[a + 1:e + 1] = numpy.concatenate((de, bc))
        elif adecbf == new:
            arr[a + 1:e + 1] = numpy.concatenate((de, numpy.flipud(bc)))
        elif aedbcf == new:
            arr[a + 1:e + 1] = numpy.concatenate((numpy.flipud(de), bc))
        else:
            assert(False)

    reorder_inplace(seg0)
    reorder_inplace(position_to_id)
    return new - orig


def threeOptLocal(seg0, nn=5, twoOpt=False):
    # Own copy from here on: _threeOptLocalMove mutates seg0 in place, and
    # the caller's input array must not be touched.
    seg0 = numpy.array(seg0, copy=True)
    n = len(seg0)
    totald = 0
    # Querying for more neighbors than exist makes scipy pad the result with
    # an out-of-bounds sentinel index (len(seg0)), which can otherwise reach
    # seg0[e]/seg0[c] below and raise IndexError.
    nn = min(nn, n)

    # A 2-/3-opt move only reorders which point sits at which tour position
    # -- it never moves a point's actual (x,y) -- so a KD-tree built on the
    # ORIGINAL, immutable coordinates never goes stale and only needs to be
    # built once. position_to_id/id_to_position track where each point
    # currently sits in the tour, updated incrementally as moves are
    # accepted, instead of rebuilding the tree (and losing that update, an
    # O(n log n) rebuild) after every single accepted move.
    identity_coords = seg0.copy()
    identity_tree = spatial.cKDTree(identity_coords)
    position_to_id = numpy.arange(n)
    id_to_position = numpy.arange(n)

    for a in range(n - 1):
        id_a = position_to_id[a]
        _, neighbor_ids = identity_tree.query(identity_coords[id_a], nn)
        neighbor_positions = numpy.sort(id_to_position[neighbor_ids])
        nn_list = list(neighbor_positions)
        while len(nn_list) > 2:
            c = nn_list.pop(0)
            # Reset per candidate c: if no e below ends up making a move
            # (e.g. every candidate fails the filters below), delta must
            # reflect that nothing happened here, not linger from an
            # unrelated earlier c or a -- otherwise this can either crash
            # (first time it happens at all) or silently break out early on
            # a stale value from a previous iteration.
            delta = 0
            if c <= a + 1:
                continue
            for e in nn_list:
                if c + 1 == e:
                    continue
                if e + 1 == n:
                    continue
                # seg0/position_to_id are mutated in place; no reassignment.
                delta = _threeOptLocalMove(seg0, position_to_id, a, c, e, twoOpt=twoOpt)
                totald += delta
                if delta < 0:
                    changed = numpy.arange(a + 1, e + 1)
                    id_to_position[position_to_id[changed]] = changed
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
