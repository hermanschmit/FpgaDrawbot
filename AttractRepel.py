import math

import numpy as np
from numba import jit


#@jit("float64(float64)", nopython=True, cache=True)
def _LennardJones(r):
    force = (r ** 12 - r ** 6)
    return force


# @jit(cache=True)
def _LennardJones2(R0, i_pt, pi2xij, xij, Fa):
    fij = (i_pt - xij) / pi2xij
    fij *= _LennardJones(R0 / pi2xij) * Fa
    return fij


@jit
def _density(pixel_val):
    x = 256 / (256 - pixel_val)
    return x


@jit
def _R0_val(i_pt, imin, R0, R1_R0):
    i_pt0 = max(min(round(i_pt[0]), imin.shape[0] - 1), 0)
    i_pt1 = max(min(round(i_pt[1]), imin.shape[1] - 1), 0)
    r0 = R0 * _density(imin[i_pt0][i_pt1])
    return r0, R1_R0 * r0


@jit(nopython=True, cache=True)
def _distABtoP(a_pt, b_pt, p_pt):
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

    if dist == 0:
        dist = 1e-10

    return dist, (x, y)


def attract_repel_segment(s, im, maze_path, kdtree, R0, R1_R0, Fa, chunk=2000):
    fi_l = []
    for i in range(s, s + chunk):
        if i >= len(maze_path):
            continue
        fi = np.array([0., 0.])
        i_pt = maze_path[i]
        r0, r1 = _R0_val(i_pt, im, R0, R1_R0)
        neighbors = kdtree.query_ball_point(i_pt, r1)
        n_set = set(neighbors)
        for x in neighbors:
            n_set.add(x - 1)

        for j in n_set:
            if j < 0 or j == len(maze_path) - 1:
                continue
            if j < i - 2 or j >= i + 2:
                j_pt = maze_path[j]
                jp1_pt = maze_path[j + 1]
                pi2xij, xij = _distABtoP(j_pt, jp1_pt, i_pt)
                if pi2xij < r1:
                    fij = _LennardJones2(r0, i_pt, pi2xij, xij, Fa)
                    fi += fij
        fi_l.append(fi)

    return fi_l

def attract_repel_global(im, maze_path, R0, R1_R0, Fa):
    fi_l = []
    for i in range(0, len(maze_path)):

        fi = np.array([0., 0.])
        i_pt = maze_path[i]
        r0, r1 = _R0_val(i_pt, im, R0, R1_R0)

        for j in range(0,len(maze_path)):
            if j == len(maze_path) - 1:
                continue
            if j < i - 2 or j >= i + 2:
                j_pt = maze_path[j]
                jp1_pt = maze_path[j + 1]
                pi2xij, xij = _distABtoP(j_pt, jp1_pt, i_pt)
                if pi2xij < r1:
                    fij = _LennardJones2(r0, i_pt, pi2xij, xij, Fa)
                    fi += fij
        fi_l.append(fi)

    return fi_l

@jit
def _repulse(r):
    force = (r ** 12)
    return force


# def attract_repel_segment(s, im, maze_path, kdtree, R0, R1_R0, Fa, chunk=2000):
#    return _attract_repel_segment(s,im,maze_path,kdtree,R0,R1_R0,Fa,chunk)

def boundary(r0_b, Fo, maze_path, boundary_seg):
    """
    This is the brute force version
    Returns:
    """
    rArray = np.empty([len(maze_path), 2])

    R1 = 2.0 * r0_b

    for i in range(0,
                   len(maze_path)):
        fi = np.array([0., 0.])
        pi = np.array(maze_path[i])
        for j in range(0,
                       len(boundary_seg) - 1):
            j_pt = boundary_seg[j]
            jp1_pt = boundary_seg[j + 1]
            pi2xij, xij = _distABtoP(j_pt, jp1_pt, pi)
            if pi2xij < R1:
                fij = (pi - xij) / max(0.00001, pi2xij)
                fij *= _repulse(r0_b / pi2xij) * Fo
                fi += fij
        rArray[i] = np.array(fi)
    return rArray
