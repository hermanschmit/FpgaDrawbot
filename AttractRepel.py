import math

import numpy as np
from numba import jit, prange


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


@jit(nopython=True, cache=True)
def _seg_dist(ax, ay, bx, by, px, py):
    """
    Distance from point (px,py) to segment (ax,ay)-(bx,by), clamped to the
    segment, plus the closest point on it. Compiled helper shared by the
    kernels below -- inlined into their prange loops by numba.
    """
    seg_x = bx - ax
    seg_y = by - ay
    seglen_sqrd = seg_x * seg_x + seg_y * seg_y
    if seglen_sqrd == 0.0:
        u = 0.0
    else:
        u = ((px - ax) * seg_x + (py - ay) * seg_y) / seglen_sqrd
    if u > 1.0:
        u = 1.0
    elif u < 0.0:
        u = 0.0
    x = ax + u * seg_x
    y = ay + u * seg_y
    dx = x - px
    dy = y - py
    dist = math.sqrt(dx * dx + dy * dy)
    if dist == 0.0:
        dist = 1e-10
    return dist, x, y


def r0_r1_vectorized(maze_path, im, R0, R1_R0):
    """Vectorized replacement for calling _R0_val once per point in a loop."""
    ix = np.clip(np.round(maze_path[:, 0]).astype(np.int64), 0, im.shape[0] - 1)
    iy = np.clip(np.round(maze_path[:, 1]).astype(np.int64), 0, im.shape[1] - 1)
    r0 = R0 * (256.0 / (256.0 - im[ix, iy]))
    return r0, R1_R0 * r0


def build_candidate_csr(maze_path, kdtree, r1_arr):
    """
    For every point i, the candidate set of path-segment start indices to
    test against: every neighbor within r1 of i, plus each neighbor's
    predecessor (so the segment leading into a nearby point is tested too),
    minus indices that can't start a segment (negative, or the last point).
    Flattened to CSR form (offsets/flat_j) so the hot loop below can run
    compiled instead of as a Python-level double loop.
    """
    n = len(maze_path)
    neighbor_lists = kdtree.query_ball_point(maze_path, r1_arr)
    offsets = np.zeros(n + 1, dtype=np.int64)
    candidates = []
    last = n - 1
    for i, nbrs in enumerate(neighbor_lists):
        # np.union1d both unions with the shifted (predecessor) indices and
        # dedupes/sorts in one C-level pass -- much faster than building a
        # Python set() per point, especially when a point has many neighbors.
        arr = np.asarray(nbrs, dtype=np.int64)
        cand = np.union1d(arr, arr - 1)
        cand = cand[(cand >= 0) & (cand != last)]
        offsets[i + 1] = offsets[i] + len(cand)
        candidates.append(cand)
    flat_j = np.concatenate(candidates) if candidates else np.zeros(0, dtype=np.int64)
    return offsets, flat_j


@jit(nopython=True, parallel=True, cache=True)
def attract_repel_kernel(maze_path, r0_arr, r1_arr, offsets, flat_j, Fa):
    """
    Compiled, multi-threaded replacement for the Python loop in
    attract_repel_segment: for every point i, sums a Lennard-Jones-style
    force from every candidate path segment (j, j+1) within r1 of i. Reads
    a frozen maze_path and writes only its own output row, so this is
    embarrassingly parallel -- no locking or synchronization needed.
    """
    n = maze_path.shape[0]
    out = np.zeros((n, 2))
    for i in prange(n):
        xi = maze_path[i, 0]
        yi = maze_path[i, 1]
        r0 = r0_arr[i]
        r1 = r1_arr[i]
        fx = 0.0
        fy = 0.0
        for k in range(offsets[i], offsets[i + 1]):
            j = flat_j[k]
            if i - 2 <= j < i + 2:
                continue
            dist, xij, yij = _seg_dist(maze_path[j, 0], maze_path[j, 1],
                                       maze_path[j + 1, 0], maze_path[j + 1, 1],
                                       xi, yi)
            if dist < r1:
                ratio = r0 / dist
                force = (ratio ** 12 - ratio ** 6) * Fa / dist
                fx += (xi - xij) * force
                fy += (yi - yij) * force
        out[i, 0] = fx
        out[i, 1] = fy
    return out


@jit(nopython=True, parallel=True, cache=True)
def boundary_kernel(maze_path, boundary_seg, R0_B, Fo):
    """
    Compiled, multi-threaded replacement for Maze.boundary_slow: a pure
    repulsive force from the (small, fixed) bounding rectangle.
    """
    n = maze_path.shape[0]
    nseg = boundary_seg.shape[0] - 1
    R1 = 2.0 * R0_B
    out = np.zeros((n, 2))
    min_dist = np.empty(n)
    for i in prange(n):
        xi = maze_path[i, 0]
        yi = maze_path[i, 1]
        fx = 0.0
        fy = 0.0
        local_min = 1.0e300
        for j in range(nseg):
            dist, xij, yij = _seg_dist(boundary_seg[j, 0], boundary_seg[j, 1],
                                       boundary_seg[j + 1, 0], boundary_seg[j + 1, 1],
                                       xi, yi)
            if dist < local_min:
                local_min = dist
            if dist < R1:
                ratio = R0_B / dist
                denom = dist if dist > 0.00001 else 0.00001
                force = (ratio ** 12) * Fo / denom
                fx += (xi - xij) * force
                fy += (yi - yij) * force
        out[i, 0] = fx
        out[i, 1] = fy
        min_dist[i] = local_min
    return out, min_dist
