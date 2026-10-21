"""
String-art-style rendering off a free point field instead of a circle of pins.

StringArt.py approximates an image by choosing, from the chords of a fixed
circle of pins, the subset whose combined anti-aliased coverage best matches
the image's tone. The circle is only there because physical string has to
wrap around something; a pen has no such constraint. This module keeps
StringArt's matching-pursuit selection algorithm (see StringArt.py's
docstring for why it's done that way -- sparse matrix-vector scoring against
a residual, a quadratic self-penalty term, incremental updates, a greedy
pin-chaining walk for stroke ordering) but drops the circle: candidate lines
connect pairs of points from an arbitrary 2D point field instead of pins on a
boundary.

Two point layouts are offered, per the two options on the table:

- 'grid': a plain rectangular lattice covering the whole canvas. No implicit
  shape bias -- good default for most source images.
- 'rings': concentric circles of points (alternating rings angularly
  staggered by half a step, so candidate lines aren't all radially aligned),
  masked to the enclosing circle like StringArt's canvas. Keeps the
  string-art "look" while allowing lines to start/stop at any ring, not just
  the outer boundary.

Candidate edges are every pair of points whose distance falls in
[min_edge_len, max_edge_len], found with a KD-tree radius query rather than
enumerating all pairs: a free 2D point field is typically denser than a
circle's pin count, so an O(n^2) candidate list stops being affordable long
before an O(n^2) pin-pair list would. min_edge_len drops near-zero-length
edges that add nothing (like StringArt's min_pin_gap); max_edge_len bounds
the candidate count and is the main performance/quality knob here -- wider
intervals let the optimizer reach for long dark strokes but roughly square
the candidate count, since points can connect to many more neighbors. A
dense grid easily reaches millions of candidates, where StringArt's circle
tops out in the tens of thousands -- see _build_render_matrix for the
memory-density tradeoffs (dtype width, avoiding redundant copies of the
same per-edge pixel data) that matter once you're at that scale and
StringArt never had to worry about.

pen_alpha, max_lines, min_score, removal_interval, removal_threshold all mean
exactly what they mean in StringArt.py. length_penalty is new here (see
__init__): a direct, tunable bias toward shorter lines, applied on top of
the SSE-driven scoring rather than by constraining the candidate set itself.

Backlog of other ways to get fine detail without an unaffordable candidate
count at high point density, not yet implemented (see __init__'s
length_penalty for the one that was): a two-tier point field (sparse+long
alongside dense+short); distance-banded candidate sampling (cap neighbors
per distance band per point, so short edges stay dense while long ones stay
sparse, all from one point field); coarse-to-fine two-pass optimization
(select on a coarse grid first, bake it into the residual, then refine with
a fine grid's short edges).
"""
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from skimage.draw import line_aa
from skimage.transform import resize

import Segments


class PointArt:
    LAYOUT_GRID = 'grid'
    LAYOUT_RINGS = 'rings'

    def __init__(self, image_matrix, layout=LAYOUT_GRID, canvas_size=600,
                 grid_rows=30, grid_cols=30, n_rings=14, points_per_ring=28,
                 min_edge_len=8, max_edge_len=120, pen_alpha=0.12,
                 length_penalty=0.0, max_lines=3000, min_score=0.0,
                 removal_interval=250, removal_threshold=0.0, margin=4):
        """
        :param image_matrix: grayscale image (e.g. from imageio mode='F'), any size
        :param layout: 'grid' (rectangular lattice) or 'rings' (concentric
            circles of points, masked to the enclosing circle)
        :param canvas_size: internal working resolution (pixels) of the
            square optimization canvas; point coordinates live in this same
            space, so it also sets the physical scale of the output before
            Segments.scaleBin() normalizes it
        :param grid_rows, grid_cols: point counts for layout='grid'
        :param n_rings, points_per_ring: point counts for layout='rings'
        :param min_edge_len: drop candidate edges shorter than this (near-
            zero-length edges add nothing and just bloat the candidate count)
        :param max_edge_len: drop candidate edges longer than this -- the
            main performance/quality knob: the candidate count grows
            roughly with the square of this relative to point spacing, and
            it caps how long a single dark stroke can be
        :param pen_alpha: darkness contributed by a single pass over a
            pixel, in (0, 1] -- see StringArt.py's docstring for why the
            default is well below 1.0
        :param length_penalty: stylistic preference for shorter lines, on
            top of (not instead of) the SSE-driven self-penalty above. That
            self-penalty already discourages long edges somewhat, since a
            longer line touches more pixels and so costs more to add, but
            only incidentally -- it has no opinion on length itself, just on
            ink cost. length_penalty is a direct, tunable "prefer short,
            all else being equal" knob: each candidate's score is reduced by
            length_penalty * (its pixel length) before ranking, and the same
            amount makes an already-selected long edge correspondingly more
            likely to be dropped in a removal sweep. 0 (default) disables
            it, matching the plain SSE-driven selection. There's no
            canvas-independent good value -- corr/self_penalty scale with
            how many pixels a line touches, so the useful range shifts with
            canvas_size and max_edge_len. Start at 0 and increase in small
            steps (e.g. by 2x) while watching the printed "selected N / M"
            count and the mean of edge_length[selected]: too high and
            selection count collapses toward zero (every candidate's score
            drops below min_score on the very first pick -- PointArtProg.py
            raises rather than trying to write an empty drawing in that
            case)
        :param max_lines: hard cap on the number of lines selected
        :param min_score: stop adding lines once the best remaining
            candidate's score drops to/below this
        :param removal_interval: after this many additions, sweep the
            active lines and drop any whose score is below
            removal_threshold; 0 disables this
        :param removal_threshold: see removal_interval
        :param margin: blank-border pixels between the point field and the
            edge of the canvas
        """
        self.layout = layout
        self.canvas_size = canvas_size
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.n_rings = n_rings
        self.points_per_ring = points_per_ring
        self.pen_alpha = pen_alpha
        self.length_penalty = length_penalty
        self.max_lines = max_lines
        self.min_score = min_score
        self.removal_interval = removal_interval
        self.removal_threshold = removal_threshold

        self._prepare_target(image_matrix, margin)
        self._place_points(margin)
        self._build_candidate_edges(min_edge_len, max_edge_len)
        self._build_render_matrix()
        self._greedy_select()
        self._build_segments()

    # ---- setup ----------------------------------------------------------

    def _prepare_target(self, image_matrix, margin):
        im = np.asarray(image_matrix, dtype=float)
        # Center-crop to square first so a 'rings' canvas isn't stretched.
        h, w = im.shape
        side = min(h, w)
        r0 = (h - side) // 2
        c0 = (w - side) // 2
        im = im[r0:r0 + side, c0:c0 + side]
        im = resize(im, (self.canvas_size, self.canvas_size), anti_aliasing=True)
        lo, hi = im.min(), im.max()
        norm = (im - lo) / (hi - lo) if hi > lo else np.zeros_like(im)
        y = 1.0 - norm

        if self.layout == self.LAYOUT_RINGS:
            yy, xx = np.mgrid[0:self.canvas_size, 0:self.canvas_size]
            cy = cx = self.canvas_size / 2.0
            radius = self.canvas_size / 2.0 - margin
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
            y[~mask] = 0.0
            self.circle_mask = mask
            self.center = (cy, cx)
            self.radius = radius
        else:
            self.circle_mask = np.ones_like(y, dtype=bool)
        self.target = y

    def _place_points(self, margin):
        if self.layout == self.LAYOUT_RINGS:
            self._place_rings(margin)
        else:
            self._place_grid(margin)
        self.n_points = len(self.points)

    def _place_grid(self, margin):
        size = self.canvas_size
        rows = np.linspace(margin, size - 1 - margin, self.grid_rows)
        cols = np.linspace(margin, size - 1 - margin, self.grid_cols)
        gy, gx = np.meshgrid(rows, cols, indexing='ij')
        self.points = np.stack([gy.ravel(), gx.ravel()], axis=1)

    def _place_rings(self, margin):
        cy, cx = self.center
        max_r = self.radius
        radii = np.linspace(max_r / self.n_rings, max_r, self.n_rings)
        pts = []
        for k, r in enumerate(radii):
            # Stagger alternate rings by half a step so candidate edges
            # between neighboring rings aren't all radially aligned.
            offset = (np.pi / self.points_per_ring) * (k % 2)
            angles = np.linspace(0, 2 * np.pi, self.points_per_ring, endpoint=False) + offset
            rows = cy + r * np.sin(angles)
            cols = cx + r * np.cos(angles)
            pts.append(np.stack([rows, cols], axis=1))
        self.points = np.concatenate(pts, axis=0)
        self.points[:, 0] = np.clip(self.points[:, 0], 0, self.canvas_size - 1)
        self.points[:, 1] = np.clip(self.points[:, 1], 0, self.canvas_size - 1)

    def _build_candidate_edges(self, min_edge_len, max_edge_len):
        tree = cKDTree(self.points)
        pairs = tree.query_pairs(r=max_edge_len)
        edges = []
        lengths = []
        for i, j in pairs:
            d = np.hypot(*(self.points[i] - self.points[j]))
            if d >= min_edge_len:
                edges.append((i, j))
                lengths.append(d)
        self.edges = edges
        self.edge_length = np.array(lengths)

    def _build_render_matrix(self):
        """
        Same construction as StringArt._build_render_matrix, with memory
        optimizations that start to matter once the candidate count gets
        into the millions (a dense grid gets there; StringArt's circle
        never did): narrower dtypes (int32/float32 instead of int64/
        float64), and avoiding two sources of redundant large allocations
        that StringArt's version has no reason to care about at its scale:

        - No parallel edge_pixels list duplicating per-edge (pixel, weight)
          data that self.A_csc below already holds, as millions of small
          Python-level numpy array objects (real overhead on top of the
          data itself). The per-edge loop below naturally produces pixel
          indices already grouped and ordered by edge -- exactly CSC's own
          layout -- so self.A_csc is built directly from (data, indices,
          indptr) with no COO sort needed; self.A (CSR, for the row-by-
          pixel slicing apply_delta also needs) is derived from it with one
          real conversion, same one-time cost StringArt already pays to
          build its single CSR matrix.
        - edge_norm_sq is computed here with a bincount over the same flat
          arrays, instead of _greedy_select building a whole second
          same-size sparse matrix via A.multiply(A) just to sum and discard
          it.
        """
        size = self.canvas_size
        n_edges = len(self.edges)
        indices_all, data_all = [], []
        counts = np.empty(n_edges, dtype=np.int64)
        for e, (i, j) in enumerate(self.edges):
            r0, c0 = self.points[i]
            r1, c1 = self.points[j]
            rr, cc, val = line_aa(int(round(r0)), int(round(c0)),
                                  int(round(r1)), int(round(c1)))
            inside = (rr >= 0) & (rr < size) & (cc >= 0) & (cc < size)
            rr, cc, val = rr[inside], cc[inside], val[inside]
            flat = rr * size + cc
            indices_all.append(flat.astype(np.int32))
            data_all.append(val.astype(np.float32))
            counts[e] = flat.shape[0]

        indices_all = np.concatenate(indices_all)
        data_all = np.concatenate(data_all)
        indptr = np.empty(n_edges + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(counts, out=indptr[1:])

        self.edge_norm_sq = np.add.reduceat(data_all.astype(np.float64) ** 2,
                                            indptr[:-1])
        self.edge_norm_sq[counts == 0] = 0.0  # reduceat's empty-slice quirk

        # Already in column-major (edge-then-pixel) order, so this is a
        # direct construction, not a sort -- see docstring above.
        self.A_csc = sparse.csc_matrix((data_all, indices_all, indptr),
                                       shape=(size * size, n_edges))
        self.A = self.A_csc.tocsr()

    # ---- greedy selection -------------------------------------------------

    def _greedy_select(self):
        """Identical matching-pursuit loop to StringArt._greedy_select -- see
        there for the reasoning behind the self-penalty term and the
        removal-sweep sign convention. Nothing here depends on points lying
        on a circle, so the algorithm carries over unchanged."""
        size = self.canvas_size
        n_edges = len(self.edges)
        mask_flat = self.circle_mask.ravel()
        raw = np.zeros(size * size)
        rendered = np.zeros(size * size)
        target_flat = self.target.ravel()
        residual = np.where(mask_flat, target_flat - rendered, 0.0)

        corr = np.asarray(self.A.T @ residual).ravel()
        self_penalty = 0.5 * self.pen_alpha * self.edge_norm_sq
        # Stylistic "prefer shorter lines" bias (see length_penalty in
        # __init__): static per edge like self_penalty, so it's computed
        # once here rather than inside the per-iteration loop. Subtracted
        # from the add-score below (long edges look less attractive to add)
        # and added back into the removal-benefit (long edges look more
        # attractive to drop in a sweep) -- same bias, opposite-signed
        # effect on the two decisions, since the two scores are themselves
        # already sign-flipped mirror images of each other (see the
        # removal_benefit comment below).
        length_bias = -self.length_penalty * self.edge_length
        active = np.zeros(n_edges, dtype=bool)

        def apply_delta(e, sign):
            start, end = self.A_csc.indptr[e], self.A_csc.indptr[e + 1]
            flat = self.A_csc.indices[start:end]
            val = self.A_csc.data[start:end]
            old_r = residual[flat]
            raw[flat] = np.clip(raw[flat] + sign * self.pen_alpha * val, 0.0, None)
            new_rendered = np.minimum(raw[flat], 1.0)
            rendered[flat] = new_rendered
            new_r = np.where(mask_flat[flat], target_flat[flat] - new_rendered, 0.0)
            residual[flat] = new_r
            delta_r = new_r - old_r
            sub = self.A[flat, :]
            corr[:] += np.asarray(sub.T @ delta_r).ravel()

        self.selected = []
        since_removal = 0
        for _ in range(self.max_lines):
            net_score = corr - self_penalty + length_bias
            net_score_masked = np.where(active, -np.inf, net_score)
            best = int(np.argmax(net_score_masked))
            if net_score_masked[best] <= self.min_score:
                break
            active[best] = True
            apply_delta(best, +1)
            self.selected.append(best)
            since_removal += 1

            if self.removal_interval and since_removal >= self.removal_interval:
                since_removal = 0
                active_idx = np.flatnonzero(active)
                removal_benefit = -(corr[active_idx] + self_penalty[active_idx]) - length_bias[active_idx]
                drop = active_idx[removal_benefit > self.removal_threshold]
                for e in drop:
                    active[e] = False
                    apply_delta(e, -1)
                    self.selected.remove(e)

        self.active = active
        self.rendered_canvas = rendered.reshape(size, size)
        print("PointArt: selected %d / %d candidate lines" %
              (len(self.selected), n_edges))

    # ---- output -----------------------------------------------------------

    def _build_segments(self):
        """
        Same bespoke greedy walk as StringArt._build_segments (hop through
        shared points while edges remain there, jump to the nearest point
        with unused edges once they run out) -- the dense point-sharing that
        made StringArt avoid EuclidMST applies here too, if anything more
        so, since a 'grid'/'rings' point can be shared by edges in every
        direction rather than just along one circle.
        """
        point_edges = [set() for _ in range(self.n_points)]
        for e in self.selected:
            i, j = self.edges[e]
            point_edges[i].add(e)
            point_edges[j].add(e)
        active_points = {p for p in range(self.n_points) if point_edges[p]}

        def other_point(e, p):
            i, j = self.edges[e]
            return j if p == i else i

        def consume(p, e):
            other = other_point(e, p)
            point_edges[p].discard(e)
            point_edges[other].discard(e)
            if not point_edges[p]:
                active_points.discard(p)
            if not point_edges[other]:
                active_points.discard(other)
            return other

        ordered = []
        if self.selected:
            current = self.edges[self.selected[0]][0]
            if current not in active_points:
                current = next(iter(active_points))
            while active_points:
                if point_edges[current]:
                    e = next(iter(point_edges[current]))
                    nxt = consume(current, e)
                    ordered.append((current, nxt))
                    current = nxt
                else:
                    cands = np.array(list(active_points))
                    d = np.sum((self.points[cands] - self.points[current]) ** 2, axis=1)
                    current = int(cands[np.argmin(d)])

        self.segments = Segments.Segments()
        for p_from, p_to in ordered:
            self.segments.append(np.array([self.points[p_from], self.points[p_to]]))
