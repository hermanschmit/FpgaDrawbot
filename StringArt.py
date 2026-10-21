"""
String-art-inspired continuous-line rendering.

Adapts "String Art: Towards Computational Fabrication of String Images"
(Birsak, Rist, Wonka, Musialski; EUROGRAPHICS 2018) for a pen-based drawbot
instead of physical string wound around pins. Pins are placed evenly around
a circle; the drawable "edges" are the straight chords between pin pairs. A
greedy algorithm selects a subset of chords whose combined anti-aliased
rendering approximates the target image, then the selected chords are
ordered into pen strokes by walking from chord to chord through shared pins
(see _build_segments) and concatenated.

Three deliberate departures from the paper, since a pen isn't a string:

- No Eulerian-path / auxiliary-edge machinery (paper Sec. 6): that exists
  solely because physical thread must wind around pins without being cut.
  A pen can simply lift and move, so disjoint strokes just need a
  reasonable travel order rather than a true Eulerian circuit -- see
  _build_segments for why that's done with a bespoke greedy walk here
  rather than this repo's usual EuclidMST (used by Canny.py/Skeleton.py).
- One canvas resolution instead of the paper's supersample+box-downfilter
  split (Sec. 4.4): that split exists because their output has to land on
  a fixed physical raster (thread pitch, requirement R2). Our output is
  vector line segments between exact pin coordinates, with no fixed output
  grid to hit -- one reasonably sized canvas gives the same "more gray
  levels via anti-aliasing" benefit without the extra tier.
- Edge selection scores every candidate at once via one sparse
  matrix-vector product against the current residual (classic matching
  pursuit), incrementally updated after each pick by touching only the
  pixels that actually changed -- instead of the paper's literal
  per-candidate evaluation of the exact clamped objective every iteration
  (Algorithm 1), which would mean a Python loop over thousands of edges
  per iteration.

`pen_alpha` generalizes the paper's assumption that a string is fully
opaque -- one pass = full black, requirement (R1) -- into "one pass
contributes this much darkness". Set it below 1 for a pen that needs
several overlapping passes over the same spot to reach full black.
"""
import numpy as np
from scipy import sparse
from skimage.draw import line_aa
from skimage.transform import resize

import Segments


class StringArt:
    def __init__(self, image_matrix, n_pins=200, canvas_size=600, min_pin_gap=4,
                 pen_alpha=0.12, max_lines=2500, min_score=0.0,
                 removal_interval=250, removal_threshold=0.0, margin=4):
        """
        :param image_matrix: grayscale image (e.g. from imageio mode='F'), any size
        :param n_pins: number of pins evenly spaced around the circle
        :param canvas_size: internal working resolution (pixels) of the square
            optimization canvas; pin/segment coordinates live in this same
            space, so it also sets the physical scale of the output before
            Segments.scaleBin() normalizes it
        :param min_pin_gap: skip candidate chords between pins fewer than
            this many positions apart around the circle -- near-tangent
            chords add little to the image and just bloat the candidate count
        :param pen_alpha: darkness contributed by a single pass over a
            pixel, in (0, 1]. 1.0 matches the paper's opaque-string
            assumption (one pass = full black); lower it for a pen that
            needs several overlapping passes to reach full black. The
            default is well below 1.0 even as a starting point: at 1.0 a
            single chord can fully saturate every pixel it touches in one
            step, which empirically starves the greedy search of room to
            build up fine tonal detail (confirmed by comparison -- see
            StringArtProg.py's out2) well before it should stop, regardless
            of how dark your actual pen is
        :param max_lines: hard cap on the number of chords selected
        :param min_score: stop adding chords once the best remaining
            candidate's score drops to/below this (0 = stop once no
            remaining chord would reduce the residual at all)
        :param removal_interval: after this many additions, sweep the
            active chords and drop any whose score is below
            removal_threshold (an over-saturated chord actively hurting the
            fit -- the paper's alternating removal phase); 0 disables this
        :param removal_threshold: see removal_interval
        :param margin: blank-border pixels between the pin circle and the
            edge of the canvas
        """
        self.n_pins = n_pins
        self.canvas_size = canvas_size
        self.pen_alpha = pen_alpha
        self.max_lines = max_lines
        self.min_score = min_score
        self.removal_interval = removal_interval
        self.removal_threshold = removal_threshold

        self._prepare_target(image_matrix, margin)
        self._place_pins(margin)
        self._build_candidate_edges(min_pin_gap)
        self._build_render_matrix()
        self._greedy_select()
        self._build_segments()

    # ---- setup ----------------------------------------------------------

    def _prepare_target(self, image_matrix, margin):
        im = np.asarray(image_matrix, dtype=float)
        # Center-crop to square first so the circle isn't stretched.
        h, w = im.shape
        side = min(h, w)
        r0 = (h - side) // 2
        c0 = (w - side) // 2
        im = im[r0:r0 + side, c0:c0 + side]
        im = resize(im, (self.canvas_size, self.canvas_size), anti_aliasing=True)
        # Dark pixels in the source photo need ink. Normalize defensively
        # by the image's own range rather than assuming a fixed 0-255
        # scale, since resize()'s output range can drift slightly.
        lo, hi = im.min(), im.max()
        norm = (im - lo) / (hi - lo) if hi > lo else np.zeros_like(im)
        y = 1.0 - norm

        yy, xx = np.mgrid[0:self.canvas_size, 0:self.canvas_size]
        cx = cy = self.canvas_size / 2.0
        radius = self.canvas_size / 2.0 - margin
        self.circle_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        self.radius = radius
        self.center = (cy, cx)
        y[~self.circle_mask] = 0.0
        self.target = y

    def _place_pins(self, margin):
        angles = np.linspace(0, 2 * np.pi, self.n_pins, endpoint=False)
        cy, cx = self.center
        rows = cy + self.radius * np.sin(angles)
        cols = cx + self.radius * np.cos(angles)
        rows = np.clip(rows, 0, self.canvas_size - 1)
        cols = np.clip(cols, 0, self.canvas_size - 1)
        self.pins = np.stack([rows, cols], axis=1)

    def _build_candidate_edges(self, min_pin_gap):
        edges = []
        n = self.n_pins
        for i in range(n):
            for j in range(i + 1, n):
                gap = min(j - i, n - (j - i))
                if gap < min_pin_gap:
                    continue
                edges.append((i, j))
        self.edges = edges

    def _build_render_matrix(self):
        """
        Precompute each candidate chord's anti-aliased pixel footprint once
        (skimage.draw.line_aa, the same Wu's-algorithm anti-aliasing the
        paper itself uses), and assemble them into one sparse
        (n_pixels x n_edges) matrix so every chord's current score can be
        read off with a single matrix-vector product instead of a Python
        loop over candidates.
        """
        size = self.canvas_size
        edge_pixels = []  # (flat_pixel_index, weight) per edge, for incremental updates
        rows_all, cols_all, data_all = [], [], []
        for e, (i, j) in enumerate(self.edges):
            r0, c0 = self.pins[i]
            r1, c1 = self.pins[j]
            rr, cc, val = line_aa(int(round(r0)), int(round(c0)),
                                  int(round(r1)), int(round(c1)))
            inside = (rr >= 0) & (rr < size) & (cc >= 0) & (cc < size)
            rr, cc, val = rr[inside], cc[inside], val[inside]
            flat = rr * size + cc
            edge_pixels.append((flat, val))
            rows_all.append(flat)
            cols_all.append(np.full(flat.shape, e, dtype=np.int64))
            data_all.append(val)

        self.edge_pixels = edge_pixels
        rows_all = np.concatenate(rows_all)
        cols_all = np.concatenate(cols_all)
        data_all = np.concatenate(data_all)
        self.A = sparse.csr_matrix((data_all, (rows_all, cols_all)),
                                   shape=(size * size, len(self.edges)))

    # ---- greedy selection -------------------------------------------------

    def _greedy_select(self):
        size = self.canvas_size
        n_edges = len(self.edges)
        mask_flat = self.circle_mask.ravel()
        raw = np.zeros(size * size)
        rendered = np.zeros(size * size)
        target_flat = self.target.ravel()
        residual = np.where(mask_flat, target_flat - rendered, 0.0)

        # Linear correlation of edge e's own pixels with the current
        # residual: sum_p weight_p * residual_p. This is the *first-order*
        # term of how adding edge e changes the squared error, and on its
        # own it is NOT enough to decide whether adding an edge actually
        # helps -- it reads as exactly zero for an untouched background
        # pixel (target=0, rendered=0, residual=0), making background look
        # "free" to cross, when in fact (target-rendered)^2 grows
        # quadratically the moment any ink lands there even though its
        # gradient at zero is zero. That second-order self-penalty term is
        # 0.5*alpha*sum_p(weight_p^2) -- bigger for edges covering more
        # pixels, i.e. longer chords -- and without subtracting it, long
        # chords that only loosely align with the target were being
        # accepted even when they made the overall fit worse (confirmed by
        # measuring SSE increasing on a synthetic test before this fix).
        # See net_score below for the corrected add/remove decision.
        corr = np.asarray(self.A.T @ residual).ravel()
        edge_norm_sq = np.asarray(self.A.multiply(self.A).sum(axis=0)).ravel()
        self_penalty = 0.5 * self.pen_alpha * edge_norm_sq
        active = np.zeros(n_edges, dtype=bool)

        def apply_delta(e, sign):
            flat, val = self.edge_pixels[e]
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
            # Net estimated SSE reduction from adding an inactive edge at
            # full strength (positive = helps); for an active edge, a very
            # negative value means it's sitting in over-inked territory and
            # removing it would help instead.
            net_score = corr - self_penalty
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
                # Removing edge e changes raw by -alpha*weight, the mirror
                # image of adding it, so its estimated SSE reduction from
                # removal is -(corr + self_penalty) -- not -net_score.
                # removal_threshold uses the same sign convention as
                # min_score: positive means "remove if it would help by
                # more than this".
                removal_benefit = -(corr[active_idx] + self_penalty[active_idx])
                drop = active_idx[removal_benefit > self.removal_threshold]
                for e in drop:
                    active[e] = False
                    apply_delta(e, -1)
                    self.selected.remove(e)

        self.active = active
        self.rendered_canvas = rendered.reshape(size, size)
        print("StringArt: selected %d / %d candidate chords" %
              (len(self.selected), n_edges))

    # ---- output -----------------------------------------------------------

    def _build_segments(self):
        """
        Order the selected chords into pen strokes directly, rather than
        with EuclidMST (used elsewhere in this repo by Canny.py/
        Skeleton.py): string art's extreme pin-sharing -- every pin can be
        the shared endpoint of dozens of chords, far denser than anything
        that shared ordering code previously had to handle -- made it
        emit some segments twice (confirmed: segment count roughly doubled
        after ordering on a real run, with ~97% of the resulting
        "connector" jumps longer than a tenth of the canvas). Chords
        sharing exact pin endpoints is exactly the structure a simple
        greedy "stay on the same pin as long as possible" walk exploits
        well: hop from chord to chord through shared pins, and only jump
        -- to the nearest pin that still has an unused chord -- once the
        current pin's own chords are exhausted.
        """
        pin_edges = [set() for _ in range(self.n_pins)]
        for e in self.selected:
            i, j = self.edges[e]
            pin_edges[i].add(e)
            pin_edges[j].add(e)
        active_pins = {p for p in range(self.n_pins) if pin_edges[p]}

        def other_pin(e, p):
            i, j = self.edges[e]
            return j if p == i else i

        def consume(pin, e):
            other = other_pin(e, pin)
            pin_edges[pin].discard(e)
            pin_edges[other].discard(e)
            if not pin_edges[pin]:
                active_pins.discard(pin)
            if not pin_edges[other]:
                active_pins.discard(other)
            return other

        ordered = []
        if self.selected:
            current = self.edges[self.selected[0]][0]
            if current not in active_pins:
                current = next(iter(active_pins))
            while active_pins:
                if pin_edges[current]:
                    e = next(iter(pin_edges[current]))
                    nxt = consume(current, e)
                    ordered.append((current, nxt))
                    current = nxt
                else:
                    cands = np.array(list(active_pins))
                    d = np.sum((self.pins[cands] - self.pins[current]) ** 2, axis=1)
                    current = int(cands[np.argmin(d)])

        self.segments = Segments.Segments()
        for p_from, p_to in ordered:
            self.segments.append(np.array([self.pins[p_from], self.pins[p_to]]))
