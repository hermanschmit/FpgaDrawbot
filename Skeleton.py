from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize, remove_small_objects, binary_closing, footprint_rectangle

import Segments
import EuclidMST


def smooth_with_function_and_mask(image, function, mask):
    bleed_over = function(mask.astype(float))
    masked_image = np.zeros(image.shape, image.dtype)
    masked_image[mask] = image[mask]
    smoothed_image = function(masked_image)
    output_image = smoothed_image / (bleed_over + np.finfo(float).eps)
    return output_image


_NEIGHBOR_OFFSETS = [(-1, 0), (0, -1), (1, 0), (0, 1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


class Skeleton:

    def count_neighbors(self):
        k = np.array([[1, 1, 1],
                      [1, 0, 1],
                      [1, 1, 1]])

        m = ndi.filters.convolve(self.skeleton, k, mode='constant', cval=0)
        nc = np.multiply(m, self.skeleton)
        self.neighbor_count = nc

    def euclidMstOrder(self):
        emst2 = EuclidMST.EuclidMST(self.segments.segmentList)  # TODO fix
        emst2.segmentOrdering()
        self.segments.segmentList = emst2.newSegmentTree  # TODO fix

    def __init__(self, image_matrix, sigma=0.5):

        if sigma >= 0.:
            mask = np.ones(image_matrix.shape, dtype=bool)
            fsmooth = lambda x: ndi.gaussian_filter(x, sigma, mode='constant')
            self.imin = smooth_with_function_and_mask(image_matrix, fsmooth, mask)
        else:
            self.imin = image_matrix

        plt.imshow(self.imin, cmap=cm.gray)
        plt.savefig("figStartOrig.png")
        plt.clf()

        # Otsu threshold on the (blurred) grayscale image separates ink from paper
        # more robustly than a global 2-cluster k-means, especially for faint/anti-
        # aliased strokes. Close 1px gaps and drop speckle noise before thinning.
        thresh = threshold_otsu(self.imin)
        ibin = self.imin < thresh
        ibin = binary_closing(ibin, footprint_rectangle((2, 2)))
        ibin = remove_small_objects(ibin, min_size=4, connectivity=2)
        self.ibin = ibin.astype(np.uint8)

        self.skeleton = skeletonize(self.ibin.astype(bool)).astype(np.uint8)
        plt.imshow(self.skeleton, cmap=cm.gray)
        plt.savefig("figStartSkel.png")
        plt.clf()

        self.segments = Segments.Segments()
        self.count_neighbors()
        self.trace_skeleton()
        print("skeleton pixels:", int(np.sum(self.skeleton)),
              "traced into", len(self.segments.segmentList), "segments")

    def _skel_neighbors(self, x, y):
        X, Y = self.skeleton.shape
        result = []
        for (i, j) in _NEIGHBOR_OFFSETS:
            nx, ny = x + i, y + j
            if 0 <= nx < X and 0 <= ny < Y and self.skeleton[nx, ny] == 1:
                result.append((nx, ny))
        return result

    def _walk_chain(self, node, first_step, ring_start=None):
        """
        Walk from a node pixel through a run of degree-2 pixels to the next
        node (or a dead end). If ring_start is given, also stop (and close
        the chain) on returning to it, for tracing loops with no junctions.
        """
        chain = [node, first_step]
        prev, cur = node, first_step
        while True:
            if ring_start is not None and cur == ring_start:
                break
            nbrs = [n for n in self._skel_neighbors(*cur) if n != prev]
            deg = len(self._skel_neighbors(*cur))
            if deg != 2 or not nbrs:
                break
            nxt = nbrs[0]
            chain.append(nxt)
            prev, cur = cur, nxt
        return chain

    @staticmethod
    def _tangent(chain, at_start, lookahead=3):
        """Unit direction vector pointing away from one end of a pixel chain."""
        pts = chain if at_start else list(reversed(chain))
        n = min(lookahead, len(pts) - 1)
        if n <= 0:
            return np.zeros(2)
        p0 = np.array(pts[0], dtype=float)
        p1 = np.array(pts[n], dtype=float)
        v = p1 - p0
        norm = np.linalg.norm(v)
        if norm == 0:
            return np.zeros(2)
        return v / norm

    def trace_skeleton(self):
        """
        Reduce the pixel skeleton to a graph -- junction/endpoint pixels as
        nodes, runs of degree-2 pixels as edges -- then walk each junction
        "straight through" by pairing its incident edges by minimum turning
        angle. This traces a stroke that crosses another stroke as one
        continuous path instead of cutting it into fragments at every
        crossing.
        """
        xs, ys = np.where(self.skeleton == 1)
        nodes = set()
        for x, y in zip(xs, ys):
            deg = self.neighbor_count[x, y]
            if deg == 1 or deg >= 3:
                nodes.add((x, y))

        chains = []  # chains[chain_id] = [pixel, pixel, ...] from node A to node B
        incident = defaultdict(list)  # node -> [(chain_id, 'A'|'B'), ...]
        claimed_start = set()  # (node, first_step) half-edges already walked

        for node in nodes:
            for nbr in self._skel_neighbors(*node):
                if (node, nbr) in claimed_start:
                    continue
                chain = self._walk_chain(node, nbr)
                end_node = chain[-1]
                chain_id = len(chains)
                chains.append(chain)
                incident[node].append((chain_id, 'A'))
                incident[end_node].append((chain_id, 'B'))
                claimed_start.add((node, nbr))
                if len(chain) >= 2:
                    claimed_start.add((end_node, chain[-2]))

        used_chain_pixels = set()
        for chain in chains:
            used_chain_pixels.update(chain[1:-1])

        # Closed loops with no junctions/endpoints at all (pure rings).
        visited = np.zeros(self.skeleton.shape, dtype=bool)
        for x, y in zip(xs, ys):
            visited[x, y] = (x, y) in used_chain_pixels or (x, y) in nodes
        for x, y in zip(xs, ys):
            if visited[x, y]:
                continue
            nbrs = self._skel_neighbors(x, y)
            if not nbrs:
                continue
            chain = self._walk_chain((x, y), nbrs[0], ring_start=(x, y))
            for (px, py) in chain:
                visited[px, py] = True
            if len(chain) > 2:
                self.segments.append(np.array(chain))

        # Pair incident edges at every node by "straightest continuation" so a
        # path passing through a junction is not cut there.
        pair_of = {}
        for node, edges in incident.items():
            if len(edges) < 2:
                continue
            dirs = {(cid, end): self._tangent(chains[cid], at_start=(end == 'A'))
                    for (cid, end) in edges}
            remaining = list(edges)
            while len(remaining) >= 2:
                best = None
                for i in range(len(remaining)):
                    for j in range(i + 1, len(remaining)):
                        d = float(np.dot(dirs[remaining[i]], dirs[remaining[j]]))
                        if best is None or d < best[0]:
                            best = (d, i, j)
                _, i, j = best
                a, b = remaining[i], remaining[j]
                pair_of[a] = b
                pair_of[b] = a
                for idx in sorted((i, j), reverse=True):
                    remaining.pop(idx)

        def other_end(cid, end):
            return 'B' if end == 'A' else 'A'

        def orient(chain, end):
            return chain if end == 'A' else list(reversed(chain))

        # Walk from every loose end (endpoints, and any unpaired junction
        # stub) through the paired junctions, merging chains into one path.
        visited_halfedge = set()
        for node, edges in incident.items():
            for (cid, end) in edges:
                if (cid, end) in pair_of or (cid, end) in visited_halfedge:
                    continue
                path = list(orient(chains[cid], end))
                visited_halfedge.add((cid, end))
                cur = (cid, other_end(cid, end))
                visited_halfedge.add(cur)
                while cur in pair_of:
                    nxt = pair_of[cur]
                    if nxt in visited_halfedge:
                        break
                    n_cid, n_end = nxt
                    seg = orient(chains[n_cid], n_end)
                    path.extend(seg[1:])
                    visited_halfedge.add(nxt)
                    cur = (n_cid, other_end(n_cid, n_end))
                    visited_halfedge.add(cur)
                if len(path) > 1:
                    self.segments.append(np.array(path))

        # Anything left over is a closed loop traced entirely through paired
        # junctions (no loose end to start from).
        for cid, chain in enumerate(chains):
            if (cid, 'A') in visited_halfedge:
                continue
            path = list(chain)
            visited_halfedge.add((cid, 'A'))
            cur = (cid, 'B')
            visited_halfedge.add(cur)
            while cur in pair_of:
                nxt = pair_of[cur]
                if nxt in visited_halfedge:
                    break
                n_cid, n_end = nxt
                seg = orient(chains[n_cid], n_end)
                path.extend(seg[1:])
                visited_halfedge.add(nxt)
                cur = (n_cid, other_end(n_cid, n_end))
                visited_halfedge.add(cur)
            if len(path) > 1:
                self.segments.append(np.array(path))
