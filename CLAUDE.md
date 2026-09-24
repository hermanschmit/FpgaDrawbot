# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Python image-processing code for the Altera FPGA Drawbot: it turns a raster image into continuous pen paths (polylines) and writes them in formats the drawbot (and other tools) consume. It is a flat collection of scripts at the repo root — no package, no build system, no requirements file.

Dependencies (not pinned anywhere): `numpy`, `scipy`, `svgwrite`, `numba`, `scikit-image`, `imageio`, `matplotlib`.

## Commands

Tests use `unittest`-style `TestCase` classes in `test_*.py` (many define `runTest`, so run them via a test runner, not by name-matching `test_` methods):

```bash
python -m pytest                          # all tests (or: python -m unittest discover)
python -m pytest test_Segments.py         # one file
python -m pytest "test_Canny.py::TestCannyI10_1"   # one test class
```

Run from the repo root — modules import each other by bare name (`import Segments`).

Generator entry points (`*Prog.py`), all taking an input image, an output preview PNG, and an optional `.bin` path (default `bfile.bin`):

```bash
python CannyProg.py in.jpg out.png [out.bin]
python HilbertProg.py in.jpg out.png [out.bin]
python SkeletonProg.py in.jpg out.png [out.bin]
python SketchyProg.py in.jpg out1.png out2.png [out.bin]
python MazeProg.py [--moore|--skeleton|--fass|--diag] [--quant_levels N] in.jpg out.png out.bin [out.svg]
```

Sample inputs live in `InputImages/`. `MazeProg` dumps per-iteration frames to `img/figNNNNN.png`; `scripts/movie1.sh <dir>` rotates those and assembles an mp4 with ImageMagick + ffmpeg (examples in `savedMovies/`). `*.png`, `*.svg`, `*.scad` are gitignored.

## Architecture

Each drawing style is a generator class that builds a `Segments` object; everything downstream (optimization, rendering, output) goes through `Segments`.

- **`Segments.py`** — the central data structure. `segmentList` is a list of numpy `(N,2)` point arrays (one per pen-down stroke) with tracked bounds (`xmin/xmax/ymin/ymax`) and `numpts`. Provides the pipeline tail used by every `*Prog.py`:
  - cleanup: `simplify()` (drop collinear points), `chaikin_smoothing()`, `concatSegments()` (merge into one continuous path), `addInitialStartPt()`
  - preview: `segment2grad()` + `renderGrad()` rasterize via Bresenham into `self.grad`, which is saved as the output PNG
  - output: `scaleBin()` normalizes coordinates to roughly `[-1, 1]` centered on origin, then `binWrite()` emits float32 pairs **in (y, x) order**, NaN-padded (20 trailing NaNs) — this is the format the FPGA reads. Also `svgwrite/svgread`, `cArrayWrite`, `openScadArrayWrite`.
- **Generators** (each owns `self.segments`):
  - `Canny.py` — edge detection → edge polylines; then `euclidMstOrder()` uses `EuclidMST.py` (Delaunay → minimum spanning tree → DFS) to order disjoint segments and minimize pen travel.
  - `Skeleton.py` — morphological skeleton → traced lines, also ordered with `EuclidMST`.
  - `Hilbert.py` — space-filling Hilbert/Moore curve whose local density follows image intensity (quantized via `Quantization.py` k-means).
  - `Maze.py` — the most elaborate: a single closed curve evolved by attraction/repulsion forces (`AttractRepel.py`, numba-jitted, KD-tree neighbor search), brownian motion, fairing and resampling in `optimize_loop2()`; initial shape chosen by `INIT_*` constants (Moore curve, skeleton, Fass, diagonal; `MazeProg` defaults to skeleton). `MazeSimple.py` is an older, reduced variant.
  - `Sketchy.py` — greedy line-drawing over a quantized image, with an optional bot-geometry coordinate transform.
- **`TSPopt.py`** — 2-opt/3-opt local path improvement shared by Hilbert/Maze.
- `LSystem.py`, `SimpleHilbertCurve.py`, `HilbertTest.py`, `CannyTestPattern1.py` are standalone experiments, not part of the main pipeline.

Note: point-path simplification lives in `Segments.simplify_segment()` (`TSPopt.simplify()` wraps it). The image/matrix coordinate convention (row, col) vs. (x, y) is swapped in several places — check `binList()` and the generator before changing axis handling.
