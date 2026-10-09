# The maps layer against the Julia port, M3 to M7

Run 2026-09-21, `bench/maps.py` and `bench/maps.jl` on one synthetic map: 600 x 600 pixels of
unit spacing, the grains the Voronoi cells of 800 random seeds (585 survive the 5 degree
threshold, 39 014 boundary segments), each grain with its own orientation about z and a
gradient of 0.002 rad per pixel along x inside. The angles are written by the Python script
and read by the Julia one, so both see one map. Python 3.12, NumPy 2.5, numba 0.61 on all
12 threads; Julia 1.12 on 1 and on 12 threads. Best of three after a warm up. Times in
seconds; "before" is the NumPy form of every verb as first written, "after" the kernels this
benchmark led to.

| verb | Python before | Python after | Julia 1 thread | Julia 12 threads |
|---|---:|---:|---:|---:|
| calcGrains | 0.93 | 0.90 | 0.60 | 0.39 |
| simplifyBoundary | 0.065 | 0.021 | 0.015 | 0.022 |
| refineBoundary | 0.178 | 0.073 | 0.114 | 0.141 |
| reduceBoundary | 0.070 | 0.024 | 0.016 | 0.016 |
| smoothBoundary, Laplace 5 | 0.160 | 0.072 | 0.074 | 0.089 |
| smoothBoundary, curvature | 0.175 | 0.086 | 0.090 | 0.105 |
| aspectRatio | 0.021 | 0.006 | 0.000 | 0.000 |
| hull | 0.038 | 0.013 | 0.006 | 0.007 |
| paris | 0.022 | 0.014 | 0.006 | 0.008 |
| caliper | 0.029 | 0.023 | 0.007 | 0.008 |
| paror | 0.013 | 0.016 | 0.010 | 0.011 |
| surfor | 0.043 | 0.038 | 0.053 | 0.065 |
| characteristicShape | 0.006 | 0.006 | 0.011 | 0.011 |
| gradient, one sided | 0.221 | 0.025 | 0.085 | 0.024 |
| gradient, full stencil | 0.713 | 0.065 | 0.307 | 0.071 |
| curvature | 0.243 | 0.045 | 0.082 | 0.036 |
| KAM | 0.230 | 0.039 | 0.112 | 0.027 |
| KAM, order 3 | 1.265 | 0.147 | 0.650 | 0.127 |
| calcGROD | 0.051 | 0.024 | 0.034 | 0.015 |
| grainMean | 0.009 | 0.008 | 0.056 | 0.084 |
| calcMisorientation, 1e5 pairs | 0.047 | 0.047 | 0.013 | 0.014 |
| calcGND, 1600 pixels | 1.845 | 0.004 | 0.017 | 0.006 |
| smooth, mean filter | 0.352 | 0.120 | 0.124 | 0.129 |
| smooth, median filter | 0.529 | 0.172 | 0.159 | 0.153 |
| smooth, Kuwahara | 0.480 | 0.122 | 0.261 | 0.181 |
| smooth, spline | 1.976 | 0.355 | 1.263 | 1.161 |
| smooth, l1TV | 5.750 | 1.79 (loaded; Julia read 1.56 alongside) | 1.440 | 1.339 |

## What it says

- **The per-pixel verbs are at Julia's 12-thread time.** The gradient, the curvature, the
  KAM and the GROD were NumPy passes over the map, one per lattice offset, with the
  symmetry reduction through `closestImage` and the tangent through `rotationVector`; they
  ran two to three times behind Julia's single thread. Fused into one numba loop per verb
  in `mtex/maps/kernels.py`, each pixel reducing its neighbours against the proper group of
  its phase in place, they run at Julia's 12-thread time. The NumPy forms stay as the
  fallback and the oracle, and the tests hold the two to 1e-12.
- **calcGND was a hundred times behind, and is now ahead.** SciPy's `linprog` costs about a
  millisecond per call whatever the size of the program, and the dislocation fit is one
  six-by-thirty-six program per pixel. The dense two-phase simplex written for Julia's
  Taylor factor is now a numba kernel here, run over all pixels in parallel, and agrees
  with HiGHS on the value of every program; the densities may differ where the optimum is
  not unique, and both reproduce the curvature. A 360 000 pixel map fits in under a second
  where it took ten minutes.
- **The loop tracing was the cost of every grain list.** `tracePolygons` walked the edges
  of each grain in Python, forty milliseconds for 585 grains and seconds for a real map,
  and every subset, merge, hull and resampling traces again. It is one compiled walk over
  all grains now, the edges sorted by grain, with the areas accumulated on the way so the
  outer loop is found without a second pass.
- **The chain structure of a boundary is computed once.** `chainId` was a property
  recomputed on every access, forty times per simplification through `isChainStart`,
  `isChainEnd` and `isClosed`; a boundary is never edited in place, so it is memoised with
  the vertex degrees. That halved the resampling verbs, which are now at or below Julia.
- **The shape measures are bincounts.** Area, perimeter, centroid, the moments, the hull
  perimeters and the widths were loops over the grains in Python at thirty microseconds a
  grain; they read one flat edge list of all loops now (`loopEdges`), the hulls come from
  one compiled pass over all outer loops (`convexHullLoops`), and the extremal widths are
  one `reduceat` over the edge-point pairs of every hull.
- **The denoising (M8, 2026-09-22).** The window filters gathered their windows in NumPy
  and the variational ones solved one SuperLU system per component and IRLS step; they
  are a fused window kernel and one conjugate gradient over the whole graph now, the
  system being block diagonal in the components anyway, with the tangent field, the
  exponential and the reweighting as kernels beside them. The spline is three times ahead
  of Julia, the window filters at par. The l1TV filter is the one that matters, and it
  went through three versions: the SuperLU solve per component and step at 5.8 s; the
  global conjugate gradient at 2.5 s, which ran every grain to the cap of a hundred steps
  because the stopping rule was global, where Julia's per-component solve stops the median
  grain after four steps and lets four grains run to the cap; and the global solve with
  every component frozen on its own convergence and skipped by every kernel, which is at
  parity: 1.79 s against Julia's 1.56 s, both measured back to back on a machine at a load
  of two to four (Julia reads 1.37 s quiet). Its residual cost is the per-step overhead of
  the kernel launches and the gathers over the whole map while four grains are still
  active, about 9 ms a step against Julia's 3 ms.
- **What is still behind.** `calcGrains` at 1.5 times Julia's single thread, spread over
  the vertex weld, the criterion and the grain table with no one step dominant, the weld
  being the known item of `benchmark-decomposition.md`. The caliper and the hull measures
  at two to three times, a few milliseconds of NumPy on the pairs. `calcMisorientation`
  at three times on fifty milliseconds. None of these is a kernel's worth.

## The routes the benchmark does not time, 2026-10-09

`bench/maps.py` at 1000 x 1000 (1523 grains) under another user's load of 10: every verb in line
with the 600 x 600 table above scaled by the pixels. Beside it, at the same map: the orientations
8 ms, the inverse pole figure key 6 ms, the axis angle key 354 ms (ten NumPy passes, none
dominant; left), the inner boundaries selected 9 ms, their misorientation angles 6 ms, the grain
areas, perimeters and boundary lengths 3 to 5 ms, `reduce` 14 ms, `gridify` 19 ms, `rotate` 14 ms,
`calcDensity` of the million orientations 63 ms. The Bunge key takes 3.75 s and is left: almost
nobody uses it.

- **`interp` at 1e5 scattered positions 1744 -> 407 ms**: the map it returns tried to infer a
  lattice of the positions, failed the affine fit and grew an index over the neighbour graph in
  Python, 1.2 s, ending in a "hex lattice" with two points in one cell. `assignIndex` now asks
  first whether the neighbouring steps of a central patch are whole cells (`onLattice`: 85 % within
  0.4 of a cell; the tests' lattice distorted by 0.6 cells keeps 97 %, scattered points 67 %) and
  refuses scattered points, which then have no lattice; a grid of positions keeps its own.
- **`fill` of a map with nothing to fill 302 -> 12 ms**, and `fill(ebsd, grains)` 249 -> 38 ms:
  without an unmeasured pixel, and with grains without padding as well, the copy is returned
  before the smoothing graph is built.
