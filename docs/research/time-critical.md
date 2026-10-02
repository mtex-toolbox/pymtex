# The time critical algorithms, their treatment and what is left, 2026-09-22

The port keeps NumPy for everything that is one array expression and compiles the loops
that are not: numba kernels with a NumPy fallback as the test oracle, finufft for the
scattered transforms, `scipy.fft` on the product grids. This is the list of the places
where the time goes, what each has today, and what is still to do; the measurements are
in the other notes of this directory.

| algorithm | treatment | state |
|---|---|---|
| quaternion products, symmetry reduction, fundamental region projection | numba kernels in `geometry/kernels.py`, NumPy fallback as oracle | at or below Julia single thread (`benchmark-geometry.md`, `benchmark-e2-kernels.md`) |
| nearest and radius search on S2 rings and on SO(3) through the k-d tree | compiled searches in `geometry/grids/` | done (`benchmark-gridsearch.md`) |
| `equispacedSO3Grid` and the radius search of `find` | the gamma ladders as one flat array, the region test of every z copy a numba kernel (`regionMembership`), the vertices of `maxAngle` one batched solve; the radius search on the grid's structure as one numba loop (`so3Radius` in `geometry/grids/gridsearch.py`, one image per coset of the rotations about z), the k-d tree its fallback and oracle, the merge of the build through it | 2026-09-26: build 16 ms and radius 11 ms against Julia's 24 and 14 (m-3m, 2.5 degree); MATLAB's C build is 1.5 ms |
| the orientation embedding, a tensor turned by many orientations | the turn index by index, `turnBlocks` in `tensors/kernels.py`; the seeds of `project` kept per frame | 2026-09-26: ahead of Julia on every embedding case but mean_m3m, `calcTensor(ori, C, 'Hill')` 87 ms for Julia's 202 |
| the grain measures (area, perimeters, centroid, moments) | one compiled pass over the loops, `loopSums` in `maps/polygon.py` | 2026-09-26: 0.2 ms for 143 grains, was 2.6 |
| Wigner and spherical coefficient transforms | parallel numba kernels on the quadrant of d(pi/2) (SO3) and by rows (S2, exact beyond L = 1023), finufft on scattered nodes with the NFFT of `../nfft` as optional adjoint, the grid route one axis at a time, direct sum below 50 nodes | done, faster than MATLAB's master and Julia (`benchmark-harmonics.md`, `docs/plans/harmonics-speed.md`) |
| RBF evaluation, `calcComponents`, `max(odf)` with hundreds of centres | none | hangs on hundreds of centres; the halo cell list is the planned fix |
| kernel density estimation `calcDensity` | harmonic route | fine on the sphere, untimed at scale on SO(3) |
| pole figure inversion `calcODF` | two `S2FourierPlan`s per iteration, Voronoi weights, dense harmonic weights | dense solve only below 4000 nodes |
| optimal sampling | re-pointed plans at the low oversampling, vectorised derivatives | behind Julia only on the large sphere case |
| clustering `classix`, `calcCluster` | NumPy classix, k-d tree, batch embedding | ahead of Julia; the hierarchical method is quadratic, binned above 1000 |
| Taylor model `calcTaylor` | compiled simplex over the orientations (`analysis/kernels.py`) | the Taylor field of the TaylorModel page 14 s for 1278 s |
| texture evolution `doEulerStep` | prepared transport | 7 s per step, explicit scheme only |
| text import, lattice detection | numba row parser in `io/kernels.py` | done (`benchmark-io.md`) |
| decomposition clipping, segmentation, boundary tracing, KAM, GROD, GND simplex | numba in `maps/kernels.py` | done (`benchmark-decomposition.md`, `benchmark-maps.md`) |
| variational denoising | numba loops | done, not timed against Julia |
| parent grain reconstruction | the vote table and the variant graph (`maps/reconstruction.py`, `maps/variantgraph.py`), the misfit by a k-d tree, the MCL expansion in `maps/kernels.py` | every step ahead of MATLAB, 3 to 9 times (`benchmark-reconstruction.md`) |
| plotting: IPF colours, map raster, PNG | the grid route as a numba kernel in `plotting/kernels.py`, the aligned raster by rounding, the byte image capped at the axes' pixels | at Julia's time for the colours, plot within two of Julia, ten times ahead of MATLAB (`benchmark-mapplot.md`) |
| plotting: spherical contours, pole figures | matplotlib | untimed |

## To do

- [ ] RBF evaluation and the steepest descent of `calcComponents` through the halo cell
  list of the SO(3) grid, so that an ODF of a thousand centres evaluates and finds its
  modes in seconds; the tests of the clustering and the characteristics check modes by
  evaluation until then.
- [x] `calcTaylor` over a map: the simplex as one numba loop over the orientations,
  `analysis/kernels.py`; the Taylor field of the TaylorModel page in 14 s for 1278 s.
- [ ] `doEulerStep`: an implicit scheme, and the transport prepared once per bandwidth
  rather than per call.
- [ ] `calcODF`: the harmonic quadrature weights above 4000 nodes through a
  preconditioned iteration rather than the dense solve.
- [ ] `optimalSample` on the sphere at bandwidth 128 and above: the S2 plan is the cost,
  a plan with a narrower window at the sampler's tolerance would close the gap to Julia.
- [ ] `calcCluster(..., method='hierarchical')` above 1000 orientations without the grid
  binning, through the k-d tree.
- [ ] the parent grain reconstruction of M9 with its variant graph search, timed against
  Julia; against MATLAB done 2026-10-02 (`benchmark-reconstruction.md`).
- [x] the map plot in three steps measured in `benchmark-mapplot.md`, done 2026-09-22, 2.6 s
  to 0.86 s on four million points: the grid route of `ipfColors` as one numba kernel over
  the quaternions (1.05 s to 0.2 s); the picture of an aligned lattice read off the lattice
  table instead of the corner search (1.2 s to 0.12 s); the image as bytes and the
  picture capped at twice the axes' pixels so that every draw, pan and zoom resamples a
  million pixels at most (0.4 s to 0.1 s). The same day the passes over every measurement
  that the picture does not need were removed (the extent from the box of the positions, the
  lattice residual remembered, the bytes of the pixels alone): sixteen million points plot in
  0.66 s, thirty five million in 1.4 s, a zoom in 40 ms.
- [ ] `latticeTable`, the last per-measurement pass of the map plot: 0.25 s at sixteen
  million points, scattering an index per measurement into the table the picture reads. It
  depends only on the lattice and the order of the map, so the lattice could keep it, at an
  int32 per measurement held for as long as the lattice lives.
- [ ] the variational denoising, the spherical plots and the pole figure plots timed
  against Julia and MATLAB.
