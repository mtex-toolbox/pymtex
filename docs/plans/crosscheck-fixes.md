# What the cross-port benchmarks found, port by port

Status: written 2026-10-02 from the runs of every cross benchmark (`bench/*_ops.py`, `density`,
`dubna_density`, `calcgrains` and the standalone ones) against MATLAB `../master` at
feature/fastHarmonicTransforms (34df8425a) and the Julia port at `main`, with the cores each case
kept busy (`tools/graph/query.py parallel`). The Python findings are issue #1 of
mtex-toolbox/pymtex. The MATLAB fixes go into develop, checked out in `../master`,
as `develop-fixes.md` does; the Julia ones into `../mtexJulia`. Every number below is a
`compare` line or a `parallel` row of that run.

## How a direction is chosen

A result that depends on what it should not (the order of a search, the size of a batch), a
crash, or a value that contradicts the function's own definition is a defect and is fixed in the
port that has it. Where the ports differ by a convention, the design decides as CLAUDE.md says:
the Julia port on the algorithm, MATLAB on the name and on the semantics already decided there.
Nearness to MATLAB is no measure where no true answer exists (which pixels a grain absorbs);
an open convention is the user's to decide.

## 0. Which port is wrong

Python and Julia agree in these and MATLAB does not, or a port has no half to break the tie. Each
is settled by a reference computation (an analytic case, a brute force sum) before anything is
fixed; the outcome decides the port. They are the open questions at the end.

## 1. Python

Correctness:

| # | what | where | state |
|---|---|---|---|
| P1 | `SO3FunRBF.eval` asks for 447 GiB when a sum of 1e7 centres is too narrow for the harmonic form: `CHUNK` bounds the evaluation points, not the centres | `mtex/functions/rbf.py` | done: a kernel of support below 120 degree sums through a k-d tree of the centres (`_treeSum`), the rest by blocks of at most 2^22 dots (`_blockSum`); the Dubna run peaks at 4.8 GB. It found that an antipodal function with unequal weights put each weight on another centre's inverse (eval and the pole figure), fixed. `dubna_density` now passes 10 degree, MATLAB's default, since this port's `calcDensity` selects a halfwidth: 1.2e-2 from MATLAB |
| P2 | `calcComponents` stops two modes short on a flat ridge (0.4 and 1.75 degree from Julia's, gradient 31.7); a mode under 1 % is not merged as Julia merges it | `mtex/functions/characteristics.py` | done: a step was judged against a value of the other kind (a large evaluation goes through the harmonic form, a small one is summed), so each iteration evaluates the modes and both trials in one call; a radial or fibre function ascended on its harmonic gradient climbs on to its peaks on its own gradient (`_ascend` twice). The values agree with Julia's to 3e-4 of 470, one mode sits 0.86 degree along a ridge that flat; 1.4 s against 0.76. The mode under 1 % stays: the flat-slope rule stops its seed by design (`test_components_freeze_light_seeds_on_flat_stretches`) |
| P3 | `smooth(ebsd, splineFilter, fill=True)` differs from Julia's by up to 3.85 degree on 5 % of the ferrite pixels; the unfilled filter agrees to 7.7e-8 | `mtex/maps/smoothing.py`, the filters in `mtex/analysis/variational.py` | no fault here: the two ports smooth different maps. After `calcGrains` this port keeps the 1671 notIndexed pixels a grain absorbed, with its phase and a NaN rotation, as MATLAB does (`absorbNotIndexed`), so `ebsd['indexed']` holds them and the fill solves for them with their neighbours; Julia's `ebsd[isIndexed(ebsd)]` leaves them out. The pixels that differ lie 1 step from an absorbed one (at most 3), the ones that agree 5. J6 |
| P4 | `calcGrains` on Forsterite a, b, d: the grain counts agree with Julia, 613 to 838 pixels sit in another grain; copper b has 389 grains against 388 | `mtex/maps/calcgrains.py` | diagnosed, the rule to decide: all 838 are notIndexed pixels whose absorption differs, and every one has a nudged corner as near to two sites (on a square lattice beside a hole the corner lies on the edge of two cells), which `absorbInteriorPixels` in both ports breaks by the order of its search. An order-free rule is the fix, in both ports alike; which one is a convention (a tie counts for neither grain or for both). Tried here: "for neither" moves this port to 729 pixels from MATLAB on Forsterite and 297 on Emsland (Julia 977 and 967), "for both" to 2174; reverted until decided; issue #2 |
| P5 | `smoothBoundary(grains, 5)`: the perimeters 1.3e-3 off MATLAB, the tolerance 1e-3 | `mtex/maps/smoothing.py` | no fault here by the rule above: the difference is the simplification (with `simplify=False` all 143 grains agree with MATLAB to 1.8e-6, without the resampling 25 still differ), and there this port follows Julia, the two agreeing but on the 2 grains where Julia is off (J2). MATLAB's simplification is unchanged since August (its 2026-10-01 commit only zeroes the pixel pair of a merged segment); the difference is the open item of BoundaryTutorial in `docs/findings.md` (3363 segments against 3359). Left to decide: whether the definition of the simplification (every vertex dropped within the tolerance of the kept chain) settles which of the two is right |

Speed:

| # | what | numbers | state |
|---|---|---|---|
| P6 | `max(odf)` of 300 centres | 2.2 to 4.7 s, MATLAB 0.88 s, Julia 0.19 s | done: the refinement evaluates the shifted grids of every mode in one call per level (14 calls, not 141), which the harmonic form serves; 0.23 s, the maximum 1.2e-5 below the direct search's, the position 3.5e-3 from MATLAB's (the accuracy 0.25 degree) |
| P7 | libraries outside numba spread over all logical cores: up to 30 cores busy at `threads()` = 16 (`youngs`, `velocity`, `mean_m3m`, `odf_max`, `halfwidth_s2_*`) | `parallel` | done: OpenBLAS (NumPy's and SciPy's) starts on every logical core and is now set to the preference at import, on `setMTEXpref('threads')` and on `reset()` (`applyThreads`); numba holds its count per Python thread, so every `_parallelKernel` launch sets it, where the tensors, maps and io kernels set none; the k-d tree searches take `threads()` for `workers=-1`. `youngs` and `schmid` 16 cores, `mean_m3m` 73 to 27 ms, `velocity` 14 to 9 ms; `velocity` and `rbf_harmonic` still read 26 to 29, OpenBLAS's threads spinning after a product while numba's run |
| P8 | one core where the work is parallel: `inv` of tensors, `trueebsd_undistort`, `export_h5`, `birefringence` (3 cores) | `parallel` | done: `inv` of rank four tensors is one compiled loop over them (gather, Gauss-Jordan, spread; `voigtInverse` in `mtex/tensors/kernels.py`), 141 to 12 ms, NumPy's three passes the fallback and oracle; `export_h5` spent 0.9 of its 1.1 s closing the copy, since ext4 writes a truncated and rewritten file through to the disk, and removes an old file first now, 206 ms (Julia 376, MATLAB 1104, which likely meets the same); `birefringence` restricts the tensor by two matrix products of n rows instead of batched 2 x 3 products, 32 to 22 ms, the rest elementwise passes; `trueebsd_undistort` stays, 106 ms in many small passes and below MATLAB's 157 |
| P14 | the map verbs of `bench/maps.py` at 1000 x 1000 (2200 grains) behind Julia: curvature 149 against 43 ms, calcMisorientation 103 against 27, caliper 55 against 16, paror 54 against 19, calcGROD 50 against 17, KAM 65 against 23, surfor 209 against 146 | `bench/maps.py` | done, the values unchanged: curvature 29 ms (the tensor in one compiled pass, the boolean NaN assignment of a (1e6, 3, 3) array gone), calcGROD 17 (the references one gather), calcMisorientation 32 (only the drawn pixels read, and drawn among the indexed ones, where it used the positions of the indexed pixels with indices into all of them), caliper 4 (the extremal widths one compiled loop over the hulls, the hulls of the loops in parallel), paror 24, surfor 26 (the projections by blocks), KAM 22 and of order 3 89 (the angle by one product and a dot per element), the lattice lookup cached on the lattice; columnMin and columnMax in geometry/kernels.py replace the reductions over the long axis of tall arrays, twenty times slower in NumPy, in the lattice, gridify, the traces, the voxel lattice, the KDE and the extents of the plots. Left: hull 20 against 13, paror 24 against 19 |
| P15 | SO(3) `symmetrise` of 432/1 at L = 128: 55 ms against MATLAB's 18 | `harmonics_user` | done, the coefficients the same to the bit: the copy the projection worked on is gone (it writes into a new array, `project(..., out=)`, and only reads the source), a block of one index array and a slice is written as one instead of by the outer index of two, and the degrees run over the thread pool, dealt out in turn: 432/1 14 ms, 432/mmm 20 ms (MATLAB 19). 622/222, which the compiled sparse projection takes, measured 5.6 ms in the morning and 29 ms later: no regression, the state of the process. Its time is the first write into a fresh 46 MB array, page by page, 24 ms on one thread; after the grid cases of `harmonics_user` the allocator happened to have the pages ready (3 ms), as `hugePages` would (7 ms). The copy into the new array runs over the cores now (`copyInto` in geometry/kernels.py, 26 to 8 ms), 622/222 12 ms in a fresh process (MATLAB 13.5) |
| P16 | SO(3) quadrature at L = 128, 1e6 nodes, m-3m: 631 ms against MATLAB's 506; the eval 366 ms | `harmonics_user` | done: the oversampling factor is finufft's own choice unless a caller names one (`_options` in `mtex/functions/transforms.py`). The rule by the density of the nodes, 1.25 below a quarter of the lattice, took the slower factor up to three times over (1e6 nodes on the 4-fold lattice of m-3m: 296 ms, 356 at 1.25, 795 at 2); finufft's choice is within the noise of the better of the two over nodes from 1e4 to 1e7 and lattices from 17 x 129^2 to 129 x 257^2, but for one case 20 % off. Quadrature 331 ms, eval 214 ms |
| P17 | the SO(3) transforms at few nodes on a large lattice: eval at 1e4 nodes, L = 128, 253 ms where the transform proper takes 106 | `harmonics_user` | done: finufft faulted its oversampled grid in on every execute (`heapBlocks` in geometry/kernels.py keeps the pages), the adjoint's plan put its short axis first (now (k, j, l)), the grid adjoint unwrapped its polar FFT and spread its weights out, and a callable is evaluated on half the grid where a 2-fold axis perpendicular to z maps it onto the rest; every SO(3) route 12 to 52 % faster, docs/research/benchmark-harmonics.md, 2026-10-09 |
| P18 | the orientation grids: the 1 degree equispaced build 170 ms, the Fibonacci build 122, `find` among a regular grid 4 ms a query, every repeated build from scratch | `geometry_ops` grid cases | done: the radius search of the build's merge filled a grid-sized array per chunk of 16 queries (now one row per thread), the last four grids are kept as MATLAB keeps its last, `find` in a plain orientation list goes through the quaternion tree from 32 queries on, the Fibonacci spiral and `checkInside` compiled; docs/research/benchmark-gridsearch.md, 2026-10-09 |
| P19 | the S2 grids: the radius search of the ring grid 400 ms for 2000 queries, the Fibonacci grid's searches 100 to 200 ms, `find` in a plain list of directions 7 s for 1e4 among 1e4; an antipodal list searched as if it were not | own measurement | done: one k-d tree over the points for every set of directions (`S2Tree`, kept on a grid), 4 to 25 ms; `find` reads the antipodal flags of both lists by default; docs/research/benchmark-gridsearch.md, 2026-10-09 |
| P20 | the evaluation of the SO(3) functions that are not harmonic: a narrow RBF (3 degree, 1e4 centres) 86 s at 1e6 orientations, the fibre ODF 1.8 s, Bingham 0.3 s | own measurement | done: a 4D cell list of the centres instead of the tree's Python lists, the closed-form kernels summed in the compiled loop; the fibre ODF's arguments one matrix product and its de la Vallee Poussin sum compiled; Bingham's copies in one compiled loop. 2.8 s, 75 ms, 27 ms; docs/research/benchmark-harmonics.md, 2026-10-09 |
| P21 | the S2 functions: `S2FunTri` 774 ms at 1e6 directions, `sphericalSeries` below 50 directions the direct sum, ten times the trigonometric sum | own measurement | done: the start triangle's tree query on the threads and the walk compiled (148 ms); the trigonometric sum from one direction on; docs/research/benchmark-harmonics.md, 2026-10-09 |
| P22 | the S1 functions: a small bandwidth on finufft's single thread, 1e6 angles at L = 16 59 ms, the plan's 0.5 ms at a few angles | own measurement | done: the series by Horner's scheme and its adjoint compiled, up to L = 64 and to 4e6 terms; 8 ms, 0.02 ms; docs/research/benchmark-harmonics.md, 2026-10-09 |
| P23 | the tensor routes: `PoissonRatio` and `shearModulus` at 1e6 direction pairs 1.4 and 0.75 s, `calcTensor` over 1e6 orientations 0.4 s and over a map 1.3 s | own measurement | done: one tensor with vector lists a compiled multilinear form in `EinsteinSum`, the mean over orientations summed without forming the turned tensors, the map's Reuss average from the turned inverses, `quat2mat` compiled; 70, 37, 36 and 76 ms; docs/research/benchmark-tensors.md, 2026-10-09 |
| P12 | `grod`: 60 ms against MATLAB's 13.4 | `maps_ops` | done: `calcGROD` itself took 4.7 ms, the reduced angle of its misorientations 10.7; that angle is the chord to the nearest image of the identity, whose images are the group's elements (24 for m-3m, not the 576 pairs of the pair search), found and measured in one compiled loop (`nearestAngle`): the angle 1.8 ms, `grod` 8.0 ms, the angles as the pair search's to 2e-15 |
| P13 | S2 `calcDensity` of 1e5: 217 ms against MATLAB's 8.4 | `harmonics_user` | mostly the benchmark: without a halfwidth this port selects one (186 ms), MATLAB takes 10 degree; `harmonics_user` passes 10 degree now. With it 16.3 ms, and dropping the NaN rows and normalising in one pass 12.6 ms (15.5 in the benchmark's best of two), MATLAB 8.8; what is left is finufft at a tolerance of 1e-8 (5.4 ms), its plan (1.4 ms) and the arithmetic around it. A plan kept between calls, or MATLAB's lower precision, would close more |
| P9 | the image searches of the symmetry reduction on one core | 10 to 13 times faster on the thread pool, bit for bit | done, `_overRows` in `mtex/geometry/kernels.py` |

Found by the Julia fixes (section 3):

| # | what | where | state |
|---|---|---|---|
| P10 | the kernel sum of an RBF ODF (`rbf_direct`) is 0.39 of 787 off MATLAB's and Julia's, which agree to 3e-6, and equal to the harmonic form cut at the kernel's bandwidth (25 for 10 degree) to 3e-13: the sum is of the kernel cut at its accuracy cutoff, not of the kernel. J1 | `mtex/functions/rbf.py`, the kernel evaluation | open |
| P11 | a closed boundary chain is simplified from its first kept vertex or where the walk entered it, so the result depends on the walk: Julia's simplified digital circle measured 80.1 to 87.4 by the start. Julia now begins every ring at its lowest vertex (J2) | `mtex/maps/boundaryresampling.py`, `orderWalk` | open |

To decide: `loadEBSD(..., dataSet=)` counts from 0 (`mtex/io/h5.py`), MATLAB and Julia from 1.

## 2. MATLAB

| # | what | where |
|---|---|---|
| M1 | `calcComponents` leaves its loop when no mode gains at the current step, while that step is still 3.3 degree: 22 modes, some 1 to 16 degree off the peaks, where the ports find 16 | `SO3Fun/@SO3Fun/calcComponents.m` |
| M2 | the EDAX reader turns an all-zero orientation matrix (a failed pattern) into a rotation and keeps the pixel indexed: 383 pixels of `h5_edax_2datasets` | `interfaces/loadEBSD_h5.m` |
| M3 | the 3D KAM adds 1e-5 to every angle so that a zero misorientation is kept in the sparse matrix; the 2D KAM may do the same | `EBSDAnalysis/@EBSD3/KAM.m` |
| M4 | `optimalSample` returns 216 nodes for 200 and 288 for 300, the size of the equispaced grid it starts from | `@SO3Fun/optimalSample.m`, `@S2Fun/optimalSample.m` |
| M5 | `calcGND` 8.1 s and `calcTaylor` 9.9 s, `taylor_fun` 19 s, nearly all in `linprog`'s interior point solver on one core; the ports' simplex takes milliseconds | `@curvatureTensor/fitDislocationSystems.m`, `calcTaylor.m` |
| M6 | the kernel sum of an RBF ODF 3.1 s against 7 ms; projecting directions into the sector 755 ms against 7 ms | `@SO3FunRBF/eval.m`, `@vector3d/project2FundamentalRegion.m` |
| M7 | `SO3FunHarmonic` of an RBF ODF caps the bandwidth asked for at the kernel's accuracy cutoff, 25 for a 10 degree de la Vallee Poussin kernel whose coefficients run to degree 91, so its harmonic form is 0.39 of 787 off its kernel sum whatever bandwidth is asked for (J1) | `@SO3FunRBF/SO3FunHarmonic.m` |
| M8 | `grainBoundary/simplify` cuts a closed chain at the vertex its walk began with, so the simplified ring depends on the walk (P11, J2) | `@grainBoundary/simplify.m` |

## 3. Julia

Fixed on `main` 2026-10-09 (08ded6e, e0f15ff, 633e276, 2544ed5), the values of every sped up case
unchanged.

| # | what | numbers | state |
|---|---|---|---|
| J1 | the kernel sum of an RBF ODF 0.39 off its harmonic form (MATLAB too; Python agrees to 3e-13) | `functions_ops`, the slow variant | done: the kernel sum was right and the harmonic form wrong. `SO3FunHarmonic` of an RBF or fibre ODF capped the bandwidth asked for at the kernel's accuracy cutoff (25), on the premise that the kernel has no degrees above it; it now caps at `degreeLimit`, the degree where the coefficients vanish, finite only for polynomial kernels. At bandwidth 64 the two agree to 1.3e-4, the transform's accuracy. Python's 3e-13 is its own defect, P10; MATLAB's cap is M7 |
| J2 | `perimeter` 0.13 and `smoothBoundary` 0.12 off MATLAB, where Python agrees to 1.7e-6 and 1.3e-3 | `maps_ops` | done: `perimeter` counted the holes, 228 for MATLAB's 170 on the grain around an inclusion; it is the outer loop now, as MATLAB's, with `withInclusions = true` for the holes. The inclusion itself was simplified from wherever the walk entered its ring; `orderWalk` begins every ring at its lowest vertex, so the cut no longer depends on the walk (Python P11, MATLAB M8). This exposed `isInclusion` testing a vertex on the hole's boundary, fixed. Not rerun against MATLAB; the remaining grains are P5's open simplification question |
| J3 | `velocity` 22 s against 14 ms, `grod` 5.8 s against 60 ms | `tensors_ops`, `maps_ops` | done: `velocity` 0.16 s, the Christoffel tensor a direct contraction and its eigen static (was a generic `EinsteinSum` and LAPACK per direction); `grod` 0.09 s, `angle` of a list reduces over groups built once, where every element rebuilt the proper point group from its symbol |
| J4 | `calcCluster` of 1e5 directions 13 s against 1.4 s, `classix` of 1e6 points 4.1 s against 0.5 s, `taylor_fun` 6.0 s against 0.57 s, `calcComponents` 5.0 s against 0.76 s, the SO(3) grid 4.2 s against 0.15 s | `cluster.py`, `analysis_ops`, `tensors_ops`, `gridsearch.py` | done but the grid: `calcCluster` 0.60 s (`knnRadius` allocated per distance, `classix` ran on a matrix a comprehension boxed); `classix` 7.2 to 0.71 s on the bench's points at `minPts = 1e4`; `calcComponents` 0.25 s, the groups derived from a named point group now built once per symbol and axes; `taylor_fun` 0.36 s, the node solves threaded and the minimum norm shears solved by a static Cholesky where a condition bound allows, so the threads no longer queue on LAPACK. The grid is no defect: `gridsearch.jl` times the first call, 99 % compilation; built again it takes 22 ms (section 4) |
| J6 | after `calcGrains!` the notIndexed pixels a grain absorbs are not in `isIndexed` (58921 of the ferrite map's pixels against 60592 in this port and MATLAB), so `smooth(..., fill = true)` has no hole to fill there and differs from both next to them | `maps_ops`, `smooth_fill_*` | done: an absorbed pixel takes its grain's phase and a NaN rotation, as MATLAB's `calcGrains`; not rerun |
| J5 | few of 32 threads busy: L1TV and half quadratic denoising 2 to 5 cores, `project_6mmm` 2.2, and the image searches one core, where the Python thread pool gave 10 to 13 times | `parallel` | open |

## 4. The benchmarks

- The MATLAB halves of `calcgrains` and `mapplot` still use the bridge of `../frameSymmetry`;
  they move to `../master`'s, and `geometry.m` and `gridsearch.m` go (written for frameSymmetry, replaced by `geometry_ops`).
- Halves still missing: Julia for density estimation, `calcODF` and `trueEbsd`; MATLAB
  (profiled) and Julia for the parent grain reconstruction.
- Algorithms still missing: the S2 grid `find`, `doEulerStep`, the pole figure and contour
  plots, the grain area, centroid and moments, the text import with lattice detection.
- `dubna_density` writes its Python density values once P1 is fixed.
- `gridsearch.jl` times the first build of the grid, which is compilation; it warms up first
  as `crossrun` does.
- `maps_ops`, `functions_ops`, `tensors_ops`, `analysis_ops` and `cluster` are to be rerun for
  the Julia fixes; the numbers in section 3 are from the scripts' inputs in this port alone.

## Order

P1; the open questions; M1 and M2; J3; then the speed items by their gain; then section 4.

## Decided, 2026-10-09

What the MATLAB fixes on develop (1dc5807b1..b23d9cc59) showed, and what the user decided:

- **The mean and spline filters.** MATLAB's `meanFilter` left the pixel's own weight out of its
  normaliser, clamped at the border and ignored `weights`; fixed, it is the ports' normalised
  window mean (the four-neighbour cross by default) and agrees with this port to 5e-6 degree on
  twins and 1e-5 on ferrite. Closed. The spline filters differ by design: MATLAB smooths by
  `smoothn` (a robust DCT spline, its alpha by GCV on the largest grain and reused), the ports by
  a graph Laplacian with a smoothing length (`docs/findings.md`). Closed as a convention.
- **The 6/mmm colour key.** MATLAB's 5e-2 was its own bug (the centre of the sector not
  normalised in `polarCoordinates`); what is left is this port's: `equaliseAzimuth` pins the
  vertices to a 1000-step grid, so a vertex of 6/mmm lands at 2.0036 thirds. Done: the vertex
  azimuths are knots of the grid, the turn integrated by the trapezoid rule, every arc between two
  vertices a third, so the vertices land on the thirds to 3e-15; the exact IPF colours are now
  4.8e-4 (m-3m) and 4.7e-4 (6/mmm) from MATLAB's, from 1.8e-2 and 5.3e-2, and the reference table of
  `tests/test_directionkeys.py` holds MATLAB's colours of today. A direction on the boundary of the
  -43m sector ([-3 -1 0]) takes either side's colour, 2e-3 apart at any grid: open. The Julia
  port, 2.1e-2 in 6/mmm, needs the same.
- **The kernel sum of an RBF ODF (P10).** The harmonic switch of `SO3FunRBF.eval` stays, also
  where the kernel's bandwidth is cut; the benchmark says that it measures the harmonic form, and its
  check compares the harmonic form with the exact kernel sum (4.9e-4 of the maximum). Done.
- **optimalSample.** From the ports' starting sequences MATLAB's L-BFGS reaches an equally good
  minimum by another path (J 0.1454 against 0.1436), so the benchmark compares the objective J,
  this port's for every port's nodes, relative to the target's: 2.7e-7 (S2) and 2.0e-7 (SO(3)). Done.
- **calcTaylor and calcGND.** MATLAB returns the analytic centre of the optimal face of each
  linear programme (`linprogCentre`), the symmetric and unique slip rates, where the ports'
  simplex returns a vertex: the Taylor factors agree, the slip rates and the spin do not. This
  port follows MATLAB's centre, and the Julia port is to. Done: `linprogCentre` (taylor.py, NumPy,
  batched) and `_centreProgram` (maps/kernels.py, compiled, used by the Taylor and the GND loop)
  stop on the duality gap, as MATLAB's does since 78bcc0897, and raise the diagonal of the normal
  equations by only 1e-16 (1e-13 shifted the badly conditioned last steps off the constraints,
  1 strain of the 1e4 and 13 of 1e5 random ones never converged); `spinSelection='centre'` is
  the default, 'basic' and 'minimumNorm' remain. The factors agree with MATLAB's to 7.8e-12. Uniaxial [001] in fcc: the 8 active systems at 0.3062 each (MATLAB's 16 x 0.1531 count
  each signed system twice). taylor 17.5 ms (MATLAB 74), taylor_fun 152 ms (184), gnd 56 ms (705).
- **The benchmarks.** `bench_maps_ops.m` maps 'hq' to `l1TVFilter`, which is right: MATLAB's
  `halfQuadraticFilter` is an obsolete subclass of it. MATLAB's EDAX reader, 3D KAM,
  calcComponents and meanFilter changed, and the comparison anew shows them agreeing: load_h5_edax
  7.3e-8, kam3d 6.8e-7 (the offset the comparison took off MATLAB's KAM is gone), the mean filter
  6e-12 on twins and 8e-10 on ferrite. calcComponents keeps 4.5e-2 against MATLAB's 16 modes, the
  flat ridge where every port's mode lies elsewhere.

## Open questions

### Why MATLAB finds 22 grains fewer on 1C_1

`calcGrains` of the 2.5 million pixel steel map gives 118930 grains in both ports and 118908 in
MATLAB, the centroids 1.3e-4 apart; MATLAB's boundary vertices sit up to 1e-6 off the lattice.

### Which calcODF and calcDensity are right on the large inputs

`calcODF` of the Dubna pole figures is 1.5e-2 from MATLAB's, `calcDensity` of 1e6 orientations
7.1e-3 (1e4 agree to 4e-8); Julia has no half for either, which would break the tie.
