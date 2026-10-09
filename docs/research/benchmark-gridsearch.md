# The SO(3) grid search for ODF reconstruction, rethought 2026-09-21

The operation ODF reconstruction from pole figures needs is the radius search: the ODF is a
sum of local kernels centred at the grid points, so evaluating it, or building the kernel
matrix, means finding all grid points within a kernel support of a query. The grid must be
near-equidistant and support that search quickly. This note records the rethink.

## The grid: the ring grid is kept, Fibonacci added as an option

Measured uniformity on full SO(3) at equal point count (N = 119088), nearest-neighbour
misorientation spacing and the covering radius (the max over random orientations of the
distance to the nearest grid point):

| grid | NN spacing mean | NN spacing cv | covering radius |
| --- | ---: | ---: | ---: |
| equispaced ring (MATLAB) | 4.99° | 0.009 | 4.08° |
| super-Fibonacci (Alexa 2022) | 3.62° | 0.077 | 4.62° |

The ring grid packs more evenly and covers slightly better, so it stays the default.
Super-Fibonacci (`superfibonacciSO3Grid`, `grids/superfibonacci.py`) is added as an
alternative: a low-discrepancy quaternion spiral trimmed to the fundamental region, trivial
to generate at any count with no ladder structure, its resolution calibrated to the mean
nearest-neighbour spacing.

## The search: a quaternion k-d tree, replacing the ring search

`grids/orientationtree.py`: a `scipy.spatial.cKDTree` over the grid quaternions. Orientation
angle w becomes the Euclidean radius 2 sin(w/4) on the double cover [q, -q], so a ball on the
tree is a misorientation ball. The grid is one fundamental region, so a query is symmetrised
and every image searched; the union of the hits is the symmetry-reduced neighbourhood. It
uses only the points, so the ring grid and the Fibonacci grid search the same way.

`SO3Grid.find` and `SO3FibonacciGrid.find` both route through it, giving the three MATLAB
forms: nearest (`eps` None), the k nearest (an integer), and all within a radius (a float,
returning the sparse incidence and the sparse reduced angles). All three are exact against a
brute-force symmetry-reduced distance to every grid point, across m-3m, 222, 6/mmm, -3m.

Measured, cubic 2.5° grids, best of three:

| grid | points | nearest | radius @ 7.5° | grid×grid neighbours |
| --- | ---: | ---: | ---: | ---: |
| ring | 39946 | 5.3 µs/query | 124 µs/query | 4.8 s |
| Fibonacci | 15192 | 6.2 µs/query | 76 µs/query | 1.4 s |

One caveat drove the design: scipy's plain nearest `query` prunes poorly on the 4-sphere
(44 s for 1e4 cubic queries), while `query_ball_point` and a distance-bounded `query` prune
to the region and run in milliseconds. So nearest and k-nearest use a distance-bounded query
widened until every query resolves; radius uses `query_ball_point`. All run with
`workers=-1`.

## The ODF radius search (deferred to P2)

The radius search the ODF reconstruction needs was rethought against the real use case
(build the grid once, then evaluate f(o) = sum_i c_i psi(dist(o, g_i)) over ~1e6
orientations). Returning the neighbour matrix does not scale - 1e6 queries times ~114
neighbours is a 114M-nonzero matrix - so the evaluation should fuse search, kernel and sum
into one compiled pass that outputs one number per query. A halo cell list does this: at
setup, symmetrise the grid and keep the images within delta of the fundamental region (the
halo), bin them by the quaternion imaginary part (no cubochoric - the region plus halo never
reaches 180 deg, so no seam); on the hot path, fold each query into the region and walk its
cell and neighbours in a numba loop. Prototyped and validated 2026-09-21: cubic 2.5 deg grid,
delta 7.5 deg, 2.3 s per 1e6 (2.1 us/query), exact vs the brute image sum, versus 105
us/query for the k-d tree radius. It is deferred until the SO3 functions (P2) are ported,
since the kernel psi belongs to that layer; the design is recorded in the agent memory
`odf-halo-search`.

## What this replaces

The bespoke compiled ring search of the previous round (a numba loop over the ring and gamma
ladders) is retired for SO(3): it optimised the nearest search, but ODF needs the radius
search, and the k-d tree does both, more simply, on any grid. The S2 ring grid keeps its
compiled search (`grids/gridsearch.py`), unchanged. Not done: the radius search of the
pole-figure S2 grids, and profiling the ODF forward operator that will drive the grid×grid
kernel matrix.

How to run:

    .venv/bin/python -m pytest -q tests/test_orientationtree.py tests/test_superfibonacci.py

## Against Julia and MATLAB

The same operations on a cubic 2.5° grid (~40000 points), best of three, this machine.
`bench/gridsearch.py`, `bench/gridsearch.jl`, `bench/gridsearch.m`. Julia's `nearest` and
`neighbours` and MATLAB's `find` are structured ring searches (MATLAB's is a compiled MEX);
Python's is the quaternion k-d tree of this note.

| operation | Python k-d tree | Julia ring | MATLAB MEX ring |
| --- | ---: | ---: | ---: |
| build | 0.9 s | 12 s | 0.45 s |
| nearest, per query | 4.4 µs | 6.2 µs | 6.1 µs |
| radius @ 7.5°, per query | 105 µs | 17 µs | 24 µs |

All three return the same ~114 neighbours per query. What the numbers say:

- **Nearest**: the k-d tree is the fastest, a touch ahead of both ring searches.
- **Build**: Python matches MATLAB's compiled constructor and is far ahead of Julia's
  (whose 12 s is a cold build including the JIT of its construction path, not steady state).
- **Radius, the ODF-critical operation**: the k-d tree is 4 to 6x slower than the compiled
  ring searches. The cost is not the tree - `query_ball_point` is fast - but the Python work
  around it: symmetrising each query into its 24 images, then the per-hit loop that unions
  the images and builds the sparse angle matrix. Julia and MATLAB do that pass in compiled
  code. At 105 µs/query the grid×grid kernel-matrix build for a 40000-point grid is about
  4 s, fine as a one-time ODF setup; it would hurt only if the radius search sat in a hot
  loop. Closing the gap means moving the union-and-sparse-build into a compiled kernel (as
  the pointwise kernels were), or keeping the ring search for radius while the tree serves
  nearest - which brings back the ring machinery this note retired.

## The builds and the searches of every orientation grid, 2026-10-09

m-3m, ms, warm, before / after. MATLAB's 1.25 ms for `equispacedSO3Grid(m-3m, 2.5 degree)` in the
geometry crosscheck is its `persistent` cache answering the same arguments, not a build.

| | before | after |
|---|---:|---:|
| `equispacedSO3Grid`, 2.5 degree (39242 nodes), again | 12.5 | 0.14 |
| `equispacedSO3Grid`, 1 degree (623685 nodes), first / again | 170 / 162 | 72 / 1.9 |
| radius search of 24761 queries at 0.5 degree on the 1 degree grid (its merge) | 57.6 | 7.5 |
| `fibonacciSO3Grid`, 2.5 degree | 122 | 12.7 |
| `superfibonacciSO3Grid`, 2.5 degree | 68 | 7.1 |
| `find(regularSO3Grid(2.5 degree), 1000 orientations)` | 4148 | 97 |
| `find` of 1000 among 1e5 orientations | 2170 | 54 |

- **The structured radius search filled a grid-sized array per chunk of 16 queries**: `best`, one
  cos(w/2) per grid point, and the list of points touched, 5 MB each on the 1 degree grid, 1548
  chunks in each of the two passes, about 15 GB written for 28 thousand pairs. Each thread now has
  its rows, allocated once per call inside `heapBlocks` and set to -1 by the thread on its first
  chunk; every query restores what it touched, so both passes share them (numba's
  `get_thread_id`, below the thread count the launch sets).
- **The last four grids built are kept** by their frames and resolution, as MATLAB keeps its last
  one, and handed out as copies with their own coordinates; above 2e6 nodes none is kept.
  `max(odf)`, calcODF, the cluster search and the plots ask for the same grid again and again.
- **`find` in a plain orientation list** built the matrix of every distance, 22 ns a pair and the
  memory of all of them (1e4 queries among the 192 thousand orientations of a 2.5 degree regular
  grid: 15 GB). From 32 queries and 1e5 pairs on it goes through the quaternion tree of the grids
  over the list, which gives the same indices and angles; an antipodal list makes the queries
  antipodal, since the tree takes the flag from the images of the queries.
- **The Fibonacci grids** generated about a million spiral points in NumPy and tested them with
  NumPy against the region: the spiral is one compiled parallel loop now (`superfib`), and
  `orientationRegion.checkInside` takes the compiled `regionMembership` of the equispaced grid
  for its planes, the axis sector, where there is one, tested after.
- Measured and left: `subGrid` of the 1 degree grid (11 ms, NumPy's boolean indexing of 624
  thousand quaternions), the tree's radius search on the Fibonacci grid (44 ms against the
  structured 4.9 on the equispaced grid of the same size).

## The S2 grids, 2026-10-09

ms, warm, 1e4 queries for the nearest, 2000 queries within three times the resolution for the
radius, before / after; the builds were not the cost (0.2 to 14 ms up to half a degree).

| grid at 2.5 degree | nearest | radius |
|---|---:|---:|
| `equispacedS2Grid` (6600 points; antipodal 3300) | 0.8 / 0.8 | 398 / 14.7 (487 / 17.0) |
| `fibonacciS2Grid` (6477) | 192 / 2.3 | 113 / 11.4 |
| `regularS2Grid`, a plain list (10512) | 7020 / 4.0 | 1408 / 23.3 |
| `HEALPixS2Grid`, a plain list (12288) | 8059 / 4.3 | 1638 / 24.5 |

At half a degree (82 to 260 thousand points) the radius search takes 10 to 66 ms and the nearest
1 to 51 ms, the tree of a plain list built in every call.

- **One k-d tree over the points serves every set of directions** (`S2Tree` in
  `grids/search.py`): on the unit sphere the Euclidean nearest point is the angular nearest and
  the angle eps the chord 2 sin(eps/2); an antipodal set holds every point as v and -v, a point
  hit both ways counted once. A grid keeps its tree; a plain list builds one from 32 queries and
  1e5 pairs on. The radius search of the ring grid and both searches of the Fibonacci grid went
  query by query through Python; the nearest search of the ring grid stays its compiled ring
  search. Crystal directions measure by their symmetry and keep the matrix of every distance.
- **A defect on the way**: `find` passed antipodal=False to `angle` unless told otherwise, which
  reads as "never", so an antipodal list was searched as if it were not. It reads the flags of
  both lists now (antipodal=None), True and False forcing either way, as MATLAB's find reads
  v.antipodal.
