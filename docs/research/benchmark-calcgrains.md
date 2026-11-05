# calcGrains against MATLAB and the Julia port

**Historical results.** See [the September 28 release audit](calcgrains-release-audit.md)
before using the tables below: the emsland vertex-weld issue is already fixed, the MATLAB
forsterite input had an incorrect phase mapping (now fixed in the benchmark), and the two
ports use different default angles. The audit contains current comparisons with explicit
parameters and identifies the remaining minPixel defect.

Run 2026-09-23, `bench/calcgrains.py`, `bench/calcgrains.jl` and `bench/matlab/bench_calcgrains.m`.
Every map is read once, by this port's reader, and exported as its real measurements (the
padding of a grid left out): position, phase and quaternion per measurement, the notIndexed
ones with a NaN rotation as the readers give them, and one line per phase with its point
group, cell and alignment. Each port builds its map from that export, puts it on its grid
itself and segments it, so the three see one input and differ only in `calcGrains`. The
partitions are compared by position, not by index.

Machine: a laptop, 13th Gen Intel Core i7-1365U, 12 logical CPUs, 30 GB. Python 3.12.13,
NumPy 2.5.3, numba 0.67 on 12 threads; Julia 1.12.7 with `-t auto`, 12 threads; MATLAB
R2024b on the frameSymmetry branch through the bridge, whose session runs with
`maxNumCompThreads` 2. Times are the best of three after a warm up, the map copied outside
the clock, MATLAB's measured with tic and toc inside MATLAB, the load excluded.

The cases, the same keywords in all three ports:

| case | angle | minPixel | alpha | removeQuadruplePoints |
|---|---:|---:|---:|---|
| a | 10° | 1 | 3.1 (the default) | no |
| b | 5° | 5 | 3.1 | no |
| c | 10° | 1 | 0 (no closing) | no |
| d | 10° | 1 | 3.1 | yes (twins and forsterite) |

The maps: `twins` (22 879 measurements, magnesium, no gaps), `copper` (a hexagonal grid,
16 116), `csl` (154 112, iron), `forsterite` (245 952, four phases, 58 485 notIndexed),
`emsland` (270 000, ferrite and austenite, 18 393 notIndexed) and `synthetic`, the
600 x 600 Voronoi map of 800 seeds of `bench/maps.py` (one phase, no gaps).

## Time and counts

P, J and M are this port, Julia and MATLAB; times in seconds.

| map | measurements | case | P s | J s | M s | grains P / J / M | segments P / J / M | triple points P / J / M |
|---|---:|---|---:|---:|---:|---|---|---|
| twins | 22879 | a | 0.055 | 0.025 | 0.112 | 125 / 125 / 125 | 3892 / 3892 / 3891 | 121 / 121 / 121 |
| twins | 22879 | b | 0.094 | 0.046 | 0.164 | 82 / 82 / 82 | 3719 / 3719 / 3708 | 115 / 115 / 115 |
| twins | 22879 | c | 0.047 | 0.013 | 0.105 | 131 / 131 / 131 | 3951 / 3951 / 3951 | 131 / 131 / 131 |
| twins | 22879 | d | 0.062 | 0.033 | 0.135 | 103 / 103 / 103 | 3895 / 3895 / 3894 | 127 / 127 / 127 |
| copper | 16116 | a | 0.069 | 0.025 | 0.119 | 777 / 777 / 777 | 13005 / 12803 / 13005 | 1304 / 1304 / 1304 |
| copper | 16116 | b | 0.110 | 0.043 | 0.239 | 389 / 388 / 389 | 10126 / 9862 / 10071 | 646 / 644 / 644 |
| copper | 16116 | c | 0.058 | 0.019 | 0.125 | 777 / 777 / 777 | 13005 / 13005 / 13005 | 1304 / 1304 / 1304 |
| csl | 154112 | a | 0.332 | 0.095 | 0.507 | 924 / 924 / 924 | 22211 / 22211 / 22211 | 1517 / 1517 / 1517 |
| csl | 154112 | b | 0.560 | 0.213 | 0.757 | 502 / 502 / 502 | 21393 / 21393 / 21365 | 875 / 875 / 875 |
| csl | 154112 | c | 0.310 | 0.083 | 0.463 | 929 / 929 / 929 | 22229 / 22229 / 22229 | 1519 / 1519 / 1519 |
| forsterite | 245952 | a | 0.505 | 0.282 | 0.920 | 3185 / 3185 / 3160 | 45840 / 45840 / 45372 | 3578 / 3578 / 3498 |
| forsterite | 245952 | b | 0.830 | 0.431 | 1.540 | 976 / 976 / 972 | 37261 / 37261 / 36651 | 1710 / 1710 / 1712 |
| forsterite | 245952 | c | 0.568 | 0.179 | 1.089 | 24099 / 24099 / 24132 | 147559 / 147559 / 147748 | 11358 / 11358 / 11412 |
| forsterite | 245952 | d | 0.562 | 0.269 | 1.142 | 3005 / 3005 / 2944 | 45931 / 45931 / 45492 | 3760 / 3760 / 3733 |
| emsland | 270000 | a | 0.626 | 0.280 | 2.089 | 14875 / 15021 / 15012 | 127504 / 124897 / 124664 | 19331 / 18639 / 18655 |
| emsland | 270000 | b | 0.979 | 0.461 | 2.691 | 4296 / 4312 / 4310 | 93187 / 90005 / 89782 | 7553 / 7436 / 7417 |
| emsland | 270000 | c | 0.770 | 0.236 | 2.604 | 27288 / 27288 / 27288 | 146395 / 146395 / 146395 | 27934 / 27934 / 27934 |
| synthetic | 360000 | a | 0.771 | 0.265 | - | 382 / 382 / - | 33594 / 33594 / - | 652 / 652 / - |
| synthetic | 360000 | b | 1.394 | 0.499 | - | 580 / 580 / - | 39007 / 39007 / - | 1043 / 1043 / - |
| synthetic | 360000 | c | 0.767 | 0.223 | - | 382 / 382 / - | 33594 / 33594 / - | 652 / 652 / - |

## The partitions

The adjusted Rand index over the measurements both ports give a grain, and the pixels that
differ: each grain of the first port matched to the grain of the second holding most of its
pixels, a pixel counted where that match fails or where only one port gives it a grain; split
into measured (indexed) pixels, which is the segmentation, and notIndexed ones, which is the
absorption of the gaps into grains. P-M, P-J, J-M are the three pairs.

| map | case | ARI P-M / P-J / J-M | indexed pixels that differ P-M / P-J / J-M | notIndexed pixels that differ P-M / P-J / J-M |
|---|---|---|---|---|
| twins | a | 1.00000 / 1.00000 / 1.00000 | 0 / 0 / 0 | 2 / 0 / 2 |
| twins | b | 1.00000 / 1.00000 / 1.00000 | 5 / 1 / 6 | 2 / 0 / 2 |
| twins | c | 1.00000 / 1.00000 / 1.00000 | 0 / 0 / 0 | 0 / 0 / 0 |
| twins | d | 1.00000 / 1.00000 / 1.00000 | 0 / 0 / 0 | 2 / 0 / 2 |
| copper | a | 1.00000 / 1.00000 / 1.00000 | 0 / 0 / 0 | 0 / 0 / 0 |
| copper | b | 1.00000 / 0.99998 / 0.99998 | 2 / 7 / 13 | 0 / 0 / 0 |
| copper | c | 1.00000 / 1.00000 / 1.00000 | 0 / 0 / 0 | 0 / 0 / 0 |
| csl | a | 1.00000 / 1.00000 / 1.00000 | 0 / 0 / 0 | 0 / 0 / 0 |
| csl | b | 1.00000 / 1.00000 / 1.00000 | 28 / 9 / 23 | 0 / 0 / 0 |
| csl | c | 1.00000 / 1.00000 / 1.00000 | 0 / 0 / 0 | 0 / 0 / 0 |
| forsterite | a | 0.99917 / 1.00000 / 0.99917 | 368 / 0 / 368 | 1282 / 838 / 1092 |
| forsterite | b | 0.99888 / 1.00000 / 0.99889 | 413 / 27 / 400 | 1347 / 586 / 1176 |
| forsterite | c | 0.99936 / 1.00000 / 0.99936 | 390 / 0 / 390 | 0 / 0 / 0 |
| forsterite | d | 0.99918 / 1.00000 / 0.99918 | 364 / 0 / 364 | 1281 / 838 / 1091 |
| emsland | a | 0.99770 / 0.99769 / 1.00000 | 660 / 660 / 3 | 1063 / 108 / 964 |
| emsland | b | 0.98976 / 0.98982 / 0.99992 | 1660 / 1095 / 570 | 831 / 154 / 685 |
| emsland | c | 1.00000 / 1.00000 / 1.00000 | 0 / 0 / 0 | 0 / 0 / 0 |
| synthetic | a | - / 1.00000 / - | - / 0 / - | - / 0 / - |
| synthetic | b | - / 1.00000 / - | - / 0 / - | - / 0 / - |
| synthetic | c | - / 1.00000 / - | - / 0 / - | - / 0 / - |

## What it says

- **Without gaps to close the three ports agree to the pixel.** twins, copper, csl and the
  synthetic map at the default, emsland and forsterite without closing (case c, but for the
  MATLAB Enstatite defect below): the same grains, the same boundary segments, the same triple
  points. On twins MATLAB gives two notIndexed pixels a grain that the ports leave at 0, and
  counts one segment less; the grains of the measured pixels are the same.
- **Time: this port is two to three times behind Julia and one and a half to four times ahead
  of MATLAB.** Julia runs on 12 threads, MATLAB on the 2 the bridge session has, so the MATLAB
  column is not MATLAB's best. On the forsterite map the absorption of the notIndexed pixels
  into grains (`absorbInteriorPixels`) was a Python loop over the pixels until this run:
  2.9 s of 3.4 s, 85 percent of the reconstruction, which is why the first run of this
  benchmark measured 9.1 s there while another session loaded the machine. It is now a compiled
  loop (`pixelKernel("absorb")` in `mtex/maps/kernels.py`) with the same grain ids, 0.5 s.
- **Where the port and Julia differ, it is the absorption or the cull, not the segmentation.**
  On forsterite they segment every measured pixel identically in all four cases (ARI 1, 0
  indexed pixels); they differ only in which of the 58 485 notIndexed pixels a grain absorbs
  (838), the corner test of `absorbInteriorPixels` deciding otherwise near boundaries. The
  minPixel cull (case b) moves a handful of pixels on twins, copper and csl in every pair.
- **emsland is the exception: the port's closing joins what Julia and MATLAB keep apart.**
  At the default the port finds 14 875 grains where Julia finds 15 021 and MATLAB 15 012, and
  Julia and MATLAB agree to 3 measured pixels. Without closing all three agree, and the NumPy
  fallback of the kernels gives the same 14 875, so it is the closing itself: 140 grains of
  the port hold two or more of Julia's, mostly a grain of one to twenty pixels joined across a
  one or two pixel notIndexed seam to a grain less than 10° away, inside the map (5 of the 140
  within a step of the rim). Reproducer: case a of `emsland` - the measurement at
  (27.2381, 0) forms a grain of one pixel in Julia and MATLAB and is joined in the port to the
  grain holding (27.2381, 3.0265), across the two notIndexed pixels between them (misorientation
  2.0°). This is the port's side of the known M3 question (MartensiteVariants in
  `docs/findings.md`: 3523 and 5201 grains on emsland for MATLAB's 3551 and 5225).
- **forsterite, port and Julia against MATLAB.** 3185 grains for MATLAB's 3160 at the default;
  368 measured pixels differ. Without closing 390 still differ, all of them Enstatite: MATLAB
  splits neighbouring Enstatite pixels whose misorientation it computes itself as 0.18°, 0.34°
  and 0.30° (the measurements at (6100, 1100), (6100, 1150), (6150, 1150) and (6200, 1150),
  related by a half turn of mmm), which the ports join. So most of the forsterite difference is
  MATLAB's criterion on that one phase, not the closing.
- **The quadruple point split (case d)** agrees between the port and Julia on both maps and
  moves forsterite from 3185 to 3005 grains (MATLAB 3160 to 2944).

## Closing the gap to Julia, 2026-09-24

The 2.53 million pixel steel map `1C_1.ctf` (Fe-BCC and Fe-FCC, 0.15 µm steps) runs on request,
`bench/calcgrains.py export 1C_1`, against Julia only. Both ports give the same grains on it
(118 930 at the defaults, 334 629 without the closing). This port first took 9.7 s where Julia
takes 5.7 s, the time spread over every stage; each stage was then made a compiled pass, the
NumPy form kept as the fallback and the oracle, and every one of the 22 map and case runs gives
counts and partitions byte-identical to the results before, the boundaries and polygons of
twins, forsterite and 1C_1 bit-identical as well.

| stage on 1C_1 | before | after | what changed |
|---|---:|---:|---|
| decomposition | 4.2 s | 1.0 s | the vertex weld one numba pass over a coarse bucket grid with a union-find, the rules of the NumPy weld kept |
| misorientation criterion | 1.5 s | 0.13 s | one parallel loop over the pairs against a padded table of the proper groups |
| grain numbering | 0.58 s | 0.29 s | numbered by first appearance in one loop instead of unique and a double argsort |
| grain table | 1.2 s | 0.17 s | compiled passes: the reference, the symmetry reduction, the sums in `np.add.at` order, the spread |
| boundary, outer and inner | 1.4 s | 0.26 s | built once in its final order, the walk and the chain reversal one permutation, the sorts counting sorts |
| grain polygons | 0.70 s | 0.15 s | the loops traced in parallel, each grain into its own slice |
| absorption of notIndexed pixels | 2.9 s (forsterite) | 0.1 s | the corner search one numba loop |

`calcGrains`, best of three, 12 threads each, the machine at a load of about 3 from other work:

| map | case | Python before | Python after | Julia |
|---|---|---:|---:|---:|
| 1C_1 | a, defaults | 9.75 s | 2.11 s | 3.44 s |
| 1C_1 | c, no closing | 10.71 s | 3.11 s | 2.68 s |
| forsterite | a | 2.93 s | 0.36 s | 0.26 s |
| forsterite | b, 5°, minPixel 5 | | 0.59 s | 0.47 s |
| emsland | a | 1.99 s | 0.38 s | 0.28 s |
| emsland | b | | 0.65 s | 0.51 s |

The port is now within 1.2 to 1.7 times of Julia on every case and ahead of it on the large map
at the defaults. What is left is mostly the NumPy glue between the kernels (gathers and
concatenations of the per-pixel arrays), not one stage.

## MATLAB develop's reconstruction, 2026-11-05

calcGrains follows MATLAB develop at 175d8c803: the boundaries through the cells the closing filled
drawn straight (`straighten=True`, MATLAB's `'straighten'`, `mtex/maps/straighten.py`), the bay mouth
reopened only as far as the largest smaller closing leaves it open and repeated while the outside grows
(`openCorridors`), a vertex where four outer faces meet split (`splitPinches`), the decomposition repeated
while the minPixel cull re-fragments a grain, the empty cells of a hole joined to its smallest measured
pixel (`joinHoles`), and the notIndexed pixels absorbed by the grain polygons they lie in. smoothBoundary
is D6 of the grain reconstruction plan: 20 Taubin pairs with every vertex of a pixel edge held in its
pixel cell, then the side constrained simplification at d/10.

### Against MATLAB

`bench/calcgrains_ops.py` (the MATLAB half `bench/matlab/bench_calcgrains_ops.m`) on small (10°, minPixel 5),
peridotite (alpha 2.1, 6.1 and 16.1, minPixel 5), forsterite (minPixel 1 and 5) and the T1 instance with
cracks; besides it the 21 instances of the T1 quick subset of the grain benchmark, both with and without
straightening.

- Grain counts, indexed grain counts and boundary segment counts are MATLAB's on every map and setting.
- Along the bisectors every segment between two grains lies within 1.3e-9 of MATLAB's (coordinates up to
  3e4 µm). The outer boundary differs only at the map corners, where MATLAB's vertices lie 1e-4 to 7e-4
  pixels off the lattice; the grain areas differ by that.
- Straightened, the vertices lie within 2e-3 to 5e-3 µm of MATLAB's (a ten thousandth of a pixel). A cut point is
  found by 16 bisections; where a midpoint falls exactly on a lattice edge the fill test is zero and
  rounding decides the side (this port takes a tie as zero, MATLAB its 1e-14 of the affine fit).
- Where the two ports hold different vertices, the cause is MATLAB's sample count of a swept area,
  `ceil(3 len / dxy)`, for a segment exactly one pixel long: rounding of `len` gives 3 or 4 samples, and the
  fourth finds a sliver of a measured cell the third misses. On forsterite one chain of five segments is
  held in MATLAB and straight here (32 µm apart), on the hexagonal T1 instance two of 400 segments in the
  first round, after which 1877 vertices differ by up to 0.22 pixels. A count with a tolerance,
  `ceil(3 len / dxy - 1e-9)`, would make it deterministic in both ports.
- The absorbed notIndexed pixels differ in two ways, both MATLAB's: MATLAB counts a nudged corner on a
  boundary as inside, so it absorbs a pixel a diagonal bisector runs through (forsterite: about 1000
  pixels it absorbs and this port does not); and it tests only the grain of the one nearest measured pixel, so
  after straightening a pixel wholly inside the other grain stays at 0 (forsterite: 597 pixels this port
  absorbs, 199 of the first 200 checked to lie wholly in one of MATLAB's own polygons).

### Time

bayes (16 cores, at a load of 11 to 13 from other users), 4 threads, best of five (1C_1 of three), the map copied
outside the clock. main is pymtex main, before the closing classification of MATLAB develop; base the
branch this work started from (the closing, dd94245c0's corridor rule, D6 smoothing); port the first
commit of the MATLAB develop port with the straightening in NumPy; now its compiled form.

| map | setting | calcGrains main | base | port | now | smoothBoundary main | base | now |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| small | 10°, minPixel 5 | 0.024 | 0.025 | 0.042 | 0.034 | 0.002 | 0.005 | 0.003 |
| peridotite | alpha 2.1, minPixel 5 | 0.213 | 0.220 | 0.582 | 0.361 | 0.024 | 0.108 | 0.037 |
| peridotite | alpha 6.1, minPixel 5 | 0.240 | 0.290 | 0.706 | 0.428 | 0.020 | 0.080 | 0.029 |
| peridotite | alpha 16.1, minPixel 5 | 0.672 | 0.722 | 1.183 | 0.881 | 0.017 | 0.077 | 0.026 |
| forsterite | defaults | 0.189 | 0.199 | 1.057 | 0.380 | 0.043 | 0.170 | 0.061 |
| forsterite | minPixel 5 | 0.331 | 0.385 | 1.378 | 0.720 | 0.033 | 0.151 | 0.043 |
| 1C_1 | 10° | 2.63 | 3.13 | 30.9 | 5.73 | 1.71 | 6.85 | 2.44 |

With `straighten=False` now takes 0.321 s on peridotite alpha 2.1, 0.309 s and 0.687 s on forsterite and 3.79 s on
1C_1: what is left beyond base there is the absorption by the grain polygons and, with minPixel, the third
decomposition the loop takes where the cull re-fragments a grain (forsterite minPixel 5, as in MATLAB).

main's smoothBoundary is the Laplacian one before D6, main's calcGrains gives other grain counts (665 for
609 on peridotite alpha 2.1, 3120 for 3104 on forsterite, 118 930 for 118 899 on 1C_1). On the laptop of
the first section, MATLAB R2024b (the bridge, 2 threads) and Julia (12 threads):

| map | setting | MATLAB calcGrains | MATLAB smoothBoundary | Julia calcGrains (closing branch, bisectors) | Julia smoothBoundary (D6 branch) |
|---|---|---:|---:|---:|---:|
| small | 10°, minPixel 5 | 0.067 | | | |
| peridotite | alpha 2.1 / 6.1 / 16.1, minPixel 5 | 1.15 / 2.07 / 2.37 | | | |
| forsterite | defaults / minPixel 5 | 2.45 / 4.21 | | 0.41 / 0.59 | 0.083 / 0.061 |
| 1C_1 | 10° | 42.7 (12.8 with `'straighten', false`) | 21.4 | 4.03 | 2.52 |

Where the time went, 1C_1 on bayes (cProfile, cumulative):

| stage | port | now | what changed |
|---|---:|---:|---|
| straightenInFill | 26.5 s | 1.9 s | the stages below |
| its crossing test, 5 rounds | 12.9 s | 0.35 s | every segment in the cells of a grid of four pixels its box meets, compiled, the grid built once with a second one of the segments moved since; a grid as large as the longest segment held thousands of segments per cell |
| its fill test, 24 calls | 7.9 s | < 0.1 s | the four lattice points around a point in one compiled loop, on the lattice's own origin and basis |
| its samples of the swept areas and the NumPy between | 3.4 s | 0.5 s | generated and tested in one compiled loop with their counts, stopping at the first sample outside |
| its solve, 5 rounds | 0.76 s | 0.5 s | over the boundary vertices alone, after the first round over the chains a held vertex belongs to, the system assembled from the half edges |
| buildSites | 0.89 s | 0.57 s | the corridor distances one compiled loop, the hull of the first and last cell of every row |
| smoothBoundary's Taubin pairs | 5.1 s | 0.5 s | the 40 steps and the projection into the pixel cells one compiled loop, the sparse product's sums in its order |

Every step but the solve's gives the vertices and the grain ids of the step before to the bit; the solve
moves them by rounding (at most 2e-11 µm). Against base, what 1C_1 still spends is the straightening
(1.9 s, a quarter of it the first round's solve over 515 000 free vertices, the rest the crossing test and
the samples of 760 000 moved segments and the setup around them) and the absorption's polygon test
(0.7 s, 0.24 s before). A raster of the cells boundary segments touch, to let a pixel far from them skip the
polygon test, cost more to build than it saved on 1C_1's small grains.

### The straightening after the Fable 5.1 proposals, 2026-11-05

Branch `claude/straighten-fable-548ece` on top of 9c12bd5. A MATLAB session had a Fable 5.1 agent propose
cheaper checks for `straightenInFill.m` (B, the same output) and a chord-based pinning rule (A, a changed
output); this is what of them pays in this port, measured stage by stage on 1C_1 (the complex before the
straightening pickled, `straightenInFill` alone, laptop, 4 threads, min of 3):

| stage of straightenInFill, 1C_1 | 9c12bd5 | now | what changed |
|---|---:|---:|---|
| whole | 1.86 s | 1.05 s | |
| crossing test, grids and queries | 0.47 | 0.25 | the sweep test runs first and the crossing test asks only for the segments that stay in the fill (the crossing finds 5490 of the 307 000 bad segments of round 1); one grid per round of the segments moved in it, where they are then, a query checking every grid, instead of a grid of all segments moved since the first, rebuilt each round (630 000 segments for the 234 queries of round 4); the grids built from the vertices and the segment list in place, within the box of the held vertices, over the cells the queries meet; two pixel cells instead of four |
| solve | 0.31 | 0.08 | the chains of the system eliminated: 55 % of the 515 000 free vertices of round 1 have two free neighbours and 10 % three or more, so every path of vertices with at most two free neighbours is reduced by the Thomas recursion onto its anchors, the 50 000 anchors solved by SuperLU, the chains substituted back (`chainSolveNumba`); the only step that moves the output, by the rounding of the elimination order: at most 1e-10 µm in a vertex, 4e-7 µm² in an area, no grain id |
| sweep samples | 0.26 | 0.15 | the fill test takes the roots of the two nearest lattice points instead of all four (a root is monotone, so the same two are nearest); the new segment sampled first, where a sweep leaves the fill; a segment whose ends stayed sampled once |
| setup: fill test, cut, half edges | 0.44 | 0.28 | the boundary faces gathered once, the two-column reductions (`max(axis=1)`, `min(axis=0)`, `any(axis=1)`) written out, the neighbour of a half edge from the segments' vertex sums, the split faces appended instead of re-gathered |
| per round | 0.3 | 0.2 | the held vertices and the chains to re-solve marked instead of `unique` and `isin` |

What of proposal B does not pay here: the raster lookup instead of four lattice points - the fill test
already reads the raster at the four points, and skipping the roots where the nearest point is filled
saved nothing measurable (0.156 to 0.150 s) through the call between compiled helpers; the line and path
samples with an island check - they change the sample set, so the output, and the early exit for a swept
quad whose lattice box is all fill applies to 1 % of them, the fill being one to three pixels wide; the
cheaper components - `connected_components` costs 0.02 s. The cut's 16 bisections and the direct assembly
of the CSC were tried and left: 0.03 s, and the sort of the indices moved into the solve.

On bayes (16 physical cores, 4 threads, at a load of 9 to 12 from other users, both versions in one job,
best of five, 1C_1 of three; now = f41d999, the state of the table above):

| map | setting | calcGrains now | straightened | smoothBoundary now | straightened |
|---|---|---:|---:|---:|---:|
| small | 10°, minPixel 5 | 0.035 | 0.030 | 0.003 | 0.002 |
| peridotite | alpha 2.1, minPixel 5 | 0.344 | 0.351 | 0.038 | 0.038 |
| peridotite | alpha 6.1, minPixel 5 | 0.419 | 0.417 | 0.027 | 0.028 |
| peridotite | alpha 16.1, minPixel 5 | 0.849 | 0.853 | 0.027 | 0.026 |
| forsterite | defaults | 0.393 | 0.376 | 0.060 | 0.064 |
| forsterite | minPixel 5 | 0.758 | 0.757 | 0.044 | 0.043 |
| 1C_1 | 10° | 5.88 | 5.05 | 2.49 | 2.47 |

The smaller maps straighten in a few hundredths of a second either way. What 1C_1 still spends beyond
`straighten=False` (3.79 s) is about 1.2 s: the two compiled tests (0.4 s), the setup and the bookkeeping
of five rounds in NumPy (0.5 s), the solve (0.1 s); beyond base, the absorption's polygon test (0.7 s).

**Proposal A** is in as the private switch `PIN_FARTHEST` of `mtex/maps/straighten.py`, off: a chain of
free vertices with a segment that sweeps out of the fill or crosses holds only the vertex whose old
position lies farthest from the chord through the chain's solved ends (Douglas-Peucker), the chains read
off the elimination, instead of both ends of every bad segment; a bad segment without a free chain vertex
holds both ends as before. Measured (`calcGrains`, the indexed pixels compared by grain):

| | default | PIN_FARTHEST |
|---|---:|---:|
| 1C_1 rounds | 5 | 12 (112 000 vertices held in round 1 for 400 000, then 90 000, 50 000, 24 000, ...) |
| 1C_1 boundary length | 162 566 µm | 161 703 µm (-0.53 %) |
| 1C_1 vertices moved | 200 530 | 308 926 |
| 1C_1 grains whose area changes | | 81 434 of 118 899, at most 0.35 µm² |
| 1C_1 pixels changing grain | | 18 271 of 2 529 549, all notIndexed in the fill, absorbed (8835) or released (9436); no indexed pixel |
| forsterite boundary length | 2 052 205 µm | 2 036 501 µm (-0.77 %) |
| forsterite pixels changing grain | | 1404 of 245 952, all notIndexed, 635 absorbed and 769 released; one one-pixel grain's polygon in a crack shrinks by 58 % |
| straightenInFill 1C_1, laptop | 1.05 s | 1.75 s |
| calcGrains, bayes, one job at 1892ccb | 1C_1 4.76 s, forsterite 0.341 s | 1C_1 5.70 s, forsterite 0.387 s |

A costs more, not less, here: every round re-solves the whole chains a held vertex belongs to, so their
vertices move for real and their segments are tested again (609 000 in round 2 for both rules, 343 000 in
round 3 for 20 000), the rounds halve from 12 instead of ending after 5, and the queries check one grid
per round; a threshold on the move changed nothing, the moves are not rounding. The straighter result and
its price are the decision; it leaves the parity with MATLAB develop (`bench/calcgrains_ops.py`).

**The 39 000 segments found out of the fill where they are.** In round 2, 39 058 segments of 1C_1 whose
ends are back at their old positions fail the sweep test at their old position itself. They are the faces
between two *direct-neighbour* measured pixels of different grains that continue into the fill: the sample
points lie 0.53 to 0.66 px from both pixels, on their shared cell edge, up to 0.44 px out of the fill, while
both ends sit on the fill's edge (fill distance -7e-12), so the bisection, which cuts only where the sign
changes between the ends, never cuts them. Legitimate: that part of the boundary must not move. The cost is a
retest per round and the inside end of each held where a cut at the fill entry would let it move; MATLAB
develop has the same, so it stays for parity (`docs/plans/develop-fixes.md`).

**The second rule under study, `CHAIN_DP`** (off, `mtex/maps/straighten.py`): the first solve places the
free junctions, which are then held; every chain of degree-two vertices between held vertices is then
straightened on its own, Douglas-Peucker on its chord (`chainsNumba` walks the chains, `chainDPNumba` runs
them in parallel: the vertices between two decided ones evenly on the chord, a failing sweep pins the vertex
farthest from the chord and does both parts again); a junction whose segment to a decided vertex sweeps out
goes back and its chains are done again (81 000 of them on 1C_1), both segments of a crossing keep their
vertices (the crossing test reports the partner; pinning only the query looped forever). Six iterations on
1C_1, four on forsterite, no global rounds after the first solve.

| | default | PIN_FARTHEST (A) | CHAIN_DP |
|---|---:|---:|---:|
| 1C_1 straightenInFill, laptop | 1.05 s | 1.75 s | 1.04 s |
| 1C_1 boundary length | 162 566 µm | 161 703 (-0.53 %) | 161 876 (-0.42 %) |
| 1C_1 vertices moved | 200 530 | 308 926 | 331 302 |
| 1C_1 pixels changing grain | | 18 271 notIndexed | 25 673 notIndexed (13 359 absorbed, 12 314 released), none between two grains, so no indexed pixel |
| forsterite calcGrains, laptop | 0.427 s | 0.468 s | 0.403 s |
| forsterite boundary length | 2 052 205 µm | 2 036 501 (-0.77 %) | 2 038 710 (-0.66 %) |
| forsterite pixels changing grain | | 1 404 notIndexed | 1 613 notIndexed, no indexed pixel |
| calcGrains, bayes (e3f5629, one job, load 11-12) | 1C_1 4.92 s, forsterite 0.366 s | 1C_1 5.63 s, forsterite 0.400 s | 1C_1 4.92 s, forsterite 0.377 s |

CHAIN_DP gives A's straightness at the default's cost: what remains of its time is round 1 (setup 0.28 s,
solve 0.1 s) and the chain pass (0.16 s), the junction redo (0.06 s), one crossing check (0.12 s). Both
rules change the result and leave the parity with MATLAB develop; the choice is Ralf's.

## MATLAB defects found on the way

Rechecked on MTEX's own sample data (`bench/matlab/probe_calcgrains_bugs.m`), one of the three
first suspected holds:

- **calcGrains fails on a map without indexed pixels**, in `grain2d` line 174 (index out of
  bounds): `calcGrains(mtexdata('forsterite')('notIndexed'))`. The maps built here from a list
  of phase numbers that did not match the phase map were all notIndexed, which is how it showed;
  filed as mtex-toolbox/mtex#2624.
- The orientation of a notIndexed pixel does not change MATLAB's grains on the loaded map
  (3106 grains with NaN and with the identity), and the Enstatite pixels at (6100, 1100) and
  its neighbours form one grain there. The 24 378 grains and the split Enstatite came from the
  way the exported map was rebuilt in MATLAB, not from `calcGrains`; the MATLAB column of
  forsterite above carries that rebuild and is to be read with it.

## Reproducing

```
.venv/bin/python bench/calcgrains.py export
.venv/bin/python bench/calcgrains.py run
julia -t auto --project=../mtexJulia/dev bench/calcgrains.jl
.venv/bin/python bench/calcgrains.py matlab        # the bridge session mtexfs, one call per map and case
.venv/bin/python bench/calcgrains.py compare
```

The export, the per-port counts and the partitions go to `bench/out/calcgrains/`. The cross-check
against MATLAB develop:

```
.venv/bin/python bench/calcgrains_ops.py export     # the T1 crack instance of the grain benchmark
.venv/bin/python bench/calcgrains_ops.py run
.venv/bin/python bench/calcgrains_ops.py matlab     # bench/matlab/bench_calcgrains_ops.m through the bridge
.venv/bin/python bench/calcgrains_ops.py compare
```
