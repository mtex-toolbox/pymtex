# Grain reconstruction benchmark: instances, metrics, first smoke run

Milestone M1 of the research plan `../master/docs/grain-reconstruction-research-plan.md` (WP1,
WP2, WP4): synthetic EBSD maps whose truth is a continuous grain geometry, scores for the
partition and for the boundary geometry, the scores checked on cases with analytic answers, and
a first run of `calcGrains` and `smoothBoundary` in this port and in MATLAB. The code is
`bench/grains/`; every length is in pixel steps d, every area in d², every angle in degrees.

```
.venv/bin/python bench/grains/checks.py               # the metrics on the analytic cases
.venv/bin/python bench/grains/run.py generate T0      # the instances from their seeds
.venv/bin/python bench/grains/run.py run T0           # this port's candidates
.venv/bin/python bench/grains/run.py matlab T0        # bench/matlab/bench_grains.m through the bridge
.venv/bin/python bench/grains/run.py score T0         # bench/out/grains/T0/scores.csv
.venv/bin/python bench/grains/run.py table T0         # the smoke table below
.venv/bin/python bench/grains/run.py compare T0 [a b] # two ports run by run
```

`run.py matlab` connects to the engine named by `SESSION_NAME` (default `mtexcc`) through
`../master/docs/agents/matlab-bridge/matlab_run.py`, one instance per call.

## Instances

### Structures

A structure is a partition of a box in the plane, written as its boundary segments
`(x1, y1, x2, y2, left, right)` with the grain `left` on the left of the walk and 0 outside
(`planar.py`). The truth and every reconstruction take this form, so one code scores both.

| id | generator | parameters |
|---|---|---|
| G1 | Poisson Voronoi | D, the equivalent circle diameter of the mean cell area |
| G2 | Laguerre cells of Neper 4.10.1, `-morpho diameq:lognormal(1,0.5)` | D as for G1 |
| G3 | discs and ellipses (aspect up to 3, half of them) in one matrix grain, a gap of 1 between them | radii log-uniform in [rmin, rmax], 30 % of the area |
| G5 | Poisson Voronoi stretched along x at constant cell area | D, aspect |
| G5L | bands along x | thickness log-uniform in [tmin, tmax] |

The Voronoi and Laguerre networks are exact. G3 is a polygon within 10⁻⁴ of every ellipse. Neper
runs on one thread with its optimisation stopped after 3000 iterations (`-morphooptistop
itermax=3000`): that is deterministic and takes 2 s for 600 cells, where full convergence takes
minutes. The cell diameters then spread by 0.49 of their mean against the 0.5 asked for.

### Orientations and the measurement model

- Grain orientations are uniform in m-3m. With probability `lowAngle` a grain is instead turned
  by 1–15° about a random axis from a neighbour drawn before it, with probability `twin` it is
  that neighbour's Σ3. A `gradient` (lo, hi) turns every grain linearly about a random axis
  along a random direction, by lo to hi degrees across its equivalent diameter.
- The grid is square or hexagonal (rows shifted by half a step) of unit step, and the structure
  is turned about the map centre by a uniform angle and shifted by up to one step, so that no
  boundary lies on the grid by construction.
- Point sampling labels a pixel by the grain at its centre. Spot sampling labels it by the
  majority of 37 sunflower points in a disc of radius `spot`; the majority's share is the
  confidence, and a pixel below `minShare` is notIndexed. That is the boundary-correlated
  notIndexed case of the plan.
- Orientation noise is exp(v) with v ~ N(0, σ²/3 · I), so σ is the RMS misorientation angle.
  A share `wild` of the pixels takes a uniform orientation. `notIndexed` drops pixels at random;
  `clusters` drops discs of radius 1–4 until they cover that share.
- The seed feeds three independent streams (structure, orientations, measurement), so changing
  a measurement parameter keeps the structure and the orientations.

### Files

| file | content |
|---|---|
| `<name>.bin`, `<name>.phases` | the measured map in the neutral format of `bench/calcgrains.py`: rows (x, y, phase, a, b, c, d), NaN rotations where notIndexed |
| `<name>.labels.bin` | the true label at every pixel centre, 0 where the map is notIndexed: the input of the label track |
| `<name>.truth.npz` | per pixel `label`, `qTrue` (noise free), `share`, `wild`; the boundary network `S`, its `junctions`, `grainQ` per grain |
| `<name>.json` | the specification with its seed, and what was drawn from it: turn, shift, counts |

They are written to `bench/out/grains/<tier>/` and regenerated from the seeds.

### Tiers

T0, 64 × 64 pixels, the cross-port check:

| instance | structure | grid | measurement |
|---|---|---|---|
| 01_voronoi | G1, D 10 | square | point, clean |
| 02_hex | G1, D 10 | hex | point, clean |
| 03_small | G1, D 5 | square | point, clean |
| 04_spot | G1, D 10 | square | spot 0.5, minShare 0.7 |
| 05_noise | G1, D 10 | square | σ 1.5°, 1 % wild |
| 06_notIndexed | G1, D 10 | square | 5 % notIndexed at random |
| 07_clusters | G1, D 10 | square | 5 % notIndexed in clusters |
| 08_lowAngle | G1, D 10 | square | 30 % low angle, 10 % Σ3, σ 0.5° |
| 09_laguerre | G2, D 10 | square | point, clean |
| 10_laguerreHex | G2, D 10 | hex | spot, σ 1.5° |
| 11_discs | G3, r 2–12 | square | point, clean |
| 12_discsSpot | G3, r 2–12 | square | spot, σ 0.5° |

T1 holds 69 instances of 128² pixels (D ≤ 10) or 256² pixels. It varies one factor at a time
around two base scenarios, each with two seeds: *clean* (G1, D 20, point, no noise) and *noisy*
(σ 1.5°, 1 % wild, 2 % notIndexed, spot 0.5 with minShare 0.7). The factors are G2, G3 (r 2–30),
G5 (aspect 4), hex, D 5, 10 and 50, the other sampling, σ 0.5° and the other σ, 2 % wild, 5 %
notIndexed, 5 % clusters, 30 % low angle with 10 % Σ3, and gradients of 1–5°. Five combined
scenarios complete the tier: a noisy hexagonal Laguerre map, noisy discs, a hexagonal map with
low-angle boundaries, small noisy Laguerre grains, and twin lamellae 1–10 thick. Generating
T0 takes 5 s, T1 2 min 30 s; spot sampling of G3 dominates, about 25 s per 256² map.

## Metrics

The partition scores count the pixels that are indexed, not wild, and in a reconstructed grain.
The geometry scores are computed in the plane from the two boundaries. The pixels enter only
through their centres: where they lie, and which grain contains them.

**Partition** (`metrics.partition`)

| score | definition |
|---|---|
| `ari`, `vi` | adjusted Rand index; variation of information in bits |
| `f1_50`, `f1_75` | 2 · (matches) / (true + reconstructed grains), a match a pixel IoU above 0.5 or 0.75 |
| `splits`, `merges` | Σ (pieces − 1): a reconstructed grain is a piece of the true grain holding half its pixels, and the other way round |
| `det<lo>_n`, `det<lo>_k`, `sep<lo>` | true boundaries with a misorientation in [lo, next) of 0, 2, 3, 5, 10, 15, 180°; found are those whose sides' dominant reconstructed grains differ, as in `scoreGrainBenchmark`; `sep` is the share of their adjacent pixel pairs put into different grains |
| `sizeW1` | Wasserstein distance of the log pixel counts, true against reconstructed |

**Geometry** (`metrics.geometry`), on the grains matched by an area IoU above 0.5

| score | definition |
|---|---|
| `disp` | Σ area(T_i Δ R_i) / Σ perimeter(T_i) over matched grains: the mean boundary displacement, needing no correspondence of curves |
| `d_rt_*`, `d_tr_*` | distance from the reconstructed boundary to the true one and back: length weighted mean, p95, max |
| `areaBias` | mean over matched grains of (area R − area T) / area T |
| `lengthRatio` | reconstructed / true boundary length between matched grains |
| `dirErr` | length weighted angle between a reconstructed segment and the nearest true segment, the smallest where several are nearest |
| `dirW1` | Wasserstein distance on the circle between the length weighted direction distributions; grid-axis locking shows here |
| `junctionErr`, `junctionFound` | distance from every true junction two steps inside the map to the nearest reconstructed one, mean of those within 2, and their share |
| `pixelWrong` | share of the noise-free pixel centres of matched grains on the wrong side of the reconstructed boundary |
| `orientOK` | every reconstructed grain has a positive area and together they cover the map: the left/right convention of the port holds |

The areas are exact. `planar.overlay` cuts both boundaries at their intersections and at the
ends of collinear overlaps. It locates the regions on either side of every piece by a ray
towards +x, and applies Green's theorem: the boundary of T_i ∩ R_j is the part of ∂T_i inside
R_j and the part of ∂R_j inside T_i, where the two coincide counted once. That gives
area(T_i ∩ R_j) for all pairs at once, and the clipping by the map comes for free: the
reconstruction's outer boundary is the map's edge. The distances are exact point-to-segment
distances, integrated by Simpson's rule on pieces of 1/32 step.

## Metric checks

`checks.py` computes every expected value without `planar.py`. For the line: integrals of
|h_i − y(x)| column by column, closed forms for the area, the length and the direction. For
the disc: quadrature of the disc's chord in every pixel the circle crosses, and the distance to
the circle at 2000 points per pixel edge. The squares are worked by hand. The reconstruction is
the pixel staircase of the true labels on a square grid.

| case | scores checked | agreement |
|---|---|---|
| staircase of y = 0.137 x + c, 0.577 x + c, x + c | disp, d_rt max, lengthRatio, dirErr, areaBias, pixelWrong | 10⁻⁹ |
| the same | d_rt mean | 6 · 10⁻⁵ relative: Simpson's rule on \|f\| where f changes sign inside a piece |
| disc r = 10 and r = 3, centre off the lattice | disp, d_rt mean, areaBias, pixelWrong | 2 · 10⁻⁵ relative: the truth is a polygon within 10⁻⁶ of the circle |
| square on the pixel edges | disp, areaBias, lengthRatio, dirErr, dirW1, pixelWrong, d max both ways | 0 exactly |
| the square shifted by (0.3, 0.2) | disp = (560 − 19.7 · 27.8)/48, d max = √0.13 both ways, length 1, areaBias 0 | 10⁻⁹ |
| the same | dirErr = 90° · 1.1 / 96 = 1.031° | 0.011°, within the 0.049° bound of the sampling |
| identical partitions; two grains merged; one split in halves | ari, vi, f1, splits, merges, sizeW1 | exact |

Where the staircase offsets are equidistributed along the line, the displacement tends to
(1/4 + tan²θ/12) cos θ for a line at θ ≤ 45° to the grid: 64 columns give 0.246 at 7.8° and
0.241 at 30° against 0.249 and 0.241. At slope 1 the staircase repeats column by column and its
displacement, 0.259 here, depends on the offset of the line. The direction error of the shifted square is the length of the reconstructed edge whose
nearest true edge is the perpendicular one near a corner, 2(s_x + s_y) + |s_x − s_y| = 1.1
steps.

## Smoke run on T0

Candidates: `calcGrains` at 5° of the measured map (track *map*) or `calcGrains(grainId)` of the
true labels (track *labels*, the geometry track), each with the raw boundary, `smoothBoundary(1)`
(simplify d/√2, refine d, one Laplace step) and `smoothBoundary(1)` without simplify. Ports:
`python` is this branch, from `main` 3323a31. `matlab` is `../master` on develop at cdf3579c2,
R2024b through the bridge, with `calcGrains` as it is: it straightens the boundaries through
filled cells. `matlabNoStraighten` is MATLAB with `'straighten', false`, the bisectors this port
also draws. Mean over the 12 instances:

| port | case | ARI | F1 .75 | splits | merges | disp | d r→t | d t→r p95 | d max | area % | length | dir ° | dirW1 ° | junction | pixel % | time s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| python | map-raw | 0.998 | 0.976 | 2 | 0 | 0.224 | 0.214 | 0.398 | 0.76 | -0.48 | 1.182 | 30.7 | 18.7 | 0.43 | 1.05 | 0.02 |
| python | map-smooth | 0.998 | 0.976 | 2 | 0 | 0.198 | 0.188 | 0.432 | 0.76 | -1.29 | 1.011 | 14.3 | 5.8 | 0.43 | 1.90 | 0.08 |
| python | map-noSimplify | 0.998 | 0.976 | 2 | 0 | 0.189 | 0.177 | 0.373 | 0.71 | -0.97 | 1.069 | 22.3 | 11.1 | 0.43 | 0.91 | 0.03 |
| python | labels-raw | 1.000 | 0.981 | 2 | 0 | 0.220 | 0.214 | 0.398 | 0.76 | -0.58 | 1.181 | 30.7 | 18.7 | 0.43 | 0.98 | 0.01 |
| python | labels-smooth | 1.000 | 0.981 | 2 | 0 | 0.195 | 0.188 | 0.431 | 0.75 | -1.43 | 1.010 | 14.3 | 5.8 | 0.43 | 1.89 | 0.02 |
| python | labels-noSimplify | 1.000 | 0.981 | 2 | 0 | 0.185 | 0.177 | 0.373 | 0.71 | -1.10 | 1.067 | 22.3 | 11.1 | 0.43 | 0.84 | 0.02 |
| matlabNoStraighten | map-raw | 0.998 | 0.976 | 2 | 0 | 0.225 | 0.214 | 0.398 | 0.77 | -0.50 | 1.183 | 30.7 | 18.7 | 0.43 | 1.05 | 0.03 |
| matlabNoStraighten | map-smooth | 0.998 | 0.976 | 2 | 0 | 0.200 | 0.189 | 0.434 | 0.76 | -1.49 | 1.010 | 14.5 | 5.9 | 0.43 | 1.85 | 0.04 |
| matlabNoStraighten | map-noSimplify | 0.998 | 0.976 | 2 | 0 | 0.189 | 0.177 | 0.373 | 0.73 | -1.00 | 1.070 | 22.3 | 11.1 | 0.43 | 0.88 | 0.03 |
| matlabNoStraighten | labels-raw | 1.000 | 0.981 | 2 | 0 | 0.221 | 0.214 | 0.398 | 0.77 | -0.61 | 1.181 | 30.7 | 18.7 | 0.43 | 0.98 | 0.03 |
| matlabNoStraighten | labels-smooth | 1.000 | 0.981 | 2 | 0 | 0.197 | 0.189 | 0.434 | 0.75 | -1.64 | 1.009 | 14.5 | 5.9 | 0.43 | 1.84 | 0.04 |
| matlabNoStraighten | labels-noSimplify | 1.000 | 0.981 | 2 | 0 | 0.186 | 0.177 | 0.373 | 0.72 | -1.12 | 1.068 | 22.3 | 11.1 | 0.43 | 0.82 | 0.03 |
| matlab | map-raw | 0.998 | 0.976 | 2 | 0 | 0.228 | 0.217 | 0.408 | 0.78 | -0.77 | 1.178 | 30.4 | 18.7 | 0.44 | 0.93 | 0.03 |
| matlab | map-smooth | 0.998 | 0.976 | 2 | 0 | 0.201 | 0.190 | 0.438 | 0.76 | -1.56 | 1.009 | 14.5 | 6.2 | 0.44 | 1.86 | 0.04 |
| matlab | map-noSimplify | 0.998 | 0.976 | 2 | 0 | 0.193 | 0.181 | 0.386 | 0.74 | -1.21 | 1.067 | 22.2 | 11.3 | 0.44 | 0.89 | 0.04 |
| matlab | labels-raw | 1.000 | 0.981 | 2 | 0 | 0.224 | 0.217 | 0.408 | 0.78 | -0.87 | 1.176 | 30.4 | 18.7 | 0.44 | 0.86 | 0.03 |
| matlab | labels-smooth | 1.000 | 0.981 | 2 | 0 | 0.198 | 0.190 | 0.438 | 0.76 | -1.71 | 1.008 | 14.5 | 6.2 | 0.44 | 1.85 | 0.04 |
| matlab | labels-noSimplify | 1.000 | 0.981 | 2 | 0 | 0.189 | 0.180 | 0.386 | 0.73 | -1.33 | 1.066 | 22.2 | 11.3 | 0.44 | 0.82 | 0.03 |

The label track of this port, instance by instance:

| instance | disp raw | smooth | noSimplify | dir ° raw | smooth | noSimplify | pixel % raw | smooth | noSimplify |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01_voronoi | 0.245 | 0.210 | 0.208 | 32.2 | 14.2 | 23.5 | 0.05 | 1.37 | 0.05 |
| 02_hex | 0.221 | 0.192 | 0.194 | 33.8 | 19.9 | 25.4 | 0.00 | 0.00 | 0.00 |
| 03_small | 0.246 | 0.217 | 0.211 | 35.4 | 18.5 | 27.5 | 0.10 | 3.15 | 0.10 |
| 04_spot | 0.158 | 0.149 | 0.132 | 22.4 | 6.9 | 16.1 | 3.42 | 3.84 | 2.67 |
| 05_noise | 0.239 | 0.206 | 0.199 | 32.5 | 14.3 | 23.4 | 0.10 | 1.71 | 0.10 |
| 06_notIndexed | 0.250 | 0.218 | 0.208 | 34.2 | 16.1 | 25.1 | 0.66 | 1.59 | 0.56 |
| 07_clusters | 0.271 | 0.224 | 0.232 | 31.6 | 13.9 | 23.2 | 1.65 | 2.75 | 1.60 |
| 08_lowAngle | 0.240 | 0.210 | 0.197 | 34.0 | 15.2 | 24.6 | 0.07 | 1.39 | 0.07 |
| 09_laguerre | 0.245 | 0.200 | 0.201 | 34.5 | 15.4 | 24.9 | 0.15 | 1.05 | 0.15 |
| 10_laguerreHex | 0.132 | 0.133 | 0.118 | 21.7 | 10.5 | 16.3 | 3.49 | 2.91 | 2.93 |
| 11_discs | 0.243 | 0.206 | 0.199 | 33.0 | 15.0 | 21.9 | 0.00 | 0.49 | 0.00 |
| 12_discsSpot | 0.153 | 0.177 | 0.123 | 23.4 | 11.9 | 15.3 | 2.12 | 2.42 | 1.81 |

What the first run shows:

- The raw staircase misplaces 0.22–0.25 d of boundary on point-sampled maps, as the line check
  predicts, and is 18 % too long. Where spot mixing leaves a notIndexed band along the
  boundaries, the filled cells put the boundary near the bisector: 0.13–0.16 d.
- One smoothing step brings the length to within 1 % with simplify and to 7 % without it, and
  lowers the displacement by 12–15 % either way. With simplify the direction error halves
  (30.7° → 14.3°, 22.3° without) and the locking to the grid axes drops to a third (dirW1
  18.7° → 5.8°, 11.1° without). Simplify also puts 1–3 % of the pixel centres on the wrong
  side of the boundary, against 0.05–0.15 % without it: the half-pixel shift that started the
  programme. And it shrinks the grains more: their area changes by −1.3 % on average, against
  −1.0 % without simplify and −0.5 % raw.
- On the discs with spot sampling, simplify raises the displacement above the raw boundary
  (0.177 against 0.153); without simplify it falls to 0.123.
- Straightened boundaries through filled cells (`matlab` against `matlabNoStraighten`) move
  further from the truth on the spot maps (raw disp 0.176 against 0.158, 0.175 against 0.153 on
  the discs) and come closer on the random notIndexed map (0.244 against 0.250).
- The partition is easy at T0: ARI 0.998 on the measured maps. The 5° threshold finds every
  boundary above 5° and none of the three of 2–5° on 08_lowAngle. The two splits are true
  grains that fall apart into two pixel components on the grid.

These are 12 small maps, one smoothing step each: a smoke test of the harness, not the M2
baseline.

## Cross-check of the ports

`run.py compare T0` matches every grain of one port to the grain of the other most of its pixels
lie in, and measures the Hausdorff distance of the two boundaries:

| case | instances | same partition | pixels differing, max | Hausdorff max | median |
|---|---|---|---|---|---|
| raw, either track | 12 | 8 | 0.51 % | 1.0 | 6 · 10⁻¹⁴ |
| noSimplify, either track | 12 | 8 | 0.51 % | 1.0 | 6 · 10⁻¹⁴ |
| smooth, either track | 12 | 8 | 0.51 % | 1.1 | 0.64 |

(`python` against `matlabNoStraighten`.) The ports agree to rounding on every map without
notIndexed pixels, hexagonal ones included, raw and with the Laplace step alone. Two
differences remain, both inherited from MATLAB develop having moved since the port:

1. On the square maps with filled notIndexed cells (04, 06, 07, 12; the hexagonal 10 agrees),
   MATLAB gives 4–21 pixels another grain, and its boundary lies up to one step away. Changes to
   MATLAB's fill after the port are the likely cause: dd94245c0 makes a hole joined to the
   outside by a short filled path part of the outside, and 89e8bbf85 calls calcPolygonsC. The
   commit responsible has not been isolated.
2. Simplify and refine differ on every map, by 0.6 d at the median, while the Laplace step
   alone agrees to 10⁻¹³. MATLAB's refine subdivides each segment since dc3e96fc2 (#2641) and
   this port resamples at equal arc length; that this alone explains the difference is not
   checked.

Every reconstruction of all three ports passes `orientOK`, so both conventions for the left
grain hold, after smoothing too.

## T1

Generating T1 takes 2 min 40 s, this port's six cases on all 69 instances under a minute, and
scoring the 414 runs 13 min: about 2 s for a 256² run, nearly all of it in `nearest` and the
overlay in NumPy. Mean over the 69 instances:

| port | case | ARI | F1 .75 | splits | merges | disp | d r→t | d t→r p95 | d max | area % | length | dir ° | dirW1 ° | junction | pixel % | time s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| python | map-raw | 0.991 | 0.973 | 6 | 2 | 0.253 | 0.198 | 0.371 | 0.94 | -0.46 | 1.171 | 28.3 | 17.1 | 0.43 | 2.00 | 0.06 |
| python | map-smooth | 0.991 | 0.973 | 6 | 2 | 0.223 | 0.182 | 0.422 | 0.89 | -0.51 | 1.016 | 10.8 | 4.1 | 0.43 | 1.91 | 0.07 |
| python | map-noSimplify | 0.991 | 0.973 | 6 | 2 | 0.209 | 0.159 | 0.341 | 0.88 | -0.47 | 1.068 | 19.8 | 9.4 | 0.43 | 1.73 | 0.07 |
| python | labels-raw | 1.000 | 0.987 | 5 | 0 | 0.204 | 0.198 | 0.371 | 0.92 | -0.29 | 1.173 | 28.3 | 17.1 | 0.43 | 1.43 | 0.05 |
| python | labels-smooth | 1.000 | 0.987 | 5 | 0 | 0.190 | 0.182 | 0.422 | 0.87 | -0.72 | 1.017 | 10.7 | 4.0 | 0.43 | 1.77 | 0.06 |
| python | labels-noSimplify | 1.000 | 0.987 | 5 | 0 | 0.168 | 0.159 | 0.340 | 0.86 | -0.50 | 1.068 | 19.8 | 9.4 | 0.43 | 1.16 | 0.06 |

The order of T0 holds: without simplify the boundary is closer to the truth (disp 0.168 against
0.190) and puts fewer pixel centres on the wrong side than the raw staircase does. With
simplify it is better aligned (10.7° against 19.8°) and of the right length. Thin
lamellae and stretched grains fall apart into pixel components on the grid: on the label track
61 splits among the 93 lamellae, 35 and 54 among the 250 grains of the clean stretched maps.
G5 is there to show this.

## Left for later

- G4 (curvature flow by multiphase thresholding at 8×), G6 (subgrain cells, walls that stop
  inside a grain), G7 (second-phase particles); cracks and pores without material;
  pseudo-symmetric misindexing; point groups other than m-3m.
- A deformed base scenario for T1: gradients are one factor of T1, not a base.
- Curvature error (G3, G4), dihedral angles at junctions, and the detection rate weighted by
  boundary length.
- MATLAB on T1, T2 and the real maps of T3.
- The two cross-port differences above, on the side of this port.
- `tools/graph/build.py` reads `ALGORITHM` from `bench/*.py` only, so the literal in
  `bench/grains/run.py` does not reach the dependency graph yet.
- The scorer in numba once T2 needs it.
