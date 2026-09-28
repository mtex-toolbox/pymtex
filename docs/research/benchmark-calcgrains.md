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

The export, the per-port counts and the partitions go to `bench/out/calcgrains/`.
