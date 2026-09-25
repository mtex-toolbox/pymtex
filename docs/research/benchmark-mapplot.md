# Benchmark S4 - the inverse pole figure colouring and the plot of large EBSD maps, against MTEX.jl and MATLAB

Run 2026-09-22 on one machine, twelve cores, the three sides one after the other: Python
3.12, NumPy 2, matplotlib 3.11 on the Agg backend, `bench/mapplot.py`; Julia 1.12.7 with
twelve threads and CairoMakie, `bench/mapplot.jl`; MATLAB R2024b with the `frameSymmetry`
branch through its bridge, `bench/matlab/bench_mapplot.sh`. The maps are the Forsterite
map of `../frameSymmetry/data/EBSD` (three phases, one of them not indexed, a square grid),
the Emsland plessite map (ferrite and austenite), and the Forsterite map tiled two by two
and four by four, which the Python script writes as `.ctf` files for the other two. Best of
three after a warm-up run, one run for the largest map; the machine had other sessions
running, so single numbers move by up to a factor of two between runs.

The columns: `grid` is `ipfColors(ebsd)`, every phase through the half degree grid of its
key, what the map plot uses; `exact` the same through the key itself; `key` is the key of
the largest phase on that phase's orientations alone, `key.orientation2color(ori)`, the
one number MATLAB has too; `raster` the picture of the map, one pixel per cell; `plot` the
whole `plot(ebsd, 'orientation')` into a fresh figure; `save` the figure written to a PNG at
150 dpi; `boundaries` the map drawn again with the grain boundaries on top and saved.

## Python

| map | points | grid | exact | key | raster | plot | save | boundaries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Forsterite | 245 952 | 0.031 s | 0.29 s | 0.21 s | 0.053 s | 0.18 s | 0.21 s | 0.50 s |
| Emsland | 270 000 | 0.061 s | 1.49 s | 1.38 s | 0.20 s | 0.26 s | 0.29 s | 0.96 s |
| Forsterite 2x2 | 983 808 | 0.24 s | 1.44 s | 1.13 s | 0.57 s | 1.33 s | 0.29 s | |
| Forsterite 4x4 | 3 935 232 | 0.71 s | 5.34 s | 4.16 s | 0.96 s | 2.59 s | 0.43 s | |

## Julia

| map | points | grid | exact | key | plot | save |
|---|---:|---:|---:|---:|---:|---:|
| Forsterite | 245 952 | 0.011 s | 0.18 s | 0.14 s | 0.045 s | 0.040 s |
| Emsland | 270 000 | 0.013 s | 0.29 s | 0.25 s | 0.052 s | 0.038 s |
| Forsterite 2x2 | 983 808 | 0.041 s | 0.64 s | 0.52 s | 0.12 s | 0.046 s |
| Forsterite 4x4 | 3 935 232 | 0.13 s | 2.31 s | 1.83 s | 0.40 s | 0.079 s |

## MATLAB

`ipfHSVKey(cs).orientation2color(ori)` on the largest phase, `plot(ebsd(phase), colors)`
with `drawnow` into an invisible figure, `print -dpng -r150`; the four by four map was not
run, the five minute rule.

| map | points | key | plot | save |
|---|---:|---:|---:|---:|
| Forsterite | 245 952 | 0.13 s | 0.78 s | 1.01 s |
| Emsland | 270 000 | 0.33 s | 1.40 s | 3.75 s |
| Forsterite 2x2 | 983 808 | 0.31 s | 1.06 s | 0.65 s |

## Reading

- The map plot is ahead of MATLAB on every map, four times on the Forsterite map, and
  behind Julia by four to ten times. Julia's advantage is the compiled loop of the colour
  grid look-up (45 ns per point against 180 ns in NumPy after the fixes below) and
  CairoMakie's raster path, which writes the PNG in 40 ms where matplotlib's image
  resampling and PIL's PNG encoding take 200 ms.
- The exact key is level with Julia within a factor of two and ahead of MATLAB; the
  Emsland map is the slow case on every side, its cubic ferrite folding every direction
  over 24 elements.
- The plot of the four million point map takes 2.6 s in Python: 0.7 s the colours, 1.0 s
  the raster, the rest the figure; a map of that size is a picture of two thousand pixels
  a side, so the cost is in the arrays and not in matplotlib.

## Found on the way

- `mapRaster` judged the Emsland map not aligned with the screen, its lattice matrix
  carrying off-diagonal entries of 4e-8 against a spacing of one, and drew it at a quarter
  of the spacing per pixel, sixteen times the pixels: 1.05 s for the raster and 1.74 s for
  the plot. The alignment test is now relative at 1e-6 of the spacing (the plotting
  session's fix); the plot takes 0.26 s.
- `S2Grid.nearestStepwise`, the look-up of the colour grid, looped over the three hundred
  rings of the grid and computed the polar angles twice through `normalize`, `theta` and
  `rho`. It is one array expression now, the ring from the polar angle and the node from
  the azimuth against the ring's first node, step and count, with the angles taken from
  the coordinates once: the grid colouring of the four million point map went from 1.53 s
  to 0.71 s.
- What is left in the grid route is in the plotting layer: the rotated directions are
  assembled through `np.tile`, checked for finiteness and normalised again although they
  are unit by construction, and the colours gathered into a fresh array per phase; about
  half of the 0.7 s. A numba kernel over the quaternions and the grid would bring the
  route to Julia's time, as `docs/research/time-critical.md` lists.
- matplotlib's `savefig` spends 0.14 s of the 0.21 s in PIL's PNG encoder at its default
  compression; `pil_kwargs={"compress_level": 1}` would write the file in a third of
  the time for a file a third larger.

## Running

```
.venv/bin/python bench/mapplot.py
(cd ../mtexJulia && julia -t auto --project=dev ../pymtex/bench/mapplot.jl)
bench/matlab/bench_mapplot.sh        # the mtexfs bridge must be running
```

The Python script writes `bench/out/forsterite_{2x2,4x4}.ctf` for the other two, and the
results land in `bench/out/mapplot_{python,julia}.csv` and `bench/out/mapplot_matlab_*.json`;
the rendered maps are next to them.

## The parts, one by one

`bench/mapplot_parts.py` times the three stages of `plot(ebsd, 'orientation')` on their own,
on the Forsterite map and its tiles, and on the screen through the Qt and Tk backends of
matplotlib (PySide6 and tkinter, DISPLAY :1). Milliseconds, best of three.

The colours of the largest phase (Forsterite, 60 % of the points), then the whole map:

| points | rotate the direction | grid look-up | gather the colours | `ipfColors` of the map | numba kernel |
|---:|---:|---:|---:|---:|---:|
| 245 952 | 7 | 18 | 4 | 64 | 6 |
| 983 808 | 26 | 66 | 13 | 240 | 40 |
| 3 935 232 | 128 | 262 | 55 | 1050 | 123 |

The three array steps are less than half of `ipfColors`; the rest is the assembly round
them, `np.tile`, finiteness checks, a second normalisation, the wrapping into `vector3d`
and the per-phase gather. The kernel is the whole route in one compiled loop over the
quaternions (the third row of the rotation matrix, the polar angles, the ring, the node,
the colour), eight times faster than `ipfColors` and at Julia's time; it agrees with the
grid route on 99.8 % of the points, the rest being ties at the middle between two nodes.

The raster:

| points | picture | lattice table | `mapRaster` | direct index | RGBA image float64 | RGBA image uint8 |
|---:|---:|---:|---:|---:|---:|---:|
| 245 952 | 732 x 336 | 16 | 116 | 15 | 17 | 12 |
| 983 808 | 1464 x 672 | 35 | 378 | 53 | 59 | 33 |
| 3 935 232 | 2928 x 1344 | 171 | 1192 | 118 | 143 | 91 |

`mapRaster` searches the nearest of the four lattice corners round every pixel, which a
hexagonal or a tilted lattice needs; a lattice aligned with the screen, which every map
in this benchmark is, has its picture in the lattice table itself, transposed and flipped
to the axes' directions. The direct index agrees with the search on every map and is ten
times faster. The image as bytes rather than doubles saves a third and a hundred
megabytes at four million points.

The drawing, the figure made and the image drawn onto the canvas; `show` is the window
opened and drawn, `zoom` a pair of redraws at another x range:

| points | backend | figure | `imshow` | `draw` | draw, uint8 | `savefig` | show | zoom pair |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 245 952 | Agg | 4 | 3 | 31 | 32 | 103 | | |
| 983 808 | Agg | 10 | 29 | 118 | 80 | 183 | | |
| 3 935 232 | Agg | 4 | 49 | 254 | 263 | 608 | | |
| 245 952 | QtAgg | 15 | 8 | 80 | 85 | | 139 | 218 |
| 983 808 | QtAgg | 9 | 13 | 68 | 73 | | 83 | 170 |
| 3 935 232 | QtAgg | 7 | 56 | 211 | 259 | | 266 | 523 |
| 245 952 | TkAgg | 134 | 3 | 38 | 42 | | 92 | 110 |
| 983 808 | TkAgg | 162 | 16 | 73 | 93 | | 198 | 181 |
| 3 935 232 | TkAgg | 132 | 47 | 155 | 255 | | 544 | 667 |

A draw is matplotlib resampling the whole picture to the pixels of the axes, seven
hundred by three hundred at the default figure size, so it costs by the source: 60 ns a
pixel, 250 ms for the four million point map, and every pan or zoom pays it again. The
window itself costs nothing on Qt and 130 ms on Tk. Saving is a draw at 150 dpi plus
the PNG encoder; the low compression level gains nothing worth having on these images.

## How to make it fast

Of the 3.2 s of the four million point plot, 1.05 s are the colours, 1.2 s the raster,
0.14 s the image, 0.25 s the draw and the rest the figure. Three changes, each measured
above as a prototype, bring the same plot to about half a second:

1. The grid route of `ipfColors` as one numba kernel over the quaternions, with the NumPy
   route as fallback and oracle: 1.05 s to about 0.2 s for the whole map.
2. The picture of a lattice aligned with the screen read off the lattice table, the
   corner search kept for the hexagonal and the tilted lattices: 1.2 s to 0.12 s.
3. The image assembled as bytes, and the picture capped at twice the pixels of the axes on
   each side unless `maxPixels` asks otherwise, so that a draw, a pan and a zoom resample a
   million pixels at most: 0.4 s to about 0.1 s for the image and the draw together, on
   the screen as in a file.

After these the plot is bounded by the raster's gather and matplotlib's draw, both a
hundred milliseconds at four million points; the on-screen work of MATLAB, a second for a
million points, is far behind, Julia's 0.4 s within reach.

## After the three changes, 2026-09-22

All three are in: `plotting/kernels.py` holds the compiled grid route, `ipfColors` takes it
where numba is there and keeps the NumPy route as fallback and oracle; `mapRaster` rounds
the lattice coordinates of the pixel centres where the lattice is aligned with the screen
and searches the corners otherwise; `plotEBSD` builds the image as bytes and, unless
`maxPixels` is given, caps the picture at twice the axes' pixels on each side. The same
runs as above, on a quieter machine:

| map | points | grid | exact | key | raster, uncapped | plot | save | boundaries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Forsterite | 245 952 | 0.010 s | 0.32 s | 0.24 s | 0.019 s | 0.071 s | 0.14 s | 0.29 s |
| Emsland | 270 000 | 0.015 s | 1.12 s | 0.95 s | 0.054 s | 0.17 s | 0.24 s | 0.39 s |
| Forsterite 2x2 | 983 808 | 0.038 s | 1.73 s | 1.17 s | 0.096 s | 0.38 s | 0.23 s | |
| Forsterite 4x4 | 3 935 232 | 0.21 s | 6.33 s | 5.35 s | 0.56 s | 0.86 s | 0.20 s | |

The grid colouring is at Julia's time now (0.21 s against 0.13 s on four million points,
the kernel itself 45 ms on the largest phase), the plot of the four million point map
went from 2.6 s to 0.86 s and the Forsterite map draws in 70 ms against Julia's 45 ms and
MATLAB's 780 ms. What is left at four million points is the capped raster and its image,
about 0.4 s of NumPy over the pixel grid, and the draw of the capped picture, 70 ms; a pan
or a zoom on the screen redraws that picture and no longer the whole map.

The exact key did not change and stays the slow route for whoever asks for it.

## Larger maps, 2026-09-22

`bench/mapplot_large.py` tiles the Forsterite map eight by eight and twelve by twelve in
memory, sixteen and thirty five million points, and times the colours, the whole plot
drawn on the Agg canvas, the file, and the window with a pair of zooms on the screen
through Qt. The machine has thirty gigabytes; the map itself takes about 270 bytes a
point, so the thirty five million point map is nine gigabytes and a sixty three million
point one would not have fitted next to the other sessions.

Right after the three changes the plot still grew faster than the map: 5.6 s at sixteen
million and 21 s at thirty five million points, the picture capped all the while. The
profile found four passes over every measurement that had nothing to do with the picture:
the positions projected into the plane of the map three times, twice for the extent of
the axes and once for the raster, the largest deviation from the lattice recomputed at
every plot, the colours of the whole map converted to bytes before the picture picked its
pixels, and the colours of every phase assembled into a temporary and copied. Each is
gone: the extent is the box round the positions, exact for a map in a coordinate plane,
with the column bounds taken a block of rows at a time since NumPy reduces a short axis
ten times slower than a long one; the lattice remembers its residual; the bytes are made
of the pixels alone; the kernel writes the colours of a phase straight into the map's
array.

| points | tiles | build | colours | plot and draw | save | window and draw | zoom pair | memory |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 15 740 928 | 8 x 8 | 7.6 s | 0.23 s | 0.66 s | 0.17 s | 0.70 s | 0.04 s | 4.2 GB |
| 35 417 088 | 12 x 12 | 22 s | 0.52 s | 1.40 s | 0.16 s | 1.9 s | 0.04 s | 9.1 GB |

The plot now grows with the map at about 40 ns a point, the colours at 15 ns, and the
picture is the same million pixels whatever the map, so a zoom costs 20 ms at thirty five
million points. What is left per point is the lattice table, the kernel and the box; the
build column is the tiling itself, the lattice inferred over every copy, and says how far
the maps layer is from the plot.

The standard maps after the same changes, `bench/mapplot.py` once more:

| map | points | grid | raster, uncapped | plot | save | boundaries |
|---|---:|---:|---:|---:|---:|---:|
| Forsterite | 245 952 | 0.009 s | 0.006 s | 0.026 s | 0.12 s | 0.19 s |
| Emsland | 270 000 | 0.013 s | 0.008 s | 0.032 s | 0.15 s | 0.27 s |
| Forsterite 2x2 | 983 808 | 0.015 s | 0.027 s | 0.079 s | 0.15 s | |
| Forsterite 4x4 | 3 935 232 | 0.064 s | 0.10 s | 0.22 s | 0.19 s | |

The Forsterite map plots in 26 ms against Julia's 45 ms and MATLAB's 780 ms, the four
million point map in 0.22 s against Julia's 0.40 s. The file is the PNG encoder now.

## Colouring what the picture shows, 2026-09-22

The wizard's map view asked for the last of the per-measurement work to go. A picture holds
at most a million or two pixels, each showing one measurement, so a map larger than that is
coloured where nobody looks. `ipfColors(x, ..., at=)` and `mapValues(x, color, ..., at=)`
now take the indices of the measurements to colour and return one colour per index, the
per-phase keys and the unindexed handling intact; `plotEBSD` passes them when the picture
holds fewer pixels than the map holds points. A route that yields numbers rather than
colours ignores `at` and stays whole, so that the colour range is the map's and not the
picture's.

Inside one plot of the sixteen million point map, profiled in the same run: colouring the
picture's measurements takes 0.09 s where colouring the map takes 0.44 s. The wizard
measured the same on its own recolour, 0.63 s to 0.11 s, and its peak allocation for one
inverse pole figure tab fell from 501 MB to 160 MB.

The whole plot of that map hardly moves, 0.71 s, because what is left is no longer the
colours: `latticeTable` scatters an index per measurement into the table the picture reads,
0.25 s of own time at sixteen million points, and the raster round it another 0.3 s. That
table depends only on the lattice and the order of the map, so a map plotted again could
keep it, at the price of an int32 per measurement held for as long as the lattice lives.
That trade is not taken here.

Two more things came out of the same exchange.

- The byte image painted an unindexed measurement opaque black. `mapValues` gives such a
  measurement a colour that is not a colour, NaN per channel, because the notIndexed frame
  carries `array([nan, nan, nan])` as its colour rather than none at all, and the old float
  image let that NaN reach matplotlib, which drew nothing. Clipping to bytes turned it into
  black. The alpha now follows the finiteness of the gathered colours: on the Forsterite
  map, 58 485 pixels transparent, the unindexed count exactly, and no opaque black pixel.
  `tests/test_mapplot.py` has both halves of this.
- `keyLegend` drew each inverse pole figure key at `plotKey`'s default half degree, tens of
  thousands of quads for an inset about a hundred pixels wide, and every canvas draw paid
  for it. It takes a `resolution` now, and chooses one from the inset's width when none is
  given, `keyResolution(px)`, about two screen pixels to a quad and never outside half a
  degree to four. The Forsterite inset is 147 px wide and meshes at 1.2 degree: three keys
  and a draw, 0.94 s to 0.31 s, where asking for two degrees outright gives 0.29 s. A key
  drawn large for a page still gets the half degree.

## The key at ten million orientations (2026-09-25)

`ipfHSVKey(cs).orientation2color(ori)` on 1e7 random orientations, twelve cores, with other
sessions on the machine (load 5 to 6, so the absolute times run up to twice those of a quiet
machine; each comparison below was taken under the same load). The exact key folds every
crystal direction into the sector and then works out its polar coordinates and its colour
in NumPy. MATLAB's `precompute`, a grid of the key looked up without the fold, is
`key.precompute()` here, and `orientation2color` takes it by itself from 1e5 orientations
on (`GRID_FROM`, MATLAB's estimate of where it pays); `ipfColors`, the map plot, always did.

| m-3m, 1e7 | nodes | built once per frame | look-up | mean error | max error |
|---|---:|---:|---:|---:|---:|
| exact key | | | 16.5 s | | |
| rings, half a degree (before) | 165 016 | 0.11 s | 236 ms | 0.011 | 0.048 |
| cube map, 230 cells an edge | 317 400 | 0.29 s | 83 ms | 0.009 | 0.043 |
| cube map, 512 cells an edge | 1 572 864 | 1.1 s | 82 ms | 0.004 | 0.026 |

- The grid is now a cube map (`cubeMapNodes`, `cubeMapIndex` and the loop in
  `plotting/kernels.py`): a direction falls on the face of its largest component, and its
  cell is the other two components over that one, two divisions. The rings needed an arccos
  for the ring and an arctan2 for the place in it, and those were the cost; the cube map is
  three times faster and, at the same widest cell, nearer the key. The edge count is
  ceil(2 / resolution), the cell at a face centre being the widest.
- Lower precision does not pay: float32 quaternions and float32 or byte colours took the
  ring look-up from 227 ms to 198 ms at best, the loop being bound by its arithmetic, not
  by memory; float32 moves 74 of 1e7 directions to another node, and converting a float64
  map costs 40 ms. numba has no float16 arithmetic.
- The fold is no longer what the exact key costs: `projectIntoSector` runs over the cores
  in 0.21 s of the 6.7 s a profiled run takes; the rest is the colour arithmetic of
  `sphereColor` and `hsv2rgb`, a dozen arrays of 1e7 x 3, memory bound.
- The unmeasured orientations are coloured inside the look-up loop (`blank`, NaN for
  `orientation2color`, white for the map), which halved the call: the separate `isfinite`
  pass over 1e7 quaternions took as long as the look-up itself.
- `bench/mapplot_parts.py`: `ipfColors` of the 3.9 million point map in 55 ms, the compiled
  loop and the array form agreeing on every point. `bench/mapplot_large.py` (rings still):
  the 15.7 million point map in 0.25 s for the colours and 0.35 s for the plot.

## The three ports again, with the cube map (2026-09-25)

Run 2026-09-25 evening, the machine quiet (load below 3), the three sides one after the
other: `bench/mapplot.py`, `bench/mapplot.jl` (Julia 1.12.7, twelve threads),
`bench/matlab/bench_mapplot.sh` (R2024b, `frameSymmetry`, now the four by four map too, 19 s
for both tiled maps). The key columns are the key on the orientations of the largest phase;
`key grid` is the key as Python calls it now (its cube map from 1e5 orientations on) and
MATLAB's after `precompute`; `map colours` is `ipfColors` of the whole map, all phases, the
route of the map plot; `plot` is the whole map plotted for Python and Julia and MATLAB's
`plot(ebsd(phase), ebsd(phase).orientations)`, colouring included, `drawnow`; `save` a PNG
at 150 dpi (Julia one pixel per unit). Seconds.

| map | points | key exact Py / Jl / ML | key grid Py / ML | map colours Py / Jl | plot Py / Jl / ML | save Py / Jl / ML |
|---|---:|---|---|---|---|---|
| Forsterite | 245 952 | 0.042 / 0.085 / 0.123 | 0.004 / 0.019 | 0.003 / 0.009 | 0.029 / 0.036 / 0.61 | 0.13 / 0.022 / 0.82 |
| Emsland | 270 000 | 0.080 / 0.197 / 0.207 | 0.003 / 0.023 | 0.004 / 0.007 | 0.032 / 0.039 / 0.80 | 0.15 / 0.032 / 2.08 |
| Forsterite 2x2 | 983 808 | 0.193 / 0.383 / 0.268 | 0.005 / 0.056 | 0.010 / 0.033 | 0.067 / 0.073 / 0.47 | 0.17 / 0.030 / 0.58 |
| Forsterite 4x4 | 3 935 232 | 0.762 / 1.570 / 0.873 | 0.010 / 0.217 | 0.044 / 0.104 | 0.120 / 0.366 / 0.78 | 0.18 / 0.059 / 0.54 |

- The colours: Python leads on every route. The map through the cube map is 2.4 to 3.3
  times Julia's ring grid, the key through its grid 5 to 22 times MATLAB's precomputed one,
  and the exact key, the parallel fold, 1.1 to 2.6 times ahead of MATLAB and 2 times of
  Julia. MATLAB's grid deviates from its exact key by 0.009 to 0.04, the cube map by 0.009
  and 0.043 (m-3m, 1e7 orientations, above).
- The plot: Python and Julia level on the small maps, Python three times faster on the four
  by four; MATLAB 5 to 25 times slower, for one phase where the other two draw the map.
- The file is where Python stays behind Julia: matplotlib's resampling and PIL's PNG
  encoding take 130 to 180 ms where CairoMakie writes in 22 to 59 ms. Plot and file together:
  Python 0.16, 0.19, 0.24 and 0.30 s, Julia 0.06, 0.07, 0.10 and 0.42 s, MATLAB 1.4, 2.9, 1.1
  and 1.3 s.
- The file, then: of the 130 to 180 ms, 96 to 134 were Pillow's zlib at its default level
  6. `mtexFigure.savefig` writes a PNG at level 1 unless the caller gives a level: the same
  pixels, the four by four map's file 677 kB for 611 kB (level 0: 2 MB, 67 ms; level 9:
  905 ms), and the save 65, 85, 82 and 91 ms for 128, 181, 172 and 180. Plot and file
  together: Python 0.09, 0.11, 0.15 and 0.22 s, Julia 0.06, 0.07, 0.10 and 0.42 s. What is
  left of the save on the Forsterite map is a third each: the compression (31 ms), the
  image resampled into the axes (27 ms) and the text laid out again (31 ms), since
  matplotlib gives a save at another resolution than the figure's a new renderer and so an
  empty text cache.
