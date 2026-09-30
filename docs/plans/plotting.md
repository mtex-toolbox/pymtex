# Subplan P5 - plotting

Status: written 2026-09-22; S1 to S6 built the same day, the crystal shapes of S5 the day after. Syntax reference: `../frameSymmetry/plotting`
with its `CLAUDE.md`, the `plot`, `scatter`, `contourf`, `contour`, `pcolor`, `quiver`, `text`
methods of `@vector3d`, `@S2Fun`, `@S2VectorField`, `@sphericalRegion`, `plotPDF` and
`plotIPDF` of `@SO3Fun`, `sphericalPlot.m`, `makeSphericalProjection.m` and the
`sphericalProjections/` classes. Design reference: `../mtexJulia/lib/MTEXPlotting/src/` and the
Makie extension `../mtexJulia/ext/MTEXMakieExt/`, with `docs/plans/plotting.md` there, S1 to
S6 done. Tests: `../mtexJulia/test/spherical.jl` P1 to P4, `test/odfsections.jl`,
`test/plotting.jl`; MATLAB has no headless test of a spherical plot, the rendered checks of
`../mtexJulia/dev/plots.jl` are looked at instead.

## 1. Deliverable

A package `mtex.plotting` whose public layer reads like the MATLAB of the branch and whose
inside is the Julia design: the geometry of a plot computed without graphics, matplotlib
drawing what the geometry hands it. Spherical plots first, since they check P1 and P2:

- directions and Miller indices as markers, on one hemisphere or both, symmetrised, in the
  fundamental sector, labelled;
- functions on the sphere as `pcolor`, `contourf` and `contour`, with the MTEX colour maps,
  a colour bar and the min/max annotation;
- the pole figures and inverse pole figures of an ODF in a row of axes, one colour range;
- vector and axis fields as arrows;
- spherical regions and fundamental sectors as outlines;
- `plottingConvention` applied to the axes: what points east, what out of the screen;
- later: ODF sections (S2), colour keys (S3), EBSD and grain maps (S4), three dimensional
  plots (S5), the pole figures of orientation lists and measured pole figures (S6).

`import mtex` does not import matplotlib (D8 of `docs/PLAN.md`): `mtex.plotting.spherical` is
NumPy only and `mtex.plotting.sphericalplot` imports matplotlib inside the functions that draw.

## 2. The two references, and the rule between them

MATLAB names the user sees, Julia the representation. The Julia plan separated the geometry
of a spherical plot from its drawing; that split is kept, and the MATLAB names are put on
the Julia objects:

| MATLAB | Python | from Julia |
| --- | --- | --- |
| `@sphericalProjection` with `sR`, `pC`, `antipodal`; `eareaProjection`, `eangleProjection`, `edistProjection`, `orthographicProjection`, `gnomonicProjection`, `plainProjection` | the same classes, `project(v)` and `iproject(x, y)`, `upper` as a fourth field | Julia's `SphericalChart`: a projection is one radial function `radius(theta)` with its inverse `polarAngle(r)`, the chart folds the lower hemisphere by `theta -> pi - theta` |
| `makeSphericalProjection(v, 'upper')`, `'projection','stereo'`, `'complete'`, `'antipodal'` | `makeSphericalProjection(v, projection='stereo', upper=True, complete=True, antipodal=True)` returning a list | Julia's `charts`: one chart, or an upper and a lower one when the region reaches into both hemispheres |
| `@sphericalPlot`, one per axes: `proj`, `boundary`, `grid`, `labels`, `bounds`, `minData`, `maxData` | `sphericalPlot(ax, proj)`, the same fields, kept on the axes as `ax._mtexSphericalPlot` | the chart grid follows the boundary; `chartGrid`, `chartBoundary`, `chartGridLines`, `chartLimits`, `chartLabels` are methods of the projection |
| `mtexFigure`, `newMtexFigure`, `gcm`, `nextAxis`, `drawNow`, `mtexColorbar` | `mtexFigure` over `matplotlib.figure.Figure` with `nextAxis()`, `children`, `colorbar()`; `newMtexFigure`, `gcm`, `mtexColorbar` | Makie's `GridLayout` did this; here a thin class since matplotlib has no layout of that kind |
| `plot(v)`, `scatter(v)`, `plot(v,'contourf')`, `contourf(v)`, `plot(v,data)`, `plot(v,rgb)`, `text(v, labels)`, `quiver(v, w)` | the same functions in `mtex.plotting` and methods on `vector3d`; flags are keywords: `contourf=True`, `upper=True`, `antipodal=True`, `fundamentalRegion=True`, `symmetrised=True`, `complete=True`, `minmax=True`, `grid=True`, `axisLabels=False` | `dynamicMarkerSize`, one rule for the marker size |
| `plot(sF)`, `contourf(sF)`, `contour(sF)`, `pcolor(sF)` | the same, functions and methods of `S2FunHarmonic` | the function is evaluated on the chart grid; the contour lines and bands on the curvilinear grid |
| `plotPDF(odf, h)`, `plotIPDF(odf, r)`, `'resolution'`, `'antipodal'`, `'complete'`, `'noTitle'` | `plotPDF(odf, h, resolution=..., antipodal=True, complete=True, title=False)`, methods of the SO(3) functions | one colour range over the row, the specimen sector of a symmetric pole figure |
| `plot(sR)`, `plot(cs.fundamentalSector())` | the same, a method of `sphericalRegion` | |
| `'MarkerSize'`, `'MarkerFaceColor'`, `'MarkerEdgeColor'`, `'Marker'`, `'MarkerColor'` | the same keywords, translated to matplotlib; lowercase matplotlib keywords pass through | |
| `mtexColorMap WhiteJet`, `LaboTeX`, `blue2red`, `white2black` | `mtexColorMap('WhiteJet')` on the current figure, `colormap='WhiteJet'` on a plot | colour maps are stops; matplotlib builds the gradient |
| `'how2plot', pC` | `how2plot=pC` | |
| `hold on; plot(v2)` | `hold(True)` or `plot(v2, parent=ax)`; `add2all=True` | |

Where MTEX.jl deliberately changed a MATLAB picture, the Python port follows Julia and the
difference is recorded once, in section 8.

## 3. Design

**A projection is a radial function.** In the frame of the plotting convention a direction has
the polar angle theta and the azimuth rho, and the point drawn is `r(theta) (cos rho, sin rho)`:
equal area `2 sin(theta/2)`, stereographic `tan(theta/2)`, equal distance `theta`, orthographic
`sin theta`, gnomonic `tan theta`; the plain projection draws rho over theta in degrees and
ignores the convention. Every subclass is two one-line methods, `radius` and `polarAngle`, and
`sphericalProjection.project` does the rest.

**A projection is a chart: a projection, a region, a convention and a hemisphere.** MATLAB's
`sphericalProjection` already carries `sR`, `pC` and `antipodal`; `upper` is added so that the
lower chart folds `theta -> pi - theta` and is seen through the sphere from above as in MTEX.
An antipodal chart draws the antipode of a direction that falls into the other hemisphere.
`project` gives NaN outside the region, which is how lines are clipped. `makeSphericalProjection`
decides the charts: one when the region lies in one hemisphere or is antipodal, two otherwise,
`upper`, `lower`, `complete` forcing the choice; the plain projection is always one chart.

**The plotting grid follows the boundary.** `proj.grid(resolution)` is regular in the azimuth
of the chart's frame and, per azimuth, runs from the lower to the upper polar angle of the
region through `thetaIntervals` of P1, one strip of columns per branch of the polar set, so a
pcolor of a sector has clean edges. `rhoRange` is the complement of the widest gap in the
azimuths of the boundary, the whole circle when a pole lies inside. matplotlib draws the
curvilinear grid as it is: `pcolormesh(X, Y, C)` for the surface and `contour`, `contourf` for
the lines and bands, so one grid serves every projection and no contour routine is written.

**Labels come from the frame.** A crystal frame labels the corners of its sector with the
rounded Miller indices, direct basis first, reciprocal if that does not round; any other frame
labels its axes with their names where they fall into the chart, on the rim in a white box.
MTEX's `annotateFrame` is this.

**The region a plot covers** is the whole sphere, or the conventional sector of a frame with a
group when `fundamentalRegion=True` is asked for or a function is symmetric; the Laue sector
for axes. `complete=True` draws a symmetric function on the whole sphere.

**Graphics state as MATLAB has it.** `gcm()` is the current `mtexFigure`; `plot` starts a new
figure unless the current axes holds (`hold(True)`) or `parent=ax` says where; `add2all=True`
draws into every axes of the figure, which is what `annotate` and the poles added to a pole
figure need. The `sphericalPlot` of an axes is kept on the axes, so a second plot into it
reuses the projection, the boundary and the labels instead of drawing them again.

**Colour maps are stops.** `WhiteJet`, `LaboTeX`, `blue2red`, `white2black` as stops with
positions in `colormaps.py`; `mtexColorMap` builds a `LinearSegmentedColormap` and registers it
once. `WhiteJet` is the default of every filled plot.

## 4. Order of work

**S1 - spherical geometry and the matplotlib drawing. BUILT 2026-09-22.**
`mtex/plotting/spherical.py`: the projection classes, `makeSphericalProjection`, `rhoRange`,
the grid, boundary, grid lines, limits and labels; `colormaps.py` with `contourLevels`;
`mtexfigure.py` with `mtexFigure`, `newMtexFigure`, `gcm`, `hold`, `mtexColorbar`,
`mtexColorMap`; `sphericalplot.py` with `sphericalPlot`, `plot`, `scatter`, `contourf`,
`contour`, `pcolor`, `smooth`, `quiver`, `text`, `annotate`, `plotPDF`, `plotIPDF` and the
methods they install on `vector3d`, `S2FunHarmonic`, the SO(3) functions, the fields and
`sphericalRegion`. Tests `tests/test_spherical.py` (the geometry, the numbers of Julia's P1 to
P4) and `tests/test_sphericalplot.py` (the drawing on the Agg backend: artists, limits,
colour ranges, one axes per chart). `bench/plots.py` writes the rendered checks to `bench/out`
for a look. The `sphericalProjection` name of `mtex.functions.s2fun`, the projection of
harmonic coefficients onto the invariant subspace, is no longer exported, so that the MATLAB
class keeps its name.

**S2 - ODF sections. BUILT 2026-09-22.** `mtex/plotting/sections.py`: `ODFSections` with
`phi1Sections`, `phi2Sections`, `alphaSections`, `gammaSections` (one `eulerSections` base,
two parameters: the convention and which end is fixed), `pfSections` and `sigmaSections` (the
direction `ori * h1`, the coordinate the twist of `ori * h2` against the reference field, which
is the smallest rotation from the pole of the convention applied to its east),
`axisAngleSections` (the axis sector of the fundamental region at each angle, on the
conventional sector of the shared rotations, scaled by `sin(w/2)`), and `newODFSectionPlot`
choosing phi2 sections against a specimen and axis angle sections between two crystals. MATLAB's
property names are kept: `oS.phi2`, `oS.sigma`, `oS.h1`, `oS.h2`, `oS.angles`, `oS.axesSectors`,
`oS.jointCS`, `oS.oR`, `oS.volumeScaling`, `oS.CS1`, `oS.CS2`, `oS.tol`, `oS.makeGrid()`; the
section values are the keywords of the family, `phi2Sections(cs, ss, phi2=...)`. Every section
is a chart of S1, so `sectionplot.py` draws sections with the code that draws pole figures:
`plotSection(odf)`, `plotSection(odf, 'sigma', sections=4, contourf=True)`, `plotSection(ori,
oS, values=..., points=2000, all=True)`, `oS.plot(x)`, with `columns`, `field=False` to leave
the reference field out, and one colour range over the sections. The degenerate middle Euler
angle puts an orientation into every section, as Julia decided. Tests `tests/test_sections.py`
(Julia's E1 to E7) and `tests/test_sectionplot.py`; rendered `bench/plots.py` 40 to 43. Left
out with Julia: `PhiSections`, `ipfSections`, `quiverSection`, `contour3`, `surf3`, `slice3`.

**S3 - colour keys. BUILT 2026-09-22.** `mtex/plotting/sectorfold.py`: the centre of a
region, the fold of a sector derived from the group as Julia designed it (the halving plane
through two characteristic points whose closure with the group is a reflection group; `-1`,
`-3` and `-4` fold once and `hasSmoothKey` says their colours jump), and MATLAB's
`polarCoordinates` with the equalised azimuth. All 32 folds agree with MTEX's table and the
polar coordinates with MTEX's values within two degrees at the corners. `directionkeys.py`:
`directionColorKey` with `direction2color(h)`, `HSVDirectionKey(cs, whiteCenter=, maxAngle=)`
(the colour sphere, the HSL saturation, the hue correction; within 10 of 255 of MTEX),
`TSLDirectionKey`, `HKLDirectionKey`, `spotDirectionKey`, `hsv2rgb`, `rgb2hsv`, `hsl2hsv`.
`orientationkeys.py`: `orientationColorKey` with `orientation2color(ori)`, `ipfColorKey(cs,
ipfDirection=, dirMap=)`, `ipfHSVKey` (within 5 of 255 of MTEX's), `ipfTSLKey`, `ipfHKLKey`,
`ipfSpotKey(cs, center=, color=, psi=)`, `axisAngleColorKey(CS1, CS2, oriRef=, maxAngle=,
thresholdAngle=)`, `BungeColorKey`, `spotColorKey(center, color, psi=)`, and the key on a grid:
`colorGrid`, held per frame, behind `ipfColors(ebsd | grains, ipfDirection=, exact=False)`,
which colours a map by the key of every phase and the background for what is not indexed.
`key.plot()` draws a key as its sector, an rgb surface on the chart grid. The white half of
`2`, `2/m` and `23` is the mirror image of MTEX's, as in Julia, and the three are pinned so.
On the way `projectIntoRegion` of P1 became one vectorised pass over the images and the
stepwise nearest search of the S2 grid runs ring by ring; both were per-point loops. Tests
`tests/test_sectorfold.py`, `test_directionkeys.py`, `test_orientationkeys.py`; rendered
`bench/plots.py` 21 to 29. Left out: `PatalaColorKey`, `spectralTransmissionColorKey` (P4).
The tensor plots came 2026-09-22 with P4's tensors, in `tensorplot.py`: `plot(T)` draws
`directionalMagnitude(T)`; `plot((r, g, b))` of three functions on the sphere, MATLAB's
`plot(sF, 'rgb')`, is one surface with the channels as its colours through `_rgbMesh` of the
keys, `rgb=True` accepted, `complete` dropping the sector; `plotSeismicVelocities(C)` is
MATLAB's two by three overview (Vp, the S-wave anisotropy, Vs1 and Vs2 with their
polarisations, Vp/Vs1, Vp/Vs2; the maximum a black square, the minimum a white circle, the
percentage anisotropy under every chart, `red2blue`, `lower=True`, `symmetriseMinMax=True`).
The extrema are read off a one degree grid and the six charts share one cache of the
Christoffel solve per grid (the composed handles were slow when this was written, since an
antipodal handle averaged over its group on every call; that is fixed in `handles.py`, the
cache still saves five solves per grid). `complete` and `upper` combine now, as in MATLAB. Tests `tests/test_tensorplot.py`; rendered `bench/plots.py` 70, 71.

**S4 - EBSD and grain maps. BUILT 2026-09-22.** `mtex/plotting/maps.py`: the geometry as
Julia designed it, `mapLayout(pC)` the signed permutation that takes the plane to the screen,
`mapRaster` the picture of a rigid lattice (a pixel belongs to the cell of the nearest site
among the four corners of its lattice parallelogram, so hexagons are hexagons; one pixel per
cell when aligned, a quarter spacing otherwise; no picture when the residual exceeds a quarter
pixel or a hundredth of a cell), `mapCells` the cells of a drifted lattice with corners at the
mean of the neighbouring centres, `mapPoints`, `mapExtent`, `boundaryPolyline` with NaN breaks,
`boundarySegments`, `grainPolygons`, `scaleBar`, `switchUnit`, `triadGeometry`, `frameOnScreen`,
`phasePalette`. `mapplot.py`: `mapPlot` on the axes with the layout and the extent,
`plot(ebsd)`, `plot(ebsd, values | rgb | ori | key | 'orientation', faceColor=, backend='imagesc' |
'patch' | 'scatter', colorbar=, legend=, micronbar=, refFrame=, how2plot=, coordinates=, region=,
edgeColor=)`, `plot(grains, property, noBoundary=, lineColor=, lineWidth=)`, `plot(gB, values |
rgb, lineColor=, lineWidth=, edgeAlpha=)`, `quiver(grains | ebsd | gB, v)`, the scale bar with
the triad in its box redrawn when a held map widens, the phase legend and the key legend as
inset axes. A map drawn into a held map plot widens it, as `hold on` does. Pixel snapping of
the boundary is matplotlib's own `snap=True`. Tests `tests/test_maps.py` (Julia's P10, S4b)
and `tests/test_mapplot.py`; rendered `bench/plots.py` 30 to 39 on the Forsterite map of the
local sample data. Left out: the data cursor, `S2Fun` on grains (S5), the inset key over the map, the wizard's
raster cap. `plot(grains, cS)` came with the crystal shapes, see S5.

**S5 - three dimensional plots. BUILT 2026-09-22, the crystal shapes 2026-09-22 too.**
`mtex/plotting/space.py`: `sphereMesh` (rings joined by triangles wound outwards, fans at
the poles), `sphereGridLines`, `smallCircle`, `frameAxes`, and orientation space as Julia
designed it: `axisAnglePoints`, `maxAngles(oR, V)` reading the region's normals at fixed axes
(`tan(w/2) dot(v, a_i) <= tan(w_i/4)`, zero outside the sector the region is cut down to),
`regionSurface`, `regionWalls`, `regionLines`. `spaceplot.py` on matplotlib's 3D axes:
`plot3d(v, data)`, `scatter3d`, `plot3d(sF, radial=True, colorbar=True)` as the sphere coloured at
its faces, `plot3d(vF)` as arrows and an axis field as segments, `plot(oR)` as the surface in
degrees with its walls and the parallels and meridians on it, `scatter(ori)` with `axisAngle`
(the default, the region behind the points), `Euler` and `Rodrigues`, `ignoreFundamentalRegion`,
`points`, a value per orientation through the colour map. `plot(ori)` and `plot(oR)` dispatch
there from the spherical `plot`. The back half of the grid is not hidden, since matplotlib's 3D
axes has no depth test for lines. Tests `tests/test_space.py` (Julia's S5 and S5b numbers) and
`tests/test_spaceplot.py`; rendered `bench/plots.py` 50 to 58. pyvista was not needed.
The crystal shapes, with G7's `crystalshape.py`: `space.py` got `visibleFaces(cS, view)`,
`shapeOutlines(cS, plane, layout)` (the front faces seen along the normal of the map, flipped
with a layout of negative determinant, as polygons on the screen) and `shapesOnMap(grains, cS)`
(one shape per grain of every phase whose frame fits the shape's crystal frame, `ori * cS *
sqrt(area) + centroid` as one shape list with the grain indices; another lattice draws nothing).
`shapeplot.py` draws them: `plot(cS)` with `faceColor=`, `faceAlpha=`, `edgeColor=`, `lineWidth=`,
`colored=True` (a colour per form with the legend of their Miller indices) or `colored=key` (the
direction key of the normals), a colour per face or per shape or a value per face; `plot(x, y,
cS)`, `plot(x, y, z, cS)`, `plot(xy, cS)`; `plot(cS, N)`, `plotInnerFace`, `plotInnerDirection`
(the arrow up to where the direction leaves the shape), `plotSlipSystem` and `plot(cS, sS)`;
`plot(grains, cS)` through `plotGrains`, `faceColor` per grain indexed by the grains that take the
shape. A shape stands in a 3D axes, `_spaceAxes` of `spaceplot.py`, fitted around what is drawn
at equal scale; drawn into a held map plot it is projected there, front faces only, as MATLAB's
lifted patch is seen from above. Every shape of one 3D axes joins one `Poly3DCollection`, with
the alpha in the colours, because matplotlib orders collections by depth as wholes and only
the faces within one against each other; a twin drawn into a held parent interleaves that way.
Tests `tests/test_shapeplot.py` and the shape test of `tests/test_space.py` (the Julia S5
numbers: a cube seen along the map normal is a square of side sqrt(area)/sqrt(3)); rendered
`bench/plots.py` 60 to 68, the MATLAB CrystalShapes page's figures. `plot(plane)` of a `plane3d`
lives there too: the polygon the plane cuts from the bounds of the 3D axes, as MATLAB clips it.

**S6 - poles of orientation lists and measured pole figures. BUILT 2026-09-22.**
`mtex/plotting/poles.py`: `poles(ori, h)` and `inversePoles(ori, r)` as the `n x m` lists over
the distinct equivalents, an axis and its antipode counted once, axes where the pole figure
cannot tell `h` from `-h` (`_antipodalPole` of P2), `orbit` for any frame with a group, and
`dynamicMarkerSize`, fifty times the mean spacing of the markers on a hemisphere.
`polefigureplot.py`: `plotPDF(ori, h)`, `plotPDF(ori, data, h)`, `plotPDF(ori, h, points=100,
contourf=True, antipodal=True, complete=True, title=False)`, `plotIPDF(ori, r)`, one marker per
orbit projected into the sector of the frame, the budget of MTEX (ten thousand markers per pole
figure, a hundred thousand per inverse pole figure) spent before the equivalents are formed by a
deterministic thinning; the styles draw `calcDensity` of the poles or `interpolate` of the
values. `plot(pf)` draws a measured pole figure as its specimen directions coloured by intensity,
one panel per pole figure titled by its Miller strings joined with ` + `, each on its own colour
range (`colorRange='individual'`) since the counts of two measurements are not comparable;
`smooth`, `contourf`, `contour` draw `S2FunHarmonic(pf.select(i))`. `plotDiff(pf, odf | pf2,
'RP' | l1=True | l2=True)` draws `calcErrorPF`. `plotPDF` and `plotIPDF` of an ODF stay in
`sphericalplot.py`; the new module dispatches on the first argument, and `plotPDF`/`plotIPDF`
are installed on `orientation`, `plot`/`plotDiff` on `PoleFigure`. On the way: `_plotRow` takes
`colorRange='individual'`, a marker below four points loses its edge, and the axis at the pole of
a chart is no longer labelled, since the label hid the data of a pole figure. Tests
`tests/test_poles.py` (Julia's S6 counts) and `tests/test_polefigureplot.py` on the Dubna
fixture; rendered `bench/plots.py` 44 to 49, the Dubna set from the local sample data. Left out:
`plotPDF!` adding poles to an existing pole figure plot beyond `hold`, the `symmetry=False` flag,
`superposition` for a list.

## 5. Keyword table

| keyword | meaning | default |
| --- | --- | --- |
| `projection` | `'earea'`, `'schmidt'`, `'eangle'`, `'stereo'`, `'edist'`, `'orthographic'`, `'gnomonic'`, `'plain'` | `'earea'` |
| `upper`, `lower`, `complete` | the hemispheres drawn | by the data |
| `antipodal` | `v` and `-v` are one point | by the data |
| `fundamentalRegion` | draw into the sector of the frame's group | `False` for lists, `True` for symmetric functions |
| `symmetrised` | draw every equivalent of a crystal direction | `False` |
| `how2plot` | a `plottingConvention` for this plot | the frame's |
| `resolution` | the grid step of a filled plot | `1*degree` |
| `contours` | the number of levels or the levels | 10 |
| `colorRange` | `(lo, hi)` or `'equal'` over a row | the data |
| `colormap` | an MTEX or matplotlib name | `'WhiteJet'` |
| `colorbar` | draw one beside the plot | `False` |
| `minmax` | annotate the data range | `False` |
| `grid`, `grid_res` | the meridians and small circles | `False`, `30*degree` |
| `axisLabels=False`, `title=False` | leave out the frame labels, the titles | `False` |
| `label` | text at every marker | none |
| `MarkerSize`, `Marker`, `MarkerFaceColor`, `MarkerEdgeColor`, `MarkerColor` | as MATLAB | 11 points, `'o'` |
| `parent`, `add2all` | where to draw | the current figure |

## 6. Tests

- `tests/test_spherical.py`: the projections and their inverses, the equal area and conformal
  properties, the charts of a sphere, a hemisphere, an antipodal region and a sector, every
  direction on exactly one chart and back again, the azimuth range of the cubic wedge, the grid
  inside the sector with finite plane coordinates, the three arcs of the cubic sector, the
  labels `[001]`, `[101]`, `[111]`, the axis labels of a specimen frame, the closed grid of a
  hemisphere, the eleven grid lines, the plain projection's rectangle, the sector chosen by a
  per-plot convention.
- `tests/test_sphericalplot.py`: on the Agg backend, `plot(v)` makes two axes for a list on
  both hemispheres and one with `upper=True`, the markers lie inside the chart limits, `plot(v,
  data)` sets the colour range, `contourf(sF)` and `pcolor(sF)` produce a filled artist per
  axes, the colour bar, `plotPDF` makes one axes per Miller index with one colour range,
  `plot(sR)` draws the outline, `hold` adds to the current axes.
- The rendered checks of `bench/plots.py`, compared by eye with `../mtexJulia/dev/out`.

## 7. Left out for now

`mtexFigure`'s resize layout and the adopted legends, the data cursor, the `arrow` markers,
`surf` with rgb, the square projection, `plot3d`, `plot(...,'3d')`, `sphericalPlot.TL/TR/BL/BR`
beyond the min/max annotation, `saveFigure`, and everything of S2 to S6 above.

## 8. Where the port follows Julia rather than MATLAB

- The grid of a filled plot follows the boundary of the region in one strip per branch of
  the polar set; MTEX draws strips of several intervals.
- A symmetric pole figure is drawn on the specimen sector and a direction projected into it,
  one marker per orbit, instead of every element of the specimen group drawn.
- The lower chart is a fold of the upper one, `theta -> pi - theta`, not a class of its own.
