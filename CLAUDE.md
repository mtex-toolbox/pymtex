# MTEX for Python

The Python port of MTEX, the crystallographic texture analysis toolbox. The plan is
`docs/PLAN.md`, the geometry subplan `docs/plans/geometry.md`, the semantics
`docs/design/semantics.md`, measurements `docs/research/`.

## Two references, and the rule between them

- **Syntax**: the MATLAB branch `../frameSymmetry` (not `../master`, which still has
  `crystalSymmetry`). Class names in MATLAB casing (`vector3d`, `rotation`, `orientation`,
  `crystalFrame`, `S2Grid`), verbs (`calcGrains`, `symmetrise`), static constructors
  (`rotation.byEuler`, `specimenFrame.rolling`), properties (`ori.CS`, `m.hkl`).
- **Design**: the Julia port `../mtexJulia` (`docs/adr/`, `docs/plans/`), which separated the
  mathematics from the MATLAB workarounds. Frames carry the point group, objects carry
  frames, frames are interned handles compared by identity, rotations are proper, point
  groups are built from their Hermann-Mauguin symbol in closure order, nothing indexes
  group elements by position, the symmetry reduction is one kernel.
- When they disagree on a name the user sees, MATLAB wins; on a representation or an
  algorithm, Julia wins. **Never transcribe code from either**; design from them.
- The exception is `docs/plans/syntax.md`: where MATLAB's name is a misnomer, a duplicate, a
  MATLAB workaround or a typo, the Python name listed there replaces it, with no alias. The
  definition cites MATLAB's name in its docstring as `MATLAB's plotPDF` (no backticks, the form
  `tools/graph/scan_python.py` reads), so the graph keeps the counterpart;
  `tools/syntaxcheck.py` lists the old spellings left, `--graph` the counterparts lost.

## Spellings decided

- What MATLAB declares as a property, and every `is*` predicate, is a Python property
  (`v.antipodal`, `ori.CS`, `cs.isLaue`). What MATLAB implements as a method is a method
  with parentheses (`rot.angle()`, `cs.numSym()`) and also a module function (`angle(r)`).
- Boolean flags are keywords named positively (`thinning=False`, not MATLAB's `'nothinning'`);
  a positional string names something (`'uvw'`, `'ZYZ'`, `'X||a'`, `'{111}'`). The keywords
  are camelCase (`markerSize`, `displayName`, `lineWidth`), a tolerance is `tol`, an iteration
  bound `maxIter`, the progress output `verbose` (default False), a symmetry `CS`/`SS`, and
  British spellings (`symmetrise`, `regularisation`).
- Everything on SO(3) carries the two frames it maps between as `frameA` (from, the crystal)
  and `frameB` (to, the specimen or the second crystal), as the Julia port; `CS` and `SS` are
  convenience properties over them.
- Julia's type parameters are fields here: `antipodal`, `isNormalized`, `dispStyle`,
  `ori.antipodal`; frame kinds and properness stay classes (`rotation` has no `i`,
  `orthogonalTransform` holds the parity as a fifth coordinate).
- `xvector`, `yvector`, `zvector` are minted on every access (module `__getattr__`), since
  a shared mutable constant once carried a frame from one test into the next.
- `find` returns the MATLAB pair: `(ind, d)`, `(ind, d)` with k columns, or two sparse
  matrices for a radius. On an SO(3) grid the nearest form's `d` is cos(w/2), as MATLAB's
  `SO3Grid/find`, the other forms angles. Internal searches (the orientation tree, the
  kernels) return cos(w/2), which needs no arccos; only a public function converts.
- Every container is an `MtexArray` over one `(..., k)` NumPy array with carried
  attributes; a single element is the zero-dimensional container. Test a container with
  `isinstance(x, MtexArray)`, never `hasattr(x, "data")`: an ndarray has `.data` too.
- A base-class method must not call `self.dot` or `self.angle`: the orientation subclass
  intercepts them with the symmetry-reduced versions.
- A user's preferences persist in `mtex_settings.json` (`geometry/settings.py`, MATLAB's
  `mtex_settings.m`), read at import, written by `setMTEXpref(..., save=True)` and
  `plottingConvention.default(..., save=True)`; a fresh installation draws `y↓→x`. The tests
  (`conftest.py`) and the doc build set `MTEX_SETTINGS=""` so that they see the defaults.
- The coefficients a user reads and passes, `fhat`, are MATLAB's: MTEX's spherical harmonics
  without the Condon-Shortley phase, MTEX's Wigner blocks, `S1FunHarmonic` over e^{-ikx}. The
  internal `_fhat` keep `docs/design/harmonics.md`; the constructors and the `fhat` property
  convert, and internal code builds with `_internal=True`.
- The padding `gridify` adds to close a grid is notIndexed (phase 0, since a phase id indexes
  the phase list) and marked in `ebsd.isPadding`, MATLAB's NaN phase: the extent, the phase
  counts and selections, the plots, the decomposition and `fill` leave it out, so a grid
  segments exactly as its list does. New code over a map's cells must do the same. The one
  exception is `fill(ebsd, grains)` and `smooth(ebsd, F, fill=grains)`: a grain's polygon puts
  the padding it holds inside the scan, so it is filled and stops being padding, as MATLAB's
  `fill(ebsd, grains, 'gridify')` fills the dropped sites of a sparse map.
- `loadEBSD` builds the grid while it reads: where the row order (or a header the positions
  confirm) gives the lattice, `mapFromColumns` in `io/load.py` writes every column into its grid
  cell and the list is never built; it must stay equal to `gridify` of the list (a test per format).
  A rigid `rotate` or `shift` carries the lattice (`lattice.moved`) instead of inferring it again.
- The preference `hugePages` (default False) is NumPy's madvise of huge pages, process wide, set at
  import: on fragmented memory it made fresh large arrays ten times slower to fill.
- `calcGrains` gives the notIndexed pixels a grain absorbs the grain's phase and a NaN rotation,
  as MATLAB's returned map does (`absorbNotIndexed`), the padding left out.
- `calcGrains` alters the map it is given, as the Julia port does and MATLAB's
  `grains = calcGrains(ebsd)` does not: the grain ids, the pixels the `minPixel` cull drops and
  the absorbed pixels stay on it. This is deliberate; reconstructions to be compared run on
  copies (`ebsd.copy()`).
- Moving a gridded map (`rotate`, `shift`, `transform`) never reorders its arrays: the positions
  move and `ebsd.layout` turns with them, where MATLAB stores a rotated map anew in its default
  layout. Deliberate (user, 2026-09-26); reordering is asked for explicitly, by `gridify(ebsd, gL)`
  or `transformReferenceFrame`.
- A function on SO(3) is evaluated at an orientation whose frames are its own or their
  siblings differing only in the group (`stripSym`, another group on the same axes), by the
  rotation, as MATLAB, whose compatibility check in `eval` is commented out; a frame with
  other axes is refused (`_sameFrame` in `functions/rbf.py`).
- `fundamentalSector` is the conventional sector, the one MATLAB and an inverse pole figure
  show, wherever `tilesSphere` confirms it is a fundamental domain of the group (every
  crystal frame); for a group turned against its axes, as the rotations two frames share,
  it is `dirichletSector`. Directions are folded into it, the orientation projection takes
  the conjugate whose axis lies in it; "the equivalent nearest a centre" is a separate call.

## Running

```
.venv/bin/python -m pytest -q            # about 70 s; the SO3Grid and CSL tests dominate
.venv/bin/python bench/g1_layout.py      # the kernel benchmark of docs/research/benchmark-g1.md
uv sync                                  # Python 3.14, NumPy 2, SciPy, finufft; pytest and numba for tests
```

The products, the searches of the symmetry reduction, the projections and the membership in a
fundamental region are numba loops in `mtex/geometry/kernels.py`, compiled on first call, with NumPy forms as fallback and test
oracle; the maps layer keeps its own in `mtex/maps/kernels.py` (the clipping, the union-find,
the loop tracing, the sums of the grain measures over the loops, the fused gradient, KAM and
GROD over the lattice, the simplex of the dislocation fit), the planar hull in `mtex/geometry/hull.py`, the inverse pole figure
colouring of a map in `mtex/plotting/kernels.py` and the NUFFT nodes and real defects of the
harmonic transforms and the direct Wigner series below 100 nodes in `mtex/functions/kernels.py` and the batched symmetric 3 x 3 eigen
decomposition of the wave velocities and the turn of flattened tensors by many rotations (the
embedding's images, `ori * T`) in `mtex/tensors/kernels.py`, on the same pattern;
`setMTEXpref('kernels', 'numpy')` selects the fallback, the preference `threads` (default the
physical cores, `threads()` in `geometry/kernels.py`) the threads of numba, finufft and
`scipy.fft`. A parallel loop called from Python is compiled by `_parallelKernel`
(`geometry/kernels.py`), which under numba's `workqueue` layer (macOS) holds one lock around
every launch; a parallel helper called from inside another compiled loop is a plain
`nb.njit(parallel=True)`, never a `_parallelKernel`. The NFFT of `../nfft` (branch feature/mtex-kernels, built with CMake as
`libnfft3_omp.so`) is an optional backend of the SO(3) adjoint through ctypes,
`mtex/functions/nfft.py`, found by the preference `NFFTLibrary` or `MTEX_NFFT_LIBRARY`;
finufft stays the dependency. Never import numba at module level. The SO(3) grids (`SO3Grid`, `SO3FibonacciGrid`) search through a quaternion
k-d tree, `mtex/geometry/grids/orientationtree.py`, for nearest, k-nearest and radius; the
radius search, what ODF reconstruction needs, is on `SO3Grid` the structured search of
`mtex/geometry/grids/gridsearch.py` (one image per coset of the rotations about z, the rings,
an alpha window per ring and a gamma window per node), the tree its fallback and oracle;
`gridRadius` picks. The nearest search of the S2 ring grid is the same pattern there: the
structured ring search compiled as one parallel loop, binary searches over the ladders, the
Python `find` as fallback and oracle.

`tools/graph/build.py` writes the dependency graph of MATLAB and both ports, the Julia port read
by `scan_julia.jl` with Julia's own parser (`tools/graph/query.py` asks it: `node`, `impact`,
`algo`, `gaps [--port julia]`, `layer`, `questions`). A research question is a
`###` subsection, or else a top level bullet, under a heading naming questions ("Open questions",
"Research questions", "research agenda") in any note the graph reads (plans, research notes,
memory, MATLAB master's `tests/*.md`, the Julia port's research notes); name the functions it
concerns in backticks and it links to them. Its algorithms are the `ALGORITHM` literals of
the benchmarks; the `bench/*_ops.py` (geometry, functions, tensors, embedding, maps, analysis,
io, plotting) check this port against MATLAB on one input through `tools/graph/crosscheck.py`,
`run`, `matlab` one op per bridge call, `julia` (the `bench/*_ops.jl` of the same name through
`tools/graph/crosscheck.jl`, warmed up, sampled profiles), `compare`, each tolerance with its
reason beside it.
One test file per source file, named `tests/test_<file>.py`, one statement per test (`docs/plans/tests.md`:
the kinds kept, when a test goes; `query.py tests <node>`, `overlap`, `untested` read a test's subject, the
functions its assertions check, and the coverage run's timings); `tests/test_semantics.py`
holds one test per statement of the semantics document and must stay green. The Julia
port runs with `julia --project=../mtexJulia/dev`; MATLAB through the bridge in
`../master/docs/agents/matlab-bridge/` (`../master/CLAUDE.local.md` has the machine's
paths) - **never start a MATLAB run that could take more than five minutes without
asking**, and never the MATLAB test tiers.

## Layout

Five slim cores and one layer of algorithms, cut by the direction of the imports
(`docs/PLAN.md` section 8): `mtex.geometry`, `mtex.functions` (the function spaces and the
`PoleFigure` container), `mtex.tensors` (the tensor container and its algebra), `mtex.maps`
(later `mtex.maps3d`), and above them `mtex.analysis` for whatever iterates or optimises:
`pf2odf`, `optimalsample`, `cluster`, `embedding`, `elasticity`, `mechanics`, `optics`,
`taylor`, `texture`, `variational`, `dislocations`, `denoising`. A core imports only cores
below it; `mtex.io` and `mtex.plotting` import from everywhere. The user sees one flat
namespace assembled in that order. The time critical algorithms and their open items are
listed in `docs/research/time-critical.md`.

## Status

P1 geometry: G0 to G7 done (containers, vectors, rotations, point groups, frames, Miller,
orientations, symmetry-aware operations, regions, grids, find, tangent vectors, fibres, planes,
slip systems, crystal shapes, hull, parent-to-child in `geometry/parent2child.py`). P3 maps (`docs/plans/maps.md`) is under way: M1, the `EBSD`
container and the lattice, M2, the decomposition with its numba clipping, M3, the
criteria, the segmentation and `calcGrains` with the grain table, M4, the boundaries with
their resampling and smoothing, M5, the grain polygons with their measures and `merge`,
M6, the shape parameters over the `polygonSet` mixin with G7's `hull.py`, M7, the
intragranular quantities and `calcGND` with G7's `slipsystem.py`, M8, the denoising
filters over the graph of the map with `fill`, `interp` and `reduce`, and M9, the parent grain
reconstruction (`maps/reconstruction.py`, `maps/variantgraph.py`: `parentGrainReconstructor`
with MATLAB's vote table, the grain and variant graphs, checked against MATLAB on the
martensite map), are done, and M10 (2026-09-26): `gridify(ebsd, gL)` and `gridify(ebsd, unitCell=)` with `gridLayout` (`geometry/gridlayout.py`), `cleanUpPseudoSym` (`maps/pseudosym.py`), `simulateEBSD` (`maps/simulate.py`) and the `spatialTransform` family (`geometry/spatialtransform.py`); MCL segmentation is `calcGrains(..., mcl=)`. M11 (2026-09-26): `mapImage` and `mapImageList` (`maps/mapimage.py`), `xcfShift` and `pairShifts` (`maps/registration.py`) and the `trueEbsd` job (`maps/trueebsd.py`), MATLAB's run of the EBSDTrueEbsd page to the digit; `mtexdata('trueEbsdWCCoSmall')` reads the HDF5 file `bench/matlab/export_trueebsd.m` writes, hosted in the data repository. Variants are numbered as MTEX numbers them (`mtexEnumeration`). `rotate(ebsd, r, 'keepEuler')` dispatches to the method, MATLAB's flags as strings. P6 interfaces (`docs/plans/io.md`): the EBSD import is done, I1 to I4,
`loadEBSD` and `EBSD.load` for `.ang`, `.ctf`, named-column tables, `.cpr`/`.crc`, `.osc`
and the vendors' HDF5 (h5py, the `io` extra, configs in `io/hdf5_config/`) in `mtex/io`;
the numeric block of a text file is a parallel numba row parser (`io/kernels.py`,
`np.loadtxt` the fallback and oracle), the lattice is read off the header grid or the order
of the positions and verified, and every format is timed against the Julia loaders in
`docs/research/benchmark-io.md`. I5 export is done too: `export(ebsd, fname)` for `.ang`,
`.ctf`, a table and the HDF5 write-back into a copy of the source file, every one tested by
its round trip, and `mtexdata('forsterite')` fetches and reads the sample data with the
recipes of the pages; the files are the joint data repository `mtex-toolbox/mtex-data` (`../mtex-data`,
`docs/plans/data-repository.md`), whose `manifest.json` at a tag is `mtex/io/sampledata.json`. I6, the pole figure import (`io/loadpolefigure.py`: `loadPoleFigure`, `PoleFigure.load`, the
Dubna, Siemens, Geesthacht, Bruker, Aachen, rw1 and generic readers, `mtexdata('dubna')` and
the other pole figure sets) is done; the ODF, orientation and tensor files wait on P4. P2, the function spaces (`docs/plans/functions.md`), is under way in
`mtex/functions`: F0 harmonics, F1 kernels, F2 transforms with the quadrature grids, F3 the
harmonic types with their operations, F4 the radial basis functions with `calcDensity` and
`interpolate`, F5 the handles, compositions and Bingham distributions, F6 the texture
characteristics and F7 the vector and axis fields are done, the subplan complete; P4 (pole
figures, tensors) is done: P4a the `PoleFigure` container (`functions/polefigure.py`, one
container holding its pole figures as the parallel lists `allH`, `allR`, `allI`, `c`, the
quadrature weights by SciPy's spherical Voronoi), P4b the inversion (`analysis/pf2odf.py`,
`MLSSolver`, `zeroRangeMethod`, `calcODF`, `calcODFIterative`), P4c the `tensor` container
(`tensors/tensor.py`, an `MtexArray` over `(..., 3^R)` with the rank carried, the physical classes
`stiffnessTensor`, `strainTensor`, ... fixing rank and Voigt factors, `EinsteinSum`, `colon`),
P4d `analysis/elasticity.py`, `mechanics.py`, `optics.py`, P4e `taylor.py` (the simplex, `calcTaylor`)
and `texture.py` (`calcTensor`, `calcLankford`, `calcKearnsFactor`, `SO3FunSBF`, `doEulerStep`),
P4f `embedding.py`; the F6 leftovers `optimalSample` (`analysis/optimalsample.py`) and `calcCluster`
with `classix` (`cluster.py`) are done too; finufft is the one dependency, imported lazily. Import cost of `mtex` above NumPy is about 200 ms, from `scipy.sparse`
imported at module level in the G6 grids; the maps layer adds 3 ms. Heavy SciPy imports
go inside the functions that use them. P5 plotting (`docs/plans/plotting.md`) is under way in
`mtex/plotting`: S1, the spherical plots, is done - the chart geometry in `spherical.py`
(`sphericalProjection` with `eareaProjection`, `eangleProjection`, ..., `makeSphericalProjection`,
the grid that follows the boundary, the labels from the frame), `colormaps.py`, `mtexfigure.py`
(`mtexFigure`, `gcm`, `hold`, `mtexColorbar`) and the matplotlib drawing in `sphericalplot.py`
(`plot`, `scatter`, `text`, `annotate`, `contourf`, `contour`, `pcolor`, `quiver`, `plotPDF`,
`plotIPDF`, installed as methods on `vector3d`, the S2 functions and fields, the SO(3)
functions and `sphericalRegion`); matplotlib is the `plot` extra and is imported only inside
the drawing functions. S2, the ODF sections, is done too: `sections.py` (`phi2Sections`,
`sigmaSections`, `axisAngleSections`, ..., `newODFSectionPlot`; a section is a chart) and
`sectionplot.py` (`plotSection` on the SO(3) functions and on `orientation`). S3, the colour
keys, too: `sectorfold.py` (the derived fold, `polarCoordinates`), `directionkeys.py`
(`HSVDirectionKey`, `TSLDirectionKey`, `HKLDirectionKey`, `direction2color`) and
`orientationkeys.py` (`ipfHSVKey`, `ipfColorKey`, `axisAngleColorKey`, `BungeColorKey`,
`spotColorKey`, `orientation2color`, `ipfColors` through a per-frame colour grid). S4, the EBSD
and grain maps, too: `maps.py` (`mapLayout`, `mapRaster`, `mapCells`, `scaleBar`, `triadGeometry`)
and `mapplot.py` (`plot(ebsd | grains | gB, ...)`, `quiver`, the scale bar with the triad, the
legends). `bench/plots.py` writes the rendered checks to `bench/out`, the maps from the
Forsterite map of `../frameSymmetry/data/EBSD`. S6 too: `poles.py` (`poles`, `inversePoles`)
and `polefigureplot.py` (`plotPDF(ori, h)`, `plotIPDF(ori, r)`, `plot(pf)`, `plotDiff(pf, odf)`).
S5 too: `space.py` and `spaceplot.py` (`plot3d`, `scatter3d`, `plot(oR)`, `scatter(ori)` in axis
angle, Euler or Rodrigues space) on matplotlib's 3D axes, and `shapeplot.py` (`plot(cS)`, `plot(cS, sS)`,
`plotInnerFace`, `plotInnerDirection`, `plot(grains, cS)`) over G7's `crystalshape.py` (`crystalShape` with the
MATLAB catalogue in `geometry/data/crystalshapes.py`; a list of shapes is one object over `(n, nV)` vertices) and
G7's `plane.py` (`plane3d`, `plot(plane)` clipped by the axes). `tensorplot.py` draws P4's tensors: `plot(T)`, `plot((r, g, b))` for
the interference colours, `plotSeismicVelocities(C)`. The plotting subplan is complete but for the pieces `docs/plans/plotting.md`
lists as left out. P3b, the maps of volumes (`docs/plans/maps3d.md`, `mtex/maps3d`, `../master` the one
reference), is done through V8, its ten doc pages checked against master: `EBSD3` over a `voxelLattice`, `calcGrains` on voxels, `grain3d` and
`grain3Boundary`, sections, the DREAM.3D and Neper readers, smoothing, the intragranular quantities,
and the drawing: a `scene3d` of NumPy arrays (`plotting/scene3d.py`) held by an `mtexFigure` and shown
as the picture PyVista renders offscreen (`render3d.py`, the `plot3d` extra), so hold and `setCamera` act
on it; `exportHTML` writes it as a three.js page and the doc build links every 3D figure to one; a 2D
map in a plane of space enters a scene as the picture of its own map plot (`volumeplot.py`). P7 documentation
(`docs/plans/docs.md`) started 2026-09-22 with the MATLAB pages of `../frameSymmetry/doc` as
the one reference, text and numbers: a page is `docs/pages/<Chapter>/<Page>.py` in the
jupytext light format, `docs/make.py` builds it into `docs/build/`, `docs/site.py <site>` exports the built pages into the Jekyll site of mtex-toolbox.github.io, `docs/preview.py` serves them locally, ordered by MATLAB's tocs, with their status, `openDoc('<Page>')` (`io/opendoc.py`) writes a page as a `# %%` cell script for a reader, the pages shipped in the wheel as `mtex/docpages` by `hatch_build.py`, `docs/matlabrun.py`
runs the MATLAB script through the bridge into `docs/matlab/<Page>/` for the comparison,
`docs/notebook.py <Page>` opens a page as a notebook in VS Code to run it section by section,
`docs/findings.md` keeps what the pages found and `docs/plans/remaining-pages.md` the status
of every page. D0, the Vectors and Rotations chapters of D1, the ODFAnalysis chapter of D2 and the EBSDAnalysis and Grains chapters of D3 are done (`docs/plans/remaining-pages.md` has every page's state, `docs/findings.md` what they found); CrystalGeometry is done too (9 checked, Operations written), PoleFigureAnalysis, the whole D4 chapter, all 11 checked, CrystalOrientations, all 16 checked (`plot3d(odf)` now draws MATLAB's isosurfaces, so the ODF pages that left it as prose can take it), and Misorientations, all 6 checked, which completes D1; PhaseTransitions, all 11 checked on M9 (2026-09-23), Tensors, Elasticity and Plasticity, all 26 checked, GrainBoundaries, 12 checked and 3 partial, and the five Partial ODF and Vectors pages taking `plot3d` and `arrow3d` (2026-09-23); SphericalFunctions, all 16 checked, and SO3Functions, all 13 checked (2026-09-23), which completes D2; the least squares fits run `functions/lsqr.py`, LSQR with MATLAB's stopping rules. GeneralConcepts, Plotting, Tutorials and EBSD3Analysis followed: every chapter is written, 221 of 235 pages checked. The release (`docs/PLAN.md` section 11): GPL-2.0, `mtex` on PyPI for Python 3.12 or later, CI in `.github/workflows`; the pages join MATLAB's on mtex-toolbox.github.io with a MATLAB | Python switch per page (D5).
The halfwidth selection (`docs/plans/halfwidth.md`) is built through H2: `mtex/functions/halfwidth.py`
(`degreeEnergies`, the cross-fitted degree energies of a sample on S2 or SO(3); `ucvHalfwidth`,
`lowerBoundHalfwidth`, `noiseFloorHalfwidth`, `selectHalfwidth`), `calcKernel(v | ori | ebsd,
'conservative' | 'UCV', groups=)`, and `ori.notIID`, set by `ebsd.orientations` and carried through
indexing and products, which every automatic halfwidth refuses without grain ids; the studies are
`bench/halfwidth_s2.py` and `bench/halfwidth_so3.py`.
P8, the import wizard (`docs/plans/wizard.md`), is built through W6: `importWizard()` opens a
PySide6 window (the `wizard` extra) over a headless core, `mtex/wizard/state.py` and
`callbacks.py`, with `inspectEBSD` in `mtex/io/inspect.py` describing a file from its header
and `buildImportWizardScript` in `mtex/io/importscript.py` writing the Python of an import;
the window is `window.py`, `browser.py`, `phases.py`, `pictogram.py`, `views.py`, `scriptview.py`, the Julia
window as the look, its pictograms and its chooser of the eight conventions included;
`bench/wizard.py` grabs it offscreen into `bench/out/wizard-<tab>.png`, which is also the
picture of the EBSDImport page. The drawing has its own compiled kernels beside the
geometry's, `mtex/plotting/kernels.py` (the colour look-up of a map, the sector fold's polar
radius), and the fold of a direction into a sector is `projectIntoSector` in
`geometry/kernels.py`, which is what an inverse pole figure key is mostly built of. A Qt signal connected to a plain Python callable runs on the
emitting thread: connect a worker's signals to a `@Slot` of an object living on the main one.

## Style

Two-space indentation, one-line lowercase comments saying what the next lines do, no
comment on what the previous version did. A file is one concept. Every edit reads as if
the previous version never existed.

What the user reaches (a public def, class or method, or a function attached under a public
name) says what it is in a lowercase docstring, which hover, completion and `help()` show;
internals keep the `#` comment above them. A module attaches a method to a class defined
elsewhere only through `install(classes, f, name, declare=)` (`geometry/install.py`), never
`setattr` or a lambda. The editor reads the package without running it, so
`.venv/bin/python tools/staticapi.py` writes out what the run assembles: the layers' explicit
imports, the attached methods declared under `TYPE_CHECKING` in each class body, comments above
public definitions moved into docstrings. `tests/test_staticapi.py` fails until it has run.
