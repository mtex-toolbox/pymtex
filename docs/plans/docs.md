# Documentation (P7)

Status: D0 done 2026-09-22, D1 done (Vectors, Rotations, CrystalGeometry, CrystalOrientations and Misorientations), D2 under way (ODFAnalysis done 2026-09-22), D3 under way (EBSDAnalysis and Grains done 2026-09-22, GrainBoundaries 2026-09-23, 12 checked and 3 partial), D4 under way (PoleFigureAnalysis done 2026-09-22, PhaseTransitions done 2026-09-23).

The MATLAB documentation, `../frameSymmetry/doc`, is 229 content scripts in twenty
chapters, publish-style: prose in comment blocks, code between, the output and the figures
of every block rendered under it. It is the one reference of this subplan: the text, the
sequence, the examples, the numbers a page quotes and the figures it shows are MATLAB's,
and the Python page is that page with Python code under it. The Julia port's pages,
`../mtexJulia/docs/literate/`, rewrote the prose in places and are not a reference here;
its status table, `../mtexJulia/docs/plans/remaining-pages.md`, is read once per page as a
hint of which feature a page needs that a port may not have.

## 1. The shape of a page

- A page is one Python script `docs/pages/<Chapter>/<Page>.py`, named as the MATLAB script
  is, in the [jupytext light format](https://jupytext.readthedocs.io/en/latest/formats-scripts.html#the-light-format):
  a comment block preceded by a blank line is prose (Markdown, with `$...$` maths), the
  code between two prose blocks is one cell, and `# ---` ends a cell without prose. This is
  Literate's shape and MATLAB's `%%` shape, and jupytext turns it into a notebook when a
  site wants one.
- The page starts with the imports it uses, `from mtex import ...`, `import numpy as np`,
  and `plottingConvention.default('y↑→x')` where the MATLAB page sets it.
- The text is the MATLAB page's, paragraph for paragraph, and the examples come in the
  MATLAB page's order; the code is the Python API, kept as short as the MATLAB code. When the Python code of an
  example comes out longer than the MATLAB code, the page stops and the API gets the
  missing verb, property or overload first; a workaround on a page is a bug in the port.
- Numbers the MATLAB page quotes stay in the text, so a differing Python result shows when
  the page is read. Every page ends with a **Technical Details** section only when there is
  something to say: what the port derives rather than stores, what is still owed.
- A chapter's own script (`Vectors.m` and its `.toc`) is navigation and waits for the site.

## 2. The build

`docs/make.py [Page ...]` executes every page (or the named ones) cell by cell in one
namespace and writes `docs/build/<Page>.md` with its PNGs beside it, as `make.jl` does:

- the cell's code in a fenced block, then its output in a fenced block: what it printed,
  and the display of a trailing expression that is not an assignment, as Jupyter shows it
  (so `v = vector3d(1,2,3)` prints nothing and `v` prints the container, which is why the
  MATLAB pages leave the semicolon off - the Python page writes the name on its own line);
  a number is displayed as MATLAB's format short, `3.7417`, an integer without decimals,
  and the handles a plot returns are not displayed, as MATLAB's semicolon hides them;
- every open matplotlib figure saved as `<Page>-<cell>.png` and closed, the Agg backend,
  120 dpi;
- an output block longer than 30 lines cut to 15 head and 10 tail lines around `⋮`;
- a failing cell stops the page, the script prints the traceback and exits nonzero at
  the end, and the other pages still build;
- `docs/build/` is not committed; `bench/out` keeps its role as the render check of the
  plotting plan.

`docs/montage.py <Page>` lays the MATLAB figures of a page beside the built ones in one
picture, `docs/build/_montage_<Page>.png`, which is how a page's figures are compared.

The loop per page: `docs/matlabrun.py <Chapter>/<Page>` (one page per call, the bridge
session `mtexfs` started from `../frameSymmetry`; a page over five minutes is killed and
needs the user's say for a longer run), write `docs/pages/<Chapter>/<Page>.py` from the
MATLAB script, `docs/make.py <Page>`, read the diary against the output blocks, the
montage by eye, then a row in `remaining-pages.md` and the findings.

No dependency beyond matplotlib. The site - Jupyter Book or MyST, the navigation of the
`.toc` files, the search - is a later step once the pages exist, as Documenter was in Julia.

## 3. Checking a page

A page is checked against MATLAB: the MATLAB script runs through the bridge of
`../master/docs/agents/matlab-bridge/` on the `frameSymmetry` checkout (`SESSION_NAME=mtexfs`,
master dropped from the path, one file per call) under `diary`, with every figure exported,
into `docs/matlab/<Page>/` - the diary and the PNGs, not committed. The Python page's output
blocks are read against the diary number for number, and the figures by eye. Most pages run
in seconds; a page over five minutes is not started without asking, and a page whose
MATLAB run needs data the machine does not have is marked so. What a page finds goes into
`docs/findings.md`, one table each for the bugs, the gaps beside MTEX, the conventions that
differ and the practicalities; a row leaves when the thing is fixed. The status of every
page is one row of `docs/plans/remaining-pages.md`.

Status words: **Checked** compared with the MATLAB run; **Adapted** prose-only or process
pages rewritten for Python; **Partial** checked with a named feature missing; **Written**
built but not compared; **Blocked** needs a feature that does not exist; **Pending** not
started.

## 4. Order

The pages are ordered by what the port has, and the chapters that need what it has not
are the reason to build it: the documentation is the check of the API, as it was in Julia.

**D0 - the build and the pilot. DONE 2026-09-22.** `docs/make.py` builds the pages,
`docs/matlabrun.py` runs the MATLAB script of a page through the bridge into
`docs/matlab/<Page>/` (the diary with a marker per cell, the anchors of MATLAB's linked
displays stripped, one PNG per figure a cell leaves open), `docs/findings.md` holds the
four tables. `Vectors/VectorDefinition` is built and checked: every display and the four
figures as MATLAB's. It found one bug (the display of a NaN container crashed) and one
missing spelling (`plottingConvention.default('y↑→x')` as a call), both fixed, and three
conventions recorded in the findings.

**D1 - the geometry.** Vectors (7), Rotations (8), CrystalGeometry (10),
CrystalOrientations (16), Misorientations (6): 47 pages on P1. Vectors DONE 2026-09-22, 5
checked and 2 partial (the `arrow3d` figure of Operations, the measured pole figure of
Axes); the chapter found five bugs (the NaN display, the `complete` flag of antipodal
functions, the seeds of the local maxima, the Laue antipodality, the Fibonacci count) and
wrote `vector3d.load`, `export(v)`, `HEALPixS2Grid`, `mtexTitle`, `nextAxis`, the `layout`
option, the kernel plot, the S2 `volume`, `angle_outer`, `dot_outer`, `orthProj` and the
phase abbreviation `ebsd['Fo']`, all in `docs/findings.md`. Crystal shapes and the
3D plots of the shapes chapter are G7 and S5 work and are the first gaps the pages meet;
`plotPDF(ori, h)` and `plotIPDF(ori, r)` of the orientation pages are S6 without the
measured pole figure. Rotations DONE 2026-09-22, all 8 checked: the chapter wrote `arrow3d`, `quiver3(t)`,
`scatter(rot)` of bare rotations with MATLAB's marker options, the flags of `plot(f)`,
`orientation(f, points=n)`, `rotation.fit(method='kabsch')`, `spinTensor(S)` and `vector3d(S)`,
`rotation(cs)` and `orthogonalTransform(cs)`, and found the Euler region projection, the frames
of `orientation.byAxisAngle(Miller, w)` and the style of `Miller(t, cs)`, all in `docs/findings.md`.
CrystalGeometry DONE 2026-09-22, 9 checked and Operations written (its Kikuchi pattern is a
MATLAB object file the frameSymmetry MATLAB cannot load; the port reads it as the data set
`quartzPattern`): the chapter wrote `plot(cs)` for the symmetry elements, `plot(key, complete=True)`,
the labelled markers and arrows, `circle` with a parent, `eq(antipodal=True)`, `ori * S2Fun`,
`crystalFrame(LaueId=k)`, `axesDual`, `det` of three vectors, the Bruker `.phl` reader, the data sets
`olivine1997` and `quartzPattern`, MATLAB's capitalised options on shapes and maps, and found the
unreconciled frames of `angle`, the equator poles, the empty icosahedral Laue sector and the rest in
`docs/findings.md`.
CrystalOrientations DONE 2026-09-22, all 16 checked: the chapter wrote `plot3d(odf)` as MATLAB's
isosurfaces (marching tetrahedra in `plotting/space.py`), `orientation.load` with the data set `tongue`,
`regularSO3Grid`, `std`, `rotation.rand(n, maxAngle=w)`, `byEuler` of triples, the embedding's MATLAB
spellings, an exact Bingham sampler, 3D plots in a `newMtexFigure` layout, and found the grain exchange
flag ignored by the regions, the projection and the sections, the sampled largest angle of a region, the
321 chart sector, the held axis angle plots taking Euler coordinates, all in `docs/findings.md`.
Misorientations DONE 2026-09-23, all 6 checked, which completes D1: the chapter wrote
`plotAngleDistribution` and `plotAxisDistribution`, `calcAngleDistribution` in MATLAB's order and three
forms, `maxAngle(cs)`, the exact uniform axis distribution, the edges of an orientation region with
`boundaryColor` and `noSurface`, MATLAB's display of a misorientation as its parallel planes and
directions, positional legend labels and MATLAB's colour order for held plots, and found
`calcMisorientation` reversed, the sign of `axis(o1, o2)`, the two-phase axis distribution averaged as
antipodal, all in `docs/findings.md`.

**D2 - the functions.** SphericalFunctions (16), SO3Functions (13), ODFAnalysis (18),
GeneralConcepts (12), Plotting (11), Tutorials (5): 75 pages on P2 and P5's S1 to S3 (the
colour keys were built 2026-09-22). The Plotting pages need the annotation verbs; the
tutorials need the sample data of `mtexdata` and, for the EBSD ones, D3.

ODFAnalysis DONE 2026-09-22: 15 checked, 3 partial (ODFTheory, ODFPlot and FibreODFs
without the isosurface plot `plot3d(odf)` of S5). The chapter found the Aachen pole figure
reader taking the tilt markers as intensities, the fibre volume biased by its cap grid,
`mtexdata('dubnaODF')` returning the pole figures, the named fibres not MTEX's, a uniform
part refusing to join a radial function, `calcComponents` half an hour on the Dubna ODF,
and a dozen plotting slips (the layout of sections, the lowest contour band, a prepared
figure ignored, `hold` beside a plain figure); it wrote `crystalFrame.load` for CIF files,
the named rolling components and fibres, `plot(oS)`, `quiver`, shapes, `annotate` and
fibres into section plots, `plot(psi)` and `plotSpektra`, the ODF file interfaces
(`export(odf, ...)`, `SO3Fun.load`, pickling with interned frames), `fundamentalRegionEuler`,
`project2EulerFR`, `calcPDF(odf, None, r)`, `BinghamODF` in MATLAB's forms, the crystal
axes as Millers, MATLAB's named figure sizes and the `silent` flag, all in
`docs/findings.md`. Radial functions with many centres now evaluate and take their
gradient through the harmonic form. Most pages build in seconds; ODFComponents and
ODFCharacteristics reconstruct the Dubna ODF (30 s each) and RandomSampling runs its
halfwidth experiment in four minutes.

**D3 - the maps.** EBSDAnalysis DONE 2026-09-22 at the user's request, ahead of D1 and D2:
12 checked, 1 adapted (the HDF5 developer guide), 4 partial (Index without the cube
coordinates, Grid without the layouts and the transforms, Inter without the regridding,
Filling without the fill by grain polygons), 3 written (SharpPlot held back by the
gridding of the calcite map, KAM and IPFMap whose MATLAB scripts exceed five minutes and
wait for a longer run), 5 blocked (MapsAndImages, SpatialTransform, TrueEbsd,
PseudoSymmetry, Simulation: the image container, M10's transforms, pseudo symmetry and
simulation). The chapter found a dozen bugs and wrote some forty verbs and options, all
in `docs/findings.md`; the builders learned to keep figures across cells, to run every
page in its own process and to hoist MATLAB's local functions. The Grains chapter is
being written by a parallel session. Originally: EBSDAnalysis (25), Grains (15),
GrainBoundaries (15): 55 pages on P3,
after S3 and S4, the EBSD and grain maps, which these chapters are the check of. The pages
Julia blocked stay blocked: the image container, TrueEBSD, Markov clustering, Dream3D.

Grains DONE 2026-09-22 (a second session): 11 checked, 2 adapted (GrainReconstruction without FMC,
HullBasedParameters on the port's own fifteen test shapes), Dream3dGrains and
GrainReconstructionMCL blocked. The chapter wrote `planarColorKey`, the polar plots of
`plotting/polarplot.py` (`histogram(v)` rose diagrams, `plot(S1Fun)`, `plotSection(S2Fun, sec)`,
`plot(shape2d)`, `setView`), `hist(grains)`, `histogram(grains)` and `plotEllipse` in
`grainplots.py`, `logAxis`, the compass needles `plot(grains, v)`, `plotPDF(fibre, h)`, `plot(fibre)`
in Euler space, `annotate(v, plane=True)`, `export(ori)` and `export_VPSC` in
`io/exportorientation.py`, `grains.N`, `grains.phase`, `id2ind`, `findByOrientation`,
`ebsd.how2plot`, `ebsd.project2FundamentalRegion()`, `merge(grains, gB)`, `caliper` over a list
of directions and `S1FunHarmonic.interpolate`; it found the bugs in `docs/findings.md` (the map
plot ignoring `newMtexFigure` and `nextAxis`, a `region` without the layout, one gridspec per
panel, the perimeter counting inclusions, the aliased S1 interpolation, the NaN fibre fit, the
per-phase threshold list, the unimported `orientation` of `quiver`).

**D4 - the rest.** PoleFigureAnalysis DONE 2026-09-22, all 11 checked: the measured data
from their files through the corrections to a reconstructed ODF and back out again, with
the solver traces, the fit errors and the recalculated pole figures of every page agreeing
with MATLAB to three digits. The chapter wrote `SantaFe`, `export(pf, fname, 'degree')` in
`io/exportpolefigure.py`, `v.resolution`, the `parula` colour map, `byMiller` with a
specimen frame, `cat(pf1, pf2)`, `linewidth` on `plot(odf, f)` and MATLAB's `FontSize`
option; it found `mtexColorMap` and `setColorRange` flattening a filled contour plot (the
bands are cut out of the map for one range, so a change of either draws the plot again now,
which corrected the contour figures of VectorsDensityEstimation, FibreODFs and ODFImport
too), `mtexColorbar` adding a bar where MATLAB's toggles, the RP threshold of `calcError`
left at the regularised measure's 0.5 instead of MTEX's 1, which ordered the two ghost
correction reconstructions the wrong way round, `plotPDF(odf, pf.allH)` failing on a
superposed entry, and the ` + ` written between
the directions of such an entry, all in `docs/findings.md`. Three differences survive and
the pages say so: the Geesthacht normalisation, where MATLAB's two scans carry different
antipodal flags after the correction; the refinement loop, where MATLAB's `refine` measures
below the cap; and the counts of the equispaced SO(3) grids. The Dubna reconstruction takes
about 11 s on the 5 degree grid, and PoleFigureSimulation runs its fourteen reconstructions
in twelve minutes. Tensors, Elasticity and Plasticity, all 26 checked on 2026-09-23, were open since P4's tensors landed on 2026-09-22 (section
11 of `docs/plans/functions.md` lists the names, MATLAB's throughout: `stiffnessTensor`,
`Voigt`, `EinsteinSum`, `C.YoungsModulus()` with parentheses, `velocity(C)`,
`calcTaylor`, `calcLankford`, `SchmidTensor`; not ported are `plot(C)`,
`plotSeismicVelocities`, the implicit `doEulerStep`, `calcEaring` and the json and mpod
tensor files, and `max` of a velocity field and `calcLankford` of an ODF are slow);
PhaseTransitions (11) done 2026-09-23, all checked, on M9; EBSD3Analysis (4) blocked with Julia.

**D5 - the site.** The chapter navigation from the `.toc` files, the function reference
from the docstrings, the executed pages as the check in CI.

Each step ends with the page counts of the status table and the findings it left, as the
other subplans end with the test count.

## 5. Decisions taken

- Scripts, not notebooks, so that a page diffs and reviews like code, as Literate and
  MATLAB's publish are; the light format so that a notebook is one `jupytext` call away:
  `docs/notebook.py <Page>` converts a page into `docs/build/<Page>.ipynb` and opens it in
  VS Code, one cell per paragraph, for running it section by section (jupytext and
  ipykernel are in the dev group, the Python and Jupyter extensions in VS Code).
- The build is a hundred lines of our own rather than sphinx-gallery or MyST-NB, so that
  the pages execute today without a site, and the site chooses its tool later against
  built pages.
- MATLAB is the one reference of the pages, text and numbers, decided 2026-09-22: the
  Julia pages drifted from the MATLAB text in places, and the syntax rule of `CLAUDE.md`
  applies to the documentation too - a MATLAB user reads the same page with Python code
  under it. Julia's design shows only in the Technical Details section, where the port
  says what it derives rather than stores.
- The MATLAB run of a page is kept beside the Python build rather than compared once and
  thrown away, so that a page rebuilt after an API change is re-checked without a new run.
