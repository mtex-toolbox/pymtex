# What the documentation pages found

Every row names the page it came from. A row of the bugs and gaps leaves when the thing is
fixed; the conventions and practicalities stay, they are what a reader of the MATLAB page
has to know.

## Bugs

| Page | Found | Status |
|---|---|---|
| VectorDefinition | the display of a container holding NaN crashed in `numString` (`int(nan)`) | fixed 2026-09-22, prints `NaN` as MATLAB does |
| VectorsAxes | `complete=True` drew one hemisphere of an antipodal function or pole figure, the flag being dropped before the chart chooser in `_plotFunction` and `_plotRow` | fixed 2026-09-22; `test_pole_figures_of_an_odf` expected the old behaviour and was changed |
| VectorsDensityEstimation | `max(pdf, numLocal=3)` returned two maxima where MATLAB finds three: the seeds of the ascent were the best nodes by value at a small exclusion, all on the flanks of the two big peaks | fixed 2026-09-22: the seeds are the local maxima of the grid, the neighbourhood taken over the copies of the grid under the group and the sign; the three maxima and their positions now match MATLAB to four digits |
| VectorsDensityEstimation | the inverse pole density of a Laue frame reported `antipodal: false` where MATLAB's is antipodal | fixed 2026-09-22: a harmonic with a centrosymmetric frame is even |
| EBSDOrientationPlots | the pole figure sample budget divided 10000 by the number of poles of h instead of the number of symmetry elements, 10000 markers for MATLAB's 1250 | fixed 2026-09-22 |
| EBSDOrientationPlots | `hold` was ignored by the pole figure rows of an orientation list, a second call opened a new figure | fixed 2026-09-22 |
| EBSDOrientationPlots | `plotSection(ori, values, 'sigma')` dropped the positional values | fixed 2026-09-22 |
| EBSDOrientationPlots | `scatter(ori)` drew axis angle space where MATLAB draws the Euler region, and every one of 152345 points | fixed 2026-09-22: the Euler region for an orientation, 2000 points sampled |
| EBSDSharpPlot | the calcite map (`mtexdata('sharp')`, positions on a lattice with basis (0.5, 0.5), (-1, 1)) is gridded onto 301 x 151 sites whose positions form a triangle in the scan's box, half the sites drawn on top of the others, in every backend; MATLAB pads an axis-aligned 301 x 151 grid instead | open |
| EBSDSharpPlot | `mean(ori)` failed on a list holding NaN orientations, as a gridded map's list does now | fixed 2026-09-22: the quaternion and orientation means skip them |
| EBSDSharpPlot | `ipfHSVKey.maxAngle` and `whiteCenter` could not be set; a colour array in the map's shape was refused by the map plot | fixed 2026-09-22 |
| EBSDInter, EBSDFilling | `ebsd['xy', x, y]` and a flat boolean mask failed on a gridded map, the flat positions being taken as row indices | fixed 2026-09-22: a flat mask or list of positions addresses a gridded map in its linear order |
| EBSD2ODF | the seeds of the local maxima failed on a Laue group, an improper element being no rotation | fixed 2026-09-22: the copies under the proper subgroups |
| EBSDGrid | `ebsd.orientations` dropped the sites of other phases and the empty sites of a gridded map, so `plot(ebsdMg, ebsdMg.orientations)` had fewer orientations than measurements | fixed 2026-09-22: one orientation per site in the shape of the map, NaN where the site is not of the phase, as MATLAB keeps it |
| EBSDPlotting | the phase legend listed a not indexed phase without a colour, and failed on grains without a phase | fixed 2026-09-22 |
| VectorGrids | `fibonacciS2Grid(resolution=)` rounded the point count before halving and gave 825 nodes at 7 degree for MATLAB's 827 | fixed 2026-09-22, MATLAB's rule |
| GrainReconstruction | `plot(ebsd)` and `plot(grains)` opened a figure of their own after `newMtexFigure(layout=...)` and `nextAxis()`, so a two-panel map ended up as three figures | fixed 2026-09-22: the map plot takes the pending axes or the first axes of a fresh figure, as the spherical plots do |
| GrainReconstruction | the phase legend listed notIndexed, whose colour on the Forsterite map is NaN | fixed 2026-09-22 |
| GrainSpatialPlots | `text(grains, labels)` failed, reading the centroid as a method, and drew labels outside a `region` | fixed 2026-09-22, clipped to the axes |
| GrainSpatialPlots | a `region` in map coordinates was set as the axes limits without the layout, so a map under `y↓→x` came out empty | fixed 2026-09-22 |
| GrainMerge | `newMtexFigure` and `nextAxis` gave every child its own gridspec, so the panels of a figure that grew by `nextAxis` overlapped, and the figure did not widen | fixed 2026-09-22: one gridspec, the figure grows with its layout |
| GrainMerge | `setColorRange` after `mtexColorbar` left one bar per axes where MATLAB redraws one for all | fixed 2026-09-22 |
| ShapeParameters | the Euler angles of a rotation list came as three rows, which the grain display read as one row per grain | fixed 2026-09-22 in the display; `Euler()` keeps its layout |
| HullBasedParameters | `perimeter` counted the loops around inclusions where MATLAB's counts the outer loop, so `deltaP` of a disc with a hole was 22 percent | fixed 2026-09-22: `perimeter` is the outer loop, `perimeterWithInclusion` the whole |
| ProjectionBasedParameters | `S1FunHarmonic.min` returned the maximum's position on a series interpolated at the session bandwidth: the samples were aliased | fixed 2026-09-22: `interpolate` fits up to the Nyquist limit of the samples |
| GrainOrientationParameters | `fibre.fit` reported a NaN fit angle and `scatter(ori)` refused a list holding NaN orientations, as the measurements a closing absorbed are | fixed 2026-09-22 |
| GrainOrientationParameters | `quiver` in `sphericalplot.py` named `orientation` without importing it | fixed 2026-09-22 |
| GrainReconstructionAdvanced | a list of thresholds, one per phase in the order of `CSList`, was taken as one threshold unless an entry was itself a pair | fixed 2026-09-22: a pair of numbers is `[high, low]`, any other list is per phase |
| RotationFibre | `project2EulerFR` took the first symmetric equivalent inside the Euler box, so a fibre segment inside the box jumped to another representative for a third of its points; MATLAB keeps the given one and folds the first and third angle by the multiplicity about z first | fixed 2026-09-22 |
| RotationSpinTensor | `orientation.byAxisAngle(Miller(..., cs), w)` without frames built an orientation into the specimen frame, so `oriRef * increment` refused the frames; MATLAB makes it a rotation of the crystal frame into itself | fixed 2026-09-22 |
| RotationSpinTensor | `Miller(t, cs)` of a right tangent vector kept the vector's UVTW style, so `round` gave (2 4 -6 9) for MATLAB's (1 2 -3 3) in hkil | fixed 2026-09-22: a plain or tangent vector takes hkl or hkil as MATLAB's `Miller(v, cs)` |
| RotationPlotting | `scatter(rot)` of a bare rotation failed on `project2FundamentalRegion`; a rotation has no region | fixed 2026-09-22: the whole Euler box, the 180 degree ball in axis angle space |
| RotationFibre | `project2EulerFR` took the first symmetric equivalent inside the Euler box, so a fibre segment inside the box jumped to another representative for a third of its points; MATLAB keeps the given one and folds the first and third angle by the multiplicity about z first | fixed 2026-09-22 |
| RotationSpinTensor | `orientation.byAxisAngle(Miller(..., cs), w)` without frames built an orientation into the specimen frame, so `oriRef * increment` refused the frames; MATLAB makes it a rotation of the crystal frame into itself | fixed 2026-09-22 |
| RotationSpinTensor | `Miller(t, cs)` of a right tangent vector kept the vector's UVTW style, so `round` gave (2 4 -6 9) for MATLAB's (1 2 -3 3) in hkil | fixed 2026-09-22: a plain or tangent vector takes hkl or hkil as MATLAB's `Miller(v, cs)` |
| RotationPlotting | `scatter(rot)` of a bare rotation failed on `project2FundamentalRegion`; a rotation has no region | fixed 2026-09-22: the whole Euler box, the 180 degree ball in axis angle space |
| ODFCharacteristics | `mtexdata('dubnaODF')` returned the pole figures, its recipe naming the wrong function | fixed 2026-09-22 |
| ODFCharacteristics | the volume about a fibre came out 92.5 percent for an exact 94.0: the cap grid of `fibreVolume` put more points at large angles, and the inverse pole figure it averaged was the truncated harmonic | fixed 2026-09-22: the tube is integrated directly, the cap rings weighted by their area |
| ODFComponents | `calcComponents` on the Dubna ODF ran for half an hour: the clustering of the modes compared orientations one pair at a time in Python, and the radial function summed its 12000 kernel copies at every seed | fixed 2026-09-22: the angles of one orientation against all at once, the harmonic form for a large kernel sum, 47 s for the page |
| ODFComponents | the third local maximum of `max(odf, numLocal=3)` differs between the direct and the harmonic evaluation of a radial function: a shoulder 11 degrees from the main peak is a grid maximum for one and not the other | open; the page shows the harmonic path's 43.5 mrd where MATLAB finds 31.9 |
| DetectionOfSampleSymmetry | the Aachen pole figure reader took the tilt marker closing every ring, `-05.00`, as an intensity, so the rings shifted and the reconstructed ODF peaked at 12.6 mrd where MATLAB's peaks at 16.9 and the twofold axes came out 25 degrees off | fixed 2026-09-22: the marker is skipped, 16.94 mrd and 3.1 degrees |
| ODFModeling | `0.73 * uniformODF + 0.27 * unimodalODF(mod, psi)` became a composition, the uniform part carrying a kernel of its own that differed from psi | fixed 2026-09-22: a uniform part joins any radial function |
| EulerAngleSections | `odf.SS = specimenFrame('222')` on a composition changed the label without reaching the components, so the next plot refused the orientations | fixed 2026-09-22: setters on the radial functions and the composition |
| EulerAngleSections | `plotSection(odf, ..., layout=[5, 2])` ignored the layout and laid the wide Euler sections out three abreast | fixed 2026-09-22: the layout option, and two columns for wide panels by default |
| ODFShapes | `plot(psi)` in `hold(True)` after `plt.figure()` opened a figure per kernel: `hold` made an MTEX figure of its own beside the plain one | fixed 2026-09-22: `hold` adopts the user's figure, the line plots draw into it |
| RadialODFs | `plotPDF(odf, h)` inside a `newMtexFigure(layout=[2, 3])` loop opened a figure of its own instead of taking the prepared axes | fixed 2026-09-22 |
| BinghamODFs | the lowest band of a `contourf` plot was coloured, so a sigma section of a Bingham ODF had a blue background | fixed 2026-09-22: a band takes the colour of its lower level, as MATLAB's does |
| FibreODFs | `fibre.beta(cs)` was not MATLAB's: the port projected the far end to the equivalent nearest the near end, so the fibre axis came out [0 -6 -7] for MATLAB's (12 6 11) and the fibre ODF twice as sharp | fixed 2026-09-22: the named fibres take the representatives `byMiller` gives, as MTEX defines them |
| CrystalReferenceSystem | `angle(ori1, ori2)` between orientations of one group in differently aligned crystal frames compared the representatives as they were, 0 for MATLAB's 30, where MATLAB's `ensureCS` re-expresses the first in the frame of the second and says so | fixed 2026-09-22 |
| CrystalReferenceSystem | poles on the equator, z = -0.0, were dropped by the upper hemisphere chart, 3 markers for MATLAB's 6; an antipodal pole on the rim is now marked at both ends | fixed 2026-09-22 |
| CrystalReferenceSystem | `plotPDF` after `newMtexFigure` opened its own figure, and after `nextAxis` another one | fixed 2026-09-22 |
| CrystalSymmetries | the proper subgroup of -4m2, 222 turned by 45 degrees, was named unknown; MATLAB names the type | fixed 2026-09-22: the signature names a group in a setting no symbol spells |
| CrystalDirections | `char` of a single crystal direction wrote the family brackets `<101>` for a frame with symmetry, MATLAB's plot label is `[101]` | fixed 2026-09-22: `char(family=False)` for labels, the families stay for legends |
| CrystalDirections | a held `plot` on a figure of two hemisphere charts went to the last chart only, so a direction of the other hemisphere vanished | fixed 2026-09-22: both charts take it, each clipping its own |
| FundamentalSector | the outline of a spherical region was drawn under the mirror circles of `plot(cs)` | fixed 2026-09-22 |
| QuasiCrystals | the fundamental sector of the centrosymmetric icosahedral group was empty, its centre a zero vector, so `ipfHSVKey` coloured the whole hemisphere with one wheel | fixed 2026-09-22: the centre of the Moebius triangle between neighbouring five-, three- and twofold axes |
| CrystalOperations | `gcm()` kept a figure closed with `plt.close`, so the next plot drew into a dead figure | fixed 2026-09-22 |
| CrystalOperations | `ori * S2Fun` warned about specimen symmetry while rotating the coefficients out of the crystal frame | fixed 2026-09-22 |
| PoleFigure2ODF | `calcError(pf, odf, 'RP')` kept the regularised measure's threshold of 0.5 where MTEX's RP measure keeps the directions above 1, so every RP value came out about a fifth too high and the ghost correction comparison of PoleFigure2ODFGhostCorrection ordered the two reconstructions the wrong way round | fixed 2026-09-22: the default is 1 for `'RP'` and 0.5 for the regularised error, MTEX's; the seven Dubna values now agree with MATLAB to three digits |
| PoleFigureRefinement | `plotPDF(odf, pf.allH)` without superposition coefficients failed on an entry of several crystal directions | fixed 2026-09-22: an entry of several directions is their equally weighted mean, as `calcPDF` gives it |
| PoleFigureSantaFe | `FontSize`, MATLAB's spelling of the option, was refused by the plots and reached matplotlib | fixed 2026-09-22: it is read as `fontSize` |
| PoleFigureImport | a superposed entry was displayed and titled with ` + ` between its crystal directions where MATLAB writes them one after the other, and the exported file name is that string | fixed 2026-09-22: the display, the pole figure titles and `char` all write `{011̅1}{101̅1}` |
| OrientationDefinition | `orientation.symmetrise(proper=...)` took a keyword it never read, the reduction groups being proper throughout | fixed 2026-09-22: the keyword is gone, in `orientation.py`, `symmetryops.py` and the orientation tree |
| DefinitionAsCoordinateTransform | the Miller and tensor displays kept four significant digits, `1.667 1.333 0.3333` and `2.485`, where MATLAB rounds to four decimals, `1.6667 1.3333 0.3333` and `2.4848` | fixed 2026-09-22: `numString` and `table` take `decimals`, the Miller rows and the Voigt matrix use it; CrystalDirections and CrystalShapes rebuilt |
| PoleFigurePlot | `mtexColorMap` after a filled contour plot flattened it: `set_cmap` replaced the bands cut out of the old map, and the plot came out in one colour with a colour bar to match. `setColorRange` on one broke it the same way, the levels having been cut for the old range | fixed 2026-09-22: a filled plot remembers its data and is drawn again for a new range or a new colour map, as MATLAB redraws it; it also corrected the LaboTeX contours of VectorsDensityEstimation and the equal ranges of FibreODFs and ODFImport |
| PoleFigurePlot | `mtexColorbar` always added a bar where MATLAB's toggles, so a page that clears the bars before setting one range ended with two | fixed 2026-09-22: a figure that has bars loses them, as MATLAB's `mtexFigure/colorbar` does; `test_mtexfigure` expected the old behaviour and was changed |
| OrientationSymmetry | `plotPDF(ori, h, complete=True)` drew the sector of the specimen group, not the sphere: the flag was popped before the chart was made | fixed 2026-09-22: the flag reaches `makeSphericalProjection`, the two hemispheres are titled once and each says which it is |
| OrientationSymmetry | the poles of a list left the specimen group out, so a complete pole figure showed 6 of MATLAB's 12 poles, and the sample budget divided 10000 by the crystal group alone | fixed 2026-09-22: `poles` is MATLAB's `ori.SS * (ori * symmetrise(h))`, the budget divides by both groups, in `plotIPDF` too |
| OrientationStandard | `round2Miller(ori)` rounded the plane and the direction over four indices where MATLAB's `round` allows twelve, so CubeND22 came out `(001)[31̅0]` for MATLAB's `(001)[52̅0]` | fixed 2026-09-22: the orientation branch rounds over twelve, the misorientation branch keeps its four; CrystalDirections now matches MATLAB's `1.6667 1.3333 0.3333` too |
| OrientationStandard | the fundamental sector of an orthorhombic frame with a mirror was not turned by the plotting convention, so under `y←↑x` the quarter disc opened west where MATLAB's opens east | fixed 2026-09-22: the in-plane normals are flipped to open east, as MATLAB's `fundamentalSector` does for mmm and mm2 |
| OrientationStandard | a line contour drawn over a surface removed the surface, and a held row of pole figures opened a second figure instead of drawing into the one it had | fixed 2026-09-22: a contour keeps its own artists in `sP.lines`, and `_plotRow` takes a held figure of as many charts |
| OrientationStandard | a single RGB triple as `MarkerColor` was passed to matplotlib as a value per marker, which warned and coloured by the colour map | fixed 2026-09-22: a triple is one colour, a list of colours is one per marker |

## Gaps beside MTEX

| Page | Missing |
|---|---|
| EBSDGrid | `gridLayout` with `'rowMajor'` and `'columnMajor'`, `ebsd.layout`, `oldId` and the second output of `gridify`, `gridify(ebsd, unitCell=)` resampling onto another cell, `spatialTransformProjective.byTilt` and `transform(ebsd, T)`; M10 |
| EBSDFilling | `fill(ebsd, grains)` and `smooth(ebsd, F, 'fill', grains)` constrained by the grain polygons: a site outside every grain stays notIndexed, a corridor between grains is never filled; the port fills from the grain of the surrounding measurements and needs `extrapolate=True` for a hole that reaches the rim | M8 |
| EBSDFilling | the `quality` property `smooth` writes, zero where no orientation was measured |
| VectorsOperations | `arrow3d`, the arrows in space of the vector sum; S5 |
| VectorsOperations | the dynamic marker size: MTEX shrinks the markers of a long list, the port draws 764 directions at the size of one; S6 names it |
| VectorsAxes | `plot(pf)` of a measured pole figure and `annotate` on it; S6 |
| VectorsImport, VectorsExport | `vector3d.load`, `loadVector3d`, `export(v, fname)` did not exist | added 2026-09-22, `io/vectors.py`; the vectors file joined the sample data as `mtexdatafile('vectors')` |
| VectorGrids | `HEALPixS2Grid` did not exist | added 2026-09-22, `geometry/grids/healpix.py`, the ring scheme centres, MATLAB's nside rule |
| EBSDImport | a phase list given with a vendor file was refused; an import script states its own frames | added 2026-09-22: `EBSD.load(fname, csList, ...)` replaces the file's phases by the list, the not indexed one first |
| EBSDImport | the data sets of an HDF5 file were not listed on import, `silent` did not exist | added 2026-09-22 |
| EBSDImport, EBSDPlotting | the display and the maps gave the phases no colour where MATLAB assigns LightSkyBlue, DarkSeaGreen, Goldenrod, ... | added 2026-09-22: MATLAB's PhaseColorOrder in `maps/phaselist.py`, shared by the display, the maps and `ebsd['phase'].color` |
| EBSDReferenceFrame, EBSDPlotting | `ebsd.EulerCorrection`, `ebsd['phase'].color` read and set, `str2rgb`, `plot(key)`, `rotate(ebsd, r, 'keepEuler')` at the package level (the function of the function spaces shadowed the method), the `location` of the scale bar, `ipfDirection` and `faceAlpha` of a map plot, the title of a colour bar, the phase legend drawn by default | added 2026-09-22 |
| EBSDGROD | `HSVDirectionKey()` of the whole sphere and `HSVDirectionKey(cs, antipodal=True)`, `plot(key)` for a direction key, `faceAlpha` as one value per measurement on a map and `edgeAlpha` as one per boundary segment | added 2026-09-22 |
| EBSDAdvancedMaps | `plot(key)` and `plot(key, 'sigma', sections=6)` for the Bunge and spot keys, the coloured sections of orientation space (`plotKeySections`); `spotColorKey(ebsd)` and `ipfSpotKey(cs)` with `center`, `color` and `psi` set afterwards; `MarkerAlpha` | added 2026-09-22 |
| EBSDAdvancedMaps | `PatalaColorKey` | S3 |
| EBSDIPFMap | `plot(ipfKey, 'complete', 'upper')` draws the whole hemisphere of a key; the port's key plot draws the fundamental sector whatever the flags | S3 |
| EBSDIPFMap | `plot(ipfKey, '3d')`, the coloured sphere of a key, and `plot(grains, ipfKey)` drawing that sphere on every grain; `ipfKey.colorPostRotation` | S3, S5 |
| EBSDExport | the Oxford binary pair `.crc`/`.cpr` on export, MATLAB's `.mat` working copy | I5 |
| EBSDInterfaceHDF5 | `EBSD.load(fname, 'headerOnly')` and the `'debug'` trace of the HDF5 reader | I4 |
| EBSDOrientationAnalysis, EBSDIndex | `circle(v, radius)`, `nextAxis(k)` choosing an axes, MATLAB's marker names (`'square'`), `plot(odf, f)` along a fibre (`plotting/fibreplot.py`), `plot(odf, 'sigma')` routed to the sections for every SO3Fun form, `text(ebsd, labels)`, `faceColor='none'` outlining the cells, no second scale bar on a held map, the sampling messages of the orientation plots | added 2026-09-22 |
| VectorGrids, VectorsDensityEstimation | `mtexTitle`, `nextAxis`, the `layout` option of a plot, a plot into a fresh `newMtexFigure`, `MarkerEdgeAlpha`, `MarkerFaceColor='none'`, `plot(psi)` of a spherical kernel, `volume` of a spherical function, `v.calcDensity()`, `calcDensity(v, antipodal=True)`, the `sum` (integral) of an S2Fun, `angle_outer`, `dot_outer`, `orthProj` | added 2026-09-22 |
| GrainSpatialPlots | `colorcet('C2')` and the other CET maps; matplotlib's cyclic `twilight` stands in, the hues differ |
| GrainReconstruction | fast multiscale clustering (`'fmc'`), the Markovian clustering and the Delaunay decomposition of `calcGrains`, as in the Julia port |
| SelectingGrains | `selectInteractive`, the mouse selection of grains |
| HullBasedParameters | MATLAB's stored `testgrains` with its forty shapes; the port draws fifteen of its own |
| ODFTheory, ODFPlot, FibreODFs | `plot3d(odf)`, the isosurfaces of an ODF in Euler or axis angle space; S5 |
| ODFTheory | `plot(ebsd, cS)` drawing the crystal shape at every measurement, MATLAB's named figure sizes `figSize='large'`, `plot(ori, 'Euler')` with the space as a positional flag | added 2026-09-22 |
| ODFPlot | `fundamentalRegionEuler(cs, ss)`, `ori.project2EulerFR()` | added 2026-09-22 |
| ODFInversePoleFigure | `calcPDF(odf, None, r)`, the inverse pole density for MATLAB's empty crystal direction | added 2026-09-22 |
| EulerAngleSections | `crystalFrame.load('Al-Aluminum.cif')` (`io/cif.py`, the MTEX CIF files shipped in `geometry/data/cif`, a COD number fetched), `orientation.brass`, `cube`, `goss`, `SR`, `PLage`, `QLage`, `invGoss` and their variants, `annotate(ori)`, `plot(f)` and `plot(ori)` into a held section plot, `coordinates='off'`, `xlabel`, `ylabel`, the `silent` flag everywhere, `LineWidth` in MATLAB's spelling | added 2026-09-22 |
| SigmaSections | `plot(oS)` with the reference fields, `quiver(ori, v)` in sections, `plotSection(ori, ori * cS, oS)` drawing crystal shapes into sections, `oS.h1`, `oS.h2`, `oS.omega` and `oS.sigma` settable, `cs.aAxis`, `cs.cAxis` and the reciprocal ones as Millers in the frame | added 2026-09-22 |
| SigmaSections | `oS.referenceField = S2VectorField.polar(xvector)`, replacing the reference field of a section object; the port's field is a method |
| ODFCharacteristics | `plotPDF(odf, pf.allH, superposition=pf.c)` with one superposed pole figure per entry, `annotate(ori)` and `plot(ori, add2all=True)` marking the poles on pole figures, `calcError(..., 'L1')` in MATLAB's case, `calcComponents(..., silent=True)` | added 2026-09-22 |
| ODFShapes | `plot(psi)` of an SO(3) kernel over -180 to 180 degree, `plot(psi, symmetric=True)` of a spherical one, `plotSpektra(psi)` and `plotSpektra(odf)` (`plotting/kernelplot.py`), `ind2color` | added 2026-09-22 |
| BinghamODFs | `BinghamODF(20, ori)`, `BinghamODF(20, f)`, `BinghamODF([10, 10, 10], quaternion.eye, cs)` with the concentrations padded, `odf.axes`, `orientation.eye`, `quaternion.eye` | added 2026-09-22 |
| ODFExport, ODFImport | `export(odf, fname, 'Bunge', resolution=)`, `export(odf, fname, S3G, 'generic')`, `interface='mtex'`, `'VPSC'` with `points` (`io/exportodf.py`); `SO3Fun.load` with `'interp'` and `'density'`, `columnNames`, `halfwidth`, the VPSC reader (`io/loadodf.py`); `pickle` of a function whose frames come back as the session's handles; the sample table `mtexdatafile('odf')` | added 2026-09-22 |
| ODFImport | the import wizard | not ported |
| PoleFigureSantaFe | `SantaFe`, the benchmark model ODF, did not exist | added 2026-09-22, `functions/rbf.py`: cubic over orthorhombic, 0.73 uniform and a von Mises Fisher component of 10 degree halfwidth at (122)[221] |
| PoleFigureExport | `export(pf, fname, 'degree')`, one table of polar angle, azimuth and intensity per pole figure, did not exist | added 2026-09-22, `io/exportpolefigure.py`; the round trip through `PoleFigure.load` is exact |
| PoleFigureRefinement | `v.resolution` of a plain list of directions did not exist | added 2026-09-22, MATLAB's rule: the whole circle up to four points, the square root of the median Voronoi cell area up to fifty thousand, the equal area estimate beyond |
| PoleFigurePlot | the `parula` colour map was not among MTEX's | added 2026-09-22, `plotting/colormaps.py` |
| PoleFigureSantaFe | `orientation.byMiller(hkl, uvw, cs, ss)` read the crystal frame given again as the specimen frame | fixed 2026-09-22: the frame of `hkl` is dropped from the trailing frames |
| PoleFigure2ODFGhostCorrection | `plot(odf, f)` refused `linewidth`, matplotlib's spelling, beside `lineWidth` | fixed 2026-09-22 |
| PoleFigureCorrection | `cat(pf1, pf2)` failed, the generic `cat` passing an axis | fixed 2026-09-22 |
| OrientationSymmetry | `ori * cs` and `ss * ori`, the equivalents on one side | added 2026-09-22, `geometry/products.py`; the group goes along a new axis, so `ss * ori * cs` is shaped (2, 6) |
| OrientationStandard | `legend('show', 'location', 'southoutside', 'numColumns', 3)` | added 2026-09-22, `plotting/mtexfigure.py`: `legend(location=..., numColumns=...)` gathers every artist a `DisplayName` named over all the charts of the figure, MATLAB's compass locations and the outside ones |
| OrientationStandard | `ind2color` reads MTEX's own 24 colours | added 2026-09-22: the table of `plotting/plotting_tools/colors.mat`, where the port took matplotlib's ten |
| OrientationStandard | `DisplayName` on markers and on a scatter in orientation space, `add2all` on a section plot, `'filled'` | added 2026-09-22 |

## Conventions that differ

| Page | MATLAB | Python |
|---|---|---|
| VectorDefinition | `vector3d(M)` with a 3 x 3 matrix warns and reads by columns | reads by rows, one vector per row of three, as `byXYZ` and the `(..., 3)` layout do; no warning |
| VectorDefinition | `size(v)` is `1 5` for five vectors, `v(2)` the second | `v.shape` is `(5,)`, `v[1]` the second, zero based |
| VectorDefinition | the grid of a spherical plot follows the frame's Z axis, so under `'z←↑y'` the meridians run between the poles at the rim | the grid is drawn about the screen centre whatever the layout |
| VectorDefinition | `plottingConvention.default('y↑→x')` sets the session default | the same call; `plottingConvention.default` read is the convention, called it sets it |
| VectorsExport | `v.rho` remembers the azimuth a vector was constructed with, so `export(v, f, 'polar')` writes -80 | the azimuth is derived from the components, in [0, 360), so 280 is written; the direction is the same |
| VectorsImport | `[v, S] = vector3d.load(...)` gives the other columns on request | `vector3d.load` returns the directions; `loadVector3d` returns `(v, S)` with the other columns as a dict |
| VectorGrids | `norm(sF)` integrates against the surface measure, the constant one having norm sqrt(4 pi) | against the normalised measure, as the Julia port has it, the constant one having norm one; MATLAB's deviations are sqrt(4 pi) times the port's |
| VectorsDensityEstimation | `ebsd('Fo')` selects a phase by an abbreviation of its mineral | `ebsd['Fo']`, the same abbreviation, now accepted when it begins one mineral only |
| VectorsDensityEstimation | `[grains, ebsd] = calcGrains(...)` | `grains = calcGrains(...)`, the map with grain ids is `grains.ebsd`-less: the port writes the ids into the map it was given |
| VectorsDensityEstimation | `max(sF)` with one output is the value | `max(sF)` is the pair `(value, position)`, as `find` returns MATLAB's pair; `min(pdf2)[0]` for the value |
| EBSDGrid | an empty site of a gridded subset has phaseId NaN | phaseId 0 with a NaN rotation and an id beyond the parent map's |
| EBSDGrid | the eclogite file warns that 92 of 613 measurements share a lattice cell | warns that the measurements span 15040 cells for 617 of them; both keep the list |
| EBSDFilling | `fill` leaves the measured properties NaN at a filled site | the floating point properties are solved on the same graph, so a filled site carries interpolated properties; the ids tell the measured sites apart |
| EBSDFilling | `length(ebsd)` is the number of measurements | `len(ebsd)` is the first dimension of a gridded map; `ebsd.size` counts the measurements |
| EBSDFilling | a gridded hexagonal map counts the measurements of the file | the sites of the lattice, the 135 empty ones of the ferrite file among the notIndexed |
| EBSDAdvancedMaps | `mean(ori, 'robust')` of the forsterite orientations puts 12.1 percent of them within 20 degrees | 13.8 percent: the robust means differ, the port trimming at 2.5 times the 80th percentile |
| EBSDSharpPlot | the calcite scan is padded onto an axis-aligned 301 by 151 grid | the same 301 by 151 sites on the square lattice the positions form, turned by 45 degrees, its extent the turned box |
| EBSDSharpPlot | `h.rho` in (-180, 180] degrees | in [0, 360); the page maps it back for MATLAB's numbers |
| EBSDExport | the `.ctf` and `.ang` readers keep each file's own crystal axis alignment, so a converted map comes back 30 degrees off | the readers align the crystal frame with the file, the round trip closes at 0.0006 degree |
| EBSDIPFMap | `'MarkerSize', 0.05*area` scales the marker area | a marker size is a diameter in points, `s = size**2`, so a page takes the square root of MATLAB's number |
| EBSDIPFMap | `ebsd('olivine').CS = cs` assigns the frame of the variable | `ebsd['olivine'].CS = cs` replaces it in the phase list every map of the session shares, as with the colour |
| EBSDDenoising | `grains(ebsd.grainId)` indexes the grains by id, ids being positions | `grains.selectByGrainId(ebsd.grainId)`, the ids counting from one and the positions from zero |
| EBSDDenoising | `splineFilter` selects its parameter by cross validation, 4.61 on the twins map, and `[ebsdS, F] = smooth(ebsd, F)` returns it; `infimalConvolutionFilter` has `lambda` and `mu` | every variational filter takes a smoothing length, four steps by default, as the Julia port has it; `alpha` and `beta` derive from it; the mean deviations after smoothing differ accordingly (0.63 for 0.58) |
| EBSDDenoising | `axisAngleColorKey` without an argument | `axisAngleColorKey(ebsd)` names the phase |
| EBSDIndex | a crop of a gridded map is ordered column-major, `ebsd(16:18)` one-based | row-major and zero-based, `ebsd[15:18]`; `ebsd.id` is the zero-based position in the parent map |
| EBSDPlotting | a colour assigned to a phase belongs to the EBSD variable | frames are interned handles, so `ebsd['Diopside'].color = ...` colours Diopside for every map of the session; the builder runs every page in its own process for that reason |
| EBSDSelect | the default colour map of a value map is parula | viridis |
| EBSDImport | `'dataSet', 1` names the first data set of an HDF5 file | `dataSet=0`, counted from zero as the port counts |
| EBSDImport | `ebsd.opt.header`, `ebsd.opt.dataSet` | `ebsd.meta['header']`, `ebsd.meta['dataSet']`; the properties are lower case, `iq`, `metric` |
| EBSDImport | the import wizard | not ported; the page keeps the section and says so |
| EBSDPlotting | `colormap gray`, `legend off` | `mtexColorMap('gray')`, `legend=False` on the plot |
| VectorsDensityEstimation | `volume(sF, v, 20*degree)` is 0.1723 on the 0.1 degree grid about the centre | 0.1721 on a local grid of the same resolution; the two grids differ |
| GrainSpatialPlots | the aspect ratio and the long axis are those of the ellipse with the same second area moments, MATLAB's fit to the boundary vertices differs on ribbon grains (17.8 for 13.3 at the extreme) | design of P3, see docs/plans/maps.md section 9 |
| GrainReconstruction | `calcGrains` writes the grain ids into the map it is given and rewrites the phase of the measurements `minPixel` culls; a sweep over `minPixel` passes `ebsd.copy()`; the pixels the closing absorbs keep their phase where MATLAB's returned map gives them the host phase | design of P3 |
| GrainNeighbours | `numNeighbors` is a property, the shape parameters MATLAB implements as methods (`aspectRatio`, `shapeFactor`, `paris`, `caliper`, `fitEllipse`, `longAxis`) are methods with parentheses; `area`, `perimeter`, `diameter`, `equivalentRadius` properties | |
| GrainMerge | `merge` returns the merged grains and writes `parentId` into the grains it was given; `accumarray` over parent ids is `np.bincount` with weights; a grain id written into a subset of a map goes through a mask on the whole map | |
| GrainOrientationParameters | `fibre.fit(ori, 'local', diagnostics=True)` returns MATLAB's triple, the eigenvalues `lam[0]` to `lam[3]`; MATLAB's `noSym2` reads as `noSymmetry` where it applies | |
| RotationDefinition | `Euler(rot, 'Roe')` without an output prints a table in degrees | returns the three angles in radians as a tuple; the page shows `np.array(Euler(rot, 'Roe')) / degree` |
| RotationOperations | `rot \ v` applies the inverse rotation; `.*` pairs entries and `*` forms every combination | `inv(rot) * v`; `*` broadcasts as NumPy does, a column of rotations times a row of directions is the outer product |
| RotationImproper | `rotation(cs)` lists the operations of a mixed group with their inversion flags, `isImproper(rot)` is a method | `orthogonalTransform(cs)`; `rotation(cs)` only for a proper group; `.isImproper` is a property of `orthogonalTransform`, a `rotation` has none (semantics S16), a `cat` mixing both is an `orthogonalTransform` |
| RotationSpinTensor | `log(r, ref, SO3TangentSpace.rightSpinTensor)` names four representations | the side is `'left'` or `'right'`, `log` returns the vector, `spinTensor(t)` its matrix, `vector3d(S)` the axial vector; `spinTensor` at the package level is the tensor class of the tensors layer |
| RotationRepresentations | there is no `rotation.byCubochoric`; the page goes through `cubo2homo` | `rotation.byCubochoric` exists; the page keeps MATLAB's route |
| ODFTheory | the display of an ODF names the specimen frame by its convention, `y↑→x (mmm)` | `specimen (mmm)`, the frame's name with its group |
| ODFModeling | `max(odf1)` with one output is the value | `max(odf1)[0]`, the pair being returned, as `min` and `max` of every function |
| ODFPoleFigure | `Miller({1,0,-1,0},{0,0,0,1},{1,1,-2,1},cs)`, one cell per direction | `Miller([[1, 0, -1, 0], [0, 0, 0, 1], [1, 1, -2, 1]], cs)`, one row per direction |
| EulerAngleSections | `odfOrtho = odf; odfOrtho.SS = ...` leaves `odf` alone | both names hold the same object, so the assignment changes both |
| EulerAngleSections | `fibre.beta(cs)` has `h || r` = (12 6 11) || (-1,-1,4) | (12 6 11) || (1,1,4): `byMiller` picks another of the equivalent directions than MATLAB's symmetrise order, a half turn about Z apart; the port does not index group elements by position |
| ODFComponents | `[ori, w] = calcComponents(odf)` takes two of three outputs | `ori, w, _ = calcComponents(odf)`: the triple is always returned |
| FibreODFs | `[f, lambda, d] = fibre.fit(ori, 'local')` | `fibre.fit(ori, 'local', diagnostics=True)` returns the triple, without the flag the fibre alone |
| ODFShapes | the bump kernel stores 1024 Chebyshev coefficients | 33, the bandwidth its halfwidth needs |
| ODFImport | `'interp'` fits a radial function and warns about the iteration limit | `SO3Fun.interpolate` fits a harmonic series by a regularised least squares, so the imported function is an `SO3FunHarmonic` and its texture index differs |
| ODFExport | `save`, `load` of a `.mat` working copy | `pickle`; a pickled frame comes back as the session's handle |
| DetectionOfSampleSymmetry | `centerSpecimen` prints its search and takes `'Fourier'` | the search is silent and always harmonic; the four outputs are always returned |
| CrystalSymmetries | `rotation(cs)` lists the operations of any group, `cs.Laue`, `cs.properGroup` are properties | `orthogonalTransform(cs)` for a mixed group, `rotation(cs)` for a proper one; `Laue()`, `properGroup()`, `properSubGroup()`, `fundamentalSector()` with parentheses |
| CrystalSymmetries | the operation tables list the half turns as (0, 180, 0) | the same rotations as (45, 180, 45), another Euler representative of a degenerate triplet |
| CrystalOperations | `vector3d(m)` drops the crystal frame | keeps it; `vector3d(m.x, m.y, m.z)` is the frame-free vector; `polar(v)` without an output prints degrees where `v.theta`, `v.rho` are radians |
| CrystalShapes | `cS.V` prints Cartesian components; `crystalShape.olivine` is a property; `{1,0,-1,0}` is an index cell | `cS.V` is a `vector3d` in the crystal frame and prints Miller indices; `crystalShape.olivine()`; `[1, 0, -1, 0]` |
| QuasiCrystals | a group built from elements reports `unknown` | the type by its signature, `532`, `-5-32/m` |
| PoleFigureCorrection | after `correct` the two Geesthacht scans carry different antipodal flags, so their quadrature weights differ although their specimen directions are identical; `normalize` then gives 0.03 to 2.15 mrd | both scans are axes, as diffraction data are, and are weighted the same way: 0.03 to 2.27 mrd, and the ODF based scale agrees with the direct one to 0.02 percent rather than 0.33 |
| PoleFigureRefinement | `refine` closes the hull with the south pole and keeps the centroids of the skirt triangles, so the adaptive loop measures directions below the 80 degree cap and ends at 367 accumulated directions | the centroids stay inside the cap, 327 directions, and the loop's interim errors differ; the port's direct reconstruction ends at 0.22 against MATLAB's 0.40, the iterative at 0.12 against 0.117 |
| PoleFigureRefinement | a pole figure remembers the spacing its specimen grid was built with, so the first round reports 14.5 degrees | the resolution is derived from the points, 15.3 degrees |
| PoleFigure2ODF | the solver's grid holds 19848 orientations at 5 degrees on the Dubna set | 20224; the equispaced SO(3) grids differ in count, the reconstructions agree to the digit |
| OrientationDefinition | `ori.symmetrise` lists the 48 x 8 products of the full point groups, `'proper'` cuts them to 24 x 4 | lists the proper equivalents only, 24 and 96, as semantics S22 has it; an orientation is a proper rotation, so a Laue group would give each equivalent twice |
| OrientationDefinition | the display names the specimen frame by its plotting convention, `orientation (Copper → y↑→x)` | names it `specimen`, the convention being a property of the plot, not of the frame's identity |
| OrientationSymmetry | the axis at the centre of a chart is labelled, `Z` in the middle of an upper hemisphere | left unlabelled, since a label there would cover the data; the axes at the rim are labelled as MATLAB's |
| OrientationStandard | the contour levels of a pole figure row are taken per panel, so every panel reaches the top of the colour map | one range over the panels of one call, as the surfaces share it |
| OrientationStandard | a plot onto held axes takes the next colour of the axes' colour order | the marker colour stays the default; a page that wants MATLAB's colours passes `MarkerColor=ind2color(i)` |

## Practicalities

| Page | Note |
|---|---|
| VectorDefinition | a MATLAB cell shows `v = vector3d (...)` after `v = vector3d(1,2,3)` without a semicolon; the Python page writes `v` on the line after the assignment, as a notebook does |
| VectorDefinition | the builder displays a trailing number as MATLAB's format short (`3.7417`, `45.0000`), an integer-valued one without decimals, and drops the handles a plot returns |
| VectorDefinition | the random directions of `vector3d.rand` differ from MATLAB's draw; only the layout of the figure is compared |
| VectorsExport | a MATLAB cell of several statements is one cell in the page too: the statements follow without a blank line, or sit between `# +` and `# -` |
| EBSDIndex | a figure lives on across the cells of a page, and `hold` in a later cell draws into it; the builders snapshot a figure after the cell that made or changed it, as MATLAB's publish does |
| EBSDOrientationAnalysis | `islocalmax(x, 'MinProminence', 1)` is `scipy.signal.find_peaks(x, prominence=1)`; `orientation(f, CS, SS)` along a fibre is `f.orientation()` |
| EBSDIndex | `hex2cube`, `cube2hex` are not ported; `rectangle` and `histogram` are matplotlib's `Rectangle` and `hist` |
| VectorsDensityEstimation | the colour bar of a `contourf` plot is ticked at the contour levels with three decimals where MATLAB's shows round ticks |
| VectorsDensityEstimation | `xnum2str` and `mtexdegchar` are an f-string; `[grains, ebsd] = calcGrains` is `grains = calcGrains`; `newMtexFigure('layout', [1 4])` is `newMtexFigure(layout=[1, 4])`; `plt.xlim` and `plt.legend` stand in for MATLAB's `xlim` and `legend` |
| Grains chapter | MATLAB's `table` displays are printed lines, `histogram`/`polarplot`/`scatter`/`subplot` are matplotlib's, `xlim` and `legend` are `plt.xlim` and `plt.legend`, `colorcet('C2')` is `twilight` |
| RotationRepresentations | a figure built over several paragraphs (a `plt.figure`, then its subplots) is one cell only between `# +` and `# -`; otherwise the builder saves and closes it after the first paragraph |
| RotationPlotting | `docs/matlabrun.py` saves a persisting MATLAB figure again after every cell that redraws it, so a page's MATLAB folder holds several copies of one figure; the montage shows them all |
| RotationPlotting | the count of the random subset is a draw: 26 here, 32 in the MATLAB run, 20 on the MATLAB page |
| ODFTheory | MATLAB's `mtexdata titanium silent` is `ebsd = mtexdata('titanium')`, which prints nothing unless displayed |
| SigmaSections | MATLAB's local function at the end of the script is defined at the top of the page, before its first call |
| ODFShapes | MATLAB's `figure('position', [...])` is `plt.figure(figsize=..., layout='constrained')`, so an outside legend gets its room; `table(...)` is a printed loop |
| RandomSampling | `fminbnd(err, 2.5, 80, optimset('TolX', 0.1))` is `minimize_scalar(err, bounds=(2.5, 80), method='bounded', options={'xatol': 0.1})`; the page builds in a few minutes |
| ODFComponents | the page's `calcODF` of the Dubna pole figures takes about 30 s, its `max`, `volume` and `calcComponents` a few seconds each through the harmonic form |
| CrystalOperations | MATLAB's `quartzPattern.mat` is a v7.3 object file of a class the frameSymmetry branch no longer has, so its MATLAB run fails on that branch; the port's `mtexdata('quartzPattern')` reads the one large array of the file with h5py as the harmonic coefficients in the quartz frame |
| SymmetryAlignment | the tensor file of MATLAB's data folder is the data set `olivine1997` here, and Bruker's `crystal.phl` ships beside the CIF files |
| PoleFigureSimulation, PoleFigureSantaFe, PoleFigureCorrection | the Poisson draw of `noisepf` and the spoiled measurements of the outlier example are not seeded, as MATLAB's are not, so their numbers move between builds; the Correction page seeds its own generator, and MATLAB's own run differs from the numbers its page quotes |
| PoleFigure2ODF | `calcODF` is silent unless `silent=False`, where MATLAB prints its iteration trace unless `'silent'` is given |
