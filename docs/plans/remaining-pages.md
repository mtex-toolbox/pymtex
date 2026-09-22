# Documentation pages

Snapshot: 2026-09-22. One row per content script one chapter below `../frameSymmetry/doc`,
ordered by chapter; the chapter navigation scripts are site work. A page is checked once its
Python cells execute and its numbers and figures match the MATLAB run of the same script
through the bridge; the findings go to `docs/findings.md`. The last column is the status the
Julia port reached on the same page, a hint of the feature a page needs, not a reference.

Status words: **Checked** compared with the MATLAB run;
**Adapted** prose-only or process pages rewritten for Python; **Partial** checked with a
named feature missing; **Written** built but not compared; **Blocked** needs a feature that
does not exist; **Pending** not started.

| Chapter | Total | Checked | Adapted | Partial | Written | Blocked | Pending |
|---|---|---|---|---|---|---|---|
| CrystalGeometry | 10 | 9 | 0 | 0 | 1 | 0 | 0 |
| CrystalOrientations | 16 | 16 | 0 | 0 | 0 | 0 | 0 |
| EBSD3Analysis | 4 | 0 | 0 | 0 | 0 | 0 | 4 |
| EBSDAnalysis | 25 | 13 | 1 | 4 | 3 | 4 | 0 |
| Elasticity | 4 | 0 | 0 | 0 | 0 | 0 | 4 |
| GeneralConcepts | 12 | 0 | 0 | 0 | 0 | 0 | 12 |
| GrainBoundaries | 15 | 0 | 0 | 0 | 0 | 0 | 15 |
| Grains | 15 | 11 | 1 | 1 | 0 | 0 | 2 |
| Misorientations | 6 | 0 | 0 | 0 | 0 | 0 | 6 |
| ODFAnalysis | 18 | 15 | 0 | 3 | 0 | 0 | 0 |
| PhaseTransitions | 11 | 0 | 0 | 0 | 0 | 0 | 11 |
| Plasticity | 14 | 0 | 0 | 0 | 0 | 0 | 14 |
| Plotting | 11 | 0 | 0 | 0 | 0 | 0 | 11 |
| PoleFigureAnalysis | 11 | 11 | 0 | 0 | 0 | 0 | 0 |
| Rotations | 8 | 8 | 0 | 0 | 0 | 0 | 0 |
| SO3Functions | 13 | 0 | 0 | 0 | 0 | 0 | 13 |
| SphericalFunctions | 16 | 0 | 0 | 0 | 0 | 0 | 16 |
| Tensors | 8 | 0 | 0 | 0 | 0 | 0 | 8 |
| Tutorials | 5 | 0 | 0 | 0 | 0 | 0 | 5 |
| Vectors | 7 | 5 | 0 | 2 | 0 | 0 | 0 |
| **All** | 229 | 88 | 2 | 10 | 4 | 4 | 121 |

| MATLAB source | Page | Status | Julia (hint) |
|---|---|---|---|
| CrystalGeometry/CrystalDirections | CrystalDirections | Checked 2026-09-22; 98.9, 31.6762, [541] and every display to the digit, four figures as MATLAB's; wrote `labeled`, `backgroundColor`, `textAboveMarker`, `plane=True`, a held plot going to both hemisphere charts; found `char` writing family brackets on a single direction | Checked |
| CrystalGeometry/CrystalOperations | CrystalOperations | Written 2026-09-22; 6 and 3, [6 12 8], the logicals, the zone law, 60, 0 and 180 as on the page; the quartz Kikuchi pattern is `mtexdata('quartzPattern')`, read from MATLAB's v7.3 object file with h5py, which the frameSymmetry MATLAB cannot load, so the nine figures have no MATLAB run; wrote `circle` with `parent`, `color` and lists, `eq(antipodal=True)`, `ori * S2Fun`, MATLAB's lower-case marker keys; `vector3d(m)` keeps the frame, the page builds the frame-free vector from the components | Checked |
| CrystalGeometry/CrystalReferenceSystem | CrystalReferenceSystem | Checked 2026-09-22; 90, 30 with MATLAB's message, (270, 0, 0), the albite displays; six figures as MATLAB's; wrote `plot(cs)` for the symmetry elements, `arrow3d(labeled=True)`, `innerPlotSpacing`, `plotPDF` into a prepared figure; found the angle between orientations in differently aligned frames silently unreconciled and the poles on the equator dropped by the upper chart, both fixed | Checked |
| CrystalGeometry/CrystalShapeSmorf | CrystalShapeSmorf | Checked 2026-09-22; the table 0.13897, 0.019766, 0.1384, 0.1346, 0.033199, 0.043238 and the form areas to the digit, the shape as MATLAB's | Checked |
| CrystalGeometry/CrystalShapes | CrystalShapes | Checked 2026-09-22; 12 vertices and 8 faces, (8, 12), the slip systems, the twin and the habit sequence as MATLAB's, seventeen figures; `cS.V` displays Miller indices where MATLAB prints components; wrote MATLAB's `FaceAlpha`, `FaceColor`, `LineWidth` on shapes and grain maps, `slipSystem.cat` with an axis | Checked |
| CrystalGeometry/CrystalSymmetries | CrystalSymmetries | Checked 2026-09-22; the logical, 48, 8, 4, 24, 4, quartz and magnetite as MATLAB's, the operation tables up to the Euler representatives of the half turns; six figures; wrote `plot(cs)`, `orthogonalTransform(cs)`, the Bruker `.phl` reader; found the proper subgroup of -4m2 named unknown, fixed by its signature | Checked |
| CrystalGeometry/FundamentalSector | FundamentalSector | Checked 2026-09-22; three normals, False, (213), the gallery of ten Laue groups as MATLAB's; wrote `crystalFrame(LaueId=k)`, `lineWidth` of a region outline; found the outline drawn under the mirror circles, fixed | Checked |
| CrystalGeometry/LatticeMetric | LatticeMetric | Checked 2026-09-22; 223.5856, the identity, 7.5563, 2, 4.9986, 3.6, 2.5456, 2.0785 and the metric matrix to the digit; wrote `cs.axesDual`, `det(v)` of three vectors | Checked |
| CrystalGeometry/QuasiCrystals | QuasiCrystals | Checked 2026-09-22; 60 and 120 elements, the sector, the dodecahedron and the two keys as MATLAB's; the port names the groups 532 and -5-32/m where MATLAB says unknown; wrote `plot(key, complete=True)`; found the sector of the centrosymmetric icosahedral group empty, its centre now the Moebius triangle of neighbouring axes | Checked |
| CrystalGeometry/SymmetryAlignment | SymmetryAlignment | Checked 2026-09-22; 10 and 0, the two Voigt matrices and 320.5 twice as MATLAB's, two figures; the tensor file is the data set `olivine1997` | Checked |
| CrystalOrientations/DefinitionAsCoordinateTransform | DefinitionAsCoordinateTransform | Checked 2026-09-22; every display, the transformed stiffness, 37.1140 and the residuals (0 and 2.2e-15 for MATLAB's 1.2e-06 and 3.4e-06) as MATLAB's, the cube with its arrows as MATLAB's up to the camera elevation; `symmetrise` gives 24 where MATLAB gives 48 and 24, the proper-only class again; found the Miller and tensor displays counting significant digits where MATLAB rounds to four decimals, fixed | Checked |
| CrystalOrientations/MTEXvsBungeConvention | MTEXvsBungeConvention | Checked 2026-09-22; the vectors, 0, (10, 50, 20), (160, 50, 170), the matrix and 0 as MATLAB's (0 for MATLAB's single precision 1.7e-06); the two cubes as MATLAB's but the arrow shows through the cube, matplotlib having no occlusion; wrote `byEuler` of an array of triples, 3D shapes into a `newMtexFigure` layout with `nextAxis`, MATLAB's `text(x, y, z, s)` | Checked |
| CrystalOrientations/OrientationDefinition | OrientationDefinition | Checked 2026-09-22; the three displays, 100 and the Goss angle (2.7e-15 for MATLAB's exact 0) as MATLAB's, one figure as MATLAB's; `symmetrise` gives 24 and 96 where MATLAB gives 48 and 384, the semantics' proper-only class, said in the page; dropped the dead `proper` keyword of `symmetrise` | Checked |
| CrystalOrientations/OrientationEmbeddings | OrientationEmbeddings | Checked 2026-09-22; the naive and intrinsic means, the embedding display, the 315 degree mean, the ones, 81 and 9 as MATLAB's; MATLAB's run fails in the de la Vallee Poussin loop asking for zero samples, the port draws both families; wrote `std`, `norm(e, normalized=True)`, `double(e, full=True)`, `orientation(e)`, `BinghamODF(kappa, cs)`, MATLAB's embedding display, an exact Bingham sampler (the page went from six minutes to eight seconds) | Checked |
| CrystalOrientations/OrientationExport | OrientationExport | Checked 2026-09-22; the headers and layouts of the four tables and both VPSC files as MATLAB's, the weight sum 1; the values are a random sample; the table writer now fits each column as MATLAB's cprintf, shared with the vector export | Checked |
| CrystalOrientations/OrientationFibre | OrientationFibre | Checked 2026-09-22; the four fibre displays, 6 copies, the volumes 7.39 and 26.32 for MATLAB's 7.26 and 25.93 (the page's 58 and 100 percent are stale in MATLAB too), all figures as MATLAB's; wrote `plot3d(odf)` as MATLAB's isosurfaces in the Euler box, the fibre in inverse pole figures, `DisplayName` on fibres, `project2FundamentalRegion` on fibre plots, MATLAB's `Location` of a legend, the Euler box ticked as MATLAB's; the port draws 96 copies of the symmetrised beta fibre where MATLAB's figure shows fewer | Checked |
| CrystalOrientations/OrientationFundamentalRegion | OrientationFundamentalRegion | Checked 2026-09-22; 6.0553 as MATLAB's, all figures as MATLAB's after the fixes; the boundary counts 11714 and 16524 for 11751 and 15872 from the default reconstruction (3185 grains for 3106, the closing of GrainReconstruction); found the grain exchange flag ignored by the region, the projection and the axis angle sections, orientations drawn into a held axis angle plot in Euler coordinates, misorientations scattered in Euler space, `all` and `ignoreFundamentalRegion` without effect, section angles after a family read as marker values, all fixed; the port's antipodal half is a different, equally valid half of MATLAB's | Checked |
| CrystalOrientations/OrientationGrid | OrientationGrid | Checked 2026-09-22; 24624, 265 and 8.75 as MATLAB's, the regular grid's pole figure ranges 0.79 to 2.62 and 0.75 to the digit; the equispaced grid has 5177 nodes for MATLAB's 4923 and is flatter, 0.98 to 1.02 for 0.93 to 1.02; wrote `regularSO3Grid` | Checked |
| CrystalOrientations/OrientationImport | OrientationImport | Checked 2026-09-22; 382 orientations and the two pole figures as MATLAB's; wrote `orientation.load` and `loadOrientation` (Euler or quaternion columns, radians, passive, positions) and the data set `tongue`; MATLAB's page warns of three defects of its importer that the port does not have, said in the page | Checked |
| CrystalOrientations/OrientationInversePoleFigure | OrientationInversePoleFigure | Checked 2026-09-22; 6 and 300 as MATLAB's, the figures as MATLAB's in structure, the random orientation differing; wrote `rotation.rand(n, maxAngle=w)`; found the 321 sector bounded by the a directions where MATLAB bounds it by the m directions, fixed in the charts | Checked |
| CrystalOrientations/OrientationPoleFigure | OrientationPoleFigure | Checked 2026-09-22; 6 and all figures as MATLAB's in structure (a random orientation); the port lays the four panels out 2x2 and shares one colour bar | Checked |
| CrystalOrientations/OrientationStandard | OrientationStandard | Checked 2026-09-22; Copper at 5.0000, the nine $(hkl)[uvw]$ labels and every marker position as MATLAB's, eight figures; wrote `legend`, MTEX's colour table behind `ind2color`, `DisplayName` on markers and in space, `add2all` on section plots; found `round2Miller` rounding over four indices where MATLAB rounds over twelve, the orthorhombic sector not turned by the plotting convention, a contour overlay replacing the surface under it and a held row of pole figures opening a second figure, all fixed; the contour levels are shared over the panels of one call where MATLAB gives each its own, and a held plot does not take the next colour of the order | Checked |
| CrystalOrientations/OrientationSymmetry | OrientationSymmetry | Checked 2026-09-22; the three equivalent lists, 12, [12, 6] and both angle columns to the digit as MATLAB's; the complete pole figure as MATLAB's but for the Z at the chart centre; wrote `ori * cs` and `ss * ori`; found `complete` never reaching the pole figure chart, the specimen group missing from the poles and the sample budget ignoring it, all fixed; `symmetrise` gives 24 for cubic where MATLAB gives 48 and 24 | Checked |
| CrystalOrientations/OrientationVisualization3d | OrientationVisualization3d | Checked 2026-09-22; the four figures as MATLAB's; the port's markers are filled where MATLAB draws hollow circles | Checked |
| CrystalOrientations/OrientationVisualizationSections | OrientationVisualizationSections | Checked 2026-09-22; the phi2, sigma and axis angle sections as MATLAB's; found the largest angle of a region sampled (61.6 for 62.8 degrees), now exact at a vertex in Rodrigues space, so the sections sit at 5.233 to 57.566 degrees and are labelled as MATLAB's | Checked |
| CrystalOrientations/SpecimenSymmetry | SpecimenSymmetry | Checked 2026-09-22; 8, 4, 1, 2, the frames and both pole figures with their ranges 6.9 and 3.4 as MATLAB's; the texture indices 27.89 and 7.27 have no MATLAB value, its NFFT MEX failing to load | Checked |
| EBSD3Analysis/Grains3D |  | Pending | Blocked |
| EBSD3Analysis/Grains3DOperations |  | Pending | Blocked |
| EBSD3Analysis/Grains3DProperties |  | Pending | Blocked |
| EBSD3Analysis/NeperInterface |  | Pending | Blocked |
| EBSDAnalysis/EBSD2ODF | EBSD2ODF | Checked 2026-09-22; the peaks 3.727, 37.26, 1.476, the halfwidths 2.7377 and 4.7247, the distances 0.160 and 0.012 as MATLAB's, the grain-based numbers within the few grains the reconstructions differ by (374 for 368), all eight figures; MATLAB's tables are printed rows Partial |
| EBSDAnalysis/EBSDAdvancedMaps | EBSDAdvancedMaps | Checked 2026-09-22; the Bunge key and its seams, the spot and fibre keys, the six lobes and the two layered maps as MATLAB's, the fibre count 24.8 to the digit; the spot count 13.8 for 12.1 because the robust means differ; needed `plot(key)` in sections for the Bunge and spot keys, a spot key constructed from a phase, its centre, colour and kernel settable, `MarkerAlpha`; `PatalaColorKey` is not ported Checked |
| EBSDAnalysis/EBSDDenoising | EBSDDenoising | Checked 2026-09-22; the raw deviation 0.70 as MATLAB's, all nine figures; the variational filters take a smoothing length where MATLAB's spline selects its parameter, so the total variation and spline deviations are 0.63 and 0.64 for MATLAB's 0.58 and 0.59, said in the page; `grains.selectByGrainId(ebsd.grainId)` for MATLAB's `grains(ebsd.grainId)` Checked |
| EBSDAnalysis/EBSDExport | EBSDExport | Checked 2026-09-22; the two inventories as MATLAB's; the round trip closes at 0.0006 degree where MATLAB's page reports the 30 degree crystal frame mismatch of its two readers, said in the page; the binary pair and the MAT-file are not ported Checked |
| EBSDAnalysis/EBSDFilling | EBSDFilling | Partial 2026-09-22; the 29.0 percent, the ten figures in their sequence; `fill(ebsd, grains)` constrained by the grain polygons is not ported, so the sparse ferrite map is filled with `extrapolate=True` and its corridors with it, and the forsterite interiors are filled only where the hole does not reach the rim; the properties are interpolated too where MATLAB leaves them NaN, `quality` is not kept Checked |
| EBSDAnalysis/EBSDGROD | EBSDGROD | Checked 2026-09-22; the 377 grains, the GOS maximum 6.6, the crystal axis density 0.86 to 1.21 for 0.83 to 1.21 and the specimen one 0.29 to 6.35 for 0.24 to 6.76, all nine figures as MATLAB's; the GROD median 1.69 for 1.55 by the filter's default length; needed `HSVDirectionKey()` of the whole sphere and `antipodal=True`, a transparency per pixel and per boundary segment, `plot(key)` for direction keys Checked |
| EBSDAnalysis/EBSDGrid | EBSDGrid | Partial 2026-09-22; the displays, the pixel lookup, the eclogite list with its warning, the gridded subset with its 46 empty sites, the gradient, the hexagonal map and the rotated grid as MATLAB's; the layouts, `oldId` and the second output of `gridify`, the resampling onto a square cell and the projective tilt (M10) are prose; found `orientations` dropping the empty sites Checked |
| EBSDAnalysis/EBSDIPFMap | EBSDIPFMap | Written 2026-09-22: the shapes on the map, the keys, the sized inverse pole figure and the two hemispheres of the trigonal TSL key build; the three dimensional key, the spheres on the grains and `colorPostRotation` are prose, the key plots ignore `complete`; the MATLAB run of the script exceeds five minutes and waits Checked |
| EBSDAnalysis/EBSDImport | EBSDImport | Checked 2026-09-22; the inventory display with the data set list, the phase map and the orientation map as MATLAB's; needed a phase list on a vendor import, the data set listing, `ebsd.meta` and the default phase colours in the display; the wizard section shows `importWizard()` and the offscreen grab of the window | Checked |
| EBSDAnalysis/EBSDIndex | EBSDIndex | Partial 2026-09-22; the 98 measurements, the 7 x 14 grid, the 'xy' lookup with its row display as MATLAB's; the list order is row-major and zero-based, said in the page; `hex2cube` waits, so the cube coordinates are prose; needed `text(ebsd, labels)` and figures kept across cells | Checked |
| EBSDAnalysis/EBSDInter | EBSDInter | Partial 2026-09-22; `interp` at a point and the `'xy'` lookup with the zero angle between them, the hexagonal block, as MATLAB's; the resampling onto another unit cell (M10) is prose; found the `'xy'` lookup and a flat mask failing on a gridded map Checked |
| EBSDAnalysis/EBSDInterfaceHDF5 | EBSDInterfaceHDF5 | Adapted 2026-09-22; the same six JSON configurations, the inventories of the EMSphInx file as MATLAB's from a full import, `headerOnly` and the debug trace not ported, `ebsd.opt` is `ebsd.meta` Adapted |
| EBSDAnalysis/EBSDKAM | EBSDKAM | Written 2026-09-22; the medians 0.60, 0.59 and 0.81 as on MATLAB's page, the denoised one 0.20 for 0.27 by the filter's default length, the neighbourhood diagrams built from scratch; the MATLAB run of the script exceeds five minutes and waits Checked |
| EBSDAnalysis/EBSDMapsAndImages |  | Blocked: the image container, the grid layouts and the trueEbsdWCCoSmall data set are not ported, as in Julia | Blocked |
| EBSDAnalysis/EBSDOrientationAnalysis | EBSDOrientationAnalysis | Checked 2026-09-22; the girdle axis, 61.7854, 27.9806, the fibre profile between 5.5218 and 19.6383 with its three peaks, and all seven figures as MATLAB's; needed `circle`, `nextAxis(k)`, MATLAB's marker names, `plot(odf, f)` along a fibre and `plot(odf, 'sigma')` on the RBF form | Checked |
| EBSDAnalysis/EBSDOrientationPlots | EBSDOrientationPlots | Checked 2026-09-22; the sample sizes 1250, 12500, 1000, 2000 as MATLAB's after the budget was set by numSym, all eight figures as MATLAB's; found `hold` ignored by the pole figure rows, the values dropped by `plotSection(ori, values, 'sigma')`, `scatter(ori)` in axis angle space where MATLAB takes the Euler region, `all=True` missing | Checked |
| EBSDAnalysis/EBSDPlotting | EBSDPlotting | Checked 2026-09-22; the Goldenrod triplet to the digit, all eight figures as MATLAB's; needed `ebsd['phase'].color` read and set, `str2rgb`, `plot(key)`, `faceAlpha`, the colour bar title and the phase legend by default | Checked |
| EBSDAnalysis/EBSDProfile | EBSDProfile | Checked 2026-09-22; the grain of spread 9.7, its 86 measurements, the two curves spike for spike and the three inverse pole figures as MATLAB's Checked |
| EBSDAnalysis/EBSDPseudoSymmetry |  | Blocked: `cleanUpPseudoSym` is M10 | Checked |
| EBSDAnalysis/EBSDReferenceFrame | EBSDReferenceFrame | Checked 2026-09-22; the Euler correction (315, 180, 45), the eight large grains, all seven figures as MATLAB's, the crystal shapes on the maps included; needed `ebsd.EulerCorrection`, `rotate(ebsd, r, 'keepEuler')` at the package level, the `location` of the scale bar and `ipfDirection` of a map | Checked |
| EBSDAnalysis/EBSDSelect | EBSDSelect | Checked 2026-09-22; 152345, 178403, 187467, 20301, 881, 4052 and 13359 to the digit, all seven figures and the histogram as MATLAB's; the region rectangle is a matplotlib patch | Checked |
| EBSDAnalysis/EBSDSharpPlot | EBSDSharpPlot | Checked 2026-09-22; the display (32 notIndexed, 20119 calcite, [0 → 200] x [-100 → 0]), the azimuth median -22.27, the 98 percent between -23.89 and -17.76 and the tail to -41.72 as MATLAB's, every map with its few purple outliers, the inverse pole figure, the keys and the forsterite grain figures as MATLAB's; the value maps in viridis for parula; found the padding counted and drawn as notIndexed (fixed, `isPadding`), the fundamental sector a Dirichlet domain other than the conventional one (fixed), the means failing on NaN sites, `maxAngle` of the HSV key without a setter, colours in the map's shape refused | Checked |
| EBSDAnalysis/EBSDSimulation |  | Blocked: `simulateEBSD` and its verbs are M10 | Checked |
| EBSDAnalysis/EBSDSpatialTransform |  | Blocked: the spatial transforms, affine and projective with their fits, are M10 | Adapted |
| EBSDAnalysis/EBSDTrueEbsd |  | Blocked: the TrueEBSD registration and the staged distortion models are not ported, as in Julia | Blocked |
| Elasticity/AnisotropicTheory |  | Pending | Checked |
| Elasticity/CPOSeismicProperties |  | Pending | Checked |
| Elasticity/IsotropicTheory |  | Pending | Checked |
| Elasticity/WaveVelocities |  | Pending | Checked |
| GeneralConcepts/ClusterDemo |  | Pending | Checked |
| GeneralConcepts/Contribute2Doc |  | Pending | Adapted |
| GeneralConcepts/DensityEstimation |  | Pending | Checked |
| GeneralConcepts/GeneralConceptsConfiguration |  | Pending | Checked |
| GeneralConcepts/GeneralConceptsOptions |  | Pending | Checked |
| GeneralConcepts/Glossary |  | Pending | Adapted |
| GeneralConcepts/ListsAndIndexing |  | Pending | Checked |
| GeneralConcepts/MTEXScripts |  | Pending | Checked |
| GeneralConcepts/NotationAndConventions |  | Pending | Checked |
| GeneralConcepts/OptimalKernel |  | Pending | Checked |
| GeneralConcepts/Properties |  | Pending | Checked |
| GeneralConcepts/changelog |  | Pending | Adapted |
| GrainBoundaries/BoundaryCurvature |  | Pending | Checked |
| GrainBoundaries/BoundaryIntersections |  | Pending | Checked |
| GrainBoundaries/BoundaryMisorientations |  | Pending | Checked |
| GrainBoundaries/BoundaryNormalDistribution |  | Pending | Checked |
| GrainBoundaries/BoundaryPlots |  | Pending | Checked |
| GrainBoundaries/BoundaryProperties |  | Pending | Checked |
| GrainBoundaries/BoundarySelect |  | Pending | Checked |
| GrainBoundaries/CSLBoundaries |  | Pending | Checked |
| GrainBoundaries/GrainSmoothing |  | Pending | Checked |
| GrainBoundaries/GrainSmoothingAdvanced |  | Pending | Checked |
| GrainBoundaries/QuadruplePoints |  | Pending | Checked |
| GrainBoundaries/SubGrainBoundaries |  | Pending | Checked |
| GrainBoundaries/TiltAndTwistBoundaries |  | Pending | Checked |
| GrainBoundaries/TriplePoints |  | Pending | Checked |
| GrainBoundaries/TwinningBoundaries |  | Pending | Checked |
| Grains/Dream3dGrains |  | Pending | Blocked |
| Grains/EllipseBasedParameters | EllipseBasedParameters | Checked 2026-09-22; ten figures as MATLAB's; the long axis peaks 72.3 and 89.5 degrees for MATLAB's 74.2 and 78.8, the caliper peaks 73.3 and 85.3 for 74.3 and 82.0, the boundary peak 82.9 at 0.94 of the grid axis for 75 at 0.97, the characteristic aspect ratios 1.344 and 1.343 for 1.329 and 1.339, the ellipse being the one of the second moments; wrote `histogram(v)` rose diagrams, `plot(S1Fun)`, `plotSection(S2Fun, sec)`, `plot(shape2d)`, `setView`, `plotEllipse`, `ebsd.how2plot`, the `periodic` flag | Checked |
| Grains/GrainExport | GrainExport | Checked 2026-09-22; the grain table, the orientation table with its extra columns, the VPSC file with its weights summing to one, the polygon file and the boundary table written and read back as MATLAB's, the first interior grain another one here; one figure; MATLAB's `table`, `writetable` and `save` are a dictionary of columns, the `csv` module and `pickle`; wrote `export(ori, fname)` and `export_VPSC` in `io/exportorientation.py` | Checked |
| Grains/GrainMerge | GrainMerge | Checked 2026-09-22; 1408 of 2698 candidate twin segments for MATLAB's 1406 of 2696, 91 grains merged into 28, 16.63 percent twin area for 16.64, the weighted GOS higher in 14 of 28 parents by 0.579 degrees for 0.581; merged grain 16 holds six children for MATLAB's nine, the merge ids differing; nine figures; wrote the function form `merge(grains, gB)` | Partial |
| Grains/GrainNeighbours | GrainNeighbours | Checked 2026-09-22; 251 pairs, the pair misorientation, the twin law at 86.2992 degrees and the twin summary (93 twin pairs, 37.1 percent, 83 and 10 in the peak bin) to the number, five figures as MATLAB's; `numNeighbors` is a property here, the table summary a printed line, the histogram `plt.hist` | Checked |
| Grains/GrainOrientationParameters | GrainOrientationParameters | Checked 2026-09-22; the fibre axes, the eigenvalues and the fit angle 1.2233 to the digit, 161 large grains with 103 accepted axes as MATLAB's, the correlations 0.923 and 0.637 for 0.922 and 0.608, the axis density maximum 4.8582 to the digit; thirteen figures; two unindexed grains that survive the closing are dropped; wrote `plotPDF(fibre, h)`, `plot(grains, v)` compass needles, `ebsd.project2FundamentalRegion()`, `fibre.fit(ori, 'local')`; found the fit angle NaN over absorbed pixels and `scatter(ori)` refusing NaN orientations, both fixed | Checked |
| Grains/GrainReconstruction | GrainReconstruction | Partial 2026-09-22; the displays, the threshold sweep (227 at 5 degrees for MATLAB's 228), the alpha and minPixel sweeps (317 grains at alpha = 1 for MATLAB's 239, the closing by distance transform absorbing less; 90 and 71 for 89 and 70), the largest grain (1624 pixels, the 51 absorbed ones with their own phase, for MATLAB's 1622 rewritten) and eight of the eleven figures as MATLAB's; the three FMC figures are prose, fast multiscale clustering being unported; found the map plot ignoring `newMtexFigure` and `nextAxis` and a NaN phase colour in the legend, both fixed | Checked |
| Grains/GrainReconstructionAdvanced | GrainReconstructionAdvanced | Checked 2026-09-22; the criterion object and the option agree, the phase-dependent thresholds give 63 forsterite grains as MATLAB's (from 56 here, 55 there), the soft criterion matches the hard 14 degree counts with 28 inner segments as MATLAB's, the c-axis grains 73 to 70 for 69 to 66, the largest unindexed region 355 pixels and 10.63 to the digit; two figures; the Delaunay route and MCL are prose; a custom criterion overrides `connectivity`; found the per-phase threshold list unrecognised, fixed | Checked |
| Grains/GrainReconstructionMCL |  | Pending | Blocked |
| Grains/GrainSpatialPlots | GrainSpatialPlots | Checked 2026-09-22; fourteen figures as MATLAB's, the eight clipped grains and the largest grain's departure 6.0553 to the digit; the aspect ratio of the extreme grain is 17.8 for MATLAB's 13.3 and the band fraction 0.499 for 0.513, the ellipse being the one of the second moments (a P3 decision); `twilight` stands in for `colorcet('C2')`; wrote `planarColorKey`, `grains.N`; found a `region` not put through the layout and `text(grains)` failing on the centroid, both fixed | Checked |
| Grains/Grain_dispersion_axes | Grain_dispersion_axes | Checked 2026-09-22; the grid axis 6.3 degrees from the fitted one, the fibre distances 1.41, 2.51 and 4.45 to the digit, 262 large grains with 147 accepted for MATLAB's 249 and 141, the specimen and crystal density maxima 2.22 and 2.83 for 1.96 and 2.92 near [010]; ten figures; wrote `annotate(v, plane=True)`, `plot(fibre)` in Euler space, the outer product `ori.reshape(-1, 1) * s2G` | Checked |
| Grains/HullBasedParameters | HullBasedParameters | Adapted 2026-09-22; the port draws its own fifteen test shapes where MATLAB loads forty from a file, so the ids and the table differ while the four measures rank the discs with holes, the clover and the convex shapes as the page says; three figures; found `perimeter` counting the loops around inclusions where MATLAB's counts the outer loop, fixed | Checked |
| Grains/ProjectionBasedParameters | ProjectionBasedParameters | Checked 2026-09-22; the grain of the largest diameter is MATLAB's grain 515 (6925.8 and 1631.7 to the digit), the PAROR minima 0.7118 and 0.7241 for 0.7118 and 0.7214, the sampled directions within a degree, SURFOR within 0.002 and a degree, the characteristic caliper angles as MATLAB's; nine figures; the rectangles come from the port's own test shapes; wrote `caliper(dirs)` over a list of directions, `S1FunHarmonic.interpolate`, `quiver(noScaling=True)` | Checked |
| Grains/SelectingGrains | SelectingGrains | Checked 2026-09-22; fourteen figures as MATLAB's; the grain at (9000, 3500) with its 323 pixels, the three similar grains with 523 pixels, the four selected grains and the edge grains as MATLAB's, the ids and the largest area (4.098e6 for 4.112e6) differ with the native reconstruction; `selectInteractive` is prose; wrote `grains.phase`, `id2ind`, `findByOrientation` | Checked |
| Grains/ShapeParameters | ShapeParameters | Checked 2026-09-22; the largest forsterite area 2.8049e6 and its share 0.1531 to the digit, the paris summary 2.766 and 43.02 for 43.30, the correlation 0.477 for 0.4715, the same extreme grain with and without the edge grains as MATLAB's, the fractal dimensions 1.145, 1.018, 0.963 for 1.128, 1.011, 0.965; eight figures; grain nine is another grain here (the order differs); wrote `hist(grains)`, `histogram(grains)`, `logAxis` | Checked |
| Misorientations/AngleDistributionFunction |  | Pending | Checked |
| Misorientations/AxisDistributionFunction |  | Pending | Checked |
| Misorientations/MisorientationDistributionFunction |  | Pending | Checked |
| Misorientations/MisorientationGrainExchangeSym |  | Pending | Checked |
| Misorientations/MisorientationTheory |  | Pending | Checked |
| Misorientations/Twinning |  | Pending | Checked |
| ODFAnalysis/BinghamODFs | BinghamODFs | Checked 2026-09-22; the concentrations of every constructor form as MATLAB's, the fitted ones of the same rule on another draw, the four axes, eight figures; wrote `BinghamODF(kappa, ori)`, `BinghamODF(kappa, fibre)`, the zero padding, `odf.axes`, `orientation.eye`, `quaternion.eye`, `plot(ori, add2all=True)` on pole figures; contour bands now take the colour of their lower level | Checked |
| ODFAnalysis/DetectionOfSampleSymmetry | DetectionOfSampleSymmetry | Checked 2026-09-22; the recovered mounting rotation 1.34 degrees for 1.32 on another draw, the Aachen ODF maximum 16.94 as MATLAB's 16.95, the correction 3.12 degrees for 2.87 with the offsets 3.12, 3.01 for 2.87, 2.87, the six figures; found the Aachen reader taking the tilt marker of every ring as an intensity; `centerSpecimen` returns its four outputs and runs its search silently; the measured pole figures draw as markers without MATLAB's dynamic size | Checked |
| ODFAnalysis/EulerAngleSections | EulerAngleSections | Checked 2026-09-22; the bounds 90 90 90, the six section galleries in MATLAB's layouts with the markers and the beta fibre drawn into them; wrote `crystalFrame.load` for CIF files, the named rolling components `orientation.brass` and the rest, `annotate(ori)` and `plot(f)` into held sections, the `layout`, `coordinates='off'`, `xlabel` and `ylabel` options; the beta fibre's specimen direction is turned by 180 degrees about Z by `byMiller`'s representative | Checked |
| ODFAnalysis/FibreODFs | FibreODFs | Partial 2026-09-22; the beta fibre's h (12 6 11), the ridge 3.9239, the halfwidth table 7.62, 2.36, 1.26 and 15.21, 3.93, 2.06 to the digit, the fitted fibre and its eigenvalue ratio of the same size, five figures; the isosurface plot is prose; the named fibres now take MATLAB's raw ends where the port had projected the far end, `fibre.fit(..., diagnostics=True)` for the three outputs | Checked |
| ODFAnalysis/ODFCharacteristics | ODFCharacteristics | Checked 2026-09-22; the texture index 72.17 exact where MATLAB's quadrature gives 72.1424, the measured one 16.35 for 16.21 and the peak 95.16 for 93.97 by the recomputed Dubna ODF, the entropy -2.8385 for -2.8402, the volume 38.15 for 37.98, the fibre volume 94.04 where MATLAB prints 95.16 and the exact integral is 94.00, the seven annotated pole figures; found the fibre volume's cap grid biased outward and `mtexdata('dubnaODF')` returning the pole figures | Checked |
| ODFAnalysis/ODFComponents | ODFComponents | Checked 2026-09-22; the peak 109.4 for 110.4, the balls 11.2, 5.2 and the uniform 0.169, the components 48.9, 22.1, 21.3, 7.7 for 48.4, 22.1, 21.3, 7.3, the four annotated section galleries as MATLAB's; the third local maximum is a shoulder at 43.5 mrd where MATLAB reports 31.9; `calcComponents` returns its three outputs; found its clustering quadratic in Python and the radial evaluation slow, both now through vectorised angles and the harmonic form | Checked |
| ODFAnalysis/ODFExport | ODFExport | Checked 2026-09-22; the generic file's first rows 3.46659 and 2.63068 and the VPSC header to the digit, the six sections; MATLAB's `.mat` copy is `pickle`, which now hands the frames back as the session's handles; wrote `export(odf, ...)` with the generic, mtex and VPSC interfaces | Checked |
| ODFAnalysis/ODFImport | ODFImport | Checked 2026-09-22; the interpolation 1.205 and 3.50 mrd, the densities 1.177, 1.117, 1.031 and 3.11, 2.35, 1.56 as MATLAB's, the sections and the row of four pole figures; wrote `SO3Fun.load` with `'interp'`, `'density'`, `columnNames` and `halfwidth`, the VPSC reader, the sample table `mtexdatafile('odf')`; the interpolation is a harmonic least squares where MATLAB fits a radial function | Checked |
| ODFAnalysis/ODFInversePoleFigure | ODFInversePoleFigure | Checked 2026-09-22; 22.9594, 0.0249 and 5.4244 to the digit, the four inverse pole figure pairs as MATLAB's; wrote `calcPDF(odf, None, r)` for MATLAB's empty crystal direction | Checked |
| ODFAnalysis/ODFModeling | ODFModeling | Checked 2026-09-22; the maximum 15.9611 for 15.9609, the mean 1, the three pole figure pairs and the rotation warning as MATLAB's; the specimen frame now displays its group, a uniform part joins any radial function, `rotate` warns when it strips the specimen symmetry | Checked |
| ODFAnalysis/ODFPlot | ODFPlot | Partial 2026-09-22; the phi2 and sigma sections, the fibre profile and the Euler bounds 90 90 90 as MATLAB's; the two `plot3d` figures are prose; wrote `fundamentalRegionEuler`, `project2EulerFR`, the two-column grid of wide Euler sections | Checked |
| ODFAnalysis/ODFPoleFigure | ODFPoleFigure | Checked 2026-09-22; 0.1982 and the mean 1 to the digit, the four figures with `complete` and `antipodal` and the fibre profile as MATLAB's; a list of four-index directions is a `Miller` from a matrix of rows | Checked |
| ODFAnalysis/ODFShapes | ODFShapes | Checked 2026-09-22; the halfwidths and bandwidths to the digit but the bump kernel's 33 for 1024, the Dirichlet minimum -8.666, the three curve figures as MATLAB's; wrote `plot(psi)` of an SO(3) kernel, `symmetric=True` of a spherical one, `plotSpektra`, `hold` adopting a plain matplotlib figure | Checked |
| ODFAnalysis/ODFTheory | ODFTheory | Partial 2026-09-22; 0.8166, 9.0004 for 9.0003, the mean 1 and the volume 0.0235 to the digit, the map with the prisms, the Euler cloud and the six sigma sections as MATLAB's; the isosurface plot `plot3d(odf)` is prose; wrote `plot(ebsd, cS)`, MATLAB's named figure sizes, `plot(ori, 'Euler')` as a positional flag | Checked |
| ODFAnalysis/RadialODFs | RadialODFs | Checked 2026-09-22; the displays, the means 2 and 1 and the 2 x 3 gallery on one colour range as MATLAB's; found `plotPDF` opening a figure of its own inside a prepared `newMtexFigure` | Checked |
| ODFAnalysis/RandomSampling | RandomSampling | Checked 2026-09-22; the seven figures, the errors, the texture indices and the fitted laws of the same size as MATLAB's on another random fibre and draw (the exponents -0.23 and -0.26 for -0.21 and -0.26, the optimized kernel about a third sharper); wrote `plot(ori)` into held sections, `ind2color`, `calcError(..., 'L1')`; the page builds in four minutes | Checked |
| ODFAnalysis/SO3FunHarmonicRepresentation | SO3FunHarmonicRepresentation | Checked 2026-09-22; the degree two block 0.0997-0.2354i, 0.9289-0.4739i, 1.1373 for 0.0990-0.2350i, 0.9290-0.4718i, 1.1372 by the recomputed Dubna ODF, `isReal` False, the minimum -1.2645 to the digit, the approximation error 1.07 for 1.27 on another draw, the five figures; the same `reshape(5, 5)` gives MATLAB's matrix | Checked |
| ODFAnalysis/SigmaSections | SigmaSections | Checked 2026-09-22; the ten figures as MATLAB's: the section geometry with its reference field, the stars of a-axes, the hexagonal prisms in their panels, the four maxima one to a panel, the girdles, the twelve panels about (10-11); wrote `plot(oS)`, `quiver(ori, v)` in sections, `plotSection(ori, ori * cS, oS)`, settable `h1`, `h2`, `omega`, `cs.cAxis` as a Miller; replacing the reference field is prose | Written |
| PhaseTransitions/GrainGraphBasedReconstruction |  | Pending | Pending |
| PhaseTransitions/LowLevelParentGrainReconstruction |  | Pending | Pending |
| PhaseTransitions/MaParentGrainReconstruction |  | Pending | Checked |
| PhaseTransitions/PhaseTransitions |  | Pending | Checked |
| PhaseTransitions/MaParentGrainReconstructionAdvanced |  | Pending | Pending |
| PhaseTransitions/MartensiteVariants |  | Pending | Checked |
| PhaseTransitions/OrientationRelationshipFit |  | Pending | Checked |
| PhaseTransitions/ParentChildVariants |  | Pending | Checked |
| PhaseTransitions/TiBetaReconstruction |  | Pending | Checked |
| PhaseTransitions/TransformationTexture |  | Pending | Checked |
| PhaseTransitions/TriplePointBasedReconstruction |  | Pending | Checked |
| Plasticity/DislocationSystems |  | Pending | Checked |
| Plasticity/GND |  | Pending | Checked |
| Plasticity/Lankford |  | Pending | Checked |
| Plasticity/SachsModel |  | Pending | Checked |
| Plasticity/SchmidFactor |  | Pending | Checked |
| Plasticity/SingleSlipModel |  | Pending | Checked |
| Plasticity/SlipSystems |  | Pending | Checked |
| Plasticity/SlipTransmission |  | Pending | Checked |
| Plasticity/TaylorHex |  | Pending | Rendered |
| Plasticity/TaylorModel |  | Pending | Checked |
| Plasticity/TextureEvolution |  | Pending | Rendered |
| Plasticity/TwinningTutorial |  | Pending | Checked |
| Plasticity/VPSCImport |  | Pending | Checked |
| Plasticity/WBV |  | Pending | Rendered |
| Plotting/Annotations |  | Pending | Checked |
| Plotting/AxesAlignment |  | Pending | Checked |
| Plotting/ColorMaps |  | Pending | Checked |
| Plotting/CombinedPlots |  | Pending | Checked |
| Plotting/ContourPlots |  | Pending | Checked |
| Plotting/Legends |  | Pending | Checked |
| Plotting/Multiplot |  | Pending | Checked |
| Plotting/PlotTypes |  | Pending | Checked |
| Plotting/PlottingExport |  | Pending | Checked |
| Plotting/SphericalProjections |  | Pending | Checked |
| Plotting/TransparencyDemo |  | Pending | Checked |
| PoleFigureAnalysis/PoleFigure2ODF | PoleFigure2ODF | Checked 2026-09-22; the eight solver iterations, the RP errors 0.826 to 0.860 and the regularised 0.239 to 0.395 to three digits, the ODF difference 0.585 against 0.587, the peaks 95.2, 27.2 and 109.4 mrd against MATLAB's 93.9, 27.4 and 110.4 and the three mean RP errors 0.59, 0.75, 0.51 to the digit; found the RP threshold, fixed; the grid holds 20224 orientations against MATLAB's 19848 | Checked |
| PoleFigureAnalysis/PoleFigure2ODFAmbiguity | PoleFigure2ODFAmbiguity | Checked 2026-09-22; twelve pole figures resolve the modes, 128.9 against 3.4 where MATLAB has 128.5 against 5.4 and the truth is 129.6 and 0; seven leave both sets near 66 mrd, 64.0 and 67.4 against MATLAB's 66.2 twice; the Friedel poles coincide antipodally and under the Laue group; degree 9 of Santa Fe parts the three spectra; wrote `SantaFe`, `cs.Laue()` on a Miller's frame | Checked |
| PoleFigureAnalysis/PoleFigure2ODFGhostCorrection | PoleFigure2ODFGhostCorrection | Checked 2026-09-22; the RP errors 0.0087/0.0088/0.0108 and 0.0251/0.0255/0.0265 against MATLAB's 0.0089/0.0088/0.0109 and 0.0256/0.0246/0.0264, L1 0.1267 and 0.0054, L2 0.3641 and 0.0315, and the fibre profile 24.96/2.83/3.73 and 37.33/1.06/1.05 against 25.23/2.82/3.71 and 37.38/1.06/1.05 | Checked |
| PoleFigureAnalysis/PoleFigureCorrection | PoleFigureCorrection | Checked 2026-09-22; counts into mrd on the Geesthacht steel: background subtraction leaving 7 to 631 with mean 273 as MTEX reports, two outlier passes and the cap; the normalisation reaches 2.27 mrd where MATLAB reaches 2.15, its two scans carrying different antipodal flags after the correction, and the ODF route agrees to 0.02 percent | Checked |
| PoleFigureAnalysis/PoleFigureDubna | PoleFigureDubna | Checked 2026-09-22; the seven solver iterations and the residuals 0.4222/0.1833/0.3523 against 0.4269/0.1820/0.3534, 1224 of 1368 directions after the band, the coarse grid 2382 against 2472 orientations | Checked |
| PoleFigureAnalysis/PoleFigureExport | PoleFigureExport | Checked 2026-09-22; wrote `export(pf, fname, 'degree')` in `io/exportpolefigure.py`, one table per pole figure named by its directions as MATLAB names them, the round trip through the generic reader exact; the rows run along the polar angle first where MATLAB's run along the azimuth | Checked |
| PoleFigureAnalysis/PoleFigureImport | PoleFigureImport | Checked 2026-09-22; the two Dubna files with the superposed pair at 0.52 and 1.23 on their 72 x 19 grids, the display as MATLAB's; found the ` + ` between the directions of a superposed entry, fixed | Checked |
| PoleFigureAnalysis/PoleFigurePlot | PoleFigurePlot | Checked 2026-09-22; the three ptx panels as markers, at another marker size, as contours and on one colour range, and the recalculated pole figures beside them; wrote the `parula` colour map; found `mtexColorMap` and `setColorRange` flattening a filled contour plot and `mtexColorbar` not toggling, both fixed, which also corrected the contour figures of VectorsDensityEstimation, FibreODFs and ODFImport | Checked |
| PoleFigureAnalysis/PoleFigureRefinement | PoleFigureRefinement | Checked 2026-09-22; the iterative reconstruction beating the direct one on the Dubna set, its errors 0.295 to 0.166 against MATLAB's 0.290 to 0.157 and the L1 distance 0.151 against 0.161; the adaptive loop takes another path, `refine` keeping the centroids inside the cap and the resolution being derived from the points, so it ends at 327 directions against 367 and at 0.22 against 0.40, the iterative estimate at 0.12 either way; wrote `v.resolution` | Checked |
| PoleFigureAnalysis/PoleFigureSantaFe | PoleFigureSantaFe | Checked 2026-09-22; ODF error 0.047 against 0.045 and 0.101 against 0.099 on another noise draw, the residuals all 0.05, the model peak 5.0395 to the digit and the minima 0.59 and 0.20 against 0.587 and 0.177; wrote `SantaFe`, the `FontSize` option and `byMiller` with a specimen frame | Checked |
| PoleFigureAnalysis/PoleFigureSimulation | PoleFigureSimulation | Checked 2026-09-22; the L1 error 0.084 against 0.082 and the error curve by pole figure count of the same shape, 0.30 at one figure falling to 0.08 at seven with the bump at five, on another noise draw | Checked |
| Rotations/RotationDefinition | RotationDefinition | Checked 2026-09-22; the angles 69.5509, the Roe and Bunge triplets, the Rodrigues vector, the matrix, the mapped directions and the quaternion identities to the digit, the fit error a random draw as MATLAB's; one figure; wrote `arrow3d`, `rotation.fit(method='kabsch')`; `Euler` returns radians as a tuple, the page divides by `degree` | Checked |
| Rotations/RotationFibre | RotationFibre | Checked 2026-09-22; the fibre displays, h, r, the zero mapping errors, the equality and the 12 samples as MATLAB's; four figures as MATLAB's; wrote `orientation(f, points=n)`, the flags and `lineColor` of `plot(f)`, `'complete'`; found `project2EulerFR` choosing another representative for points already inside the box, fixed | Checked |
| Rotations/RotationImproper | RotationImproper | Checked 2026-09-22; every logical, 8, 4, 4, 4, 8 and the determinants as MATLAB's; one figure; MATLAB's `rotation(cs)` over a mixed group is `orthogonalTransform(cs)`, `isImproper` a property, a mixed `cat` promotes; `rotation` has no `isImproper` by the semantics | Checked |
| Rotations/RotationOperations | RotationOperations | Checked 2026-09-22; 82.8192, 104.4775, the axis, the 60 degree distances and the Matthies triplet 270, 60, 180 to the digit; two figures as MATLAB's; `inv(rot) * v` for `rot \ v`, broadcasting for `.*` and a column times a row for the outer `*` | Checked |
| Rotations/RotationPlotting | RotationPlotting | Checked 2026-09-22; the four scatter plots as MATLAB's after `scatter(rot)` learned bare rotations (the Euler box, the 180 degree ball, `'noBoundary'`), MATLAB's marker options; the subset count is a random draw (26 here, 32 in the MATLAB run, 20 on the page), 5.7669 to the digit | Checked |
| Rotations/RotationRepresentations | RotationRepresentations | Checked 2026-09-22; the Rodrigues lengths, the homochoric radius 1.3307 to the digit, the cubochoric round trip 3.7e-13 for MATLAB's 3.4e-6; the histogram and the two boundaries as MATLAB's, drawn with matplotlib; `cubo2homo` exported, `rotation.byCubochoric` exists here | Checked |
| Rotations/RotationSpinTensor | RotationSpinTensor | Checked 2026-09-22; T, the two skew matrices, the axes (1,2,3) and (-0.318, 1.03, 3.58), the zero residuals and the crystal axis (1 2 -3 3) to the digit; one figure; `spinTensor(S)` reads the axial vector, `vector3d(S)` too, `exp(S, ref, side)`; found `orientation.byAxisAngle(Miller, w)` built against the specimen frame and `Miller(tangentVector, cs)` keeping the UVTW style, both fixed | Checked |
| Rotations/RotationTangentSpace | RotationTangentSpace | Checked 2026-09-22; the residuals at round-off, the displays of the left, right and summed vectors 0.2, 0.581, 0.427 as MATLAB's; one figure; wrote `quiver3(t)` at the base point, `plot(rot)` into axis angle space; the side is `'left'`/`'right'` for `SO3TangentSpace.leftVector` | Checked |
| SO3Functions/HarmonicApproximationTheory |  | Pending | Checked |
| SO3Functions/RBFApproximationTheory |  | Pending | Pending |
| SO3Functions/SO3FunApproximationTheory |  | Pending | Checked |
| SO3Functions/SO3FunConcept |  | Pending | Checked |
| SO3Functions/SO3FunConvolution |  | Pending | Checked |
| SO3Functions/SO3FunDefinition |  | Pending | Checked |
| SO3Functions/SO3FunOperations |  | Pending | Checked |
| SO3Functions/SO3FunQuadrature |  | Pending | Checked |
| SO3Functions/SO3FunSymmetricFunctions |  | Pending | Checked |
| SO3Functions/SO3FunVectorField |  | Pending | Pending |
| SO3Functions/SO3FunVectorValued |  | Pending | Adapted |
| SO3Functions/SO3Kernels |  | Pending | Checked |
| SO3Functions/WignerFunctions |  | Pending | Checked |
| SphericalFunctions/S1FunHarmonics |  | Pending | Checked |
| SphericalFunctions/S2Bingham |  | Pending | Checked |
| SphericalFunctions/S2FunApproximationInterpolation |  | Pending | Checked |
| SphericalFunctions/S2FunAxisField |  | Pending | Checked |
| SphericalFunctions/S2FunConcept |  | Pending | Checked |
| SphericalFunctions/S2FunHarmonicRepresentation |  | Pending | Checked |
| SphericalFunctions/S2FunOperations |  | Pending | Checked |
| SphericalFunctions/S2FunPlotting |  | Pending | Checked |
| SphericalFunctions/S2FunQuadrature |  | Pending | Checked |
| SphericalFunctions/S2FunRadon |  | Pending | Checked |
| SphericalFunctions/S2FunSampling |  | Pending | Checked |
| SphericalFunctions/S2FunSym |  | Pending | Checked |
| SphericalFunctions/S2FunVectorField |  | Pending | Checked |
| SphericalFunctions/S2FunVectorValued |  | Pending | Adapted |
| SphericalFunctions/S2Kernels |  | Pending | Checked |
| SphericalFunctions/SphericalHarmonics |  | Pending | Checked |
| Tensors/BirefringenceDemo |  | Pending | Checked |
| Tensors/MagneticAnisotropy |  | Pending | Checked |
| Tensors/PiezoElectricity |  | Pending | Checked |
| Tensors/TensorArithmetics |  | Pending | Checked |
| Tensors/TensorAverage |  | Pending | Checked |
| Tensors/TensorDefinition |  | Pending | Checked |
| Tensors/TensorImport |  | Pending | Checked |
| Tensors/TensorVisualisation |  | Pending | Checked |
| Tutorials/BoundaryTutorial |  | Pending | Checked |
| Tutorials/EBSDTutorial |  | Pending | Checked |
| Tutorials/GrainTutorial |  | Pending | Checked |
| Tutorials/ODFTutorial |  | Pending | Checked |
| Tutorials/PoleFigureTutorial |  | Pending | Checked |
| Vectors/VectorDefinition | VectorDefinition | Checked 2026-09-22; every display and the four figures as MATLAB's, the grid under `z←↑y` about the screen centre where MATLAB's follows Z | Checked |
| Vectors/VectorGrids | VectorGrids | Checked 2026-09-22; the four grids and their counts as MATLAB's after `HEALPixS2Grid` was written and the Fibonacci count rule taken from MATLAB; the L1 deviations to the digit, the L2 ones sqrt(4 pi) smaller by the port's norm convention, said in the page | Checked |
| Vectors/VectorsAxes | VectorsAxes | Partial 2026-09-22; every angle and mean to the digit, ten of twelve figures as MATLAB's; `plot(pf)` and the annotation on it wait on S6; found the `complete` flag dropped for antipodal functions | Checked |
| Vectors/VectorsDensityEstimation | VectorsDensityEstimation | Checked 2026-09-22; the halfwidth table, the three local maxima, the basis densities, the Dirichlet minimum and the observed fraction to the digit, the volume 0.1721 for 0.1723, 1152 grains for 1151, all ten figures as MATLAB's; found the local maxima seeds and the Laue antipodality | Checked |
| Vectors/VectorsExport | VectorsExport | Checked 2026-09-22; the three tables digit for digit, the round trip errors 3.3461e-05 and 3.3333e-07 as on the page; the azimuth in [0, 360) where MATLAB writes the constructed angle, said in the page | Checked |
| Vectors/VectorsImport | VectorsImport | Checked 2026-09-22; `vector3d.load` written for it, the thousand directions and both figures as MATLAB's | Checked |
| Vectors/VectorsOperations | VectorsOperations | Partial 2026-09-22; every number to the digit and the selection figure as MATLAB's; the `arrow3d` figure waits on S5 | Checked |
