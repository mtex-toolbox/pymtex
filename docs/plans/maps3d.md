# Subplan P3b - maps3d

Status: written 2026-09-26; V1 done the same day (`mtex/maps3d/voxellattice.py`, `ebsd3.py`,
`mtex/io/load3d.py`: the IN100 and xnovo volumes and two slices as MATLAB's, voxel for voxel);
V2 too (`calcgrains3d.py`, `grain3d.py`, `grain3boundary.py`: IN100's grains as MATLAB's, face for face);
V3 on 2026-09-26 (`section3d.py`; the measures, selections and the z section as MATLAB's);
V4 the same day (`io/loadgrains3d.py`, `orientfaces.py`, `neper.py`); V5 too (`smoothing3d.py`,
`analysis/gbnd3d.py`); V6 too (`intragranular3d.py`); V7 too (`plotting/scene3d.py`, `render3d.py`,
`volumeplot.py`, section 4); V8 the same day: the nine EBSD3Analysis pages and `Grains/Dream3dGrains` checked against
MATLAB master (findings in `docs/findings.md`). Syntax and algorithm reference: `../master/EBSDAnalysis/@EBSD3`,
`@EBSD3square`, `@grain3d`, `@grain3Boundary`, `@neper`, the loaders
`../master/interfaces/loadEBSD_dream3d.m`, `loadEBSD_xnovo.m`, `loadNeperTess.m`,
`loadGrains_Dream3d.m`, and the nine pages of `../master/doc/EBSD3Analysis` with
`doc/Grains/Dream3dGrains.m`. Master, not frameSymmetry, is the reference here: only master
has the voxel `calcGrains` (frameSymmetry's `EBSD3.calcGrains` is a stub), the boundary
smoothing of `grain3d` and the chapter as it now stands. The Julia port has no 3D part, so
the design follows this port's 2D maps layer (`docs/plans/maps.md`) wherever the two meet.
Tests: `../master/tests/slow/check_gbnd3d.m`, `check_orientFaces.m`, and MATLAB through the
master bridge on the IN100 volume.

## 1. Deliverable

A core `mtex.maps3d` above `mtex.maps` (CLAUDE.md's layout reserves it), in MATLAB's names:

- `EBSD3`: voxels in a specimen frame, one rotation, phase id and property row each, on a
  lattice of three translations `d1, d2, d3` or as a list; selection as `EBSD`; `gridify`
  to the `(n1, n2, n3)` array and back; `slice(plane)` to a 2D `EBSD`; `EBSD3.load`;
- `calcGrains(ebsd3, ...)`: the segmentation over the voxel face pairs with the 2D criteria
  (angle, per phase, soft, custom, `minPixel`), `grainId` on every voxel;
- `grain3d`: closed polyhedra given by the incidence `I_GF` of grains and boundary faces,
  `meanOrientation`, `GOS`, `numPixel`; `volume`, `surfaceArea`, `centroid`, `diameter`,
  `principalComponents`, `neighbors`, `numNeighbors`, `rotate`, subsets, `orientFaces`,
  `triangulate`, `slice` and `intersected` to `grain2d`, `grain3d.load` of a DREAM.3D mesh
  and a Neper `.tess`;
- `grain3Boundary`: triangles or polygons over shared vertices with `grainId`, `ebsdId`,
  `phaseId`, `misrotation`, `misorientation`, `N`, `area`, `centroid`, `edges`,
  `nodeType`, `perimeter`, `flip`, the incidences `I_FG`, `I_VF`, `I_VG`, `calcGBND`;
- `smoothBoundary`, `reduceBoundary`, `refineBoundary` with the Taubin and Laplace filters;
- on the volume: `KAM`, `fill`, `smooth`, `interp`, `calcGND`, `spatialProfile`;
- `neper`: `simulateGrains`, `simulateTwinGrains`, `simulateChildGrains` through the `neper`
  executable (`getSlice` left out, decided 2026-09-26: `grains.slice` cuts the loaded grains);
- drawing: `plot(ebsd3)`, `plot(grains3d)`, `plot(gB3)`, `quiver`, `slice` views, on a
  renderer-independent scene with PyVista and matplotlib behind it, and interactive 3D
  figures on the doc pages.

## 2. Data model

- A volume is an `EBSD3` whose gridded form is an `(n1, n2, n3)` array with the translations
  `d1, d2, d3` (MATLAB's `EBSD3square`); a list keeps the lattice it came from, as master's
  "a selection from a volume is a list that keeps its lattice". Voxel `(i, j, k)` sits at
  `pos0 + i d1 + j d2 + k d3`; the neighbour pairs are the three axis shifts.
- Grains and boundaries share one vertex list (`allV`, a `vector3d` in the specimen frame);
  faces are an `(n, 3)` integer array of triangles, or polygons as offsets into one index
  array (never a Python list of lists). `I_GF` is a `scipy.sparse` matrix with +1 where the
  face normal points out of the grain, -1 where it points in, as MATLAB's.
- `calcGrains` writes every boundary face of the voxel surface as two triangles, as master;
  the reduction of coplanar faces is a separate call for drawing and export.
- Everything returned is in the P1 types: positions and normals `vector3d`, orientations
  `orientation` framed by the phase and the specimen, misorientations with both crystal
  frames.

## 3. Steps

- **V1 - EBSD3.** The container, the lattice of three translations inferred from positions
  or given, `gridify`, selection, `slice(plane)` into a 2D `EBSD` on the plane's lattice,
  `EBSD3.load` for DREAM.3D cell data (h5py), the `.ang` stack of `data/EBSD/3dData` with
  its slice spacing, and xnovo HDF5. Test against MATLAB: the IN100 volume and the stack
  read to the digit, one slice of each. DONE: `bench/matlab/probe_ebsd3.m`; HDF5's C order is
  turned to (x, y, z) so that voxel [i, j, k] is MATLAB's (i+1, j+1, k+1); a trigonal or
  hexagonal xnovo phase takes the hexagonal angles, as MATLAB reads Corundum's rhombohedral
  cell; a slice keeps the specimen frame (MATLAB turns it to a convention facing the plane);
  the map plots show it face on anyway, in the coordinates of its `mapPlane`, and V7 lifts it
  into space by that plane.
- **V2 - calcGrains.** The face pairs of the lattice, the criteria of `mtex.maps.criteria`
  on them, the union-find of `mtex.maps.kernels`, `minPixel` by one resegmentation as
  master, the boundary faces as triangles with their outward orientation, the mean
  orientation and GOS per grain. Numba for the face extraction. Test: IN100 at 5 and 10
  degrees, the grain count, volumes, face count and misorientations against MATLAB. DONE
  (`bench/matlab/probe_calcgrains3d.m`): 751 and 809 grains, 748 886 and 757 554 triangles,
  343 561 and 346 729 vertices, the numbering, the GOS, the misorientations and the centroids
  as MATLAB's, in 1.3 s for MATLAB's 1.4 and 2.1. The voxels are taken in MATLAB's
  column-major order and the grains numbered by their last voxel, as MTEX's
  connectedComponents numbers the trees of the elimination tree; the mean is taken about
  the first voxel where MATLAB takes the last, the same orientation in another symmetric
  equivalent; a grain the cull made notIndexed has no mean and GOS 0 where MATLAB computes
  them; the padding of a gridded list is in no grain and its faces are faces of the surface.
  `calcGrains` of the flat namespace is the one of `mtex.maps3d`, which hands a 2D map to
  `mtex.maps`. The face extraction stays NumPy: vectorised, it is not the bottleneck.
- **V3 - grain3d and grain3Boundary.** The measures and operations of section 1, `slice`
  and `intersected` of the polyhedra with a plane into `grain2d`. Test against MATLAB on
  IN100 and on a Neper tessellation. DONE (`bench/matlab/probe_grain3d.m`; the Neper half
  comes with V4): a face is a triangle or a polygon over offsets into one index array, every
  measure one pass over the fan triangles; `diameter`, `principalComponents`, `neighbors`,
  `numNeighbors`, the selection of 226 grains with its 657 170 faces, `perimeter`,
  `nodeType`, `edges`, `intersected` and the z section (67 pieces of 63 grains, area 625,
  perimeter 907.5, the same `Id3d`) as MATLAB's. The section turns every crossing face into
  a segment with the grain its normal leaves on the left, splits a grain the plane cuts
  apart into one 2D grain per connected part and gives the holes to the part holding them;
  the 2D boundary is put in walk order by `mtex.maps`' own `_walkOrder`. Found in master:
  `principalComponents` fails on triangulated grains (`c = c.xyz` on a double; the probe
  passes the triangles as closed polygons); on the tilted (1,1,1) section of IN100 its
  pieces add up to 814.52 where the section of the box is 809.7134, which this port's
  pieces cover exactly (354 pieces for MATLAB's 504); `g.numNeighbors(1:10)` fails in its
  subsref. `histogram`, `quiver` and `plot` came with V7; `cat` of grain sets is not needed yet.
- **V4 - loading grains and Neper.** `grain3d.load` for DREAM.3D meshes (the 758 564
  triangles of IN100) and `.tess`, `orientFaces`, the `neper` interface with its three
  simulations and `mtexdata('NeperGrain3d')`. DONE (`bench/matlab/probe_loadgrains3d.m`) but
  for `mtexdata('NeperGrain3d')`, left for V8 where the pages need it (the
  pages' orientations are random, so V8 takes MATLAB's grains through a neutral file):
  `grain3d.fromFaces` is MATLAB's `grain3d(V, F, I_GF, ori, CSList, phaseId)`; the IN100 mesh
  (794 grains, 757 564 triangles, the volumes, every inner face +1 and -1 after
  `orientFaces`) and `my100grains.tess` (1000 polyhedra, 7131 polygons, volumes, means,
  centroid, neighbours, diameters, the z = 1 section of 232 grains) as MATLAB's. DREAM.3D
  writes `AvgQuats` as (x, y, z, w) of the passive rotation, as the Euler angles of the voxels
  show; the rotation is (w, -x, -y, -z) and the means lie within a degree of their voxels.
  MATLAB's `loadGrains_Dream3d` reads (x, y, z, w) as (a, b, c, d), which turns every grain
  away from its voxels. `orientFaces` as MATLAB's, but a face that gets other sides than
  its old ones swapped (a face given with the same sign for both grains) takes the new
  sides with their phases and misorientation, where MATLAB only swaps. `neper` runs the
  command of the preference `neperCommand` with `np.env` added to its environment; the
  lamellae of a host are searched without MATLAB's window of 50 cells, which fails for a
  host of more (`simulateChildGrains` at MATLAB's default width 0.015 in a unit cube). The
  Neper on this machine is built against scotch 7.0.4 and the system has 7.0.14: it runs
  with a link `libscotch.so.7.0.4` on `LD_LIBRARY_PATH`, and `tests/test_neper.py` skips
  where it does not run.
- **V5 - boundary smoothing and GBND.** `smoothBoundary`, `reduceBoundary`,
  `refineBoundary`, the Taubin and Laplace filters on the vertex graph, the volume they keep;
  the 3D `calcGBND` against `check_gbnd3d`.
- **V6 - operations on the volume.** `KAM`, `fill`, `smooth`, `interp`, `calcGND`,
  `spatialProfile`, sharing the filters of `mtex.analysis.denoising` over the 3D graph. DONE
  but for `smooth`: only master's `gradientX/Y/Z` (`gradientDim`) are three dimensional, and
  this port's agree with them on an IN100 block voxel for voxel, NaN across the grain
  boundaries included; master's `curvature` and `calcGND` of a volume fail on its unit cell
  ("unit cell must have 4 or 6 corners"), its `KAM`, `fill`, `reduce` and `fillByGrainId`
  work on x and y alone, and `smooth` and `spatialProfile` are stubs. This port's:
  `curvature` is complete, the sum of the three gradients against their unit steps;
  `calcGND` is the 2D one, which now asks `ebsd.curvature()`; `KAM` over the voxels within
  `order` steps along the axes, the same grain or below the threshold; `fill` from the
  nearest measured voxel by a distance transform, the empty voxels the outside reaches over
  empty voxels left unless `extrapolate`; `spatialProfile` the voxels along a polyline at
  the voxel spacing, through `interp`. `smooth` of a volume waits for the 3D graph of the
  denoising filters.
- **V7 - drawing.** See section 4.
- **V8 - documentation.** The nine EBSD3Analysis pages and `Grains/Dream3dGrains`, checked
  against MATLAB, with interactive figures.

## 4. Drawing

Three files in `mtex/plotting`, none importing a renderer at module level:

- `scene3d.py`, what is drawn, as NumPy arrays: `mesh3d` (polygons over ptr/idx with a colour
  per face, face and edge transparency), `slices3d` (three orthogonal slices of a voxel volume,
  MATLAB's `volshow` 'SlicePlanes'), `picture3d` (an RGBA picture on a rectangle in space),
  `lines3d`, `arrows3d` (shafts and cone heads, as `quiver3` draws them), and `scene3d`, the
  items with the plotting convention they are seen by. The builders follow MATLAB:
  `grainMesh` draws the outer faces of a grain set in its grains' colours (the faces between
  two of the grains only when translucent, grey), `boundaryMesh` light blue or by a value per
  face, `volumeSlices` the three middle slices by phase, value or orientation. The colours
  come from the map plots' `mapValues`.
- `render3d.py`, the drawing: PyVista (the extra `plot3d`) for the offscreen picture and the
  interactive window, unlit and flat as MATLAB draws a patch without a light, with a parallel
  projection; matplotlib's 3D axes when PyVista is missing; and `sceneHTML`/`writeHTML`, a
  standalone page drawing the scene with three.js from jsdelivr, the arrays compressed and in
  base64 inside the page, decompressed by the browser's `DecompressionStream`.
- `volumeplot.py`, the figure: `plot(grains3d | grain3Boundary | EBSD3)` and
  `quiver(grains3d | grain3Boundary, dir)` put their items into a `scene3d` that an
  `mtexFigure` holds and shows in its axes as the picture PyVista renders, cropped to what it
  shows; `hold`, `setCamera`, `mtexColorbar` act on it as on every figure. `show3d()` opens
  the scene in a window, `exportHTML(fname)` writes the page. A two dimensional map lying in
  a plane of space (a slice) is lifted into a scene as the picture of its own 2D plot on its
  rectangle, so every option of the map plots holds there: when it is plotted into a held
  scene, into a held map of another plane, or when `setCamera` looks at a map from off its
  normal. Maps in one plane are drawn into one picture, one over the other. The map plots
  record what they drew on their axes for this (`_liftable` in `mapplot.py`).

`histogram(grains3d, ...)` weights the grains by their volume, as master's.

The doc build (`docs/make.py`) writes a figure holding a scene as its PNG linked to the
interactive page, `[![](Page-k.png)](Page-k.html)`.

The spike (2026-09-26) on IN100, 794 grains of 757 564 triangles: the outer faces are 120 000
triangles over 60 002 vertices, 2.5 MB raw and 0.5 MB compressed. PyVista's `export_html`
(trame, trame-vtk, trame-pyvista, deprecated in PyVista 0.49) wrote 2.7 MB, of which 1.07 MB is
the vtk.js viewer repeated in every page; the three.js page holds 1.2 MB, builds in 0.34 s
and renders as the PyVista picture does (checked in headless Chromium). The offscreen picture
takes 0.16 s, the whole `plot` 0.6 s. Decided with the user: the own three.js writer, the scene
figure with PyVista.

## 5. Kernels

Numba loops in `mtex/maps3d/kernels.py`, NumPy forms as fallback and oracle: the face
extraction of the segmentation, the merge of coplanar faces, the plane cut of a mesh, the
vertex smoothing. The union-find and the criteria are the 2D ones.

## 6. Left out

`EBSD3.calcGrains` on an unstructured cloud through a 3D Voronoi decomposition (master's
`spatialdecomposition3d` 'voronoi' branch; master itself segments voxels), `calcTraces`,
`plotUnitCells`, `loadDream3d_sketch`, the MATLAB `volshow` viewer's own controls.
