# Subplan P3b - maps3d

Status: written 2026-09-26; V1 done the same day (`mtex/maps3d/voxellattice.py`, `ebsd3.py`,
`mtex/io/load3d.py`: the IN100 and xnovo volumes and two slices as MATLAB's, voxel for voxel);
V2 too (`calcgrains3d.py`, `grain3d.py`, `grain3boundary.py`: IN100's grains as MATLAB's, face for face). Syntax and algorithm reference: `../master/EBSDAnalysis/@EBSD3`,
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
- `neper`: `simulateGrains`, `simulateTwinGrains`, `simulateChildGrains`, `getSlice`
  through the `neper` executable;
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
  cell; a slice keeps the specimen frame (MATLAB turns it to a convention facing the plane,
  left for V7).
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
  IN100 and on a Neper tessellation.
- **V4 - loading grains and Neper.** `grain3d.load` for DREAM.3D meshes (the 758 564
  triangles of IN100) and `.tess`, `orientFaces`, the `neper` interface with its three
  simulations and `mtexdata('NeperGrain3d')`.
- **V5 - boundary smoothing and GBND.** `smoothBoundary`, `reduceBoundary`,
  `refineBoundary`, the Taubin and Laplace filters on the vertex graph, the volume they keep;
  the 3D `calcGBND` against `check_gbnd3d`.
- **V6 - operations on the volume.** `KAM`, `fill`, `smooth`, `interp`, `calcGND`,
  `spatialProfile`, sharing the filters of `mtex.analysis.denoising` over the 3D graph.
- **V7 - drawing.** See section 4.
- **V8 - documentation.** The nine EBSD3Analysis pages and `Grains/Dream3dGrains`, checked
  against MATLAB, with interactive figures.

## 4. Drawing

A scene layer in `mtex/plotting/scene3d.py` computes what is drawn as NumPy arrays and does
not import a renderer: surface meshes with per-face or per-vertex colours, the reduced
voxel surfaces (coplanar faces of one boundary merged, only the visible part of a clipped
volume), three orthogonal slices of a volume as textured planes (MATLAB's `volshow`
'SlicePlanes'), lines, arrows and labels. Two renderers draw a scene:

- **PyVista** (the optional extra `plot3d`, `pyvista` with `pyvistaqt` on the PySide6 the
  wizard already uses): interactive windows, offscreen pictures for the doc build, and the
  export of a scene as a standalone HTML page (vtk.js) that the doc pages embed.
- **matplotlib** `Axes3D` for small scenes and figures mixing 3D and 2D axes, always
  available.

The doc build (`docs/make.py`) learns one hook: a figure that holds a 3D scene is written
as an interactive HTML fragment beside its PNG. How the fragment is produced (PyVista's
`export_html` through trame, or our own writer of vtk.js scenes) is settled by a spike at the
start of V7, measured on the 758 564 triangles of IN100: file size, load time in the browser,
and the pages' build time.

## 5. Kernels

Numba loops in `mtex/maps3d/kernels.py`, NumPy forms as fallback and oracle: the face
extraction of the segmentation, the merge of coplanar faces, the plane cut of a mesh, the
vertex smoothing. The union-find and the criteria are the 2D ones.

## 6. Left out

`EBSD3.calcGrains` on an unstructured cloud through a 3D Voronoi decomposition (master's
`spatialdecomposition3d` 'voronoi' branch; master itself segments voxels), `calcTraces`,
`plotUnitCells`, `loadDream3d_sketch`, the MATLAB `volshow` viewer's own controls.
