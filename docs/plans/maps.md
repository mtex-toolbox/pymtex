# Subplan P3 - maps

Status: written 2026-09-21; M1 to M7 done the same day, M8 on 2026-09-22, M9 on 2026-09-23, M10 and M11 on 2026-09-26. Syntax reference:
`../frameSymmetry/EBSDAnalysis/` with its `CLAUDE.md` and the ADRs 0002 (walk order) and
0004 (FMC defaults) of that branch. Design reference: `../mtexJulia/lib/MTEXMaps/` and its
`docs/plans/ebsd.md`, M0 to M11 done there, which this document follows package by
package. Tests: `../frameSymmetry/tests/core/check_ebsd`, `check_ebsdGrid`, `check_find`,
`check_calcGrainsCases`, `check_boundaryChains`, `check_chainOrder`, `check_grainMerge`,
`check_calcComponents`, `check_gradient`, `check_gbnd`, and `../mtexJulia/test/` for
`ebsd`, `lattice`, `decomposition`, `calcgrains`, `quadruple`, `boundary`, `grains`,
`shape`, `smoothing`, `intragranular`, `denoising`, `reconstruction`, `gbnd`.

## 1. Deliverable

A package `mtex.maps` whose public layer reads like the MATLAB of the branch and whose
inside is the Julia design on NumPy arrays:

- `EBSD`: positions in a specimen frame, one rotation per pixel, a phase id per pixel with
  a list of crystal frames, named properties, selection by phase, property, position and
  orientation, `rotate`, `shift`, `transform`, `extent`;
- the lattice: inference of the square or hexagonal lattice, rotated, sheared or smoothly
  distorted, from the positions; `gridify` to map shape and back; neighbours, gradients,
  subgrids, filling and interpolation;
- `calcGrains`: the local decomposition with the alpha closing, a pluggable criterion
  (angle, two levels, per phase, soft, by property, custom), `minPixel`, the quadruple
  point removal, a grain id on every pixel;
- `grain2d`: polygons with inclusions, mean orientation, GOS, `numPixel`, area,
  perimeter and the shape parameters, `merge`, subsets, neighbours, hull, `inpolygon`;
- `grainBoundary`: segments in walk order with grain, pixel and phase ids, the
  misorientation with grain exchange, chains, triple points, direction, length,
  curvature, coarsening, smoothing, the grain boundary normal distribution;
- the intragranular quantities `mis2mean`, `KAM`, `calcGROD`, `calcGND`, `gradient`,
  `grainMean`, `spatialProfile`;
- the EBSD filters and the boundary filters, `fill`, `interp`;
- `parentGrainReconstructor` with the votes, the variant graph, `calcParentEBSD`;
- `simulateEBSD`, `cleanUpPseudoSym`, the two spatial transforms, `weightedBurgersVec`;
- the tests of section 8 passing.

Everything returned is in the P1 types: positions are `vector3d` in the specimen frame,
orientations are `orientation` framed by the phase and the specimen, boundary
misorientations carry `antipodal=True` within one phase, a GBND is an `S2FunHarmonic` of
P2 once P2 exists.

## 2. The two references, and the rule between them

MATLAB names the user sees, Julia the representation and the algorithm. The Julia plan
lists in every package what it designed rather than ported, with the reason; those
designs are taken here without rediscussion and the list below records where a Python
reader will meet a MATLAB name on a Julia object. Where MTEX.jl deliberately changed a
MATLAB result, the Python port follows Julia and the difference is recorded once, in
section 9.

| MATLAB | Python | from Julia |
| --- | --- | --- |
| `@EBSD`, `@EBSDgrid`, `@EBSDsquare`, `@EBSDhex` | `EBSD`, one class; a gridded map has two-dimensional arrays | matrix storage is an array shape, not a class; the cell shape lives on the lattice |
| `ebsd.unitCell`, `latticeBasis`, `ebsd.lattice`, `gridIndex`, `assignGridIndex` | `lattice` object with `A`, `origin`, `ij`, `shape` `'square'` or `'hex'`, `stencil`, `dxy`, `unitCell` | inferred by direction, the index one checked rounding grown over the neighbour graph where the check fails |
| `phaseList`, `CSList`, `phaseMap`, `phaseId`, `phase` | `phaseList` of interned frames, the not indexed frame first; `ebsd.CSList`, `ebsd.phaseId`, `ebsd.phase` | colours and names read off the frames |
| `ebsd.prop.mad`, `dynProp` | `ebsd.prop` dict, reachable as `ebsd.mad` | properties as a dict; options as keywords |
| `ebsd(cond)`, `ebsd('Forsterite')`, `ebsd('indexed')`, `ebsd(i,j)` | `ebsd[mask]`, `ebsd['Forsterite']`, `ebsd['indexed']`, `ebsd[i, j]` on a gridded map | `EBSD` is a table: indexing returns a sub-map |
| `spatialDecompositionGrid`, `jcvoronoi`, `spatialDecomposition` | `decompose(ebsd, alpha)` returning a cell complex | the local clipping of unit cells, the alpha closing as a distance transform, no Voronoi library |
| `spatialDecompositionAlpha` (`'delaunay'`) | `decompose(..., method='alphaComplex')` | needs a general triangulation, `scipy.spatial.Delaunay` behind an option |
| `gbcAngle`, `gbcSoft`, `gbcVariants`, `gbcCustom`, `gbcFMC` | `gbcAngle`, `gbcSoft`, `gbcEqual`, `gbcCustom`, `gbcFMC` (`maps/fmc.py`, `calcGrains(..., fmc=)`, MATLAB's hierarchy pair for pair on the same pixel order) | one `connectivity(crit, ebsd, i, j)` per pair, levels as the threshold, `byPhase` |
| `doSegmentation`, `connectedComponents`, `I_DG` | `segment(complex, ebsd, criterion)` | union-find with path halving |
| `[grains, ebsd.grainId] = calcGrains(ebsd)` | `grains = calcGrains(ebsd)`, the grain id written into the map, `ebsd.copy()` to keep it | the two-pass cull of `minPixel` |
| `calcPolygons`, `EulerCycles`, `inclusionId` | `tracePolygons`, a grain is its loops, `keyhole` derived | winding from the walk order, nothing repaired |
| `@grain2d` | `grain2d`, an array of grains: `grains[i]`, `grains[mask]`, `grains['Forsterite']`, `for g in grains` | the loops, the boundary, the inner boundary |
| `@grainBoundary`, `gB.order`, `chainId`, `componentId` | `grainBoundary` in walk order; chains, junctions and neighbours derived from the row index | ADR 0002 as an unconditional invariant |
| `@triplePointList`, `gB.triplePoints` | `triplePoints(gB)`, `grains.triplePoints` derived, not stored | |
| `removeQuadruplePoints` | `calcGrains(..., removeQuadruplePoints=True)` | the pairing by the measurements alone, the face added and the segmentation run once more, the two vertices pulled apart |
| `reduce`, `simplify`, `refine` on a boundary, `reduceBoundary`, `simplifyBoundary`, `refineBoundary` on grains | the grain names on both, one merge step with two policies | `mustKeep` and `mergeRuns` |
| `smoothBoundary`, `laplaceFilter`, `taubinFilter`, `curvatureFilter`, `huberFilter` | the same names, `boundaryFilter` base | simplify, resample, smooth; a smoothing length, not a count |
| `fitEllipse`, `longAxis`, `aspectRatio`, `hull`, `paris`, `shapeFactor`, `caliper`, `paror`, `surfor`, `characteristicShape`, `@shape2d` | the same names on `grain2d` and `shape2d` | the ellipse of the second area moments in closed form, a hole with the opposite sign |
| `gradient`, `gradientX`, `gradientY`, `curvature`, `KAM`, `calcGROD`, `calcGND`, `grainMean`, `spatialProfile`, `calcMisorientation` | the same names; `gradient(ebsd, stencil='oneSided')` | one gradient through the lattice, returned as a framed `vector3d` |
| `EBSDFilter`, `meanFilter`, `medianFilter`, `KuwaharaFilter`, `splineFilter`, `halfQuadraticFilter`, `l1TVFilter`, `infimalConvolutionFilter`, `smooth`, `fill`, `fillByGrainId`, `interp` | the same names | one graph model, a smoothing length, filling as the same solve pinned |
| `@parentGrainReconstructor` and its `calc*`, `merge*`, `revert` | `parentGrainReconstructor`, the one handle class of the port | the votes as a distribution per grain and variant |
| `calcGBND` | `calcGBND` | symmetrised over the crystal group; refused where a coarsening invalidated the pixel pairs |
| `simulateEBSD` handle | `simulateEBSD(cs, ...)` a map, `addNoise`, `addStep`, `addGradient`, `addCircularSubgrain` functions | |
| `@spatialTransform` family | `affineTransform`, `projectiveTransform` | the seven other models wait |

Dropped from MATLAB, as the Julia port dropped them: `cube2hex`, `hex2cube`, `squarify`,
`hexify`, `pos2ind`, `id2ind`, `subGrid`, `subSet`, `subsind`, `subsref`, `subsasgn`,
`end`; the reconciliation of `phase` against a `CSList` of the wrong length, which is the
importer's business in P6; `dynProp` and `dynOption`; the sparse incidence matrices,
`remapIFD`, `gridComponents`, `calcGrainsOld`; the disk structuring element and its
repairs; the four mex kernels; `rotate_outer`, `plus`, `minus`, `cat`, `repmat` as method
files; `A_D` cached on the object; `loadobj`, `export_*`, `getCPRInfo`,
`selectInteractive`, and every `plot`, `quiver`, `text`, `plotUnitCells`, which are P5.

## 3. The Julia decisions and their Python spelling

The rule of `docs/PLAN.md` D2 again: a distinction that changes which algorithm runs is a
class, one that changes a parameter of the same algorithm is a field.

| Julia | Python | kept from Julia |
| --- | --- | --- |
| `Lattice{S}`, the cell shape a type | `lattice` with `shape` `'square'` or `'hex'`; the stencil and the cell neighbours read off it | the basis holds the two shortest translations, the index is `Int32`, the origin the foot of the specimen origin on the map plane |
| `MapPlane` | `mapPlane` with `origin`, `e1`, `e2` | a map out of the xy plane needs no rotation to it |
| `EBSD{T,N}` | `EBSD` with arrays of one shape, `ndim` 1 or 2; `dtype` of the rotations | not a container: `ebsd[i]` returns a sub-map; `ebsd.orientations` the framed orientations of the one phase present |
| `PhaseList` | `phaseList`, a list of frames with the not indexed frame at index 0 | `phaseId` is a 0-based index into `CSList`, `phase` the importer's number |
| `props::Dict{Symbol,Array}` | `ebsd.prop`, a dict of arrays in map shape, `ebsd.mad` through `__getattr__` | a property is one value per measurement, `meta` is what describes the map |
| `CellComplex` | `cellComplex` with `V`, `E`, `cells`, `site2pixel` as `Int32` arrays | a partition, each face once, the site pair of every face |
| `AbstractGBCriterion` subtypes | `grainBoundaryCriterion` subclasses | `connectivity` answers one, one half or zero per pair, the levels transformed once, the largest `abs(dot)` compared, never an angle in the loop |
| `Grains <: AbstractVector{Grain}` | `grain2d`, indexable, iterable, `grains[i]` one grain | the loops, `inclusionId` derived |
| `GrainBoundary` in walk order | `grainBoundary` | `grainId[:, 0]` on the left; chains and neighbours from `k ± 1` |
| boundary misorientations `Orientations{...,true}` | `orientation` with `antipodal=True` within one phase | `angle` and `project2FundamentalRegion` right without a flag |
| `EBSDFilter`, `BoundaryFilter` subtypes | `EBSDFilter`, `boundaryFilter` subclasses | one graph model for the pixels, one network for the vertices |
| `ParentGrainReconstructor` mutable | `parentGrainReconstructor`, a handle | the votes as a distribution |
| `Int32` ids and faces, `Int8` phase ids | the same NumPy dtypes | a ten million pixel map is the design point |
| threads | numba `parallel=True` in the kernels that are embarrassingly parallel | the criterion evaluation, the distance transform, the clipping |

Mutability: as in P1, property writes MATLAB scripts make (`ebsd.CSList = ...`,
`ebsd.prop.mad = ...`, `grains.meanOrientation` read only) mutate in place; `calcGrains`
writes `grainId` into the map it was given, and `ebsd.copy()` keeps the original.

## 4. The data model

`EBSD` holds `pos` (a `vector3d` of shape `(n,)` or `(nx, ny)` in the specimen frame),
`rotations` (a `rotation` of the same shape), `phaseId` (`int8`, same shape), `CSList`
(a `phaseList`), `lattice` (a `lattice` or `None`), `prop` (a dict of arrays of the same
shape), `meta` (a dict), `scanUnit`. `unitCell` and `N` are read off the lattice and the
map plane. `id` is the flat index, `x`, `y`, `z`, `xyz` views of `pos`, `isIndexed` a mask,
`indexedPhaseId` the ids of the indexed phases present, `orientations` the framed
orientations when the map holds one indexed phase and an error naming the phases
otherwise, `grainId` and `mis2mean` properties once `calcGrains` has written them.

`ebsd[idx]` slices every array and keeps the phase list, the lattice index and the
metadata; a 0-dimensional selection is a map of one measurement, not a scalar class.
`ebsd['Forsterite']` selects by phase name, `ebsd['indexed']` and `ebsd['notIndexed']`
by indexing, a `crystalFrame` by identity. Selection by a mask of map shape and by a
list of ids; `findByLocation` and `findByOrientation` return masks.

The lattice inference, the decomposition, the union-find and the chain ordering are the
kernels of this package; each has a NumPy form and a numba loop, as in `geometry/kernels.py`,
with `setMTEXpref('kernels', 'numpy')` selecting the fallback. The NumPy forms are the
oracles and must stay correct on every test; the loops are what runs.

## 5. Order of work

Numbered as in the Julia plan so that the two can be read side by side.

**M1 - the container and the lattice. DONE 2026-09-21** (`phaselist.py`, `lattice.py`, `polygon.py`, `ebsd.py`, `gridify.py`, `funcs.py`; the Julia L1 to L6 and E1 to E6 ported, the shear, drift and indexing cases of `check_ebsdGrid` and `check_ebsd`; 359 tests; the maps package adds 3 ms to the import). Three things met on the way: two inferences of one lattice picked different equivalent bases, so `reduceBasis` ends in a canonical choice, `a1` the shortest translation pointing most along the first axis of the plane and `a2` the one turning least anticlockwise from it, which is what lets `ebsd.lattice.ij` survive a shear or a drift as `check_ebsdGrid` asks; the geometry constructors cast to float64, so the map builds its vectors and rotations on the arrays as they are, which is also what keeps a map of millions from being copied; and `scipy.spatial` costs 300 ms at import, so the k-d tree is imported where it is used. `id` is a stored field, as in MATLAB, kept through selection and numbered on past the last one by the padding of `gridify`. A sheared square with six Voronoi neighbours reads as `'hex'`, the Julia rule. The neighbour graph growth of the index is a Python loop over levels and is the fallback for distorted maps only. `phaselist.py` (`phaseList`, names, colours,
`isIndexed`, the not indexed frame), `ebsd.py` (`EBSD`, the properties above, selection,
`rotate`, `shift`, `transform`, `extent`, `findByLocation`, `findByOrientation`, `copy`,
`cat`, display), `lattice.py` (`mapPlane`, `lattice` with the inference by direction, the
shape by the edges of the Voronoi cell, the index by the checked rounding and the graph
growth, `stencil`, `spacing`, `position`, `unitCell`, `latticeBasis`), `polygon.py` (the
half-plane clipping, the seed box, `voronoiCell`, `signedArea`, shared with M2),
`gridify.py` (`gridify` to map shape with `NaN` and not indexed in the holes, and back).
Tests: `test_lattice.py` with the Julia L1 to L6 (shape, basis and index for square,
rectangular and hexagonal lattices at four rotations, under a trapezoidal distortion, out
of the xy plane, in float32; the cell area against `|det A|`); `test_ebsd.py` with E1 to
E6 (the table, selection, framed orientations, `gridify` with holes, finding by location
and orientation) and `check_ebsd`, `check_ebsdGrid`, `check_find` as far as they need no
import, plotting or grains.

**M2 - the spatial decomposition. DONE 2026-09-21** (`decomposition.py` with the NumPy forms, `kernels.py` with the clipping as one numba loop over chunks of sites; the Julia D1 to D6 on both backends and the two backends against each other; 384 tests). E2 decided: on one thread numba is within 1.4 of a C loop and faster than the Julia kernel, on 12 threads at the Julia time, so P3 builds no extension (`docs/research/benchmark-decomposition.md`). The vertex weld is `np.unique` on one integer key per grid cell, twice the Julia dictionary; a nine million pixel map waits on a counting pass that sizes the face buffer exactly. `decomposition.py`: `alphaClose` as the De Morgan pair
of dilations by the disk of lattice offsets, `floodBorder`, `siteSet` classifying every
lattice cell, `localVoronoi` clipping each cell against its candidates until it closes,
`meshVertices` welding the copies of a vertex, `decompose(ebsd, alpha)` returning a
`cellComplex`. The clipping is the first numba kernel of P3 and the E2 measurement:
numba against a compiled extension on the clipping, the budget the Julia numbers, a
million pixels in 0.55 s square and 1.06 s hex end to end. Tests D1 to D6: the stencil
on square, rectangular and hexagonal lattices at two rotations, the partition, the
closing on holes, invariance under a rigid motion, the closing against the morphological
one on slits, float32.

**M3 - segmentation. DONE 2026-09-21** (`criteria.py`, `segment.py` with the union-find as a numba loop and `scipy.sparse.csgraph` as the fallback, `calcgrains.py`, `grains.py` with the grain table of `grain2d`; the Julia S1 to S10 but for the areas of S10, and from `check_calcGrainsCases` the block grids with holes and a cropped corner on both lattices with their counts under a trapezoidal distortion and a rotation, the diagonal stray pixel of the cull on both lattices, and gridify padding segmenting like not indexed pixels; 389 tests in the geometry and maps suites). The grain mean and spread run vectorised over all pixels of a phase with the reference of every pixel that of its grain, through `closestImage`. Grain ids count from 1 with 0 for a pixel without a grain, as in MATLAB, while `grains[k]` is positional. `calcGrains` takes MATLAB's `angle` as its keyword, a criterion positionally, and `threshold` as an alias. The areas of S10 and of the commutation checks, and the quadruple point cases, wait for M4 and M5. `criteria.py` (`grainBoundaryCriterion`, `gbcAngle`, `gbcSoft`,
`gbcEqual`, `gbcCustom`, `connectivity`), `segment.py` (union-find with path halving in
numba, `components`, `segment`, `faceConnectivity`, `segmentFrom`), `calcgrains.py`
(`calcGrains(ebsd, threshold, criterion, minPixel, alpha, removeQuadruplePoints)` with the
two-pass cull, `absorbInteriorPixels` by the nearest site, the grain table with phase,
`numPixel`, mean orientation and GOS). Tests S1 to S9 and `check_calcGrainsCases` for the
grain counts and the commutation with a transform and a rotation.

**M4 - boundaries. DONE 2026-09-21** (`boundary.py` with `grainBoundary` in walk order, the chain ordering as a numba loop that is also the Python fallback, chains, junctions, `triplePointList`, subsets by rows, by phase and by grain, `flip`, `curvature`, `calcMeanDirection`, `intersect`, `inpolygon`; `quadruple.py`; `calcGrains` building `grains.boundary` and `grains.innerBoundary`; the Julia B1 to B13 and Q1 to Q7 but for the areas, 410 tests). One difference from a row subset: `selectByGrainId`, which `grains[k]` uses, walks the subset again, since a junction of the whole boundary may have lost its third segment there; a plain `gB[rows]` keeps the rows as they are, as in Julia. The resampling and the smoothing followed M5 (`boundaryresampling.py`, `smoothing.py`; the Julia B14 to B17 and SM1 to SM5; 479 tests): `reduceBoundary`, `simplifyBoundary` and `refineBoundary` on a boundary, with MATLAB's `reduce`, `simplify`, `refine` as the same methods, and on the grains, which coarsen both boundaries over one pool of vertices and trace their loops again; `mustKeep` and `mergeRuns` as the one merge step, `keepGrains` the ids whose polygon may not collapse, the Douglas-Peucker recursion as the third numba loop over the flat vertex list of all chains, the arc-length resampling vectorised over all pieces as in MATLAB. One design change against both references: the vertex the walk cuts a closed chain at is an accident of the numbering, and both MATLAB and Julia keep it as if it were a junction, their tests passing only because their lowest vertex is a corner; here `mustKeep` frees the end of a closed chain, `anchorClosedChains` rotates the chain's rows to its first kept head before the merge or the resampling, and the Douglas-Peucker loop walks a closed chain from that anchor round to itself, so a square grain simplifies to its four sides whatever the numbering. `smoothBoundary(grains, iter | filter, simplify=, refine=, noSimplify=, noRefine=, moveTriplePoints=, moveOuterBoundary=)` is the three steps over a `boundaryNetwork`, the compressed neighbourhood of both boundaries with what may not move; `laplaceFilter(iter, lambda_)`, `taubinFilter(iter, lambda_, mu)`, `curvatureFilter(smoothingLength, alpha)` and `huberFilter(smoothingLength, threshold, alpha, iterMax, tol)` under `boundaryFilter`, `F.smooth(V, net, h)` and `F.weight(h)`; the Laplacian is one `scipy.sparse` matrix per call, the variational filters one Dirichlet solve. MATLAB's `lambda` is `lambda_`, a reserved word. On a 600 x 600 map of 585 grains: simplify 0.06 s, refine 0.17 s, the smoothing 0.16 s (Laplace, Taubin), 0.2 s (curvature), 0.5 s (Huber). The distance weighted kernels of MATLAB's `laplaceFilter` (`gauss`, `exp`, `umbrella`) and its `secondOrder` are left out, as in Julia. `boundary.py` (`grainBoundary` from the complex in walk order,
`chainOrder`, chains, junctions, `triplePoints`, `direction`, `midPoint`, `segLength`,
`perimeter`, `misorientation` with exchange, `selectByGrainId`, `hasPhase`, `isTwinning`,
`curvature`, `calcMeanDirection`, `angle`, `flip`, `intersect`, `inpolygon`),
`quadruple.py` (`quadruplePoints`, `splitQuadruplePoints`), `boundaryresampling.py`
(`reduce`, `simplify`, `refine` with `mustKeep` and `mergeRuns`), `smoothing.py`
(`boundaryNetwork`, `laplaceFilter`, `taubinFilter`, `curvatureFilter`, `huberFilter`,
`smoothBoundary`). The chain ordering is the second numba kernel. `gbnd.py` waits for P2
(`S2FunHarmonic`, the SO(3) density, the convolution). Tests B1 to B17, Q1 to Q7, SM1 to
SM5, `check_boundaryChains`, `check_chainOrder`.

**M5 - grains. DONE 2026-09-21** (`grains.py` with the loops traced from the walk order, `area`, `perimeter`, `boundarySize`, `centroid`, `diameter`, `equivalentRadius`, `equivalentPerimeter`, `hasHole`, `isInclusion`, `neighbors`, `numNeighbors`, `isBoundary`, `subBoundarySize`, `subBoundaryLength`, `checkInside`, `inpolygon`, `grains(x, y)`, `selectByGrainId`, `keyhole` and `poly`, `merge` by pairs, by a boundary or by a threshold writing `parentId` and carrying `childId`; the Julia G1 to G8 and the area assertions M3 and M4 had left open: `check_calcGrainsCases` commutation of the areas with a trapezoidal distortion within 5e-3 and a rotation within 1e-3 on both lattices, the closed-ring parity of every grain, positive areas with and without the quadruple point split, the Julia S10, Q3 and Q6; 431 tests). Two things met on the way and fixed in M2: a site without a measurement is now placed one lattice step per index from its nearest measured cell rather than on the affine fit of the whole map, so the rings and the holes follow a smooth distortion as the measurements do, which is what brings the distorted rim within the MATLAB tolerance; and the vertex weld missed copies of one vertex lying within rounding of a boundary of its grid, so it welds on a fine grid first, again on the half-shifted grid for the few candidates near a boundary, and then on the tolerance grid, at 0.7 s square and 1.5 s hex for a million sites. `grains.py` (`grain2d` with its loops, `tracePolygons`, `area`,
`perimeter`, `boundarySize`, `centroid`, `diameter`, `equivalentRadius`,
`equivalentPerimeter`, `hasHole`, `isInclusion`, `neighbors`, `numNeighbors`,
`isBoundary`, `checkInside`, `inpolygon`, subsets, `merge`, `keyhole`). Tests G1 to G8,
`check_grainMerge`, `check_calcComponents`.

**M6 - the shape parameters. DONE 2026-09-21** (`shape.py`, `hull.py` of G7; the Julia H1 to H11, 491 tests). The measures of M5 (`area`, `perimeter`, `centroid`, `diameter`, `checkInside`, `keyhole`...) moved from `grain2d` into the mixin `polygonSet` of `polygon.py`, what a list of outlines answers through `loops`, `V`, `plane`, `frame` and `size`; `grain2d` and `shape2d` are its two classes, so `shape2d` needs no id, phase or pixel count and no boundary with a nan misorientation, as MATLAB's does. `shape.py` attaches to the mixin: `polygonMoments` and `principalComponents` (the ellipse of the second area moments in closed form, the half axes as framed antipodal `vector3d`s of that length, as MATLAB returns them), `fitEllipse` returning `(c, a, b)`, `longAxis`, `shortAxis`, `aspectRatio`, `hullLoops` through `convexHull` (Andrew's chain as a numba loop with the Python loop as oracle, one hull per grain), `paris`, `shapeFactor`, `convexity`, `caliper` in MATLAB's four forms (none or `'longest'`, `'shortest'`, `'shortestPerp'` as framed axes of the width, a `vector3d` or an angle for the width across it), `paror`, `surfor` on a boundary and on grains, `characteristicShape` returning a `shape2d`, `grains.hull()` returning grains with convex polygons over a boundary with the outside on the other side, and `shape2d(V)`, `shape2d.byRhoTheta`, `shape2d.byFV`, `Vs`, `rho`, `theta`. All of these are methods with parentheses, as MATLAB implements them, where the measures of M5 stay properties. One choice between the references: `caliper('shortestPerp')` carries MATLAB's length, the width measured across the direction normal to the narrowest one, where Julia gave it the narrowest width itself; the direction is the same in both. The `hullArea`, `hullPerimeter`, `outerPerimeter` helpers are module functions of `shape.py`, MATLAB's `convexhullProps` split in two. On the 600 x 600 map of 585 grains every parameter runs between 0.03 s and 0.2 s. The 3D hull and the spherical Voronoi diagram of Julia's geometry `hull.jl` wait for the pole figures of P2.

**M7 - intragranular quantities. DONE 2026-09-21** (`intragranular.py`, `dislocations.py`, `slipsystem.py` of G7; the Julia I1 to I10, I9a, I9b and the slip system tests; 512 tests without the io suites). `gradient(ebsd, stencil='oneSided' | 'lattice' | 'full')` returns the pair `(gx, gy)` of framed vectors along the axes of the plane of the map, `gradient(ebsd, v)` the derivative along a specimen direction and NaN where it leaves the plane, `gradientX`, `gradientY`, `gradientZ` MATLAB's spellings of it; the three stencils are one set of normal equations over the neighbours a `mapPairs` predicate admits (indexed, one phase, one grain where the map knows its grains), assembled with `bincount` over all pixels per offset, with the symmetric equivalent of every neighbour taken through `closestImage` phase by phase. `curvature` is an `(..., 3, 3)` array with the columns along the map normal NaN, until P4 has a tensor class. `KAM(order=, radius=, threshold=, max=, weights=)` with MATLAB's `max`; `calcGROD(ebsd, grains, rotRef)`; `grainMean(ebsd, prop, grains, method, ulim=, llim=)`; `spatialProfile` returning MATLAB's pair `(ebsdLine, dist)`; `calcMisorientation` with a `rng` keyword. `slipsystem.py` holds `slipSystem` and `dislocationSystem` as one object over arrays, as MATLAB's classes are, with the static families (`slipSystem.fcc(cs)`, `basal`, `pyramidalCA`... and `hcp` refusing to guess), `symmetrise(antipodal=)`, `SchmidFactor(v, relative=)`, `deformationTensor`, `SchmidTensor`, `mPrime`, `residualBurgersVector`, `trace`, `ori * sS`; `dislocationSystem(sS)` builds the edge systems with energy 2 and the screw ones with energy 1 over the half or third Burgers vector, `isEdge`, `isScrew`, `tensor()`. For `ori * sS` to reach the system's own `rotate`, the quaternion and orientation products now return NotImplemented for any object with a `rotate` method, so `rot * ebsd` works the same way. `dislocations.py`: `fitDislocationSystems(kappa, dS)` and `calcGND(ebsd, dS, stencil=)` as the linear program of least line energy over the known components of the curvature, solved per pixel with SciPy's HiGHS. On the 600 x 600 map: the one sided gradient 0.37 s, the full stencil 1.3 s, KAM 0.65 s (order 3: 3.4 s), GROD 0.1 s; `calcGND` runs at 1.8 ms per pixel, MATLAB's per-pixel `linprog` shape, and wants a batched solve or P4's own simplex before it is used on a whole map. `weightedBurgersVec` waits for M10.

**Benchmark, 2026-09-21.** `docs/research/benchmark-maps.md`, `bench/maps.py` against `bench/maps.jl` on one synthetic map: after the fused per-pixel kernels (gradient, KAM, GROD, the simplex of the dislocation fit), the compiled loop tracing and hull, the memoised chain structure and the bincount shape measures, every verb of M3 to M7 is at Julia's 12-thread time or below it, except `calcGrains` at 1.5 times Julia's single thread and a few millisecond shape measures. `calcGND` went from a hundred times behind to ahead.

**M8 - denoising and repair. DONE 2026-09-22** (`denoising.py`, `variational.py`; the Julia D1 to D9; 684 tests). `smooth(ebsd, F=l1TVFilter(), fill=, extrapolate=, props=)`, `fill(ebsd, F, extrapolate=)` (MATLAB's name for Julia's `fillMissing`), `fillByGrainId`, `reduce(ebsd, fak)` (MATLAB's name, `subsample` the alias, the sublattice rebuilt fak times coarser) and `interp(ebsd, pos | x, y, F=, fill=, extrapolate=)`; the filters `meanFilter`, `medianFilter`, `KuwaharaFilter` with MATLAB's `numNeighbours`, `splineFilter`, `halfQuadraticFilter` with MATLAB's `alpha`, `eps`, `threshold`, `iterMax`, `tol`, `l1TV`, `l1DataFit`, `l1TVFilter`, `infimalConvolutionFilter` with `alpha` and `beta` (its two weights are derived from the smoothing length as Julia has them, not MATLAB's `lambda` and `mu`), every variational one taking a smoothing length first as the boundary filters do. The Julia design throughout: every filter is a vector field on the graph of the map within the components (grains or phases), `smoothingGraph` deciding the components, the edges and which pixel a hole is solved from, `tangentField` the field at the mean of every component; filling is the same model with the data term off and the measurements pinned; the properties are fields on the same graph. Two things changed against Julia for the machine: the variational solve runs over the whole graph at once instead of component by component, since the edges never cross a component and the one system is block diagonal in them, and the surrogate of every IRLS step is solved by a Jacobi preconditioned conjugate gradient kernel over the CSR graph (`_cgLoop`, three passes over the map per step) rather than a Cholesky per component, with SuperLU on the whole system as the fallback and oracle, agreed to 1e-6 degrees. The one thing a global solve must keep from the per-component one is that every component stops on its own: on the benchmark map the median grain converges in four IRLS steps and four grains run to the cap of a hundred, in both ports, and a global stopping rule would run every grain to the cap; so the IRLS freezes a component once nothing in it moves by more than `tol`, and the norms, the weights, the operator and the conjugate gradient skip frozen pixels, so a step costs what the active pixels cost. A component the weights cut off from every measurement is left at the mean of the previous iterate by both solvers. The window filters, the tangent field, the exponential and the reweighting are kernels too. On the 600 x 600 map against Julia's single thread, both measured on a machine at a load of two to four (the mean filter read 0.23 s there against 0.12 s quiet): mean 0.12 s quiet (Julia 0.12), median 0.17 (0.16), Kuwahara 0.12 (0.26), spline 0.36 (1.26), l1TV 1.79 s loaded against Julia's 1.56 s under the same load and 1.37 s quiet, so about parity; its remaining cost after the first twenty steps is the per-step overhead of the kernel launches and the gathers over the whole map for the four grains still active, about 9 ms a step against Julia's 3 ms.

**M9 - parent grain reconstruction. DONE 2026-09-23.** `reconstruction.py` and `variantgraph.py`, MATLAB's
`@parentGrainReconstructor` (2500 lines, the one handle class of the port) against Julia's
`reconstruction.jl`; the doc pages of `PhaseTransitions` use all of it, so the MATLAB surface
is ported but for `selectInteractive`, `plotVariantPF`, `calcGrainVotes` (no page calls it, and
its reshape of the fits is broken in MATLAB), the `includeTwins` option of the clustering and
`peakFitting` of `calcParent2Child`. The steps change the job and return nothing.

- `parentGrainReconstructor(ebsd, grains, p2c, useBoundaryOrientations=)` with the phases
  guessed as MATLAB guesses them (the most frequent indexed phase the child, the rarest the
  parent, even at zero pixels), the map projected next to the grain means, the parent frame
  appended to the phase lists of the job's copies; the MATLAB properties (`csParent`,
  `csChild`, `parentPhaseId`, `childPhaseId`, `numChilds`, `isChild`, `isParent`,
  `isTransformed`, `isMerged`, `parentGrains`, `childGrains`, `transformedGrains`, `ebsd`,
  `parentEBSD`, `variantId`, `packetId`, `bainId`, `parentId`, `variantMap`, `c2c`) and its
  display with the fit quintiles and the closest ideal OR.
- one extraction of the grain pairs (c2c, p2c with the parent first, p2p) with the grain
  means or, with `useBoundaryOrientations`, the means of their pixels along the boundary, each
  moved next to its grain mean, a child pixel of a grain that is a parent now counted with the
  parent it gives nearest that mean (MATLAB computes the whole parent map for this).
- `calcGBFit`, `calcParent2Child` (the p2c average and G7's fit of the c2c misorientations,
  combined by count, `variantMap` kept), `calcGBVotes` (`p2c`, `c2c`, `reconsiderAll`,
  `curvatureFactor`, `bestFit`, `count`, `minFit`, `maxFit`, `numFit`), `calcTPVotes`,
  `calcParentFromVote` (`minProb`, `minDelta`, `minFit`, `minAlpha`, `minCount`, `maxCount`),
  `mergeSimilar`, `mergeInclusions`, `revert` in its four cases, `calcParentEBSD` (`exactOR`,
  per pixel `variantId`, `packetId`, `fit`), `calcVariants`, `parent2ChildInfo`;
  `calcGraph`, `clusterGraph` over `mclComponents`, `calcParentFromGraph`, `mergeByGraph`,
  `plotGraph`, `calcVariantGraph`, `clusterVariantGraph` (`includeSimilar`, `mergeSimilar`,
  `noPseudoDiagonal`, `sym`, `keepGraph`).
- `grain2d.merge` takes a label per grain, a graph of grains, `inclusions`/`host`/`maxSize`
  (an inclusion merges across a phase boundary and the grain keeps the host's phase and
  orientation, MATLAB's `'inclusion'` rule) and `testRun`; `grains.inclusionHost` is
  MATLAB's second output of `isInclusion`, `grains.update()` reads the boundary phases off
  the grains; the threshold merge is one vectorised angle per phase.

The votes are MATLAB's table (decided with the user): the best `numFit` variants per grain,
sorted, `job.votes.parentId` and `.prob` (or `.fit`, `.count`), accumulated as MATLAB does,
so a `minProb` of a MATLAB script means the same. Julia's normalised distribution is not
taken. Four things had to be MATLAB's exactly for the numbers to agree:

- MTEX shadows MATLAB's `quantile` with its own, the `round(n q)`-th sorted value without
  interpolation; the votes take the 0.875 quantile per grain and variant with it, the OR
  refinement and the display too.
- `probMax .* prob ./ sum(prob, 2)` divides by the sum of the unscaled probabilities.
- the variants are numbered as MTEX numbers them: the proper group enumerated as the
  products of the cyclic groups of its Laue class's generators (first factor fastest), the 24
  child variants of a cubic parent reordered to Morito's numbering, and a variant id counted
  from the parent's representative in the fundamental region (`mtexEnumeration` and
  `variants` in `standardorientations.py`; G7's test built its children from other
  representatives). Before this, packets came out 4865/1725/1812/2201 instead of MATLAB's
  2445/2434/2852/2867.
- `calcParent` sorts the variant pairs and triples column-major, the first index fastest, which
  is how MATLAB breaks the many exact ties of the KS pairs.

Ties the port cannot reproduce: for Kurdjumov-Sachs up to four parent candidates fit a pair of
variants alike, and which two a pair keeps depends on which grain of the pair comes first,
that is on the order the segmentation numbers the grains in, which differs between the ports.
The synthetic map of the tests is built so that no grain depends on such a tie (the left
parent is variants 2, 7, 14, the Julia map's 1, 2, 3 has two parents that fit alike), and there
all six grains carry MATLAB's variant ids and probabilities to 1e-12.

The angle between the parent candidates of two children depends only on their
misorientation, so `variantAngles` in `geometry/kernels.py` is one compiled loop, `PV_k^-1 m
PV_l` reduced over the parent group with the angle taken from the vector part (acos loses the
ties near zero), the NumPy blocks the fallback and oracle; `calcParent`'s pairs and triples
(the triples in blocks of 256) and the variant graph use it.

Martensite map of MaParentGrainReconstruction against the MATLAB bridge
(`bench/reconstruction.py --matlab-p2c --routes`, `bench/matlab/probe_reconstruction.m`,
`probe_votes.m`), with MATLAB's refined OR so that G7's own fit (3.50 degrees from KS against
MATLAB's 3.42, lower misfit quintiles) does not enter: fit quintiles, variant graph entries
402272 (401934, the port has 9 more grains), cluster votes to four digits, parents after the
vote 10514 (10508), after the growth 10578 (10573), grains after `mergeSimilar` 437 (433) and
`mergeInclusions` 329/280 parents (325/280), packets 2448/2433/2853/2869 (2445/2434/2852/2867),
parent map fit quintiles to four digits, boundary votes [0.2967 0.419 0.5349 0.6554]
([0.2966 0.4188 0.5349 0.6551]), triple point votes [0.1001 0.2072 0.3398 0.5213] ([0.0997
0.2074 0.3397 0.5213]), `calcParentFromGraph` 10538 (10532) parents. Times, Python against
MATLAB: `calcVariantGraph` 0.46 s (4.8), `clusterVariantGraph` 3.6 (5.4), `mergeSimilar` 0.16
(0.40), `calcParentEBSD` 1.0 (1.15), `calcGBVotes` 2.1 (5.9), `calcTPVotes` 27 (30),
`calcParentFromGraph` 33 (96), `calcParent2Child` 17 (19.5).

Tests `tests/test_reconstruction.py` (the Julia R1 to R7 on the synthetic two-parent map, R2
pinned to MATLAB, R2b the growth, the accumulation of the votes), `tests/test_variantgraph.py`
(R4, R4b, the vote restriction, MCL, the grain graph route, the compiled angles against their
oracles), `tests/test_grains.py` G8b for the merges.

**M10 - the rest. DONE 2026-09-26** (`gridify.py`, `geometry/gridlayout.py`, `pseudosym.py`,
`simulate.py`, `geometry/spatialtransform.py`; `tests/test_gridify.py`, `test_gridlayout.py`,
`test_pseudosym.py`, `test_simulate.py`, `test_spatialtransform.py`). `gridify(ebsd, gL)` and
`gridify(ebsd, unitCell=uC)`, the resampling by the nearest measurement with MATLAB's lattice
order and layout, `return_index` for MATLAB's `newId`, `ebsd.layout`, `dPos` as MATLAB's
shortest cell edge; `cleanUpPseudoSym` in place as `calcGrains`, MATLAB's three outputs;
`simulateEBSD` as MATLAB's class with an `rng`; the `spatialTransform` family of MATLAB's
frameSymmetry branch (the `affineTransform` and `projectiveTransform` of this plan's first
draft are `spatialTransformShift` and `spatialTransformProjective` there), the homography
fitted after Hartley's normalisation; MCL segmentation came along as `calcGrains(..., mcl=)`.
EBSDGrid, EBSDInter, EBSDPseudoSymmetry, EBSDSimulation, EBSDSpatialTransform and
GrainReconstructionMCL are checked against MATLAB (`docs/findings.md` has what differs).

**M11 - TrueEBSD and map images. DONE 2026-09-26** (`maps/mapimage.py`, `maps/registration.py`,
`maps/trueebsd.py`, `plotMapImage` in `plotting/mapplot.py`, `gridLayout.assumedFor`, the
`trueEbsdWCCoSmall` recipe in `io/sampledata.py`; `tests/test_mapimage.py`, `test_registration.py`,
`test_trueebsd.py`, one drawing test in `test_mapplot.py`). The EBSDTrueEbsd page runs as MATLAB's
to the digit (`bench/matlab/probe_trueebsd.m`): the measured settings, the retry of the first hop,
every tile's shift, the peaks to 1e-14, the resized and undistorted images to 1e-13, and the
undistorted map's phases, positions and rotations identical. Decisions on the way: a list of
images is a `mapImageList`, a list displaying as MATLAB's table, and the hops a
`spatialTransformList` likewise; `xcfShift` keeps MATLAB's tile arithmetic counted from one
internally and answers in positions counted from zero; the EBSD map is resampled by its nearest
measurement within the reach of its cell (`resampleNearest`, frameSymmetry's `EBSD/interp`, not the
port's tangent space `interp`); the job's verbs return nothing, as `parentGrainReconstructor`'s.
The upsampling DFT of all tiles is two GEMMs, the band placed directly into the doubled array.
Times against MATLAB: `pixelSizeMatch` 0.07 s (0.31), `calcDistortion` 2.7 s (2.2), `undistort`
0.7 s (0.4). Both pages checked.

The plan as it was written: unblocks EBSDTrueEbsd and EBSDMapsAndImages, the last two
blocked EBSDAnalysis pages. MATLAB only (`../frameSymmetry`, about 2350 lines); the Julia
port has none of it, so the design follows MATLAB's classes, which are already split along
the right lines. The distortion models are M10's `spatialTransform` family and stay as they
are; M11 adds what produces their point pairs and what applies them.

The pieces, in the order they depend on each other:

1. `mapImage` (MATLAB `EBSDAnalysis/@mapImage`, 11 files): an image on a REGULAR grid, an
   origin and two perpendicular step vectors with the positions derived, never stored; k
   channels, integer images scaled to [0, 1]. Constructors `mapImage(img, dxy=, origin=,
   name=, scanUnit=)` and `mapImage(ebsd.bc, ebsd)` (a channel of a gridded square map, the
   map riding along). Methods: `pos`, `layout` (M10's `gridLayout`), `how2plot`, `dx`, `dy`,
   `extent`, `pos2ind`, `interp(mg, pos)` (bilinear on the regular grid, NaN outside),
   `subGrid`, `rescale`, `imboxfilt` (SciPy's `uniform_filter`), `edgeMap` (the edge
   transform registration runs on), `transformReferenceFrame(mg, frame, 'byScreenAlignment')`,
   display and `plot` (one image, or a list with `layout=`). A list of images is a Python
   list, as M10's hops are. File: `maps/mapimage.py`; the drawing in `plotting/mapplot.py`.
2. `xcfShift(A, B, ROISize=, numROI=, XCFMesh=)` (MATLAB `tools/registration_tools`): tile-wise
   phase correlation of two images on one grid, the peak refined by the matrix-multiply DFT
   upsampling of Guizar-Sicairos et al. (XCFMesh 250); only where both images are finite;
   returns `(u, peak, pos)`, in specimen units for two `mapImage`. With it `pairShifts`, the
   shifts one fit stage was fitted to (`tools/registration_tools/@pairShifts`). File:
   `maps/registration.py`. Checked against MATLAB tile for tile on two images of the data set
   (the sub-pixel shifts to 1e-3 pixel).
3. `trueEbsd(imgList, T)` (MATLAB `EBSDAnalysis/@trueEbsd`, 7 files and 5 private): the job
   over a chain of images, most distorted first, one unfitted transform per hop (`+` for a
   staged hop, never `*`, which would drop an identity prototype). A handle class in MATLAB;
   here an object whose methods alter it and return it, as `parentGrainReconstructor`.
   `setOptions(n, roiSize=, numROI=, registerOn='edge'|'raw', edgeWidth=, highContrast=)`,
   `pixelSizeMatch` (every map onto one grid), `calcDistortion(fitErr=, silent=, retryMax=)`
   (per hop and stage: correlate, fit the stage by `fitStage`, resample, measure again; the
   ROI doubled on retry), `undistort` (the hops composed and every map resampled into the
   ground truth's grid; the EBSD map comes out with the images as per-pixel properties), the
   properties `resizedList`, `undistortedList`, `shifts`, `fitError`, and the progress table.
   The private `autoTune`, `fitTo`, `measureShift`, `resample` follow. File:
   `maps/trueebsd.py`.
4. The pages: EBSDMapsAndImages first (it needs only `mapImage`, layouts and `interp`), then
   EBSDTrueEbsd, each against `docs/matlabrun.py`.

Open before starting:

- The sample data (decided 2026-09-26): `mtexdata('trueEbsdWCCoSmall')` is read from an HDF5
  file of plain arrays, `~/.mtex/data/EBSD/trueEbsdWCCoSmall.h5` (3.7 MB), written through the
  bridge by `bench/matlab/export_trueebsd.m` from MATLAB's `.mat` (a v7.3 file holding an EBSD
  object, which h5py cannot unpack). The original `.h5oina` is not on this machine; the small
  set is `../trueEBSD/tools/makeWCCoSmall.m`'s centred half, every fourth point, of
  `../master/data/EBSD/trueEbsdWCCo.mat` (Zenodo 13870131, a v5 file of 180 MB). The file
  holds `/ebsd/x`, `/ebsd/y`, `/ebsd/rotation` (a, b, c, d), `/ebsd/prop/<name>` (bc, bs,
  bands, MAD, quality, oldId), `/ebsd/phaseId` (a list in MATLAB's column order), attributes
  `scanUnit` and `how2plot`; `/phases/<k>` with `mineral`, `pointGroup`, `axes`, `angles`,
  `alignment`; `/images/fsdB3` (RGB), `fsdT3`, `fsdT1`, `fsdT10` and `pixSzImg`. h5py reads
  MATLAB's arrays transposed: the map is 96 x 128 and the images 192 x 256 in MATLAB's order.
  The recipe goes into `io/sampledata.py` as a local file; where it is hosted for
  `mtexdatafile` to fetch is still open.
- Speed: `calcDistortion` correlates 24 x 18 tiles per stage and hop on images of about
  1000 x 1000; the FFTs go through `scipy.fft` with the `threads` preference, and the
  upsampling DFT is two small matrix products per tile. Time it against MATLAB on the page.
- Which of MATLAB's defaults are measured on this one data set (`autoTune`,
  `highContrast='auto'`, `PARAMETERS.md` in the MATLAB folder) - port them as they are and
  say so in the page, rather than re-deriving them.

Tests: `tests/test_mapimage.py` (the grid, `interp` exact at pixel centres, `pos2ind` round
trip, a channel of a map, the layouts), `tests/test_registration.py` (a known sub-pixel shift
recovered from a synthetic image pair, the peak as weight, NaN padding excluded; the MATLAB
tiles pinned where the data folder is present), `tests/test_trueebsd.py` (a synthetic chain:
an image, a drifted and sheared copy, a tilted copy; the fitted hops recover the imposed
transforms and `undistort` returns the ground truth's grid).

Each step ends with the import-time measurement and the test count, as in P1, and each
kernel with its budget row against the Julia numbers.

## 6. Source layout

```
mtex/maps/
  phaselist.py        phaseList
  ebsd.py             EBSD, selection, properties, rotate, shift, transform
  lattice.py          mapPlane, lattice, inference, index, neighbours
  polygon.py          clipping, voronoiCell, signedArea, the polygonSet mixin of the measures
  gridify.py          gridify and back, resampling
  decomposition.py    the alpha closing, the cell complex
  criteria.py         the grain boundary criteria
  segment.py          union-find, segmentation
  quadruple.py        quadruple points
  calcgrains.py       calcGrains, minPixel, the grain table
  grains.py           grain2d, polygons, merge, subsets
  boundary.py         grainBoundary, chains, triple points
  boundaryresampling.py  reduce, simplify, refine
  smoothing.py        boundary network, boundary filters, smoothBoundary
  shape.py            shape parameters, shape2d
  intragranular.py    gradient, curvature, KAM, GROD, GND, profiles
  dislocations.py     fitDislocationSystems
  denoising.py        EBSD filters, fill, interp
  variational.py      the graph solver
  reconstruction.py   parentGrainReconstructor, the votes
  variantgraph.py     the grain graph and the variant graph, MCL
  gbnd.py             calcGBND
  simulate.py         simulateEBSD
  mapimage.py         mapImage, mapImageList, the image verbs
  registration.py     xcfShift, pairShifts
  trueebsd.py         trueEbsd, the automatic settings
  kernels.py          the numba loops with their NumPy forms: clipping, union-find, loop tracing, gradient, KAM, GROD, the simplex, the window filters, the graph CG
tests/test_<file>.py  one per source file
tests/fixtures/maps/  the small fixture maps
```

`mtex.maps` imports from `mtex.geometry` only (the denoising and the dislocation fit moved
to `mtex.analysis` in the layout of 2026-09-22), and from `mtex.functions` inside the two
functions that need it, `calcGBND` and the GBND kernel.

## 7. Experiments this subplan depends on

- **E2, the grain kernels.** Decided per `docs/PLAN.md` D5 in M2 on the clipping and in
  M3 on the union-find: numba against a compiled extension, budget the Julia table. If
  numba reaches the budget within a factor of two the extension is not built.
- **E5, the general triangulation.** Only for scattered maps, the alpha complex variant
  and the hulls. `scipy.spatial.Delaunay` and `ConvexHull` are available and cost
  nothing; the hull of M6 is Andrew's chain in `hull.py` regardless, so that a grain list
  does not call Qhull per grain.
- **Real maps.** The tests are synthetic through M3. A real map enters with M4, when the
  chain invariants want a map with holes and not indexed regions; a minimal ctf reader is
  pulled forward from P6 then, and the maps of `../master/data/EBSD` become fixtures under
  `tests/fixtures/maps/`.

## 8. Tests

- One test file per source file; every `check_*` named in the header becomes a test of
  the same name with the same assertions, or a line in `tests/TODO.md` with the reason.
- Maps are synthetic and small: a block map of known grains on both lattices, holes of
  known size, a distorted grid, a two-phase map, a map with not indexed regions. The one
  MATLAB case that needs `mtexdata small` becomes a fixture.
- The invariants: the segmentation commutes with a rigid motion and a smooth distortion;
  the grain count does not depend on the order of the measurements; every pixel has one
  grain id or zero; the chains close; the polygons wind positively and their areas sum to
  the map area; `grains.boundary` and the pixel adjacency agree.
- Numbers pinned from MATLAB through the bridge on request only, never a run above five
  minutes: grain counts and boundary lengths of the fixture map at a few thresholds.
- Every kernel test runs the NumPy form and the numba loop against each other.

## 9. Left out of P3

As in Julia: 3D (`EBSD3`, `grain3d`, `grain3Boundary`, the 3D `calcGBND`, `neper`);
registration (`mapImage`, `trueEbsd`, `pairShifts`, the seven other spatial transforms);
import and export (P6); plotting and colour keys (P5); `calcTensor` (P4); the distance-weighted Laplace kernels; generalised cross
validation for the spline filter; the `maxFit` guard of `mergeInclusions`.

Where the Julia port changed a MATLAB result and this port follows it: the alpha closing
by distance transform at integer alpha; the pixel absorbed by the closing keeps its phase;
`minProb` of the reconstruction is a share, not a sum; the GBND is symmetrised over the
crystal group and refused on invalidated pixel pairs; the ellipse is the one of the second
area moments; the boundary filters take a smoothing length; the smooth and fill filters
are one model.

## 10. Open items to settle during M1 to M3

- Whether `ebsd.phaseId` is written 0-based into `CSList` (decided yes, 2026-09-21) and
  what `ebsd.phase` returns for a synthetic map without an importer (the same number).
- Whether `calcGrains` returns the grains alone with the grain id written into the map
  (Julia) or the MATLAB pair. Decided: the grains alone; `ebsd.grainId` after the call.
- Float32 maps: the rotations decide the dtype; positions follow; the kernels are
  compiled for both.
