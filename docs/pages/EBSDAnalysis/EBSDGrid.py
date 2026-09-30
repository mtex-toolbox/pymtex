# # Gridded EBSD Data
#
# Most EBSD maps are measured on a square or hexagonal scan lattice. By default,
# [EBSD.load](EBSD.load.html) keeps that structure: it returns a map on its square or
# hexagonal grid whenever every measurement fits on one lattice. Otherwise it keeps the
# measurements as a plain `EBSD` list and explains why.
#
# This page assumes basic [EBSD selection](EBSDSelect.html) and
# [plotting](EBSDPlotting.html). See [Select by Index](EBSDIndex.html) first if row and
# column indexing is unfamiliar.
#
# A scan lattice, a matrix layout, and a reference frame answer different questions. The
# lattice says which measurements are neighbours. The layout says which specimen
# directions the matrix indices follow. The reference frame says which axes the positions
# and orientations are expressed in.

import numpy as np

from mtex import *

plottingConvention.default('y↑→x')
ebsd = mtexdata('twins')
ebsd

# The summary identifies `ebsd` as a square grid. Its 22879 measurements form a 137 by
# 167 matrix, with one entry per scan position. Apart from its matrix shape, it behaves
# like any other `EBSD` variable.

plot(ebsd['Magnesium'], ebsd['Magnesium'].orientations)

# The rectangular outline follows the square measurement grid. The four corners of one
# pixel give the same information directly.

ebsd.unitCell

# ## What a Grid Is Good For
#
# * Per-pixel data can be handed to image-processing and registration tools as a matrix,
#   and `ebsd[i, j]` addresses one scan position.
# * [Plotting](EBSDPlotting.html) and [denoising](EBSDDenoising.html) are considerably
#   faster because the raster does not have to be reconstructed.
#
# Matrix indexing means what it says. The measurement in row 49 and column 99, counted
# from zero, is

ebsd[49, 99]

# In the default layout, the first matrix dimension follows the grid direction closest to
# y. The second follows the direction closest to x. Both indices advance towards
# increasing coordinates. Thus `ebsd[0, 0]` is the corner with the smallest coordinates,
# and `ebsd[i, j]` is the j-th pixel in the i-th scan row.
#
# This layout belongs to the map, not to the file traversal order. It is the same
# whichever corner the acquisition started from.
#
# A stored matrix is not required nearly as often as it once was. The
# [orientation gradient](EBSD.gradientX.html), [curvature](EBSD.curvature.html),
# [GND](EBSD.calcGND.html), and [fill](EBSD.fill.html) operate on the virtual lattice
# derived by [lattice](EBSD.lattice.html). They also work on plain lists, phase subsets,
# and arbitrarily aligned maps.

# ## Choosing the Layout
#
# A matrix is useful for image processing only when it is stored the same way round as
# the image. A forescatter or BSE image follows the order in which its detector wrote the
# pixels. That order need not match the map. [Maps and Images](EBSDMapsAndImages.html)
# compares a real map with SEM images of the same area.
#
# `'columnMajor'` and `'rowMajor'` are the two layouts aligned with x and y. Both are
# [gridLayout](gridLayout.gridLayout.html) objects. A layout names the row direction
# first, in the same order as the shape of the array. Suppose an image's rows run against
# x and its columns run along y, as for a detector mounted a quarter turn from the scan.

gL = gridLayout(-xvector, yvector)

# Handing that layout to `gridify` changes the matrix shape and ordering.

ebsdI = gridify(ebsd, gL)
ebsdI

# The map records the layout in which it is stored.

ebsdI.layout

# Every per-pixel property follows the same layout. For this pair of layouts, band
# contrast is related by a quarter turn.

np.array_equal(ebsdI.bc, np.rot90(ebsd.bc))

# No value was resampled or invented. A transpose and flips are sufficient, so conversion
# back to the original layout is exact.

np.array_equal(gridify(ebsdI, ebsd.layout).bc, ebsd.bc)

# Layout changes storage, not the specimen. The positions and orientations are unchanged,
# and the plotting convention still draws both maps in the same specimen frame.

# +
newMtexFigure(layout=[1, 2])
plot(ebsd, ebsd.bc)
mtexColorMap('gray')
mtexTitle('columnMajor')
nextAxis()
plot(ebsdI, ebsdI.bc)
mtexColorMap('gray')
mtexTitle('row against x, column along y')
# -

# The same features occupy the same screen positions in both panels. Only the labels
# describe a different order in memory.

# ## Data That Cannot Be Put on a Grid
#
# A rectangular raster is faithful only when every measurement occupies a distinct site
# of one lattice. If two measurements land in the same cell, one would be lost. MTEX
# therefore keeps such data as a list and reports the problem rather than silently
# dropping a measurement.

ebsd = EBSD.load(mtexdatafile('eclogite'), eulerCorrection=rotation.byAxisAngle(zvector, 180 * degree))
ebsd

# The warning above says why the data stays a list. MTEX also keeps a list when the
# positions are too irregular to span a sensible raster. You may request a list
# independently of the data, either for one import,
#
#     ebsd = EBSD.load(fname, grid=False)
#
# or for the whole session.
#
#     setMTEXpref('gridifyOnImport', False)
#
# You may change representation later. `ebsd.flatten()` flattens a gridded map into a
# list, while [gridify](EBSD.gridify.html) puts a list on its grid.

# ## Selecting a Subset Drops the Matrix Shape
#
# Selecting a phase, region, or indexed measurements usually leaves a shape that is not
# rectangular. The result is therefore a plain list, although every retained measurement
# still lies on the original lattice.

# +
ebsd = mtexdata('twins')

ebsdMg = ebsd['Magnesium']
ebsdMg
# -

# Reapplying [gridify](EBSD.gridify.html) restores the matrix shape.

ebsdMg = ebsd['Magnesium'].gridify()
ebsdMg

# The variables `ebsd` and `ebsdMg` differ at the 46 positions that were not indexed. In
# `ebsd` those are real measurements in the separate `'notIndexed'` phase: a diffraction
# pattern was recorded but could not be indexed. Selecting magnesium removes those
# measurements and creates gaps, meaning missing sites within scan lines.
#
# After `gridify`, each gap is an empty lattice site in `ebsdMg`. Its orientation is
# `NaN` and its id lies beyond the ids of the parent map, because no selected measurement
# belongs there. It is not a notIndexed measurement.

[np.sum(ebsd.id >= ebsd.size), np.sum(ebsdMg.id >= ebsd.size)]

# The empty sites complete the rectangle. Row and column ranges can therefore select and
# plot a rectangular subregion.

plot(ebsdMg[49:100, 4:100], ebsdMg[49:100, 4:100].orientations)

# ## Gridding Reorders the Measurements
#
# [gridify](EBSD.gridify.html) does not preserve input order, and generally cannot. The
# layout fixes the first matrix dimension to y, whereas `.ctf` and `.ang` files usually
# write x fastest. Consequently, the k-th entry of the flattened grid is generally not
# line k of the input file.
#
# Nothing is lost. The `id` of every measurement keeps its number in the imported list,
# the record MATLAB keeps as the property `oldId`. Its upper left corner shows
# consecutive ids running along matrix rows.

ebsd.id[0:3, 0:4]

# The index `return_index=True` adds, MATLAB's second output of `gridify`, translates in
# the other direction. In particular, `ebsdGrid.pos` at `newId` returns the gridded
# positions in the order of the input list.

# +
ebsdGrid, newId = gridify(ebsd.flatten(), return_index=True)

np.array_equal(ebsdGrid.pos.flatten()[newId].data, ebsd.flatten().pos.data)
# -

#
# This distinction matters only to code that depends on input order. Grain reconstruction
# does not: [calcGrains](EBSD.calcGrains.html) returns the same grains from the list and
# from the map.

# ## The Gradient
#
# The orientation gradient, the incomplete Nye tensor, and the gradient form of the
# weighted Burgers vector are computed on the virtual lattice. They therefore do not
# require a stored matrix. The default integral form of the weighted Burgers vector is a
# raster algorithm and grids internally.
#
# A grid does no harm, so this example continues with `ebsdMg`. The result has the same
# matrix shape as the map.

# +
gradX = ebsdMg.gradientX()

plot(ebsdMg, norm(gradX))
setColorRange([0, 4 * degree])
# -

# The colour field follows the same rectangular raster, including its empty sites. No
# separate gridding step was needed for the derivative itself.

# ## Hexagonal Grids
#
# The same principles apply to a hexagonal scan. MTEX imports it on its hexagonal grid
# and provides the same row and column indexing.

# +
ebsd = mtexdata('copper')

grains = calcGrains(ebsd)

ebsd
# -

# ---

plot(ebsd[0:20, 0:40], ebsd[0:20, 0:40].orientations, micronbar=False, edgeColor='black')

# The black cell edges reveal the alternating half-step offset between scan rows. The
# matrix is rectangular even though its pixel footprints are hexagons.

# ## Switching from a Hexagonal to a Square Grid
#
# Some external image-processing tools require square pixels. Passing a square `unitCell`
# to [gridify](EBSD.gridify.html) resamples the hexagonal measurements onto such a grid.

# +
# define a square unit cell
unitCell = 2.5 * vector3d([-1, -1, 1, 1], [-1, 1, 1, -1], 0)

# use the square unit cell for gridify
ebsdS = ebsd.gridify(unitCell=unitCell)
ebsdS
# -

# ---

# +
# visualize the result
newMtexFigure(layout=[1, 2])
plot(ebsd, ebsd.orientations)
nextAxis()
plot(ebsdS, ebsdS.orientations)
# -

# This operation differs fundamentally from restoring a map's own lattice. The new grid
# sites do not coincide with the measured ones, so every site copies the values of the
# nearest measurement. The left panel shows the measured hexagons; the right shows the
# square pixels after resampling.
#
# The square cell above has approximately the same size as the hexagonal one. Squares
# cannot reproduce hexagonal outlines, so grain boundaries in the right panel are visibly
# staircased. A smaller square cell reduces the size of the steps.

# +
# a smaller unit cell
unitCell = 0.5 * vector3d([-1, -1, 1, 1], [-1, 1, 1, -1], 0)

# use the small square unit cell for gridify
ebsdS = ebsd.gridify(unitCell=unitCell)
ebsdS
# -

# ---

# +
plot(ebsdS, ebsdS.orientations)
hold(True)
plot(grains.boundary, lineWidth=2)
hold(False)
# -

# The smaller cells follow the original boundary more closely. They do not add spatial
# resolution: every new orientation still comes from a nearest measured hexagon.
# [Regridding and Interpolation](EBSDInter.html) develops this distinction in detail.
#
# Pixels that remain notIndexed correspond to measurements that were not indexed in the
# source map. [fill](EBSD.fill.html) may replace them by interpolation;
# [smooth](EBSD.smooth.html) offers more sophisticated methods; see
# [Filling Missing Data](EBSDFilling.html).

# ## Rotated Grids
#
# Rotating a map rotates its unit cell together with its positions. The lattice therefore
# survives, and the map remains on its square or hexagonal grid. Nothing needs repair or
# interpolation.

ebsdR = rotate(ebsd, 20 * degree)
ebsdR

# ---

plot(ebsdR, ebsdR.orientations)

# The map is turned on screen, while the object summary still identifies a hexagonal
# grid. If the same data arrives as a plain list, `gridify` recovers the rotated lattice
# without requiring axis alignment.

gridify(ebsdR.flatten())

# ## Robustness to Distorted Grids
#
# EBSD is measured on a tilted specimen. Seen from a finite working distance, the far
# edge is farther away and appears smaller. The measured positions can therefore depart
# smoothly from any single rigid lattice. MTEX still reconstructs the underlying grid
# indices of an [EBSD](EBSD.EBSD.html) object. This matters to `gridify` and to every
# operation that needs neighbours, including [calcGrains](EBSD.calcGrains.html) and the
# cell drawing of a map.
#
# Grid reconstruction uses a local deformation model. MTEX first fits an ideal affine
# grid. It then interpolates the local deviation between the measured positions and that
# grid wherever a cell has no measurement. Thus a gap, a notIndexed hole, and the dummy
# cells used to bound the map follow the measured distortion rather than an unrelated
# rigid lattice.
#
# These terms are distinct. A gap is a run of measurements removed from a scan line, such
# as by selecting one phase. A hole is a connected notIndexed area inside the scanned
# region. A dummy cell is a synthetic cell beyond the scanned edge; it has no id and never
# becomes a grain.
#
# The example uses a real map because the failure appears only when the map is
# realistically wide. A small synthetic grid remains safe at distortion levels that already
# misindex a wide map.

ebsd = mtexdata('small')

# [transform](EBSD.transform.html) moves every pixel and its unit cell. It leaves
# orientations and all other per-pixel properties untouched. The command takes a
# [spatialTransform](spatialTransform.html), which records the mapping as an object; see
# [Spatial Transforms](EBSDSpatialTransform.html).
#
# A tilt is a projective transform. Straight lines remain straight, but parallel lines need
# not, and scale varies across the frame. Three values specify it: the surface tilt, the
# working distance, and the point left in place. `byTilt` tilts about the x axis, so the
# tilt angle foreshortens the map across that axis, along y. The working distance controls
# perspective, which vanishes as that distance grows.
#
# Here the tilt is known and imposed. Recovering an unknown tilt from two measured images
# is the inverse problem. [spatialTransformTilt](spatialTransformTilt.html) fits it in
# stages.

# +
theta = 20 * degree                                  # surface tilt
wd = 8 * (ebsd.pos.y.max() - ebsd.pos.y.min())       # working distance
centre = mean(ebsd.pos)                              # what the tilt leaves in place

distort = spatialTransformProjective.byTilt(theta, wd, centre)
distort
# -

# The map is compressed along y, from 3000 to 2820 micrometres, while the x extent is left
# alone by the tilt and only stretched by perspective. It tapers towards the edge that is
# farther away.

# +
ebsdDistorted = transform(ebsd, distort)

plot(ebsdDistorted['Fo'], ebsdDistorted['Fo'].orientations)
hold(True)
plot(ebsdDistorted['En'], ebsdDistorted['En'].orientations)
plot(ebsdDistorted['Di'], ebsdDistorted['Di'].orientations)
hold(False)
# -

# The following pair measures the imposed distortion in cell units. Its first value is the
# maximum pixel displacement. Its second is the largest residual after the best affine
# grid has been removed.

# +
pos0 = ebsd.pos.data.reshape(-1, 3)[:, :2]
posD = ebsdDistorted.pos.data.reshape(-1, 3)[:, :2]
ijD = ebsdDistorted.lattice.ij.astype(float)
isIndexed = ebsdDistorted.isIndexed.reshape(-1)
affineDesign = np.column_stack([np.ones(len(ijD)), ijD])
affineFit = np.linalg.lstsq(affineDesign[isIndexed], posD[isIndexed], rcond=None)[0]
cellSize = ebsd.lattice.dxy
distortionInCells = np.array([np.linalg.norm(posD - pos0, axis=1).max(),
                              np.linalg.norm(posD - affineDesign @ affineFit, axis=1).max()]) / cellSize
distortionInCells
# -

# Every pixel has moved by up to 2.48 cells. Once the affine foreshortening is removed,
# 0.80 cell remains. That is enough for a rigid reconstruction to round a pixel onto its
# neighbour's site. Even so, the same grid indices are recovered as for the undistorted
# map.

np.array_equal(ebsdDistorted['Fo'].lattice.ij, ebsd['Fo'].lattice.ij)

# Drawing each measurement as its unit cell shows the result. The cells form one
# continuous deformed mesh, with no cell sheared independently of its neighbours. A rigid
# reconstruction would make the mesh wavy and would give
# [calcGrains](EBSD.calcGrains.html) the wrong neighbours.

# +
plot(ebsdDistorted['Fo'], ebsdDistorted['Fo'].orientations, edgeColor='black')
hold(True)
plot(ebsdDistorted['En'], ebsdDistorted['En'].orientations, edgeColor='black')
plot(ebsdDistorted['Di'], ebsdDistorted['Di'].orientations, edgeColor='black')

# grain reconstruction consequently sees the distorted map as the same neighbourhood graph
grains = calcGrains(ebsdDistorted, minPixel=3)
grains = smoothBoundary(grains, simplify=False)

plot(grains.boundary, lineWidth=2)
hold(False)
grains
# -

# The boundary overlay remains closed and follows the tapered map. No duplicated or
# missing grain strips appear along its edges.

# ## Further Reading
#
# * R. C. Staunton, [_Hexagonal Sampling in Image Processing_](https://doi.org/10.1016/S1076-5670(08)70188-5),
#   Advances in Imaging and Electron Physics 107, 231--307 (1999), reviews the geometry
#   and image-processing consequences of hexagonal sampling.
# * V. S. Tong and T. B. Britton,
#   [_TrueEBSD: Correcting spatial distortions in electron backscatter diffraction maps_](https://doi.org/10.1016/j.ultramic.2020.113130),
#   Ultramicroscopy 221, 113130 (2021), separates tilt and drift distortion and
#   demonstrates pixel-scale correction.
# * Y. B. Zhang, A. Elbrønd and F. X. Lin,
#   [_A method to correct coordinate distortion in EBSD maps_](https://doi.org/10.1016/j.matchar.2014.08.003),
#   Materials Characterization 96, 158--165 (2014), treats nonlinear drift by registering
#   an EBSD map to a reference image.
# * [ISO 13067:2020](https://www.iso.org/standard/74309.html), _Microbeam analysis -
#   Electron backscatter diffraction - Measurement of average grain size_, defines EBSD
#   grain-size measurement on two-dimensional sections. Consult it before treating a
#   resampled raster as quantitative evidence about spatial resolution or grain size.

# ## Next
#
# [Maps and Images](EBSDMapsAndImages.html) uses layouts to compare an EBSD map with
# detector images pixel by pixel. [Regridding and Interpolation](EBSDInter.html) develops
# resampling onto a different lattice. [Spatial Transforms](EBSDSpatialTransform.html)
# introduces the transform models used above, and
# [TrueEBSD Distortion Correction](EBSDTrueEbsd.html) fits them to measured images.
# Continue with [Grain Reconstruction](GrainReconstruction.html) when the goal is to turn
# measurements into grains.
