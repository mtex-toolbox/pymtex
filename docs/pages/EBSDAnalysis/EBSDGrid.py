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

from mtex import EBSD, calcGrains, degree, gridify, mtexColorMap, mtexdata, mtexdatafile, newMtexFigure, nextAxis, norm, plot, plottingConvention, rotate, rotation, setColorRange, mtexTitle, zvector

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
# MATLAB offers `gridLayout` objects, `'columnMajor'` and `'rowMajor'`, to store the
# matrix in another orientation; `gridify(ebsd, gL)` reindexes the map without
# resampling, every per-pixel property following, and the conversion back is exact. The
# layouts are not ported: the port stores every gridded map with rows along y and columns
# along x, and a transposed or flipped array is one NumPy call on a property.
#
# Layout changes storage, not the specimen. The positions and orientations are
# unchanged, and the plotting convention still draws the map in the same specimen frame.

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
#     ebsd = EBSD.load(fname, noGrid=True)
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
# so `ebsd.id` translates between the two orders. MATLAB keeps the same record under the
# name `oldId` and returns the reverse translation as the second output of `gridify`.
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
# Some external image-processing tools require square pixels. In MATLAB, passing a
# square `unitCell` to `gridify` resamples the hexagonal measurements onto such a grid;
# the new grid sites do not coincide with the measured ones, so `interp` copies values
# from the nearest measurements, and squares cannot reproduce hexagonal outlines, so
# grain boundaries are visibly staircased. A smaller square cell reduces the size of the
# steps without adding spatial resolution. The resampling onto another unit cell is not
# ported yet. [Regridding and Interpolation](EBSDInter.html) develops the distinction
# between restoring a map's own lattice and resampling.

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
# MATLAB's page imposes a known tilt on the forsterite map with a projective
# `spatialTransform`, `spatialTransformProjective.byTilt`, and shows that every pixel
# moves by up to 2.48 cells, 0.80 cell remaining after the affine foreshortening is
# removed, and that the grid indices, the cell mesh and the grain boundaries come out the
# same as for the undistorted map. The spatial transforms are not ported yet, so that
# demonstration is left out here; [Spatial Transforms](EBSDSpatialTransform.html)
# introduces the transform models and [TrueEBSD Distortion Correction](EBSDTrueEbsd.html)
# fits them to measured images.

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
