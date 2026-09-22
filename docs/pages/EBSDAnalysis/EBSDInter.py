# # Regridding and Interpolating EBSD Data
#
# An EBSD map stores one measurement at each acquired position.
# [Denoising](EBSDDenoising.html) changes noisy orientations without moving those
# positions. [Filling Missing Data](EBSDFilling.html) supplies values at missing
# positions while leaving the measurement grid unchanged.
#
# This page considers the other operation: reading the map at positions that were not
# measured, then sampling it on a different grid. MTEX uses nearest neighbours for this
# operation. It does not average orientations.
#
# The examples assume basic [EBSD selection](EBSDSelect.html) and the distinction
# between a list and a [gridded EBSD map](EBSDGrid.html).

import matplotlib.pyplot as plt

from mtex import angle, degree, hold, interp, mtexdata, plot, plottingConvention

plottingConvention.default('y↑→x')
ebsd = mtexdata('twins')

# ## Reading the Map at One Point
#
# [interp](EBSD.interp.html) evaluates a map at arbitrary coordinates. It works with a
# plain `EBSD` list, a phase subset, and a rotated or sheared map. The imported map in
# this example happens to be a square grid.

# +
x = 30.5
y = 5.5

plot(ebsd, ebsd.orientations)
hold(True)
plt.plot(x, y, 'ko', markerfacecolor='w', markersize=7)
hold(False)
# -

# The marker identifies the requested position. It lies between the centres of the
# coloured source pixels, so the result must be sampled from one of them.

e1 = interp(ebsd, x, y)
e1

# The summary shows a new, one-entry `EBSD` variable at the requested position. MTEX
# finds the nearest source measurement and copies its phase, orientation, and per-pixel
# properties.
#
# The query is accepted only when the nearest measurement is no farther away than the
# furthest corner of `ebsd.unitCell`. This circular distance cutoff is not a
# polygon-containment test. A point beyond that reach returns as `notIndexed` instead of
# being extrapolated, whether it is outside the map or sufficiently far inside a hole.
#
# The `'xy'` selector also finds the nearest source measurement.

e2 = ebsd['xy', x, y]
e2

# For this point, both commands therefore report the same orientation.

angle(e1.orientations, e2.orientations) / degree

# The zero-degree result confirms the match. The two EBSD summaries expose the important
# difference: `e2` retains the source position and id, whereas `e1` sits at the requested
# position and receives a new id.
#
# With a list of query positions, `interp` returns one new entry per query in the same
# order. That behavior makes resampling possible; `'xy'` remains a selector for original
# measurements.

# ## Resampling onto a Different Grid
#
# A unit cell describes the shape and scale of a grid. In MATLAB,
# `gridify(ebsd, 'unitCell', uC)` derives its cell-to-cell translations, builds a lattice
# over the map extent, and calls `interp` at the new positions; with a unit cell twice as
# large and rotated by 45 degrees the orientation regions occupy the same specimen
# positions, but their pixels are coarser and tilted, and the tilted lattice needs a
# larger rectangular matrix, 108 by 109, to cover the source extent, so that fewer than
# half of its cells are indexed. The resampling of a map onto another unit cell is not
# ported yet; `interp` at a list of positions of your own is the same operation by hand.

# ## From a Hexagonal to a Square Grid
#
# A custom unit cell can also change the grid type. The ferrite data was measured on a
# hexagonal grid.

# +
plottingConvention.default('y↓→x')
ebsd = mtexdata('ferrite')

hexGridSize = ebsd.shape
hexGridSize
# -

plot(ebsd[0:50, 0:100], ebsd[0:50, 0:100].orientations)

# The staggered pixel rows reveal the hexagonal sampling lattice. Their orientations are
# measurements. In MATLAB a square cell with half the measurement spacing, passed to
# `gridify`, redraws those values on a square lattice of 808 by 809 cells in place of the
# 270 by 234 hexagonal cells, about ten times as many, each new cell repeating its
# nearest hexagonal measurement; no orientation is invented on the way, and the smaller
# cell does not improve the map's spatial resolution. That resampling waits with the one
# above.

# ## What Regridding Does Not Change
#
# Resampling changes the array size and pixel outlines, not the acquisition step or the
# spatial resolution of the experiment. Repeated cells are not independent measurements.
# Keep the original map for quantitative counts, orientation statistics, and grain-size
# measurements unless the analysis explicitly requires a new raster.
#
# Every copied quality value still describes the original diffraction pattern. It is not
# evidence that a pattern was acquired at the new cell. Use
# [Filling Missing Data](EBSDFilling.html) to recover orientations at missing positions,
# or [Denoising](EBSDDenoising.html) to reduce orientation noise. Neither task is
# performed by `interp`.

# ## Further Reading
#
# * R. C. Staunton, [_Hexagonal Sampling in Image Processing_](https://doi.org/10.1016/S1076-5670(08)70188-5),
#   Advances in Imaging and Electron Physics 107, 231--307 (1999), reviews the geometric
#   consequences of converting between hexagonal and square sampling lattices.
# * F. J. Humphreys, [_Review: Grain and subgrain characterisation by electron backscatter diffraction_](https://doi.org/10.1023/A:1017973432592),
#   Journal of Materials Science 36, 3833--3854 (2001), quantifies how EBSD step size
#   limits grain-size measurements.
# * [ISO 13067:2020](https://www.iso.org/standard/74309.html), _Microbeam analysis -
#   Electron backscatter diffraction - Measurement of average grain size_, defines EBSD
#   grain-size measurement on two-dimensional sections. Consult it before using a
#   resampled raster for grain statistics.

# ## Next
#
# [ODF Estimation](EBSD2ODF.html) turns measured map orientations into an orientation
# distribution function. Use the measured map rather than a densified copy so repeated
# cells do not acquire extra statistical weight.
