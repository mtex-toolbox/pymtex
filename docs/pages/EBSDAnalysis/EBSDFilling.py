# # Fill Missing Data in Orientation Maps
#
# A `notIndexed` pixel is a real scan measurement whose diffraction pattern could not be
# indexed. Patterns measured at grain boundaries may combine signal from both
# neighbouring grains. Cracks, pores, scratches, surface contamination, or a phase
# omitted during indexing may also prevent a usable solution.
#
# A missing lattice site is different: no measurement occupies that position. This can
# happen when a scan is interrupted or when selecting or cropping a map leaves an
# irregular footprint. The experimental meanings of the two cases differ.
#
# For filling, both are the same: a position without an orientation.
# [fill](EBSD.fill.html) assigns an orientation to every such position inside the
# scanned area, unless a mask excludes it. The assigned values are reconstructions, not
# measurements. Keep that provenance when the filled map is used for quantitative
# analysis.
#
# This page assumes familiarity with [selecting EBSD data](EBSDSelect.html),
# [orientation maps](EBSDIPFMap.html), and
# [grain reconstruction](GrainReconstruction.html). Reducing random error at positions
# that already have orientations is the separate subject of
# [Denoising Orientation Maps](EBSDDenoising.html). Filling also cannot correct an
# indexed but wrong orientation. The next page,
# [Correcting Pseudo Symmetry](EBSDPseudoSymmetry.html), treats one important source of
# that different error.

import numpy as np

from mtex import *

# ## A complete map for comparison
#
# The first example is an orientation map of ferrite. The object summary reports the
# indexed ferrite measurements and the sites whose phase is `notIndexed`: one measurement
# the indexing rejected, and the 135 sites of the hexagonal lattice the file leaves empty.

# +
# import the data in its measurement frame
plottingConvention.default('y↓→x')
ebsd = mtexdata('ferrite')
ebsd
# -

# +
# reconstruct the grain structure
grains = calcGrains(ebsd, angle=10 * degree, minPixel=5)

# smooth the pixel staircase of the grain boundaries
grains = smoothBoundary(grains, 5)

# plot the orientation map and its reconstructed grain boundaries
plot(ebsd, ipfDirection=zvector)
hold(True)
plot(grains.boundary, lineWidth=1.5)
hold(False)
# -

# The boundary overlay supplies the geometric constraint used below. It separates missing
# positions in grain interiors from unresolved corridors along grain boundaries.

# ## A very sparse measured data set
#
# To make the two filling methods easy to compare, we mark approximately 75 percent of
# the measurements as `notIndexed` and keep approximately 25 percent. A marked pixel keeps
# its position and its other properties, but its orientation no longer counts as
# measured.

# +
ebsdSub = ebsd.copy()
ebsdSub[np.random.default_rng(0).random(ebsd.shape) > 0.25] = 'notIndexed'
ebsdSub
# -

# +
# plot the retained data
plot(ebsdSub, ipfDirection=zvector)
# -

# The object summary gives the exact retained counts. The plot now consists of isolated
# coloured measurements, so the original grain shapes cannot be read from the
# orientation map alone.
#
# We first reconstruct grains from the retained quarter of the map. The option `alpha`
# closes narrow notIndexed regions during segmentation; it does not itself assign
# orientations. See [Grain Reconstruction](GrainReconstruction.html) for that
# distinction.

# +
# reconstruct the grain structure
grainsSub = calcGrains(ebsdSub, angle=9 * degree, minPixel=2, alpha=15)

grainsSub = smoothBoundary(grainsSub, 5)

hold(True)
plot(grainsSub.boundary, lineWidth=1.5)
hold(False)
# -

# The boundaries recover the main grain outlines despite the sparse input. They also
# divide the map into interiors that may be filled and boundary corridors that should
# remain unresolved.

# ## Filling within the grains
#
# [fill](EBSD.fill.html) repeats the orientation of the nearest measured pixel. It fills
# every `notIndexed` pixel and every missing lattice site inside the convex hull of the
# measurements. Pass the reconstructed grains whenever boundaries are known. A pixel then
# takes an orientation only from its own grain, so no orientation is carried across a
# grain boundary.
#
# Some pixels have no grain to fill from:
#
# * A pixel a grain boundary passes through belongs to no grain and stays `notIndexed`.
# * The pixels of a `notIndexed` grain stay `notIndexed`.
#
# Without the grains, a missing lattice site takes the grain of its nearest pixel instead,
# which fills most of the corridors along the boundaries as well.

# +
ebsdSub_filled = fill(ebsdSub, grainsSub)

plot(ebsdSub_filled['indexed'], ipfDirection=zvector)

hold(True)
plot(grainsSub.boundary, lineWidth=1.5)
hold(False)
# -

# The grain interiors are now coloured. Sharp patches remain because a nearest-neighbour
# estimate repeats a measured orientation instead of creating a gradual orientation field.
# The white pixels lie on grain boundaries: they belong to no grain and remain
# `notIndexed`.

# ## Filling with a denoising filter
#
# A denoising filter can estimate the missing orientations while smoothing the
# orientation field. [smooth](EBSD.smooth.html) fills the same positions as `fill` before
# it filters, so pass it the grains as well. The `halfQuadraticFilter` used here is a
# total-variation method for rotation-valued data; its mathematical basis is described by
# [Bergmann et al. (2016)](https://doi.org/10.3934/ipi.2016001).
#
# In contrast to nearest-neighbour interpolation, the filter can create a smooth
# transition between the estimated orientations. It is still an estimate, and the
# reconstructed grain boundaries still constrain it. Neither method reconstructs the
# grains again. The supplied boundaries remain the segmentation model for the filled map.

# +
F = halfQuadraticFilter()
F.alpha = 0.25

# interpolate and smooth the missing orientations
ebsdSub_smoothed = smooth(ebsdSub, F, grainsSub)

plot(ebsdSub_smoothed['indexed'], ipfDirection=zvector)

hold(True)
plot(grainsSub.boundary, lineWidth=1.5)
hold(False)
# -

# The repeated colour patches have become continuous fields inside the grains. The white
# corridors still follow the reconstructed boundaries, so the smoother result has not
# erased the segmentation constraint.

# ## Which data is interpolated
#
# In these grain-constrained calls, both `fill` and `smooth` recover the orientation,
# phase, and `grainId` of a filled position. The other per-pixel properties keep what the
# pixel had. The ferrite map carries these properties:

list(ebsdSub_filled.prop)

# The properties `ci`, `fit`, and `iq` describe the recorded pattern or its indexing
# solution. The property `sem_signal` is the simultaneously recorded SEM signal. A pixel
# marked `notIndexed` keeps the values recorded with it. A lattice site that was never
# measured has none, so MTEX leaves each of them as `NaN` there.

plot(ebsdSub_filled, ebsdSub_filled.ci)
mtexColorbar(title='confidence index')

# The confidence index therefore does not tell which orientations were filled. Take that
# from the map before filling: a pixel without an orientation then and with one afterwards
# was filled. A pixel `calcGrains` absorbed into a grain has the grain's phase but no
# orientation, so both conditions are needed.

noOrientation = ~ebsdSub.isIndexed | ebsdSub.rotations.isnan()
isFilled = noOrientation & ebsdSub_filled.isIndexed & ~ebsdSub_filled.rotations.isnan()
print(f'{np.sum(isFilled)} of {ebsdSub_filled.size} positions have a filled orientation')

# MATLAB's `smooth` provides a second marker, a per-pixel property named `quality` set to
# zero at every position that had no orientation before filtering; it is not ported.

# ## Filling is not resampling
#
# If all per-pixel properties are needed at arbitrary query positions, use
# [interp](EBSD.interp.html). It transfers the complete record of the nearest existing
# measurement to a requested position. The page
# [Interpolating EBSD Data](EBSDInter.html) develops that operation.
#
# The requested position must be covered by an existing measurement cell. Thus `interp`
# resamples a map but does not fill a hole. Conversely, `fill` reconstructs missing
# orientations but leaves the other properties as they were.

# ## A multiphase geological map
#
# Geological samples often contain many notIndexed measurements because of relief,
# fractures, or phases that are difficult to index. The forsterite example also shows why
# phase and grain boundaries must constrain filling.

# +
plottingConvention.default('y↑→x')
ebsd = mtexdata('forsterite')
ebsd = ebsd[inpolygon(ebsd, np.array([10, 4, 5, 3]) * 10**3)]

print(f'The submap contains {100 * np.sum(~ebsd.isIndexed) / ebsd.size:.1f} percent notIndexed measurements')

plot(ebsd, ipfDirection=zvector)
hold(True)

# compute and smooth grains
grains = calcGrains(ebsd, angle=10 * degree, minPixel=3, alpha=3)
grains = smoothBoundary(grains, 5)

# plot the boundary of all grains
plot(grains.boundary, lineWidth=2)
hold(False)
# -

# The printed fraction is 29.0 percent, and most of that white is speckle inside the
# grains. Some grains are peppered with it while their neighbours are almost clean, and
# 48 percent of all white pixels sit more than one 50 micrometre step from any boundary
# between two indexed grains. The two large white patches are areas that could not be
# indexed at all, which segmentation kept as notIndexed grains of their own.
#
# Each phase is drawn with its own IPF key, so a colour identifies an orientation within
# a phase and not the phase itself. The figure shows where orientations are missing, not
# which mineral sits where.
#
# The white that does follow the black outlines is the case filling must respect. There
# the interaction volume straddles two crystals, and giving such a position one
# neighbour's orientation would erase the uncertainty that marks the boundary.

# ## Fill only grain interiors
#
# The command `smooth` fills every `notIndexed` pixel that `calcGrains` assigned to an
# indexed grain. The `notIndexed` measurements on grain boundaries belong to no grain, and
# the two large patches are `notIndexed` grains; both remain `notIndexed`.

# +
F = halfQuadraticFilter()
F.alpha = 10

ebsdS = smooth(ebsd, F, grains)

plot(ebsdS, ipfDirection=zvector)
hold(True)

# plot the boundary of all grains
plot(grains.boundary, lineWidth=1.5)
hold(False)
# -

# Colour now extends through holes in the grain interiors. The narrow white corridors
# that remain coincide with reconstructed boundaries, so the filled map does not claim a
# phase or orientation where the segmentation was deliberately unresolved.

# ## Filling selected pixels
#
# A mask restricts filling to the pixels it marks. It never fills a pixel that `smooth`
# would leave alone, so grain boundaries and `notIndexed` grains stay `notIndexed` whatever
# the mask says. A mask can select by phase, here the pixels of the enstatite grains.

# +
# the pixels of the enstatite grains
isEnstatite = np.isin(ebsd.grainId, grains['Enstatite'].id)
ebsdEn = smooth(ebsd, F, grains, fill=isEnstatite)

plot(ebsdEn, ipfDirection=zvector)
hold(True)
plot(grains.boundary, lineWidth=1.5)
hold(False)
# -

# The enstatite grains are filled, while the holes in the forsterite and diopside grains
# remain.
#
# A mask can also use any property of the pixels. The band contrast `bc` measures the
# quality of the diffraction pattern. A `notIndexed` pixel with a low band contrast
# probably did not see a crystal at all, for example above a pore. The next call fills
# only the pixels whose band contrast is at least 80; nine in ten indexed measurements
# exceed 85.

# +
# the pixels with a pattern of at least moderate contrast
hasPattern = ebsd.bc >= 80
ebsdBC = smooth(ebsd, F, grains, fill=hasPattern)

plot(ebsdBC['notIndexed'], faceColor='black')
hold(True)
plot(ebsdBC['indexed'], ipfDirection=zvector)
plot(grains.boundary, lineWidth=1.5)
hold(False)
# -

# The black pixels are `notIndexed`: they lie on grain boundaries or in the `notIndexed`
# grains. The white holes inside the grains are the pixels with a weak pattern, which the
# mask left unfilled.

# ## Read the recovered orientation field
#
# Absolute orientation colours can hide small intragranular changes. An
# `axisAngleColorKey` compares every orientation with its grain's mean orientation. The
# misorientation axis determines the hue, and the angle controls the colour saturation
# up to the chosen 2.5 degree maximum.

# +
colorKey = axisAngleColorKey(ebsdS['Fo'])
colorKey.oriRef = grains.selectByGrainId(ebsdS['Fo'].grainId).meanOrientation
colorKey.maxAngle = 2.5 * degree

color = colorKey.orientation2color(ebsdS['Fo'].orientations)
plot(ebsdS['Fo'], color, micronbar=False)

hold(True)
colorKey.oriRef = grains.selectByGrainId(ebsdS['En'].grainId).meanOrientation

plot(ebsdS['En'], colorKey.orientation2color(ebsdS['En'].orientations))

# plot boundaries
plot(grains.boundary, lineWidth=4)
plot(grains['En'].boundary, lineWidth=4, lineColor='r')
hold(False)
# -

# The colour changes smoothly through positions that previously lacked an orientation.
# This continuity is the filter's reconstruction of the intragranular orientation
# gradient, not newly measured deformation.
#
# Compare it with the same view of the original map.

# +
colorKey.oriRef = grains.selectByGrainId(ebsd['Fo'].grainId).meanOrientation
colorKey.maxAngle = 2.5 * degree

color = colorKey.orientation2color(ebsd['Fo'].orientations)
plot(ebsd['Fo'], color, micronbar=False)

hold(True)
colorKey.oriRef = grains.selectByGrainId(ebsd['En'].grainId).meanOrientation

plot(ebsd['En'], colorKey.orientation2color(ebsd['En'].orientations))

# plot boundaries
plot(grains.boundary, lineWidth=4)
plot(grains['En'].boundary, lineWidth=4, lineColor='r')
hold(False)
# -

# The original view breaks those smooth colour fields with white regions. The comparison
# shows exactly where filling supplied continuity and guards against reading the
# reconstructed pixels as independent observations.

# ## Further reading
#
# * F. Bachmann, R. Hielscher and H. Schaeben,
#   [Grain detection from 2d and 3d EBSD data - specification of the MTEX algorithm](https://doi.org/10.1016/j.ultramic.2011.08.002),
#   _Ultramicroscopy_ 111, 1720-1733, 2011, explains the spatial grain reconstruction that
#   makes the boundary constraint possible.
# * R. Hielscher, C. B. Silbermann, E. Schmidl and J. Ihlemann,
#   [Denoising of crystal orientation maps](https://doi.org/10.1107/S1600576719009075),
#   _Journal of Applied Crystallography_ 52, 984-996, 2019, compares filters, hole
#   filling, and their effects on derived quantities.
# * A. J. Schwartz, M. Kumar, B. L. Adams and D. P. Field, editors,
#   [Electron Backscatter Diffraction in Materials Science](https://doi.org/10.1007/978-0-387-88136-2),
#   second edition, Springer, 2009, provides the experimental background to indexing and
#   cleanup.
# * [ASTM E2627-13(2019)](https://doi.org/10.1520/E2627-13R19) standardizes EBSD
#   grain-size measurements for fully recrystallized materials and requires a high
#   proportion of reliably indexed patterns. It is a reminder that filled orientations do
#   not increase the measured indexing rate.
