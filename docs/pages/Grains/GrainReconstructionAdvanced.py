# # Advanced Grain Reconstruction
#
# [Grain Reconstruction](GrainReconstruction.html) introduced the threshold angle,
# `alpha`, and `minPixel`. This page shows how to replace parts of the reconstruction
# rather than tune those three settings.
#
# A *grain* is a phase-homogeneous, spatially connected region of EBSD pixels produced by
# segmentation. This page assumes that definition and the basic reconstruction workflow.
# If misorientation is new to you, first see
# [Misorientation Theory](MisorientationTheory.html).
#
# It helps to separate `calcGrains` into three operations:
#
# 1. partition the measured surface into cells
# 2. assign a connectivity to each pair of neighbouring cells
# 3. collect connected cells into grains
#
# The first sections replace the second operation. The later sections compare two spatial
# decompositions used by the first. Denoising orientations and filling missing
# orientations are separate preprocessing decisions; see
# [Denoising Orientation Maps](EBSDDenoising.html) and
# [Fill Missing Data in Orientation Maps](EBSDFilling.html).
#
# We use the same subregion of the forsterite data set as the basic page.

import time

import numpy as np

from mtex import Miller, calcGrains, degree, gbcAngle, gbcCustom, gbcSoft, hold, ipfColorKey, mtexColorbar, mtexdata, plot, plottingConvention

# +
plottingConvention.default('y↑→x')
ebsd = mtexdata('forsterite')
ebsd = ebsd[ebsd.inpolygon(np.array([5, 2, 10, 5]) * 1e3)]
# -

# ## The grain boundary criterion
#
# A `grainBoundaryCriterion` evaluates arrays of neighbouring pixel indices at once. It
# returns one connectivity per pair, from `0` for separated to `1` for fully connected.
# The discrete angle criterion uses three values:
#
# * `1` means no boundary
# * `0.5` means a boundary drawn inside a connected grain
# * `0` means a grain boundary that separates the cells
#
# Without clustering, every positive connectivity joins the graph used to form grains. A
# connectivity at or below `0.5` is also drawn as a boundary. A continuous criterion can
# therefore draw a boundary between two cells that remain connected through the grain.
# The soft-threshold example below makes this distinction visible.
#
# Normally `calcGrains` constructs a criterion from its options. It also accepts a
# criterion object directly. These calls perform the same reconstruction, so their
# summaries would only repeat one another.

# +
ebsdUniform = ebsd.copy()
grainsUniform = calcGrains(ebsdUniform, angle=10 * degree, minPixel=5)
ebsdFromObject = ebsd.copy()
grainsFromObject = calcGrains(ebsdFromObject, gbcAngle(10 * degree), minPixel=5)

print(f'Option and criterion object give identical grain ids: {int(np.array_equal(ebsdUniform.grainId, ebsdFromObject.grainId))}')
# -

# MTEX supplies `gbcAngle`, `gbcSoft`, `gbcCustom`, `gbcVariants`, and `gbcFMC`. The
# default is `gbcAngle`. The [basic page](GrainReconstruction.html) uses `gbcFMC` for
# deformed microstructures. The criterion `gbcVariants` separates variant ids during
# [parent grain reconstruction](MaParentGrainReconstruction.html). The three remaining
# alternatives are developed here.

# ## Phase-dependent thresholds
#
# Different phases may require different boundary angles. A list gives one threshold per
# phase, in the order of `ebsd.CSList`. Its displayed summary makes that order explicit.

ebsd.CSList

# The first entry is `notIndexed`. It is never used as an angular threshold, but it must
# be present. The other four entries belong to the four minerals. Here forsterite, the
# phase that carries the deformation in this specimen, is separated at 5 degrees. The
# other phases keep the 10 degree threshold.

# +
threshold = [10 * degree,   # notIndexed, never used
             5 * degree,    # Forsterite
             10 * degree,   # Enstatite
             10 * degree,   # Diopside
             10 * degree]   # Silicon

grainsPhase = calcGrains(ebsd.copy(), angle=threshold, minPixel=5)

print(f"Forsterite grains: uniform 10 degree = {len(grainsUniform['Forsterite'])}; phase-dependent = {len(grainsPhase['Forsterite'])}")
# -

# The printed counts show that the lower forsterite threshold splits that phase more
# finely. The other phases use exactly the same criterion as in the uniform
# reconstruction and are unchanged.
#
# An entry may itself be a pair `[highAngle, lowAngle]`. The first angle separates grains;
# the second records internal boundaries. See
# [Subgrain Boundaries](SubGrainBoundaries.html) for that distinction.

# ## Soft thresholds
#
# A hard threshold is discontinuous in the data. Neighbouring pixels 9.9 degrees apart
# remain together, while pixels 10.1 degrees apart separate, although the physical
# distinction is unlikely to be that sharp.
#
# `gbcSoft` replaces the step with an error function. Its two parameters are the centre
# angle and transition width. Connectivity then decreases gradually across the band.

# +
grainsSoft = calcGrains(ebsd.copy(), gbcSoft(10 * degree, 2 * degree), minPixel=5)
grainsHard14 = calcGrains(ebsd.copy(), angle=14 * degree, minPixel=5)

softPhaseCounts = np.bincount(grainsSoft.phaseId, minlength=len(ebsd.CSList))
hardPhaseCounts = np.bincount(grainsHard14.phaseId, minlength=len(ebsd.CSList))

print(f'Soft [10 2] and hard 14 degree phase counts agree: {int(np.array_equal(softPhaseCounts, hardPhaseCounts))}')
print(f'Soft-criterion boundary segments inside grains: {len(grainsSoft.innerBoundary)}')
# -

# On its own, the soft criterion still forms grains from positive graph connections. The
# error function reaches numerical zero only after its transition has saturated, so this
# example merely raises the effective splitting angle. Its phase counts match a hard 14
# degree threshold.
#
# The fractional values still record the transition band. A pair past the centre can have
# a boundary drawn while remaining connected inside its grain. Those weights become useful
# when [Markovian Clustering](GrainReconstructionMCL.html) replaces connected components
# with a weighted graph clustering.

# ## Segmenting by another per-pixel property
#
# `gbcCustom` separates neighbouring pixels when a supplied per-pixel property differs by
# more than a threshold. The property may be numeric, a `vector3d`, or a `quaternion`.
# Vectors and quaternions are compared by their angle.
#
# The custom criterion compares only that property; it does not add a phase check. Subset
# a multiphase map to one phase, as below, or enforce phase separation in a criterion of
# your own. This keeps the resulting grains phase-homogeneous.
#
# Here grains are defined by their c-axis alone. Two forsterite pixels may belong to the
# same grain whenever their c-axes agree, regardless of the rotation about that axis.

# +
fo = ebsd['Forsterite']
cAxis = fo.orientations * Miller(0, 0, 1, fo.CS)

grainsC = calcGrains(fo.copy(), gbcCustom(cAxis, 10 * degree, antipodal=True), minPixel=5)
grainsA = calcGrains(fo.copy(), angle=10 * degree, minPixel=5)

print(f'Full-orientation grains: {len(grainsA)}; c-axis grains: {len(grainsC)}')
# -

# The flag `antipodal` is passed to `angle`. It makes the c-axis an axis rather than a
# direction. Without it, opposite senses of the same axis would be treated as different.
#
# Draw the ordinary reconstruction in black and the c-axis reconstruction in red. Black
# segments that remain visible mark boundaries across which the c-axes agree but the
# rotations about them differ.

# +
ipfKey = ipfColorKey(fo.CS)
ipfColor = ipfKey.orientation2color(fo.orientations)

plot(fo, ipfColor, micronbar=False)
hold(True)
plot(grainsA.boundary, lineWidth=3, lineColor='black')
plot(grainsC.boundary, lineWidth=1, lineColor='red')
hold(False)
# -

# Only a few black segments remain on this specimen. The printed counts show that 69
# full-orientation grains become 66 c-axis grains. Such mergers would be more common in a
# material with a strong fibre texture.

# ## Writing your own criterion
#
# A custom criterion is a subclass of `grainBoundaryCriterion`. Implement the method
# `connectivity(ebsd, i, j)`. The arrays `i` and `j` list neighbouring pixel indices and
# the output must have the same size.
#
# This example requires both a small misorientation and adequate band contrast. The
# second condition stops a grain from leaking through a chain of poorly indexed pixels.
# The value `minBC = 50` is specific to the data and acquisition settings; inspect the
# band-contrast distribution before choosing it.
#
# ```python
# class gbcAngleAndBC(grainBoundaryCriterion):
#
#   def __init__(self, threshold=10 * degree, minBC=50):
#     self.threshold, self.minBC = threshold, minBC
#
#   def connectivity(self, ebsd, i, j):
#
#     # delegate the misorientation part
#     out = gbcAngle(self.threshold).connectivity(ebsd, i, j)
#
#     # veto connections through low band contrast
#     bad = (ebsd.bc.reshape(-1)[i] < self.minBC) | (ebsd.bc.reshape(-1)[j] < self.minBC)
#     out[bad] = 0
#     return out
# ```
#
# It can then be passed like any other criterion:
#
# ```python
# grains = calcGrains(ebsd, gbcAngleAndBC(), minPixel=5)
# ```

# ## The spatial decomposition
#
# The option `alpha` changes the first reconstruction operation: the spatial support from
# which cells are built. By default, MTEX closes the indexed region with a raster disk
# whose radius is `alpha` pixel spacings. A `notIndexed` region narrower than `2*alpha` is
# crossed by surrounding cells; a wider region remains a separate `notIndexed` grain. The
# default is `alpha = 3.1`.
#
# The work of raster closing grows with the area of the disk. MATLAB's flag `'delaunay'`
# instead computes a Delaunay triangulation of the indexed pixels and keeps a triangle
# when its circumradius is at most `alpha` pixel spacings, an exact alpha complex whose
# cost changes little as `alpha` grows; that route is not part of this port, the closing
# here is by a distance transform whose cost does not grow with the disk either.

# +
t = time.time()
grainsRaster = calcGrains(ebsd.copy(), angle=10 * degree, minPixel=5, alpha=60)
print(f'Elapsed time is {time.time() - t:.3f} seconds.')
# -

# ## Diagnose the `notIndexed` regions first
#
# Choosing one `alpha` for a whole map need not be guesswork. `calcGrains` returns
# connected `notIndexed` areas as grains, so their size and shape can be inspected with
# the tools from [Shape Parameters](ShapeParameters.html). Setting `alpha = 0` closes
# nothing, so narrow `notIndexed` areas stay where they are instead of being absorbed into
# their neighbours.

# +
grainsOpen = calcGrains(ebsd.copy(), angle=10 * degree, alpha=0)
notIndexedGrains = grainsOpen['notIndexed']
notIndexedGrains
# -

# Pixel counts give a first scale estimate. The equivalent-circle radius of a region with
# `n` pixels is `sqrt(n/pi)` pixel spacings.

# +
numPixel = notIndexedGrains.numPixel
print(f'median pixels: {np.median(numPixel):g}, 99th percentile: {np.nanquantile(numPixel, 0.99):g}, '
      f'maximum: {np.max(numPixel)}, maximum equivalent radius in pixels: {np.sqrt(np.max(numPixel) / np.pi):.2f}')
# -

# The table shows that most areas are isolated failed measurements, while a few are
# regions in their own right. The largest contains 355 pixels and has an equivalent-circle
# radius of about 10.6 pixels. This radius suggests the scale to inspect; it does not
# predict the closing value by itself. Closing responds to the narrowest width and
# therefore also to shape.
#
# The ratio below is a quick size-and-compactness diagnostic. It is high for large,
# compact blobs and low for small or ragged areas. It is not a scale-free compactness
# measure because the numerator grows with area and the denominator counts boundary
# segments. Use `shapeFactor` when shape must be compared apart from size.

plot(notIndexedGrains, np.log(notIndexedGrains.numPixel / notIndexedGrains.boundarySize))
mtexColorbar(title='log(pixels / boundary segments)')

# High colours identify the larger, more compact holes that can justify preservation. Low
# colours mark isolated pixels and ragged indexing failures that a closing may reasonably
# absorb. The map is a diagnostic, not an automatic choice of `alpha`.
#
# A `notIndexed` pixel and an absent grid position have different experimental meanings.
# The first is a real measurement whose pattern could not be indexed; the second has no
# retained measurement. The spatial decomposition treats both as positions without an
# indexed orientation, so `alpha` acts on both. Deleting measurements therefore creates a
# gap for segmentation; it does not assign orientations there. To reconstruct
# orientations, see [Fill Missing Data in Orientation Maps](EBSDFilling.html).

# ## Choosing the next step
#
# Use a custom criterion when the physical separator is another per-pixel quantity. Use
# the spatial options when the uncertainty is where the map has support. In either case,
# repeat the reconstruction over defensible nearby settings before treating a grain count
# or size distribution as measured.
#
# Continue with [Markovian Clustering](GrainReconstructionMCL.html) when fractional
# criterion weights should influence the grouping itself. The resulting boundaries and
# their internal boundaries are developed in [Grain Boundaries](GrainBoundaries.html) and
# [Subgrain Boundaries](SubGrainBoundaries.html). Standards for reporting EBSD grain size
# are listed on the [basic reconstruction page](GrainReconstruction.html).

# ## References
#
# * F. Bachmann, R. Hielscher, and H. Schaeben,
#   [Grain Detection from 2d and 3d EBSD Data - Specification of the MTEX Algorithm](https://doi.org/10.1016/j.ultramic.2011.08.002),
#   *Ultramicroscopy* 111 (2011), 1720-1733. This paper derives the Voronoi
#   reconstruction used by `calcGrains`.
# * H. Edelsbrunner, D. G. Kirkpatrick, and R. Seidel,
#   [On the Shape of a Set of Points in the Plane](https://doi.org/10.1109/TIT.1983.1056714),
#   *IEEE Transactions on Information Theory* 29 (1983), 551-559. This paper introduces
#   planar alpha shapes.
# * P. Soille,
#   [Morphological Image Analysis: Principles and Applications](https://doi.org/10.1007/978-3-662-05088-0),
#   2nd ed., Springer, 2003. The chapters on dilation, erosion, opening, and closing
#   provide the image-processing basis for the raster decomposition.

# ## Technical details
#
# `calcGrains` writes into the map it is given, so every reconstruction here runs on
# `ebsd.copy()` to keep the comparisons on the same map. `gbcSoft([10 2]*degree)` is
# `gbcSoft(10 * degree, 2 * degree)`, and a custom criterion overrides `connectivity`
# where MATLAB's overrides `doEvaluate`. The Delaunay route and the Markovian clustering
# are not ported; the closing is a distance transform at integer radius, which absorbs
# less than MATLAB's raster disk at small `alpha`.
