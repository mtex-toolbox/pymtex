# # Grain Size and Basic Shape Parameters
#
# The outline of a grain is a polygon with hundreds of vertices. A shape parameter
# compresses that outline into a number that can be plotted, histogrammed, and compared
# between specimens. The price is that almost all other information about the outline is
# lost.
#
# This page covers direct measurements of a grain in a two-dimensional section. It assumes
# that the grains have been [reconstructed](GrainReconstruction.html) and explains why
# their boundaries are [smoothed](GrainSmoothing.html). Use
# [Selecting Grains](SelectingGrains.html) first if selecting complete grains away from
# the map edge is unfamiliar.
#
# A measured section is not a three-dimensional grain. Large grains are more likely to be
# cut, and most cuts miss the widest part of a grain. Quantities such as `area` and
# `equivalentRadius` therefore describe the observed section unless a stereological model
# or a standard says how to infer a three-dimensional size from them.
#
# The most useful scalar properties are listed below. Lengths are returned in the map's
# measurement unit and areas in its square.
#
# | | | | |
# |---|---|---|---|
# | `numPixel` | number of measurements in the grain | `area` | area of the section |
# | `boundarySize` | number of outline segments | `perimeter` | length of the outline |
# | `subBoundarySize` | number of subgrain boundary segments | `subBoundaryLength` | length of subgrain boundaries |
# | `diameter` | largest vertex-to-vertex distance | `caliper` | caliper or Feret diameter |
# | `equivalentPerimeter` | perimeter of the equal-area circle | `equivalentRadius` | radius of the equal-area circle |
# | `shapeFactor` | perimeter divided by equivalent perimeter | `paris` | indentation relative to the convex hull |
# | `sphericity` | boundary irregularity in the ice-grain example | `isBoundary` | does the grain touch the map edge? |
# | `hasHole` | does the grain enclose another grain? | `isInclusion` | is the grain enclosed by another grain? |
# | `numNeighbors` | number of neighbouring grains | `triplePoints` | list of triple points |
# | `boundary` | list of grain boundary segments | `innerBoundary` | list of subgrain boundary segments |
# | `x`, `y` | coordinates of the outline vertices | `centroid` | area centroid |
#
# A hole and an inclusion are the same enclosure viewed from opposite sides. The
# containing grain has a hole, and the contained grain is an inclusion.

import matplotlib.pyplot as plt
import numpy as np

from mtex import calcGrains, degree, hist, histogram, hold, logAxis, mtexColorbar, mtexdata, plot, plottingConvention, setColorRange, smoothBoundary, zvector

# +
# load sample EBSD data in its specimen plotting frame
plottingConvention.default('y↑→x')
ebsd = mtexdata('forsterite')

# restrict the map to a region of interest and indexed measurements
ebsd = ebsd[ebsd.inpolygon(np.array([5, 2, 10, 5]) * 1e3)]
ebsd = ebsd['indexed']

# reconstruct and smooth the grains
grains = calcGrains(ebsd, angle=5 * degree, minPixel=5)
grains = smoothBoundary(grains, 5)
mapGrains = grains

# plot forsterite orientations and the reconstructed boundary network
plot(ebsd['Fo'], ebsd['Fo'].orientations, ipfDirection=zvector)
hold(True)
plot(grains.boundary, lineWidth=2)
hold(False)
# -

# The black network separates the reconstructed grains. Smoothing has removed the
# square-grid staircase while retaining the larger-scale turns of the boundaries. It is a
# measurement choice, not recovered sub-pixel detail.
#
# A grain touching the map edge continues beyond the measured field. Its visible area and
# outline are truncated, so we exclude such grains from the per-grain comparisons below. A
# standardised average grain-size method may prescribe a different edge-counting rule.

grains = grains[~grains.isBoundary]

# ## Pixel count and area
#
# Size comes in two forms: the number of measurements assigned to a grain and the area
# occupied by its observed section.

grains[8].numPixel

grains[8].area

# `numPixel` is a count, while `area` is in the square of the map unit. They are
# proportional only when one step size is used throughout the map. The count is useful
# during reconstruction because `minPixel` is expressed in it. Physical area is usually the
# quantity to report.
#
# The number-weighted area distribution is strongly skewed.

plt.figure()
plt.hist(grains.area)
plt.xlabel('grain area')
plt.ylabel('number of grains')

# Nearly every grain falls into the first bar. The plot answers how many grains have each
# area, so numerous small grains dominate even when they occupy little of the section.
#
# Weighting each grain by its area asks a different question: what fraction of the
# observed section belongs to grains in each size class? `hist(grains)` draws grouped
# bars, one colour per indexed phase.

hist(grains)

# The small-grain bar is much less dominant after area weighting. Each bar height is now
# a percentage of the analysed section rather than a grain count. `histogram(grains)`
# draws the same values as overlaid phase histograms instead of side-by-side bars.

histogram(grains)

# The phase distributions overlap; they are not stacked and their heights should not be
# added by eye. The many tiny grains remain in the list, but they contribute little area.
# The largest forsterite grain and its fraction of the analysed interior-grain area are
# printed below.

# +
forsteriteArea = grains['Fo'].area
largestForsteriteArea = max(forsteriteArea)
largestForsteriteArea
# -

largestForsteriteFraction = largestForsteriteArea / np.sum(grains.area)
largestForsteriteFraction

# The largest forsterite section covers 2.805 mm² and accounts for 15.31 percent of the
# analysed interior-grain area. The many small grains have not gone away; area weighting
# has changed how much each one counts.

# ## Boundary segments and perimeter
#
# `boundarySize` and `perimeter` are the same pair one dimension lower: a count of
# outline segments and their total length.

grains[8].boundarySize

grains[8].perimeter

# Both are shortcuts for asking the grain boundary itself, which is a list of segments
# with one `segLength` each.

len(grains[8].boundary)

np.sum(grains[8].boundary.segLength)

# The two pairs react differently to processing. `numPixel` remains tied to the
# measurements, whereas smoothing resamples the outline and can change both
# `boundarySize` and `perimeter`. By default, `perimeter` measures only the outer loop.
# The option `withInclusion` adds the loops around grains enclosed inside it.

# ## Diameter and the equivalent circle
#
# The `diameter` is the longest distance between any two vertices of the outline. It is
# one of the directional caliper or Feret measures developed in
# [Projection Parameters](ProjectionBasedParameters.html).

grains[8].diameter

# Another characteristic length comes from the circle with the same area. Its
# `equivalentRadius` gives the diameter below. No shape of that area has a shorter
# perimeter than the circle, and no grain of that area has a smaller maximum diameter.

2 * grains[8].equivalentRadius

grains[8].equivalentPerimeter

# ## Departure from a circle
#
# The `shapeFactor` is $F=P/P_{\mathrm{eq}}$, where $P$ is the perimeter and
# $P_{\mathrm{eq}}$ is the equivalent perimeter. It is at least 1 and grows when a grain
# is elongated, indented, or ragged.
#
# Other literature sometimes calls $4\pi A/P^2=1/F^2$ the circularity or even the shape
# factor. State the formula when comparing values between software or publications.

# +
shapeFactorSummary = np.array([np.min(grains.shapeFactor()), np.median(grains.shapeFactor()), np.max(grains.shapeFactor())])
shapeFactorSummary

plot(grains, grains.shapeFactor())
mtexColorbar(title='shape factor')
# -

# The minimum, median, and maximum are 1.058, 1.221, and 1.662. In the map, the
# high-value yellow grains are visibly lobed or elongated, while compact grains are dark
# blue. The measure cannot tell those causes apart: a smooth ellipse and a round grain
# with a frayed boundary can have the same value.
#
# The same information can be scaled as the relative difference between the `perimeter`
# and the `equivalentPerimeter`.

# +
relativePerimeter = (grains.perimeter - grains.equivalentPerimeter) / grains.perimeter
plot(grains, relativePerimeter)
setColorRange([0, 0.5])
mtexColorbar(title='relative perimeter difference')
# -

# Round shapes are near zero in this map. The quantity equals $1-1/F$ and is bounded
# above by 1 rather than being unbounded.
#
# A third measure compares the perimeter with the grain's own convex hull. `paris`, the
# Percentile Average Relative Indented Surface, is reported in percent. An elongated but
# smoothly bounded grain has a small `paris` because its hull follows it closely. Bays and
# inlets increase it. Inclusion loops are excluded from both `paris` and the default
# perimeter. [Convex Hull Parameters](HullBasedParameters.html) develops this comparison.

# +
parisSummary = np.array([np.median(grains.paris()), np.max(grains.paris())])
parisSummary
# -

shapeParisCorrelation = np.corrcoef(grains.shapeFactor(), grains.paris())[0, 1]
shapeParisCorrelation

# +
shapeFactorExtreme = np.argmax(grains.shapeFactor())
parisExtreme = np.argmax(grains.paris())
sameExtremeGrain = shapeFactorExtreme == parisExtreme
sameExtremeGrain
# -

# +
mapShapeFactorExtreme = np.argmax(mapGrains.shapeFactor())
mapParisExtreme = np.argmax(mapGrains.paris())
sameExtremeWithEdgeGrains = mapShapeFactorExtreme == mapParisExtreme
sameExtremeWithEdgeGrains
# -

plot(grains, grains.paris())
mtexColorbar(title='paris')

# The median `paris` is 2.77 percent, its maximum is 43.30, and its correlation with
# `shapeFactor` is 0.472. Among the complete grains, both measures select the same extreme
# grain. In the full plotted map, including truncated edge grains, they select different
# extremes. Their partial agreement is the point: `shapeFactor` counts elongation as a
# departure from a circle, whereas `paris` does not. The changed selection also shows why
# the edge-grain rule belongs in a reported method.

# ## Perimeter-area fractal dimension
#
# Boundary length depends on the scale at which it is measured. This is the coastline
# problem. A perimeter-area fractal dimension describes how rapidly perimeter grows with
# grain size across a population and has been related to dynamic-recrystallisation
# conditions.
#
# This is a population estimator, not a box-counting dimension of one grain. It assumes
# that the grains are statistically self-similar over the fitted size range. If
# $P\propto r^D$, a straight line fitted to $\log(P)$ against $\log(r)$ has slope $D$. A
# family of geometrically similar smooth grains has $D=1$; increasingly convoluted
# boundaries can give a slope between 1 and 2.

# +
# reconstruct all indexed grains in the full data set once
ebsd = mtexdata('forsterite')
rawGrains = calcGrains(ebsd['indexed'])
rawGrains = rawGrains[~rawGrains.isBoundary]

# use five smoothing iterations for the main estimate
grains = smoothBoundary(rawGrains, 5)

plt.figure()
plt.scatter(grains.equivalentRadius, grains.perimeter)
plt.xlabel('equivalent radius')
plt.ylabel('perimeter')

# set both axes to logarithmic scales
logAxis(plt.gca(), [10, 10 ** 4], [10 ** 2, 10 ** 5])

# fit and draw a straight line in log space
ab = np.polyfit(np.log(grains.equivalentRadius), np.log(grains.perimeter), 1)
fitRadius = np.array([10, 10 ** 4])
plt.plot(fitRadius, np.exp(ab[1] + ab[0] * np.log(fitRadius)), linewidth=3)
# -

# The points scatter around the fitted line because individual grains are not scaled
# copies of one shape. The fitted slope, 1.011, is the estimated perimeter-area fractal
# dimension.

fractalDimension = ab[0]
fractalDimension

# Treat the estimate with care. Pixel spacing sets the smallest visible feature, small
# grains provide very few boundary samples, and smoothing removes exactly the excursions
# that the measure is intended to count. The next comparison reuses one reconstruction so
# that only the boundary processing changes. The zero-iteration case is the actual pixel
# staircase; calling `smoothBoundary(rawGrains, 0)` would still simplify and resample it.

for iter in [0, 5, 25]:
  g = rawGrains if iter == 0 else smoothBoundary(rawGrains, iter)
  ab = np.polyfit(np.log(g.equivalentRadius), np.log(g.perimeter), 1)
  print(f'{iter:2d} smoothing iterations: fractal dimension {ab[0]:.3f}')

# The pixel staircase gives 1.128 because grid corners add artificial length. Five
# iterations give 1.011, while 25 iterations push the estimate to 0.965, below the range
# of a self-similar plane curve. That is a diagnostic that the estimator's assumptions
# have failed, not a physical property of the rock. Compare specimens only after matching
# the pixel resolution, grain-size cutoff, segmentation, and smoothing procedure.

# ## Further reading
#
# * [ISO 13067:2020](https://www.iso.org/standard/74309.html), *Microbeam analysis -
#   Electron backscatter diffraction - Measurement of average grain size*. It distinguishes
#   measurements on a two-dimensional section from inferences about three-dimensional
#   grain size.
#
# * R. Heilbronner and S. Barrett,
#   [Image Analysis in Earth Sciences: Microstructures and Textures of Earth Materials](https://doi.org/10.1007/978-3-642-10343-8),
#   Springer, 2014. The textbook develops two- and three-dimensional grain-size
#   distributions together with particle and surface fabrics.
#
# * B. B. Mandelbrot,
#   [How Long Is the Coast of Britain? Statistical Self-Similarity and Fractional Dimension](https://doi.org/10.1126/science.156.3775.636),
#   *Science* 156 (1967), 636-638. This paper explains why measured length depends on
#   scale.
#
# * M. Takahashi and H. Nagahama,
#   [The Sections' Fractal Dimension of Grain Boundary](https://doi.org/10.1016/S0169-4332(01)00417-2),
#   *Applied Surface Science* 182 (2001), 297-301. It gives the perimeter-diameter
#   construction used above for dynamically recrystallised quartz.
#
# * S. E. Johnson et al.,
#   [EBSD-Based Calibration of Differential Stress From Experimentally Deformed Black Hills Quartzite Using the Perimeter-Area Fractal Dimension](https://doi.org/10.1029/2024JB030866),
#   *Journal of Geophysical Research: Solid Earth* 130 (2025), e2024JB030866. The study
#   tests pixel spacing, minimum grain size, and MTEX smoothing before calibration.

# ## Next
#
# [Ellipse Based Shape Parameters](EllipseBasedParameters.html) replaces each grain by a
# moment-equivalent ellipse and introduces shape preferred orientation.
# [Convex Hull Parameters](HullBasedParameters.html) isolates indentations, while
# [Projection Parameters](ProjectionBasedParameters.html) measures directional widths.
# Continue to [Grain Boundaries](GrainBoundaries.html) when the segments and junctions,
# rather than whole-grain outlines, are the subject.

# ## Technical details
#
# The shape parameters MATLAB implements as methods, `shapeFactor`, `paris`, `caliper`,
# are methods with parentheses here; `area`, `perimeter`, `boundarySize`, `diameter`,
# `equivalentRadius` and `equivalentPerimeter` are properties as in MATLAB. Grain nine of
# MATLAB is `grains[8]`. MATLAB's `histogram(grains.area)` and `scatter` are matplotlib's
# `plt.hist` and `plt.scatter`, `polyfit` is `np.polyfit`.
