# # Grain Boundary Smoothing
#
# A *grain* is a phase-homogeneous, spatially connected region of EBSD pixels produced by
# segmentation. This page assumes that the grains have already been reconstructed as in
# [Grain Reconstruction](GrainReconstruction.html).
# [Grain Boundary Properties](BoundaryProperties.html) introduces the boundary segments
# and their links to the neighbouring pixels.
#
# EBSD data is measured on a regular grid, so a grain boundary comes out of
# [calcGrains](EBSD.calcGrains.html) as a staircase of pixel edges. Every segment runs
# along one of the grid axes, whatever direction the boundary actually has. The staircase
# makes the boundary too long, quantizes its direction to a few values, and makes its
# curvature meaningless.
#
# [smoothBoundary](grain2d.smoothBoundary.html) repairs this geometry.

# +
import matplotlib.pyplot as plt
import numpy as np
from mtex import *

ebsd = mtexdata('csl')
grains = calcGrains(ebsd, minPixel=3)

grainsSmooth = smoothBoundary(grains)

ipfKey = ipfColorKey(grainsSmooth)
grainColor = ipfKey.orientation2color(grainsSmooth.meanOrientation)
plot(grainsSmooth, grainColor, micronbar='off')
# -

# The colours distinguish the mean grain orientations. Smoothing changes the polygonal
# outlines, but it does not change the EBSD measurements, their orientations, or which
# measurements belong to each grain.
#
# This page is about choosing and using the smoothing. How the stages work, and the full
# list of algorithms, is covered in [Smoothing Algorithms](GrainSmoothingAdvanced.html).

# ## What it does
#
# Boundary smoothing has two stages. It first moves the boundary vertices towards a smooth
# curve. It then removes the vertices the smooth curve no longer needs.
#
# The first stage never moves the boundary across the centre of a pixel it was measured
# next to. A vertex of the staircase is a corner of four pixels. It may only move within
# the square spanned by those four pixel centres, shrunk by a twentieth of the pixel
# spacing. So every pixel stays on the side of the boundary it was measured on.
#
# The measured boundary is grey below, the result is magenta, and the black dots are the
# pixel centres.

# +
plot(grains.boundary, lineWidth=4, lineColor='LightGray', displayName='measured', micronbar='off', region=[313, 353, 140, 156])
hold(True)
plot(grainsSmooth.boundary, lineWidth=2, lineColor='Fuchsia', displayName='smoothed')
line(ebsd.x, ebsd.y, lineStyle='none', marker='.', markerSize=6, color='k', label='pixel centers')
hold(False)
legend()
# -

# The magenta curves cut across the pixel steps, but each passes between the same pixel
# centres as the grey staircase. The junctions move a little, each within the square
# around it.
#
# The second stage straightens what is straight. It drops every vertex whose removal moves
# the boundary by less than a tenth of the pixel spacing, unless that brings a pixel centre
# closer than a twentieth of the spacing.
#
# A staircase along a boundary at 45 degrees is $\sqrt{2}$ times as long as the diagonal
# it represents. The whole map contains many boundary directions, so its measured change
# is data-dependent.

lenRaw = np.sum(grains.boundary['indexed'].segLength)
lenSmooth = np.sum(grainsSmooth.boundary['indexed'].segLength)
print(f'total boundary length: {lenRaw:.0f} {grains.scanUnit} measured, {lenSmooth:.0f} {grains.scanUnit} smoothed'
      f' - {100 * (1 - lenSmooth / lenRaw):.0f}% shorter')

# The direction distribution shows the same grid artifact. The raw boundary follows two
# antipodal grid axes, drawn as four spikes at multiples of 90 degrees. Smoothing spreads
# boundary length over all directions. A few directions still collect more length than
# their neighbours: the grid axes and their diagonals, and those of long straight
# boundaries in this map. Each panel has its own radial scale, so compare the angular
# spread rather than the bar heights.

# +
plt.figure(figsize=(9, 4.5), layout='constrained')
plt.subplot(1, 2, 1, projection='polar')
histogram(grains.boundary['indexed'].direction, 180, weights=grains.boundary['indexed'].segLength)
plt.title('measured')
plt.subplot(1, 2, 2, projection='polar')
histogram(grainsSmooth.boundary['indexed'].direction, 180, weights=grainsSmooth.boundary['indexed'].segLength)
plt.title('smoothed')
# -

# ## A measurement choice, not recovered truth
#
# Smoothing suppresses a sampling artifact. It cannot recover the unmeasured sub-pixel
# path of the physical boundary. It replaces grid bias with a smaller, method-dependent
# geometric bias.
#
# For quantitative comparisons, use the same reconstruction threshold, filter, and
# smoothing parameter for every map. Report those choices and repeat important
# measurements at nearby settings to check their sensitivity.

# ## How much to smooth
#
# The second argument is the number of smoothing steps, 20 by default. Each step is a
# smoothing pass followed by a slightly larger unshrinking pass. More steps give a
# straighter boundary, but no step moves a vertex out of its square. A step count is not a
# physical length, so the same count can have a different effect when the boundary
# sampling changes.

# +
iterations = [2, 5, 20, 40]
color = plt.get_cmap('copper')(np.linspace(0, 1, len(iterations) + 1))

plot(grains.boundary, lineWidth=1, lineColor='LightGray', displayName='measured', micronbar='off', region=[313, 353, 140, 156])
for i, n in enumerate(iterations):
  hold(True)
  plot(smoothBoundary(grains, n).boundary['indexed'], lineWidth=2, lineColor=color[i], displayName=f'{n} steps')
hold(False)
legend()
# -

# Two and five steps still follow the pixel corners. At 20 steps the curve runs straight
# between the bends of the boundary. Forty steps change it little more, because every
# vertex is held in its square.

# ## Which algorithm - the short version
#
# A [boundaryFilter](boundaryFilter.html) selects the algorithm of the first stage. The
# default is [taubinFilter](taubinFilter.html), the pair of passes described above. It
# suits plots, directions and lengths as well as grain areas and shape parameters.
#
# [laplaceFilter](laplaceFilter.html) applies the smoothing passes alone. It shrinks: every
# pass pulls a convex grain inward, until the squares stop it. Taubin's unshrinking pass
# gives the area back.

# +
A0 = grains.area
d = np.median(grains.boundary.segLength)
big = A0 > 10 * d ** 2
A0 = A0[big]

grainsLaplace = smoothBoundary(grains, laplaceFilter(25))
aL = grainsLaplace.area[big]
grainsTaubin = smoothBoundary(grains, 25)
aT = grainsTaubin.area[big]

print('grain area after 25 steps')
print(f'  laplaceFilter {100 * np.mean((aL - A0) / A0):+6.2f}% on average, '
      f'{100 * np.min((aL - A0) / A0):+6.1f}% for the worst grain')
print(f'  taubinFilter  {100 * np.mean((aT - A0) / A0):+6.2f}% on average, '
      f'{100 * np.min((aT - A0) / A0):+6.1f}% for the worst grain')
# -

# The areas are compared with those of the measured staircase. The comparison excludes
# grains with an area no larger than $10 d^2$, where $d$ is the pixel spacing. The
# Laplacian loses area on average and much more on its worst grain. Taubin keeps the mean
# area within a fraction of a percent, without promising that every grain keeps its area
# exactly.
#
# The difference is visible on the grain whose area is closest to 30 square map units. The
# measured staircase is grey, the Laplacian is magenta, and Taubin is blue.

# +
A = grains.area
gid = int(np.argmin(np.abs(A - 30)))
c = grains[gid].centroid

plot(grains[gid].boundary, lineWidth=4, lineColor='LightGray', displayName='measured', micronbar='off',
     region=[c.x[0] - 6, c.x[0] + 6, c.y[0] - 6, c.y[0] + 6])
hold(True)
plot(grainsLaplace[gid].boundary, lineWidth=2.5, lineColor='Fuchsia', displayName='Laplace, 25 steps')
plot(grainsTaubin[gid].boundary, lineWidth=2.5, lineColor='DodgerBlue', displayName='Taubin, 25 steps')
hold(False)
legend()
# -

# The magenta Laplacian cuts inside the blue Taubin curve along the right edge of this
# grain. Both stay close to the staircase, because the squares hold their vertices.

# ## A physical smoothing length
#
# [curvatureFilter](curvatureFilter.html) replaces the step count with a length: the
# wavelength whose amplitude is reduced by half. Finer detail is attenuated more strongly,
# while coarser detail is preserved more strongly. Because the parameter is a length, it
# has the same meaning for scans made at different step sizes.

# +
F = curvatureFilter()
F.smoothingLength = 8   # in the units of the map, here um

plot(grains.boundary, lineWidth=4, lineColor='LightGray', displayName='measured', micronbar='off', region=[313, 353, 140, 156])
hold(True)
plot(smoothBoundary(grains, F).boundary, lineWidth=2.5, lineColor='Orange', displayName='curvatureFilter, 8 um')
hold(False)
legend()
# -

# The orange curve removes pixel-scale turns but follows the larger bends of the measured
# boundary. Use one physical smoothing length across scans when their spatial resolutions
# differ.
#
# [huberFilter](huberFilter.html) is available for faceted materials. It keeps genuine
# corners sharp instead of rounding them off. It is not a good default; see
# [Smoothing Algorithms](GrainSmoothingAdvanced.html).

# ## When to switch the second stage off
#
# The second stage changes the number of boundary segments. A segment that no longer runs
# between one specific pair of pixels gets the `gB.ebsdId` 0. Wherever `gB.ebsdId` is read
# per segment, for example to look up the orientations on either side, switch the second
# stage off.

# +
grainsPlain = smoothBoundary(grains, simplify=False)

print(f'boundary segments: {len(grains.boundary)} measured, {len(grainsSmooth.boundary)} after smoothing, '
      f'{len(grainsPlain.boundary)} without the second stage')
# -

# Only the last count remains equal to the measured count. It therefore keeps one valid
# `ebsdId` pair for every original pixel-edge segment.

# ## Junctions and triple points
#
# A *junction* is a boundary vertex at which the number of meeting segments is not two. It
# includes triple and quadruple points as well as a loose end. A triple or quadruple point
# moves with the smoothing, but only within the square of the pixel centres around it, or
# within half a pixel spacing where no pixel edge meets it. A loose end, where an inner
# boundary fades out, stays where it is. Segment connectivity is unchanged, so the same
# grains meet.
#
# The junctions on the outer boundary and the corners of its convex hull remain fixed
# unless `moveOuterBoundary=True` is passed, so the map keeps its extent.

# ## References
#
# * S. F. F. Gibson, [Constrained Elastic Surface Nets: Generating Smooth Surfaces from
#   Binary Segmented Data](https://doi.org/10.1007/BFb0056277), _MICCAI 1998_, LNCS 1496,
#   888-898, smooths a voxel surface while holding every vertex in its cell, the idea of
#   the first stage.
# * D. H. Douglas and T. K. Peucker, "Algorithms for the Reduction of the Number of Points
#   Required to Represent a Digitized Line or Its Caricature", _The Canadian Cartographer_
#   10 (1973), 112-122, [doi:10.3138/FM57-6770-U75U-7727](https://doi.org/10.3138/FM57-6770-U75U-7727).
# * M. de Berg, M. van Kreveld and S. Schirra, [Topologically Correct Subdivision
#   Simplification Using the Bandwidth Criterion](https://doi.org/10.1559/152304098782383007),
#   _Cartography and Geographic Information Systems_ 25 (1998), 243-257, simplify a line
#   without moving it across given points, the constraint of the second stage.
# * G. Taubin, "A Signal Processing Approach to Fair Surface Design", _Proceedings of
#   SIGGRAPH 1995_, 351-358, [doi:10.1145/218380.218473](https://doi.org/10.1145/218380.218473).
# * P. J. Huber, "Robust Estimation of a Location Parameter", _The Annals of Mathematical
#   Statistics_ 35 (1964), 73-101, [doi:10.1214/aoms/1177703732](https://doi.org/10.1214/aoms/1177703732).
# * S. Fan et al., "Using Grain Boundary Irregularity to Quantify Dynamic
#   Recrystallization in Ice", _Acta Materialia_ 209 (2021), 116810,
#   [doi:10.1016/j.actamat.2021.116810](https://doi.org/10.1016/j.actamat.2021.116810).
# * For standardized EBSD grain-size definitions and scope, see
#   [ISO 13067:2020](https://www.iso.org/standard/74309.html) and
#   [ASTM E2627-13(2019)](https://doi.org/10.1520/E2627-13R19).

# ## Next
#
# Continue with [Smoothing Algorithms](GrainSmoothingAdvanced.html) for the two stages in
# detail, filter parameters, and quantitative comparisons. Then see
# [Boundary Curvature](BoundaryCurvature.html) for a measurement that depends directly on
# the smoothed geometry. The grain-level consequences for area, perimeter, and outline
# descriptors continue in [Shape Parameters](ShapeParameters.html).

# ## Technical Details
#
# MATLAB's flag `'noSimplify'` is the keyword `simplify=False`, and `'moveOuterBoundary'` is
# `moveOuterBoundary=True`. A filter is constructed with parentheses,
# `laplaceFilter(25)` and `curvatureFilter()`, and its smoothing length is set on the
# object. The rose diagram of boundary directions is `histogram(v, numBin, weights=)` on a
# polar axes, MATLAB's `histogram(v, weights, n)`. The pixel centres are drawn by
# `line(x, y, ...)` with a marker and no line, which places them on the map as the map is
# drawn, MATLAB's `plot(x, y, '.k')`. `grains.scanUnit` carries the unit of the map the
# grains came from. MATLAB's zoom `axis([x0 x1 y0 y1])` after a map plot is
# `region=[x0, x1, y0, y1]` on its first plot, in the coordinates of the map whatever the
# plotting convention.
