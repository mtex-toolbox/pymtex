# # Quadruple Points
#
# A *junction* is a boundary vertex where the number of meeting grain boundary segments is
# not two. On a square measurement grid, four pixels meet at every interior vertex. A
# checkerboard arrangement of two labels can therefore create a junction with four
# boundary segments - a *quadruple point*.
#
# The ambiguity is diagonal connectivity. Connecting one pair of like pixels separates the
# other pair, while connecting the other pair reverses that decision. The resulting grain
# count can depend on which diagonal is chosen because a grain is a spatially connected
# region.
#
# Exact four-way contacts in a two-dimensional physical network are usually unstable and
# split into three-segment junctions. The grid contact on this page is instead a
# digital-topology ambiguity introduced by sampling. In three dimensions, a physical
# quadruple point has a different meaning.
#
# [calcGrains](EBSD.calcGrains.html) resolves this ambiguity during reconstruction, and
# `removeQuadruplePoints=False` keeps the four-way contacts. This page assumes the
# reconstruction concepts introduced in [Grain Reconstruction](GrainReconstruction.html).

import numpy as np
from mtex import *

# display matrix rows downwards and columns to the right
plottingConvention.default('y↓→x')

# ## Construct a diagonal contact
#
# This artificial map contains one crystal phase and two orientations. The pixels selected
# by `ring` form a ring whose ends touch diagonally at its lower corner.

# +
cs = crystalFrame('1', mineral='test')

ring = np.array([
  [0, 0, 0, 0, 0, 0],
  [0, 1, 1, 1, 1, 0],
  [0, 1, 1, 1, 1, 0],
  [0, 1, 0, 0, 1, 0],
  [0, 1, 0, 0, 1, 0],
  [0, 1, 1, 1, 0, 0],
  [0, 0, 0, 0, 0, 0]]) == 1

# one orientation on the ring and the identity on the background
rot = rotation.byEuler(np.where(ring, 130 * degree, 0.0), np.where(ring, 120 * degree, 0.0), np.where(ring, 110 * degree, 0.0))

ebsd = EBSDsquare(rot, np.ones(ring.shape, dtype=int), ['notIndexed', cs])

# compute orientation colours explicitly to keep the output focused
colorKey = ipfColorKey(ebsd)
ebsdColors = colorKey.orientation2color(ebsd.orientations)
plot(ebsd, ebsdColors, micronbar='off')
# -

# The two colours meet in a checkerboard pattern at the lower contact. The picture alone
# cannot say which diagonal should be connected.

# ## Reconstruct without resolving the junction
#
# With the quadruple points kept, the reconstruction closes the oriented ring at the
# diagonal contact. It then treats the identity-oriented interior and exterior as separate
# grains, so the map contains three grains rather than two.

# +
grainsWithQuadruple = calcGrains(ebsd, angle=10 * degree, removeQuadruplePoints=False)

print(f'ordinary reconstruction: {len(grainsWithQuadruple)} grains')

grainColors = colorKey.orientation2color(grainsWithQuadruple.meanOrientation)
plot(grainsWithQuadruple, grainColors, micronbar='off')
hold(True)
plot(grainsWithQuadruple.boundary, lineWidth=2)
hold(False)
# -

# The interior patch has its own outline. At the diagonal contact, four thick boundary
# segments end at the same vertex.

# ## Resolve the junction during reconstruction
#
# By default `calcGrains` joins the diagonal pair whose pixels lie in two grains the
# criterion connects, so that joining them merges the grains: here the like-oriented
# interior and exterior, since the ring pixels were already connected to each other
# elsewhere. The expected two grains remain.

# +
grains = calcGrains(ebsd, angle=10 * degree)

print(f'resolved reconstruction: {len(grains)} grains')
print(f'total boundary length: {np.sum(grainsWithQuadruple.boundary.segLength):g} before, '
      f'{np.sum(grains.boundary.segLength):g} after')
print(f'strict triple points: {len(grains.triplePoints)}')

grainColors = colorKey.orientation2color(grains.meanOrientation)
plot(grains, grainColors, micronbar='off')
hold(True)
plot(grains.boundary, lineWidth=2)
hold(False)
# -

# The interior outline has joined the exterior network. The four-segment vertex has become
# two three-segment junctions a quarter of a pixel apart, each moved an eighth of a pixel
# into the wedge of the cell it was not joined to, so the two joined pixels meet along a
# narrow neck and the total boundary length changes by a fraction of a pixel. In this
# two-grain example the junctions are not strict *triple points*, because a triple point
# must separate three distinct real grains. See [Triple Points](TriplePoints.html) for
# that distinction. A contact whose both diagonals the criterion separates stays a
# quadruple point.

# ## Smooth the opened contact
#
# [smoothBoundary](grain2d.smoothBoundary.html) moves every interior junction within the
# square of the pixel centres around it, so the neck widens into a smooth waist. Its default
# [taubinFilter](taubinFilter.html) suppresses the pixel staircase while limiting the
# systematic area loss of Laplacian smoothing, and every grain is given its measured area
# back afterwards.

# +
grains = smoothBoundary(grains)

grainColors = colorKey.orientation2color(grains.meanOrientation)
plot(grains, grainColors, lineWidth=2, micronbar='off')
# -

# The contact has opened into a waist. Smoothing moves the geometry; the topological
# correction was made in the reconstruction.

# ## Curvature at the opened contact
#
# Signed [curvature](grainBoundary.curvature.html) makes the opened pinch visible as
# neighbouring bends in opposite directions.

# +
gB = grains[1].boundary

plot(gB, gB.curvature(10), lineWidth=6, micronbar='off')
mtexColorMap('blue2red')
setColorRange(0.5 * np.array([-1, 1]))
mtexColorbar(title='signed curvature in 1/grid unit')
# -

# The blue and red extrema beside the pinch have opposite signs. The sign depends on the
# stored walk direction, so use [Boundary Curvature](BoundaryCurvature.html) before
# interpreting it as convex or concave relative to a particular grain.

# ## References
#
# * T. Y. Kong and A. Rosenfeld,
#   [Digital topology: Introduction and survey](https://doi.org/10.1016/0734-189X(89)90147-3),
#   _Computer Vision, Graphics, and Image Processing_ 48 (1989), 357--393, develops the
#   adjacency choices behind diagonal connectivity on a digital grid.
# * C. Herring,
#   [Surface Tension as a Motivation for Sintering](https://doi.org/10.1007/978-3-642-59938-5_2),
#   in _The Physics of Powder Metallurgy_ (1951), 143--179, gives the classical capillary
#   balance at physical junctions.
# * P. R. Rios and M. E. Glicksman,
#   [Grain boundary, triple junction and quadruple point mobility controlled normal grain growth](https://doi.org/10.1080/14786435.2015.1050476),
#   _Philosophical Magazine_ 95 (2015), 2092--2127, distinguishes the roles of boundaries,
#   triple junctions, and quadruple points in grain-growth models.

# ## Next
#
# Continue with [Boundary Intersections](BoundaryIntersections.html) for geometric
# crossings between boundaries and other curves. For the two operations used here, see
# [Grain Boundary Smoothing](GrainSmoothing.html) and
# [Boundary Curvature](BoundaryCurvature.html).

# ## Technical Details
#
# `EBSDsquare(rotations, phaseId, CSList)` builds a map on a square grid from an array of
# rotations, MATLAB's constructor of the same name; the rows of the array are the rows of
# pixels. An array of identities is written as `rotation.byEuler` of three arrays, where
# MATLAB has `rotation.id(size)`.
#
# MATLAB numbers the ring grain 1 and the surrounding grain 2, the port the other way
# round, so the curvature is drawn for `grains[1]` where MATLAB writes `grains(1)`. The
# smaller id is the first side of a segment in both, so MATLAB walks the ring with the ring
# on its left and the port with the surrounding grain on its left, and the signs, the
# colours of the figure, are reversed. The inverse pole figure key of the triclinic frame colours the two orientations
# grey and purple where MATLAB colours them red and mint.
