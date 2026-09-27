# # Convex Hull Based Shape Parameters
#
# Stretch a rubber band around a grain and let it snap tight: what you get is the convex
# hull. A convex grain is its own hull, however large or elongated it is. A bay or inlet
# leaves a gap between the grain and its hull, so comparing the two isolates lobateness:
# departure from a convex outline.
#
# This page assumes the direct measurements introduced in
# [Shape Parameters](ShapeParameters.html). The shapes below are artificial, chosen so
# that each measure has something to react to.

import numpy as np

from mtex import *

# +
# import the artificial grain shapes
plottingConvention.default('y↑→x')
grains = mtexdata('testgrains')

# select and smooth a few interesting grains
grains = smoothBoundary(grains['id', [2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]], 3)
# -

# [Boundary smoothing](GrainSmoothing.html) is not optional here. On an unsmoothed square
# grid every boundary segment is horizontal or vertical. The `perimeter` then measures the
# cityblock distance and comes out too long, while the convex hull cuts the corners. The
# difference between the two would be dominated by the grid. For very small grains even
# smoothing does not save it, and the numbers below should not be trusted.
#
# `hull` returns the hulls as grains, so everything that works on grains works on them.

# +
# compute convex hull grains
chGrains = grains.hull()

# plot the original grains
plot(grains, micronbar=False, legend=False)

# and on top of them the convex hull
hold(True)
plot(chGrains.boundary, lineWidth=2, lineColor='r')
text(grains, grains.id)
hold(False)
# -

# Where a grain is convex the red line follows its outline; where it is indented the line
# cuts across. The labels are grain ids used in the table below. Hulls may overlap even
# when the grains do not. A hull outline no longer separates two grains. The second phase
# of every hull boundary segment is therefore set to `notIndexed`.

# ## Four ways to measure an indentation
#
# The relative difference between the perimeter of the grain and that of its hull reacts
# most strongly to thin, narrow indentations - a crack that does not quite cut the grain in
# two adds a great deal of perimeter and almost no area. The result, `deltaP`, is a
# dimensionless percentage.

deltaP = 100 * (grains.perimeter - chGrains.perimeter) / grains.perimeter

# `paris`, the Percentile Average Relative Indented Surface, compares the same two
# perimeters relative to the hull and doubles that ratio. It is the conventional name in
# the literature, and `grains.paris()` computes it directly.

paris = 200 * (grains.perimeter - chGrains.perimeter) / chGrains.perimeter

# Comparing the areas instead of the perimeters reacts to broad, shallow lobes - a big
# bite out of a grain, which changes the area a lot and the perimeter hardly at all. The
# result, `deltaA`, is 100 times one minus the quantity often called solidity, the grain
# area divided by its hull area.

deltaA = 100 * (chGrains.area - grains.area) / chGrains.area

# Since the two react to different kinds of indentation, combining them gives `radiusD`,
# a measure that responds to both.

radiusD = np.sqrt(deltaP ** 2 + deltaA ** 2)

# ## Compare the measures
#
# The table makes the values behind the colour maps explicit. A zero in a perimeter-based
# column means that the outer outline is convex. `deltaA` is zero only when the grain
# fills its hull, with no hole. `deltaP` and `deltaA` are below 100, whereas `paris` has
# no upper bound despite being reported as a percentage.

print(f"{'grainId':>8}{'hasHole':>9}{'deltaP':>9}{'paris':>9}{'deltaA':>9}{'radiusD':>9}")
for k in range(grains.size):
  print(f"{grains.id[k]:>8}{str(bool(grains.hasHole[k])):>9}{deltaP[k]:>9.3f}{grains.paris()[k]:>9.3f}{deltaA[k]:>9.3f}{radiusD[k]:>9.3f}")

# Drawn side by side on the same shapes, the measures separate narrow from broad
# indentations:

# +
newMtexFigure(layout=[2, 2])
plot(grains, deltaP, micronbar=False)
mtexTitle('deltaP')

nextAxis()
plot(grains, grains.paris(), micronbar=False)
mtexTitle('paris')

nextAxis()
plot(grains, deltaA, micronbar=False)
mtexTitle('deltaA')

nextAxis()
plot(grains, radiusD, micronbar=False)
mtexTitle('radiusD')
mtexColorbar()
# -

# The first two maps rank the grains identically, as they must, being the same ratio
# written two ways - only the scale differs, `paris` running well beyond `deltaP`. The
# third map ranks them differently: the lobed clover scores high on `deltaP`, all
# perimeter and almost no missing area, while the two discs with a hole score low on
# `deltaP` - as convex as the plain disc - and high on `deltaA`. `radiusD` puts both kinds
# high, which is what combining them is for.
#
# The discs with holes are worth a second look, because they show what these measures are
# and are not counting. A hole and an inclusion are the same enclosure viewed from
# opposite sides. It is not an indentation of the outer outline, so the perimeter-based
# measures ignore it; `paris` explicitly removes inclusion loops before it measures. The
# area-based measure cannot ignore it, since the hull contains the hole and the grain does
# not. If a hole is not what you mean by lobateness, use `deltaP` or `paris`; if it is,
# use `deltaA`.
#
# Which measure to report otherwise depends on the process being described. A
# dissolution front eats broad bays and shows up in `deltaA`; a partly healed fracture is
# a thin slot and shows up in `deltaP`.
#
# These values also depend on pixel spacing and boundary smoothing. Use the same
# `smoothBoundary` settings for every map being compared, report them with the
# segmentation and resolution, and state the formula because other software uses names
# such as convexity and solidity for several different ratios.

# ## References
#
# * Panozzo, R. and Hürlimann, H. (1983),
#   [A simple method for the quantitative discrimination of convex and convex-concave lines](https://micro.earth.unibas.ch/contact/RH.html),
#   *Microscopica Acta* 87, 169-176, introduced the PARIS factor.
# * Heilbronner, R. and Keulen, N. (2006),
#   [Grain size and grain shape analysis of fault rocks](https://doi.org/10.1016/j.tecto.2006.05.020),
#   *Tectonophysics* 427, 199-216, developed the hull-area difference and applied both
#   descriptors to fault rocks.
# * Back, A. L., Kana Tepakbong, C., Bédard, L. P. and Barry, A. (2025),
#   [From rocks to pixels](https://doi.org/10.3389/feart.2025.1508690), *Frontiers in Earth
#   Science* 13, reviews grain-shape descriptors and their inconsistent nomenclature
#   across fields.

# ## Next
#
# A convex hull discards every indentation. The
# [Projection Parameters](ProjectionBasedParameters.html) page instead keeps the width of
# each grain as a function of direction and develops shape-preferred orientation from
# those widths.

# ## Technical details
#
# MATLAB's `testgrains` is a stored map of forty artificial shapes; this port draws its
# own map of fifteen (a disc, discs with a small and a large hole, an ellipse, a
# rectangle and a turned one, a clover, a disc with a slit, an L, a triangle, a hexagon
# and a disc with a bay, in one background grain), so the ids and the table differ from
# the MATLAB page while the measures rank the same kinds of shape the same way. MATLAB's
# `table` is a printed table here.
