# # Line intersections
#
# A straight line drawn across a microstructure is a one-dimensional probe. Every grain
# boundary it crosses ends one intercept and starts the next. Counting these crossings is
# one of the oldest measurements in microscopy. On a micrograph it needs neither labelled
# grain regions nor a model for grain shape, only visible boundaries and a calibrated
# ruler.
#
# In MTEX the boundary network comes from [grain reconstruction](GrainReconstruction.html).
# A grain boundary starts as a list of short segments. Each segment lies between
# neighbouring EBSD pixels assigned to different grains. The
# [intersect](grainBoundary.intersect.html) command finds where a test line crosses that
# reconstructed boundary network.
#
# We use a magnesium map containing thin twin lamellae.
# [Smoothing](GrainSmoothing.html) removes the pixel-grid staircase, but it also moves the
# boundary. Use the same documented smoothing settings when comparing measurements between
# maps.

# +
import numpy as np
from mtex import calcGrains, hold, line, mtexdata, plot, plottingConvention, scatter, smoothBoundary

plottingConvention.default('y↑→x')
ebsd = mtexdata('twins')

grains = calcGrains(ebsd)
grains = smoothBoundary(grains)
gB = grains.boundary

plot(grains, faceColor=[0.8, 0.8, 0.8])
hold(True)
plot(gB, lineWidth=2)
hold(False)
# -

# The black network is the smoothed grain boundary. Notice the thin, elongated twin
# lamellae enclosed by much broader grains.

# ## Define a test line
#
# A line is specified by its start and end points in map coordinates.

# +
xy1 = [10, 10]   # start point
xy2 = [41, 41]   # end point

hold(True)
line([xy1[0], xy2[0]], [xy1[1], xy2[1]], LineStyle=':', LineWidth=4, Color='w')
hold(False)
# -

# The white diagonal crosses the lamellae obliquely. Its direction is part of the
# measurement, not merely a plotting choice.

# ## Find the crossed boundary segments
#
# `intersect` returns the segments the line crosses and the crossing points, in the order
# in which the line meets them.

# +
x, y = gB.intersect(xy1, xy2)
isIntersection = ~np.isnan(x)

hold(True)
scatter(x[isIntersection], y[isIntersection], 36, 'b', 'filled')
hold(False)
# -

# Each blue marker is a reported crossing. Adjacent boundary segments share vertices. A
# line through a vertex crosses two segments at the same place, so two markers can land on
# one geometric crossing. Avoid such vertices when placing test lines, or consolidate
# coincident points before counting them.

# ## Mean lineal intercept
#
# For a traverse of length $L$ with $N$ boundary crossings, the mean lineal intercept is
# $\bar{\ell} = L/N$. It is a standard measure of apparent grain size. Here the three
# quantities are printed because they are the result of the example.

# +
nIntersections = np.count_nonzero(isIntersection)
nIntersections
# -

# ---

# +
lineLength = np.linalg.norm(np.subtract(xy2, xy1))
lineLength
# -

# ---

# +
meanIntercept = lineLength / nIntersections
meanIntercept
# -

# This 43.84 µm traverse crosses 18 boundary segments. Its mean lineal intercept is
# 2.44 µm. This is apparent grain size on a two-dimensional section, not a mean grain
# diameter in three dimensions.

# ## From one traverse to a measurement
#
# One line is a thin estimate, and this one runs diagonally across a map full of twin
# lamellae, every one of which it counts. In practice, add the lengths of many parallel
# test lines and divide by their total number of crossings. Repeat this measurement in
# several directions for an elongated microstructure. A line along the elongation crosses
# fewer boundaries than one drawn across it.
#
# This directional sensitivity is useful, but it means that a single line must not be
# presented as a direction-independent grain size. A formal measurement also needs a
# sampling design. Follow the boundary-counting rules of the applicable standard.
#
# A single line still gives a useful reconstruction check. Compare its crossing count with
# what [the grain sizes](ShapeParameters.html) imply. If they disagree badly, the
# reconstruction is finding boundaries that are not there, or missing ones that are.

# ## Further reading
#
# * [ASTM E112-24, Standard Test Methods for Determining Average Grain Size](https://store.astm.org/e0112-24.html)
#   includes the Heyn linear-intercept procedure and sampling rules for metallic
#   microstructures.
# * [ISO 643:2024, Steels - Micrographic determination of the apparent grain size](https://www.iso.org/standard/82221.html)
#   covers intercept measurements for ferritic and austenitic steels.
# * J. C. Russ and R. T. DeHoff,
#   [Practical Stereology, 2nd ed.](https://doi.org/10.1007/978-1-4615-1233-2), 2000,
#   develops line probes, unbiased sampling, and anisotropy.
# * A. Thorvaldsen,
#   [The intercept method - 2. Determination of spatial grain size](https://doi.org/10.1016/S1359-6454(96)00198-X),
#   Acta Materialia 45 (1997), 595-600, explains why mean lineal intercept and mean
#   three-dimensional grain size are not generally proportional.

# ## Next
#
# A test line samples how often boundaries occur along one chosen direction.
# [Boundary Normal Distribution](BoundaryNormalDistribution.html) instead uses many
# boundary traces to estimate which interface planes are preferred.

# ## Technical Details
#
# `line` and `scatter` of two coordinate lists draw into the current map under its layout,
# as MATLAB's do.
