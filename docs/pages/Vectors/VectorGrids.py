# # Spherical Grids
#
# A spherical grid is a finite list of directions used to represent the sphere in a
# numerical calculation. Each direction in the list is a node. Such nodes are needed to
# integrate a spherical function, sample one, or draw it.
#
# A sphere cannot be covered by one rectangular angular mesh without distortion or a
# coordinate singularity. Grid constructions balance node spacing against other useful
# properties. These include equal-area cells, hierarchical structure, and regular indexing
# in spherical angles.
#
# Read [Vectors](Vectors.html) first for the MTEX direction model.
# [Spherical Projections](SphericalProjections.html) explains the upper-hemisphere view.
# The examples construct directions on the complete sphere. The plot option `upper` hides
# the lower hemisphere; it does not identify a direction with its negative. That
# distinction is explained in [Axes](VectorsAxes.html).
#
# For these figures, the plotting convention lays the reference frame out with y up and x
# to the right. It does not alter the constructed nodes.

from mtex import (HEALPixS2Grid, calcDensity, degree, equispacedS2Grid, fibonacciS2Grid, mtexColorbar, mtexTitle, nextAxis,
                  norm, plot, plottingConvention, regularS2Grid, setColorRange)

plottingConvention.default('y↑→x')

# ## Four Constructions
#
# MTEX offers the [regularS2Grid](regularS2Grid.html) and the
# [equispacedS2Grid](equispacedS2Grid.html). Two further choices are the
# [HEALPixS2Grid](HEALPixS2Grid.html) and [fibonacciS2Grid](fibonacciS2Grid.fibonacciS2Grid.html).
#
# The regular grid is a tensor product of polar and azimuth angles. The equispaced grid
# changes the number of azimuth steps from one latitude to the next. HEALPix supplies the
# centres of hierarchical, equal-area, iso-latitude pixels. The Fibonacci grid follows a
# golden-angle spiral.
#
# Each constructor accepts a target angular `resolution`. The target does not guarantee
# one exact nearest-neighbour distance. Each construction adjusts or rounds it according to
# its own geometry.

# + 
# a regular grid in the two spherical angles, the MTEX equispaced grid, the HEALPix grid
# and the Fibonacci grid
grids = [regularS2Grid(resolution=7 * degree),
         equispacedS2Grid(resolution=7 * degree),
         HEALPixS2Grid(resolution=7 * degree),
         fibonacciS2Grid(resolution=7 * degree)]

gridNames = ['regular', 'equispaced', 'HEALPix', 'Fibonacci']
# -

# The upper-hemisphere view makes the different node patterns visible.

# +
plot(grids[0], upper=True, layout=[1, 4])
mtexTitle(gridNames[0])

for k in range(1, 4):
  nextAxis()
  plot(grids[k], upper=True)
  mtexTitle(gridNames[k])
# -

# Notice how the latitude rows of the regular grid converge near the centre. At the pole,
# every azimuth represents the same direction. The equispaced and HEALPix patterns reduce
# the number of nodes on shorter latitude rings. The Fibonacci spiral has no latitude-ring
# structure.
#
# The requested resolution also produces different numbers of nodes.

nodeCounts = [len(g) for g in grids]
nodeCounts

# At $7^{\circ}$ the four constructions contain 1404, 812, 768 and 827 nodes,
# respectively. These counts include both hemispheres. They reflect how each constructor
# interprets the target resolution and are not by themselves a ranking of grid quality.

# ## Comparison of Uniformity
#
# Node uniformity can be measured instead of only judged from a plot. Here every node
# receives the same weight. [Density Estimation](VectorsDensityEstimation.html) smooths
# the resulting discrete measure into a function on the sphere. MTEX normalizes that
# function to have mean value 1. A uniform equal-weight node measure should therefore be
# close to the constant 1.
#
# This diagnostic measures equal-weight node placement at the chosen halfwidth. It does
# not test cell areas, nearest-neighbour distances, or the accuracy of a quadrature rule.

# +
density = [calcDensity(g, halfwidth=5 * degree) for g in grids]

plot(density[0], upper=True, layout=[2, 2])
mtexTitle(gridNames[0])
for k in range(1, 4):
  nextAxis()
  plot(density[k], upper=True)
  mtexTitle(gridNames[k])
setColorRange('equal')
mtexColorbar()
# -

# All four panels use the same colour range. The regular grid has a strong maximum at the
# pole, whereas the other three panels remain close to the uniform value 1. The numerical
# comparison below uses the complete sphere, not only the displayed upper hemisphere.
#
# The $L^2$ norm gives the square root of the integrated squared deviation.

l2Deviation = [norm(d - 1) for d in density]
l2Deviation

# The integrated absolute deviation is an $L^1$ measure of the same error.

l1Deviation = [abs(d - 1).sum() for d in density]
l1Deviation

# At a $5^{\circ}$ halfwidth, the $L^2$ deviations are 1.1324, 0.0089, 0.0120 and 0.0057.
# The $L^1$ deviations are 5.7668, 0.0600, 0.0674 and 0.0320. The regular grid is about
# two orders of magnitude less uniform by both measures in this experiment. The Fibonacci
# grid has the smallest deviation.
#
# This comparison is not a universal ranking. Changing the resolution or smoothing
# halfwidth changes the numbers, and a downstream algorithm may require cell areas,
# quadrature weights, or a particular grid structure.

# ## Choosing a Grid
#
# Use a regular grid when values must sit on a rectangular raster in the two spherical
# angles, for example for a surface or contour representation. Use an equispaced or
# Fibonacci grid when nearly uniform equal-weight nodes matter more than rectangular
# indexing. Use HEALPix when the centres of its standard equal-area pixelization are
# required.
#
# A nearly uniform point set is not automatically a quadrature rule. If an integral must
# be exact for a specified class of functions, use the nodes and weights prescribed by that
# integration method.

# ## Further Reading
#
# * E. B. Saff and A. B. J. Kuijlaars,
#   [Distributing many points on a sphere](https://doi.org/10.1007/BF03024331),
#   _The Mathematical Intelligencer_ 19(1), 5-11, 1997. This article surveys competing
#   meanings of a well-distributed point set.
# * K. M. Górski et al.,
#   [HEALPix: A Framework for High-Resolution Discretization and Fast Analysis of Data Distributed on the Sphere](https://doi.org/10.1086/427976),
#   _The Astrophysical Journal_ 622, 759-771, 2005. This paper defines HEALPix.
# * R. Swinbank and R. J. Purser,
#   [Fibonacci grids: A novel approach to global modelling](https://doi.org/10.1256/qj.05.227),
#   _Quarterly Journal of the Royal Meteorological Society_ 132, 1769-1793, 2006. This
#   paper develops the Fibonacci construction.
# * V. I. Lebedev and D. N. Laikov,
#   [A quadrature formula for a sphere of the 131st algebraic order of accuracy](https://www.mathnet.ru/eng/dan3035),
#   _Doklady Mathematics_ 59(3), 477-481, 1999. This paper illustrates nodes designed
#   together with weights and an exactness criterion.

# ## Next
#
# Grids on the rotation group are covered in [Orientation Grids](OrientationGrid.html).
# Those grids can respect crystal symmetry. Choosing informative nodes for a spherical
# function is a different question, treated in [Sampling](S2FunSampling.html).

# ## Technical Details
#
# The $L^2$ norm of a spherical function is taken here with respect to the normalised
# surface measure, so that the constant one has norm one, as its mean is one. MATLAB
# integrates against the surface measure itself and reports the same deviations larger by
# $\sqrt{4\pi}$: 4.0141, 0.0317, 0.0426 and 0.0201. The $L^1$ deviations are integrals
# on both sides and agree.
