# # Density Estimation
#
# Density estimation turns a finite sample into a continuous function. In texture analysis,
# the samples may be EBSD orientations, grain mean orientations, misorientation axes, or
# simulated orientations.
#
# Keep three parts of the calculation distinct:
#
# * **input:** samples $x_n$, possibly with statistical weights;
# * **assumption:** nearby samples belong to a smoothly varying density;
# * **result:** an estimate $f_N$ of an unknown density $f$.
#
# Pole-figure intensities are different input. They are sampled values of a projected pole
# density rather than random orientations. Recovering an ODF from them is an inverse problem,
# not the kernel estimate described here.
#
# The worked sequence starts with real numbers. It then applies the same construction to
# directions and orientations.

# +
import matplotlib.pyplot as plt
import numpy as np

from mtex import *

plottingConvention.default('y↑→x')
rng = np.random.default_rng(1)
# -

# ## Begin with a finite sample
#
# Use a mixture of two Gaussian densities as a known reference. The second argument of
# `Gaussian` is a width parameter $\delta$. The profile is $\exp(-(x-m)^2/\delta^2)$, so
# $\delta$ is the standard deviation times $\sqrt{2}$.

# +
f = lambda x: (Gaussian(0.2, 0.05, x) + Gaussian(0.5, 0.2, x)) / 2
x = np.linspace(0, 1, 1000)

# draw a sample from the density on [0,1]
N = 20
xN = discreteSample(f, N, range=[0, 1], rng=rng)

plt.figure()
plt.plot(x, f(x), linewidth=2)
plt.plot(xN, np.zeros_like(xN), 'o', markeredgewidth=2, markeredgecolor='r', markerfacecolor='none')
plt.xlabel('x')
plt.ylabel('density')
plt.legend(['true density', 'sample'])
# -

# The blue curve has a narrow mode near 0.2 and a broad mode near 0.5. More red samples occur
# where that curve is high. With only 20 samples, random gaps and clusters are still
# prominent.
#
# ## A histogram depends on its bins
#
# A histogram is the simplest density estimate. The `density=True` normalization puts the bar
# areas on the same scale as the reference density.

plt.figure()
plt.hist(xN, 10, density=True)
plt.plot(x, f(x), linewidth=2)
plt.plot(xN, np.zeros_like(xN), 'o', markeredgewidth=2, markeredgecolor='r', markerfacecolor='none')
plt.xlabel('x')
plt.ylabel('density')
plt.legend(['true density', 'sample', 'histogram'])

# The estimate is piecewise constant, and changing the bin edges changes its steps. The bars
# hint at two modes, but they do not follow either mode smoothly. Kernel density estimation
# removes the bin edges.
#
# ## Replace each sample by a kernel
#
# A *kernel* is a small density profile placed at one observation. Start with a Gaussian of
# mean 0 and width 0.05.

# +
psi = Gaussian(0, 0.05)

plt.figure()
plt.plot(xN, np.zeros_like(xN), 'o', markeredgewidth=2, markeredgecolor='r', markerfacecolor='none')
for n in range(N):
  plt.plot(x, psi(x - xN[n]), 'k')
plt.xlabel('x')
plt.ylabel('kernel contribution')
# -

# Every black curve has the same shape and total mass. Only its centre changes. Their mean is
# the kernel density estimate.

# +
fN = lambda x: np.mean(psi(x[None, :] - xN[:, None]), axis=0)

plt.figure()
plt.plot(x, f(x), linewidth=3, color=ind2color(0))
plt.plot(x, fN(x), linewidth=3, color=ind2color(1))
plt.xlabel('x')
plt.ylabel('density')
plt.legend(['true density', 'kernel estimate'])
# -

# The estimate is smooth, but it is not the true function. Its small peaks record the finite
# sample as well as the two features of the source.
#
# ## Kernel width decides which detail survives
#
# Repeat the estimate with widths 0.01, 0.05, and 0.25.

# +
delta = [0.01, 0.05, 0.25]

plt.figure()
plt.plot(x, f(x), linewidth=3)
for d in delta:
  psiTrial = Gaussian(0, d)
  plt.plot(x, np.mean(psiTrial(x[None, :] - xN[:, None]), axis=0), linewidth=2)
plt.xlabel('x')
plt.ylabel('density')
plt.legend(['$f$', '$f_{0.01}$', '$f_{0.05}$', '$f_{0.25}$'])
# -

# The 0.01 estimate retains a noisy peak for almost every observation. The 0.25 estimate
# merges the two source modes and lowers their peaks. The middle curve balances sample-scale
# variation against lost structure. This is the bias--variance trade-off behind kernel
# selection.
#
# For one-dimensional real data, [calcDensity](calcDensity.html) selects a smoothing width
# automatically. The estimate carries the width it chose as `bandwidth`.

# +
fAuto = calcDensity(xN, range=[0, 1])
autoBandwidth = fAuto.bandwidth
autoBandwidth
# -

# +
plt.figure()
plt.plot(x, f(x), linewidth=2)
plt.plot(x, fAuto(x), linewidth=2)
plt.xlabel('x')
plt.ylabel('density')
plt.legend(['true density', 'automatic estimate'])
# -

# The automatic estimate smooths the sample without reproducing every observation as a
# separate peak. In MATLAB's draw its left mode sits on the true one at $x = 0.2$, its right
# mode near $x = 0.67$, well to the right of the true mode at $x = 0.5$, and the taller of the
# two; the draw here is another one, so its modes sit elsewhere.
#
# The bandwidth is estimated from the same twenty samples, and twenty samples do not locate
# the broad component of the mixture. Read the estimate as one draw rather than as the density
# itself.
#
# For one-dimensional data, the `bandwidth` option fixes the smoothing parameter instead.

fFixed = calcDensity(xN, range=[0, 1], bandwidth=0.004)

# [Optimal Kernel Selection](OptimalKernel.html) explains why no single width is best for
# every sample.
#
# ## The construction also works in several dimensions
#
# Draw 100 independent pairs whose x and y coordinates follow the same mixture. A row of
# `np.column_stack([xN, yN])` is now one two-dimensional observation.

# +
N = 100
xN = discreteSample(f, N, range=[0, 1], rng=rng)
yN = discreteSample(f, N, range=[0, 1], rng=rng)

plt.figure()
plt.scatter(xN, yN, facecolors='none', edgecolors='r', linewidths=2)
plt.axis('scaled')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('x')
plt.ylabel('y')
# -

# The sample cloud is concentrated near combinations of the two one-dimensional modes. The
# empty space between points is not assigned zero density; the next step estimates it from
# neighbouring observations.
#
# For $d$-dimensional data, the range has one column per coordinate. The first row contains
# the minima, and the second contains the maxima.

# +
fN = calcDensity(np.column_stack([xN, yN]), range=[[0, 0], [1, 1]], rng=rng)

xGrid, yGrid = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100), indexing='ij')
z = fN(xGrid, yGrid)
plt.figure()
levels = np.arange(0, z.max() + 2, 2)
plt.contourf(xGrid, yGrid, z, levels=levels)
plt.contour(xGrid, yGrid, z, levels=levels, colors='k', linewidths=0.5)
mtexColorMap('LaboTeX')
plt.axis('scaled')
plt.scatter(xN, yN, s=4, c='r')
plt.xlabel('x')
plt.ylabel('y')
# -

# The estimate is positive everywhere, but most of the square carries very little of it. The
# `LaboTeX` colour map starts at white, and most of the grid falls in the lowest contour band,
# so the field reads as a few pink islands on a white ground.
#
# White here means density below the first contour, not zero. The islands sit at the four
# combinations of the two one-dimensional modes. The pair of narrow modes near $x = 0.2$,
# $y = 0.2$ is by far the densest, and the pair of broad modes near $x = 0.5$, $y = 0.5$ is the
# faintest.
#
# ## Directions need kernels on the sphere
#
# A directional kernel uses angular distance on the sphere rather than ordinary distance on a
# line. As a crystallographic example, estimate the distribution of misorientation axes along
# one phase boundary population. A misorientation axis is the direction about which one
# crystal must be rotated to match the other.

# +
ebsd = mtexdata('forsterite', verbose=False)
grains = calcGrains(ebsd)

# Select boundaries between forsterite and enstatite grains.
gB = grains.boundary['Forsterite', 'Enstatite']
# the axes are taken with respect to the forsterite frame, the first of the misorientation
misAxes = gB.misorientation.axis()

plot(misAxes, fundamentalRegion=True, markerFaceAlpha=0.1)
# -

# Symmetry maps equivalent axes into the same fundamental region. The points cluster instead
# of covering that region uniformly, so a density can summarize their preferred directions.

# +
# boundary segments of one grain pair repeat an axis, so the halfwidth is named
axisDensity = calcDensity(misAxes, halfwidth=10 * degree)

contourf(axisDensity)
mtexColorMap('LaboTeX')
mtexColorbar()
hold(True)
plot(misAxes, markerEdgeAlpha=0.25, markerFaceColor='none', markerEdgeColor='k')
hold(False)
# -

# The contours join nearby black observations into broad maxima. The result is an
# `S2FunHarmonic` with the symmetry of the frame and supports the operations in
# [Spherical Function Operations](S2FunOperations.html).
#
# Repeat the calculation with a 5 degree halfwidth.

# +
axisDensity5 = calcDensity(misAxes, halfwidth=5 * degree)

contourf(axisDensity5)
mtexColorMap('LaboTeX')
mtexColorbar()
hold(True)
plot(misAxes, markerEdgeAlpha=0.25, markerFaceColor='none', markerEdgeColor='k')
hold(False)
# -

# The smaller halfwidth keeps narrower, more fragmented maxima around the observations. That
# extra detail is not automatically extra information; it may be sampling variation.
#
# ## Orientations need kernels in orientation space
#
# Applying the construction to orientations produces an *orientation distribution function*
# (ODF). An ODF is a density over the possible crystal orientations of one phase in the
# specimen.

# +
forsteriteOri = ebsd['Forsterite'].orientations
odf = calcDensity(forsteriteOri, halfwidth=10 * degree)
odf
# -

plotSection(odf, contourf=True)
mtexColorMap('LaboTeX')
hold(True)
plot(forsteriteOri, markerEdgeAlpha=0.25, markerFaceColor='none', markerEdgeColor='k', markerSize=10)
hold(False)

# Each section cuts through the continuous ODF. The black measurements concentrate around the
# same section maxima, while the kernel fills the space between them.
# [ODF Estimation from EBSD Data](EBSD2ODF.html) treats phase selection, correlated EBSD
# pixels, and ODF interpretation in detail.
#
# ## Weights decide what population the density describes
#
# A weighted sample gives some observations more mass than others. Grain orientations make
# the physical choice clear. Equal weights answer "what fraction of grains has this
# orientation?" Pixel-count weights approximate "what fraction of the mapped area has this
# orientation?"

# +
ebsd = mtexdata('titanium', verbose=False)
grains = calcGrains(ebsd)
indexedGrains = grains['indexed']

# one halfwidth for both, MATLAB's default, so that only the weights differ
odfEqual = calcDensity(indexedGrains.meanOrientation, halfwidth=10 * degree)
odfByPixel = calcDensity(indexedGrains.meanOrientation, weights=indexedGrains.numPixel, halfwidth=10 * degree)

weighting = ['equal grains', 'grain pixel count']
peakMRD = [max(odfEqual)[0], max(odfByPixel)[0]]
for w, p in zip(weighting, peakMRD):
  print(f'{w:<20}{p:.4f}')
# -

# The unequal peaks, 7.6274 and 8.9863 mrd in MATLAB, show that weighting changes the
# estimated population, not just the plotting style. On a regular map, pixel count is
# proportional to mapped area. State the sampling unit before interpreting an ODF.
#
# ## Parametric estimation answers a different question
#
# Kernel estimation assumes smoothness but does not prescribe one global shape. *Parametric
# density estimation* instead assumes that the unknown density belongs to a chosen family and
# estimates that family's parameters.
#
# For a Gaussian, those parameters are the mean and standard deviation. The analogous models
# on spheres and in orientation space are Bingham distributions. See
# [Spherical Bingham Distribution](S2Bingham.html) and [Bingham ODFs](BinghamODFs.html) for
# fitting and checking those models.
#
# ## The maths behind kernel density estimation
#
# For equally weighted samples $x_1,\ldots,x_N$ and a normalized kernel $\psi$, the estimate
# is
#
# $$ f_N(x) = \frac{1}{N} \sum_{n=1}^N \psi(x-x_n). $$
#
# Each observation contributes the same total mass. The kernel decides how far that mass
# spreads. For non-negative weights $w_n$, MTEX replaces the equal average by a normalized
# weighted sum.
#
# On the rotation group, let $o_n$ be orientations and let $\psi$ be a radially symmetric
# kernel. The weighted ODF estimate is
#
# $$ f(o) = \frac{1}{\sum_{j=1}^{N} w_j} \sum_{n=1}^{N} w_n \psi(o o_n^{-1}). $$
#
# MTEX also accounts for the symmetry-equivalent representatives carried by each
# orientation. Different phases generally have different crystal symmetries, so estimate an
# EBSD-derived ODF from one selected phase at a time. [Kernels on SO(3)](SO3Kernels.html)
# compares the kernel families.
#
# ## References
#
# * Z. I. Botev, J. F. Grotowski, and D. P. Kroese,
#   [Kernel density estimation via diffusion](https://doi.org/10.1214/10-AOS799),
#   _The Annals of Statistics_ 38 (2010), 2916--2957, gives the automatic estimator used by
#   `calcDensity` for one-dimensional real data.
#
# * R. Hielscher, [Kernel density estimation on the rotation group and its application to
#   crystallographic texture analysis](https://doi.org/10.1016/j.jmva.2013.03.014),
#   _Journal of Multivariate Analysis_ 119 (2013), 119--143, derives the orientation-space
#   estimator and its fast algorithms.
#
# * C. Bingham, [An antipodally symmetric distribution on the
#   sphere](https://doi.org/10.1214/aos/1176342874), _The Annals of Statistics_ 2 (1974),
#   1201--1225, introduces the parametric directional model used here as the alternative to a
#   kernel estimate.
#
# ## Next
#
# [Optimal Kernel Selection](OptimalKernel.html) turns the visual bias--variance trade-off
# into practical choices for directional and orientation data. It also explains when
# automatic selection is reliable.
