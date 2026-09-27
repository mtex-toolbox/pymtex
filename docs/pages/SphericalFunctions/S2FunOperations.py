# # Operations on Spherical Functions
#
# A spherical function can be evaluated, combined, searched, integrated, differentiated and
# rotated. The [Concept](S2FunConcept.html) page introduces their common `S2Fun` interface
# and evaluates one function at a chosen direction. This page builds on that step with two
# functions.
#
# ## Two functions to compare
#
# The examples use one patterned function and one narrow peak. Both are harmonic spherical
# functions and therefore support the same operations.

# +
import numpy as np

from mtex import *

plottingConvention.default('y↑→x')

# a patterned function and one narrow peak
sF1 = S2Fun.smiley()
sF2 = S2FunHarmonic.unimodal(halfwidth=10 * degree)

newMtexFigure(layout=[1, 2])
plot(sF1, upper=True)
mtexTitle('Patterned function')
nextAxis()
plot(sF2, upper=True)
mtexTitle('Narrow peak')
# -

# The first function has several broad features. The second concentrates its values around
# one direction. This contrast makes the effect of each operation visible.
#
# ## Basic arithmetic
#
# Adding functions or multiplying them by scalars produces another spherical function.
# Adding the constant one shifts every value equally.

# +
combined = 15 * sF1 + sF2
shifted = 1 + combined

# check the shift at one direction
shiftAtX = shifted.eval(xvector) - combined.eval(xvector)
print(f'Shift at specimen X: {shiftAtX:.1f}')

plot(combined, upper=True)
# -

# The combined plot retains the pattern of the first function and adds the sharp peak of the
# second. The scalar factor lifts the broad pattern towards the scale of that peak; the peak,
# a density of mean one concentrated within 10 degrees, still reaches about 98 and dominates
# the colour scale. The printed difference of 1.0 confirms that adding one shifts the value
# without changing the pattern.
#
# ## Pointwise operations
#
# The basic operations `-`, `*`, `**` and `/` also work with `S2Fun` objects. Functions such
# as [min](S2Fun.min.html), [max](S2Fun.max.html), [abs](S2Fun.abs.html) and
# [sqrt](S2Fun.sqrt.html) likewise return spherical functions when used pointwise.
#
# For two functions, `*` and `/` act pointwise; with a number, `*` scales a function and
# `sF / a` divides it by a scalar.

# +
newMtexFigure(layout=[1, 2])

# the maximum between two functions
plot(max(15 * sF1, sF2), upper=True)
mtexTitle('Pointwise maximum')

nextAxis()

# the minimum between two functions
plot(min(15 * sF1, sF2), upper=True)
mtexTitle('Pointwise minimum')
# -

# The pointwise maximum keeps whichever surface is higher at each direction. The pointwise
# minimum keeps the lower surface. The two plots therefore partition the same pair of
# functions in complementary ways.
#
# ## Global and local extrema
#
# [min](S2Fun.min.html) and [max](S2Fun.max.html) select their behaviour from their inputs:
#
# * Two spherical functions produce their pointwise minimum or maximum.
# * A spherical function and one number produce its pointwise minimum or maximum with that
#   value.
# * One spherical function returns its global minimum or maximum.
# * The additional `numLocal` option requests several local extrema.
#
# As the [Concept example](S2FunConcept.html) shows, `numLocal` is an upper limit rather
# than a promise that the requested number exists.

# +
newMtexFigure()
plot(combined, upper=True)

# compute and mark the global maximum
maxValue, maxNodes = max(combined)
annotate(maxNodes)

# compute and mark up to two local minima
minValue, minNodes = min(combined, numLocal=2)
annotate(minNodes)

print(f'Global maximum: {maxValue:.3f}\nlocal minima  : {minValue[0]:.3f} and {minValue[1]:.3f}')
# -

# The maximum marker sits on the top of the peak at the north pole. The minimum markers
# identify the two eyes, separate basins instead of merely the lowest sampled pixels in the
# plot. The maximum is 97.852, while the two basins have equal minima of -7.470, fifteen
# times the eyes' depth of -0.5.
#
# ## Integration and norms
#
# [sum](S2Fun.sum.html) returns the surface integral over the sphere. Thus the constant
# function one integrates to $4\pi$. [mean](S2Fun.mean.html) divides that integral by
# $4\pi$, so the constant function one has mean value one. These two normalisations
# therefore give the same result for any function.

# +
meanValue = mean(sF1)
normalisedIntegral = sum(sF1) / (4 * np.pi)

print(f'Mean: {meanValue:.4f}\nintegral/(4*pi): {normalisedIntegral:.4f}')
# -

# Both values are 0.0064. This agreement checks the $4\pi$ normalisation rather than
# asserting that `sF1` itself is normalized to mean one.
#
# Integration also defines the $L^2$ norm of a spherical function $f$:
#
# $$\Vert f\Vert_2 = \left(\int_{\mathrm{sphere}} \vert f(\xi)\vert^2\,\mathrm d\xi\right)^{1/2}.$$
#
# It can be assembled directly from the surface integral.

normFromIntegral = sqrt(sum(sF1 ** 2))

# [norm](S2Fun.norm.html) computes the norm from the coefficients, for the normalised
# surface measure $\mathrm d\xi/4\pi$, that is from the mean of $|f|^2$ rather than its
# integral.

directNorm = norm(sF1)
print(f'From integral: {normFromIntegral:.4f}; norm: {directNorm:.4f}; '
      f'times sqrt(4*pi): {directNorm * np.sqrt(4 * np.pi):.4f}')

# The integral gives 0.4229, the norm 0.1193, and the two agree after the factor
# $\sqrt{4\pi}$. The result is a single measure of the function's overall magnitude, not its
# maximum value.
#
# ## Differentiation
#
# The differential at one direction is the tangential gradient. MTEX returns it as a
# [three-dimensional vector](vector3d.vector3d.html) with [grad](S2Fun.grad.html).

gradientAtX = grad(sF1, xvector)
gradientAtX

# ---

gradientMagnitude = norm(gradientAtX)
gradientMagnitude

# The vector lies in the plane tangent to the sphere at X, so its X component is zero. Its
# magnitude of about $10^{-4}$ is no feature of the smiley, which is flat around X, but the
# residue of expanding a function with sharp cap edges into harmonics up to degree 250.
#
# Gradients at all directions form a spherical vector field. Calling
# [grad](S2Fun.grad.html) without an evaluation direction returns an
# `S2VectorFieldHarmonic`.

# +
# compute the gradient as a vector field
G = grad(sF1)

# plot the gradient on top of the function
newMtexFigure()
plot(sF1, upper=True)
hold(True)
plot(G)
hold(False)
# -

# Long arrows mark large changes in intensity. Arrows become almost invisible where the
# function is nearly constant.
#
# ## Rotate a function
#
# [rotate](S2Fun.rotate.html) moves a spherical function by a specified rotation.

# +
# define a rotation
rot = rotation.byAxisAngle(yvector, -30 * degree)

# plot the rotated spherical function
newMtexFigure()
plot(rotate(combined, rot), upper=True)
# -

# The entire pattern, including its maxima and minima, moves together by $-30$ degrees about
# the $y$ axis.
#
# ## Symmetrise a function
#
# A special case of rotation is symmetrising it with respect to some symmetry. The next
# example symmetrises the smiley with respect to a twofold axis in the $z$ direction.

# +
# define the symmetry
sym = specimenFrame('112')

# compute the symmetrised function
sFs = symmetrise(sF1, sym)
sFs
# -

# ---

# plot it
newMtexFigure()
plot(sFs, upper=True, complete=True)

# The result carries the frame of its symmetry. The complete plot shows that the original
# pattern has been repeated by the twofold symmetry operation.
#
# ## References
#
# * F. Bachmann, R. Hielscher and H. Schaeben,
#   [Texture Analysis with MTEX - Free and Open Source Software Toolbox](https://doi.org/10.4028/www.scientific.net/SSP.160.63),
#   _Solid State Phenomena_ 160, 63--68, 2010. This article connects MTEX spherical
#   calculations to pole figures and orientation distributions.
#
# ## Next
#
# Continue with [Plotting](S2FunPlotting.html) to choose a spherical projection, plot style,
# hemisphere and colour scale for these functions.
#
# ## Technical Details
#
# `S2FunHarmonic.unimodal` is a density, mean one, as MTEX documents it; MATLAB's reuses
# the degree variable in its normalisation, and its peak has mean 0.1629, 9.72 at its centre
# and still 0.30 at the eyes, 34 degrees away. MATLAB therefore reports the maximum 15.717
# and the minima -7.203, where the port has 97.852 and -7.470; the smiley itself agrees
# with MATLAB's coefficients to $2\cdot10^{-7}$, the aliasing of two quadratures. That
# difference is what moves the gradient at X, MATLAB's $(0, 1.26, 1.26)\cdot10^{-4}$, which
# the port reproduces from MATLAB's coefficients; MATLAB's text says 0.0012. `norm` is that of the normalised measure, $1/\sqrt{4\pi}$ times MATLAB's.
# MATLAB's `.*`, `./` and `.^` are `*`, `/` and `**`; `sum` is the surface integral as a
# function, `sF.sum()` as a method.
