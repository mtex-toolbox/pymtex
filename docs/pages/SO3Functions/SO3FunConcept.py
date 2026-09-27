# # Orientation-Dependent Functions
#
# An orientation-dependent function assigns a numerical value to every rotation or crystal
# orientation. The set of rotations is the rotation group $SO(3)$, which gives the MTEX class
# `SO3Fun` its name.
#
# An important example is the [orientation density function (ODF)](ODFTheory.html). It
# assigns a density to each crystal orientation in a specimen. Other examples include the
# Schmid factor and the Taylor factor as functions of crystal orientation.
#
# ## A First Orientation-Dependent Function
#
# MTEX represents a scalar function on $SO(3)$ by an object of type [SO3Fun](SO3Fun.SO3Fun.html).
# Its symmetries determine which rotations represent the same argument.
#
# Consider the smallest rotational angle between an orientation and the identity, including
# its cubic symmetry equivalents. The [angle](orientation.angle.html) command returns this
# angle in radians. Dividing by `degree` makes the function value a number in degrees.

# +
import numpy as np

from mtex import *

# define cubic crystal symmetry
cs = crystalFrame('432')

# wrap the angle formula in an SO3Fun
SO3F = SO3FunHandle(lambda ori: angle(ori) / degree, cs)
SO3F
# -

# ## Evaluate the Function
#
# `SO3FunHandle` turns a Python function into an `SO3Fun`. The variable `SO3F` now stores
# the formula together with its symmetry.
#
# Use [eval](SO3FunHandle.eval.html) to evaluate it at one or many orientations. Here the
# input is fixed so that the result is reproducible.

ori = orientation.byEuler(20 * degree, 30 * degree, 10 * degree, cs)
ori

# ---

angleInDegrees = SO3F.eval(ori)
angleInDegrees

# The returned scalar is the smallest angle, in degrees, among the symmetrically equivalent
# representatives of `ori`. The next page develops this construction and the other ways to
# define an `SO3Fun`.
#
# ## Plot Euler-Angle Sections
#
# A scalar function on $SO(3)$ depends on three coordinates. MTEX can show it as a stack of
# sections at fixed third Euler angle $\varphi_2$.

plotSection(SO3F, sections=4)
mtexColorbar()

# Each panel covers the first two Euler angles at one value of $\varphi_2$. The colour
# changes within and between panels because the smallest symmetry-reduced angle depends on
# all three Euler angles.
#
# ## Plot Axis--Angle Sections
#
# The same function is especially simple in axis--angle coordinates. Each section fixes the
# rotational angle and varies the rotational axis.

plotSection(SO3F, 'axisAngle', np.arange(15, 61, 15) * degree, 'pcolor')
mtexColorbar()
mtexColorMap('parula')

# Every panel has one colour because `SO3F` returns the angle that labels that panel. A
# rotation by one of these angles may have any axis, so each angle fills the whole sphere
# and needs an upper and a lower panel. The two section plots show the same function in
# different coordinates; neither changes the underlying data.
#
# ## Analyse an Orientation-Dependent Function
#
# The common `SO3Fun` interface provides arithmetic, integration, differentiation and
# searches for extrema. For this angle function, a local maximum is an orientation farthest
# from a cubic symmetry equivalent of the identity.
#
# The [max](SO3Fun.max.html) command below requests up to ten distinct local maxima. The
# `accuracy` option controls the final angular search tolerance.

value, oriMax = max(SO3F, numLocal=10, accuracy=0.001 * degree)
value

# ---

oriMax

# The calculation finds exactly six symmetrically inequivalent maxima. Their function
# values are about 62.799 degrees, and their positions are the vertices of the fundamental
# region in orientation space.

color = ind2color(np.tile(np.arange(1, oriMax.size + 1), (numSym(cs), 1)))
plot(oriMax.symmetrise(), color, 'axisAngle', filled=True, markerSize=20, restrict2FundamentalRegion=True)

# The six colours distinguish the six maxima. The markers lie on the outer vertices because
# those points have the greatest possible distance from the identity after cubic symmetry
# has been taken into account.
#
# ## Representations of Orientation-Dependent Functions
#
# MTEX can store a function in several ways. The representation controls how the function
# is constructed and how expensive an operation is, but all representations share the
# `SO3Fun` interface.
#
# | representation | MTEX class or documentation |
# |---|---|
# | harmonic series expansion | [SO3FunHarmonic](SO3FunHarmonicRepresentation.html) |
# | superposition of radial functions | [SO3FunRBF](RadialODFs.html) |
# | superposition of fibre elements | [SO3FunCBF](FibreODFs.html) |
# | Bingham distribution | [SO3FunBingham](BinghamODFs.html) |
# | sum of different components | `SO3FunComposition` |
# | formula evaluated on demand | `SO3FunHandle` |
#
# Thus functions with different internal representations can be added, multiplied,
# averaged, integrated or differentiated through the same API.
#
# ## Related Function Types
#
# [SO3VectorField](SO3FunVectorField.html) assigns a vector instead of a scalar to each
# rotation. [SO3Kernel](SO3Kernels.html) represents a radial function whose value depends
# only on rotational angle.
#
# ## References
#
# * H.-J. Bunge, [Texture Analysis in Materials Science: Mathematical Methods](https://doi.org/10.1016/C2013-0-11769-2),
#   Butterworths, 1982, develops the orientation-space and ODF framework used here.
#
# ## Next
#
# Continue with [Defining Orientation-Dependent Functions](SO3FunDefinition.html) to
# construct functions from formulas, harmonic coefficients and sampled values.
#
# ## Technical Details
#
# The maxima lie within $10^{-4}$ degree of the vertices of the fundamental region, where the
# region's faces meet; the symmetric copies of one maximum are therefore folded onto
# different equivalent vertices of the region's surface, which the plot shows in the colour
# of that maximum, where MATLAB's run shows one marker per maximum. MATLAB lists the maxima
# in other symmetric equivalents.
