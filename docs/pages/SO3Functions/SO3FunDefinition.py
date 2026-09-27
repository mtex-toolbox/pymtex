# # Defining Orientation-Dependent Functions
#
# [Orientation-Dependent Functions](SO3FunConcept.html) introduced the `SO3Fun` interface
# and the representations that implement it. This page shows how to construct the
# representation that matches the information you already have.
#
# ## Choose a Representation
#
# All `SO3Fun` representations support nearly the same operations. Different
# representations can also be combined in one expression. Choose one by the form of the
# available data rather than by the operation you plan next.
#
# | starting point | representation | constructor or guide |
# |---|---|---|
# | an explicit formula or algorithm | formula evaluated on demand | [SO3FunHandle](SO3FunHandle.SO3FunHandle.html) |
# | a general function or harmonic coefficients | harmonic series | [SO3FunHarmonic](SO3FunHarmonicRepresentation.html) |
# | centres with radial peaks | radial basis functions | [SO3FunRBF](RadialODFs.html) |
# | preferred orientation fibres | fibre components | [SO3FunCBF](FibreODFs.html) |
# | an elliptic distribution on the quaternion sphere | Bingham distribution | [SO3FunBingham](BinghamODFs.html) |
# | existing functions that should remain separate | arbitrary sum | [SO3FunComposition](SO3FunComposition.SO3FunComposition.html) |
#
# The harmonic representation is the most general numerical representation. Any `SO3Fun`
# can be converted with `SO3FunHarmonic(SO3F)`. This conversion is the
# [quadrature](SO3FunQuadrature.html) problem.
#
# Some operations require harmonic coefficients, and many others are much faster with
# them. The conversion is an approximation whenever only a finite harmonic bandwidth is
# retained.
#
# ## From an Explicit Formula
#
# Use `SO3FunHandle` when a formula or algorithm already returns one value for each input
# orientation. The formula could compute a Taylor factor or another physical property.
#
# The concept page used the rotational angle as its first example. Here the same example
# makes the two construction steps explicit. First assign the formula to a Python function.

# +
import numpy as np

from mtex import *

angleFormula = lambda ori: angle(ori) / degree
# -

# Second, attach the cubic crystal symmetry and wrap the formula in an `SO3FunHandle`.
# Dividing by `degree` makes the returned values numerical angles in degrees.

cs = crystalFrame('cubic')
SO3FHandle = SO3FunHandle(angleFormula, cs)
SO3FHandle

# ---

plot(SO3FHandle, sections=4)
mtexColorbar()

# Each panel fixes the third Euler angle. The colour varies because the smallest angle to a
# cubic symmetry equivalent depends on all three Euler angles.
#
# ## From a Harmonic Expansion
#
# `SO3FunHarmonic` stores a finite harmonic series on $SO(3)$. Converting the handle above at
# bandwidth 16 retains harmonic degrees from 0 through 16.

SO3FHarmonic = SO3FunHarmonic(SO3FHandle, bandwidth=16)
SO3FHarmonic

# ---

plot(SO3FHarmonic, sections=4)
mtexColorbar()

# The broad pattern matches the handle plot. Sharp changes are rounded and may show
# oscillations because the series has been cut off at degree 16. Raising the bandwidth
# reduces this cut-off error at additional cost.
#
# Currently, a bandwidth of up to 128 works reasonably fast in MTEX.
#
# The power spectrum shows how much squared coefficient magnitude belongs to each harmonic
# degree.

plotSpektra(SO3FHarmonic, linewidth=2, figSize='small')

# The plotted spectrum stops at degree 16 because higher coefficients were not retained. Its
# decay indicates how strongly the finer angular scales contribute to this approximation. See
# [Harmonic Representation](SO3FunHarmonicRepresentation.html) for coefficients and bandwidth
# in detail.
#
# ## From Radial Functions
#
# A radial function depends only on angular distance from a centre orientation. Examples
# include the de la Vallee Poussin, Abel--Poisson, Gauss--Weierstrass and von Mises--Fisher
# kernels.
#
# Their common size parameter is the halfwidth. It is the angular distance at which the
# kernel value is half its value at the centre.

# +
# define a de la Vallee Poussin kernel with 15 degree halfwidth
psi = SO3DeLaValleePoussinKernel(halfwidth=15 * degree)
psi
# -

# ---

plot(psi)

# The curve is highest at zero angular distance and falls to half that height at 15
# degrees. A smaller halfwidth therefore creates a narrower peak around every centre
# orientation.
#
# A superposition of radial kernels can approximate a general function. Such `SO3FunRBF`
# objects arise naturally in ODF reconstruction from pole figures and in kernel density
# estimation from discrete orientations.

# +
ori = orientation.rand(200, cs, rng=np.random.default_rng(1))

SO3FRBF = calcDensity(ori, kernel=psi)
SO3FRBF
# -

# ---

plot(SO3FRBF, sections=4)
mtexColorbar()

# The density contains overlapping peaks centred at the 200 sampled orientations. Their 15
# degree halfwidth smooths the individual samples into one continuous function. See
# [Radial ODFs](RadialODFs.html) for the weights, centres and kernels stored by `SO3FunRBF`.
#
# ## From Fibre Components
#
# A fibre is a one-dimensional family of orientations. `SO3FunCBF` represents a function as
# a superposition of components distributed along such fibres, which is useful for modelling
# a fibre ODF.

betaFibre = fibre.beta(cs)
betaFibre

# ---

SO3FCBF = SO3FunCBF(betaFibre, halfwidth=10 * degree)
SO3FCBF

# ---

plot(SO3FCBF, sections=4)
mtexColorbar()

# The high values follow the beta fibre through successive sections rather than forming an
# isolated orientation peak. The 10 degree halfwidth sets the spread transverse to that
# fibre. See [Fibre ODFs](FibreODFs.html) for further constructions.
#
# ## From a Bingham Distribution
#
# A [Bingham distribution](SO3FunBingham.SO3FunBingham.html) is described by four
# orientation axes `U` and four concentration values `kappa`. Together they specify the
# directions and relative lengths of the half-axes of a four-dimensional ellipsoid.

kappa = [100, 90, 80, 0]
U = orientation.cat(orientation.byAxisAngle(xvector, np.array([0, 180]) * degree, cs),
                    orientation.byAxisAngle(vector3d.cat(yvector, zvector), 180 * degree, cs))
U

# ---

SO3FBingham = BinghamODF(kappa, U)
SO3FBingham

# ---

plot(SO3FBingham, sections=4)
mtexColorbar()

# The unequal concentration values produce an anisotropic peak. Its shape differs along the
# four axes instead of depending only on distance from one centre. See
# [Bingham ODFs](BinghamODFs.html) for fitting and evaluation.
#
# ## Vector-Valued Functions
#
# The constructors above return scalar functions. Use [SO3VectorField](SO3FunVectorField.html)
# when every orientation should map to a vector instead of one number.
#
# ## References
#
# * H.-J. Bunge, [Texture Analysis in Materials Science: Mathematical Methods](https://doi.org/10.1016/C2013-0-11769-2),
#   Butterworths, 1982, develops harmonic representations of orientation density functions.
# * C. Bingham, [An Antipodally Symmetric Distribution on the Sphere](https://doi.org/10.1214/aos/1176342874),
#   _The Annals of Statistics_ 2 (1974), 1201-1225, introduces the distribution used by
#   `SO3FunBingham`.
#
# ## Next
#
# Continue with [Operations on Orientation-Dependent Functions](SO3FunOperations.html) to
# evaluate, combine, differentiate and integrate the objects constructed here.
#
# ## Technical Details
#
# NumPy's generator cannot reproduce MATLAB's `rng(1)`, so the 200 random orientations and
# the density built from them differ from MATLAB's.
