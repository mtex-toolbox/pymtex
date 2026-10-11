# # Symmetry of Orientation-Dependent Functions
#
# A rotational function can repeat under symmetry operations on either side of its argument.
# Every `SO3Fun` therefore stores the frame `frameA` of its right symmetry and the frame `frameB`
# of its left symmetry, MATLAB's `SRight` and `SLeft`. For
# an orientation distribution function (ODF), the right side is the crystal symmetry and the
# left side is the specimen symmetry.
#
# Symmetry is the point group under which the data is invariant. It is not a reference
# frame. Each stored symmetry is attached to the reference frame of its side, so replacing a
# symmetry may also change which frame the function reports.

# +
import numpy as np

from mtex import *

plottingConvention.default('y↑→x')
SO3F = SO3Fun.dubna()
SO3F
# -

# ---

cs = SO3F.frameA
cs

# ---

ss = SO3F.frameB
ss

# `CS` and `SS` are convenient aliases for `frameA` and `frameB`. The Dubna ODF has quartz
# crystal symmetry `321` on the right and no specimen symmetry on the left.
#
# ## Left and right actions
#
# If $s_L$ belongs to `frameB` and $s_R$ belongs to `frameA`, a symmetric function satisfies
#
# $$ f(s_L R s_R)=f(R). $$
#
# Rotation composition is not commutative, so the two point groups cannot be exchanged.
# [symmetrise](orientation.symmetrise.html) constructs the complete orbit $s_L R s_R$ of an
# orientation.

# +
ori = orientation.rand(cs, ss, rng=np.random.default_rng(2))
equivalentOrientations = ori.symmetrise()
equivalentValues = SO3F.eval(equivalentOrientations)
maximumOrbitDifference = np.max(np.abs(equivalentValues - SO3F.eval(ori)))
maximumOrbitDifference
# -

# ---

manualOrbit = ss * ori * cs
manualOrbitValues = SO3F.eval(manualOrbit)
manualConstructionDifference = np.max(np.abs(np.sort(equivalentValues.ravel()) - np.sort(manualOrbitValues.ravel())))
manualConstructionDifference

# Both printed differences are at numerical precision. The first verifies that all
# symmetry-equivalent orientations have the same density. The second verifies that left
# multiplication by `ss` and right multiplication by `cs` construct the same orbit as
# `ori.symmetrise()`.
#
# ## Symmetry reduces the plot domain
#
# By default, MTEX plots only one fundamental region. A fundamental region contains one
# representative from every symmetry-equivalent orbit.

plot(SO3F, 'sigma')

# The displayed sigma sections stop at the boundary of the quartz fundamental region. This
# smaller domain avoids showing repeated copies; it does not discard part of the ODF. Every
# orientation left outside the picture has an equivalent inside it, carrying the same
# density.
#
# ## A symmetry label is not a projection
#
# In most `SO3Fun` representations, the symmetry properties are stored separately from
# coefficients, centres, or other model parameters. This makes reassignment easy, but it
# does not mean that arbitrary stored data automatically has the newly claimed invariance.
#
# The effect of reassignment depends on the representation. For an RBF model it changes
# which symmetry-related kernel copies contribute to an evaluation. For a harmonic model it
# leaves the Fourier coefficients untouched until the function is explicitly symmetrised.
#
# On an ODF, assign a `specimenFrame` to the left side. A general rotational function may
# instead describe a relation between two crystal sides, for which an assignment such as the
# following is meaningful:
#
# ```python
# SO3F.frameB = crystalFrame('432')
# ```
#
# Applying that line to the Dubna ODF would change its physical meaning. It would no longer
# describe quartz orientations relative to an unsymmetric specimen frame.
#
# ## Relabel harmonic coefficients
#
# Construct a reproducible real-valued harmonic function without a nontrivial point-group
# symmetry. Its random coefficients make violations of a proposed twofold symmetry easy to
# detect.

# +
rng = np.random.default_rng(1)
SO3F2 = SO3FunHarmonic(rng.standard_normal(1000))
SO3F2.isReal = True
coefficientsBefore = SO3F2.fhat

twoFold = crystalFrame('2')
SO3F2.frameA = twoFold

coefficientChangeAfterRelabelling = np.linalg.norm(SO3F2.fhat - coefficientsBefore)
coefficientChangeAfterRelabelling
# -

# The zero coefficient change confirms that assignment only relabelled the existing series.
# Test the claimed invariance at a random orientation and its twofold orbit.

# +
probe = orientation.rand(twoFold, SO3F2.frameB, rng=rng)
probeOrbit = probe.symmetrise()
valuesBeforeProjection = SO3F2.eval(probeOrbit)
orbitSpreadBeforeProjection = np.max(np.abs(valuesBeforeProjection - valuesBeforeProjection.flat[0]))
orbitSpreadBeforeProjection
# -

# ---

plot(SO3F2)

# The nonzero orbit spread shows that the relabelled coefficients do not yet define a
# twofold-symmetric function. The plot uses the newly labelled fundamental region, so a
# smaller plot domain is not evidence of actual invariance.
#
# ## Project onto symmetric functions
#
# [SO3FunHarmonic.symmetrise](SO3FunHarmonic.symmetrise.html) averages the function over
# its left and right point groups. In coefficient space this projects the Fourier
# coefficients onto the subspace with the requested invariance.

# +
SO3F2Sym = SO3F2.symmetrise()
relativeCoefficientChange = np.linalg.norm(SO3F2Sym.fhat - SO3F2.fhat) / np.linalg.norm(SO3F2.fhat)
relativeCoefficientChange
# -

# ---

valuesAfterProjection = SO3F2Sym.eval(probeOrbit)
orbitSpreadAfterProjection = np.max(np.abs(valuesAfterProjection - valuesAfterProjection.flat[0]))
orbitSpreadAfterProjection

# ---

plot(SO3F2Sym)

# The coefficients now change, while the orbit spread falls to numerical precision. Features
# that disagreed between symmetry-related orientations have been averaged. This projection
# loses their differences, so the original nonsymmetric function cannot be recovered from
# `SO3F2Sym`.

plot(SO3F2Sym, complete=True)

# In the complete plot, every twofold-related position now carries the same value. This
# visual repetition and the small printed orbit spread test the same property in
# complementary ways.
#
# Harmonic invariance is encoded directly in the Fourier coefficients. Changing only
# `frameA` or `frameB` does not encode it; applying `symmetrise` does.
#
# ## Convert another representation before projection
#
# Every `SO3Fun` can be expanded as an `SO3FunHarmonic`. The constructor uses the quadrature
# procedure from [Quadrature of Orientation-Dependent Functions](SO3FunQuadrature.html) when
# the representation offers nothing quicker. The Dubna ODF is an `SO3FunRBF` and takes its
# own route, from its centres and kernel. Either way the same explicit projection can then
# be applied.

SO3F3 = SO3FunHarmonic(SO3F, bandwidth=14)
SO3F3

# `SO3F3` inherits the left and right symmetries of the Dubna ODF. Because the source is
# already invariant, its computed coefficients are symmetrised during construction. Calling
# `SO3F3.symmetrise()` again would therefore leave it unchanged apart from numerical
# accuracy.
#
# ## The maths behind symmetrisation
#
# Let $G_L$ and $G_R$ be the proper rotations in the left and right point groups.
# Symmetrisation replaces a function $f$ by the group average
#
# $$ f_{\mathrm{sym}}(R)=\frac{1}{\lvert G_L\rvert\lvert G_R\rvert}
# \sum_{s_L\in G_L}\sum_{s_R\in G_R}f(s_L R s_R). $$
#
# Applying any member of either group merely permutes the terms in the sum. The averaged
# function therefore has the required left and right invariance. Components that are
# incompatible with the point groups cancel, which explains both the coefficient change and
# the loss of information.
#
# ## References
#
# * H.-J. Bunge, [Texture Analysis in Materials Science: Mathematical Methods](https://doi.org/10.1016/C2013-0-11769-2),
#   Butterworths (1982), develops crystal and specimen symmetry for ODFs and their
#   generalized harmonic coefficients.
#
# ## Next
#
# Continue with [Convolution](SO3FunConvolution.html) to combine rotational functions. The
# left and right sides introduced here determine whether two functions can be convolved and
# which symmetries the result inherits.
#
# ## Technical Details
#
# NumPy's generator cannot reproduce MATLAB's `rng(1)`, so the random coefficients, the
# orbit spread 9.84 for MATLAB's 4.71 and the coefficient change 0.70 for 0.74 differ from
# MATLAB's in the digits.
# `plot(f, 'complete')` draws the sections of the frames without their groups.
