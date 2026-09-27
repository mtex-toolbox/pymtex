# # ODF Modeling
#
# An orientation distribution function (ODF) does not have to come from a measurement. A
# *model ODF* is built from a few chosen ingredients: a preferred orientation and its
# spread, a fibre, or a mixture of these. Because its ingredients are known, a model ODF
# can serve as a reference for measured textures, as a starting point for
# texture-evolution simulations, or as test data with a known answer.
#
# This page assumes the normalization and multiples of a random distribution (mrd)
# introduced in [ODF Theory](ODFTheory.html). Every ODF in MTEX follows the
# [SO3Fun](SO3FunConcept.html) interface for functions on the rotation group $SO(3)$. The
# physical model and its numerical representation are related, but they are not the same
# choice:
#
# | construction | meaning |
# |---|---|
# | [uniform](RadialODFs.html) | constant, the untextured reference |
# | [unimodal](RadialODFs.html) | a radial peak about one orientation |
# | [multimodal](RadialODFs.html) | several radial peaks |
# | [fibre](FibreODFs.html) | a peak spread along a curve in orientation space |
# | [Bingham](BinghamODFs.html) | a parametric peak with three independent spreads |
# | [harmonic representation](SO3FunHarmonicRepresentation.html) | a series expansion, the classical form used for pole figure inversion |
#
# Harmonic names a representation, not another physical peak shape. The current
# [calcODF](PoleFigure.calcODF.html) normally returns a radial-basis ODF. It can then be
# converted to a harmonic series. All of these objects share one interface for
# evaluation, plotting, scaling, and addition. This is why components with different
# representations can be mixed in one model.

from mtex import *

plottingConvention.default('y↑→x')

# ## The Uniform ODF
#
# The simplest model is the constant function
#
# $$f(g) = 1,\quad g \in SO(3).$$
#
# It needs only the crystal and specimen symmetries. The returned summary is useful here:
# it records both symmetries and identifies the constant component.

# +
cs = crystalFrame('cubic')
ss = specimenFrame('orthorhombic')

odf = uniformODF(cs, ss)
odf
# -

# A value of 1 mrd everywhere is an untextured specimen. This uniform ODF is the reference
# against which every other mrd value is measured.

# ## A Single Component
#
# A unimodal ODF is a peak about one preferred orientation. A [kernel](SO3Kernels.html)
# sets the shape, and its halfwidth sets the angular distance at which the kernel falls
# to half its maximum. The halfwidth is a spread parameter, not a cutoff: the component
# continues beyond that angle.

# +
psi = SO3vonMisesFisherKernel(halfwidth=10 * degree)

mod1 = orientation.byMiller([1, 2, 2], [2, 2, 1], cs, ss)

odf1 = unimodalODF(mod1, psi)
odf1
# -

# The summary records the kernel, centre, and component weight. The maximum sits at the
# preferred orientation and its symmetry-equivalent copies. Its value measures
# concentration rather than volume fraction. A narrower normalized peak has a higher
# maximum because its mean must remain one.

odfMax = max(odf1)[0]
odfMax

# ---

plotPDF(odf1, cat(Miller(1, 0, 0, cs), Miller(1, 1, 0, cs)), antipodal=True, colorRange='equal')
mtexColorbar(title='mrd')

# The localized spots are projections of one preferred orientation and its
# symmetry-equivalent copies. They are not separate components.

# ## Mixtures
#
# ODFs are added and scaled like functions, so a textured component can sit on a uniform
# background. The classical Santa Fe standard is 27 percent of the component above and 73
# percent uniform background.

odf = 0.73 * uniformODF(cs, ss) + 0.27 * unimodalODF(mod1, psi)
odf

# The printed summary separates the uniform and unimodal terms. Both are individually
# normalized, so their coefficients act as mixture volume fractions. They must add up to
# one if the mixture is to remain normalized.

mean(odf)

# The mean is 1. The component peaks may overlap in orientation space, but the
# coefficients still describe the fractions assigned to the two terms. They are not
# volumes of disjoint regions drawn around the maxima.

plotPDF(odf, cat(Miller(1, 0, 0, cs), Miller(1, 1, 0, cs)), antipodal=True, colorRange='equal')
mtexColorbar(title='mrd')

# The uniform term contributes a 0.73 mrd background, while the unimodal term produces the
# spots. This known model is commonly used to test pole figure inversion;
# [The Santa Fe Example](PoleFigureSantaFe.html) simulates pole figures from it and scores
# the reconstruction against the answer.

# ## Rotating a Model
#
# [rotate](SO3Fun.rotate.html) actively moves a model relative to the specimen axes. By
# default, the rotation acts on the specimen side of every component orientation.

# +
odfRot = rotate(odf, rotation.byAxisAngle(vector3d.Z, 30 * degree))

plotPDF(odfRot, cat(Miller(1, 0, 0, cs), Miller(1, 1, 0, cs)), antipodal=True, colorRange='equal')
mtexColorbar(title='mrd')
# -

# Compared with the preceding pole figures, every feature turns by $30^\circ$ about the
# centre. The original `mmm` specimen symmetry was tied to x, y, and z. It is no longer
# coordinate-aligned after this rotation, so MTEX drops the specimen-symmetry label and
# issues a warning. The physical twofold axes have rotated with the texture.
#
# A frame change is different: it re-expresses the same physical texture in another
# reference frame and leaves the texture itself untouched. Use
# [transformReferenceFrame](SO3Fun.transformReferenceFrame.html) when the crystal frame
# changes. The corresponding coordinate transformation is inverse to an active rotation.

# ## Further Reading
#
# * [Bunge, Texture Analysis in Materials Science](https://doi.org/10.1016/C2013-0-11769-2) develops the mathematical foundations of ODFs and their representations.
# * [Matthies, Vinel, and Helming, Standard Distributions in Texture Analysis](https://doi.org/10.1515/9783112736173) is an atlas of cubic-orthorhombic model textures.
# * [Roe (1965)](https://doi.org/10.1063/1.1714396) gives the classical harmonic solution of the pole figure inversion problem.
# * [Kunze and Schaeben (2004)](https://doi.org/10.1023/B:MATG.0000048799.56445.59) develop quaternion Bingham distributions for texture analysis.

# ## Next
#
# [Plotting an ODF](ODFPlot.html) compares the views used to inspect these models. The
# model-family pages begin with [Radial ODFs](RadialODFs.html).
# [Fibre ODFs](FibreODFs.html) and [Bingham ODFs](BinghamODFs.html) cover the other shapes
# listed above. [Random Sampling](RandomSampling.html) turns a model back into discrete
# orientations, while [Properties](ODFCharacteristics.html) extracts the numbers that
# describe any ODF.

# ## Technical Details
#
# The sum of a uniform and a unimodal radial function stays one `SO3FunRBF`, as in MATLAB;
# the port lists the uniform part and the kernel centres in its display without MATLAB's
# table of Euler angles. `max(odf)` returns the pair of the value and the orientation.
