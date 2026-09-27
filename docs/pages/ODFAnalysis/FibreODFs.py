# # Fibre ODFs
#
# A fibre ODF is a density concentrated around a one-dimensional
# [orientation fibre](OrientationFibre.html). Every orientation on it maps one fixed
# crystal direction onto one fixed specimen direction. Rotation about that aligned
# direction remains free, so the ODF is constant along the fibre and decays across it.
#
# Wire drawing gives the classical example. A crystal direction from the
# $\langle111\rangle$ family aligns with the drawing axis. Rotation about that axis
# remains unconstrained. The ideal fibre has zero width. A fibre ODF replaces it by a
# normalized tube of finite angular width.
#
# This page assumes the ODF normalization introduced in [ODF Theory](ODFTheory.html).
# [ODF Modeling](ODFModeling.html) introduces the model families. The geometry and named
# rolling-texture fibres are developed in [Fibres of Orientations](OrientationFibre.html).
#
# The plotting convention below draws specimen Y upward and specimen X to the right. It
# changes only the screen layout, not the ODF or its reference frames.

import numpy as np

from mtex import *

plottingConvention.default('y↑→x')

# ## Defining a Fibre ODF
#
# A fibre is represented by an object of type `fibre`. Many of the named fibres of rolling
# textures are built in. This example uses the built-in beta fibre for a cubic rolling
# texture. Its printed summary gives the endpoint orientations. It also gives the crystal
# and specimen directions that remain parallel.

cs = crystalFrame('432')
f = fibre.beta(cs)
f

# The displayed h parallel r row puts the two directions in different reference frames:
# `f.h` belongs to the crystal frame and `f.r` to the specimen frame. The beta object also
# has two endpoints. [fibreODF](fibreODF.html) uses only the direction pair, so the
# density continues around the corresponding full fibre.
#
# The `halfwidth` option controls how quickly the density decays away from that fibre.
# The returned summary identifies the `SO3FunCBF` representation, its kernel, halfwidth,
# defining directions, and weight.

odf = fibreODF(f, halfwidth=10 * degree)
odf

# ## The Tube in Orientation Space
#
# A three-dimensional contour plot shows surfaces of equal density around the fibre.

plot3d(odf)

# The translucent branches are sections of one tubular ridge folded by cubic symmetry.
# They are symmetry-equivalent descriptions of the same fibre component, not additional
# physical components.

# ## Sections Through the Tube
#
# A [sigma-section](SigmaSections.html) plot cuts through the same tube. The ridge passes
# from panel to panel because a section shows only a slice of the three-dimensional
# orientation space.

plotSection(odf, 'sigma')
mtexColorbar(title='mrd')

# The brightest curves trace the fibre through the section panels. Their intensity stays
# constant along the centreline. The surrounding colour bands show the decay across it.
# Evaluate 25 orientations on the original beta segment to check that constancy directly.

oriOnFibre = orientation(f, points=25)
ridgeValueRange = np.array([np.min(odf.eval(oriOnFibre)), np.max(odf.eval(oriOnFibre))])
ridgeValueRange

# ## Pole Figures of a Fibre ODF
#
# A pole figure projects an ODF by integrating along orientation fibres. Plot the
# defining crystal direction first, followed by the standard $(100)$ and $(111)$
# directions. [Pole Figures of an ODF](ODFPoleFigure.html) develops this projection as
# the crystallographic Radon transform.

h = cat(f.h, Miller(1, 0, 0, cs), Miller(1, 1, 1, cs))
plotPDF(odf, h, contourf=True)
mtexColorbar(title='mrd')

# For the ideal fibre, its defining direction collapses to the specimen direction `f.r`
# and its symmetry-equivalent copies. The finite halfwidth broadens each point into a
# spot in the first panel. The other directions sweep rings or ring segments about `f.r`.
# A ring is therefore a projection of the fibre, not a second component.

# ## The Effect of the Halfwidth
#
# This example uses the default de la Vallee Poussin kernel. Its halfwidth is the angular
# distance at which density falls to half its ridge value. It is a spread parameter, not
# a cutoff: the density continues beyond that angle. [fibreODF](fibreODF.html) also
# accepts a custom [S2Kernel](S2Kernel.S2Kernel.html) when halfwidth alone does not
# describe the required shape. [ODF Shapes](ODFShapes.html) compares the kernels.
#
# Each component below remains normalized with mean 1 mrd. Narrower fibres concentrate
# that same total volume into a smaller tube, so their maximum and texture index are
# larger. Neither number is a volume fraction.

# +
halfwidths = np.array([5, 10, 20]) * degree
odfByHalfwidth = []
for hw in halfwidths:
  odfByHalfwidth.append(fibreODF(f, halfwidth=hw))
  print(f'halfwidth {hw / degree:4.1f} degree : mean {mean(odfByHalfwidth[-1]):4.2f}, texture index {norm(odfByHalfwidth[-1]) ** 2:6.2f}, '
        f'maximum {max(odfByHalfwidth[-1])[0]:6.2f}')
# -

# The pole density around the defining specimen direction broadens as the halfwidth
# increases. A shared colour range also makes the fall in peak density visible from left
# to right.

# +
mtexFig = newMtexFigure(layout=[1, 3], figSize='large')
for i, hw in enumerate(halfwidths):
  plotPDF(odfByHalfwidth[i], f.h, contourf=True, noTitle=True)
  mtexTitle(f'${hw / degree:g}^{{\\circ}}$')
  if i < len(halfwidths) - 1:
    nextAxis()
setColorRange('equal')
mtexColorbar(title='mrd')
# -

# ## Fitting a Fibre to Data
#
# The inverse problem asks which fibre best describes an ODF or a set of orientations.
# [fibre.fit](fibre.fit.html) returns a candidate centreline. Start with a low-symmetry
# fibre so that every sampled orientation has one consistent representative. The
# `'local'` branch then reads the fibre from the orientation tensor without a global grid
# search. The sampling step is developed separately in
# [Random Sampling](RandomSampling.html).

# +
csFit = crystalFrame('1')
fTrue = fibre(Miller(1, 1, 1, csFit), vector3d.Z)
odfFit = fibreODF(fTrue, halfwidth=10 * degree)
ori = discreteSample(odfFit, 1000)
fFit, lam, fitDistance = fibre.fit(ori, 'local', diagnostics=True)
fFit
# -

# The two printed distances are the sample's mean angular distance from the true fibre
# and from the fitted fibre. They should be similar for this synthetic low-symmetry
# example.

meanDistanceDegrees = np.array([mean(angle(ori, fTrue)), fitDistance]) / degree
meanDistanceDegrees

# The eigenvalues returned by the local branch test whether a fibre is a sensible model.
# The ratio below compares spread along the fibre with scatter away from it. A ratio
# near 1 describes a blob rather than a line; a larger ratio supports a fibre
# interpretation.

linearityRatio = lam[2] / lam[1]
linearityRatio

# Two warnings are essential. First, the fit always returns a fibre even when the data do
# not follow one. Applied to a unimodal ODF, it can simply return a fibre through the
# mode. [Dispersion Axes](Grain_dispersion_axes.html) develops the eigenvalue diagnostic
# with measured grain orientations.
#
# Second, the global search is not reliable for highly symmetric groups. On cubic data,
# compare the mean distance to the fitted fibre with the distance to any physically
# expected fibre. A larger fitted distance is the clear failure symptom seen in the
# original cubic example. Treat a cubic global fit as a starting point for manual
# inspection.
#
# The local branch is much faster, but it requires low symmetry. It can also use a good
# starting guess that has already put symmetry-equivalent observations into one locally
# consistent set. The API does not accept the starting guess itself.

# ## The Maths Behind a Fibre ODF
#
# For a crystal direction $h$ and a specimen direction $r$, the full fibre is
#
# $$F_{h,r}=\{g \in SO(3):g h=r\}.$$
#
# Away from symmetry equivalents, a single fibre component has the form
#
# $$f(g)=\psi(\angle(g h,r)),$$
#
# where $\psi$ is the spherical kernel. Crystal and specimen symmetry make the
# construction invariant under equivalent descriptions. MTEX stores the object as an
# `SO3FunCBF`. It still uses the evaluation, plotting, scaling, and addition interface
# shared by every [SO3Fun](SO3FunConcept.html).

# ## Further Reading
#
# * H.-J. Bunge, [Texture Analysis in Materials Science: Mathematical Methods](https://doi.org/10.1016/C2013-0-11769-2),
#   Butterworths, 1982. Chapter 5 develops fibre textures and their orientation
#   distributions.
# * A. Morawiec, [Orientations and Rotations: Computations in Crystallographic Textures](https://doi.org/10.1007/978-3-662-09156-2),
#   Springer, 2004, develops rotation-space geometry and symmetry-reduced regions.
# * L. A. I. Kestens and H. Pirgazi,
#   [Texture formation in metal alloys with cubic crystal structures](https://doi.org/10.1080/02670836.2016.1231746),
#   _Materials Science and Technology_ 32 (2016), 1303--1315. This review covers named
#   cubic rolling fibres. It warns that the same Greek label can mean different fibres in
#   FCC and BCC materials.
# * H. Schaeben, [The de la Vallee Poussin Standard Orientation Density Function](https://doi.org/10.1155/TSM.33.365),
#   _Textures and Microstructures_ 33 (1999), 365--373. This paper relates the kernel
#   halfwidth to its finite harmonic representation.
# * [ISO 3785:2023](https://www.iso.org/standard/82165.html), _Metallic materials --
#   Designation of test specimen axes in relation to product texture_, standardises
#   specimen-axis language for rolled products.

# ## Next
#
# Continue with [Bingham ODFs](BinghamODFs.html) for a compact parametric model with
# three independent spreads. The preceding model family is [Radial ODFs](RadialODFs.html),
# and the curves without a density around them are
# [Fibres of Orientations](OrientationFibre.html). [ODF Properties](ODFCharacteristics.html)
# explains texture index and fibre volume. [Density Estimation](DensityEstimation.html)
# estimates an ODF from measured orientations instead of fitting a single parametric
# centreline.

# ## Technical Details
#
# `fibre.fit` returns the fibre alone
# unless `diagnostics=True` asks for MATLAB's three outputs, the eigenvalues and the
# mean distance. The random sample differs from MATLAB's draw, so the fitted numbers are
# of the same size but not the same digits.
