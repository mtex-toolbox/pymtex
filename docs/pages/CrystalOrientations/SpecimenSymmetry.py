# # Specimen Symmetry

# +
from mtex import Miller, crystalFrame, degree, mtexColorbar, norm, numProper, numSym, orientation, plotPDF, plottingConvention, \
    specimenFrame, unimodalODF

plottingConvention.default('y↑→x')
# -

# [Crystal symmetry](CrystalSymmetries.html) describes operations that leave a crystal
# lattice unchanged. *Specimen symmetry* describes operations that leave the texture of a
# specimen unchanged. The forming process is often the physical reason for this
# invariance.
#
# Rolling, for example, is commonly modelled by orthorhombic specimen symmetry. A texture
# with this symmetry is unchanged by $180^\circ$ rotations about the rolling direction
# (RD), transverse direction (TD), and normal direction (ND).
#
# This is a model of the texture, not a consequence of the specimen having a rectangular
# shape. A shear texture, a through-thickness gradient, or a misaligned specimen may not
# have that symmetry. Imposing symmetry that is absent averages distinct components
# together and hides real information.
#
# An orientation maps coordinates from the crystal frame into the specimen frame. Specimen
# symmetry therefore acts from the left, while crystal symmetry acts from the right. See
# [Symmetrically Equivalent Orientations](OrientationSymmetry.html).
#
# ## Defining a Specimen Symmetry
#
# A `specimenFrame` is defined by its point group. Only a small number of point groups
# occur in practice. The trivial group is the default when no specimen symmetry is
# specified.

ss = specimenFrame('1')
ss

# Orthorhombic specimen symmetry is the common choice for rolled material. It may be
# specified by its point group `'mmm'` or by the name `'orthorhombic'`.

ss = specimenFrame('mmm')
ss

# The full point group `mmm` has eight operations. Its proper subgroup has four rotations:
# the identity and one $180^\circ$ rotation about each specimen axis. Orientations live in
# the rotation group, so these proper operations are the ones that identify equivalent
# orientations.

numberOfOperations = numSym(ss)
numberOfProperRotations = numProper(ss)
numberOfOperations, numberOfProperRotations

# Note the difference between `'1'` and `'triclinic'`. The first denotes the identity
# alone. The lattice-type name `'triclinic'` selects the point group $\bar 1$, which also
# contains inversion.

numberInIdentityGroup = numSym(specimenFrame('1'))
numberInTriclinicGroup = numSym(specimenFrame('triclinic'))
numberInIdentityGroup, numberInTriclinicGroup

# ## The Effect on an ODF
#
# An [orientation distribution function](ODFAnalysis.html) (ODF) is a density over
# orientations. We begin with one smooth component and trivial specimen symmetry.

cs = crystalFrame.load('quartz.cif')
odf = unimodalODF(orientation.byEuler(30 * degree, 50 * degree, 10 * degree, cs), halfwidth=15 * degree)

# The `complete` and `upper` flags keep the full upper hemisphere visible. This matters for
# the comparison below, because a plot with nontrivial specimen symmetry otherwise defaults
# to its specimen fundamental sector.

plotPDF(odf, Miller(1, 0, -1, 0, cs), contourf=True, complete=True, upper=True)
mtexColorbar()

# The peaks are equivalent crystal poles. Notice that their pattern is not forced to be
# symmetric about both horizontal and vertical specimen axes.
#
# The texture index is the squared [norm](SO3Fun.norm.html) of the ODF. The first value is
# the baseline before specimen symmetry is imposed.

textureIndexTrivial = norm(odf) ** 2
textureIndexTrivial

# Assigning orthorhombic specimen symmetry makes the ODF invariant under its proper
# rotations. This changes the represented function rather than merely changing its plot.

odf.SS = specimenFrame('mmm')

# ---

plotPDF(odf, Miller(1, 0, -1, 0, cs), contourf=True, complete=True, upper=True)
mtexColorbar()

# The second pole figure is symmetric about the horizontal and vertical specimen axes. The
# density is shared among the equivalent positions, so the peaks are lower than in the
# first plot.

textureIndexOrthorhombic = norm(odf) ** 2
textureIndexOrthorhombic

# The second texture index is also smaller. Quantities derived from the ODF therefore
# change when specimen symmetry is imposed.
#
# ## Specimen Symmetry and the Specimen Frame
#
# A [reference frame](referenceFrame.referenceFrame.html) is the coordinate system in which
# data are expressed. A symmetry is the point group under which those data are invariant.
# The two are attached, but they are not the same concept.
#
# Which physical direction a symmetry axis denotes depends on the specimen frame. The point
# group `'112'` has its twofold rotation about the $z$ axis.

ss = specimenFrame('112')
ss

# The rolling frame names its axes RD, TD, and ND and supplies their plotting convention.
# Asking it for the same point group therefore declares that the twofold $z$ axis is ND,
# and those names appear in summaries and plots.

ss = specimenFrame.rolling('112')
ss

# Assigning a frame does not rotate an ODF or correct a mounting error. Use
# [Sample Symmetry](DetectionOfSampleSymmetry.html) to test and align a measured texture
# before imposing a nontrivial symmetry.
#
# ## References
#
# * H.-J. Bunge, [Texture Analysis in Materials Science: Mathematical Methods](https://doi.org/10.1016/C2013-0-11769-2),
#   Butterworths, English ed., 1982, develops ODFs with crystal and specimen symmetry.
# * A. Morawiec, [Orientations and Rotations: Computations in Crystallographic Textures](https://doi.org/10.1007/978-3-662-09156-2),
#   Springer, 2004, treats orientation space under crystal and specimen symmetries.
# * U. F. Kocks, C. N. Tomé and H.-R. Wenk, [Texture and Anisotropy: Preferred Orientations in Polycrystals and Their Effect on Materials Properties](https://assets.cambridge.org/97805217/94206/frontmatter/9780521794206_frontmatter.pdf),
#   Cambridge University Press, 1998.
# * J. S. Kallend, U. F. Kocks, A. D. Rollett and H.-R. Wenk, [Operational texture analysis](https://doi.org/10.1016/0921-5093(91)90355-Q),
#   _Materials Science and Engineering A_ 132 (1991), 1--11, treats quantitative texture
#   analysis with general specimen symmetry.
# * [ISO 3785:2023](https://www.iso.org/standard/82165.html), _Metallic materials --
#   Designation of test specimen axes in relation to product texture_, standardises how
#   specimen directions are reported.
#
# ## Next
#
# [Orientation Grids](OrientationGrid.html) use symmetry to restrict sampling to the
# fundamental region. The ODF chapter develops [texture characteristics](ODFCharacteristics.html)
# and the consequences of imposing specimen symmetry on them.
