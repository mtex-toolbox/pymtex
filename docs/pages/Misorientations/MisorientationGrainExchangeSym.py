# # Grain Exchange Symmetry
#
# A [misorientation](MisorientationTheory.html) maps crystal coordinates of one crystal
# into those of another. For grains of the *same* phase, neither side of their shared
# boundary is intrinsically first. Swapping the grains replaces the misorientation by its
# inverse, but it does not describe a different boundary, and no measurement can tell the
# two apart.
#
# This identification is *grain exchange symmetry*. It is a symmetry of an unordered pair
# of grains, not an additional symmetry of either crystal. MTEX records it in the
# `antipodal` property of a misorientation.
#
# The name is shared with the flag that identifies opposite directions, described in
# [Axes and Antipodal Symmetry](VectorsAxes.html). For a vector it means that $\mathbf{v}$
# and $-\mathbf{v}$ are equivalent. For a misorientation it means that $m$ and $m^{-1}$ are
# equivalent.

# +
from mtex import *

plottingConvention.default('y↑→x')

cs = crystalFrame('622')
# -

# ## Without Grain Exchange Symmetry
#
# Start with a $40^\circ$ misorientation and compare it with its inverse. The `antipodal`
# property is initially false, so the order of the two crystals matters.

mori = orientation.byAxisAngle(Miller(1, 2, -3, 1, cs), 40 * degree, cs, cs)
angle(mori, inv(mori)) / degree

# The two descriptions are about $14.1^\circ$ apart after crystal symmetry has been
# applied. They are therefore distinct when MTEX treats the pair as ordered. That is the
# right answer between two different phases, where "first" and "second" mean something -
# the parent and the child of a transformation, say.
#
# In axis--angle coordinates, inversion keeps the rotation angle and reverses the axis. The
# two markers below consequently occupy opposite points on the upper and lower hemispheres.

# +
axMori = axis(mori, noSymmetry=True)
axInverse = axis(inv(mori), noSymmetry=True)

plot(axMori, complete=True, MarkerSize=12, DisplayName='m', grid=True)
hold(True)
plot(axInverse, complete=True, MarkerSize=12, DisplayName='inverse(m)')
hold(False)
legend(location='southeast')
# -

# The blue and orange points are antipodal: they mark the same $40^\circ$ angle about
# opposite axes. Grain exchange symmetry will identify this pair; it will not change the
# stored rotation.
#
# Point group `622` has 12 proper rotations. Acting from both crystal frames gives
# $12 \times 12 = 144$ symmetrically equivalent descriptions for this ordered pair.

numEquivalentOrdered = len(mori.symmetrise())
numEquivalentOrdered

# ## With Grain Exchange Symmetry
#
# Set `antipodal` when the grain order is arbitrary.

mori.antipodal = True
angle(mori, inv(mori)) / degree

# The symmetry-aware distance is now zero because the inverse belongs to the same
# equivalence class. This does not say that the misorientation is the identity; its
# rotation angle is still $40^\circ$.
#
# For this generic example, [symmetrise](orientation.symmetrise.html) now includes the
# inverse of every crystal-symmetry equivalent. The displayed list therefore doubles from
# 144 to 288 descriptions.

numEquivalentUnordered = len(mori.symmetrise())
numEquivalentUnordered

# ## The Fundamental Region Becomes Smaller
#
# A [fundamental region](OrientationFundamentalRegion.html) keeps one representative of each
# equivalence class. Adding grain exchange symmetry identifies each misorientation with its
# inverse, so fewer representatives are needed.

# +
orderedRegion = fundamentalRegion(cs, cs)
exchangeRegion = fundamentalRegion(cs, cs, antipodal=True)

plot(orderedRegion, boundaryColor=[0.2, 0.4, 0.8], noSurface=True, figSize='small')
hold(True)
plot(exchangeRegion, boundaryColor=[0.85, 0.3, 0.1], noSurface=True)
hold(False)
# -

# The blue outline is the region for an ordered pair. The orange outline is the region after
# inverse misorientations have been identified. It has half the volume, and here it lies
# inside the blue one. The number of faces is not a measure of region size.
#
# ## Where MTEX Sets the Flag Automatically
#
# A grain is a phase-homogeneous, spatially connected region of EBSD pixels produced by
# segmentation. See [grain reconstruction](GrainReconstruction.html) for that step.
# Boundary misorientations between grains of one phase receive grain exchange symmetry
# automatically.

# +
ebsd = mtexdata('twins', silent=True)

grains = calcGrains(ebsd, threshold=5 * degree, minPixel=5)

grains.boundary['Mag', 'Mag'].misorientation.antipodal
# -

# Between two phases, the phase labels distinguish the source and target crystal frames.
# Reversing a forsterite-to-enstatite mapping gives an enstatite-to-forsterite mapping, so
# MTEX leaves the flag off.

# +
ebsd = mtexdata('forsterite', silent=True)

grains = calcGrains(ebsd)

grains.boundary['Fo', 'En'].misorientation.antipodal
# -

# The same distinction applies inside one grain. A misorientation between the grain mean
# and an orientation measured within that grain has a fixed reference and target, so
# `antipodal` belongs off. See [Grain Orientation Parameters](GrainOrientationParameters.html)
# for this grain-reference orientation deviation.
#
# ## Why Swapping Gives the Inverse
#
# Let $g_1$ and $g_2$ be orientations that map the two crystal frames into the specimen
# frame. The misorientation of the ordered pair maps crystal 2 coordinates into crystal 1,
#
# $$m_{12} = g_1^{-1}g_2.$$
#
# Swapping the grains reverses the composition:
#
# $$m_{21} = g_2^{-1}g_1 = m_{12}^{-1}.$$
#
# The algebra always gives the inverse. Grain exchange symmetry identifies the two results
# only when the grain pair is physically unordered and both grains have the same phase.
#
# ## References
#
# * A. Morawiec, [Orientations and Rotations: Computations in Crystallographic Textures](https://doi.org/10.1007/978-3-662-09156-2),
#   Springer, 2004, develops rotation-space geometry and crystalline-interface symmetry.
# * S. Patala and C. A. Schuh, [Topology of Homophase Grain Boundaries in Two-Dimensional Crystals: The Role of Grain Exchange Symmetry](https://doi.org/10.3970/cmc.2010.017.001),
#   _Computers, Materials & Continua_ 17 (2010), 1--18, develops the topological
#   consequences of indistinguishable adjoining grains.
# * R. Krakow _et al._, [On Three-Dimensional Misorientation Spaces](https://doi.org/10.1098/rspa.2017.0274),
#   _Proceedings of the Royal Society A_ 473 (2017), 20170274, shows how grain exchange
#   symmetry halves a same-phase misorientation fundamental zone.
#
# ## Next
#
# Continue with [Twinning](Twinning.html), where same-phase boundary misorientations are
# compared with an ideal orientation relationship. The geometric effect of the flag is
# developed further in [Fundamental Region](OrientationFundamentalRegion.html), and its
# effect on populations appears in [Axis Distribution](AxisDistributionFunction.html).
