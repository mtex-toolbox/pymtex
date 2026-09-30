# # Twinning
#
# A *twin law* is a special crystallographic orientation relationship between parts of the
# same crystal species. Growth, transformation, and deformation twins arise by different
# processes. Each repeats one discrete lattice relationship.
#
# MTEX represents the rotational part of a twin law as a same-phase misorientation. This
# page constructs an ideal twin law, explains why its quoted angle depends on crystal
# symmetry, and compares it with an EBSD map.
#
# Start with [Theory of Misorientations](MisorientationTheory.html) if symmetry-equivalent
# misorientations and disorientation are unfamiliar. The treatment of an unordered pair of
# same-phase grains is introduced in [Grain Exchange Symmetry](MisorientationGrainExchangeSym.html).

# +
import matplotlib.pyplot as plt
import numpy as np

from mtex import *

plottingConvention.default('y↑→x')
ebsd = mtexdata('twins', verbose=False)

# use the crystal symmetry and lattice parameters stored with the data
CS = ebsd['Magnesium'].CS
# -

# ## Constructing an Ideal Twin Relationship
#
# A rotation is fixed by two non-parallel correspondences. For the magnesium extension twin,
# choose a parent plane normal and direction. Their counterparts describe the same features
# in the twin.

# +
parentPlane = Miller(1, -1, 0, 1, CS)
twinPlane = Miller(1, 0, -1, -1, CS)
parentDirection = Miller(0, 1, -1, 1, CS, 'uvw')
twinDirection = Miller(1, -1, 0, 1, CS, 'uvw')

# map the parent correspondences onto the twin correspondences
twinning = orientation.map(parentPlane, twinPlane, parentDirection, twinDirection)
# -

# [round2Miller](orientation.round2Miller.html) recovers one low-index description of the
# relationship. Crystal symmetry permits several equally valid descriptions. The reported
# indices therefore need not repeat the input representatives.

round2Miller(twinning)

# ## Seeing the Reorientation
#
# The same hexagonal crystal shape can represent the parent and the twin. The twin law
# rotates the orange copy while leaving its lattice and shape unchanged.

# +
cS = crystalShape.hex(CS)

plot(cS, faceColor='LightSkyBlue', faceAlpha=0.55, figSize='large')
hold(True)
plot(0.9 * (twinning * cS), faceColor='orange', faceAlpha=0.55)
hold(False)
plt.gca().view_init(elev=20, azim=35 - 90)
# -

# The two prisms have the same faces and proportions. Their discrete relative placement is
# the orientation relationship; the picture does not show the shear or the plane of their
# physical interface.
#
# ## Why 86.3 Degrees and 180 Degrees Are Both Correct
#
# A twin law is an equivalence class of rotations under crystal symmetry.
# [angle](orientation.angle.html) returns the smallest-angle representative by default,
# while the `max` option returns the largest-angle representative.

twinAngles = np.array([angle(twinning), twinning.angle('max')]) / degree
twinAngles

# The magnesium extension twin therefore has a disorientation angle of $86.299^\circ$, but
# the same twin law also has a $180^\circ$ representative. A textbook using the 180 degree
# description and MTEX reporting 86.3 degrees are not in conflict.
#
# The rotation axis changes with the representative as well. Always state which
# representative and crystal frame an axis belongs to.
# [Axis Distribution](AxisDistributionFunction.html) develops this distinction for
# populations of misorientations.
#
# ## The Twin as a Near Coincidence
#
# A coincidence site lattice (CSL) misorientation takes $\Sigma$ times every lattice vector
# onto a lattice vector, so that one lattice site in $\Sigma$ is shared by both crystals.
# Which rotations do this is a property of the lattice alone. For magnesium,
# $c^2/a^2 = 2.636$ is not a ratio of small integers, and the exact coincidences up to
# $\Sigma 19$ are only the rotations about $[0001]$: $\Sigma 7$, $\Sigma 13$ and $\Sigma 19$.

np.asarray(CSL(range(1, 20), CS).angle()) / degree

# The option `delta` allows a relative misfit of the lattice. Within two percent the metric
# is replaced by the ideal ratio $c^2/a^2 = 8/3$, and the coincidences of that lattice are
# the near coincidences of magnesium. Among them is $\Sigma 17a$, $86.63^\circ$ about
# $\langle 2\bar1\bar10\rangle$, a third of a degree from the extension twin.

csl17 = CSL(17, CS, delta=0.02)
np.hstack([angle(csl17), angle(csl17, twinning)]) / degree

# This is why the extension twin is described as a near $\Sigma 17a$ boundary. The CSL
# names the nearby coincidence, but the twin law itself, $86.3^\circ$, is fixed by the
# twinning elements and the real lattice.
#
# ## Comparing the Twin Law with Measured Boundaries
#
# A grain is a phase-homogeneous, spatially connected region of EBSD pixels produced by
# segmentation. The map is segmented at $5^\circ$, and grains with fewer than five indexed
# pixels are excluded. Boundary geometry is then smoothed for five iterations before trace
# lengths are compared.

# +
grains = calcGrains(ebsd['indexed'], angle=5 * degree, minPixel=5)
grains = smoothBoundary(grains, 5)

# retain only boundaries between magnesium grains
gB = grains.boundary['Magnesium', 'Magnesium']
# -

# A same-phase boundary is unordered. Marking the ideal relationship as antipodal makes its
# inverse equivalent. This is the grain exchange symmetry required for the comparison.

# +
twinBoundaryRelation = twinning.copy()
twinBoundaryRelation.antipodal = True

# compare the complete misorientation with the ideal relationship
twinDeviation = angle(gB.misorientation, twinBoundaryRelation) / degree

plt.figure()
plt.hist(twinDeviation, np.arange(0, 91, 2))
plt.xlabel('deviation from ideal twin (degree)')
# -

# The strong concentration near zero is the repeated extension-twin relationship. Comparing
# only boundary angles near 86.3 degrees would discard the axis information and can admit
# unrelated boundaries.
#
# ## Candidate Twin Boundaries
#
# A five degree tolerance selects candidate boundaries in this example. The tolerance is an
# analyst choice, not a universal property of the twin law, and should reflect orientation
# uncertainty and the scientific aim.

# +
isCandidate = twinDeviation < 5

# weight the result by boundary trace length rather than segment count
candidateTracePercent = 100 * np.sum(gB[isCandidate].segLength) / np.sum(gB.segLength)
candidateTracePercent
# -

# With this segmentation, smoothing, and tolerance, candidates account for about 49 percent
# of the magnesium-to-magnesium boundary trace length.

plot(grains, grains.meanOrientation, ipfDirection=zvector, micronbar='off')
hold(True)
plot(gB[isCandidate], lineColor='w', lineWidth=3)
hold(False)

# The white traces follow the thin lamellae in the orientation map. This spatial agreement
# supports the crystallographic classification.
#
# A misorientation match alone does not prove a twinning mechanism. A full boundary
# description also needs the interface-plane orientation. A two-dimensional EBSD map
# measures only its trace. Morphology, loading, and other evidence may also be needed.
# [Twinning Analysis](TwinningBoundaries.html) infers a relationship from measured
# boundaries and inspects the selected traces.
#
# ## References
#
# * Th. Hahn and H. Klapper, [Twinning of Crystals](https://doi.org/10.1107/97809553602060000644),
#   _International Tables for Crystallography_, Vol. D, ch. 3.3, pp. 393--448, 2006. This
#   chapter treats twin laws, morphology, origins, and interfaces.
# * J. W. Christian and S. Mahajan, [Deformation Twinning](https://doi.org/10.1016/0079-6425(94)00007-7),
#   _Progress in Materials Science_ 39 (1995), 1--157. This review covers twinning shear,
#   modes, and deformation mechanisms.
# * Y. Zhang _et al._, [A General Method to Determine Twinning Elements](https://doi.org/10.1107/S0021889810037180),
#   _Journal of Applied Crystallography_ 43 (2010), 1426--1430. This paper connects
#   measured orientation relationships with classical twinning elements.
#
# ## Next
#
# A single ideal relationship appears as a peak in a population.
# [Misorientation Distribution Function](MisorientationDistributionFunction.html) compares
# correlated boundary misorientations with the uncorrelated distribution expected from
# texture.
#
# The same distinction reduced to rotation angle alone is developed in
# [Angle Distribution](AngleDistributionFunction.html). Twin laws between parent and
# product phases lead into [Phase Transitions](PhaseTransitions.html).
