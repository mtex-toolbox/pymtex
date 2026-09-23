# # Inferring Twin Boundaries
#
# A twin is a part of a crystal whose lattice has a special mirrored orientation relative
# to its host. The crystallographic relation between the two orientations is the *twin
# law*.
#
# Deformation twins form in magnesium and other hexagonal metals when slip alone cannot
# accommodate the imposed strain. Their abundance and geometry can therefore help to
# reconstruct what happened to a specimen.
#
# This page infers a candidate twin law from a repeated misorientation in an EBSD map. It
# then selects the boundary segments that match the complete relationship.
# [Twinning](Twinning.html) takes the complementary route and starts from a known
# crystallographic law.
#
# The example assumes familiarity with [grain reconstruction](GrainReconstruction.html)
# and [misorientation symmetry](MisorientationTheory.html).
# [Misorientations at Grain Boundaries](BoundaryMisorientations.html) introduces the
# boundary properties used below.

import numpy as np

from mtex import (Miller, angle, axis, calcGrains, degree, histogram, hold, ipfColorKey, mean, mtexColorbar, mtexdata,
                  orientation, plot, plottingConvention, round, round2Miller, scatter, smoothBoundary, xlabel)

# +
# load the example in its specimen plotting frame
plottingConvention.default('y↑→x')
ebsd = mtexdata('twins')

# reconstruct and smooth the grains
grains = calcGrains(ebsd, angle=5 * degree, minPixel=3)
grains = smoothBoundary(grains, 5)

# compute the colours explicitly to keep the published output quiet
colorKey = ipfColorKey(grains)
grainColor = colorKey.orientation2color(grains.meanOrientation)
plot(grains, grainColor)

# store the crystal symmetry of magnesium
CS = grains.CS
# -

# The narrow lamellae cross several larger grains and have a contrasting mean orientation.
# That morphology suggests twinning, but it does not prove a twin mechanism. The following
# analysis identifies the repeated lattice relationship between each lamella and its host.

# ## Select Same-Phase Boundaries
#
# A grain boundary is one segment between neighboring EBSD measurements that belong to
# different grains. The summary splits the complete network by the phases on either side.

gB = grains.boundary
gB

# Of these segments, some are boundaries against the edge of the scan, where a grain is cut
# off and has no neighbour. MTEX reports the missing neighbour as `notIndexed`; this map
# contains no `notIndexed` grains of its own.
#
# Only segments with magnesium grains on both sides have the same-phase misorientations
# needed here. The object summary reports their number and total trace length. Segment
# count is controlled by boundary sampling and is not a count of physical twins.

gB_MgMg = gB['Magnesium', 'Magnesium']
gB_MgMg

# ## Find the Repeated Misorientation
#
# Colour each segment by its disorientation angle.

plot(gB_MgMg, gB_MgMg.misorientation.angle() / degree, linewidth=2)
mtexColorbar(title='misorientation angle in degree')

# The connected yellow traces near 86 degrees outline most of the narrow lamellae. Cooler
# colours mark boundaries spread over lower angles.
#
# The histogram makes the concentration explicit. The narrow peak just below 90 degrees
# dominates a background spread over many angles.

histogram(gB_MgMg.misorientation.angle() / degree, 40)
xlabel('misorientation angle (degree)')

# Isolate the 85--87 degree peak and print its share of the boundary segments. About one
# third of all magnesium-to-magnesium segments lie in this two-degree interval. A
# concentration this sharp is not expected from a random population; it points to one
# relationship repeated across the map.

# +
inPeak = (gB_MgMg.misorientation.angle() > 85 * degree) & (gB_MgMg.misorientation.angle() < 87 * degree)
mori = gB_MgMg.misorientation[inPeak]

peakSummary = {'segments': len(mori), 'percentOfMgMgSegments': 100 * len(mori) / len(gB_MgMg)}
peakSummary
# -

# ## Identify the Relationship
#
# Angle alone does not identify a misorientation. Plot the peak in the full axis--angle
# space.

scatter(mori)

# The points form one tight cluster rather than several clusters with similar angles. This
# compact cloud is the evidence for one complete relationship.
#
# The robust mean locates the centre without giving isolated measurements undue influence.
# [`round2Miller`](orientation.round2Miller.html) then finds low-index planes and
# directions that the measured mean makes parallel.

moriMean = mean(mori, robust=True)
round2Miller(moriMean)

# The fit is within half a degree. Use those low-index correspondences to define the ideal
# relationship without the small experimental deviation of the measured mean.

twinning = orientation.map(Miller(1, -1, 0, 1, CS), Miller(1, 0, -1, -1, CS),
                           Miller(0, 1, -1, 1, CS, 'uvw'), Miller(1, -1, 0, 1, CS, 'uvw'))
twinning

# ## Two Equivalent Descriptions
#
# By default MTEX returns the *disorientation*: the symmetry-equivalent representative with
# the smallest rotation angle. For this magnesium twin law it is 86.3 degrees about the
# (11-20) crystal axis.

twinDisorientationAxis = round(twinning.axis())
twinDisorientationAxis

twinDisorientationAngle = twinning.angle() / degree
twinDisorientationAngle

# The same twin law also has a 180 degree representative about a twin axis. The `max`
# option of [`angle`](orientation.angle.html) and [`axis`](orientation.axis.html) selects
# that representative.

twinOperationAngle = angle(twinning, 'max') / degree
twinOperationAngle

twinOperationAxis = round(Miller(axis(twinning, 'max'), 'UVTW'))
twinOperationAxis

# Both descriptions belong to the same misorientation under magnesium crystal symmetry. The
# 86.3 degrees reported for the disorientation and the 180 degrees quoted for the twin
# operation are therefore not in conflict. An angle and its axis must always be read from
# the same representative.

# ## Select Candidate Twin Boundaries
#
# Compare every boundary misorientation with the complete ideal law, not only with its
# 86.3 degree angle. A boundary has no intrinsic first side, so its misorientation carries
# grain-exchange symmetry: the law and its inverse are treated as the same relationship.
# See [Grain Exchange Symmetry](MisorientationGrainExchangeSym.html).
#
# A five-degree tolerance admits the experimental spread in this map. It is an analyst
# choice rather than a universal property of the twin law, and it should reflect
# orientation uncertainty and the purpose of the analysis.

# +
twinDeviation = angle(gB_MgMg.misorientation, twinning)
isTwinning = twinDeviation < 5 * degree
twinBoundary = gB_MgMg[isTwinning]
twinBoundary
# -

# report the fraction of trace length rather than only segment count
candidateTracePercent = 100 * np.sum(twinBoundary.segLength) / np.sum(gB_MgMg.segLength)
candidateTracePercent

# The two object summaries show how many magnesium-to-magnesium segments pass the test.
# They account for about half of the total trace length.

plot(grains, grainColor)
hold(True)
plot(twinBoundary, linecolor='w', linewidth=4, displayName='candidate twin boundary')
hold(False)

# The white traces follow the lamellae seen in the first figure. This spatial agreement
# checks that the relationship inferred from the peak is the one repeated in the map.

# ## What the Selection Establishes
#
# The selected objects are candidate twin-boundary segments, not a count of physical
# twins. Consecutive segments can belong to one lamella, and boundary smoothing changes
# their number. Trace length is less sensitive to resampling, but it still does not count
# twin domains.
#
# A misorientation match also does not identify which side is the parent or prove that
# deformation twinning produced the boundary. A two-dimensional EBSD map supplies the
# boundary trace, not the full interface-plane orientation;
# [Tilt and Twist Boundaries](TiltAndTwistBoundaries.html) develops that limitation.
# Morphology, loading, and evidence for later slip, detwinning, or secondary twinning may
# be needed for a mechanistic interpretation.

# ## References
#
# * Th. Hahn and H. Klapper,
#   [Twinning of Crystals](https://doi.org/10.1107/97809553602060000917),
#   *International Tables for Crystallography*, Vol. D, ch. 3.3, pp. 413--487, 2013. This
#   chapter defines twin laws, equivalent operations, and twin boundaries.
# * J. W. Christian and S. Mahajan,
#   [Deformation Twinning](https://doi.org/10.1016/0079-6425(94)00007-7),
#   *Progress in Materials Science* 39 (1995), 1--157. This review covers deformation-twin
#   crystallography and mechanisms.
# * Y. Zhang *et al.*,
#   [A General Method to Determine Twinning Elements](https://doi.org/10.1107/S0021889810037180),
#   *Journal of Applied Crystallography* 43 (2010), 1426--1430. This paper connects an
#   observed relationship with classical twinning elements.
# * V. Randle,
#   [Application of EBSD to the Analysis of Interface Planes: Evolution over the Last Two Decades](https://doi.org/10.1111/j.1365-2818.2008.02000.x),
#   *Journal of Microscopy* 230 (2008), 406--413. This review explains what
#   two-dimensional and stereological EBSD measurements reveal about boundary planes.

# ## Next
#
# If the twin law is known in advance, [Twinning](Twinning.html) shows how to construct it
# directly. The next reconstruction step is often to restore parent-grain footprints by
# merging across the selected boundaries; see [Merging Grains](GrainMerge.html).
