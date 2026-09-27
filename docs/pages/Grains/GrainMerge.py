# # Merging Grains
#
# Grain segmentation correctly identifies a twin domain as a grain: it is a
# phase-homogeneous, spatially connected region of EBSD measurements. For an analysis of
# the grain before twinning, however, that domain belongs to its host. Merging removes the
# selected internal boundaries and reconstructs this parent-grain footprint.
#
# In parent-phase reconstruction, merging is the second of two distinct steps. The child
# grains are first transformed to candidate parent orientations and only then merged where
# those candidates are compatible. See
# [Grain Graph Based Reconstruction](GrainGraphBasedReconstruction.html) for that complete
# workflow.
#
# This page uses deformation twins in magnesium. It assumes familiarity with
# [grain reconstruction](GrainReconstruction.html), [grain IDs](SelectingGrains.html), and
# [boundary misorientations](BoundaryMisorientations.html).

import numpy as np

from mtex import *

# +
# load the example in its specimen plotting frame
plottingConvention.default('y↑→x')
ebsd = mtexdata('twins')

# reconstruct and smooth the grains
grains = calcGrains(ebsd, angle=5 * degree, minPixel=3)
grains = smoothBoundary(grains, 5)
grains = grains['indexed']

# compute the mean-orientation colours without printing a colour-key notice
colorKey = ipfColorKey(grains)
grainColor = colorKey.orientation2color(grains.meanOrientation)
plot(grains, grainColor)
# -

# The narrow lamellae have mean-orientation colours unlike their surrounding grains. Their
# morphology suggests twins, but morphology alone does not establish a twin relationship.

# ## Select candidate twin boundaries
#
# A twin law maps two pairs of crystallographic directions onto each other.
# `orientation.map` constructs that complete relationship. Comparing complete
# misorientations is more selective than comparing their rotation angles alone.

# +
# define the ideal twinning misorientation
CS = grains.CS
twinning = orientation.map(Miller(0, 1, -1, -2, CS), Miller(0, -1, 1, -2, CS), Miller(2, -1, -1, 0, CS), Miller(2, -1, -1, 0, CS))

# extract magnesium-to-magnesium grain boundaries
gB = grains.boundary['Magnesium', 'Magnesium']

# select segments within 5 degrees of the twin law
isTwinning = angle(gB.misorientation, twinning) < 5 * degree
twinBoundary = gB[isTwinning]
twinBoundary
# -

# report how much of the sampled boundary network passed the test
print(f'Mg-Mg segments: {len(gB)}, candidate twin segments: {len(twinBoundary)}, percent: {100 * len(twinBoundary) / len(gB):.3f}')

# overlay the candidate twin boundaries
hold(True)
plot(twinBoundary, lineColor='w', lineWidth=4, DisplayName='candidate twin boundary')
hold(False)

# The white traces follow the narrow lamellae. With this boundary sampling, 1406 of 2696
# magnesium-to-magnesium segments pass the five-degree test. A segment count is not a count
# of physical twins, and smoothing or resampling the same traces can change it. The
# tolerance is also an analyst choice: these are candidate twin boundaries, not proof of
# the mechanism or of which side is the parent.
# [Inferring Twin Boundaries](TwinningBoundaries.html) develops those limitations.

# ## Merging along selected boundaries
#
# `merge` joins the grains on both sides of every supplied boundary. It treats the
# selected boundaries as connections in a graph and merges each connected component. The
# operation is therefore transitive: if A is joined to B and B to C, all three become one
# grain. One false connecting boundary can consequently fuse a much larger parent than
# expected.

# +
mergedGrains = merge(grains, twinBoundary)
parentId = grains.parentId

print(f'grains before: {len(grains)}, after: {len(mergedGrains)}')

# overlay the reconstructed parent-grain boundaries
hold(True)
plot(mergedGrains.boundary, lineColor='k', lineWidth=2.5, DisplayName='merged grains')
hold(False)
# -

# The 91 segmented grains have become 28 merged grains. Black lines are the reconstructed
# parent-grain outlines. Each white trace that no longer has a black line on it is now
# internal to one merged grain.

# ## Keeping track of the child grains
#
# `merge` writes `parentId` into the grains it was given, with one entry for every grain in
# `grains`. `parentId[k]` is the ID of the merged grain that contains `grains[k]`. This is
# the same kind of bookkeeping that `ebsd.grainId` provides between measurements and
# grains; it is a mapping of persistent IDs, not a list position.

selectedParentId = mergedGrains[15].id
selectedParentId

# The following object summary lists the child grains assigned to merged grain 16.

childs = grains[parentId == selectedParentId]
childs

# ## Which child domains are twins
#
# Merging establishes which domains belong together. It does not identify the original
# host or the twin. One simple rule works while twins occupy less area than their host:
# cluster each parent's child orientations and label the largest orientation cluster by
# area as the original grain.

# +
# cache the child-grain areas
gArea = grains.area

# classify the smaller orientation clusters as twin domains
isTwin = np.ones(len(grains), dtype=bool)
for i in range(len(mergedGrains)):

  # find the children of this merged-grain ID
  parent = mergedGrains[i].id
  childInd = np.flatnonzero(parentId == parent)

  # cluster children with similar mean orientations
  fId, _ = calcCluster(grains.meanOrientation[childInd], maxAngle=15 * degree, method='hierarchical')

  # find the orientation cluster with the largest total area
  clusterArea = np.bincount(fId, gArea[childInd])
  fParent = np.argmax(clusterArea)
  isTwin[childInd[fId == fParent]] = False

# report the mapped-area fraction assigned to twins
twinAreaPercent = 100 * np.sum(grains[isTwin].area) / np.sum(grains.area)
twinAreaPercent
# -

# visualize the classification
plot(grains[~isTwin], faceColor='darkgray', DisplayName='not twin')
hold(True)
plot(grains[isTwin], faceColor='red', DisplayName='twin')
plot(mergedGrains.boundary, lineColor='k', lineWidth=2, DisplayName='merged grains')
mtexTitle('twin classification')
hold(False)

# The rule assigns 17 percent of the mapped area to twins. The red domains are lamellae
# inside the grey hosts, which is a useful spatial check on the classification. The rule
# fails if a twin consumes more than half of its host. This map alone cannot reveal that
# history.

# ## Properties of the merged grains
#
# A merged grain receives a new shape and the summed pixel count. Other grain properties
# have no universal merge rule. In particular, directly averaging host and twin
# orientations usually does not recover the host orientation, so the mean orientation of a
# merged grain is not appropriate for this example. The intended physical quantity must
# determine how each property is carried from children to parents.
#
# For [grain orientation spread (GOS)](GrainOrientationParameters.html), `parentId`
# supplies the groups to `np.bincount`. An unweighted mean gives every child grain one
# vote.

# +
# compute an unweighted child-grain mean for each parent
unweightedGOS = np.bincount(parentId, grains.GOS, len(mergedGrains) + 1)[1:] / np.bincount(parentId, minlength=len(mergedGrains) + 1)[1:]
mergedGrains.GOS = unweightedGOS

# compare child and unweighted parent values
plot(grains, grains.GOS / degree)
hold(True)
plot(mergedGrains.boundary, lineColor='white', lineWidth=2)
mtexTitle('child-grain GOS')
hold(False)

nextAxis()
plot(mergedGrains, unweightedGOS / degree)
mtexTitle('unweighted merged GOS')
setColorRange([0, 1.5])
# -

# This mean counts a two-pixel twin as much as the grain that contains it. That is rarely
# the intended statistic. Weighting each child by area gives each measured part of the
# parent equal influence.

# +
# cache GOS and area for the weighted accumulation
childGOS = grains.GOS
childArea = grains.area

# compute the area-weighted child-grain mean for each parent
weightedGOS = np.bincount(parentId, childGOS * childArea, len(mergedGrains) + 1)[1:] / np.bincount(parentId, childArea, len(mergedGrains) + 1)[1:]
mergedGrains.GOS = weightedGOS

nextAxis()
plot(mergedGrains, weightedGOS / degree)
mtexTitle('area-weighted merged GOS')
mtexColorbar()
setColorRange([0, 1.5])

print(f'weighted value higher in {np.count_nonzero(weightedGOS > unweightedGOS)} of {len(mergedGrains)} parents, '
      f'maximum increase {np.max(weightedGOS - unweightedGOS) / degree:.5f} degrees')
# -

# The two parent maps differ in both directions. The weighted value is larger in 14 of the
# 28 parents, by as much as 0.6 degrees. In those parents the larger children carry the
# higher GOS values, while small twins pull the unweighted mean down. More generally,
# orientation spread can depend on grain size, so the weighting rule must be reported.

# ## Pointing EBSD measurements at the merged grains
#
# The EBSD measurements still carry the `grainId` values from the original segmentation.
# Those IDs refer to a different grain list after merging. Indexing `ebsd` with a merged
# grain therefore selects the wrong measurements without raising an error.

# +
targetMergedGrain = mergedGrains['id', 22]
staleMeasurements = ebsd[targetMergedGrain]
staleColor = colorKey.orientation2color(staleMeasurements.orientations)

plot(targetMergedGrain.boundary, lineWidth=2)
hold(True)
plot(staleMeasurements, staleColor)
hold(False)
# -

# The coloured measurements do not fill the outlined merged grain. They are the
# measurements whose old `grainId` happens to equal the new ID 22. Replace every indexed
# measurement's old grain ID by the parent ID of its original grain.

# +
# copy the EBSD data so the original mapping remains available
ebsd_merged = ebsd.copy()

# map old grain IDs to indices in grains, then from children to parent IDs
indexed = ebsd_merged.isIndexed
ebsd_merged.grainId[indexed] = parentId[grains.id2ind(ebsd.grainId[indexed])]

correctedMeasurements = ebsd_merged[targetMergedGrain]
correctedColor = colorKey.orientation2color(correctedMeasurements.orientations)
# -

# The same lookup now returns the measurements inside the outline. Updating `grainId` is
# essential whenever subsequent EBSD analysis uses the merged grains.

plot(correctedMeasurements, correctedColor)
hold(True)
plot(targetMergedGrain.boundary, lineWidth=3)
hold(False)

# ## References
#
# * F. Bachmann, R. Hielscher, and H. Schaeben,
#   [Grain Detection from 2d and 3d EBSD Data - Specification of the MTEX Algorithm](https://doi.org/10.1016/j.ultramic.2011.08.002),
#   *Ultramicroscopy* 111 (2011), 1720-1733. This paper derives the grain and boundary
#   network on which merging operates.
# * J. W. Christian and S. Mahajan,
#   [Deformation Twinning](https://doi.org/10.1016/0079-6425(94)00007-7), *Progress in
#   Materials Science* 39 (1995), 1-157. This review develops the crystallography and
#   mechanisms of deformation twins.
# * W. Pantleon, W. He, T. O. Johansson, and C. Gundlach,
#   [Orientation Inhomogeneities within Individual Grains in Cold-Rolled Aluminium Resolved by Electron Backscatter Diffraction](https://doi.org/10.1016/j.msea.2006.08.139),
#   *Materials Science and Engineering A* 483-484 (2008), 668-671. This paper discusses
#   orientation spread and its dependence on grain size.
# * [ASTM E2627-13(2019)](https://doi.org/10.1520/E2627-13R19), *Standard Practice for
#   Determining Average Grain Size Using Electron Backscatter Diffraction (EBSD) in Fully
#   Recrystallized Polycrystalline Materials*. Its scope is fully recrystallized materials,
#   but it shows why boundary and grain definitions must be stated when merged grains enter
#   a size analysis.

# ## Next
#
# Continue with [Inferring Twin Boundaries](TwinningBoundaries.html) when the twin law
# must be inferred from the data. For a phase transformation, see
# [Grain Graph Based Reconstruction](GrainGraphBasedReconstruction.html) for the
# transform-then-merge workflow.

# ## Technical details
#
# `merge` returns the merged grains and writes `parentId` into the grains it was given,
# where MATLAB returns it as a second output. MATLAB's `accumarray` over the parent ids is
# `np.bincount` with weights, and the grain id written into a subset of a map goes through
# a mask on the whole map, since `ebsd['indexed']` is a copy.
