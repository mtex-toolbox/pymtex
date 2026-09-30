# # EBSD Simulation
#
# A map-processing method is easiest to test on a map whose answer is known in advance.
# The [simulateEBSD](simulateEBSD.simulateEBSD.html) class constructs such a map one
# feature at a time. It can add orientation noise, a low-angle boundary, or an orientation
# gradient to a uniform field.
#
# This makes it useful for checking [denoising](EBSDDenoising.html) and [KAM](EBSDKAM.html).
# It also supplies known input for [GND](GND.html) and [WBV](WBV.html) calculations.
# Synthetic and experimental examples appear in
# [Hielscher et al. (2019)](https://doi.org/10.1107/S1600576719009075).
#
# First read about [orientations](OrientationDefinition.html) and
# [misorientations](MisorientationTheory.html). The page also assumes familiarity with
# [grain reconstruction](GrainReconstruction.html).
#
# ## What is being simulated
#
# `simulateEBSD` constructs an `EBSD` variable containing positions, a phase, and
# orientations. It does not generate diffraction patterns or model the detector, pattern
# indexing, spatial distortion, or failed measurements. A program such as
# [EMsoft](https://github.com/EMsoft-org/EMsoft) is needed for physics-based EBSD pattern
# simulation.
#
# The class is designed for a single-grain orientation field. Use the
# [Neper interface](NeperInterface.html) when polycrystal topology matters.
#
# ## The object and its defaults
#
# The object holds the map settings and the current simulated map. Its defaults specify
# coordinate limits of 100 by 100 and a unit step size. They do not create a map or add
# noise until the corresponding methods are called.

import numpy as np

from mtex import *

# +
plottingConvention.default('y↓→x')

eS = simulateEBSD()
eS
# -

# ## A uniform orientation field
#
# Set the coordinate limit, crystal symmetry, and reference orientation. The method
# `makeMap` then creates the `EBSD` variable in `eS.EBSDsim`.

# +
eS.xdim = 200
eS.CS = crystalFrame('mmm', mineral='Kryptonite')
eS.ori0 = orientation.byEuler(np.array([0, np.pi / 4, 0]) * degree, eS.CS)

eS.makeMap()
eS.EBSDsim
# -

# ---

ipfKey = ipfColorKey(eS.CS)
plot(eS.EBSDsim, ipfKey.orientation2color(eS.EBSDsim.orientations))

# The summary reports 20,000 measurements of one indexed phase. The map has one colour
# because every measurement carries `ori0` exactly. This is the known reference field to
# which the following features are added.
#
# ## Adding orientation noise
#
# Noise is a random rotation about a random axis in the specimen frame. `noiseFun` selects
# a uniform or lognormal distribution of rotation angles. For `'logn'`, the sampled angles
# are rescaled so that the largest equals `noiseMax`; `noiseMax` is not a parameter of a
# lognormal distribution. The noise is drawn from `eS.rng`, a NumPy generator, and from a
# fresh one when it is `None`; a seeded generator makes the page reproducible.

# +
eS.noiseFun = 'logn'
eS.noiseMax = 2 * degree
eS.rng = np.random.default_rng(1)
eS.addNoise()

noiseAngle = angle(eS.ori0, eS.EBSDsim.orientations) / degree
print(f'Noise deviation: median {np.median(noiseAngle):.2f} degree; 90th percentile '
      f'{np.quantile(noiseAngle, 0.9):.2f} degree; maximum {noiseAngle.max():.2f} degree')

plot(eS.EBSDsim, noiseAngle)
mtexColorbar(title='deviation from ori0 in degree')
# -

# The deviation map shows many small rotations and a sparse long tail. Its printed maximum
# reaches the requested 2°. The median and 90th percentile describe the reproducible sample
# used here. This is orientation-level noise, not noise in simulated diffraction patterns.
#
# ## A known low-angle boundary
#
# A step feature is specified by a misorientation axis in the specimen frame and a total
# misorientation angle. `addFeature('singleStep')` rotates one stepped domain relative to the
# other.

eS.axS = yvector
eS.moriAngle = 3 * degree
eS.addFeature('singleStep')

# Feature methods modify the orientations already in `eS.EBSDsim`. They therefore
# accumulate: this 3° step is added to the noisy map rather than to a fresh uniform map.
#
# The boundary segments name their pixels by id, `ebsdId`, while `domainID` is a mask over
# the gridded map, so the mask is read through the ids of the map.

# +
newMtexFigure(layout=[1, 2])

plot(eS.EBSDsim, angle(eS.ori0, eS.EBSDsim.orientations) / degree)
mtexTitle('deviation from ori0')

# classify the known step as an inner boundary
grains = calcGrains(eS.EBSDsim, angle=[10 * degree, 1 * degree])
inDomain = np.zeros(eS.EBSDsim.size, dtype=bool)
inDomain[eS.EBSDsim.id.reshape(-1)] = eS.domainID.reshape(-1)
ids = grains.innerBoundary.ebsdId
isStepSegment = inDomain[ids[:, 0]] ^ inDomain[ids[:, 1]]
stepBoundary = grains.innerBoundary[np.flatnonzero(isStepSegment)]
boundaryAngle = stepBoundary.misorientation.angle() / degree
print(f'Inner boundary: {len(boundaryAngle)} segments; mean {boundaryAngle.mean():.2f} degree; '
      f'range {boundaryAngle.min():.2f} to {boundaryAngle.max():.2f} degree')

nextAxis()
plot(grains)
hold(True)
plot(stepBoundary, boundaryAngle, lineWidth=3)
hold(False)
setColorRange([2.75, 3.25])
mtexColorbar(title='boundary angle in degree')
# -

# The left map contains one domain scattered around `ori0` and another scattered around the
# 3° offset. The right map uses 10° as the grain-boundary threshold and 1° as the lower,
# subgrain threshold. The connected field is one grain. Its known step is stored in
# `innerBoundary`.
#
# The narrow colour range makes variation along the boundary visible. The printed mean
# stays close to the imposed 3°. Its range records the noise that was added before the
# step. The count is the number of individual grain boundary segments along the stepped
# feature. Other short entries in `innerBoundary` can be caused by neighbouring noise
# rotations that differ by more than 1°. The `domainID` mask isolates the segments that
# cross the feature whose true location is known.
#
# Synthetic noisy maps are particularly useful when the imposed boundary is close to the
# angular noise. See [Germain et al. (2014)](https://doi.org/10.1016/j.matchar.2014.10.007).
#
# ## Starting over with an orientation gradient
#
# `makeMap` discards the accumulated features and restores a uniform field. For a
# gradient, `gradDir` gives the direction of increase in the specimen frame. Here
# `moriAngle` is the angle increment per spatial grid step, not the total angle used by
# the step feature.

# +
eS.makeMap()
eS.axS = yvector
eS.gradDir = xvector
eS.moriAngle = 0.03 * degree
eS.addFeature('simpleGradient')

gradientAngle = angle(eS.ori0, eS.EBSDsim.orientations) / degree
print(f'Gradient deviation: {gradientAngle.min():.2f} to {gradientAngle.max():.2f} degree')

plot(eS.EBSDsim, gradientAngle)
mtexColorbar(title='deviation from ori0 in degree')
# -

# The colour changes smoothly from left to right because `gradDir` is `xvector`. With the
# default unit step, each column adds 0.03° about `yvector`. The first column is already one
# increment from `ori0` because the default map coordinates start at one.
#
# Further gradients can be superposed by changing `axS`, `gradDir`, or `moriAngle` and
# calling `addFeature('simpleGradient')` again. In contrast, `addFeature('circularSubgrain')`
# applies `moriAngle` as one total rotation. It changes the orientations inside a circular
# domain. Assign an existing map to `eS.EBSDsim` to start from measured or separately
# generated data.
