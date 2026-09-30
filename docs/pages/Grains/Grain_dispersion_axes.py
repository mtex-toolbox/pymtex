# # Using fibres to evaluate grain dispersion axes
#
# A grain is a phase-homogeneous, spatially connected part of an EBSD map. It contains
# many measured orientations rather than one exact orientation. When those orientations
# follow one dominant rotation, they trace a fibre in orientation space. The axis of that
# fibre is the grain's dispersion axis.
#
# This page develops the construction for one grain and then repeats it over a map. Read
# [Grain Reconstruction](GrainReconstruction.html) and
# [Selecting Grains](SelectingGrains.html) first if grains and grain ids are new to you.
# [Orientation Parameters](GrainOrientationParameters.html) introduces the grain mean,
# GROD, and grain orientation spread.
#
# A dispersion axis is an axis, not a directed vector. Its two ends are equivalent. Its
# crystal-frame representation can constrain a deformation mechanism, while its
# specimen-frame representation can constrain the kinematics. Neither interpretation
# follows from the fit alone.

import numpy as np

from mtex import *

# +
plottingConvention.default('y↑→x')
ebsd = mtexdata('forsterite')
grains = calcGrains(ebsd, minPixel=5)
# -

# ## Find a grain with coherent intragranular rotation
#
# First colour each forsterite measurement by its misorientation from the mean orientation
# of its own grain. The hue gives the misorientation axis in the specimen frame, and the
# saturation gives the angle. The reference orientation is supplied per measurement
# through `ck.oriRef`.

# +
ck = axisAngleColorKey(ebsd['f'].CS)
ck.oriRef = grains['id', ebsd['f'].grainId].meanOrientation
plot(ebsd['f'], ck.orientation2color(ebsd['f'].orientations))

hold(True)
plot(grains.boundary, lineWidth=2)
plot(grains[['En', 'Di']], alpha=0.7)
hold(False)
# -

# Pale grains have little rotation relative to their means. A grain with one strong hue
# has rotated mainly about one specimen axis. Bands of two or three hues instead warn that
# more than one axis may be active.
#
# The white outline selects one large, strongly coloured grain. The rest of the
# single-grain analysis asks whether its apparent axis survives a quantitative fit.

grainSelected = grains(5095, 7803)
hold(True)
plot(grainSelected.boundary, lineWidth=3, lineColor='w')
hold(False)

# ## See the dispersion in a pole figure
#
# Start with a grid of crystal directions. Every orientation in the selected grain maps
# every grid direction into the specimen frame. Each grid direction becomes a small cloud,
# and how far that cloud spreads measures how much the intragranular rotation moves the
# direction.

# +
s2G = equispacedS2Grid(resolution=15 * degree)
s2G = Miller(s2G, ebsd['f'].CS)
ori = ebsd[grainSelected].orientations
directions = ori.reshape(-1, 1) * s2G.reshape(1, -1)

plot(directions, MarkerSize=3, upper=True)
# -

# At this scale the clouds all look alike. What separates them is how far each one
# spreads, so colour every cloud by its mean angular deviation.

# +
poleDispersion = np.nanmean(angle(mean(directions, axis=0), directions, symmetry=False), axis=0)

plot(directions, np.tile(poleDispersion, (len(ori), 1)) / degree, MarkerSize=3)
mtexColorbar(title='average pole dispersion in degree')
# -

# The darkest blue cloud moves least. A rotation leaves its own axis fixed, so that
# crystal direction gives a grid-based estimate of the dispersion axis in the specimen
# frame.

# +
idMin = np.nanargmin(poleDispersion)
axisGrid = grainSelected.meanOrientation * s2G[idMin]
hold(True)
annotate(axisGrid)
annotate(axisGrid, plane=True, lineStyle='--', lineWidth=2)
hold(False)
# -

# The dashed great circle is normal to the black axis. The streaks follow that circle, as
# rotation about the axis requires. This construction is deliberately coarse: it can
# return only one of the 15 degree grid nodes. The spacing is a sampling resolution, not a
# 15 degree uncertainty bound.

# ## Fit a fibre without a direction grid
#
# Orientations produced by one continuous rotation lie on an
# [orientation fibre](OrientationFibre.html). `fibre.fit` fits that curve directly. The
# `'local'` algorithm is intended for a concentrated orientation cloud such as the
# orientations inside one grain.

fib = fibre.fit(ori, 'local')
fib

# The fitted fibre reports the same physical axis in two reference frames. `fib.h` is the
# crystal direction and `fib.r` is the specimen direction.
# [Reference Frame Alignment](EBSDReferenceFrame.html) explains why the specimen
# interpretation depends on a correctly calibrated frame.

fib.h

fib.r

# +
hold(True)
annotate(fib.r, MarkerFaceColor='r')
annotate(fib.r, plane=True, lineStyle='-.', lineWidth=2, lineColor='r')
hold(False)

gridFitDifference = angle(axisGrid, fib.r, antipodal=True) / degree
print(f'Grid axis to fitted axis: {float(np.asarray(gridFitDifference).reshape(-1)[0]):.1f} degree')
# -

# The printed separation is smaller than the grid spacing. The black and red estimates
# therefore agree at the resolution of the grid, while the fitted result is not restricted
# to a grid node.

# ## Decide whether the grain really has one axis
#
# A best fit always exists, even for a round cloud with no meaningful axis. The two
# additional outputs from `fibre.fit` diagnose that case. `lam` contains the four
# eigenvalues of the quaternion orientation tensor in ascending order. `fitAngle` is the
# mean angular distance from the fitted fibre.

# +
fib, lam, fitAngle = fibre.fit(ori, 'local', diagnostics=True)

lam
# -

fitAngle / degree

# The largest eigenvalue, $\lambda_4$, records overall concentration. The next,
# $\lambda_3$, records extension along the fibre, while $\lambda_2$ records scatter away
# from it. Their ratio is a useful screening statistic for this example.

fibreRatio = lam[2] / lam[1]
fibreRatio

# Here the printed ratio is well above one, so the cloud is more line-like than round. The
# cutoff of three used below is a heuristic for this page, not a universal MTEX threshold.
# A defensible cutoff also depends on EBSD angular precision, grain size, cleaning, and
# the deformation expected.
#
# Small-angle rotation axes are especially sensitive to EBSD error. Do not interpret a
# stable-looking axis from a nearly uniform grain without first checking the angle scale
# and the fit.

# ## Locate departures from the fitted fibre
#
# The distance from each orientation to the fibre shows where the one-axis model works and
# where it fails. The left panel shows the line in orientation space. The right panel
# returns the same residual to the map.

# +
distanceFromFibre = angle(fib, ori) / degree
plot(ori, distanceFromFibre, all=True)
ax = gcm().children[-1]
ax.set_xlim(0, 30)
ax.set_ylim(20, 70)
ax.set_zlim(80, 120)
hold(True)
plot(fib, lineWidth=2)
hold(False)

nextAxis()
plot(ebsd[grainSelected], distanceFromFibre)
mtexColorbar(title='distance from fibre in degree')

print(f'Distance from fibre: median {np.nanmedian(distanceFromFibre):.2f} degree, 90th percentile '
      f'{np.nanquantile(distanceFromFibre, 0.9):.2f} degree, maximum {np.nanmax(distanceFromFibre):.2f} degree')
# -

# Most points lie close to the fitted line. A band across the middle and the tail at the
# bottom lie farther away. Those coherent residuals are where a single dispersion axis is
# least adequate; they may record a second rotation or a subgrain boundary.

# ## Fit all sufficiently sampled grains
#
# A fit is made separately for every forsterite grain with more than 100 measurements.
# Small grains are excluded because their orientation clouds do not sample a fibre well
# enough for this comparison.

# +
grainsLarge = grains['fo']
grainsLarge = grainsLarge[grainsLarge.numPixel > 100]

axisCrystal = np.full((len(grainsLarge), 3), np.nan)
axisSpecimen = np.full((len(grainsLarge), 3), np.nan)
fibreRatio = np.full(len(grainsLarge), np.nan)

for k in range(len(grainsLarge)):

  fib, lam, _ = fibre.fit(ebsd[grainsLarge[k]].orientations, 'local', diagnostics=True)

  axisCrystal[k] = fib.h.data.reshape(3)
  axisSpecimen[k] = fib.r.data.reshape(3)
  fibreRatio[k] = lam[2] / lam[1]

axisCrystal = Miller(vector3d(axisCrystal), grainsLarge.CS)
axisSpecimen = vector3d(axisSpecimen, frame=ebsd.frame)

isFibre = fibreRatio > 3
print(f'Large forsterite grains: {len(grainsLarge)}; ratio above three: {np.count_nonzero(isFibre)} '
      f'({100 * np.count_nonzero(isFibre) / len(grainsLarge):.1f} percent)')
# -

# Only the grains that pass the stated screen enter the aggregate plots. The others are
# not proved to have no physical rotation axis; this orientation data simply do not
# support reporting one by this criterion.

# ## Compare axes in the specimen frame
#
# The dots show individual antipodal axes. The contours show a kernel density estimate
# with a 15 degree halfwidth. Its units are multiples of a uniform distribution, so one is
# the uniform reference.

# +
specimenDensity = calcDensity(axisSpecimen[isFibre], halfwidth=15 * degree, antipodal=True)
maxSpecimenDensity, peakSpecimenAxis = specimenDensity.max()

plot(axisSpecimen[isFibre], contourf=True, antipodal=True, upper=True, halfwidth=15 * degree)
hold(True)
plot(axisSpecimen[isFibre], antipodal=True, upper=True, MarkerSize=4)
hold(False)
mtexColorbar()

print(f'Maximum specimen-axis density: {maxSpecimenDensity:.2f} multiples of uniform')
peakSpecimenAxis
# -

# The preference is mild rather than uniform. The axes avoid the centre of the projection,
# which is the section normal, and gather near the rim towards X. They therefore tend to
# lie in the section and point roughly east-west. Such a specimen-frame cluster can
# indicate a common kinematic rotation axis, but calling it a vorticity axis requires the
# geological and deformation context.

# ## Compare axes in the crystal frame
#
# The crystal-frame plot asks a different question. A preferred crystal direction can
# constrain a common rotation mechanism across differently oriented grains.

# +
crystalDensity = calcDensity(axisCrystal[isFibre], halfwidth=15 * degree)
maxCrystalDensity, peakCrystalAxis = crystalDensity.max()

plot(axisCrystal[isFibre], contourf=True, antipodal=True, fundamentalRegion=True, halfwidth=15 * degree)
mtexColorbar()

print(f'Maximum crystal-axis density: {maxCrystalDensity:.2f} multiples of uniform')
peakCrystalAxis = Miller(peakCrystalAxis, grainsLarge.CS)
peakCrystalAxis
# -

# This maximum is stronger and lies near [010], with density falling towards [100]. The
# forsterite grains therefore bend about their own [010] direction more often than about
# other crystal directions in this map. A rotation axis can narrow the candidate slip
# mechanisms, but it does not identify a unique slip system without boundary geometry or
# dislocation evidence.
#
# Keep the two frames separate. A specimen-frame cluster supports a shared kinematic
# frame, while a crystal-frame cluster supports a shared crystallographic mechanism. This
# data set shows both preferences, with the crystal-frame preference the stronger of the
# two.

# ## References
#
# * Z. D. Michels, B. Tikoff, S. C. Kruckenberg and J. R. Davis, "Determining vorticity
#   axes from grain-scale dispersion of crystallographic orientations", *Geology* 43
#   (2015), 803-806, [doi:10.1130/G36868.1](https://doi.org/10.1130/G36868.1). This paper
#   develops crystallographic vorticity-axis analysis from grain-scale dispersion axes.
#
# * S. M. Reddy and C. Buchan, "Constraining kinematic rotation axes in high-strain zones:
#   a potential microstructural method", *Geological Society, London, Special
#   Publications* 243 (2005), 1-10,
#   [doi:10.1144/GSL.SP.2005.243.01.02](https://doi.org/10.1144/GSL.SP.2005.243.01.02).
#
# * D. J. Prior, "Problems in determining the misorientation axes, for small angular
#   misorientations, using electron backscatter diffraction in the SEM", *Journal of
#   Microscopy* 195 (1999), 217-225,
#   [doi:10.1046/j.1365-2818.1999.00572.x](https://doi.org/10.1046/j.1365-2818.1999.00572.x).
#   This is the measurement-precision caution behind the small-angle warning above.

# ## Next
#
# [Grain Reference Orientation Deviation](EBSDGROD.html) develops the angle and axis of
# every measurement relative to its grain reference.
# [Axis Distribution Function](AxisDistributionFunction.html) treats axes of
# grain-boundary misorientations and explains their symmetry and random references. The
# next page in this chapter, [Grain Neighbours](GrainNeighbours.html), changes from
# intragranular orientation structure to the network formed by adjacent grains.

# ## Technical details
#
# `ori * s2G` over a list of orientations and a list of directions is the outer product
# `ori.reshape(-1, 1) * s2G.reshape(1, -1)`, and the mean of the directions of every cloud
# is `mean(directions, axis=0)`. The eigenvalues come as `lam[0]` to `lam[3]`, so MATLAB's
# `lambda(3)/lambda(2)` is `lam[2] / lam[1]`. The measurements the closing absorbed carry
# no orientation, so the statistics skip them with NumPy's `nan` forms.
