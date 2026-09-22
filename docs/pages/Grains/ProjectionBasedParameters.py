# # Projection Based Shape Parameters
#
# Hold a grain up against the light and measure the width of its shadow. Turn it a little
# and measure again. The family of parameters on this page comes from that operation: the
# width of a grain as a function of direction.
#
# | quantity | meaning |
# |---|---|
# | `caliper` | caliper or Feret diameter in the map's length unit |
# | `diameter` | longest caliper in the map's length unit |
#
# Unlike a fitted [ellipse](EllipseBasedParameters.html), which describes a grain by four
# numbers, the projection function keeps one value for every direction. It can therefore
# describe fabrics that one ellipse cannot.
#
# This page assumes that the grains have been [reconstructed](GrainReconstruction.html)
# and that selecting complete grains is familiar from
# [Selecting Grains](SelectingGrains.html). [Grain Smoothing](GrainSmoothing.html)
# explains why a boundary reconstructed from a pixel grid should be smoothed before its
# directions are measured. All measurements below describe two-dimensional grain
# sections, not the full three-dimensional grains.

import matplotlib.pyplot as plt
import numpy as np

from mtex import (S1FunHarmonic, angle, calcGrains, characteristicShape, degree, hold, mtexColorbar, mtexdata, norm, paror, plot,
                  plottingConvention, quiver, smoothBoundary, surfor, vector3d)

# +
# load sample EBSD data in its specimen plotting frame
plottingConvention.default('y↑→x')
ebsd = mtexdata('forsterite')

# reconstruct grains, discard boundary grains, and smooth their outlines
grains = calcGrains(ebsd, angle=5 * degree, minPixel=5)
grains = grains[~grains.isBoundary]
grains = smoothBoundary(grains['indexed'], 10, moveTriplePoints=True)

# plot all grains and highlight one complete grain
plot(grains)

ind = int(np.argmax(grains.diameter))
hold(True)
plot(grains[ind].boundary, lineWidth=5, lineColor='blue')
hold(False)
# -

# The blue outline identifies the grain used in the first measurements. Removing grains
# at the map edge prevents a truncated section from being mistaken for an unusually narrow
# or elongated grain. Smoothing suppresses the horizontal and vertical directions imposed
# by the measurement grid.

# ## The caliper
#
# The `diameter` is the longest distance between any two vertices of the outline. For the
# selected grain it is:

grainDiameter = grains[ind].diameter
grainDiameter

# A `caliper`, or Feret diameter, is the span of the grain after its vertices are
# projected onto a supplied direction. Traced over all directions, the caliper becomes a
# function. The direction is an axis, so values 180 degrees apart are identical.

# +
omega = np.linspace(0, 180, 100)
dir = vector3d.byPolar(90 * degree, omega * degree)
plt.figure()
plt.plot(omega, grains[ind].caliper(dir).reshape(-1), linewidth=2)
plt.ylabel('length in µm')
plt.xlabel('projection direction in degrees')
plt.xlim(0, 180)
# -

# The curve of this grain runs between its shortest and its longest width. Its one broad
# maximum and one minimum over 180 degrees are characteristic of an elongated, roughly
# convex outline. The extrema are also available as vectors whose `norm` is the caliper
# length. The longest one is the diameter again.

caliperExtrema = np.array([norm(grains[ind].caliper('shortest')), norm(grains[ind].caliper('longest'))]).reshape(-1)
caliperExtrema

# +
plot(grains[ind], micronbar=False, legend=False)

hold(True)
quiver(grains[ind], grains[ind].caliper('longest'), noScaling=True)
quiver(grains[ind], grains[ind].caliper('shortest'), noScaling=True)
hold(False)
# -

# The long arrow joins the two most distant vertices. The short arrow shows the minimum
# width and is not generally perpendicular to the long one.

# ## Directional elongation
#
# The relative difference between the two extrema is a simple measure of how far a grain
# is from having the same width in every direction. It is zero for a constant-width
# outline and approaches one as the minimum width becomes small. It measures directional
# elongation, not lobes or boundary roughness.

# +
cMin = grains.caliper('shortest')
cMax = grains.caliper('longest')
caliperAnisotropy = (norm(cMax) - norm(cMin)) / norm(cMax)

plot(grains, caliperAnisotropy, micronbar=False)
mtexColorbar(title='(c_max - c_min) / c_max')
# -

# Grains with similar maximum and minimum widths plot near zero. High values mark
# sections whose longest span is much greater than their minimum width.

# ## Three ways to say which way a grain points
#
# The longest caliper and the long axis of the fitted ellipse usually agree, but not
# always. For rectangular particles they disagree badly: the longest caliper of a
# rectangle follows a diagonal. A strong alignment of rectangles can therefore produce a
# bimodal distribution of diagonals that no grain actually points along. The direction
# normal to the *shortest* caliper, `'shortestPerp'`, does not have this problem.

# +
# load two artificial grains, a rectangle and a turned one
testgrains = mtexdata('testgrains')
testgrains = smoothBoundary(testgrains['id', [6, 8]], 10)

# compute the longest caliper and the direction normal to the shortest one
cMax = testgrains.caliper('longest')
cMinPerp = testgrains.caliper('shortestPerp')

# compare the three candidate long directions
plot(testgrains, micronbar=False, lineWidth=2)
hold(True)
quiver(testgrains, cMax, DisplayName='longest caliper', lineWidth=3)
quiver(testgrains, testgrains.longAxis(), DisplayName='long axis', lineWidth=3)
quiver(testgrains, cMinPerp, DisplayName='perp. to shortest', lineWidth=3)
hold(False)
plt.legend(loc='center right')
# -

# On a rectangle the longest caliper follows a diagonal while the long axis and the
# perpendicular to the shortest caliper both follow the long side, so those two arrows
# lie on top of each other and only the longest caliper stands apart.
#
# Which one to use is a decision about the material: use the long axis for roughly
# elliptical grains and the perpendicular to the shortest caliper for grains whose flat
# faces define their alignment.

# ## From grains to a particle fabric: PAROR
#
# A shape preferred orientation, or SPO, is an alignment of grains as bodies. Projection
# lengths can be added over a population without first assigning one direction to every
# grain and without assuming that each grain is an ellipse. This construction is the
# PAROR method introduced by Panozzo.
#
# `caliper` accepts a list of directions and returns one projection length per grain and
# direction. The individual functions and their average can therefore be drawn directly.

# +
omega = np.linspace(0, 360 * degree, 361)
dir = vector3d.byPolar(90 * degree, omega)
c = grains['Fo'].caliper(dir)

fig = plt.figure(figsize=(9, 4.5))
ax = fig.add_subplot(1, 2, 1, projection='polar')
ax.plot(omega, c.T, linewidth=2, color=(0, 0.25, 0.5, 0.25))
ax.set_title('Forsterite')

# draw the average five times larger than its true scale
ax.plot(omega, 5 * np.mean(c, axis=0), linewidth=3, color='k')

ax = fig.add_subplot(1, 2, 2, projection='polar')
c = grains['Enstatite'].caliper(dir)
ax.plot(omega, c.T, linewidth=2, color=(0, 0.25, 0.5, 0.25))
ax.set_title('Enstatite')

# draw the average five times larger than its true scale
ax.plot(omega, 5 * np.mean(c, axis=0), linewidth=3, color='k')
# -

# Each faint line is one grain and the black line is their average, drawn five times too
# large so that it remains visible. Individual grains scatter over many shapes and
# directions. The smooth, slightly flattened average reveals the particle fabric.
#
# `paror` sums the same projection lengths over the grains and divides the curve by its
# maximum. A larger grain contributes a wider projection. Since a sum and a mean differ
# only by the number of grains, this is also the same normalized average drawn above. The
# normalization removes the total scale, so the result compares fabric direction and
# anisotropy, not grain count, phase amount, or mean grain size.
#
# The supplied angle is the axis onto which the vertices are projected. Zero degrees is
# the specimen x axis, angles increase counterclockwise, and the result repeats after 180
# degrees. Sampling through 360 degrees merely closes the polar curve. This is the direct
# form of the traditional description in which a right-handed coordinate system is
# rotated around the particle and the particle is projected onto its x axis.

# +
cumplF = paror(grains['fo'], omega)
cumplE = paror(grains['en'], omega)

fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(2, 2, 1)
ax.plot(omega / degree, cumplF, linewidth=3, color='k')
ax.set_xlim(0, 180)
ax.set_title('PAROR forsterite')
ax = fig.add_subplot(2, 2, 2, projection='polar')
ax.plot(omega, cumplF, linewidth=3, color='k')
ax = fig.add_subplot(2, 2, 3)
ax.plot(omega / degree, cumplE, linewidth=3, color='k')
ax.set_xlim(0, 180)
ax.set_title('PAROR enstatite')
ax = fig.add_subplot(2, 2, 4, projection='polar')
ax.plot(omega, cumplE, linewidth=3, color='k')
# -

# The Cartesian panels make the extrema easy to read over the independent interval from 0
# to 180 degrees. The polar panels turn the same values into a visual fabric axis. The
# curves are normalized, so only their shape and direction should be compared.
#
# The minimum plays the role of an average axial ratio $b/a$ for the whole fabric. It is 1
# for an isotropic projection function and becomes smaller as the fabric becomes more
# anisotropic.

parorMinimum = np.array([np.min(cumplF), np.min(cumplE)])
parorMinimum

# The forsterite and enstatite minima are 0.7118 and 0.7214, a difference of about 0.01.
# Both phases therefore have a moderate projection anisotropy of similar strength.
#
# The maximum marks the preferred direction of the longest projection. The normal to the
# minimum marks the preferred direction inferred from the shortest projection. For an
# orthorhombic fabric these axes coincide in the section, so their difference measures
# departure from orthogonal fabric axes.
#
# Harmonic interpolation estimates the extrema between the sampled angles.

# +
sF_Fo = S1FunHarmonic.interpolate(omega, cumplF)
_, maxposfo = sF_Fo.max()
_, minposfo = sF_Fo.min()

harmonicDirectionsFo = np.array([np.mod(maxposfo, np.pi), np.mod(minposfo - np.pi / 2, np.pi)]) / degree
harmonicDirectionsFo
# -

# +
sF_En = S1FunHarmonic.interpolate(omega, cumplE)
_, maxposen = sF_En.max()
_, minposen = sF_En.min()

harmonicDirectionsEn = np.array([np.mod(maxposen, np.pi), np.mod(minposen - np.pi / 2, np.pi)]) / degree
harmonicDirectionsEn
# -

# For forsterite the two fitted directions are 71.5 and 74.5 degrees; for enstatite they
# are 87.5 and 88.8 degrees. The two directions of each phase lie within a few degrees, so
# both fabrics are close to orthorhombic in this section. The 16-degree difference between
# the preferred longest projections of the two phases is the more interesting comparison.
#
# The same directions can be read directly from the sampled values. The sampling is one
# degree, so these estimates agree with the harmonic result to within that resolution.

# +
idMax = np.argmax(cumplF)
idMin = np.argmin(cumplF)
sampledDirectionsFo = np.array([np.mod(omega[idMax] / degree, 180), np.mod(omega[idMin] / degree - 90, 180)])
sampledDirectionsFo
# -

# +
idMax = np.argmax(cumplE)
idMin = np.argmin(cumplE)
sampledDirectionsEn = np.array([np.mod(omega[idMax] / degree, 180), np.mod(omega[idMin] / degree - 90, 180)])
sampledDirectionsEn
# -

# ## From boundaries to a surface fabric: SURFOR
#
# `surfor` applies the same projection idea to a list of boundary segments instead of to
# whole grains. It weights every segment by its length and normalizes the summed curve to
# one. Because it needs no closed outline, it also works for selections that are not
# grains: subgrain boundaries, twin boundaries, or contacts between selected phases.
#
# Segment directions inherited from a square pixel grid would dominate this calculation.
# The grains at the start of the page were therefore smoothed before their boundaries
# were selected. Use the same segmentation, spatial resolution, and smoothing procedure
# when comparing maps.

# +
pairs = [('Fo', 'Fo'), ('Fo', 'En'), ('Fo', 'Di'), ('En', 'Di')]
pairName = ['Fo-Fo', 'Fo-En', 'Fo-Di', 'En-Di']
surforCurves = np.zeros((len(pairs), len(omega)))

fig = plt.figure(figsize=(5, 5.5))
ax = fig.add_subplot(1, 1, 1, projection='polar')
for i, pair in enumerate(pairs):
  gB = grains.boundary[pair]
  surforCurves[i] = surfor(gB, omega)
  ax.plot(omega, surforCurves[i], linewidth=2, label=pairName[i])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncols=4)

surforMinimum = np.min(surforCurves, axis=1)
surforLongDirection = np.mod(omega[np.argmax(surforCurves, axis=1)] / degree, 180)
print(f"{'boundaryPair':>14}{'minimumToMaximum':>18}{'longDirectionDegree':>21}")
for name, mn, d in zip(pairName, surforMinimum, surforLongDirection):
  print(f"{name:>14}{mn:>18.5f}{d:>21.0f}")
# -

# The four curves differ in both shape and direction. The contact between the two
# pyroxenes, enstatite and diopside, is the roundest, with a minimum at 0.87 of its
# maximum. It is the least anisotropic boundary population in this rock. The most
# anisotropic is the forsterite-diopside contact at 0.68, not the forsterite-forsterite
# boundary at 0.76. What sets the forsterite-forsterite boundary apart is its direction
# rather than its strength: its long direction is 63 degrees, whereas the other three lie
# between 75 and 85 degrees.

# ## Characteristic shape
#
# `characteristicShape` takes every selected boundary segment together with its opposite,
# sorts those vectors by direction, and lays them end to end. They close into a
# fabric-equivalent polygon without requiring closed grain outlines. It is often read as
# an average grain outline, but it is not an observed grain or an arithmetic average
# grain. It represents the selected boundary-length distribution.

# +
shapeF = characteristicShape(grains.boundary['Fo', 'Fo'])
plot(shapeF, normalize=True, lineWidth=2, plain=True, DisplayName='Fo-Fo')
hold(True)
shapeE = characteristicShape(grains.boundary['En', 'En'])
plot(shapeE, normalize=True, lineWidth=2, plain=True, DisplayName='En-En')
shapeEF = characteristicShape(grains.boundary['En', 'Fo'])
plot(shapeEF, normalize=True, lineWidth=2, plain=True, DisplayName='En-Fo')
hold(False)

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02), ncols=3)
# -

# The three polygons are normalized to equal area, so compare their direction and shape
# rather than their absolute size. The result is a `shape2d`, which accepts the same shape
# commands as a grain.
#
# A shape with a mirror line normally has perpendicular longest and shortest fabric axes.
# The angle between those calipers is then 90 degrees and departs from 90 degrees for a
# skewed shape. More generally, the angle tests whether the two extrema are perpendicular;
# it does not by itself prove mirror symmetry. It therefore supplies an asymmetry measure
# that a single axial ratio cannot.

characteristicCaliperAngle = np.array([angle(shapeF.caliper('longest'), shapeF.caliper('shortest')),
                                       angle(shapeE.caliper('longest'), shapeE.caliper('shortest')),
                                       angle(shapeEF.caliper('longest'), shapeEF.caliper('shortest'))]).reshape(-1) / degree
characteristicCaliperAngle

# The two single-phase shapes both give 79.3 degrees. Each is skewed by about 10.7
# degrees, and the two skews run in opposite senses: the outlines are close to mirror
# images of one another. The mixed forsterite-enstatite shape is nearly symmetric at 88.5
# degrees.
#
# These values describe this map. Whether a difference between specimens is larger than
# sampling variation requires replicate maps, subdivision, or a justified resampling
# procedure; normalization alone supplies no uncertainty.

# ## Further reading
#
# * R. Panozzo,
#   [Two-dimensional analysis of shape-fabric using projections of digitized lines in a plane](https://doi.org/10.1016/0040-1951(83)90073-2),
#   *Tectonophysics* 95 (1983), 279-294, introduces the PAROR construction.
#
# * R. Panozzo,
#   [Two-dimensional strain from the orientation of lines in a plane](https://doi.org/10.1016/0191-8141(84)90098-1),
#   *Journal of Structural Geology* 6 (1984), 215-221, develops the SURFOR interpretation.
#
# * R. Heilbronner and S. Barrett,
#   [Image Analysis in Earth Sciences: Microstructures and Textures of Earth Materials](https://doi.org/10.1007/978-3-642-10343-8),
#   Springer, 2014. The chapters on particle and surface fabrics place both methods in a
#   broader image-analysis workflow.
#
# * [ISO 13322-1:2014](https://www.iso.org/standard/51257.html), *Particle size analysis
#   - Image analysis methods - Part 1: Static image analysis methods*, standardizes
#   static-image measurements including maximum and minimum Feret dimensions.
#
# * [ISO 9276-6:2008](https://www.iso.org/standard/39389.html), *Representation of
#   results of particle size analysis - Part 6: Descriptive and quantitative
#   representation of particle shape and morphology*, gives shape terminology and
#   emphasizes that most image-based definitions are two-dimensional.
#
# The opening shadow analogy also echoes Edwin A. Abbott's
# [*Flatland*](https://www.gutenberg.org/ebooks/201) (1884).

# ## Next
#
# [Ellipse Based Shape Parameters](EllipseBasedParameters.html) compares these projection
# directions with fitted long axes. Continue to
# [Grain Orientation Parameters](GrainOrientationParameters.html) to compare shape fabric
# with orientation variation inside the grains, or to
# [Grain Boundaries](GrainBoundaries.html) when the boundary segments and their
# crystallographic relations are the subject.

# ## Technical details
#
# MATLAB's grain 515 is the grain of the largest diameter here, since the order of the
# reconstructed grains differs, so the first numbers describe another grain. The
# artificial rectangles come from this port's own `testgrains`. MATLAB's `polarplot` and
# `subplot` are matplotlib's polar axes, its `table` a printed one.
