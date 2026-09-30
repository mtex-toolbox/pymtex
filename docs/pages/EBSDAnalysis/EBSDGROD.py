# # Grain Reference Orientation Deviation (GROD)
#
# The grain reference orientation deviation asks how far each measurement has turned
# away from a chosen orientation for its own grain. The grain mean is the usual
# reference.
#
# [KAM](EBSDKAM.html) compares a point with its immediate neighbours and asks how
# sharply the lattice is bent. GROD instead compares the point with the grain reference.
# It reveals orientation variation across the whole grain. GROD records lattice rotation
# relative to that reference. It is not a strain measurement or a measure of accumulated
# deformation.
#
# Both the angle and the axis of the deviation carry information. The angle shows its
# magnitude, while the axis can distinguish rotation patterns that the angle alone hides.
#
# The example is a deformed ferrite specimen. The misorientation axes are very sensitive
# to noise, so the orientations are denoised first. See [Denoising](EBSDDenoising.html).
#
# The page assumes that the map has already been segmented into grains. See
# [Grain Reconstruction](GrainReconstruction.html) for that step. Misorientation angle
# and axis should also be familiar; see [Misorientations](MisorientationTheory.html).

import numpy as np

from mtex import *

# +
# use the rolling-direction frame carried by the ferrite specimen
plottingConvention.default('y↑→x')

ebsd = mtexdata('ferrite')

highAngle = 10 * degree
lowAngle = 1 * degree
grains = calcGrains(ebsd, angle=[highAngle, lowAngle], minPixel=3)

# smooth grain boundaries
grains = smoothBoundary(grains, 5)

# denoise the orientations
F = halfQuadraticFilter()
ebsd = smooth(ebsd, F, fill=True)

ipfKey = ipfColorKey(ebsd.CS)
plot(ebsd, ipfKey.orientation2color(ebsd.orientations))
hold(True)
plot(grains.boundary, lineWidth=2)
hold(False)
# -

# The colour changes within the larger grains are the orientation gradients that GROD
# will quantify. The overlaid outlines show which grain supplies the reference
# orientation for each measurement.

# ## The deviation angle
#
# [calcGROD](EBSD.calcGROD.html) computes the deviation as a misorientation, one per
# measurement. It needs the reconstructed grains as a second argument. It also needs
# `ebsd.grainId`, which `calcGrains` set.

# compute the grain reference orientation deviation
grod = ebsd.calcGROD(grains)

# The first plot maps its angle. Subgrain boundaries are drawn on top and faded in
# proportion to their own misorientation.

# +
# plot the misorientation angle of the GROD
plot(ebsd, grod.angle() / degree, micronbar=False)
mtexColorbar(title='misorientation angle in degree')
mtexColorMap('LaboTeX')

# overlay grain and sub-grain boundaries
hold(True)
plot(grains.boundary, lineWidth=1.5)
plot(grains.innerBoundary, edgeAlpha=np.minimum(grains.innerBoundary.misorientation.angle() / (5 * degree), 1))
hold(False)

grodAngle = angle(grod) / degree
print(f'GROD angle: median {np.nanmedian(grodAngle):.2f} degree; maximum {np.nanmax(grodAngle):.1f} degree')
# -

# Half the measurements are within about 1.6° of their grain mean and the worst reach
# about 19°. The large values are not scattered. They fill extended parts of grains and
# often end at the inner boundaries.
#
# Those boundaries were detected from local neighbour misorientations, not from GROD. A
# [subgrain boundary](SubGrainBoundaries.html) can coincide with a jump in GROD, but a
# GROD jump is not its definition.

# ## Grain orientation spread
#
# Averaging the deviation angle over each grain gives the grain orientation spread.
# [grainMean](EBSD.grainMean.html) performs that averaging.
#
# Although `calcGrains` stored `grains.GOS`, that value belongs to the raw orientations.
# The calculation below deliberately uses the denoised data.

# +
GOS = grainMean(ebsd, grod.angle(), grains)

plot(grains, GOS / degree)
mtexColorbar(title='GOS in degree')

print(f'GOS for {len(grains)} grains: median {np.nanmedian(GOS / degree):.2f} degree; maximum {np.nanmax(GOS / degree):.1f} degree')
# -

# Of the 377 grains the median spread is below a degree and the largest 6.6°. The map is
# not uniform. Some grains have much more internal orientation variation than their
# neighbours, producing an order-of-magnitude spread.
#
# GOS depends on segmentation, denoising, reference orientation, and grain size.
# Comparisons require the same choices.

# ## The misorientation axis in crystal coordinates
#
# Every deviation also has an axis, and it matters in which reference frame that axis is
# expressed. The crystal frame is attached to the lattice of the phase. In this frame the
# axis is an $(hk\ell)$ crystal direction, obtained with [axis](orientation.axis.html).

# +
axCrystal = grod.axis()

h = plot(axCrystal, contourf=True, fundamentalRegion=True, antipodal=True)
mtexColorbar(title='mrd')

crystalDensityRange = h[0].get_clim()
print(f'Crystal-frame axis density: {crystalDensityRange[0]:.2f} to {crystalDensityRange[1]:.2f} mrd')
# -

# The range is close to uniform, about 0.8 to 1.2 times. In crystal coordinates the axes
# are therefore close to evenly distributed, with a slight preference for $[101]$. A
# colour key shows where those directions occur in the map.

# +
colorKey = HSVDirectionKey(ebsd.CS, antipodal=True)

plot(colorKey)
# -

# Directions related by ferrite crystal symmetry have the same colour in this key. The
# deviation angle serves as transparency in the map below, so that points which have
# barely turned fade to white.
#
# A small-angle axis is poorly constrained by conventional EBSD. Whitening it prevents
# uncertain axes from dominating the picture, but it is a visualization choice rather
# than an uncertainty correction.

# +
# compute the color from the misorientation axis
color = colorKey.direction2color(axCrystal)

# and set the transparency from the misorientation angle
alpha = np.minimum(grod.angle() / degree / 7.5, 1)

# plot the data
plot(ebsd, color, micronbar=False, faceAlpha=alpha)

hold(True)
plot(grains.boundary, lineWidth=2)
plot(grains.innerBoundary, edgeAlpha=np.minimum(grains.innerBoundary.misorientation.angle() / (5 * degree), 1))
hold(False)
# -

# Whole regions of a grain share one colour, that is one misorientation axis. This is
# what makes the crystal frame worth looking at.
#
# A low-angle boundary formed by a single slip system has its axis fixed by the geometry
# of that system. A pure tilt wall turns about an axis in the slip plane and
# perpendicular to the Burgers vector. A pure twist wall turns about the slip plane
# normal. This rule applies directly to the misorientation across a boundary. GROD
# compares a point with its reference.
#
# A coherent GROD-axis cluster can nevertheless constrain candidate slip systems when
# combined with the angle map and independent slip geometry. V. Tong, E. Wielewski and
# B. Britton use long-range rotations in
# [Characterisation of Slip and Twinning in High-Rate-Deformed Zirconium with EBSD](https://doi.org/10.48550/arXiv.1803.00236).
#
# A cluster alone does not identify a unique active system. The
# [slip system](SlipSystems.html) and [dislocation system](DislocationSystems.html)
# chapters describe how to set up candidates in MTEX. See
# [Tilt and Twist Boundaries](TiltAndTwistBoundaries.html) for the boundary geometry
# itself.

# ## The misorientation axis in specimen coordinates
#
# The specimen frame is the reference frame in which the map and sample are expressed.
# The same axis in this frame is the crystal axis carried over by the orientation of the
# measurement.
#
# The option `symmetry=False` is essential. The axis must be the representative that belongs
# to this orientation, not a symmetric equivalent.

# +
axSpecimen = ebsd.orientations * grod.axis(symmetry=False)

h = plot(axSpecimen, contourf=True, halfwidth=2.5 * degree)
mtexColorbar(title='distribution of misorientation axes in mrd')

specimenDensityRange = h[0].get_clim()
print(f'Specimen-frame axis density: {specimenDensityRange[0]:.2f} to {specimenDensityRange[1]:.2f} mrd')
# -

# This distribution runs from about 0.2 to nearly 7 times uniform. The same axes that
# were spread evenly over the crystal are strongly clustered in the specimen frame.
# Loading is defined in this frame, which makes the clustering useful. Texture and a few
# highly deformed grains can also produce maxima; the map below distinguishes those
# possibilities.
#
# In the specimen frame the axes have no crystal symmetry and are directional rather
# than antipodal. The colour key therefore uses the whole sphere.

# +
colorKey = HSVDirectionKey()

plot(colorKey)
# -

# Opposite specimen directions have different colours in this key. The spatial plot
# follows the same construction as the crystal-frame map.

# +
# compute color and transparency
omega = np.minimum(grod.angle() / degree / 7.5, 1)
color = colorKey.direction2color(axSpecimen)

# plot the data
plot(ebsd, color, micronbar=False, faceAlpha=omega)

hold(True)
plot(grains.boundary, lineWidth=2)
plot(grains.innerBoundary, edgeAlpha=np.minimum(grains.innerBoundary.misorientation.angle() / (5 * degree), 1))
hold(False)
# -

# ## Choosing another reference orientation
#
# The grain mean is convenient, but it removes a rigid rotation shared by the whole
# grain. [calcGROD](EBSD.calcGROD.html) accepts a third argument: one reference rotation
# for every grain ID, or one scalar for all grains.
#
# A registered map of the undeformed specimen can supply one initial orientation per
# grain. GROD then retains whole-grain rotation relative to that state. Such a comparison
# first requires spatial registration; see
# [TrueEBSD Distortion Correction](EBSDTrueEbsd.html). Changing the reference changes
# the angle, axis, and GOS. State that choice in every reported method.

# ## The definition
#
# For the orientation $o_{i,j}$ at position $(i,j)$ and the chosen reference orientation
# $r_g$ of its grain,
#
# $$\mathrm{GROD}_{i,j} = \mathbf S_{i,j} \cdot \mathrm{inv}(r_g) \cdot o_{i,j}$$
#
# where the symmetry element $\mathbf S_{i,j}$ is the one that makes the misorientation
# angle as small as possible. With no third input to `calcGROD`, $r_g$ is the grain mean
# orientation.

# ## Further reading
#
# * S. I. Wright, M. M. Nowell and D. P. Field,
#   [A Review of Strain Analysis Using Electron Backscatter Diffraction](https://doi.org/10.1017/S1431927611000055),
#   _Microscopy and Microanalysis_ 17 (2011), 316--329. It reviews what
#   orientation-gradient measures can and cannot say about deformation.
# * D. J. Prior, [Problems in Determining the Misorientation Axes for Small Angular Misorientations Using EBSD](https://doi.org/10.1046/j.1365-2818.1999.00572.x),
#   _Journal of Microscopy_ 195 (1999), 217--225. It quantifies the rapid loss of axis
#   precision at small angles.
# * V. Tong, E. Wielewski and B. Britton,
#   [Characterisation of Slip and Twinning in High-Rate-Deformed Zirconium with EBSD](https://doi.org/10.48550/arXiv.1803.00236)
#   (2018). It demonstrates the slip-system interpretation discussed above.

# ## Next
#
# Continue with [Line Profiles](EBSDProfile.html) to turn a colour gradient into a curve
# along a chosen path. The broader family of grain-scale statistics is described in
# [Grain Orientation Parameters](GrainOrientationParameters.html).
