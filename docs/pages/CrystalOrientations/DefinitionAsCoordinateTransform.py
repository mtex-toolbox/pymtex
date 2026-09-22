# # Crystal Orientation as Coordinate Transformation
#
# An orientation in MTEX maps crystal coordinates to specimen coordinates. It takes a
# direction or tensor expressed in the crystal frame and returns the same object expressed
# in the specimen frame.
#
# A *reference frame* is the coordinate system in which data are expressed. The *crystal
# frame* is the Cartesian frame fixed to a phase's lattice, while the *specimen frame*
# describes the sample in a measurement, rolling, or geological frame.
#
# This page assumes the Miller indices introduced in
# [Crystal Directions](CrystalDirections.html) and the active rotations from
# [Rotation Operations](RotationOperations.html). Everything below follows from the
# direction of the coordinate map, including which side a rotation acts on and what
# happens when the specimen is turned.

from mtex import (Miller, angle, arrow3d, cat, crystalFrame, crystalShape, degree, hold, inv, normalize, orientation,
                  plot, plottingConvention, rotation, round, stiffnessTensor, vector3d)

plottingConvention.default('y↑→x')

# ## The Two Ingredients
#
# An orientation combines a [rotation](rotation.rotation.html) with the symmetry, lattice
# metric, and crystal frame stored by a [crystalFrame](crystalFrame.crystalFrame.html).

rot = rotation.byEuler(10 * degree, 20 * degree, 30 * degree, 'Bunge')

cs = crystalFrame.load('Al-Aluminum.cif')
cs

# The summary identifies aluminium, its point group, lattice parameters, and crystal-frame
# alignment. Combining `rot` and `cs` gives an orientation.

ori = orientation(rot, cs)
ori

# The arrow in the summary reads from the crystal frame on the left to the specimen frame
# on the right. An orientation is also a rotation, so every
# [rotation operation](RotationOperations.html) applies to it.

# ## From Crystal Coordinates to Specimen Coordinates
#
# Take the crystal direction $[100]$.

h = Miller(1, 0, 0, cs, 'uvw')

# In a grain with orientation `ori`, that direction has the following Cartesian components
# in the specimen frame.

r = ori * h
r

# The picture shows the same map. The translucent cube is the crystal where `ori` places
# it. The black arrows are the specimen axes X, Y and Z, and the red arrow is the crystal
# direction `h` expressed in specimen coordinates. The direction is fixed in the lattice;
# what the orientation supplies is where the lattice is pointing.

cS = crystalShape.cube(cs)

plot(ori * cS, faceAlpha=0.35, faceColor=[0.6, 0.75, 0.9])
hold(True)
arrow3d(0.75 * normalize(r), faceColor='red')
arrow3d(0.75 * cat(vector3d.X, vector3d.Y, vector3d.Z), faceColor='black')
hold(False)

# ## Other Crystal Objects Transform the Same Way
#
# The same multiplication applies to a stiffness tensor. This example starts with tensor
# components in the crystal frame.

C = stiffnessTensor([[2, 1, 1, 0, 0, 0],
                     [1, 2, 1, 0, 0, 0],
                     [1, 1, 2, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 1]], cs)
C

# After the coordinate transform, the summary names the specimen frame and displays the
# transformed components.

Cspecimen = ori * C
Cspecimen

# Everything defined in the crystal frame travels in the same direction:
#
# * [crystal directions](Miller.html)
# * [tensors](tensor.tensor.html)
# * [slip systems](slipSystem.slipSystem.html)
# * [twinning systems](twinningSystem.twinningSystem.html)
# * [dislocation systems](dislocationSystem.dislocationSystem.html)
# * [crystal shapes](crystalShape.crystalShape.html)

# ## And Back Again
#
# The inverse orientation maps specimen coordinates to crystal coordinates. Applying it to
# `r` therefore returns the direction we started from.

hBack = inv(ori) * r
hBack

# The displayed coefficients do not resemble $[100]$ yet. A `Miller` made from a specimen
# direction displays as $(hkl)$ unless told otherwise, while the original $[100]$
# direction has the aluminium lattice-vector length of 4.04958 Angstrom. Selecting
# lattice-direction notation and rounding recovers the original indices.

hBack.dispStyle = 'uvw'
hRounded = round(hBack)
hRounded

# Much of the literature defines an orientation in the opposite direction, from specimen
# to crystal coordinates. That is what MTEX calls `inv(ori)`. Both conventions are in use,
# and reading Euler angles with the wrong one inverts every orientation in the data. See
# [MTEX vs. Bunge Convention](MTEXvsBungeConvention.html) for the practical consequences.

# ## Turning the Specimen
#
# Putting the sample on the stage in another position actively turns the crystal relative
# to the fixed measurement frame. A rotation expressed in specimen coordinates therefore
# multiplies every orientation from the left.

rotSpecimen = rotation.byAxisAngle(vector3d.X, 60 * degree)
oriNew = rotSpecimen * ori

# Every crystal direction moves with the specimen. Going through the new orientation and
# turning the old specimen direction must agree.

leftConsistency = angle(oriNew * h, rotSpecimen * r) / degree
leftConsistency

# The residual is numerically zero. The same rotation written on the right is interpreted
# in crystal coordinates: it turns the direction inside the lattice before `ori` maps that
# direction into the specimen frame.

rightDifference = angle(ori * (rotSpecimen * h), oriNew * h) / degree
rightDifference

# The nonzero result confirms that left and right multiplication describe different
# operations. The rotation on the right acts first, just as in ordinary matrix
# multiplication.

# ## Crystal Symmetry Also Acts from the Right
#
# Right multiplication has a second, symmetry-aware meaning. A point-group operation
# changes the crystal-frame representative but not the physical crystal setting.
# [symmetrise](orientation.symmetrise.html) lists those equivalent descriptions.

equivalentCount = len(ori.symmetrise())
equivalentCount

# The count is 24, the number of proper operations in aluminium's `m-3m` point group; only
# those are rigid rotations and only those give another setting of the same crystal. The
# 24 improper operations remain symmetries of the lattice and are relevant when opposite
# plane normals are treated as equivalent.
#
# A symmetry-aware orientation comparison regards all 24 descriptions as equivalent, so
# their largest angular difference from `ori` vanishes.

symmetryResidual = max(angle(ori.symmetrise(), ori)) / degree
symmetryResidual

# ## Rotating Is Not Changing Frame
#
# The stage rotation above moves the physical object in a fixed frame. A *frame change*
# instead re-expresses the same physical object in another reference frame and leaves the
# object untouched. Use
# [transformReferenceFrame](orientation.transformReferenceFrame.html) when crystal data
# use a different Cartesian crystal frame. Orientations also depend on how the Cartesian
# crystal frame $\vec x$, $\vec y$, $\vec z$ is inscribed into the crystal axes $\vec a$,
# $\vec b$, $\vec c$.
#
# A *plotting convention* only states how a reference frame is laid out on screen.
# Changing it does not rotate the specimen or re-express the data.
# [The Crystal Reference System](CrystalReferenceSystem.html) develops both distinctions
# for crystal axes.

# ## The Maths Behind the Multiplication Order
#
# Let $\mathbf{G}$ be the matrix of `ori`, and let $\mathbf{h}$ and $\mathbf{r}$ contain
# the crystal and specimen components of one direction. Then
#
# $$ \mathbf{r} = \mathbf{G}\mathbf{h}, \qquad
#    \mathbf{h} = \mathbf{G}^{\mathrm{T}}\mathbf{r}. $$
#
# The transpose appears because a rotation matrix is orthogonal, so
# $\mathbf{G}^{-1}=\mathbf{G}^{\mathrm{T}}$. A specimen rotation $\mathbf{Q}$ gives
# $\mathbf{QG}$, while a crystal-frame rotation $\mathbf{P}$ gives $\mathbf{GP}$. This is
# the matrix form of the two multiplication examples above.
#
# The same rule determines the order of a misorientation product. See
# [Misorientations](MisorientationTheory.html).

# ## References
#
# * H.-J. Bunge, [Texture Analysis in Materials Science: Mathematical Methods](https://doi.org/10.1016/C2013-0-11769-2),
#   Butterworths, English ed., 1982, establishes the Euler-angle and orientation
#   conventions used in texture analysis.
# * A. Morawiec, [Orientations and Rotations: Computations in Crystallographic Textures](https://doi.org/10.1007/978-3-662-09156-2),
#   Springer, 2004, develops coordinate maps, symmetry, and the geometry of orientation
#   space.
# * D. Rowenhorst et al., [Consistent representations of and conversions between 3D rotations](https://doi.org/10.1088/0965-0393/23/8/083501),
#   *Modelling and Simulation in Materials Science and Engineering* 23, 083501, 2015,
#   compares active and passive conventions and gives reproducible conversion rules.
# * [ISO 24173:2024](https://www.iso.org/standard/82749.html), *Microbeam analysis --
#   Guidelines for orientation measurement using electron backscatter diffraction*, gives
#   current guidance for reproducible EBSD orientation measurements.

# ## Next
#
# [Symmetry](OrientationSymmetry.html) develops the equivalent rotations that an
# orientation represents. [Pole Figures](OrientationPoleFigure.html) then draws the
# specimen directions computed here, while
# [Inverse Pole Figures](OrientationInversePoleFigure.html) asks the inverse question.
