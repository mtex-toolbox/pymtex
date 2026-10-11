# # Standard Orientations
#
# Rolling, drawing and recrystallisation produce the same few orientations again and
# again. These ideal orientations have names such as Cube, Goss, Brass and Copper. Naming
# the dominant components describes a texture in one sentence. MTEX provides them so that
# measured orientations can be compared with the conventional ideals.
#
# This page assumes the orientation defined in
# [Defining Orientations](OrientationDefinition.html). It also uses the plane and
# direction notation from [Miller Indices](CrystalDirections.html).
# [Symmetry](OrientationSymmetry.html) explains why one physical component has several
# equivalent coordinate descriptions.

import numpy as np

from mtex import *

# ## Predefined Components and Their Frame
#
# The complete set of predefined orientation constructors is
#
# * Cube, CubeND22, CubeND45, CubeRD
# * Goss, invGoss
# * Copper, Copper2
# * SR, SR2, SR3, SR4
# * Brass, Brass2
# * PLage, PLage2, QLage, QLage2, QLage3, QLage4
#
# These names are conventions for cubic rolling and recrystallisation textures. Their
# Euler angles require the same Bunge convention and frames. See
# [MTEX vs. Bunge Convention](MTEXvsBungeConvention.html). The numbered constructors
# retain variants that may be distinct under weak specimen symmetry. With orthorhombic
# specimen symmetry, several are equivalent.
#
# A *reference frame* is the coordinate system in which data are expressed. Here the
# specimen frame is the rolling frame. Specimen X is the rolling direction (RD), and Y is
# the transverse direction (TD). Specimen Z is the normal direction (ND). The plotting
# convention below draws RD north and TD west. Consequently, ND points out of the page.

plottingConvention.default('y←↑x')

# cubic crystal symmetry, and orthorhombic specimen symmetry in the rolling frame
cs = crystalFrame('m-3m')
ss = specimenFrame.rolling('orthorhombic')

# ## A Representative Selection
#
# The following subset contains common ideal components and several rotated Cube
# components.

components = cat(orientation.goss(cs, ss),
                 orientation.brass(cs, ss),
                 orientation.cube(cs, ss),
                 orientation.cubeND22(cs, ss),
                 orientation.cubeND45(cs, ss),
                 orientation.cubeRD(cs, ss),
                 orientation.copper(cs, ss),
                 orientation.PLage(cs, ss),
                 orientation.QLage(cs, ss))

componentNames = ['Goss', 'Brass', 'Cube', 'CubeND22', 'CubeND45', 'CubeRD', 'Copper', 'PLage', 'QLage']

# ## Plane and Direction Notation
#
# Each legend entry below is generated in the conventional form $(hkl)[uvw]$. The plane
# $(hkl)$ faces the sheet normal, and the direction $[uvw]$ points along the rolling
# direction. Equivalently, the orientation maps the plane normal to ND and the lattice
# direction to RD. [orientation.byMiller](orientation.byMiller.html) uses this argument
# order. [round2Miller](orientation.round2Miller.html) reads those two indices back off an
# orientation.

legendNames = [''.join(m.char() for m in round2Miller(components[i])) for i in range(len(components))]
legendNames

# ## Matching a Measured Orientation
#
# A component match is an angular comparison, not a comparison of three Euler-angle
# columns. As a reproducible stand-in for a measurement, make an orientation five degrees
# from the Copper entry. Then compare it with every component.

phi1, Phi, phi2 = Euler(components[6])
measured = orientation.byEuler(phi1 + 5 * degree, Phi, phi2, cs, ss)

componentDistance = angle(measured, components) / degree
bestId = int(np.argmin(componentDistance))

bestMatch = componentNames[bestId]
bestMatch

# ---

bestDifference = componentDistance[bestId]
bestDifference

# The closest entry is Copper at $5^\circ$. [angle](orientation.angle.html) uses the
# attached crystal and specimen symmetries. It therefore compares physical components
# rather than arbitrary stored representatives. A real analysis must also state the
# angular tolerance used to call an orientation part of a component.

# ## Three-Dimensional Euler Angle Space
#
# The first view places every component at its three Bunge Euler angles.

# +
for i in range(len(components)):
  plot(components[i], 'bunge', markerSize=10, markerFaceColor=ind2color(i), displayName=legendNames[i])
  hold(True)
hold(False)
legend(location='southoutside', numColumns=3)
# -

# Notice Cube at the zero-angle corner. CubeND22 and CubeND45 move along the $\varphi_1$
# edge, whereas CubeRD moves along the $\Phi$ edge. This plot is a coordinate chart.
# Proximity near a chart boundary need not mean a small physical orientation difference.

# ## Two-Dimensional phi2 Sections
#
# The classical paper view uses sections of fixed $\varphi_2$. This is how rolling
# components are usually recognized in an ODF. See
# [Euler Angle Sections](EulerAngleSections.html).

# +
for i in range(len(components)):
  plotSection(components[i], add2all=True, markerSize=10, markerFaceColor=ind2color(i), displayName=legendNames[i])

legend(location='southeast')
# -

# Notice that one colour can occur at several coordinate positions. These are
# symmetry-equivalent representatives of one component. They are not additional physical
# components.

# ## Three-Dimensional Axis-Angle Space
#
# Axis-angle space places the same components inside the fundamental region of the
# cubic-orthorhombic symmetry pair. See
# [Fundamental Region](OrientationFundamentalRegion.html).

# +
for i in range(len(components)):
  plot(components[i], 'axisAngle', markerSize=10, markerFaceColor=ind2color(i), displayName=legendNames[i])
  hold(True)
hold(False)
legend(location='southoutside', numColumns=3)
# -

# Cube is the identity orientation at zero rotation. Every other marker is one
# representative chosen from its symmetry class. All nine components therefore fit into
# the single fundamental region.

# ## Pole Figures
#
# A pole figure shows where each component puts selected families of lattice planes. This
# is the view against which a measured pole figure is compared.

h = Miller([[1, 0, 0], [1, 1, 0], [1, 1, 1], [3, 1, 1]], cs)

# +
for i in range(len(components)):
  plotPF(components[i], h, markerSize=10, markerFaceColor=ind2color(i), displayName=legendNames[i])
  hold(True)
hold(False)
legend(location='northeast', numColumns=2)
# -

# For Goss, the $\{110\}$ family has a pole in the centre because its $(011)$ plane faces
# ND. Cube puts $\{100\}$ poles in the centre and on the RD and TD axes of the rim.

# ## Inverse Pole Figures
#
# Inverse pole figures ask the opposite question: which crystal direction lies along
# specimen X, Y or Z? In this rolling frame those directions are RD, TD and ND.

r = cat(vector3d.X, vector3d.Y, vector3d.Z)

# +
for i in range(len(components)):
  plotIPF(components[i], r, markerSize=(12 - i - 1) ** 1.5, markerFaceColor=ind2color(i), displayName=legendNames[i])
  hold(True)
hold(False)
legend(location='northeast', numColumns=2)
# -

# Several components put the same crystal direction along one specimen axis but differ by
# rotation about it. Their markers coincide in that panel, so the decreasing marker sizes
# keep all components visible. An inverse pole figure loses the rotation about the plotted
# direction. Therefore, it cannot identify a full orientation by itself.

# ## From a Component to a Model Texture
#
# A named component is a single ideal orientation. A real texture has a spread around that
# ideal. Giving the component a halfwidth turns it into a model ODF. This is the quantity
# fitted against measured data; see [Modeling ODFs](ODFModeling.html).

odf = unimodalODF(components[2], halfwidth=7.5 * degree)
odf

# +
plotPF(odf, h)
hold(True)
plotPF(odf, h, contour=True, lineColor='k', lineWidth=2)
hold(False)
# -

# The isolated Cube markers have become density peaks at the same pole positions. At
# $7.5^\circ$, the radial kernel falls to half its maximum. This halfwidth is a radius,
# not a $15^\circ$ uniform band.

# ## The Model in Inverse Pole Figures
#
# The same spread appears around the Cube directions in the inverse pole figures.

# +
plotIPF(odf, r)
hold(True)
plotIPF(odf, r, contour=True, lineColor='k', lineWidth=2)
hold(False)
# -

# Notice that the peak centres agree with the Cube markers in the earlier inverse pole
# figures. Only the ideal point has changed into a continuous neighbourhood.

# ## The Model in phi2 Sections
#
# Finally, plot the model in $\varphi_2$ sections and overlay all nine ideal components to
# identify the centre used to build it.

# +
plotSection(odf)
hold(True)
plotSection(odf, contour=True, lineColor='k', lineWidth=2)

for i in range(len(components)):
  plotSection(components[i], markerSize=10, filled=True, markerFaceColor=ind2color(i), displayName=legendNames[i])

hold(False)
# -

# The density maximum sits on Cube. The other component markers remain isolated reference
# points and do not contribute to this model ODF.

# ## References
#
# * H.-J. Bunge, [Texture Analysis in Materials Science: Mathematical Methods](https://doi.org/10.1016/C2013-0-11769-2),
#   Butterworths, 1982. It establishes the Euler-angle and ODF conventions used in texture
#   analysis.
# * U. F. Kocks, C. N. Tomé and H.-R. Wenk, [Texture and Anisotropy](https://assets.cambridge.org/97805217/94206/excerpt/9780521794206_excerpt.pdf),
#   Cambridge University Press, 2nd ed., 2000. It connects named components with
#   processing and material anisotropy.
# * O. Engler and V. Randle, [Introduction to Texture Analysis](https://doi.org/10.1201/9781420063660),
#   CRC Press, 2nd ed., 2010. It gives a practical treatment of macrotexture, microtexture
#   and orientation mapping.
# * L. A. I. Kestens and H. Pirgazi, [Texture formation in metal alloys with cubic crystal
#   structures](https://doi.org/10.1080/02670836.2016.1231746), *Materials Science and
#   Technology* 32 (2016). It reviews the common rolling components of cubic alloys.

# ## Next
#
# [MTEX vs. Bunge Convention](MTEXvsBungeConvention.html) explains which published
# Euler-angle triplets can be copied directly. Pole figures are developed in
# [Pole Figures](OrientationPoleFigure.html). The reverse view is
# [Inverse Pole Figures](OrientationInversePoleFigure.html).
#
# [Fibres of Orientations](OrientationFibre.html) follows the named fibre components
# through orientation space. [Modeling ODFs](ODFModeling.html) develops the model textures
# introduced above.
