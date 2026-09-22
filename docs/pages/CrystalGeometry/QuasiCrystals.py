# # Quasicrystal Symmetry
#
# A quasicrystal is an aperiodic crystal with long-range order. Five- or tenfold symmetry
# is common, but forbidden rotational symmetry is not part of the definition.
#
# A periodic lattice admits only one-, two-, three-, four- and sixfold rotation axes. An
# icosahedral quasicrystal can therefore have point symmetry that is not among the 32
# [crystallographic point groups](CrystalSymmetries.html).
#
# This page shows how to supply such a finite point group to MTEX. It covers point
# symmetry only; it does not model quasiperiodic translations or the higher-dimensional
# indexing used in quasicrystal structure analysis. Read
# [Crystal Symmetries](CrystalSymmetries.html) and [Fundamental Sector](FundamentalSector.html)
# before this page.

import matplotlib.pyplot as plt
import numpy as np

from mtex import cat, crystalFrame, degree, hold, ipfHSVKey, normalize, plot, plottingConvention, rotation, vector3d

plottingConvention.default('y↑→x')

# ## Build the Proper Icosahedral Group
#
# [crystalFrame.byElements](crystalFrame.byElements.html) repeatedly composes the supplied
# operations until they close into a group. The generators must describe a finite group;
# arbitrary rotations need not do so.
#
# The proper icosahedral group is the rotation group of both the icosahedron and the
# dodecahedron. One fivefold rotation and one threefold rotation generate it. Their axes
# are written in the crystal frame.

phi = (1 + np.sqrt(5)) / 2

fiveFoldAxis = normalize(vector3d(0, 1 / phi, 1))
rot5 = rotation.byAxisAngle(fiveFoldAxis, 72 * degree)

threeFoldAxis = normalize(vector3d(1 / phi ** 2, 0, 1))
rot3 = rotation.byAxisAngle(threeFoldAxis, 120 * degree)

cs = crystalFrame.byElements(cat(rot5, rot3))
cs

# The summary reports 60 elements. These are all proper rotations, so this is the chiral
# icosahedral group rather than the group with inversion.

plot(cs, symbolSize=0.5, projection='eangle', grid=True)

# The pentagons mark six fivefold axes, the triangles mark ten threefold axes, and the
# lenses mark fifteen twofold axes. Opposite ends of an axis represent the same rotation
# axis.

# ## Reduce Directions and Orientations
#
# Most MTEX algorithms ask a symmetry for its elements rather than for a crystallographic
# name. Direction and orientation reductions therefore also work for this custom group.
#
# A [fundamental sector](FundamentalSector.html) contains one representative of each
# general direction family. The proper icosahedral group has 60 elements, so its sector
# covers one sixtieth of the sphere.

hold(True)
plot(cs.fundamentalSector(), color='red')
hold(False)

# The red spherical triangle is bounded by three great circles. It is the
# non-crystallographic counterpart of the cubic fundamental sector.
#
# The corresponding fundamental region in orientation space is a dodecahedron. Its twelve
# faces separate the region from twelve neighbouring symmetry-equivalent copies.

oR = cs.fundamentalRegion()
plot(oR)
plt.gca().set_axis_off()

# The heavy outer edges reveal the twelve-faced region. The finer curves are the
# axis-angle coordinate grid drawn on its faces.

# ## Build an Inverse Pole Figure Colour Key
#
# An [ipfHSVKey](ipfHSVKey.html) assigns a colour to each direction in the fundamental
# sector. For an EBSD orientation, the key first expresses its chosen specimen direction
# in the crystal frame and then colours the resulting crystal direction.

ipfKey = ipfHSVKey(cs)

plot(ipfKey, complete=True, upper=True, resolution=0.5 * degree, noLabel=True)
hold(True)
plot(cs, symbolSize=0.6)
hold(False)

# The upper hemisphere shows the colours repeated over the symmetry-equivalent sectors
# visible there. An EBSD map of a quasicrystalline phase can use this key in the same way
# as [a crystallographic IPF key](EBSDIPFMap.html).

# ## Align the Fivefold Axis with the Crystal Frame
#
# A crystal frame is the Cartesian reference frame in which the symmetry axes are
# expressed. Changing the axes below changes their alignment in that frame, not the
# abstract group.
#
# Here the fivefold axis is placed along $\mathbf{z}$. The threefold axis is placed at
# the angle between the original generators, $37.377^\circ$.

rot5 = rotation.byAxisAngle(vector3d.Z, 72 * degree)
rot3 = rotation.byAxisAngle(vector3d.byPolar(37.377 * degree, 0), 120 * degree)

cs = crystalFrame.byElements(cat(rot5, rot3))
cs

# The summary still reports 60 elements, confirming that only the alignment changed.

plot(cs, symbolSize=0.5, projection='eangle', grid=True)

# The central pentagon shows that the plot now looks directly down the fivefold axis. The
# surrounding threefold and twofold axes have rotated with it.

# ## Add Inversion
#
# Many commonly studied icosahedral quasicrystals are centrosymmetric. Adding
# [rotation.inversion](rotation.inversion.html) combines every proper rotation with
# inversion.

cs = crystalFrame.byElements(cat(rot5, rot3, rotation.inversion))
cs

# The summary now reports 120 elements: 60 proper and 60 improper operations. It does not
# report 120 physical rotations.

plot(cs, symbolSize=0.6, mirrorLineWidth=2)

# The hollow circle at the centre marks inversion. The great circles mark mirror planes
# produced by combining inversion with twofold rotations.

ipfKey = ipfHSVKey(cs)
plot(ipfKey, complete=True, resolution=0.5 * degree, noLabel=True)

# The upper- and lower-hemisphere panels now carry the same colour pattern. Inversion
# identifies every direction with its opposite, as explained in
# [Axes and Antipodal Symmetry](VectorsAxes.html).

# ## Further Reading
#
# * The [IUCr Online Dictionary of Crystallography](https://dictionary.iucr.org/Quasicrystal)
#   defines quasicrystals and explains why forbidden rotational symmetry is common but not
#   required.
# * Th. Hahn, H. Klapper, U. Mueller, and M. I. Aroyo, [Point groups and crystal classes](https://doi.org/10.1107/97809553602060000930),
#   _International Tables for Crystallography A_, ch. 3.2, 2016, treats crystallographic
#   and non-crystallographic point groups.
# * D. Shechtman, I. Blech, D. Gratias, and J. W. Cahn, [Metallic phase with long-range orientational order and no translational symmetry](https://doi.org/10.1103/PhysRevLett.53.1951),
#   _Physical Review Letters_ 53, 1951-1953, 1984, reports the first icosahedral
#   quasicrystal.
# * W. Steurer and S. Deloudi, [Crystallography of Quasicrystals](https://doi.org/10.1007/978-3-642-01899-2),
#   Springer, 2009, develops quasiperiodic structure analysis and the higher-dimensional
#   approach beyond MTEX's point-group treatment.

# ## Next
#
# [Fundamental Region](OrientationFundamentalRegion.html) develops the orientation-space
# reduction used above. Continue with [Inverse Pole Figure](OrientationInversePoleFigure.html)
# and [IPF Maps](EBSDIPFMap.html) to use the custom symmetry for orientation data.

# ## Technical Details
#
# The port names the icosahedral groups by their signature, `532` and `-5-32/m`, where
# MATLAB reports `unknown`; the group is built from its elements in closure order in both.
