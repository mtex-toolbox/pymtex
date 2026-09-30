# # Inverse Pole Figures
#
# An inverse pole figure fixes a direction in the specimen and asks which crystal
# direction points along it. It asks the [pole-figure](OrientationPoleFigure.html) question
# in reverse. It is the natural view when one specimen direction is physically important.
# Examples include the normal of a rolled sheet, a loading axis, and the surface normal of
# an EBSD map.
#
# This page assumes the crystal-to-specimen coordinate map from
# [Theory](DefinitionAsCoordinateTransform.html) and the equivalent directions from
# [Symmetry](OrientationSymmetry.html). It also assumes the projections from
# [Spherical Projections](SphericalProjections.html). A *reference frame* is the coordinate
# system in which data are expressed. The plotting convention below lays specimen Y upward
# and specimen X right. It changes only the screen layout. It does not rotate the specimen
# or re-express the data.

# +
import numpy as np

from mtex import *

plottingConvention.default('y↑→x')

rng = np.random.default_rng(635)
cs = crystalFrame('321')

ori = orientation.rand(cs, rng=rng)
ori
# -

# ## Building One by Hand
#
# Fix the specimen Z direction. The orientation maps crystal coordinates into specimen
# coordinates, so its inverse maps this direction back into the crystal frame.

r = vector3d.Z

# ---

h = inv(ori) * r
h

# The displayed Miller indices are the crystal-frame coordinates of the specimen Z
# direction. [symmetrise](vector3d.symmetrise.html) applies the crystal point group. It
# returns every equivalent description.

hSym = h.symmetrise()
symmetryCopyCount = len(hSym)
symmetryCopyCount

# The count is six for a general direction in point group 321. A direction on a symmetry
# axis can have fewer distinct copies. Several operations may give the same result there.
#
# Plot the copies in the [fundamental sector](FundamentalSector.html), the part of the
# sphere that retains one representative from each equivalent family.

plot(hSym, fundamentalRegion=True, markerFaceColor='red')

# Notice one marker rather than six. Symmetry reduction maps all six copies onto the same
# position in the sector. They describe one crystal direction family, not six
# orientations.
#
# ## The Shortcut
#
# [plotIPF](orientation.plotIPDF.html) performs the inverse map, symmetrisation,
# reduction, and projection for several specimen directions at once.

plotIPF(ori, [vector3d.X, vector3d.Y, vector3d.Z])

# The three sectors correspond to specimen X, Y, and Z. Each marker shows which crystal
# direction lies along the axis named above its panel.
#
# One marker does not determine the complete orientation. The plot loses rotation about
# the aligned direction. Distinct orientations can therefore occupy the same inverse pole
# figure position. Use [3D Plots](OrientationVisualization3d.html) or
# [Section Plots](OrientationVisualizationSections.html) when that missing information
# matters.
#
# ## A Population as Contours
#
# For a population, `contourf` replaces the individual markers with a kernel density
# estimate on the sphere. Construct 300 orientations within $12^\circ$ of the orientation
# above. This gives the density a feature worth reading.

oriPopulation = rotation.rand(300, maxAngle=12 * degree, rng=rng) * ori
populationSize = len(oriPopulation)
populationSize

# ---

plotIPF(oriPopulation, [vector3d.X, vector3d.Y, vector3d.Z], contourf=True)
mtexColorbar()

# Each panel now contains one concentrated lobe rather than one marker. Its position
# changes because X, Y, and Z point along different crystal directions. The colour scale is
# in multiples of a random distribution (m.r.d.). It is a density, not a percentage of
# orientations at one point. Quantitative inverse pole densities from an ODF are developed
# in [Inverse Pole Figures of an ODF](ODFInversePoleFigure.html).
#
# ## Seeing the Symmetry Copies
#
# By default `plotIPF` draws only the fundamental sector. The option `complete` ignores
# that reduction, while `upper` restricts the result to the upper hemisphere.

plotIPF(oriPopulation, r, contourf=True, complete=True, upper=True)
mtexColorbar()

# Notice that the one Z-direction lobe is repeated three times on this upper hemisphere.
# The complete sphere would also contain the three copies in the lower hemisphere. The six
# lobes are the point-group equivalents that the fundamental-sector plot folds onto one
# another.
#
# The `antipodal` flag makes the additional identification $\mathbf{h}\sim-\mathbf{h}$.
# Use it only when the measurement or model cannot distinguish a crystal direction from
# its opposite. This is a modelling choice, not another operation of point group 321. See
# [Axes and Antipodal Symmetry](VectorsAxes.html).
#
# If an orientation carries a nontrivial specimen symmetry, `plotIPF` also applies it to
# the fixed specimen direction. Crystal symmetry acts after the inverse map in the crystal
# frame. Specimen symmetry acts before it in the specimen frame. See
# [Specimen Symmetry](SpecimenSymmetry.html).
#
# ## Inverse Pole Figures and EBSD Colours
#
# An [IPF map](EBSDIPFMap.html) uses the same construction point by point. A colour key
# assigns a colour to each fundamental-sector position. The key uses one chosen specimen
# direction. The map therefore inherits the same information loss. Equal colours do not by
# themselves prove equal orientations.
#
# ## The Maths Behind an Inverse Pole Figure
#
# Let $\mathbf{O}$ map crystal coordinates into specimen coordinates. Let $\mathbf{C}$ be
# a crystal-symmetry operation, $\mathbf{P}$ a specimen-symmetry operation, and
# $\mathbf{r}$ the fixed specimen direction. Every crystal direction represented by
# `plotIPF` has the form
#
# $$ \mathbf{h} = \mathbf{C}\,\mathbf{O}^{-1}\mathbf{P}\,\mathbf{r},
#    \qquad \mathbf{C} \in \mathrm{S}_{\mathrm{c}}, \quad
#    \mathbf{P} \in \mathrm{S}_{\mathrm{s}}. $$
#
# This example has identity specimen symmetry. The hand construction is therefore
# $\mathbf{h}=\mathbf{C}\mathbf{O}^{-1}\mathbf{r}$. Reduction to the fundamental sector
# keeps one representative of this family.
#
# ## References
#
# * H.-J. Bunge, [Texture Analysis in Materials Science: Mathematical Methods](https://doi.org/10.1016/C2013-0-11769-2),
#   Butterworths, English ed., 1982, develops pole and inverse pole figures from
#   orientation densities.
# * U. F. Kocks, C. N. Tomé, and H.-R. Wenk, [Texture and Anisotropy](https://assets.cambridge.org/97805217/94206/excerpt/9780521794206_excerpt.pdf),
#   Cambridge University Press, 1998, connects these texture representations to processing
#   and anisotropic properties.
# * A. Morawiec, [Orientations and Rotations: Computations in Crystallographic Textures](https://doi.org/10.1007/978-3-662-09156-2),
#   Springer, 2004, develops rotations, symmetry, and elementary regions of orientation
#   space.
# * D. Chateigner, L. Lutterotti, and M. Morales, [Quantitative texture analysis and combined analysis](https://doi.org/10.1107/97809553602060000968),
#   _International Tables for Crystallography H_, ch. 5.3, 2019, relates orientation
#   distributions and diffraction pole figures.
# * G. Nolze and R. Hielscher, [Orientations - perfectly colored](https://doi.org/10.1107/S1600576716012942),
#   _Journal of Applied Crystallography_ 49, 1786-1802, 2016, explains the topology and
#   limitations of inverse-pole-figure colour keys.
# * [ISO 24173:2024](https://www.iso.org/standard/82749.html), _Microbeam analysis --
#   Guidelines for orientation measurement using electron backscatter diffraction_, gives
#   current guidance for reproducible EBSD orientation measurements.
#
# ## Next
#
# Continue with [3D Plots](OrientationVisualization3d.html). They retain the orientation
# information that an inverse pole figure discards. The sector itself, and how symmetry
# determines its shape, is [Fundamental Sector](FundamentalSector.html). Its counterpart
# for whole orientations is the [Fundamental Region](OrientationFundamentalRegion.html). For
# measured maps, continue with [IPF Maps](EBSDIPFMap.html).
