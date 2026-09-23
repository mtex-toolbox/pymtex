# # Piezoelectricity in Quartz
#
# Mechanical stress creates electric displacement in a piezoelectric crystal. This is the
# *direct piezoelectric effect*. An electric field can also create strain, which is the
# converse effect.
#
# The direct effect relates a symmetric stress tensor $\sigma$ to electric displacement
# $D$ through the rank three piezoelectric strain tensor $d$:
#
# $$ D_i = d_{ijk}\,\sigma_{jk}. $$
#
# This page assumes the tensor rank and compact notation introduced in
# [Defining Tensorial Properties](TensorDefinition.html). Read
# [Importing Tensor Data](TensorImport.html) first for units, crystal frames, and the
# `doubleConvention` used by the file below.

import numpy as np

from mtex import (WhiteJetColorMap, blue2redColorMap, crystalFrame, mtexColorbar, mtexdatafile, orientation, plot,
                  plotSection, plottingConvention, setMTEXpref, surf, tensor, vector3d)

# +
plottingConvention.default('y↑→x')

# define the quartz crystal symmetry and crystal frame
csQuartz = crystalFrame('32', [4.916, 4.916, 5.4054], 'X||a*', 'Z||c', mineral='Quartz')

# load the right-handed quartz piezoelectric strain tensor
quartzFile = mtexdatafile('quartzPiezo')
P = tensor.load(quartzFile, csQuartz, rank=3, name='piezoelectric strain', unit='pC/N', doubleConvention=True)

setMTEXpref('defaultColorMap', blue2redColorMap)
# -

# ## Why crystal symmetry matters
#
# A material-property tensor must be invariant under every operation of its crystal
# point group. Inversion changes the sign of a polar rank three tensor. A
# centrosymmetric crystal must therefore have $d=0$.
#
# Piezoelectricity is allowed in 20 of the 32 crystallographic point groups: every
# noncentrosymmetric group except `432`. Quartz belongs to point group `32`, so symmetry
# permits the effect and repeats it about the threefold crystal axis.
#
# Handedness remains important even though the point-group symbol is the same. Inverting
# right-handed quartz produces left-handed quartz and reverses the piezoelectric tensor.
# The filename identifies `P` as the right-handed form.

# ## The signed longitudinal response
#
# [directionalMagnitude](tensor.directionalMagnitude.html) contracts the same unit
# direction $n$ into all three indices:
#
# $$ q(n) = d_{ijk}\,n_i n_j n_k. $$
#
# For a uniaxial stress along $n$, $q(n)$ is the electric-displacement component along
# that direction per unit stress. It is a signed longitudinal coefficient, not the
# magnitude of the displacement vector.

# +
q = P.directionalMagnitude()

# plot one symmetry-reduced sector
plot(P)
mtexColorbar(title='longitudinal coefficient (pC/N)')
# -

# The sector contains both positive and negative response. Blue and red therefore mean
# opposite signs of the longitudinal component, not weak and strong polarization.
#
# A rank three directional response is odd: $q(-n)=-q(n)$. Plotting both hemispheres
# makes that sign reversal explicit.

plot(P, 'complete', 'smooth', 'upper', 'lower')
mtexColorbar(title='longitudinal coefficient (pC/N)')

# The same threefold pattern occurs on the two hemispheres with red and blue exchanged at
# antipodal directions. This is the sign reversal expected for an odd-rank tensor.

# ## A radial surface
#
# [surf](S2Fun.surf.html) can use the absolute response as distance from the origin and
# the colour as its sign. The `noScaling` option keeps the physical zero and the original
# pC/N values.

surf(q, noScaling=True)
mtexColorbar(title='longitudinal coefficient (pC/N)')

# The six lobes have the same radial magnitude in antipodal pairs, while their colours
# have opposite signs. The surface meets the origin in directions where the longitudinal
# response is zero.

# ## Planar sections
#
# [plotSection](S2Fun.plotSection.html) draws the signed response as a polar radius in a
# chosen plane. A negative radius is placed in the opposite direction, so each curve is
# traced twice. Use the coloured hemisphere plots above, rather than these outlines, to
# read the sign.
#
# The basal plane is normal to z.

plotSection(q, vector3d.Z)

# The basal section is a three-petal rose. Its threefold repetition is the clearest planar
# expression of quartz point group `32`.
#
# A vertical section normal to x has a different outline.

plotSection(q, vector3d.X)

# This section is a single oval rather than a three-petal rose because its plane contains
# the threefold z axis instead of cutting across it.

# ## A polycrystal average needs handedness
#
# The Tongue quartzite data contain one orientation for each of 382 grains. Mainprice,
# Lloyd, and Casey (1993) explain a decisive limitation of these measurements: routine
# electron-channelling patterns did not determine the handedness of each quartz grain.
# Every grain was indexed as right-handed. The paper therefore states that
# piezoelectricity cannot be calculated from this orientation set.

# +
orientationFile = mtexdatafile('tongue')
ori = orientation.load(orientationFile, csQuartz, columnNames=['Euler 1', 'Euler 2', 'Euler 3'])

orientationCount = len(ori)
orientationCount
# -

# The calculation below is still instructive as a counterexample. It uses
# [calcTensor](orientation.calcTensor.html) to rotate the right-handed tensor by every
# orientation and take their unweighted arithmetic mean. The result is an *apparent*
# aggregate in which all 382 grains have been assumed to be right-handed. It is not a
# prediction for the rock.

# +
apparentMean = ori.calcTensor(P)
qApparent = apparentMean.directionalMagnitude()

plot(apparentMean, 'complete', 'smooth', 'upper', 'lower')
mtexColorbar(title='apparent longitudinal coefficient (pC/N)')

# rows: the single crystal and the all right-handed average, columns: minimum, maximum
responseRanges = np.array([[q.min()[0], q.max()[0]], [qApparent.min()[0], qApparent.max()[0]]])
responseRanges
# -

# The single-crystal range is -2.3000 to 2.3000 pC/N. Under the deliberately false
# all-right-handed assumption, the range is -0.5940 to 0.5940 pC/N, or 25.8 percent of
# the single-crystal extreme.
#
# Differently oriented grains partly cancel, which explains the reduction. Unknown
# left-handed grains can reverse additional contributions. Their number and orientations
# are absent from this data set, so the apparent average cannot be corrected without new
# handedness information.

# ## Next
#
# [Birefringence](BirefringenceDemo.html) continues with a rank two optical property,
# whose even rank makes its directional response antipodal.
# [Tensor Averages](TensorAverage.html) develops Voigt, Reuss, and Hill estimates for
# elastic stiffness and explains their mechanical assumptions.

# ## Further reading
#
# * D. Mainprice, G.E. Lloyd, and M. Casey,
#   [Individual orientation measurements in quartz polycrystals: advantages and limitations for texture and petrophysical property determinations](https://doi.org/10.1016/0191-8141(93)90162-4),
#   *Journal of Structural Geology* 15 (1993), 1169-1187, documents the 382-grain data
#   and its handedness limitation.
# * [IUCr Online Dictionary of Crystallography: Piezoelectricity](https://dictionary.iucr.org/Piezoelectricity)
#   lists the 20 piezoelectric point groups and relates the direct and converse tensors.
# * [IEEE Std 176-1987](https://standards.ieee.org/ieee/176/356/), *IEEE Standard on
#   Piezoelectricity*, specifies quartz axes, signs, and contracted notation. The standard
#   was withdrawn in 2000.
# * J.F. Nye, [Physical Properties of Crystals: Their Representation by Tensors and Matrices](https://search.worldcat.org/title/11114089),
#   Oxford University Press, 1985, develops tensor representation surfaces and crystal
#   symmetry.
# * C. Frondel, [The System of Mineralogy, Volume III: Silica Minerals](https://search.worldcat.org/title/The-System-of-Mineralogy-%3A-vol.-III-Silica-Minerals/oclc/500448822),
#   7th ed., Wiley, 1962, is the source named in the bundled quartz coefficient file.

setMTEXpref('defaultColorMap', WhiteJetColorMap)

# ## Technical Details
#
# The quartz file of MATLAB's data folder is the sample data set `quartzPiezo` here and
# the Tongue quartzite orientations the data set `tongue`, both fetched on first use.
# `min` and `max` of a function on the sphere return the value and the direction where it
# is attained, as MATLAB's two outputs.
