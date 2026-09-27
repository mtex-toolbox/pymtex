# # Axes and Antipodal Symmetry
#
# A *direction* distinguishes its two ends: north is not south. An *axis* does not. A
# plane normal is an axis when the two sides of the plane are physically equivalent.
# Reversing the axis of a twofold rotation likewise describes the same $180^\circ$
# rotation.
#
# MTEX represents directions and axes with [vector3d](vector3d.vector3d.html). The logical
# property `antipodal` records the difference. When it is true, `v` and `-v` represent the
# same axis.
#
# This page assumes the vector construction and angle conventions from
# [Defining Three-Dimensional Vectors](VectorDefinition.html) and
# [Vector Operations](VectorsOperations.html). See
# [Spherical Projections](SphericalProjections.html) if upper- and lower-hemisphere plots
# are new to you.

import numpy as np

from mtex import *

plottingConvention.default('y↑→x')

# ## Plotting Directions and Axes
#
# Take two directions that differ only in the sign of their z coordinate.

# +
v1 = vector3d(1, 1, 2)
v2 = vector3d(1, 1, -2)

plot(cat(v1, v2), label=['v_1', 'v_2'], grid=True)
# -

# The labels appear on different hemisphere plots. Selecting only the upper hemisphere
# would hide `v2`. It would not turn either direction into an axis.

plot(cat(v1, v2), label=['v_1', 'v_2'], antipodal=True, grid=True)

# The plotting option represents both inputs on the upper hemisphere, using `-v2` for the
# second one. The two points remain separated because `v1` and `v2` are different axes.
# This option affects this plot only; it does not change either input variable.

# ## Angles and Axial Means
#
# The [angle](vector3d.angle.html) between directed vectors ranges from $0^\circ$ to
# $180^\circ$. The angle between axes is measured to whichever end is closer and is
# therefore never greater than $90^\circ$.

directedAngle = angle(v1, v2) / degree
directedAngle

# ---

axisAngle = angle(v1, v2, antipodal=True) / degree
axisAngle

# The directed angle is $109.4712^\circ$, whereas the axial angle is $70.5288^\circ$. The
# two add to $180^\circ$. Forgetting `antipodal` produces a plausible but obtuse answer
# instead of an error.
#
# Signs matter for an ordinary mean as well. Two opposite unit directions cancel, while
# the same observations interpreted as axes have a mean axis.

directedMeanLength = norm(mean(cat(vector3d.X, -vector3d.X)))
directedMeanLength

# ---

axisMeanMisfit = angle(mean(cat(vector3d.X, -vector3d.X), antipodal=True), vector3d.X, antipodal=True) / degree
axisMeanMisfit

# The directed mean has length 0. The axial mean is aligned with the X axis, so its axial
# misfit is $0^\circ$. The [mean](vector3d.mean.html) of axes uses their unoriented lines
# rather than averaging signed components.

# ## Attaching the Flag to the Data
#
# Repeating an option at every call is unnecessary. Attach the flag by assigning the
# property. The `vector3d` constructor also accepts `antipodal=True`.

# +
v2.antipodal = True

storedAxisAngle = angle(v1, v2) / degree
storedAxisAngle
# -

# ---

sameAxis = (v2 == -v2)
sameAxis

# The stored flag gives the same $70.5288^\circ$ angle without an option and makes
# `v2 == -v2` true. For a binary operation, an antipodal flag on either operand requests
# the axial interpretation.
#
# One `vector3d` variable carries one `antipodal` value for its entire array. Do not mix
# directed observations and axes in the same variable. Separate them before calculating
# angles, means, or densities.

# ## Densities of Axes
#
# [Density Estimation](VectorsDensityEstimation.html) turns a list of directions into a
# continuous function on the sphere. These 100 deterministic directions form a short band
# in the upper hemisphere.

# +
rho = np.linspace(20, 70, 100) * degree
theta = np.linspace(10, 30, 100) * degree
v = vector3d.byPolar(theta, rho)

directionDensity = v.calcDensity(halfwidth=10 * degree)
plot(directionDensity, complete=True)
# -

# The upper hemisphere contains the concentration band. The lower one has no
# point-reflected copy because these observations are still directions.

# +
axisDensity = v.calcDensity(antipodal=True, halfwidth=10 * degree)
plot(axisDensity, complete=True)
# -

# The lower hemisphere now repeats the upper pattern through the centre of the sphere. An
# axial density must satisfy $f(v)=f(-v)$.

# ## Measured Pole Figures
#
# Under Friedel's law, opposite reflections have equal intensities when the crystal is
# centrosymmetric or resonant scattering is absent. Conventional pole-figure measurements
# normally use this axial model. Resonant scattering can distinguish the signs. Antipodal
# symmetry is therefore an experimental assumption rather than a property of every ODF.
#
# The [Pole Figure Tutorial](PoleFigureTutorial.html) introduces measured pole figures
# and the experiment behind them.

# +
pf = mtexdata('dubna')

CS = pf.CS

plot(pf[0])
# -

# MTEX draws this measured pole figure on the upper hemisphere because its specimen
# directions already represent axes. The lower hemisphere would repeat the same
# measurements.

annotate(vector3d(1, 0, -1), labeled=True, backgroundColor='w')

# Although the annotated direction points downwards, its equivalent upper endpoint is
# labelled in the plot.

# ## Pole Figures Computed from an ODF
#
# A pole figure computed from an ODF need not be antipodal. The quartz point group `321`
# does not contain inversion, and the model below does not impose Friedel's law. The
# [Pole Figures of an ODF](ODFPoleFigure.html) page develops this calculation.

# +
center = orientation.byEuler(20 * degree, 30 * degree, 0, 'ZYZ', CS)
odf = unimodalODF(center)
h = Miller(1, 2, 2, CS)

plotPDF(odf, cat(h, -h))
# -

# The two pole figures have different intensity patterns. The ODF distinguishes the
# $(122)$ plane normal from its opposite for this non-Laue point group.

plotPDF(odf, h, antipodal=True)

# With antipodal symmetry imposed, MTEX draws only the upper hemisphere. The omitted lower
# hemisphere is now a point-reflected copy.

# ## Inverse Pole Figures
#
# An inverse pole figure fixes a specimen direction and displays crystal directions. The
# [Inverse Pole Figures of an ODF](ODFInversePoleFigure.html) page explains this
# complementary view.

plotIPDF(odf, cat(vector3d.Y, -vector3d.Y), complete=True, noLabel=True)

# The complete inverse pole figures for Y and -Y differ. Without an antipodal assumption,
# reversing the specimen direction changes the question.

plotIPDF(odf, vector3d.Y, antipodal=True, complete=True, noLabel=True)

# The complete antipodal plot repeats the same crystal-direction pattern on opposite sides
# of the sphere.

# ## Fundamental Sectors
#
# Inverse pole figures are usually reduced to the
# [fundamental sector](FundamentalSector.html). This is the patch of the sphere that
# crystal symmetry leaves inequivalent.

plotIPDF(odf, vector3d.Y)

# Without antipodal symmetry, MTEX must retain the larger fundamental sector of point group
# `321`.

plotIPDF(odf, vector3d.Y, antipodal=True)

# Identifying opposite directions reduces the region further. Every omitted crystal
# direction is equivalent to one inside the smaller plotted sector.

# ## Further Reading
#
# * K. V. Mardia and P. E. Jupp, [Directional Statistics](https://doi.org/10.1002/9780470316979),
#   Wiley, 1999, develops statistical methods for both directional and axial data.
# * [IUCr Online Dictionary of Crystallography: Friedel's law](https://dictionary.iucr.org/Friedel%27s_law)
#   states the diffraction conditions under which opposite reflections have equal
#   intensity.
# * [ASTM E81-96(2024), Standard Test Method for Preparing Quantitative Pole Figures](https://doi.org/10.1520/E0081-96R24)
#   covers quantitative X-ray pole-figure acquisition.
# * H.-J. Bunge, [Texture Analysis in Materials Science: Mathematical Methods](https://doi.org/10.1016/C2013-0-11769-2),
#   Butterworths, English ed., 1982, develops pole figures, inverse pole figures, and ODFs
#   together.

# ## Next
#
# Continue with [Density Estimation](VectorsDensityEstimation.html) to work with c-axes
# from an EBSD map. Crystal axes written as Miller indices are treated in
# [Miller Indices](CrystalDirections.html). The same flag records grain-exchange symmetry
# in [Grain Exchange Symmetry](MisorientationGrainExchangeSym.html).
