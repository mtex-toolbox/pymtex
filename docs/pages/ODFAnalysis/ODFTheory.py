# # The Orientation Distribution Function
#
# A texture is a population of orientations. The orientation distribution function (ODF)
# describes that population as a *density* over orientation space, rather than as a list
# of measured orientations.
#
# This page assumes the crystal-to-specimen map introduced in
# [Orientation Theory](DefinitionAsCoordinateTransform.html) and the symmetry-equivalent
# representatives from [Orientation Symmetry](OrientationSymmetry.html).
#
# Density is the word to hold on to. No finite fraction of a continuous distribution sits
# at one exact orientation. An ODF instead says how much material occurs per unit of
# orientation space near that orientation. Values are reported in *multiples of a random
# distribution*, mrd. A uniform texture is 1 mrd everywhere; 9 mrd means nine times the
# random density near that orientation, not nine percent of the material.
#
# A *plotting convention* states how a reference frame is laid out on screen. The
# following convention draws Y upward and X to the right. It does not rotate the specimen
# or change any orientation.

from mtex import (calcDensity, crystalShape, degree, hold, max, mean, mtexColorMap, mtexdata, orientation, plot, plot3d, plotSection,
                  plottingConvention, reduce, volume, zvector)

plottingConvention.default('y↑→x')

# ## From a Map to Orientations
#
# The example is a titanium alloy measured by EBSD on a hexagonal grid. An orientation is
# stored at every indexed measurement point.

# import the data without printing its EBSD summary
ebsd = mtexdata('titanium')

# The map below uses an inverse pole figure key for the specimen Z direction. The overlaid
# hexagonal prisms make the crystal orientation behind selected colours explicit.

# +
# define the habitus of titanium as a sample hexagonal prism
cS = crystalShape.hex(ebsd.CS)

# keep a regular spatial subsample for the crystal and orientation plots
ebsdPlot = reduce(ebsd, 4)

# plot coloured orientations
plot(ebsd, ebsd.orientations, ipfDirection=zvector, micronbar=False, figSize='large')

# overlay the orientations as rotated hexagonal prisms
hold(True)
plot(ebsdPlot, 40 * cS)
hold(False)
# -

# Neighbouring points with similar colours have similar orientations, and the prisms show
# the corresponding lattice directions. This spatial arrangement is useful for EBSD
# analysis, but an ODF deliberately discards it.
#
# Keep only the orientations and draw the regular subsample in Bunge Euler angle space.

oriPlot = ebsdPlot['indexed'].orientations
plot(oriPlot, 'Euler')

# The same measurements now form a point cloud. Dense clusters and sparse regions reveal
# the texture, while their former positions in the map are no longer present.
#
# Orientation space is curved, so no drawing of it is canonical. The alternatives are
# compared in [3D Plots](OrientationVisualization3d.html).

# ## From Orientations to a Density
#
# [calcDensity](rotation.calcDensity.html) places a kernel at every orientation and adds
# the kernels. The full indexed pixel list is used here, so equal-area pixels have equal
# statistical weight.

oriData = ebsd['indexed'].orientations
odf = calcDensity(oriData)
odf

# The printed summary identifies the harmonic representation and its bandwidth. Other ODF
# representations share the same [SO3Fun](SO3FunConcept.html) interface.
#
# This conversion is an estimate, not a unique rewriting of the points. Its most important
# choice is the kernel halfwidth. A small halfwidth keeps sample-scale peaks. A large one
# can merge real components; see [Density Estimation](DensityEstimation.html).
#
# The weighting also answers a physical question. Pixel orientations describe mapped
# area, whereas equally weighted grain mean orientations describe the fraction of grains.
# The `weights` option can make grain areas supply the weights when area fraction is
# wanted instead.

# ## Reading Density and Volume
#
# An ODF is a function, so it can be evaluated at an orientation that was never measured.
# Here the orientation has zero Bunge Euler angles.

ori0 = orientation.byEuler(0, 0, 0, ebsd.CS)
valueAtOri0 = odf.eval(ori0)
valueAtOri0

# The value is 0.8166 mrd, slightly less common than the random density of 1 mrd. The
# strongest density in this texture is about 9.0003 mrd.

maxValue, oriMax = max(odf)
maxValue

# The mean is 1 mrd by construction, whatever the texture. This is why an isolated ODF
# value only has meaning relative to the random density.

meanValue = mean(odf)
meanValue

# A volume fraction comes from integrating over a region. The region below contains every
# orientation within $10^\circ$ of the strongest one.

volumeFraction = volume(odf, oriMax, 10 * degree)
volumeFraction

# The result is 0.0235, or about 2.35 percent, not nine percent. The 9 mrd value is a
# density, and the orientation region around the maximum is small.

# ## Looking at an ODF
#
# An ODF is a function on a three-dimensional curved space. A 3-D plot can show how the
# measured orientations relate to its high-density regions.

# +
plot3d(odf, 'Euler')
hold(True)
plot(oriPlot, 'Euler', MarkerEdgeColor='k')
hold(False)
# -

# The black point clusters pass through the coloured density lobes. This is the visual
# connection between the discrete measurements and their ODF.
#
# Equal-looking boxes in Euler angle space do not represent equal volumes of orientation
# space. Use this plot to locate components, not to estimate their volume by eye.
#
# For hexagonal crystals, [sigma sections](SigmaSections.html) often give a more direct
# view.

plotSection(odf, 'sigma')
mtexColorMap('LaboTeX')

# Position within a section fixes the c-axis direction, while the section angle records
# the remaining rotation about it. The panels separate the components. They do not invite
# a volume estimate from distorted boxes.
#
# Both plots evaluate the ODF itself. A [pole density function](ODFPoleFigure.html)
# instead integrates the ODF along orientation fibres and therefore loses information.

# ## The Maths Behind the Normalization
#
# For a region $A$ of orientation space, its material volume fraction is
#
# $$\frac{V(A)}{V} = \int_A \mathrm{odf}(g)\,\mathrm{d}g.$$
#
# In differential form this is
#
# $$\mathrm{odf}(g) = \frac{1}{V}\frac{\mathrm{d}V(g)}{\mathrm{d}g}.$$
#
# Here $\mathrm{d}g$ is the normalized volume measure on the symmetry-reduced orientation
# space. Integrating 1 over the whole space gives 1, so the uniform ODF is
# $\mathrm{odf}(g)=1$ everywhere.
#
# Crystal and specimen symmetry identify equivalent numerical rotations. MTEX carries both
# symmetries with the ODF. Equivalent copies are not counted as distinct material during
# evaluation or integration.

# ## References
#
# * H.-J. Bunge, [Texture Analysis in Materials Science: Mathematical Methods](https://doi.org/10.1016/C2013-0-11769-2),
#   Butterworths, English ed., 1982. It develops orientation distributions, symmetry and Euler sections.
# * U. F. Kocks, C. N. Tomé and H.-R. Wenk, [Texture and Anisotropy](https://books.google.com/books?id=vkyU9KZBTioC),
#   Cambridge University Press, 2000. It connects quantitative texture to anisotropic material properties.
# * H. Schaeben, [Towards statistics of crystal orientations in quantitative texture analysis](https://doi.org/10.1107/S0021889892009270),
#   _Journal of Applied Crystallography_ 26 (1993), 112--121. It introduces kernel density estimation for orientations.

# ## Next
#
# Continue with [Density Estimation](DensityEstimation.html) for kernels and halfwidth
# selection. [Modeling](ODFModeling.html) builds known ODFs, and
# [Properties](ODFCharacteristics.html) extracts texture index, entropy and volume
# fractions. Diffraction projections begin with [Pole Figures](ODFPoleFigure.html). The
# next chapter continues with [pole-figure reconstruction](PoleFigureAnalysis.html).

# ## Technical Details
#
# `max(odf)` returns
# the pair of the value and the orientation, as MATLAB's two outputs do.
