# # Annotations
#
# A colour-coded pole figure needs a colour scale before its values can be
# read. Markers, labels, legends, and grids then make particular features
# identifiable. Together these annotations turn a finished plot into a figure
# that another reader can interpret and compare.
# Without marked directions, only a reader who already knows the figure's
# contents can identify them.
#
# This page starts with numerical colour scales. It then marks known
# directions and orientations, identifies plotted objects, and adds an angular
# grid. Each operation modifies an existing figure.

# +
import matplotlib.pyplot as plt

from mtex import *

plottingConvention.default('y↑→x')
# -

# ## Colour scales for numerical data
#
# [mtexColorbar](mtexColorbar.html) differs from matplotlib's `colorbar` in
# that it addresses the whole MTEX figure. Every axis in a multi-plot gets a
# colourbar. The example uses two model components so that the plotted pole
# densities have recognisable maxima.

# +
# define a two-component model ODF
cs = crystalFrame('-3m')

mod1 = orientation.byEuler(110 * degree, 30 * degree, 80 * degree, cs)
mod2 = orientation.byEuler(310 * degree, 70 * degree, 40 * degree, cs)
odf = 0.7 * unimodalODF(mod1) + 0.3 * unimodalODF(mod2)

# plot two pole figures
plotPDF(odf, Miller([[1, 0, 0], [1, 1, 1]], cs))

# add a colourbar to each pole figure
mtexColorbar()
# -

# Notice that the two colourbars have different ranges. Equal colours do not
# yet represent equal pole densities in the two panels. Calling
# `mtexColorbar` again removes the colourbars. The `location` option places
# the replacements below the panels, while `title` names their unit. Pole
# density is measured in multiples of a random distribution, abbreviated mrd.

# +
# remove the vertical colourbars
mtexColorbar()

# add horizontal colourbars
mtexColorbar(location='southoutside', title='mrd')
# -

# For a direct comparison, first give every axis the same colour range. One
# colourbar can then describe the entire figure. The same colour now means the
# same pole density in both panels. See [Colour Coding](ColorMaps.html) for
# choosing the colours themselves.

mtexColorbar()               # remove the colourbars
setColorRange('equal')       # use one colour range for all panels
mtexColorbar()               # create one colourbar for the figure

# ## Marking directions and orientations
#
# [annotate](annotate.html) adds to an existing figure without disturbing
# it. It needs no `hold(True)` and does not change the colour range. First mark
# a specimen direction on both pole figures above. A specimen direction is a
# direction expressed in the specimen frame. The text passed through `label`
# is arbitrary; here it names the vector drawn at $(1,1,1)$.

annotate(vector3d(1, 1, 1), label=['(111)'], backgroundColor='w')

# In an inverse pole figure, a crystal direction is the natural annotation.
# It is a direction expressed in the crystal frame. The `all` option draws
# every symmetrically equivalent direction. The `labeled` option writes the
# Miller indices beside those markers.

# +
plotIPDF(odf, [xvector, zvector], antipodal=True, marginx=10)
mtexColorMap('white2black')

annotate(Miller([[2, -1, -1, 0], [2, -1, -1, 1]], cs), all=True, labeled=True, backgroundColor='yellow')
# -

# The axes of an inverse pole figure are the fundamental sector of the
# crystal symmetry, so most symmetrically equivalent copies of a direction
# fall outside them. Each supplied index is labelled where it lands inside
# the sector.
#
# A whole orientation can also be marked. MTEX draws it wherever that
# orientation appears. It places one marker per pole in every axis. The next
# figure marks the two components used to construct the model ODF.

# +
plotIPDF(odf, [xvector, zvector], antipodal=True)
mtexColorMap('white2black')
annotate(mod1, marker='s', markerSize=6, markerFaceColor='r', label='A', color='w')

annotate(mod2, marker='s', markerSize=6, markerFaceColor='g', label='B')

gcm().drawNow(figSize='normal')
# -

# The red squares sit on the maxima, while the green ones mark the weaker
# component. These known orientations provide a direct check on the plot.
#
# The same annotation works on ODF sections. In a section, an orientation is
# represented by one point rather than by one pole in every axis.

# +
plotSection(odf, 'sigma')
mtexColorMap('white2black')
annotate(mod1, label='A', textColor='r', markerSize=15, markerEdgeColor='r', markerFaceColor='none')

annotate(mod2, label='B', textColor='b', markerSize=15, markerEdgeColor='b', markerFaceColor='none')
# -

# The outlined labels identify the two component positions without covering
# the density underneath. The same annotation also works on a scatter plot of
# individual orientations. It shows which model components the sampled point
# cloud lies around.

ori = odf.discreteSample(200)
scatter(ori)
annotate(mod1, markerSize=10, markerEdgeColor='r', markerFaceColor='r')
annotate(mod2, markerSize=10, markerEdgeColor='g', markerFaceColor='g')

# ## Legends
#
# A colourbar explains numerical values encoded by colour. A legend instead
# identifies plotted objects by a label and symbol. Anything plotted with a
# `displayName` enters the legend, while everything else stays out. In a
# multi-plot figure, combine `displayName` with `add2all`. The object and
# its legend entry then appear on every panel.

# +
plotPDF(odf, Miller([[1, 0, 0], [1, 1, 1]], cs))
plot(ori, markerFaceColor='k', markerEdgeColor='black', add2all=True, displayName='randomSample')

f = fibre(Miller(1, 1, -2, 1, cs), vector3d.Y)
plot(f, color='red', lineWidth=2, add2all=True, displayName='fibre')

legend('show')
# -

# The black points and red curve now have the same names in both pole figures.
# See [Legends](Legends.html) for legend placement and formatting.
#
# A line plot also needs a legend when several curves share one axis. Here the
# harmonic coefficients of the two-component ODF are compared with those of a
# fibre ODF through their power at each harmonic degree.

# +
plt.close('all')
plotSpectrum(FourierODF(odf, 32), displayName='Two-component ODF', figSize='small')
hold(True)
fodf = fibreODF(Miller(1, 0, 0, cs), zvector)
plotSpectrum(FourierODF(fodf, 32), displayName='Fibre ODF')
hold(False)
legend('show')
# -

# The fibre ODF decays faster, as a function with a rotational symmetry
# must. It has fewer degrees of freedom to spend at each harmonic degree. The
# legend makes it possible to assign that decay to the correct curve.
#
# ## A spherical grid
#
# A grid of latitude and longitude lines makes the projection legible and
# lets angles be read from the figure. The `grid` flag switches it on, and
# `grid_res` sets the angular spacing.

plotPDF(odf, [Miller(1, 0, 0, cs), Miller(0, 0, 1, cs)], grid=True, grid_res=15 * degree, antipodal=True)
mtexColorMap('white2black')

# At 15 degrees the lines are close enough for approximate measurements and
# far enough apart not to compete with the pole-density pattern. The grid is
# part of the spherical projection; it does not change the sampled data.
#
# ## References
#
# * S. R. Midway,
#   [Principles of Effective Data Visualization](https://doi.org/10.1016/j.patter.2020.100141),
#   _Patterns_ 1 (2020), 100141, explains how labels, legends,
#   and consistent visual scales support comparisons between plots.
#
# ## Next
#
# Once the figure carries the information needed to read it, continue with
# [Exporting Figures](PlottingExport.html). It explains how figure size,
# renderer, and file format affect the exported result.
#
# ## Technical Details
#
# The 200 orientations of `discreteSample` are random draws, so the point clouds differ from
# MATLAB's in their details. The MATLAB reference run of the spectrum cell stopped in the
# `wignerTrafoAdjoint` MEX file (a static TLS allocation of `libgomp`), so that figure is
# compared with the MATLAB script alone.
