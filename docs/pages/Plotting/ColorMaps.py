# # Color Mapping
#
# A colour range is the numerical interval represented by a plot's colours.
# Its lower and upper limits receive the end colours of the colormap. Two
# plots can be compared by colour only when they use the same range and the
# same colormap.
#
# This page shows how MTEX chooses the range, how to fix it, and how a
# colormap translates values within that range into colours. Legends for
# discrete objects and colour keys for directions were distinguished on
# [Legends](Legends.html).

# +
import matplotlib.pyplot as plt
import numpy as np

from mtex import *

plottingConvention.default('y↑→x')
# -

# ## Create two quantities to compare
#
# An [orientation distribution function (ODF)](ODFAnalysis.html) describes
# the relative frequency of crystal orientations. This model ODF supplies
# two simulated [pole figures](PoleFigureAnalysis.html) whose densities can
# be compared.

# +
cs = crystalFrame('-3m')
odf = fibreODF(Miller(1, 1, 0, cs), zvector)
pf = calcPoleFigure(odf, [Miller(1, 0, 0, cs), Miller(1, 1, 1, cs)], equispacedS2Grid(points=500, antipodal=True))

odf
# -

# ## The default range is tight and per axis
#
# Without a `colorRange` option, MTEX uses `'tight'`. Each axis spans the
# range of its own data. This uses the available colours fully, but it does
# not guarantee that colours are comparable between axes.

plt.close('all')
plot(pf)
mtexColorbar()

# Read the two colour bars before comparing the patterns. The $(100)$ panel
# reaches approximately 3.5 multiples of a uniform distribution (mrd),
# whereas the $(111)$ panel reaches approximately 2.1 mrd. The same colour
# therefore denotes a different pole density in each panel. Nothing in the
# maps alone warns about that mismatch.
#
# ## Use one range for one figure
#
# `colorRange='equal'` chooses the smallest common range containing the
# tight range of every axis in the figure.

plot(pf, colorRange='equal')
mtexColorbar()

# One colour bar now serves the figure, and both panels run to approximately
# 3.5 mrd. The $(111)$ panel is visibly paler. The common range reveals its
# lower density, which the separate tight ranges hid.
#
# ## Fix one range across separate figures
#
# Separate figures cannot discover each other's limits. State the same
# numerical range in each plotting command. Here the original ODF and a
# mixture containing half uniform ODF both use the interval from 0 to 4 mrd.

# +
plt.close('all')
plotPF(odf, [Miller(1, 0, 0, cs), Miller(1, 1, 1, cs)], colorRange=[0, 4], antipodal=True)
mtexColorbar()

odfMixed = 0.5 * odf + 0.5 * uniformODF(cs)
plotPF(odfMixed, [Miller(1, 0, 0, cs), Miller(1, 1, 1, cs)], colorRange=[0, 4], antipodal=True)
mtexColorbar()
# -

# Mixing with the uniform ODF halves the density contrast above 1 mrd.
# Because the figures share a range, the second texture looks weaker. Tight
# ranges would map these two affinely related fields to the same colours and
# make them look identical.
#
# ## Use explicit contour levels
#
# A contour level is a value at which a contour line or colour boundary is
# drawn. Explicit levels take the place of an explicit colour range for a
# contour plot. Reusing the levels makes separate contour plots comparable.

plt.close('all')
plotPF(odf, [Miller(1, 0, 0, cs), Miller(1, 1, 1, cs)], contourf=np.arange(0, 6), antipodal=True)
mtexColorbar()

# ## Change the range after plotting
#
# [setColorRange](setColorRange.html) adjusts a figure that has already
# been drawn. This is convenient when the useful limits become clear only
# after inspecting the data.

setColorRange([0.38, 3.9])

# The colour bar now spans 0.38 to 3.9 mrd. Values outside that interval use
# an end colour. Existing contour boundaries stay at their original levels;
# changing the colour range does not recompute the contours.
#
# ## Use a logarithmic scale
#
# A sharp texture puts most values near zero and a few at much larger
# values. A linear scale can then show one small bright area against an
# almost empty background. `logarithmic` spreads the positive low values
# across more colours. Its lower colour-range limit must be positive.

plt.close('all')
plotPF(odf, [Miller(1, 0, 0, cs), Miller(1, 1, 1, cs)], antipodal=True, logarithmic=True)
setColorRange([0.01, 12])
mtexColorbar()

# Weak parts of the pole figures now show structure. The colour bar is no
# longer linear, so equal distances in colour no longer represent equal
# differences in density. That loss of linear distance is the trade-off for
# making weak structure visible.
#
# ## Choose a colormap
#
# A colormap is the ordered set of colours assigned across the colour range.
# [mtexColorMap](mtexColorMap.html) sets it for a figure. MTEX supplies
# `white2black`, `blue2red`, and `LaboTeX` in addition to matplotlib's colormaps.

plotPF(odf, [Miller(1, 0, 0, cs), Miller(1, 1, 1, cs)], antipodal=True)
mtexColorMap('white2black')
mtexColorbar()

# Colormap choice is not decoration. A monotone map such as `white2black`
# is an honest default for a density, which has no natural middle. A
# diverging map such as `blue2red` suits a quantity with a meaningful middle,
# such as signed curvature or the difference between two pole figures. Its
# neutral colour lies at the middle of the range. Set that range symmetrically,
# or the neutral colour marks a meaningless value.
#
# ## Use different colormaps in one figure
#
# Without an axes handle, `mtexColorMap` changes every axis in the figure.
# Pass an axes handle to colour only that axis. Independent colormaps are
# appropriate when the axes show different quantities rather than repeated
# views of one quantity.

# +
mtexFig = newMtexFigure()

v = vector3d.rand(100)

for cm in ['hot', 'cool', 'parula']:
  nextAxis()
  plot(v, smooth=True, grid=True, grid_res=90 * degree, upper=True)

  mtexColorMap(mtexFig.gca, cm)
  mtexTitle(cm)

mtexColorbar()
# -

# These are three plots of the same random directions, yet each colormap
# gives a different visual impression. Every panel gets a colour bar of its
# own, because one bar carries a single colormap and would describe only one
# of the three axes. That is the cost of using several colour mappings in
# one figure.
#
# ## References
#
# * S. R. Midway,
#   [Principles of Effective Data Visualization](https://doi.org/10.1016/j.patter.2020.100141),
#   _Patterns_ 1 (2020), 100141, explains how colour scales
#   support honest comparisons between plots.
#
# ## Next
#
# Continue with [Contour Plots](ContourPlots.html) to choose filled or line
# contours and to apply the levels introduced here.
#
# ## Technical Details
#
# MATLAB draws the markers of `plot(pf)` in its default `parula` map; the port uses MTEX's
# `WhiteJet` for every plot coloured by values. The 100 random directions of the last figure
# are not MATLAB's.
