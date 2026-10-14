# # Plotting Pole Figures
#
# A measured pole figure stores an intensity at every sampled specimen direction for one
# family of crystal-plane normals. [plot](PoleFigure.plot.html) places those samples on a
# projection of the specimen hemisphere. It does not turn the measurements into a
# continuous function.
#
# This page assumes the pole-figure idea introduced in
# [Pole Figures](PoleFigureAnalysis.html) and a `PoleFigure` object imported as in
# [Import](PoleFigureImport.html). Review [Miller Indices](CrystalDirections.html) if the
# plane labels are unfamiliar.

from mtex import *

# +
plottingConvention.default('y↑→x')
pf = mtexdata('ptx')

pf
# -

# The display reports three measured pole figures, for (104), (110), and (202). Each one
# contains a regular $72 \times 17$ grid of specimen directions. The convention above
# draws y upwards and x to the right. It makes the layout explicit instead of inheriting
# the session default. See [Axes Alignment](AxesAlignment.html) for how that convention
# relates to the specimen frame; it does not rescale the intensities.

# ## Measured directions as markers
#
# With no plot-type option, each measured direction becomes a circle coloured by its
# stored intensity.

plot(pf)
mtexColorMap('parula')
mtexColorbar(title='intensity')

# The circles expose the sampling grid as well as the intensity pattern. Each panel has
# its own colour range, so the same colour does not yet mean the same value in all three
# panels.
#
# MTEX estimates a marker size from the angular spacing and the figure size. The
# `markerSize` option replaces that estimate when it hides gaps or makes neighbouring
# circles overlap.

plot(pf, markerSize=2)
mtexColorMap('parula')
mtexColorbar(title='intensity')

# The smaller circles separate neighbouring samples more clearly. Their size has no
# physical meaning, and the intensity pattern is unchanged.

# ## Contour plots
#
# Contours need values between the measured directions. On a regular grid, `contourf`
# lets the plotting routine interpolate the stored intensities. This is useful for a quick
# visual summary.

plot(pf, contourf=True)
mtexColorMap('parula')
mtexColorbar(title='intensity')

# The isolated spots now appear as continuous patches. That continuity is interpolation,
# not additional measurement, and it can be misleading on an irregular grid. When a smooth
# pole-density function is required, reconstruct an ODF and recalculate the pole figure as
# shown below.

# ## One colour range for all
#
# Comparing pole figures by eye only works when they share a colour range.
# [setColorRange](setColorRange.html) with `'equal'` gives all axes of the figure the
# union of their ranges. A single colorbar then applies to every panel.

mtexColorbar()  # remove the colorbars
setColorRange('equal')
mtexColorbar(title='intensity')  # add a single colorbar

# The colours can now be compared directly. The (202) figure reaches 15.8, whereas (104)
# stops at 9.8. The same orientation population projects differently for different
# crystal-plane normals, which is why several pole figures provide more information than
# one.
#
# The colourbar shows the values stored in `pf`. Do not call them multiples of a random
# distribution (mrd), or compare maxima as material fractions, until the pole figures have
# been corrected and normalized.
#
# Some stored intensities are negative, down to -1.8. Diffracted intensity cannot be
# negative, so the background correction has subtracted too much.
# [Data Correction](PoleFigureCorrection.html) covers that problem and the normalization
# step.

# ## Recalculated pole figures
#
# An [orientation distribution function](ODFAnalysis.html) (ODF) is a density over crystal
# orientations. At one point of a recalculated pole figure, MTEX integrates that density
# over every orientation that carries the chosen crystal-plane normal into the
# corresponding specimen direction.
#
# The result can be calculated for any crystal-plane normal, including one that was never
# measured. Reconstructing the ODF from measured pole figures is the subject of
# [ODF Estimation](PoleFigure2ODF.html).

odf = calcODF(pf)

# [plotPF](SO3Fun.plotPDF.html) recalculates the same three pole figures. Diffraction
# identifies a plane normal with its opposite, so `antipodal` folds the two directions
# into the same hemisphere.

plotPF(odf, pf.h, antipodal=True)
mtexColorMap('parula')
setColorRange('equal')
mtexColorbar(title='mrd')

# These are smooth because the ODF is, not because the data were. Comparing them with the
# measured figures is the standard first check on a reconstruction, and what to compare is
# where the maxima sit.
#
# Their heights need not agree. Read the two colour bars here: the measured peaks are two
# to three times the recalculated ones, because the finite width of the reconstruction
# kernel spreads a sharp measured maximum out.
#
# A displaced or missing maximum points to a reconstruction or data problem. Continue with
# [ODF Estimation](PoleFigure2ODF.html) for quantitative error measures and
# [Data Correction](PoleFigureCorrection.html) for systematic defects. A visual match does
# not make the reconstructed ODF unique; the ambiguity is explained in
# [The Ghost Effect](PoleFigure2ODFAmbiguity.html).

# ## Further reading
#
# * B. B. He, [Pole density and pole figures](https://onlinelibrary.wiley.com/iucr/itc/Ha/ch2o5v0001/sec2o5o4o2o1/),
#   _International Tables for Crystallography_, Volume H, section 2.5.4.2.1, 2019. It
#   defines the spherical directions, projection, and pole-density plot used here.
# * ASTM International,
#   [ASTM E81-96(2024): Standard Test Method for Preparing Quantitative Pole Figures](https://doi.org/10.1520/E0081-96R24).
#   It distinguishes measured complete and partial pole figures from calculated pole
#   figures.
# * H.-J. Bunge,
#   [Texture Analysis in Materials Science: Mathematical Methods](https://doi.org/10.1016/C2013-0-11769-2),
#   Butterworths, English ed., 1982. It gives the classical relation between pole figures
#   and orientation distributions.

# ## Technical Details
#
# `mtexColorbar` toggles, as MATLAB's does: it removes the colour bars of the figure when
# it has some and adds them otherwise, which is why the call before `setColorRange` takes
# the three panel bars away. `calcODF` is silent by default here; `verbose=True` prints
# MATLAB's iteration trace.
