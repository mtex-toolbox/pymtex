# # Data Correction
#
# Diffraction counts are not pole densities. The detector also records background
# radiation, and tilting the specimen may reduce the measured intensity through
# defocusing. Isolated bad measurements and an unknown scale introduce further errors.
#
# This page prepares measured pole figures for ODF reconstruction. It assumes the
# pole-figure idea from [Pole Figures](PoleFigureAnalysis.html) and the import checks from
# [Import](PoleFigureImport.html).
#
# A `PoleFigure` behaves like an array of measured values. Pole figures can be selected,
# added and scaled, while individual values can be selected, overwritten or deleted.

import numpy as np
from mtex import (calcODF, cat, correct, degree, isOutlier, max, mean, min, mtexColorbar, mtexdata, normalize, plot, plottingConvention,
                  rotate, rotation, setColorRange, xvector)

# +
plottingConvention.default('y←↑x')
pf = mtexdata('geesthacht')

pf
# -

# ---

# plot the four entries in the imported object
plot(pf)
mtexColorbar()

# ## Splitting signal and background
#
# The display and plot show four entries. The first and third are the two complete
# intensity scans, with 679 specimen directions each. The second and fourth are sparse
# background scans with 16 directions each.
#
# Each background panel is a single radial line of points running from the centre of the
# disc outward: the 16 directions share one azimuth and differ only in polar angle. They
# belong to the complete scan with the same Miller index. A list of positions in brackets
# selects entries of the `PoleFigure` object.

pf_complete = pf[[0, 2]]
pf_background = pf[[1, 3]]

# ## Arithmetic with pole figures
#
# Arithmetic uses the same syntax as arithmetic with numbers. The following weighted sum
# demonstrates addition and scaling without changing `pf`.

pf_weighted = 2 * pf[0] + 3 * pf[2]

# The weights above are arbitrary and the two entries have different Miller indices. This
# is an API example, not a physical correction or an example of an unresolved diffraction
# peak.

# ## Background and defocusing
#
# [correct](PoleFigure.correct.html) applies correction measurements that have already
# been obtained. It does not infer background or defocusing from the measured pole figure.
#
# With `background`, MTEX interpolates each sparse background scan onto the directions of
# its complete scan and subtracts it. Because these background points differ only in
# polar angle, that interpolation is a spline in the polar angle alone. The Geesthacht
# data contain the background measurements needed for this operation.

# +
pf = correct(pf_complete, background=pf_background)

# compare the first scan before and after background subtraction
plot(cat(pf_complete[0], pf[0]), layout=[1, 2])
setColorRange('equal')
mtexColorbar()

correctedI = pf.intensities
print(f'Corrected counts: min {np.min(correctedI):.0f}, mean {np.mean(correctedI):.0f}, max {np.max(correctedI):.0f}')
# -

# The left panel is the raw (104) scan and the right panel is the corrected scan on the
# same colour range. Background subtraction preserves the broad intensity pattern while
# lowering every value. The corrected counts run from 7 to 631, with an arithmetic mean of
# 273.
#
# Defocusing is different: it is a tilt-dependent loss of signal, so the correction is a
# division. Supply a defocusing measurement and, when available, its own background
# measurement:
#
#     pf = correct(pf, background=pf_bg, defocusing=pf_def, defocusingBackground=pf_def_bg)
#
# MTEX subtracts `pf_bg` from `pf`, subtracts `pf_def_bg` from `pf_def`, and divides the
# first result by the second. The correction measurements must match the pole figures
# physically; interpolation only adapts their sampled directions.

# ## Normalization
#
# Pole density is reported in multiples of a random distribution (mrd). Its mean over a
# complete pole figure is 1, so [normalize](PoleFigure.normalize.html) divides each
# complete scan by its quadrature-weighted mean.

# +
pf_normalized = normalize(pf)
plot(pf_normalized)
mtexColorbar()

normalizedI = pf_normalized.intensities
normalizedMean = mean(pf_normalized)
print(f'Normalized mrd: min {np.min(normalizedI):.2f}, max {np.max(normalizedI):.2f}; '
      f'pole-figure means {normalizedMean[0]:.2f} and {normalizedMean[1]:.2f}')
# -

# The pattern has not moved; only the colour scale has changed. Values now run from 0.03
# to 2.27 mrd, and both pole figures have mean 1. They can therefore be compared with
# normalized measurements from another specimen.
#
# MATLAB reports 2.15 for the same data. Its two scans carry different antipodal flags
# after the correction, so their quadrature weights differ although their specimen
# directions are identical. Here both scans are axes, as diffraction data are, and both
# are weighted the same way.

# ## Incomplete pole figures
#
# Direct normalization fails for an incomplete pole figure. The unmeasured part of the
# sphere also carries pole density, and its contribution is exactly what is unknown.
#
# One route is to reconstruct an ODF first. The ODF fills in the unmeasured part, and
# `normalize(pf, odf)` determines the scale against recalculated values at the measured
# directions. ODF reconstruction is explained in
# [ODF Reconstruction](PoleFigure2ODF.html).

# +
odf = calcODF(pf)
pf_normalized_odf = normalize(pf, odf)

plot(pf_normalized_odf)
mtexColorbar()

odfNormalizedMean = mean(pf_normalized_odf)
normalizationDifference = 100 * np.max(np.abs(odfNormalizedMean - 1))
print(f'Largest ODF-based scale difference: {normalizationDifference:.2f} percent')
# -

# These pole figures are complete, so the direct and ODF-based scales differ by at most
# 0.02 percent. Their plots are consequently almost indistinguishable. On incomplete data
# the two procedures need not agree.

# ## Outliers
#
# A single bad measurement can distort a reconstruction because the solver has no reason
# to distrust it. [isOutlier](PoleFigure.isOutlier.html) marks values that disagree with
# the mean of their neighbourhood.
#
# The default threshold is two standard deviations for each pole figure. Review the
# flagged points in the measurement context instead of treating the default as an
# automatic quality criterion.
#
# To make the operation visible, first spoil 100 random measurements.

# +
rng = np.random.default_rng(1)
ind = rng.permutation(pf.intensities.size)[:100]
factor = 3 + rng.random(100)
I = pf.intensities
I[ind] *= factor
pf.intensities = I

plot(pf)
mtexColorbar()
# -

# The spoiled measurements appear as isolated bright dots above the broad texture
# pattern. Deleting selected entries keeps the complement of a logical condition, as it
# does for an ordinary NumPy array.

# +
condition = isOutlier(pf)
nFirstPass = np.count_nonzero(condition)
print(f'Outliers removed on first pass: {nFirstPass}')
pf = pf[~condition]

plot(pf)
mtexColorbar()
# -

# The first pass removes about half of the 100 inserted values. An outlier beside another
# outlier raises the local mean and can hide behind it. Recomputing the condition after
# deletion catches roughly half of the rest.

# +
condition = isOutlier(pf)
nSecondPass = np.count_nonzero(condition)
print(f'Outliers removed on second pass: {nSecondPass}')
pf = pf[~condition]

plot(pf)
mtexColorbar()
# -

# Most isolated bright points have disappeared after the second pass. The remaining
# inserted values were not separated far enough from their local neighbourhood to pass
# this particular threshold.
#
# Deletion removes both the intensity and its specimen direction. It does not replace the
# value by an interpolated estimate.

# ## Any other condition
#
# Nothing about the selection is specific to outliers. Any logical condition on the
# intensities can select values for deletion or replacement.
#
# The next threshold is deliberately artificial: it demonstrates replacement by capping
# counts at 500, not a recommended experimental correction. `min(pf, 500)` clips every
# intensity above 500 to that value.

# +
condition = pf.intensities > 500
nCapped = np.count_nonzero(condition)
nRemaining = pf.intensities.size
print(f'Values capped at 500: {nCapped} of {nRemaining}')
pf = min(pf, 500)

plot(pf)
mtexColorbar()
# -

# The cap affects about a hundred of the remaining measurements. They now share the same
# top colour, flattening the bright parts of the plot and showing why a numerical
# condition needs a physical justification.

# ## Rotating pole figures
#
# A reference frame is the coordinate system in which the specimen directions are
# expressed. If import assigned the wrong specimen frame, fix the import as described in
# [Import](PoleFigureImport.html) whenever possible.
#
# The following example deliberately rotates the measured directions by 100 degrees about
# the x-axis. This moves the data; it is not a change of plotting convention, which only
# controls where axes are drawn.

# +
rot = rotation.byAxisAngle(xvector, 100 * degree)
pf_rotated = rotate(pf, rot)

plot(pf_rotated, antipodal=True)
mtexColorbar()
# -

# [rotate](PoleFigure.rotate.html) leaves the intensities unchanged and applies the
# rotation to their measured directions. The measured cap tips across the equator, and
# `antipodal` folds the part below the equator back into the same disc.
#
# Both panels come out filled edge to edge, and the vertical arcs across them are the
# rotated ring sampling rather than gaps in it. Drop `antipodal` and the (110) panel opens
# over the part of the sphere that was never measured. That is why correcting the specimen
# frame at import is preferable to an avoidable rotation later.

# ## Further reading
#
# * ASTM International,
#   [ASTM E81-96(2024): Standard Test Method for Preparing Quantitative Pole Figures](https://doi.org/10.1520/E0081-96R24).
#   It distinguishes complete, partial and calculated pole figures and describes
#   experimental preparation.
# * H.-J. Bunge,
#   [Texture Analysis in Materials Science: Mathematical Methods](https://doi.org/10.1016/C2013-0-11769-2),
#   Butterworths, English ed., 1982. It gives the classical definitions of pole density,
#   normalization and ODF inversion.
# * A. A. Saleh, V. Q. Vu and A. A. Gazder,
#   [Correcting intensity loss errors in the absence of texture-free reference samples during pole figure measurement](https://doi.org/10.1016/j.matchar.2016.06.018),
#   _Materials Characterization_ 118, 425-430, 2016. It explains background,
#   tilt-dependent intensity loss and reference-sample corrections.
# * D. Chateigner, L. Lutterotti and M. Morales,
#   [Quantitative texture analysis and combined analysis](https://doi.org/10.1107/97809553602060000968),
#   _International Tables for Crystallography_, Volume H, chapter 5.3, 2019. It connects
#   diffraction intensity, instrumental corrections, normalized pole density and the ODF
#   forward model.

# ## Technical Details
#
# An integer, a slice or a list in brackets picks pole figures, a boolean mask picks
# measurements, so MATLAB's `pf({1,3})` is `pf[[0, 2]]` and `pf(condition) = []` is
# `pf = pf[~condition]`. `pf.intensities` is a copy of all intensities in one array;
# changed values go back through the same property. The cap is `min(pf, 500)`, MATLAB's
# `min(pf, 500)` as well. The spoiled measurements are drawn from a seeded generator so
# that the page rebuilds the same; MATLAB's `randperm` is not seeded, and the counts of
# the two passes vary with the draw.
