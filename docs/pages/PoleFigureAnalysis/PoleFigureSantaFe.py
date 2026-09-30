# # The Santa Fe Benchmark
#
# The Santa Fe orientation distribution function (ODF) is a model texture agreed on at a
# workshop in Santa Fe so that pole-figure inversion methods could be compared on the same
# problem. An ODF is a density over orientation space. Because this one is known exactly,
# a reconstruction can be judged by what it recovers rather than only by how well it fits
# the data.
#
# This page assumes the forward experiment from
# [Simulating Pole Figure Data](PoleFigureSimulation.html) and the inverse problem from
# [ODF Estimation](PoleFigure2ODF.html). The origin of the ghost effect is explained in
# [The Ghost Effect](PoleFigure2ODFAmbiguity.html). Here the complete controlled
# experiment is used to measure how much
# [ghost correction](PoleFigure2ODFGhostCorrection.html) recovers.
#
# A plotting convention states how the specimen reference frame is laid out on screen. The
# benchmark uses Y upward and X to the right. This convention does not rotate the specimen
# or change the ODF.

# +
import matplotlib.pyplot as plt
import numpy as np
from mtex import *

plottingConvention.default('y↑→x')
# -

# ## The known model
#
# [SantaFe](SantaFe.html) has cubic crystal symmetry and orthorhombic specimen symmetry.
# It combines a 0.73 mrd uniform background with one component containing 27 percent of
# the volume. Values are in multiples of a random distribution (mrd), so a uniform ODF has
# the value 1 mrd everywhere; see [ODF Theory](ODFTheory.html).

odf = SantaFe()

# ## Simulate noisy pole figures
#
# Four lattice directions are sampled on an antipodally symmetric specimen grid with 5
# degree spacing.

# +
# crystal directions
h = Miller([[1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 1, 1]], odf.CS)

# specimen directions
r = equispacedS2Grid(resolution=5 * degree, antipodal=True)

# pole figures
pf = calcPoleFigure(odf, h, r)
# -

# [noisepf](PoleFigure.noisepf.html) draws Poisson counts. At a direction with pole
# density $P_h(r)$, the mean count is $100P_h(r)$ in this example; 100 is an intensity
# scale, not the same mean at every point. No background is added.

# +
pf = noisepf(pf, 100)

plot(pf, markerSize=5)
mtexColorMap('LaboTeX')
# -

# The four panels retain the model's broad pattern, while individual grid values
# fluctuate. Relative fluctuations are most conspicuous in weak regions because Poisson
# noise is large compared with a small signal.

# ## Reconstruct with and without ghost correction
#
# [calcODF](PoleFigure.calcODF.html) applies ghost correction by default. The second call
# disables it explicitly. Solver output is suppressed here because
# [ODF Estimation](PoleFigure2ODF.html) explains the iteration history.

rec = calcODF(pf)
rec2 = calcODF(pf, ghostCorrection=False)

# ## Compare fit with recovery
#
# Two different comparisons are needed. The ODF error from
# [calcError](SO3Fun.calcError.html) is half the mean absolute difference on a 5 degree
# orientation grid. For normalized ODFs, it is the fraction of material that would have to
# move through orientation space to turn one distribution into the other.
#
# The mean pole-figure residual instead compares each function with the noisy data. It is
# the regularised relative error used by [calcError](PoleFigure.calcError.html), averaged
# over the four pole figures. The extrema give a second reading of the reconstructed ODFs.

# +
odfError = [0, calcError(rec, odf, resolution=5 * degree), calcError(rec2, odf, resolution=5 * degree)]

meanPoleFigureResidual = [np.mean(calcError(pf, odf)), np.mean(calcError(pf, rec)), np.mean(calcError(pf, rec2))]

minimumMrd = [min(odf)[0], min(rec)[0], min(rec2)[0]]
maximumMrd = [max(odf)[0], max(rec)[0], max(rec2)[0]]

print(f'{"":24s}{"odfError":>10s}{"meanPFResidual":>16s}{"minimumMrd":>12s}{"maximumMrd":>12s}')
for name, a, b, c, d in zip(['model', 'withGhostCorrection', 'withoutGhostCorrection'], odfError, meanPoleFigureResidual, minimumMrd, maximumMrd):
  print(f'{name:24s}{a:10.4f}{b:16.4f}{c:12.4f}{d:12.4f}')
# -

# The corrected reconstruction has an ODF error of about 0.045, so between four and five
# percent of the volume is in the wrong place. The uncorrected error is about 0.10, more
# than twice as large. All three mean pole-figure residuals round to 0.05, and the
# uncorrected reconstruction fits the noisy data as well as the model itself. A good fit to
# the projections therefore does not prove that an ODF is unique or true.
#
# The counting noise is drawn afresh on every run, so these numbers move in the third
# digit. The ordering of the two reconstructions does not.

# ## Where the corrected reconstruction misses the data
#
# [plotDiff](PoleFigure.plotDiff.html) shows the regularised relative difference at every
# simulated measurement direction.

plotDiff(pf, rec)

# The residuals are scattered rather than concentrated in one coherent patch. That pattern
# is consistent with the counting noise added above. With measured data it would be a
# diagnostic clue, not proof that noise is the only cause.

# ## Recalculated pole figures
#
# The pole figures recalculated from the corrected ODF reproduce the broad maxima in the
# noisy simulation.

plotPF(rec, pf.h, antipodal=True)

# Their smoothness is expected: the reconstruction uses finite-width kernels and is not
# intended to reproduce every random fluctuation. This agreement is necessary, but the
# table above shows why it is not a sufficient validation when the true ODF is unknown.

# ## Read the three ODFs
#
# First plot the corrected reconstruction in Euler-angle sections.

plot(rec, sections=18, contourf=True, fontSize=10, verbose=False, figSize='large', minmax=True)
mtexColorMap('white2black')

# The known model provides the reference.

plot(odf, sections=18, contourf=True, fontSize=10, verbose=False, figSize='large', minmax=True)
mtexColorMap('white2black')

# Finally, plot the reconstruction without ghost correction.

plot(rec2, sections=18, contourf=True, fontSize=10, verbose=False, figSize='large', minmax=True)
mtexColorMap('white2black')

# The components occupy the right places in both reconstructions. Read the `minmax` labels
# instead of comparing independently scaled contour shades. The model peaks at 5.0 mrd and
# the corrected reconstruction a little below it. Its minimum is about 0.6 mrd against the
# model background of 0.73 mrd. Without ghost correction the minimum falls to about 0.2
# mrd: the reconstruction digs holes in the background to pay for intensity missing from
# the peaks.

# ## Read the harmonic spectrum
#
# [plotSpectrum](SO3Fun.plotSpektra.html) groups the magnitude of the harmonic coefficients
# by degree. This is the most direct view of the even--odd defect.

# +
plt.close('all')
plotSpectrum(odf, bandwidth=32, lineWidth=2, figSize='small')
hold(True)
plotSpectrum(rec, bandwidth=32, lineWidth=2)
plotSpectrum(rec2, bandwidth=32, lineWidth=2)
plt.legend(['true ODF', 'with ghost correction', 'without ghost correction'])
hold(False)
# -

# The uncorrected curve zig-zags: its odd degrees lie below the true ones, while its even
# degrees follow them. That is the ghost effect in harmonic space. The corrected curve
# follows the model much more smoothly because ghost correction estimates information that
# the pole figures do not determine.

# ## Further reading
#
# * K. Pawlik, J. Pospiech and K. Lücke,
#   [The development of a new direct method of ODF reproduction from pole figures and its testing with the help of model functions](https://labosoft.com.pl/download/adcmethod.htm),
#   in J. S. Kallend and G. Gottstein (eds.), _ICOTOM 8_, The Metallurgical Society,
#   105--110, 1988. This is the model-function comparison presented at the Santa Fe
#   conference.
# * S. Matthies and G. W. Vinel,
#   [On the reproduction of the orientation distribution function of texturized samples from reduced pole figures using the conception of a conditional ghost correction](https://doi.org/10.1002/pssb.2221120254),
#   _physica status solidi (b)_ 112, K111--K114, 1982. This paper introduces the
#   conditional ghost-correction idea.
# * R. Hielscher and H. Schaeben,
#   [A novel pole figure inversion method: specification of the MTEX algorithm](https://doi.org/10.1107/S0021889808030112),
#   _Journal of Applied Crystallography_ 41, 1024--1037, 2008. This paper specifies the
#   estimator and numerical reconstruction used here.

# ## Next
#
# Continue with [the Dubna example](PoleFigureDubna.html) to apply the same validation
# sequence to measured neutron pole figures. Once an ODF has been validated,
# [ODF Analysis](ODFAnalysis.html) introduces its properties and derived quantities.

# ## Technical Details
#
# `min(odf)` and `max(odf)` return the pair of the extreme value and the orientation where
# it is reached, so the table reads their first element. MATLAB's `table` is a printed
# loop. The Poisson draw of `noisepf` is not seeded, as MATLAB's is not, so the errors
# vary in the third digit between builds.
