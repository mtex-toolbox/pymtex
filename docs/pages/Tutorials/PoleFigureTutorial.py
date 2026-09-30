# # Pole Figure Tutorial
#
# X-ray, synchrotron and neutron diffraction measure many crystals at once. For each
# selected lattice plane, a pole figure records intensity over specimen directions. Each
# value combines all crystal orientations that place that plane normal in the measured
# direction.
#
# This tutorial imports four pole figures, corrects their intensities, reconstructs an
# orientation distribution function (ODF), and checks the reconstruction against the
# measurements. New MTEX users may first want [General Concepts](GeneralConcepts.html).
# Miller indices are introduced in [Miller Indices](CrystalDirections.html), and specimen
# symmetry in [Specimen Symmetry](SpecimenSymmetry.html).

import numpy as np

from mtex import *

# ## Describe the experiment
#
# The import wizard, `importWizard()`, asks for scientific inputs that a numeric file may
# not contain, then writes a reproducible script.
# [Import Pole Figure Data](PoleFigureImport.html) explains those choices in detail.
#
# The point group of this ZnCuTi phase is `6/mmm`. The lattice parameters and alignment
# options state how its lattice is expressed in the crystal frame. That alignment belongs
# to the frame, not to the symmetry itself.

CS = crystalFrame('6/mmm', [2.633, 2.633, 4.8], 'X||a*', 'Y||b', 'Z||c')

# Specimen symmetry describes a physical invariance of the specimen. Choosing `1` imposes
# no such invariance on this reconstruction.

# +
SS = specimenFrame('1')

# plotting convention: z out of the screen, x pointing north
plottingConvention.default('y←↑x')
# -

# The first four files contain the specimen measurements. The second four contain
# measurements of a texture-free reference specimen made with the same instrument.

# +
files = mtexdatafile('zncuti')

# ZnCuTi_Wal_50_5x5_PF_002_R.UXD, ..._100_R, ..._101_R, ..._102_R
fname = files[:4]

# ZnCuTi_defocusing_PF_002_R.UXD, ..._100_R, ..._101_R, ..._102_R
fnameDef = files[4:]
# -

# ## Match files to lattice planes
#
# Each Miller index must describe the reflection in the file at the same position. A wrong
# assignment still produces an ODF, but it gives the reconstruction the wrong physical
# measurement model.

h = [Miller(0, 0, 2, CS), Miller(1, 0, 0, CS), Miller(1, 0, 1, CS), Miller(1, 0, 2, CS)]

# ## Import and correct the intensities
#
# [PoleFigure.load](PoleFigure.load.html) combines the four files in one `PoleFigure`
# object. Its display reports four pole figures with 1152 specimen directions each, for
# 4608 measured intensities in total.

pf = PoleFigure.load(fname, h, CS, SS, interface='uxd')
pf

# Keep the reference measurement quiet because it is only an input to the correction.
# [correct](PoleFigure.correct.html) divides the specimen data by this reference to
# compensate for intensity lost as the specimen tilts.

pfDef = PoleFigure.load(fnameDef, h, CS, SS, interface='uxd')
pf = correct(pf, defocusing=pfDef)

# ## Check the corrected data
#
# Plot measurements before attempting an inversion. Check that each panel has the intended
# Miller index and that the plotted specimen axes match the experimental alignment. A
# plotting convention controls only where directions appear on screen; it does not repair
# a wrong reference frame.

plot(pf)

# The four panels share the same sampling grid, while their broad intensity maxima occur at
# different specimen directions. Those distinct patterns provide independent constraints
# on the ODF.
#
# Corrections can create negative intensities, which are not physical diffraction
# measurements. Count them before clipping them to zero.

# +
numNegative = np.count_nonzero(pf.intensities < 0)

pf[pf.intensities < 0] = 0
numNegative
# -

# `numNegative` is 0 for this dataset, so clipping changes no values and a second plot
# would be identical. Outlier removal, rotation, scaling, and other corrections are covered
# in [Modify Pole Figures](PoleFigureCorrection.html).

# ## Reconstruct an ODF
#
# [calcODF](PoleFigure.calcODF.html) finds an ODF whose recalculated pole figures fit the
# corrected measurements. `silent=True` suppresses the solver iteration history, while the
# returned object remains visible.

odf = calcODF(pf, verbose=False)
odf

# The display identifies the result as an [SO3FunRBF](SO3FunRBF.SO3FunRBF.html). Radial
# basis functions are its numerical representation, not a different scientific quantity.
# MTEX plotting and analysis commands operate through the common ODF interface.
#
# Reconstruction is not unique. Distinct ODFs can have identical pole figures, even with
# perfect measurements. More independent pole figures constrain the result, but they do not
# remove the fundamental ambiguity. [The Ghost Effect](PoleFigure2ODFAmbiguity.html)
# explains what the measurement cannot determine.

# ## Check the reconstruction
#
# Recalculate the four measured pole figures from the ODF and compare them with the
# corrected data above.

plotPDF(odf, pf.allH)

# The recalculated panels are smooth fields rather than discrete dots. Compare their broad
# high- and low-intensity regions with the measurements above; the reconstruction should
# follow the structure without reproducing every point-to-point fluctuation.
#
# [calcError](PoleFigure.calcError.html) quantifies the same comparison. It returns one
# regularised relative error for each measured pole figure.

reconstructionError = calcError(pf, odf)
reconstructionError

# The four errors range from 0.0412 to 0.0548. They show that this ODF reproduces the
# measured pole figures closely after intensity scaling. A small error does not prove that
# the ODF is unique or physically true. [ODF Reconstruction](PoleFigure2ODF.html) covers
# error measures and solver choices in detail.

# ## Inspect the ODF

# +
plot(odf)
mtexColorMap('LaboTeX')
# -

# An ODF value is measured in multiples of a random distribution, mrd. A value of 1 is
# random density, while 10 means ten times the random density near that orientation.
# Compute the largest value rather than estimating it from the colour scale.

odfMaximum, _ = max(odf, numLocal=1)
odfMaximum

# The printed maximum rounds to 1.9 mrd. The sections show broad, modest maxima rather than
# sharp isolated peaks, so this ZnCuTi sheet is weakly textured. Continue with
# [ODF Analysis](ODFAnalysis.html) to choose other views, identify components, and calculate
# texture-dependent properties.

# ## The maths behind the reconstruction
#
# Let $f(g)$ be the ODF, $h$ a crystal-plane normal, and $r$ a specimen direction. The
# corresponding pole density is the integral
#
# $$P_h(r) = \int_{\{g:\,g h=r\}} f(g)\,\mathrm{d}g.$$
#
# The integration set is an orientation fibre: every orientation that maps $h$ to $r$.
# This spherical Radon transform explains both why diffraction measures many crystals
# together and why its inversion is ambiguous.
#
# Further reading:
#
# * [ASTM E81-96(2024)](https://doi.org/10.1520/E0081-96R24), _Standard Test Method for
#   Preparing Quantitative Pole Figures_, covers X-ray acquisition.
# * [Bunge (1969)](https://doi.org/10.1016/C2013-0-11769-2), _Texture Analysis in Materials
#   Science_, develops the classical pole-figure and ODF theory.
# * [Matthies, Vinel and Helming (1987)](https://doi.org/10.1515/9783112736173), _Standard
#   Distributions in Texture Analysis_, gives standard distributions and the conventions
#   used to read them.
# * [Hielscher and Schaeben (2008)](https://doi.org/10.1107/S0021889808030112), _A novel
#   pole figure inversion method_, specifies the MTEX reconstruction algorithm.

# ## Technical Details
#
# The reconstruction places its kernels on 9978 orientations where MATLAB's grid for 6/mmm
# has 9924, with a uniform portion of 0.5466 for 0.54. The errors, 0.0416, 0.0417, 0.0548
# and 0.0417 for MATLAB's 0.0412, 0.0416, 0.0548 and 0.0418, and the maximum, 1.8067 mrd
# for 1.8528, differ accordingly; the recalculated pole figures and the sections agree by
# eye. `pf[mask] = 0` is MATLAB's `pf(mask) = 0`.
