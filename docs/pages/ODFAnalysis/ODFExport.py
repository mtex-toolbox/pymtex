# # ODF Export
#
# Exporting an orientation distribution function (ODF) means choosing what the receiving
# program needs. MTEX supports four common representations:
#
# * a pickle file containing the MTEX object, which preserves it as a Python object;
# * an MTEX ASCII file describing supported ODF components in readable text;
# * a generic table of ODF values on an orientation grid;
# * a VPSC table of discrete orientations and their volume fractions.
#
# The last two are finite approximations to the continuous ODF introduced in
# [ODF Theory](ODFTheory.html). Record the Euler-angle convention, angle units, grid
# resolution or number of orientations, crystal symmetry and specimen symmetry whenever
# the result must be reproducible.

import os
import pickle
import tempfile

import numpy as np

from mtex import *

# ## Define a Model ODF
#
# The examples use one mixture of uniform, fibre and unimodal components. Keeping the ODF
# fixed makes the differences between the file formats visible.

# +
cs = crystalFrame('cubic')
mod1 = orientation.byAxisAngle(xvector, 45 * degree, cs)
mod2 = orientation.byAxisAngle(yvector, 65 * degree, cs)
model_odf = (0.5 * uniformODF(cs)
             + 0.05 * fibreODF(Miller(1, 0, 0, cs), xvector, halfwidth=10 * degree)
             + 0.05 * fibreODF(Miller(0, 1, 0, cs), yvector, halfwidth=10 * degree)
             + 0.05 * fibreODF(Miller(0, 0, 1, cs), zvector, halfwidth=10 * degree)
             + 0.05 * unimodalODF(mod1, halfwidth=15 * degree)
             + 0.3 * unimodalODF(mod2, halfwidth=25 * degree))
plot(model_odf, sections=6, silent=True)
# -

# The six sections show a smooth density with localized maxima and fibre ridges on a
# uniform background. The grid and VPSC exports below replace this continuous function by
# finitely many rows.

# ## Save a Python Object
#
# Use Python's `pickle` when the next step also runs in Python with MTEX. Unlike a table
# export, this stores `model_odf` itself. MATLAB's counterpart is `save` into a `.mat`
# file.

# +
tempdir = tempfile.mkdtemp()
pickleName = os.path.join(tempdir, 'odf.pkl')
with open(pickleName, 'wb') as f:
  pickle.dump(model_odf, f)
# -

# Loading the file returns the stored ODF object rather than reconstructing one from
# sampled values. The assertion makes that round trip executable.

# +
with open(pickleName, 'rb') as f:
  saved = pickle.load(f)
assert np.allclose(saved.eval(cat(mod1, mod2)), model_odf.eval(cat(mod1, mod2)))
# -

# ## Export Values on a Generic Grid
#
# By default, [export](SO3Fun.export.html) writes four columns. The first three are
# [Bunge Euler angles](RotationRepresentations.html) on a regular $5^\circ$ grid, in
# degrees, and the fourth is the ODF value at that orientation. These values are density
# values in multiples of a uniform distribution, not volume fractions.
#
# Sampling does not preserve the internal ODF representation. A grid that is too coarse
# can miss a narrow component, so choose the resolution from the smallest feature that
# the receiving calculation must resolve. Here we request $10^\circ$ to keep the example
# file compact.

genericName = os.path.join(tempdir, 'odf-generic.txt')
export(model_odf, genericName, 'Bunge', resolution=10 * degree)

# The header records the symmetries and names the four columns. The first data rows then
# contain angles and the sampled ODF value.

# +
print('Beginning of the generic grid file:')
with open(genericName) as f:
  for k in range(6):
    print(f.readline().rstrip())
# -

# ## Pass a Grid Directly
#
# Other Euler-angle conventions and resolutions are available as options to `export`.
# For complete control, construct an orientation grid and pass it directly. This example
# uses an equispaced grid with a nominal resolution of $10^\circ$.

S3G = equispacedSO3Grid(cs, resolution=10 * degree)
gridName = os.path.join(tempdir, 'odf-equispaced.txt')
export(model_odf, gridName, S3G, 'Bunge', 'generic')

# ## Export an MTEX Component Description
#
# The `'mtex'` interface writes a human-readable description of the ODF components. It
# records the components themselves instead of replacing them by grid samples, and it is
# not a general interchange format.
#
# The current source tree has no matching reader: [SO3Fun.load](SO3Fun.load.html) offers
# loaders for generic and VPSC ODF files only, so this format is write-only. Use `pickle`
# when an exact round trip matters. Not every representation is supported either, and
# the exporter records that harmonic components cannot be written in this format.

mtexName = os.path.join(tempdir, 'odf.mtex')
export(model_odf, mtexName, 'Bunge', interface='mtex')

# The beginning of the file identifies the symmetries and the uniform component. Later
# blocks describe the fibre and radial components.

# +
print('Beginning of the MTEX component file:')
with open(mtexName) as f:
  for k in range(8):
    print(f.readline().rstrip())
# -

# ## Export a Synthetic Polycrystal for VPSC
#
# The [VPSC code](https://github.com/lanl/VPSC_code) and other crystal plasticity
# programs operate on discrete crystal orientations rather than directly on an ODF. The
# `'VPSC'` interface therefore draws orientations from the ODF and writes their Bunge
# Euler angles and volume fractions.

vpscName = os.path.join(tempdir, 'odf-vpsc.txt')
export(model_odf, vpscName, 'VPSC', points=5000)

# A VPSC block has four header lines. Its fourth line gives the Euler-angle convention
# and orientation count: `B` means Bunge. Each following row contains three Euler angles
# in degrees and one volume fraction.

# +
print('Beginning of the VPSC file:')
with open(vpscName) as f:
  for k in range(6):
    print(f.readline().rstrip())
# -

# The `points` option controls the number of orientations and defaults to 10000. More
# orientations usually represent the continuous density more finely, but they also make
# the receiving calculation larger. A VPSC file carries no crystal symmetry, so pass that
# information separately when it is read. [Import from VPSC](VPSCImport.html) shows the
# return path.

# ## Clean Up
#
# All examples wrote to a temporary directory. Remove every file now that the previews
# and round-trip check are complete.

# +
os.remove(pickleName)
os.remove(genericName)
os.remove(gridName)
os.remove(mtexName)
os.remove(vpscName)
# -

# ## Choosing a Format
#
# Use `pickle` while continuing an analysis in Python and MTEX. Use MTEX ASCII when a
# readable description of supported components is useful. Use a generic grid when the
# receiving program expects function values, and use VPSC when it expects a synthetic
# polycrystal.
#
# Grid spacing and sample size are accuracy parameters, not merely file-format options.
# [Random Sampling](RandomSampling.html) explains why a random statistical sample and an
# optimized numerical representation are not interchangeable.
# [ODF Import](ODFImport.html) explains how MTEX interprets tabulated values and weights
# when files are read back. [Orientation Export](OrientationExport.html) is the
# corresponding page when the starting data are already individual orientations rather
# than an ODF.

# ## References
#
# * H.-J. Bunge, [Texture Analysis in Materials Science: Mathematical Methods](https://doi.org/10.1016/C2013-0-11769-2),
#   Butterworths, 1982. This is the standard reference for ODFs and the Bunge Euler-angle
#   convention.
# * R. A. Lebensohn and C. N. Tomé,
#   [A self-consistent anisotropic approach for the simulation of plastic deformation and texture development of polycrystals](https://doi.org/10.1016/0956-7151(93)90130-K),
#   _Acta Metallurgica et Materialia_ 41 (1993), 2611--2624. This paper introduces the
#   VPSC formulation used by the discrete-orientation export.

# ## Technical Details
#
# MATLAB's `.mat` working copy is Python's `pickle`; a pickled frame comes back as the
# session's own handle, so the loaded function lives in the same frames as the original.
# The three writers are `exportODF_generic`, `exportODF_mtex` and `exportODF_VPSC` in
# `mtex/io/exportodf.py`, chosen by the flag or the `interface` keyword. The VPSC sample
# is a random draw, so its rows differ from MATLAB's.
