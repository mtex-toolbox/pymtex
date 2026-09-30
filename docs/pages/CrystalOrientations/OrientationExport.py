# # Exporting Crystal Orientations
#
# Exporting an orientation means choosing a numerical representation that the receiving
# program understands. MTEX writes plain ASCII tables, so any program can read the numbers
# once their columns and conventions are known. This page is the counterpart of
# [Importing Crystal Orientations](OrientationImport.html).
#
# The table does not contain the complete MTEX `orientation` object. In particular, it
# does not store crystal symmetry or the crystal and specimen reference frames. A reference
# frame is the coordinate system in which data are expressed. The direction of the
# orientation map is also not stored.
#
# Record those choices beside the file. Their meaning is introduced in
# [Crystal Orientation as Coordinate Transformation](DefinitionAsCoordinateTransform.html)
# and compared with Bunge's map direction in [MTEX vs. Bunge Convention](MTEXvsBungeConvention.html).
#
# ## Define an Orientation Sample
#
# We use a random sample of 100 orientations from a model orientation distribution
# function (ODF).

# +
import os
import tempfile

import numpy as np

from mtex import *

cs = crystalFrame.load('quartz.cif')

odf = unimodalODF(orientation.byEuler(30 * degree, 50 * degree, 10 * degree, cs), halfwidth=10 * degree)

ori = odf.discreteSample(100)
numOrientations = len(ori)
numOrientations
# -

# The output confirms that the list has 100 entries. The record count in a VPSC header
# later on must agree with this value.
#
# ## Exporting Euler Angles
#
# [export](quaternion.export.html) writes one orientation per row. Here we name the Bunge
# convention explicitly, so a session preference cannot change the file. The angles are
# written in degree unless `radians=True` is passed.

fname = os.path.join(tempfile.gettempdir(), 'orientations.txt')
export(ori, fname, 'Bunge')

# The first line names the three columns. The following lines contain the Bunge Euler
# angles $(\varphi_1,\Phi,\varphi_2)$ in degree.

with open(fname) as f:
  for k in range(4):
    print(f.readline(), end='')

# ## Other Conventions and Units
#
# Without an explicit convention, `export` follows the session's Euler-angle preference.
# Any convention described in [Defining Rotations](RotationDefinition.html) may be named
# instead. The next file uses Matthies angles in radians.

export(ori, fname, 'Matthies', radians=True)
with open(fname) as f:
  for k in range(4):
    print(f.readline(), end='')

# The new header and values describe the same orientations in a different convention and
# unit. The file does not label the unit, so it must be recorded separately for the
# receiver.
#
# ## Exporting Quaternions
#
# Passing `quaternion=True` writes the four quaternion components instead of Euler angles.
# This is the only format on this page that performs no angle conversion.

export(ori, fname, quaternion=True)
with open(fname) as f:
  for k in range(3):
    print(f.readline(), end='')

# MTEX writes the scalar component first, in the order `a`, `b`, `c`, `d`. Other programs
# may use another order or sign convention, and a unit quaternion and its negative
# describe the same rotation. Check the receiver's contract before exchanging quaternion
# columns.
#
# ## Exporting Additional Columns
#
# Often an orientation needs an associated weight, grain size, or another quantity. A
# dictionary passed to [export](quaternion.export.html) appends one column per entry and
# uses the keys as column headers. Every entry must supply one value per orientation.

# +
S = {'angle': ori.angle() / degree,
     'weight': np.ones(ori.shape) / len(ori)}

export(ori, fname, S, 'Bunge')
with open(fname) as f:
  for k in range(3):
    print(f.readline(), end='')
# -

# The preview shows that `angle` and `weight` remain aligned with the Euler angles on each
# row.
#
# ## The VPSC Format
#
# [export](orientation.export_VPSC.html) with `interface='VPSC'` writes individual orientations in the
# texture format expected by the VPSC crystal plasticity code. It writes three Euler
# angles and one relative volume fraction per row.
#
# VPSC supports the Bunge, Kocks, and Roe conventions. MTEX defaults to Bunge for this
# format regardless of the session preference. The fourth header line records `B`, `K`, or
# `R` together with the number of rows.

fnameVPSC = os.path.join(tempfile.gettempdir(), 'orientations_vpsc.txt')
export(ori, fnameVPSC, interface='VPSC')
with open(fnameVPSC) as f:
  for k in range(6):
    print(f.readline(), end='')

# The line `B 100` identifies Bunge angles and the 100 orientations. The following rows
# contain three angles in degree and a uniform weight.
#
# ## Unequal VPSC Weights
#
# Pass weights that differ between orientations with the `weights` option. VPSC
# interprets them as relative volume fractions, so use nonnegative values with a positive
# sum. MTEX divides the supplied values by their sum before writing them.

# +
weights = np.arange(1, numOrientations + 1).reshape(ori.shape)
export(ori, fnameVPSC, interface='VPSC', weights=weights)
with open(fnameVPSC) as f:
  for k in range(6):
    print(f.readline(), end='')

vpscData = np.loadtxt(fnameVPSC, skiprows=4)
writtenWeightSum = np.sum(vpscData[:, 3])
writtenWeightSum
# -

# The first two written weights now differ. The final output checks their sum after the
# values have been rounded for the text file.
#
# ## Choosing a Format
#
# Use Euler angles when the receiver specifies a convention and unit. Use quaternions when
# both programs agree on component order and signs. Use the VPSC format only when a
# polycrystal code expects its weighted texture table.
#
# If the data will stay in MTEX, Python's `pickle` preserves the orientation object more
# completely than a numeric table. A whole ODF, rather than a list of individual
# orientations, is exported by the commands in [ODF Export](ODFExport.html).

# Remove the temporary files.

os.remove(fname)
os.remove(fnameVPSC)

# ## References
#
# * H.-J. Bunge, [Texture Analysis in Materials Science: Mathematical Methods](https://doi.org/10.1016/C2013-0-11769-2),
#   Butterworths, English ed., 1982, establishes the Euler-angle convention used in
#   texture analysis.
# * D. Rowenhorst et al., [Consistent representations of and conversions between 3D rotations](https://doi.org/10.1088/0965-0393/23/8/083501),
#   _Modelling and Simulation in Materials Science and Engineering_ 23, 083501, 2015,
#   compares Euler, matrix, axis--angle, and quaternion conventions.
# * R. A. Lebensohn and C. N. Tomé, [A self-consistent anisotropic approach for the simulation of plastic deformation and texture development of polycrystals](https://doi.org/10.1016/0956-7151(93)90130-K),
#   _Acta Metallurgica et Materialia_ 41, 2611--2624, 1993, introduces the VPSC
#   formulation.
# * The [VPSC project and manual](https://public.lanl.gov/lebenso/) describe the weighted
#   orientation input used by the code.
#
# ## Next
#
# [Embeddings of Orientations](OrientationEmbeddings.html) is the next page in this
# chapter. It replaces coordinate representations with tensors for statistics and machine
# learning. For a continuous orientation density, continue with [ODF Export](ODFExport.html).
