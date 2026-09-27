# # Importing Crystal Orientations
#
# A list of crystal orientations usually arrives as a text file with one orientation per
# row. Unlike an EBSD map, the list has no spatial positions. Importing it requires the
# numeric representation, the crystal symmetry, and the reference frames in which the
# numbers were defined.
#
# This page assumes the orientation map introduced in
# [Defining Orientations](OrientationDefinition.html) and the Euler-angle conventions from
# [Defining Rotations](RotationDefinition.html). It uses a three-column file of Bunge Euler
# angles in degrees.

# +
from mtex import *

plottingConvention.default('y↑→x')

# load the quartz symmetry and its crystal reference frame from a CIF file
cs = crystalFrame.load('quartz.cif')
# -

# ## Name the Columns
#
# The file contains three numeric columns but no header. Their names tell
# [orientation.load](orientation.load.html) that the columns are the Bunge angles
# $(\varphi_1,\Phi,\varphi_2)$. The file is one of the sample data sets, fetched on first
# use.

fname = mtexdatafile('tongue')
ori = orientation.load(fname, cs, columnNames=['phi1', 'Phi', 'phi2'])
ori

# The display reports a list of 382 [orientations](orientation.orientation.html). It also
# identifies the quartz crystal symmetry and the specimen frame. MTEX stores all 382
# orientations in this one vectorized object.
#
# ## Check the Imported Texture
#
# A pole figure is a useful first sanity check, although it cannot prove a convention by
# itself. Here the $(0001)$ and $(10\bar{1}0)$ poles should show the texture carried by
# the imported orientations.

plotPDF(ori, Miller([[0, 0, 0, 1], [1, 0, -1, 0]], cs))

# Notice that the $(0001)$ poles concentrate around ND, whereas many $(10\bar{1}0)$ poles
# lie closer to the rim. The imported population is therefore textured rather than
# randomly distributed.
#
# ## What the Options Are For
#
# For Euler input, the three angle columns are mandatory. Further named columns are
# returned in a dictionary by `loadOrientation`, as in
# `ori, properties = loadOrientation(...)`. The keys are converted to lower case and have
# whitespace removed.
#
# | option | meaning |
# |---|---|
# | `columnNames` | what each imported column contains |
# | `columns` | positions of those columns in the file, counted from zero |
# | `radians=True` | angles are in radians rather than degrees |
# | `header` | number of header lines to skip |
# | `delimiter` | character that separates the numbers |
# | `passive=True` | the rows are the inverse map |
#
# The names and positions solve different problems. For example, `columns=[3, 1, 6]`
# selects physical columns 4, 2, and 7, while `columnNames=['phi1', 'Phi', 'phi2']`
# assigns their meanings in that order.
#
# Four columns named `['Quat real', 'Quat i', 'Quat j', 'Quat k']` are read as the
# quaternion components. Quaternions avoid the Euler-angle sequence and angular-unit
# questions, but they do not identify the mapping direction or either reference frame.
#
# ## Four Questions to Answer Before Trusting the Result
#
# A plain numeric file does not contain enough information to distinguish several valid
# interpretations. All four questions below must be answered from its header,
# accompanying documentation, or a known physical feature.
#
# * *Which Euler-angle convention?* This example uses the Bunge column names; the columns
#   `alpha`, `beta`, `gamma` are read as Matthies angles. Equal angle triplets in other
#   conventions describe different orientations. Import those numeric columns and pass them
#   to [orientation.byEuler](orientation.byEuler.html) with the convention named
#   explicitly; see [Defining Rotations](RotationDefinition.html).
#
# * *Degrees or radians?* The importer reads degrees unless `radians=True` is passed; it
#   does not guess the unit from the values. A file of angles in radians read as degrees
#   gives valid-looking results, so the unit must come from the file's documentation.
#
# * *Active or passive?* MTEX orientations map coordinates from the crystal frame into the
#   specimen frame. Do not select `passive=True` merely because a source calls its
#   convention Bunge: reported Bunge Euler angles are copied directly when the frames
#   agree. The distinction is developed in [MTEX vs. Bunge Convention](MTEXvsBungeConvention.html).
#
# * *Which crystal and specimen frames?* The alignment between Cartesian crystal axes and
#   lattice axes belongs to the `crystalFrame` loaded above; see
#   [The Crystal Reference System](CrystalReferenceSystem.html). The file documentation must
#   also say which physical specimen directions its axes denote. The explicit plotting
#   convention on this page states TD upward and RD to the right; it does not infer those
#   directions from the three columns.
#
# A known direction provides the strongest check. Verify that one indexed crystal direction
# maps to the specimen direction observed in the experiment. A plausible pole figure alone
# cannot distinguish every wrong combination of convention and frame.
#
# ## References
#
# * H.-J. Bunge, [Texture Analysis in Materials Science: Mathematical Methods](https://doi.org/10.1016/C2013-0-11769-2),
#   Butterworths, English ed., 1982, establishes the Euler-angle convention used in
#   texture analysis.
# * G. Nolze, [Euler angles and crystal symmetry](https://doi.org/10.1002/crat.201400427),
#   _Crystal Research and Technology_ 50, 188--201, 2015, explains why unit-cell settings
#   and specimen axes can give different Euler triplets for the same orientation.
# * D. Rowenhorst et al., [Consistent representations of and conversions between 3D rotations](https://doi.org/10.1088/0965-0393/23/8/083501),
#   _Modelling and Simulation in Materials Science and Engineering_ 23, 083501, 2015, gives
#   reproducible conversion rules for Euler angles, matrices, axis--angle pairs, and
#   quaternions.
# * [ISO 24173:2024](https://www.iso.org/standard/82749.html), _Microbeam analysis --
#   Guidelines for orientation measurement using electron backscatter diffraction_, covers
#   reliable and reproducible orientation measurements when a list originates from EBSD.
#
# ## Next
#
# Writing orientations back to a file is [Export](OrientationExport.html). Orientations
# measured on a grid across a specimen are imported as a map instead; see
# [Importing EBSD Data](EBSDImport.html).
#
# ## Technical Details
#
# MATLAB's generic importer ignores `'radians'` and guesses the unit from the values
# (degrees only when an angle exceeds 15), applies `'passive'` twice so that it has no
# effect, and fails on the quaternion columns it advertises; its page warns about all
# three. The port reads the unit it is told, inverts once, and reads the quaternion
# columns.
