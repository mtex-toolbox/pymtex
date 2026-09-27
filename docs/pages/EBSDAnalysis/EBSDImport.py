# # Importing EBSD Data
#
# An EBSD map represents a list of measurements, whatever layout the file uses on disk.
# Each measurement has a position, a phase and an orientation. Importing turns that list
# into an `EBSD` variable and supplies the crystallography and reference frames needed to
# interpret it.
#
# Before starting, identify the specimen directions and check the phase names, lattice
# parameters and crystal-axis alignments reported by the acquisition software.
# [Orientations](OrientationDefinition.html) explains how an orientation relates the
# crystal and specimen frames.
#
# [EBSD.load](EBSD.load.html) chooses the interface from the file extension. The first
# import needs only a filename. The file is one of the sample data files, fetched on first
# use.

from mtex import EBSD, crystalFrame, mtexdatafile, notIndexedFrame, plot, plottingConvention, rotation, xvector, zvector

plottingConvention.default('y↓→x')

# +
fileName = mtexdatafile('EMSphinx')
ebsd = EBSD.load(fileName)
ebsd
# -

# Read this display once. It is the inventory for everything that follows. The file
# contains four scans, and the triangle in the data-set list marks the first as the one
# imported. Three indexed phases arrived with lattice parameters, and the cobalt phase
# states its crystal-axis alignment. Two phases are cubic, and 99% of the measurements are
# gamma iron. The square grid has 508 × 955 cells and covers about 382 × 203 microns.
#
# The display also lists `iq` and `metric` as *properties*. A property has one value per
# measurement and is subset together with the map. By contrast, the selected HDF5 path
# and the file header are scan-level *options* under `ebsd.meta`; they are not resized by
# a selection.
#
# A phase map is the quickest check that the scan footprint and phase inventory are
# plausible.

plot(ebsd)

# Almost the entire map has the gamma-iron phase colour, as the 99% figure predicts. The
# sparse alpha-iron and cobalt measurements remain visible against it, so a wrong phase
# identifier would not be hidden by the total.
#
# This variable is the starting point for
# [grain reconstruction](GrainReconstruction.html), [ODF estimation](EBSD2ODF.html) and
# [misorientation analysis](Misorientations.html).

# ## What Import Cannot Guess
#
# A *reference frame* is the coordinate system in which data are expressed. Two reference
# frames can be involved in an EBSD measurement: the frame of the map coordinates and the
# frame used for the Euler angles. Vendors align them differently, and a file does not
# always record their relationship.
#
# The failure mode is quiet. A map imported with the wrong relationship still plots, still
# reconstructs into grains and still yields pole figures. Those results are simply rotated
# or mirrored with respect to the specimen, and no number in the data set reveals the
# mistake.
#
# A *plotting convention* only states how a reference frame is laid out on screen.
# Changing it can turn the displayed map, but it cannot repair an incorrect relationship
# between the two frames. Supply that relationship during import, then follow
# [Reference Frame Alignment](EBSDReferenceFrame.html) to validate it against a known
# specimen direction or microstructural feature.

# ## The Import Wizard
#
# The wizard is the interactive route through those choices. It opens a window over a
# folder, describes a file from its header before reading it, imports a map on a click and
# shows it beside its phases and the two reference frames. It needs the `wizard` extra,
# `pip install mtex[wizard]`, which brings PySide6.
#
# ```python
# from mtex import importWizard
# ebsd = importWizard()               # the current folder
# ebsd = importWizard('/data/ebsd')   # a folder, or a file to open on start
# ```
#
# Without Python open, the wizard starts from a terminal as `mtex-wizard`, or
# `mtex-wizard /data/ebsd`. Started this way it has no session to return the map to; save
# or copy the import script it writes instead.
#
# ![](importWizard.png)
#
# The folder is browsed at the left, the file's description under it and the data sets of a
# project file between them. The phase table across the top is edited in place, the two
# frames at its left are the map coordinates and, in red, the coordinates the Euler angles
# were written in; each opens a menu of the eight axis aligned conventions. Changing the
# Euler frame turns every orientation in place, so the map and the pole figures answer at
# once, and you can compare the result with the specimen rather than guess a vendor
# convention.
#
# The wizard can bind the map to a variable directly, but its more useful output is an
# import script. The script records every choice, runs without the wizard and provides a
# reproducible start for the rest of the analysis.

# ## The Import Script
#
# The essential part of such a script is shown below. It records the phases, the screen
# alignment, the source file, the frame correction and a first plot.

# +
# crystal symmetry
csList = [
  notIndexedFrame(),
  crystalFrame('m-3m', [2.8665, 2.8665, 2.8665], mineral='Fe(alpha-iron)', color='LightSkyBlue'),
  crystalFrame('m-3m', [3.591, 3.591, 3.591], mineral='Fe(gamma-iron)', color='DarkSeaGreen'),
  crystalFrame('6/mmm', [2.5071, 2.5071, 4.0686], 'X||a', 'Y||b*', 'Z||c', mineral='Co(alpha-cobalt)', color='Goldenrod')]

# how the map is aligned on screen
plottingConvention.default('y↓→x')

# which file to be imported
fname = mtexdatafile('EMSphinx')

# rotates the Euler angle reference frame onto the map reference frame
EulerCorrection = rotation.map(xvector, xvector, zvector, -zvector)

# create an EBSD variable containing the data
ebsd = EBSD.load(fname, csList, dataSet=0, eulerCorrection=EulerCorrection, silent=True)
# -

# The crystal-symmetry calls state both a point group and a crystal frame. For cobalt, the
# X parallel a, Y parallel b-star and Z parallel c choices align the Cartesian crystal
# frame with the lattice basis; they are not part of the 6/mmm point group.
#
# `EulerCorrection` is a frame change from the Euler-angle frame to the map frame. Here it
# maps Euler $x$ onto map $x$ and Euler $z$ onto minus map $z$. This is the correction for
# a map with $y$ pointing down when the Euler angles were measured with $y$ pointing up.
#
# The plotting convention is set as the session default before the import. Everything
# derived later that states no frame of its own follows it.
#
# The script ends with an orientation plot. Unlike the phase map above, this plot checks
# that orientations and positions have arrived together as a spatially coherent map.

plot(ebsd['Fe(gamma-iron)'], ebsd['Fe(gamma-iron)'].orientations, ipfDirection=zvector)

# Coherent elongated domains and smooth colour changes inside them show that the
# orientations have not been scrambled across the scan. An equally coherent map can still
# be rotated or mirrored, so this picture alone does not validate the reference frame. Use
# the known-direction tests on [Reference Frame Alignment](EBSDReferenceFrame.html) for
# that.

# ## Supported Data Formats
#
# | | |
# |---|---|
# | [.ang](loadEBSD_ang.html) | EDAX and EMSphInx text files |
# | [.ctf](loadEBSD_ctf.html) | Oxford / HKL text files |
# | [.osc](loadEBSD_osc.html) | EDAX binary files |
# | [.crc, .cpr](loadEBSD_crc.html) | Oxford binary files |
# | [HDF5](loadEBSD_h5.html) | .h5, .hdf5, .oh5, .h5oina, .edaxh5 and .dream3d |
# | [generic text](loadEBSD_generic.html) | columns in a user-defined order |
#
# The HDF5 interface reads Bruker, EDAX, Oxford, ThermoFisher, EMsoft and EMSphInx
# layouts. Prefer a documented, non-proprietary format when the acquisition software
# offers one. A published specification preserves the units, hierarchy and coordinate
# definitions needed to read the data later. Add a new HDF5 layout through the JSON
# configuration described in [HDF5 Interface](EBSDInterfaceHDF5.html) rather than writing
# another HDF5 reader.
#
# The generic loader is the fallback for a text file that no vendor interface recognises.
# If it contains Euler angles, a phase and spatial coordinates as columns,
#
#     alpha_1 beta_1 gamma_1 phase_1 x_1 y_1
#     alpha_2 beta_2 gamma_2 phase_2 x_2 y_2
#     alpha_3 beta_3 gamma_3 phase_3 x_3 y_3
#     .       .      .       .       .   .
#     alpha_M beta_M gamma_M phase_M x_M y_M
#
# then `columnNames` assigns each column. Orientations without spatial coordinates do not
# form a map; import them as described in [Importing Orientations](OrientationImport.html).

# ## HDF5 Files With Several Data Sets
#
# An HDF5 project file can hold more than one map. EDAX paths commonly contain
# `Area N/OIM Map N`, Oxford uses numbered slices, and EMSphInx uses `Scan N`. The opening
# import prints every available data set and marks the selected one. When no selection is
# supplied, the first entry is imported.
#
# Select another by its number, counted from zero, or by an unambiguous part of its name.
#
#     ebsd = EBSD.load(fname, dataSet=1)
#     ebsd = EBSD.load(fname, dataSet='OIM Map 7')
#
# The full imported HDF5 path is the scan-level option `ebsd.meta['dataSet']`. The short
# names of all choices are stored in `ebsd.meta['dataSets']`, and the header of the file,
# the scan-level information from its preamble such as acquisition settings and vendor
# records, in `ebsd.meta['header']`. Its field names retain the vendor's native layout;
# MTEX does not impose a common header schema.

# ## Raw and Post-Processed Data
#
# An Oxford `h5oina` file may store the same map twice: as recorded by the detector under
# `EBSD` and after vendor processing under `Data Processing`. They are two data sets of
# one file. MTEX lists them together and puts the processed version first.
#
#     ebsd = EBSD.load(fname)                     # post-processed
#     ebsd = EBSD.load(fname, dataSet='EBSD')     # as recorded
#
# The recorded version usually retains more per-pixel properties, such as band contrast,
# band slope, pattern quality and pattern-centre values. The processed version may retain
# fewer properties but includes the vendor's bad-pixel cleanup. Both refer to the same
# specimen frame, so their orientations can be compared directly.
#
# A file that was never processed contains the recorded version alone and lists one data
# set per map.

# ## Writing Your Own Interface
#
# A format not covered above needs a loader function. The established name is
# `loadEBSD_xxx`, and the loaders in `mtex/io` are examples. Return an `EBSD` variable and
# call the new function directly while developing it.
#
# The dispatch of `EBSD.load` is explicit: placing a function in that folder does not by
# itself register a new extension. Full integration also requires a dispatch case in
# `loadEBSD` for the extension.

# ## References
#
# * A. J. Schwartz, M. Kumar, B. L. Adams and D. P. Field, editors,
#   [Electron Backscatter Diffraction in Materials Science](https://doi.org/10.1007/978-0-387-88136-2),
#   second edition, Springer, 2009, develops EBSD measurement, calibration and orientation
#   mapping.
# * T. B. Britton et al.,
#   [Tutorial: Crystal orientations and EBSD -- Or which way is up?](https://doi.org/10.1016/j.matchar.2016.04.008),
#   _Materials Characterization_ 117, 113--126, 2016, gives a practical calibration chain
#   between detector, scan, specimen and crystal frames.
# * [ISO 24173:2024](https://www.iso.org/standard/82749.html), _Microbeam analysis --
#   Guidelines for orientation measurement using electron backscatter diffraction_,
#   specifies current guidance for reproducible orientation measurements, including
#   specimen alignment and calibration.
# * M. A. Jackson et al.,
#   [h5ebsd: an archival data format for electron back-scatter diffraction data sets](https://doi.org/10.1186/2193-9772-3-4),
#   _Integrating Materials and Manufacturing Innovation_ 3, 44--55, 2014, defines a
#   vendor-neutral HDF5 archival layout.
# * The [Oxford Instruments NanoAnalysis HDF5 File Specification](https://github.com/oinanoanalysis/h5oina/blob/master/H5OINAFile.md)
#   documents the public `h5oina` hierarchy, units and coordinate systems used by that
#   format.

# ## Next
#
# Continue with [Reference Frame Alignment](EBSDReferenceFrame.html) before trusting any
# specimen-relative result. Then use [Plotting](EBSDPlotting.html) to inspect map
# quantities and [Selecting EBSD Data](EBSDSelect.html) to restrict the measurement list.
# [Exporting EBSD Data](EBSDExport.html) covers writing the checked map back to disk.
