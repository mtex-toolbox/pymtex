# # Exporting EBSD Data
#
# Exporting writes an `EBSD` map to another file. Begin with a map whose phases and
# [reference frame](EBSDReferenceFrame.html) have already been checked.
# [Importing EBSD Data](EBSDImport.html) introduces the properties, scan-level options
# and header information discussed below.
#
# [export](EBSD.export.html) chooses the exporter from the filename extension.
#
#     # write an Oxford text file
#     export(ebsd, 'myFile.ctf')

import os
import tempfile

import numpy as np

from mtex import EBSD, angle, degree, export, mtexdata

# ## Choose the output for its purpose
#
# No EBSD file format can represent every part of an MTEX variable. A *property* has one
# value per measurement and is subset with the map. A *header* is scan-level information
# from the imported file, such as acquisition settings and vendor bookkeeping. It remains
# in the vendor's native layout under `ebsd.meta['header']`.
#
# | *Output* | *Use* | *Important limit* |
# |---|---|---|
# | [.ang](exportEBSD_ang.html) or [.ctf](exportEBSD_ctf.html) | vendor text exchange | fixed columns and format-specific header entries only |
# | [HDF5 extensions](exportEBSD_h5.html) | return changed data to a vendor container | requires the imported HDF5 provenance and a reference file |
# | another extension, such as `.txt` | a plain numeric table | writes Euler angles, phase ids and numeric properties, but not a complete map archive |
#
# MATLAB writes the Oxford binary pair `.crc` / `.cpr` as well and moves a complete
# variable between sessions in a `.mat` file; neither is ported. Converting between
# formats can therefore lose or rename properties, header entries, phase descriptions and
# acquisition data. Keep the original measurement file, write to a new name and re-import
# the result before relying on it.
#
# The `.ang` and `.ctf` exporters undo the correction applied by the corresponding
# importer between the Euler-angle and map reference frames. This preserves the
# specimen-frame interpretation when the same setting is used on import. It does not
# guarantee that another format can represent the same crystal frame attached to each
# phase.
#
# Both exporters take as much of the rest along as the format allows: whatever the header
# of the imported file stated is kept in `ebsd.meta['header']` and written back out, so
# entries MTEX does not model - the pattern centre, the working distance, the operator -
# are carried over rather than written as zeros.

# ## Verify a text-format conversion
#
# This example converts a bundled CTF map to ANG and imports the new file. The two object
# summaries are useful output: compare the measurement and phase inventories, then
# inspect which per-pixel properties the target format retained.

ebsd = mtexdata('twins')
ebsd

# +
exportFile = os.path.join(tempfile.mkdtemp(), 'twins.ang')
export(ebsd, exportFile)
ebsdRoundTrip = EBSD.load(exportFile)
ebsdRoundTrip
# -

# +
isIndexed = ebsd.isIndexed & ebsdRoundTrip.isIndexed
roundTripError = np.max(angle(ebsd[isIndexed].orientations, ebsdRoundTrip[isIndexed].orientations)) / degree
roundTripError
# -

# Both summaries contain 22,879 measurements: 22,833 indexed magnesium measurements and
# 46 notIndexed measurements. Their property lists differ because ANG and CTF define
# different columns. A missing property name does not always mean that its values
# vanished. The exporter may map a compatible quantity to the target format's name.
#
# Export and import both use ANG setting 2 here, so the relationship between the
# Euler-angle and map reference frames is consistent. The original CTF phase uses X
# parallel to a-star and Y parallel to b, whereas an ANG file supplies X parallel to a
# and Y parallel to b-star: the two files do not carry the same hexagonal crystal frame.
# The readers keep each file's own alignment, the comparison warns that the two frames
# differ, and the `roundTripError` of $30^\circ$ exposes that loss. A successful import and equal
# measurement counts are still not enough to validate a converted map: compare the
# crystal reference frames of the two phases as well.

os.remove(exportFile)

# ## HDF5: write into a copy of the imported file
#
# HDF5 is a container, not one EBSD data format. Every vendor defines its own hierarchy,
# units and data sets. A vendor file may also contain raw diffraction patterns, electron
# images and acquisition settings that MTEX never imported. Creating a new hierarchy from
# the `EBSD` variable would discard those contents.
#
# The HDF5 exporter instead copies the file from which the map was imported and patches
# the changed values into that copy. The output remains in the vendor's layout.
#
#     ebsd = EBSD.load('myfile.h5oina')
#     ebsd = smooth(ebsd, halfQuadraticFilter())
#
#     # copy myfile.h5oina and replace its orientations
#     export(ebsd, 'denoised.h5oina')
#
# [EBSD.load](EBSD.load.html) records the source file and resolved data-set paths in the
# scan-level option `ebsd.meta['h5']`. A different reference file can be named
# explicitly, but the `EBSD` variable must still carry that HDF5 provenance. The named
# file must contain the recorded paths.
#
#     export(ebsd, 'denoised.h5oina', reference='myfile.h5oina')
#
# The output and reference filenames must differ. The exporter refuses to overwrite the
# reference because a failed write would destroy the only complete copy.
#
# Only measurements that remain in the `EBSD` variable are patched. Other rows remain as
# the reference file stored them. So does every data set that MTEX did not read.
#
# The exporter updates orientations, phases, phase names and lattice values. It does not
# translate point-group symmetry between vendor coding schemes. Per-pixel numeric
# properties return to their original paths. Header contents that MTEX does not model
# survive because the reference file is copied.
#
# A property computed in MTEX is added beside the imported properties when the vendor
# layout has an extensible data group. A compound record has no room for another column.
# The exporter warns and leaves that property out. Pass `properties=False` to update
# orientations and phases while leaving every property in the reference copy unchanged.
#
# A reference file is required. Data that was not imported from HDF5 has no record of
# the paths to patch. Supplying another HDF5 filename still raises an error.

# ## Preserve the complete MTEX variable
#
# MATLAB carries the map between sessions in a MAT-file, which preserves the full
# variable, including properties, scan-level options, reference frames and the imported
# header. Python's `pickle` module plays that role for an `EBSD` variable here. Such a
# file is the lossless working copy, not a substitute for the original acquisition file
# or a documented interchange format. Preserve those alongside it when the data must
# remain usable outside MTEX.

# ## References
#
# * [ISO 24173:2024](https://www.iso.org/standard/82749.html), _Microbeam analysis --
#   Guidelines for orientation measurement using electron backscatter diffraction_,
#   concerns reliable and reproducible measurements.
# * M. A. Jackson et al., [h5ebsd: an archival data format for electron back-scatter diffraction data sets](https://doi.org/10.1186/2193-9772-3-4),
#   _Integrating Materials and Manufacturing Innovation_ 3, 44--55, 2014. The paper
#   distinguishes archival and workflow files.
# * The [HDF5 Data Model and File Structure](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_m__u_g.html)
#   explains why HDF5 supplies containers and objects rather than an EBSD-specific
#   hierarchy.
# * The [Oxford Instruments NanoAnalysis HDF5 File Specification](https://github.com/oinanoanalysis/h5oina/blob/master/H5OINAFile.md)
#   documents one vendor hierarchy, including units and coordinate definitions.

# ## Next
#
# Export completes the EBSD file workflow. Continue with [Grains](Grains.html) to measure
# reconstructed grains. [Grain Boundaries](GrainBoundaries.html) treats their interfaces.
# [ODF Analysis](ODFAnalysis.html) describes the orientation distribution.
