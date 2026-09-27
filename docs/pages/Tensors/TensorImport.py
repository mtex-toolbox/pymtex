# # Importing Tensor Data
#
# Importing a coefficient table is only the first step in defining a tensor.
# The physical property, unit, compact-matrix convention, crystal symmetry,
# and crystal frame must agree with the source that published the values.
# MTEX cannot infer that scientific context from plausible-looking numbers.
#
# This page assumes the ranks and physical classes introduced in
# [Defining Tensorial Properties](TensorDefinition.html).
# Read [Crystal Reference System](CrystalReferenceSystem.html) first if the
# alignment between lattice axes and Cartesian axes is new to you.
#
# A reference frame is the coordinate system in which data are expressed.
# It is distinct from crystal symmetry, which states the point group under
# which the property is invariant.

from mtex import *

plottingConvention.default('y↑→x')

# ## Load a generic tensor
#
# [tensor.load](tensor.load.html) reads the coefficient table of a text file.
# Reading the table does not establish what the coefficients mean.
#
# The bundled quartz file contains descriptive header lines, but only its
# compact $3\times6$ coefficient table is imported.
# The point group and alignment options below attach the crystal frame used
# by the source. The alignment belongs to the frame, not to point group `32`.

# +
# define the quartz crystal symmetry and frame
csQuartz = crystalFrame('32', [4.916, 4.916, 5.4054], 'X||a*', 'Z||c', mineral='Quartz')

# define the file name
quartzFile = mtexdatafile('quartzPiezo')

# import and display the piezoelectric strain tensor
P = tensor.load(quartzFile, csQuartz, rank=3, name='piezoelectric strain', unit='pC/N', doubleConvention=True)
P
# -

# ## Read the import summary
#
# The display is an import audit. It identifies a rank 3 tensor in the
# quartz crystal frame, labels its values in pC/N, and reports
# `doubleConvention: true` before printing the compact coefficient table.
#
# The `name` and `unit` options are labels stored with `P`.
# They do not convert the numbers, so the file values must already be in
# pC/N. These labels are the imported object's record of the physical meaning
# and unit; retain the publication itself as the record of provenance.

# ## Check the compact-matrix convention
#
# The six compact columns represent the index pairs
# $(11,22,33,23,13,12)$. With `doubleConvention`, columns 4 to 6 contain
# twice the corresponding off-diagonal tensor components.
# MTEX divides those entries by two when expanding the table to
# $3\times3\times3$.
#
# This is the engineering-shear convention used by the quartz file.
# Do not infer it from the shape of a table: omitting the option for this
# source would silently double part of the tensor.

# ## Load a physically specific tensor
#
# Common physical tensors have dedicated loaders. The
# [stiffnessTensor.load](stiffnessTensor.stiffnessTensor.html) method returns
# the physically typed class, fixes rank 4, uses the stiffness form of the
# Voigt convention, and supplies GPa as the default unit label.
#
# Those defaults are correct for this olivine file, but they are still
# assumptions rather than unit conversion or validation. The file also gives
# a density of 3355 kg/m$^3$, which is 3.355 g/cm$^3$. The loader reads it from
# the file; giving it explicitly records it with the import.

# +
# define the file name
olivineFile = mtexdatafile('olivine1997')

# define the olivine crystal symmetry and frame
csOlivine = crystalFrame('mmm', [4.7646, 10.2296, 5.9942], mineral='Olivine')

# import and display the elastic stiffness tensor
C = stiffnessTensor.load(olivineFile, csOlivine, density=3.355)
C
# -

# ## Read the typed result
#
# The display now reports a rank 4 `stiffnessTensor` in GPa, the attached
# density, and the $6\times6$ Voigt matrix. The typed result provides elastic
# moduli and wave-velocity operations that a generic `tensor` does not.
# [Wave Velocities](WaveVelocities.html) explains why density is required for
# seismic velocities.

# ## Check compatibility with crystal symmetry
#
# [checkSymmetry](tensor.checkSymmetry.html) tests whether each imported
# tensor is invariant under its attached crystal point group.
# The two logical values below correspond to quartz and olivine.

symmetryMatches = [checkSymmetry(P), checkSymmetry(C)]
symmetryMatches

# Both values are true for these files. This check can detect an incompatible
# point group, and for the quartz file it also catches a missing
# `doubleConvention`. It cannot verify units, axis sense, handedness, or
# whether the correct physical property was selected.
# Compare those items with the source publication before using the tensor.

# ## Next
#
# Continue with [Tensor Arithmetic](TensorArithmetics.html) to rotate and
# contract imported tensors. [Tensor Visualization](TensorVisualisation.html)
# shows how to inspect their directional dependence, and
# [Tensor Averages](TensorAverage.html) combines a single-crystal property with
# orientations or an ODF.
#
# The quartz example continues in [Piezoelectricity](PiezoElectricity.html).
# The olivine example continues in [Anisotropic Elasticity](AnisotropicTheory.html)
# and [Wave Velocities](WaveVelocities.html).

# ## Further reading
#
# * J. F. Nye, [Physical Properties of Crystals: Their Representation by Tensors and
#   Matrices](https://search.worldcat.org/title/11114089), Oxford University Press, 1985,
#   develops crystal axes, tensor components, and contracted matrix notation.
# * A. Authier, editor, [International Tables for Crystallography, Volume D: Physical
#   Properties of Crystals](https://doi.org/10.1107/97809553602060000113), 2nd ed., IUCr,
#   2014, tabulates crystal tensor symmetries and Voigt notation.
# * [IEEE Std 176-1987](https://standards.ieee.org/ieee/176/356/), _IEEE Standard on
#   Piezoelectricity_, specifies historical quartz axis, sign, and compact-matrix
#   conventions. The standard was withdrawn in 2000.
# * E. H. Abramson, J. M. Brown, L. J. Slutsky, and J. Zaug, [The elastic constants of San
#   Carlos olivine to 17 GPa](https://doi.org/10.1029/97JB00682), _Journal of Geophysical
#   Research_ 102 (1997), 12253-12263, is the source of the olivine stiffness example.
#
# ## Technical Details
#
# The quartz file of MATLAB's data folder is the sample data set `quartzPiezo` here, the
# olivine file the data set `olivine1997`, both fetched on first use; `mtexdatafile` gives
# their paths. `mtexdata('quartzPiezo')` returns the tensor exactly as imported above.
