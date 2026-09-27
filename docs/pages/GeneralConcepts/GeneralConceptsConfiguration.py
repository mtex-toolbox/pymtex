# # Configuration
#
# An [option](GeneralConceptsOptions.html) changes one command. A preference supplies a
# default for the whole MTEX session. Preferences are read with `getMTEXpref` and set with
# `setMTEXpref`; MATLAB keeps its startup values in the central file `mtex_settings.m`, which
# the port does not have (see below).
#
# The diagram separates the three scopes. An option is local to one command, while
# `setMTEXpref` changes the current session. MATLAB's `mtex_settings.m` makes a setting take
# effect when MTEX starts in future sessions.
#
# ![](configuration-scope.svg)
#
# ## Inspect the current preferences
#
# Pass a preference name to `getMTEXpref` to read its current value. With no input, the
# function returns a dict containing all current preferences. A preference that has not been
# set is read as `None`, or as the default a second argument names; the commands that use it
# supply their own default then.

# +
from mtex import *

getMTEXpref()
# -

# ## Change one preference temporarily
#
# For example, `getMTEXpref('FontSize')` reads the current default font size, while
# `setMTEXpref('FontSize', 14)` would change it to 14 for this session. Documentation scripts
# should not change global figure preferences, so the executable example instead uses the
# preference for UTF-8 output.
#
# The sequence saves the current value, changes it, reads the changed value, and restores the
# original. Keeping the restoration in the same section makes the example safe to run
# repeatedly.

oldUTF8Output = getMTEXpref('UTF8Output')
setMTEXpref('UTF8Output', False)
print(getMTEXpref('UTF8Output'))
setMTEXpref('UTF8Output', oldUTF8Output)

# ## Make a preference persistent
#
# MATLAB runs `mtex_settings.m` at startup, so an active setting there applies to every later
# session. The port reads no settings file: every session starts from the defaults, and
# `mtex.reset()` returns to them. A setting meant for every session is a `setMTEXpref` call in
# a module of your own that your scripts import first, or in the file `PYTHONSTARTUP` names
# for interactive sessions:
#
#     # my_mtex_settings.py
#     from mtex import setMTEXpref
#     setMTEXpref('FontSize', 14)
#
# The preferences the port reads include:
#
# * `FontSize` and `markerSize`, the font size and the marker size of every plot;
# * `showMicronBar` and `showRefFrame`, whether an EBSD map shows a micron bar and a
#   reference-frame indicator;
# * `defaultColorMap`, the default colormap;
# * `EulerAngleConvention`, the Euler-angle convention, which is Bunge by default;
# * `mtexDataPath`, the folder the sample data sets are fetched into;
# * `gridifyOnImport`, whether an imported map is [gridified](EBSDGrid.html) on import;
# * `stopOnSymmetryMissmatch`, whether a symmetry mismatch stops with an error;
# * `kernels`, `threads`, `NUFFT` and `NFFTLibrary`, the compiled loops, the number of
#   threads and the optional third-party transform used for the harmonic transforms;
# * `maxS2Bandwidth`, `maxSO3Bandwidth`, the largest harmonic degrees used by default.
#
# ## Keep scripts reproducible
#
# `setMTEXpref` changes only the running session. A settings module imported at the start of
# a script makes the same [MTEX script](MTEXScripts.html) behave differently on another
# computer that does not have it. A script shared or published for others should therefore
# pass important settings as command options where possible instead of assuming a local
# configuration.
#
# ## Plot alignment is not an ordinary preference
#
# The alignment of plots is no longer an ordinary preference. It belongs to the
# [reference frame](referenceFrame.referenceFrame.html) in which the data is expressed. A
# reference frame identifies the coordinate system and basis of the data; it is distinct
# from the symmetry attached to that frame.
#
# The frame supplies a plotting convention that determines how its axes are laid out on
# screen. See [On-Screen Coordinate System Alignment](AxesAlignment.html) to set that
# convention explicitly instead of relying on a legacy axis-direction preference.
#
# ## References
#
# This page documents MTEX configuration behavior and does not rely on an external method
# reference.
#
# ## Next
#
# Continue with [Vectors](VectorDefinition.html) to create the basic geometric objects used
# for directions, axes, and positions throughout MTEX.
