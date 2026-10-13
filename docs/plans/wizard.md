# Subplan P8 - the import wizard

Status: W0 to W4 and W6 done, W5 the design pass open, W7 (volumes, section 4b) done but for its later pieces. Design reference: the Julia wizard,
`../mtexJulia/docs/plans/wizard.md` (W0 to W18 done there) and `../mtexJulia/ext/MTEXWizardExt/`
with `lib/MTEXInterfaces/src/{wizard,inspect}.jl`; its window as it stands after the W9,
W10 and W14 design passes is the look to reproduce. Syntax reference:
`../frameSymmetry/interfaces/import_wizard/` (`import_wizard.m`, `buildImportWizardScript.m`)
for the names only. `PLAN.md` section 1 named the wizard a non-goal; this subplan takes it
up now that every loader it sits on is done (I1 to I7).

Decided with the project owner on 2026-09-22: Qt widgets, not QML; the verb is
`importWizard()`, the Julia name; the inverse pole figures are three tabs of the strip, as
in both references, and a colouring of the map underneath.

## 1. Deliverable

One window that browses a folder, describes a file before reading it, imports the map on a
click, lets the crystallography and the two reference frames be corrected against a picture
of the result, and hands back either the map or the Python that reproduces it:

```
from mtex import importWizard
importWizard()                      # the current folder
importWizard('/data/ebsd')          # a folder, or a file to open on start
```

The three requirements the Julia port was held to hold here too: keep the design and the
functionality, be more responsive than MATLAB's window, and let the panels be resized with
the mouse.

## 2. Design

**The chrome is Qt widgets, the data is matplotlib.** The same split the Julia port settled
on (its section 2), with the reasons that made it hard there absent here. PySide6 ships the
tree view, the table view with editable delegates, the splitter, the tab bar and a file
system model, so no piece of chrome is written by hand; matplotlib ships a Qt canvas, so
the views are the package's own `plot(ebsd, ...)`, `plotPF` and `plot(image)` drawn into
supplied axes. A widgets window lays itself out and grabs itself as a picture under
`QT_QPA_PLATFORM=offscreen`, which the QML and Makie pairing never could, so the window is
checked by a script and not only by opening it. PySide6 is the `wizard` extra of
`pyproject.toml` beside `plot`, imported only inside the wizard, and a session without it is
told which package to install in the words `plotPF` uses for matplotlib.

**The core is headless.** Everything the wizard settles is an argument of `loadEBSD`: the
data set, the phases, the Euler correction, the scan unit. The state is those arguments plus
the map they produced and what the views derive from it; every control reads a field of it
and writes back through one function of the callbacks module; the window is a view. So the
whole wizard is driven and tested with no window open, and the window layer holds nothing
but bindings. Python has no observables: the state carries a `changed` hook per field
that the window connects to a Qt signal and the tests to nothing.

**The eight conventions are the whole coordinate model.** Two dropdowns over one table of
axis aligned conventions: *map coordinates*, which sets the session convention and only
re-views, and *Euler coordinates*, the frame the vendor writes its angles in, the correction
being `mapRot * inv(eulerRot)`. Both vendor corrections are members of the table, the EDAX
half turn about `(1, -1, 0)` and the Oxford half turn about `z`, so the wizard opens on the
convention that reproduces what the loader applied and the user moves away from it. Each
dropdown is labelled in the axis names of its own frame, `Y1↑→X1` for an Oxford file, `A2↑→A1`
for EDAX, `RD, TD, ND` for a rolling frame.

**A vendor file declares its own phases, so the crystallography is assigned after the load.**
The generated script assigns `ebsd.CSList` after the load and only where the user changed
something.

**A header is not a map.** Selecting a file is instant on a 200 MB one: nothing about the
preview reads a numeric block.

**The preview is a picture, not a mesh.** The map is drawn as one RGB image resampled off
the lattice, which `mapRaster` of the plotting layer already does, so a change of phase,
property or IPF direction is one image update and a change of map convention only moves the
view. A map on no lattice keeps the cell drawing.

**An Euler change is applied in place.** The rotations are turned by `new * inv(old)`, one
multiplication per measurement, and the density is rotated by the same, never re-estimated.

**The pole figure preview is estimated from a binned map.** Above a few hundred
orientations the centres are binned onto a regular grid of Euler angles, one representative
per cell weighted by its count, and `calcDensity(..., weights=, harmonic=True)` runs over
the cells: on a 150k orientation phase about 3000 cells and eight times faster, with the
maxima where the full estimate puts them.

**No compilation in the repaint.** The Julia window froze for its first draw because the
compilation ran inside Qt's paint callback and needed a precompile workload to move. Here
the numba kernels compile on their first call, once per process, and the load runs on a
worker thread; the drawing is matplotlib and compiles nothing. What is owed is only that
the kernels the loaders and the raster use carry `cache=True`, so the first import of a
session costs a read and not a compile.

## 3. The window

The Julia layout, panel for panel. A `QSplitter` with the browser side left and the analysis
side right, opening maximized and coming back to 1600 x 900.

*The browser side*, top to bottom: the folder as the end of its path, the last three folders
with the rest behind a leading ellipsis that lists them, each a target and each elided at
110 px, so that the panel's width is the browser's and not the depth of the folder open; the tree of the
folder, folders first, a folder unfolding where it stands on a click and walked into on a
double click, the vendor extensions as the filter, the arrow keys moving the current row
and the file the cursor stops on read, 300 ms after it stopped, a load under way dropped when it moves on to another file, its pole figures drawn ahead only after a rest of 700 ms, a finished job letting go of the map it held, Right unfolding a folder where it stands
and stepping into it, Left stepping out and folding it, Backspace walking up onto the folder
left, a letter going to the first entry starting with it and again to the next; below the tree a status band coloured by the state of the load, idle with the hint, amber while
loading, green with the time when loaded, red with the message when failed; the data set list, shown only when the
file holds more than one map, picking one being the import; the file information and the
metadata of the map as one tree of name and value, the file's own rows at its top - the
vendor, the date, the grid with its step, the extent, the Euler correction, what the window
shows nowhere else (`infoRows`: the name, size, points, phases, properties and data sets are
the list's, the band's, the table's and the tabs') - the header of the loaded map after them.
The script of the import is saved and copied from its panel beside the phase table (W4h).

No panel carries a heading, at the project owner's asking on 2026-09-22: what a panel
holds is said by what is in it - the file's own rows, the names of the data sets, the
columns of the phase table - and four headings over four panels are four labels to read
before the window can be. The phase table keeps its column names, which name the data and
not the panel.

*The analysis side*: the two convention pictograms stacked at the left, map coordinates
above and Euler coordinates below in red, each a framed square drawing the frame it stands
for and offering the eight as pictures when clicked; the phase table beside them, inline editable, with the columns Plot,
the phase cell filled with its colour and printed with its id, mineral, pixels, per cent,
symmetry as a dropdown over the point groups, `a b c α β γ` each under its own heading with
the cells the lattice forces greyed background and all, and the alignment as a dropdown over
the twelve setups; at the right end of the row the inverse pole figure key of one phase, at a
fixed size, the phase picked by a click on its row of the table and named in bold there; and under them the tab
strip over one canvas: *phase*, *IPF X*, *IPF Y*, *IPF Z*, one tab per property, *pole
figures*, *images* where the file brought one, coloured by category, a strip of
one tab hidden. A property map carries a colour bar; the pole figures carry editable Miller
labels and the Euler frame's axes in red; the window opens on *IPF Z* when anything is
indexed. Every panel in a `#c8c8c8` frame, the numeric columns right aligned, two font sizes.

## 4. Order of work

**W0 - the spike. DONE 2026-09-22.** `bench/wizard_spike.py`: a splitter, a tree over a file system model, an
editable table cell, a tab strip and a map on a matplotlib canvas recoloured on a click,
grabbed offscreen to `bench/out/`. Measures a recolour through the raster and a load on a
worker thread against the window staying live. Adds the `wizard` extra. Half a day; it
settles the stack for real and is thrown away.

*What it measured*, offscreen on `Forsterite.ctf` (245952 points): the load on the worker
thread 1.35 s with the window live throughout (511 event loop ticks meanwhile); the map
drawn through `plot(ebsd)` into the canvas's figure 0.77 s the first time; the raster
336 x 732; a recolour 100 ms of colours (the IPF lookup, 2.8 s the first time for the key)
plus 100 to 230 ms of Agg redraw; the editable cell, the tree over `QFileSystemModel` and
the tab strip all native; the grab `bench/out/wizard_spike.png`. The redraw is the lever
if the window ever feels slow: blitting the image alone, or a smaller figure.

**W1 - the core. DONE 2026-09-22.** All of it headless, one test file per module.

- `mtex/io/inspect.py`: `inspectEBSD(path)` returning an `EBSDFileInfo` with the vendor, the
  phases as crystal frames, the properties, the data sets, the grid, the step, the extent,
  the Euler correction, the date and the size, read from the header each loader already
  parses (`angHeader`, `ctfHeader`, the `.cpr` ini, the `.osc` header, the HDF5 configs with
  `h5DataSets`), plus `ebsdFiles(folder)` and `isEBSDFile`. Pinned to the point counts the io
  plan pins and timed against the load.
- `AXIS_CONVENTIONS`, `conventionNames(names)` and `closestConvention(rot)` beside
  `plottingConvention` in the geometry; a test that both vendor corrections are members to a
  deviation of zero.
- `closestAlignment(cs)` and `alignmentSetups()` in the frames: the twelve `X||_, Z||_` pairs
  that construct on every lattice, the setup a frame is reported in found by rebuilding it in
  each and comparing axes.
- `mtex/io/importscript.py`: `buildImportWizardScript(state)`, MATLAB's name, a pure
  function writing the shortest floats that read back exactly, naming the phase list
  `csList`, carrying the data set only where the file offers more than one and the phases
  only where they differ from the header. Tested by executing what it writes and comparing
  with the direct load: positions and phase ids identical, orientations to float noise.
- `mtex/wizard/state.py` and `mtex/wizard/callbacks.py`: the state and `selectEntry`,
  `openEntry`, `startImport`, `setDataSet`, `recolor`, `setMapConvention`,
  `setEulerConvention`, `applyEuler`, `pickTab`, `setMiller`, `pickMeta`, `showImage`,
  `editPhase`, `toVariable`, `writeScript`, `copyCall`, each a function of the state.
- The binned density in `mtex/wizard/binning.py`; the IPF key is already cached per frame
  by `ipfColors`, so nothing was added for it.

*What W1 found.* `inspectEBSD` describes every file of the MATLAB data folder in 7 to
180 ms, the phases, properties, grid, step and correction agreeing with the load on every
fixture; the head of a text file is read in 256 kB chunks until the header ends, the
`.osc` until its closing marker, the HDF5 through its metadata with the count off a data
set's shape. A map built from rotation matrices never recorded its Euler correction
(`_fromRotations`), so `ebsd.EulerCorrection` read as the identity on those files; fixed.
`ebsd.CSList = [...]` stored a bare list, since the setter did not wrap it; it does now.
The register interns a crystal frame within a 1 % shape tolerance (`frameShapeTolerance`),
so an edit of a lattice parameter below that came back as the file's frame; `editPhase`
mints the edited frame outside the register when the register hands back something else,
and the header phases are kept as a value snapshot (`frameKey`) so an edit on a shared
handle, a colour set on the frame itself, still reads as a change. The script therefore
reproduces an edit in a fresh session, where its `csList` is registered before the file is
read, and not in the session that already holds the file's frame - which is the register's
rule, tested as such. The binning of the Forsterite phase gives 2875 cells of 152345
orientations, the same count the Julia port measured, the pass 21 ms.

**W2 - the shell. DONE 2026-09-22.** `mtex/wizard/window.py` and `browser.py`: the window,
the breadcrumb, the status band, the tree over a flat list with a depth per row, the data
sets, the file information, the exports. The load on a `QThread`, the result committed
through a slot of a main-thread object. `bench/wizard.py` opens the window offscreen, reads
the Forsterite map through the worker, draws every tab into `bench/out/wizard-<tab>.png`
and `wizard.png`, and prints the panel widths, the column widths and the point sizes.

*What the shell taught.* **A signal connected to a plain Python callable runs on the
emitting thread.** The worker's `done` was first connected to a closure, so the commit and
every widget refresh ran on the worker thread and their own signals were queued back to the
main thread after the refresh had finished; the table then read its own filled cells as
edits, `editPhase` ran 88 times in one bench pass and Qt warned about a timer stopped from
another thread. The receiver is a `QObject` living on the main thread with a `@Slot`, the
connection `QueuedConnection`. **Every field announced redraws once**: the canvas hooks
schedule one draw through a queued `invokeMethod`, since a zero timer is not run by a single
`processEvents` under the offscreen platform. `closestAlignment` rebuilt twelve frames per
call, 0.75 s, on every table refresh; it is cached per frame shape. The table fills with its
signals blocked and a re-entrancy counter.

*Measured*, offscreen on `Forsterite.ctf`: the load 0.12 to 0.6 s with the window live; a
tab of the map 170 to 400 ms of which the Agg redraw and the grab are most; the first IPF
colouring 1 s for its key; the pole figures 1 s the first time, 300 ms after.

**W3 - the phase table and the coordinate panel. DONE 2026-09-22.** `phases.py`: the
`QTableWidget` of thirteen columns with the widths the Julia window measured (mineral 118,
alignment 108, the lengths 56, the angles 49), rows 28 high, the numeric columns right
aligned, the forced cells greyed background and all and not editable, the symmetry and the
alignment as combo box delegates over the point groups and the twelve setups with a `▾` on
the cell, the phase cell filled with the phase's colour and printed with its id in black or
white by luma and opening the colour dialog, a cell an editor on a single click.

`pictogram.py`, the two reference frames, **to the Julia window's design in both the
drawing and the choosing** (its `FramePictogram.qml` and `ConventionButton.qml`), which the
project owner asked for on 2026-09-22 after seeing the first pass:

- *A convention is a picture, so the button is the picture and the chooser is eight of
  them.* Clicking a frame opens the eight as pictograms, four to a row, the one in force
  marked, each with its name as a tooltip - not a menu of pictogram strings.
- *Each axis is drawn where it points.* `conventionArrows(pC)` gives the three components of
  every reference axis along east, north and out of the screen, so the drawing reads the
  frame rather than a string. The axis that leaves the screen is **named in the middle of
  the crossing** - no ring, no cross, no line under the letters - and the shafts start clear
  of whatever that name needs; the other two are arrows with their names past the heads.
- *The drawing is laid out in units of its own and scaled as a whole*, not as fractions of
  the square: a font taken as one fraction and an arrow as another do not keep their
  proportions together. `LEN 46, HEAD 14, LABEL 18, GAP 8, WEIGHT 4.5`, as Julia has them.
- *It is centred on its ink, not on its crossing*, which sits wherever the two arrows leave
  it - so `x` and `RD` would otherwise come out in different places. Measured over the eight
  conventions and three sets of axis names, the ink box is within 5 px of the centre of a
  104 px square and spans more than 80 per cent of it.
- The frame's name is drawn into the free quadrant, light and a little transparent, so the
  square needs no label of its own; the map is in ink and the Euler frame in the red the
  pole figures carry its axes in. The square is a framed panel and not a raised button, the
  ground lightening under the pointer. `bench/wizard.py` writes `wizard-frames.png` and
  `wizard-conventions.png`, since a popup is its own window and not in the grab of this one.

*A small map is drawn as cells, not as a picture*, by the package's own plot, so the
recolouring sets the collection's face colours where there is no raster, and the artist's
own alpha is cleared first or it overrides the transparency of a phase not drawn.

**W4 - the views. DONE 2026-09-22.** `views.py`: `ViewCanvas` over one matplotlib figure,
the map drawn once per import and map convention through `plotEBSD` and recoloured through
the raster index, a property with a colour bar, the pole figures of the dominant drawn phase
from the binned density through the package's `plotPF` with the Euler frame's axes as red
marks, the three crystal directions as fields over the canvas (`MillerBar`) shown on that
tab alone, the images the metadata holds, the metadata as `MetaTree` with the top level
open. The strip: phase, IPF X, IPF Y, IPF Z, one tab per property, pole figures, images
where there are any, coloured by kind; opens on IPF Z. (A *metadata* tab that only pointed at
the tree was dropped with W4f, when the tree moved into the browser side.)

**W4a - the redraw path. 2026-09-22.** The parallel session's `benchmark-mapplot.md` made
`plot(ebsd)` several times faster and pointed out that the wizard's redraw kept the per
measurement work the plot had dropped. Three things, now taken: the colours of the pixels
are gathered before they are made (`values[pick]`) instead of an `(n, 4)` buffer over every
measurement being built and then gathered; the picture is handed over as `uint8` where
`plotEBSD` had made one; and `commitMap` kept a second raster at the map's full resolution
beside the view's, which nothing read, so the state holds none and the canvas owns the one
it draws. Measured offscreen on the Forsterite map tiled to size, against that session's
numbers for the same three maps:

| points | commit | first draw | redraw phase | redraw IPF | peak on a recolour |
|---|---:|---:|---:|---:|---:|
| 245 952 | 0.13 (0.47) | 1.69 | 0.06 (0.03) | 0.52 (0.06) | 35 MB |
| 3.9 M | 0.10 (0.74) | 1.34 (0.87) | 0.17 (0.29) | 0.69 (0.56) | 206 MB |
| 15.7 M | 0.40 (2.98) | 2.70 (1.67) | 0.33 (0.71) | 0.93 (1.56) | 501 MB |

*Only the measurements the picture shows are coloured.* `mapValues(x, color, ..., at=)` and
`ipfColors(..., at=)` take the measurements to colour, which the parallel session added on
the same day at this session's asking, since `plotEBSD` gathered the same way and gained
the same. `drawMap` hands over the raster's own index: the colouring of the 15.7 M point
map falls from 0.63 s to 0.11 s and the peak allocation of a recolour from 501 MB to
160 MB. A route that yields numbers stays whole, so that the colour range is the map's and
not the picture's.

*A measurement with no colour is drawn through, not black.* `mapValues` answers NaN per
channel for an unindexed measurement under an orientation colouring, so the `nan_to_num`
that makes the bytes turns it into black unless the alpha is taken from the finite test on
the gathered colours. The same line in `plotEBSD` paints all 58485 unindexed points of the
Forsterite map opaque black; reported to the session that owns `mapplot.py`, not fixed
here.

**W5 - the outputs and the design pass.** The outputs are in place (W2); the IPF key of every
phase present stands beside an IPF map, drawn once and hidden on the other tabs. Still open: To variable through the caller's frame, the
script to a file in the working directory, the load call to the clipboard. Then the Julia
W9, W10 and W14 items as a checklist against `bench/out/wizard.png`: no elided lattice
parameter, every column wide enough for `-5-32/m` and `X||a*, Z||c*`, two font sizes,
three titles on one band, the rows of the browser the height of their line.

**W4f - one key, in a fixed place. 2026-09-26.** The project owner did not like the keys
over the map, whose size and place changed with the map and the canvas, and asked to join the
file information with the metadata browser so that the corner the browser held could show one
key. The file's rows and the metadata are one tree on the browser side (`InfoTree` in
`browser.py`); `KeyPanel` in `views.py` shows the key of `st.keyPhase` at 190 px, drawn at the
screen's pixels and cut to what is drawn so that its labels stand whole; a click on a row of
the phase table picks the phase (`pickKeyPhase`), the load picks the one with the most pixels,
and the key stays on every tab. The canvas carries no keys and gives the map its whole width.
A key of a group without the inversion whose sector reaches into the lower hemisphere (1, m,
mm2, 3, 4mm, 6, ...) is drawn as both hemispheres, each named, beside or over one another as
fills the square best; the key legend of a map plot (`keyLegend`) had drawn the upper one only
too and stacks the two in the phase's row now.
The same pass found the artefact the canvas showed around a map: a map plot resized a fresh
figure to its own shape, the canvas's figure too, and left the rest of the widget unpainted;
a figure pyplot does not manage is fixed in size now (`mtexFigure.fixedSize`).

**W4g - the pole figure tab, opened at once. 2026-09-26.** The project owner asked for the tab
to open immediately, the Euler axes as names on a white ground without a marker, the headings
close to the charts, a colour bar under each, the preview computed faster, and then for headings
in the window's font with the bars of a negative index in place. Measured first: the density is
7 ms once warm, and what costs is matplotlib, 220 to 380 ms to render three charts at the
canvas's size with or without the layout engine, which no faster plotting removes. So the
density runs on the worker behind the map (`ViewCanvas.density`, a result for a map since
replaced dropped), from at most 100 000 measurements drawn at random but the same every time
(`previewDensity`: 1.8 s to 96 ms on the 2.5 million points of 1C_1, the pole figure 1.7 % of
its maximum off), and when it arrives the pole figures are drawn off the screen on the next turn
of the event loop, at the canvas's size, as a picture the tab shows pixel for pixel
(`figimage`): the tab opens in 30 to 50 ms, 120 ms on 1C_1. Not on the worker thread, since
matplotlib's fonts are shared between figures. The picture is keyed by the density, the
directions, the Euler names and the canvas's size, so the Miller bar keeps its room when hidden
and the canvas has one size on every tab. The charts are evaluated at 1.5 degree (3 showed the facets of the grid), the row laid out
in a band as tall as it needs across the middle, the titles without padding. The window's font
reaches matplotlib through `guiFont` (the family Qt resolves its generic one to, mathtext in the
same family); the bar is mathtext's `\bar` now wherever a spherical plot writes a Miller index
(`plotText`), since matplotlib sets the combining overline `Miller.char` writes beside the
digit, and a label on a marker of size zero stands on its point.

**W4h - the script beside the phase table. 2026-09-26.** The room between the phase table and
the key, a hundred pixels at 1600 and several hundred on a wider screen, shows the import script
as it stands (`ScriptView` in `scriptview.py`): `scriptOf` rewritten with every change that alters
it - the map, the phases, the two frames, the phases drawn, the data set, the name the map is
bound to - read only in a small fixed-width font, comments grey and strings green, *save…* and
*copy* in its corner, children of the text so that they stay over it, each saying on itself for a
moment that it did its work. Narrower than 240 px a line cannot be read, and a note asks for a
wider window instead. The exports under the browser - the name, *to variable*, *write script*,
*copy load call* - went with it: the script is copied or saved from here, and `importWizard()`
returns the map. The vertex labels of a crystal sector stand centred over or under their corner
(beside it at half height), so that the key takes no width beside the sector and fills its panel.
A choice in a dropdown of the phase table is committed as it is picked, not when the focus leaves
the cell, so the key follows the click. Every import is numbered and only the latest is taken
(`st.imports`): a map clicked while a slower one was read had been replaced by it when that
finished, which looked like a click that did not open; and a double click on a file reads it
when it is not the one selected, since two quick clicks near one another come as a double click.

**W6 - the pages. DONE 2026-09-22** for what exists: the wizard section of
`EBSDAnalysis/EBSDImport` shows `importWizard()` and the offscreen grab, kept as
`docs/pages/EBSDAnalysis/importWizard.png` and copied into the build by `docs/make.py`
with every picture beside a page; the status in `CLAUDE.md` and `PLAN.md`.

**W4c - the pictures a file holds. 2026-09-22.** The project owner: the images view is what
the Julia window shows the SEM images in, picked from the tree of the metadata beside it,
and an Oxford file holds several of them. Neither was true here - the view looked for an
array in `ebsd.meta['header']`, the tree offered nothing to pick, and the vendors keep
their pictures where that never finds them.

*Where a vendor puts a picture.* Bruker writes `EBSD/Header/SEM Image` as a stack of one
channel, `(1, 1332, 2000)`, beside a second copy under `SEM/` and a diagram of its
coordinate system. Oxford writes eleven electron images and a layered colour one as **flat
runs** - `1/Electron Image/Data/FSE/Lower Left 26` is 3 145 728 int16 - whose width and
height are two fields of the nearest enclosing `Header`. EDAX writes its PRIAS images and
its coordinate diagrams as planes.

*So the loader indexes them and does not read them.* `h5ImageIndex` in `io/h5.py` walks the
file for anything that is a picture - a plane, a plane of three or four channels either way
round, a stack of one, or a flat run the nearest header counts - and records the path, the
shape and the type in `ebsd.meta['images']`. **The map's own columns are flat runs of
exactly that kind**, so anything with one value per measurement is not a picture of the map;
that rule alone takes the Oxford file from 29 found to 12. `h5Image(fname, entry)` reads one
when it is looked at, a plane for a grey picture and `(height, width, 3)` for a colour one.
Reading none of them keeps 70 MB of electron images out of a 195 000 point map.

*And the tree is the picker.* Every picture is a row of the metadata tree - the indexed ones
in a branch of their own, since nothing of them is in the metadata to hang them on, the
read ones on their own row - named by the last two parts of its path that say anything,
`FSE/Lower Left 26` and not `Data/Color`. A row says what the picture is, `picture 2048 x
1536`, and clicking it shows it and moves to the images view. The wizard keeps the pixels
it has read, so looking at one again is free.

**W4b - four things the project owner asked for on 2026-09-22, at the window.**

- *The path was the panel's width.* The breadcrumb drew every folder of it, so the browser
  could not be made narrower than the path was long - 433 px on this machine's data folder.
  It draws the last three, each elided at 110 px, with the rest behind a leading ellipsis
  that lists them from the root down, and states no width of its own: the panel's minimum
  is 193 px and it opens at 360, which holds the longest file name and the file table.
- *Every phase has a plot box, and the boxes are centred.* The phase that was never indexed
  had none, and `drawn` did not carry it; it does, and the drawing reads it like any other,
  so the unindexed measurements can be taken off the map. An item draws its check indicator
  against the left edge whatever its alignment says, so the column has a delegate that
  draws the cell without one and puts the box in the middle, and a click anywhere in the
  cell ticks it.
- *The keys lie over a map that fills the width* - replaced by the key panel of W4f.
- *Return walks into a folder*, as a double click does; a single click still unfolds it
  where it stands.

*What was slow was the keys, not the map.* Three inverse pole figure keys were drawn at
`plotKey`'s default half a degree over a sector about 100 px wide, tens of thousands of
quads for ten thousand pixels, and matplotlib re-renders them on every draw: 0.86 s of a
0.96 s draw, against 0.095 s for the map picture and everything else. Asked of the session
that owns `mapplot.py`, and answered the same day with a better rule than the fixed step
this session proposed: `keyLegend(..., resolution=None)` calls `keyResolution(px)`, which
gives a quad about two screen pixels over a sector some ninety degrees across and clamps
between 0.5° and 4°. The wizard's insets are 147 px, so they mesh at 1.23° and a draw with
the three keys falls from 0.96 s to 0.43 s with nothing visible in it; a key drawn large
for a page still gets the half degree. What is left is matplotlib re-rendering three
sectors that have not changed, which only caching them as pictures would take away.

**W4d - a map of two and a half million points, clicked to seen. 2026-09-22.** The project
owner asked for the path from a click on
`h5Data/Oxford/EBSD_incl_EDS.h5oina` - 281 MB, 2 476 571 points, five phases, 24 EDS
columns - to its IPF Z map, and named the keys as the thing to drop if need be. Profiled;
nothing was dropped.

| | before | after |
|---|---:|---:|
| click to loaded, the window live | 1.6 s | 2.0 s |
| the IPF Z map on screen | 2.3 s | 0.46 s |
| a later IPF tab | 1.8 s | 0.21 s |
| any other tab | 0.3 s | 0.17 s |

The load carries 0.4 s it did not: the colour grids of the five phases are built beside it
(4 below). Every number here is with the machine otherwise quiet; a parallel session's test
run moves them by a factor of two.

Four things, none of which is a smaller window of what the wizard shows:

1. *The keys were on the critical path and are not.* Five of them cost a second to build,
   and the user was looking at nothing while they did. They are built on the turns of the
   event loop after the map is drawn, **one to a turn**, so that neither the map nor the
   window waits for them.
2. *A key is a picture of its phase, so it is drawn once as one.* Matplotlib re-renders a
   mesh of tens of thousands of quads on every draw, which is what made a tab switch cost a
   second where the colouring cost a tenth. Each key is drawn once into a figure of its own
   at the resolution that picture wants, and shown as an image afterwards; a phase's picture
   is kept for the session, since a key does not change with the map or the window.
3. *The picture is drawn in the colouring the tab asks for.* It was drawn by phase and then
   recoloured, which built the raster twice and coloured the map twice; the raster of the
   recolour is built when a second tab asks for it and not before.
4. *The colour of an orientation is a look-up over a grid built once per phase*, 1.6 s for
   five, and it was built when the map was first drawn, with the window frozen for it. The
   phases are known from the header before the map is read, so the grids are now built
   **beside the read** on a thread of their own - numpy holds no lock over its own arrays and
   the register of the grids has one of its own.

What is left is the load, 1.6 s of the 2.9 s: 0.5 s of reading 24 columns of 2.5 M points,
0.4 s of gridding them, 0.2 s of quaternions. That is `EBSD.load` itself and the same for
any caller; the wizard shows it on a coloured band with the window live throughout.

*One lever not pulled.* The colour look-up grid is `0.5°`, and a map drawn on a screen does
not need it: at `1°` the grids cost 0.4 s instead of 1.6 s and the median colour differs by
2 of 255, at `2°` 0.1 s and 4 of 255. That default is `orientationkeys.py`, the parallel
session's file, and a question of what a colour is allowed to be off by; it is theirs to
answer, not this window's to force.

**W4e - where the time goes in an inverse pole figure key. 2026-09-22.** The project owner
asked where, and whether numba or a key held on disc would help. Measured, then answered
with a kernel.

*What a key is.* The colour of a direction changes smoothly over the sphere, so the key is
evaluated once on an equispaced grid - 165 016 directions at the half degree
`COLOR_GRID_RESOLUTION` - and every measurement afterwards is a look-up. **Julia does
exactly this**, `ColorGrid` in its `orientationkeys.jl`, cached by the frame's identity,
with the same arithmetic look-up, and it holds nothing on disc. MATLAB precomputes the same
way. The difference was the build: Julia's comment says a fifth of a second, ours took
1.85 s for a cubic key.

*Where those 1.85 s went.* Not in the colouring. **1.04 s was the symmetry fold**:
`projectIntoRegion` built the `(n, |G|, 3)` block of all 24 images of all 165 016
directions and tested each against the sector's planes, 0.53 s of it in `ufunc.reduce` over
temporaries. Another 0.24 s was `polarRadius`, the sector fold's radius, which builds a
handful of `(n, 3)` temporaries per boundary. The rest was spread thin over the hue and the
conversions.

*So the fold is a kernel.* `projectIntoSector` in `geometry/kernels.py`: a loop that takes
the images of one direction in turn and **leaves at the first one inside**, where the array
form has to build them all. 1.33 s to 0.048 s, **28 times**, the same answer to 1e-9. The
sector fold's radius is the same shape and went to `plotting/kernels.py` beside the colour
look-up that was already there: 0.24 s to 0.045 s. A cubic key is now **0.32 s**, five
phases of the Oxford EDS file 4.15 s to about 1.5 s, and what is left of a key is spread
over `sphereColor` 0.09 s, `azimuthAbout` 0.024 s and the colour conversions.

*The accuracy is not the wizard's to trade.* Against the exact key on 65 000 measurements
the median error is 0.9 of 255 at the half degree, 1.9 at one, 3.7 at two, where the grids
cost 0.32, 0.09 and 0.03 s. The session that owns `orientationkeys.py` was asked and kept
the half degree: under one level is invisible, the cost is once per crystal frame for the
session, and the same colours go into a saved file, so they should not depend on the size
of a window. A caller who wants it cheaper says `resolution=`.

*And a key on disc?* Not worth it now. A grid is 165 016 nodes, 0.5 MB as bytes, and the
common groups could be shipped - but a key belongs to a **frame**, not to a point group: the
sector is expressed in the crystal's own axes, so a monoclinic phase with another angle
between a and c is another key, and only the orthogonal, hexagonal and cubic groups have a
frame their group fixes. What a disc cache would save is now 0.3 s per phase, once per
session, against a cache that must be invalidated whenever the colour of a direction
changes - and a stale one shows wrong colours rather than failing. The kernel closed the
gap that made the question worth asking.

*One mistake, and what it cost.* `plotting/kernels.py` already held the map's colour
look-up, and this session wrote the file without looking first; there is no version control
here, so the original was gone. It was reconstructed from its call site, from
`S2Grid.nearestStepwise` and from its own test, which passes, and on 200 000 random cubic
orientations the rebuilt loop picks the identical node for every one. The session that
wrote it had the original in hand and diffed it: everything that changes an answer was
faithful, and what was lost was that the colour loop is compiled `parallel=True` over
`prange`, which it has restored. **Read a file before writing it, even to add one function.**

## 4a. What is owed

- The window has been driven by its callbacks and by Qt's item signals offscreen and
  grabbed as a picture; nothing has been clicked on a screen from this side.
- `importWizard()` blocks in a plain session and returns the map; under a running Qt loop it
  returns the window. Neither has been tried from IPython.

## 4b. W7 - volumes (2026-10-12)

MATLAB's wizard knows no volume, so this is this port's design: every view of the wizard is a
view of a map, and a layer of a volume is a map, so a volume is shown one section at a time.

1. **The description. DONE.** `inspectEBSD` describes a DREAM.3D file and the LabDCT volume of
   an xnovo file from the header: `info.is3d`, a grid and a step of three, `info.unit`, the
   properties `EBSD3.load` will give, and `info.grains` (feature ids, surface mesh). An HDF5
   file is asked whether it is a volume before the vendor reader sees it; `dream3dVolume` of
   `io/load3d.py` is the one description the loader and the inspector share.
2. **The import and the state. DONE.** `startImport` reads a volume with `loadEBSD3`. `st.ebsd`
   is the volume, `st.section` the map the views draw (the map itself for a 2D file),
   `st.sectionAxis` and `st.sectionIndex` say which layer, the middle one across z on import;
   `sectionOf(ebsd, axis, index)` takes the layer as a 2D `EBSD` with the two coordinates along
   it as x and y, `setSection` moves it and announces `section`.
3. **The views. DONE.** The canvas draws `st.section`, always as a picture; `SectionBar` under
   it is the x | y | z choice, a slider through the layers and the layer in words (its number,
   where it lies, and which axes of the volume the picture shows as x and y). Another layer
   across the same axis keeps the picture and takes new colours, 55 to 130 ms a step on a
   333 x 333 layer of 13 million voxels, where drawing it anew took 170 to 600. The pole
   figures are those of the whole volume, thinned as a large map is. The triad of the picture
   still names its axes x and y for a section across x or y: a frame cannot rename them.
4. **The script. DONE.** `buildImportWizardScript(..., volume=True)` and `loadCall` write
   `EBSD3.load(fname)`; the wizard returns the `EBSD3`.
5. **The frames. DONE.** The Euler coordinates of a volume are corrected as those of a map:
   `applyEuler` turns the orientations of the volume and takes the section anew, the script
   writes `ebsd = ebsd.rotate(eulerCorrection, keepXY=True)` after the load, the reader taking
   no correction. For this a moved volume stays a volume: `EBSD3._moved` carries the voxel
   lattice through `rotate` and `shift`, which returned a 2D `EBSD` before.
6. **Later.** A 3D tab of three slices, the grains of the feature ids or the mesh in the
   script, a stack of 2D files as a volume. Open.

## 5. Left out

The generic column assignment dialog, since `loadEBSD` on a table takes the columns as an
argument and says so when it cannot read them. Pole figures, ODFs and tensors: the MATLAB
wizard of this branch imports EBSD only. The `'wizard'` flag the MATLAB loaders thread
through. Nothing of MATLAB's render signatures, lazy axes or warm-up timer: a state with
hooks and a tab built on first activation are the same laziness without the bookkeeping.

## 6. Open

- The IPF key per phase costs a second on first use; whether it is worth computing on the
  worker thread ahead of the tab.
- Whether `inspectEBSD` belongs on `EBSD` as `EBSD.inspect` as well, the way `load` does.
