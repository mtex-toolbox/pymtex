# Subplan P6 - interfaces

Status: written 2026-09-21, I1 to I4 done the same day, which is the whole EBSD import, I5
the export on 2026-09-22; the pole figures, the lists and the tensors follow when their
consumers exist. Syntax reference:
`../frameSymmetry/interfaces/` with its `CLAUDE.md` and `info.md`, and
`../frameSymmetry/EBSDAnalysis/@EBSD/load.m`. Design reference:
`../mtexJulia/lib/MTEXInterfaces/` and its `docs/plans/interfaces.md`, I1 to I8 done there,
whose rules this document takes over and whose MTEX pins it reuses.

## 1. Deliverable

A package `mtex.io` whose public layer reads like the MATLAB of the branch:

```
ebsd = EBSD.load('map.ang')                       the format from the extension
ebsd = loadEBSD('map.ctf', eulerCorrection=rot)   the same verb as a function
ebsd = loadEBSD('map.txt', CS=[cs1, cs2], columnNames=['x', 'y', 'Euler1', 'Euler2', 'Euler3', 'phase'])
```

and whose inside is the Julia design: the format is the extension, a convention the vendor
states is an argument with a documented default, a phase block is placed by the number it
declares, what describes the map goes to `ebsd.meta`, one map constructor for every format.

## 2. Decisions

**MATLAB names.** `EBSD.load` and `loadEBSD` both exist in MATLAB and both here, one a
class method importing `mtex.io` lazily so that the maps layer stays below it. The keywords
are MATLAB's options: `eulerCorrection`, `setting` (0 to 4, the EDAX alignments), `noGrid`,
`interface`, `columnNames`, `columns`, `CS`, `radians`, `delimiter`, `header`, the flags
`EDAX` and `EMSphInx`, the Euler convention a positional string `'Bunge'` or `'ABG'`. The
header a loader reads goes to `ebsd.meta['header']` as the flat dict MATLAB builds in
`ebsd.opt.header`, since that is what an exporter writes back; the acquisition surface of
an Oxford file to `ebsd.meta['surface']`.

**Julia rules, taken over without rediscussion** (`interfaces.md` sections 2 and 3):
the phase numbering read off the non-negative values, a number outside the list a
measurement of no phase; a negative confidence index a failed pattern; the EMSphInx
fingerprint and its unmeasured rows; the `.ang` columns fixed by the format for the first
eight and `extra1, ...` beyond; the `.ctf` read by the names it gives; the Oxford surface
tilt undone before the half turn about z with a warning; gridify on load with MATLAB's
three guards. Where the two references differ on a user-visible name the MATLAB spelling
is taken; the Julia `grid = false` is MATLAB's `noGrid=True`.

**The numeric block is a numba row parser, `np.loadtxt` its fallback and oracle.** The
Julia design: a row is scanned over its bytes and each field read where it lies, the rows
in parallel, the decimal comma decided once for the block, a row of the wrong width naming
itself. `io/kernels.py` in the D5 shape, both backends tested against each other. The first
version was `np.loadtxt` alone, 2.5 s on 2.5 million rows; the parser takes 0.33 s.

**The grid a header states is the lattice, verified against the positions.** MATLAB prefers
`XSTEP`/`YSTEP` over the estimate from the positions as a unit cell hint; this port goes
further: `XCells`/`YCells` or `NCOLS_ODD`/`NCOLS_EVEN`/`NROWS` with the steps fix the index
of every measurement in file order, one pass checks the positions are on it, and the map
gets its lattice without the inference, which took 3.5 s of the 9.6 s the first version
needed on `1C_1.ctf` (`gridLattice` in `load.py`). Where the positions disagree with the
header, a cut file for instance, the unit cell hint and the inference take over as before.

**No warning for a documented default.** MATLAB warns on every load that it assumed the
vendor's alignment; the Julia port does not, and neither does this one. The default is in
the docstring and `eulerCorrection` overrides it. The tilt of an acquisition surface warns,
since there the loader changes what the file says.

## 3. Order of work

**I1 - the text EBSD formats. DONE 2026-09-21.** `mtex/io/table.py` (the header lines,
the numeric block, the flat header dict), `kernels.py` (the row parser), `load.py`
(`loadEBSD`, the dispatch, the phase numbering, `ebsdFromColumns`, `gridLattice`,
`tryGridify`), `ang.py`, `ctf.py`, `generic.py`; `EBSD.load` as a class method in
`maps/ebsd.py`. Fixtures cut from `../frameSymmetry/data/EBSD` into `tests/fixtures/ebsd/`;
the full files pinned to the MTEX numbers of the Julia plan and skipped where the data
folder is absent, and every one agrees: `ferrite.ang` 63045 rows on a 270 x 234 hexagonal
grid, first orientation Bunge (27.6983, 113.4183, 92.6414)°; `Forsterite.ctf` 245952 rows,
phases 58485 / 152345 / 26058 / 9064, first orientation (264.6750, 137.6500, 113.6700)°;
`olivineopticalmap.ang` 49729 rows with olivine 44953, Dolomite 1370, Enstatite 2311,
Chalcopyrite 1095; `85_829grad_07_09_06.txt` 60775 rows, 11411 not indexed, 48184 Fe,
1180 Mg; `1C_1.ctf` 2529549 rows, 1916888 indexed. 80 tests. Timed against the Julia
loaders on one machine in `docs/research/benchmark-io.md`: 0.86 s for the 2.5 million rows
against 2.5 s, and two to three times faster on the sample files. Import cost of
`mtex.io`: nothing measurable, numba is compiled on the first load. Three things met on the
way: MATLAB's `makeValidName` turns `#` into `x_`, which the header dict follows; a cut of a
map disagrees with its own header, so the grid path has to fall back rather than trust the
header; and NumPy's `np.unique` on 2.5 million integer keys costs 2 s in the maps layer's
`hasCollision`, which is what any map without a usable header pays and belongs to P3.

**I2 - the Oxford pair. DONE 2026-09-21.** `crc.py`: the `.cpr` INI as nested dicts in
`ebsd.meta['header']`, its `[Fields]` as a NumPy structured dtype, the `.crc` one
`np.fromfile` with each column a strided view of the records; either name of the pair
reads it, the space group the sharper statement over the Laue code, a phase without a
lattice not indexed with a warning, the acquisition surface kept and not applied. Pinned
against the Julia reader on `martensite.cpr` (343602 pixels, 92415 / 251187 / 0, first
indexed orientation (332.4892, 44.3015, 64.4717)°, mean MAD 0.470577) and `EDXLMDTi64.cpr`
(196608, 10449 / 437 / 185722, surface (-90, 0, 0)); the Julia fixture `martensite_cut`
copied over. 12 tests. What the step brought to the layers below: the Euler angles to
quaternions were 0.37 s of every large load in NumPy, so `matthies2quat` now goes through
a numba loop in `geometry/kernels.py` (`eulerQuat`) for a thousand triples or more, tested
against its NumPy form; with it `1C_1.ctf` reads in 0.61 s, `martensite.cpr` in 0.062 s
against Julia's 0.049 s, `EDXLMDTi64.cpr` in 0.033 s against 0.039 s.

**I3 - the EDAX binary map. DONE 2026-09-21.** `osc.py`: the marker cut, the stride read
off the second pixel, the block one `np.frombuffer`, the phase records found where a TSL
symmetry code can sit in the header and kept where the bytes before it read as a name and
the bytes after it as a cell. What MATLAB reads out of the fixed places of the header,
operator, sample and scan id, the calibration, goes to `ebsd.meta['header']` with the cell
counts and the steps. Pinned against the Julia reader on `copper.osc`: 16116 points,
first orientation (235.9327, 95.8342, 134.1963)°, mean image quality 53823.9668; the
Julia fixture `copper_cut.osc` copied over. 7 tests.

**I4 - the HDF5 maps. DONE 2026-09-21.** `h5.py` on h5py, the optional extra `mtex[io]`,
with the six vendor configs of the Julia port in `mtex/io/hdf5_config/`, which are MTEX's
with the two entries the Julia plan added. The reader is the resolver of `h5.jl` on h5py:
the four match modes, the outermost matches as the data sets, the phase groups with the
cell in any of its three shapes and the point group by space group number or EDAX code,
the Euler angles as one data set or three or a stack, EDAX's table of matrices, positions
measured or implied, the phase numbering with the negative marker, every per pixel data
set a property, EDAX's unplaced pixels dropped, the correction as a constant, a scan
rotation or a row of a table, the tilt of an Oxford surface undone, the header group in
`ebsd.meta['header']` and the provenance record in `ebsd.meta['h5']` for the export.
`dataSet` counts from zero, as `columns` does. Fixtures: `bench/cut_h5.py` cuts the first
rows of a vendor file with its whole tree, so `tests/fixtures/ebsd/h5/` holds one real
layout per vendor at 4 to 200 kB, eight files; the corpus in `~/mtex/data/h5Data` is
pinned to the Julia reader's counts, first orientations and property means and skipped
where absent. 26 tests. Left out until P5: the pictures beside a map (`electron_image`).

**Checked against MTEX on the paper corpus, 2026-09-22.** The 22 EBSD files of
`../mtex-paper`, 59 million measurements, read by both and compared number by number:
all agree; `docs/research/benchmark-io.md` has the table and `bench/matlab/` the probes.

**What the four steps did to the load path.** Every format ends in `ebsdFromColumns`, and
a map whose file states no grid now gets one from the order of its positions
(`scanGridFromPositions`: a row ends where x jumps back, a hexagonal scan alternates two
row lengths half a step apart) before the inference of the maps layer is asked; a
statement, verified against every position. Where the positions are not a raster, Bruker's
sample coordinates wobble in y by whole steps, the inference runs, and two of its passes
were NumPy's hash based `np.unique` on millions of keys: `hasCollision` in
`maps/lattice.py` and the collision test of `maps/gridify.py` mark cells instead, which
took the largest Bruker map from 3.9 s to 1.2 s. The numbers of every format against the
Julia loaders are in `docs/research/benchmark-io.md`.

**I4 - the HDF5 maps.** `h5.py` on `h5py`, an optional extra `mtex[io]`, with the six vendor
configs copied into `mtex/io/hdf5_config/`, the resolver of the Julia port.

**I5 - export. DONE 2026-09-22.** `exportebsd.py`: `export(ebsd, fname)` and
`ebsd.export(fname)`, the format from the extension, `.ang`, `.ctf`, the HDF5 formats, and
any other name a table `x y phi1 Phi phi2 phase` plus the properties under a line of names,
which `loadEBSD` reads back. An exporter undoes what its loader applies, the setting's
rotation for `.ang`, the half turn about z and the tilt of a stated surface for `.ctf`, and
writes back the header the loader kept in `ebsd.meta`; only the phases present are
declared, counted from one, since a declared and unused phase is what makes a file read
zero based. The HDF5 write-back is the Julia design: the file the map came from, or
`reference`, is copied and only the data sets the provenance record of the import names
are written into the copy, the angles in the file's own unit and layout, the phase column
in the vendor's numbering with its marker, a renamed mineral and a corrected cell into the
phase groups, the properties into their data sets and a property the map gained beside
them; the EDAX table of records is written field by field, its matrices column by column.
The rows of a text file are formatted by a numba loop in `io/kernels.py` (`formatTable`,
the mirror of the parser: two passes, one sizing every row and one writing them in
parallel, printf's rounding and sign), `np.savetxt` the fallback and oracle. Round trips
on every fixture: a `.ctf` comes back bit for bit, an `.ang` to the 1e-5 rad its five
decimals allow, the eight HDF5 layouts to float32 with the map rotated, a property changed,
one gained and a mineral renamed, and everything else the file held still there. 19
tests. On `1C_1.ctf`, 2.5 million rows: `.ctf` 1.8 s, `.ang` 1.3 s, the table 1.6 s, the
Julia writer 4.9 s and 3.5 s; the HDF5 write-back of the 2.66 million point Bruker map
1.05 s. Not written: the Oxford binary pair (`export_crc`), as in the Julia port.

**The sample data. DONE 2026-09-22.** `sampledata.py`: `mtexdata()` the table,
`mtexdata('forsterite')` the map its page uses, `mtexdatafile` the file alone. The files
come from the pinned MTEX revision of the Julia port with its digests (`sampledata.json`,
generated from `Artifacts.toml`), kept under `getMTEXpref('mtexDataPath')`, `MTEX_DATA`
or `~/.mtex/data`, and a local copy of MTEX's data folder, or its `EBSD` folder, serves
as well (`root=`, `download=False`). The recipes are MATLAB's: the phases and columns of
the text maps, the Euler correction of each vendor file, the crop of `small`, the turn of
`olivine`, the proper groups of `martensite` and `alphaBetaTitanium`, the names of the
EMSphInx phases, the two phases of `epidote` the page leaves out; `testgrains` is the
Julia port's synthetic map of twelve shapes through `calcGrains`, no file. Two things
the recipes needed: the generic loader's `replaceExpr` for the CSL map that spells a
failed angle in Italian, and `nan` as a value the row parser reads, an angle that is not
a number being no orientation. The VPSC run is listed and says it waits on P4. Every
recipe pinned on the local data folder, 30 tests.

**I6 - pole figure import. DONE 2026-09-22.** `mtex/io/loadpolefigure.py`: `loadPoleFigure`
and `PoleFigure.load` for one file or a list, the format read off the extension or the
first bytes, `interface=` naming it; the readers of the sample data and the pages -
`loadPoleFigure_dubna` (`.cnv` on the goniometer's own `DubnaGrid`, `.cns`),
`loadPoleFigure_siemens` (`.ptx` dumps), `loadPoleFigure_geesthacht` (the TEX EINZEL
table), `loadPoleFigure_uxd` (Bruker), `loadPoleFigure_aachen_exp` (AUTEX), `loadPoleFigure_rw1`
(the Philips binary) and `loadPoleFigure_generic` (columns named `polar angle`, `azimuth
angle`, `intensity` or `x`, `y`, `z`, a block of more than fifteen columns a regular grid);
`string2Miller` reads the reflection off a name, four indices with signs before three
without, several groups a superposition. The crystal directions and structure coefficients
given to `loadPoleFigure` replace what the files say. The recipes of `mtexdata`: `dubna`
(quartz `321`, a = 4.916, c = 5.4054 from MTEX's cif, the coefficients of the third file),
`geesthacht`, `ptx`, `steel` (turned by 90°), `zncuti` (zinc's lattice, the page states
none), `aachenexp`, `dubnaODF`. Fixtures of five formats under `tests/fixtures/polefigure`,
the Geesthacht and Seifert files read from the local data folder; 7 tests plus the sample
data test. The MATLAB `'HEADER', 21` of the generic page example skips the first data row
of the Seifert file; `header` counts the lines above the block, and without it the block
is found by itself. Not ported: the twenty other `loadPoleFigure_*` readers of MATLAB (they
follow the same pattern when a file asks for them), the import wizard.

**I7 - tensor files. DONE 2026-09-22.** `mtex/io/loadtensor.py`: `loadTensor` and
`tensor.load`, `stiffnessTensor.load`, ... for the text files of `data/tensor`: the last
block of numeric rows of one width is the tensor (six rows of six a Voigt matrix, three of
six the Voigt table of a rank three, three of three a rank two; the lattice line above
the block has the width of the block and is left out), the density the number below the
word `density` (kg/m3 turned into g/cm3), the crystal frame given positionally. Fixtures
of the olivine and quartz files, 3 tests. Not ported: `loadTensor_json`, `loadTensor_mpod`.

**I8 - the pictures an HDF5 file holds. DONE 2026-09-22.** `h5ImageIndex` and `h5Image` in
`io/h5.py`: the loader indexes what a file holds as a picture - a plane, a plane of three or
four channels either way round, a stack of one channel, or a flat run whose nearest
enclosing `Header` counts its cells - into `ebsd.meta['images']` as a path, a shape and a
type, and `h5Image(fname, entry)` reads one when it is asked for. The pixels stay out of the
map: one Oxford file carries eleven 2048 x 1536 electron images and a layered colour one, 70
MB, beside a 195 000 point map. A data set with one value per measurement is the map and not
a picture of it, which is what tells an electron image from a band contrast column, both of
them flat runs. Bruker's `SEM Image` is a stack of one channel, EDAX's PRIAS images are
planes. The import wizard lists them and draws one on demand.

**Later.** `loadODF`, `loadOrientations`, the VPSC file, the Neper tessellation, the pole
figure export.

## 4. Left out

The import wizard (now `docs/plans/wizard.md`), `loadData`'s search over every loader, the forty-phase rule of
`loadEBSD_generic`, mineral names in the data block of an `.ang` (MATLAB substitutes
numbers for them; no file here has them), `.mat` files, `functionSignatures.json`, the
`dream3d` and `xnovo` loaders marked WIP in the branch.
