# Benchmark I1 to I4 - reading the EBSD formats, against MTEX.jl

Run 2026-09-21 on one machine, twelve cores, both sides warm and run one after the other:
Python 3.12, NumPy 2, numba, h5py, `bench/load_text.py`; Julia 1.12.7 with twelve threads,
`julia -t auto --project=dev` on the `mtexJulia` loaders (`bench/load_text.jl`). The Julia
numbers are what its own `loadEBSD` takes here today, not the numbers of its plan. `list`
is the map as the file holds it (`grid=False`, `grid = false`), `grid` the map on its
raster, which is what both return by default. Best of three, at the end of I4.

| file | rows | Python list | Python grid | Julia list | Julia grid |
|---|---:|---:|---:|---:|---:|
| `Forsterite.ctf` | 245 952 | 0.057 s | 0.083 s | 0.209 s | 0.201 s |
| `ferrite.ang` (hex) | 63 045 | 0.020 s | 0.029 s | 0.058 s | 0.113 s |
| `olivineopticalmap.ang` | 49 729 | 0.021 s | 0.025 s | 0.039 s | 0.063 s |
| `1C_1.ctf`, 151 MB | 2 529 549 | 0.63 s | 0.92 s | 2.51 s | 2.22 s |
| `martensite.cpr` | 343 602 | 0.058 s | 0.092 s | 0.049 s | 0.080 s |
| `EDXLMDTi64.cpr` | 196 608 | 0.033 s | 0.052 s | 0.039 s | 0.042 s |
| `copper.osc` (hex) | 16 116 | 0.008 s | 0.010 s | 0.006 s | 0.011 s |
| `h5_edax.h5` (hex) | 56 615 | 0.027 s | 0.035 s | 0.279 s | 0.611 s |
| `KamilaTi.oh5` | 70 760 | 0.040 s | 0.047 s | 0.037 s | 0.037 s |
| `oli_test.h5oina` | 30 132 | 0.021 s | 0.024 s | 0.031 s | 0.029 s |
| `oxford.h5` | 138 890 | 0.052 s | 0.066 s | 0.088 s | 0.038 s |
| EMSphInx `..._BW88.h5` | 485 140 | 0.102 s | 0.153 s | 0.286 s | 0.125 s |
| `BCWQ10min2.h5` (Bruker) | 666 000 | 0.202 s | 0.345 s | 0.316 s | 0.298 s |
| `h5_bruker.h5` | 2 663 999 | 0.73 s | 1.18 s | 0.527 s | 0.492 s |
| `KamilaTi.edaxh5` | 70 760 | 0.047 s | 0.055 s | 1.064 s | 0.177 s |

The same numbers come out on both sides: the phase counts, the first orientation, the
property means and, on `1C_1.ctf`, the 1 916 888 indexed points the Julia plan records.
The first call in a session compiles the row parser, 1.6 s once with `cache=True`, and the
first point group of a session compiles the quaternion product, 0.4 s.

The binary and HDF5 formats are a few tens of milliseconds on either side, where the fixed
cost of a load, the point groups, the frame register, the plane fit, is what is measured.
The two Bruker maps are the one place Julia stays ahead: their sample coordinates wobble
in y by whole steps, so no scan grid is read off them and the lattice is inferred, which
in this port is `maps/lattice.py` on NumPy against a Julia loop.

## Where the time went, and where it goes

The first version took 9.6 s on `1C_1.ctf`. Its profile, and what replaced each part:

| part | before | after | what changed |
|---|---:|---:|---|
| the numeric block | 2.46 s | 0.33 s | `np.loadtxt` replaced by the numba row parser of `io/kernels.py`, the rows in parallel, each field read where it lies |
| cutting the file into lines | 1.54 s | 0 | only the header is cut into lines; the block is handed over by its offset |
| the lattice | 3.52 s | 0.30 s | read off the grid the header states and verified against the positions, instead of inferred; `hasCollision`'s `np.unique` alone took 2.1 s |
| the Euler angles to quaternions | 0.37 s | 0.37 s | `matthies2quat`, NumPy |
| the correction product, the map | 0.5 s | 0.4 s | unchanged |
| `gridify` | 0.71 s | 0.27 s | the lattice comes with its index, so the fast path reshapes |

What remains is spread thin: the parser 0.33 s, the Euler conversion 0.37 s, the grid
check 0.3 s, the rest in allocations of a 2.5 million row table. The two that would pay
are outside `mtex.io`: `matthies2quat` as a numba kernel in the geometry layer, and the
lattice inference of the maps layer, whose `hasCollision` goes through NumPy's hash based
`unique` on 2.5 million keys where a `bincount` on the bounded linear index would take a
few milliseconds - it is what any map without a usable header still pays.

## A stopped scan (2026-09-28)

`Scan1.osc` of the Rohrer data set, 1 229 173 hexagonal points of a scan stopped in its 615th
row, took 17.5 s. The file keeps the count it announced and fills the 1750 pixels never
measured with placeholders at (0, 0), all on the first pixel's site; the affine index refused
the collision, the neighbour graph was grown in Python for 14 s, and the map stayed a list.
Warm, one process:

| part | before | after | what changed |
|---|---:|---:|---|
| the placeholders | - | 25 ms | dropped by `oscTrim` |
| the lattice | 14 s, then 330 ms | 15 ms | a partial last row allowed in the row order; the row index one compiled pass (`gridKernel('rowOrder')`) that also shifts it; float32 rounding of the positions within the tolerance of the header steps, so the least squares confirmation is skipped; an exact affine fit with duplicates is kept rather than grown |
| `byEuler` | 63 ms | 30 ms | `eulerQuat` over the cores |
| `EBSD(...)` | 57 ms | 40 ms | the NaN test of the positions over the whole array first |
| `gridify` | 200 ms | 90 ms | the scan cells once, in one compiled pass, with their extent; the cell indices of the array likewise; `np.take` for the padded copy |
| `loadEBSD` | 17.5 s | 0.23 s | 0.5 s for the first call of a process |

In the import wizard the largest of the set, `Scan15.osc` (3.7 M points), then spent 2.5 s on
the main thread after its load: `mapRaster` 1.25 s (the lattice residual measured again, 0.56 s,
now carried from the loader's check; the hexagonal pixel lookup, now a compiled loop), the phase
counts sorted fifteen times per commit, 0.37 s (now one `bincount` per map), and the pole
figures, 0.43 s. It is 0.26 s of drawing and the pole figures now. The Euler correction no longer
copies its one rotation out to the map's length (`qmul` with a constant left factor), 114 MB.
The load peaks at about 1 GB for a map of 379 MB: the file, the list map, and the gridded copy
made while the list is alive; on a machine that swaps this is what makes a load of 1.1 s take 2
to 4 s.

Then (2026-09-28, second pass):

- The grid is written while the file is read. `loadEBSD` tells the loaders it wants one
  (`_gridOnRead`); where the row order gives the lattice, `mapFromColumns` scatters every column
  straight into its grid array, so the list map is never built. Bruker's wobbling positions
  still infer and grid afterwards. Checked against the list gridded afterwards for every format.
- A header grid the positions refute falls back to the row order before inference, and the
  HDF5 path reads the row order too.
- NumPy asks the kernel for huge pages on large arrays. On this machine, with fragmented
  memory, every page fault then waited for compaction: a fresh 1 GB array took 2.4 s to fill,
  0.24 s without the request. The preference `hugePages` (default False) switches the request
  off at import. This, not the arithmetic, was most of the 2 to 7 s loads.
- `Scan15.osc`, 3.7 M points: 0.82 s, a traced peak of 791 MB (was 997 MB), stable across runs.
- In the wizard every file of the set is on screen within 1.2 s of being selected.

## Why the parser is exact

A field is an integer mantissa and a power of ten. Up to fifteen significant digits the
mantissa is exact below 2^53, the powers to 1e22 are exact doubles, and one division or
product is one rounding, which is what strtod does; so every value a vendor writes comes
out bit for bit as `np.loadtxt` reads it, which the test checks on the fixtures and on a
block of edge cases. From the sixteenth to the eighteenth digit the mantissa rounds once
more, an ulp or two, and beyond that digits are dropped; an exponent past ±22 goes through
the nearest double of the power, which is what such a value's own rounding already cost.

## Writing (I5, 2026-09-22)

The same machine, both warm. The text rows are formatted by the numba loop of
`io/kernels.py`, two passes over the table; the HDF5 write-back copies the file and writes
the resolved data sets into the copy.

| map | Python | Julia |
|---|---:|---:|
| `1C_1.ctf` to `.ctf`, 150 MB | 1.77 s | 4.91 s |
| `1C_1.ctf` to `.ang`, 160 MB | 1.29 s | 3.50 s |
| `1C_1.ctf` to a table, 244 MB | 1.59 s | |
| `h5_bruker.h5` written back, 2 663 999 points | 1.05 s | |

## The paper corpus against MTEX (2026-09-22)

Every EBSD file of `../mtex-paper`, 22 files, 59.3 million measurements: twenty `.ctf`
of 2.5 to 3.3 million points, an Oxford `.cpr`/`.crc` pair of 3 million and a Bruker
HDF5 map. Read by MTEX on the `frameSymmetry` branch through its bridge (`EBSD.load(f,
'noGrid')`, R2024b, master removed from the path since its `@Miller` shadows the
branch's) and by the port (`loadEBSD(f, grid=False)`), one bridge call per file, and
compared on the number of points, the phase counts, the mineral names and point groups,
the cells, the first indexed measurement with its Euler angles and position, the mean
rotation matrix over the indexed points, the extent and the mean of every property
(`bench/matlab/probe_ebsd.m`, `probe_python.py`, `compare.py`, `sweep_paper.sh`).

**All 22 agree in every number.** MTEX took 99 s for the corpus, the port 37 s; per file
MTEX reads a 3.3 million point `.ctf` in 3.2 s where the port takes 1.2 s, and the
slowest, `C75_350_02.ctf`, 9.6 s against 2.6 s. The Oxford pair is the one file MTEX reads
faster, 0.58 s against 0.76 s, the fixed cost of the port's frames and point groups on a
map of three phases.
