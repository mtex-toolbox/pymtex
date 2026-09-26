# MTEX for Python - rough plan

Status: draft, 2026-09-21. Nothing below is implemented. This is the top-level plan: what
the port covers, which decisions to take, and which experiments should take them. Each
piece gets its own subplan under `docs/plans/` once the decisions here are settled.

## 1. The starting point is the Julia port, not the MATLAB tree

MTEX exists twice: the MATLAB toolbox (`../master`, 150k lines, native code for NFFT,
Voronoi and chain ordering) and MTEX.jl (`../mtexJulia`, 32k lines plus a 4k Makie
extension), which is complete through geometry, function spaces, maps, pole figures,
tensors, interfaces, plotting and 199 of 228 documentation pages.

MTEX.jl already did the expensive part of a port: it separated the mathematics from the
MATLAB workarounds, wrote the design down (11 ADRs in `../mtexJulia/docs/adr/`, one subplan
per module in `../mtexJulia/docs/plans/`), measured the alternatives (`docs/research/`),
and mirrored every MATLAB `check_*.m` as a test. So the source of truth for the Python port
is layered:

| question | answered by |
| --- | --- |
| what the user writes: class names, constructors, properties, verbs, flags | the MATLAB branch `../frameSymmetry`, which already spells ADR 0008 (`crystalFrame` carries the group, `crystalSymmetry` is gone, `Miller` returns a `vector3d` in a crystal frame) |
| what a function means, which model of frames and symmetry | MTEX.jl and its ADRs, which carry MATLAB ADR 0003 and 0008 |
| the order of work, what is dropped, what is left out | `../mtexJulia/docs/plans/*.md` |
| which numbers must hold | the MATLAB `tests/core` and the Julia `test/` files |
| the documentation | `../mtexJulia/docs/literate/` |
| the MATLAB code itself | consulted only where the Julia port says it left something out |

`../master` is on `develop` and still has `crystalSymmetry`; it is consulted for numbers
and for code the branch has not touched, never for syntax. When the two references
disagree on a name the user sees, MATLAB wins; on a representation or an algorithm,
Julia wins.

The rule of MTEX.jl holds unchanged: **no unit is transcribed.** Not from MATLAB, and not
from Julia either. Julia's type-parameter machinery (`Vector3d{V,F,B,N,A,E}`, `Val`
dispatch, isbits atoms, package extensions, precompile workloads) exists to make Julia
fast and does not exist in Python. What transfers is the design: which concepts exist,
what each one carries, the kernels, the mathematics, the tests, the pages.

## 2. Goals and non-goals

Goals:

- The user layer is the MATLAB one: the same class names, constructors, properties,
  verbs and doc pages as the frameSymmetry branch, so that a MATLAB user reads Python
  MTEX without a glossary. Only what Python cannot spell changes (zero-based brackets,
  keyword options, `cat` for brackets).
- NumPy is the substrate. One object holds many entities, as in MATLAB; the arrays inside
  are struct-of-arrays NumPy arrays; broadcasting is NumPy broadcasting.
- Pure Python and NumPy/SciPy wherever it is fast enough; compiled kernels only where an
  experiment shows they are needed, and then behind one small, replaceable seam.
- `pip install mtex` on Linux, macOS and Windows with wheels for every dependency.
- Compact, readable code; one file is one concept; one test file per source file.

Non-goals:

- Bit-identical numbers. Same mathematics, tested against MATLAB and Julia where a number
  is worth pinning.
- MATLAB-only machinery (`dynOption`, option string parsing, `obsolete/`,
  `compatibility/`, mex) and Julia-only machinery (see section 1).
- Reproducing either file layout.
- Nothing of the GUI beyond the import wizard, which is `docs/plans/wizard.md` (P8),
  planned 2026-09-22 on Qt widgets.
- GPU. Design the kernels so a NumPy-compatible array library could be swapped in later,
  do nothing else.
- MATLAB functions decided not to be ported (user, 2026-09-26): `calcFEMODF` (the finite
  element pole figure inversion) and `neuralgas`; `hex2cube` and `cube2hex` (the conversion
  between a hexagonal and a square grid) and `calcInvTaylor`; `S2FunGrid`, whose one use in
  MATLAB, the colour cache of `ipfColorKey`, is `colorGrid` in `plotting/orientationkeys.py`
  (a cube map looked up by its nearest node). `tools/graph/query.py gaps`
  leaves them out (`NOT_PORTED` there).

## 3. Phases

The phases are the Julia ones. Each is a usable milestone and each has a finished Julia
subplan to follow, so the order of work inside a phase is largely settled.

**P1 - geometry core.** `mtex.geometry`: vectors, axes, Miller indices, quaternions,
rotations, orientations, point groups from the Hermann-Mauguin symbol, reference frames,
plotting conventions, spherical and orientation regions, fundamental regions,
symmetry-reduced distances, grids on the sphere and on SO(3), crystal shapes, slip
systems. Follows `../mtexJulia/docs/plans/geometry.md`. Every type decision of this port
lives here; P1 is written by hand and reviewed before anything is built on it.

**P2 - function spaces.** `mtex.functions`: S1Fun, S2Fun, SO3Fun families (harmonic,
RBF, handle, Bingham, fibre), kernels, the Wigner-d tables and coefficient transforms,
NUFFT evaluation, density estimation, quadrature, extrema, sampling, Wasserstein. Follows
`functions.md`, whose sections 9 to 15 record the decided numerics.

**P3 - maps.** `mtex.maps`: EBSD container, phase lists, lattice and gridify, the local
spatial decomposition (no Voronoi library, Julia D10), `calcGrains`, boundaries, chains,
triple and quadruple points, grain shapes, smoothing, denoising, parent grain
reconstruction, GBND, dislocations. Follows `ebsd.md`. Largest module, and the one where
the compiled-kernel question (D5) is decided.

**P4 - pole figures and tensors.** The `PoleFigure` container in `mtex.functions`, the
tensor container in `mtex.tensors`, pf2odf, elasticity, mechanics, optics, Taylor and the
embedding in `mtex.analysis`. Follows
`polefigures-tensors.md`.

**P5 - plotting.** `mtex.plotting`: matplotlib backend, imported lazily. Spherical
projections, pole figures, inverse pole figures, ODF sections, colour keys, maps, grain
overlays, annotations, scale bars, the plotting convention applied to the axes. Follows
`plotting.md`.

**P6 - interfaces.** `mtex.io`: ang, ctf, cpr/crc, osc, HDF5 (h5ebsd, Dream3D, Bruker,
Oxford), pole figure formats, tensor and CIF files, export, the sample data catalogue.
Follows `interfaces.md`.

**P7 - documentation.** The 199 Literate pages become executable pages (Jupyter Book or
MyST with `jupyter-execute`), the datasets a downloadable catalogue.

P2 and P3 do not depend on each other; P4 depends on P2; P5 can start on P1 output;
P6 needs only the containers. Effort is well below the Julia effort because the design
work is done: P1 is weeks, P2 to P4 weeks to a few months each, P5 and P6 parallelisable.

## 4. Design decisions

Each decision names the options, the recommendation, and the experiment that should
settle it. Decisions the Julia port already took and that transfer unchanged are listed
first, briefly.

### 4.1 Taken over from MTEX.jl without change

- **Frames carry the point group; objects carry frames; `crystalSymmetry` is never a
  type** (MATLAB ADR 0008, Julia D1). `CrystalFrame`, `SpecimenFrame`, `NotIndexedFrame`
  under an abstract `ReferenceFrame`; a `Miller` is a vector in a crystal frame; an
  orientation names two frames.
- **Frames are interned handles** (Julia ADR 0004). Python objects with identity equality,
  a session register keyed on cell, alignment, group, name and convention, sealed after
  registration except name and colour, one session object with `reset()`. This maps
  onto Python without loss; it is how orix shares `Phase` objects too.
- **Rotations are proper; the improper element is a separate class** (ADR 0003).
  `Quaternion` (the algebra), `Rotation` (unit, `q ~ -q`), `OrthogonalTransform`
  (rotation times ±1, the point group element).
- **Antipodal vectors are a class**, `Axis`, and crystal directions carry no antipodal
  state (ADR 0002).
- **Grain exchange, index basis, unit length are part of the object, never a flag read
  off a name** (ADR 0006 to 0008). How each is spelled in Python is D2.
- **Point group elements come in closure order; nothing indexes them by position.**
- **Concatenation joins frames by the ADR 0008 rule 7 and raises otherwise** (ADR 0009).
- **Tangent vectors carry a side and a reference** (ADR 0010).
- **Kernels**: the non-normalising quaternion product, the chord angle, the
  `max_k |<r, ref * inv(s_k)>|` symmetry reduction, tolerances compared on the dot or
  the chord, the Wigner-d at π/2 coefficient transforms followed by a plain NUFFT (D9),
  the local lattice decomposition for grains (D10), the return-value rule of
  `return-values.md` (one result, accessors for what follows from it, a named tuple for
  coupled results, the Base idiom where one exists).
- **What is dropped**: `subSet`, `repmat`, the `times`/`mtimes` split (NumPy
  broadcasting is both), option-string parsing, `check_option`, `dynOption` (Julia D12a,
  D14). The `_outer` names stay, see D4.

### 4.2 To decide for Python

**D1 - The array model.** MATLAB and NumPy agree: one object, many entities. Every
geometric class is a container over NumPy arrays with a `shape`, `__len__`,
`__getitem__` returning the same class, `reshape`, `flatten`, `ndim`, and elementwise
methods that broadcast. A single element is the zero-dimensional container (Julia ADR
0011), never a separate scalar class. Storage: `Vector3d` holds one `(..., 3)` float array
(or three columns, see E1), `Rotation` one `(..., 4)` array, `Orientation` a `Rotation`
plus two frame handles. No per-element Python objects anywhere on a hot path.
Recommendation: `(..., k)` trailing-axis arrays, since every NumPy and SciPy routine takes
them and `einsum` handles the group reductions. Experiment E1 checks that against
struct-of-arrays columns for float32.

**D2 - Where Julia's type parameters go. DECIDED 2026-09-21 with the MATLAB syntax.**
Python has classes, fields and `dtype`, not type parameters, and the MATLAB syntax
already names the fields:

| Julia parameter | Python |
| --- | --- |
| precision `T` | `dtype` of the array, float64 default, float32 allowed |
| polarity (`Axis`) | the field `v.antipodal`; the Julia semantics (comparison antipodal when either side is, arithmetic clears it, `Miller` stores none and uses the Laue sibling) are kept |
| unit length (`UnitVec3`) | the field `v.isNormalized`, which skips normalisation in `theta`, `rho`, `angle` and the projections |
| index basis (hkl vs uvw) | the field `v.dispStyle`; storage stays Cartesian |
| frame kind | the class of the frame handle; behaviour that differs by kind is a method of the frame, not an `isinstance` at the call site |
| properness | `rotation` versus `orthogonalTransform`, two classes; `rotation` has no `i` |
| grain exchange `X` | the field `ori.antipodal`, allowed only when `frameA is frameB` |
| storage `A` | the dtype and layout of the array, not visible in the class |

The rule behind the table: a distinction that changes *which algorithm runs* is a class;
one that changes *a parameter of the same algorithm* is a field. Julia made the flags
types to protect inference, which Python does not have; the MATLAB field names are kept
and the Julia semantics behind them are kept. Detail in `docs/plans/geometry.md` 3.

**D3 - Dispatch.** No multiple dispatch. Binary operations are methods on the left
operand plus `__radd__`-style reflection; anything that dispatches on both frame kinds
lives on the frame (the join rule, `isAligned`, `sim`). Constructors that accept many
forms (`Miller`, `crystalFrame`, `specimenFrame`, `orientation`) sort their positional
arguments by type once, as MATLAB's `getClass` does, and the `by*` class methods take one
form each. Never dispatch on a class name string or an integer id.

**D4 - Naming. DECIDED 2026-09-21.** MATLAB names throughout, in MATLAB casing:
classes `vector3d`, `rotation`, `orientation`, `crystalFrame`, `S2Grid`; verbs
`calcGrains`, `project2FundamentalRegion`, `symmetrise`; static constructors
`rotation.byEuler`, `orientation.byMiller`; properties `ori.CS`, `m.hkl`, `rot.angle`.
Verbs exist as methods and as module functions of the same name, so `angle(o1, o2)` and
`o1.angle(o2)` are one call. Boolean flags become keyword arguments with their MTEX
names (`noSymmetry=True`); a positional string that names a convention, a basis, a group
or an alignment stays positional (`'uvw'`, `'ZYZ'`, `'X||a'`). The `_outer` names stay as
one-line conveniences over broadcasting. Angles in radians with the `degree` constant;
no unit library on the hot path. A snake_case alias layer can be added later at no cost.

**D5 - Compiled kernels.** The known hot spots are the symmetry reduction, `calcGrains`
on 1e6 to 1e7 pixels (decomposition, union-find, chain ordering), the Wigner-d recursions
and coefficient transforms, boundary smoothing, and `unique` on orientations. Options:

- (a) NumPy and SciPy only: `einsum` for group reductions, `scipy.sparse.csgraph` for
  segmentation, `scipy.ndimage` for lattice operations, `scipy.fft`.
- (b) (a) plus numba for the loops that will not vectorise (chain ordering, clipping).
- (c) (a) plus a small compiled extension (nanobind or Rust via maturin) for the same
  loops.

Recommendation: (a) throughout P1 and P2, and decide between (b) and (c) in E2 at the
start of P3 on the chain ordering and the decomposition. Whatever is chosen sits in one
`mtex/_kernels/` package with a pure-NumPy reference implementation of every kernel, so
the tests run without the accelerator and the accelerator is replaceable. numba's
support for the current CPython (3.14 on this machine) is one of the things E2 checks.

Decided 2026-09-21 for the geometry: (b). `docs/research/benchmark-e2-kernels.md` measured
numba and C at the same speed, half of Julia's time on the product and the image search,
and cache-blocked NumPy at 1.6 to 2x Julia. The kernels live next to their consumers,
`mtex/geometry/kernels.py` for the products, the searches of the symmetry reduction and
the projections, `mtex/geometry/grids/gridsearch.py` for the S2 ring-grid nearest search, and
`mtex/geometry/grids/orientationtree.py`, a quaternion k-d tree, for the SO(3) grid
nearest/k-nearest/radius search (the radius search ODF reconstruction needs); the pointwise
and S2 kernels each with a NumPy form as fallback and oracle; numba is the optional extra
`mtex[numba]`, imported on the first kernel call, `setMTEXpref('kernels', 'numpy')` turns
it off. numba 0.67 runs on 3.14. The grain kernels of P3 decide (b) against (c) for
themselves in E2. Decided 2026-09-21 for the maps: (b) again, `docs/research/benchmark-decomposition.md`.

**D6 - Fast transforms.** Julia D9 stands: own Wigner-d tables and coefficient transforms,
then a plain NUFFT. The NUFFT candidates are `finufft` (maintained, wheels on every
platform, threaded, types 1 to 3 in 1 to 3 dimensions) and `pynfft` (bindings of libnfft3,
unmaintained). Recommendation: finufft, with the equispaced route through `scipy.fft` on
quadrature grids as Julia section 13 built it. The coefficient transforms are matrix
products per degree and stay NumPy. E3 reproduces the Julia numbers (1e6 orientations at
L = 64 in 0.35 s, 1e6 directions at L = 256 in 0.05 s) as the budget.

**D7 - Relation to orix.** orix (pyxem) has `Quaternion`, `Rotation`, `Orientation`,
`Misorientation`, `Symmetry`, `Phase`, `CrystalMap`, fundamental regions, IPF colour keys
and readers for ang, ctf and h5ebsd. Options: (a) build on it; (b) independent, orix as a
test oracle and a conversion adapter. Recommendation: (b). orix attaches the point group
to the orientation, not to a frame, which is the model ADR 0008 replaces; it canonicalises
quaternion signs; and its containers are not the function-space or frame model. The
reasoning is the Julia ADR 0001 argument again. An optional `mtex.compat.orix` with
`from_orix`/`to_orix` keeps the two ecosystems interoperable. The same holds for
`scipy.spatial.transform.Rotation`: a conversion, not a foundation.

**D8 - Plotting backend.** matplotlib, in `mtex.plotting`, imported lazily so that
`import mtex` never imports matplotlib. `mtexFigure`'s multi-axis layout, colour bars and
annotations become a thin layout class over `matplotlib.figure`. 3D (crystal shapes,
grain3d, ODF in 3D) through matplotlib's 3D axes first, pyvista as an optional extra if E4
of the plotting subplan finds matplotlib too slow. Alternative: plotly for notebooks. Not
before P5.

**D9 - Package layout.** One distribution `mtex`, one import, five subpackages mirroring
the Julia libraries and their dependency order:

```
mtex/
  geometry/     P1
  functions/    P2, P4 (polefigures/, tensors/ inside)
  maps/         P3
  io/           P6
  plotting/     P5, lazy
  _kernels/     D5, reference NumPy implementation plus optional accelerator
  _session.py   the frame register and the session defaults
```

`mtex/__init__.py` re-exports the user layer. A subpackage imports only from those below
it; a test checks it. File names follow the Julia ones where the concept is the same
(`symreduce.py`, `wignerd.py`, `decomposition.py`), so the two ports can be read side by
side.

**D10 - Toolchain.** `pyproject.toml`, `uv` for environments and locking, `pytest`,
`ruff` for formatting and linting, `mypy` or `pyright` in strict mode on the public
layer, NumPy 2, SciPy, Python >= 3.11 (or whichever floor D5's accelerator forces).
Type hints on every public signature; `numpy.typing.NDArray` for arrays. Optional extras:
`mtex[plot]`, `mtex[io]` (h5py), `mtex[fast]` (the accelerator), `mtex[orix]`.

**D11 - Global state.** The frame register and the session defaults (default specimen
frame, default plotting convention) are one `Session` object in `_session.py` with
`reset()`, as the Julia port has. Thread safety through a lock; no process-level sharing.

**D12 - Mutability.** Every operation returns a new object, slicing returns a view
where NumPy would, and the property writes MATLAB scripts make (`v.antipodal = True`,
`ori.CS = cs`, `cs.color = 'red'`, `ebsd.CS = ...`) are allowed and mutate in place, as
NumPy and orix do. `copy()` exists. The pages say once that `ori2 = ori` shares, unlike
MATLAB. The in-place map operations the Julia port marks with `!` are the MATLAB
methods with the MATLAB signatures. **Discussion.**

**D13 - Testing against MATLAB and Julia.** Every Julia `test/*.py` counterpart holds the
assertions of the MATLAB `check_*.m` it mirrors plus the Julia ones. Numbers are pinned
against Julia by running MTEX.jl (`julia --project=../mtexJulia/dev`) where a table or a
convention is the subject; MATLAB through the bridge in `../master/docs/agents/` only after
asking, never the test tiers.

## 5. Experiments

Throwaway scripts under `bench/`, reported in the subplan that uses them.

**E1 - storage layout.** 1e6 and 1e7 orientations: `(n, 4)` float64, `(n, 4)` float32,
four separate columns. Measure the symmetry-reduced `angle` against one reference under
cubic symmetry, `symmetrise`, `unique`, `mean`, slicing, concatenation, memory. Decides
D1 and whether a `Direction` subclass is worth its constructor cost (D2). Budget: Julia
does the reduction at 46 ns per element single-threaded. Run 2026-09-21
(`docs/research/benchmark-geometry.md`): the layout stays; the reduction over one crystal
frame is 5x Julia and 2 to 5x faster than MATLAB, the products 10x Julia and 7x slower than
MATLAB (memory traffic of the temporaries), the misorientation search 8 to 10x Julia because
it forms the (n, 288) dot matrix in one piece. Candidates: a blocked search in `symreduce`,
a preallocated column kernel for the products.

**E2 - compiled kernels.** The chain ordering and the local decomposition on a 1e6 and a
4e6 pixel map, as pure NumPy, as numba, as a nanobind or Rust extension. Measure wall
time, wheel size, build complexity, CPython version support. Decides D5. Budget: Julia
does 1e6 sites in 0.93 s on one thread. Run 2026-09-21 on the geometry kernels instead
(`docs/research/benchmark-e2-kernels.md`): numba and C both at 0.5x Julia, numba taken.

**E3 - harmonics.** finufft against pynfft against the equispaced `scipy.fft` route for
the SO(3) and the S2 evaluation and adjoint at L = 64 and L = 256 with 1e6 nodes.
Decides D6 against the Julia budget.

**E4 - import and first-call time.** `import mtex`, first `Orientation`, first
`calcGrains`, with and without the accelerator loaded. Not a decision, a budget: NumPy
and SciPy alone import in about 0.3 s, and the plotting layer must add nothing to that.

**E5 - float32.** Julia found float32 pays only in struct-of-arrays layout. Repeat on the
symmetry reduction and the ODF evaluation; decides how far the `dtype` freedom of D2 is
pushed into the kernels.

## 6. Dependencies, candidates

Core: numpy, scipy. Transforms: finufft. IO: h5py, and the standard library for text.
Sample data: pooch. Plotting: matplotlib; pyvista optional. Accelerator: numba or a
nanobind/maturin extension, per E2. Oracles in the tests only: orix,
scipy.spatial.transform, quaternion (moble). Docs: Jupyter Book or Sphinx with MyST and
sphinx-gallery; the Literate pages of MTEX.jl are the source.

## 7. Tests

- One `tests/test_<file>.py` per source file, with the assertions of the MATLAB `check_*`
  and of the Julia test of the same name.
- The doc pages execute in CI, so they cannot rot.
- The frame register is reset in a fixture before every test.
- A structural test that no subpackage imports from above it and that every public name
  has exactly one owner, as `test/packages.jl` does.

## 8. Layout, decided 2026-09-22

Five slim cores and one layer of algorithms above them, cut by the direction of the
imports and not by difficulty: a core holds representations, transforms, evaluation and
arithmetic, and imports only cores below it; whatever iterates or optimises sits in
`mtex.analysis` and may import every core.

- `mtex.geometry`: vectors, rotations, point groups, frames, Miller indices, regions, grids,
  the symmetry reduction, tangent vectors, fibres, planes, slip systems, crystal shapes,
  the parent-to-child relationship.
- `mtex.functions`: the S1, S2 and SO(3) functions with their transforms, kernels,
  quadrature, density estimation, characteristics, vector and axis fields, and the
  `PoleFigure` container.
- `mtex.tensors`: the tensor container over the geometry, its algebra and the physical
  tensor classes.
- `mtex.maps`: the EBSD container, lattice, decomposition, segmentation, grains,
  boundaries, shapes and the intragranular quantities in two dimensions; `mtex.maps3d`
  the same in three when it comes.
- `mtex.analysis`: the pole figure inversion, optimal sampling, clustering, the
  embedding, the elastic and optical functions, the Taylor model, texture evolution,
  variational denoising, the dislocation fit, and the parent grain reconstruction when
  written.
- `mtex.io` and `mtex.plotting` import from everywhere.

The user sees one flat namespace, `from mtex import *`, assembled in this order, so the
move of a file changes only the module lists of the `__init__` files. Two upward imports
remain inside methods, `vector3d.calcDensity` into the functions and the tensor methods
`velocity`, `SchmidFactor`, `calcTensor` into the analysis; the plotting layer shows the
alternative, installing the methods from above at import. The time critical algorithms
and what is left to do for them are in `docs/research/time-critical.md`.

## 9. Subplans

One document per piece, written after the decisions above are taken, each starting from
the Julia subplan of the same name and stating: what transfers unchanged, what is
respelled for Python (D2, D3), what the Julia subplan left out and is still left out, the
tests, and the experiments it depends on.

1. `docs/plans/geometry.md` - P1. WRITTEN 2026-09-21.
2. `docs/plans/functions.md` - P2 and P4. WRITTEN 2026-09-21 (P2; P4 pole figures and tensors still to add).
3. `docs/plans/maps.md` - P3. WRITTEN 2026-09-21.
4. `docs/plans/plotting.md` - P5.
5. `docs/plans/io.md` - P6.
6. `docs/plans/docs.md` - P7. WRITTEN 2026-09-22; the page table in `docs/plans/remaining-pages.md`.

Subplans 2 to 5 can be executed in parallel once the geometry layer is stable, each
against the geometry API, each by a separate agent.

## 10. Risks

- **Python loops.** The Julia port leans on cheap scalar loops in the maps layer
  (decomposition, chains, union-find). If E2 finds that neither NumPy nor numba reaches
  the budget, P3 needs a compiled extension and the wheel build that comes with it.
- **The two sources disagree.** Where MTEX.jl deliberately changed a MATLAB result (its
  subplans' "Dropped from MATLAB" and "Left out" sections), the Python port follows Julia,
  and the disagreement is recorded once in the subplan, not rediscovered per function.
- **A third implementation drifts.** The mitigation is the shared test set and the shared
  pages: a change in one port that is not mirrored in the tests of the others is visible.
- **Type-hint completeness versus the array model.** Containers whose `__getitem__`
  returns their own class and whose arrays have free `dtype` are awkward to type strictly;
  the public layer is typed, the kernels are not, and this is accepted.
