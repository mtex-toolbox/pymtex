# A more Pythonic pymtex: the plan

The laptop session's read-only review of 2026-10-11 (all of `mtex/`, ten reviewers; the summary in
`~/mtex/mailbox/to-server.md`, the full reports in `~/mtex/mailbox/review/` once sent) found where
the Python reads like Julia or MATLAB. The user decided the scope and the open questions on
2026-10-11; bayes executes the whole plan without further questions, commits locally phase by
phase, and pushes once at the end.

## Decisions

- **Public indices count from zero.** `nextAxis`, `ind2color`, the boundary chain ids (`chainId`,
  `entry`, `kOf`), `loadODF`'s column numbers and every other public index that is one-based as in
  MATLAB. The docstring names MATLAB's one-based meaning where a MATLAB user would look for it.
  Labels stay as they are: grain ids (0 is no grain), variant and packet numbers in MTEX's
  numbering, the phase numbers a vendor file writes. One exception kept: the tile centres of
  `xcfROIGrid` follow MATLAB's pixel arithmetic, so that the `trueEbsd` registration stays
  MATLAB's tile for tile, as it was checked.
- **Keywords only for flags.** A string that duplicates a keyword (`'antipodal'` beside
  `antipodal=True`, `'complete'`, `'smooth'`, ...) goes; a positional string stays only where it
  names something (`'uvw'`, `'ZYZ'`, `'X||a'`, `'Reuss'`, a phase name), as CLAUDE.md says.
- **Tolerant equality is `isclose(other, tol=)`.** `==` is exact and takes no `tol`; every
  `__eq__(self, other, tol=)` goes, and `vector3d.isclose`, which called `__eq__` with a `tol` it
  does not take, is repaired by it.
- **Public API changes are allowed** beyond these three; each is entered in `docs/plans/syntax.md`
  with MATLAB's name where one exists, and no alias is kept.
- **Speed is never traded for readability**: a hot path is measured before and after a change,
  and a change that costs time is not made.

## Phases

Every phase ends with the full suite, `tools/staticapi.py` and a local commit; a phase that
changes what users call also updates the doc pages that call it.

1. **Bugs**, each with a test that fails before the fix:
   `vector3d.isclose(tol=)` (through the equality decision), `isCompatible` used as a boolean in
   `plotting/poles.py`, the negated `verbose` in `parent2child.py`, the method check of
   `maps/traces.py` outside the cluster loop, the dead `x if False else y` branches and the
   duplicated `__rmul__`; then the reported ones, checked first: the indexing of `grainsPrior` in
   `reconstruction.py`, the `_harmonicCache` of `SO3FunRBF` never invalidated when the weights
   or the centres change.
2. **The three API decisions**: `isclose` only; zero-based indices; keywords only. A search of
   every positional string the functions accept (the `*args` scans of pattern 8) lists the flags
   to remove; the tests and the doc pages move with them.
3. **Speed items**, each measured before and kept only if it gains: the direction key's per-node
   `azimuthAbout2`, `calcLankford`'s LP per orientation, `id2ind`/`selectByGrainId`'s dict over all
   ids (and the `maps3d` and `reconstruction` copies), `triplePoints` and `chainV`, `merge`'s list
   comprehensions, `findByLocation`'s scan per point, `symmetry.lattice` recomputed per access,
   `_clenshawCurtisWeights`, `_euler` for a dtype, `characteristics.py:797`, the median per group
   of `spatialtransform.py`, `vector3d`'s `_isFrame` on every argument. Hot paths named by the
   review (reconstruction's cluster masks, `variantgraph`, `mapplot`'s polygon rows,
   `_array._new`, `orientationtree.knn`) are measured and left unless a change gains.
4. **Shared helpers** (pattern 1): `qconj` and the identity beside `qmul` in `geometry/kernels.py`
   replace the twelve `_conj` and the inline `[1, -1, -1, -1]`; `asVector3d(v)` and `xyz(v)` in
   `vector3d.py` replace the coercions and `.data.reshape(-1, 3)`; one `_unit` per meaning; one
   `_complexType`; the two `_readFrames` and `_leadingAxis` of different behaviour renamed to say
   what they do.
5. **Types instead of `hasattr`** (pattern 3): `SO3Fun` becomes a real base class of every SO(3)
   function (and `S2Fun` likewise, with an `S2FunGeneric` beside `SO3FunGeneric`), so the four
   `_isSO3Fun` become one `isinstance`; `hasattr(x, "data")` and `type(x).__name__` tests go.
6. **Dispatch** (pattern 2): the long `isinstance` ladders of `sphericalplot.plot`/`annotate`, the
   funops verbs, the exporters and `traces.py` become `functools.singledispatch` or a method on
   the type where the type is ours; the long constructors split their argument reading into
   named helpers (pattern 8: one argument reader for the `*args` scans, which rejects leftovers).
7. **Records** (pattern 4): `NamedTuple` or `dataclass` for `scanGrid`, `_spaceFlags`, the flat
   arrays of `gridsearch.py`, `parent2child`'s fit dict and the wizard's rows; the plot state set
   in `__init__` instead of probed with `getattr`.
8. **Copy and replace** (pattern 5): one `_with(**changes)` per container on the model of
   `grainBoundary._with`, used for `EBSD`, `grain2d` and the `maps3d` classes; no
   `object.__setattr__` from outside a class.
9. **Caches** (pattern 6): `functools.cached_property` where an attribute is computed once;
   module caches keyed by `id(...)` keyed by the object or weakly; the deliberate caches
   (`_TRIG_CACHE`, `_HALF_PI_TABLES`, `_PROJECTIONS`, `_GRIDS`) stay.
10. **Twins** (pattern 7): the arithmetic of the S1/S2/SO(3) harmonic classes from one mixin;
    `max_`/`min_` one function; the S2/SO(3) discrepancy functions sharing their skeleton; one
    table of EBSD formats by extension; one "which axes" helper in plotting; the kernel dispatch
    of the maps layer through one registry.
11. **The HDF5 reader** (pattern 9): one reader object carrying the file, the paths, the config
    and the scope instead of threading them through every helper.
12. **Small things** (pattern 10): assigned lambdas to `def`, nested conditional expressions
    unfolded, `__all__` at the end, imports at the top where they cost nothing at import, `pathlib`
    in the wizard, io and settings, keywords that shadow builtins renamed (`max=`, `all=`,
    `range=`, `iter`), `halfwidth.py`'s `space == "S2"` by two classes.

Left as the review names them as models: the compiled loop beside its NumPy oracle in every
`kernels.py`, `_array.py`, `representations.py`, `latticetype.py`, `kernelfunctions.py`,
`lsqr.py`, `criteria.py`, `keywords.py`, the `FORMATS` table of `loadpolefigure.py`, the
projections of `spherical.py`, the cost constants of `transforms.py`.

## After the plan

The whole documentation is built (`docs/make.py`, every page) and every page checked against its
MATLAB reference (`docs/matlab/<Page>/`, `docs/plans/remaining-pages.md`, `docs/findings.md`):
the refactor must not have changed what a page shows.

## Done

- Phase 2, the decisions: `==` exact and `isclose(tol=)` (7af70b9), zero-based public indices
  (9911d0b), flags as keywords with `refuseFlags` (4b51309).
- Phase 1, the bugs, each with a test that failed before (5736f94).
- Phase 3, the speed items, each measured, equal to what it replaces: `calcLankford` over
  orientations 42 s to 0.55 s (50 orientations), `id2ind` 72 to 0.6 ms, `triplePoints` 159 to 26 ms,
  `chainV` 25 to 3.5 ms, `merge` 39 to 23 ms, the S2 sample's symmetry 2.4 s to 96 ms (1e5),
  the k nearest on a grid 306 to 96 ms (1e4), `symmetry.lattice` cached, the scan line medians
  by group, the direction key's azimuths in one call, the pole figure's Clenshaw-Curtis weights
  from `quadrature.clenshawCurtis`. Measured and left: `findByLocation` (a tree costs 300 ms to
  build on 1e6 positions against 10 ms per scanned query), `vector3d`'s `_isFrame` (a fraction of
  2.4 us), the plot of grains (26 ms of its own), the reconstruction loops (checked against MATLAB).
- Phase 4, the shared helpers: `qconj` in `geometry/kernels.py` replaces the twelve `_conj` and the
  inline `[1, -1, -1, -1]`; `asVector3d` in `vector3d.py` the 85 coercions; `unitRows` in
  `representations.py` the nine row normalisations (a zero row stays zero); one `_complexType` in
  `functions/kernels.py`; the lenient frame reader of `rbf.py` is `_framesAmong`, and the two
  `_leadingAxis` are NumPy's `normalize_axis_index`. Left: a helper for `.data.reshape(-1, 3)`,
  which is plain NumPy and reads as well; `xyz` stays MATLAB's (..., 3) property.
- Phase 5, types: `SO3Fun` and `S2Fun` are the base classes of every representation, so the
  `_isSO3Fun`/`_isS2Fun` probes are `isinstance`; the pole figure, tensor, field and planar key
  probes of the plots and exporters likewise, and a frame is tested as `referenceFrame` (the probe
  `hasattr(x, "sym")` took a tensor, whose `sym` is its symmetric part, for a frame). Left:
  `vector3d._isFrame` and `_isConvention`, which cannot import the classes (frames import
  vector3d); the map probes of `texture.py` and `orientationkeys.py`, which mean any container
  with phases (EBSD, grains, phaseList).
- Phase 6, dispatch: `funops.rotate` folded into the methods it served, the public `rotate` the one
  dispatcher; the space plots take `complete=` and `region=` as keywords (the strings 'complete'
  and 'noBoundary' were flags phase 2 missed); `export` tests its kinds by class. Left: the ladder
  of `plot`, which is ordered by the state of the figure as much as by type, and `conv`, which
  dispatches on both arguments.
- Phase 7, records: `scanGrid` (`io/load.py`), `flatSO3Grid` (`gridsearch.py`), parent2child's
  `_Misfit`; `mtexFigure.sections` set in `__init__`, the axes drawn into last one method,
  `_axesDrawnInto`, beside the `gca` property.
- Phase 8: no `object.__setattr__` outside a class; the rebuilds of a map change most fields at
  once, so no `_with` was added.
- Phase 9: the search tables of `SO3Grid` are a `cached_property`; the colour grid cache is keyed by
  the frame itself, which hashes by identity.
- Phase 10, twins: `harmonicArithmetic` (`functions/harmonicarithmetic.py`) holds the operators the
  three harmonic classes shared; `max_`/`min_` one `_extremum`; the two discrepancies share
  `_difference`; `VENDOR_FORMATS` in `io/load.py` is the one table of the vendor formats, which
  `inspectEBSD` reads too. Left: the maps kernels, which compile through the two registries
  `pixelKernel` and `grainKernel` already.
- Phase 11: `h5Reader` (`io/h5.py`) carries the file, its paths, the config, the data set and the
  scope; `_loadEBSD_h5` and `inspectEBSD` read through it, which removed `inspect.py`'s copies of
  the property scan and of the correction rule. Checked equal to the previous reader on nine
  vendor files (Oxford, EDAX, EMSphInx, Bruker), every field of the map and of `meta`.
- Phase 12: the 71 assigned lambdas are `def`s, six dead assignments gone, `__all__` last in every
  module but the packages, `degreeEnergies` an S2 and an SO(3) class (bit for bit the previous
  results). Left: keywords named like builtins (`all=`, `max=`, `iter=`, `range=`, `id=`), which
  are MATLAB's option names and shadow nothing a body calls; `pathlib` for `os.path`, a change of
  no gain.

## Second pass (2026-10-12)

Six read-only reviews by layer, then each layer's findings applied and checked: every change on a
hot path timed before and after and kept only if not slower, results equal (or to rounding where
the order of a sum changed), figures pixel-identical on 29 render cases and 16 rebuilt pages, the
readers and exporters equal or byte-identical on every fixture and sample map. Shared helpers
replace retyped idioms (cluster dedup, unique bookkeeping, degree index vectors, id position
tables, pair keys, gathers, run starts, table starts, header parsers, plot holders' attributes,
colour limits); Python loops over rows, chains, grains or degrees became array operations where
that was faster. Measured: `ori * rot` 278 to 4.3 ms (1e4), `round2Miller`'s rounding 40 to 0.6,
grain `caliper` 1569 to 58, `diameter` 554 to 36, `smoothBoundary` 364 to 195, reconstruction
boundary weights 1836 to 130, `S2FunHarmonic.truncate` 469 to 2.5 (L = 512), Bingham's symmetric
axes 23 to 0.19, the component seeds 98 to 53. Reverted where slower or not bit-safe: the halved 3D
KAM (3e-11 rad), the SO(3) `power` by reduceat, the kernel handles' outer products, the table
formatting mask, the shared line generator, `anchorClosedChains`.

Bugs found and fixed, each with a test that failed before: `S2FunTri` with complex values, the int32
pair key of the reconstruction above 46341 grains, 3D `calcGrains` ignoring the kernel preference,
the 3D `meanOrientation` setter's IndexError, `closestAlignment`'s cached None, the export of a
number beyond int64 (an endless loop in the table writer). The analysis layer was not done: the
permission check refused the applier's edits there; tensors and the two kernels glue files were.
