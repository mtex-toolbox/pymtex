# Subplan - the Python syntax where MATLAB's is a workaround

Status: written 2026-09-30 on the branch `syntax-review`, with the user's decisions of the
same day; R0, R1 and R2 built. Choices made on the way: MATLAB's `hist` is
`histogram(grains, grouped=True)`; `eqTol` of frames is `isclose(frA, frB)`; the flat `byMiller` went
with the standard orientations (`orientation.byMiller`); `antipodal` is one tri-state (None
from the objects, True forces, False forbids) where MATLAB has `antipodal` and `noAntipodal`;
`noLabel` is `axisLabels=False`, since `label` names the text of the points; `noP2C` is
`useP2C=False`; `smoothBoundary(simplify=False, refine=False)` and `calcODFIterative(thinning=0)`
absorb their `no*` flags; `plotSection(..., field=False)` replaces `noGrid`; `polar=False` the
`notPolar` of the circle plots; `verbose` keeps MATLAB's default where MATLAB reports unasked.
The plot keywords are checked in `mtex/plotting/keywords.py`, a variant raising TypeError.
`m:@rotation.alpha` left the graph's counterparts: it was a false link, from a docstring citing
MATLAB's alpha of the grain maps. Until now the rule was "on a name the user sees, MATLAB wins" (`CLAUDE.md`). This
plan replaces it for the names below: where MATLAB's name is a misnomer, a duplicate, a MATLAB
workaround or a typo, the Python name is the better one, and MATLAB's name goes. Every other
name stays MATLAB's.

## 1. Rules

- **Hard rename, before 1.0.0.** The old name is removed, with no alias and no deprecation
  period. 1.0.0 is not tagged yet (the CHANGELOG says unreleased), so the renames go into the
  first release and break no user. The CHANGELOG's 1.0.0 notes say where the names depart
  from MATLAB. The page `MATLABtoPython` lists every rename in its table "Names that changed",
  filled step by step, and `PythonDevelopers` states the rule.
- **MATLAB's name stays findable.** A renamed public definition cites its MATLAB name in its
  docstring in the form the graph reads, `MATLAB's plotPDF` without backticks
  (`MATLAB_REF` in `tools/graph/scan_python.py`), so the counterpart edge survives as
  `resolution="cited"` where the name match is lost.
- **Keep the graph current.** After every step: `.venv/bin/python tools/graph/build.py`, then
  `tools/graph/query.py gaps` compared with its output before the step. A MATLAB node that
  had a Python counterpart before a step must have one after it. The `bench/*_ops.py` ops
  carry the new names, and the crosscheck runs of the touched ops stay green.
- **Every place the name appears changes in the same step.** That covers `mtex/`, `tests/`,
  `docs/pages/`, `bench/`, `docs/findings.md`, the wizard's script writer
  (`io/importscript.py`) and `openDoc`. Afterwards run `tools/staticapi.py`, then the tests of
  the touched files (never the full suite unless asked), and rebuild the touched doc pages.
- `CLAUDE.md` "Spellings decided" gets the new rules in R0: the keywords (section 4 below),
  `frameA`/`frameB`, and "a Python name may differ from MATLAB's where this plan says so".
- A doc page shows the Python name. The MATLAB | Python switch of the site makes the
  difference visible; `docs/plans/release.md`'s hand-off doc gets the rename table.

## 2. Decided renames

### 2.1 Pole figures (R3)

| old | new |
|---|---|
| `plotPDF(ori \| odf \| pf, h)` | `plotPF` |
| `plotIPDF(ori \| odf, r)` | `plotIPF` |
| `plotPoleFigure(pf)` | removed, `plotPF(pf)` / `plot(pf)` |
| `odf.pdf(h)`, `odf.calcPDF(h)` | removed; `odf.radon(h)` gives the S2 function, `calcPoleFigure(odf, h, r)` the `PoleFigure` |
| `odf.ipdf(r)` | removed, `odf.radon(r=r)` |

`calcPDF`'s `superposition=` moves to `radon`. Sizes: `plotPDF` occurs 154 times in the pages,
`plotIPDF` 42 times.

### 2.2 Duplicates (R2)

| old | new |
|---|---|
| `gB.reduceBoundary`, `gB.simplifyBoundary`, `gB.refineBoundary` | removed; `gB.reduce`, `gB.simplify`, `gB.refine` |
| `grains.reduceBoundary`, ... (grain2d, grain3d) | unchanged: on grains the word says what is reduced |
| `grains.hist(prop, nbins)` | removed; `histogram(grains, prop, bins=)` |
| `quaternion.eye`, `orientation.eye`, `rotation.eye` | removed; `id` (`tensor.eye` stays, a matrix) |
| `eq`, `eqTol` | removed; `isclose(a, b, tol=)` |
| module functions `copper(cs)`, `cube`, `goss`, `SR2`, ..., `Bain`, `KurdjumovSachs`, ... | removed from the flat namespace; `orientation.copper(cs)` and so on |

The numbered orientations (`SR2` to `SR4`, `QLage2` to `QLage4`, `PLage2`, `brass2`, `copper2`)
keep their names, which are the literature's.

### 2.3 Projections (R4)

| old | new |
|---|---|
| `project2FundamentalRegion` | `projectIntoFundamentalRegion` |
| `project2EulerFR` | `projectIntoEulerRegion` |
| `projectIntoRegion` | unchanged |
| `sR.restrict2Upper`, `restrict2Lower` | `restrictToUpper`, `restrictToLower` |

### 2.4 Frames (R6)

`orientation` already stores `frameA` (from, the crystal) and `frameB` (to, the specimen, or the
second crystal of a misorientation), with `CS` and `SS` as properties over them (Julia ADR 0001).
Every object on SO(3) gets the same pair: the SO(3) functions (`SRight` -> `frameA`, `SLeft` ->
`frameB`), `orientationRegion` (`CS1`/`CS2` -> `frameA`/`frameB`), `SO3TangentVector`, the SO(3)
vector fields, `fibre`, the SO(3) grids, `embedding` and `MLSSolver`. `CS` and `SS` stay on all
of them as convenience properties; `SRight`, `SLeft`, `CS1` and `CS2` are removed. First check
that `frameA = CS = SRight` holds in every class; `mori = inv(o1) * o2` gives
`frameA = o2.frameA` and `frameB = o1.frameA`.

### 2.5 Defects (R1)

| old | new |
|---|---|
| `plotSpektra` | `plotSpectrum` |
| `S2Triangulation.calcBario` | `barycentric` |
| `parentGrainReconstructor.numChilds` | `numChildren` |
| `noisepf(pf, fak=)` | `noisepf(pf, factor=)` |
| `SO3Grid.nnodes`, `S2Grid.nrings`, `boundaryNetwork.nNeighbors` | `numNodes`, `numRings`, `numNeighbors` |
| `angle_outer(a, b)`, `dot_outer(a, b)` | `angle(a, b, outer=True)`, `dot(a, b, outer=True)` |
| `simulateEBSD.addnoise` | `addNoise` |
| `addFeature_singleStep`, `_circularSubgrain`, `_simpleGradient` | `addFeature('singleStep')`, ... |
| `crystalShape.fluorite_complex` | `fluoriteComplex` |
| `textureindex` | `textureIndex` |

`runtimeCounterparts` in `tools/graph/build.py` maps MATLAB's `_outer` methods to their base
name. Once `angle_outer` and `dot_outer` are gone, that mapping is the only link; check it still
resolves.

### 2.6 Workarounds (R5)

- **Per-format loaders private.** `loadEBSD_ang`, `_ctf`, `_crc`, `_osc`, `_h5`, `_generic`,
  `_dream3d`, `_xnovo`, `exportEBSD_*`, `loadODF_*`, `exportODF_*`, `export_VPSC`,
  `loadPoleFigure_generic` and `loadGrains_Dream3d` become `_`-prefixed. Only `loadEBSD(f,
  interface=)`, `loadEBSD3`, `loadODF`, `loadPoleFigure`, `loadGrains3d`, `export(x, f,
  interface=)` and the `load` class methods stay public. Their keywords (`setting=`, `EDAX=`,
  ...) pass through `**kwargs` and are documented on the public loader. Private definitions
  drop out of the graph's name match, so their docstrings cite the MATLAB `interfaces/` file
  name (`MATLAB's loadEBSD_ang`).
- **`plotx2east` family removed.** `plotx2east`, `plotx2north`, `plotx2south`, `plotx2west`,
  `plotzIntoPlane` and `plotzOutOfPlane` go; `plottingConvention.default(...)` is the one way
  (the pages: 3 uses, the tests: 10).
- **3D plots by dispatch.** `plot`, `quiver` and `scatter` dispatch on `EBSD3`, `grain3d`,
  `grain3Boundary` and the ellipsoids. `plotEBSD3`, `plotGrains3d`, `plotBoundary3d`,
  `plot3dMap`, `quiver3dMap`, `quiverVectors`, `plotPoints3d`, `plotEllipsoid`, `scatter3` and
  `quiver3` stop being public; `plot3d`, `scatter3d` and `arrow3d` (drawing into 3D space)
  stay. `quiver3(t)` on tangent vectors becomes `quiver(t)` if nothing else claims that input.

## 3. Keywords (R1)

| theme | old | new |
|---|---|---|
| tolerance | `tolerance=` (70 in mtex) | `tol=` |
| iterations | `maxit=`, `iterMax=`, `itermax=` | `maxIter=` (`iterMin=` -> `minIter=`) |
| progress | `silent=` | `verbose=` (default False), with the sense inverted |
| plotting | `MarkerSize`, `DisplayName`, `EdgeAlpha`, `Length`, `linewidth` | `markerSize`, `displayName`, `edgeAlpha`, `length`, `lineWidth` |
| symmetry | `cs=`, `ss=` as the plain symmetry keyword | `CS=`, `SS=`; role names (`csParent=`, `csChild=`) stay |
| flags | every `no*` keyword: `noSymmetry` (46 in mtex), `noAntipodal` (20), `noLabel`, `noIntern`, `noRobust`, `noOrientFaces`, `noGrid`, `noGhostCorrection`, `noTitle`, `noSurface`, `noScaling`, `noSimplify`, `noRefine`, `noCenter`, `noPseudoDiagonal`, `noP2C`, `noDiagonal`, `noBoundary`, `nothinning` | the positive name `=False`: `symmetry=False`, `antipodal=False`, ...; where the positive name is already a keyword (`antipodal`, `label`, `title`), the two merge, `False` meaning off |
| spelling | `regularization` | `regularisation`, British as `symmetrise` |

MATLAB's flags as strings (`'noBoundary'` positional, `CLAUDE.md` on `rotate(ebsd, r,
'keepEuler')`) are not affected. Where a function takes a MATLAB flag string, the string keeps
MATLAB's spelling.

Before R1 starts, write a script in `tools/` that lists every public signature's keywords
grouped by these themes. It checks what is left after each step and becomes a test in
`tests/test_staticapi.py`: no public keyword spelled the old way.

## 4. Steps

Each step is one commit on `syntax-review`, with the graph rebuilt and the gaps compared.

- **R0** The rules: `CLAUDE.md`, `tools/syntaxcheck.py` (the old spellings left; `--graph`
  compares the graph's counterparts with the baseline `bench/out/syntax-counterparts-before.txt`,
  2485 MATLAB nodes on 2026-09-30), the pages `PythonDevelopers` and `MATLABtoPython`, and the
  CHANGELOG.
- **R1** Defects and keywords (sections 2.5 and 3). This is the largest step by occurrences;
  do it one theme per commit.
- **R2** Duplicates (section 2.2).
- **R3** Pole figures (section 2.1), with every page that plots a pole figure rebuilt.
- **R4** Projections (section 2.3).
- **R5** Workarounds (section 2.6).
- **R6** Frames (section 2.4).
- **R7** Merge: the full test suite (asked for here), all doc pages rebuilt, `syntaxcheck.py`
  empty and `--graph` losing nothing, and the MATLAB | Python hand-off doc of the site updated.

## 5. Open questions

### Module functions shadowing builtins

`from mtex import *`, which the doc pages use, puts `sum`, `max`, `min`, `round`, `abs`-like
names over Python's builtins. There are three ways: methods only (`v.sum()`), flat functions
that fall back to the builtin for other input, or leaving them. The user is still thinking
about this; it is not part of R0 to R7.

### Not changed, for the record

- The `calc` prefix (`calcGrains`, `calcODF`, ...) stays. New functions avoid it where no
  property clashes.
- The numbered standard orientations stay.
