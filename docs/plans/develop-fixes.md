# The port's findings fixed in MATLAB

The audit of 2026-10-01 (`docs/research/matlab-defects-audit.md`) lists 28 defects the port found
in MATLAB MTEX that are still on develop. The user decided on each. The fixes go into develop,
made in the worktree `~/mtex/mtexDevelop` and checked through a bridge session of its own; never
into `frameSymmetry`. Every run stays under five minutes; MATLAB's test tiers only when asked.

## Decisions

| # | Defect | MATLAB | Decision |
|---|---|---|---|
| 1 | `S2FunHarmonic.unimodal` not normalised: the degrees overwritten by the coefficient indices | `S2Fun/@S2Fun/unimodal.m` | fix |
| 2 | `SO3FunHarmonic.interpolate` fits with an NFSOFT cutoff of 1, `eval` uses 4 | `SO3Fun/@SO3FunHarmonic/interpolate.m` | fix |
| 3 | `S2DirichletKernel` halfwidth always 180 degrees | `S2Fun/S2KernelFunctions/S2DirichletKernel.m` | fix |
| 4 | `S2FunBingham` normalised by a saddle point approximation, mean 1.0377 | `S2Fun/@S2FunBingham/S2FunBingham.m` | fix |
| 5 | `vector3d/refine` appends the south pole before the cap filter | `geometry/@vector3d/refine.m` | fix |
| 6 | pole figure scans with different antipodal flags (`allR{1}` alone) | `PoleFigureAnalysis/@PoleFigure/PoleFigure.m` | fix |
| 7 | generic orientation import: unit guessed, `'passive'` twice, quaternion columns crash | `interfaces/loadOrientation_generic.m`, `interfaces/tools/loadHelper.m` | fix |
| 8 | `.ctf` exported to `.ang` and read back turned by 30 degrees | `interfaces/exportEBSD_ang.m` | fix |
| 9 | `stiffnessTensor.load` drops the density the file states | `interfaces/loadTensor_generic.m` | fix |
| 10 | `mtexdata`'s cached ODF keeps a stale plotting convention | `tools/mtexdata.m` | fix |
| 11 | `parentGrainReconstructor` keeps stale votes and graph after a merge | `@parentGrainReconstructor/merge*.m`, `calcParentFromVote.m` | fix |
| 12 | `smoothBoundary` keeps stale pixel pairs in `gB.ebsdId` | `@grain2d/smoothBoundary.m`, `@grainBoundary/refine.m` | fix |
| 13 | the S2FunOperations page states a gradient of 0.0012, MATLAB computes 1.8e-4 | `doc/SphericalFunctions/S2FunOperations.m` | fix |
| 14 | `.osc` of a stopped scan keeps its placeholder records | `interfaces/loadEBSD_osc.m` | fix |
| 15 | `EBSD/interp` takes the padding of a grid as a nearest measurement | `EBSDAnalysis/@EBSD/interp.m` | fix |
| 16 | `spatialTransformProjective.fit` without normalised coordinates | `geometry/spatialTransforms/spatialTransformProjective.m` | fix |
| 17 | `SO3FunCBF/discreteSample` reads cos theta in [0, 1], an unnormalised distribution | `SO3Fun/@SO3FunCBF/discreteSample.m` | fix |
| 18 | `discreteSample(..., 'range', ...)` ignores the range | `tools/math_tools/discreteSample.m` | fix |
| 19 | `kde1d`'s `'bandwidth'` sets the diffusion time, not the bandwidth it returns | `tools/math_tools/kde1d.m` | fix |
| 20 | the OrientationEmbeddings page asks for zero samples for 13 halfwidths | `doc/.../OrientationEmbeddings.m` | check, then fix |
| 21 | FMC splits a uniform grain | `gbcFMC` | later: study the hierarchy, then fix MATLAB and the port together |
| 22 | `grain3d/principalComponents` fails on triangulated grains | `@grain3d/principalComponents.m` | fix |
| 23 | `neper.simulateChildGrains` searches a fixed window of 50 children | `@neper/simulateChildGrains.m` | fix |
| 24 | `curvature` and `calcGND` of a voxel map fail on the 8-corner cell | `latticeBasis.m` via `EBSD3square` | fix |
| 25 | `KAM`, `fill`, `reduce`, `fillByGrainId` of an `EBSD3` work in x and y only | `EBSDAnalysis/@EBSD3/...` | fix |
| 26 | `grain3d/orientFaces` only swaps sides | `@grain3d/orientFaces.m` | fix |
| 27 | LSCV's weights not normalised, 1./(1-w) infinite | `LSCV.m:17` | fix the weights; the truncation at degree 25 and the stop at the first increase stay, the method's design |
| 28 | the INEL pole figure file `data/PoleFigure/a52214s.int` has no reader | `interfaces/` | add a reader |

## Order

One commit per fix on develop, each checked through the bridge against the port's behaviour or an
analytic value: the function spaces (1-4, 17-19, 27, 2), the geometry and pole figures (5, 6, 16),
the interfaces (7-9, 14, 28), the maps (10-12, 15) and the volumes (22-26), then the pages (13, 20).

## Status

Decisions taken 2026-10-01. All 27 decided fixes are done on develop (2026-10-01, unpushed), each
checked through the bridge session `mtexdev` against the port or an analytic value:

| # | Commit | Note |
|---|---|---|
| 1 | `7eac7a837` | also: the rotation took v to the pole instead of the pole to v |
| 3 | `f99d91d63` | the base `S2Kernel.halfwidth` takes the first half-maximum crossing |
| 4 | `3e082fb74`, `9dd0427a5` | `mhyper` cancelled for negative arguments (the port too, fixed `d6d7b5d`); `fit` set N to the constant and projected on the rows of `a.xyz`; the fit uses `mhyper`'s moments, no MEX |
| 17 | `29cd16be0` | |
| 18 | `c5fb52d5f` | also the undefined default of `'weights'` |
| 19 | `6688a5124` | the DensityEstimation page's 0.004 is now a bandwidth, as in the port |
| 27 | `f8bdade28` | LSCV did not run at all: `eodf.components` of the old ODF class |
| 2 | `3098c17f3` | the options reached the LSQR operators as one nested cell, so no option had an effect |
| 5 | `630909039` | the PoleFigureRefinement page's numbers rerun: 327 directions, errors 0.20 and 0.067 |
| 6 | `bd97d9a4d` | Geesthacht after `correct`: both flags set, 2.27 mrd |
| 16 | `6b10822f6` | the row-wise sample recovered to 1e-14 (7193 off before) |
| 7 | `3564bc50b` | degrees unless `'radians'` (user, 2026-10-01), as the port; the OrientationImport page drops its three limitations |
| 8 | `15a0101ce`, `02b0d42cf` | both exporters write in the reader's frame (EDAX's for `.ang`, the default for `.ctf`); the port had the same defect, fixed `dba5b7e`; also `loadEBSD_ctf`'s space group fallback passed the string `',abc(:)'` as the lattice |
| 9 | `ea4bd3d70` | the TensorImport page no longer passes the density by hand for want of it |
| 14 | `c7e82623e` | checked on a copy of copper.osc with its last 100 records blanked |
| 28 | `2f63839b9` | the polar angle is \|khi\|, the reflection a Miller passed in or guessed from the name with a warning |
| 10 | `b89c258aa` | the cache is keyed by the MTEX version and the default plotting convention; a switch costs one rebuild |
| 11 | `2ee5f6ff3` | every merge clears votes and graph, as `revert` did; the martensite scenario now asks for new votes instead of an index error |
| 12 | `e633e0c04` | the stale pair becomes 0, the outer boundary's no-pixel; `calcGBND` uses the segments with a pair; two smoothing pages reworded |
| 15 | `0bfe3b64e` | gridded and list copper maps now interpolate alike (68 sites differed) |
| 22 | `28832ceeb` | triangulated and polygon grains agree to 3e-12 |
| 23 | `c75c71c2b` | the grains scanned one by one, as the port; 4 hosts give [49 57 74 97] where the window gave [50 0 0 0] and failed |
| 24 | `7fc28c74f` | `EBSD3square/curvature`, the dyads of the three gradients; calcGND runs |
| 25 | `f47dcd3a5` | all four on the lattice of `gridify`; `reduce` had returned the volume unchanged, `fillByGrainId` indexed by an undefined `ci` (its 2D twin `@EBSD/fillByGrainId.m:37` still does) |
| 26 | `51b3d87d2` | faces with new sides take them with phases and misorientation; the pair read as grain ids, not positions |
| 13 | `73ea397a2` | develop's gradient at X is 0.0012 (frameSymmetry 1.8e-4), aliasing; the page calls it a residue, and its extrema follow fix 1 |
| 20 | `4d8a4ab9e` | develop gave NaN points, no error; a floor of 1000 samples, the port's page too (`0312e72`) |

Found on the way, not fixed: `interpolate` merges nodes within 0.02 rad and averages their values,
which bounds the fit (0.005 on 20000 random nodes of a bandwidth 16 function); `export(copper, '.ctf')`
of the hexagonal grid `.osc` map loses 67 pixels and moves positions by up to 5 um; no reader on
develop claims `data/PoleFigure/Rigaku.DAT`; `data/Neper/my100grains.tess`, which three 3D pages load, was not on
develop (tracked since `abac772d9`, the stray `data/Neeper` removed); `@EBSD/fillByGrainId.m:37` indexes by an undefined `ci`.

Left: 21 (FMC), decided for later. Nothing pushed.

## Found by the function reference (2026-10-14)

Found while writing the calling forms of `docs/plans/reference.md`, fixed in the port, open on develop:

- `fibre.alpha` and `fibre.tau` (`geometry/@fibre/fibre.m`): `byMiller([111],[110])` picked a copy
  of [110] in (111) the first end did not share, so alpha's geodesic turned about [0 -8 -11] over
  102.2 degrees, tau's about [12 5 12]; fixed on develop `8ddab621b` (2026-10-14, unpushed):
  alpha (001)[1-10] to (111)[1-10], tau (001)[110] to (110)[00-1], checked through the bridge
  (h || RD over 54.74 degrees, h || TD over 90, both ends on the fibre, the fibreODF's maximum at
  them), the port's ends to the digit.
- `orientation/axis` of two orientations of different crystal frames: the port reduces over the
  products of both groups, as MATLAB's general branch; to check against MATLAB on the bridge.

## Found while rewriting the MATLAB help texts (2026-10-14)

The help of 127 functions and classes was rewritten on develop (`9b2059d42`). Bugs the agents
noticed in the code, not fixed:

- `S2VectorFieldTri/cross.m`, `dot.m`, `times.m` use an undefined `b` (should be `a`); `times.m`
  multiplies `sVF.values .* sVF`; `plus.m` uses an undefined `a` and `SVF.vertices`.
- `S2VectorFieldTri/angle.m` takes `acos(dot(...))` without normalising.
- `S2VectorFieldHarmonic/mean.m` returns `fhat(1,:)` without the `/sqrt(4*pi)` of
  `S2FunHarmonic/mean`; check `SO3VectorFieldHarmonic/mean` too.
- `S2VectorFieldHarmonic/mtimes.m` leaves its output undefined for two non-numeric arguments.
- `spinTensor.m`: the rotation branch uses an undefined `M`.
- `S1FunHarmonic/power.m`, `S2FunHarmonic/power.m`: `a.^sF` takes `a * bandwidth` as the bandwidth.
- `quaternion/power.m` flips to `a >= 0` first, wrong for general quaternions and odd n.
- `vector3d/plot.m` 'custom' asks `plotCustom` for outputs it does not return.
- `planarColorKey`: the constructor does not take `'periode'`, its first argument is always a
  colour map, and `property2color` scales by `max(v1)` instead of the period.
- `calcGBND`: the documented `nonneg` is not read here, the `oneSide` flag it reads was undocumented.
- `uniformComponent.loadobj` turns an old uniform component into one RBF at the identity.
- `coolColorMap.m`, `parulaColorMap.m` define functions named `cool`, `parula`.
- `getUnitScale('au')` is the Ångström, not the astronomical unit.
- the stale reference pages `trueEbsd2.*` (class renamed to `trueEbsd` in f903375b2) remain on the
  site; `remapShifted` (trueEBSD repository) has a help continuation line read as an entry.
