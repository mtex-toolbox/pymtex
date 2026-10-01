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

Decisions taken 2026-10-01; nothing built yet.
