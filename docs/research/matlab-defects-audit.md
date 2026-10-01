# MATLAB defects found by the Python port: are they still on develop?

Audit of 2026-10-01. Target: MATLAB `develop` at `/home/hielscher/mtex/mtexDevelop`, HEAD `d86e0507f`
(2026-10-01), clean checkout. Read-only; MATLAB was not run. "Reproduced on develop" below
refers to MATLAB's own `../master/TODO.md` rows O30-O34, F14, F15, I10-I13, G43, G44, C22,
which were reproduced on develop through the bridge on 2026-09-24. For every item the
develop code was read, and `git log -1 -- <file>` showed no change to it after its finding.
Line numbers refer to develop.

Sources read: `docs/findings.md` (all four sections), `docs/research/*.md`
(benchmark-calcgrains, calcgrains-release-audit, test-review), `docs/plans/*.md` (halfwidth,
maps3d, io, remaining-pages, ...), the pymtex git log, `../master/TODO.md` (the rows the
port's pages produced), `../master/tests/*Discrepanc*.md` and
`mtexDevelop/tests/S2Discrepancies.md`. The discrepancy notes are about the discrepancy
statistics, not discrepancies with the ports, and contain no defect reports.

Develop already holds the cherry-picks of the master fixes: `34e444be6` (calcKernel, BCV,
kde1d; master `186e53d2f`), `f3e70a78a` (Dream3d, neper, slice; master `e79185787`) and
`be7a27187` (#2624).

Checked afterwards: items 1 and 3 by reading the develop code, item 10 by running
develop through the bridge. A cached Dubna ODF read under `y↑→x` keeps `y↓→x`, and so does
`SO3Fun.dubna`; cached pole figures (`mtexdata dubna`) follow the session in both directions, so
the defect is the ODF's. The matching fixes in the other direction, MATLAB's into the port, are in
`docs/plans/matlab-fixes.md`.

## Summary

| # | Defect | Source | Verdict | MATLAB location (develop) |
|---|---|---|---|---|
| 1 | `S2FunHarmonic.unimodal` not normalised: the degree vector is overwritten by coefficient indices before `sqrt(2*l+1)` (mean 0.1629, 9.72 at the centre for 1 and 97.85) | findings.md:336; TODO O30; remaining-pages S2FunOperations | STILL IN MATLAB | `S2Fun/@S2Fun/unimodal.m:33-36` |
| 2 | `SO3FunHarmonic.interpolate` runs LSQR with an NFSOFT of cutoff 1 while `eval` uses 4; the fit is optimal for another operator | findings.md:352; TODO O31 | STILL IN MATLAB | `SO3Fun/@SO3FunHarmonic/interpolate.m:167,177,189` vs `eval.m:111` |
| 3 | `S2DirichletKernel(N).halfwidth` is always 180 degrees | findings.md:342; TODO O32 | STILL IN MATLAB | `S2Fun/S2KernelFunctions/S2DirichletKernel.m:38` |
| 4 | `S2FunBingham` normalised by a saddle point approximation, mean 1.0377 | findings.md:343; TODO O33 | STILL IN MATLAB | `S2Fun/@S2FunBingham/S2FunBingham.m:81,91-93` |
| 5 | `vector3d/refine` appends the south pole before the hull, so the cap filter removes nothing and the grid keeps the pole and the skirt centroids | findings.md:308; TODO F14 | STILL IN MATLAB | `geometry/@vector3d/refine.m:12-21` |
| 6 | Geesthacht scans after `correct` carry different `antipodal` flags, so their quadrature weights and `normalize` differ (2.15 for 2.27 mrd) | findings.md:307; TODO F15 | STILL IN MATLAB | `PoleFigureAnalysis/@PoleFigure/PoleFigure.m:83` (only `allR{1}` made antipodal) |
| 7 | Generic orientation import: unit guessed (`'radians'` ignored), `'passive'` applied twice, quaternion columns crash | findings.md:318; TODO I10 | STILL IN MATLAB | `interfaces/tools/loadHelper.m:115-124`, `interfaces/loadOrientation_generic.m:83` |
| 8 | A `.ctf` map exported to `.ang` and read back is turned by 30 degrees (hexagonal) | findings.md:267; TODO I11 | STILL IN MATLAB | `interfaces/exportEBSD_ang.m:62-65,165-167`; `interfaces/loadEBSD_ang.m:82` |
| 9 | `stiffnessTensor.load` drops the density the file states | findings.md:331; TODO I12 | STILL IN MATLAB | `interfaces/loadTensor_generic.m` (txt2mat only) |
| 10 | `mtexdata` returns a cached `.mat`, so frames keep the plotting convention of the session that wrote the cache (the Dubna ODF `y↓→x`; the pole figures follow the session) | findings.md:356, 367; TODO I13 | STILL IN MATLAB | `tools/mtexdata.m:67-69,295` |
| 11 | `parentGrainReconstructor`: `mergeSimilar`/`mergeInclusions` keep `votes` (one row per old grain) and `graph`; `mergeByGraph` keeps `votes` | findings.md:326; TODO G43 | STILL IN MATLAB | `@parentGrainReconstructor/mergeSimilar.m:19-20`, `mergeInclusions.m:20`, `mergeByGraph.m:113`, `calcParentFromVote.m:34-80` |
| 12 | `smoothBoundary` keeps stale pixel pairs in `gB.ebsdId` on simplified/resampled segments | findings.md:322; TODO G44 | STILL IN MATLAB (documented behaviour) | `@grain2d/smoothBoundary.m:61-64`, `@grainBoundary/refine.m:28-33` |
| 13 | S2FunOperations page states a gradient of 0.0012 at X; MATLAB's run gives 1.8e-4 (aliasing) | findings.md:337; TODO C22 | STILL IN MATLAB (doc) | `doc/SphericalFunctions/S2FunOperations.m:162` |
| 14 | `.osc` of a scan stopped early: the unmeasured trailing records (position (0,0), Euler 4π) are kept as measurements, so the map does not grid | findings.md:247 | STILL IN MATLAB | `interfaces/loadEBSD_osc.m:35,54-57`, `loadHelper.m:110-111` |
| 15 | `interp(ebsd, pos)` on a gridded map takes the NaN-phase padding as a nearest measurement, so the result depends on whether the map was gridded | findings.md:400 | STILL IN MATLAB | `EBSDAnalysis/@EBSD/interp.m:45,54,62`; padding from `@EBSD/private/squarify.m:25` |
| 16 | `spatialTransformProjective.fit` solves the DLT on raw coordinates (no Hartley normalisation); with outliers the row-wise sample is not recovered | findings.md:405 | STILL IN MATLAB | `geometry/spatialTransforms/spatialTransformProjective.m:142-151` |
| 17 | `SO3FunCBF/discreteSample`: `theta = acos(t./M)` reads the histogram index as t in [0,1] instead of [-1,1]; fibre samples too close to the fibre | findings.md:421 | STILL IN MATLAB | `SO3Fun/@SO3FunCBF/discreteSample.m:6-11` |
| 18 | `discreteSample(fun, n, 'range', [a b])` returns index/10000 in [0,1], ignoring the range | findings.md:423 | STILL IN MATLAB | `tools/math_tools/discreteSample.m:33-36` |
| 19 | `kde1d(..., 'bandwidth', b)` sets the diffusion time t, while the returned bandwidth is sqrt(t)·R, so a returned value cannot be passed back | findings.md:423 | STILL IN MATLAB | `extern/kde1d.m:71-72,92` |
| 20 | OrientationEmbeddings page asks `discreteSample` for zero orientations for the 13 smallest halfwidths | findings.md:395; remaining-pages | STILL IN MATLAB (doc; crash on develop unconfirmed) | `doc/CrystalOrientations/OrientationEmbeddings.m:186-190` |
| 21 | FMC splits a uniform, undeformed grain into several pieces (40x40, 0.1 degree noise: 683/661/190/60/3/3) | findings.md:430 | STILL IN MATLAB | `grainBoundaryCriteria/@gbcFMC/private/FMC_Coarsen.m` (identical to frameSymmetry) |
| 22 | `principalComponents` of triangulated 3D grains fails: `c = c.xyz` on a double | maps3d.md:99-103 | STILL IN MATLAB | `EBSDAnalysis/@grain3d/principalComponents.m:32-33` |
| 23 | `neper.simulateChildGrains` searches the lamellae of a host in a fixed window of 50; a host with more children breaks the count | maps3d.md:120-122 | STILL IN MATLAB | `EBSDAnalysis/@neper/simulateChildGrains.m:75-90` |
| 24 | `curvature` and `calcGND` of a voxel map fail: "unit cell must have 4 or 6 corners" | maps3d.md:131-135 | STILL IN MATLAB | `@EBSD/curvature.m:37` → `gradientDir.m:28` → `@EBSD/gradient.m:86` → `lattice.m:34` → `latticeBasis.m:77`; 8-corner cell `@EBSD3square/EBSD3square.m:84`; `@EBSD3square/calcGND.m:25` |
| 25 | `KAM`, `fill`, `reduce`, `fillByGrainId` of `EBSD3` work on x and y only | maps3d.md:133-135 | STILL IN MATLAB | `@EBSD3/KAM.m:32`, `fill.m:27`, `reduce.m:26`, `fillByGrainId.m:15` |
| 26 | `grain3d/orientFaces` only swaps a face's sides; a face given with the same sign for both grains keeps wrong `grainId`s | maps3d.md:115-120 | STILL IN MATLAB (edge case) | `EBSDAnalysis/@grain3d/orientFaces.m:146` |
| 27 | `calcKernel(..., 'method','LSCV')` judges every candidate on data truncated at degree 25 and stops at the first increase | halfwidth.md:66-70 | STILL IN MATLAB (method weakness) | `geometry/@orientation/LSCV.m:20-21,49` |
| 28 | `data/PoleFigure/a52214s.int` (INEL) ships without any reader | io.md:209; pymtex `1cf1bfa` | STILL IN MATLAB (minor) | `data/PoleFigure/a52214s.int`; no INEL reader in `interfaces/` |
| 29 | `calcKernel` subsampled with replacement, biasing KLCV narrow | findings.md:422; halfwidth.md:739 | FIXED IN MATLAB | `34e444be6`, `geometry/@orientation/calcKernel.m:36` (`randperm`) |
| 30 | `BCV` did not run (undefined `weights`, `psi(i)` on a cell, printed every value) | halfwidth.md:70,739 | FIXED IN MATLAB | `34e444be6`, `geometry/@orientation/BCV.m:23,41` |
| 31 | `kde1d` paired `(1:n).^2` with `a(1:n)`; printed `t_star` | findings.md:423; halfwidth.md:76-80 | FIXED IN MATLAB | `34e444be6`, `extern/kde1d.m:79-82` |
| 32 | `calcGrains` fails on a map without indexed pixels (#2624) | benchmark-calcgrains.md:173-177 | FIXED IN MATLAB | `be7a27187` (`@grain2d/grain2d.m`) |
| 33 | `loadGrains_Dream3d` read AvgQuats (x,y,z,w, passive) as (a,b,c,d) | findings.md:452; maps3d.md | FIXED IN MATLAB | `f3e70a78a`, `interfaces/loadGrains_Dream3d.m` |
| 34 | `grain3d/slice` took a hole for a piece over the grain inside it | findings.md:453 | FIXED IN MATLAB | `f3e70a78a`, `EBSDAnalysis/@grain3d/slice.m` |
| 35 | `simulateGrains` and friends never passed `neper.id` | findings.md:454 | FIXED IN MATLAB | `f3e70a78a`, `@neper/simulate*.m` |
| 36 | Triclinic sigma sections draw the upper hemisphere only | findings.md:353; TODO O34 | ONLY ON frameSymmetry | `plotting/ODFSections/pfSections.m:58` identical; differs by the branch's antipodal `cAxisRec` |
| 37 | `symmetrise(sF, specimenFrame('222'))` returns a plain function plotted on the mmm octant | findings.md:341; TODO O34 | ONLY ON frameSymmetry | per TODO O34 (reproduced on develop) |
| 38 | `S2FunHarmonic.quadrature(f,'bandwidth',48)` displays 49 | findings.md:351; TODO O34 | ONLY ON frameSymmetry | develop sets it: `S2Fun/@S2FunHarmonic/quadrature.m:72` |
| 39 | `fibre.beta(cs,ss,'full').symmetrise` with specimen symmetry fails | findings.md:319; TODO O34 | ONLY ON frameSymmetry | per TODO O34 |
| 40 | `data/quartzPattern.mat` holds a class the branch no longer has | findings.md:391; TODO O34 | ONLY ON frameSymmetry | develop `c69f255d7` ships the pattern the page loads |
| 41 | Measured pole figures drawn in parula instead of MTEX's default map | findings.md:330, 442 | ONLY ON frameSymmetry (fixed on develop) | `018c99cf1`, `geometry/@vector3d/scatter.m:142` |
| 42 | `newMtexFigure('layout',[1,2]); plotPDF; nextAxis; plotPDF` overlays both in one axes | findings.md:426; remaining-pages GeneralConceptsOptions | ONLY ON frameSymmetry (likely fixed on develop, not run) | nextAxis rework `f2393f9a6`, `da793f8af`, `b3a1a6c96`, none on frameSymmetry; `tests/plotting/check_nextAxis.m` |
| 43 | `calcKernel(ebsd.orientations)` treats correlated pixels as independent | findings.md:248; halfwidth.md | NOT A DEFECT (statistical limitation, design) | `geometry/@orientation/calcKernel.m` |
| 44 | `mean(ori,'robust')` keeps another share than the port | findings.md:264 | NOT A DEFECT (another trimming rule) | — |
| 45 | Aspect ratio from an ellipse fitted to the boundary vertices | findings.md:281 | NOT A DEFECT (another definition) | — |
| 46 | Bump kernel stores 1024 Chebyshev coefficients | findings.md:298 | NOT A DEFECT (default `maxS1Bandwidth`) | `SO3Fun/SO3KernelFunctions/SO3BumpKernel.m:37-41` |
| 47 | ODF import `'interp'` fits a radial function and warns about the iteration limit | findings.md:299 | NOT A DEFECT | — |
| 48 | A pole figure remembers the spacing of its grid (14.5 for 15.3 degrees) | findings.md:309 | NOT A DEFECT | — |
| 49 | `equispacedSO3Grid` count and flatness differ | findings.md:317 | NOT A DEFECT (other construction) | — |
| 50 | Uniform ADF expanded to bandwidth 500 | findings.md:323 | NOT A DEFECT (approximation) | — |
| 51 | `calcParent2Child` ends in a higher local basin than the port | findings.md:327 | NOT A DEFECT (local optimiser) | — |
| 52 | Quadrature at bandwidth 1 of a degree-2 function errs 0.6667 (aliasing), not 0.5 | findings.md:339 | NOT A DEFECT (grid exact to 2N by design; the page's "nearly zero ... truncation error" wording is loose) | `S2Fun/@S2FunHarmonic/quadrature.m:56-61`; `doc/SphericalFunctions/S2FunQuadrature.m:52-60` |
| 53 | `linprog` returns an interior point of a degenerate face (Taylor) | findings.md:344 | NOT A DEFECT | — |
| 54 | Wave velocities interpolated on an 18338 node grid | findings.md:345 | NOT A DEFECT | — |
| 55 | `calcTaylor` spin field stored in left components, evaluated as right spin tensors | findings.md:357 | NOT A DEFECT (`internTangentSpace` vs `tangentSpace` by design) | `TensorAnalysis/@strainTensor/calcTaylor.m:68-70`; `SO3VectorFieldHarmonic.m:72,97-98` |
| 56 | `ebsd.prop.x = v` never checks the length | findings.md:427 | NOT A DEFECT (no validation, design) | `tools/dynProp.m:165-169` |
| 57 | `grain3d` faces coloured by values use the axes' parula | findings.md:457 | NOT A DEFECT (as MATLAB's 2D maps; only `EBSD3/plot.m:89` applies the default map) | `EBSDAnalysis/@grain3d/plot.m:62-78` |
| 58 | `g.numNeighbors(1:10)` fails | maps3d.md:103 | NOT A DEFECT (a method without an index argument; `g(1:10).numNeighbors` is the call; same in 2D) | `@grain3d/numNeighbors.m`, `@grain3d/subsref.m` |
| 59 | `smooth` and `spatialProfile` of `EBSD3` are stubs | maps3d.md:135 | NOT A DEFECT (explicit "not yet implemented") | `@EBSD3/smooth.m`, `@EBSD3/spatialProfile.m` |
| 60 | A held `plot(ori)` after `plot3d(odf)` replaced the isosurfaces | findings.md:329 | UNCLEAR | `SO3Fun/@SO3Fun/plot3d.m`; `doc/ODFAnalysis/ODFTheory.m:134-137` |
| 61 | OrientationFibre page's "58 percent" and "100 percent" stale (run gave 7.26 and 25.93) | remaining-pages OrientationFibre | UNCLEAR | `doc/CrystalOrientations/OrientationFibre.m:309-319` |
| 62 | NFFT and Wigner MEX fail to load after a long session ("cannot allocate memory in static TLS block") | findings.md:397, 398, 420 | UNCLEAR (environment or MEX linking) | `mex/` |
| 63 | `LSCV` does not normalise its weights, so with the default unit weights `1./(1-w)` is Inf (found in this audit, not by the port) | this audit, while checking #27 | UNCLEAR | `geometry/@orientation/LSCV.m:17,44-46` vs `BCV.m:19` |

Counts: STILL IN MATLAB 28, FIXED IN MATLAB 7, ONLY ON frameSymmetry 7, NOT A DEFECT 17, UNCLEAR 4.

## STILL IN MATLAB: evidence

**1. `unimodal` normalisation.** `S2Fun/@S2Fun/unimodal.m:33-36`:
`l = 0:bw; l = 1+l.^2+l;` then `f_hat(l) = psi.A ./ reshape(sqrt(2*l+1),...)`. The second
line wants the degree n and gets the index n²+n+1. `S2FunHarmonic.unimodal('halfwidth',10*degree)`:
mean 0.1629, 9.72 at the centre (should be 1 and 97.85). Last change 2026-05-15.

**2. `interpolate` cutoff.** `interpolate.m:167` `cutOff = get_option(varargin,'cutOffParameter',1)`,
used by the adjoint (:177) and the forward (:189) inside LSQR; `eval.m:111` defaults to 4.
Dubna values, bandwidth 32, λ = 5e-7: relative residual 0.0060 under the cutoff 1 transform,
0.0239 evaluated with `f.eval`; exact transforms give 0.0062.

**3. Dirichlet halfwidth.** `S2DirichletKernel.m:36-40`: `get.halfwidth` returns `pi` for every N.
`S2DirichletKernel(10).halfwidth` is 180 degrees, the half maximum angle 11.6 degrees (21.4 at N = 5).

**4. Bingham normalisation.** `S2FunBingham.m:81` `BS2.N = 4*pi./BS2.normalizationConst`,
:93 `numericalSaddlepointWithDerivatives(...)*exp(1)`: an approximation.
`mean(S2FunBingham([-10 -4 0],a))` is 1.0377; the page calls the 3.8 percent normalisation error.

**5. `refine`.** `refine.m:12` `v = [v(:); -zvector];` precedes `:19`
`r(r.theta > max(v.theta(:))) = [];` so `max(v.theta)` is π and nothing is removed; the pole
itself is returned too (`:21`). PoleFigureRefinement ends at 367 directions where 327 are expected.

**6. Antipodal flags of a pole figure.** `PoleFigure.m:83` sets
`pf.allR{1}.antipodal = true` only for the first specimen grid; `get.antipodal` (:189) reads
`allR{1}` alone. `loadPoleFigure_geesthacht.m:30-37` passes one grid per pole figure, so the
second and later grids stay non-antipodal, and `mean` (`mean.m:16`) weights them through
`calcQuadratureWeights`, whose `unique`/Voronoi step depends on the flag. Reproduced on develop
(TODO F15): `correct(pf({1,3}),'background',pf({2,4}))` on `mtexdata geesthacht` gives flags 1 and
0, max weights 0.0015 and 0.0024, `normalize` 0.03 to 2.15 mrd instead of 2.27. Root cause read
from the code, not run.

**7. Generic orientation import.** `loadHelper.m:115-117`: `if any(any(rotData>15)), rotData = rotData*pi/180; end`
decides the unit from the values; `loader.unit` (the `'radians'` flag, :54) is used only for the 4π
check (:110). A file `5 10 12; 3 7 9; 11 2 4` read with `'ColumnNames',{'phi1','Phi','phi2'}` is
taken as radians (47.16 degrees off). `'passive'` inverts in `loadHelper.m:123-124` and again in
`loadOrientation_generic.m:83`, so it has no effect. `loadHelper.m:120`
`rotation(quaternion(rotData))` calls the constructor with one N×4 matrix, which reads
`varargin{2..4}`: "Index exceeds the number of array elements".

**8. CTF to ANG, 30 degrees.** `exportEBSD_ang.m:62-65` only undoes the Euler correction and
:165-167 writes `g.rotations.phi1/Phi/phi2` as they are, relative to the map's crystal frame;
nothing re-expresses them in the EDAX alignment (X||a for trigonal and hexagonal) that
`loadEBSD_ang.m:82` assumes when reading (`crystalSymmetry(...,'EDAX')`). A CTF phase has MTEX's
default alignment, so `mtexdata twins` (Mg) exported to `.ang` and read back is 30.00 degrees off.

**9. Tensor density.** `loadTensor_generic.m` is `T = txt2mat(fname); T = tensor(T,varargin{:})`;
the `density kg/m3 3355.0` line of `data/tensor/Olivine1997PC.GPa` is never read.
`doc/Tensors/TensorImport.m:72-83` passes it by hand.

**10. `mtexdata` cache.** `mtexdata.m:67-69` loads `data/<name>.mat` if it exists, :295 writes it
after the first read. The stored objects keep the frames and plotting convention of the writing
session and survive reader fixes. `plottingConvention.default('y↑→x'); odf = SO3Fun.dubna` shows
`y↓→x` with an old cache. The `'force'` option is the only escape.

**11. Stale votes after a merge.** `mergeSimilar.m:19-20` and `mergeInclusions.m:20` replace
`job.grains` and `job.mergeId` but leave `job.votes` (one row per grain) and `job.graph`;
`mergeByGraph.m:113` clears the graph only. `calcParentFromVote.m:34-80` then forms
`doTransform` from the old rows and indexes `job.grains(doTransform)` (:75, :80). Scenario:
`job.calcGBVotes; job.mergeSimilar; job.calcParentFromVote` gives an index error or transforms
other grains.

**12. Stale `ebsdId`.** `@grainBoundary/refine.m:28-33` documents that resampled segments inherit
`ebsdId` from the segment covering their midpoint; `smoothBoundary.m:61-64` advises `'noSimplify'`
and `'noRefine'`. The pairs are kept, not cleared, so `calcGBND` (`calcGBND.m:90-91`) silently reads
wrong pixel pairs after a default `smoothBoundary`. It is documented, but TODO G44 files it as a bug.

**13. S2FunOperations text.** `doc/SphericalFunctions/S2FunOperations.m:162` "At specimen X the
gradient magnitude is 0.0012"; MATLAB's run prints 1.8e-4, which is quadrature aliasing on a flat
stretch.

**14. `.osc` of a stopped scan.** `loadEBSD_osc.m` keeps every record the file announces:
`loadHelper(...,'Radians')` (:35) turns the placeholder Euler angles 4π into NaN rotations
(`loadHelper.m:110-111`), but their positions (0,0) and phases stay (`:54-57`
`EBSD(pos,rot,phase,...)`). The placeholders collide on the first pixel, the map cannot be gridded,
and a 1.2 M point Rohrer scan reads in 17 s as a list (the port trims them, 2.5 s, gridded).

**15. `interp` and the padding.** `interp.m:45` `knnsearch` over all positions of the map, padding
included; :54 accepts a hit within the cell radius; :62 copies its `phaseId`. A gridded map's
padding has phase NaN (`squarify.m:25`). A new position nearest to a padding cell becomes
unindexed although a measurement lies within reach: on the copper map 19 of 14042 sites at
2.5 times the cell (101 of 346326 at 0.5) are unindexed in MATLAB. The result depends on whether
the map was gridded first.

**16. Homography fit.** `spatialTransformProjective.m:142-151` builds the DLT rows from the raw
`x, y, xB, yB` and calls `robustLsq` without centring and scaling. The algebraic residual weighs
a point by its distance from the origin. With every tenth point moved by (30,-40), the page's
row-wise sample is not recovered (7193 off); the port with Hartley's normalisation recovers it
to 1e-13.

**17. CBF sampling.** `SO3FunCBF/discreteSample.m:7` `t = linspace(-1,1,M)`, :10 `[~,t] = histc(rand,c)`
returns indices 1..M, :11 `theta = acos(t ./ M)` maps them to cos θ in [0,1] instead of
`t_grid(index)` in [-1,1]. The CDF `c` (:8) is not normalised to end at 1 either. Fibre samples
come out about √2 too close; CLASSIX on ClusterDemo then separates fibre and peak only because of it.

**18. `discreteSample` of a handle.** `tools/math_tools/discreteSample.m:33-36`: the density is
sampled on `linspace(range(1),range(2),10000)` but the return is
`discretesample(...)/length(density)`, a value in (0,1]. With `'range',[2 5]` every sample lies in [0,1].

**19. `kde1d` bandwidth option.** `kde1d.m:71-72` `t_star = get_option(varargin,'bandwidth')`,
:92 `bandwidth = sqrt(t_star) * R`. The option is the diffusion time on the rescaled interval, the
output a bandwidth in data units, so passing the output back changes the estimate.

**20. OrientationEmbeddings.** `doc/CrystalOrientations/OrientationEmbeddings.m:186-190` loops
`hw = logspace(-1,1.75,40)*degree` and draws `round(1000*(hw*6)^3)` samples. That is 0 for
hw < 0.76 degree, the first 13 of 40 values. MATLAB's run on frameSymmetry failed there; develop's
`SO3FunRBF/discreteSample` was rewritten on 2026-09-26 (`c836b404a`), so whether develop errors
or yields NaN points is not confirmed. The request for zero samples is in the page either way.

**21. FMC.** The FMC files on develop are byte-identical to frameSymmetry's
(`@gbcFMC/gbcFMC.m`, `private/FMC_*.m`). The port reproduces MATLAB's hierarchy pair for pair
(59650 pairs), and findings.md:430 records the split "in MATLAB and in the port alike": a 40x40
map with 0.1 degree noise becomes 683/661/190/60/3/3 pixels. A plausible mechanism, not
verified: `FMC_Coarsen.m` part0 `EdgeDilution` zeroes every edge below `sumW/(gammaW*n)` at both
ends, which on near-uniform weights cuts a random subset of edges.

**22. `principalComponents`, triangulated.** `@grain3d/principalComponents.m:19` takes the
`isnumeric(grains.F)` branch; :32 computes `c` as a double matrix, :33 `c = c.xyz` fails ("Dot
indexing is not supported"). The polygon branch (:85-88) builds a `vector3d` first and works.

**23. `simulateChildGrains`.** `:75` `dk = 50`; :89-91, when no non-lamella face is found within
the next 50 grains, `numChilds(l) = dk; break`, so the remaining parents get no children and
`pId = repelem(1:numParents,numChilds)` (:103) is shorter than the grains. At the default width
0.015 in a unit cube a host holds about 66 lamellae.

**24. `curvature`/`calcGND` of a voxel map.** `EBSD3square` has an 8-corner unit cell
(`EBSD3square.m:84`) and no own `curvature`, `gradient` or `lattice`. `@EBSD/curvature.m:37` →
`private/gradientDir.m:28` → `@EBSD/gradient.m:86` `ebsd.lattice.A` → `lattice.m:34`
`latticeBasis(ebsd.unitCell)` → `latticeBasis.m:77` error "unit cell must have 4 or 6 corners".
`@EBSD3square/calcGND.m:25` calls `ebsd.curvature` and fails the same way.

**25. EBSD3 in-plane methods.** `@EBSD3/KAM.m:32` `spatialDecomposition([pos.x, pos.y],...)`,
`fill.m:27` `scatteredInterpolant([x, y],...)`, `reduce.m:26` indexes by x and y only,
`fillByGrainId.m:15` a 2D `meshgrid`. On a volume, voxels sharing x and y are treated as one
column, so the neighbours and filled values are wrong.

**26. `orientFaces`.** `@grain3d/orientFaces.m:146`
`grains.boundary = flip(grains.boundary, ~all(newId == oldId,2))` swaps the two sides of every
changed face. A face whose input carried the same sign for both grains had `oldId = [g, 0]`. Its
new sides `[g1 g2]` are not a swap of the old, so `boundary.grainId` (and the phases and
misorientation that go with it) disagree with `I_GF` after the call.

**27. LSCV.** `LSCV.m:20-21` evaluates every candidate on `calcFourierODF(...,SO3DirichletKernel(25))`,
so candidates sharper than about 8 degrees are judged as if they were 8 degrees. `:49`
`if i>1 && c(i)>c(i-1), break; end` stops at the first local minimum of the ten-point grid.
This is a weakness of the method rather than a crash (see also #63).

**28. Unreadable sample file.** `data/PoleFigure/a52214s.int` is an INEL "Fdt File" with no reader
among `interfaces/loadPoleFigure_*.m`, and no `.m` file refers to it, so it is a sample file
MTEX cannot load.

## UNCLEAR: what would settle it

**60. Held `plot3d(odf)`.** `SO3Fun/@SO3Fun/plot3d.m` draws via `newOrientationPlot` and
`contour3s`; the overlay goes through `orientation/plot` → `newMtexFigure`. Static reading does
not show whether `hold on` keeps the isosurface patch. Run on develop:
`plot3d(odf,'Euler'); hold on; plot(oriPlot,'Euler','MarkerEdgeColor','k'); hold off` (ODFTheory
cell) and check that `findobj(gca,'type','patch')` is non-empty.

**61. OrientationFibre percentages.** The page text (`OrientationFibre.m:312,319`) is the same on
develop and frameSymmetry; the frameSymmetry run printed 7.26 and 25.93. Run on develop:
`volume(odf,betaFull,5*degree)` and `10*degree` from that page. If they are not 0.58 and 1.00,
the text is stale.

**62. MEX static TLS.** "cannot allocate memory in static TLS block" when loading the NFFT and
`wignerTrafoAdjointmex` MEX after a long session. The usual cause is a library with initial-exec
TLS (often libgomp) loaded late by `dlopen`. To settle it: `ldd` and `readelf -d` on the MEX files
for the OpenMP runtime, and a test with `LD_PRELOAD` of that runtime at MATLAB start. It is not
a code defect of the toolbox functions.

**63. LSCV weights.** `LSCV.m:17` `w = get_option(varargin,'weights',ones(size(ori)))` is not
normalised (BCV does `w = w ./ sum(w(:))` at `BCV.m:19`). The criterion (:44-46) contains
`1./(1-w)` and `w./(1-w)`, which are Inf for unit weights, so `c(i)` is NaN or Inf unless
`calcFourierODF` or a caller supplies normalised weights. Run
`calcKernel(orientation.rand(1000,crystalSymmetry('m-3m')),'method','LSCV')` and print the vector `c`.

## Skipped: the port's own defects, gaps and conventions

These are the remaining rows of `docs/findings.md`: the port's own bugs, gaps beside MTEX, and
conventions or practicalities that do not claim a MATLAB defect. Each was considered and has no
MATLAB counterpart to check. Other sources considered and skipped:
`calcgrains-release-audit.md` (the default angle and the final `minPixel` cull were the port's;
MATLAB's `enforceMinPixel` is the reference), `benchmark-calcgrains.md` (the Enstatite split and
the 24 378 grains came from the rebuild of the exported map, not from MATLAB),
`test-review.md` (no MATLAB defect), `harmonics-speed.md`/`benchmark.md` (MATLAB speed
references only), and the pymtex git log (its MATLAB defects are #2624 and the six master fixes
above).

- findings.md:11 (port bug) VectorDefinition: the display of a container holding NaN crashed in `numString` (`int(nan)`)
- findings.md:12 (port bug) VectorsAxes: `complete=True` drew one hemisphere of an antipodal function or pole figure, the flag being dropped before ...
- findings.md:13 (port bug) VectorsDensityEstimation: `max(pdf, numLocal=3)` returned two maxima where MATLAB finds three: the seeds of the ascent were the best ...
- findings.md:14 (port bug) VectorsDensityEstimation: the inverse pole density of a Laue frame reported `antipodal: false` where MATLAB's is antipodal
- findings.md:15 (port bug) EBSDOrientationPlots: the pole figure sample budget divided 10000 by the number of poles of h instead of the number of symmetry ...
- findings.md:16 (port bug) EBSDSharpPlot: `project2FundamentalRegion(h)` of calcite directions put 537 of 20119 azimuths below -25 degrees where ...
- findings.md:17 (port bug) EBSDOrientationPlots: `hold` was ignored by the pole figure rows of an orientation list, a second call opened a new figure
- findings.md:18 (port bug) EBSDOrientationPlots: `plotSection(ori, values, 'sigma')` dropped the positional values
- findings.md:19 (port bug) EBSDOrientationPlots: `scatter(ori)` drew axis angle space where MATLAB draws the Euler region, and every one of 152345 points
- findings.md:20 (port bug) EBSDSharpPlot: the calcite map (`mtexdata('sharp')`) is on a lattice turned by 45 degrees against its axis aligned box, ...
- findings.md:21 (port bug) EBSDSharpPlot: `mean(ori)` failed on a list holding NaN orientations, as a gridded map's list does now
- findings.md:22 (port bug) EBSDSharpPlot: `ipfHSVKey.maxAngle` and `whiteCenter` could not be set; a colour array in the map's shape was refused by ...
- findings.md:23 (port bug) EBSDInter, EBSDFilling: `ebsd['xy', x, y]` and a flat boolean mask failed on a gridded map, the flat positions being taken as row ...
- findings.md:24 (port bug) EBSD2ODF: the seeds of the local maxima failed on a Laue group, an improper element being no rotation
- findings.md:25 (port bug) EBSDGrid: `ebsd.orientations` dropped the sites of other phases and the empty sites of a gridded map, so ...
- findings.md:26 (port bug) EBSDPlotting: the phase legend listed a not indexed phase without a colour, and failed on grains without a phase
- findings.md:27 (port bug) VectorGrids: `fibonacciS2Grid(resolution=)` rounded the point count before halving and gave 825 nodes at 7 degree for ...
- findings.md:28 (port bug) GrainReconstruction: `plot(ebsd)` and `plot(grains)` opened a figure of their own after `newMtexFigure(layout=...)` and ...
- findings.md:29 (port bug) GrainReconstruction: the phase legend listed notIndexed, whose colour on the Forsterite map is NaN
- findings.md:30 (port bug) GrainSpatialPlots: `text(grains, labels)` failed, reading the centroid as a method, and drew labels outside a `region`
- findings.md:31 (port bug) GrainSpatialPlots: a `region` in map coordinates was set as the axes limits without the layout, so a map under `y↓→x` came ...
- findings.md:32 (port bug) GrainMerge: `newMtexFigure` and `nextAxis` gave every child its own gridspec, so the panels of a figure that grew by ...
- findings.md:33 (port bug) GrainMerge: `setColorRange` after `mtexColorbar` left one bar per axes where MATLAB redraws one for all
- findings.md:34 (port bug) ShapeParameters: the Euler angles of a rotation list came as three rows, which the grain display read as one row per grain
- findings.md:35 (port bug) HullBasedParameters: `perimeter` counted the loops around inclusions where MATLAB's counts the outer loop, so `deltaP` of a ...
- findings.md:36 (port bug) ProjectionBasedParameters: `S1FunHarmonic.min` returned the maximum's position on a series interpolated at the session bandwidth: the ...
- findings.md:37 (port bug) GrainOrientationParameters: `fibre.fit` reported a NaN fit angle and `scatter(ori)` refused a list holding NaN orientations, as the ...
- findings.md:38 (port bug) GrainOrientationParameters: `quiver` in `sphericalplot.py` named `orientation` without importing it
- findings.md:39 (port bug) GrainReconstructionAdvanced: a list of thresholds, one per phase in the order of `CSList`, was taken as one threshold unless an entry ...
- findings.md:40 (port bug) RotationFibre: `project2EulerFR` took the first symmetric equivalent inside the Euler box, so a fibre segment inside the ...
- findings.md:41 (port bug) RotationSpinTensor: `orientation.byAxisAngle(Miller(..., cs), w)` without frames built an orientation into the specimen frame, ...
- findings.md:42 (port bug) RotationSpinTensor: `Miller(t, cs)` of a right tangent vector kept the vector's UVTW style, so `round` gave (2 4 -6 9) for ...
- findings.md:43 (port bug) RotationPlotting: `scatter(rot)` of a bare rotation failed on `project2FundamentalRegion`; a rotation has no region
- findings.md:44 (port bug) RotationFibre: `project2EulerFR` took the first symmetric equivalent inside the Euler box, so a fibre segment inside the ...
- findings.md:45 (port bug) RotationSpinTensor: `orientation.byAxisAngle(Miller(..., cs), w)` without frames built an orientation into the specimen frame, ...
- findings.md:46 (port bug) RotationSpinTensor: `Miller(t, cs)` of a right tangent vector kept the vector's UVTW style, so `round` gave (2 4 -6 9) for ...
- findings.md:47 (port bug) RotationPlotting: `scatter(rot)` of a bare rotation failed on `project2FundamentalRegion`; a rotation has no region
- findings.md:48 (port bug) ODFCharacteristics: `mtexdata('dubnaODF')` returned the pole figures, its recipe naming the wrong function
- findings.md:49 (port bug) ODFCharacteristics: the volume about a fibre came out 92.5 percent for an exact 94.0: the cap grid of `fibreVolume` put more ...
- findings.md:50 (port bug) ODFComponents: `calcComponents` on the Dubna ODF ran for half an hour: the clustering of the modes compared orientations ...
- findings.md:51 (port bug) ODFComponents: the third local maximum of `max(odf, numLocal=3)` differs between the direct and the harmonic evaluation ...
- findings.md:52 (port bug) DetectionOfSampleSymmetry: the Aachen pole figure reader took the tilt marker closing every ring, `-05.00`, as an intensity, so the ...
- findings.md:53 (port bug) ODFModeling: `0.73 * uniformODF + 0.27 * unimodalODF(mod, psi)` became a composition, the uniform part carrying a ...
- findings.md:54 (port bug) EulerAngleSections: `odf.SS = specimenFrame('222')` on a composition changed the label without reaching the components, so the ...
- findings.md:55 (port bug) EulerAngleSections: `plotSection(odf, ..., layout=[5, 2])` ignored the layout and laid the wide Euler sections out three abreast
- findings.md:56 (port bug) ODFShapes: `plot(psi)` in `hold(True)` after `plt.figure()` opened a figure per kernel: `hold` made an MTEX figure of ...
- findings.md:57 (port bug) RadialODFs: `plotPDF(odf, h)` inside a `newMtexFigure(layout=[2, 3])` loop opened a figure of its own instead of ...
- findings.md:58 (port bug) BinghamODFs: the lowest band of a `contourf` plot was coloured, so a sigma section of a Bingham ODF had a blue background
- findings.md:59 (port bug) FibreODFs: `fibre.beta(cs)` was not MATLAB's: the port projected the far end to the equivalent nearest the near end, ...
- findings.md:60 (port bug) CrystalReferenceSystem: `angle(ori1, ori2)` between orientations of one group in differently aligned crystal frames compared the ...
- findings.md:61 (port bug) CrystalReferenceSystem: poles on the equator, z = -0.0, were dropped by the upper hemisphere chart, 3 markers for MATLAB's 6; an ...
- findings.md:62 (port bug) CrystalReferenceSystem: `plotPDF` after `newMtexFigure` opened its own figure, and after `nextAxis` another one
- findings.md:63 (port bug) CrystalSymmetries: the proper subgroup of -4m2, 222 turned by 45 degrees, was named unknown; MATLAB names the type
- findings.md:64 (port bug) CrystalDirections: `char` of a single crystal direction wrote the family brackets `<101>` for a frame with symmetry, MATLAB's ...
- findings.md:65 (port bug) CrystalDirections: a held `plot` on a figure of two hemisphere charts went to the last chart only, so a direction of the ...
- findings.md:66 (port bug) FundamentalSector: the outline of a spherical region was drawn under the mirror circles of `plot(cs)`
- findings.md:67 (port bug) QuasiCrystals: the fundamental sector of the centrosymmetric icosahedral group was empty, its centre a zero vector, so ...
- findings.md:68 (port bug) CrystalOperations: `gcm()` kept a figure closed with `plt.close`, so the next plot drew into a dead figure
- findings.md:69 (port bug) CrystalOperations: `ori * S2Fun` warned about specimen symmetry while rotating the coefficients out of the crystal frame
- findings.md:70 (port bug) PoleFigure2ODF: `calcError(pf, odf, 'RP')` kept the regularised measure's threshold of 0.5 where MTEX's RP measure keeps ...
- findings.md:71 (port bug) PoleFigureRefinement: `plotPDF(odf, pf.allH)` without superposition coefficients failed on an entry of several crystal directions
- findings.md:72 (port bug) PoleFigureSantaFe: `FontSize`, MATLAB's spelling of the option, was refused by the plots and reached matplotlib
- findings.md:73 (port bug) PoleFigureImport: a superposed entry was displayed and titled with ` + ` between its crystal directions where MATLAB writes ...
- findings.md:74 (port bug) OrientationDefinition: `orientation.symmetrise(proper=...)` took a keyword it never read, the reduction groups being proper ...
- findings.md:75 (port bug) DefinitionAsCoordinateTransform: the Miller and tensor displays kept four significant digits, `1.667 1.333 0.3333` and `2.485`, where ...
- findings.md:76 (port bug) PoleFigurePlot: `mtexColorMap` after a filled contour plot flattened it: `set_cmap` replaced the bands cut out of the old ...
- findings.md:77 (port bug) PoleFigurePlot: `mtexColorbar` always added a bar where MATLAB's toggles, so a page that clears the bars before setting ...
- findings.md:78 (port bug) OrientationSymmetry: `plotPDF(ori, h, complete=True)` drew the sector of the specimen group, not the sphere: the flag was ...
- findings.md:79 (port bug) OrientationSymmetry: the poles of a list left the specimen group out, so a complete pole figure showed 6 of MATLAB's 12 poles, ...
- findings.md:80 (port bug) OrientationStandard: `round2Miller(ori)` rounded the plane and the direction over four indices where MATLAB's `round` allows ...
- findings.md:81 (port bug) OrientationStandard: the fundamental sector of an orthorhombic frame with a mirror was not turned by the plotting convention, ...
- findings.md:82 (port bug) OrientationStandard: a line contour drawn over a surface removed the surface, and a held row of pole figures opened a second ...
- findings.md:83 (port bug) OrientationStandard: a single RGB triple as `MarkerColor` was passed to matplotlib as a value per marker, which warned and ...
- findings.md:84 (port bug) OrientationFundamentalRegion: the grain exchange flag of a region was stored but never used: `fundamentalRegion(cs, cs, antipodal=True)` ...
- findings.md:85 (port bug) OrientationFundamentalRegion: orientations drawn into a held axis angle plot, the fundamental region for instance, came in Euler ...
- findings.md:86 (port bug) OrientationFundamentalRegion: `plotSection(ori, 'axisAngle', angles)` read the angles as marker values
- findings.md:87 (port bug) OrientationVisualizationSections: the largest angle of a region was sampled, 61.6 degrees for the cubic 62.8, so axis angle sections sat at ...
- findings.md:88 (port bug) OrientationInversePoleFigure: the 321 sector of the charts ran between two a directions where MATLAB's runs between two m directions
- findings.md:89 (port bug) OrientationEmbeddings: `discreteSample` of a Bingham distribution evaluated it on a 1.5 degree grid, 0.8 s per call, six minutes ...
- findings.md:90 (port bug) OrientationEmbeddings: an embedding of an empty list failed in the batch rotation, its mean divided by zero
- findings.md:91 (port bug) OrientationExport: the orientation table wrote fixed `%10.5g` columns where MATLAB's cprintf fits each column to its entries
- findings.md:92 (port bug) MisorientationDistributionFunction: `calcMisorientation(ebsd1, ebsd2)` returned inv(o2) o1 labelled (1 → 2), where MATLAB returns inv(o1) o2 ...
- findings.md:93 (port bug) MisorientationDistributionFunction: the axis of a misorientation between two phases was labelled with its second frame, MATLAB labels it with ...
- findings.md:94 (port bug) MisorientationDistributionFunction: `calcAxisDistribution` of a two-phase MDF was built as an antipodal function without a frame: h and -h ...
- findings.md:95 (port bug) AxisDistributionFunction: `axis(o1, o2)` took the axis of the best quaternion without making its scalar part positive, so the axis ...
- findings.md:96 (port bug) AxisDistributionFunction: the uniform axis distribution was a bandwidth-64 expansion of a sampled integral, min 0.568 and max 1.598 ...
- findings.md:97 (port bug) AngleDistributionFunction: the uniform angle distribution measured its sector areas on a latitude-longitude grid, so the hexagonal ...
- findings.md:98 (port bug) MisorientationGrainExchangeSym: a region was drawn as a surface with a latitude grid and no edges; `boundaryColor` and `noSurface` were ...
- findings.md:99 (port bug) MisorientationTheory: held marker plots all took the first colour of the order where MATLAB's take the next
- findings.md:100 (port bug) MisorientationTheory: a misorientation displayed its Euler angles where MATLAB shows its parallel planes and directions when ...
- findings.md:101 (port bug) ParentChildVariants: `variants(p2c)` and `variants(p2c, 'parent')` held MATLAB's set in another order, so variant, packet and ...
- findings.md:102 (port bug) ParentChildVariants: `nextAxis(k)` counted from zero and reached only axes that existed, and a held plot drew into the last ...
- findings.md:103 (port bug) ParentChildVariants: an open marker (`MarkerFaceColor='none'`) was drawn black and took no colour of the order, so the next ...
- findings.md:104 (port bug) ParentChildVariants, OrientationRelationshipFit: positional flags (`plotPDF(ori, h, 'upper')`, `plot(v, 'contourf', 'fundamentalRegion')`) and a positional ...
- findings.md:105 (port bug) ParentChildVariants: `variants(p2c, childOri, 'parent')` and `orientation.parents()` were missing
- findings.md:106 (port bug) MartensiteVariants: a four-index direction given with `'uvw'` displayed in three indices, `<100>` for MATLAB's ⟨2̅110⟩
- findings.md:107 (port bug) MartensiteVariants: `plotPDF(ori, 'add2all')` and `plotPDF(ori)` into held pole figures asked for the directions; `ind2color` ...
- findings.md:108 (port bug) OrientationRelationshipFit: `calcParent2Child` could return a relationship worse than the start refined locally: the scan scores its ...
- findings.md:109 (port bug) TransformationTexture: `variants(p2c, odf)`, MATLAB's `transformODF`, was missing
- findings.md:110 (port bug) MaParentGrainReconstruction: `smooth(ebsd, F, fill=grains)` took the grains as a plain flag; `gB.selectByGrainId(pairs)` had no pair ...
- findings.md:111 (port bug) LowLevelParentGrainReconstruction: `calcParent(child, parent, p2c)` and the pair and triple forms returned variant ids where MATLAB returns ...
- findings.md:112 (port bug) LowLevelParentGrainReconstruction: `majorityVote`, `g1.neighbors(g2)`, `tP('Ti (alpha)', ...)`, `grains.cs2phaseId`, `ebsd.name2id` and an ...
- findings.md:113 (port bug) MartensiteVariants: `calcGrains(ebsd, minPixel=2)` on the emsland map leaves 32462 austenite measurements in the map where ...
- findings.md:114 (port bug) MaParentGrainReconstructionAdvanced: `calcGrains(..., minPixel=5, removeQuadruplePoints=True)` on the martensite map gives 4467 grains and ...
- findings.md:115 (port bug) EBSDReferenceFrame: the package-level `rotate(ebsd, r, 'keepEuler')` passed the flag by position, so it landed on `keepXY` and ...
- findings.md:116 (port bug) VectorsOperations, RotationOperations, RotationDefinition: `arrow3d` in fresh axes drew a tilted perspective with its labels cut, where MATLAB shows the top view
- findings.md:117 (port bug) TensorArithmetics: the stress of `C * eps` carried the density of the stiffness
- findings.md:118 (port bug) TensorDefinition: a complete chart was labelled with the crystal axes where MATLAB labels the sector corners in all their ...
- findings.md:119 (port bug) TensorVisualisation: the `defaultColorMap` preference was set by pages but never read
- findings.md:120 (port bug) TensorAverage: the needles of an axis field on the equator stepped over the rim, the antipodal chart put the step's end ...
- findings.md:121 (port bug) MagneticAnisotropy, PiezoElectricity: `plotPDF` and `plot` of a function dropped MATLAB's positional flags (`'antipodal', 'contourf'`); ...
- findings.md:122 (port bug) PiezoElectricity: the rim labels of a trigonal chart are the a directions `[-12-10]`, four of them, where MATLAB writes ...
- findings.md:123 (port bug) TensorAverage: the phase legend of a map lists the phases present where MATLAB lists all four of the epidote map's
- findings.md:124 (port bug) S2FunQuadrature: `S2FunHarmonic.quadrature(S2G, values)` expanded to degree 250 instead of the bandwidth of the grid, a ...
- findings.md:125 (port bug) S2FunQuadrature: a series of low degree on a few thousand directions went through the NUFFT, exact only to 1e-9 where ...
- findings.md:126 (port bug) S2FunOperations: `sqrt` of a plain number raised
- findings.md:127 (port bug) S2FunPlotting: `plotSection(sF, N)` drew a polar axes with its grid, clipping negative values; MATLAB draws the ring ...
- findings.md:128 (port bug) S2FunPlotting: `nextAxis(i, j)`, `setCamera(how2plot)`, `FontSize` of `plotSpektra`, the axes arrows of `plot3d(sF)` and ...
- findings.md:129 (port bug) S2FunApproximationInterpolation, EBSDIPFMap: a marker size per marker was read as a diameter where MATLAB's scatter reads it as an area, and the ...
- findings.md:130 (port bug) S2FunApproximationInterpolation: the scatter of many directions kept the full marker size; MATLAB caps it at 50 times the spacing
- findings.md:131 (port bug) S2FunSampling, S2Kernels: `legend` and `hold` after `plt.figure()`/`plt.subplot` went to the last MTEX figure
- findings.md:132 (port bug) S2FunSym: `specimenFrame('222', pC)` dropped the convention
- findings.md:133 (port bug) S2Kernels: `arrow3d` into a held surface cropped it by its zoom; `arrowWidth` was absolute where MATLAB's is relative ...
- findings.md:134 (port bug) S2Bingham: `S2FunBingham.fit` returned the axes with the mode first and `Z` descending, `a` was a matrix
- findings.md:135 (port bug) S2Bingham: an unused cell of a `newMtexFigure` grid showed empty axes with ticks; a curve on an antipodal chart was ...
- findings.md:136 (port bug) IsotropicTheory: the chart labels of a frame without sector vertices were projected unnormalised, the triclinic [001] at ...
- findings.md:137 (port bug) AnisotropicTheory: `inv` kept the unit and a product carried the left factor's
- findings.md:138 (port bug) WaveVelocities: `_arrows` scaled the needles by the largest on-screen length, so almost radial axes drew long, in every ...
- findings.md:139 (port bug) WaveVelocities: `velocity(C, harmonic=True)` was accepted and ignored
- findings.md:140 (port bug) CPOSeismicProperties: `text` after `nextAxis` in a held figure wrote into every panel
- findings.md:141 (port bug) SchmidFactor: `SchmidFactor(r)` of many directions was transposed against MATLAB's matrix; the Schmid factor function ...
- findings.md:142 (port bug) SchmidFactor, TensorDefinition: a complete chart labelled every equivalent axis whatever was drawn; MATLAB labels the equivalent sector ...
- findings.md:143 (port bug) SlipTransmission: `mPrime` and `residualBurgersVector` formed all pairs where MATLAB goes entry by entry
- findings.md:144 (port bug) SchmidFactor: the grain quiver used a fixed in-plane length where MATLAB draws 0.2 times the diameter along the 3D ...
- findings.md:145 (port bug) TaylorModel: `calcTaylor` over orientation space took 1278 s, a Python simplex per node
- findings.md:146 (port bug) TaylorHex: `optimalSample` left the sample unmoved for a specimen group with a two-fold about z, the left symmetry ...
- findings.md:147 (port bug) TaylorHex: `plotPDF` of an ODF shared one colour range over the row where MATLAB gives each pole figure its own
- findings.md:148 (port bug) VPSCImport: `setColorRange` redrew the contour levels over the common range, emptying the weak panels
- findings.md:149 (port bug) SingleSlipModel: `sS.deformationTensor()` of several systems was one rank three tensor
- findings.md:150 (port bug) TaylorModel: map `quiver` ignored `displayName` and drew black where MATLAB takes the colour order
- findings.md:151 (port bug) TextureEvolution: `plotSection` read a style flag such as `'contourf'` as a section family
- findings.md:152 (port bug) S1FunHarmonics: the coefficients of `S1FunHarmonic` were taken for e^{ikx}, so `S1FunHarmonic([-1i, 0.5, 4, 0.5, 1i])` was ...
- findings.md:153 (port bug) S1FunHarmonics: `S1FunHarmonic.interpolate` was the quadrature, right for equispaced angles only; `smooth`, `sum`, ...
- findings.md:154 (port bug) S2FunVectorField: `norm(vF)` was a number where MATLAB's is the function of the lengths; `quiver3` put arrows on a mesh far ...
- findings.md:155 (port bug) S2FunAxisField: an axis field was drawn on one hemisphere, its argument taken for an axis
- findings.md:156 (port bug) S2FunVectorValued: `interpolate` of several value columns and `plot` of a list of functions failed
- findings.md:157 (port bug) HarmonicApproximationTheory: `setColorRange('tight')` was refused; a section plot with `newMtexFigure(layout=...)` and `nextAxis` ...
- findings.md:158 (port bug) SO3FunApproximationTheory: the harmonic fit did not merge nodes within 0.02 as MATLAB does, so 1e5 random nodes gave bandwidth 41 for ...
- findings.md:159 (port bug) SO3FunApproximationTheory: `min` of a density fit was -0.0064: the extrema of a sum of many kernels were valued by its harmonic form
- findings.md:160 (port bug) SO3FunVectorField: `norm(W)` of an SO(3) vector field was its L2 norm, a number; `dot(W, Miller(1,0,0,cs))` of a left field ...
- findings.md:161 (port bug) SO3FunVectorField: `quiver3` of an SO(3) vector field failed; `SO3VectorFieldHarmonic` of a list of three `SO3FunHarmonic` ...
- findings.md:162 (port bug) RBFApproximationTheory: the fits ran SciPy's LSQR, which also stops on its condition estimate and on a residual test widened by ...
- findings.md:163 (port bug) BoundarySelect: the notIndexed pixels inside a grain kept phase 0 after `calcGrains`, MATLAB giving them the grain's phase ...
- findings.md:164 (port bug) BoundaryPlots: `translucent` on grains was a constant alpha; ±∞ values drew nothing; the boundary quiver drew centred ...
- findings.md:165 (port bug) BoundaryMisorientations, SubGrainBoundaries: grain outlines always black; a fresh `newMtexFigure(figSize=)` square around a map; the `edgeAlpha` of ...
- findings.md:166 (port bug) TriplePoints: the triple points of a grain or boundary selection were computed from the selection alone and empty; a ...
- findings.md:167 (port bug) BoundaryCurvature: `mtexColorMap`, `setColorRange` and the colour bar reached only the last value-coloured plot of a map
- findings.md:168 (port bug) BoundaryIntersections: grains of one `faceColor` drew a phase legend
- findings.md:169 (port bug) TwinningBoundaries, TiltAndTwistBoundaries, GrainSmoothing: `xlabel` wrote over the tick labels of a plain axes; `scatter(mori)` framed a ±100 degree cube; ...
- findings.md:170 (port bug) BoundarySelect: Euler angle displays round to three significant digits (25.3, 260) where MATLAB shows six (239.134); the ...
- findings.md:171 (port bug) BoundaryNormalDistribution: the tension twin GBCD peaks 4.8 degrees from the (10-12) plane where MATLAB's text says 1.3 and Julia ...
- findings.md:172 (port bug) TriplePoints, BoundaryCurvature, CSLBoundaries: original reports: 78 triple points for 70, 924 grains for 885 on csl, a titanium lobe 10.55 degrees across ...
- findings.md:173 (port bug) GrainReconstruction: after the absorption, `calcGrains` on its own result made every absorbed pixel (a phase, no rotation) a ...
- findings.md:174 (port bug) QuasiCrystals: the sector of an icosahedral group generated from the rounded axis angle 37.377 degrees, which closes only ...
- findings.md:175 (port bug) RotationFibre: a rounded direction printed a negative zero, `(-0,0,1)`
- findings.md:176 (port bug) EBSDSharpPlot, EBSDGrid: the shares of the phase table were printed to three digits (99.8%) where MATLAB writes two (100%)
- findings.md:177 (port bug) GrainMerge: 20 triple points for MATLAB's 15 after the merge; the other pages now agree (SelectingGrains 6, 133, 43, ...
- findings.md:178 (port bug) AxisDistributionFunction: the bandwidth hint of the ADF moved from 512 to 250 where MATLAB's is 500
- findings.md:179 (port bug) TriplePoints, ShapeParameters: after the display rework the headers of `triplePointList` and `shape2d` still named the specimen frame ...
- findings.md:185 (port gap) EBSDGrid: `gridLayout` with `'rowMajor'` and `'columnMajor'`, `ebsd.layout`, `oldId` and the second output of ...
- findings.md:186 (port gap) EBSDFilling: `fill(ebsd, grains)` constrained by the grain polygons: a site outside every grain stays notIndexed, a ...
- findings.md:187 (port gap) EBSDFilling: the `quality` property `smooth` writes, zero where no orientation was measured
- findings.md:188 (port gap) VectorsOperations: `arrow3d`, the arrows in space of the vector sum; S5
- findings.md:189 (port gap) VectorsOperations: the dynamic marker size: MTEX shrinks the markers of a long list, the port draws 764 directions at the ...
- findings.md:190 (port gap) VectorsAxes: `plot(pf)` of a measured pole figure and `annotate` on it; S6
- findings.md:191 (port gap) VectorsImport, VectorsExport: `vector3d.load`, `loadVector3d`, `export(v, fname)` did not exist
- findings.md:192 (port gap) VectorGrids: `HEALPixS2Grid` did not exist
- findings.md:193 (port gap) EBSDImport: a phase list given with a vendor file was refused; an import script states its own frames
- findings.md:194 (port gap) EBSDImport: the data sets of an HDF5 file were not listed on import, `silent` did not exist
- findings.md:195 (port gap) EBSDImport, EBSDPlotting: the display and the maps gave the phases no colour where MATLAB assigns LightSkyBlue, DarkSeaGreen, ...
- findings.md:196 (port gap) EBSDReferenceFrame, EBSDPlotting: `ebsd.EulerCorrection`, `ebsd['phase'].color` read and set, `str2rgb`, `plot(key)`, `rotate(ebsd, r, ...
- findings.md:197 (port gap) EBSDGROD: `HSVDirectionKey()` of the whole sphere and `HSVDirectionKey(cs, antipodal=True)`, `plot(key)` for a ...
- findings.md:198 (port gap) EBSDAdvancedMaps: `plot(key)` and `plot(key, 'sigma', sections=6)` for the Bunge and spot keys, the coloured sections of ...
- findings.md:199 (port gap) EBSDAdvancedMaps: `PatalaColorKey`
- findings.md:200 (port gap) EBSDIPFMap: `plot(ipfKey, 'complete', 'upper')` draws the whole hemisphere of a key; the port's key plot draws the ...
- findings.md:201 (port gap) EBSDIPFMap: `plot(ipfKey, '3d')`, the coloured sphere of a key, and `plot(grains, ipfKey)` drawing that sphere on ...
- findings.md:202 (port gap) EBSDExport: the Oxford binary pair `.crc`/`.cpr` on export, MATLAB's `.mat` working copy
- findings.md:203 (port gap) EBSDInterfaceHDF5: `EBSD.load(fname, 'headerOnly')` and the `'debug'` trace of the HDF5 reader
- findings.md:204 (port gap) EBSDOrientationAnalysis, EBSDIndex: `circle(v, radius)`, `nextAxis(k)` choosing an axes, MATLAB's marker names (`'square'`), `plot(odf, f)` ...
- findings.md:205 (port gap) VectorGrids, VectorsDensityEstimation: `mtexTitle`, `nextAxis`, the `layout` option of a plot, a plot into a fresh `newMtexFigure`, ...
- findings.md:206 (port gap) GrainSpatialPlots: `colorcet('C2')` and the other CET maps; matplotlib's cyclic `twilight` stands in, the hues differ
- findings.md:207 (port gap) GrainReconstruction: fast multiscale clustering (`'fmc'`), the Markovian clustering and the Delaunay decomposition of ...
- findings.md:208 (port gap) SelectingGrains: `selectInteractive`, the mouse selection of grains
- findings.md:209 (port gap) HullBasedParameters: MATLAB's stored `testgrains` with its forty shapes; the port draws fifteen of its own
- findings.md:210 (port gap) ODFTheory, ODFPlot, FibreODFs: `plot3d(odf)`, the isosurfaces of an ODF in Euler or axis angle space; S5
- findings.md:211 (port gap) ODFTheory: `plot(ebsd, cS)` drawing the crystal shape at every measurement, MATLAB's named figure sizes ...
- findings.md:212 (port gap) ODFPlot: `fundamentalRegionEuler(cs, ss)`, `ori.project2EulerFR()`
- findings.md:213 (port gap) ODFInversePoleFigure: `calcPDF(odf, None, r)`, the inverse pole density for MATLAB's empty crystal direction
- findings.md:214 (port gap) EulerAngleSections: `crystalFrame.load('Al-Aluminum.cif')` (`io/cif.py`, the MTEX CIF files shipped in `geometry/data/cif`, a ...
- findings.md:215 (port gap) SigmaSections: `plot(oS)` with the reference fields, `quiver(ori, v)` in sections, `plotSection(ori, ori * cS, oS)` ...
- findings.md:216 (port gap) SigmaSections: `oS.referenceField = S2VectorField.polar(xvector)`, replacing the reference field of a section object; the ...
- findings.md:217 (port gap) ODFCharacteristics: `plotPDF(odf, pf.allH, superposition=pf.c)` with one superposed pole figure per entry, `annotate(ori)` and ...
- findings.md:218 (port gap) ODFShapes: `plot(psi)` of an SO(3) kernel over -180 to 180 degree, `plot(psi, symmetric=True)` of a spherical one, ...
- findings.md:219 (port gap) BinghamODFs: `BinghamODF(20, ori)`, `BinghamODF(20, f)`, `BinghamODF([10, 10, 10], quaternion.eye, cs)` with the ...
- findings.md:220 (port gap) ODFExport, ODFImport: `export(odf, fname, 'Bunge', resolution=)`, `export(odf, fname, S3G, 'generic')`, `interface='mtex'`, ...
- findings.md:221 (port gap) ODFImport: the import wizard
- findings.md:222 (port gap) PoleFigureSantaFe: `SantaFe`, the benchmark model ODF, did not exist
- findings.md:223 (port gap) PoleFigureExport: `export(pf, fname, 'degree')`, one table of polar angle, azimuth and intensity per pole figure, did not exist
- findings.md:224 (port gap) PoleFigureRefinement: `v.resolution` of a plain list of directions did not exist
- findings.md:225 (port gap) PoleFigurePlot: the `parula` colour map was not among MTEX's
- findings.md:226 (port gap) PoleFigureSantaFe: `orientation.byMiller(hkl, uvw, cs, ss)` read the crystal frame given again as the specimen frame
- findings.md:227 (port gap) PoleFigure2ODFGhostCorrection: `plot(odf, f)` refused `linewidth`, matplotlib's spelling, beside `lineWidth`
- findings.md:228 (port gap) PoleFigureCorrection: `cat(pf1, pf2)` failed, the generic `cat` passing an axis
- findings.md:229 (port gap) OrientationSymmetry: `ori * cs` and `ss * ori`, the equivalents on one side
- findings.md:230 (port gap) OrientationStandard: `legend('show', 'location', 'southoutside', 'numColumns', 3)`
- findings.md:231 (port gap) OrientationStandard: `ind2color` reads MTEX's own 24 colours
- findings.md:232 (port gap) OrientationStandard: `DisplayName` on markers and on a scatter in orientation space, `add2all` on a section plot, `'filled'`
- findings.md:233 (port gap) MaParentGrainReconstruction: `job.selectInteractive`, `job.plotVariantPF`, `calcGrainVotes`, the `includeTwins` option of ...
- findings.md:234 (port gap) S2FunApproximationInterpolation: `interp(v, y, 'linear' / 'harmonic' / 'nearest' / 'inverseDistance')`, `S2FunTri` and `S2Triangulation`
- findings.md:235 (port gap) S2FunSampling: `S2FunHarmonic.adjointNFSFT`, `sum` of a function, `norm` of a lazy sum
- findings.md:236 (port gap) S2Bingham: `ellipse(rot, a, b)`
- findings.md:237 (port gap) S2FunRadon: `radon(sF, delta)`, the means over small circles
- findings.md:238 (port gap) BoundaryPlots, EBSDAdvancedMaps: `PatalaColorKey`
- findings.md:239 (port gap) BoundaryNormalDistribution: `calcGBND`, `calcGBNDKernel` (`nonneg`, `bump`, the GBCD)
- findings.md:240 (port gap) BoundaryCurvature: `testgrains.mat` is a MATLAB grain object
- findings.md:241 (port gap) BoundaryIntersections, BoundarySelect, CSLBoundaries: MATLAB's `[x, y] = gB.intersect(xy1, xy2)`, `line` and `scatter(x, y)` on maps, `legend('off')`, ...
- findings.md:249 (convention) VectorDefinition: `vector3d(M)` with a 3 x 3 matrix warns and reads by columns
- findings.md:250 (convention) VectorDefinition: `size(v)` is `1 5` for five vectors, `v(2)` the second
- findings.md:251 (convention) VectorDefinition: the grid of a spherical plot follows the frame's Z axis, so under `'z←↑y'` the meridians run between the ...
- findings.md:252 (convention) VectorDefinition: `plottingConvention.default('y↑→x')` sets the session default
- findings.md:253 (convention) VectorsExport: `v.rho` remembers the azimuth a vector was constructed with, so `export(v, f, 'polar')` writes -80
- findings.md:254 (convention) VectorsImport: `[v, S] = vector3d.load(...)` gives the other columns on request
- findings.md:255 (convention) VectorGrids: `norm(sF)` integrates against the surface measure, the constant one having norm sqrt(4 pi)
- findings.md:256 (convention) VectorsDensityEstimation: `ebsd('Fo')` selects a phase by an abbreviation of its mineral
- findings.md:257 (convention) VectorsDensityEstimation: `[grains, ebsd] = calcGrains(...)`
- findings.md:258 (convention) VectorsDensityEstimation: `max(sF)` with one output is the value
- findings.md:259 (convention) EBSDGrid: an empty site of a gridded subset has phaseId NaN
- findings.md:260 (convention) EBSDGrid: the eclogite file warns that 92 of 613 measurements share a lattice cell
- findings.md:261 (convention) EBSDFilling: `fill` leaves the measured properties NaN at a filled site
- findings.md:262 (convention) EBSDFilling: `length(ebsd)` is the number of measurements
- findings.md:263 (convention) EBSDFilling: a gridded hexagonal map counts the measurements of the file
- findings.md:265 (convention) EBSDSharpPlot: the padding of a gridded map has the phase NaN, `isnan(ebsd.phaseId)` finds it
- findings.md:266 (convention) EBSDSharpPlot: `h.rho` in (-180, 180] degrees
- findings.md:268 (convention) EBSDIPFMap: `'MarkerSize', 0.05*area` scales the marker area
- findings.md:269 (convention) EBSDIPFMap: `ebsd('olivine').CS = cs` assigns the frame of the variable
- findings.md:270 (convention) EBSDDenoising: `grains(ebsd.grainId)` indexes the grains by id, ids being positions
- findings.md:271 (convention) EBSDDenoising: `splineFilter` selects its parameter by cross validation, 4.61 on the twins map, and `[ebsdS, F] = ...
- findings.md:272 (convention) EBSDDenoising: `axisAngleColorKey` without an argument
- findings.md:273 (convention) EBSDIndex: a crop of a gridded map is ordered column-major, `ebsd(16:18)` one-based
- findings.md:274 (convention) EBSDPlotting: a colour assigned to a phase belongs to the EBSD variable
- findings.md:275 (convention) EBSDSelect: the default colour map of a value map is parula
- findings.md:276 (convention) EBSDImport: `'dataSet', 1` names the first data set of an HDF5 file
- findings.md:277 (convention) EBSDImport: `ebsd.opt.header`, `ebsd.opt.dataSet`
- findings.md:278 (convention) EBSDImport: the import wizard
- findings.md:279 (convention) EBSDPlotting: `colormap gray`, `legend off`
- findings.md:280 (convention) VectorsDensityEstimation: `volume(sF, v, 20*degree)` is 0.1723 on the 0.1 degree grid about the centre
- findings.md:282 (convention) GrainReconstruction: `calcGrains` writes the grain ids into the map it is given and rewrites the phase of the measurements ...
- findings.md:283 (convention) GrainNeighbours: `numNeighbors` is a property, the shape parameters MATLAB implements as methods (`aspectRatio`, ...
- findings.md:284 (convention) GrainMerge: `merge` returns the merged grains and writes `parentId` into the grains it was given; `accumarray` over ...
- findings.md:285 (convention) GrainOrientationParameters: `fibre.fit(ori, 'local', diagnostics=True)` returns MATLAB's triple, the eigenvalues `lam[0]` to `lam[3]`; ...
- findings.md:286 (convention) RotationDefinition: `Euler(rot, 'Roe')` without an output prints a table in degrees
- findings.md:287 (convention) RotationOperations: `rot \ v` applies the inverse rotation; `.*` pairs entries and `*` forms every combination
- findings.md:288 (convention) RotationImproper: `rotation(cs)` lists the operations of a mixed group with their inversion flags, `isImproper(rot)` is a method
- findings.md:289 (convention) RotationSpinTensor: `log(r, ref, SO3TangentSpace.rightSpinTensor)` names four representations
- findings.md:290 (convention) RotationRepresentations: there is no `rotation.byCubochoric`; the page goes through `cubo2homo`
- findings.md:291 (convention) ODFTheory: the display of an ODF names the specimen frame by its convention, `y↑→x (mmm)`
- findings.md:292 (convention) ODFModeling: `max(odf1)` with one output is the value
- findings.md:293 (convention) ODFPoleFigure: `Miller({1,0,-1,0},{0,0,0,1},{1,1,-2,1},cs)`, one cell per direction
- findings.md:294 (convention) EulerAngleSections: `odfOrtho = odf; odfOrtho.SS = ...` leaves `odf` alone
- findings.md:295 (convention) EulerAngleSections: `fibre.beta(cs)` has `h
- findings.md:296 (convention) ODFComponents: `[ori, w] = calcComponents(odf)` takes two of three outputs
- findings.md:297 (convention) FibreODFs: `[f, lambda, d] = fibre.fit(ori, 'local')`
- findings.md:300 (convention) ODFExport: `save`, `load` of a `.mat` working copy
- findings.md:301 (convention) DetectionOfSampleSymmetry: `centerSpecimen` prints its search and takes `'Fourier'`
- findings.md:302 (convention) CrystalSymmetries: `rotation(cs)` lists the operations of any group, `cs.Laue`, `cs.properGroup` are properties
- findings.md:303 (convention) CrystalSymmetries: the operation tables list the half turns as (0, 180, 0)
- findings.md:304 (convention) CrystalOperations: `vector3d(m)` drops the crystal frame
- findings.md:305 (convention) CrystalShapes: `cS.V` prints Cartesian components; `crystalShape.olivine` is a property; `{1,0,-1,0}` is an index cell
- findings.md:306 (convention) QuasiCrystals: a group built from elements reports `unknown`
- findings.md:310 (convention) PoleFigure2ODF: the solver's grid holds 19848 orientations at 5 degrees on the Dubna set
- findings.md:311 (convention) OrientationDefinition: `ori.symmetrise` lists the 48 x 8 products of the full point groups, `'proper'` cuts them to 24 x 4
- findings.md:312 (convention) OrientationDefinition: the display names the specimen frame by its plotting convention, `orientation (Copper → y↑→x)`
- findings.md:313 (convention) OrientationSymmetry: the axis at the centre of a chart is labelled, `Z` in the middle of an upper hemisphere
- findings.md:314 (convention) OrientationStandard: the contour levels of a pole figure row are taken per panel, so every panel reaches the top of the colour map
- findings.md:315 (convention) OrientationStandard: a plot onto held axes takes the next colour of the axes' colour order
- findings.md:316 (convention) OrientationFundamentalRegion: the grain exchange half of a same-phase region is the y >= 0 half of the box
- findings.md:320 (convention) OrientationVisualization3d: orientations in space are hollow circles
- findings.md:321 (convention) MisorientationTheory, AxisDistributionFunction: grains are numbered in pixel order along y first, so the twins' grains 57 and 58 are neighbours, and the ...
- findings.md:324 (convention) MaParentGrainReconstructionAdvanced: `ebsd(ind).orientations = ori`, `grains(ind).meanOrientation = ori`
- findings.md:325 (convention) MaParentGrainReconstructionAdvanced: `[grains, parentId] = merge(grains, ...)`
- findings.md:328 (convention) ParentChildVariants: the Burgers relationship prints as `(110)
- findings.md:332 (convention) TensorDefinition: `propertyname`
- findings.md:333 (convention) TensorVisualisation: blue2red
- findings.md:334 (convention) TensorVisualisation: `[V, D] = eig(T)`
- findings.md:335 (convention) SphericalHarmonics, SO3FunHarmonicRepresentation: the coefficients `fhat` are in MTEX's basis, no Condon-Shortley phase, `Y_l^{-m} = conj(Y_l^m)`
- findings.md:338 (convention) S2FunHarmonicRepresentation, S2FunOperations: a callable is expanded to degree 250
- findings.md:340 (convention) S2FunSampling: `[v, c] = optimalSample(...)`
- findings.md:346 (convention) IsotropicTheory, CPOSeismicProperties: the several outputs of `mean` and `calcTensor`
- findings.md:347 (convention) SchmidFactor, DislocationSystems: the outer `ori * sS`
- findings.md:348 (convention) TwinningTutorial, SachsModel: the Schmid factor of one slip system is a number
- findings.md:349 (convention) Lankford: `[grains, ebsd.grainId] = calcGrains(ebsd)`
- findings.md:350 (convention) S1FunHarmonics: the periodic density is cut at round(4/sigma)
- findings.md:354 (convention) RBFApproximationTheory: the solvers `'mlrl'`, `'lsqnonneg'`, `'lsqlin'`, `'nnls'` of `SO3FunRBF.interpolate`
- findings.md:355 (convention) RBFApproximationTheory: `SO3FunRBF.eval` sums the kernels through the sparse kernel matrix at any number of points: the tight ...
- findings.md:358 (convention) GrainSmoothing: the zoom `axis([x0 x1 y0 y1])` after a map plot
- findings.md:359 (convention) BoundaryNormalDistribution: the GBND a function on the sphere
- findings.md:360 (convention) QuadruplePoints, BoundaryProperties: the grain ids and the sides of a segment
- findings.md:361 (convention) LowLevelParentGrainReconstruction, MaParentGrainReconstructionAdvanced and the other pages taking quantiles: `quantile(x, p)` skips NaN, so the absorbed pixels without a rotation drop out
- findings.md:362 (convention) every page: an object prints as `x = class (frames)`, a blank line, the fields, a blank line, the data; a size `1 × ...
- findings.md:363 (convention) PoleFigureDubna, PoleFigurePlot, PoleFigureImport, PoleFigureExport, ...: a pole figure is headed by the specimen convention alone, `PoleFigure (y↑→x)`
- findings.md:364 (convention) WaveVelocities: the wave velocity functions are headed by the specimen convention
- findings.md:365 (convention) RotationTangentSpace, RotationSpinTensor, SO3FunVectorField: `SO3TangentVector (crystal)`
- findings.md:366 (convention) TaylorModel, TextureEvolution, DislocationSystems: a crystal frame without a mineral is written `crystal`
- findings.md:373 (practicality) VectorDefinition: a MATLAB cell shows `v = vector3d (...)` after `v = vector3d(1,2,3)` without a semicolon; the Python page ...
- findings.md:374 (practicality) VectorDefinition: the builder displays a trailing number as MATLAB's format short (`3.7417`, `45.0000`), an integer-valued ...
- findings.md:375 (practicality) VectorDefinition: the random directions of `vector3d.rand` differ from MATLAB's draw; only the layout of the figure is compared
- findings.md:376 (practicality) VectorsExport: a MATLAB cell of several statements is one cell in the page too: the statements follow without a blank ...
- findings.md:377 (practicality) EBSDIndex: a figure lives on across the cells of a page, and `hold` in a later cell draws into it; the builders ...
- findings.md:378 (practicality) EBSDOrientationAnalysis: `islocalmax(x, 'MinProminence', 1)` is `scipy.signal.find_peaks(x, prominence=1)`; `orientation(f, CS, ...
- findings.md:379 (practicality) EBSDIndex: `hex2cube`, `cube2hex` are not ported; `rectangle` and `histogram` are matplotlib's `Rectangle` and `hist`
- findings.md:380 (practicality) VectorsDensityEstimation: the colour bar of a `contourf` plot is ticked at the contour levels with three decimals where MATLAB's ...
- findings.md:381 (practicality) VectorsDensityEstimation: `xnum2str` and `mtexdegchar` are an f-string; `[grains, ebsd] = calcGrains` is `grains = calcGrains`; ...
- findings.md:382 (practicality) Grains chapter: MATLAB's `table` displays are printed lines, `histogram`/`polarplot`/`scatter`/`subplot` are matplotlib's, ...
- findings.md:383 (practicality) RotationRepresentations: a figure built over several paragraphs (a `plt.figure`, then its subplots) is one cell only between `# +` ...
- findings.md:384 (practicality) RotationPlotting: `docs/matlabrun.py` saves a persisting MATLAB figure again after every cell that redraws it, so a page's ...
- findings.md:385 (practicality) RotationPlotting: the count of the random subset is a draw: 26 here, 32 in the MATLAB run, 20 on the MATLAB page
- findings.md:386 (practicality) ODFTheory: MATLAB's `mtexdata titanium silent` is `ebsd = mtexdata('titanium')`, which prints nothing unless displayed
- findings.md:387 (practicality) SigmaSections: MATLAB's local function at the end of the script is defined at the top of the page, before its first call
- findings.md:388 (practicality) ODFShapes: MATLAB's `figure('position', [...])` is `plt.figure(figsize=..., layout='constrained')`, so an outside ...
- findings.md:389 (practicality) RandomSampling: `fminbnd(err, 2.5, 80, optimset('TolX', 0.1))` is `minimize_scalar(err, bounds=(2.5, 80), ...
- findings.md:390 (practicality) ODFComponents: the page's `calcODF` of the Dubna pole figures takes about 30 s, its `max`, `volume` and `calcComponents` ...
- findings.md:392 (practicality) SymmetryAlignment: the tensor file of MATLAB's data folder is the data set `olivine1997` here, and Bruker's `crystal.phl` ...
- findings.md:393 (practicality) PoleFigureSimulation, PoleFigureSantaFe, PoleFigureCorrection: the Poisson draw of `noisepf` and the spoiled measurements of the outlier example are not seeded, as ...
- findings.md:394 (practicality) PoleFigure2ODF: `calcODF` is silent unless `silent=False`, where MATLAB prints its iteration trace unless `'silent'` is given
- findings.md:396 (practicality) OrientationInversePoleFigure, OrientationPoleFigure, OrientationExport: the pages draw random orientations; NumPy's generator cannot reproduce MATLAB's `rng(635)`, so their ...
- findings.md:401 (practicality) EBSDInter, EBSDGrid: `ebsd.dPos` was the lattice spacing; it is now MATLAB's shortest edge of the unit cell, the same on a ...
- findings.md:402 (practicality) EBSDInter: a map resampled on a lattice turned by 45 degrees is drawn as polygons, and matplotlib's antialiasing ...
- findings.md:403 (practicality) EBSDPseudoSymmetry: `calcGrains(ebsd['indexed'], angle=10*degree)` on the forsterite map gives 3176 grains for MATLAB's 3165 ...
- findings.md:404 (practicality) EBSDPseudoSymmetry: a map zoomed by `xlim` and `ylim` lost its scale bar, which stayed at the corner of the whole map; the bar ...
- findings.md:406 (practicality) EBSDSpatialTransform: `spatialTransformField` interpolates linearly over the Delaunay triangulation and takes the nearest node ...
- findings.md:407 (practicality) EBSDGrid: `calcGrains(mtexdata('small'), minPixel=3)` leaves a one-pixel enstatite grain after the cull on the plain ...
- findings.md:408 (practicality) EBSDMapsAndImages: `rotate(ebsd, 90*degree)` keeps the 96 × 128 array of the gridded map and turns its layout with the ...
- findings.md:409 (practicality) EBSDMapsAndImages, EBSDTrueEbsd: a sequence of images is a `mapImageList`, a Python list displaying as MATLAB's table of a `mapImage` ...
- findings.md:410 (practicality) EBSDTrueEbsd: the whole run is MATLAB's to the digit: the measured settings, the retry of the first hop, every tile's ...
- findings.md:411 (practicality) EBSDTrueEbsd: MATLAB draws the NaN padding the correction leaves at the edges of the undistorted images black, the first ...
- findings.md:412 (practicality) EBSDTutorial: `grains('Forsterite', grains.numPixel > 100)` had no port form: `grains['Forsterite', mask]` failed ...
- findings.md:413 (practicality) GrainTutorial: `ebsd[grains]` returned the padding cells the alpha closing had given a grain id (803 cells for 451 ...
- findings.md:414 (practicality) GrainTutorial: the phase shares in the EBSD display were divided by all grid cells, padding included (mylonite 11% for ...
- findings.md:415 (practicality) GrainTutorial: `calcGrains(angle=15°)` on the mylonite region gives 373 andesina grains for MATLAB's 371 and 4647 ...
- findings.md:416 (practicality) BoundaryTutorial: `smoothBoundary()` on the twins grains leaves 3363 segments for MATLAB's 3359; open, M4
- findings.md:417 (practicality) PoleFigureTutorial: `calcODF` on the ZnCuTi 6/mmm pole figures places 9978 kernel centres for MATLAB's 9924 (uniform weight ...
- findings.md:418 (practicality) PoleFigureTutorial: `pf(pf.intensities < 0) = 0` had no port form; fixed 2026-09-26 (`PoleFigure.__setitem__` with a mask)
- findings.md:419 (practicality) ODFTutorial: `calcODF(mtexdata('ptx'))` peaks at 17.21 mrd for MATLAB's 17.72: the equispaced mmm grid has 29818 nodes ...
- findings.md:424 (practicality) DensityEstimation: `calcDensity(x)` of plain numbers is now MATLAB's real-valued estimate. The circle needs `periodic=True` ...
- findings.md:425 (practicality) DensityEstimation: Section plots counted the NaN orientations `calcGrains` leaves on the absorbed pixels ("out of 181146"); ...
- findings.md:428 (practicality) MTEXScripts: `calcGrains(ebsd, minPixel=3)` on forsterite gives 1014 grains (541/228/232) for MATLAB's 950 ...
- findings.md:429 (practicality) GeneralConceptsConfiguration: The port has no startup settings file (MATLAB's `mtex_settings.m`), and `getMTEXpref()` lists only the ...
- findings.md:431 (practicality) PlotTypes: SO3FunCBF / fibreODF for a non-cubic Laue group (-3m) left out -h: 30.6 at one end of the fibre and 0 at ...
- findings.md:432 (practicality) SphericalProjections: plotPDF(odf, h, antipodal=True) of a polar crystal (321) showed only the upper hemisphere (max 30.6), not ...
- findings.md:433 (practicality) CombinedPlots: plotDiff defaulted to RP where MATLAB defaults to the relative error, and refused plotDiff(odf, pf). ...
- findings.md:434 (practicality) ContourPlots: blue2red was a 3-stop map; MATLAB's runs dark blue, blue, white, red, dark red. Fixed; pages using ...
- findings.md:435 (practicality) PlotTypes: equispacedS2Grid defaulted to 5°, MATLAB to 2.5°. Fixed (no caller relied on the default)
- findings.md:436 (practicality) ColorMaps: contourf with explicit levels spread the colours over the data range instead of the levels; the band below ...
- findings.md:437 (practicality) CombinedPlots: symmetrised labelled Millers on an antipodal inverse pole figure were labelled by the antipode ((000-1)), ...
- findings.md:438 (practicality) TransparencyDemo, CombinedPlots, Annotations: markers had black edges where MATLAB edges them in the face colour; orientation scatters were filled where ...
- findings.md:439 (practicality) AxesAlignment: frame-free directions were labelled X/Y/Z after specimenFrame.rolling.makeDefault(); refFrame='off' (and ...
- findings.md:440 (practicality) AxesAlignment, Multiplot: nextAxis followed by a two-hemisphere plot opened a new figure; nextAxis(i, j) left axes with ticks; a ...
- findings.md:441 (practicality) Legends, Annotations: a legend was drawn under the chart labels; a second legend added to the first. Fixed
- findings.md:443 (practicality) ContourPlots, AxesAlignment: calcODF of the Dubna data has 20040 centres vs MATLAB's 19848
- findings.md:444 (practicality) SphericalProjections: minmax of the lower hemisphere: 4.18 vs MATLAB 4.3; the 3D sphere grid behind the surface shows through ...
- findings.md:445 (practicality) PlotTypes: plotPDF(..., resolution=10°) is drawn coarse where MATLAB's figure looks unaffected; plot(odf, fibre) ...
- findings.md:446 (practicality) Legends: MATLAB moves the legend with lgd.Location / lgd.Title; the page calls legend again instead
- findings.md:447 (practicality) docs/make.py: a figure after plt.close was missed when its reused number matched an old snapshot, and moving a colour ...
- findings.md:448 (practicality) GrainReconstruction: the seed order of FMC breaks ties by pixel order, so a map stored with x running fastest (the port) and ...
- findings.md:449 (practicality) GrainReconstruction: on the full map the first FMC port produced NaNs: its row normalisation took 1/Psum, which overflows for a ...
- findings.md:450 (practicality) FibreODFs: `fibre.fit` returned (h, r) or (-h, -r) depending on the sample (the sign of the eigenvectors), so the ...
- findings.md:451 (practicality) EBSD3Analysis (all pages): The bridge starts MATLAB with `-nodisplay`, which leaves it without OpenGL: every figure is printed by ...
- findings.md:455 (practicality) NeperInterface: the (1,-1,0) section: MATLAB turns it by the shortest rotation taking z to the normal, a rectangle at 45° ...
- findings.md:456 (practicality) EBSD3Plotting: a slice through z = 0 lies halfway between two voxel layers; the port's `interp` rounded half to even and ...
- findings.md:458 (practicality) EBSD3Analysis, Grains3DReconstruction: `histogram(x)` of numbers used 10 bins; MATLAB bins automatically. Fixed ('auto')
- findings.md:459 (practicality) EBSD3Analysis: the fourth and fifth largest grains (550 voxels each) come in the other order, a tie of the sort
- findings.md:460 (practicality) Grains3DBoundaries: `ebsdId` counts the voxels from 0 in the port's array order (200000, 210000) where MATLAB counts from 1 in ...
