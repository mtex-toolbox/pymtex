# calcGrains release investigation, 2026-09-28

The recorded differences have several causes. They are not all defects in closing.
This audit used Python at `b4d628f` (the subsequent `464c205` changes only compiled-kernel
dispatch) and the local frameSymmetry checkout at `238c4ac96`, MATLAB R2024b.
The MATLAB checkout also has local edits, including its CTF reader. The neutral-input
comparisons below bypass that reader. Julia comparisons use the saved September 23
partitions, not a fresh Julia run.

## Different default thresholds

Python's `calcGrains` and `gbcAngle` default to 10 degrees, as Julia does.
MATLAB's `gbcAngle` defaults to 15 degrees. Comparing calls without an explicit angle
therefore compares different reconstructions.

| Recipe | Python at 10 degrees | Python at 15 degrees | MATLAB at 15 degrees |
|---|---:|---:|---:|
| csl, default reconstruction: grains | 924 | 885 | 885 |
| forsterite, minPixel=3: grains | 1014 | 962 | 950 |
| small, alpha=5, removeQuadruplePoints: triple points | 78 | 68 | 70 |

MATLAB at 10 degrees also returns 924 csl grains, with the same measured-pixel
partition, 22,211 boundary segments and 1,517 triple points as Python.
At 15 degrees both return 21,982 segments and 1,451 triple points.
The CSL discrepancy is entirely the default threshold. The other two rows retain a
smaller difference after matching it. MATLAB at 10 degrees returns 80 triple points
for the small-map recipe, so there is still a topology difference there.

The defaults were aligned with MATLAB on 2026-10-01: 15 degrees in Python and in Julia.
Cross-port comparisons still state their angles.

## Correct the benchmark input before comparing algorithms

`bench/matlab/bench_calcgrains.m` passed one-based phase indices to an `EBSD` constructor
that, without an explicit `phaseMap`, interprets them as external phase numbers.
Forsterite's phase list includes Silicon although no Silicon measurements are present.
That missing phase triggers remapping: the reconstructed map carries the wrong phase
indices and crystal groups.

The benchmark now supplies `phaseMap=0:numel(CSList)-1` and asserts both the phase indices
and indexed mask before gridifying. Nine MATLAB map/case runs passed those assertions.
The earlier apparent Enstatite symmetry defect is invalid evidence about `calcGrains`;
the old forsterite MATLAB table must not be used as the reference.

## Current comparisons on identical exported measurements

Cases are those in `bench/calcgrains.py`: a = 10 degrees, minPixel=1, alpha=3.1;
b = 5 degrees, minPixel=5, alpha=3.1; c = 10 degrees, minPixel=1, alpha=0.
No case below removes quadruple points. Each reconstruction gets a fresh map copy.

| Map | Case | Python grains | Corrected MATLAB grains |
|---|---|---:|---:|
| csl | a | 924 | 924 |
| csl | b | 502 | 502 |
| csl | c | 929 | 929 |
| forsterite | a | 3185 | 3171 |
| forsterite | b | 976 | 965 |
| forsterite | c | 24099 | 24099 |
| emsland | a | 15021 | 15012 |
| emsland | b | 4312 | 4310 |
| emsland | c | 27288 | 27288 |

With closing disabled, all three maps have identical measured-pixel partitions, including
notIndexed measurements, after matching positions and ignoring grain numbering.
This removes the previously reported 390-pixel Enstatite mismatch.

The large Python-versus-Julia emsland discrepancy was already fixed in `fcef5d2`:
the vertex weld rounds from the origin, dropping vanishing faces between nearly cocircular
sites. Python now matches the saved Julia indexed-pixel partitions in all three cases,
and their grain, segment and triple-point counts. Default forsterite also matches Julia's
indexed partition. Its minPixel case still has 27 indexed assignments different from Julia.

Remaining Python-versus-MATLAB differences are localized around closing, absorption and
culling. At case a, the directional majority-match disagreement counts on indexed
measurements are 3 / 26 for forsterite and 2 / 16 for emsland (Python to MATLAB / reverse).
Both directions matter: a split can score zero in one direction. At case b those counts
are 85 / 116 and 570 / 591 respectively; csl has 28 / 28 despite equal grain counts.
Some originally indexed pixels are culled and absorbed in this case, so these numbers
include differences in the returned map's assignments.

The closing rules are different in source: MATLAB supplements the sampled disk with
lateral companions at its tips and iteratively fills single-cell concavities. Python
and Julia use the sampled disk and treat the area outside the scan footprint as material
while closing. MATLAB also uses a different reconstruction of unmeasured site positions.
These differences must be evaluated as algorithm choices, not erased to match counts.
As an isolation experiment, adding only the eight square-lattice tip companions for
alpha=3.1 moves Python's default forsterite count from 3185 to 3170 (MATLAB 3171), but
leaves emsland at 15021. That experiment was not applied to the library.

## Confirmed defect: minPixel is not enforced after the second segmentation

Python runs two decomposition/segmentation passes. It culls undersized grains after
the first, then unconditionally accepts the second. Changed site geometry can disconnect
part of a previously large grain, producing a new indexed grain below minPixel.

At an explicit 15-degree angle and minPixel=3:

- `mtexdata('small')` leaves one indexed one-pixel grain at (36000, 6650). Its pixel
  belonged to a 239-pixel grain before the cull.
- `mtexdata('forsterite')` leaves six undersized indexed grains, sizes 2, 1, 1, 2, 1, 1.
  Their original grains had 4, 16 or 13 pixels.
- MATLAB leaves zero undersized indexed grains on both maps. Its final `enforceMinPixel`
  marks these components notIndexed without recomputing their geometry.

Minimal public-API reproduction, using locally available sample data:

```python
from mtex import mtexdata, calcGrains, degree

ebsd = mtexdata('small', silent=True)
grains = calcGrains(ebsd, angle=15 * degree, minPixel=3)
print(grains.numPixel[grains.isIndexed & (grains.numPixel < 3)])  # [1]
```

Fixed 2026-10-01 in both ports by MATLAB's policy: the fragments become notIndexed where
they lie (`test_no_indexed_grain_below_minPixel_survives_the_second_pass`, and its Julia
twin in `test/calcgrains.jl`).

This was the first algorithm defect to fix. Add a regression that verifies the final
indexed-grain sizes after culling changes connectivity, including on a transformed map.
Decide whether to classify the final fragments as notIndexed, as MATLAB does, or repeat
the geometry reconstruction; those policies produce different boundaries. MATLAB's source
(`calcGrains.m`, `enforceMinPixel`) records that merging the fragments into a neighbour was
measured and dropped: they lie in notIndexed regions by construction, and 14 of 15 on
forsterite and small had no neighbour of their own phase. That favours MATLAB's policy. A final phase
correction alone will not remove all remaining differences in total grain counts.

## Scope and reproduction

The corrected MATLAB benchmark and current Python pipeline were run on all nine rows
above; their outputs were not kept. The cases are regenerated with the `run`, `matlab` and
`compare` commands of `bench/calcgrains.py`, which overwrite the historical outputs.
The public-API default probes used each port's native sample-data loader. On csl and
forsterite the original measured positions and phase indices agree between loaders;
quaternion absolute-dot errors are at most 4.5e-16.

The advanced martensite quadruple-point example, mylonite GrainTutorial, and GrainMerge
were not rerun in this audit and remain open. The general absorption and boundary topology
differences also remain open. No segmentation implementation or default was changed.
