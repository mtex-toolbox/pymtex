# MATLAB's recent fixes in the port

An audit on 2026-10-01 compared MATLAB's commits since 2026-09-01 (develop, and develop..
feature/fastHarmonicTransforms and develop..docs-next) with the port. Of 67 fixes and behaviour
changes, 44 were already in the port, 20 did not apply to its design, and 13 were missing. The
user decided on each; this plan holds the decisions and their state.

## Decisions

| # | MATLAB | What | Decision |
|---|---|---|---|
| 1 | a3e47891a | the fundamental sector of a frame with c* off z built around c*, not z; the port falls back to the Dirichlet sector for 34 of 50 group and alignment pairs | port |
| 2 | 8454bf73c | `plot(ebsd, 'ipfz')` | not ported: the port's spelling is `plot(ebsd, ipfDirection=zvector)`; `'ipfz'` fails with an error that names it |
| 3 | dcbe9a24b | an `ipfDirection` alone colours maps and grains by orientation | port |
| 4 | 488801503 | the wizard's sanity plot of the whole map | skip |
| 5 | 36a53dba9 | `EinsteinSum` refuses a summed index not used exactly twice | port |
| 6 | 6110d9210 | an `.osc` symmetry code that is a point group id, 100 to 131 (#2621) | port |
| 7 | 9a51196b2 | the Thermo Fisher HDF5 map correction, a rotation [90, 180, 270] | port |
| 8 | efe8b2bd7 | the on-boundary tolerance of `inpolygon` with a floor in the coordinates' magnitude | port |
| 9 | 3195d4961 | `mtexColorbar` shares a bar only between panels of one colour map (#2614) | port |
| 10 | f2393f9a6 | `nextAxis(k)` without hold empties the cell it picks (#2615) | port |
| 11 | d76744c74 | `calcGrains(ebsd, 'grainId')` on a map of the plane | port |
| 12 | 93203281b, 7fb788f1a | the discrepancies on S2 and SO(3): D_2, D_cap, L2, kernel; `sphericalDiscrepancy`, `rotationalDiscrepancy` | port |
| 13 | 909ba8263 | `fibonacciSO3Grid` under MATLAB's name, with its node count round(8π²/res³) and `antipodal=` | add beside `superfibonacciSO3Grid` |
| - | 8e137e216 | `angle(r, -r)`, π in MATLAB | keep 0: a rotation is proper by type, the improper case is `orthogonalTransform`'s |

## Order

The small ones first, each with its test: 5, 8, 6, 7, 11, 3 with 2, 10, 9; then 1, checked
against MATLAB's sectors for the failing pairs; then 13; then 12, checked against MATLAB on a
few inputs.

## Status

Decisions taken 2026-10-01. Built: all eleven, 5, 8, 6, 7, 11, 3 with 2, 10, 9, 1, 13 and 12.

Item 1 against MATLAB develop: the sectors of all 36 point group symbols in three alignments
agree with MATLAB's in 105 of 108 cases (49 before), and the HSV key colours agree in every
alignment tried, where turned lattices were off before. The three left are 321's: the port starts
its wedge at the two-fold axis along a, MATLAB 30 degrees below; turning it to MATLAB's start
changed the colours, which the port's derived fold then no longer matched, so the outline stays
and `docs/findings.md` records it. Repeated sector planes (222, -4) are given once.

Item 13: `fibonacciSO3Grid` gives MATLAB develop's node counts (4994 at 5 degrees, 5034 for
points=5000, 619 at 10 degrees) and its nodes; antipodal=True keeps 2499 where MATLAB keeps 2503,
four nodes on the boundary of the region that the two inside tests round apart.

Item 12: `discrepancy`, `sphericalDiscrepancy` and `rotationalDiscrepancy` (`mtex/functions/discrepancy.py`)
with D_2, D_cap / D_ball, L2 and the kernel objective on S2 and SO(3), every call form, methods
on vectors, rotations and the function classes. Against MATLAB develop: to 1e-12 and better where
MATLAB's transforms are exact; with atoms on SO(3) MATLAB's adjoint transforms are off by 3e-8 and
3e-6 against a direct sum, which the port takes. The default D_ball centres differ (the port's and
MATLAB's equispaced 72 rotations are not the same), so that lower bound differs slightly. MATLAB's
checkStandardDiscrepancies still expects an error for unequal masses that develop's code no longer
raises; the port follows the code.
