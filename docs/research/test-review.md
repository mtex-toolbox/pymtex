# The review of the test suite, 2026-09-28

Every test was read against `docs/plans/tests.md`, layer by layer, with the graph's `overlap`
and `tests` queries as hints and a fresh coverage run (`tools/graph/coverage_python.py`, 1385
passed, 10 skipped) as the measurement.

## What it found

The suite was not redundant: of 1102 test functions 4 went as duplicates and a handful of
lines inside tests, while splits of grab-bag tests added some; 1098 test functions, 1422
collected cases (1395 before). The `overlap` pairs the graph flagged were, on reading, different
statements over the same code (nearest, k-nearest and radius search; least squares, minimum
norm and the stopping flag), so a shared subject with shared coverage does not make a
duplicate by itself.

What it found instead were assertions that could not fail, all fixed:

- `test_intragranular` I5 compared `KAM(radius=1.5)` with itself (the reference multiplied by 0);
- `test_sampledata` compared `mtexdata("steel")` with itself;
- `test_importscript` accepted any half turn (`or angle > 3.1`), `test_ctf` either of two headers;
- `test_miller` `close(uvw, ...) or close(UVTW, ...)` with a first clause that was always false;
- `test_symmetryops::test_axis` asserted `angle(a, a) == 0`;
- `test_sections` E6 `jointCS is None or numSym() >= 1` (m-3m and 6/mmm share 222: now `== 4`);
- `A and B or C` and `X if hasattr(...) else True` forms in `test_sectionplot`, `test_space`,
  `test_spaceplot`, `test_spherical`, `test_render3d`, `test_views`, `test_window`, `test_gridify`;
- the numba oracles of `test_table`, `test_exportebsd` and `test_kernels` returned early without
  numba; they now skip or fail.

And tests far slower than their statement needs: the precomputed colour key (40 s to 5 s), the
fibre volume of C1 (14 s to 0.15 s), the FMC seed loop (16 s to 3 s), `calcParent2Child` (16 s
to 7 s), the orientation tree oracles (6° to 10° grids), the Bingham fit, the density halfwidth
selection, the pole figure plots (one fixture for the quartz tests), the ODF round trip.

## What went

`test_oracles` S9/S10 (a formula in the test file against SciPy, no MTEX; the semantics tests
and `test_euler_against_scipy` state it), `test_representations::test_euler_definitions`
(semantics S9 to S13, left as the refusal of an unknown convention), `test_quadruple` Q2 (Q6
on a larger map), `test_load::test_loadEBSD_dispatches_on_the_extension` (every format test
loads by extension), `test_state::test_file_row`, the weaker compiled grid oracle of the
orientation keys, and single duplicated lines in tangent, symreduce, transforms, kernelfunctions,
s1fun, so3vectorfield, tensor, directionkeys and install. Merged into parametrized tests: the
vendor corrections of `test_h5`, the `readTable` inputs, the `phaseShift` cases, the tracing
oracles of `test_grains`, the projection into the orientation region over nine groups.

## Done after the review

- `CSL` (`geometry/csl.py`) searches any lattice over its lattice vectors instead of the
  integer quaternions of a cubic one: exact for a metric with rational ratios, the constrained
  coincidences of any other with a misfit `delta`, Sigma counted as the index of the coincidence
  lattice in a primitive cell of the Bravais lattice: `cs.centering` (read off a space group
  symbol, a CIF file, or `centering=`; unstated it agrees with any stated one) or
  `CSL(sigma, cs, 'I')`. The cubic P, I and F lattices share their coincidences, a centred
  lattice of lower symmetry (hR, tI, oC, ...) has its own. Its tests take 2.5 s where the cubic test took 27 s, and the axis sectors of the
  orientation projection are cached per group pair, which was 0.1 s on every projection.
- The semantics tests assert S5 near pi, S6's properties, S21's compatible frames, S22's sides,
  S25's landing frame and S26's render time default.
- The IN100 volume is segmented once per run (`in100` in `conftest.py`).
- The grab-bag tests are split into statement-named tests: 1098 test functions became 2592
  (3906 collected cases), every assertion kept; shared setup is a module fixture where it is
  expensive and built per test where a test mutates it (a module fixture must not hold frames a
  test compares by identity: `reset()` clears the register before every test).

## Open

- **Slow on their own:** `test_IN100_as_MATLAB_smooths_it` (14 s, the refinement to 3 million
  faces), `test_loadgrains3d::test_the_dream3d_mesh` (`orientFaces` twice), the NumPy cases of
  decomposition D1 and D3.
- **Untested:** `query.py untested` lists 1186 public functions and methods no assertion names,
  279 of them never run by any test; much is display, Qt and property plumbing, the rest are gaps.
