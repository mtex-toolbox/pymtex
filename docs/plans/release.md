# Roadmap to 1.0.0

The decisions are in `docs/PLAN.md` section 11: GPL-2.0, `mtex` on PyPI, Python 3.12 or later,
the pages on mtex-toolbox.github.io beside MATLAB's. This note orders what is left, as of
2026-09-27. Each step ends in something that can be checked.

## Already done from the section 11 list

- `mtexdata('vpsc')` loads.
- `calcGBND` has its default kernel by inversion, MATLAB's to the digit.
- `fill(ebsd, grains)` and `smooth(ebsd, F, fill=grains)` fill by the grain polygons.
- The RBF evaluation and `max(odf)` no longer hang. With 500 centres, m-3m, 5 degrees:
  1000 evaluations take 0.4 s and `max` takes 3.3 s. `calcComponents` still takes 39 s (R3).

## R0. One history

- The rewritten local main is 24 commits ahead of `origin/main` and 14 behind it. Push it with
  `git push --force-with-lease=main:89f65f1 origin main` (the user runs this). The last CI run
  tested the old history, which still holds the piecewise steps (`test_s2piecewise` fails there).
- Commit the halfwidth changes in the working tree.
- `singular-functions` stays a branch and is not part of 1.0.0.

Done when `origin/main` is the release history and the working tree is clean.

## R1. CI green on the matrix

The run of 2026-09-26 passed 1227 tests with 1 to 8 failures per job. The failures fall in
three groups.

- **Everywhere, local too:** `test_ang.py::test_olivine_grids_with_padding`. The display no
  longer prints `652 (73%)`. Either the display changed or the padding count did; find out
  which.
- **Numerics that depend on the platform or the random draw:** `test_fibre.py::test_fit_recovers_a_fibre`
  fails on Linux, macOS and Windows but passes here. The sign of the fitted fibre is not
  fixed, so fix the seed or compare up to the antipode. Some failures appear on macOS only:
  - `test_boundary.py::test_B13` (19 segments for 18);
  - `test_exportebsd.py`: the hexagonal lattice check on `thermofisher_cut.hdf5`, and the scan
    size read back;
  - `test_load.py::test_the_numpy_backend_reads_the_same_map`.

  These look like float rounding in the lattice inference. Make it tolerant, not the tests.
- **Windows:** path separators in `sampledata` (`/` joined into a `\` path) and in the wizard's
  `callbacks` browser; files written without `encoding="utf-8"` (`importscript`, `callbacks`:
  the `y↓→x` arrows); the vector3d display comparison in `test_representations.py::test_rodrigues`.

Done when every job of `test.yml` passes, including the NumPy-fallback job.

## R2. Known differences: fix or state

The largest group of open rows in `docs/findings.md` concerns `calcGrains` (M3).
The [September 28 audit](../research/calcgrains-release-audit.md) separates a default-angle
mismatch (Python 10 degrees, MATLAB 15), an incorrect MATLAB benchmark phase mapping (fixed),
and remaining closing/culling differences. The table below records the original reports,
not comparisons at identical parameters: csl agrees exactly when the angle is matched.

Done 2026-10-01, in Python and in the Julia port: the default angle is MATLAB's 15 degrees
(`DEFAULT_ANGLE` beside `gbcAngle`), and a fragment the second segmentation leaves below
`minPixel` becomes notIndexed where it lies, its geometry kept, as MATLAB's
`enforceMinPixel`; csl's default reconstruction gives MATLAB's 885 grains, small and
forsterite keep no indexed grain below minPixel=3. The doc pages that call `calcGrains`
without an angle (about 30 in each port) are to be rebuilt. Next: the remaining closing and
quadruple-point differences with explicit parameters.

| Page | Here | MATLAB |
|---|---|---|
| TriplePoints | 78 triple points | 70 |
| CSLBoundaries | 924 grains | 885 |
| MTEXScripts (`minPixel=3`) | 1014 grains | 950 |
| MaParentGrainReconstructionAdvanced (with `removeQuadruplePoints`) | 4467 grains | 4443 |
| GrainTutorial | 373 grains | 371 |
| GrainMerge | 20 triple points | 15 |

Work on this first; it is the one item that can change numbers users see. The other open
rows are small, and each is either fixed or moved to a "known differences" section of the
README or CHANGELOG:

- the third maximum of `max(odf, numLocal=3)`;
- the GBCD twin peak (4.8 degrees where the text says 1.3);
- the trigonal rim labels;
- the phase legend listing only the phases present;
- the Euler display digits;
- the ADF bandwidth hint;
- 3363 boundary segments for 3359 after `smoothBoundary`;
- 9978 kernel centres for 9924 in `calcODF` on 6/mmm.

Done when `docs/findings.md` has no bug row that is still "open" without a stated difference.

## R3. Performance floor

- `calcComponents` on a few hundred centres: 39 s. Use the halo cell list
  (`docs/research/time-critical.md`), or at least cut the steepest ascent to the centres near
  each seed.
- The other unchecked items in `time-critical.md` wait until after the release.

Done when the ODF pages and the example above run in seconds.

## R4. Documentation

- The 14 pages that are not Checked (Written, Partial or Adapted in
  `docs/plans/remaining-pages.md`): check them, or mark their missing piece in the page itself.
- D5: `docs/site.py` exports the built pages into the Jekyll site. The switch and the layout
  on the homepage are done by the other person; agree a date with them. Do not push `../web`
  without asking (it deploys on push).
- README: install lines per extra, and one short example. Name the platforms without finufft
  wheels (Linux aarch64, macOS x86_64).

Done when the Python pages are live beside MATLAB's, or ready to go live on the tag.

## R5. Package and publish

- The tag's commit must pass the full test matrix and wheel smoke check before publishing.
  Implemented locally: `release.yml` calls the reusable `test.yml`; release builds depend
  on its success and publishing depends on the release build. Validate this gate on GitHub
  before tagging the public release.
- Packaging checked locally on Linux / Python 3.12: the sdist now includes `tools/staticapi.py`,
  required by its shipped tests. `uv build` builds the wheel from that sdist.
  `python tools/check_dist.py --dist dist --python 3.12` installs each artifact in a fresh
  environment outside the checkout, checks core-only EBSD reconstruction, ODF calculation
  and bundled tutorials, then adds the I/O/plot extras for HDF5 import and PNG output.
  Both artifacts passed 16 static-API/documentation tests and collected 1330 tests.
  The reusable CI workflow now runs these packaging checks on Linux, macOS and Windows;
  the latter two remain to be verified on GitHub.
- Set up the environment `pypi` for trusted publishing, on both PyPI and GitHub.
- Upload a release candidate to TestPyPI. Then, on each of the three OSes, install it with
  `pip install mtex[all]` into a clean environment and run a smoke script: load
  `mtexdata('forsterite')`, run `calcGrains`, do a plot, and do `calcODF` on the Dubna pole
  figures.
- Write the CHANGELOG entry for 1.0.0 (the version is bumped, 2026-09-28).
- Tag `v1.0.0`. `release.yml` publishes it.

## Order

R0, then R1 (a day, mostly mechanical). R2 is the long one, and R3 and R4 run beside it.
R5 comes last. Everything else from section 11's "after it" list waits:

- H3 to H6;
- the singular functions;
- W5;
- the remaining time-critical items;
- D12.
