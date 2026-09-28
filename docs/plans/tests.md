# The test suite

What a test is for, which kinds the suite keeps, and how the graph tells a needed test from a
redundant one. The review of 2026-09-28 applied this to every test; `docs/research/test-review.md`
keeps its verdicts.

## A test states one thing

A test checks one statement about one piece of functionality, and its name is that statement:
`test_fundamentalSector_projectionIsIdempotent`, not `test_orientation_misc`. The same
statement over several inputs (point groups, formats, kernels) is one test with `parametrize`,
never copies of it. A test that needs a sentence with "and" to describe is two tests, unless the
second clause is the setup the first depends on. Tests sharing an expensive setup share it through
a `scope="module"` fixture (or a session one in `conftest.py`, as the IN100 volume); a test that
mutates what it is given builds its own. A shared fixture holds no frame a test compares by
identity, since `reset()` clears the register before every test.

## The kinds of test

- **Property**: an identity of the mathematics over random input (the projection into a
  fundamental sector is idempotent and lands in the sector, `inv(r) * r` is the identity,
  symmetric equivalents have distance zero). The backbone of the geometry and function layers.
- **Oracle**: a fast path against the slow one it replaces: the numba kernel against its NumPy
  fallback, the structured grid search against the tree, the NUFFT against the direct sum. One
  per kernel, in the test file of the module that holds the kernel, never repeated downstream.
- **Reference**: numbers MATLAB (or the literature) gives, with where they came from beside them.
- **Round trip**: a reader and its writer, one per format; a reader without a writer is checked
  on a fixture file against values read off it by hand or by MATLAB.
- **Regression**: a bug once found, named after the bug, kept only while no property test
  covers it.
- **Semantics**: `tests/test_semantics.py`, one test per statement of `docs/design/semantics.md`.
- **Drawing**: that a plot is drawn and its axes, labels and artists are what they should be;
  never pixels, never a second test of the numbers the plot draws.

## When a test goes

- Its statement is checked already, by a test of the same kind with the same subject (typically
  the same identity once in the module's own test file and again in a downstream one).
- It checks the layout of private state rather than behaviour a user sees.
- It checks NumPy, SciPy or Python rather than this package.
- It only runs code ("does not raise") that a stronger test with the same subject runs too.
- It is slow for what it states: a smaller input states the same thing.

## The graph

`tools/graph/scan_python.py` gives a test an `asserts` edge to its *subject*: the functions
called inside its assertions (`assert`, `np.testing.*`, `close`, the body of `pytest.raises`) and
one assignment back from the names they use; everything else it calls is setup.
`tools/graph/coverage_python.py` records what every test runs and how long it takes (the whole suite
under it takes about fourteen minutes; run it on the files at hand otherwise). On these,
`tools/graph/query.py` answers

- `tests <node>`: the tests whose subject is the node, then those that only run it;
- `overlap [<file|layer>]`: pairs of tests in one file with the same subject and nearly the
  same measured coverage, the candidates to merge or drop;
- `untested [<layer>]`: public functions and methods no test has as its subject.

A new test is written against `untested`, and before it is added `overlap` says whether its
statement is stated already.
