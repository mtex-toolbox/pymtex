# E2 on the geometry kernels: how close to Julia a Python kernel can get, 2026-09-21

`bench/e2_kernels.py` with `bench/e2_kernels.c`, one million elements, float64, best of
five, one thread unless stated. The two kernels are the Hamilton product and the search for
the best |<q, e>| over the 288 images of a cubic to hexagonal reference, the rows the
geometry benchmark (`docs/research/benchmark-geometry.md`) found 10x and 9x behind Julia.

| kernel | product ns/elem | search ns/elem |
| --- | ---: | ---: |
| NumPy as in the port, `(n, 4)` rows, one array pass per operation | 153 | 3745 |
| NumPy, four columns, cache blocked (16384 elements), preallocated | 23 | 2074 (blocked 4096) |
| numba `@njit`, `(n, 4)` rows, one thread | 7.6 | 690 |
| numba `parallel=True`, 12 threads | 5.9 | |
| C through ctypes, `(n, 4)` rows, `gcc -O3 -march=native` | 7.5 | 617 |
| Julia, one thread (from the geometry benchmark) | 14 | 1037 |

Why the port is slow: NumPy executes one operation over the whole array, so the 28 flops
of the product are 28 passes over arrays that do not fit in cache. One multiply of two
contiguous columns costs 2.8 ns per element, on a column of the `(n, 4)` array 9.5 ns
because the stride drags all four coordinates through the cache. 28 x 9.5 is the 153
measured. The search adds the same on an (n, 288) matrix formed whole, and at one million
elements that is 2.3 GB three times over.

What the table says:

- Cache blocking alone, in pure NumPy, gives 23 ns: the chunk stays in L2 across the 28
  passes and the Python loop over chunks costs 1 ns per element. That is 1.6x Julia for
  the product and 2x for the search, with no dependency.
- A compiled loop beats Julia's single thread on both kernels, at 0.5x and 0.6x, on the
  existing `(n, 4)` row layout: the layout is not the limit once the loop is fused, so D1
  stands either way.
- numba and C are the same speed. numba costs a 195 MB install (llvmlite 172 MB), 264 ms
  of import on top of NumPy's 173, and a first call of 390 ms per kernel with the disk
  cache warm (610 cold); 0.67.0 supports Python 3.14. A C or Rust extension costs the
  build and per-platform wheels and nothing at import.

The options for the port, from the numbers:

1. NumPy only, restructured: the kernels take columns, work in chunks, write into
   preallocated output. Ceiling 1.6 to 2x Julia. No dependency, no build, every kernel
   rewritten in the chunked style.
2. numba kernels with the NumPy kernel as fallback and oracle: 0.5x Julia, imported
   lazily on the first kernel call to keep `import mtex` under its budget, `cache=True`.
   The precedent is orix, which depends on numba.
3. A compiled extension (nanobind, Cython, or Rust with maturin): the same 0.5x Julia,
   cleanest for users through binary wheels, a second language and a build matrix for
   the repository.

Decision, the same day: option 2. `mtex/geometry/kernels.py` holds the loops (the two
products, the two searches, the three projections) with their NumPy forms, numba is an optional dependency imported on the first kernel call, the
preference `kernels` selects the backend, `tests/test_kernels.py` compares them. The geometry
benchmark rerun with the kernels is in `docs/research/benchmark-geometry.md`.

The kernel surface is small either way: the products, the rotation of vectors, the image
search and the pair search of `symreduce.py`, and later the representation conversions.
