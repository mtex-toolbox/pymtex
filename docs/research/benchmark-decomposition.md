# E2 for the maps layer - the spatial decomposition, numba against C

Run 2026-09-21, `bench/decomposition.py` with `bench/decomposition_kernel.c`, Python 3.12,
NumPy 2, numba 0.61, gcc, 12 threads, the machine of `benchmark-geometry.md`. The
question of `docs/PLAN.md` D5 for P3: do the grain kernels reach the Julia budget with
numba, or does P3 need a compiled extension?

Setup as the Julia E5 (`../mtexJulia/docs/research/benchmark-decomposition.md`):
synthetic square and hexagonal maps of `n x n` cells with three round holes of radius 1.5,
4 and 9 cells and 0.5 % speckle, the alpha closing at `alpha = 3.1`, sites for the
measurements and the enclosed holes, a band and a dummy ring outside. The clipping of
`mtex/maps/kernels.py` runs as one numba loop over chunks of 4096 sites, each chunk writing
into its own slice of a face buffer; the C loop is the same algorithm on one thread,
called through `ctypes`. Times in seconds.

| grid | sites | closing and sites | clipping, numba 1 thread | numba 12 threads | C 1 thread | vertex weld | faces | vertices |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| square | 91 949 | 0.07 | 0.07 | 0.03 | 0.07 | 0.06 | 184 180 | 93 436 |
| square | 1 002 985 | 0.29 | 0.68 | 0.26 | 0.56 | 0.53 | 2 013 902 | 1 014 922 |
| hex | 91 894 | 0.02 | 0.07 | 0.03 | 0.05 | 0.07 | 273 102 | 182 417 |
| hex | 1 003 009 | 0.30 | 0.72 | 0.28 | 0.51 | 1.10 | 2 990 295 | 1 991 295 |

The Julia numbers on the same maps: a million sites clipped in 0.93 s (square) and 0.86 s
(hex) on one thread, 0.31 and 0.32 s on 12; the closing in 0.23 and 0.29 s; the weld in
0.30 and 0.47 s; end to end 0.55 s square and 1.06 s hex.

## What it says

- **numba is C.** On one thread the numba loop is within 1.2 (square) and 1.4 (hex) of
  the C loop, and faster than the Julia kernel on one thread. On 12 threads it matches
  Julia's 12 threads. D5 is settled for P3 as it was for P1: numba loops with NumPy forms
  as oracles, no extension.
- **The face counts are the ones the lattice demands**, 2 faces and 1 vertex per square
  cell, 3 and 2 per hexagonal one, and identical to the Julia counts, so the complex is
  structurally right and not merely fast. The tests hold every interior site to exactly
  its stencil, the cocircular case a general Voronoi code has to tie-break.
- **The weld is the slow step now** (2026-09-21, later the same day: a straddle bug made it a two-grid weld, 0.70 s square and 1.47 s hex, see `docs/plans/maps.md` M5). `np.unique` over the rounded vertex coordinates,
  written as one integer key per grid cell, sorts four to six million keys in 0.5 to 1.1 s
  where Julia's dictionary does it in 0.3 to 0.5 s. `np.unique` on the `(n, 2)` array
  itself took 3.9 and 6.9 s, which is why the key is one column. A hash in the numba loop
  would take the weld below the clipping; it is not needed for the budget of M3.
- **The closing and the sites** through `scipy.ndimage` and NumPy slicing are at the
  Julia time.

End to end at a million sites: 1.1 s square and 1.7 s hex against Julia's 0.55 and 1.06,
the difference being the weld. A nine million pixel map was not run: the chunked face
buffers of the clipping would take 1.5 GB at that size, and the buffer should shrink to
the exact count with a counting pass before that map is a target.
