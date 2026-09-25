# Thread pools: OpenBLAS, OpenMP and numba on the same cores

Found 2026-09-25 while tuning the harmonic transforms (`docs/plans/harmonics-speed.md`, the
numbers in `benchmark-harmonics.md`). A process running the port holds up to four thread
pools, each sized to the machine, and each waits for work in its own way:

| pool | who uses it | waits after a call by | set by |
|---|---|---|---|
| OpenBLAS of NumPy (`libscipy_openblas64_`) | `@`, `np.dot`, `np.linalg` on arrays past a size | busy waiting, about 0.1 s | `blasThreads(n)` |
| OpenBLAS of SciPy (`libscipy_openblas`, a second library) | `scipy.linalg` | the same | `blasThreads(n)` |
| OpenMP of finufft (its own `libgomp`) and of the NFFT (the system's) | `SO3FourierPlan`, `S2FourierPlan`, `S1FourierPlan` | spinning, unless `OMP_WAIT_POLICY=PASSIVE` | `passiveOpenMP()`, `threads()` |
| numba, TBB layer | the parallel kernels of `mtex/functions/kernels.py`, `mtex/geometry/kernels.py`, `mtex/maps/kernels.py` | short spinning | `threads()` |

A pool that waits by spinning keeps its cores after the call that woke it, and the next
parallel call of another pool finds them taken. One call alone is fast; a loop alternating
the pools runs each at a fraction of its speed, and a benchmark of the transform in
isolation does not show it.

## What it cost

- **finufft's OpenMP spinning** (before `passiveOpenMP`): a 2D transform of 1e5 nodes on a
  97 x 97 lattice took 115 to 265 ms after a numba kernel instead of 3; `calcODF` of the
  Dubna pole figures 14 s, `calcDensity` of 1e7 orientations 4 s.
- **OpenBLAS busy waiting** (before `blasThreads`): `MLSSolver.iterate` does products of
  vectors between the 224 transforms of `calcODF`; each transform ran four times slower than
  alone (12 / 14 ms against 2.4 / 3.5), `calcODF` 1.77 s against 0.69 with BLAS on one
  thread. `optimalSample` of a 222 fibre at L = 64: 3.79 s against 2.21.
- **Not a cost**: numba's TBB against finufft once OpenMP waits passively (numba off is no
  faster); the threading layers omp and workqueue are slower than TBB.

## The rule

- `passiveOpenMP()` runs before finufft or the NFFT library loads (`_finufft` in
  `mtex/functions/transforms.py`, `_load` in `mtex/functions/nfft.py`); a policy the user set
  stays.
- A loop that interleaves transforms or numba kernels with BLAS on vectors runs under
  `blasThreads(1)`: `MLSSolver.iterate` in `mtex/analysis/pf2odf.py`, `optimalSample` in
  `mtex/analysis/optimalsample.py`. BLAS stays threaded everywhere else, where a large matrix
  product needs the threads.
- The environment variables act only before the library loads: `OPENBLAS_NUM_THREADS` before
  NumPy, `OMP_WAIT_POLICY` before finufft. `blasThreads` sets the loaded libraries through
  ctypes, found in `/proc/self/maps`; on a system without it, it does nothing.

## Suspected, not measured

The same mix, a transform plan as the operator of an iteration with norms and dot products
of the iterates between the calls:

- `lsqr` in `mtex/functions/lsqr.py`, which `interpolate` in `mtex/functions/density.py` runs
  over `S2FourierPlan` and `SO3FourierPlan`;
- `calcQuadratureWeights` in `mtex/functions/polefigure.py`, SciPy's LSQR over a NUFFT
  operator;
- the least squares fit of `S1FunHarmonic` in `mtex/functions/s1fun.py`.

The test is the one that found it: time the iteration with and without
`OPENBLAS_NUM_THREADS=1` in the environment, on a quiet machine; a gain says the loop wants
`blasThreads(1)`.

## How to see it

Time the transform inside the loop and alone, on the same plan and the same input (a
wrapper around `execute` that records both); if the loop is slower with no other difference,
another pool holds the cores. `/proc/self/maps` lists the runtimes a process has loaded
(`openblas`, `gomp`, `tbb`).
