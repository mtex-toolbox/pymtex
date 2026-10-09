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
  `mtex/analysis/optimalsample.py`, the least squares fits below. BLAS stays threaded
  everywhere else, where a large matrix product needs the threads.
- The environment variables act only before the library loads: `OPENBLAS_NUM_THREADS` before
  NumPy, `OMP_WAIT_POLICY` before finufft. `blasThreads` sets the loaded libraries through
  ctypes, found in `/proc/self/maps`; on a system without it, it does nothing.

## The least squares fits, measured 2026-09-25

The same mix, a transform plan as the operator of an iteration with norms and dot products
of the iterates between the calls, 2e4 nodes, s:

| | before | under `blasThreads(1)` |
|---|---:|---:|
| `interpolate` on S2 (`lsqr` in `mtex/functions/lsqr.py` over `S2FourierPlan`) | 3.3 | 0.63 |
| `interpolate` on SO(3), L = 16 (over `SO3FourierPlan`) | 0.8 | 0.38 |
| `S1FunHarmonic.interpolate` (`mtex/functions/s1fun.py`) | 1.1 to 1.4 | 0.28 |
| `calcQuadratureWeights` with `method='harmonic'` (SciPy's LSQR over a NUFFT operator) | 8.4 | 2.3 |

The three fits share `_lsqrFull` in `mtex/functions/density.py`, which runs `lsqr` under
`blasThreads(1)`; `calcQuadratureWeights` does it around its LSQR branch in
`mtex/functions/polefigure.py`, the dense least squares below 4000 nodes keeping its BLAS
threads. `lsqr` itself stays unwrapped: `mtex/functions/rbfinterpolation.py` runs it over a
matrix, where the threads help.

The test that finds the next one: time the iteration with and without
`OPENBLAS_NUM_THREADS=1` in the environment, on a quiet machine; a gain says the loop wants
`blasThreads(1)`.

## How to see it

Time the transform inside the loop and alone, on the same plan and the same input (a
wrapper around `execute` that records both); if the loop is slower with no other difference,
another pool holds the cores. `/proc/self/maps` lists the runtimes a process has loaded
(`openblas`, `gomp`, `tbb`).

## The bound of a single thread on two machines, 2026-10-09

`SINGLE_THREAD_WORK` in `functions/kernels.py` sends a compiled loop to one thread below a work
in inner steps: the spherical transform and its adjoint ((L+1)^3) and `s2Nodes` (40 per node).
It was set to 1 << 16 from the laptop's break-even (twelve threads). On sixteen cores the threads
break even at about 5000 to 15000, also with 20 ms idle between the calls, where they start
cold: the spherical transform of L = 40 takes 37 us on the threads against 131 on one, s2Nodes
of 1600 nodes 22 against 86. The bound is 1 << 14 now, between the two; the small transforms of
optimalSample (bandwidth 16, a few hundred nodes, works of 4913 and 12000) stay on one thread on
both machines, and optimal_s2 measures the same at any bound from 0 to 1 << 16 here (220 ms).
