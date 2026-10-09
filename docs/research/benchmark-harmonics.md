# E3: the harmonic transforms, measured 2026-09-21

The question of `docs/plans/functions.md` 8: does finufft at scattered nodes, with the
coefficient transforms of H7 in front of it, reach the budget of the Julia port
(`../mtexJulia/docs/research/benchmark-harmonics.md`: 1e6 orientations at L = 64 in about
0.35 s of NFFT plus 0.07 s of coefficient transform, 1e6 directions at L = 256 in about
0.05 s), and is the equispaced FFT on the Clenshaw-Curtis grid the exact and faster route
there, as Julia section 13 found. `bench/harmonics.py`, 12 cores under a load average of
8 from a second session, finufft with its default threads, numba loops compiled once.

## The coefficient transforms alone

| | L | forward | adjoint | Julia, two threads |
|---|---|---:|---:|---|
| SO(3), numba | 64 | 0.027 s | 0.038 s | 0.013 s / 0.018 s |
| SO(3), numba | 128 | 0.43 s | 0.56 s | 0.24 s / 0.41 s |
| SO(3), NumPy fallback | 64 | 0.13 s | 0.11 s | |
| SO(3), NumPy fallback | 128 | 2.5 s | 2.1 s | |
| S2, NumPy | 256 | 0.074 s | 0.23 s | 0.025 s / 0.016 s |

The Wigner pair is the O(L^4) of the pipeline and runs as two numba loops per degree,
serial; Julia threads within a degree from degree 16, which is where its factor of two
comes from. The NumPy fallback (one `einsum` per degree) is four to five times slower and
stays the oracle. The spherical pair is O(L^3) and stays NumPy; its adjoint could take the
same shape as the forward if it ever matters.

## Scattered nodes through finufft, 1e6 nodes

| | L | plan | evaluate | adjoint | Julia budget |
|---|---|---:|---:|---:|---|
| SO(3), real | 64 | 0.20 s | 0.30 s | 0.45 s | 0.42 s |
| S2, complex | 256 | 0.12 s | 0.11 s | 0.19 s | 0.05 s |

Evaluate and adjoint include the coefficient transform. finufft's `eps = 1e-8` is the
accuracy knob; the oversampling factor and the kernel width are its own choice, so Julia's
`nfftParameters` rule (section 11) has no counterpart here. Against the direct sum the
evaluation is accurate to 1e-7 relative, and `eps = 1e-11` reaches 1e-10, the tolerance
below which finufft warns in double precision.

## The Clenshaw-Curtis grid through scipy.fft

| | L | nodes | evaluate | adjoint | finufft on the same nodes |
|---|---|---:|---:|---:|---|
| SO(3) | 64 | 2 180 100 | 0.075 s | 0.094 s | about 0.6 s / 0.9 s |
| S2 | 256 | 263 682 | 0.087 s | 0.16 s | about 0.05 s / 0.1 s |

Exact to 1e-15 where finufft is accurate to 1e-7, so it is the route of every quadrature
(`SO3FunHarmonic` of a callable, `symmetrise` by quadrature, the pointwise operations of
F3), as in Julia. The folded grid shrinks the two periodic transforms by the folds exactly.

## Two things found on the way

- **Thread contention is the outlier.** finufft runs on every core by default; under the
  other session's load an evaluation that takes 0.06 s alone took 12 s once, and the test
  file's slowest test went from 240 s to 3 s when the load dropped. The plan holds no thread
  policy; if it becomes a problem, finufft's `nthreads` option is the knob.
- **`numpy.polynomial.legendre.leggauss` solves a dense eigenproblem**, 0.66 s for 400
  nodes; `scipy.special.roots_legendre` takes a millisecond (F1).

## Verdict

D6 holds: finufft for scattered nodes, `scipy.fft` on the Clenshaw-Curtis grids, the direct
sums below 50 nodes. The one gap against Julia is the missing threading inside a degree of
the Wigner transforms, a factor of two at L = 64 and above; a `prange` over l in the numba
loops would close it and is left for the day the transform is the bottleneck of a caller.

## Against MATLAB's master of 2026-09-25, and T1 of `docs/plans/harmonics-speed.md`

`bench/harmonics_user.py` and `bench/matlab/bench_harmonics.m`, the user facing calls, times
in ms, 10 physical cores; the 1e6 node rows vary by up to four times between runs. The
baseline and the MATLAB column are the table of the plan's section 1.

| case | before | T1 | MATLAB |
|---|---:|---:|---:|
| SO3 eval L=64, 1e6, 1 / m-3m | 1280 / 1666 | 217 / 239 | 351 / 232 |
| SO3 eval L=128, 1e4, 1 / m-3m | 1364 / 495 | 620 / 180 | 326 / 101 |
| SO3 eval L=128, 1e6, 1 / m-3m | 2467 / 1567 | 821 / 448 | 660 / 397 |
| SO3 quadrature L=64, 1e6, 1 / m-3m | 1626 / 2635 | 278 / 288 | 292 / 247 |
| SO3 quadrature L=128, 1e4, 1 / m-3m | 2051 / 920 | 877 / 337 | 321 / 161 |
| SO3 quadrature L=128, 1e6, 1 / m-3m | 3710 / 2220 | 1268 / 817 | 1170 / 665 |
| SO3 Clenshaw-Curtis L=127, 1/1 | 1842 / 2525 | 755 / 908 | 506 / 587 |
| SO3 Clenshaw-Curtis L=127, 432/1 | 509 / 848 | 190 / 351 | 113 / 210 |
| SO3 Clenshaw-Curtis L=128, 622/222 | 224 / 275 | 93 / 211 | 53 / 88 |
| SO3 isReal L=128 | 36 | 3.2 | 3.6 |
| S2 eval / quadrature L=128, 1e6 | 197 / 170 | 84 / 91 | 98 / 130 |
| S2 symmetrise m-3m L=128 | 2813 | 3.8 | 24 |
| S2 calcDensity 1e5 | 30 | 20 | 17 |

What T1 did: the nodes and their phase from one compiled loop (`mtex/functions/kernels.py`),
a plan building only the direction a call takes, finufft on the physical cores with the
oversampling factor by density, `scipy.fft` on the cores, the real defect in one pass over
half of every block, and the S2 projection from the held projections of H5. At L=64 the
scattered routes are level with MATLAB; what is left at L=128 and on the grids is the serial
Wigner transform (T2) and the lattice of the grid route (T4).

## T2: the Wigner transforms as parallel kernels, 2026-09-25

The quadrant `d_n^{jk}(pi/2)`, j, k >= 0, as one flat table by a compiled recursion, and
the two transforms parallel over the planes l of the Fourier array, each thread reading the
column l of every degree block and writing its plane once (`mtex/functions/kernels.py`).
Equal to the NumPy form to 3e-15.

| real, ms | L=64 forward / adjoint | L=128 forward / adjoint |
|---|---:|---:|
| before, serial per degree | 27 / 37 | 419 / 569 |
| T2, no folds | 4.1 / 4.1 | 51 / 52 |
| T2, fold 4 | 1.2 / 1.6 | 13 / 14 |

The user facing rows after T2 (ms, MATLAB in brackets): SO3 eval L=128 1e4 221 (326) and
91 (101) for m-3m, 1e6 480 (660) and 453 (397); the Clenshaw-Curtis grid L=127 1/1 396 (506)
and 468 (587), 432/1 87 (113) and 207 (210), 622/222 32 (53) and 121 (88). The quadrature at
scattered nodes is now the NUFFT alone: L=128 1e4 431-479 (321), where finufft's type 1 on
the half lattice is FFT bound.

## finufft against the NFFT of `../nfft` (feature/mtex-kernels), 2026-09-25

The branch built with CMake as `libnfft3_omp.so`, driven through a ctypes shim the way
`nfftmex` drives it (`NFFT_REAL`, `NFFT_PRUNED_FFT`, `NFFT_OMP_BLOCKWISE_ADJOINT`, no
`PRE_PSI`, `nfftParameters`' m and sigma), against finufft on the half lattice at eps 1e-8;
plan, nodes and transform, 10 threads, random nodes, ms.

| | NFFT real trafo / adjoint | finufft type 2 / type 1 |
|---|---:|---:|
| L=64, 1e6 nodes | 228 / 214 | 179 / 245 |
| L=128, 1e4 | 182 / 164 | 192 / 345 |
| L=128, 1e6 | 449 / 438 | 427 / 614 |

The evaluation is level; the adjoint is 15 to 50 % faster in the NFFT, most at few nodes on
a large lattice, where the pruned FFT counts.

The NFFT as the optional backend of the SO(3) adjoint (`mtex/functions/nfft.py`, used where
the library is configured and the tolerance is the default), the compiled Wigner adjoint
reading the NFFT's lattice in its own axis order without a copy; `SO3FunHarmonic.quadrature`,
ms, MATLAB in brackets:

| | finufft | NFFT backend |
|---|---:|---:|
| L=64, 1e6, 1 / m-3m | 252 / 304 | 240 / 175 (292 / 247) |
| L=128, 1e4, 1 / m-3m | 480 / 224 | 258 / 161 (321 / 161) |
| L=128, 1e6, 1 / m-3m | 693 / 661 | 556 / 465-670 (1170 / 665) |

## T3: the spherical transforms by rows, 2026-09-25

The column m of d_l(pi/2) is its row m up to sign, and the rows come degree by degree from
the recursion at a fixed row: a thread owns an order m, carries three rows of length L + 1,
and reads only the row 0 of every degree and the factors of the recursion, which split into
row and column vectors so that an entry costs four products. No quadrant table, which
streamed from memory was the whole cost (2.9 GB at L = 1024). Entries born below 2^-1000,
in a band around k = m from m of about 1000 on, are carried as mantissa and exponent until
they grow to 2^-60, as MATLAB does since `a91aef87f`; before that the transform was 12 % off
at L = 1100, now 7e-14 against the direct sum (1e-11 at L = 2000, the direct sum's own
rounding being of that order). Equal to the NumPy form to 2e-15 (1.5e-14 at L = 300, where
the NumPy recursion's cos(pi/2) = 6e-17 accumulates).

| ms | L=256 | L=512 | L=1024 | L=2000 |
|---|---:|---:|---:|---:|
| before, forward / adjoint | 72 / 147 | 3056 / 3516 | 25939 / 30444 | |
| T3 | 5 / 6 | 23 / 24 | 234 / 209 | 2449 / 3426 |
| MATLAB mex, forward | | | 400 | 3300 |

User rows after T3 (MATLAB in brackets): S2 eval L=128 1e6 67 (98), quadrature 58 (89-130),
Clenshaw-Curtis eval L=256 9.6 (28), L=512 84 (83), quadrature of a callable L=256 22 (19),
calcDensity 1e5 20 (17). What is left on the grid is its FFT route, T4.

## T4: the Clenshaw-Curtis grids one axis at a time, 2026-09-25

The grid plans transform one axis at a time, each on the data it has: beta by an FFT of
length 4Lg on the occupied planes (l, k), cut to the grid's rows, alpha on those rows,
gamma last as one real valued inverse FFT for a real function (`irfft` of the conjugated
half array with the plane l = 0 doubled back), placement buffers held by the plan and zero
where nothing is written; the adjoint backwards, `rfft` in gamma first. Before, a lattice of
(2L+2)^2 4L numbers, 532 MB at L = 127, was zeroed and transformed on every call. Exact
against the direct sums at the grid nodes to 1e-12.

Two things found on the way:
- `quadratureSO3Grid.fullValues` found the distinct nodes of the grid (MATLAB's uniqueGrid)
  on every quadrature to tell values at the distinct nodes from values at all of them, 28 s
  for 622/222 at L = 128; values at all nodes now skip it. `_unique` itself is still that
  slow where it is asked for (`size`, `flatten`, values at the distinct nodes).
- The symmetry projection of the dihedral groups has 1.5 % (222) to 5 % (622) of its block
  entries above rounding; `project` runs a compiled loop over the nonzeros, parallel over
  the degrees, where both sides are below a quarter full (MATLAB's `symmetriseWignermex`),
  and the matrix products of BLAS for the dense cubic blocks: 622/222 20 -> 4 ms. The checks
  that a reshape is a view took `np.shares_memory`, milliseconds on a coefficient vector;
  a contiguous array is the test now.

| ms, MATLAB in brackets | eval | quadrature |
|---|---:|---:|
| SO3 L=127 1/1 | 286 (506) | 300 (587) |
| SO3 L=127 432/1 | 85 (113) | 125 (210) |
| SO3 L=128 622/222 | 21 (53) | 44 (88) |
| S2 L=256 / 512 / 1024 | 13 / 60 / 351 (28 / 83 / 517) | |

SO3 symmetrise L=128: 9 / 26 / 14 ms for 622/222, 432/mmm, 432/1 (15 / 24 / 22). The ring
length 2L+2 with a large prime factor costs 25 to 35 % (L = 96, 194 = 2 * 97: 149 ms against
115 for the neighbours); it defines the grid, whose node count MATLAB shares, so it stays.

## Closing: Python, MATLAB and Julia in one sitting, 2026-09-25

`bench/harmonics_user.py`, `bench/matlab/bench_harmonics.m` (MATLAB R2024b, master of
2026-09-25 through the bridge) and `bench/harmonics_user.jl` (`../mtexJulia/dev`, 10
threads, with the uncommitted state of that checkout), one after the other, times in ms.
The 1e6 node rows move by up to a factor of two between runs on this hybrid CPU; the NFFT
column is the same Python with the NFFT backend of the SO(3) adjoint. Julia crashed in
FFTW's threads (segfault) at L = 128 with 1e6 m-3m nodes, eval and quadrature alike.

| case | Python | Python, NFFT | MATLAB | Julia |
|---|---:|---:|---:|---:|
| SO3 eval L=64 1e6, 1 / m-3m | 201 / 231 | | 282 / 239 | 399 / 1445 |
| SO3 eval L=128 1e4, 1 / m-3m | 250 / 101 | | 313 / 104 | 487 / 1438 |
| SO3 eval L=128 1e6, 1 / m-3m | 450 / 304 | | 1096 / 607 | 1049 / crash |
| SO3 quadrature L=64 1e6, 1 / m-3m | 275 / 296 | 287 / 243 | 302 / 352 | 583 / 1589 |
| SO3 quadrature L=128 1e4, 1 / m-3m | 419 / 182 | 376 / 208 | 417 / 184 | 948 / 1651 |
| SO3 quadrature L=128 1e6, 1 / m-3m | 708 / 614 | 962 / 663 | 1298 / 696 | 1698 / crash |
| SO3 grid L=127 1/1, eval / quadrature | 335 / 397 | | 513 / 609 | 955 / 989 |
| SO3 grid L=127 432/1 | 96 / 142 | | 115 / 202 | 211 / 317 |
| SO3 grid L=128 622/222 | 22 / 46 | | 56 / 89 | 105 / 172 |
| SO3 symmetrise L=128 622/222, 432/mmm, 432/1 | 6 / 23 / 18 | | 18 / 28 / 32 | 36 / 25 / 27 |
| SO3 isReal L=128 | 3.0 | | 3.7 | |
| SO3 rotate L=128 / 64 | 135 / 20 | | 119 / 17 | |
| S2 eval / quadrature L=128 1e6 | 70 / 98 | | 103 / 95 | 272 / 186 |
| S2 grid eval L=256 / 512 / 1024 | 13 / 75 / 508 | | 22 / 85 / 747 | 17 / 275 / 2960 |
| S2 quadrature of a callable L=256 | 26 | | 30 | |
| S2 symmetrise m-3m L=128 | 3.8 | | 27 | 2682 |
| S2 calcDensity 1e5 | 15 | | 21 | 20 |

Python is faster than both on every row but three, where it is level with MATLAB within the
run to run spread: the quadrature at 1e4 nodes at L = 128 (419 against 417, the NFFT backend
376), the S2 quadrature at 1e6 nodes (98 against 95, 58 in an earlier run) and `rotate`
(135 against 119; the phases around a real d_n(beta) from a compiled table, the O(L^4) of the
products left). The NFFT backend is ahead at few nodes on a large lattice and within the
spread elsewhere.

## The Dubna chain: SO3Fun.dubna, discreteSample(1e7), calcDensity, 2026-09-25

`bench/dubna_density.py`, `bench/dubna_density.jl`, `bench/matlab/bench_dubna_density.m`;
warm runs, s. MATLAB's `SO3Fun.dubna` answers 0.31 s from the second call on; its
`calcODF(pf)` alone is 1.0 to 1.2 s, the fair comparison of the first step.

| step | Python before | Python after | MATLAB | Julia |
|---|---:|---:|---:|---:|
| odf = SO3Fun.dubna (calcODF of the seven pole figures) | 11.6 | 1.85 (3.0 under load) | 1.0-1.2 | 2.3 |
| discreteSample(1e7) | 5.5 | 0.46 (0.8 under load) | 1.5 | 2.3 |
| calcDensity(ori) | 1.26 | 1.3 (2.3 under load) | 1.06 | 4.3 |

What changed:
- **The OpenMP runtimes of finufft (its own libgomp) and the NFFT spin their threads after a
  transform**, and numba's TBB pool runs on the same cores: a 2D transform of 1e5 nodes on a
  97 x 97 lattice took 115 to 265 ms after a numba kernel instead of 3. `OMP_WAIT_POLICY` is
  set to `PASSIVE` before either library loads (`passiveOpenMP` in `geometry/kernels.py`, a
  policy the user set stays); calcODF 14 -> 2 s, calcDensity 4 -> 1.3 s against `ACTIVE`.
- **The pole figure operator takes real plans**: its values are real, the half lattice
  serves, 1.95 -> 1.85 s, the ODF equal to 1e-8.
- **The radial basis function sampler is one compiled loop** (`rbfSample` in
  `functions/kernels.py`): a centre by its weight, the angle by the kernel's inverse cdf, a
  uniform axis, the group elements on both sides, uniforms from the caller's generator in
  chunks of 2^20, so a seeded generator gives the same sample; 5.5 -> 0.46 s.

Measured and left:
- **The NFFT with precomputed windows (PRE_PSI) for the kept S2 plans of the solver** is slower
  than finufft at these sizes (60 to 120 thousand nodes, 97 x 97): 5.9 / 5.3 ms, 8.1 / 7.1 ms
  with PRE_PSI, against finufft's 2.7 / 4.2 ms.
- **The operator through one SO(3) transform on the grid** (Y_n(g h) = conj(D_n(g)) Y_n(h),
  the grid's Wigner coefficients shared by every pole figure, the backward summed over them
  into one series) gives the same ODF to 9e-9 but is slower, 2.4 s: the 3D spreading of the
  20040 grid nodes costs more than the 2D spreading of their 60 to 120 thousand images.
- **finufft with upsampfac 2, ten threads and FFTW_ESTIMATE** took 355 ms for a transform that
  takes 38 ms with FFTW_MEASURE (20040 nodes, 17 x 97 x 97); the density rule picks 1.25
  there, but the combination is a trap for other sizes.

## The gridded adjoint, the NFFT for the whole chain, and calcODF against MATLAB, 2026-09-25

**The gridded adjoint** (`wignerSeriesAdjointGridded`, MATLAB's `adjoint(..., 'gridded')`):
above 2 L^3 centres (10 L^3 until 2026-09-27, MATLAB's crossover) `SO3FunRBF._harmonic` rounds them onto a Clenshaw-Curtis grid of
bandwidth min(2L, max(L, 128)), folded by the z folds of the frames, and runs the grid's
exact FFT adjoint. calcDensity of 1e7 Dubna samples 1.27 -> 0.77 s (MATLAB 1.06); the
density's values move by 0.7 %, the rounding MATLAB accepts on the same grid.

**The NFFT for every transform** (preference `NUFFT = 'nfft'`, equal to finufft to 3.5e-8):
calcODF 2.8-3.1 s, 2.2-2.3 s with PRE_PSI, against finufft's 1.7-2.6 s; calcDensity 3.4-4.0 s
against 1.2-2.2 s. The spreading of many nodes is finufft's strength; under `auto` the NFFT
now takes the SO(3) adjoint only for fewer nodes than lattice points, where its pruned FFT
counts. A regression found on the way: `nfftLibrary()` without a configured path asked
`ctypes.util.find_library` on every transform, which starts processes; calcODF went from
1.7 to 8 s until the lookup was held.

**calcODF against MATLAB.** The same algorithm with the same counts: 7 iterations, 56 forward
and 56 backward operator calls, 224 transforms, bandwidth 48, gh nodes 59544 or 119088 per
pole figure (Python 60120 or 120240, its grid 20040 orientations against MATLAB's 19848),
1368 specimen directions.

| | MATLAB | Python |
|---|---:|---:|
| total | 1.14 s | 1.68 s |
| transforms | 1.09 s (nfsftmex, 729 calls, plan precomputation included) | 1.22-1.24 s (224 finufft executes) |
| the rest | about 0.3 s (init 0.29) | 0.45 s: the grid 0.15, the plans 0.06, the S2 coefficient transforms 0.09, the solver's vector arithmetic 0.11 |

The large transforms take 11.9 / 14.3 ms (type 1, 60 / 120 thousand nodes) and 5.5 / 8.5 ms
(type 2) inside the loop, against 5.6 / 7.6 and 2.2 / 4.1 ms for the same seven plans called
in turn in isolation; at the isolated times the transforms would be 0.55 s and calcODF about
1.1 s. Not the cause: the order of the plans (measured), the numba pool (numba off is no
faster), the threading layer (TBB is the fastest of the three). The runs were made at a load
average of 5 to 9 from other processes, with the same configuration varying from 1.7 to 2.8 s,
so the cause of the in-loop factor is left for a quiet machine.

**The cause, on a quiet machine.** The large transforms of calcODF took 12 / 14 ms (type 1)
and 5 / 8 ms (type 2) inside the solver and 2.4 / 3.5 and 1.2 / 2.2 ms alone, on the same
plans and the same inputs; not the numba pool (numba off: as slow), not subnormal inputs (none),
not the plans (the solver's own plan alone is as fast as a fresh one). It was OpenBLAS: the
solver's products of vectors run on NumPy's OpenBLAS, whose ten threads busy wait after
every call and take the cores of the transforms. With `OPENBLAS_NUM_THREADS=1` calcODF went
from 1.77 to 0.69 s. The environment variable acts only before NumPy loads, so
`blasThreads(n)` in `geometry/kernels.py` sets every loaded OpenBLAS (NumPy's and SciPy's are
separate libraries) through ctypes for a block and restores it; the solver's iteration and
optimalSample's run under `blasThreads(1)`. calcODF 0.65 to 0.77 s (MATLAB 1.14), optimalSample
of the 222 fibre at L = 64 3.79 -> 2.21 s; the quadrature rows of `harmonics_user.py`, which
call BLAS once per transform, do not move.

The Dubna chain after it, s: `SO3Fun.dubna` 1.0 to 1.13 (the pole figures read and calcODF;
MATLAB 0.31 from its cache, its calcODF 1.14), `discreteSample(1e7)` 0.54 to 0.59 (MATLAB
1.5), `calcDensity` 0.92 to 1.0 (MATLAB 1.06); Julia 2.3 / 2.3 / 4.3.

## The crossover of the gridded adjoint lowered to 2 L^3, 2026-09-27

The 1C_1 map (2.2 million Fe-BCC pixels) with the UCV kernel of its grain means, 3.68
degrees and L = 64, sat at 8.4 L^3, under MATLAB's 10 L^3: calcDensity ran finufft over every
pixel, 1.05 s, where the grid takes 0.16 s and moves the density by 0.4 % in L2. The sweep over
the Fe-BCC orientations of 1C_1, `calcDensity` both ways, best of three, the difference at
5000 random orientations:

| L | N / L^3 | finufft | gridded | rel. L2 |
|---|---|---|---|---|
| 25 | 1.3 | 0.012 s | 0.005 s | 4.7e-3 |
| 25 | 96 | 0.202 s | 0.100 s | 2.3e-3 |
| 38 | 1.8 | 0.035 s | 0.015 s | 4.2e-3 |
| 38 | 27 | 0.213 s | 0.106 s | 2.4e-3 |
| 50 | 0.8 | 0.047 s | 0.025 s | 6.3e-3 |
| 50 | 12 | 0.216 s | 0.123 s | 3.3e-3 |
| 64 (3.7 deg) | 1.9 | 0.797 s | 0.073 s | 5.2e-3 |
| 64 (3.7 deg) | 5.7 | 0.894 s | 0.183 s | 4.2e-3 |
| 64 (3.0 deg, cut) | 1.9 | 0.630 s | 0.098 s | 8.8e-3 |
| 64 (2.4 deg, cut) | 1.9 | 0.587 s | 0.069 s | 1.5e-2 |
| 64 (2.4 deg, cut) | 5.7 | 0.705 s | 0.147 s | 1.3e-2 |

The grid wins from about 0.2 L^3 on; its error falls with N / L^3 and grows as the kernel
narrows below what L = 64 resolves. 2 L^3 keeps the error under one percent but for kernels
cut by the bandwidth cap; `exact=True` takes finufft at any size.


## The symmetries the transforms use, and three costs outside the transforms, 2026-10-09

Which symmetries each port uses in the evaluation and the quadrature of the harmonic functions:

| symmetry | Python | Julia | MATLAB |
|---|---|---|---|
| real values, half the array (SO(3) and S2, both routes) | yes | yes | yes |
| the rotations about z of both groups, the array shrunk by the folds | grid and finufft | grid and NFFT | grid; the NFFT array is full |
| a 2-fold axis perpendicular to z in the coefficient transform | no | no | yes (`wignerTrafo`, `sym`) |
| a callable on the grid evaluated at the distinct nodes only | half the grid, by the 2-fold | no | yes (`iuniqueGrid`) |
| exchange symmetry f(g) = f(g^-1), S2 antipodal, S2 rotations about z | no | no | no |

Left out, measured: the perpendicular 2-fold in the coefficient transform saves at most half of
7.6 ms in an evaluation of 200 ms at L = 128 (m-3m, 1e6 nodes), where finufft's interpolation
takes 120 ms with its kernel 14 points wide in 3D at the tolerance 1e-8, which no symmetry
narrows; the exchange symmetry turns alpha into gamma and shrinks no array; the S2 transforms
take 27 ms at 1e6 nodes and L = 128. The other elements of the cubic groups (the 3-folds) map no
node of the Euler lattice onto a node. MATLAB's NFFT runs at m = 4, sigma = 2, the same accuracy
as finufft's 1e-8; a lower tolerance is a choice of accuracy, 1e-6 would take 25 to 30 % off.

What changed (`bench/harmonics_user.py`, ms, before / after):

| case | before | after |
|---|---:|---:|
| SO3 eval L=128 1e4, 1 / m-3m | 253 / 80 | 147 / 51 |
| SO3 eval L=128 1e6, 1 / m-3m | 363 / 218 | 263 / 192 |
| SO3 quadrature L=128 1e4, 1 / m-3m | 301 / 192 | 207 / 93 |
| SO3 quadrature L=128 1e6, 1 / m-3m | 497 / 290 | 366 / 222 |
| SO3 quadrature L=64 1e6, 1 / m-3m | 220 / 130 | 180 / 95 |
| SO3 grid quadrature L=127 1/1, 432/1, L=128 622/222 | 281 / 85 / 48 | 197 / 70 / 35 |
| SO3FunHarmonic of a callable, m-3m, L = 128 | 300 | 218 |

- **finufft allocates its oversampled grid in every execute** and frees it after; glibc maps a
  block that large fresh and unmaps it on free, so every call faulted the whole grid in again:
  66 thousand pages, 272 MB, 120 of the 230 ms of an evaluation at 1e4 nodes on the lattice of
  L = 128. `heapBlocks()` (geometry/kernels.py) takes such blocks from the heap during an
  execute, whose top keeps up to 1 GB after a free, so the next call finds the pages mapped;
  outside it large blocks are mapped as before, so an array freed below a small one is still
  given back. Reusing finufft plans across calls gains nothing: the buffer is per execute.
- **The adjoint's finufft plan has the short axis last**: finufft deals its sorted nodes to the
  threads in slabs of its slowest axis and spreads every slab into a box of the other axes,
  widened by the kernel; with the halved and folded axis l first (48 points for m-3m) the boxes
  covered the grid several times over, 151 against 83 ms at 1e4 nodes. The plan of type 1 runs on
  (k, j, l), which the adjoint Wigner transform reads as a view; the evaluation (type 2) gains
  6 to 9 ms by the same and keeps its order. On S2 the present order is the faster one.
- **The grid adjoint reads the polar FFT in place**: the kernel takes the index of the order
  j = 0 (`centre=0`, negative orders wrap around), which spares the concatenation (30 ms at
  L = 127), and the plan weighs by the rule's beta weights broadcast instead of a weight per node
  spread out (28 ms). Conjugating while the coefficients are copied, to spare the forward's pass
  over its array, measured no gain and was left.
- **A callable sees half the grid** where a 2-fold axis perpendicular to z maps the nodes beta <
  pi/2 onto the rest (`quadratureSO3Grid.halfNodes`, `fromHalf`): on the crystal side at the angle
  psi from x (alpha, beta, gamma) -> (alpha + pi, pi - beta, -gamma - 2 psi), on the specimen side
  (2 psi - alpha, pi - beta, gamma + pi), on the grid where 2 psi is a multiple of its step, which
  some axis of every such group is. MATLAB's distinct nodes (`fullGrid`, `_unique` here) take
  27 s to find at L = 128 by comparing every image; the mirror needs no search.

## The circle, 2026-10-09

`fourierSeries` summed directly up to 2e6 terms (nodes times 2L+1), the bound of the sphere's
trigonometric sum, and its adjoint below 50 nodes. The direct sum costs about 27 ns a term and a
plan 0.45 ms whatever the size, so both cross at about 2e4 terms (`DIRECT_SUM_WORK_S1`): the
evaluation at 1000 nodes and L = 64 4.8 -> 0.5 ms, at L = 1024 the sum would have taken 49 ms
against the plan's 0.56. The S1 grids themselves cost nothing that matters: `S1Grid.nearest` of
1e6 angles 41 to 158 ms, `SO3Grid.gamma` of the 2.5 degree grid 17 ms.

## The evaluation of the SO(3) functions that are not harmonic, 2026-10-09

m-3m, ms, before / after, at 1e4 and 1e6 orientations; measured under another user's load (5 to 9),
so the absolute times are high, the ratios not.

| function | 1e4 | 1e6 |
|---|---:|---:|
| `SO3FunRBF`, 1e4 centres, de la Vallee Poussin of 3 degree | 679 / 30 | 86 s / 2.8 s |
| `fibreODF` (`SO3FunCBF`), 10 degree | 17.3 / 1.0 | 1849 / 75 |
| `BinghamODF` | 3.0 / 1.3 | 293 / 27 |

The others were not the cost: an RBF of a kernel above the cell sum's 120 degree support, or of
many terms, goes through its harmonic form (65 to 87 ms at 1e6), a handle 23 ms, the uniform 1 ms.

- **A narrow kernel's sum went through a k-d tree** whose radius search returns a Python list per
  query image (24 per query for m-3m): 240 thousand lists for 1e4 queries, flattened by `fromiter`,
  with the threads of the search started per call. A 4D cell list over the centres (c and -c, cubes
  of side at least the support's chord, `cellList` in functions/kernels.py) gives each image its
  partners in the 3^4 cubes about it. The de la Vallee Poussin and the von Mises-Fisher kernel are
  summed inside the loop (`cellKernelSum`), any other kernel over the pairs it returns in batches
  (`cellPairs`), equal to 5.7e-14. The bandwidth of a 3 degree kernel is above maxSO3Bandwidth, so
  the harmonic form, which would take some 300 ms at 1e6, is not taken: the preference decides.
- **The fibre ODF** rotated its fibre vectors through container objects in chunks of 1024: the
  arguments <g h, r> are the products of the rotation matrices with the outer products r h, and
  for the de la Vallee Poussin kernel on the sphere the whole sum is one compiled loop
  (`fibreSum`); any other kernel takes the products and its own evaluation.
- **The Bingham distribution** summed its symmetric copies one NumPy pass each; one compiled loop
  over the rotations sums them all (`binghamSum`).

## The evaluation of the S2 functions, 2026-10-09

Measured under another user's load (6 to 9). Harmonic functions evaluate in 31 to 46 ms at 1e6
directions and 2 to 10 ms at 1e4, the pole figures and inverse pole figures of the RBF, fibre and
unimodal ODFs build in 1 to 24 ms warm (144 and 608 ms were their first calls, compiling),
`S2FunBingham` takes 16 ms and a handle 1.3 ms at 1e6. Two routes were not where they should be:

- **`S2FunTri`, 1e6 directions 774 -> 148 ms**: the k-d tree query for the start triangle ran on one
  thread (507 ms, 53 on the threads), the walk to the triangle was a NumPy loop of passes over the
  directions still walking, and the evaluation built a sparse matrix of the coordinates to multiply
  it once. The walk is one compiled loop (`triangleWalk` in geometry/kernels.py, the NumPy walk the
  fallback, the same triangles), the value the barycentric mean of the three corners.
- **`sphericalSeries` below 50 directions** summed the harmonics directly: the trigonometric sum
  over the transformed coefficients, which keeps the transform of the last coefficient vectors,
  takes 0.17 ms against 2.1 at L = 64 and 0.55 against 7 at L = 256 for ten directions, and wins
  with the transform included. The thresholds between the trigonometric sum and the plan are left
  for a quiet machine, their differences being within the load's noise.

## The evaluation of the S1 functions, 2026-10-09

`S1FunHarmonic.eval` is `fourierSeries`, which summed in NumPy below 2e4 terms and otherwise took a
finufft plan; a small bandwidth landed on the plan's single thread, so 1e6 angles at L = 16 took
59 ms against 16 at L = 1024. The series is now summed by Horner's scheme in z = e^{ix} as one
compiled loop (`fourierSum`, functions/kernels.py), stable on the unit circle (1.6e-13 at L = 1024),
and the adjoint by the powers of e^{-ix} with a block of coefficients per thread
(`fourierAdjointSum`); both win at any number of angles up to L = 64 and up to 4e6 terms above it.
ms, complex / real coefficients, under load 8:

| | 10 | 1e3 | 1e4 | 1e6 |
|---|---:|---:|---:|---:|
| L = 16, before | 0.01 | 0.50 | 0.93 | 59 |
| L = 16, after | 0.02 | 0.02 | 0.09 | 8.1 |
| L = 64, before | 0.03 | 0.54 | 0.91 | 52 |
| L = 64, after | 0.02 | 0.04 | 0.24 | 22 |
| L = 1024, before / after | 0.52 / 0.03 | 0.63 / 0.32 | 0.95 / 1.07 | 16 / 17 |

The S1 kernels evaluate their closed forms or series in NumPy, 10 to 101 ms at 1e6 angles; left.
