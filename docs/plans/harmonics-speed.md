# Subplan - the harmonic transforms at MATLAB's speed and beyond

Status: planned 2026-09-25; T1 done 2026-09-25, the L=64 scattered rows level with MATLAB;
T2.1-T2.3 done 2026-09-25, the coefficient transforms ten times faster, T2.4 and T2.5 left
since the transforms are no longer what a call waits for; the NFFT of `../nfft` an optional
backend of the SO(3) adjoint, every quadrature row at or below MATLAB with it; T3 done
2026-09-25, the spherical transforms by rows without a table, exact beyond L = 1023; T4 done
2026-09-25, every grid row below MATLAB, with T5.3 (the sparse projection) pulled in and T4.4
(fast ring lengths) left since the ring length defines the grid. Open: the distinct nodes of
a quadratureSO3Grid (`_unique`) take seconds (`docs/research/benchmark-harmonics.md`). T5.1
done (rotate by phases around a compiled d_n(beta)), T5.2 in part (no double realize), T5.3
with T4. Closed 2026-09-25 with the three implementations in one sitting: Python faster on
every row but three, level with MATLAB there. Goal: every row of the table in section 1 faster
than MATLAB's master and than the Julia port, the code still readable - one kernel per transform, the
NumPy forms kept as fallback and oracle as in `mtex/geometry/kernels.py`.

The reference is the speed work of `../master` of 2026-09-24/25 (commits `8e51de07f` to
`3a2c5897b`: `SO3Fun/@SO3FunHarmonic/private/wignerTrafomex.c`, `wignerTrafoAdjointmex.c`,
`wigner_d_quadrant_at_pi_half.c`, `nfftNodesmex.c`, `realDefectmex.c`,
`symmetriseWignermex.c`, `tools/math_tools/circFold.m`, `fftLength.m`, `wignerdHalfPi.m`,
`S2Fun/@S2FunHarmonic/private/sphericalHarmonicTrafo*.cpp`). Design from them, never
transcribe. The Julia port has not moved since 2026-09-14; its numbers are
`../mtexJulia/docs/research/benchmark-harmonics.md`.

## 1. Baseline

`bench/harmonics_user.py` and `bench/matlab/bench_harmonics.m` (one tag per bridge call),
2026-09-25, 10 physical / 12 logical cores of a hybrid CPU, times in ms. The 1e6 node
rows vary by up to four times between runs under load; every step re-measures both sides
in one sitting.

| case | Python | MATLAB | Julia |
|---|---:|---:|---:|
| SO3 eval L=64, 1e6, 1 / m-3m | 1280 / 1666 | 351 / 232 | 270 |
| SO3 eval L=128, 1e4, 1 / m-3m | 1364 / 495 | 326 / 101 | |
| SO3 eval L=128, 1e6, 1 / m-3m | 2467 / 1567 | 660 / 397 | |
| SO3 quadrature L=64, 1e6, 1 / m-3m | 1626 / 2635 | 292 / 247 | 600 |
| SO3 quadrature L=128, 1e4, 1 / m-3m | 2051 / 920 | 321 / 161 | |
| SO3 quadrature L=128, 1e6, 1 / m-3m | 3710 / 2220 | 1170 / 665 | |
| SO3 Clenshaw-Curtis L=127, 1/1, eval / quadrature | 1842 / 2525 | 506 / 587 | |
| SO3 Clenshaw-Curtis L=127, 432/1 | 509 / 848 | 113 / 210 | |
| SO3 Clenshaw-Curtis L=128, 622/222 | 224 / 275 | 53 / 88 | |
| SO3 symmetrise L=128, 622/222, 432/mmm, 432/1 | 23 / 25 / 18 | 15 / 24 / 22 | |
| SO3 isReal L=128 | 36 | 3.6 | |
| SO3 rotate L=128 / L=64 | 185 / 28 | 125 / 42 | |
| S2 eval / quadrature L=128, 1e6 | 197 / 170 | 98 / 130 | |
| S2 Clenshaw-Curtis eval L=256 / 512 / 1024 | 85 / 3071 / > 26000 | 28 / 83 / 517 | |
| S2 quadrature of a callable L=256 | 100 | 19 | |
| S2 symmetrise m-3m L=128 | 2813 | 24 | |
| S2 calcDensity 1e5 | 30 | 17 | |

Where the time goes in Python (SO3, 1e6 nodes, L=64): `quat2euler` 100 ms, the two finufft
plans 160-220 ms although one call uses one, the NUFFT itself 120-460 ms depending on the
thread count and the load; at L=128 the serial Wigner transforms, 419 / 569 ms. On the
sphere above L=256 the half-pi recursion streamed in NumPy per degree; `S2FunHarmonic.symmetrise`
rebuilds the Wigner matrices of every group element in Python on every call.

## 2. Decisions

- **finufft stays the dependency; the NFFT of `../nfft` is an optional backend.** finufft has
  no real valued transform, no pruned FFT and no precomputed window; the NFFT branch
  (`NFFT_REAL`, `NFFT_PRUNED_FFT`) computes the adjoint 15 to 50 % faster and the evaluation
  level. It is GPL and has no Python package, so it is used, through ctypes
  (`mtex/functions/nfft.py`), where the preference `NFFTLibrary` or `MTEX_NFFT_LIBRARY` points
  at a `libnfft3_omp` that computes real plans, for the SO(3) adjoint at the default
  tolerance; everything else stays on finufft (decided 2026-09-25).
- **One thread count.** A preference `threads`, default the physical cores (read from
  `/sys/devices/system/cpu/*/topology/core_cpus_list`, `os.cpu_count()` where that is
  missing), passed to finufft (`nthreads`), `scipy.fft` (`workers`) and numba
  (`set_num_threads` inside the calls). Twelve threads on ten cores made finufft warn and
  run slower; eight beat ten at times on the hybrid CPU, which the benchmark decides.
- **The kernels live in `mtex/functions/kernels.py`**, compiled on first call, NumPy forms
  beside them as fallback and oracle, `setMTEXpref('kernels', 'numpy')` selecting them - the
  pattern of the geometry, maps and plotting layers. `transforms.py` keeps the plans and the
  routing and loses its inline loops.
- **Every step is exact against the previous code**: the coefficient transforms and the
  grid routes to 1e-13 relative, the NUFFT routes to finufft's tolerance against the direct
  sum, on the symmetry pairs 1/1, m-3m/1, 432/mmm, 622/222, 321/2, -3m/1, real and complex,
  even and odd bandwidths, bandwidths 0 to 3 included. The tests go into
  `tests/test_transforms.py` (and `test_so3fun.py`, `test_s2fun.py`); per step only the
  test files of the changed modules and their importers run.

## 3. Steps

### T1 - the cheap wins

1. `threads` preference as above, `threads()` in `geometry/kernels.py`; `_options` in
   `transforms.py` reads it.
2. Nodes in one kernel (`nfftNodesmex`): quaternions to the folded nodes
   `(rA gamma, rB alpha, beta) mod 2pi` and the phase `e^{-i s x}` in one parallel loop,
   directions to `(-rho mod 2pi, theta)` likewise; a non finite node becomes node 0 with
   phase 1 (MATLAB `40fbc7212`). Replaces `_euler`, the `np.mod`s and the `exp`.
3. A plan builds the direction a call uses: `wignerSeries` builds type 2 only,
   `wignerSeriesAdjoint` type 1 only; a kept `SO3FourierPlan` / `S2FourierPlan` builds each
   direction on first use.
4. The oversampling factor by density, MATLAB's `nfftParameters` rule measured on finufft:
   `upsampfac = 1.25` when the nodes are fewer than a quarter of the lattice, 2 otherwise
   (L=128: 920 -> 650 ms; L=64 at 1.25: 143 -> 533 ms, so the rule matters both ways).
   Checked against the direct sum at 1e-7 relative.
5. `workers=` on every `scipy.fft` call of the grid plans.
6. `isRealWigner`, `isRealSpherical` in one compiled pass (`realDefectmex`): the sum of
   `|f - s conj(f_mirror)|^2` against `|f|^2`, no gather.
7. `sphericalProjection` through the held projection of `harmonics.py`: the proper group's
   one sided projection, cached by the group's identity, and for an improper group on the odd
   degrees `2|H|/|G| P(H) - P(K)`, H the proper elements, K the rotation parts of all
   (MATLAB `f22a7d224`). 2.8 s -> milliseconds.

Gate: the eval and quadrature rows at L=64 within a factor 1.5 of MATLAB, S2 symmetrise
below it.

### T2 - the Wigner transforms as parallel kernels

The O(L^4) of the pipeline; at L=128 it alone exceeds MATLAB's whole call.

1. **The half-pi quadrant.** `d_n^{jk}(pi/2)` for `0 <= j, k <= n` by the three term
   recursion in the degree, one degree from the two before, as a compiled loop over rows in
   parallel; the other quadrants by the signs `(-1)^{n+j}`, `(-1)^{n+k}`. A quarter of the
   table and of its memory traffic; the table cached to `HALF_PI_CACHE_MAX` as quadrants,
   streamed above.
2. **Forward** `G[l, k, j] = sum_n c_n^{kl} d_n^{jk} d_n^{jl}`: parallel over j, degrees in
   blocks of four so that a row of G stays in cache across a block, the innermost loop over k
   with the coefficient row and the quadrant row read in order (the present loop over j reads
   `d[j, k+n]` with a stride of `2n+1`). `sqrt(2n+1)` and `i^{k-l}` applied as the terms are
   formed, the plane l = 0 halved for a real function, the mirror
   `G[l, k, -j] = (-1)^{k+l} G[l, k, j]` written in the same kernel: no `_blockPhase`
   array per degree, no copy of a coefficient block, no pass over the whole array.
3. **Adjoint** the same shape, parallel over degrees or over rows of a degree: reads the
   NUFFT output as it is (half lattice, folded), applies the phase and the normalisation in
   the pass, fills the conjugate half of a real function in the kernel.
4. **Two fold axes perpendicular to z.** A group with a two fold axis along y (along x with
   the sign `(-1)^{n+k}`) has `fhat_n^{-k,l} = (-1)^n fhat_n^{k,l}`: the forward reads only
   k >= 0 (l >= 0 on the other side), the adjoint computes half and mirrors (MATLAB
   `62a24ddb6`). The flag comes from the frame's group next to the z fold of `folds`.
5. **No projection where the folds are the whole group**: after the adjoint of a cyclic
   group about z the coefficients are already invariant, the constructor skips `project`.

Gate: the coefficient transforms at L=128 below 60 ms on the cores, both SO3 eval rows and
the quadrature rows at or below MATLAB, the NumPy oracle equal to 1e-13.

### T3 - the sphere at large bandwidth

1. The spherical transform pair as kernels on the quadrant of T2.1: forward
   `G[m, j] = sum_l c_l^m d_l^{jm} d_l^{j0}`, `d_l^{j0}` zero for odd `l + j`, parallel over j,
   the real half and the mirror inside, as in T2.2 and T2.3.
2. Beyond L of about 1000 the frame values `sqrt(binom(2n, a)) 2^{-n}` underflow in double.
   First a test that the transform at L=1100 matches the direct sum at a few nodes (it
   probably does not today); then MATLAB's cure (`a91aef87f`): an entry that starts below
   2^-1000 is carried as mantissa and binary exponent until it has grown above, checked every
   16 degrees, in the band around the diagonal where that happens.
3. The callable quadrature, `calcDensity` and `S2FunHarmonic` of a grid follow from T3.1
   and T4.

Gate: L=512 and L=1024 Clenshaw-Curtis rows below MATLAB; S2 quadrature of a callable at
L=256 below 19 ms.

### T4 - the Clenshaw-Curtis grids

The grid is the fundamental region of the z folds, so its FFTs have the lengths of the
grid and every frequency lands at its index modulo that length (MATLAB `circFold`,
`d579e9b99`, `37493a318`, `86073c4db`).

1. **Evaluate**: beta first on the Fourier array itself, zero padded to 4L and cut to the
   2L+1 grid rows, then alpha on the occupied planes, then gamma as a Hermitian transform
   (`scipy.fft.irfft` family, output real) for a real function - no zeroed lattice of
   `(2L+2)^2 4L` numbers (532 MB at L=127), no `2 * real` pass. Frequencies beyond the grid
   length summed into their residue (a grid coarser than the bandwidth).
2. **Adjoint**: the inverse, each inverse FFT keeping only the orders the Wigner adjoint
   reads before the next (of a real function only `k <= 1`), beta last, straight into the
   folded layout of T2.3.
3. The same for `S2GridPlan`: theta first cut to `[0, pi]`, then rho, Hermitian for a real
   function.
4. `fftLength`: `quadratureSO3Grid` and `quadratureS2Grid` take a ring length of at least
   2L+2, divisible by the folds, without prime factors above 13 (`scipy.fft.next_fast_len`
   plus the fold condition); L=96 gives 194 = 2 * 97 today. The weights follow the length.

Gate: every Clenshaw-Curtis row below MATLAB.

### T5 - the remaining operations

1. `rotate`: `conj(D_n) F = diag(e^{ik alpha}) d_n(beta) diag(e^{il gamma}) F`, the phases
   as row and column scalings and one real matrix product per degree, `d_n(beta)` by the
   recursion (MATLAB `3a2c5897b`). Below MATLAB at L=128.
2. `SO3FunHarmonic` construction: trim zero degrees from the top without computing the power
   twice, no copy for an unchanged bandwidth (MATLAB `86073c4db`); `realizeWigner` compiled
   with T1.6.
3. SO3 `symmetrise` is level with MATLAB; `symmetriseWignermex` visits only the nonzeros of
   the projection blocks, which matters for the dihedral groups. Take it only if the
   benchmark after T2 shows the projection above 10 % of a caller.

## 4. Closing

After each step: `bench/harmonics_user.py` and `bench/matlab/bench_harmonics.m` in one
sitting, the table appended to `docs/research/benchmark-harmonics.md` with the date, the
row of `docs/research/time-critical.md` updated. After T5 the Julia rows are re-measured
(`julia --project=../mtexJulia/dev`) so that the claim "faster than both" stands on numbers
of the same day.
