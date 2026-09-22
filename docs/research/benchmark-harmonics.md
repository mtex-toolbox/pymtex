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
