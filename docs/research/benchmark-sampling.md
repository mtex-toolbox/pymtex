# Benchmark F6 extension - optimal sampling and clustering, against MTEX.jl

Run 2026-09-22 on one machine, twelve cores, both sides warm and run one after the other on
a quiet machine: Python 3.14, NumPy 2, finufft 2.4, `bench/optimalsample.py` and
`bench/cluster.py`; Julia 1.12.7 with twelve threads and six BLAS threads,
`julia -t auto --project=dev` on `bench/optimalsample.jl` and `bench/cluster.jl`. Best of
three after a warm-up run. The Julia numbers are what the port takes here today.

## optimalSample

The same targets and the same deterministic starting nodes on both sides, a fixed number
of L-BFGS steps with every tolerance at zero, so that both sides do the same iterations.
The discrepancy before and after is identical to six digits in every row, so the two ports
solve the same problem by the same steps; the columns are the target, the bandwidth L,
the number of nodes M, the steps, whether the weights are optimised.

| target | L | M | steps | weights | Python | Julia | final discrepancy |
|---|---:|---:|---:|---:|---:|---:|---:|
| S2 1 + 0.6 z | 32 | 100 | 20 | no | 0.156 s | 0.239 s | 4.86e-4 |
| S2 1 + 0.6 z | 32 | 100 | 20 | yes | 0.163 s | 0.431 s | 4.81e-4 |
| S2 1 + 0.6 z | 128 | 1000 | 20 | no | 1.56 s | 0.616 s | 1.72e-5 |
| SO3 1 + 0.6 (R z).z | 16 | 100 | 20 | no | 0.140 s | 0.435 s | 2.43e-3 |
| SO3 1 + 0.6 (R z).z | 16 | 100 | 20 | yes | 0.562 s | 0.915 s | 2.39e-3 |
| SO3 222 fibre | 16 | 500 | 20 | no | 0.176 s | 0.172 s | 1.06e-5 |
| SO3 222 fibre | 32 | 500 | 20 | no | 0.602 s | 0.727 s | 3.35e-5 |
| SO3 222 fibre | 32 | 2000 | 50 | no | 1.94 s | 2.02 s | 2.51e-6 |
| SO3 222 fibre | 64 | 500 | 20 | no | 7.36 s | 61.7 s | 4.78e-5 |

Python is at or ahead of Julia on the group and on the small sphere cases, behind on the
large sphere case (bandwidth 128, 1000 nodes), where the Julia plan of the two-dimensional
NFFT with its own window is faster than finufft's at the same size and the derivative
coefficients of the sphere are three ladder products per iteration. The Julia bandwidth 64
row is an outlier of its plan construction at that size, not of its iteration, and was
reproducible over three runs.

## Clustering

Five Gaussian clouds in three dimensions, `classix` on the raw coordinates with radius 0.1
and `minPts` a hundredth of the points, `calcCluster` on the same points as directions with
its defaults (the classix of the axis tensor), and four orientation clouds of a cubic
crystal through the embedding (`calcCluster(o, method='classix', radius=0.1)`). The point
sets are written by the Python script and read by the Julia one.

| case | n | Python | Julia | clusters |
|---|---:|---:|---:|---:|
| `classix` | 10 000 | 0.034 s | 0.052 s | 5 |
| `classix` | 100 000 | 0.121 s | 0.718 s | 1 |
| `classix` | 1 000 000 | 1.09 s | 9.40 s | 1 |
| `calcCluster` directions | 10 000 | 0.068 s | 0.756 s | 4 |
| `calcCluster` directions | 100 000 | 2.10 s | 18.2 s | 4 |
| `calcCluster` orientations | 2 000 | 0.019 s | 0.011 s | 4 |
| `calcCluster` orientations | 20 000 | 0.146 s | 0.181 s | 4 |

At radius 0.1 the clouds of 0.4 standard deviation merge into one cluster above 10 000
points on both sides, the same count, so the two implementations agree on the merge; the
direction case counts four because the clouds at the origin and along the axes fold onto
each other on the sphere.

## Three things found on the way

- finufft starts its thread pool on every execute; on a plan of 100 nodes at bandwidth 32
  that is 23 ms against 0.7 ms of transform. `_threads` in `transforms.py` asks for one
  thread below a work of nodes times modes of 2e8; the S2 100-node case went from 2.83 s
  to 0.16 s, and the plans of the maps and the pole figures are large enough not to be
  touched.
- The FFT of the oversampled lattice dominates the plans on SO(3) as long as the nodes are
  few against the modes. finufft's low oversampling factor 1.25 halves the lattice per
  dimension and costs accuracy below 1e-8, where the quadrature test asks for 1e-10, so it
  is not a default of the transforms: the sampler, whose plan runs four times an
  iteration, asks for `eps=1e-7, upsampfac=1.25` itself, as the Julia plans run at that
  window; the bandwidth 32 row went from 1.58 s to 0.60 s.
- `angularDerivatives` on the sphere and the batch rotation of the embedding were
  Python loops over the degrees and over the orientations; both are now one array
  expression, with the ladders of the degrees cached by bandwidth. The 20 000 embedding
  takes 0.23 s and the orientation clustering went from 1.4 s to 0.15 s.

## Running

```
.venv/bin/python bench/optimalsample.py
.venv/bin/python bench/cluster.py
(cd ../mtexJulia && julia -t auto --project=dev ../pymtex/bench/optimalsample.jl)
(cd ../mtexJulia && julia -t auto --project=dev ../pymtex/bench/cluster.jl)
```

The results land in `bench/out/optimalsample_{python,julia}.csv` and
`bench/out/cluster_{python,julia}.csv`; run the Python side first, it writes the point
sets. The times move by a factor of two under another session's test run.
