# The geometry benchmark against Julia and MATLAB, 2026-09-21

`bench/geometry.py`, `bench/geometry.jl`, `bench/geometry.m`: the same eleven operations on
one million elements, float64, crystal frame `m-3m`, the misorientations from `m-3m` to
`6/mmm` (24 x 12 = 288 images), best of five after one warm-up call, each language alone on
the machine (12 threads, 30 GB). Python 3.14 with NumPy 2.5 on one thread, once with the
NumPy kernels and once with the numba kernels of `kernels.py` (numba 0.67, one thread);
Julia 1.12.7 with one thread and BLAS pinned to one; MATLAB R2024b on the `frameSymmetry`
branch through the bridge, `maxNumCompThreads` reporting 2.

| operation | NumPy ms | numba ms | Julia ms | MATLAB ms | numba / Julia | numba / MATLAB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `r * s` | 144 | 7.7 | 14 | 20 | 0.6 | 0.4 |
| `r * v` | 133 | 7.9 | 14 | 24 | 0.6 | 0.3 |
| `dot(ori, ori2)` | 637 | 130 | 131 | 124 | 1.0 | 1.0 |
| `dot(ori, ref)` | 382 | 53 | 49 | 115 | 1.1 | 0.5 |
| `dot(mori, mori2)` | 8507 | 858 | 1095 | 2999 | 0.8 | 0.3 |
| `dot(mori, mref)` | 10067 | 688 | 1037 | 675 | 0.7 | 1.0 |
| `project2FundamentalRegion(m)` Miller | 404 | 120 | 122 | 1447 | 1.0 | 0.1 |
| `project2FundamentalRegion(ori, ref)` | 670 | 103 | 115 | 284 | 0.9 | 0.4 |
| `project2FundamentalRegion(ori, ori2)` | 901 | 160 | 181 | 325 | 0.9 | 0.5 |
| `project2FundamentalRegion(mori, mref)` | 9831 | 752 | 1112 | 1042 | 0.7 | 0.7 |
| `project2FundamentalRegion(mori, mori2)` | 6686 | 884 | 1138 | 7711 | 0.8 | 0.1 |

Milliseconds per million elements are nanoseconds per element. `ref` is one orientation,
`ori2` a list of references, one per element; `m` are one million random crystal
directions. The MATLAB `dot(mori, mori2)` row is the two-phase pair reduction of
`orientation/dot`; the Julia `project2FundamentalRegion(ori, ori2)` row is composed in the
script from the batched pair kernel, since the Julia API takes a single reference.

Each script prints a check per row, the mean of the result. The pair rows agree across the
three languages to four digits (`dot(ori, ori2)` 0.9329, `dot(mori, mori2)` 0.9794, the
Miller projection 0.8312, the products 0.424 and 0.500), so they compute the same thing. The
single-reference rows do not agree (0.9634, 0.9675, 0.9807 for `dot(mori, mref)`) and are
not meant to: each language drew its own reference, and the mean reduced dot against a fixed
reference depends on how that reference's 288 images spread over SO(3). MATLAB's
single-reference form was checked against the explicit maximum over `symmetrise(mref)` on
two thousand misorientations and is exact.

What the numbers say:

- **The NumPy column is the execution model.** One operation per pass over arrays that do
  not fit in cache: the 28 flops of the product are 28 passes, at 9.5 ns each on a strided
  column of the `(n, 4)` array, and the search against one reference forms the (n, 288)
  matrix of dots whole, 2.3 GB at one million. `docs/research/benchmark-e2-kernels.md`
  separates the two effects.
- **The numba column is a fused loop over rows** and lands at 0.6x Julia on the products
  and 0.7x to 1.1x on every reduction and projection, on the same `(n, 4)` layout. The
  layout was never the limit once the loop is fused, so D1 stands.
- **The projections are one loop each**: the search and the assembly of the chosen image
  (`projectToRef`, with the conjugation into the axis sector for the identity reference),
  the pair search with its image (`closestImage`, which the pair angle and the mean share),
  and the direction into its sector (`projectDirections`). Before the fold, the assembly
  after the search was NumPy passes of gathers and products and held those rows at 2.4x to
  2.7x Julia.
- **Against MATLAB** the port is now faster or equal on every row; MATLAB's two threads and
  its exact single-reference form keep `dot(ori, ori2)` and `dot(mori, mref)` level.
- Rows outside the table, same run: `project2FundamentalRegion(ori)` without a reference
  99 ns, cubic to cubic 300 ns, `angle(ori, ori2)` 300 ns, `mean(ori)` 567 ns per element.

How to run:

    .venv/bin/python bench/geometry.py                 # optional size as first argument
    ~/.juliaup/bin/julia --project=../mtexJulia/dev bench/geometry.jl
    cd ../frameSymmetry && docs/agents/matlab-bridge/start_session.sh
    docs/agents/matlab-bridge/.venv/bin/python docs/agents/matlab-bridge/matlab_run.py \
      "restoredefaultpath; cd('/home/hielscher/mtex/frameSymmetry'); startup_mtex; addpath('/home/hielscher/mtex/mtexPython/bench'); geometry(1000000)"

The NumPy column is `setMTEXpref('kernels', 'numpy')` before the run, or a venv without
numba. The MATLAB path reset is needed: the saved path on this machine carries `../master`,
whose `@Miller` class shadows the `Miller` function of `frameSymmetry` and fails on
`dispStyle`.
