# Subplan B - the geometry benchmark against Julia and MATLAB

Status: run 2026-09-21 on the eleven rows of `docs/research/benchmark-geometry.md` (the
quaternion products, the reduced dot and the projection into the fundamental region, for
orientations and for cubic to hexagonal misorientations, against one reference and against
a list). The rest of the table in section 2 is still open. `docs/research/benchmark-g1.md`
has the frame-free kernels, Python only.

## 1. Deliverable

`docs/research/benchmark-geometry.md`: one table, the same operations at the same sizes
in Python, Julia and MATLAB, on this machine, with the ratio to Julia. And the three
scripts that produced it, so it can be rerun after every optimisation:

```
bench/geometry.py       Python, prints one line per operation
bench/geometry.jl       Julia, run with julia --project=../mtexJulia/dev bench/geometry.jl
bench/geometry.m        MATLAB, run through the bridge in ../master/docs/agents/matlab-bridge/
```

Templates exist on the Julia side: `../mtexJulia/bench/orientation_dot.jl` with its MATLAB
twin `orientation_dot.m` and the write-up `../mtexJulia/docs/research/benchmark-orientation-dot.md`,
and `../mtexJulia/bench/layout_and_libraries.jl` for E1. Read them first; the MATLAB side
of the dot benchmark is already written and the MTEX.jl names are shown there.

## 2. The operations

One million elements unless stated, float64, cubic `m-3m` with a specimen `222` where a
frame pair is needed, the same seed everywhere so the inputs agree. Best of three.

| operation | Python | Julia | MATLAB |
| --- | --- | --- | --- |
| `r * s` product | `rotation.rand(n) * rotation.rand(n)` | `rand(Rotation, n) .* rand(Rotation, n)` | `rotation.rand(n) .* rotation.rand(n)` |
| `r * v` rotate | `r * vector3d.rand(n)` | `r .* v` | `r .* v` |
| `Euler(r)`, `rotation.byEuler` | | `Bunge.(r)`, `Rotation.(Bunge.(...))` | `Euler(r)`, `rotation.byEuler` |
| `angle(ori, ref)` symmetry reduced, one reference | `angle(os, ref)` | `angle(os, ref)` | `angle(ori, ref)` |
| `angle(ori1, ori2)` symmetry reduced, pairs | `angle(o1, o2)` | `angle(o1, o2)` | `angle(o1, o2)` |
| `dot_outer` 1e4 x 1e3 | `dot(o1[:, None], o2[None, :])` | `dot.(o1, permutedims(o2))` | `dot_outer(o1, o2)` |
| `projectIntoFundamentalRegion(ori)` | | | |
| `symmetrise(ori)` 1e5 | | | |
| `unique(ori)` 1e5 | | | |
| `mean(ori)` | | | |
| `Miller` indices: `hkl(m)` 1e6, `Miller(h,k,l,cs)` | | | |
| `angle(m1, m2)` crystal directions 1e6 | | | |
| `equispacedSO3Grid(cs, resolution=2.5°)` construction | | | |
| `find(grid, ori)` nearest, 1e4 queries | | `nearest(g, o)` | `find(S3G, ori)` |
| `equispacedS2Grid(resolution=1°)` and its `find` 1e5 | | | |
| `crystalFrame('m-3m', ...)` first and repeated construction | | | |
| `symmetry('m-3m')`, `symmetry('-5-32/m')` | | `PointGroup(...)` | `crystalFrame(...)` |
| `import mtex`, `using MTEX`, MATLAB startup | | | |

Report nanoseconds per element for the elementwise rows, milliseconds for the rest.

## 3. How to run each side

- **Python**: `.venv/bin/python bench/geometry.py`. `time.perf_counter`, best of three,
  the array built outside the timed call.
- **Julia**: `julia --project=../mtexJulia/dev bench/geometry.jl`, `@belapsed` from
  BenchmarkTools if it is in `dev`, else `@elapsed` best of three after one warm-up call.
  Julia 1.12.7 is at `~/.juliaup/bin/julia`. The MTEX.jl API is the Julia spelling
  (`Orientation`, `crystalFrame`, `Bunge`, `angle`, `symmetrise`, `equispacedSO3Grid`,
  `nearest`); `../mtexJulia/docs/plans/geometry.md` section 3 has the MATLAB-to-Julia map.
- **MATLAB**: the bridge keeps one session warm. `../master/CLAUDE.local.md` has
  `MATLAB_ROOT` and the three commands; the branch to run against is `../frameSymmetry`
  (start the bridge from that repo root so its `startup.m` is used). `timeit` where a
  call fits, `tic/toc` best of three otherwise. **Ask before starting the bridge or any
  run that could take more than five minutes.** Do not run `runTests`.

## 4. What to do with the numbers

- Record every ratio; a Python operation slower than Julia by more than 5x is a candidate
  for a struct-of-arrays or preallocated kernel (`docs/PLAN.md` D1, E1), one slower than
  MATLAB is a defect, since MATLAB is also array code without a compiled kernel.
- The known slow spots before measuring: `rotation.byMatrix` (evaluates four candidates),
  `rotation.byHomochoric` (Newton over the whole array), `unique` (a sort), the SO3Grid
  `find` (a Python loop per query), `CSL` (a projection per candidate).
- Update `docs/PLAN.md` E1 with the outcome and `docs/plans/geometry.md` with what was
  changed.
