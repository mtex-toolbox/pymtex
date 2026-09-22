# E1 on the frame-free layer, 2026-09-21

`bench/g1_layout.py`, one million elements, float64 unless stated, best of three, on the
12-thread machine, NumPy 2.5, single-threaded kernels.

| kernel | ns per element |
| --- | ---: |
| `r * s` | 137 |
| `r * v` | 159 |
| `angle(r, s)` | 135 |
| `dot(r, s)` | 38 |
| `Euler(r)` | 259 |
| `rotation.byEuler` | 472 |
| `matrix(r)` | 245 |
| `rotation.byMatrix` | 1243 |
| `axis(r)` | 104 |
| `homochoric(r)` | 232 |
| `rotation.byHomochoric` | 2836 |
| `unique(r)` | 3011 |
| `mean(r)` | 27 |
| `angle(v, w)` | 135 |
| `v + v` | 4 |
| `cat(r, s)` | 20 |
| float32 `r * s` | 54 |
| float32 `r * v` | 63 |

What the numbers say:

- The `(..., k)` layout is memory bound: the Hamilton product is sixteen multiplications
  and twelve additions per element and runs at 137 ns, so each temporary of the
  `moveaxis` and `stack` form costs a pass over memory. float32 halves the traffic and
  runs 2.5 times faster, which is the Julia finding for a struct-of-arrays store. A
  kernel written on the four columns with preallocated output would be the next step if
  P3 needs it; the layout itself is not the limit.
- `byMatrix` and `byHomochoric` are the slow ones: the first evaluates all four candidate
  quaternions and selects, the second iterates Newton over the whole array twenty times
  when the last elements are still converging. Both are exact and both have an obvious
  vectorised fix (select first, mask the converged), deferred until a consumer needs
  them at this size.
- `unique` is a rounding key through `np.unique` with `axis=0`, which sorts rows; three
  microseconds per element is the sort. MATLAB's `uniquetol` is the same order.
- Indexing and concatenation are NumPy's; nothing on the container side shows up.

Decision for D1: the trailing-axis layout stays. Struct-of-arrays columns are a kernel
internal option, as in Julia, to be taken in G5 if the symmetry reduction asks for it.
