# The tensor routes, measured

## Directions, orientations and maps, 2026-10-09

m-3m stiffness, ms, warm, before / after; measured under another user's load (7 to 9), so the
absolute times are high and the ratios hold.

| route | before | after |
|---|---:|---:|
| `shearModulus(C, h, u)`, 1e6 pairs | 746 | 37 |
| `PoissonRatio(C, x, y)`, 1e6 pairs | 1405 | 70 |
| `linearCompressibility(C, v)`, 1e6 | 85 | 9 |
| `calcTensor(ori, C)`, 1e6 orientations | 397 | 36 |
| `calcTensor(ebsd, C1, C2, 'all')`, Forsterite, 246 thousand pixels | 1273 | 76 |
| `ori * C`, 1e6 | 260 | 128 |
| `quat2mat`, 1e6 | 125 | 9 |

Already where they should be: `YoungsModulus` and `directionalMagnitude` (4 ms at 1e6, the laptop's
`directionalPolynomial`), `birefringence` (7 ms), `calcTensor(odf, C)` (9 ms). At memory speed and
left: `colon` and `EinsteinSum` of 1e6 turned tensors with one (60 ms, 648 MB read), `inv` of 1e6
(101 ms, compiled), `velocity` at 1e6 directions (118 ms, the compiled eigen decomposition).

- **One tensor contracted with lists of vectors** went through einsum's broadcasting loop, 350 ms
  per contraction at 1e6, and einsum's `optimize` does not help with the ellipsis. `EinsteinSum`
  hands such a contraction, every index of the one tensor summed with its own vector list (a trace
  in the tensor taken first), to a compiled multilinear form (`multilinear` in tensors/kernels.py):
  per row the nonzero coordinates times the components their digits pick, 21 of 81 for a cubic
  stiffness.
- **The mean over orientations** formed every turned tensor, 648 MB at 1e6, and reduced them;
  `turnSum` turns the one tensor by each quaternion, the matrix formed in the loop, and adds it into
  its block's row. Over a map the Reuss average is the inverse of the mean of the turned inverses,
  since turning commutes with inversion: one inversion a phase instead of one a pixel.
- **`quat2mat`** wrote nine strided arrays; above 16384 quaternions one compiled loop writes each
  matrix once. The weighted mean of a tensor list is one product with the weights, as NumPy
  reduces the long axis of a tall array slowly.
