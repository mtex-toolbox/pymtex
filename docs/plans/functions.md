# Subplan P2 - functions on the circle, the sphere and SO(3)

Status: F0 to F7 done 2026-09-21, the subplan complete (`docs/design/harmonics.md`, `mtex/functions/harmonics.py`
with the held projection, `kernelfunctions.py`, `quadrature.py`, `transforms.py`, `so3fun.py`,
`s2fun.py`, `s1fun.py`, `funops.py`, `rbf.py`, `density.py`, `handles.py`, `bingham.py`,
`characteristics.py`, `so3vectorfield.py`, `s2vectorfield.py`, `s2axisfield.py`, `funcs.py`;
finufft added as the dependency of D6; the E3 numbers are `docs/research/benchmark-harmonics.md`;
the least squares of `interpolate` run on `scipy.sparse.linalg.lsqr`, decision 7 needing no
own solver; `optimalSample` and the CLASSIX clustering of Julia are deferred until a caller
needs them). What is left of P2 beyond the subplan: the MATLAB `check_*` tests not yet
mirrored, and the pole figures and tensors of P4 (`docs/plans/functions.md` is to grow a P4
section). Starts from the Julia subplan
`../mtexJulia/docs/plans/functions.md` (F0 to F7 all done there) and the conventions
`../mtexJulia/docs/design/harmonics.md`. Syntax from the MATLAB branch `../frameSymmetry`
(`S1Fun/`, `S2Fun/`, `SO3Fun/`, the kernel folders). The geometry layer P1 is the base;
the quadrature grids deferred from G6 land here.

## 1. Deliverable

`mtex.functions` on top of P1, in float64 and float32:

- harmonic functions on S1, S2 and SO(3): evaluation at arbitrary points (fast for many
  points), coefficients from samples by exact quadrature or least squares, and the algebra
  - sum, product, scalar multiple, convolution, rotation, inversion, mean, norm, gradient,
  Laplacian, Radon transform (`pdf`, `ipdf`, `radon`), sections;
- kernels on S1, S2 and SO(3) as parameter objects with derived coefficients, halfwidths,
  derivatives, convolutions and Radon transforms;
- the radial basis `SO3FunRBF` and the fibre `SO3FunCBF`, with `calcDensity` estimating an
  ODF from orientations and a density on the sphere from directions and axes;
- handles, sums of heterogeneous functions (`SO3FunComposition`), Bingham distributions on
  both spaces;
- texture characteristics: entropy, texture index, volume, fibre volume, local maxima and
  components, misorientation distribution, M-index, angle and axis distributions, sampling;
- vector fields on S2 and SO(3) with gradient, curl and divergence, and axis fields;
- the MATLAB `check_*` function tests ported and passing, plus the transform tests MATLAB
  has no counterpart for.

Everything a function returns is in the P1 types: samples come back framed, extrema as
`Miller` or `orientation`, symmetries are the groups of the frames.

## 2. Design

The mathematics is stated first, the MATLAB workarounds named and dropped, the flags become
fields. The conventions H1 to H7 of the Julia `harmonics.md` are ported verbatim into a new
`docs/design/harmonics.md` (Varshalovich Wigner-d, Condon-Shortley spherical harmonics, the
layout with `deg2dim`, the series with rows the specimen side and columns the crystal side,
the group projections as the mean of `D_n(s)^T` over the proper group, the `pi/2` Fourier
identity the transforms rest on). They transfer unchanged; only the code spelling differs.

### 2.1 What transfers from Julia unchanged

- The whole F0 to F7 order of work and the file split (section 3).
- The decided numerics of the Julia plan sections 9 to 15: the half-pi table cached to a
  bandwidth ceiling; real and antipodal by the half array; the NFFT oversampling as a cost,
  not a bandwidth; the symmetry projection held in its support and folded into the adjoint;
  the equispaced FFT on the quadrature grid as the exact route; kernel selection and
  specimen centering. These are read from the Julia plan and re-measured in E3, not
  redesigned.
- The type story: a value type (real vs complex), an exchange parameter on SO(3), a domain
  element (direction vs axis) on S2 - in Python these are fields and properties, not type
  parameters (D2).

### 2.2 Types (Python spelling)

MATLAB class names, MATLAB casing. Julia's type parameters become fields, as in P1.

| type | Julia parameters -> Python |
|---|---|
| `SO3FunHarmonic` | Wigner series; coefficient dtype carries real vs complex (`isReal` a property); `antipodal` a field for grain exchange (ADR 0008); `CS`, `SS` the frames |
| `S2FunHarmonic` | spherical harmonic series; `antipodal` a field selecting the axis domain (even degrees only); one frame or none |
| `S1FunHarmonic` | Fourier series on the circle |
| `SO3FunRBF` | `c0 + sum w_i psi(., o_i)`, centres an `orientation` list, one kernel |
| `SO3FunCBF` | fibre components `sum w_i psi(<g h_i, r_i>)`, one S2 kernel |
| `SO3FunHandle`, `S2FunHandle`, `S1FunHandle` | a callable with frames and a bandwidth hint |
| `SO3FunComposition`, `S2FunSum` | sum of heterogeneous functions (MATLAB's `SO3FunComposition`); what `+` returns when coefficients cannot be added |
| `SO3FunBingham`, `S2FunBingham` | the Bingham distributions |
| `SO3VectorFieldHarmonic`, `S2VectorFieldHarmonic`, `...Handle`, `SO3VectorFieldRBF` | vector fields, three coefficient columns, the tangent side of ADR 0010 |
| `S2AxisFieldHarmonic` | axis valued fields |
| `SO3DeLaValleePoussinKernel`, `S2...`, `S1...` and the other kernels | parameter objects, `A_n` derived, MATLAB names (with the `SO3`/`S2`/`S1` prefix); `SO3Kernel(A)` and its siblings are the kernels given by coefficients |
| `quadratureSO3Grid`, `quadratureS2Grid`, `quadratureS1Grid` | nodes and weights exact to a degree, merged over orbits |
| `SO3FourierPlan`, `S2FourierPlan`, `S1FourierPlan` | the finufft plan for a node set and bandwidth, reusable across coefficient vectors |

Every function is callable: `f(ori)` for one orientation, `f(oris)` for a container (the
fast path), through `__call__`. MATLAB's `eval` stays as the method `f.eval(ori)`; there is
no module-level `eval` (it shadows the builtin). Frames are `frameA` (crystal, acts on the
right) and `frameB` (specimen, left), the accessors `f.CS`, `f.SS` as on `orientation`.

`isReal` and `antipodal` are MATLAB properties, so Python properties, but they are enforced
by construction, not numerically detected with a tolerance: a function built from real
coefficients is real, an axis function has even degrees only, an exchange function has
`fhat_n = reverse(fhat_n)^T`.

### 2.3 Dropped from MATLAB (as Julia dropped them)

Option strings and `dynOption` (the `nfsoft`/`noNFFT`/`direct`/`gridded` choices become one
cost rule); the `createPlan`/`keepPlan`/`killPlan` persistent caches (a plan is a value the
caller holds); `subsref`/`subsasgn`, `rotate_outer`, `reshape`/`cat` overloads; numeric
detection of `isReal` and `antipodal`; the point-group-id branches of
`symmetriseWignerCoefficients`/`symmetriseFourierCoefficients` (derived from the elements);
the NFSOFT, NFSFT and FPT paths and every `evalNFSOFT`/`adjointNFSFT` (the coefficient
transform plus a plain NUFFT is the only route, so no compiled spherical-transform binary);
`WignerD` by matrix exponential and by spherical harmonics (one recursion); `cs.fhatRef`.

Not ported in P2 (same as Julia): `SO3FunMLS`, `S2FunMLS`, `SO3FunHomochoric`, `SO3FunSBF`,
`doEulerStep`, `S2FunTri` and the triangulated fields (need P3's triangulation),
`S2FunGrid`, the precomputed optimal S2 quadrature designs, `export`, all plotting.

## 3. Order of work

Each step ships one concept per file, one `tests/test_<file>.py`, and updates this status
line. The Julia step of the same name is the design; the MATLAB `check_*` tests are the
oracles, run through the bridge where a number must be pinned.

- **F0 - conventions and the Wigner machinery.** `docs/design/harmonics.md` (H1 to H7) and
  `mtex/functions/harmonics.py`: `deg2dim`, `dim2deg`, `degreeRange`, `degreeBlock`;
  `wignerd` (all degrees at once by the three-term recursion in `n`, the Jacobi closed form
  as oracle and outer frame); `WignerD` for a rotation, a group and a frame (the projection
  `P_n = mean of D_n(s)^T`); `wignerSeries` the direct sum (reference for F2);
  `normalizedLegendre`, `sphericalY`; `clebschGordan`, `wigner3j`, `clebschGordanTensor` by
  the Racah formula in exact arithmetic. Tests: the Jacobi oracle to degree 12,
  orthonormality of both bases by product quadrature, invariance of a projected function
  under every element of several groups in closure order.
- **F1 - kernels.** `mtex/functions/kernelfunctions.py` (MATLAB's folder name, since `kernels.py` is the geometry's compiled loops): the base kernels per domain and one class
  per family (`SO3DeLaValleePoussinKernel`, `SO3AbelPoissonKernel`,
  `SO3GaussWeierstrassKernel`, `SO3vonMisesFisherKernel`, `fibreVonMisesFisherKernel`,
  `SO3DirichletKernel`, `SO3BumpKernel`, `SO3LaplaceKernel`, `SO3SobolevKernel`,
  `SO3SquareSingularityKernel`, `SO3RestrictedDistanceKernel`, the S2 and S1 members, the
  coefficient kernels `SO3Kernel(A)`/`S2Kernel(A)`/`S1Kernel(fhat)` as the base classes, the
  handles, `S1GaussKernel`, and `radon` mapping an SO(3) kernel to its S2 kernel). `calcFourier(psi, L)` from
  the family rule, never stored; `bandwidth` by MTEX's cutoff; `psi(t)` by closed form or
  Clenshaw; `halfwidth` and construction from a halfwidth by bisection; `derivative`, `conv`,
  `radon`, `mean`, `norm`. The square-singularity coefficients use the stable backward
  Legendre-Q recurrence. Dropped: `SO3RieszKernel` (no consumer), Schulz defocusing (P4),
  the GBND kernel (P3). Tests K1 to K6.
- **F2 - transforms and quadrature.** `mtex/functions/transforms.py`: the coefficient
  transforms of H7 (`wignerTransform`, `sphericalTransform`, the circle) and their adjoints,
  the half-pi Wigner-d tables cached per precision to a bandwidth ceiling, the plans wrapping
  finufft (D6) on the nodes `(beta, alpha, gamma)/2pi`, `(theta, -rho)/2pi`, `-x/2pi`, the
  direct sums with the small-node cost rule. `mtex/functions/quadrature.py`: `gaussLegendre`,
  `clenshawCurtis`, `legendreP`, and `quadratureSO3Grid`/`quadratureS2Grid`/`quadratureS1Grid`
  exact to a degree, the z-fold reduction, nodes merged over orbits. The equispaced FFT on
  the quadrature grid is the exact route (Julia section 13). Tests T1 to T6, and E3.
- **F3 - harmonic functions.** `mtex/functions/so3fun.py`, `s2fun.py`, `s1fun.py`: the
  harmonic types, construction from coefficients, from a kernel (the diagonal embedding),
  from a callable by quadrature, from weighted samples by the adjoint; callable on rotations,
  orientations and vectors in their frames and containers; `symmetrise` by the projections;
  `inv`, `withExchange`, `conj`, `real`, `truncate`, the algebra, `mean`, `norm`, `dot`;
  `funops` - `rotate` on either side, `conv` in the SO(3)xSO(3), SO(3)xkernel, SO(3)xS2,
  S2xS2, S2xkernel pairings, `grad`, `laplace`, `pdf`/`ipdf`/`radon`, the Funk transform on
  S2, the pointwise `*`,`/`,`**`,`sqrt`,`log`,`exp`,`abs`,`max` by quadrature at the needed
  bandwidth. Tests F3a to F3j (the MATLAB `check_S1Fun`, `check_SO3FunRotate`, `check_odfGrad`
  on harmonics, `check_radonOptions` without option strings, the symmetrise test).
- **F4 - radial basis and density estimation.** `mtex/functions/rbf.py`: `SO3FunRBF` (the
  kernel sum over the reduction groups, the exact harmonic form as the adjoint of the point
  measure convolved with the kernel, the gradient, the pole figure), `SO3FunCBF`,
  `uniformODF`, `unimodalODF`, `fibreODF`, `calcDensity` on orientations and on
  directions/axes, `calcPDF`, `interpolate` by least squares with the Sobolev penalty through
  `scipy.sparse.linalg.lsqr` (or an own Paige-Saunders if the divergence of the open item
  below needs it). Tests R1 to R7.
- **F5 - handles, sums, Bingham.** `mtex/functions/handles.py` (handles and
  `SO3FunComposition`/`S2FunSum`) and `mtex/functions/bingham.py` (`mhyper` by the
  Hillier-Kan-Wang series, `SO3FunBingham`, `S2FunBingham`, `calcBinghamODF` with the large
  concentration approximation and the exact Newton). Tests H5a to H5c, B1 to B4, pinned
  against MATLAB through the bridge.
- **F6 - texture characteristics.** `mtex/functions/characteristics.py`: `textureIndex`,
  `entropy`, `volume`, `fibreVolume`, `max`/`min` derivative free, `calcComponents` by
  steepest ascent, `calcMDF`, `calcAngleDistribution`, `calcAxisDistribution`, `calcMIndex`,
  `discreteSample`, `optimalSample`. Tests C1 to C6.
- **F7 - vector and axis fields.** `mtex/functions/so3vectorfield.py`, `s2vectorfield.py`,
  `s2axisfield.py`: one coefficient derivative for grad/div/curl/antiderivative, the storage
  side of ADR 0010, `left`/`right`, `symmetrise` by the projection, `rotate`, `inv`, the
  algebra, `SO3VectorFieldRBF`, the sphere gradient, `S2AxisFieldHarmonic`. Tests V1 to V9.

## 4. MATLAB to Python

| MATLAB | Python |
|---|---|
| `SO3F.eval(ori)`, `eval(SO3F, ori)` | `SO3F(ori)`, `SO3F.eval(ori)`; a quadrature grid as the argument takes the FFT route |
| `SO3FunHarmonic(fhat, cs, ss)`, `SO3FunHarmonic(psi, cs, ss)`, `SO3FunHarmonic(@fun, cs, ss)` | the same three, the callable by exact quadrature on the folded rule, `bandwidth=` on all |
| `SO3F.isReal`, `SO3F.antipodal` | `SO3F.isReal`, `SO3F.antipodal` (by construction; `real=`, `antipodal=` keywords, `symmetrise=False` to skip the projection) |
| `SO3FunHarmonic.quadrature(nodes, values, 'bandwidth', L, 'weights', w)` | `SO3FunHarmonic.quadrature(nodes, values, bandwidth=L, weights=w)`, `approximate` its alias |
| `calcFourier(SO3F, 'bandwidth', L)` | `SO3FunHarmonic(SO3F, bandwidth=L)`, `SO3F.truncate(L)` |
| `rotate(SO3F, rot, 'right')` | `rotate(SO3F, rot, 'right')`, `SO3F.rotate(rot, 'right')`; a bare rotation keeps a frame only when it normalises the group, an orientation moves the frame |
| `conv(SO3F, psi)`, `conv(SO3F, sF)`, `conv(sF, sF2)`, `conv(SO3F, A)` | the same pairings, `SO3F.conv(x)` |
| `laplace`, `norm(SO3F)` | `laplace(SO3F)`, `norm(SO3F)`, `SO3F.norm(s)` the Sobolev norm |
| `SO3F.grad`, `SO3F.grad(ori, 'right')` | `grad(SO3F)` a minimal `SO3VectorFieldHarmonic` (F7 extends it), `grad(SO3F, ori, 'right')` the tangent vectors of P1 |
| `S1F.diff`, `calcDensity(angles)` | `S1F.diff()`, `grad(S1F)`; `calcDensity(rho, sigma=, kernel=, weights=, antipodal=)` on angles |
| `S2F * S2F`, `sqrt(S2F)`, `SO3F.^2`, `1./SO3F` | the operators, `f.sqrt()`, `f.exp()`, `f.log()`, `abs(f)`, `f ** p`, `1 / f`, all by `mapFun` on the quadrature grid |
| `'createPlan'`/`'keepPlan'`/`'killPlan'` | `p = SO3FourierPlan(ori, L)`, `p(fhat)`, `p.adjoint(values, weights)` |
| `'nfsoft'`/`'noNFFT'`/`'direct'`/`'gridded'` | gone, one cost rule: the direct sum below 50 nodes, the FFT on a Clenshaw-Curtis grid, finufft elsewhere |
| `quadratureSO3Grid(N, CS, SS, 'GaussLegendre')`, `[SO3G, W]` | `g = quadratureSO3Grid(N, CS, SS, 'GaussLegendre')`, `g.nodes()`, `g.weights`; the grid is a parameter object, not an orientation |
| NFFT `m`, `sigma` (Julia's `nfftParameters`) | finufft's `eps` (1e-8 by default), the oversampling being finufft's own choice |
| `SO3DeLaValleePoussinKernel('halfwidth', 10*degree)` | `SO3DeLaValleePoussinKernel(halfwidth=10*degree)` |
| `psi.A`, `psi.bandwidth`, `calcFourier(psi, L)` | `psi.A`, `psi.bandwidth` (both derived from the rule of the family, never stored), `calcFourier(psi, L)` |
| `psi.eval(t)`, `psi.halfwidth`, `grad(psi, t)` | `psi(t)`, `psi.eval(t)`, `psi.halfwidth()` (a method in MATLAB's base classes), `psi.grad(t)` |
| `SO3Kernel(A)`, `S2Kernel(A)`, `S1Kernel(fhat)` | the same: the kernels given by coefficients are the base classes, as in MATLAB (not Julia's `ChebyshevKernel`) |
| `calcDensity(ori, 'halfwidth', hw, 'weights', w)` | `calcDensity(ori, halfwidth=hw, weights=w)`, the kernel form up to 200 orientations and the harmonic form above or with `harmonic=True`; `calcDensity(v)` on directions and axes, `calcDensity(rho)` on angles |
| `calcDensity(ori, 'antipodal')` | `calcDensity(ori.withExchange())` |
| `unimodalODF(ori, 'halfwidth', hw)`, `fibreODF(h, r, SS)`, `fibreODF(f)`, `uniformODF(CS, SS)` | the same names, `SO3FunRBF(center, psi, weights, c0)` and `SO3FunCBF(h, r, weights, psi, SS)` behind them; `SO3FunHarmonic(odf, bandwidth=L)` the harmonic form |
| `calcPDF(odf, h, r, 'superposition', c)` | `calcPDF(odf, h, r, superposition=c)`, `odf.calcPDF(h)` |
| `SO3FunHandle(@fun, CS, SS)`, `S2FunHandle(@fun)`, `S1FunHandle(@fun)` | the same, `bandwidth=` the hint the harmonic form uses (the session maximum by default), `symmetrise=True` averaging the callable over the groups on every call |
| `SO3FunComposition(F1, F2)`, `F1 + F2` | `SO3FunComposition`, what `+` returns when coefficients cannot be added; `S2FunSum` on the sphere (no MATLAB class) |
| `BinghamODF(kappa, ori)`, `SO3FunBingham(kappa, A, cs)`, `calcBinghamODF(ori)` | the same; `calcBinghamODF(ori, exact=True)` Newton on the moments; `S2FunBingham(Z, a, cs)`, `S2FunBingham.fit(v)`; `mhyper(kappa)` the exact constant |
| `[v, pos] = max(SO3F, 'numLocal', k)`, `min` | `v, pos = max(odf, numLocal=k, resolution=, accuracy=)`, `odf.max()`; the module `max`/`min` fall through to the builtins on anything else |
| `[modes, weights, centerId] = calcComponents(SO3F, 'seed', ori)` | `modes, weights, centerId = calcComponents(odf, seed=ori, radius=, tol=, exact=)`, `centerId` -1 for a dropped seed |
| `volume(odf, center, radius)`, `fibreVolume(odf, h, r, radius)`, `entropy`, `textureIndex` | the same names, also on the sphere where they apply, and as methods |
| `calcMDF(odf1, odf2, 'bandwidth', L)`, `calcAngleDistribution(odf | cs)`, `calcAxisDistribution(odf, v)`, `calcMIndex` | the same; the angle distribution returns `(omega, density)`, the axis distribution a spherical function or values |
| `discreteSample(odf, n)`, `calcKernel(ori)`, `calcError(odf1, odf2)`, `centerSpecimen(odf)` | `discreteSample(odf, n, rng=)`, `calcKernel(ori, 'klcv')`, `calcError(odf1, odf2, 'l2')`, `centerSpecimen(odf)` returning `(odf, rot, v1, v2)` |
| `SO3VectorFieldHarmonic(fhat, CS, SS, SO3TangentSpace.rightVector)`, `SO3VF.tangentSpace` | `SO3VectorFieldHarmonic(fhat, CS, SS, 'right')`, `SO3VF.tangentSpace`, `SO3VF.left()`, `SO3VF.right()`; the storage side changes exactly by quadrature at the next bandwidth, the outer `tangentSpace` option is gone |
| `SO3VF.curl`, `SO3VF.div`, `SO3VF.antiderivative`, `SO3F.grad` | `curl(SO3VF)`, `div(SO3VF)`, `antiderivative(SO3VF)` (the curl of opposite sign, the antiderivative checked to be a gradient), `grad(SO3F)` a harmonic field or the central difference handle, `grad(SO3F, ori, 'right')` tangent vectors |
| `SO3VectorFieldHandle(fun, CS, SS, ...)`, `SO3VectorFieldRBF(SO3VF)` | the same; `SO3VectorFieldRBF(tangentVectors, kernel=)` fits through given vectors, `SO3VectorFieldRBF(SO3VF, resolution=)` approximates on an equispaced grid |
| `S2VectorFieldHarmonic(fun)`, `S2AxisFieldHarmonic(fun)`, `S2VectorFieldHandle` | the same, `grad(sF)` the surface gradient, `dot`/`cross`/`abs`/`normalize` by quadrature, `thetaVector`, `rhoVector`; the axis field stores the six components of a a^T and evaluates the leading eigenvector |
| `SO3FunHarmonic.interpolate(ori, values)` | `interpolate(ori, values, bandwidth=, lambda_=, s=)`, `SO3FunHarmonic.interpolate`, the same on the sphere |
| `radon(SO3F, h)`, `radon(SO3F, [], r)` | `SO3F.pdf(h)`, `SO3F.ipdf(r)`, `SO3F.radon(h, r)` |
| `symmetrise(SO3F)` | `SO3F.symmetrise()` (coefficient projection) |
| `grad(F, ori, 'right')` | `F.grad(ori, 'right')` |
| `SO3VF.curl`, `.div`, `.antiderivative` | `curl(v)`, `div(v)`, `antiderivative(v)` (curl of opposite sign) |
| `norm(SO3VF)` (pointwise) | `abs(v)`; `v.norm()` is the L2 norm |

## 5. Source layout

```
mtex/functions/harmonics.py       deg2dim, Wigner-d recursion, WignerD, projections, Clebsch-Gordan
mtex/functions/kernelfunctions.py kernel classes, coefficient rules, Clenshaw, halfwidth
mtex/functions/transforms.py      coefficient transforms, finufft plans, direct sums, cost rule
mtex/functions/quadrature.py      quadrature grids on S1, S2, SO(3)
mtex/functions/s1fun.py, s2fun.py, so3fun.py   harmonic types and algebra
mtex/functions/rbf.py             SO3FunRBF, SO3FunCBF, uniformODF, unimodalODF, fibreODF
mtex/functions/density.py         calcDensity on all three spaces, calcPDF, interpolate
mtex/functions/handles.py         handles, SO3FunComposition and S2FunSum, the SO3FunGeneric base of the non-harmonic kinds
mtex/functions/bingham.py         mhyper, SO3FunBingham, calcBinghamODF, S2FunBingham
mtex/functions/characteristics.py
mtex/functions/so3vectorfield.py, s2vectorfield.py, s2axisfield.py
docs/design/harmonics.md          H1 to H7, ported from the Julia design
bench/harmonics.py                E3
```

## 6. Dependencies

- **finufft** (D6, decided in `docs/PLAN.md`): the 2- and 3-dimensional type-2 NUFFT and its
  type-1 adjoint, the evaluation of the Fourier array at arbitrary nodes and the adjoint for
  coefficients from weighted samples; wheels on every platform, threaded. This is the one new
  runtime dependency P2 adds. The spherical and SO(3) transforms are the coefficient
  transforms we write ourselves; finufft never sees a sphere.
- **scipy**: `scipy.special` (Bessel, gamma, beta for the kernels), `scipy.fft` (the
  equispaced route on quadrature grids and the Clenshaw-Curtis weights), `scipy.sparse.linalg.lsqr`
  (least squares for `interpolate`, unless the open item forces an own solver).
- **numpy**: `numpy.polynomial.legendre.leggauss` for the Gauss-Legendre nodes; the
  coefficient transforms are matrix products per degree and stay NumPy.
- **Own, exact**: the Wigner-d recursion at `pi/2` and the Clebsch-Gordan by the Racah
  formula. Test-only oracles: `sympy.physics.wigner` (Clebsch-Gordan, Wigner-3j) and the
  MATLAB bridge, both in the dev group only.

## 7. Decisions to discuss before F0

1. **finufft as the NUFFT** (D6) versus the equispaced `scipy.fft` route as the primary,
   with finufft only for scattered nodes. E3 measures both against the Julia budget below.
   Recommendation: finufft for scattered nodes, `scipy.fft` on the quadrature grids, as Julia
   section 13.
2. **Type parameters as fields** (`isReal`, `antipodal` properties enforced by construction).
   Recommendation: yes, as P1 did with the geometry parameters.
3. **Callable functions**: `f(ori)` through `__call__`, `f.eval(ori)` kept as the MATLAB
   method, no module-level `eval`. Recommendation: yes.
4. **Arrays of functions**: a single function holds a coefficient array with an optional
   leading batch axis (vector-valued functions and batch transforms sharing one plan), the
   `MtexArray` container reserved for the geometric `(..., k)` objects. Alternative: an
   `MtexArray` of function objects. Recommendation: the batch axis on the coefficient array,
   the vector fields the typed exception, matching Julia's "arrays of functions plus batch
   transforms".
5. **Kernel names**: MATLAB's `SO3DeLaValleePoussinKernel` (with the domain prefix) rather
   than Julia's `DeLaValleePoussinKernel{D}`. Recommendation: MATLAB names, since the user
   sees them (CLAUDE.md rule).
6. **`SO3FunComposition` versus `SO3FunSum`**: MATLAB's name is `SO3FunComposition`, Julia's
   is `SO3FunSum`. Recommendation: MATLAB's `SO3FunComposition` for the SO(3) sum, since the
   user sees it; note the Julia design in the file.
7. **`interpolate` least squares**: `scipy.sparse.linalg.lsqr` first; fall back to an own
   Paige-Saunders only if the open item (the divergence on measured grids, Julia section 8)
   reproduces here.

## 8. Experiments this subplan depends on

- **E3 - harmonics** (`docs/PLAN.md`, `bench/harmonics.py`): finufft against the equispaced
  `scipy.fft` route for the SO(3) and S2 evaluation and adjoint at L = 64 and L = 256 with
  1e6 nodes. Budget from the Julia run (`../mtexJulia/docs/research/benchmark-harmonics.md`):
  1e6 orientations at L = 64 in about 0.35 s (NFFT) plus 0.07 s (coefficient transform), 1e6
  directions at L = 256 in about 0.05 s. The write-up is `docs/research/benchmark-harmonics.md`.
- **E4 - import and first-call time**: `import mtex.functions` stays within the P1 budget
  above NumPy and SciPy; finufft imported lazily on the first transform, never at module
  level, as numba is in P1. Measured at the end of each step.

## 9. Left out of P2

Plotting of functions (P5), `export` (P6), the MLS and homochoric representations,
`SO3FunSBF` and `doEulerStep` (P4 with the tensors), `S2FunTri` and the triangulated fields
(P3 with the triangulation), pole figure containers and `pf2odf` (P4), the embedding (P4).

## 9a. F6 extension - optimal sampling and clustering. DONE 2026-09-22

Moved to `mtex/analysis` in the layout of 2026-09-22 (`docs/PLAN.md` section 8).

`mtex/analysis/optimalsample.py`: `optimalSample(f, n | nodes, bandwidth=, maxIter=, tol=,
memory=, method='steepestDescent', optimizeWeights=True, warmUp=, innerIter=, tolWeights=,
minWeight=, diagnostics=True)` on S2 and SO(3) after Julia's `optimalsample.jl`: the kernel
discrepancy of the restricted distance kernel from degree one, the plan following the
nodes by resetting the finufft points and the half array phase, L-BFGS with an Armijo line
search, geodesic steps with parallel transport on the sphere and left exponential steps on
the group, the weights through a softmax moved jointly on the sphere and by the
multiplicative simplex step alternating on the group. MATLAB's two outputs are the keyword:
`optimalSample(f, n)` returns the nodes, `optimizeWeights=True` the pair `(nodes, c)`, and
`diagnostics=True` the result with discrepancy, history, iterations, stop reason. Tests
`tests/test_optimalsample.py`: gradients against finite differences, L-BFGS ahead of
gradient descent, scaling invariance, the moving plan against a fresh one, the symmetric
and antipodal cases, a large sample that keeps optimising. 200 nodes at bandwidth 16 take
4 s for 80 iterations.

`mtex/analysis/cluster.py`: `classix(X, radius, minPts=, mergeScale=)` in NumPy (the
window inner products as one matrix-vector product per group, the merge by union-find,
small clusters dissolved), `knnRadius` by a k-d tree, and `calcCluster(x, method=, radius=,
minPoints=, numCluster=, maxAngle=, halfwidth=, weights=)` returning `(cId, center)` with the
ids counted from zero: `'classix'` on the coordinates, the axis tensor or the embedding
(MATLAB's default for directions), `'odf'` the modes of the density the orientations
ascend to (MATLAB's default for orientations), `'hierarchical'` the closest pair of
centres merged until the count or the angle, a list above a thousand binned on a grid
first, both for directions and orientations. Tests `tests/test_cluster.py`.

Found on the way: an `S2FunHandle` with `antipodal=True` averaged over the group of its
frame on every evaluation, 96 calls per point for a cubic crystal, which made the
composed velocity handles of a page run for minutes; `symmetrise` is now three valued
(`None` averages an antipodal handle, `False` says the callable is symmetric by
construction), the elastic and optical functions and the arithmetic of handles pass
`False`, and a composed velocity handle evaluates on 60 000 directions in 0.35 s.

Benchmarked against Julia 2026-09-22 (`docs/research/benchmark-sampling.md`, `bench/optimalsample.py`,
`bench/cluster.py` and their `.jl` twins): the sampler reaches the same discrepancy to six
digits on both sides and is at or ahead of Julia except on the large sphere case; `classix`
is three to nine times ahead, the orientation clustering level. Found on the way: finufft
starts its thread pool on every execute, 23 ms on a plan of 100 nodes, so `_threads` in
`transforms.py` asks for one thread below a work of 2e8; the low oversampling factor 1.25
halves the lattice on SO(3) but costs the 1e-10 of the quadrature test, so only the
sampler's plans ask for it (`eps=1e-7, upsampfac=1.25`); `angularDerivatives` and the
batch rotation of the embedding are array expressions now.

## 10. P4 - pole figures

Status: written 2026-09-22; P4a, P4b and I6 done the same day (`mtex/functions/polefigure.py`,
`pf2odf.py`, tests P1 to P8 and S1 to S6, 14 tests in 22 s). Starts from the pole figure half of
the Julia subplan `../mtexJulia/docs/plans/polefigures-tensors.md` (P4a, P4b, P4h, P4j done
there) and the MATLAB folder `../frameSymmetry/PoleFigureAnalysis/` (`@PoleFigure`,
`@MLSSolver`, `@zeroRangeMethod`) with the doc chapter `doc/PoleFigureAnalysis/`. The
tensor half (P4c to P4g) is planned separately when a caller asks for it.

### 10.1 Deliverable

The MATLAB chapter *Pole Figures*: a container for measured diffraction intensities with
its arithmetic, corrections and errors; the simulation of pole figures from any SO(3)
function; the reconstruction of an ODF by the modified least squares algorithm with ghost
correction, zero range method and regularisation; the successive refinement. The importers
(P6, I6) and the plots (P5, S6) of the chapter are steps of their own subplans and are
listed there.

### 10.2 Design

The mathematics is Julia's section 2.1 unchanged: the specimen directions are axes unless
the measurement is `complete`; simulation is `calcPDF` on the even part; the inverse
problem is the convolution `Ψ_i c` evaluated through two `S2FourierPlan`s and the kernel
diagonal, the adjoint the same plans in the other order; the projected gradient with the
exact rational line search; ghost correction; the two densities of the zero range method;
the Laplacian regularisation.

Where Python differs from Julia:

- **The data set is one `PoleFigure`.** Julia stores a data set as a `Vector` of single
  pole figures because its vectors broadcast and print; a Python list does neither, so the
  MATLAB container returns: `pf.allH`, `pf.allR`, `pf.allI`, `pf.c` are parallel lists, one
  entry per pole figure, and `pf.select(i)`, `pf[i]`, iteration give Julia's single pole
  figure as a one-entry container. `len(pf)` is `pf.numPF`; `pf[mask]` with a boolean mask
  over the measured directions is MATLAB's `pf(ind)`. What Julia dropped stays dropped:
  `SSprivate` (the frame of `allR` is the specimen frame, the crystal frame that of
  `allH`), `dynProp`, `subsasgn`, `end`, the `bankId` branch.
- **Quadrature weights on scattered points** come from `scipy.spatial.SphericalVoronoi`
  instead of Julia's own hull (P4h): the areas of the Voronoi cells, an axis list doubled
  and halved, duplicates sharing a cell, no cell above twice the 80 % quantile. A product
  grid in `theta x rho` gets the exact interpolatory weights in `cos theta` and the
  trapezoidal weights on the rings. `method='harmonic'` is the least norm correction.
- **Names are MATLAB's**: `calcQuadratureWeights`, `zeroRangeMethod` with
  `checkZeroRange`, `MLSSolver` with `init`, `initIter`, `doIter`, `calcODF`, `union` for
  Julia's `merge`, `SchulzDefocusingKernel`; the flags `zeroRange=True`,
  `noGhostCorrection=True`, `quadratureWeights=True`, `intensityWeights=True`,
  `nothinning=True`, `silent=True`; the error modes as positional strings,
  `calcError(pf, odf, 'RP')`. Structure coefficients default to `1/n` per superposed
  direction (Julia's rule, MATLAB's `calcPoleFigure`), so a superposed pole figure is a
  pole density.
- **`calcODF` returns the ODF alone**; the scaling factors are `solver.alpha` or
  `calcNormalization(pf, odf)`. Printing is off unless `silent=False`.
- **`calcODFIterative`** starts each level from the previous ODF evaluated on the grid
  (Julia), since the `mlsq`/`density` route of `SO3FunRBF.interpolate` is not ported.

### 10.3 Order of work

- **P4a - the container. DONE 2026-09-22.** `mtex/functions/polefigure.py`: `PoleFigure` with its
  constructor from one or several crystal directions, specimen directions and intensities,
  the frames read off the directions or given positionally; the accessors; the arithmetic
  against numbers, per-pole-figure factors and other pole figures; `max`, `min`, `mean` by
  quadrature, `normalize` against one and an ODF; `rotate` dropping the group the rotation
  breaks; `unique`, `union`, `isOutlier`, `noisepf`, `correct` with the spherical or the
  polar spline interpolation; `calcPoleFigure` from any SO(3) function (a Miller list one
  per element, a Python list one per entry, an entry of several a superposition), also as a
  method of the functions and of a pole figure; `calcNormalization`, `calcErrorPF`,
  `calcError` (dispatching to the ODF version of F6); `calcQuadratureWeights`; `refine`;
  `SchulzDefocusingKernel`; `S2FunHarmonic(pf)` through the `_harmonicS2` hook. Tests
  `tests/test_polefigure.py` P1 to P8 after Julia's.
- **P4b - the inversion. DONE 2026-09-22.** `mtex/analysis/pf2odf.py`: the operator of one pole figure,
  `zeroRangeMethod`, `MLSSolver`, `calcODF`, `calcODFIterative`. Tests
  `tests/test_pf2odf.py` S1 to S6 after Julia's: the operator against the pole figure of
  the RBF, the adjoint identity, the round trip, ghost correction, the zero range, weights
  and regularisation. Found on the way: the gradient of the solver is tangent to the
  normalisation, `<c, v> = 0`, so MATLAB's `talpha = 1/sum(tc)` of the regularisation
  block is an infinity its floating point happens to survive; the block is written in its
  limit form, the Dirichlet energy moving linearly along the direction and keeping its
  scaling. The F6 extrema (`max(odf)`, `calcComponents`) are slow to hopeless on an RBF of
  600 centres; the tests check the mode by evaluation instead.
- **I6 - pole figure import. DONE 2026-09-22** (`docs/plans/io.md`): `loadPoleFigure` /
  `PoleFigure.load` with the generic column reader, the Dubna, Siemens, Geesthacht, Bruker,
  Aachen and rw1 formats, and the recipes of `mtexdata('dubna')`, `'geesthacht'`, `'ptx'`,
  `'steel'`, `'zncuti'`, `'aachenexp'`, `'dubnaODF'`. The Dubna reconstruction lands where
  the Julia page did: relative residuals 0.42, 0.37, 0.25, ... against Julia's 0.39, 0.30,
  0.23 (the data lacks a defocusing correction), 27 s on the 42056 node grid of the 5°
  data (`pf.resolution` reads the ring step of a list measured on evenly spaced rings).
- **S6 - pole figure plots** (`docs/plans/plotting.md`): `plot(pf)`, `plotDiff`; left to
  the plotting session, which edits `mtex/plotting` concurrently.

### 10.4 MATLAB to Python

| MATLAB | Python |
|---|---|
| `PoleFigure(h,r,I,cs,ss,'superposition',c)` | `PoleFigure(h, r, I, cs, ss, superposition=c)`; `complete=True` keeps directions |
| `pf.allH{i}`, `pf.c{i}`, `pf.numPF`, `pf.select(i)`, `pf({1,3})` | `pf.allH[i]`, `pf.c[i]`, `pf.numPF`, `pf.select(i)`, `pf[[0, 2]]` |
| `pf(ind)`, `pf(ind) = []` | `pf[mask]`, `pf[~mask]` |
| `calcPoleFigure(odf,{h1,[h2,h3]},r)` | `calcPoleFigure(odf, [h1, h23], r)` with `h23` a two-element Miller |
| `calcError(pf,rec,'RP')`, `'l1'`, `'l2'` | `calcError(pf, rec, 'RP')`; the default is the damped relative error |
| `max(pf,0)`, `min(pf)`, `mean(pf)` | `max(pf, 0)`, `min(pf)`, `mean(pf)`; one value per pole figure |
| `calcODF(pf,'zeroRange','noGhostCorrection')` | `calcODF(pf, zeroRange=True, noGhostCorrection=True)` |
| `[odf,alpha] = calcODF(pf)` | `odf = calcODF(pf)`; `MLSSolver(pf).calcODF()` then `solver.alpha` |
| `calcQuadratureWeights(r)` | `calcQuadratureWeights(r, method='voronoi'/'harmonic')` |
| `[v,vNew] = refine(v)` | `v, vNew = refine(v)` |

## 11. P4 - tensors

Status: written 2026-09-22; P4c to P4f and I7 done the same day (`mtex/tensors/tensor.py`,
`elasticity.py`, `mechanics.py`, `optics.py` with `data/cie1931.py`, `taylor.py`, `texture.py`,
`embedding.py`, `mtex/io/loadtensor.py`; 24 tests in about 40 s). Starts from the tensor half of
`../mtexJulia/docs/plans/polefigures-tensors.md` (P4c to P4g, P4i, P4k) and its sources
`lib/MTEXFunctions/src/tensors/`, the MATLAB folder `../frameSymmetry/TensorAnalysis/`
with its class per physical quantity, and the doc chapters `Tensors`, `Elasticity`,
`Plasticity`.

### 11.1 Deliverable

A rank-carrying tensor type in a reference frame, the physical tensors as its subclasses
with the derived quantities, and the coupling to texture: `calcTensor` on ODFs, orientation
lists and maps, Schmid factors, Taylor with its linear program, Lankford, Kearns, the shear
banding function and the Euler step of a texture, the isometric embedding, the tensor
files of the sample data. The plots (`plot(C)`, `plotSeismicVelocities`, `quiverSection`)
are P5's, `arrow3d` and the crystal shapes S5's.

### 11.2 Design

The mathematics is Julia's section 3.1: a rotation acts on every index; Voigt and Kelvin
are two bases of the symmetric subspace and therefore conversions; the texture average is
a quadrature of degree `2R` after truncating the ODF to degree `R`; the Taylor solve is a
two-phase simplex; the Hashin-Shtrikman bounds sit on the boundary of the semidefinite
references. What Python takes from each side:

- **A `tensor` is an `MtexArray`** over one `(..., 3^R)` array, the rank a carried
  attribute so that a list of tensors, `ori * C`, has the array interface of every other
  container; `T.M` is the coefficient view `(..., 3, ..., 3)` with the list dimensions in
  front (MATLAB puts them behind). The rank is read off the trailing axes of `M` (a 6 x 6 or
  3 x 6 matrix is a Voigt matrix) or given as `rank=`.
- **Julia's kinds are MATLAB's classes**: `stiffnessTensor`, `complianceTensor`,
  `stressTensor`, `strainTensor`, `strainRateTensor`, `deformationGradientTensor`,
  `velocityGradientTensor`, `spinTensor`, `curvatureTensor`, `dislocationDensityTensor`,
  `ChristoffelTensor`, `refractiveIndexTensor`, subclasses fixing the rank and the Voigt
  factors (`doubleConvention` is a class property, a keyword only on the plain `tensor`),
  carrying their methods, and swapping under `inv`. Contractions return a plain `tensor`;
  the physical products name their class (`C * eps` is a `stressTensor`).
- **Products** follow MATLAB's `mtimes`: rank two by rank two the matrix product, rank two
  by a vector the image, rank four by rank two the double contraction, a number a scaling,
  a rotation or orientation on the left `rotate`. The double contraction is `colon(A, B)`,
  MATLAB's name, with `doubleContract` an alias; `EinsteinSum` takes MATLAB's index
  vectors. `T1 * T2` on lists is elementwise over the leading axes.
- **The frame** is the tensor's one attribute besides its numbers; a tensor turned by
  anything but an element of the group of its frame is written in the frame without the
  group, and `ori * T` lands in the specimen frame stripped of its group. No `dynOption`:
  `density` and `unit` are keywords stored on the tensor, nothing else.
- **Return values**: `bulkModulus(C)` and the isotropic `shearModulus(C)` return the
  Voigt bound, MATLAB's first output; `'Reuss'`, `'Hill'` and `'all'` name the others.
  `calcTensor(odf, C)` returns the Voigt average, `calcTensor(odf, C, 'Hill')` the Hill
  one, `'all'` the triple. `velocity(C, n)` returns the six-tuple `vp, vs1, vs2, pp, ps1,
  ps2`, `velocity(C)` the same as spherical functions and axis fields. `calcTaylor` returns
  `M, b, W` as MATLAB does, the spin as a `spinTensor` list, or as functions for a strain in
  the specimen frame.
- **Kept from Julia**: `maxStrain` for the largest principal strain (`norm` stays the
  Frobenius norm on every class), the embedding basis computed from the group,
  `hashinShtrikmanBounds` by golden section beside MATLAB's mesh form
  `HashinShtrikmanModulus(C, K0, G0)`, `SO3FunSBF` as a handle, the minimum-norm shear
  selection of the Taylor solve.

### 11.3 Order of work

- **P4c - the tensor type. DONE 2026-09-22.** `mtex/tensors/tensor.py`: `tensor` and the physical
  subclasses, `EinsteinSum`, `dyad`, `tensor.leviCivita`, `Voigt`, `Kelvin`, the algebra,
  `trace`, `det`, `transpose`, `sym`, `antiSym`, `isSymmetric`, `inv`, `eig`, `rotate`,
  `transformReferenceFrame`, `symmetrise`, `checkSymmetry`, `directionalMagnitude`,
  `tensor.eye`, `.ones`, `.zeros`, `.rand`, `mean`. Tests T1 to T3, T9 (`check_EinsteinSum`,
  `check_tensorFactories`).
- **P4d - the physical tensors. DONE 2026-09-22.** `elasticity.py` (Hooke, `ChristoffelTensor`,
  `velocity`, `energyVector`, the moduli, `HashinShtrikmanModulus`,
  `hashinShtrikmanBounds`), `mechanics.py` (stress and strain measures, `SchmidTensor`,
  `calcShearStress`, `stressTensor.uniaxial`, `SchmidFactor` of a stress and as a function,
  the deformation and flow factories, spin and vorticity, Nye's relation), `optics.py`
  (`refractiveIndex`, `birefringence`, `opticalAxis`, `spectralTransmission` with the CIE
  1931 table). Tests T4 to T8 and P4i.
- **P4e - texture and tensors. DONE 2026-09-22.** `texture.py`: `calcTensor` on an ODF, on orientations and
  on a map with one tensor per phase, `calcLankford`, `calcKearnsFactor`, `SO3FunSBF`,
  `doEulerStep`; `taylor.py`: `linearProgram`, the prepared model, `calcTaylor` for a
  crystal strain, a list of orientations and a specimen strain. Tests X1 to X6,
  `check_calcLankford` pinned.
- **P4f - the embedding. DONE 2026-09-22.** `embedding.py`: `embedding(ori)`, `double`, `setDouble`,
  `mean`, `project` and `orientation` back, for the eleven Laue classes. Test T10
  (`check_embedding`).
- **I7 - tensor files. DONE 2026-09-22** (`docs/plans/io.md`): `tensor.load`,
  `stiffnessTensor.load` for the `.GPa`, `.P` and `.TEXP` files of `data/tensor`.

Found on the way:

- The fixed-direction functions on the sphere (`PoissonRatio(C, p)`, `shearModulus(C, h)`)
  are not invariant under the crystal group, and an `S2FunHandle` with `antipodal=True`
  averages over the group of its frame, so they are written in the frame without the
  group, as Julia's Schmid factor function is.
- The Taylor fields are fitted on the Clenshaw-Curtis rule of the component frames and
  projected before the truncation, as the vector field's own quadrature route does; a
  Gauss-Legendre rule handed to the grid plan and a truncation before the projection gave
  a spin field that did not reproduce its samples.
- `calcTaylor` checks the strain's crystal frame by the basis object the sibling frames
  share, so the frame `inv(ori) * eps` writes in passes and a different crystal does not.
- The transport of a density prepares its rule and the sampled spin once and reuses the
  density's plan; the naive route rebuilt a projected vector field per step and took ten
  times longer. Only the explicit scheme is ported; Julia's implicit GMRES is not.
- `ChristoffelTensor(C, n)` and `SchmidTensor(n, b)` are the class constructors, computing
  from a stiffness or a slip system, so the class and MATLAB's function share the name.
- The spherical colour functions of `spectralTransmission` are fitted on a quadrature grid
  from one vectorised pass over the directions; the per-direction loop took 10 s.
- The package level is one namespace: a later module's `symmetrise`, `union`,
  `transformReferenceFrame`, `curvature` or `project` shadows the geometry's verb, so the
  tensor modules export none of these and the earlier `_method` dispatchers reach the
  tensor methods. `spinTensor` is MATLAB's class and takes a vector as well; the geometry
  keeps its `spinTensor(v)` matrix function inside `tangent.py`.

### 11.4 MATLAB to Python

| MATLAB | Python |
|---|---|
| `tensor(M,cs,'rank',4)` | `tensor(M, cs, rank=4)`; `M` with the list dimensions in front |
| `stiffnessTensor(M,cs,'density',rho)` | `stiffnessTensor(M, cs, density=rho)`, `M` a 6 x 6 Voigt matrix or 3x3x3x3 |
| `T.M`, `T.rank`, `T.CS`, `T.doubleConvention` | the same, `doubleConvention` a class property of the physical classes |
| `EinsteinSum(T,[1 -1],v,-1)` | `EinsteinSum(T, [1, -1], v, [-1])` |
| `A : B` | `colon(A, B)`, `doubleContract(A, B)` |
| `rotate(T,ori)`, `ori * T`, `inv(ori) * sigma` | the same |
| `[KV,KR,KH] = bulkModulus(C)` | `bulkModulus(C)`, `bulkModulus(C, 'all')` |
| `[CV,CR,CH] = calcTensor(ebsd,C1,C2)`, `mean(C,odf)` | `calcTensor(ebsd, C1, C2, 'all')`, `calcTensor(odf, C)` |
| `[M,b,W] = calcTaylor(eps,sS,'plasticWork')` | `M, b, W = calcTaylor(eps, sS, plasticWork=True)` |
| `[vp,vs1,vs2,pp,ps1,ps2] = velocity(C,n)` | `vp, vs1, vs2, pp, ps1, ps2 = velocity(C, n)` |
| `[R,M,minM] = calcLankford(ori,sS,'rho',rho)` | `R, M, minM = calcLankford(ori, sS, rho=rho)` |
| `SchulzDefocusingKernel`, `tensor.load` | `SchulzDefocusingKernel` (P4a), `tensor.load` (I7) |
