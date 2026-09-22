# Harmonic conventions

Status: F0 of `docs/plans/functions.md`, 2026-09-21. The statements are those of
`../mtexJulia/docs/design/harmonics.md`, respelled for Python: zero-based indices, the
MATLAB call forms `wignerd(beta, l)` and `WignerD(g, bandwidth=L)`. Every statement has a
test set of the same name in `tests/test_harmonics.py`. The conventions are the canonical
ones (Varshalovich, *Quantum Theory of Angular Momentum*; Condon-Shortley phase); where
MTEX differs, the mapping is stated so numbers can be compared.

**H1 - layout.** The coefficients of degree `n` on SO(3) occupy `deg2dim(n) : deg2dim(n+1)`
with `deg2dim(n) = n(2n-1)(2n+1)/3`, as the `(2n+1) x (2n+1)` matrix `f̂_n` with the row
index `k` fastest; `f̂_n^{kl}` sits at `deg2dim(n) + (k+n) + (2n+1)(l+n)`. `dim2deg` is
the inverse in closed form, `degreeRange(n)` the slice and `degreeBlock(fhat, n)` the
matrix as a view of the coefficient array, so writing the block writes the coefficients. A
coefficient array may carry leading batch axes, `(..., d)`; the block is then
`(..., 2n+1, 2n+1)`. On S2 degree `l` occupies `l² : (l+1)²`, `k = -l..l`.

**H2 - Wigner-d.** `d_n^{kl}(β)` is the real matrix of Varshalovich 4.3.4,
`d_n^{kl}(β) = ξ √(binom(2n-s, s+a) / binom(s+b, b)) (sin β/2)^a (cos β/2)^b P_s^{(a,b)}(cos β)`
with `a = |k-l|`, `b = |k+l|`, `s = n - (a+b)/2`, `ξ = (-1)^{k-l}` for `k > l` and `1`
otherwise, so that `d_1^{10}(β) = -sin β / √2` and `d_1^{01}(β) = sin β / √2`. (The
sign rule in the `Test` block of MTEX's `WignerD.m` is not this one and that block
asserts nothing.) Consequences: `d_n(0) = I`, `d_n(β)` is orthogonal,
`d_n^{kl}(-β) = d_n^{lk}(β)`, `d_n^{-k,-l} = (-1)^{k-l} d_n^{kl}`. The code computes all
degrees at once by the three-term recursion in `n` (Vollrath 2006, Potts, Prestin,
Vollrath 2009) with the outer frame `|k| = n` or `|l| = n` from the closed form, where
`s = 0`: `wignerd(beta, bandwidth=L)` is the list `d_0 .. d_L`, `wignerd(beta, l)` the one
matrix. The precision follows `beta`. The Fourier identity that the transforms of F2 rest
on: `d_n^{kl}(β) = i^{k-l} Σ_j d_n^{jk}(π/2) d_n^{jl}(π/2) e^{-ijβ}`.

**H3 - Wigner-D.** For `R = Rz(α) Ry(β) Rz(γ)` (the Matthies angles of P1, active),
`D_n^{kl}(R) = e^{-ikα} d_n^{kl}(β) e^{-ilγ}`. Then `D_n(R1 R2) = D_n(R1) D_n(R2)`,
`D_n(R⁻¹) = D_n(R)^H`, `D_n(1) = I`, and `D_n` is unitary. `WignerD(r, l)` is the one
matrix, `WignerD(r, bandwidth=L)` the list, and a container of rotations gives the
matrices stacked in front, `(..., 2n+1, 2n+1)`. An improper transform is represented by
its rotation part. MTEX's `WignerD.m` builds `d` as `expm(-β J_y)` with the opposite sign
pattern, which is `d_MTEX = dᵀ`, and pairs `k` with `γ`, so `D_MTEX(R) = D(R)ᵀ`, an
anti-homomorphism.

**H4 - spherical harmonics.** `Y_l^m(θ, ρ) = P̄_l^m(cos θ) e^{imρ}` with the fully
normalised associated Legendre functions including the Condon-Shortley phase, so
`∫ Y_l^m conj(Y_l'^m') dS = δ δ`, `Y_0^0 = 1/√(4π)`, `Y_1^1 = -√(3/8π) sin θ e^{iρ}`, and
`Y_l^{-m} = (-1)^m conj(Y_l^m)`. `sphericalY(l, v)` is the vector `m = -l..l` in front of
the container shape, `(2l+1, ...)`; `normalizedLegendre(l, x)` the `P̄_l^m(x)`, `m = 0..l`,
in front of the shape of `x`. MTEX's `sphericalY.m` omits the `(-1)^m` for `m < 0`.
Rotation: `Y_l^m(R ξ) = Σ_{m'} conj(D_l^{m m'}(R)) Y_l^{m'}(ξ)`, as vectors
`Y_l(R ξ) = conj(D_l(R)) Y_l(ξ)` and `Y_l(R⁻¹ ξ) = D_l(R)ᵀ Y_l(ξ)`; the test tries the
four placements and this is the one that holds.

**H5 - the series and the sides.** A function on SO(3) is
`f(R) = Σ_n √(2n+1) Σ_{k,l} f̂_n^{kl} D_n^{kl}(R)`, `wignerSeries(fhat, r)` by the direct
sum; the basis `√(2n+1) D_n^{kl}` is orthonormal for the normalised Haar measure, so
`mean(f) = f̂_0` and `‖f‖² = Σ |f̂|²`. A specimen rotation on the left, `f(s R)`, maps
`f̂_n ← D_n(s)ᵀ f̂_n` (rows); a crystal rotation on the right, `f(R s)`, maps
`f̂_n ← f̂_n D_n(s)ᵀ` (columns). Hence the rows of a block are the specimen side and the
columns the crystal side. The projection onto the functions invariant under a group `G`
acting on one side is `P_n = (1/|G|) Σ_{s ∈ G} D_n(s)ᵀ` applied on that side; it is
idempotent and Hermitian, and it is `WignerD(group, l)` respectively
`WignerD(group, bandwidth=L)`; a frame delegates to its group, whose elements are written
in the frame's axes. An improper element acts on SO(3) as its rotation part, so the sum
runs over the proper group (`symmetry.properGroup`). The projection held in its support
(the fold about z) is the concern of the transforms, F2.

**H6 - Clebsch-Gordan.** `⟨j1 m1 j2 m2 | J M⟩` by the Racah formula in exact rational
arithmetic, `clebschGordan(j1, m1, j2, m2, J, M)`; `wigner3j` from it;
`clebschGordanTensor(l1, l2, J)` the array `C[m1, m2, M]`. Identity (the MATLAB test):
`D_{l1}(R) ⊗ D_{l2}(R) = Σ_J C_J D_J(R) C_Jᵀ` in the sense
`D_{l1}^{m1 m1'} D_{l2}^{m2 m2'} = Σ_J Σ_{M M'} C[m1,m2,M] D_J^{M M'} C[m1',m2',M']`.

**H7 - the circle and the Fourier arrays.** On S1 a function is `f(x) = Σ_{k=-L}^{L} c_k e^{ikx}`,
real when `c_{-k} = conj(c_k)`, with `mean(f) = c_0`; the even kernels of F1 have
`c_k = c_{-k}` real. The transforms of F2 rest on H2 and H4: a Wigner series is the
trivariate Fourier series `f(α, β, γ) = Σ ĝ_{kjl} e^{-i(kα + jβ + lγ)}` with
`ĝ_{kjl} = Σ_n √(2n+1) i^{k-l} f̂_n^{kl} d_n^{jk}(π/2) d_n^{jl}(π/2)`, a spherical harmonic
series the bivariate one `f(θ, ρ) = Σ ĝ_{jm} e^{i(mρ - jθ)}` with
`ĝ_{jm} = Σ_l √((2l+1)/4π) i^m f̂_l^m d_l^{jm}(π/2) d_l^{j0}(π/2)`, and both adjoints carry
the same phase `i^{k-l}` respectively `i^m` on the Fourier array at `-j`. A real function
has `f̂_n^{-k,-l} = (-1)^{k-l} conj(f̂_n^{kl})` on SO(3) and `f̂_l^{-m} = (-1)^m conj(f̂_l^m)`
on S2, so its Fourier array is conjugate symmetric and half of it suffices.
