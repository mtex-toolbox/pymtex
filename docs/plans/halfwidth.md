# Subplan - automatic halfwidth selection on S2 and SO(3)

Status: written 2026-09-26; H0, H1 and H2 are built the same day (section 8): the energy vector
and the selectors C1 to C3 on S2 and SO(3) in `mtex/functions/halfwidth.py`, `calcKernel` with
`'conservative'` and `'UCV'` for directions and orientations, the flag `notIID` on the
orientations of a map, the studies `bench/halfwidth_s2.py` and `bench/halfwidth_so3.py` (their
tables in `bench/out/halfwidth/`). H3 to H6 are open; `calcDensity` still defaults to 10 degrees
and `calcKernel` of orientations to KLCV until H5. It takes the densities of directions on S2 first and generalises to orientations
with crystal and specimen symmetry afterwards. The existing selectors are MATLAB's
`calcKernel` on the frameSymmetry branch (`../frameSymmetry/geometry/@orientation/calcKernel.m`
with `KLCV.m`, `LSCV.m`, `BCV.m`) and their port in `mtex/functions/characteristics.py`. The
discrepancy, Wasserstein and goodness-of-fit work that shares the mathematics lives in MATLAB
master on the branch feature/fastHarmonicTransforms (`../master/tests/S2Discrepancies.md`,
`SO3Discrepancies.md`, the harness in `../master/tests/lib/`), with earlier pilots in the Julia
research notes (`../mtexJulia/docs/research/spherical-goodness-of-fit-overview.md`,
`spherical-estimation-sampling-quadrature-testing.md`). Section 6 connects the two lines.

## 1. Goal, and what "fast" and "conservative" mean

The goal is a halfwidth selector that improves on what MTEX offers today, in two roles:

- a **default** that `calcDensity` of directions and of orientations can call without being
  asked: it must take about one second at n = 1e6 orientations and scale linearly to 1e7, and
  it must be **conservative**: when in doubt it oversmooths. The guarantee has to be stated,
  not claimed: "the selected halfwidth is at least the MISE-optimal one of the family, with
  probability at least 1 - alpha", or "with probability at least 1 - alpha no fluctuation of
  the estimate exceeds tau mrd", whichever of the two can be proven for the method;
- an **accurate** selector, allowed a few seconds more, that aims at the oracle halfwidth
  (the minimiser of the integrated squared error for the true density) and is judged by how
  close it gets.

Everything below is written in one harmonic language, the one of
`spherical-estimation-sampling-quadrature-testing.md`. On S2 with real harmonics orthonormal in
L2 of the uniform probability U, a density with mean one is
f = 1 + sum_{l>=1} f_l, f_l its degree l component, A_l = ||f_l||^2 the degree energy,
d_l = 2l + 1, lambda_l = l(l+1). The empirical measure has degree components mu_l with
||mu_l||^2 = n^-2 sum_{i,j} d_l P_l(X_i . X_j). A zonal kernel with multipliers b_l (b_0 = 1)
gives the estimate fhat = 1 + sum_l b_l mu_l, and for a fixed kernel the risk is exactly

    MISE(b) = sum_{l>=1} (1 - b_l)^2 A_l + (1/n) sum_{l>=1} b_l^2 (d_l - A_l),

the variance term exact because sum_m Y_lm(x)^2 = d_l. The unbiased estimate of the energy is the
U-statistic A^_l = (n ||mu_l||^2 - d_l) / (n - 1), whose i = j terms are known. On SO(3) the same
holds with the Wigner blocks, d_l = (2l+1)^2 and lambda_l = l(l+1); symmetry changes d_l
(section 4). The halfwidth families of MTEX (de la Vallee Poussin, von Mises-Fisher,
Gauss-Weierstrass, Abel-Poisson) are one-parameter families b_l(h), each b_l nonincreasing in h,
which is the only property section 5 needs.

Two measurements fix the budget. The adjoint harmonic transform of the port takes the empirical
coefficients of n = 1e6 points in 0.06 s at bandwidth 32 or 64 and 0.07 s at 128 on S2, and
0.25 to 0.30 s at bandwidth 16 to 48 on SO(3) with cubic symmetry (this machine, 2026-09-26,
`S2FunHarmonic.quadrature`, `SO3FunHarmonic.quadrature`). Everything the risk formula needs
after that is a sum over degrees, microseconds. A selector that works on the degree energies is
therefore inside the one-second budget with room for several transforms; one that evaluates a
kernel sum at every data point is not, unless it too goes through transforms.

## 2. What exists

### 2.1 MATLAB (`../frameSymmetry`)

`@orientation/calcKernel.m` returns an `SO3DeLaValleePoussinKernel`. Above 5000 orientations it
draws a subsample of 5000 **with replacement** (`discretesample`) and multiplies the selected
kappa by (N/5000)^(1/7) (the comment: "true is 2/7 but let us stay on the save side"; kappa
scales as h^-2, so h ~ N^(-1/7) is the AMISE rate in dimension 3 and the scaling of kappa by
the 1/7 power is the conservative half of the correct 2/7). The methods:

| Method | What it minimises or computes | Cost | Weaknesses |
|---|---|---|---|
| `magicRule` | kappa = 3 (numProper(CS) numProper(SS) N)^(2/7); no data | O(1) | ignores the density entirely; the symmetry factor assumes every symmetric copy is an independent observation; measured 2 to 4 times the oracle halfwidth on a sharp unimodal texture (below); also ignores the subsample (uses N) |
| `ruleOfThumb` | 1.5 times the 0.9 quantile of nearest-neighbour disorientations, at least 2 degrees, then the (N/5000)^(1/7) scaling | O(m^2) on the m = 5000 subsample (`angle_outer`) | a resolution, not a risk; the 0.005 rad threshold drops exact duplicates only; with replacement sampling creates duplicates it then drops; 10 s in the port at m = 5000 |
| `KLCV` (default) | leave-one-out log likelihood sum_i log fhat_{-i}(X_i), over ten candidates 30 deg / 2^(k/3), k = 1..10 (23.8 down to 2.98 deg) | O(sN * m) kernel evaluations, sN = `SamplingSize` 1000 | coarse grid (factor 1.26) with a floor of 2.98 deg before the rescaling; the subsample with replacement holds exact duplicates which a leave-one-out does not remove, so it rewards narrow kernels (docs/findings.md, rows OptimalKernel and DensityEstimation); correlated EBSD pixels of one grain do the same; KL loss is dominated by the lowest density values, i.e. by the background, not the peaks |
| `LSCV` | least squares cross validation, through `calcFourierODF` with an `SO3DirichletKernel(25)` and then `conv` with each candidate | NFFT at L = 25 per call | the data are truncated at degree 25 before the kernel is applied, so any candidate sharper than about 8 deg is judged as if it were 8 deg; the loop stops at the first increase, i.e. at a local minimum over a grid of ten |
| `BCV` | biased cross validation with an `SO3SobolevKernel` | NFFT at L = 16 | does not run: uses an undefined `weights`, indexes the cell `psi(i)` as an array, prints every value; the constants (pi/8, kappa^(3/2)) are an AMISE for a specific kernel only |

`vector3d/calcDensity.m` has no selector at all: the halfwidth defaults to 10 degrees, and
`doc/Vectors/VectorsDensityEstimation.m` says so ("MTEX does not select a halfwidth
automatically for directional data"). The one-dimensional `extern/kde1d.m` (Botev's diffusion
estimator, used by `calcAngleDistribution` and `tools/statistic_tools/calcDensity.m`) pairs the
coefficient of degree k with the weight of degree k + 1: `I = (1:n).^2` against `a.^2/4` over all
n coefficients, where Botev's original uses degrees 1..n-1 against `a(2:end)`. The port
(`mtex/functions/kde.py`, `_fixedPoint`) has the corrected pairing. The same diffusion plug-in is
one of the S2 candidates below, so the bug is worth fixing in MATLAB independently.

`doc/GeneralConcepts/OptimalKernel.m` measured on the 321 model (a fibre, a peak, 25 %
uniform): KLCV picked 5.67 deg at N = 1e4 with error 0.104; at N = 30, 100, 300, 1000 the three
methods gave the halfwidths 23.8/54.8/26.1, 11.9/30.9/22.0, 11.9/26.7/18.8, 7.5/17.8/15.9
(KLCV, RuleOfThumb, magicRule) and errors in the same order with KLCV always best. The page
states the full study (N up to 1e6) was not run. `doc/EBSDAnalysis/EBSD2ODF.m` compares the same
three on forsterite pixels and grain means.

### 2.2 The port

`calcKernel(o, method="klcv", samplingSize=1000, rng=None)` in `mtex/functions/characteristics.py`
reproduces the three working methods. It subsamples by a permutation, i.e. **without**
replacement, which already differs from MATLAB and removes the duplicate bias; there is no
`LSCV` or `BCV`. `calcDensity` in `mtex/functions/density.py` defaults to 10 degrees on both
spaces (`_densityKernel` in `mtex/functions/rbf.py`) and never selects. On one sample
(m-3m, one dlVP peak of 10 deg halfwidth; the ISE computed in the harmonic domain at bandwidth
64 over nine halfwidths from 3 to 15 deg):

| N | oracle hw | KLCV | magicRule | KLCV time |
|---:|---:|---:|---:|---:|
| 1e3 | 4 deg | 4.7 deg | 13.0 deg | |
| 2e4 | 3 deg or less | 3.4 deg | 8.5 deg | 3.0 s |
| 2e5 | 3 deg or less | 2.9 deg | 6.1 deg | |

`ruleOfThumb` took 10 s at 2e4. So KLCV without replacement is good on clean independent data;
`magicRule` is conservative by a factor 2 to 4 and blind to sharpness. Neither is fast at 1e6
in the port except `magicRule`.

`docs/pages/GeneralConcepts/OptimalKernel.py` and `DensityEstimation.py` follow the MATLAB
pages. `mtex/analysis/optimalsample.py` holds the restricted distance kernel weights of the
discrepancy (the same `kernelWeights` as MATLAB master's private helpers).

## 3. Ideas from the literature, on S2

### 3.1 Reference rules

Taylor's rule (Taylor 2008) derives the AMISE of the circular von Mises KDE for a von Mises
population and plugs in its estimated concentration; García-Portugués (2013) carries this to
S^q (the ROT selector) and improves it by fitting a von Mises-Fisher **mixture** (selected by
AIC) and plugging it into the asymptotic (AMI) or the exact (EMI) MISE, EMI being the best in
his simulation over many scenarios. Oliveira, Crujeiras and Rodríguez-Casal (2012) do the same
on the circle with a mixture. What this gives us: a reference rule is one moment and a formula,
O(n), and for a multimodal density fitted by one vMF it oversmooths (the fitted concentration is
low), which is the right direction of failure. It fails badly for sharp textures around a
uniform background, where the single vMF fit is nearly uniform. The EMI idea, exact MISE of the
mixture, is the parametric version of candidate C2 below.

### 3.2 Plug-in and diffusion

Sheather-Jones plug-in and Botev's diffusion estimator (Botev, Grotowski, Kroese 2010, the
improved Sheather-Jones) estimate the roughness functional of the AMISE, ||f''||^2 on the line,
by a kernel estimate whose own bandwidth solves a fixed-point equation. On the circle Tenreiro
(2022) writes the plug-in in Fourier series and gets relative rate n^(-1/2); Ameijeiras-Alonso
(2024) gives the circular Sheather-Jones. On S2 the AMISE of a second-order kernel is
h^4 c_K^2 ||Delta f||^2 / 4 + R(K_h)/n with Delta the Laplace-Beltrami operator, and
||Delta f||^2 = sum_l lambda_l^2 A_l: the roughness is a weighted sum of degree energies, which the
U-statistics A^_l estimate without bias. The heat kernel (`SO3GaussWeierstrassKernel`, and its
S2 counterpart) is the natural diffusion; Botev's fixed point transfers verbatim with
exp(-t lambda_l) in place of exp(-k^2 pi^2 t). The plug-in is fast (one transform) and
accurate for smooth f; for sharp textures the AMISE itself is wrong (h is not small against the
feature), which is why the exact MISE of the harmonic form is preferable wherever it is
available, and on the sphere it always is.

### 3.3 Cross validation

Hall, Watson and Cabrera (1987) introduced least squares and likelihood cross validation for
spherical KDE and showed which kernels are admissible for KL loss. LSCV is unbiased for
MISE - ||f||^2 and notoriously variable, with a tendency to undersmooth (Terrell 1990 is the
standard complaint); KLCV is consistent but sensitive to the tails and, for data with ties or
near ties, degenerate towards h -> 0. In the harmonic domain LSCV **is** the risk formula of
section 1 with A^_l in place of A_l, restricted to the bandwidth of the transform:
UCV(h) = sum_l [(b_l^2 - 2 b_l) A^_l + b_l^2 (d_l - A^_l)/n], exactly, costing one transform plus a
scan over h. KLCV needs fhat_{-i}(X_i) = (n fhat(X_i) - K_h(1)) / (n - 1) at every data point,
which is one forward transform (evaluation at the n points) per candidate, 0.1 to 0.3 s each at
1e6 in the port, so full-data KLCV without subsampling is affordable too.

### 3.4 Oversmoothing and maximal smoothing

Terrell and Scott (1985) and Terrell (1990): among all densities with a given scale, the one of
least roughness gives the largest AMISE-optimal bandwidth, so plugging in that roughness bounds
the optimal bandwidth from above by a statistic that is as stable as a variance estimate. The
sphere is compact and the uniform density has zero roughness, so the literal transfer (fix a
spread, minimise ||Delta f||^2) degenerates for weakly concentrated data. What transfers is the
structural fact behind it: **an under-estimate of the roughness is an over-estimate of the
bandwidth**. In the exact risk this is a clean monotone comparative statics statement (section
5, C2), not an asymptotic one, and it is the basis of the proposed conservative default.

### 3.5 The harmonic domain: risk estimates and smoothness priors

Because the sphere has an orthonormal basis in which rotation-invariant smoothing is diagonal,
the density estimation problem is a Gaussian sequence problem in the limit: coefficients with
noise variance about d_l / n per degree. Everything from that literature applies: unbiased risk
(Stein) estimates, which is UCV again; monotone or blockwise James-Stein shrinkage per degree
(b_l = (1 - d_l / (n ||mu_l||^2))_+ is the degree-wise James-Stein multiplier, an estimate of the
oracle A_l / (A_l + (d_l - A_l)/n)); Pinsker tapers for a Sobolev ellipsoid (Klemelä); and the
oracle inequalities of adaptive estimation (Pham Ngoc 2019 on S^d, by Goldenshluger-Lepski). The
point for MTEX is that the kernel family is fixed by the user's expectation (a halfwidth), so
we select within a one-parameter family, but the free degree-wise shrinkage is the lower bound
any family is compared against in the evaluation.

### 3.6 Bootstrap and Bayesian bandwidth

The smoothed bootstrap estimate of MISE (Taylor 1989, Marron 1992) is, for a kernel family,
available in closed form in the harmonic domain: it is the risk formula with A_l replaced by the
energies of a pilot estimate, i.e. a plug-in, and adds nothing computationally new. Its use here
is for **uncertainty**: resampling grains (a cluster bootstrap) gives an interval for the
selected halfwidth that honours intragranular correlation, which no formula does. Bayesian
bandwidth selection (Zhang, King, Hyndman 2006) puts a prior on h and uses the leave-one-out
likelihood as its likelihood; on the sphere this is KLCV with a prior, and its merit is again
the posterior spread, not speed.

### 3.7 Spurious peaks

On the line the Gaussian kernel is the only one whose number of modes cannot increase with h
(Silverman's critical bandwidth test rests on it). On S2 and SO(3) there is no such causality:
two-dimensional Gaussian scale space creates new extrema. So "conservative" in the sense of "no
spurious peaks" cannot be argued from monotonicity; it needs a deviation bound. The estimate
at halfwidth h is a polynomial of degree about L(h), its fluctuation fhat - K_h * f has variance
(1/n)(K_h^2 * f - (K_h * f)^2) <= ||K_h||_inf (K_h * f) / n pointwise, and a Bernstein inequality
at the points of a Marcinkiewicz-Zygmund grid of the degree L(h) with a Bernstein-type
inequality for spherical polynomials between them bounds the supremum with probability
1 - alpha. That gives a halfwidth below which the noise alone can raise bumps of height tau
(candidate C3).

### 3.8 Adaptive and local bandwidth, later

Abramson's square-root law (a pilot estimate raises the bandwidth where the density is low),
Lepski-type local selection (Pham Ngoc 2019), and Botev's diffusion with a pilot drift are the
natural extensions. They break the diagonal harmonic structure (the estimate is no longer a
convolution), so they cost an RBF evaluation or an iterative transform and belong to a later
stage, after the global selector is settled. Textures with a sharp component on a broad
background are exactly where they would help.

## 4. From S2 to SO(3)

**Harmonics.** Wigner blocks D^l of size (2l+1)^2, lambda_l = l(l+1), the addition theorem
sum_{m,k} |D^l_{mk}(g)|^2 (2l+1) = (2l+1)^2 for the normalisation orthonormal in L2 of Haar
probability, so d_l = (2l+1)^2 and everything in section 1 carries over verbatim. The kernels
MTEX uses are all zonal on SO(3), b_l given by `SO3Kernel.A` / (2l+1); the de la Vallée Poussin
kernel has the finite bandwidth that makes the transforms exact, the Gauss-Weierstrass kernel is
the heat kernel for the diffusion plug-in.

**Symmetry.** With proper groups G_S (specimen) and G_C (crystal) the density lives on the
invariant subspace; the estimate is the projection of the empirical coefficients, and the
variance term becomes (1/n) sum_l b_l^2 (E_f c_l(X) - A_l), where
c_l(g) = |G_S|^-1 |G_C|^-1 sum_{a in G_S, b in G_C} (2l+1) chi_l(g^-1 a g b) is the diagonal of
the degree-l reproducing kernel of the invariant subspace, chi_l the character. For generic g
only a = b = id contributes much and E c_l ~ d_l^G = (2l+1)^2 / (|G_S||G_C|), which is where
`magicRule`'s factor numProper(CS) numProper(SS) comes from; for densities concentrated near
orientations with a nontrivial stabiliser (the cube orientation with orthorhombic specimen
symmetry, Goss) the gain is smaller, and the exact U-statistic needs sum_i c_l(X_i), not the
generic value. c_l is a sum of |G_S||G_C| characters computable for all l by the three-term
recursion; on a subsample of 1e4 points it is an unbiased estimate of E c_l at negligible cost.
Antipodal (misorientation, Laue) symmetry is one more group element of the same kind.

**Reference distributions** for the rule of thumb and for pilots: on S2 the vMF; on SO(3) the
matrix Fisher (or the Bingham on quaternions) and the dlVP unimodal ODF, each symmetrised by the
groups. Symmetrisation kills the low degrees for many groups (degree 1 survives only for the cyclic
proper groups, the tetrahedral group keeps nothing below degree 3, the octahedral nothing
below degree 4),
so a moment fit must use the lowest degree with a nonzero invariant subspace, i.e. it is again a
fit of degree energies.

**Clustered data.** EBSD pixels of one grain are not independent observations of the texture:
they are one grain orientation plus a small intragranular scatter. The degree energies expose
this directly: for grain-clustered data E ||mu_l||^2 = A_l + (d_l - A_l)/n + (design effect)
where the extra term is about sum_g m_g^2 / n^2 times the energy of the intragranular
distribution at degree l. For low degrees (above the grain scale) the effective n is the Kish
number of grains n_eff = (sum_g m_g)^2 / sum_g m_g^2, m_g the pixels of grain g; for
degrees that resolve the intragranular scatter it approaches the pixel count. A selector that
takes the pixel count as n undersmooths to the grain scale; that is the failure the doc pages
warn about ("use grain means"). Cross-fitting across **grain-disjoint folds** removes the bias
from the energy estimate without modelling it: E <mu_l^(1), mu_l^(2)> = A_l exactly when the two
halves share no grain, and the spread across folds gives the variance with the correlation in
it. Where no grains are computed, spatial blocks of the map serve as folds.

**Weights.** Area-weighted grain samples and weighted `calcDensity` calls change the variance
term to sum_i w_i^2 (d_l - ...) with sum w_i = 1, i.e. n -> n_eff = 1 / sum w_i^2, and the U-statistic
to its weighted form; the cross-fitted estimate carries over unchanged.

## 5. Candidate methods

Five candidates, sketched with their cost; T_S2 and T_SO3 denote one adjoint transform of all n
points (0.06 to 0.09 s and 0.25 to 0.3 s at n = 1e6 as measured), F the number of folds, H the
number of candidate halfwidths.

### C1. Spectral UCV: the exact risk with unbiased energies

Minimise UCV(h) = sum_{l<=L} [(1 - b_l(h))^2 A^_l + b_l(h)^2 (d_l^G - A^_l)/n] over h, with the
U-statistic A^_l (or its cross-fitted version, C2) and L the largest bandwidth of the family. This
is LSCV computed exactly and without truncation at 25, the natural replacement of MATLAB's
`LSCV`. Cost: one transform at bandwidth L, then O(L H). At n = 1e6: 0.1 s on S2 (L = 128),
0.3 to 1 s on SO(3) (L = 64 with symmetry). Expected behaviour: good for moderately smooth f,
the LSCV variance at small n, undersmoothing on clustered data unless the energies are
cross-fitted. It is the accuracy candidate.

### C2. Truncated-energy oversmoothing with lower confidence bounds (the fast conservative candidate)

Replace A_l by a **lower** estimate: a_l = max(A^_l - z se_l, 0) for l <= L0 and a_l = 0 above
L0, with A^_l cross-fitted over F folds (grain-disjoint for EBSD) and se_l their jackknife error;
then minimise the risk formula with a_l.

Why it is conservative. Write R(h; A) = B(h; A) + V(h; A) with
B = sum_l (1 - b_l(h))^2 A_l nondecreasing in h and V = (1/n) sum_l b_l(h)^2 (d_l - A_l)
nonincreasing in h, for every A >= 0 (each b_l is nonincreasing in h). Lowering A_l to a_l removes
from R the function G(h) = sum_l (A_l - a_l)[(1 - b_l)^2 - b_l^2/n], which is nondecreasing in h
because both of its parts are. If G is nondecreasing, the largest minimiser of R - G is at least
the largest minimiser of R (Topkis; the two-line proof: if h1 minimises R - G and h0 < h1
minimises R, adding the two optimality inequalities gives G(h1) <= G(h0)). Hence: **if a_l <= A_l
for every l, the selected halfwidth is at least the MISE-optimal halfwidth of the family**, for
every n, without asymptotics, for any density. With the one-sided bounds holding jointly with
probability 1 - alpha (Bonferroni over the L0 degrees, or a simultaneous bootstrap bound over the
folds), the statement holds with probability 1 - alpha. The error of the truncation shrinks as L0
grows, so L0 is the dial between the guarantee and efficiency; the degrees above L0 carry no
information at small n anyway.

Cost: F transforms of n/F points at bandwidth L0 (the NFFT cost is linear in the number of
points, so F folds cost the same point work plus F times the FFT part), then O(F L0) for the fold
energies and O(L0 H) for the scan. At n = 1e6: about 0.1 s on S2 and 0.3 to 0.6 s on SO(3) with
F = 8 and L0 = 32. The guarantee is about the MISE optimum; it does not by itself exclude spurious
peaks, hence C3.

### C3. Noise floor: the smallest halfwidth whose fluctuations stay below tau

Given a tolerated bump height tau (in mrd, or relative to the estimate's maximum) and alpha, take
the smallest h for which the supremum bound of section 3.7 is below tau, using the maximum of a
pilot estimate (C2's estimate itself) for ||K_h * f||_inf. The guarantee is exactly the one users
mean by "no spurious peaks": with probability 1 - alpha, no feature of height above tau is noise.
Cost: O(H) formula evaluations and one evaluation of the pilot maximum, which the port's
`max` does with its grid-seeded ascent. For clustered data n is the n_eff of the lowest degrees.
Used as a floor: h = max(h_C2, h_C3).

### C4. Harmonic plug-in and diffusion (Botev on the sphere)

The AMISE plug-in with the roughness sum_l lambda_l^2 A_l estimated by a pilot heat smoothing
whose time solves Botev's fixed point in the degree domain, or Tenreiro's direct Fourier
plug-in. Cost as C1. Kept because it is the best-studied accurate selector in one dimension and
because the diffusion estimator is what `kde1d` already is; expected to be dominated by C1/C2
where the exact risk is available, which is always here, and to fail for sharp textures where
the AMISE does not describe the risk. It is mainly the bridge to the one-dimensional selector
MTEX already ships.

### C5. KLCV without subsampling, leave-one-grain-out

fhat_{-i}(X_i) for all n points by one forward transform per candidate (evaluate the estimate at
the data, subtract the self term), and, for EBSD, subtract the whole grain's contribution
(leave-one-grain-out) through per-grain partial sums. Cost H forward transforms: 1 to 3 s at 1e6
on SO(3) with H = 10; a golden-section search needs about 12. The compatible upgrade of the
current default: same criterion, no duplicate bias, no subsample, correct under clustering.

| | cost at n | wall time at n = 1e6 (S2 / SO(3), estimated from the measured transforms) | guarantee |
|---|---|---|---|
| C1 spectral UCV | O(n + L^d log L + L H) | 0.1 s / 0.3 to 1 s | unbiased risk estimate |
| C2 truncated LCB | O(n + F L0^d log L0 + L0 H) | 0.1 s / 0.3 to 0.6 s | h >= h_MISE with probability 1 - alpha |
| C3 noise floor | O(H) + one maximum | under 0.1 s after C2 | no noise bump above tau, probability 1 - alpha |
| C4 plug-in | as C1 | 0.1 s / 0.3 to 1 s | asymptotic, relative rate n^(-1/2) for smooth f |
| C5 KLCV all points | O(H (n + L^d log L)) | 0.5 s / 1 to 3 s | consistency only |

The fast conservative default proposed is **C2 with the C3 floor**; C1 (and C5 for MATLAB
compatibility) are the accurate options. On S2 a small pilot (below, section 5.1) supports the
ordering; on SO(3) nothing has been measured beyond the transforms.

### 5.1 A pilot on S2

A short Python experiment (not in the repo) compared the rules on exactly sampled mixtures of a
uniform part and dlVP bumps, the true energies known in closed form, H = 60 halfwidths between 2
and 60 degrees, bandwidth 192, F = 8 random folds, z = 2, reporting the median selected halfwidth,
the median and worst MISE relative to the oracle (the exact risk at the selected halfwidth, not
the realised ISE), and how often the rule chose a halfwidth below the oracle. The folds of the
n = 100 rows have fewer than 50 points, where `S2FunHarmonic.quadrature` takes its direct sum
(`DIRECT_SUM_NODES` in `mtex/functions/transforms.py`) and needs about 10 s at bandwidth 192
against 0.02 s through the NFFT at 50 points; a selector with folds must avoid that path.

Model A, 0.4 uniform + 0.4 dlVP 10 deg + 0.2 dlVP 5 deg (two bumps 37 deg apart), 8 replicates:

| n | oracle | UCV (C1) | C2, L0 = 16 | C2, L0 = 32 |
|---:|---:|---|---|---|
| 1e2 | 5.0 | 4.5 [3.8, 7.5], 1.10 / 1.30, 62 % below | 7.5 [6.3, 15.9], 1.30 / 2.73, 0 % | 7.3 [6.3, 15.9], 1.26 / 2.73, 0 % |
| 1e3 | 3.0 | 3.0 [2.5, 3.2], 1.01 / 1.09, 38 % | 3.8 [3.8, 4.0], 1.18 / 1.28, 0 % | 3.6 [3.2, 3.8], 1.10 / 1.18, 0 % |
| 1e4 | 2.0 * | 2.0, 1.00 / 1.00, 0 % | 2.4, 1.18 / 1.18, 0 % | 2.0 [2.0, 2.1], 1.00 / 1.03, 0 % |
| 1e5, 1e6 | 2.0 * | 2.0, 1.00 | 2.0, 1.00 | 2.0, 1.00 |

Model B, 0.7 uniform + three dlVP 20 deg bumps on the axes, 8 replicates:

| n | oracle | UCV (C1) | C2, L0 = 16 or 32 |
|---:|---:|---|---|
| 1e2 | 23.9 | 28.6 [22.5, 60], 1.05 / 1.34, 12 % below | 60 (the widest candidate), 1.34 / 1.34, 0 % |
| 1e3 | 12.7 | 11.9 [11.3, 13.4], 1.01 / 1.04, 62 % | 14.6 [13.4, 18.9], 1.06 / 1.45, 0 % |
| 1e4 | 8.0 | 8.2 [8.0, 9.0], 1.00 / 1.04, 0 % | 8.7 [8.5, 9.0], 1.02 / 1.04, 0 % |
| 1e5 | 5.3 | 5.3 [5.0, 5.6], 1.00 / 1.01, 12 % | 5.3, 1.00 / 1.00, 0 % |
| 1e6 | 3.6 | 3.6 [3.4, 3.6], 1.00 / 1.01, 12 % | 3.6, 1.00 / 1.00, 0 % |

Each cell: median halfwidth in degrees [range], median / worst MISE relative to the oracle, share
of replicates below the oracle. * the oracle sits on the smallest candidate, 2 deg, so these rows
say nothing about undersmoothing. The cross-fitted UCV behaved like the plain one. Read: C2 never
chose a halfwidth below the oracle in 80 paired runs, at a median cost of 10 to 30 % in MISE at
n <= 1e3 and none from 1e4 on; plain UCV chose below the oracle in up to three quarters of the runs
at n = 1e3, cheaply in MISE, which is exactly the undersmoothing that shows as spurious peaks. At
n = 100 on model B the lower bounds were all zero and C2 fell back to the widest candidate, the
honest answer to "no degree is resolved". This is iid data; clustering is untested.

On a very sharp mixture (a 3 deg bump of weight 0.15, peak 200 mrd) a first version whose
standard errors used the supremum of the density collapsed to the widest candidate at n = 1e3,
which is why the plan takes the fold jackknife for se_l: the conservative rule is only as good as
its lower bound is tight.

## 6. Relation to discrepancies, Wasserstein distances and tests

**The shared object.** A kernel density estimate, a Sobolev or discrepancy norm and a weighted
harmonic test all act on the same vector: the degree energies of the sample,
e_l = n ||mu_l - q_l||^2 (q_l the reference coefficients; zero for the uniform null or for
estimation). The KDE multiplies mu_l by b_l; `discrepancy` with `degreeWeights` (MATLAB master
`S2Fun/@S2Fun/discrepancy.m`, `SO3Fun/@SO3Fun/discrepancy.m`, `geometry/@vector3d/discrepancy.m`)
sums w_l ||mu_l - q_l||^2; the adaptive heat test standardises T_t = sum_l exp(-t lambda_l) e_l. The
unbiased energy A^_l of section 1 is (e_l - d_l)/(n-1): **the centred per-degree test statistic
and the unbiased energy estimate are the same number**. UCV, the C2 bound, the Sobolev
statistics and the heat maximum are linear functionals of one vector (e_l)_l, and a heat KDE at
time t has multipliers b_l = exp(-t lambda_l) whose squared-norm weights are exp(-2 t lambda_l),
the factor two of the Julia note.

**Shared machinery.** One transform per sample (per fold) produces (e_l)_l, and the fold
cross-products give the unbiased and the clustered versions. The same code serves `calcKernel`,
`discrepancy(..., 'metric', 'L2', 'degreeWeights', w)`, the heat-scale test and `optimalSample`,
whose objective is the restricted distance kernel weights of `kernelWeights.m`. The symmetric
trace c_l of section 4 is what the SO(3) tests need for their null moments under crystal
symmetry (`SO3Discrepancies.md` notes that symmetry "changes the effective dimensions" without
computing them). The exact samplers of the MATLAB harness (the smiley mixture sampler, the dlVP
perturbation "smoothing" alternative of `S2DiscrepancyPowerExperiment.m`, which draws cos of the
angle as 2U^(1/(kappa+1)) - 1, the same sampler the pilot above used) are the test densities of
section 7.

**Where they differ.** Estimation minimises a risk, testing maximises power at a fixed size. The
optimal weights solve different problems (the Julia note's table: b_l = A_l/(A_l + (d_l - A_l)/n)
for estimation against w_l proportional to A_l/d_l for the mean-shift criterion of a test), and
the scales differ: a cutoff B_est ~ n^(1/(2s+2)) balances the estimation error, B_test ~
n^(1/(2s+1)) detection, so a departure is detectable before it is estimable and the test wants
finer resolution than the KDE. A selector chooses once on the data and is judged by its risk; a
data-chosen test weight must be recalibrated with the choice inside the calibration (the
adaptive heat rule does this). Reusing a KDE bandwidth as the test scale is not justified in
general, and the reverse neither.

**Wasserstein.** `SO3Fun/@SO3FunHarmonic/wasserstein.m` (spectral W1, ADMM, analytic lower
bounds), `SO3Fun/slicedWasserstein.m` (closed-form pushforward to a circle, linearised to an
H^-2 norm) and the S2 sliced W1 are location-sensitive: a peak displaced by 2 degrees costs 2
degrees, where L2 charges the whole peak twice. They belong in the evaluation (section 7) as a
second error measure that does not reward undersmoothing. The master note records that the
spectral solver cannot resolve a sharp peak against a Dirac; sliced W1 on atoms and a harmonic
density is cheap and exact for atoms.

**Designed samples.** For points from `optimalSample` the iid variance formula does not apply;
the Julia note's discrepancy bound ||K_b * (Q - P)|| <= D_w(Q, P) sup_l |b_l| / sqrt(w_l) gives a
halfwidth from the achieved discrepancy instead. A separate, deterministic selector, noted in
section 9.

## 7. Evaluation protocol

**Test densities.** On S2: a single vMF/dlVP bump at 5, 10, 20 deg; two and three bumps of
different widths; the girdle (a fibre on S2, `S2Fun` of a great circle smoothed); uniform plus a
5 % sharp peak (the "uniform plus peaks" case, where reference rules fail); the smiley
(`(abs(S2Fun.smiley) + 0.1)/Z`, with the harness's exact sampler); axial versions of each.
On SO(3): unimodal (dlVP 5, 10, 20 deg) in m-3m, 6/mmm and -1; a fibre ODF; the `SantaFe` model;
the 321 model of the OptimalKernel page; a sharp EBSD-like texture (many narrow grain components,
weights from a lognormal grain-size law) with and without intragranular scatter at the pixel
level; and real data from `mtexdata`: forsterite pixels and grain means, titanium, copper, where
the truth is approximated by the estimate from all grains of a large map and judged against
disjoint halves of the map.

**Sample sizes.** 1e2, 1e3, 1e4, 1e5, 1e6, 1e7 (the last on S2 and for timing on SO(3)), at
least 50 replicates below 1e5 and 10 above, all methods on the same samples (paired).

**Error measures.** ISE in the harmonic domain at a bandwidth that resolves the truth (exact for
dlVP and harmonic truths), Hellinger distance and KL divergence on a quadrature grid, the ratio
MISE(selected)/MISE(oracle), the height and position error of the largest peaks (position by
the port's `max` with `numLocal`), the count of spurious local maxima above tau = 0.5 mrd that
do not match a true maximum within the halfwidth, sliced W1, and the fraction of replicates in
which the rule chose h below the oracle (the direct test of "conservative").

**References for comparison.** The oracle halfwidth of the family (exact risk with the true
energies), the degree-wise oracle filter (the lower bound for any kernel family), MATLAB's
KLCV with its with-replacement subsampling, the port's KLCV without replacement, `magicRule`,
`ruleOfThumb`, and the fixed 10 degrees. MATLAB's numbers come through fixed cases run by the
bridge, never a MATLAB benchmark of more than five minutes without asking.

**Timing.** The selector alone and selector plus `calcDensity`, at 1e5, 1e6, 1e7, cold and warm
(numba compilation separate), thread counts 1 and the default.

## 8. Work plan

**H0, the energy vector** (done): `degreeEnergies(v | ori | ebsd, bandwidth, weights=, groups=,
folds=8)` holds ||mu_l||^2, the energies A^_l cross-fitted over F folds (E <C_f, C_g> = W_f W_g A_l
for disjoint folds, so the estimate is unbiased whatever the dependence inside a fold), their
jackknife errors over the folds, s2 = sum w^2 and the largest weight over the independent units
(points, or groups), and the symmetric traces c_l on a subsample of 2000 (Legendre sums over the
frame's group on S2, characters over the pairs of the proper groups on SO(3), the antipode as one
more copy). Equal points go to random folds, groups and weighted points to folds of equal weight.
The coefficients come from one adjoint transform per fold at bandwidth L and tolerance 1e-6
(the energies move by about that relative amount), orthonormal for the uniform probability and
projected as the estimate is. With `folds=1` independent points take a single transform, the
energies being the U-statistic (||mu_l||^2 - s2 c_l) / (1 - s2) with the traces averaged by w^2
over the subsample; UCV, which needs no errors, runs this way unless there are groups. Checked against the direct pairwise sums
at n = 150 to 200 (weights; m-3m directions; axes; -1, m-3m, 6/mmm with 222 and antipodal
cubic misorientations on SO(3)) to 1e-9, `tests/test_halfwidth.py`.

**H1, S2 selectors** (done): `ucvHalfwidth` (C1), `lowerBoundHalfwidth` (C2),
`noiseFloorHalfwidth` (C3) and `selectHalfwidth(E, 'UCV' | 'conservative', noiseFloor=True)`,
either method bounded below by C3. The risk of a candidate adds the energies above the kernel's
degree extent as bias (b below 1e-3 counts as zero); for groups the variance term is the measured
||mu_l||^2 - A^_l below the energy bandwidth, from the unbiased energies also for C2, and the iid
form times the design effect of the upper degrees above it. C3 bounds the fluctuation at x, a sum
of independent terms below b = wmax B (B the peak of the symmetrised kernel) with variance
s2 R y (R = sum b_l^2 c_l its squared norm, y = K_h*f(x)), by Bennett's inequality at every one of
the N independent kernel footprints of the fundamental region (2/(1 - cos h) on S2, pi/(h - sin
h) on SO(3), over the symmetry), with the tolerance tau + tauRel y, tau = tauMax M, M the maximum
of the pilot estimate (the selector's own halfwidth) at 200 points of a subsample. These are
realistic, not proven constants (section 10, item 6): the first version, Bernstein with the
variance B y and a union bound over a covering of mesh delta / L, was a proof and about ten
times loose (below). `bench/halfwidth_s2.py`: eight test densities (model A and B of 5.1, single bumps of 5,
10, 20 degrees, 5 % of a 3 degree peak on a uniform background, a girdle, axial model A), exact
samplers, the exact risk against the oracle of 60 candidates from 1 to 60 degrees, 20 replicates
below 1e5 and 4 above, and the fluctuation max_x (|fhat - K_h*f| - tauRel K_h*f) / max K_h*f on
20000 points. Results (`bench/out/halfwidth/s2_study.txt`), median MISE over the oracle's:

| | n = 1e2 | 1e3 | 1e4 | >= 1e5 | below the oracle |
|---|---|---|---|---|---|
| UCV | 1.02 to 1.07 (worst 2.2) | 1.00 to 1.02 | 1.00 to 1.01 | 1.00 | 25 to 50 % of the runs at n <= 1e3 |
| C2, z = 2 | 1.00 to 1.57 | 1.01 to 1.15, sharp 1.87 | 1.00 to 1.10 | 1.00 to 1.11 | never (720 runs) |
| C2, Bonferroni z | 1.00 to 2.19 | 1.01 to 1.90 | 1.01 to 1.19 | 1.00 to 1.11 | never |
| C2 with the C3 floor, tauMax = 0.2 | 1.0 to 8.2 | 1.7 to 5.1 | 1.0 to 1.7 | 1.00 to 1.11 | never |
| fixed 10 degrees | 1.0 to 9.8 | 1.2 to 40 | 1.2 to 174 | 3.6 to 1100 | |

At C2's own halfwidth the measured fluctuation beyond tauRel K_h*f is already small: median 0.00
to 0.12 of the maximum (0.23 for the girdle at n = 100). The C3 bound is about ten times loose
(with tauMax = 0.2 the realised excess is 0.00 to 0.03): Bernstein takes K^2 <= B K where the
variance is R y with R = ||K||^2, about B/2, and the union bound counts grid points where about
4/h^2 kernel footprints are independent (the measured maxima are 2.3 to 4.8 local standard
deviations against sqrt(2 Lambda) = 5.2 to 5.8). So the floor binds at n <= 1e4 and dominates the
risk there; tauMax = 0.05 or 0.1 cost 4 to 14 times the oracle's MISE. The bandwidth L0 = 32 beat
16. On the sharp model at n = 1e3 every lower bound is zero and C2 falls back to 33 degrees
(Bonferroni: 60), the honest answer that no degree is resolved.

With the realistic constants (`bench/out/halfwidth/s2_study_floor.txt`) the bound lies 1.3 to 1.8
times above the measured maxima of model A (5 and 10 degrees, n = 1e3 and 1e4). UCV with the floor,
median MISE over the oracle's, and the median realised excess:

| | n = 1e2 | 1e3 | >= 1e4 | below the oracle, n = 1e3 | excess, n = 1e2 |
|---|---|---|---|---|---|
| UCV alone | 1.02 to 1.10 | 1.00 to 1.02 | 1.00 to 1.01 | 20 to 55 % | 0.05 to 0.41 |
| UCV + floor, tauMax = 0.05 | 1.0 to 6.7 | 1.2 to 3.6 | 1.00 to 1.26 | 0 % | -0.27 to 0.05 |
| UCV + floor, tauMax = 0.1 | 1.0 to 3.6 | 1.0 to 1.9 | 1.00 to 1.01 | 0 to 40 % | -0.26 to 0.08 |
| UCV + floor, tauMax = 0.2 | 1.0 to 1.9 | 1.00 to 1.16 | 1.00 to 1.01 | 0 to 40 % | -0.26 to 0.11 |
| C2 alone | 1.0 to 1.4 | 1.02 to 1.85 | 1.00 to 1.10 | 0 % | -0.28 to 0.22 |

The floor removes UCV's worst noise at small n (the girdle at n = 100: 0.41 to 0.03 of the
maximum) at a price that tauMax sets; 0.2 costs at most 1.9 at n = 100 and 1.16 from 1e3 on, where
UCV's own undersmoothing is cheap (at most 1.28) and quiet (excess at most 0.10).

Decided: L0 = 32, F = 8, z = 2 (Bonferroni over alpha = 0.05 as `z=None`, the stated
probability), 60 geometric candidates from 1 to 60 degrees on S2, 40 from 2 to 40 on SO(3),
alpha = 0.05, tauMax = 0.2, tauRel = 0.5; the default method UCV with the floor. Timing on S2 (`timing`, energies plus selection): the conservative choice 0.32 s at 1e6 and
1.7 s at 1e7; UCV at bandwidth 256 in one transform 0.17 s and 1.1 s.

**H2, SO(3)** (done): the same functions with the Wigner coefficients projected by both groups,
the characters for the traces, the pilot maximum over the symmetric copies.
`bench/halfwidth_so3.py`: dlVP components of 5, 10, 20 degrees in m-3m, 10 degrees in -1 and in
6/mmm with 222, SantaFe, a cubic mixture (0.5 uniform, 7 and 15 degree components), and pixels
of lognormal grains (mean 50 pixels, 1 degree scatter) drawn from the mixture, 10 replicates at
1e3, 1e4, 1e5; the truth in its harmonic form at bandwidth 96, the realised ISE of every
candidate from the projected coefficients, the oracle the candidate of least mean ISE (which
restricts the candidates to 2.5 degrees and above). Results (`bench/out/halfwidth/so3_study.txt`),
median ISE over the oracle's in the same replicate:

| | n = 1e3 | 1e4 | 1e5 | below the oracle |
|---|---|---|---|---|
| UCV | 1.00 to 1.04 | 1.00 to 1.01 | 1.00 | 10 to 50 % |
| C2 | 1.00 to 1.10 | 1.00 to 1.03 | 1.00 to 1.01 | never |
| conservative (C2 + C3) | 2.6 to 7.4 | 1.0 to 2.5 | 1.00 to 1.01 | never |
| KLCV (the port's, without replacement) | 1.0 to 1.7 | 1.03 to 2.6 | | 0 to 100 % |
| magicRule | 2.3 to 20 | 3.4 to 54 | 4.2 to 74 | |
| fixed 10 degrees | 1.5 to 8.9 | 4.2 to 30 | 16 to 75 | |

Clustered pixels (15, 237, 1976 grains at 1e3, 1e4, 1e5 pixels): taken as independent, every
selector collapses to the smallest candidate, 6 to 31 times the oracle's ISE, the failure section
10 guards against. Folded by grains, UCV is at the oracle (1.00 to 1.02, worst 3.4 with 15 grains),
C2 is safe but wide (1.2, 1.9, 1.5: 40, 22, 6 degrees against 13, 7, 4), and the C3 floor is far
too wide (2.2 to 4.9), since with groups it runs on the Kish number of the area-weighted grains
and the grain's full weight as the largest summand. Timing, m-3m (after the speed-up of 2026-09-27: tolerance 1e-6, UCV in one transform, the
pilot's kernel cut at three halfwidths where its powers ran into denormal numbers, the fold
rotations taken without renormalising): `calcKernel(ori, 'conservative')` 0.50 s at 1e6 (the
eight fold transforms at bandwidth 32 0.34 s, the pilot maximum 0.17 s) and about 3 s at 1e7;
`'UCV'` at bandwidth 64 0.37 s and 2.3 s, plus 0.3 s once per session for the projections of
the group at that bandwidth. The z-folds of the group would shrink the transforms, but finufft
takes a slow path on the folded lattice at bandwidth 65 (1.1 s for 125000 nodes against 0.13
s unfolded), so they are not used. On the copper map of EBSD2ODF (16116 pixels, 375 grains) the grain folds give 'conservative'
40 degrees and 'UCV' 3.7 degrees (the smallest candidate bandwidth 64 resolves), the grain means
14.5, 3.7 and KLCV 4.7.

With the realistic floor (`bench/out/halfwidth/so3_study_floor.txt`): on the independent SO(3)
samples UCV with the floor at tauMax = 0.2 costs at most 1.03 at n = 1e3 and nothing above (0.1:
up to 1.41, 0.05: up to 2.5); KLCV lies at 1.0 to 1.8 and below the oracle in 90 to 100 % of the
SantaFe runs. On the grain pixels, folded by grains, UCV alone is at the oracle (1.03, 1.01 at
227 and 1981 grains, worst 4.4 once with 227), C2 with the unbiased variance at 1.60 and 1.10, but
the floor widens UCV to 1.55 and 1.73: Bennett's bound on the largest term reads the largest
grain, whose weight is 7 to 17 times the mean for lognormal sizes. A Gaussian tail (no
largest-term correction) gives 1.16 and 1.02, the 99 % quantile of the grain weights 1.47 and
1.24. What the largest term measures is real, a single large grain raising its own spike at
a small halfwidth, which the ISE hardly charges and the floor is meant to prevent; so Bennett's
bound stays (section 10, item 7).

Done 2026-09-27 (H5, first part): `calcKernel` defaults to `'UCV'` with the floor (`noiseFloor=True`,
tauMax = 0.2), the new methods seeded so that a default call is reproducible; `calcDensity`
without a halfwidth or kernel selects that way, takes a map with its grains (`groups=` its grain
ids), and falls back to MATLAB's 10 degrees with a warning for flagged pixel orientations
without grain ids and for fewer than two points. The plots keep their fixed halfwidths (5 and
10 degrees). The pages compared with MATLAB name 10 degrees where their densities are pixels or
figures compared with MATLAB's (VectorsDensityEstimation, VectorsAxes, DensityEstimation,
ODFTheory, EBSDOrientationAnalysis, MisorientationDistributionFunction, TransformationTexture,
EllipseBasedParameters); OptimalKernel, VectorsDensityEstimation and EBSD2ODF show the
selection. One observation: C2 moves with the random split into folds (5.52 against 4.07
degrees on the 10000 orientations of OptimalKernel after the draws were reordered).

The bandwidth of UCV on SO(3) (2026-09-27): at bandwidth 64 the narrowest resolved candidate is
3.9 degrees, and on the cubic mixture UCV sits there from about 1e5 orientations on (2.5 at 96,
1.8 at 128). `calcKernel` therefore starts at 64 and computes the energies once more at 128 when
the choice sits on that edge (`ucvAtEdge`); 128 resolves every candidate down to 2 degrees. The
energies at 128 cost 0.95 s at 1e6 and, once per session and group, 3.1 s for the projections.
The z folds of the groups now shrink the lattice wherever the points are fewer than a quarter
of its nodes, with 1.25 oversampling: finufft spreads onto a thin lattice at 2.0 oversampling
on ten threads up to a hundred times slower (37 s against 0.4 s for 125000 points on the
(17, 131, 131) lattice). Measured on the transform alone: 0.075 s against 0.171 s at 1e4 points
and bandwidth 64, 0.70 against 1.10 s at 1e6 points and bandwidth 128; at 1e6 points and bandwidth
64 folding would cost 0.73 against 0.34 s, where the rule does not fold. The folded energies agree
to 7e-7.

**H3, C4 and C5**: the diffusion plug-in (and the MATLAB `kde1d` fix recorded in
docs/findings.md), full-data KLCV with leave-one-grain-out. Decided: which accurate method becomes
`method='...'` beside the default.

**H4, joint with the tests**: the energy vector exposed so that the discrepancy/test code can use
it; the questions of section 9 that join the lines run on the MATLAB master harness or its
Python counterpart, depending on the decision in section 10.

**H5, the API and the pages**: `calcKernel` for vector3d and orientation (done with H2), the
OptimalKernel, VectorsDensityEstimation and EBSD2ODF pages extended by the new methods (done
2026-09-26); `calcDensity(..., halfwidth='auto')` or a keyword, the switch of the defaults, and
the full N study on the pages; MATLAB's `calcKernel` gets the same methods if the user
wants them there.

**H6, later**: local bandwidth (section 3.8), designed samples, uncertainty of the selected
halfwidth by the grain bootstrap.

## 9. Research questions

### Is the C2 bound tight enough to be useful as a default?

How far above the oracle does truncated-energy oversmoothing land as a function of L0, z and n,
on S2 and on SO(3) with symmetry, and does the gap close at the rate the lower confidence bounds
shrink? The comparison is against `calcKernel` with `magicRule`, which is conservative too but
by a factor that does not shrink. Concerns `calcKernel`, `@orientation/calcKernel`,
`mtex/functions/density.py`. Measured (H1, H2): with L0 = 32 and z = 2 C2 lands at 1.0 to 1.6
times the oracle's MISE at n = 100 and within 10 % from 1e3 on (sharp S2 texture at 1e3: 1.9),
never below the oracle, where magicRule is 2 to 74 times the oracle's ISE on SO(3).

### Can the conservative guarantee be made simultaneous and finite-sample?

The Topkis argument needs a_l <= A_l for all degrees at once. Bonferroni over L0 degrees is
simple and loose; is a bootstrap over folds, or a chi-square bound on the U-statistic, tight and
valid at n = 100? Concerns `calcKernel` and the degree energies behind `discrepancy`.

### Does a sup-norm deviation bound give a usable noise floor on S2 and SO(3)?

Constants of the Marcinkiewicz-Zygmund and Bernstein inequalities for spherical and Wigner
polynomials, and whether the resulting h_C3 is below the oracle for realistic tau, i.e. whether
C3 is a floor that rarely binds or the dominant term. Concerns `calcDensity`, `S2FunHarmonic`,
`SO3FunHarmonic` and the maximum search `max`. Measured (H1): with Bernstein and a covering the
floor dominates below n = 1e4 and is about ten times loose; the variance R y in place of B y and
the number of independent kernel footprints in place of the covering are the two factors.

### What is the effective sample size of EBSD pixels, degree by degree?

Measure E ||mu_l||^2 - A_l on real maps against n, n_grains and the Kish number, and test whether
grain-disjoint folds remove the bias at all degrees, including those that resolve the
intragranular spread; compare with spatial blocks when no grains exist. Concerns `calcGrains`,
`calcDensity`, `calcKernel`, `@EBSD/calcDensity`. Measured on simulated grains (H2): taken as
independent the pixels drive every selector to the smallest candidate; the grain folds put UCV at
the oracle, so the cross-fitted energies do remove the bias; C2 with groups is wide by a factor of
1.5 to 3 in halfwidth, since the jackknife over eight folds of a few hundred grains is coarse.

### How much does crystal symmetry really multiply the sample?

The exact variance term uses E_f c_l(X), not (2l+1)^2/(|G_S||G_C|). How large is the difference for
textures at special orientations (cube, Goss), and should `magicRule`'s factor numProper(CS)
numProper(SS) be replaced by the measured trace? Concerns `@orientation/calcKernel`,
`SO3FunHarmonic`, `symmetrise`.

### Does exact spectral UCV beat KLCV on clean data, and by how much on EBSD?

UCV is LSCV without the degree-25 truncation of MATLAB's `LSCV.m`; KLCV is the current default.
Paired over the test densities, with and without clustering. Concerns `KLCV`, `LSCV`,
`@orientation/KLCV`, `@orientation/LSCV`, `calcKernel`.

### Is Botev's diffusion plug-in on the sphere a competitor or only a bridge?

The heat kernel gives the diffusion estimator directly in the degree domain; does its fixed point
beat UCV for smooth f, and how badly does it fail for sharp textures? Concerns `kde1d`,
`SO3GaussWeierstrassKernel`, `calcDensity`.

### Can one choose the test weight and the density bandwidth from the same energy vector, and when should they differ?

The heat test picks t by a calibrated maximum, the KDE picks h by risk; is there a relation
t_test ~ c(s) h^2 with the rates n^(-1/(2s+1)) against n^(-1/(2s+2)), and does a KDE-selected scale
lose much power as a fixed test scale chosen on independent training data? Concerns
`S2Fun/@S2Fun/discrepancy.m`, `SO3Fun/@SO3Fun/discrepancy.m`, `calcKernel`,
`benchmark_S2DiscrepancyPower.m`.

### Does the null covariance of the test give the variance term of the KDE under clustering?

The nonuniform-null covariance of the harmonic residual (the test side) and the cross-fold
variance of the energies (the estimation side) are the same second moments; can one estimator
serve both, including the symmetric traces on SO(3)? Concerns `SO3Fun/@SO3Fun/discrepancy.m`,
`S2DiscrepancyPowerExperiment.m`, `calcKernel`.

### Which halfwidth should a KDE use on points designed by `optimalSample`?

For deterministic designs the risk formula does not apply; the discrepancy bound gives
B ~ (R/delta)^(1/(s+r)). Is that bound sharp on the designs `optimalSample` produces, and does
it select a different halfwidth than C1 on the same number of iid points? Concerns
`optimalSample`, `mtex/analysis/optimalsample.py`, `S2Fun/@S2Fun/discrepancy.m`.

### Is sliced W1 a better error measure than ISE for judging a selector?

ISE rewards resolving sharp peaks at any price; W1 charges displacement. Do the rankings of the
selectors change when judged by `slicedWasserstein`, and which measure matches what users read
off a pole figure? Concerns `SO3Fun/slicedWasserstein.m`,
`SO3Fun/@SO3FunHarmonic/wasserstein.m`, `calcError`.

## 10. Decisions (user, 2026-09-26)

1. **Where.** The selectors are prototyped and benchmarked in the Python port; the joint
   estimation/testing experiments stay in MATLAB master's harness (`tests/lib/`), with the energy
   vector implemented once there too; the chosen selector goes to MATLAB's `calcKernel` last.
2. **Names.** `'conservative'` for C2 with the C3 noise floor, `'UCV'` for C1; `'KLCV'`,
   `'magicRule'`, `'ruleOfThumb'` stay; `calcKernel(v)` for vector3d is new in both ports.
3. **The default.** `calcDensity` without a halfwidth selects with `'conservative'` above a sample
   size to be fixed by the evaluation, and keeps 10 degrees below it; `calcKernel`'s default
   switches from KLCV to `'conservative'`.
4. **Clustered data - be extremely careful.** Orientations taken from an EBSD map are not
   independent: neighbouring pixels of one grain repeat nearly the same orientation, and a
   selector that treats them as n independent draws undersmooths badly. So:
   - a map with grain ids is folded by its grains (`calcDensity(ebsd)`, or the grain ids passed
     with the orientations);
   - orientations taken from a map carry a flag `notIID` (set by `ebsd.orientations` and kept
     through indexing and selection), and an automatic halfwidth on flagged orientations without
     grain ids is refused, with a message asking for the grain ids or an explicit halfwidth;
   - bare orientations without the flag are treated as independent.
   The flag is the simple, fast safeguard; detection from the data (the effective sample size
   from the degree energies) may come later as a second line.
5. **The MATLAB defects are fixed now,** independently of this plan, in `../master` only (branch
   feature/fastHarmonicTransforms, commit 186e53d2f), not in the frameSymmetry reference: `calcKernel`
   subsamples with `randperm` instead of `discretesample`; `BCV.m` passes its weights `w` and
   evaluates `psi{i}.eval(1)` and prints nothing; `kde1d.m` pairs the degrees `(1:n-1).^2` with
   `a(2:end)` and no longer prints `t_star`. Checked on the `mtexcc` bridge: 6000 orientations of a
   10 degree cubic peak give KLCV 3.70 degree (2.2 s) and BCV 2.94 degree (0.5 s, it did not run
   before); 2000 Gaussian samples give the `kde1d` bandwidth 0.2453 against Silverman's 0.2374.

6. **Revised after H2 (user, 2026-09-27).** The default of `calcKernel` and of `calcDensity`
   without a halfwidth or kernel is UCV bounded below by the noise floor C3, the floor with
   realistic constants (the variance R y with R = ||K||^2, an effective count of kernel
   footprints) in place of the proof; `'conservative'` stays as a method. `calcDensity` selects
   always, not above a sample size. Directions made from flagged pixel orientations do not
   carry `notIID`; the pages tell the user to pass the grain ids. Next: H3 (the diffusion
   plug-in, KLCV on all points with leave-one-grain-out), the bandwidth cap of UCV on SO(3),
   and MATLAB's `calcKernel`; the width of the selection with grain folds is under discussion.
7. **The floor on grain maps (user, 2026-09-27).** With grain folds the noise floor keeps Bennett's
   bound on the largest term, the largest grain: it prevents single grains from raising spikes of
   their own, at 1.5 to 1.7 times the oracle's ISE on the simulated grain maps. For `'conservative'`
   with grain folds the variance comes from the unbiased energies.

## 11. References

- Ameijeiras-Alonso, J. (2024). A reliable data-based smoothing parameter selection method for
  circular kernel estimation. Statistics and Computing 34, 73. arXiv:2211.10212.
- Botev, Z. I., Grotowski, J. F., Kroese, D. P. (2010). Kernel density estimation via diffusion.
  Annals of Statistics 38(5), 2916-2957.
- García-Portugués, E. (2013). Exact risk improvement of bandwidth selectors for kernel density
  estimation with directional data. Electronic Journal of Statistics 7, 1655-1685.
  arXiv:1306.0517.
- Hall, P., Watson, G. S., Cabrera, J. (1987). Kernel density estimation with spherical data.
  Biometrika 74(4), 751-762.
- Hielscher, R. (2013). Kernel density estimation on the rotation group and its application to
  crystallographic texture analysis. Journal of Multivariate Analysis 119, 119-143.
- Klemelä, J. (2003). Lower bounds for the asymptotic minimax risk with spherical data. Journal
  of Statistical Planning and Inference 113, 113-136.
- Marron, J. S. (1992). Bootstrap bandwidth selection. In Exploring the Limits of Bootstrap,
  Wiley, 249-262.
- Oliveira, M., Crujeiras, R. M., Rodríguez-Casal, A. (2012). A plug-in rule for bandwidth
  selection in circular density estimation. Computational Statistics and Data Analysis 56,
  3898-3908. arXiv:1202.6076.
- Pham Ngoc, T. M. (2019). Adaptive optimal kernel density estimation for directional data.
  Journal of Multivariate Analysis 173, 248-267. arXiv:1808.02361.
- Sheather, S. J., Jones, M. C. (1991). A reliable data-based bandwidth selection method for
  kernel density estimation. JRSS B 53, 683-690.
- Taylor, C. C. (1989). Bootstrap choice of the smoothing parameter in kernel density
  estimation. Biometrika 76, 705-712.
- Taylor, C. C. (2008). Automatic bandwidth selection for circular density estimation.
  Computational Statistics and Data Analysis 52(7), 3493-3500.
- Tenreiro, C. (2022). Kernel density estimation for circular data: a Fourier series-based
  plug-in approach for bandwidth selection. Journal of Nonparametric Statistics 34(2), 377-406.
- Terrell, G. R., Scott, D. W. (1985). Oversmoothed nonparametric density estimates. JASA 80,
  209-214.
- Terrell, G. R. (1990). The maximal smoothing principle in density estimation. JASA 85,
  470-477.
- Topkis, D. M. (1978). Minimizing a submodular function on a lattice. Operations Research 26,
  305-321.
- Zhang, X., King, M. L., Hyndman, R. J. (2006). A Bayesian approach to bandwidth selection for
  multivariate kernel density estimation. Computational Statistics and Data Analysis 50(11),
  3009-3031.
- For the testing side: Giné (1975), Jupp (2005, 2008), Fernández-de-Marcos and García-Portugués
  (arXiv:2304.04519), Schrab et al. (2023), Hagrass, Sriperumbudur and Li (2024), Fromont and
  Laurent (2006), as collected in `../mtexJulia/docs/research/spherical-goodness-of-fit-overview.md`.
