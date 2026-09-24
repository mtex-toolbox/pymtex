# # RBF-Kernel Approximation from Discrete Data
#
# [Approximating Orientation-Dependent Functions from Discrete Data](SO3FunApproximationTheory.html)
# defines the shared approximation problem and compares harmonic, RBF, and Bingham models.
# This page assumes that an RBF model is appropriate and shows how to choose density
# constraints, kernel halfwidth, centre orientations, and a least-squares solver.
#
# RBF approximation is often useful for an ODF or another density when the number of
# observations and their noise level are not too large. It can also fit a general scalar
# response, but only `'density'` imposes the nonnegativity and mean-1 constraints required
# of a density.

# +
import time

import matplotlib.pyplot as plt
import numpy as np

from mtex import (SO3AbelPoissonKernel, SO3FunRBF, degree, equispacedSO3Grid, interp, loadOrientation, mean, min,
                  mtexColorbar, mtexdatafile, norm, plot, plotSection, plottingConvention)

plottingConvention.default('y↑→x')
# -

# ## A noisy density data set
#
# Start with the data used on the preceding two pages. This time the noise standard
# deviation is 20 percent of the standard deviation of the supplied values.

# +
rng = np.random.default_rng(1)
fname = mtexdatafile('dubnaValues')
ori, S = loadOrientation(fname, columnNames=['phi1', 'Phi', 'phi2', 'values'])
val = S['values'] + rng.standard_normal(S['values'].shape) * 0.2 * np.std(S['values'], ddof=1)

plotSection(ori, val, 'sigma', all=True)
# -

# The broad high-value regions remain visible, but individual markers vary much more than on
# the harmonic page. The noise-free values `S['values']` are retained only so that this
# controlled example can later measure recovery error. They would not be available for an
# unknown experimental response.
#
# ## Fit a density
#
# [interp](rotation.interp.html) uses an RBF model by default and calls
# [SO3FunRBF.interpolate](SO3FunRBF.interpolate.html). The `'density'` flag selects modified
# least squares, `'mlsq'`. This solver constrains the kernel weights to be nonnegative, and
# MTEX normalizes the fitted function to mean 1.

SO3FDensity = interp(ori, val, 'density')
SO3FDensity

# ---

minimumDensityValue = min(SO3FDensity)[0]
minimumDensityValue

# ---

meanDensityValue = mean(SO3FDensity)
meanDensityValue

# ---

relativeDataResidual = norm(SO3FDensity.eval(ori) - val) / norm(val)
relativeDataResidual

# ---

plot(SO3FDensity, 'sigma')
mtexColorbar()

# The fit retains the broad peaks while smoothing the marker-to-marker fluctuations. Its
# printed minimum and mean verify the density constraints. A nonzero residual is expected
# because a smooth constrained model cannot reproduce every noisy value. If the fit reaches
# the 100-iteration `'mlsq'` limit, its coefficients may not yet be optimal.
#
# ## Fit without density constraints
#
# Omitting `'density'` selects unconstrained LSQR. The kernel grid still smooths the data,
# so the result is not literally "undenoised." It simply has more freedom to reduce the
# sample residual, including by using negative weights or changing the mean.

SO3FFree = interp(ori, val)
SO3FFree

# ---

minimumFreeValue = min(SO3FFree)[0]
minimumFreeValue

# ---

meanFreeValue = mean(SO3FFree)
meanFreeValue

# ---

relativeFreeResidual = norm(SO3FFree.eval(ori) - val) / norm(val)
relativeFreeResidual

# ---

plot(SO3FFree, 'sigma')
mtexColorbar()

# The unconstrained residual is smaller, but the printed minimum and mean show that this fit
# is not automatically a density. Fine-scale features that only reduce the training residual
# are possible overfitting, not evidence that the unconstrained model is physically better.
#
# ## Separate halfwidth from centre spacing
#
# The kernel halfwidth controls how far each centre influences neighbouring orientations. A
# large halfwidth produces a smooth fit. A halfwidth that is small relative to the data
# spacing can reproduce noise.
#
# By default, the `halfwidth` option also sets the resolution of the equispaced centre grid.
# That coupling changes both the kernel shape and the number of unknown weights. Specify
# `resolution` separately when the aim is to isolate the effect of halfwidth.

SO3FNarrow = interp(ori, val, 'density', halfwidth=2 * degree, resolution=5 * degree)
SO3FNarrow

# ---

plot(SO3FNarrow, 'sigma')

# The 2 degree kernels are narrower than the fixed 5 degree centre spacing. The resulting
# small-scale peaks track local samples rather than the broad structure. As a rule of thumb,
# start with a halfwidth at least as large as the resolution of the data, then validate that
# choice.

SO3FSmooth = interp(ori, val, 'density', halfwidth=10 * degree, resolution=5 * degree)
SO3FSmooth

# ---

plot(SO3FSmooth, 'sigma')

# The 10 degree kernels overlap much more. The broad peaks remain while narrow fluctuations
# merge into a smoother surface.
#
# ## Supply the centre orientations
#
# Use `SO3Grid` when the centre grid must be controlled directly. Its symmetry should match
# the sample orientations.

S3G = equispacedSO3Grid(ori.CS, resolution=5 * degree)
S3G

# ---

SO3FGrid = interp(ori, val, 'density', SO3Grid=S3G)
SO3FGrid

# ---

plot(SO3FGrid, 'sigma')

# The fitted function reports the supplied 5 degree centre grid. The plot resembles the
# default density fit because that fit makes the same grid choice; the explicit form is
# useful when several fits must share centres.
#
# ## Measure a halfwidth sweep
#
# Keep the centre resolution at 5 degrees while changing only halfwidth. The error is
# measured against the noise-free values in this simulation, not against the noisy training
# values.

# +
hw = np.array([20, 15, 12.5, 10, 7.5, 5, 2.5])
err = np.zeros(len(hw))
for k in range(len(hw)):
  SO3Fhw = interp(ori, val, 'density', halfwidth=hw[k] * degree, resolution=5 * degree)
  err[k] = norm(SO3Fhw.eval(ori) - S['values']) / norm(S['values'])

plt.figure(figsize=(6, 3.5))
plt.plot(hw, err, 'o--')
plt.gca().invert_xaxis()
plt.xlabel('halfwidth [deg]')
plt.ylabel('relative recovery error')
# -

# Read the curve from broad kernels on the left towards narrow kernels on the right. The
# minimum balances smoothing bias against sensitivity to noise. A sweep that also changed
# the centre-grid resolution would mix two effects; this one changes the kernel alone. If
# the fits reach the iteration cap, the result is the best of the capped fits rather than a
# claim that every solver reached its optimum.

bestIndex = int(np.argmin(err))
bestHalfwidth = hw[bestIndex]
bestHalfwidth

# ---

bestRecoveryError = err[bestIndex]
bestRecoveryError

# ---

SO3FBest = interp(ori, val, 'density', halfwidth=bestHalfwidth * degree, resolution=5 * degree)
plot(SO3FBest, 'sigma')

# The selected fit retains the broad regions while suppressing much of the added noise. A
# small training residual alone could not select it; the known noise-free target makes this
# a validation experiment. With real data, use held-out samples or independent physical
# knowledge instead.
#
# If too few observations constrain too many centre weights, the system is
# underdetermined. A small halfwidth does not supply the missing information; use fewer
# centres, a stronger constraint, or additional smoothness assumptions.
#
# ## Choose another kernel
#
# `halfwidth` constructs a [SO3DeLaValleePoussinKernel](SO3DeLaValleePoussinKernel.html).
# Pass `kernel` to use another [rotational kernel function](SO3Kernels.html).

psi = SO3AbelPoissonKernel(halfwidth=5 * degree)
psi

# ---

SO3FAbel = interp(ori, val, 'density', kernel=psi, resolution=5 * degree)
plot(SO3FAbel, 'sigma')

# The Abel-Poisson fit places its peaks in the same broad regions, but its tails and peak
# shapes differ from the de la Vallee Poussin result. Kernel family and halfwidth are
# separate modelling choices.
#
# ## Exact interpolation
#
# If the values are noise-free, the input orientations themselves can be used as kernel
# centres with `exact=True`. For distinct nodes and a positive-definite kernel, the
# resulting kernel matrix is symmetric and positive definite, so the linear system has a
# unique exact solution.
#
# That matrix is generally dense. Construction, solution, and later evaluation can therefore
# become prohibitively expensive for the complete data set. The example uses a reproducible
# subset to make the cost visible without attempting the original all-node dense system.

exactNodes = ori[::20]
exactValues = S['values'][::20]
numberOfExactNodes = exactNodes.size
numberOfExactNodes

# ---

t = time.perf_counter()
SO3FExact = SO3FunRBF.interpolate(exactNodes, exactValues, exact=True, halfwidth=7.5 * degree)
exactFitTime = time.perf_counter() - t

plot(SO3FExact, 'sigma')

# The exact-centre fit follows the noise-free subset closely. Every value moves the weight
# of the kernel at its own node, so noise in the values would go straight into the fit, which
# is why exact interpolation is unsuitable for noisy data. Because these centres are not a
# structured grid, later evaluations are slower than for a grid-centred RBF model.
#
# The iterative solver stops at a requested tolerance or iteration limit, so its computed
# residual need not reach machine precision. Exact centres make an exact solution available;
# they do not force the iterative solver to reach it.

relativeExactResidual = norm(SO3FExact.eval(exactNodes) - exactValues) / norm(exactValues)
relativeExactResidual

# ---

minimumExactValue = min(SO3FExact)[0]
minimumExactValue

# The printed residual checks how closely this run solved the system. Exact interpolation of
# nonnegative values takes the positive solver here as well, so the minimum stays positive;
# with values of both signs the fit is unconstrained and may become negative.
#
# ## Choose a least-squares solver
#
# The harmonic page introduces LSQR stopping conditions. RBF interpolation uses `tol` and
# `maxit` in the same spirit, but its defaults depend on the solver: unconstrained `'lsqr'`
# uses at most 30 iterations, whereas density-constrained `'mlsq'` uses at most 100. Both
# default to `tol=1e-3`.
#
# The two solvers serve different constraints:
#
# * `'lsqr'` is the fast default for an unconstrained least-squares fit.
# * `'mlsq'` enforces positive normalized weights and is selected by `'density'`, or by
#   values of one sign at fewer than 10,000 nodes.
#
# With `withIterations=True`, `SO3FunRBF.interpolate` also returns the iteration count, not
# a convergence flag. Compare residuals and fits when the count reaches the selected limit.

# +
f1, iter1 = SO3FunRBF.interpolate(ori, val, withIterations=True)
residualNorm1 = norm(f1.eval(ori) - val)
print(f'default: iterations {iter1}, residual norm {residualNorm1:.6g}')

f2, iter2 = SO3FunRBF.interpolate(ori, val, tol=1e-15, maxit=100, withIterations=True)
residualNorm2 = norm(f2.eval(ori) - val)
print(f'tight tol: iterations {iter2}, residual norm {residualNorm2:.6g}')
# -

# If the second run reaches 100 iterations, it has stopped at `maxit` rather than satisfying
# the very small tolerance. Increase the limit only when the additional accuracy matters to
# validation or interpretation.
#
# ## The maths behind RBF approximation
#
# An RBF model places one rotational kernel $\Psi$ at each centre $R_n$:
#
# $$ f(x) = \sum_{n=1}^N c_n\, \Psi\!\left(\cos\frac{\omega(x,R_n)}{2}\right). $$
#
# Here $\omega(x,R_n)$ is the rotation angle between the evaluation orientation and the
# centre. The coefficients $c_n$ are the unknown weights. With sample pairs $(x_m,v_m)$,
# unconstrained fitting solves
#
# $$ \min_c \lVert Kc-v\rVert_2^2, \qquad
# K_{mn}=\Psi\!\left(\cos\frac{\omega(x_m,R_n)}{2}\right). $$
#
# The kernel matrix is sparse in the usual grid-centred approximation because MTEX neglects
# interactions outside a halfwidth-dependent angular neighbourhood. The `exact` flag
# evaluates every interaction, which is why that matrix loses the computational advantage.
#
# For a density fit, modified least squares additionally requires nonnegative coefficients
# with a prescribed sum. MTEX then normalizes the resulting RBF function to mean 1. These
# constraints explain why a density fit can have a larger sample residual than
# unconstrained LSQR.
#
# ## References
#
# * H. Schaeben, F. Bachmann, and J.-J. Fundenberger,
#   [Construction of weighted crystallographic orientations capturing a given orientation density function](https://doi.org/10.1007/s10853-016-0496-1),
#   _Journal of Materials Science_ 52 (2017), 2077-2090, develops the positive normalized
#   RBF approximation implemented by `'mlsq'`.
#
# ## Next
#
# Continue with [Approximation and Quadrature](SO3FunQuadrature.html) to replace scattered
# observations by values on a quadrature grid and compute a harmonic representation of a
# known orientation-dependent function.
#
# ## Technical Details
#
# MATLAB's second output of `SO3FunRBF.interpolate` is `withIterations=True` here. MATLAB
# also offers the solvers `'mlrl'`, `'lsqnonneg'`, `'lsqlin'` and `'nnls'`; the port has
# `'lsqr'` and `'mlsq'`. The noise is drawn by NumPy. MATLAB evaluates a sum of kernels
# through the sparse matrix of the kernel at every point; the port evaluates one of more
# than two million kernel terms through its harmonic form at the bandwidth of the kernel,
# which reproduces the sum to about $4\mathbin{\cdot}10^{-4}$. The residuals at the samples
# cannot fall below that: the tight LSQR run converges in 41 iterations as MATLAB's does,
# which reports a residual norm of $7.9\mathbin{\cdot}10^{-13}$, while the port's evaluation
# gives 0.37. MATLAB's density fits keep 71802, 119075 and 69937 centres where the port's
# keep 73610, 119086 and 69896.
