# optimal sampling of a density on the sphere or the rotation group (MATLAB
# S2Fun/optimalSample, SO3Fun/optimalSample; docs/plans/functions.md F6, ported 2026-09-22)
#
#   ori = optimalSample(odf, 500)                          the nodes, equally weighted
#   ori, c = optimalSample(odf, 200, optimizeWeights=True) the nodes and their weights
#   v = optimalSample(sF, 1000, bandwidth=128)             on the sphere
#   optimalSample(f, nodes, maxIter=, tol=, memory=, method='steepestDescent', diagnostics=True)
#
# The density is represented by the discrete measure sum_j c_j delta_{x_j} and the nodes are
# moved so that the kernel discrepancy sum_l beta_l |mu_l - f_l|^2 with the coefficients of
# the restricted distance kernel becomes small, by a limited memory BFGS iteration with an
# Armijo line search: along the geodesics of the sphere with parallel transport of the
# stored steps, in the left components of the rotation group. Degree zero is left out,
# its kernel coefficient being negative and its term vanishing for weights that sum to
# one. The weights, when optimised, stay on the simplex: through a softmax on the sphere,
# moved jointly with the nodes, and by a multiplicative step on the rotation group,
# alternating with the node steps.
from __future__ import annotations

import math
from collections import namedtuple

import numpy as np

from ..geometry.kernels import blasThreads
from ..geometry.frames import referenceFrame
from ..geometry.orientation import orientation
from ..geometry.quaternion import quaternion, rotation
from ..geometry.units import degree
from ..geometry.vector3d import vector3d
from ..functions.harmonics import invariantProjection, project
from ..functions.kernelfunctions import S2RestrictedDistanceKernel, SO3RestrictedDistanceKernel
from ..functions.quadrature import folds
from ..functions.s2fun import S2FunHarmonic, evenize, projectSpherical, realizeSpherical
from ..functions.s2vectorfield import angularDerivatives
from ..functions.so3fun import SO3FunHarmonic, invertedCoefficients, realizeWigner
from ..functions.so3vectorfield import differentiate
from ..functions.transforms import (DIRECT_SUM_NODES, S2FourierPlan, SO3FourierPlan, sphericalSeries, sphericalSeriesAdjoint,
                         wignerSeries, wignerSeriesAdjoint)

OptimalSampleResult = namedtuple("OptimalSampleResult", ["nodes", "weights", "discrepancy", "history", "iterations", "converged", "reason", "bandwidth"])


def _isSO3(f):
  return hasattr(f, "pdf") and hasattr(f, "CS") and hasattr(f, "SS")


# ------------------------------------------------------------------ the problem

class _Problem:
  """the target coefficients, the kernel weights, the plan on the moving nodes and the
  projections onto the symmetric functions"""

  def __init__(self, f, xs, L):
    self.sphere = not _isSO3(f)
    self.L = L
    if self.sphere:
      h = S2FunHarmonic(f, bandwidth=L).truncate(L)
      mass = 4 * np.pi * np.real(h.mean())
      if not (np.isfinite(mass) and mass > 0):
        raise ValueError("the density must have positive finite mass")
      self.target = S2FunHarmonic(h._fhat / mass, h.frame, real=True, antipodal=h.antipodal, symmetrise=False, _internal=True)
      A = S2RestrictedDistanceKernel().calcFourier(L)
      beta = np.zeros((L + 1) ** 2)
      for l in range(1, L + 1):
        if h.antipodal and l % 2:
          continue
        beta[l * l:(l + 1) ** 2] = 4 * np.pi * A[l] / (2 * l + 1)
      fr = h.frame
      self.projection = fr.sym if fr is not None and fr.numSym() > 1 else None
    else:
      h = SO3FunHarmonic(f, bandwidth=L).truncate(L)
      mass = float(np.real(h.mean()))
      if not (np.isfinite(mass) and mass > 0):
        raise ValueError("the density must have positive finite mass")
      self.target = SO3FunHarmonic(h._fhat / mass, h.CS, h.SS, real=True, antipodal=h.antipodal, symmetrise=False, _internal=True)
      A = SO3RestrictedDistanceKernel().calcFourier(L)
      beta = np.zeros(len(h._fhat))
      for l in range(1, L + 1):
        beta[_range(l)] = A[l] / (2 * l + 1)
      # the gradient evaluates left derivatives, which a left symmetry does not keep, so the
      # plan folds by the crystal side alone
      self.folds = (folds(h.CS, h.SS)[0], 1)
      self.projection = (invariantProjection(h.SS, L), invariantProjection(h.CS, L))
    self.beta = beta
    self.ones = np.ones(len(xs))
    # the plan runs four times an iteration, so the low oversampling factor pays whatever
    # the size, at the tolerance the wide window of Julia's plans has
    self.plan = None
    if L > 0 and len(xs) >= DIRECT_SUM_NODES:
      nufft = dict(eps=1e-7, upsampfac=1.25)
      self.plan = S2FourierPlan(vector3d(xs), L, real=True, **nufft) if self.sphere else SO3FourierPlan(rotation(xs), L, real=True, folds=self.folds, **nufft)
    if not np.all(np.isfinite(self.target._fhat)):
      raise ValueError("the density coefficients have to be finite")

  # the plan follows the nodes
  def setNodes(self, xs):
    if self.plan is not None:
      self.plan.setNodes(vector3d(xs) if self.sphere else rotation(xs))

  # the coefficients onto the symmetric, even or exchange invariant real functions
  def project(self, c):
    if self.sphere:
      if self.projection is not None:
        projectSpherical(c, self.projection)
      if self.target.antipodal:
        evenize(c)
      realizeSpherical(c)
    else:
      project(c, *self.projection)
      if self.target.antipodal:
        c[:] = (c + invertedCoefficients(c)) / 2
      realizeWigner(c)
    return c

  # the coefficients of the discrete measure with the weights c
  def moment(self, xs, c):
    if self.plan is None:
      h = sphericalSeriesAdjoint(vector3d(xs), self.ones, self.L, weights=c) if self.sphere else wignerSeriesAdjoint(rotation(xs), self.ones, self.L, weights=c)
    else:
      h = self.plan.adjoint(self.ones, c)
    return self.project(np.array(h, dtype=complex))

  def values(self, xs, c):
    if self.plan is None:
      return np.real(sphericalSeries(c, vector3d(xs), real=True)) if self.sphere else np.real(wignerSeries(c, rotation(xs), real=True))
    return np.real(self.plan(c))

  def objective(self, d):
    return float(np.sum(self.beta * np.abs(d) ** 2))

  def residual(self, xs, c):
    self.setNodes(xs)
    d = self.moment(xs, c) - self.target._fhat
    return self.objective(d), d

  # the gradient with respect to the nodes, and to the softmax logits of the weights
  def gradient(self, xs, c, residual, weights=False):
    coeff = self.beta * residual
    dc = angularDerivatives(coeff) if self.sphere else differentiate(coeff, "left")
    vals = np.stack([self.values(xs, dc[:, i]) for i in range(3)], axis=-1)
    if self.sphere:
      vals = np.cross(vals, xs)
    vals = vals * (2 * c)[:, None]
    if weights:
      gc = 2 * self.values(xs, coeff)
      return np.concatenate([vals.ravel(), c * (gc - c @ gc)])
    return vals.ravel()


def _range(l):
  from ..functions.harmonics import deg2dim
  return slice(deg2dim(l), deg2dim(l + 1))


# ------------------------------------------------------------------ the steps

# a step along the direction: geodesics on the sphere, the exponential map on the left of the
# rotation group
def _step(p, xs, d, alpha):
  M = len(xs)
  V = d[:3 * M].reshape(M, 3)
  speed = np.linalg.norm(V, axis=1)
  if p.sphere:
    out = xs.copy()
    move = speed > 0
    s = speed[move]
    out[move] = np.cos(alpha * s)[:, None] * xs[move] + (np.sin(alpha * s) / s)[:, None] * V[move]
    return out
  out = xs.copy()
  move = speed > 0
  if np.any(move):
    axis = vector3d(V[move] / speed[move][:, None])
    r = rotation.byAxisAngle(axis, alpha * speed[move])
    out[move] = (r * rotation(xs[move])).data
  return out


# the vector w carried along the step: unchanged in the global coordinates of the rotation
# group, parallel transported along the geodesic on the sphere
def _transport(p, w, xs, d, alpha):
  out = w.copy()
  if not p.sphere:
    return out
  M = len(xs)
  W = out[:3 * M].reshape(M, 3)
  V = d[:3 * M].reshape(M, 3)
  speed = np.linalg.norm(V, axis=1)
  j = np.flatnonzero(speed > 0)
  u = V[j] / speed[j, None]
  t = alpha * speed[j, None]
  a = np.sum(W[j] * u, axis=1, keepdims=True)
  W[j] = W[j] - a * (np.sin(t) * xs[j] + 2 * np.sin(t / 2) ** 2 * u)
  return out


# the L-BFGS two loop recursion
def _direction(g, S, Y):
  if not S:
    return -g
  q = g.copy()
  a, rho = np.zeros(len(S)), np.zeros(len(S))
  for j in range(len(S) - 1, -1, -1):
    rho[j] = 1 / (Y[j] @ S[j])
    a[j] = rho[j] * (S[j] @ q)
    q -= a[j] * Y[j]
  q *= (S[-1] @ Y[-1]) / (Y[-1] @ Y[-1])
  for j in range(len(S)):
    q += S[j] * (a[j] - rho[j] * (Y[j] @ q))
  return -q


def _softmax(z):
  c = np.exp(z - np.max(z))
  return c / np.sum(c)


# a multiplicative feasible direction on the simplex and the exact quadratic line minimum
def _weightStep(p, xs, c, residual, innerIter):
  for _ in range(innerIter):
    g = 2 * p.values(xs, p.beta * residual)
    lam = g - c @ g
    d = -c * lam
    delta = p.moment(xs, d)
    denom = 2 * p.objective(delta)
    slope = g @ d
    if not (denom > 0 and slope < 0):
      break
    alpha = min(-slope / denom, 1 / np.max(np.abs(lam)))
    cn = np.maximum(c + alpha * d, 0)
    cn /= np.sum(cn)
    dn = p.moment(xs, cn) - p.target._fhat
    if p.objective(dn) > p.objective(residual):
      break
    c, residual = cn, dn
  return c, residual


# ------------------------------------------------------------------ the optimisation loop

def _optimize(p, xs, c, optimizeWeights, maxIter, tol, memory, warmUp, tolWeights, tolJ, innerIter, minWeight):
  M = len(xs)
  optimizeWeights = optimizeWeights and M > 1
  joint = optimizeWeights and p.sphere
  z = np.log(c) if joint else np.zeros(0)
  value, residual = p.residual(xs, c)
  history = [value]
  S, Y = [], []
  prevStep = 1.0
  g = None
  activeOld = False
  iterations, reason = 0, "max_iterations"
  for i in range(1, maxIter + 1):
    iterations = i
    active = optimizeWeights and i > warmUp
    entering = active and not activeOld
    if entering:
      S, Y, g = [], [], None
    activeOld = active
    oldc = c.copy()
    if active and not p.sphere:
      c, residual = _weightStep(p, xs, c, residual, innerIter)
      value = p.objective(residual)
      g = None
    if g is None:
      g = p.gradient(xs, c, residual, weights=joint and active)
    if np.linalg.norm(g) == 0:
      if optimizeWeights and not active:
        warmUp = i
        g = None
        continue
      reason = "stationary"
      break
    d = _direction(g, S, Y)
    if p.sphere:
      V = d[:3 * M].reshape(M, 3)
      V -= np.sum(V * xs, axis=1)[:, None] * xs
    slope = d @ g
    if not slope < 0:
      S, Y = [], []
      d = -g
      slope = -float(g @ g)
    speeds = np.linalg.norm(d[:3 * M].reshape(M, 3), axis=1)
    cap = np.pi / np.max(speeds) if np.max(speeds) > 0 else np.inf
    if joint and active:
      dzmax = np.max(np.abs(d[3 * M:]))
      if dzmax > 0:
        cap = min(cap, 1 / dzmax)
    alpha = min(cap if (entering and joint) else (2 * prevStep if not S else 1.0), cap)
    accepted = False
    xn, cn, zn, dn, vn = xs, c, z, residual, value
    for _ in range(60):
      xn = _step(p, xs, d, alpha)
      zn = z + alpha * d[3 * M:] if joint and active else z
      cn = _softmax(zn) if joint and active else c
      vn, dn = p.residual(xn, cn)
      if np.isfinite(vn) and vn < value + 1e-4 * alpha * slope:
        accepted = True
        break
      alpha /= 2
      if alpha < 1e-16:
        break
    if not accepted:
      p.setNodes(xs)
      if optimizeWeights and not active:
        warmUp = i
        g = None
        continue
      reason = "line_search_failed"
      break
    history.append(vn)
    # a small step is convergence only when the objective has also stopped falling
    small = (not entering) and alpha * np.max(speeds) < tol and np.max(np.abs(cn - oldc)) < tolWeights
    small = small and value - vn < tolJ * max(value, np.finfo(float).eps)
    if small and not (optimizeWeights and not active):
      xs, c, z, residual, value = xn, cn, zn, dn, vn
      reason = "step_tolerance"
      break
    gn = p.gradient(xn, cn, dn, weights=joint and active)
    if memory > 0:
      s = _transport(p, alpha * d, xs, d, alpha)
      y = gn - _transport(p, g, xs, d, alpha)
      S = [_transport(p, sk, xs, d, alpha) for sk in S]
      Y = [_transport(p, yk, xs, d, alpha) for yk in Y]
      if s @ y > 1e-12 * np.linalg.norm(s) * np.linalg.norm(y):
        S.append(s)
        Y.append(y)
        if len(S) > memory:
          S.pop(0)
          Y.pop(0)
    xs, c, z, residual, value, g = xn, cn, zn, dn, vn, gn
    prevStep = alpha
    if small and optimizeWeights and not active:
      warmUp = i
  if history[-1] != value:
    history.append(value)
  # pruning changes the measure, so the discrepancy is that of what is kept
  if optimizeWeights and minWeight > 0:
    keep = c >= minWeight
    if np.any(keep) and not np.all(keep):
      xs, c = xs[keep], c[keep]
      c = c / np.sum(c)
      p = _Problem(p.target, xs, p.L)
      value, residual = p.residual(xs, c)
  return xs, c, value, history, iterations, reason


# ------------------------------------------------------------------ the initial sample and the output

# spherical Fibonacci points, and a quasi uniform sequence of rotations
def _initial(f, initial):
  sphere = not _isSO3(f)
  if isinstance(initial, (int, np.integer)):
    n = int(initial)
    if n <= 0:
      raise ValueError("the sample count must be positive")
    j = np.arange(1, n + 1)
    if sphere:
      z = 1 - 2 * (j - 0.5) / n
      phi = j * np.pi * (3 - math.sqrt(5))
      return np.stack([np.sqrt(1 - z * z) * np.cos(phi), np.sqrt(1 - z * z) * np.sin(phi), z], axis=-1)
    a = 2 * np.pi * np.mod(j * math.sqrt(2), 1)
    b = np.arccos(2 * np.mod(j * math.sqrt(3), 1) - 1)
    g = 2 * np.pi * np.mod(j * math.sqrt(5), 1)
    return rotation.byEuler(a, b, g, "Matthies").data.reshape(-1, 4)
  if sphere:
    v = initial if isinstance(initial, vector3d) else vector3d(initial)
    return v.normalize().data.reshape(-1, 3).astype(float)
  r = initial if isinstance(initial, quaternion) else rotation(initial)
  return r.data.reshape(-1, 4).astype(float)


def optimalSample(f, n, *args, optimizeWeights=False, bandwidth=None, maxIter=None, tol=None, weights=None, memory=5,
                  method="lbfgs", warmUp=None, tolWeights=None, tolJ=1e-4, innerIter=5, minWeight=0.0, diagnostics=False):
  """the sample of n nodes that reproduces a density best up to the bandwidth: the nodes,
  and with optimizeWeights=True the nodes and their weights (volume fractions summing to
  one); n may be the starting nodes instead; diagnostics=True returns the result with
  the discrepancy, its history, the iteration count and the stop reason

    ori = optimalSample(odf, 500)
    ori, c = optimalSample(odf, 200, optimizeWeights=True, bandwidth=64)
    v = optimalSample(sF, 1000, bandwidth=128)
    optimalSample(f, nodes, method='steepestDescent', maxIter=1000, tol=0.05*degree, minWeight=1e-4)
  """
  sphere = not _isSO3(f)
  for a in args:
    if isinstance(a, str) and a.lower() == "steepestdescent":
      method = "steepestDescent"
  if not getattr(f, "isReal", True):
    raise ValueError("sampling requires a real density")
  L = (128 if sphere else 32) if bandwidth is None else int(bandwidth)
  maxIter = (1000 if sphere else 100) if maxIter is None else int(maxIter)
  tol = (0.01 * degree if sphere else 0.1 * degree) if tol is None else float(tol)
  warmUp = -(-maxIter // 5) if warmUp is None else int(warmUp)
  if L < 0 or maxIter < 0 or memory < 0 or warmUp < 0 or innerIter < 0:
    raise ValueError("the bandwidth and the iteration counts are nonnegative")
  if not all(np.isfinite(x) and x >= 0 for x in (tol, tolJ, minWeight)):
    raise ValueError("the tolerances are finite and nonnegative")
  if method not in ("lbfgs", "steepestDescent"):
    raise ValueError("the method is 'lbfgs' or 'steepestDescent'")
  if method == "steepestDescent":
    memory = 0
  xs = _initial(f, n)
  if len(xs) == 0:
    raise ValueError("the sample cannot be empty")
  if not np.all(np.isfinite(xs)):
    raise ValueError("the nodes have to be finite")
  M = len(xs)
  c = np.full(M, 1 / M) if weights is None else np.asarray(weights, dtype=float).ravel()
  if c.size != M:
    raise ValueError("one weight is required per node")
  if not (np.all(np.isfinite(c)) and np.all(c >= 0) and np.isfinite(np.sum(c)) and np.sum(c) > 0):
    raise ValueError("the weights are nonnegative with a positive finite sum")
  if optimizeWeights and sphere and np.any(c == 0):
    raise ValueError("the starting weights of the softmax have to be positive")
  c = c / np.sum(c)
  tw = 1e-3 / M if tolWeights is None else float(tolWeights)
  p = _Problem(f, xs, L)
  # the iteration's BLAS calls are products of vectors, and OpenBLAS's waiting threads would
  # take the cores of the transforms between them
  with blasThreads(1):
    xs, c, value, history, iterations, reason = _optimize(p, xs, c, optimizeWeights, maxIter, tol, memory,
                                                          min(warmUp, max(0, maxIter - 1)), tw, tolJ, innerIter, minWeight)
  if sphere:
    nodes = vector3d(xs, antipodal=bool(p.target.antipodal), frame=p.target.frame)
  else:
    nodes = orientation(rotation(xs), p.target.CS, p.target.SS, antipodal=bool(p.target.antipodal))
  if diagnostics:
    return OptimalSampleResult(nodes, c, value, np.array(history), iterations, reason in ("stationary", "step_tolerance"), reason, L)
  return (nodes, c) if optimizeWeights else nodes


def _install():
  from ..functions.bingham import S2FunBingham, SO3FunBingham
  from ..functions.handles import S2FunHandle, S2FunSum, SO3FunGeneric
  from ..functions.rbf import SO3FunCBF, SO3FunRBF
  for cls in (SO3FunHarmonic, SO3FunGeneric, SO3FunRBF, SO3FunCBF, SO3FunBingham, S2FunHarmonic, S2FunHandle, S2FunSum, S2FunBingham):
    cls.optimalSample = lambda self, n, *a, **kw: optimalSample(self, n, *a, **kw)


_install()

__all__ = ["optimalSample", "OptimalSampleResult"]
