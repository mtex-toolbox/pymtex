# the Taylor model of crystal plasticity and its linear program (MATLAB
# TensorAnalysis/@strainTensor/calcTaylor; docs/plans/functions.md P4e)
#
#   M, b, W = calcTaylor(eps, sS)                       a strain in the crystal frame, or a list of them
#   M, b, W = calcTaylor(inv(ori) * eps, sS, plasticWork=True)
#   MFun, _, WFun = calcTaylor(eps, sS)                 a strain in the specimen frame: functions of the orientation
#   x = linearProgram(c, A, b)                          min c.x with A x = b, x >= 0, a vertex
#   x = linprogCentre(c, A, y)                          the same at the analytic centre, many at once
#
# The Taylor factor is the least sum of shears (or of plastic work) on the slip systems
# that accommodates the strain: a linear program in the shears. Its optimal face is mostly
# more than a point, so the shears are its analytic centre, the symmetric and unique ones, as
# MATLAB's linprogCentre gives them; a vertex of it by a two-phase simplex with Bland's rule,
# or the shears of least norm on it, on request. The plastic spin is the antisymmetric part
# of the slip.
from __future__ import annotations

from collections import namedtuple

import numpy as np

from ..geometry.frames import crystalFrame, referenceFrame
from ..geometry.kernels import backend
from ..geometry.orientation import orientation
from ..geometry.quaternion import rotation
from ..geometry.slipsystem import slipSystem
from ..geometry.vector3d import vector3d
from ..tensors.tensor import spinTensor, strainTensor
from .kernels import taylorKernel
from ..geometry.install import install
from ..geometry.display import aligned, frameLabel

LinearProgramResult = namedtuple("LinearProgramResult", ["solution", "status", "iterations", "residual"])
TaylorResult = namedtuple("TaylorResult", ["M", "b", "W"])
# the selections of the shears on the optimal face, in the order the compiled loop numbers them
SELECTIONS = ("basic", "minimumNorm", "centre")


# ------------------------------------------------------------------ the simplex

def _pivot(T, basis, r, j):
  T[r] /= T[r, j]
  col = T[:, j].copy()
  col[r] = 0
  T -= np.outer(col, T[r])
  basis[r] = j


def _cost(T, basis, c):
  # the reduced costs of every column, the objective in the last one
  T[-1, :] = 0
  T[-1, :-1] = c[basis] @ T[:-1, :-1] - c
  T[-1, -1] = c[basis] @ T[:-1, -1]


def _solve(T, basis, columns, tol, budget):
  it = 0
  while True:
    entering = np.flatnonzero(T[-1, :columns] > tol)
    if entering.size == 0:
      return "optimal", it
    if it == budget:
      return "iteration_limit", it
    j = entering[0]                                   # Bland's rule: the first improving column
    col = T[:-1, j]
    rows = np.flatnonzero(col > tol)
    if rows.size == 0:
      return "unbounded", it
    ratio = T[rows, -1] / col[rows]
    best = np.min(ratio)
    ties = rows[ratio <= best + tol]
    r = ties[np.argmin(basis[ties])]                  # and the smallest basic index on a tie
    _pivot(T, basis, r, j)
    it += 1


def linearProgram(c, A, b, tol=1e-9, maxIter=10000, diagnostics=False):
  """the minimiser of c.x under A x = b, x >= 0, by a scaled two-phase simplex with Bland's
  rule; None when the program is infeasible, unbounded or out of iterations, or with
  diagnostics=True the result with its status, pivot count and relative residual"""
  c, A, b = np.asarray(c, dtype=float).ravel(), np.asarray(A, dtype=float), np.asarray(b, dtype=float).ravel()
  m, n = A.shape
  if c.size != n or b.size != m:
    raise ValueError("incompatible dimensions of the linear program")
  if not (np.isfinite(tol) and tol > 0 and maxIter >= 0):
    raise ValueError("tol has to be positive and maxIter nonnegative")
  if not (np.all(np.isfinite(A)) and np.all(np.isfinite(b)) and np.all(np.isfinite(c))):
    raise ValueError("the data of a linear program have to be finite")
  finish = lambda status, it, x=None, res=np.inf: LinearProgramResult(x, status, it, res) if diagnostics else x
  # phase one on the artificial variables, every row scaled to unit size and a nonnegative right hand side
  T = np.zeros((m + 1, n + m + 1))
  scale = np.maximum(np.max(np.abs(A), axis=1, initial=0.0), np.abs(b))
  scale = np.where(scale == 0, 1.0, scale) * np.where(b < 0, -1.0, 1.0)
  T[:m, :n] = A / scale[:, None]
  T[:m, -1] = b / scale
  T[np.arange(m), n + np.arange(m)] = 1
  basis = np.arange(n, n + m)
  _cost(T, basis, np.concatenate([np.zeros(n), np.ones(m)]))
  status, it = _solve(T, basis, n + m, tol, maxIter)
  if status != "optimal":
    return finish(status, it)
  if np.sum(T[:-1, -1][basis > n - 1]) > 10 * tol:
    return finish("infeasible", it)
  # the artificial variables still basic at zero level leave the basis, redundant rows go
  i = 0
  while i < len(basis):
    if basis[i] >= n:
      candidates = [k for k in range(n) if k not in basis and abs(T[i, k]) > tol]
      if not candidates:
        T = np.delete(T, i, axis=0)
        basis = np.delete(basis, i)
        continue
      if it >= maxIter:
        return finish("iteration_limit", it)
      _pivot(T, basis, i, candidates[0])
      it += 1
    i += 1
  # phase two on the original columns
  T = T[:, list(range(n)) + [T.shape[1] - 1]]
  cscale = max(float(np.max(np.abs(c), initial=0.0)), np.finfo(float).eps)
  _cost(T, basis, c / cscale)
  status, steps = _solve(T, basis, n, tol, maxIter - it)
  it += steps
  if status != "optimal":
    return finish(status, it)
  x = np.zeros(n)
  x[basis] = np.maximum(T[:-1, -1], 0)
  residual = np.max(np.abs(A @ x - b), initial=0.0) / max(1.0, np.max(np.abs(b), initial=0.0))
  if residual > 100 * tol:
    return finish("numerical_failure", it, None, residual)
  return finish("optimal", it, x, residual)


# ------------------------------------------------------------------ the analytic centre

def linprogCentre(c, A, y, tol=1e-9, maxIter=100):
  """the solutions of many linear programs min c.x under A x = y, x >= 0 at once, each the
  analytic centre of its optimal face, which is symmetric where the program is (a simplex
  gives any vertex), NaN where one does not converge (MATLAB's linprogCentre). c is (n,) or
  (N, n), A (m, n) or (N, m, n), y (m,) or (N, m); a primal dual path following method, its
  Newton steps towards the central path at 0.3 mu"""
  y = np.asarray(y, dtype=float)
  single = y.ndim == 1
  y = y.reshape(-1, y.shape[-1])
  N, m = y.shape
  A = np.broadcast_to(np.asarray(A, dtype=float), (N,) + np.shape(A)[-2:])
  n = A.shape[-1]
  c = np.broadcast_to(np.asarray(c, dtype=float), (N, n))
  At = np.swapaxes(A, 1, 2)
  mul = lambda B, v: np.einsum("kij,kj->ki", B, v)
  x, s, lam = np.ones((N, n)), np.ones((N, n)), np.zeros((N, m))
  ny, nc = np.linalg.norm(y, axis=1), np.linalg.norm(c, axis=1)
  done = np.zeros(N, dtype=bool)
  for _ in range(maxIter):
    rp = y - mul(A, x)
    rd = c - mul(At, lam) - s
    gap = np.sum(x * s, axis=1)
    mu = gap / n
    # the duality gap is the error of the objective
    dualDone = (np.linalg.norm(rd, axis=1) <= tol * (1 + nc)) & (gap <= tol * (1 + np.abs(np.sum(c * x, axis=1))))
    done = dualDone & (np.linalg.norm(rp, axis=1) <= tol * (1 + ny))
    # near the end the normal equations lose the primal residual, which the steps, cut short by
    # the entries close to zero, no longer remove: the least correction on the support restores it
    for k in np.flatnonzero(dualDone & ~done):
      S = x[k] > 1e-6 * np.max(x[k])
      z = x[k].copy()
      z[S] += A[k][:, S].T @ np.linalg.lstsq(A[k][:, S] @ A[k][:, S].T, rp[k], rcond=None)[0]
      if np.min(z[S]) >= 0 and np.linalg.norm(y[k] - A[k] @ z) <= 1e-12 * (1 + ny[k]):
        x[k], done[k] = z, True
    if done.all():
      break
    rc = 0.3 * mu[:, None] - x * s
    M = (A * (x / s)[:, None, :]) @ At
    M += 1e-13 * np.max(np.diagonal(M, axis1=1, axis2=2), axis=1)[:, None, None] * np.eye(m)
    dlam = np.linalg.solve(M, (rp - mul(A, (rc - x * rd) / s))[..., None])[..., 0]
    ds = rd - mul(At, dlam)
    dx = (rc - x * ds) / s
    dx[done], ds[done], dlam[done] = 0, 0, 0
    ap = np.minimum(1, 0.95 / np.maximum(np.max(-dx / x, axis=1), 1e-300))
    ad = np.minimum(1, 0.95 / np.maximum(np.max(-ds / s, axis=1), 1e-300))
    x, s, lam = x + ap[:, None] * dx, s + ad[:, None] * ds, lam + ad[:, None] * dlam
  x[~done] = np.nan
  return x[0] if single else x


# ------------------------------------------------------------------ the model

class TaylorModel:
  """the slip systems prepared for the Taylor solve: the distinct signed systems with their
  critical resolved shear stresses, the strain each shear produces and its spin;
  spinSelection 'centre' takes the analytic centre of the optimal face, as MATLAB does,
  'basic' a vertex of it by the simplex, 'minimumNorm' the shears of least norm on it"""

  def __init__(self, sS, spinSelection="centre"):
    if isinstance(sS, TaylorModel):
      self.__dict__.update(sS.__dict__)
      self.spinSelection = spinSelection
      return
    if not isinstance(sS, slipSystem) or len(sS) == 0:
      raise ValueError("at least one slip system is needed")
    fr = sS.frame
    if not isinstance(fr, crystalFrame):
      raise ValueError("the slip systems of a Taylor model live in a crystal frame")
    if not np.all(np.isfinite(sS.CRSS)) or np.any(sS.CRSS <= 0):
      raise ValueError("every critical resolved shear stress has to be finite and positive")
    if spinSelection not in SELECTIONS:
      raise ValueError("spinSelection is 'centre', 'basic' or 'minimumNorm'")
    full = sS.symmetrise()
    D = full._dyads()
    keys = np.concatenate([np.round(D.reshape(len(full), 9) * 1e10), full.CRSS[:, None]], axis=1)
    _, first = np.unique(keys, axis=0, return_index=True)
    keep = np.sort(first)
    self.frame, self.systems = fr, full[keep]
    D = D[keep]
    self.A = np.stack([D[:, 0, 0], (D[:, 1, 0] + D[:, 0, 1]) / 2, (D[:, 2, 0] + D[:, 0, 2]) / 2, D[:, 1, 1], (D[:, 2, 1] + D[:, 1, 2]) / 2])
    self.crss = full.CRSS[keep].astype(float)
    self.spins = (D - np.swapaxes(D, 1, 2)) / 2
    self.spinSelection = spinSelection

  def __len__(self):
    return len(self.crss)

  def __repr__(self):
    return aligned(f"TaylorModel ({frameLabel(self.frame)})\n  slip systems: {len(self)}\n  spin selection: {self.spinSelection}")


# the shears of least norm on the work-optimal face, from a vertex of it
def _minimumNormShears(A, c, y, x, tol=1e-9):
  B = np.vstack([A, c[None, :] / np.linalg.norm(c)])
  rhs = np.concatenate([y, [c @ x / np.linalg.norm(c)]])
  free = np.ones(len(x), dtype=bool)
  for _ in range(20 * len(x) + 100):
    F = B[:, free]
    lam = np.linalg.pinv(F @ F.T, rcond=1e-12) @ rhs
    z = np.zeros(len(x))
    z[free] = F.T @ lam
    if np.min(z, initial=0.0) < -tol:
      d = z - x
      alpha, j = 1.0, -1
      for k in range(len(x)):
        if free[k] and d[k] < -tol and -x[k] / d[k] < alpha:
          alpha, j = max(0.0, -x[k] / d[k]), k
      if j < 0:
        raise RuntimeError("the minimum norm selection of the Taylor shears failed")
      x = np.maximum(x + alpha * d, 0)
      free[j] = False
    else:
      x = np.maximum(z, 0)
      dual = B.T @ lam
      score = np.where(free, -np.inf, dual)
      j = int(np.argmax(score))
      if free[j] or dual[j] <= tol:
        if np.max(np.abs(B @ x - rhs)) > 1e-7:
          raise RuntimeError("the minimum norm Taylor shears do not reproduce the strain")
        return x
      free[j] = True
  raise RuntimeError("the minimum norm selection of the Taylor shears did not converge")


def _checkStrain(E):
  E = np.asarray(E, dtype=float).reshape(-1, 3, 3)
  if not np.all(np.isfinite(E)):
    raise ValueError("the strain has to be finite")
  scale = np.linalg.norm(E, axis=(1, 2))
  tol = max(1e-9, 16 * np.finfo(float).eps)
  asym = np.linalg.norm(E - np.swapaxes(E, 1, 2), axis=(1, 2))
  if np.any(asym > tol * scale) or np.any(np.abs(np.trace(E, axis1=1, axis2=2)) > tol * scale):
    raise ValueError("the Taylor solve needs a symmetric, traceless strain")


# one crystal: the factor, the shears and the spin of a strain in the crystal frame
def _taylorCrystal(E, model, plasticWork=False, spinSelection=None):
  spinSelection = model.spinSelection if spinSelection is None else spinSelection
  _checkStrain(E)
  scale = float(np.max(np.abs(np.linalg.eigvalsh((E + E.T) / 2))))
  if scale == 0:
    return 0.0, np.zeros(len(model)), np.zeros((3, 3))
  En = E / scale
  y = np.array([En[0, 0], En[1, 0], En[2, 0], En[1, 1], En[2, 1]])
  if spinSelection == "centre":
    x = linprogCentre(model.crss, model.A, y)
    if np.any(np.isnan(x)):
      raise ValueError("the Taylor solve failed: the interior point method did not converge")
    x = np.maximum(x, 0)
  else:
    result = linearProgram(model.crss, model.A, y, diagnostics=True)
    if result.status != "optimal":
      raise ValueError(f"the Taylor solve failed: {result.status}")
    x = result.solution
  if spinSelection == "minimumNorm":
    x = _minimumNormShears(model.A, model.crss, y, x)
  factor = float(model.crss @ x) if plasticWork else float(np.sum(x))
  x = x * scale
  W = np.einsum("k,kij->ij", x, model.spins)
  return factor, x, W


# many strains in the crystal frame: the factors, the shears and the spins, through the
# compiled loop or one simplex after the other
def _taylorMany(E, model, plasticWork, spinSelection):
  E = np.ascontiguousarray(E, dtype=float).reshape(-1, 3, 3)
  _checkStrain(E)
  n, k = len(E), len(model)
  M, b, W = np.zeros(n), np.zeros((n, k)), np.zeros((n, 3, 3))
  loop = taylorKernel() if backend() == "numba" else None
  if loop is None:
    for j in range(n):
      M[j], b[j], W[j] = _taylorCrystal(E[j], model, plasticWork, spinSelection)
    return M, b, W
  status = np.zeros(n, dtype=np.int64)
  loop(E, np.ascontiguousarray(model.A), model.crss, np.ascontiguousarray(model.spins), SELECTIONS.index(spinSelection),
       plasticWork, 1e-9, 10000, M, b, W, status)
  if np.any(status != 0):
    raise ValueError(f"the Taylor solve failed for {np.count_nonzero(status)} of {n} strains")
  return M, b, W


def calcTaylor(eps, sS, *args, plasticWork=False, spinSelection=None, bandwidth=32, smoothing=0.0):
  """the Taylor factor, the shears of the slip systems and the plastic spin of a strain

    M, b, W = calcTaylor(eps, sS)                     eps in the crystal frame, one or a list of strains
    M, b, W = calcTaylor(inv(ori) * eps, sS)          the strain of every orientation, one row of shears each
    MFun, _, WFun = calcTaylor(eps, sS)               eps in the specimen frame: the factor as an SO3Fun and
                                                      the spin as a vector field of the orientation

  plasticWork=True takes the work instead of the sum of the shears. The shears are the
  analytic centre of the optimal face, which makes them and the spin unique and symmetric, as
  MATLAB's; spinSelection 'basic' takes a vertex of the face, 'minimumNorm' the shears of least
  norm on it. The spin is a spinTensor in the crystal frame, so ori.exp(-W) turns the crystal.
  """
  if any(isinstance(a, str) and a.lower() == "plasticwork" for a in args):
    plasticWork = True
  model = sS if isinstance(sS, TaylorModel) else TaylorModel(sS)
  if not isinstance(eps, strainTensor):
    raise TypeError("calcTaylor needs a strainTensor")
  fr = eps.frame
  if isinstance(fr, crystalFrame):
    # the frame of the systems, or the sibling without its group that inv(ori) * eps writes in
    if not (fr is model.frame or fr._basis is model.frame._basis):
      raise ValueError("the strain and the slip systems are of different crystals")
    M, b, W = _taylorMany(eps.M, model, plasticWork, model.spinSelection if spinSelection is None else spinSelection)
    if eps.ndim == 0:
      return TaylorResult(float(M[0]), b[0], spinTensor(W[0], fr))
    return TaylorResult(M.reshape(eps.shape), b.reshape(eps.shape + (len(model),)), spinTensor(W.reshape(eps.shape + (3, 3)), fr))
  return _taylorField(eps, model, plasticWork, model.spinSelection if spinSelection is None else spinSelection, bandwidth, smoothing)


# the factor and the spin over all orientations, as harmonic functions fitted on a
# quadrature grid: the spin in left components, exp(-W) turning the crystals
def _taylorField(eps, model, plasticWork, spinSelection, bandwidth, smoothing):
  from ..functions.harmonics import deg2dim
  from ..functions.kernelfunctions import SO3DeLaValleePoussinKernel
  from ..functions.quadrature import bandwidthFor, folds, quadratureSO3Grid
  from ..functions.so3fun import SO3FunHarmonic
  from ..functions.so3vectorfield import SO3VectorFieldHarmonic, _adjointColumns
  _checkStrain(eps.M)
  cs, ss = model.frame, eps.frame
  ca, cb = cs, ss.stripSym()
  rA, rB = folds(ca, cb)
  L = max(1, bandwidthFor(bandwidth, rA, rB))
  grid = quadratureSO3Grid(L, ca, cb)
  nodes = grid.nodes()
  Q = rotation(nodes.data).matrix().reshape(-1, 3, 3)
  M, _, W = _taylorMany(np.einsum("nji,jk,nkl->nil", Q, eps.M, Q), model, plasticWork, spinSelection)
  values = np.column_stack([M, np.einsum("nij,nj->ni", Q, np.stack([W[:, 2, 1], W[:, 0, 2], W[:, 1, 0]], axis=-1))])
  # the projections onto the invariant and the equivariant functions remove what the
  # folded rule aliases, so they come before the truncation
  coeff = _adjointColumns(grid, values, L, grid.weights)
  factor = SO3FunHarmonic(coeff[:, 0], cs, ss, real=True, _internal=True).truncate(bandwidth)
  spin = SO3VectorFieldHarmonic(coeff[:, 1:4], cs, ss, "left").truncate(bandwidth)
  if smoothing > 0:
    A = SO3DeLaValleePoussinKernel(halfwidth=smoothing).calcFourier(bandwidth)
    for l in range(bandwidth + 1):
      spin._fhat[deg2dim(l):deg2dim(l + 1), :] *= A[l] / (2 * l + 1)
  return TaylorResult(factor, None, spin)


def _install():
  install(strainTensor, calcTaylor)
  # a spin tensor is the rotation it generates: its axial vector as a tangent vector
  install(spinTensor, _spinExp, "exp")


def _spinExp(W, ref=None, side=None):
  """the rotation of a spin tensor, or the orientation a spin at a reference points to; a spin
  in the crystal frame acts on the right of the orientation"""
  from ..geometry.tangent import SO3TangentVector, expVector
  M = W.M
  v = vector3d(np.stack([M[..., 2, 1], M[..., 0, 2], M[..., 1, 0]], axis=-1))
  if ref is None:
    return expVector(v)
  if side is None:
    side = "right" if isinstance(W.frame, crystalFrame) else "left"
  return SO3TangentVector(v, ref, side).exp()


_install()

__all__ = ["calcTaylor", "linearProgram", "linprogCentre", "TaylorModel"]
