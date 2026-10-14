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

from ..geometry.arguments import refuseFlags
from ..geometry.frames import crystalFrame, referenceFrame
from ..geometry.kernels import backend
from ..geometry.orientation import orientation
from ..geometry.quaternion import rotation
from ..geometry.slipsystem import slipSystem
from ..geometry.vector3d import vector3d
from ..tensors.tensor import spinTensor, strainTensor
from .kernels import taylorKernel
from .mechanics import _axial
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
  """the minimiser of a linear program in standard form, min c.x under A x = b, x >= 0

    x = linearProgram(c, A, b)                     # the solution, None when there is none
    res = linearProgram(c, A, b, diagnostics=True)  # with its status, pivot count and residual

  A scaled two phase simplex with Bland's rule, the kind of program the Taylor model solves
  for every strain: phase one finds a feasible vertex on artificial variables, every row scaled to unit
  size, phase two walks to the optimum. Without diagnostics the result is None when the
  program is infeasible, unbounded, out of iterations or numerically failed; with diagnostics a
  LinearProgramResult (solution, status, iterations, residual), the residual the largest
  entry of |A x - b| relative to max(1, |b|).

  Input
    c - the cost vector, n numbers
    A - the m x n matrix of the constraints
    b - the right hand side, m numbers
  Output
    x - the minimiser, n numbers, or None
    res - LinearProgramResult
  Keywords
    tol - the tolerance below which a reduced cost or a pivot counts as zero
    maxIter - the largest number of pivots
    diagnostics - return a LinearProgramResult instead of the bare solution
  See also
    calcTaylor"""
  c, A, b = np.asarray(c, dtype=float).ravel(), np.asarray(A, dtype=float), np.asarray(b, dtype=float).ravel()
  m, n = A.shape
  if c.size != n or b.size != m:
    raise ValueError("incompatible dimensions of the linear program")
  if not (np.isfinite(tol) and tol > 0 and maxIter >= 0):
    raise ValueError("tol has to be positive and maxIter nonnegative")
  if not (np.all(np.isfinite(A)) and np.all(np.isfinite(b)) and np.all(np.isfinite(c))):
    raise ValueError("the data of a linear program have to be finite")
  def finish(status, it, x=None, res=np.inf):
    return LinearProgramResult(x, status, it, res) if diagnostics else x
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
  analytic centre of its optimal face; MATLAB's linprogCentre

    x = linprogCentre(c, A, y)                         # one program
    x = linprogCentre(C, A, Y)                         # one program per row of C and Y
    x = linprogCentre(c, A, y, tol=1e-6, maxIter=20)   # a coarser stop

  Where the optimum is not unique the solution is the centre of the optimal face, which is
  symmetric where the program is, where a simplex gives any vertex of it; this is the choice
  of the shears in the Taylor model. The programs are solved together by a primal dual path
  following method, its Newton steps aimed at the central path at 0.3 mu.

  Input
    c - (n,) or (N, n), the costs
    A - (m, n) or (N, m, n), the constraint matrix
    y - (m,) or (N, m), the right hand sides
  Output
    x - (n,) or (N, n), the solutions; NaN for a program that does not converge
  Keywords
    tol - the relative tolerance of the primal and dual residuals and the duality gap
    maxIter - the largest number of Newton steps
  See also
    linearProgram, calcTaylor, TaylorModel"""
  y = np.asarray(y, dtype=float)
  single = y.ndim == 1
  y = y.reshape(-1, y.shape[-1])
  N, m = y.shape
  A = np.broadcast_to(np.asarray(A, dtype=float), (N,) + np.shape(A)[-2:])
  n = A.shape[-1]
  c = np.broadcast_to(np.asarray(c, dtype=float), (N, n))
  At = np.swapaxes(A, 1, 2)
  def mul(B, v):
    return np.einsum("kij,kj->ki", B, v)
  x, s, lam = np.ones((N, n)), np.ones((N, n)), np.zeros((N, m))
  ny, nc = np.linalg.norm(y, axis=1), np.linalg.norm(c, axis=1)
  done = np.zeros(N, dtype=bool)
  for _ in range(maxIter):
    rp = y - mul(A, x)
    rd = c - mul(At, lam) - s
    gap = np.sum(x * s, axis=1)
    mu = gap / n
    # the duality gap is the error of the objective
    done = (np.linalg.norm(rp, axis=1) <= tol * (1 + ny)) & (np.linalg.norm(rd, axis=1) <= tol * (1 + nc)) & \
           (gap <= tol * (1 + np.abs(np.sum(c * x, axis=1))))
    if done.all():
      break
    rc = 0.3 * mu[:, None] - x * s
    M = (A * (x / s)[:, None, :]) @ At
    # 1e-16 of the diagonal keeps dependent rows solvable; more shifts the last steps off the constraints
    M += 1e-16 * np.max(np.diagonal(M, axis1=1, axis2=2), axis=1)[:, None, None] * np.eye(m)
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
      # the step until the first free shear reaches zero, which leaves the free set
      d = z - x
      falling = free & (d < -tol)
      ratio = np.full(len(x), np.inf)
      ratio[falling] = -x[falling] / d[falling]
      j = int(np.argmin(ratio))
      if not ratio[j] < 1.0:
        raise RuntimeError("the minimum norm selection of the Taylor shears failed")
      alpha = max(0.0, ratio[j])
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
  """the Taylor factor, the shears of the slip systems and the plastic spin of a strain the
  crystal accommodates by slip alone

    M, b, W = calcTaylor(eps, sS)                       # eps in the crystal frame, one or a list of strains
    M, b, W = calcTaylor(inv(ori) * eps, sS)            # the strain of every orientation, one row of shears each
    oriNew = ori.exp(-W)                                # the orientation the spin turns the crystal to
    work, b, W = calcTaylor(inv(ori) * eps, sS, plasticWork=True)  # the plastic work instead of the factor
    MFun, _, WFun = calcTaylor(eps, sS, bandwidth=32)   # eps in the specimen frame: functions of the orientation

  Taylor's model: every grain takes the macroscopic strain, which its slip systems must
  produce. Of all shears that do, the linear program takes those of least total shear, the
  absolute shears weighted by the critical resolved shear stresses. That sum divided by the
  size of the strain is the Taylor factor M.

  The least total shear is often reached by a whole face of shears. The shears returned are
  the analytic centre of that face, which makes them and the spin unique and symmetric, as
  MATLAB's; spinSelection takes another point of the face. The spin W is a spinTensor in the
  crystal frame, so ori.exp(-W) turns the crystal.

  What comes back depends on the frame of the strain:

  - crystal frame, one strain or a list: the factor, the shears and the spin as numbers.
  - specimen frame: the factor as an SO3FunHarmonic and the spin as an SO3VectorFieldHarmonic
  of the orientation, fitted on a quadrature grid of the bandwidth (MATLAB's 'SO3Fun' flag);
  the shears are None.

  Input
    eps - strainTensor, in the crystal frame of sS or in the specimen frame
    sS - slipSystem, the systems in the crystal frame with their critical resolved shear
        stresses, symmetrised; or a TaylorModel prepared from them
  Output
    M - the Taylor factor, a number or an array of the shape of eps; SO3FunHarmonic for a
        strain in the specimen frame
    b - the shear of every slip system, an array (..., number of systems); None for a strain
        in the specimen frame
    W - spinTensor, the plastic spin in the crystal frame; SO3VectorFieldHarmonic for a strain
        in the specimen frame
  Keywords
    plasticWork - take the plastic work, the shears weighted by the critical resolved shear
        stresses, instead of the geometric Taylor factor
    spinSelection - which shears of an optimal face: 'centre' its analytic centre, 'basic' a
        vertex, 'minimumNorm' the shears of least norm; by default the model's, 'centre'
    bandwidth - the bandwidth of the functions for a strain in the specimen frame
    smoothing - the halfwidth of a de la Vallee Poussin kernel the spin field is smoothed
        with; zero for none
  References
    G. I. Taylor, Plastic strain in metals, J. Inst. Metals 62 (1938) 307-324
  See also
    slipSystem, strainTensor, linearProgram, TaylorModel, TaylorHex"""
  refuseFlags(args, "calcTaylor")
  model = sS if isinstance(sS, TaylorModel) else TaylorModel(sS)
  if not isinstance(eps, strainTensor):
    raise TypeError("calcTaylor needs a strainTensor")
  selection = model.spinSelection if spinSelection is None else spinSelection
  fr = eps.frame
  if isinstance(fr, crystalFrame):
    # the frame of the systems, or the sibling without its group that inv(ori) * eps writes in
    if not (fr is model.frame or fr._basis is model.frame._basis):
      raise ValueError("the strain and the slip systems are of different crystals")
    M, b, W = _taylorMany(eps.M, model, plasticWork, selection)
    if eps.ndim == 0:
      return TaylorResult(float(M[0]), b[0], spinTensor(W[0], fr))
    return TaylorResult(M.reshape(eps.shape), b.reshape(eps.shape + (len(model),)), spinTensor(W.reshape(eps.shape + (3, 3)), fr))
  return _taylorField(eps, model, plasticWork, selection, bandwidth, smoothing)


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
  values = np.column_stack([M, np.einsum("nij,nj->ni", Q, _axial(W))])
  # the projections onto the invariant and the equivariant functions remove what the
  # folded rule aliases, so they come before the truncation
  coeff = _adjointColumns(grid, values, L, grid.weights)
  factor = SO3FunHarmonic(coeff[:, 0], cs, ss, real=True, _internal=True).truncate(bandwidth)
  spin = SO3VectorFieldHarmonic(coeff[:, 1:4], cs, ss, "left").truncate(bandwidth)
  if smoothing > 0:
    A = SO3DeLaValleePoussinKernel(halfwidth=smoothing).calcFourier(bandwidth)
    l = np.arange(bandwidth + 1)
    spin._fhat[:deg2dim(bandwidth + 1)] *= np.repeat(A[:bandwidth + 1] / (2 * l + 1), (2 * l + 1) ** 2)[:, None]
  return TaylorResult(factor, None, spin)


def _install():
  install(strainTensor, calcTaylor)
  # a spin tensor is the rotation it generates: its axial vector as a tangent vector
  install(spinTensor, _spinExp, "exp")


def _spinExp(W, ref=None, side=None):
  """the rotation a spin tensor generates, the exponential of its axial vector, or the
  orientation a spin at a reference orientation leads to

    rot = W.exp()                      # the rotation of the spin
    ori = W.exp(ori0)                  # the orientation reached from ori0
    ori = W.exp(ori0, side='right')    # the spin applied on the crystal side

  A spin in a crystal frame acts on the right of the orientation, ori0 * exp(W), one in the
  specimen frame on the left, exp(W) * ori0, unless side says otherwise.

  Input
    W - spinTensor
  Output
    rot - rotation, or orientation with a reference
  Keywords
    ref - orientation, the reference the spin starts at, ori0 above
    side - 'left' or 'right', the side the spin acts on; by default from the frame of W
  See also
    spinTensorOf, SO3TangentVector, doEulerStep"""
  from ..geometry.tangent import SO3TangentVector, expVector
  v = vector3d(_axial(W.M))
  if ref is None:
    return expVector(v)
  if side is None:
    side = "right" if isinstance(W.frame, crystalFrame) else "left"
  return SO3TangentVector(v, ref, side).exp()


_install()

__all__ = ["calcTaylor", "linearProgram", "linprogCentre", "TaylorModel"]
