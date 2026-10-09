# the compiled kernels of the analysis layer: the Taylor solve of many strains as one
# parallel loop, each strain's linear program at the analytic centre of its optimal face, or at
# a vertex by the dense simplex of the maps layer and, for a unique spin there, the shears of
# least norm on the optimal face, and the two refractive indices and polarisations of light
# along many directions. The NumPy forms in taylor.py and optics.py are the fallback without
# numba and the oracle of the tests.
from __future__ import annotations

import numpy as np

from ..geometry.kernels import _jit, _parallelKernel, backend

prange = range
_COMPILED = {}


# the shears of least norm under B x = rhs, x >= 0, from the feasible x, by the active set
# on the zero shears started at the support of x; BT holds the columns of B as rows;
# 0 with x overwritten, or a code
def _minimumNorm(BT, rhs, x, tol):
  n, m = BT.shape
  free = np.empty(n, dtype=np.bool_)
  for j in range(n):
    free[j] = x[j] > 0.0
  z = np.empty(n)
  for _ in range(20 * n + 100):
    G = np.zeros((m, m))
    for j in range(n):
      if free[j]:
        G += np.outer(BT[j], BT[j])
    lam = np.linalg.pinv(G, 1e-12) @ rhs
    zmin = 0.0
    for j in range(n):
      z[j] = (BT[j] @ lam) if free[j] else 0.0
      zmin = min(zmin, z[j])
    if zmin < -tol:
      # step towards z until the first free shear reaches zero, which leaves the free set
      alpha, k = 1.0, -1
      for j in range(n):
        d = z[j] - x[j]
        if free[j] and d < -tol and -x[j] / d < alpha:
          alpha, k = max(0.0, -x[j] / d), j
      if k < 0:
        return 1
      for j in range(n):
        x[j] = max(x[j] + alpha * (z[j] - x[j]), 0.0)
      free[k] = False
    else:
      for j in range(n):
        x[j] = max(z[j], 0.0)
      # the fixed shear of the largest multiplier rejoins the free set, or the set is optimal
      k, best = -1, -np.inf
      for j in range(n):
        if not free[j]:
          dual = BT[j] @ lam
          if dual > best:
            k, best = j, dual
      if k < 0 or best <= tol:
        r = BT.T @ x - rhs
        res = np.max(np.abs(r))
        return 0 if res <= 1e-7 else 2
      free[k] = True
  # a shear cycling in and out at the tolerance leaves every iterate feasible and the norm
  # optimal to rounding, so the last one stands when it still reproduces the strain
  r = BT.T @ x - rhs
  return 0 if np.max(np.abs(r)) <= 1e-7 else 3


# the factor, the shears and the spin of every strain E[k] in the crystal frame: A holds
# the five independent strain components of each system, crss the cost, spins the
# antisymmetric part of each dyad; selection 0 a vertex of the optimal face, 1 the shears of
# least norm on it, 2 its analytic centre
def _taylorLoop(E, A, crss, spins, selection, plasticWork, tol, maxIter, M, X, W, status):
  n, k = E.shape[0], A.shape[1]
  cnorm = np.sqrt(crss @ crss)
  BT = np.empty((k, 6))
  BT[:, :5] = A.T
  BT[:, 5] = crss / cnorm
  for p in prange(n):
    e = E[p]
    lam = np.linalg.eigvalsh(0.5 * (e + e.T))
    scale = max(abs(lam[0]), abs(lam[2]))
    x = np.zeros(k)
    if scale == 0.0:
      M[p] = 0.0
      X[p] = x
      W[p] = 0.0
      status[p] = 0
      continue
    y = np.array([e[0, 0], e[1, 0], e[2, 0], e[1, 1], e[2, 1]]) / scale
    s = _centreProgram(A, y, crss, tol, maxIter, x) if selection == 2 else _linearProgram(A, y, crss, tol, maxIter, x)
    if s == 0 and selection == 1:
      rhs = np.empty(6)
      rhs[:5] = y
      rhs[5] = (crss @ x) / cnorm
      s = 10 + _minimumNorm(BT, rhs, x, 1e-9)
      if s == 10:
        s = 0
    status[p] = s
    M[p] = (crss @ x) if plasticWork else np.sum(x)
    for j in range(k):
      X[p, j] = x[j] * scale
    w = np.zeros((3, 3))
    for j in range(k):
      w += X[p, j] * spins[j]
    W[p] = w


def taylorKernel():
  """the compiled Taylor loop over the pixels, the linear program and the minimum norm solve per
  orientation, compiled on first call; None where numba is not available"""
  from ..maps import kernels as mk
  nb = mk._preparePixel()
  if nb is None:
    return None
  if "taylor" not in _COMPILED:
    globals()["prange"] = nb.prange
    globals()["_linearProgram"] = mk._COMPILED["_linearProgram"]
    globals()["_centreProgram"] = mk._COMPILED["_centreProgram"]
    globals()["_minimumNorm"] = nb.njit(cache=True)(_minimumNorm)
    _COMPILED["taylor"] = _parallelKernel(_taylorLoop)
  return _COMPILED["taylor"]


# the refractive index tensor restricted to the plane normal to each unit direction of U: the
# axes p1 = (-y, x, 0) of vector3d/orth and p2 = u x p1, the symmetric 2 x 2 matrix on them,
# its eigen decomposition in closed form. M holds one tensor, broadcast, or one per direction
def _birefringenceLoop(U, M, delta, fast, slow, n1, n2):
  many = np.int64(M.shape[0] > 1)
  for q in prange(U.shape[0]):
    k = many * q
    ux, uy, uz = U[q, 0], U[q, 1], U[q, 2]
    y = -1.0 if abs(ux) <= 1e-10 else uy
    r = np.sqrt(y * y + ux * ux)
    p = np.empty((2, 3))
    p[0, 0], p[0, 1], p[0, 2] = -y / r, ux / r, 0.0
    cx, cy, cz = uy * p[0, 2] - uz * p[0, 1], uz * p[0, 0] - ux * p[0, 2], ux * p[0, 1] - uy * p[0, 0]
    r = np.sqrt(cx * cx + cy * cy + cz * cz)
    p[1, 0], p[1, 1], p[1, 2] = cx / r, cy / r, cz / r
    a = b = c = 0.0
    for i in range(3):
      for j in range(3):
        a += p[0, i] * M[k, i, j] * p[0, j]
        c += p[1, i] * M[k, i, j] * p[1, j]
        b += (p[0, i] * M[k, i, j] * p[1, j] + p[1, i] * M[k, i, j] * p[0, j]) / 2
    d = 2 * np.hypot((a - c) / 2, b)
    mid = (a + c) / 2
    theta = np.arctan2(2 * b, a - c) / 2
    co, si = np.cos(theta), np.sin(theta)
    for i in range(3):
      slow[q, i] = co * p[0, i] + si * p[1, i]
      fast[q, i] = -si * p[0, i] + co * p[1, i]
    delta[q], n1[q], n2[q] = d, mid - d / 2, mid + d / 2


def birefringenceKernel(U, M):
  """the index difference, the fast and the slow axis and the two indices of light along the n unit
  directions U through the tensors M, one or n; None without numba"""
  nb = _jit() if backend() == "numba" else None
  if not nb:
    return None
  if "birefringence" not in _COMPILED:
    globals()["prange"] = nb.prange
    _COMPILED["birefringence"] = _parallelKernel(_birefringenceLoop)
  n = len(U)
  delta, n1, n2 = (np.empty(n) for _ in range(3))
  fast, slow = np.empty((n, 3)), np.empty((n, 3))
  _COMPILED["birefringence"](np.ascontiguousarray(U, dtype=np.float64), np.ascontiguousarray(M, dtype=np.float64), delta, fast, slow, n1, n2)
  return delta, fast, slow, n1, n2


__all__ = []
