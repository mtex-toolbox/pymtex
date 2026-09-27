# the compiled kernels of the analysis layer: the Taylor solve of many strains as one
# parallel loop, each strain's linear program through the dense simplex of the maps layer
# and, for a unique spin, the shears of least norm on the optimal face. The NumPy forms in
# taylor.py are the fallback without numba and the oracle of the tests.
from __future__ import annotations

import numpy as np

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
# antisymmetric part of each dyad
def _taylorLoop(E, A, crss, spins, minimumNorm, plasticWork, tol, maxit, M, X, W, status):
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
    s = _linearProgram(A, y, crss, tol, maxit, x)
    if s == 0 and minimumNorm:
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
    globals()["_minimumNorm"] = nb.njit(cache=True)(_minimumNorm)
    _COMPILED["taylor"] = nb.njit(cache=True, parallel=True)(_taylorLoop)
  return _COMPILED["taylor"]


__all__ = []
