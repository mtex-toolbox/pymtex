# the compiled kernels of the tensors layer: the eigen decomposition of many symmetric 3 x 3
# matrices, the Christoffel matrices of the wave velocities, as one parallel loop of cyclic
# Jacobi rotations, and the rotation of flattened tensors on every index, a tensor turned by
# many orientations and the images of the orientation embedding. NumPy's eigh, the Kronecker
# products of tensor.py and the batched products of analysis/embedding.py are the fallbacks
# without numba and the oracles of the tests.
from __future__ import annotations

import numpy as np

from ..geometry.kernels import _jit, backend

prange = range
_COMPILED = {}


# the eigenvalues in ascending order into lam, the eigenvectors as the columns of V, as eigh;
# the rotation of the pair (p, q) zeroes a_pq, the sweeps end when what is left off the
# diagonal is below the rounding of the largest entry
def _eigh3Loop(E, lam, V):
  n = E.shape[0]
  for k in prange(n):
    a = np.empty((3, 3))
    v = np.zeros((3, 3))
    scale = 0.0
    for i in range(3):
      v[i, i] = 1.0
      for j in range(3):
        a[i, j] = 0.5 * (E[k, i, j] + E[k, j, i])
        scale = max(scale, abs(a[i, j]))
    tol = (2.2e-16 * scale) ** 2
    for sweep in range(50):
      if a[0, 1] ** 2 + a[0, 2] ** 2 + a[1, 2] ** 2 <= tol:
        break
      for p in range(2):
        for q in range(p + 1, 3):
          apq = a[p, q]
          if apq == 0.0:
            continue
          theta = (a[q, q] - a[p, p]) / (2.0 * apq)
          t = 1.0 / (abs(theta) + np.sqrt(theta * theta + 1.0))
          if theta < 0.0:
            t = -t
          c = 1.0 / np.sqrt(t * t + 1.0)
          s = t * c
          for r in range(3):
            arp, arq = a[r, p], a[r, q]
            a[r, p] = c * arp - s * arq
            a[r, q] = s * arp + c * arq
          for r in range(3):
            apr, aqr = a[p, r], a[q, r]
            a[p, r] = c * apr - s * aqr
            a[q, r] = s * apr + c * aqr
          for r in range(3):
            vrp, vrq = v[r, p], v[r, q]
            v[r, p] = c * vrp - s * vrq
            v[r, q] = s * vrp + c * vrq
    # sorted ascending by three comparisons
    o0, o1, o2 = 0, 1, 2
    if a[o0, o0] > a[o1, o1]:
      o0, o1 = o1, o0
    if a[o1, o1] > a[o2, o2]:
      o1, o2 = o2, o1
    if a[o0, o0] > a[o1, o1]:
      o0, o1 = o1, o0
    lam[k, 0], lam[k, 1], lam[k, 2] = a[o0, o0], a[o1, o1], a[o2, o2]
    for r in range(3):
      V[k, r, 0], V[k, r, 1], V[k, r, 2] = v[r, o0], v[r, o1], v[r, o2]


def _eigh3Kernel():
  nb = _jit()
  if not nb:
    return None
  if "eigh3" not in _COMPILED:
    globals()["prange"] = nb.prange
    _COMPILED["eigh3"] = nb.njit(cache=True, parallel=True)(_eigh3Loop)
  return _COMPILED["eigh3"]


# the eigenvalues ascending and the eigenvectors as columns of symmetric 3 x 3 matrices over
# any leading dims, as np.linalg.eigh
def eigh3(E):
  E = np.asarray(E, dtype=float)
  loop = _eigh3Kernel() if backend() == "numba" else None
  if loop is None or E.ndim == 2:
    return np.linalg.eigh(E)
  lead = E.shape[:-2]
  X = np.ascontiguousarray(E.reshape(-1, 3, 3))
  lam, V = np.empty((len(X), 3)), np.empty((len(X), 3, 3))
  loop(X, lam, V)
  return lam.reshape(lead + (3,)), V.reshape(lead + (3, 3))


# the rows of U turned by Q on every index of each block of ranks alpha, into out: the index
# of stride s comes in triples (i, i + s, i + 2 s), so a block is turned index by index in
# place of transposing it; U and Q one row, broadcast, or one per row of out
def _turnLoop(U, alpha, offsets, Q, out, chunk):
  n, D = out.shape
  for c in prange((n + chunk - 1) // chunk):
    a = np.empty(D)
    b = np.empty(D)
    for k in range(c * chunk, min(n, (c + 1) * chunk)):
      ku = 0 if U.shape[0] == 1 else k
      kq = 0 if Q.shape[0] == 1 else k
      q00, q01, q02 = Q[kq, 0, 0], Q[kq, 0, 1], Q[kq, 0, 2]
      q10, q11, q12 = Q[kq, 1, 0], Q[kq, 1, 1], Q[kq, 1, 2]
      q20, q21, q22 = Q[kq, 2, 0], Q[kq, 2, 1], Q[kq, 2, 2]
      for t in range(len(alpha)):
        o0 = offsets[t]
        m = offsets[t + 1] - o0
        for i in range(m):
          a[i] = U[ku, o0 + i]
        s = 1
        for _ in range(alpha[t]):
          for base in range(0, m, 3 * s):
            for i in range(base, base + s):
              x, y, z = a[i], a[i + s], a[i + 2 * s]
              b[i] = q00 * x + q01 * y + q02 * z
              b[i + s] = q10 * x + q11 * y + q12 * z
              b[i + 2 * s] = q20 * x + q21 * y + q22 * z
          a, b = b, a
          s *= 3
        for i in range(m):
          out[k, o0 + i] = a[i]


def _turnKernel():
  nb = _jit()
  if not nb:
    return None
  if "turn" not in _COMPILED:
    globals()["prange"] = nb.prange
    _COMPILED["turn"] = nb.njit(cache=True, parallel=True)(_turnLoop)
  return _COMPILED["turn"]


# the rows of U, blocks of flattened tensors of ranks alpha starting at offsets, turned by the
# rotation matrices Q on every index; U one row or n, Q one matrix or n; None without numba
def turnBlocks(U, alpha, offsets, Q, chunk=64):
  loop = _turnKernel() if backend() == "numba" else None
  if loop is None:
    return None
  U = np.ascontiguousarray(np.atleast_2d(U), dtype=np.float64)
  Q = np.ascontiguousarray(np.asarray(Q, dtype=np.float64).reshape(-1, 3, 3))
  out = np.empty((max(len(U), len(Q)), U.shape[1]))
  loop(U, np.asarray(alpha, dtype=np.int64), np.asarray(offsets, dtype=np.int64), Q, out, np.int64(chunk))
  return out


__all__ = []
