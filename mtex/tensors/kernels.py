# the compiled kernels of the tensors layer: the eigen decomposition of many symmetric 3 x 3
# matrices, the Christoffel matrices of the wave velocities, as one parallel loop of cyclic
# Jacobi rotations, and the rotation of flattened tensors on every index, a tensor turned by
# many orientations and the images of the orientation embedding, and the contraction of one
# tensor with many directions on every index, each index with its own, and the sum of one tensor
# turned by many rotations. NumPy's eigh, the Kronecker products of
# tensor.py and the batched products of analysis/embedding.py are the fallbacks without
# numba and the oracles of the tests.
from __future__ import annotations

import numpy as np

from ..geometry.kernels import _jit, _parallelKernel, backend

prange = range
_COMPILED = {}


# the parallel loop compiled once under its name, prange made numba's for the loops of this
# module; None without numba
def _compiled(name, loop):
  nb = _jit() if backend() == "numba" else None
  if not nb:
    return None
  if name not in _COMPILED:
    globals()["prange"] = nb.prange
    _COMPILED[name] = _parallelKernel(loop)
  return _COMPILED[name]


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


def eigh3(E):
  """the eigenvalues ascending and the eigenvectors as columns of symmetric 3 x 3 matrices over
  any leading dims, as np.linalg.eigh"""
  E = np.asarray(E, dtype=float)
  loop = _compiled("eigh3", _eigh3Loop)
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


def turnBlocks(U, alpha, offsets, Q, chunk=64):
  """the rows of U, blocks of flattened tensors of ranks alpha starting at offsets, turned by the
  rotation matrices Q on every index; U one row or n, Q one matrix or n; None without numba"""
  loop = _compiled("turn", _turnLoop)
  if loop is None:
    return None
  U = np.ascontiguousarray(np.atleast_2d(U), dtype=np.float64)
  Q = np.ascontiguousarray(np.asarray(Q, dtype=np.float64).reshape(-1, 3, 3))
  out = np.empty((max(len(U), len(Q)), U.shape[1]))
  loop(U, np.asarray(alpha, dtype=np.int64), np.asarray(offsets, dtype=np.int64), Q, out, np.int64(chunk))
  return out


# the inverse of every rank four tensor through its Voigt matrix, rows of 81 coordinates in and
# out, M[i, j, k, l] at 9 (3 i + j) + 3 k + l: the 36 entries gathered with the factors wIn on
# both sides (flat holds the pair 3 i + j of each Voigt index), Gauss-Jordan with partial
# pivoting, the inverse spread over the pair symmetry (voigt holds the Voigt index of each of
# the nine pairs) with the factors wOut taken out; a tensor whose pivot vanishes is marked singular
def _voigtInverseLoop(D, flat, voigt, wIn, wOut, out, singular):
  n = D.shape[0]
  for k in prange(n):
    a = np.empty((6, 12))
    for p in range(6):
      for q in range(6):
        a[p, q] = D[k, 9 * flat[p] + flat[q]] * wIn[p] * wIn[q]
        a[p, 6 + q] = 1.0 if p == q else 0.0
    for c in range(6):
      piv = c
      for r in range(c + 1, 6):
        if abs(a[r, c]) > abs(a[piv, c]):
          piv = r
      if piv != c:
        for j in range(12):
          a[c, j], a[piv, j] = a[piv, j], a[c, j]
      d = a[c, c]
      if d == 0.0:
        singular[k] = True
        break
      for j in range(12):
        a[c, j] /= d
      for r in range(6):
        if r != c:
          f = a[r, c]
          if f != 0.0:
            for j in range(12):
              a[r, j] -= f * a[c, j]
    if singular[k]:
      continue
    for i in range(9):
      p = voigt[i]
      for j in range(9):
        q = voigt[j]
        out[k, 9 * i + j] = a[p, 6 + q] / (wOut[p] * wOut[q])


def voigtInverse(D, voigtPairs, wIn, wOut):
  """the inverses of rank four tensors, rows of 81 coordinates, through their Voigt matrices of the
  index pairs voigtPairs with the off diagonal factors wIn and wOut; None without numba or where
  one is singular, so that NumPy's inverse raises for it"""
  loop = _compiled("voigtInverse", _voigtInverseLoop)
  if loop is None:
    return None
  flat = np.array([3 * a + b for a, b in voigtPairs], dtype=np.int64)
  voigt = np.empty(9, dtype=np.int64)
  for p, (a, b) in enumerate(voigtPairs):
    voigt[3 * a + b] = voigt[3 * b + a] = p
  D = np.ascontiguousarray(D, dtype=float).reshape(-1, 81)
  out, singular = np.empty_like(D), np.zeros(len(D), dtype=bool)
  loop(D, flat, voigt, np.asarray(wIn, dtype=float), np.asarray(wOut, dtype=float), out, singular)
  return None if singular.any() else out


# a tensor of rank R contracted with u on every index is a homogeneous polynomial of degree R
# in u: the 3^R coordinates fold onto the (R + 1)(R + 2) / 2 monomials x^i y^j z^k, and the
# value at a direction of any length is that of the vector over its norm to the power R, so
# nothing is normalised beforehand
_MONOMIALS = {}


def monomials(R):
  """the monomial each flat coordinate of a rank R tensor belongs to, and the exponents (i, j, k)"""
  if R not in _MONOMIALS:
    digits = np.stack(np.unravel_index(np.arange(3 ** R), (3,) * R), axis=-1)
    expo = np.stack([np.sum(digits == a, axis=-1) for a in range(3)], axis=-1)
    expo, ids = np.unique(expo, axis=0, return_inverse=True)
    _MONOMIALS[R] = ids.ravel(), np.ascontiguousarray(expo, dtype=np.int64)
  return _MONOMIALS[R]


def _polynomialLoop(v, expo, coef, R, out, chunk):
  n = v.shape[0]
  for c in prange((n + chunk - 1) // chunk):
    p = np.ones((3, R + 1))
    for q in range(c * chunk, min(n, (c + 1) * chunk)):
      r2 = v[q, 0] * v[q, 0] + v[q, 1] * v[q, 1] + v[q, 2] * v[q, 2]
      scale = 1 / np.sqrt(r2)
      for a in range(3):
        x = v[q, a] * scale
        for r in range(1, R + 1):
          p[a, r] = p[a, r - 1] * x
      acc = 0.0
      for m in range(coef.shape[0]):
        acc += coef[m] * p[0, expo[m, 0]] * p[1, expo[m, 1]] * p[2, expo[m, 2]]
      out[q] = acc


def directionalPolynomial(M, v, chunk=1024):
  """the real tensor with the 3^R flat coordinates M contracted on every index with the unit
  vectors along the rows of v, which need not be normalised; None without numba"""
  loop = _compiled("polynomial", _polynomialLoop)
  if loop is None:
    return None
  M = np.asarray(M, dtype=np.float64).ravel()
  R = round(np.log(M.size) / np.log(3))
  ids, expo = monomials(R)
  out = np.empty(len(v))
  loop(np.ascontiguousarray(v, dtype=np.float64), expo, np.bincount(ids, M, len(expo)), np.int64(R), out, np.int64(chunk))
  return out


# a tensor of rank R contracted on every index with its own vector at each row: the sum over the
# nonzero flat coordinates of M times the components the digits of the coordinate pick, the last
# digit from the last vector
def _multilinearLoop(M, nz, V, out):
  R, n = V.shape[0], V.shape[1]
  for q in prange(n):
    acc = 0.0
    for t in range(nz.shape[0]):
      m = nz[t]
      p = M[m]
      for k in range(R - 1, -1, -1):
        p *= V[k, q, m % 3]
        m //= 3
      acc += p
    out[q] = acc


def multilinear(M, V):
  """the real tensor with the 3^R flat coordinates M contracted on index k with the rows of V[k],
  V of shape (R, n, 3): M_{i1..iR} V[0]_{i1} .. V[R-1]_{iR} at every row; None without numba"""
  loop = _compiled("multilinear", _multilinearLoop)
  if loop is None:
    return None
  M = np.ascontiguousarray(M, dtype=np.float64).ravel()
  V = np.ascontiguousarray(V, dtype=np.float64)
  out = np.empty(V.shape[1])
  loop(M, np.flatnonzero(M).astype(np.int64), V, out)
  return out


# the weighted sum of one flattened tensor of rank R turned by every quaternion, the matrix formed
# in the loop and the tensor turned index by index as in _turnLoop, each block of quaternions
# adding into its own row of acc: no turned tensor is kept
def _turnSumLoop(U, R, q, w, blocks, acc):
  n, D = q.shape[0], U.shape[0]
  for c in prange(blocks):
    a = np.empty(D)
    b = np.empty(D)
    for k in range(c * n // blocks, (c + 1) * n // blocks):
      qa, qb, qc, qd = q[k, 0], q[k, 1], q[k, 2], q[k, 3]
      q00, q01, q02 = 2 * qb * qb + 2 * qa * qa - 1, 2 * qb * qc - 2 * qa * qd, 2 * qb * qd + 2 * qa * qc
      q10, q11, q12 = 2 * qc * qb + 2 * qa * qd, 2 * qc * qc + 2 * qa * qa - 1, 2 * qc * qd - 2 * qa * qb
      q20, q21, q22 = 2 * qd * qb - 2 * qa * qc, 2 * qd * qc + 2 * qa * qb, 2 * qd * qd + 2 * qa * qa - 1
      for i in range(D):
        a[i] = U[i]
      s = 1
      for _ in range(R):
        for base in range(0, D, 3 * s):
          for i in range(base, base + s):
            x, y, z = a[i], a[i + s], a[i + 2 * s]
            b[i] = q00 * x + q01 * y + q02 * z
            b[i + s] = q10 * x + q11 * y + q12 * z
            b[i + 2 * s] = q20 * x + q21 * y + q22 * z
        a, b = b, a
        s *= 3
      for i in range(D):
        acc[c, i] += w[k] * a[i]


def turnSum(U, R, q, w):
  """sum_k w_k of the flattened real tensor U of rank R turned by the quaternion q_k on every index,
  without forming the turned tensors; None without numba"""
  loop = _compiled("turnSum", _turnSumLoop)
  if loop is None:
    return None
  from ..geometry.kernels import threads
  q = np.ascontiguousarray(q, dtype=np.float64).reshape(-1, 4)
  blocks = max(1, min(4 * threads(), len(q) // 256))
  acc = np.zeros((blocks, 3 ** R))
  loop(np.ascontiguousarray(U, dtype=np.float64).ravel(), np.int64(R), q, np.ascontiguousarray(w, dtype=np.float64),
                       np.int64(blocks), acc)
  return acc.sum(axis=0)


__all__ = []
