# the compiled kernels of the harmonic transforms: the NUFFT nodes of rotations and
# directions, and the conjugate symmetry of real valued coefficients, each a loop compiled
# by numba on its first call with a NumPy form beside it that is the fallback and the
# oracle of the tests; the pattern is the geometry's, mtex/geometry/kernels.py
#
#   x, y, z, phase, bad = so3Nodes(q, folds, shift)   the folded Euler nodes of quaternions,
#   x, y, phase, bad = s2Nodes(xyz, shift)             the polar nodes of directions; bad the
#                                                      mask of the nodes not finite, or None
#   realDefectWigner(fhat, tol), realDefectSpherical   is a coefficient vector real valued
#   realizeWigner(fhat), realizeSpherical(fhat)        the real part's coefficients, in place
from __future__ import annotations

import math

import numpy as np

from ..geometry.kernels import _jit, _loop, backend, threads
from ..geometry.representations import quat2euler
from .harmonics import dim2deg

_COMPILED = {}
prange = range


# the small functions the loops call, compiled before the first loop is
_HELPERS = ("_pairDefect", "_ipow", "_store", "_rowAdvance", "_rowState", "_qmul4")


def _compileHelpers():
  if "helpers" not in _COMPILED:
    for name in _HELPERS:
      globals()[name] = _jit().njit(cache=True, inline="always")(globals()[name])
    _COMPILED["helpers"] = True


# a loop compiled over the cores, run on the threads of the preference
def _parallel(name, f):
  nb = _jit()
  _compileHelpers()
  if name not in _COMPILED:
    global prange
    prange = nb.prange
    _COMPILED[name] = nb.njit(cache=True, parallel=True, nogil=True)(f)
  nb.set_num_threads(min(threads(), nb.config.NUMBA_NUM_THREADS))
  return _COMPILED[name]


def _complexType(dtype):
  return np.result_type(np.dtype(dtype), 1j)


# ------------------------------------------------------------------ the nodes of a NUFFT
# a rotation with the Matthies angles (alpha, beta, gamma) is the node
# (rA gamma, rB alpha, beta) mod 2pi of the lattice folded by the z folds rA, rB, and
# e^{-i s x} the phase that moves a half lattice onto the centred index set of finufft.
# A node that is not finite becomes node 0 with phase 1, and is counted


def _so3NodesLoop(q, rA, rB, s, x, y, z, phase):
  twoPi = 2 * math.pi
  bad = 0
  for i in prange(q.shape[0]):
    a, b, c, d = q[i, 0], q[i, 1], q[i, 2], q[i, 3]
    at1, at2 = math.atan2(d, a), math.atan2(b, c)
    beta = 2 * math.atan2(math.sqrt(b * b + c * c), math.sqrt(a * a + d * d))
    u, v = rA * (at1 + at2), rB * (at1 - at2)
    u -= twoPi * math.floor(u / twoPi)
    v -= twoPi * math.floor(v / twoPi)
    if math.isfinite(u) and math.isfinite(v) and math.isfinite(beta):
      x[i], y[i], z[i] = u, v, beta
      phase[i] = complex(math.cos(s * u), -math.sin(s * u))
    else:
      x[i], y[i], z[i] = 0, 0, 0
      phase[i] = 1
      bad += 1
  return bad


def _so3NodesNumpy(q, rA, rB, s, x, y, z, phase):
  alpha, beta, gamma = quat2euler(q, "Matthies")
  good = np.isfinite(alpha) & np.isfinite(beta) & np.isfinite(gamma)
  x[:] = np.where(good, np.mod(rA * gamma, 2 * np.pi), 0)
  y[:] = np.where(good, np.mod(rB * alpha, 2 * np.pi), 0)
  z[:] = np.where(good, beta, 0)
  phase[:] = np.exp(-1j * s * x)
  return int(np.sum(~good))


# the mask of the nodes that are not finite, None when there are none
def _badNodes(p, count):
  return None if count == 0 else ~np.all(np.isfinite(p), axis=1)


def so3Nodes(q, folds=(1, 1), shift=0):
  q = np.ascontiguousarray(q.reshape(-1, 4))
  x, y, z = (np.empty(len(q), dtype=q.dtype) for _ in range(3))
  phase = np.empty(len(q), dtype=_complexType(q.dtype))
  args = (q, int(folds[0]), int(folds[1]), int(shift), x, y, z, phase)
  bad = _parallel("so3Nodes", _so3NodesLoop)(*args) if backend() == "numba" else _so3NodesNumpy(*args)
  return x, y, z, phase, _badNodes(q, bad)


# a direction with the polar angles (theta, rho) is the node (-rho mod 2pi, theta)
def _s2NodesLoop(v, s, x, y, phase):
  twoPi = 2 * math.pi
  bad = 0
  for i in prange(v.shape[0]):
    a, b, c = v[i, 0], v[i, 1], v[i, 2]
    u = -math.atan2(b, a)
    u -= twoPi * math.floor(u / twoPi)
    theta = math.atan2(math.sqrt(a * a + b * b), c)
    if math.isfinite(u) and math.isfinite(theta):
      x[i], y[i] = u, theta
      phase[i] = complex(math.cos(s * u), -math.sin(s * u))
    else:
      x[i], y[i] = 0, 0
      phase[i] = 1
      bad += 1
  return bad


def _s2NodesNumpy(v, s, x, y, phase):
  rho = np.arctan2(v[:, 1], v[:, 0])
  theta = np.arctan2(np.hypot(v[:, 0], v[:, 1]), v[:, 2])
  good = np.isfinite(rho) & np.isfinite(theta)
  x[:] = np.where(good, np.mod(-rho, 2 * np.pi), 0)
  y[:] = np.where(good, theta, 0)
  phase[:] = np.exp(-1j * s * x)
  return int(np.sum(~good))


def s2Nodes(xyz, shift=0):
  v = np.ascontiguousarray(xyz.reshape(-1, 3))
  x, y = (np.empty(len(v), dtype=v.dtype) for _ in range(2))
  phase = np.empty(len(v), dtype=_complexType(v.dtype))
  args = (v, int(shift), x, y, phase)
  bad = _parallel("s2Nodes", _s2NodesLoop)(*args) if backend() == "numba" else _s2NodesNumpy(*args)
  return x, y, phase, _badNodes(v, bad)


# ------------------------------------------------------------------ real valued coefficients
# on SO(3) a function is real when fhat_n^{-k,-l} = (-1)^{k-l} conj(fhat_n^{kl}), on S2 when
# fhat_l^{-m} = (-1)^m conj(fhat_l^m). In the storage of a degree block the mirrored
# coefficient is the block read backwards, and the sign the parity of the storage index
# r = i (2n+1) + j, i + j, on S2 of the order index i, i + l: one pass per block, no gather


# a coefficient and its mirror are one pair, tested once: off by more than tol times the
# larger of 1 and the smaller modulus, in squares
def _pairDefect(a, b, s, tol2):
  d = b - s * a.conjugate()
  e = d.real * d.real + d.imag * d.imag
  return e > tol2 and e > tol2 * min(a.real * a.real + a.imag * a.imag, b.real * b.real + b.imag * b.imag)


def _wignerDefectLoop(F, L, tol):
  tol2 = tol * tol
  for row in range(F.shape[0]):
    o = 0
    for n in range(L + 1):
      m = 2 * n + 1
      last = o + m * m - 1
      for i in range(n + 1):
        s = 1.0 if i % 2 == 0 else -1.0
        for j in range(m if i < n else n + 1):
          r = o + i * m + j
          if _pairDefect(F[row, r], F[row, last - (r - o)], s, tol2):
            return False
          s = -s
      o += m * m
  return True


def _sphericalDefectLoop(F, L, tol):
  tol2 = tol * tol
  for row in range(F.shape[0]):
    for l in range(L + 1):
      o, m = l * l, 2 * l + 1
      s = 1.0 if l % 2 == 0 else -1.0
      for i in range(l + 1):
        if _pairDefect(F[row, o + i], F[row, o + m - 1 - i], s, tol2):
          return False
        s = -s
  return True


def _realizeWignerLoop(F, L):
  for row in range(F.shape[0]):
    o = 0
    for n in range(L + 1):
      m = 2 * n + 1
      for r in range((m * m + 1) // 2):
        a, b = F[row, o + r], F[row, o + m * m - 1 - r]
        s = 1.0 if (r // m + r % m) % 2 == 0 else -1.0
        F[row, o + r] = (a + s * b.conjugate()) / 2
        F[row, o + m * m - 1 - r] = (b + s * a.conjugate()) / 2
      o += m * m


def _realizeSphericalLoop(F, L):
  for row in range(F.shape[0]):
    for l in range(L + 1):
      o, m = l * l, 2 * l + 1
      for i in range((m + 1) // 2):
        a, b = F[row, o + i], F[row, o + m - 1 - i]
        s = 1.0 if (i + l) % 2 == 0 else -1.0
        F[row, o + i] = (a + s * b.conjugate()) / 2
        F[row, o + m - 1 - i] = (b + s * a.conjugate()) / 2


# the NumPy forms, degree by degree
def _wignerDefectNumpy(F, L, tol):
  o = 0
  for n in range(L + 1):
    m = 2 * n + 1
    B = F[:, o:o + m * m]
    r = np.arange(m * m)
    sign = np.where((r // m + r % m) % 2, -1.0, 1.0)
    if not np.all(np.abs(B[:, ::-1] - sign * np.conj(B)) <= tol * np.maximum(1, np.abs(B))):
      return False
    o += m * m
  return True


def _sphericalDefectNumpy(F, L, tol):
  for l in range(L + 1):
    B = F[:, l * l:(l + 1) ** 2]
    sign = np.where((np.arange(2 * l + 1) + l) % 2, -1.0, 1.0)
    if not np.all(np.abs(B[:, ::-1] - sign * np.conj(B)) <= tol * np.maximum(1, np.abs(B))):
      return False
  return True


def _realizeWignerNumpy(F, L):
  o = 0
  for n in range(L + 1):
    m = 2 * n + 1
    B = F[:, o:o + m * m]
    r = np.arange(m * m)
    B[...] = (B + np.where((r // m + r % m) % 2, -1.0, 1.0) * np.conj(B[:, ::-1])) / 2
    o += m * m


def _realizeSphericalNumpy(F, L):
  for l in range(L + 1):
    B = F[:, l * l:(l + 1) ** 2]
    B[...] = (B + np.where((np.arange(2 * l + 1) + l) % 2, -1.0, 1.0) * np.conj(B[:, ::-1])) / 2


def _rows(fhat):
  F = np.asarray(fhat)
  return F.reshape(-1, F.shape[-1]) if F.ndim else F.reshape(1, 1)


def _dispatch(name, loop, numpyForm, F, *args):
  if backend() == "numba" and F.dtype.kind == "c" and F.flags.c_contiguous:
    _compileHelpers()
    return _loop(name, loop)(F, *args)
  return numpyForm(F, *args)


def realDefectWigner(fhat, tol=1e-10):
  F = _rows(fhat)
  if F.dtype.kind != "c":
    F = F.astype(_complexType(F.dtype))
  return bool(_dispatch("wignerDefect", _wignerDefectLoop, _wignerDefectNumpy, F, dim2deg(F.shape[1]), tol))


def realDefectSpherical(fhat, tol=1e-10):
  F = _rows(fhat)
  if F.dtype.kind != "c":
    F = F.astype(_complexType(F.dtype))
  return bool(_dispatch("sphericalDefect", _sphericalDefectLoop, _sphericalDefectNumpy, F, math.isqrt(F.shape[1]) - 1, tol))


# in place on a complex array, rows of coefficients on the last axis
def realizeWigner(fhat):
  F = fhat.reshape(-1, fhat.shape[-1]) if fhat.flags.c_contiguous else np.ascontiguousarray(fhat).reshape(-1, fhat.shape[-1])
  _dispatch("realizeWigner", _realizeWignerLoop, _realizeWignerNumpy, F, dim2deg(F.shape[1]))
  if not fhat.flags.c_contiguous:
    fhat[...] = F.reshape(fhat.shape)
  return fhat


def realizeSpherical(fhat):
  F = fhat.reshape(-1, fhat.shape[-1]) if fhat.flags.c_contiguous else np.ascontiguousarray(fhat).reshape(-1, fhat.shape[-1])
  _dispatch("realizeSpherical", _realizeSphericalLoop, _realizeSphericalNumpy, F, math.isqrt(F.shape[1]) - 1)
  if not fhat.flags.c_contiguous:
    fhat[...] = F.reshape(fhat.shape)
  return fhat


# ------------------------------------------------------------------ Wigner-d at pi/2, the quadrant
# the transforms need d_n^{jk}(pi/2) for j, k >= 0 only: d^{j,-k} = (-1)^{n+j} d^{jk} and
# d^{-j,k} = (-1)^{n+k} d^{jk}. The quadrants of the degrees 0..L are one flat table, degree
# n at quadrantOffset(n) as rows j and columns k of length n + 1; the interior of a degree by
# the three term recursion in the degree at cos(pi/2) = 0, the row and column n from the
# closed form sqrt(binom(2n, n - k)) 2^{-n}, with the sign (-1)^{n-k} on the row


def quadrantOffset(n):
  return n * (n + 1) * (2 * n + 1) // 6


# the degrees 0 .. L into T. The factors of the recursion split into rows and columns, sqrt((n^2 - j^2)(n^2 - k^2)) = a_j a_k, so that
# an entry costs a few products:
#   d_n = -(2n-1)/(n-1) (j/a_j)(k/a_k) d_{n-1} - n/(n-1) (b_j/a_j)(b_k/a_k) d_{n-2}
# with a_j = sqrt(n^2 - j^2), b_j = sqrt((n-1)^2 - j^2)
def _halfPiLoop(T, L):
  for n in range(L + 1):
    o = n * (n + 1) * (2 * n + 1) // 6
    o1 = (n - 1) * n * (2 * n - 1) // 6
    o2 = (n - 2) * (n - 1) * (2 * n - 3) // 6
    w = n + 1
    # the frame: F[a] = sqrt(binom(2n, a)) 2^-n, row n with k = n - a, column n with j = n - a
    f = 2.0 ** -n
    for a in range(n + 1):
      if a > 0:
        f *= math.sqrt((2 * n - a + 1) / a)
      T[o + n * w + n - a] = f if a % 2 == 0 else -f
      T[o + (n - a) * w + n] = f
    if n < 2:
      if n == 1:
        T[o] = 0.0
      continue
    x, y = np.empty(n), np.zeros(n)
    for k in range(n):
      ia = 1.0 / math.sqrt(n * n - k * k)
      x[k] = k * ia
      if k <= n - 2:
        y[k] = math.sqrt((n - 1) ** 2 - k * k) * ia
    c1, c2 = -(2 * n - 1) / (n - 1), n / (n - 1)
    for j in prange(n):
      s1, s2 = c1 * x[j], c2 * y[j]
      r, r1, r2 = o + j * w, o1 + j * n, o2 + j * (n - 1)
      if j <= n - 2:
        for k in range(n - 1):
          T[r + k] = s1 * x[k] * T[r1 + k] - s2 * y[k] * T[r2 + k]
        T[r + n - 1] = s1 * x[n - 1] * T[r1 + n - 1]
      else:
        for k in range(n):
          T[r + k] = s1 * x[k] * T[r1 + k]


def _halfPiNumpy(T, L):
  from .harmonics import wignerdDegrees
  for n, d in wignerdDegrees(T.dtype.type(np.pi / 2), L):
    T[quadrantOffset(n):quadrantOffset(n + 1)] = d[n:, n:].ravel()


_HALF_PI_QUADRANTS = {}


# the table held per precision, grown to the largest bandwidth asked
def halfPiQuadrants(L, dtype=np.float64):
  dtype = np.dtype(dtype)
  T = _HALF_PI_QUADRANTS.get(dtype.name)
  if T is None or len(T) < quadrantOffset(L + 1):
    L0 = max(L, 64)
    T = np.empty(quadrantOffset(L0 + 1), dtype=dtype)
    if backend() == "numba":
      _parallel("halfPi", _halfPiLoop)(T, L0)
    else:
      _halfPiNumpy(T, L0)
    _HALF_PI_QUADRANTS[dtype.name] = T
  return T


# ------------------------------------------------------------------ the Wigner transforms
# G[l, k, j] = i^{k-l} sum_n sqrt(2n+1) fhat_n^{kl} d_n^{jk} d_n^{jl}, the axes of l and k
# indexed by l/rA and k/rB, j at L + j; a real function keeps l >= 0 with the plane l = 0
# halved. Each thread owns planes l: it reads the column l of every degree block, which is
# contiguous in k, sums over the degrees into a buffer of rows j and columns k, and writes the
# plane once with the phase and the mirror G[l, k, -j] = (-1)^{k+l} G[l, k, j]


# the planes in an order that spreads the long ones (small |l| run over most degrees) over
# the contiguous chunks the parallel loop hands its threads
def planeOrder(real, La):
  lp = np.arange(0, La + 1) if real else np.concatenate([[0], np.stack([np.arange(1, La + 1), -np.arange(1, La + 1)], 1).ravel()])
  order = np.empty_like(lp)
  order[0::2], order[1::2] = lp[:(len(lp) + 1) // 2], lp[(len(lp) + 1) // 2:][::-1]
  return order


# v i^e
def _ipow(v, e):
  e %= 4
  if e == 0:
    return v
  if e == 1:
    return complex(-v.imag, v.real)
  if e == 2:
    return -v
  return complex(v.imag, -v.real)


def _wignerForwardLoop(G, fhat, T, L, real, rA, rB, order):
  La, Kb = L // rA, L // rB
  nk = 2 * Kb + 1
  for t in prange(order.shape[0]):
    lp = order[t]
    l = lp * rA
    il = lp if real else lp + La
    al = abs(l)
    B = np.zeros((L + 1, nk), dtype=G.dtype)
    for n in range(al, L + 1):
      m, w, o = 2 * n + 1, n + 1, n * (n + 1) * (2 * n + 1) // 6
      col = n * (2 * n - 1) * (2 * n + 1) // 3 + (l + n) * m + n
      s = math.sqrt(m)
      kb = n // rB
      for j in range(n + 1):
        row = o + j * w
        sg = 1.0 if (n + j) % 2 == 0 else -1.0
        dl = s * T[row + al] * (sg if l < 0 else 1.0)
        if dl == 0:
          continue
        b = B[j]
        for kp in range(0, kb + 1):
          b[kp + Kb] += fhat[col + kp * rB] * (dl * T[row + kp * rB])
        dl *= sg
        for kp in range(1, kb + 1):
          b[Kb - kp] += fhat[col - kp * rB] * (dl * T[row + kp * rB])
    half = 0.5 if real and l == 0 else 1.0
    for kp in range(-Kb, Kb + 1):
      k = kp * rB
      mirror = 1.0 if (k + l) % 2 == 0 else -1.0
      for j in range(L + 1):
        v = _ipow(B[j, kp + Kb], k - l) * half
        G[il, kp + Kb, L + j] = v
        if j > 0:
          G[il, kp + Kb, L - j] = mirror * v


# fhat_n^{kl} = sqrt(2n+1) i^{k-l} sum_j d_n^{jk} d_n^{jl} h[l, k, -j], the pairs j, -j folded
# first into U[j] = h[-j] + (-1)^{k+l} h[j]; a real function is given for l >= 0 and the
# columns -l follow from fhat_n^{-k,-l} = (-1)^{k-l} conj(fhat_n^{kl})
def _wignerAdjointLoop(fhat, H, T, L, real, rA, rB, order):
  La, Kb = L // rA, L // rB
  nk = 2 * Kb + 1
  for t in prange(order.shape[0]):
    lp = order[t]
    l = lp * rA
    il = lp if real else lp + La
    al = abs(l)
    U = np.empty((L + 1, nk), dtype=H.dtype)
    for kp in range(-Kb, Kb + 1):
      mirror = 1.0 if (kp * rB + l) % 2 == 0 else -1.0
      U[0, kp + Kb] = H[il, kp + Kb, L]
      for j in range(1, L + 1):
        U[j, kp + Kb] = H[il, kp + Kb, L - j] + mirror * H[il, kp + Kb, L + j]
    acc = np.empty(nk, dtype=H.dtype)
    for n in range(al, L + 1):
      m, w, o = 2 * n + 1, n + 1, n * (n + 1) * (2 * n + 1) // 6
      base = n * (2 * n - 1) * (2 * n + 1) // 3
      kb = n // rB
      acc[:] = 0
      for j in range(n + 1):
        row = o + j * w
        sg = 1.0 if (n + j) % 2 == 0 else -1.0
        dl = T[row + al] * (sg if l < 0 else 1.0)
        if dl == 0:
          continue
        u = U[j]
        for kp in range(0, kb + 1):
          acc[kp + Kb] += (dl * T[row + kp * rB]) * u[kp + Kb]
        dl *= sg
        for kp in range(1, kb + 1):
          acc[Kb - kp] += (dl * T[row + kp * rB]) * u[Kb - kp]
      s = math.sqrt(m)
      for kp in range(-kb, kb + 1):
        k = kp * rB
        v = _ipow(s * acc[kp + Kb], k - l)
        fhat[base + (l + n) * m + k + n] = v
        if real and l > 0:
          fhat[base + (n - l) * m + n - k] = v.conjugate() if (k - l) % 2 == 0 else -v.conjugate()


def wignerForward(G, fhat, T, L, real, folds):
  rA, rB = folds
  _parallel("wignerForward", _wignerForwardLoop)(G, fhat, T, L, bool(real), int(rA), int(rB), planeOrder(real, L // rA))


def wignerAdjoint(fhat, H, T, L, real, folds):
  rA, rB = folds
  _parallel("wignerAdjoint", _wignerAdjointLoop)(fhat, H, T, L, bool(real), int(rA), int(rB), planeOrder(real, L // rA))


# ------------------------------------------------------------------ the spherical transforms
# G[m, j] = i^m sum_l sqrt((2l+1)/4pi) fhat_l^m d_l^{jm} d_l^{j0}, m at m + L (a real function
# m >= 0 at m, the column m = 0 halved), j at L + j, G[m, -j] = (-1)^m G[m, j], and
# d_l^{j0} vanishes for odd l + j. The column m of d_l(pi/2) is its row m up to the sign
# (-1)^{j-m}, and the rows come degree by degree from the recursion at a fixed row: a thread
# owns an order m, carries the rows m of three degrees, of length L + 1, and needs nothing
# else but the row 0 of every degree and the factors of the recursion, held per bandwidth.
# No table of the quadrants, which is 2.9 GB at L = 1024


# the factors x_k = k/a_k, y_k = b_k/a_k of the recursion of degree n at n(n-1)/2 + k, and
# the row 0 of degree n at n(n+1)/2 + j
def _rowTablesLoop(X, Y, R0, L):
  for n in range(1, L + 1):
    o = n * (n - 1) // 2
    for k in range(n):
      ia = 1.0 / math.sqrt(n * n - k * k)
      X[o + k] = k * ia
      Y[o + k] = math.sqrt((n - 1) ** 2 - k * k) * ia if k <= n - 2 else 0.0
  d2, d1, d, E, band = _rowState(L, X.dtype)
  f, ef = 1.0, 0
  for n in range(L + 1):
    f, ef = _rowAdvance(d, d1, d2, E, band, n, 0, X, Y, f, ef)
    R0[n * (n + 1) // 2:(n + 1) * (n + 2) // 2] = d[:n + 1]
    d2, d1, d = d1, d, d2


# the row m of degree n into d from the rows of the two degrees before; (f, ef) is the frame
# value F_{n-1}[n-1-m] = sqrt(binom(2n-2, n-1-m)) 2^{1-n} as mantissa and binary exponent,
# returned as F_n[n-m]. At n = m the row is the frame (-1)^{m-k} F_m[m-k].
# From m of about 1000 on, the start of the row and the frame near k = m fall below the
# smallest double, while the recursion grows them to O(1/sqrt(n)) degrees later. An entry
# born below 2^SCALED_BELOW is held as d[k] 2^-E[k] instead, in the band [band[0], band[1])
# around k = m, the recursion being linear in the entry at fixed k; it returns to plain
# numbers when it reaches 2^-60, is rescaled by 2^-500 on the way up, and counts as zero
# until then. Below the underflow nothing changes, the rescalings being exact
SCALED_BELOW = -1000


def _store(d, E, band, k, g, eg):
  if eg >= SCALED_BELOW:
    d[k], E[k] = math.ldexp(g, eg), 0
  else:
    d[k], E[k] = g, -eg
    band[0], band[1] = min(band[0], k), max(band[1], k + 1)


def _rowAdvance(d, d1, d2, E, band, n, m, X, Y, f, ef):
  if n == m:
    g, eg = 1.0, -m
    for a in range(m + 1):
      if a > 0:
        g, x = math.frexp(g * math.sqrt((2 * m - a + 1) / a))
        eg += x
      _store(d, E, band, m - a, g if a % 2 == 0 else -g, eg)
    return 1.0, -m
  f, x = math.frexp(f * math.sqrt(2 * n * (2 * n - 1) / ((n - m) * (n + m))) / 2)
  ef += x
  if n == 1:
    d[0] = 0.0
  else:
    o = n * (n - 1) // 2
    s1 = -(2 * n - 1) / (n - 1) * X[o + m]
    s2 = n / (n - 1) * Y[o + m]
    for k in range(n):
      d[k] = s1 * X[o + k] * d1[k] - s2 * Y[o + k] * d2[k]
  _store(d, E, band, n, f, ef)
  for k in range(band[0], band[1]):
    if E[k] > 0:
      a = abs(d[k])
      if a >= math.ldexp(1.0, E[k] - 60):
        d[k], d1[k], E[k] = math.ldexp(d[k], -E[k]), math.ldexp(d1[k], -E[k]), 0
      elif a > 2.0 ** 500:
        d[k], d1[k], E[k] = math.ldexp(d[k], -500), math.ldexp(d1[k], -500), E[k] - 500
  while band[0] < band[1] and E[band[0]] == 0:
    band[0] += 1
  while band[1] > band[0] and E[band[1] - 1] == 0:
    band[1] -= 1
  return f, ef


# the state of one row: three degrees, the exponents and the band of the scaled entries
def _rowState(L, dtype):
  return (np.zeros(L + 1, dtype), np.zeros(L + 1, dtype), np.zeros(L + 1, dtype),
          np.zeros(L + 1, np.int64), np.array([L + 1, 0], np.int64))


_ROW_TABLES = {}


def rowTables(L, dtype=np.float64):
  dtype = np.dtype(dtype)
  held = _ROW_TABLES.get(dtype.name)
  if held is None or held[0] < L:
    L0 = max(L, 64)
    X, Y = np.zeros(L0 * (L0 + 1) // 2 + 1, dtype), np.zeros(L0 * (L0 + 1) // 2 + 1, dtype)
    R0 = np.empty((L0 + 1) * (L0 + 2) // 2, dtype)
    _compileHelpers()
    _loop("rowTables", _rowTablesLoop)(X, Y, R0, L0)
    held = _ROW_TABLES[dtype.name] = (L0, X, Y, R0)
  return held[1:]


# the orders in an order that spreads the long ones (small m run over most degrees) over the
# contiguous chunks of the parallel loop
def orderOrder(L):
  m = np.arange(L + 1)
  out = np.empty_like(m)
  out[0::2], out[1::2] = m[:(L + 2) // 2], m[(L + 2) // 2:][::-1]
  return out


def _sphericalForwardLoop(G, fhat, X, Y, R0, L, real, order):
  M0 = 0 if real else L
  for t in prange(order.shape[0]):
    m = order[t]
    d2, d1, d, E, band = _rowState(L, X.dtype)
    acc, accn = np.zeros(L + 1, G.dtype), np.zeros(L + 1, G.dtype)
    f, ef = 1.0, 0
    for n in range(m, L + 1):
      f, ef = _rowAdvance(d, d1, d2, E, band, n, m, X, Y, f, ef)
      s = math.sqrt((2 * n + 1) / (4 * math.pi))
      cp = s * fhat[n * n + n + m]
      cn = s * fhat[n * n + n - m]
      r0 = n * (n + 1) // 2
      b0, b1 = band[0], band[1]
      # d_n^{jm} d_n^{j0} = (-1)^m d_n^{mj} d_n^{0j}, and d_n^{j,-m} = d_n^{jm} for even n + j
      for j in range(n % 2, n + 1, 2):
        if b0 <= j < b1 and E[j] > 0:
          continue
        w = d[j] * R0[r0 + j]
        acc[j] += cp * w
        if not real:
          accn[j] += cn * w
      d2, d1, d = d1, d, d2
    sign = 1.0 if m % 2 == 0 else -1.0
    half = 0.5 if real and m == 0 else 1.0
    for j in range(L + 1):
      v = _ipow(acc[j], m) * (sign * half)
      G[M0 + m, L + j] = v
      if j > 0:
        G[M0 + m, L - j] = sign * v
    if not real and m > 0:
      for j in range(L + 1):
        v = _ipow(accn[j], -m)
        G[M0 - m, L + j] = sign * v
        if j > 0:
          G[M0 - m, L - j] = v


# fhat_l^m = sqrt((2l+1)/4pi) i^m sum_j d_l^{jm} d_l^{j0} h[m, -j], the pairs j, -j folded into
# h[m, -j] + (-1)^m h[m, j]; a real function has h for m >= 0 and fhat_l^{-m} = (-1)^m conj(fhat_l^m)
def _sphericalAdjointLoop(fhat, H, X, Y, R0, L, real, order):
  M0 = 0 if real else L
  for t in prange(order.shape[0]):
    m = order[t]
    sign = 1.0 if m % 2 == 0 else -1.0
    d2, d1, d, E, band = _rowState(L, X.dtype)
    u, un = np.empty(L + 1, H.dtype), np.empty(L + 1, H.dtype)
    u[0] = H[M0 + m, L]
    for j in range(1, L + 1):
      u[j] = H[M0 + m, L - j] + sign * H[M0 + m, L + j]
    if not real:
      un[0] = H[M0 - m, L]
      for j in range(1, L + 1):
        un[j] = H[M0 - m, L - j] + sign * H[M0 - m, L + j]
    f, ef = 1.0, 0
    for n in range(m, L + 1):
      f, ef = _rowAdvance(d, d1, d2, E, band, n, m, X, Y, f, ef)
      r0 = n * (n + 1) // 2
      b0, b1 = band[0], band[1]
      ap, an = 0j, 0j
      for j in range(n % 2, n + 1, 2):
        if b0 <= j < b1 and E[j] > 0:
          continue
        w = d[j] * R0[r0 + j]
        ap += w * u[j]
        if not real:
          an += w * un[j]
      s = math.sqrt((2 * n + 1) / (4 * math.pi)) * sign
      v = _ipow(s * ap, m)
      fhat[n * n + n + m] = v
      if m > 0:
        if real:
          fhat[n * n + n - m] = sign * v.conjugate()
        else:
          fhat[n * n + n - m] = _ipow(s * an, -m)
      d2, d1, d = d1, d, d2


def sphericalForward(fhat, L, real, cdtype):
  X, Y, R0 = rowTables(L, np.dtype(cdtype).type(0).real.dtype)
  G = np.empty((L + 1 if real else 2 * L + 1, 2 * L + 1), dtype=cdtype)
  _parallel("sphericalForward", _sphericalForwardLoop)(G, fhat, X, Y, R0, L, bool(real), orderOrder(L))
  return G


def sphericalAdjoint(H, L, real):
  X, Y, R0 = rowTables(L, H.real.dtype)
  fhat = np.empty((L + 1) ** 2, dtype=H.dtype)
  _parallel("sphericalAdjoint", _sphericalAdjointLoop)(fhat, H, X, Y, R0, L, bool(real), orderOrder(L))
  return fhat


# ------------------------------------------------------------------ the symmetry projection
# fhat_n <- P_n fhat_n Q_n on every degree, visiting the nonzeros of the blocks only (the
# dihedral groups keep a few percent of them). A block is stored column by column, B[k, l]
# at l (2n+1) + k, so the left factor runs down the columns and the right one combines them


def _projectLoop(F, L, order, Pptr, Pr, Pc, Pv, rP, Qptr, Qr, Qc, Qv, rQ, left, right):
  for t in prange(order.shape[0]):
    n = order[t]
    m = 2 * n + 1
    o = n * (2 * n - 1) * (2 * n + 1) // 3
    # the rows k and columns l the folds keep, block indices n mod r + r i
    k0, l0 = n % rP, n % rQ
    T = np.empty(m * m, dtype=F.dtype)
    for row in range(F.shape[0]):
      B = F[row, o:o + m * m]
      if left:
        for l in range(l0, m, rQ):
          for k in range(k0, m, rP):
            T[l * m + k] = 0
          for e in range(Pptr[n], Pptr[n + 1]):
            T[l * m + Pr[e]] += Pv[e] * B[l * m + Pc[e]]
      else:
        for l in range(l0, m, rQ):
          for k in range(m):
            T[l * m + k] = B[l * m + k]
      B[:] = 0
      if right:
        for e in range(Qptr[n], Qptr[n + 1]):
          r, b, v = Qr[e], Qc[e], Qv[e]
          for k in range(k0, m, rP):
            B[b * m + k] += v * T[r * m + k]
      else:
        for l in range(m):
          for k in range(k0, m, rP):
            B[l * m + k] = T[l * m + k]


_EMPTY = (np.zeros(1, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, complex))


# True when done: numba there and a complex array of coefficients rows
def projectBlocks(fhat, P, Q):
  left = P is not None and not P.isIdentity
  right = Q is not None and not Q.isIdentity
  if not (left or right):
    return True
  if backend() != "numba" or fhat.dtype != np.complex128:
    return False
  # a contiguous array reshapes to a view, which the loop writes through
  if not fhat.flags.c_contiguous:
    return False
  F = fhat.reshape(-1, fhat.shape[-1])
  L = dim2deg(F.shape[1])
  if (left and len(P.blocks) < L + 1) or (right and len(Q.blocks) < L + 1):
    return False
  Pn = P.nonzeros() if left else _EMPTY
  Qn = Q.nonzeros() if right else _EMPTY
  # dense blocks, the cubic groups', are faster as the matrix products of BLAS
  if (left and _density(P, Pn, L) > PROJECT_SPARSE) or (right and _density(Q, Qn, L) > PROJECT_SPARSE):
    return False
  _parallel("project", _projectLoop)(F, L, np.arange(L + 1)[::-1].copy(), *Pn, P.fold if left else 1,
                                     *Qn, Q.fold if right else 1, left, right)
  return True


# the share of the entries of the blocks up to degree L that are nonzero
PROJECT_SPARSE = 0.25


def _density(P, nz, L):
  return nz[0][L + 1] / sum(len(P.support(n)) ** 2 for n in range(L + 1))


# ------------------------------------------------------------------ Wigner-d at any angle
# d_n(beta) for n = 0..L as full matrices, degree n at deg2dim(n) in rows j and columns k:
# the recursion of wignerdDegrees in harmonics.py compiled, the frame from the closed form
# sqrt(binom(2n, a)) sin(beta/2)^a cos(beta/2)^(2n-a), the interior from the two degrees
# before, parallel over the rows


def _wignerdLoop(T, L, x, s, c):
  T[0] = 1.0
  for n in range(1, L + 1):
    m = 2 * n + 1
    o = n * (2 * n - 1) * (2 * n + 1) // 3
    o1 = (n - 1) * (2 * n - 3) * (2 * n - 1) // 3
    o2 = (n - 2) * (2 * n - 5) * (2 * n - 3) // 3 if n >= 2 else 0
    F = np.empty(m)
    g = 1.0
    for a in range(m):
      if a > 0:
        g *= math.sqrt((2 * n - a + 1) / a) * s
      F[a] = g * c ** (2 * n - a)
    # row -n is F, row n is (-1)^b F[b] backwards, column n is F backwards, column -n (-1)^a F[a]
    for a in range(m):
      b = m - 1 - a
      T[o + a] = F[a]
      T[o + (m - 1) * m + a] = F[b] if b % 2 == 0 else -F[b]
      T[o + a * m + m - 1] = F[b]
      T[o + a * m] = F[a] if a % 2 == 0 else -F[a]
    if n == 1:
      T[o + 4] = x
      continue
    mm = n - 1
    for i in prange(1, m - 1):
      j = i - n
      wj = (mm + 1) ** 2 - j * j
      for jj in range(1, m - 1):
        k = jj - n
        w = math.sqrt(wj * ((mm + 1) ** 2 - k * k))
        v = (mm + 1) * (2 * mm + 1) / w * (x - j * k / (mm * (mm + 1))) * T[o1 + (i - 1) * (m - 2) + jj - 1]
        if abs(j) <= mm - 1 and abs(k) <= mm - 1:
          v -= (mm + 1) * math.sqrt((mm * mm - j * j) * (mm * mm - k * k)) / (mm * w) * T[o2 + (i - 2) * (m - 4) + jj - 2]
        T[o + i * m + jj] = v


def wignerdTable(beta, L):
  T = np.empty(max(1, (L + 1) * (2 * L + 1) * (2 * L + 3) // 3))
  x, s, c = math.cos(beta), math.sin(beta / 2), math.cos(beta / 2)
  _parallel("wignerd", _wignerdLoop)(T, L, x, s, c)
  return T


# ------------------------------------------------------------------ sampling a radial basis function
# one orientation per row of uniforms U: a centre by its weight or the uniform part (the last
# entry of the cumulative weights), the angle by inverse transform of the kernel's density on
# the table omega, a uniform axis, the inverse with probability one half for an antipodal
# function, and a random element of each group on its side:
#   q = l * c * (cos w/2, sin w/2 axis) * r


def _qmul4(a0, a1, a2, a3, b0, b1, b2, b3):
  return (a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3, a1 * b0 + a0 * b1 - a3 * b2 + a2 * b3,
          a2 * b0 + a3 * b1 + a0 * b2 - a1 * b3, a3 * b0 - a2 * b1 + a1 * b2 + a0 * b3)


def _rbfSampleLoop(out, U, cs, cumP, omega, cdf, L, R, antipodal):
  nc, nw = cs.shape[0], omega.shape[0]
  for i in prange(out.shape[0]):
    u = U[i]
    k = np.searchsorted(cumP, u[0], side="right")
    if k < nc:
      w = omega[min(np.searchsorted(cdf, u[1]), nw - 1)]
      z, phi = 2 * u[2] - 1, 2 * math.pi * u[3]
      s, r = math.sin(w / 2), math.sqrt(max(0.0, 1 - z * z))
      q = _qmul4(cs[k, 0], cs[k, 1], cs[k, 2], cs[k, 3], math.cos(w / 2), s * r * math.cos(phi), s * r * math.sin(phi), s * z)
    else:
      a, b = math.sqrt(1 - u[1]), math.sqrt(u[1])
      q = (a * math.sin(2 * math.pi * u[2]), a * math.cos(2 * math.pi * u[2]), b * math.sin(2 * math.pi * u[3]), b * math.cos(2 * math.pi * u[3]))
    if antipodal and u[4] < 0.5:
      q = (q[0], -q[1], -q[2], -q[3])
    l = L[min(int(u[5] * L.shape[0]), L.shape[0] - 1)]
    g = R[min(int(u[6] * R.shape[0]), R.shape[0] - 1)]
    q = _qmul4(l[0], l[1], l[2], l[3], q[0], q[1], q[2], q[3])
    q = _qmul4(q[0], q[1], q[2], q[3], g[0], g[1], g[2], g[3])
    out[i, 0], out[i, 1], out[i, 2], out[i, 3] = q


# the uniforms come from the caller's generator in chunks, so a seeded generator gives the
# same sample and the uniforms of ten million draws are never held at once
SAMPLE_CHUNK = 1 << 20


def rbfSample(n, rng, cs, probs, omega, cdf, L, R, antipodal):
  out = np.empty((n, 4))
  cumP = np.cumsum(probs / probs.sum())
  loop = _parallel("rbfSample", _rbfSampleLoop)
  args = (np.ascontiguousarray(cs, dtype=float), cumP, omega, cdf, np.ascontiguousarray(L, dtype=float),
          np.ascontiguousarray(R, dtype=float), bool(antipodal))
  for a in range(0, n, SAMPLE_CHUNK):
    b = min(n, a + SAMPLE_CHUNK)
    loop(out[a:b], rng.random((b - a, 7)), *args)
  return out
