# the harmonic conventions of docs/design/harmonics.md: the coefficient layout, Wigner-d
# and Wigner-D matrices, the group projections, the direct series, spherical harmonics and
# Clebsch-Gordan coefficients (MATLAB tools/math_tools: deg2dim, WignerD, sphericalY)
#
#   wignerd(beta, l), wignerd(beta, bandwidth=L)      d_l(beta), or the list d_0 .. d_L
#   WignerD(rot, l), WignerD(rot, bandwidth=L)        D_l(rot) = e^{-ik alpha} d_l e^{-il gamma}
#   WignerD(cs, l), WignerD(cs, bandwidth=L)          the projection mean of D_l(s)^T over the group
#   wignerSeries(fhat, rot)                           sum_n sqrt(2n+1) <fhat_n, D_n(rot)>
#   sphericalY(l, v), normalizedLegendre(l, x)        Y_l^m and P_l^m, m in front
#   clebschGordan(j1, m1, j2, m2, J, M), wigner3j, clebschGordanTensor(l1, l2, J)
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from ..geometry.frames import referenceFrame
from ..geometry.quaternion import orthogonalTransform, quaternion
from ..geometry.representations import quat2euler
from ..geometry.symmetry import symmetry
from ..geometry.vector3d import vector3d

# ------------------------------------------------------------------ H1 layout


def deg2dim(n):
  return n * (2 * n - 1) * (2 * n + 1) // 3


# the bandwidth of a coefficient array of length d, so that dim2deg(deg2dim(n + 1)) == n
def dim2deg(d):
  n = max(0, round((3 * d / 4) ** (1 / 3)) - 1)
  while deg2dim(n + 2) <= d:
    n += 1
  while deg2dim(n + 1) > d:
    n -= 1
  return n


def degreeRange(n):
  return slice(deg2dim(n), deg2dim(n + 1))


# the block of degree n as a view, rows the specimen side k, columns the crystal side l
def degreeBlock(fhat, n):
  fhat = np.asarray(fhat)
  m = 2 * n + 1
  return fhat[..., degreeRange(n)].reshape(*fhat.shape[:-1], m, m).swapaxes(-1, -2)


# the one degree or the whole list, from the MATLAB pair of call forms
def _degrees(degree, bandwidth):
  if (degree is None) == (bandwidth is None):
    raise TypeError("give either the degree l or bandwidth=L")
  return (degree, True) if bandwidth is None else (bandwidth, False)


# ------------------------------------------------------------------ H2 Wigner-d


# the closed form on the outer frame of degree n, sqrt(binom(2n, a)) s^a c^b for a = 0..2n
# and b = 2n - a, as one cumulative product so that neither factor overflows on its own
def _frameValues(n, s, c):
  i = np.arange(1, 2 * n + 1, dtype=s.dtype)
  g = np.concatenate([np.ones(1, dtype=s.dtype), np.cumprod(np.sqrt((2 * n - i + 1) / i) * s)])
  return g * c ** np.arange(2 * n, -1, -1, dtype=s.dtype)


# the Wigner-d matrices d_0(beta) .. d_L(beta) one degree at a time by the three-term
# recursion in the degree, the interior of degree n from the two previous degrees, the frame
# from the closed form; a generator, so that a transform walking the degrees in order holds
# three matrices and not the whole table
def wignerdDegrees(beta, L):
  T = beta.dtype
  x, s, c = np.cos(beta), np.sin(beta / 2), np.cos(beta / 2)
  d = [np.ones((1, 1), dtype=T)]
  yield 0, d[0]
  for n in range(1, L + 1):
    D = np.empty((2 * n + 1, 2 * n + 1), dtype=T)
    F = _frameValues(n, s, c)
    sign = np.where(np.arange(2 * n + 1) % 2, -1, 1).astype(T)
    D[0, :] = F
    D[-1, :] = (sign * F)[::-1]
    D[:, -1] = F[::-1]
    D[:, 0] = sign * F
    if n == 1:
      D[1, 1] = x
    else:
      m = n - 1
      k = np.arange(-m, m + 1, dtype=T)
      w = np.sqrt(np.outer((m + 1) ** 2 - k * k, (m + 1) ** 2 - k * k))
      a = (m + 1) * (2 * m + 1) / w
      b = (m + 1) * np.sqrt(np.outer(m * m - k * k, m * m - k * k)) / (m * w)
      Q = np.zeros((2 * m + 1, 2 * m + 1), dtype=T)
      Q[1:-1, 1:-1] = d[-2]
      D[1:-1, 1:-1] = a * (x - np.outer(k, k) / (m * (m + 1))) * d[-1] - b * Q
    d = [d[-1], D]
    yield n, D


def _wignerdAll(beta, L):
  return [D for _, D in wignerdDegrees(beta, L)]


def wignerd(beta, degree=None, *, bandwidth=None):
  L, single = _degrees(degree, bandwidth)
  beta = np.asarray(beta)
  beta = beta if beta.dtype.kind == "f" else beta.astype(float)
  if beta.ndim:
    raise ValueError("wignerd takes one angle beta")
  d = _wignerdAll(beta, L)
  return d[-1] if single else d


# ------------------------------------------------------------------ H3 Wigner-D


# the phases of one rotation on the list of its Wigner-d matrices
def _wignerDRotation(q, L, dtype):
  alpha, beta, gamma = (np.asarray(e, dtype=dtype) for e in quat2euler(q, "Matthies"))
  out = []
  for n, dn in enumerate(_wignerdAll(beta, L)):
    k = np.arange(-n, n + 1, dtype=dtype)
    out.append(np.exp(-1j * k * alpha)[:, None] * dn * np.exp(-1j * k * gamma)[None, :])
  return out


# the projection onto the functions invariant under a group, one recursion per element
# carrying every degree
def _wignerDGroup(pg, L, dtype):
  rots = pg.properGroup().rotations().data.reshape(-1, 4)
  P = [np.zeros((2 * n + 1, 2 * n + 1), dtype=np.result_type(dtype, 1j)) for n in range(L + 1)]
  for q in rots:
    for M, D in zip(P, _wignerDRotation(q, L, dtype)):
      M += D.T
  return [M / len(rots) for M in P]


def WignerD(g, degree=None, *, bandwidth=None, dtype=None):
  L, single = _degrees(degree, bandwidth)
  if isinstance(g, referenceFrame):
    g = g.sym
  if isinstance(g, symmetry):
    out = _wignerDGroup(g, L, np.float64 if dtype is None else np.dtype(dtype))
    return out[-1] if single else out
  if isinstance(g, orthogonalTransform):
    g = g.rot
  if not isinstance(g, quaternion):
    raise TypeError("WignerD takes a rotation, a point group or a frame")
  dtype = g.data.dtype if dtype is None else np.dtype(dtype)
  qs = g.data.reshape(-1, 4)
  per = [_wignerDRotation(q, L, dtype) for q in qs]
  out = [np.stack([p[n] for p in per]).reshape(*g.shape, 2 * n + 1, 2 * n + 1) for n in range(L + 1)]
  return out[-1] if single else out


# ------------------------------------------------------------------ H5 the held projection


# the same projection, held and kept in its support. P_n is nonzero only where both indices
# are 0 mod r, r the order of the group's axis along z: the group contains that rotation,
# whose Wigner matrix is diagonal with e^{-ik alpha}, so P = P_z P = P P_z. Only that r-th
# part of a coefficient block is ever read or written. A group acting on orientations
# through the identity alone (1 and -1, whose elements all have the identity as their
# rotation part) projects with the identity and is held as no matrices at all
class InvariantProjection:

  def __init__(self, fold, blocks):
    self.fold, self.blocks = int(fold), blocks

  @property
  def isIdentity(self):
    return not self.blocks

  @property
  def bandwidth(self):
    return len(self.blocks) - 1

  # the indices an r fold about z leaves at degree n, k = 0 mod r, as block indices
  def support(self, n):
    return foldRange(self.fold, n)


def foldRange(r, n):
  m = n // r
  return np.arange(n - r * m, n + r * m + 1, r)


def _buildProjection(pg, L, dtype):
  if pg.nfold() == 1:
    return InvariantProjection(1, [])
  r = pg.multiplicityZ
  full = _wignerDGroup(pg, L, dtype)
  if r == 1:
    return InvariantProjection(1, full)
  return InvariantProjection(r, [M[np.ix_(foldRange(r, n), foldRange(r, n))] for n, M in enumerate(full)])


# the projection depends on the group, the bandwidth and the precision only, while applying
# it costs a fraction of building it, so it is held: keyed by the group's identity, since
# frames are interned and two groups of one symbol in different axes compare equal with
# different elements. What is held is (2 floor(n/r) + 1)^2 complex numbers per degree, and
# the bound is on those bytes; above it the build is paid per call
_PROJECTIONS = {}
PROJECTION_CACHE_MAX_BYTES = 48 * 1024 ** 2


def _projectionBytes(r, L, dtype):
  return 2 * np.dtype(dtype).itemsize * sum((2 * (n // r) + 1) ** 2 for n in range(L + 1))


def invariantProjection(g, L, dtype=np.float64):
  pg = g.sym if isinstance(g, referenceFrame) else g
  dtype = np.dtype(dtype)
  if pg.nfold() == 1:
    return InvariantProjection(1, [])
  if _projectionBytes(pg.multiplicityZ, L, dtype) > PROJECTION_CACHE_MAX_BYTES:
    return _buildProjection(pg, L, dtype)
  key = (id(pg), dtype.name)
  held = _PROJECTIONS.get(key)
  if held is None or held[1].bandwidth < L:
    held = (pg, _buildProjection(pg, L, dtype))
    _PROJECTIONS[key] = held
  return InvariantProjection(held[1].fold, held[1].blocks[:L + 1])


# B <- P_n B, B <- B Q_n and B <- P_n B Q_n on one degree block, in place: what the
# projection kills is written zero and what it ignores is never read
def _projectBlock(B, P, Q, n):
  if P is not None and P.isIdentity:
    P = None
  if Q is not None and Q.isIdentity:
    Q = None
  if P is None and Q is None:
    return B
  rows = slice(None) if P is None or P.fold == 1 else P.support(n)
  cols = slice(None) if Q is None or Q.fold == 1 else Q.support(n)
  C = B[rows][:, cols]
  if P is not None:
    C = P.blocks[n] @ C
  if Q is not None:
    C = C @ Q.blocks[n]
  if isinstance(rows, slice) and isinstance(cols, slice):
    B[...] = C
  else:
    B[...] = 0
    B[np.ix_(np.arange(B.shape[0])[rows], np.arange(B.shape[1])[cols])] = C
  return B


def project(fhat, P, Q):
  for n in range(dim2deg(fhat.shape[-1]) + 1):
    _projectBlock(degreeBlock(fhat, n), P, Q, n)
  return fhat


def projectLeft(fhat, P):
  return project(fhat, P, None)


def projectRight(fhat, Q):
  return project(fhat, None, Q)


# ------------------------------------------------------------------ H5 the series


# the series evaluated by the direct sum at every rotation of a container, the reference
# for the transforms of F2; leading batch axes of the coefficients come first
def wignerSeries(fhat, r):
  fhat = np.asarray(fhat)
  L = dim2deg(fhat.shape[-1])
  D = WignerD(r, bandwidth=L)
  out = np.zeros((*fhat.shape[:-1], *r.shape), dtype=np.result_type(fhat.dtype, D[0].dtype))
  for n in range(L + 1):
    block = degreeBlock(fhat, n).reshape(*fhat.shape[:-1], *([1] * r.ndim), 2 * n + 1, 2 * n + 1)
    out += math.sqrt(2 * n + 1) * np.sum(block * D[n], axis=(-2, -1))
  return out


# ------------------------------------------------------------------ H4 spherical harmonics


# the fully normalised associated Legendre functions P_l^m(x), m = 0..l, with the
# Condon-Shortley phase, so that Y_l^m = P_l^m e^{im rho}; m in front of the shape of x
def normalizedLegendre(l, x):
  x = np.asarray(x)
  x = x if x.dtype.kind == "f" else x.astype(float)
  T = x.dtype
  P = np.empty((l + 1, *x.shape), dtype=T)
  st = np.sqrt(np.maximum(0, 1 - x * x))
  pmm = np.full(x.shape, 1 / math.sqrt(4 * math.pi), dtype=T)
  for m in range(l + 1):
    if m > 0:
      pmm = pmm * (-math.sqrt((2 * m + 1) / (2 * m)) * st)
    # climb from degree m to l at fixed order m
    p2, p1 = np.zeros(x.shape, dtype=T), pmm
    for n in range(m + 1, l + 1):
      p = math.sqrt((4 * n * n - 1) / (n * n - m * m)) * (x * p1 - math.sqrt(((n - 1) ** 2 - m * m) / (4 * (n - 1) ** 2 - 1)) * p2)
      p2, p1 = p1, p
    P[m] = p1
  return P


# the spherical harmonics Y_l^m, m = -l..l, in front of the container shape
def sphericalY(l, v):
  v = v if isinstance(v, vector3d) else vector3d(v)
  theta, rho = v.theta, v.rho
  P = normalizedLegendre(l, np.cos(theta))
  Y = np.empty((2 * l + 1, *v.shape), dtype=np.result_type(P.dtype, 1j))
  for m in range(l + 1):
    y = P[m] * np.exp(1j * m * rho)
    Y[l + m] = y
    Y[l - m] = -np.conj(y) if m % 2 else np.conj(y)
  return Y


# ------------------------------------------------------------------ H6 Clebsch-Gordan


# <j1 m1 j2 m2 | J M> by the Racah formula in exact rational arithmetic
def clebschGordan(j1, m1, j2, m2, J, M):
  if not (m1 + m2 == M and abs(j1 - j2) <= J <= j1 + j2 and abs(m1) <= j1 and abs(m2) <= j2 and abs(M) <= J):
    return 0.0
  f = math.factorial
  A = Fraction((2 * J + 1) * f(J + j1 - j2) * f(J - j1 + j2) * f(j1 + j2 - J), f(j1 + j2 + J + 1))
  B = f(J + M) * f(J - M) * f(j1 - m1) * f(j1 + m1) * f(j2 - m2) * f(j2 + m2)
  S = Fraction(0)
  for k in range(max(0, j2 - J - m1, j1 - J + m2), min(j1 + j2 - J, j1 - m1, j2 + m2) + 1):
    t = Fraction(1, f(k) * f(j1 + j2 - J - k) * f(j1 - m1 - k) * f(j2 + m2 - k) * f(J - j2 + m1 + k) * f(J - j1 - m2 + k))
    S += -t if k % 2 else t
  return (1 if S > 0 else -1 if S < 0 else 0) * math.sqrt(A * B * S * S)


def wigner3j(j1, j2, j3, m1, m2, m3):
  return (-1 if (j1 - j2 - m3) % 2 else 1) / math.sqrt(2 * j3 + 1) * clebschGordan(j1, m1, j2, m2, j3, -m3)


# the array C[m1, m2, M] of degrees l1, l2 coupled to J
def clebschGordanTensor(l1, l2, J):
  C = np.zeros((2 * l1 + 1, 2 * l2 + 1, 2 * J + 1))
  for M in range(-J, J + 1):
    for m2 in range(-l2, l2 + 1):
      for m1 in range(-l1, l1 + 1):
        C[m1 + l1, m2 + l2, M + J] = clebschGordan(l1, m1, l2, m2, J, M)
  return C


__all__ = ["deg2dim", "dim2deg", "degreeRange", "degreeBlock", "wignerd", "wignerdDegrees", "WignerD", "wignerSeries",
           "InvariantProjection", "invariantProjection", "project", "projectLeft", "projectRight",
           "normalizedLegendre", "sphericalY", "clebschGordan", "wigner3j", "clebschGordanTensor"]
