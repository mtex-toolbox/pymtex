# the discrepancy between two measures on the sphere or on SO(3), densities or weighted point
# sets in any pairing (MATLAB S2Fun/discrepancy, vector3d/discrepancy, SO3Fun/discrepancy,
# rotation/discrepancy, sphericalDiscrepancy, rotationalDiscrepancy; develop, 2026-09-19)
#
#   d = discrepancy(v, w, weights=c, weights2=e)              D_2, the default
#   d = discrepancy(f, v, metric='L2', bandwidth=32, degreeWeights=lambda l: exp(-0.1 * l * (l + 1)))
#   d, info = discrepancy(f, v, metric='D_cap', numCenters=4097, numHeights=513, info=True)
#   d = discrepancy(odf, ori, metric='D_ball', numCenters=4097, numRadii=513)
#   d = discrepancy(odf, ori, metric='kernel')                 the squared objective of optimalSample
#
# The difference nu = mu - eta is held as a signed harmonic density plus signed atoms; a point
# set keeps its atoms, it is never smoothed. With C(m, t) = {x : m.x >= t} on the sphere and
# B(C, r) = {R : angle(C, R) <= r} on SO(3):
#   D_cap = sup |nu(C(m, t))|,  D_ball = sup |nu(B(C, r))|      by a finite search, a lower bound
#   D_2^2 = int int |nu(C(m, t))|^2 dsigma(m) dt                 Stolarsky: the chordal energy
#         = nu(S2)^2 - 1/4 int int |x - y| dnu dnu
#   D_2^2 = nu(SO3)^2 / 2 - 1/(2 sqrt(2) pi) int int |R - S|_F dnu dnu,  |R - S|_F = sqrt(8) sin(w/2)
#   L2^2  = sum_l w_l sum_k |nu_hat_lk|^2                         in the bandlimited orthonormal basis
# All three count a difference in total mass. A function keeps its integral, a sample takes the
# one of the function it is compared with (mass one against a sample or a vanishing integral).
# An antipodal point set is half its mass at every +-x, an orientation the uniform measure on its
# orbit, a bare rotation one atom. 'kernel' is the historical squared and truncated objective
# of optimalSample, degree zero dropped, so it takes two functions of equal integral only.
#
# The energies are sums of |c|^2 over a degree and of inner products within a degree, which the
# unitary change between MTEX's basis and the internal one leaves alone: everything runs on _fhat.
from __future__ import annotations

import math
import types

import numpy as np

from .so3funclass import SO3Fun
from .s2funclass import S2Fun
from ..geometry.install import install
from ..geometry.orientation import orientation
from ..geometry.quaternion import rotation
from ..geometry.symmetryops import symmetriseOrientation
from ..geometry.vector3d import vector3d
from .characteristics import S2_KINDS, SO3_KINDS
from .harmonics import _s2Degrees, _so3Degrees, deg2dim
from .kernelfunctions import S2RestrictedDistanceKernel, SO3RestrictedDistanceKernel
from .s2fun import S2FunHarmonic
from .so3fun import SO3FunHarmonic
from .transforms import sphericalSeries, sphericalSeriesAdjoint, wignerSeries, wignerSeriesAdjoint

# the atoms the chordal energy takes at a time
BLOCK = 512

# the centers the cap and ball searches take at a time
CENTER_BLOCK = 256

_S2_METRICS = {"d_cap": "D_cap", "dcap": "D_cap", "d_2": "D_2", "d2": "D_2", "l2": "L2", "e_l": "L2"}
_SO3_METRICS = {"d_ball": "D_ball", "dball": "D_ball", "d_2": "D_2", "d2": "D_2", "l2": "L2", "e_l": "L2"}

# what a measure on the sphere and one on SO(3) is given as
_S2_INPUTS = (S2Fun, vector3d)
_SO3_INPUTS = (SO3Fun, rotation)


# ------------------------------------------------------------------ the inputs


def _metricName(metric, table, space):
  if not isinstance(metric, str):
    raise ValueError(f"the metric must be {', '.join(sorted(set(table.values())))} or kernel")
  m = table.get(metric.lower())
  if m is None:
    raise ValueError(f"unknown {space} discrepancy metric: {metric}")
  return m


def _bandwidth(bw):
  if isinstance(bw, bool) or not np.isscalar(bw) or not np.isreal(bw) or not np.isfinite(bw) or bw < 0 or bw != int(bw):
    raise ValueError("the bandwidth must be a nonnegative integer")
  return int(bw)


def _sampleWeights(n, weights=None):
  """the weights of n points as a probability distribution, equal when none are given"""
  if weights is None:
    return np.full(n, 1.0 / n)
  c = np.asarray(weights, dtype=float).ravel()
  if c.size != n:
    raise ValueError("the number of weights does not match the number of points")
  if np.any(c < 0):
    raise ValueError("the weights have to be non negative")
  s = c.sum()
  if not (np.all(np.isfinite(c)) and s > 0):
    raise ValueError("point weights must be finite and have positive total weight")
  return c / s


def _mass(f):
  """the integral of a scalar real function, refusing lists and complex or infinite values"""
  if getattr(f, "shape", ()) not in ((), (1,)):
    raise ValueError("discrepancy requires scalar functions")
  if hasattr(f, "sum"):
    m = f.sum()
  elif hasattr(f, "mean"):
    m = 4 * math.pi * f.mean()
  else:
    m = S2FunHarmonic(f).sum()
  m = complex(np.asarray(m).ravel()[0])
  if abs(m.imag) > 1e-12 * max(1.0, abs(m.real)) or not np.isfinite(m.real):
    raise ValueError("the function integral must be real and finite")
  return m.real


def _sampleMass(partner):
  # a sample represents the function it is compared with, hence carries its integral, but a
  # vanishing one represents nothing
  return 1.0 if partner is None or partner == 0 else partner


def _cut(fhat, n):
  """the coefficients padded with zeros or truncated to the length n"""
  out = np.zeros(n, dtype=complex)
  m = min(n, fhat.shape[-1])
  out[:m] = fhat.ravel()[:m]
  return out


def _directions(v):
  """the unit directions of a point set as an (n, 3) array"""
  x = np.asarray(v.data, dtype=float).reshape(-1, 3)
  if x.shape[0] == 0:
    raise ValueError("point sets must not be empty")
  r = np.linalg.norm(x, axis=1)
  if not (np.all(np.isfinite(x)) and np.all(r > 0)):
    raise ValueError("points must have finite nonzero coordinates")
  return x / r[:, None]


def _quaternions(r):
  """the unit quaternions of a rotation set as an (n, 4) array"""
  q = np.asarray(r.data, dtype=float).reshape(-1, 4)
  if q.shape[0] == 0:
    raise ValueError("orientation sets must not be empty")
  return q / np.linalg.norm(q, axis=1)[:, None]


def _degreeWeights(dw, L):
  """the energy weights of the degrees 0..L, ones by default"""
  w = np.ones(L + 1) if dw is None else np.asarray(dw(np.arange(L + 1)) if callable(dw) else dw)
  w = np.ravel(w)
  if w.size != L + 1 or not np.isrealobj(w) or not np.all(np.isfinite(w)) or np.any(w < 0):
    raise ValueError("degreeWeights must contain L+1 finite nonnegative energy weights")
  return w.astype(float)


def _withInfo(res, info, wanted):
  return (res, types.SimpleNamespace(**info)) if wanted else res


# ------------------------------------------------------------------ the sphere


def _s2Representation(a, L, mass, weights):
  """the signed harmonic density h, the atoms x and their masses c, and whether nothing was lost"""
  h, x, c = np.zeros((L + 1) ** 2, dtype=complex), np.zeros((0, 3)), np.zeros(0)
  if isinstance(a, S2Fun):
    exact = isinstance(a, S2FunHarmonic) and a.bandwidth <= L
    f = S2FunHarmonic(a, bandwidth=L)
    if not f.isReal:
      raise ValueError("functions must be real valued")
    h = _cut(f._fhat, (L + 1) ** 2)
    if not np.all(np.isfinite(h)):
      raise ValueError("harmonic coefficients must be finite")
    h[0] = mass / math.sqrt(4 * math.pi)
    return h, x, c, exact
  x = _directions(a)
  c = _sampleWeights(x.shape[0], weights)
  # an antipodal point is half its mass at either sign
  if a.antipodal:
    x, c = np.concatenate([x, -x]), np.concatenate([c, c]) / 2
  return h, x, mass * c, True


def _s2Moments(x, c, L):
  """the coefficients int conj(Y_lk) dnu of the atoms, degree zero exact"""
  h = np.zeros((L + 1) ** 2, dtype=complex)
  if c.size == 0:
    return h
  if L > 0:
    h = np.asarray(sphericalSeriesAdjoint(vector3d(x), c, L, weights=np.ones(c.size)), dtype=complex)
  h[0] = c.sum() / math.sqrt(4 * math.pi)
  return h


def _s2DegreeValues(fhat, m, L):
  """the real values of every degree of fhat at the centers m, (n, L + 1)"""
  values = np.zeros((m.shape[0], L + 1))
  values[:, 0] = fhat[0].real / math.sqrt(4 * math.pi)
  v = vector3d(m)
  for l in range(1, L + 1):
    block = fhat[l * l:(l + 1) ** 2]
    if np.any(block):
      g = np.zeros((l + 1) ** 2, dtype=complex)
      g[l * l:] = block
      values[:, l] = np.real(sphericalSeries(g, v, real=False)).ravel()
  return values


def _maximumCap(fhat, x, c, L, numCenters, numHeights, centers):
  nz = np.flatnonzero(fhat)
  L = 0 if nz.size == 0 else min(L, math.isqrt(int(nz[-1])))
  if not (_isCount(numCenters, 1) and _isCount(numHeights, 2)):
    raise ValueError("use positive integer numCenters and numHeights >= 2")
  if centers is None:
    from ..geometry.grids.fibonacci import fibonacciS2Grid
    centers = np.concatenate([vector3d(fibonacciS2Grid(int(numCenters))).data.reshape(-1, 3), np.eye(3), -np.eye(3), x, -x])
  else:
    if not isinstance(centers, vector3d):
      raise ValueError("centers must be finite nonzero vector3d directions")
    centers = _directions(centers)
  # every degree once at all centers; Funk-Hecke turns it into the cap integral
  # 2 pi sum_l f_l(m) int_t^1 P_l(s) ds
  values = _s2DegreeValues(fhat, centers, L)
  grid = np.linspace(-1, 1, int(numHeights))

  def continuous(V, t):
    out = 2 * np.pi * V[:, :1] * (1 - t)
    prev, cur = np.ones_like(t), t
    for l in range(1, L + 1):
      nxt = ((2 * l + 1) * t * cur - l * prev) / (l + 1)
      out += 2 * np.pi * V[:, l:l + 1] * (prev - nxt) / (2 * l + 1)
      prev, cur = cur, nxt
    return out

  # the mass of the atoms with height >= t, and > t
  def masses(q, cq, t):
    order = np.argsort(-q, kind="stable")
    s = np.concatenate([[0.0], np.cumsum(cq[order])])
    neg = -q[order]
    return s[np.searchsorted(neg, -t, "right")], s[np.searchsorted(neg, -t, "left")]

  def heights(m):
    return np.clip(x @ m.T, -1, 1).T
  return _search(centers, values, heights, grid, c, continuous, masses)


def _isCount(n, least):
  return np.isscalar(n) and not isinstance(n, bool) and np.isfinite(n) and n >= least and n == int(n)


def _search(centers, values, heights, grid, c, continuous, masses):
  """the largest |nu| over the closed and open sets of every center and height: the grid of
  heights and the heights of all atoms, both sides of every jump"""
  best = dict(value=0.0, center=centers[0], height=grid[-1], closed=True, signed=0.0, numCenters=centers.shape[0])
  for first in range(0, centers.shape[0], CENTER_BLOCK):
    m = centers[first:first + CENTER_BLOCK]
    q = heights(m) if c.size else np.zeros((m.shape[0], 0))
    t = np.concatenate([np.broadcast_to(grid, (m.shape[0], grid.size)), q], axis=1)
    cont = continuous(values[first:first + CENTER_BLOCK], t)
    for j in range(m.shape[0]):
      closed, opened = masses(q[j], c, t[j]) if c.size else (0.0, 0.0)
      d = np.stack([cont[j] + closed, cont[j] + opened])
      a = np.abs(d)
      value = a.max()
      if value > best["value"]:
        # the first among equals: the closed side, then the smallest height, as MATLAB scans
        side, k = min(zip(*np.nonzero(a == value)), key=lambda sk: (sk[0], t[j][sk[1]]))
        best.update(value=float(value), center=m[j], height=float(t[j][k]), closed=side == 0, signed=float(d[side, k]))
  return best


# the two sides at one bandwidth, a function's own where the other is a point set: the
# difference of their harmonic parts and their atoms with signed weights; each function keeps its
# integral, a point set takes its partner's
def _difference(a, b, family, defaultBandwidth, represent, bandwidth, weights, weights2):
  isFun = isinstance(a, family), isinstance(b, family)
  if isFun[0] != isFun[1]:
    L = defaultBandwidth(a if isFun[0] else b)
  else:
    L = max(defaultBandwidth(a), defaultBandwidth(b))
  L = _bandwidth(L if bandwidth is None else bandwidth)
  massA = _mass(a) if isFun[0] else None
  massB = _mass(b) if isFun[1] else None
  massA = _sampleMass(massB) if massA is None else massA
  massB = _sampleMass(massA) if massB is None else massB
  hA, xA, cA, exactA = represent(a, L, massA, weights)
  hB, xB, cB, exactB = represent(b, L, massB, weights2 if not any(isFun) else weights)
  return L, hA - hB, np.concatenate([xA, xB]), np.concatenate([cA, -cB]), exactA and exactB


def sphericalDiscrepancy(a, b, metric="D_2", *, weights=None, weights2=None, bandwidth=None, degreeWeights=None,
                         numCenters=2049, numHeights=257, centers=None, squared=False, info=False):
  """the discrepancy of two spherical densities or weighted point sets, D_2, D_cap or L2 (E_L);
  the shared form of S2Fun/discrepancy and vector3d/discrepancy, info=True adds the details"""
  metric = _metricName(metric, _S2_METRICS, "spherical")
  if not (isinstance(a, _S2_INPUTS) and isinstance(b, _S2_INPUTS)):
    raise TypeError("inputs must be S2Fun or vector3d")
  L, fhat, x, c, exact = _difference(a, b, S2Fun, lambda f: f.bandwidth if isinstance(f, S2FunHarmonic) else 128, _s2Representation,
                                     bandwidth, weights, weights2)
  out = dict(method="", isExact=exact, bandwidth=L)

  if metric == "D_cap":
    best = _maximumCap(fhat, x, c, L, numCenters, numHeights, centers)
    out.update(method="finite center and height search", isExact=False, center=vector3d(best["center"]),
               height=best["height"], closed=best["closed"], signedDifference=best["signed"],
               numCenters=best["numCenters"], numHeights=int(numHeights))
    res = best["value"] ** 2 if squared else best["value"]
    return _withInfo(res, out, info)
  if metric == "D_2":
    # Stolarsky: the chordal energy carries the mass with a negative weight, the cap integral
    # with a positive one, hence the two constants; the atoms keep their whole spectral tail
    C = c.sum()
    energy = 8 * np.pi / 3 * abs(fhat[0] + C / math.sqrt(4 * np.pi)) ** 2 + C * C / 3
    for first in range(0, c.size, BLOCK):
      d = np.sqrt(np.sum((x[first:first + BLOCK, None, :] - x[None, :, :]) ** 2, axis=-1))
      energy -= c[first:first + BLOCK] @ d @ c / 4
    if np.any(fhat[1:]):
      l = _s2Degrees(L)
      w = np.where(l > 0, 4 * np.pi / ((2.0 * l - 1) * (2 * l + 1) * (2 * l + 3)), 0.0)
      moments = _s2Moments(x, c, L)
      energy += np.sum(w * (np.abs(fhat) ** 2 + 2 * np.real(np.conj(fhat) * moments)))
    out["method"] = "chordal atomic energy and harmonic continuous terms"
  else:
    diff = fhat + _s2Moments(x, c, L)
    dw = _degreeWeights(degreeWeights, L)
    energy = np.sum(dw[_s2Degrees(L)] * np.abs(diff) ** 2)
    out["method"] = "weighted bandlimited harmonic norm"
  res = max(0.0, float(np.real(energy)))
  return _withInfo(res if squared else math.sqrt(res), out, info)


def _s2KernelWeights(L, antipodal):
  """the weights the restricted distance kernel puts on the coefficients, degree zero dropped,
  the odd degrees too for an antipodal function"""
  A = S2RestrictedDistanceKernel(L + 1).calcFourier(L)
  l = _s2Degrees(L)
  kept = (l > 0) & ~(antipodal & (l % 2 == 1))
  w = np.zeros(l.size)
  w[kept] = np.sqrt(4 * np.pi * A[l[kept]] / (2 * l[kept] + 1))
  return w


def _s2SampleMoments(v, c, L):
  """the adjoint transform of a weighted point set, even for an antipodal one"""
  x = _directions(v)
  if v.antipodal:
    x, c = np.concatenate([x, -x]), np.concatenate([c, c]) / 2
  return _s2Moments(x, c, L)


def _legacyS2(a, b, weights=None, weights2=None, bandwidth=None):
  """the squared truncated kernel objective of S2Fun/optimalSample"""
  if isinstance(a, vector3d) and isinstance(b, S2Fun):
    return _legacyS2(b, a, weights=weights, bandwidth=bandwidth)
  if not (isinstance(a, _S2_INPUTS) and isinstance(b, _S2_INPUTS)):
    raise TypeError("the second input must be an S2Fun or a vector3d point set")
  if isinstance(a, vector3d):
    L = _bandwidth(128 if bandwidth is None else bandwidth)
    if a.size == 0 or b.size == 0:
      raise ValueError("point sets must not be empty")
    c, d = _sampleWeights(a.size, weights), _sampleWeights(b.size, weights2)
    # two probability measures share the constant coefficient
    diff = _s2SampleMoments(b, d, L) - _s2SampleMoments(a, c, L)
    w = _s2KernelWeights(L, False)
    return float(np.sum(np.abs(w * diff) ** 2))
  L = a.bandwidth if isinstance(a, S2FunHarmonic) else 128
  if isinstance(b, S2Fun):
    L = max(L, b.bandwidth if isinstance(b, S2FunHarmonic) else 128)
  L = _bandwidth(L if bandwidth is None else bandwidth)
  if isinstance(b, S2Fun):
    # a quadrature integral is checked to the tolerance of its grid, before the projection
    m1, m2 = _mass(a), _mass(b)
    if not abs(m1 - m2) <= 1e-6 * max(abs(m1), abs(m2)):
      raise ValueError(f"dropping degree zero only measures a difference if the integrals agree, these are {m1:g} and "
                       f"{m2:g} - scale one of them, or use a metric that keeps the mass")
  f = S2FunHarmonic(a, bandwidth=L)
  fhat = _cut(f._fhat, (L + 1) ** 2)
  if isinstance(b, S2Fun):
    g = S2FunHarmonic(b, bandwidth=L)
    antipodal = f.antipodal and g.antipodal
    diff = _cut(g._fhat, (L + 1) ** 2) - fhat
  else:
    if b.size == 0:
      raise ValueError("point sets must not be empty")
    c = _sampleWeights(b.size, weights)
    antipodal = f.antipodal
    diff = _mass(a) * _s2SampleMoments(b, c, L) - fhat
  w = _s2KernelWeights(L, antipodal)
  return float(np.sum(np.abs(w * diff) ** 2))


# ------------------------------------------------------------------ the rotation group


def _so3Representation(a, L, mass, weights):
  """the signed harmonic density h in the basis orthonormal for the Haar measure of mass 8 pi^2,
  sqrt(8 pi^2) times MTEX's, the atoms as quaternions and their masses, and whether it is exact"""
  h, x, c = np.zeros(deg2dim(L + 1), dtype=complex), np.zeros((0, 4)), np.zeros(0)
  if isinstance(a, SO3Fun):
    exact = isinstance(a, SO3FunHarmonic) and a.bandwidth <= L
    f = SO3FunHarmonic(a, bandwidth=L)
    if not f.isReal:
      raise ValueError("functions must be real valued")
    h = math.sqrt(8 * np.pi ** 2) * _cut(f._fhat, deg2dim(L + 1))
    if not np.all(np.isfinite(h)):
      raise ValueError("Wigner coefficients must be finite")
    h[0] = mass / math.sqrt(8 * np.pi ** 2)
    return h, x, c, exact
  x = _quaternions(a)
  c = _sampleWeights(x.shape[0], weights)
  # a symmetric orientation is the uniform measure on its orbit
  if isinstance(a, orientation):
    s = symmetriseOrientation(a.flatten())
    x = _quaternions(s)
    c = np.tile(c / s.shape[0], s.shape[0])
  return h, x, mass * c, True


def _so3Moments(x, c, L):
  """the Haar orthonormal coefficients of the atoms, degree zero exact"""
  h = np.zeros(deg2dim(L + 1), dtype=complex)
  if c.size == 0:
    return h
  if L > 0:
    h = np.asarray(wignerSeriesAdjoint(rotation(x), c, L, weights=np.ones(c.size)), dtype=complex) / math.sqrt(8 * np.pi ** 2)
  h[0] = c.sum() / math.sqrt(8 * np.pi ** 2)
  return h


def _so3DegreeValues(fhat, r, L):
  """the real values of every degree of fhat at the rotations r, (n, L + 1)"""
  values = np.zeros((r.shape[0], L + 1))
  values[:, 0] = fhat[0].real
  rot = rotation(r)
  for l in range(1, L + 1):
    block = fhat[deg2dim(l):deg2dim(l + 1)]
    if np.any(block):
      g = np.zeros(deg2dim(l + 1), dtype=complex)
      g[deg2dim(l):] = block
      values[:, l] = np.real(wignerSeries(g, rot, real=False)).ravel()
  return values


def _maximumBall(fhat, x, c, L, numCenters, numRadii, centers):
  nz = np.flatnonzero(fhat)
  if nz.size == 0:
    L = 0
  else:
    while L > 0 and deg2dim(L) > nz[-1]:
      L -= 1
  if not (_isCount(numCenters, 1) and _isCount(numRadii, 2)):
    raise ValueError("use positive integer numCenters and numRadii >= 2")
  if centers is None:
    from ..geometry.frames import crystalFrame, specimenFrame
    from ..geometry.grids.so3grid import equispacedSO3Grid
    g = equispacedSO3Grid(crystalFrame("1"), specimenFrame("1"), points=int(numCenters))
    centers = np.concatenate([_quaternions(rotation(g)), [[1.0, 0, 0, 0]], x])
  else:
    if not isinstance(centers, rotation):
      raise ValueError("centers must be proper rotations")
    centers = _quaternions(centers)
  # every degree once at all centers; Funk-Hecke on SO(3) turns it into the ball integral
  # sqrt(8 pi^2) sum_l beta_l(r) f_l(C) / (2l + 1)
  values = _so3DegreeValues(fhat, centers, L)
  grid = np.linspace(0, np.pi, int(numRadii))
  s = math.sqrt(8 * np.pi ** 2)

  def continuous(V, t):
    out = s * V[:, :1] * (t - np.sin(t)) / np.pi
    for l in range(1, L + 1):
      out += s * V[:, l:l + 1] / (2 * l + 1) * (np.sin(l * t) / l - np.sin((l + 1) * t) / (l + 1)) / np.pi
    return out

  # the mass of the atoms at an angle <= t, and < t
  def masses(q, cq, t):
    order = np.argsort(q, kind="stable")
    p = np.concatenate([[0.0], np.cumsum(cq[order])])
    qs = q[order]
    return p[np.searchsorted(qs, t, "right")], p[np.searchsorted(qs, t, "left")]

  def heights(m):
    return 2 * np.arccos(np.clip(np.abs(m @ x.T), 0, 1))
  return _search(centers, values, heights, grid, c, continuous, masses)


def rotationalDiscrepancy(a, b, metric="D_2", *, weights=None, weights2=None, bandwidth=None, degreeWeights=None,
                          numCenters=2049, numRadii=257, centers=None, squared=False, info=False):
  """the discrepancy of two rotational densities or weighted orientation sets, D_2, D_ball or
  L2 (E_L); the shared form of SO3Fun/discrepancy and rotation/discrepancy, info=True adds the details"""
  metric = _metricName(metric, _SO3_METRICS, "rotational")
  if not (isinstance(a, _SO3_INPUTS) and isinstance(b, _SO3_INPUTS)):
    raise TypeError("inputs must be SO3Fun or rotation")
  L, fhat, x, c, exact = _difference(a, b, SO3Fun, lambda f: f.bandwidth if isinstance(f, SO3FunHarmonic) else 64, _so3Representation,
                                     bandwidth, weights, weights2)
  out = dict(method="", isExact=exact, bandwidth=L)

  if metric == "D_ball":
    best = _maximumBall(fhat, x, c, L, numCenters, numRadii, centers)
    out.update(method="finite center and radius search", isExact=False, center=rotation(best["center"]),
               radius=best["height"], closed=best["closed"], signedDifference=best["signed"],
               numCenters=best["numCenters"], numRadii=int(numRadii))
    res = best["value"] ** 2 if squared else best["value"]
    return _withInfo(res, out, info)
  if metric == "D_2":
    # Stolarsky: the chordal energy carries the mass with a negative weight, the ball integral
    # with a positive one, hence the two constants; the atoms keep their whole spectral tail
    C = c.sum()
    energy = (4 * np.pi ** 2 - 64 / 3) * abs(fhat[0] + C / math.sqrt(8 * np.pi ** 2)) ** 2 + 8 * C * C / (3 * np.pi ** 2)
    for first in range(0, c.size, BLOCK):
      # sin(w/2) = |q - p| |q + p| / 2 from the quaternions
      qa = x[first:first + BLOCK, None, :]
      minus = np.sum((qa - x[None, :, :]) ** 2, axis=-1)
      plus = np.sum((qa + x[None, :, :]) ** 2, axis=-1)
      energy -= c[first:first + BLOCK] @ np.sqrt(minus * plus) @ c / (2 * np.pi)
    if np.any(fhat[1:]):
      n = _so3Degrees(L)
      w = np.where(n > 0, 64 / ((2.0 * n - 1) * (2 * n + 1) ** 2 * (2 * n + 3)), 0.0)
      moments = _so3Moments(x, c, L)
      energy += np.sum(w * (np.abs(fhat) ** 2 + 2 * np.real(np.conj(fhat) * moments)))
    out["method"] = "chordal atomic energy and harmonic continuous terms"
  else:
    diff = fhat + _so3Moments(x, c, L)
    dw = _degreeWeights(degreeWeights, L)
    energy = np.sum(dw[_so3Degrees(L)] * np.abs(diff) ** 2)
    out["method"] = "weighted bandlimited harmonic norm"
  res = max(0.0, float(np.real(energy)))
  return _withInfo(res if squared else math.sqrt(res), out, info)


def _so3KernelWeights(L):
  """the weights the restricted distance kernel puts on the Wigner coefficients, degree zero dropped"""
  A = SO3RestrictedDistanceKernel(L + 1).calcFourier(L)
  n = _so3Degrees(L)
  kept = n > 0
  w = np.zeros(n.size)
  w[kept] = np.sqrt(8 * np.pi ** 2 * A[n[kept]] / (2 * n[kept] + 1))
  return w


def _so3SampleMoments(r, c, L, CS=None, SS=None, antipodal=False):
  """the adjoint transform of a weighted rotation set, MTEX's normalisation, projected onto the
  functions symmetric under the frames given"""
  q = _quaternions(r)
  h = np.zeros(deg2dim(L + 1), dtype=complex)
  if L > 0:
    h = np.asarray(wignerSeriesAdjoint(rotation(q), c, L, weights=np.ones(c.size)), dtype=complex)
    if CS is not None:
      h = SO3FunHarmonic(h, CS, SS, real=True, antipodal=antipodal, _internal=True)._fhat
  h[0] = c.sum()
  return h


def _legacySO3(a, b, weights=None, weights2=None, bandwidth=None):
  """the squared truncated kernel objective of SO3Fun/optimalSample"""
  if isinstance(a, rotation) and isinstance(b, SO3Fun):
    return _legacySO3(b, a, weights=weights, bandwidth=bandwidth)
  if not (isinstance(a, _SO3_INPUTS) and isinstance(b, _SO3_INPUTS)):
    raise TypeError("the second input must be an SO3Fun or a rotation set")
  if isinstance(a, rotation):
    L = _bandwidth(32 if bandwidth is None else bandwidth)
    if a.size == 0 or b.size == 0:
      raise ValueError("orientation sets must not be empty")
    c, d = _sampleWeights(a.size, weights), _sampleWeights(b.size, weights2)
    # two probability densities share the constant coefficient
    def frames(r):
      return (r.CS, r.SS, r.antipodal) if isinstance(r, orientation) else (None, None, False)
    diff = (_so3SampleMoments(b, d, L, *frames(b)) - _so3SampleMoments(a, c, L, *frames(a))) / (math.sqrt(8) * np.pi)
    return float(np.sum(np.abs(_so3KernelWeights(L) * diff) ** 2))
  L = a.bandwidth if isinstance(a, SO3FunHarmonic) else 32
  if isinstance(b, SO3Fun):
    L = max(L, b.bandwidth if isinstance(b, SO3FunHarmonic) else 32)
  L = _bandwidth(L if bandwidth is None else bandwidth)
  if isinstance(b, SO3Fun):
    # a quadrature integral is checked to the tolerance of its grid, before the projection
    m1, m2 = _mass(a), _mass(b)
    if not abs(m1 - m2) <= 1e-6 * max(abs(m1), abs(m2)):
      raise ValueError(f"dropping degree zero only measures a difference if the integrals agree, these are {m1:g} and "
                       f"{m2:g} - scale one of them, or use a metric that keeps the mass")
  f = SO3FunHarmonic(a, bandwidth=L)
  fhat = _cut(f._fhat, deg2dim(L + 1))
  if isinstance(b, SO3Fun):
    diff = math.sqrt(8) * np.pi * (_cut(SO3FunHarmonic(b, bandwidth=L)._fhat, deg2dim(L + 1)) - fhat)
  else:
    if b.size == 0:
      raise ValueError("orientation sets must not be empty")
    c = _sampleWeights(b.size, weights)
    # the nodes carry the symmetries of the density, as in optimalSample
    mu = _so3SampleMoments(b, c, L, f.CS, f.SS, b.antipodal if isinstance(b, orientation) else False)
    diff = _mass(a) / (math.sqrt(8) * np.pi) * mu - math.sqrt(8) * np.pi * fhat
  return float(np.sum(np.abs(_so3KernelWeights(L) * diff) ** 2))


# ------------------------------------------------------------------ the call


def discrepancy(a, b, metric="D_2", **kwargs):
  """the discrepancy between two densities and/or weighted point sets on the sphere (S2Fun,
  vector3d) or on SO(3) (SO3Fun, rotation, orientation), in either order

    discrepancy(v, w, weights=c, weights2=e)             D_2, the root cap or ball L2 discrepancy
    discrepancy(f, v, metric='D_cap', numCenters=4097)   the finite search for the largest cap
    discrepancy(odf, ori, metric='D_ball', numRadii=513)
    discrepancy(f, v, metric='L2', bandwidth=32, degreeWeights=w)   'E_L' the same
    discrepancy(f, v, squared=True), discrepancy(f, v, info=True) -> (d, info)
    discrepancy(f, v, metric='kernel')                   the squared objective of optimalSample"""
  spherical = any(isinstance(x, _S2_INPUTS) for x in (a, b))
  if spherical and any(isinstance(x, _SO3_INPUTS) for x in (a, b)):
    raise TypeError("a measure on the sphere cannot be compared with one on SO(3)")
  if isinstance(metric, str) and metric.lower() == "kernel":
    wanted = kwargs.pop("info", False)
    kwargs.pop("squared", None)
    unused = set(kwargs) - {"weights", "weights2", "bandwidth"}
    if unused:
      raise TypeError(f"the kernel metric takes no {', '.join(sorted(unused))}")
    res = (_legacyS2 if spherical else _legacySO3)(a, b, **kwargs)
    return _withInfo(res, dict(method="legacy squared truncated kernel", isExact=False), wanted)
  return (sphericalDiscrepancy if spherical else rotationalDiscrepancy)(a, b, metric, **kwargs)


# ------------------------------------------------------------------ the methods


install((vector3d, rotation) + S2_KINDS + SO3_KINDS, discrepancy)

__all__ = ["discrepancy", "sphericalDiscrepancy", "rotationalDiscrepancy"]
