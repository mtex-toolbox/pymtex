# the compiled nearest search of the S2 ring grid: the same structure as the Python search -
# a binary search for the polar ring and the azimuth point, then the rings within the current
# best angle - written as one numba loop over the queries, so the logarithmic scaling is kept
# and the per-query cost leaves the interpreter. The grid passes its sorted ladders as flat
# arrays and the search binary-searches them. The Python search is the fallback without numba
# and the oracle of the tests. (The SO(3) grids search through the quaternion k-d tree of
# orientationtree.py instead, since their radius search is what the ODF reconstruction needs.)
from __future__ import annotations

import numpy as np

from ..kernels import _jit
prange = range
_COMPILED = {}


# the flat, sorted ladders of an S2 ring grid: the ring polar angles, all azimuth points
# concatenated in grid order, and per ring its start, count, minimum and period
def flatS2(g):
  betaPts = np.ascontiguousarray(g.thetaGrid.points, dtype=np.float64)
  rhoPts = np.ascontiguousarray(np.concatenate([r.points for r in g.rhoGrid]) if g.rhoGrid else np.zeros(0), dtype=np.float64)
  rmin = np.array([float(r.min) for r in g.rhoGrid])
  rper = np.array([float(r.period) if r.periodic else 0.0 for r in g.rhoGrid])
  nrho = np.array([len(r) for r in g.rhoGrid], dtype=np.int64)
  offset = np.ascontiguousarray(g.offset[:-1], dtype=np.int64)
  return betaPts, rhoPts, offset, nrho, rmin, rper


# ---------------------------------------------------------------- binary-search primitives

# the first index in [lo, hi) with pts[i] >= x
def _firstGE(pts, lo, hi, x):
  while lo < hi:
    mid = (lo + hi) // 2
    if pts[mid] < x:
      lo = mid + 1
    else:
      hi = mid
  return lo


# the local index of the azimuth point of a ring nearest to x, periodic when the ring is
def _nearestJ(rhoPts, off, n, rmin, per, x):
  if n == 1:
    return 0
  if per > 0:
    x = x - np.floor((x - rmin) / per) * per
  i = _firstGE(rhoPts, off, off + n, x) - off
  best = 0
  bd = 1e18
  for c in (i - 1, i, 0, n - 1):
    cc = 0 if c < 0 else (n - 1 if c > n - 1 else c)
    d = abs(rhoPts[off + cc] - x)
    if per > 0:
      d = d - np.floor(d / per) * per
      d = min(d, per - d)
    if d < bd:
      bd, best = d, cc
  return best


# the closest point of one ring for a query and its dot, from the spherical law of cosines
def _ringBest(k, theta, rho, ct, st, betaPts, rhoPts, offset, nrho, rmin, rper):
  off, n = offset[k], nrho[k]
  per = rper[k]
  j = _nearestJ(rhoPts, off, n, rmin[k], per, rho)
  d = abs(rho - rhoPts[off + j])
  if per > 0:
    d = d - np.floor(d / per) * per
    d = min(d, per - d)
  tk = betaPts[k]
  return off + j, ct * np.cos(tk) + st * np.sin(tk) * np.cos(d)


# ---------------------------------------------------------------- S2 nearest

def _s2NearestLoop(xyz, betaPts, rhoPts, offset, nrho, rmin, rper, antipodal, out):
  nring = betaPts.shape[0]
  for i in prange(xyz.shape[0]):
    bestIdx = -1
    bestDot = -2.0
    for pas in range(2 if antipodal else 1):
      s = -1.0 if pas == 1 else 1.0
      x, y, z = s * xyz[i, 0], s * xyz[i, 1], s * xyz[i, 2]
      theta = np.arccos(min(max(z, -1.0), 1.0))
      rho = np.arctan2(y, x)
      # the ring nearest in polar angle, a binary search with its two neighbours
      lo = _firstGE(betaPts, 0, nring, theta)
      k = lo if lo < nring else nring - 1
      if lo - 1 >= 0 and abs(betaPts[lo - 1] - theta) <= abs(betaPts[k] - theta):
        k = lo - 1
      ct, st = np.cos(theta), np.sin(theta)
      bi, bd = _ringBest(k, theta, rho, ct, st, betaPts, rhoPts, offset, nrho, rmin, rper)
      omega = np.arccos(min(max(bd, -1.0), 1.0))
      for k2 in range(nring):
        if k2 == k:
          continue
        if abs(betaPts[k2] - theta) > omega:
          continue
        i2, d2 = _ringBest(k2, theta, rho, ct, st, betaPts, rhoPts, offset, nrho, rmin, rper)
        if d2 > bd:
          bi, bd = i2, d2
      if bd > bestDot:
        bestDot, bestIdx = bd, bi
    out[i] = bestIdx


def s2Nearest(g, xyz, backend):
  if backend != "numba":
    return np.array([g._nearestIndex(x) for x in xyz], dtype=np.int64)
  xyz = np.ascontiguousarray(xyz, dtype=np.float64)
  out = np.empty(len(xyz), dtype=np.int64)
  _compileS2()
  _COMPILED["s2Nearest"](xyz, *g._flat(), bool(g.antipodal), out)
  return out


# ---------------------------------------------------------------- compilation

# the shared helpers njit-compiled once and swapped into the module namespace, so the two
# parallel loops that reference them by name compile in nopython mode
_HELPERS = ("_firstGE", "_nearestJ", "_ringBest")


def _compileHelpers():
  nb = _jit()
  globals()["prange"] = nb.prange
  for name in _HELPERS:
    if name not in _COMPILED:
      _COMPILED[name] = nb.njit(cache=True)(globals()[name])
      globals()[name] = _COMPILED[name]
  return nb


def _compileS2():
  if "s2Nearest" not in _COMPILED:
    nb = _compileHelpers()
    _COMPILED["s2Nearest"] = nb.njit(cache=True, parallel=True)(_s2NearestLoop)


__all__ = []
