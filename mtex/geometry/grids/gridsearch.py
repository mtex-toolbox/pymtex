# the compiled searches of the ring grids. The nearest search of the S2 ring grid has the same
# structure as the Python search - a binary search for the polar ring and the azimuth point,
# then the rings within the current best angle - written as one numba loop over the queries,
# so the logarithmic scaling is kept and the per-query cost leaves the interpreter. The grid
# passes its sorted ladders as flat arrays and the search binary-searches them. The Python
# search is the fallback without numba and the oracle of the tests.
#
# The radius search of the equispaced SO(3) grid walks the same structure (MATLAB SO3Grid.c,
# the Julia port's neighbours): a query is expanded over one representative per coset of
# the rotations about z on either side, the rotations about z being the periods of alpha and
# gamma; the images of z within eps pick the rings and on each ring an alpha window, and on
# each node the angle is affine in cos of the gamma difference, which picks a gamma window.
# The quaternion k-d tree of orientationtree.py is its fallback and oracle.
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ..kernels import _jit, _parallelKernel, heapBlocks, threads
prange = range
get_thread_id = None
_COMPILED = {}


def flatS2(g):
  """the flat, sorted ladders of an S2 ring grid: the ring polar angles, all azimuth points
  concatenated in grid order, and per ring its start, count, minimum and period"""
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
  """the index of the nearest node of a ring grid for every row of xyz, the compiled ring search
  under numba, the grid's Python search otherwise"""
  if backend != "numba":
    return np.array([g._nearestIndex(x) for x in xyz], dtype=np.int64)
  xyz = np.ascontiguousarray(xyz, dtype=np.float64)
  out = np.empty(len(xyz), dtype=np.int64)
  _compileS2()
  _COMPILED["s2Nearest"](xyz, *g._flat(), bool(g.antipodal), out)
  return out


# ---------------------------------------------------------------- SO(3) radius

# the first index in [lo, hi) with pts[i] > x
def _firstGT(pts, lo, hi, x):
  while lo < hi:
    mid = (lo + hi) // 2
    if pts[mid] <= x:
      lo = mid + 1
    else:
      hi = mid
  return lo


# the points of a sorted periodic ladder pts[lo:hi] on [pmin, pmin + per) within w of x, as
# the two index ranges [a0, a1) and [b0, b1)
def _window(pts, lo, hi, pmin, per, x, w):
  if 2 * w >= per:
    return lo, hi, hi, hi
  xl = x - w
  xl = xl - np.floor((xl - pmin) / per) * per
  xh = xl + 2 * w
  if xh < pmin + per:
    return _firstGE(pts, lo, hi, xl), _firstGT(pts, lo, hi, xh), hi, hi
  return _firstGE(pts, lo, hi, xl), hi, lo, _firstGT(pts, lo, hi, xh - per)


# the Hamilton product of two quaternions given by their components
def _qm(a0, a1, a2, a3, b0, b1, b2, b3):
  return (a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3, a1 * b0 + a0 * b1 - a3 * b2 + a2 * b3,
          a2 * b0 + a3 * b1 + a0 * b2 - a1 * b3, a3 * b0 - a2 * b1 + a1 * b2 + a0 * b3)


# the grid points within eps of one image (alpha, beta, gamma), their cos(w/2) merged into best
def _imageNeighbours(al, be, ga, ringBeta, ringOff, ringAlpha, ringNode, ringMin, nodeCb, nodeSb, gOff, gPts, gMin,
                     gp, ap, nZL, eps, best, touched, t):
  cE, cH = np.cos(eps), np.cos(eps / 2)
  cb1, sb1 = np.cos(be), np.sin(be)
  tol = 1e-9
  nR = ringBeta.shape[0]
  for rg in range(_firstGE(ringBeta, 0, nR, be - eps - tol), nR):
    br = ringBeta[rg]
    if br > be + eps + tol:
      break
    k0, k1 = ringOff[rg], ringOff[rg + 1]
    # the alpha half width in which the images of z are within eps
    den = sb1 * np.sin(br)
    w = np.pi
    if den > 1e-12:
      ca = (cE - cb1 * np.cos(br)) / den
      if ca > 1 + tol:
        continue
      if ca > -1:
        w = np.arccos(min(ca, 1.0))
    a0, a1, b0, b1 = _window(ringAlpha, k0, k1, ringMin[rg], ap, al, w + tol)
    for side in range(2):
      lo, hi = (a0, a1) if side == 0 else (b0, b1)
      for kk in range(lo, hi):
        k = ringNode[kk]
        cbk, sbk = nodeCb[k], nodeSb[k]
        for m in range(nZL):
          dA = ringAlpha[kk] + m * ap - al
          cD, sD = np.cos(dA), np.sin(dA)
          # tr(R1' R2) = A + P cos(phi) + S sin(phi) with phi the gamma difference
          A = cb1 * cbk + sb1 * sbk * cD
          P = cD * (1 + cb1 * cbk) + sb1 * sbk
          S = -sD * (cb1 + cbk)
          B = np.sqrt(P * P + S * S)
          if A - 1 + B < 2 * cE - tol or B <= 1e-15:
            continue
          # tr = A + B cos(gamma' - x): the window about x, and the copy of gamma' by the
          # rotations about z on the right nearest to x, the closest one
          cg = (1 + 2 * cE - A) / B
          wg = np.pi if cg <= -1 else np.arccos(min(cg, 1.0))
          x = ga + np.arctan2(S, P)
          g0, g1, h0, h1 = _window(gPts, gOff[k], gOff[k + 1], gMin[k], gp, x, wg + tol)
          for side2 in range(2):
            lo2, hi2 = (g0, g1) if side2 == 0 else (h0, h1)
            for j in range(lo2, hi2):
              u = gPts[j] - x
              u -= np.round(u / gp) * gp
              d = np.sqrt(max(0.0, (A + B * np.cos(u) + 1) / 4))
              if d >= cH:
                if best[j] < 0:
                  touched[t] = j
                  t += 1
                if d > best[j]:
                  best[j] = min(d, 1.0)
  return t


# every query's grid points within eps, up to symmetry: the counts first (fill False), then
# the sorted columns and cos(w/2) into the rows indptr gives (fill True). A thread keeps its
# cos(w/2) per grid point and the points it touched in its own rows of bests and touches, set
# to -1 on its first chunk and left so by every query, so both passes share them
def _so3RadiusLoop(Q, Lr, Rr, ringBeta, ringOff, ringAlpha, ringNode, ringMin, nodeCb, nodeSb, gOff, gPts, gMin, gp, ap,
                   nZL, eps, chunk, fill, counts, indptr, cols, vals, bests, touches, ready):
  nq, nb = Q.shape[0], Q.shape[1]
  for c in prange((nq + chunk - 1) // chunk):
    # one row per thread of the launch, which so3Radius counts as the launch sets them
    tid = get_thread_id() % ready.shape[0]
    best, touched = bests[tid], touches[tid]
    if not ready[tid]:
      best[:] = -1.0
      ready[tid] = True
    for i in range(c * chunk, min(nq, (c + 1) * chunk)):
      t = 0
      for b in range(nb):
        for li in range(Lr.shape[0]):
          for ri in range(Rr.shape[0]):
            l0, l1, l2, l3 = _qm(Lr[li, 0], Lr[li, 1], Lr[li, 2], Lr[li, 3], Q[i, b, 0], Q[i, b, 1], Q[i, b, 2], Q[i, b, 3])
            p0, p1, p2, p3 = _qm(l0, l1, l2, l3, Rr[ri, 0], Rr[ri, 1], Rr[ri, 2], Rr[ri, 3])
            # the Matthies angles of the image: alpha + gamma, alpha - gamma and beta
            sp, dm = 2 * np.arctan2(p3, p0), 2 * np.arctan2(-p1, p2)
            be = 2 * np.arctan2(np.sqrt(p1 * p1 + p2 * p2), np.sqrt(p0 * p0 + p3 * p3))
            t = _imageNeighbours((sp + dm) / 2, be, (sp - dm) / 2, ringBeta, ringOff, ringAlpha, ringNode, ringMin, nodeCb,
                                 nodeSb, gOff, gPts, gMin, gp, ap, nZL, eps, best, touched, t)
      if fill:
        idx = np.sort(touched[:t])
        o = indptr[i]
        for u in range(t):
          cols[o + u] = idx[u]
          vals[o + u] = best[idx[u]]
      else:
        counts[i] = t
      for u in range(t):
        best[touched[u]] = -1.0


class flatSO3Grid(NamedTuple):
  """the ladders of an equispaced SO(3) grid as flat arrays, in the order the compiled search takes them"""
  theta: np.ndarray        # the polar angle per ring
  ringOff: np.ndarray      # the first node of every ring, and the end
  ringAlpha: np.ndarray    # the alpha of the nodes, folded and sorted within their ring
  ringNode: np.ndarray     # the node of every sorted alpha
  ringMin: np.ndarray      # the lower end of the alpha period per ring
  cb: np.ndarray           # cos and sin of the beta per node
  sb: np.ndarray
  offset: np.ndarray       # the first gamma of every node, and the end
  gammaPoints: np.ndarray
  gammaMin: np.ndarray
  gammaPeriod: float
  alphaPeriod: float


def flatSO3(g):
  """the ladders of an equispaced SO(3) grid as flat arrays: per ring its polar angle, its nodes
  with their alpha folded into [min, min + period) and sorted, per node its gamma ladder"""
  ab = g.alphabeta
  ringOff = np.ascontiguousarray(ab.offset, dtype=np.int64)
  ringMin = np.array([float(r.min) for r in ab.rhoGrid])
  ap = float(g.alphaPeriod)
  # the alpha of every node folded into the period of its ring, sorted ring by ring
  ring = np.repeat(np.arange(len(ringMin)), np.diff(ringOff))
  a = g.alpha - np.floor((g.alpha - ringMin[ring]) / ap) * ap
  ringNode = np.lexsort((a, ring)).astype(np.int64)
  ringAlpha = a[ringNode]
  return flatSO3Grid(np.ascontiguousarray(ab.thetaGrid.points, dtype=np.float64), ringOff, ringAlpha, ringNode, ringMin,
                     np.ascontiguousarray(g.cb), np.ascontiguousarray(g.sb), g.offset, g.gammaPoints, g.gammaMin, g.gammaPeriod, ap)


def so3Radius(flat, Q, Lr, Rr, nZL, eps, chunk=16):
  """the radius search of an equispaced SO(3) grid, Q the queries (nq, nb, 4) turned back by the
  grid's centre with nb = 2 holding the inverse under exchange, Lr and Rr the coset
  representatives, nZL the rotations about z on the left; the CSR arrays of the incidence"""
  _compileSO3()
  Q = np.ascontiguousarray(Q, dtype=np.float64)
  Lr, Rr = np.ascontiguousarray(Lr, dtype=np.float64), np.ascontiguousarray(Rr, dtype=np.float64)
  nq = Q.shape[0]
  args = (*flat[:-2], float(flat.gammaPeriod), float(flat.alphaPeriod))
  counts = np.zeros(nq, dtype=np.int64)
  indptr = np.zeros(nq + 1, dtype=np.int64)
  empty = np.zeros(0, dtype=np.int64), np.zeros(0)
  # the rows of the threads, uninitialised: a thread fills its own on first use
  nt, n = min(threads(), _jit().config.NUMBA_NUM_THREADS), len(flat.gammaPoints)
  with heapBlocks():
    rows = np.empty((nt, n)), np.empty((nt, n), dtype=np.int64), np.zeros(nt, dtype=np.bool_)
  loop = _COMPILED["so3Radius"]
  loop(Q, Lr, Rr, *args, np.int64(nZL), float(eps), np.int64(chunk), False, counts, indptr, *empty, *rows)
  indptr[1:] = np.cumsum(counts)
  cols, vals = np.empty(indptr[-1], dtype=np.int64), np.empty(indptr[-1])
  loop(Q, Lr, Rr, *args, np.int64(nZL), float(eps), np.int64(chunk), True, counts, indptr, cols, vals, *rows)
  return indptr, cols, vals


# ---------------------------------------------------------------- compilation

# the shared helpers njit-compiled once and swapped into the module namespace, so the two
# parallel loops that reference them by name compile in nopython mode
_HELPERS = ("_firstGE", "_nearestJ", "_ringBest", "_firstGT", "_window", "_qm", "_imageNeighbours")


def _compileHelpers():
  nb = _jit()
  globals()["prange"] = nb.prange
  globals()["get_thread_id"] = nb.get_thread_id
  for name in _HELPERS:
    if name not in _COMPILED:
      _COMPILED[name] = nb.njit(cache=True)(globals()[name])
      globals()[name] = _COMPILED[name]
  return nb


def _compileS2():
  if "s2Nearest" not in _COMPILED:
    _compileHelpers()
    _COMPILED["s2Nearest"] = _parallelKernel(_s2NearestLoop)


def _compileSO3():
  if "so3Radius" not in _COMPILED:
    _compileHelpers()
    _COMPILED["so3Radius"] = _parallelKernel(_so3RadiusLoop)


__all__ = []
