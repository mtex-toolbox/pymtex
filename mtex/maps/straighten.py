# grain boundaries drawn straight through the cells the alpha closing filled (MATLAB
# EBSD/private/straightenInFill)
#
#   c, seg = straightenInFill(c, seg)    the complex with its boundary vertices in the fill moved
#
# The closing leaves the filled cells without sites, so the Voronoi decomposition shares them
# between the measured pixels around them along bisectors, which bends a boundary around the
# middle of a crack or a cluster of notIndexed pixels. A boundary vertex inside the filled cells
# moves onto the straight line between its neighbours, a harmonic solve with the other vertices
# held, while a vertex on the edge of a measured, an unfilled or an exterior cell stays. A segment
# that would sweep over anything but filled cells, or cross another boundary, keeps its vertices,
# and the solve is repeated, so every measured pixel stays in its grain.
from __future__ import annotations

import numpy as np

from .decomposition import cellComplex
from .segment import segmentation


def straightenInFill(c, seg):
  """the complex with the boundary segments that enter the filled cells split there and every
  boundary vertex strictly inside them on the straight line between its neighbours, and the
  segmentation with the split faces"""
  inFill, tol, dxy, grainId = c.inFill, c.dxy / 100, c.dxy, seg.grainId
  if inFill is None:
    return c, seg
  V, F, cells = c.V.astype(float), c.F, c.cells
  px = c.sites.pixel[cells]
  g = np.where(px >= 0, grainId[np.maximum(px, 0)], 0)
  nC = (px >= 0).sum(axis=1)
  isB = (nC == 1) | ((nC == 2) & (g[:, 0] != g[:, 1]))

  # a boundary segment entering the filled cells gets a vertex there, by bisection
  dE = inFill(V[F[isB]].reshape(-1, 2)).reshape(-1, 2)
  cut = np.flatnonzero(isB)[(dE.max(axis=1) > tol) & (dE.min(axis=1) < -tol)]
  a, b = V[F[cut, 0]], V[F[cut, 1]]
  lo, hi = np.zeros((len(cut), 1)), np.ones((len(cut), 1))
  aIn = inFill(a) > 0
  for _ in range(16):
    mid = (lo + hi) / 2
    isA = (inFill(a + mid * (b - a)) > 0) == aIn
    lo[isA], hi[~isA] = mid[isA], mid[~isA]
  nV = len(V)
  new = nV + np.arange(len(cut))
  V = np.vstack([V, a + (lo + hi) / 2 * (b - a)])
  F = np.vstack([F, np.column_stack([new, F[cut, 1]])])
  F[cut, 1] = new
  cells = np.vstack([cells, cells[cut]])
  isB = np.concatenate([isB, np.ones(len(cut), dtype=bool)])

  e = F[isB].astype(np.int64)
  from scipy.sparse import coo_matrix
  from scipy.sparse.linalg import spsolve
  A = coo_matrix((np.ones(2 * len(e)), (e.reshape(-1), e[:, ::-1].reshape(-1))), shape=(len(V), len(V))).tocsr()
  deg = np.asarray(A.sum(axis=1)).reshape(-1)
  isFixed = np.ones(len(V), dtype=bool)
  vB = np.unique(e)
  isFixed[vB] = inFill(V[vB]) <= tol
  V0, Vlast = V, V
  while True:
    V = V0.copy()
    m = ~isFixed & (deg > 0)
    mi, fi = np.flatnonzero(m), np.flatnonzero(~m)
    Amm = A[mi][:, mi]
    L = (coo_matrix((deg[mi] + 1e-3, (np.arange(len(mi)), np.arange(len(mi)))), shape=Amm.shape) - Amm).tocsc()
    V[mi] = spsolve(L, A[mi][:, fi] @ V0[fi] + 1e-3 * V0[mi]).reshape(-1, 2) if len(mi) else V[mi]

    # samples, a third of a pixel apart, of the area every segment sweeps that moved since the last round
    chg = np.any(V != Vlast, axis=1)
    isM = chg[e].any(axis=1)
    em = e[isM]
    dV = np.linalg.norm(V - V0, axis=1)
    T = np.maximum(1, np.ceil(3 * np.maximum(np.linalg.norm(V0[em[:, 1]] - V0[em[:, 0]], axis=1),
                                             np.linalg.norm(V[em[:, 1]] - V[em[:, 0]], axis=1)) / dxy)).astype(np.int64)
    K = np.maximum(1, np.ceil(3 * dV[em].max(axis=1, initial=0) / dxy)).astype(np.int64)
    n = (T + 1) * (K + 1)
    sid = np.repeat(np.arange(len(em)), n)
    r = np.arange(len(sid)) - np.repeat(np.cumsum(n) - n, n)
    t = (r % (T[sid] + 1) / T[sid])[:, None]
    l = (r // (T[sid] + 1) / K[sid])[:, None]
    P = (1 - l) * ((1 - t) * V0[em[sid, 0]] + t * V0[em[sid, 1]]) + l * ((1 - t) * V[em[sid, 0]] + t * V[em[sid, 1]])
    bad = crossing(V, e, isM)
    bad[isM] |= np.bincount(sid, weights=inFill(P) < -tol, minlength=len(em)) > 0
    if not bad.any():
      break
    isFixed[e[bad].reshape(-1)] = True
    Vlast = V
  ix = np.concatenate([np.arange(len(c.F)), cut])
  return (cellComplex(V.astype(c.V.dtype), F.astype(c.F.dtype), cells, c.sites, c.A, c.plane, c.shape, V0.astype(c.V.dtype)),
          segmentation(grainId, seg.nGrains, seg.connect[ix], seg.isInner[ix], seg.isOuter[ix]))


def crossing(V, e, isM):
  """the segments of e with isM that cross another one of e properly, without a common vertex"""
  isX = np.zeros(len(e), dtype=bool)
  q = np.flatnonzero(isM)
  if len(q) == 0:
    return isX
  # candidates from the 3 x 3 buckets, as large as the longest segment, around the midpoint
  a = V[e[:, 0]]
  d = V[e[:, 1]] - a
  c = np.floor((a + d / 2) / np.linalg.norm(d, axis=1).max()).astype(np.int64)
  c -= c.min(axis=0) - 1
  nx = int(c[:, 0].max()) + 2
  key = c[:, 0] + c[:, 1] * nx
  ord_ = np.argsort(key, kind="stable")
  ks = key[ord_]
  I, J = [], []
  for o1 in (-1, 0, 1):
    for o2 in (-1, 0, 1):
      k = key[q] + o1 + o2 * nx
      s, t = np.searchsorted(ks, k, "left"), np.searchsorted(ks, k, "right")
      cnt = t - s
      I.append(np.repeat(q, cnt))
      J.append(ord_[np.repeat(s - np.cumsum(cnt) + cnt, cnt) + np.arange(cnt.sum())])
  i, j = np.concatenate(I), np.concatenate(J)
  w = a[j] - a[i]
  with np.errstate(divide="ignore", invalid="ignore"):
    den = d[i, 0] * d[j, 1] - d[i, 1] * d[j, 0]
    s = (w[:, 0] * d[j, 1] - w[:, 1] * d[j, 0]) / den
    t = (w[:, 0] * d[i, 1] - w[:, 1] * d[i, 0]) / den
  x = (s > 0) & (s < 1) & (t > 0) & (t < 1) & ~np.any((e[i] == e[j, 0][:, None]) | (e[i] == e[j, 1][:, None]), axis=1)
  isX[i[x]] = True
  return isX


__all__ = []
