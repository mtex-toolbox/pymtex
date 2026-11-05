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
from .kernels import pixelKernel, segmentGridNumba
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

  # the boundary vertices alone, numbered from 0 in e
  vB, e = np.unique(F[isB], return_inverse=True)
  e = e.reshape(-1, 2).astype(np.int64)
  from scipy.sparse import coo_matrix
  from scipy.sparse.csgraph import connected_components
  from scipy.sparse.linalg import spsolve
  nB = len(vB)
  deg = np.bincount(e.reshape(-1), minlength=nB).astype(float)
  I, J = np.concatenate([e[:, 0], e[:, 1]]), np.concatenate([e[:, 1], e[:, 0]])
  W0 = V[vB]
  isFixed = inFill(W0) <= tol
  # every round solves only the chains a vertex held in the last one belongs to
  solve, lab, W = ~isFixed, np.full(nB, -1), W0
  while True:
    Wlast, W = W, W.copy()
    W[isFixed] = W0[isFixed]
    mi = np.flatnonzero(solve & ~isFixed)
    if len(mi):
      # (D + 1e-3) W - A W = A W0 over the held neighbours + 1e-3 W0, the free vertices in their order
      loc = np.full(nB, -1)
      loc[mi] = np.arange(len(mi))
      li, lj = loc[I], loc[J]
      ff = (li >= 0) & (lj >= 0)
      n = len(mi)
      M = coo_matrix((np.concatenate([deg[mi] + 1e-3, -np.ones(ff.sum())]), (np.concatenate([np.arange(n), li[ff]]),
                      np.concatenate([np.arange(n), lj[ff]]))), shape=(n, n)).tocsc()
      fx = (li >= 0) & isFixed[J]
      key, cnt = np.unique(li[fx].astype(np.int64) * nB + J[fx], return_counts=True)
      r, col = key // nB, key % nB
      rhs = np.column_stack([np.bincount(r, weights=cnt * W0[col, k], minlength=n) for k in (0, 1)]) + 1e-3 * W0[mi]
      W[mi] = spsolve(M, rhs).reshape(-1, 2)
      lab[mi] = connected_components(M, directed=False)[1] + lab.max() + 1

    # samples, a third of a pixel apart, of the area every segment sweeps that moved since the last round
    chg = np.any(W != Wlast, axis=1)
    isM = chg[e[:, 0]] | chg[e[:, 1]]
    bad = crossing(W, e, isM, 4 * dxy)
    bad[isM] |= sweepsOut(W0, W, e[isM], dxy, inFill, tol)
    if not bad.any():
      break
    v = e[bad].reshape(-1)
    solve = np.isin(lab, lab[v])
    isFixed[v] = True
  V0, V = V, V.copy()
  V[vB] = W
  ix = np.concatenate([np.arange(len(c.F)), cut])
  return (cellComplex(V.astype(c.V.dtype), F.astype(c.F.dtype), cells, c.sites, c.A, c.plane, c.shape, V0.astype(c.V.dtype)),
          segmentation(grainId, seg.nGrains, seg.connect[ix], seg.isInner[ix], seg.isOuter[ix]))


def sweepsOut(V0, V, em, dxy, inFill, tol):
  """whether the area each segment em sweeps from V0 to V leaves the fill by more than tol, at
  samples a third of a pixel apart along and across it; compiled on a rigid lattice, sampled in
  NumPy otherwise and as the oracle"""
  out = np.zeros(len(em), dtype=bool)
  loop = pixelKernel("sweep") if inFill.rigid else None
  if loop is not None:
    loop(np.ascontiguousarray(em), np.ascontiguousarray(V0), np.ascontiguousarray(V), dxy, *inFill.lattice, tol, out)
    return out
  T = np.maximum(1, np.ceil(3 * np.maximum(np.linalg.norm(V0[em[:, 1]] - V0[em[:, 0]], axis=1),
                                           np.linalg.norm(V[em[:, 1]] - V[em[:, 0]], axis=1)) / dxy)).astype(np.int64)
  K = np.maximum(1, np.ceil(3 * np.linalg.norm(V - V0, axis=1)[em].max(axis=1, initial=0) / dxy)).astype(np.int64)
  n = (T + 1) * (K + 1)
  sid = np.repeat(np.arange(len(em)), n)
  r = np.arange(len(sid)) - np.repeat(np.cumsum(n) - n, n)
  t = (r % (T[sid] + 1) / T[sid])[:, None]
  l = (r // (T[sid] + 1) / K[sid])[:, None]
  P = (1 - l) * ((1 - t) * V0[em[sid, 0]] + t * V0[em[sid, 1]]) + l * ((1 - t) * V[em[sid, 0]] + t * V[em[sid, 1]])
  return np.bincount(sid, weights=inFill(P) < -tol, minlength=len(em)) > 0


def crossing(V, e, isM, h):
  """the segments of e with isM that cross another one of e properly, without a common vertex:
  the pairs whose bounding boxes share a cell of a grid of spacing h, compiled, or those whose
  midpoints lie in neighbouring cells of a grid as large as the longest segment"""
  isX = np.zeros(len(e), dtype=bool)
  q = np.flatnonzero(isM)
  if len(q) == 0:
    return isX
  a, b = np.ascontiguousarray(V[e[:, 0]]), np.ascontiguousarray(V[e[:, 1]])
  grid = segmentGridNumba(a, b, q, h)
  if grid is not None:
    pixelKernel("crossing")(a, b, np.ascontiguousarray(e, dtype=np.int64), q, grid[0], grid[1], h, *grid[2:], isX)
    return isX
  d = b - a
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
