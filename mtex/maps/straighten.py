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
  # the boundary: the faces between two grains or against the outside, the segmentation's outer ones
  V, F, cells, isB = c.V.astype(float), c.F, c.cells, seg.isOuter.copy()

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
  nB, m = len(vB), len(e)
  # the half edges by their first vertex: the neighbour and the segment of each
  I = np.concatenate([e[:, 0], e[:, 1]])
  order = np.argsort(I, kind="stable")
  ptr = np.concatenate([[0], np.cumsum(np.bincount(I, minlength=nB))])
  nbr, hseg = np.concatenate([e[:, 1], e[:, 0]])[order], np.concatenate([np.arange(m), np.arange(m)])[order]
  deg = np.diff(ptr)
  W0 = V[vB]
  isFixed = inFill(W0) <= tol
  # every round solves only the chains a vertex held in the last one belongs to, and samples and
  # tests the segments that moved
  W, mi, reset, lab, loc, grid, since = W0.copy(), np.flatnonzero(~isFixed), np.zeros(0, dtype=np.int64), np.full(nB, -1), np.full(nB, -1), None, []
  while True:
    moved = [reset[np.any(W[reset] != W0[reset], axis=1)]]
    W[reset] = W0[reset]
    if len(mi):
      # (D + 1e-3) W - A W = A W0 over the held neighbours + 1e-3 W0, the free vertices in their order
      n = len(mi)
      rows, j = np.repeat(np.arange(n), deg[mi]), nbr[_ranges(ptr, mi)]
      loc[mi] = np.arange(n)
      lj = loc[j]
      loc[mi] = -1
      ff, fx = lj >= 0, isFixed[j]
      M = coo_matrix((np.concatenate([deg[mi] + 1e-3, -np.ones(ff.sum())]), (np.concatenate([np.arange(n), rows[ff]]),
                      np.concatenate([np.arange(n), lj[ff]]))), shape=(n, n)).tocsc()
      key, cnt = np.unique(rows[fx].astype(np.int64) * nB + j[fx], return_counts=True)
      r, col = key // nB, key % nB
      rhs = np.column_stack([np.bincount(r, weights=cnt * W0[col, k], minlength=n) for k in (0, 1)]) + 1e-3 * W0[mi]
      Wm = spsolve(M, rhs).reshape(-1, 2)
      moved.append(mi[np.any(Wm != W[mi], axis=1)])
      W[mi] = Wm
      lab[mi] = connected_components(M, directed=False)[1] + lab.max() + 1
    moved = np.concatenate(moved)
    isM = np.zeros(m, dtype=bool)
    isM[hseg[_ranges(ptr, moved)]] = True
    if grid is not None:
      since.append(moved)
    bad, grid = crossing(W, e, isM, 4 * dxy, grid, np.unique(hseg[_ranges(ptr, np.concatenate(since))]) if since else None)
    # samples, a third of a pixel apart, of the area every segment sweeps that moved since the last round
    bad[isM] |= sweepsOut(W0, W, e[isM], dxy, inFill, tol)
    if not bad.any():
      break
    v = e[bad].reshape(-1)
    reset = np.unique(v[~isFixed[v]])
    isFixed[v] = True
    mi = np.flatnonzero(np.isin(lab, lab[v]) & ~isFixed)
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


# the concatenated index ranges ptr[v] .. ptr[v + 1] - 1 of the vertices v
def _ranges(ptr, v):
  n = ptr[v + 1] - ptr[v]
  return np.repeat(ptr[v] - np.cumsum(n) + n, n) + np.arange(n.sum())


def crossing(V, e, isM, h, grid=None, since=None):
  """the segments of e with isM that cross another one of e properly, without a common vertex,
  and the grid of spacing h the compiled test lists the segments in: the one given, built where
  the segments were, with those moved since, the segments since, in a second one, or a new one.
  Without numba the pairs whose midpoints lie in neighbouring cells of a grid as large as the
  longest segment"""
  isX = np.zeros(len(e), dtype=bool)
  q = np.flatnonzero(isM)
  ends = lambda k: (np.ascontiguousarray(V[e[k, 0]]), np.ascontiguousarray(V[e[k, 1]]))
  loop = pixelKernel("crossing")
  if loop is not None:
    if grid is None:
      grid = segmentGridNumba(*ends(slice(None)), np.arange(len(e)), h)
    two = (grid[0], grid[1], 1, np.zeros(2, dtype=np.int64), np.zeros(0, dtype=np.int64))
    if since is not None and len(since):
      x1, y1, ny1, start1, items1 = segmentGridNumba(*ends(since), np.arange(len(since)), h)
      two = (x1, y1, ny1, start1, since[items1])
    if len(q):
      loop(np.ascontiguousarray(V), np.ascontiguousarray(e, dtype=np.int64), q, grid[0], grid[1], h, *grid[2:], *two, isX)
    return isX, grid
  if len(q) == 0:
    return isX, None
  a, b = ends(slice(None))
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
  return isX, None


__all__ = []
