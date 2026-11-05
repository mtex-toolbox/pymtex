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
from .kernels import adjacencyNumba, chainSolveNumba, farthestOfChainsNumba, pixelKernel, segmentGridNumba
from .segment import segmentation

# the pinning rule under study: True holds, of a chain of free vertices with a segment that
# sweeps out of the fill or crosses, only the vertex farthest from the chord between its ends,
# as Douglas-Peucker, instead of both ends of every such segment; it changes the result and
# leaves the parity with MATLAB develop, so it stays off
PIN_FARTHEST = False


def straightenInFill(c, seg):
  """the complex with the boundary segments that enter the filled cells split there and every
  boundary vertex strictly inside them on the straight line between its neighbours, and the
  segmentation with the split faces"""
  inFill, tol, dxy, grainId = c.inFill, c.dxy / 100, c.dxy, seg.grainId
  if inFill is None:
    return c, seg
  # the boundary: the faces between two grains or against the outside, the segmentation's outer ones
  V, F, cells, isB = c.V.astype(float), c.F, c.cells, seg.isOuter.copy()

  # the fill test at every boundary vertex once
  FB = F[isB]
  used = np.zeros(len(V), dtype=bool)
  used[FB] = True
  dV = np.full(len(V), np.nan)
  dV[used] = inFill(V[used])
  # a boundary segment entering the filled cells gets a vertex there, by bisection
  d0, d1 = dV[FB[:, 0]], dV[FB[:, 1]]
  cut = np.flatnonzero(isB)[(np.maximum(d0, d1) > tol) & (np.minimum(d0, d1) < -tol)]
  a, b = V[F[cut, 0]], V[F[cut, 1]]
  lo, hi = np.zeros((len(cut), 1)), np.ones((len(cut), 1))
  aIn = dV[F[cut, 0]] > 0
  for _ in range(16):
    mid = (lo + hi) / 2
    isA = (inFill(a + mid * (b - a)) > 0) == aIn
    lo[isA], hi[~isA] = mid[isA], mid[~isA]
  nV = len(V)
  new = nV + np.arange(len(cut))
  V = np.vstack([V, a + (lo + hi) / 2 * (b - a)])
  dV = np.concatenate([dV, inFill(V[nV:])])
  F = np.vstack([F, np.column_stack([new, F[cut, 1]])])
  F[cut, 1] = new
  cells = np.vstack([cells, cells[cut]])
  isB = np.concatenate([isB, np.ones(len(cut), dtype=bool)])

  # the boundary vertices alone, numbered from 0 in e in the order of V
  used = np.concatenate([used, np.ones(len(cut), dtype=bool)])
  vB = np.flatnonzero(used)
  e = (np.cumsum(used, dtype=np.int64) - 1)[F[isB]]
  from scipy.sparse import coo_matrix
  from scipy.sparse.csgraph import connected_components
  from scipy.sparse.linalg import spsolve
  nB, m = len(vB), len(e)
  # the half edges by their first vertex: the segment and the neighbour of each, counting sorted
  ptr = np.concatenate([[0], np.cumsum(np.bincount(e.reshape(-1), minlength=nB))])
  hseg = adjacencyNumba(e, ptr)
  if hseg is None:
    hseg = np.repeat(np.arange(m), 2)[np.argsort(e.reshape(-1), kind="stable")]
  deg = np.diff(ptr)
  nbr = (e[:, 0] + e[:, 1])[hseg] - np.repeat(np.arange(nB), deg)
  W0 = V[vB]
  isFixed = dV[vB] <= tol
  # every round solves only the chains a vertex held in the last one belongs to, and samples and
  # tests the segments that moved
  W, mi, reset, lab, loc, grids, chains = W0.copy(), np.flatnonzero(~isFixed), np.zeros(0, dtype=np.int64), np.full(nB, -1), np.full(nB, -1), [], None
  # the solve keeps every vertex in the hull of the held ones, so the box of W0 holds every round
  box = W0.T.min(axis=1), W0.T.max(axis=1)
  while True:
    moved = [reset[(W[reset, 0] != W0[reset, 0]) | (W[reset, 1] != W0[reset, 1])]]
    W[reset] = W0[reset]
    if len(mi):
      # (D + 1e-3) W - A W = A W0 over the held neighbours + 1e-3 W0, the free vertices in their order
      n = len(mi)
      rows, j = np.repeat(np.arange(n), deg[mi]), nbr[_ranges(ptr, mi)]
      loc[:] = -1
      loc[mi] = np.arange(n)
      lj = loc[j]
      ff, fx = lj >= 0, isFixed[j]
      M = coo_matrix((np.concatenate([deg[mi] + 1e-3, -np.ones(ff.sum())]), (np.concatenate([np.arange(n), rows[ff]]),
                      np.concatenate([np.arange(n), lj[ff]]))), shape=(n, n)).tocsc()
      key, cnt = np.unique(rows[fx].astype(np.int64) * nB + j[fx], return_counts=True)
      r, col = key // nB, key % nB
      rhs = np.column_stack([np.bincount(r, weights=cnt * W0[col, k], minlength=n) for k in (0, 1)]) + 1e-3 * W0[mi]
      Wm = chainSolveNumba(M, rhs, chains=PIN_FARTHEST)
      if Wm is None:
        Wm = spsolve(M, rhs).reshape(-1, 2)
      elif PIN_FARTHEST:
        Wm, chains = Wm[0], Wm[1:]
      moved.append(mi[(Wm[:, 0] != W[mi, 0]) | (Wm[:, 1] != W[mi, 1])])
      W[mi] = Wm
      lab[mi] = connected_components(M, directed=False)[1] + lab.max() + 1
    moved = np.concatenate(moved)
    isM = np.zeros(m, dtype=bool)
    isM[hseg[_ranges(ptr, moved)]] = True
    # samples, a third of a pixel apart, of the area every segment sweeps that moved since the last
    # round, and the crossings of those that stay in the fill
    q = np.flatnonzero(isM)
    bad = np.zeros(m, dtype=bool)
    bad[q] = sweepsOut(W0, W, e[q], dxy, inFill, tol)
    isX, grids = crossing(W, e, q[~bad[q]], 2 * dxy, box, grids, q)
    bad |= isX
    if not bad.any():
      break
    v = e[bad].reshape(-1)
    if chains is not None:
      v = pinFarthest(e[bad], chains, loc, mi, W, W0)
    held = np.zeros(nB, dtype=bool)
    held[v[~isFixed[v]]] = True
    reset = np.flatnonzero(held)
    isFixed[v] = True
    hit = np.zeros(lab.max() + 2, dtype=bool)
    hit[lab[v] + 1] = True
    mi = np.flatnonzero(hit[lab + 1] & ~isFixed)
  V0, V = V, V.copy()
  V[vB] = W
  split = lambda x: np.concatenate([x, x[cut]])
  return (cellComplex(V.astype(c.V.dtype, copy=False), F.astype(c.F.dtype, copy=False), cells, c.sites, c.A, c.plane, c.shape, V0.astype(c.V.dtype, copy=False)),
          segmentation(grainId, seg.nGrains, split(seg.connect), split(seg.isInner), split(seg.isOuter)))


def pinFarthest(eb, chains, loc, mi, W, W0):
  """the vertices to hold for the bad segments eb under PIN_FARTHEST: of every chain one of them
  touches, the vertex whose old position lies farthest from the chord through the solved ends,
  and both ends of a segment that touches no chain"""
  chain, prv, first, last = chains
  l = loc[eb]
  c = np.where(l >= 0, chain[np.maximum(l, 0)], -1)
  cs = np.unique(c[c >= 0])
  v = farthestOfChainsNumba(cs, first, last, prv, mi, W, W0)
  return np.concatenate([v, eb[np.all(c < 0, axis=1)].reshape(-1)])


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


def crossing(V, e, q, h, box, grids=(), moved=None):
  """the segments q of e that cross another one of e properly, without a common vertex, and the
  grids of spacing h over box the compiled test lists the segments in: the ones given, the first
  of every segment where it was when it was built, the others of the segments moved in a round,
  where they were then, and one more of the segments moved now, each over the cells the segments
  q meet. Without numba the pairs whose midpoints lie in neighbouring cells of a grid as large as
  the longest segment"""
  isX = np.zeros(len(e), dtype=bool)
  loop = pixelKernel("crossing")
  if loop is not None:
    (x0, y0), (nx, ny) = box[0], (np.floor((box[1] - box[0]) / h).astype(np.int64) + 1)
    grids = [*grids, segmentGridNumba(V, e, moved if grids else np.arange(len(e)), q, x0, y0, h, int(nx), int(ny))]
    ioff = np.concatenate([[0], np.cumsum([len(g[1]) for g in grids])])
    if len(q):
      loop(np.ascontiguousarray(V), np.ascontiguousarray(e, dtype=np.int64), q, x0, y0, h, int(ny), np.vstack([g[0] for g in grids]),
           np.concatenate([g[1] for g in grids]), ioff, isX)
    return isX, grids
  if len(q) == 0:
    return isX, grids
  a, b = V[e[:, 0]], V[e[:, 1]]
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
  return isX, grids


__all__ = []
