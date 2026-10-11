# the boundary resampled: coarsened, simplified and refined along its chains, alone and under
# the grains (MATLAB @grainBoundary/reduce, simplify, refine and @grain2d/reduceBoundary,
# simplifyBoundary, refineBoundary; Julia boundaryresampling.jl)
#
#   gB.reduce(n), gB.simplify(epsilon), gB.refine(delta)
#   grains.reduceBoundary(n), grains.simplifyBoundary(epsilon), grains.refineBoundary(delta)
#
# Coarsening is one merge step with two policies: mustKeep states which heads may never be
# dissolved, a junction, a vertex where the pair of grains changes, a vertex another boundary
# walks over; keeping every n-th vertex and Douglas-Peucker decide which of the others
# survive; mergeRuns lets every survivor swallow the run of segments before it. Refining
# resamples every piece between two protected vertices at a constant arc length. A closed
# chain has no junction: the vertex the walk cuts it at is an accident of the numbering, so
# the chain is anchored at a vertex that has to stay and cut there instead. A grain list
# coarsens both of its boundaries at once, over one pool of vertices, and traces its loops
# again.
from __future__ import annotations

import numpy as np

from ..geometry.kernels import _jit, backend
from .boundary import grainBoundary, orderWalk
from .decomposition import NO_PIXEL
from .grains import grain2d
from ..geometry.install import install

_COMPILED = {}


# ------------------------------------------------------------------ the heads that survive

def mustKeep(gB, protect=None):
  """the heads a coarsening may not dissolve: the end of an open chain is a junction and stays
  where it is, a vertex where the pair of grains changes is a corner of both grain polygons
  even where only two segments meet there, and a protected vertex is one another boundary
  over the same vertices still walks over"""
  closed = gB.isClosed
  keep = gB.isChainEnd & ~closed
  pair = np.sort(gB.grainId, axis=1)
  keep[:-1] |= np.any(pair[1:] != pair[:-1], axis=1)
  # a closed chain wraps: its last segment neighbours its first
  starts, ends = np.flatnonzero(gB.isChainStart), np.flatnonzero(gB.isChainEnd)
  wrap = closed[starts]
  keep[ends[wrap]] |= np.any(pair[ends[wrap]] != pair[starts[wrap]], axis=1)
  if protect is not None and len(protect):
    p = np.zeros(len(gB.V), dtype=bool)
    p[np.asarray(protect, dtype=np.int64)] = True
    keep |= p[gB.F[:, 1]]
  return keep


def anchorClosedChains(gB, keep):
  """every closed chain turned so that it ends at a head that stays: its rows are rotated to
  the first kept one, and a chain with nothing kept is cut where the walk cut it. After this
  every chain ends with a kept row, so a run of dissolved segments never crosses a chain"""
  keep = keep.copy()
  closed = gB.isClosed
  starts, ends = np.flatnonzero(gB.isChainStart), np.flatnonzero(gB.isChainEnd)
  order = np.arange(len(gB))
  for s, e in zip(starts[closed[starts]], ends[closed[starts]]):
    if keep[e]:
      continue
    hit = np.flatnonzero(keep[s:e])
    if len(hit) == 0:
      keep[e] = True
    else:
      k = s + hit[0]
      order[s:e + 1] = np.concatenate([np.arange(k + 1, e + 1), np.arange(s, k + 1)])
  if np.array_equal(order, np.arange(len(gB))):
    return gB, keep
  return gB[order], keep[order]


def keepClosedChains(gB, keep):
  """a closed chain has no junction to anchor it, so it is given three vertices"""
  closed = gB.isClosed
  for s, e in zip(np.flatnonzero(gB.isChainStart), np.flatnonzero(gB.isChainEnd)):
    n = e - s + 1
    if not closed[s] or n < 3 or np.count_nonzero(keep[s:e + 1]) >= 3:
      continue
    for j in np.unique(np.rint(np.linspace(n / 3, n, 3)).astype(int)):
      keep[s + j - 1] = True
  return keep


def keepGrainPolygons(gB, keep, ids):
  """no grain of the list is coarsened out of existence: a polygon needs three vertices, and a
  grain smaller than the tolerance loses all of them, so such a grain is left as it was.
  Which grains have a polygon is the caller's to say; on a bare boundary a grain appears
  with whatever part of its boundary was selected and none is restored"""
  nG = int(gB.grainId.max(initial=0))
  ids = np.asarray(ids, dtype=np.int64).reshape(-1) if ids is not None else np.zeros(0, dtype=np.int64)
  if nG == 0 or len(ids) == 0:
    return keep
  edge = (gB.grainId > 0) & (gB.grainId[:, 0] != gB.grainId[:, 1])[:, None]
  g = gB.grainId[edge]
  total = np.bincount(g, minlength=nG + 1)
  kept = np.bincount(g, weights=np.broadcast_to(keep[:, None], edge.shape)[edge], minlength=nG + 1)
  wanted = np.zeros(nG + 1, dtype=bool)
  wanted[ids[(ids >= 1) & (ids <= nG)]] = True
  # a grain that had fewer than three segments to begin with cannot be helped
  short = wanted & (kept < 3) & (total >= 3)
  if not short.any():
    return keep
  keep |= np.any(edge & short[gB.grainId], axis=1)
  return keep


def mergeRuns(gB, keep):
  """every surviving head swallows the run of segments before it, so the boundary stays
  connected and keeps its length up to the straightening of the dissolved corners. The per
  segment data is inherited from the last segment of each run; a segment that swallowed a
  run runs between no one pair of pixels any more and says so"""
  if keep.all():
    return gB
  gB, keep = anchorClosedChains(gB, keep)
  surv = np.flatnonzero(keep)
  out = gB[surv]
  runStart = np.concatenate([[0], surv[:-1] + 1])
  F = np.stack([gB.F[runStart, 0], gB.F[surv, 1]], axis=1)
  eid = out.ebsdId.copy()
  eid[runStart != surv] = NO_PIXEL
  return orderWalk(out._with(F=F, ebsdId=eid))


def _finish(gB, keep, keepGrains):
  return mergeRuns(gB, keepGrainPolygons(gB, keepClosedChains(gB, keep), keepGrains))


# ------------------------------------------------------------------ the two policies

def reduceBoundary(gB, n=2, protect=None, keepGrains=None):
  """keep every n-th vertex along each chain, whatever the shape: it thins a straight run and
  a corner alike, so prefer simplify, which spends its vertices where the shape
  needs them"""
  if gB.isempty or n <= 1:
    return gB
  starts = np.flatnonzero(gB.isChainStart)
  pos = np.arange(len(gB)) - starts[gB.chainId] + 1
  keep = mustKeep(gB, protect) | (pos % n == 0)
  return _finish(gB, keep, keepGrains)


def simplifyBoundary(gB, epsilon=None, protect=None, keepGrains=None):
  """drop the vertices that carry no shape, chain by chain: a vertex goes when leaving it out
  moves the boundary by less than epsilon, which on a pixel grid turns a staircase into the
  straight line it approximates. The default tolerance is exactly that: a staircase of
  spacing d never strays further than d / sqrt(2) from the line it approximates"""
  if gB.isempty:
    return gB
  if epsilon is None:
    epsilon = defaultEpsilon(gB)
  if epsilon <= 0:
    return gB
  keep = keepShapeChains(gB, float(epsilon), mustKeep(gB, protect))
  return _finish(gB, keep, keepGrains)


def defaultEpsilon(gB):
  """the default tolerance of simplifyBoundary: the median segment length over sqrt(2), the
  furthest a staircase of that spacing strays from its line"""
  return 0.0 if gB.isempty else float(np.median(gB.segLength)) / np.sqrt(2)


def keepShapeChains(gB, epsilon, keep=None):
  """the heads Douglas-Peucker keeps on every chain, on top of the ones given: the vertices of
  a chain are the tails of its segments and the head of the last one, and keeping vertex j
  means keeping the head of segment j - 1"""
  starts, ends = np.flatnonzero(gB.isChainStart), np.flatnonzero(gB.isChainEnd)
  offset = np.concatenate([[0], np.cumsum(ends - starts + 2)]).astype(np.int64)
  # the flat vertex list of all chains: the tail of every first segment, then every head
  isHead = np.ones(offset[-1], dtype=bool)
  isHead[offset[:-1]] = False
  chainV = np.zeros(offset[-1], dtype=np.int32)
  chainV[~isHead] = gB.F[starts, 0]
  chainV[isHead] = gB.F[:, 1]
  closed = gB.F[ends, 1] == gB.F[starts, 0]
  keepV = np.zeros(offset[-1], dtype=np.bool_)
  if keep is not None:
    keepV[isHead] = keep
  _compiledKeepShape()(np.ascontiguousarray(gB.V, dtype=np.float64), chainV, offset, closed, epsilon, keepV)
  return keepV[isHead]


# Douglas-Peucker on every chain, with an explicit stack of the pieces still to examine:
# the two ends of a chain are kept, and a piece whose interior strays further than epsilon
# from its chord is cut at the vertex furthest from it. A closed chain is walked from its
# first kept vertex, or from where the walk cut it, round to the same vertex: the chord of
# that piece has no length, so the recursion starts at the vertex furthest from the anchor,
# and a loop left with fewer than three vertices is given them back
def _keepShapeLoop(V, chainV, offset, closed, epsilon, keepV):
  for c in range(offset.shape[0] - 1):
    a, b = offset[c], offset[c + 1] - 1
    n = b - a + 1
    ring = closed[c] and n > 3
    # the positions of the chain, a closed one rotated to its anchor
    idx = np.empty(n, dtype=np.int64)
    p = a
    if ring:
      keepV[a] = keepV[a] or keepV[b]
      for i in range(a, b):
        if keepV[i]:
          p = i
          break
      for i in range(n):
        idx[i] = a + (p - a + i) % (n - 1)
    else:
      for i in range(n):
        idx[i] = a + i
    keepV[idx[0]] = True
    keepV[idx[n - 1]] = True
    stack = np.empty((n + 2, 2), dtype=np.int64)
    stack[0, 0], stack[0, 1] = 0, n - 1
    top = 1
    while top > 0:
      top -= 1
      lo, hi = stack[top, 0], stack[top, 1]
      if hi - lo < 2:
        continue
      ax, ay = V[chainV[idx[lo]], 0], V[chainV[idx[lo]], 1]
      cx, cy = V[chainV[idx[hi]], 0] - ax, V[chainV[idx[hi]], 1] - ay
      length = np.sqrt(cx * cx + cy * cy)
      dmax, imax = 0.0, -1
      # the distance of every interior point from the chord, or from the point itself
      # where the chord has no length
      for i in range(lo + 1, hi):
        rx, ry = V[chainV[idx[i]], 0] - ax, V[chainV[idx[i]], 1] - ay
        d = np.sqrt(rx * rx + ry * ry) if length == 0 else abs(rx * cy - ry * cx) / length
        if d > dmax:
          dmax, imax = d, i
      if dmax > epsilon:
        keepV[idx[imax]] = True
        stack[top, 0], stack[top, 1] = lo, imax
        stack[top + 1, 0], stack[top + 1, 1] = imax, hi
        top += 2
    if not ring:
      continue
    # a loop needs three vertices: the farthest from the one kept, then the farthest off
    # the chord between the two
    while True:
      count, q, r = 0, -1, -1
      for i in range(n - 1):
        if keepV[idx[i]]:
          count += 1
          if q < 0:
            q = i
          elif r < 0:
            r = i
      if count >= 3:
        break
      qx, qy = V[chainV[idx[q]], 0], V[chainV[idx[q]], 1]
      cx, cy = 0.0, 0.0
      if r >= 0:
        cx, cy = V[chainV[idx[r]], 0] - qx, V[chainV[idx[r]], 1] - qy
      best, dbest = -1, 0.0
      for i in range(n - 1):
        if keepV[idx[i]]:
          continue
        rx, ry = V[chainV[idx[i]], 0] - qx, V[chainV[idx[i]], 1] - qy
        d = np.sqrt(rx * rx + ry * ry) if r < 0 else abs(rx * cy - ry * cx)
        if d > dbest:
          best, dbest = i, d
      if best < 0:
        break
      keepV[idx[best]] = True
    keepV[b] = keepV[b] or keepV[a]


def _compiledKeepShape():
  if backend() == "numpy":
    return _keepShapeLoop
  if "keepShape" not in _COMPILED:
    _COMPILED["keepShape"] = _jit().njit(cache=True)(_keepShapeLoop)
  return _COMPILED["keepShape"]


# ------------------------------------------------------------------ refining

def refineBoundary(gB, delta=None, protect=None):
  """resample every chain at a constant spacing along its length: the new vertices lie on the
  old boundary, so its shape does not change, only how it is sampled. The vertices that may
  not move stay exactly where they are and cut the chain into pieces that are resampled on
  their own, which is what lets a smoothing that follows work on degrees of freedom that
  are not tied to the grid. grainId and phaseId are constant along a piece and carry over;
  the pair of measurements and the misorientation are inherited from the original segment
  covering the midpoint of the new one, and the pair is dropped where the new segment spans
  more than one of them"""
  if gB.isempty:
    return gB
  if delta is None:
    delta = defaultDelta(gB)
  if delta <= 0:
    return gB
  gB, keep = anchorClosedChains(gB, mustKeep(gB, protect))
  T = gB.V.dtype
  L = gB.segLength
  Lc = np.concatenate([[0.0], np.cumsum(L)])
  iEnd = np.flatnonzero(keep)
  iStart = np.concatenate([[0], iEnd[:-1] + 1])
  nCh = len(iStart)
  # the pieces and how many samples each of them takes
  pieceOf = np.repeat(np.arange(nCh), iEnd - iStart + 1)
  total = np.bincount(pieceOf, weights=L, minlength=nCh)
  m = np.maximum(1, np.rint(total / delta)).astype(np.int64)
  M = int(m.sum())
  cOf = np.repeat(np.arange(nCh), m)
  kOf = np.arange(1, M + 1) - np.repeat(np.concatenate([[0], np.cumsum(m[:-1])]), m)
  tPrev = total[cOf] * (kOf - 1) / m[cOf]
  tEnd = total[cOf] * kOf / m[cOf]
  gStart = Lc[iStart][cOf]
  eps = total[cOf] * 1e-9
  def locate(g):
    return _locate(g, Lc, iStart[cOf], iEnd[cOf])
  src = locate(gStart + (tPrev + tEnd) / 2)[0]
  within = (locate(gStart + tPrev + eps)[0] == src) & (locate(gStart + tEnd - eps)[0] == src)
  # every new segment contributes its head, except the last one of a piece
  isLast = kOf == m[cOf]
  iSeg, lam = _locate(gStart[~isLast] + tEnd[~isLast], Lc, iStart[cOf[~isLast]], iEnd[cOf[~isLast]])
  P1, P2 = gB.V[gB.F[iSeg, 0]], gB.V[gB.F[iSeg, 1]]
  V = np.concatenate([gB.V, (P1 + lam[:, None] * (P2 - P1)).astype(T)])
  head = np.zeros(M, dtype=np.int32)
  head[~isLast] = len(gB.V) + np.arange(np.count_nonzero(~isLast))
  head[isLast] = gB.F[iEnd[cOf[isLast]], 1]
  tail = np.zeros(M, dtype=np.int32)
  first = kOf == 1
  tail[first] = gB.F[iStart[cOf[first]], 0]
  tail[~first] = head[np.flatnonzero(~first) - 1]
  out = gB[src]
  eid = out.ebsdId.copy()
  eid[~within] = NO_PIXEL
  return orderWalk(out._with(V=V, F=np.stack([tail, head], axis=1), ebsdId=eid))


def defaultDelta(gB):
  """half the median segment, which follows the polyline closely where a sample per median
  length may cut a corner"""
  return 0.0 if gB.isempty else float(np.median(gB.segLength)) / 2


# which segment an arc length falls on, confined to the piece it belongs to, and how far
# along it; a sample sitting exactly on a segment start belongs to that segment
def _locate(g, Lc, lo, hi):
  i = np.clip(np.searchsorted(Lc, g, side="right") - 1, lo, hi)
  d = Lc[i + 1] - Lc[i]
  with np.errstate(invalid="ignore", divide="ignore"):
    lam = np.where(d > 0, (g - Lc[i]) / d, 0.0)
  return i, np.clip(lam, 0, 1)


# ------------------------------------------------------------------ the grains

def sharedVertices(grains):
  """the vertices both boundaries walk over: a vertex where an inner boundary ends on the outer
  one has degree two to either and would be dissolved, leaving the other hanging off a
  vertex its walk no longer reaches"""
  n = len(grains.boundary.V)
  outer, inner = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
  outer[grains.boundary.F] = True
  inner[grains.innerBoundary.F] = True
  return np.flatnonzero(outer & inner)


def withBoundary(grains, b, ib):
  """the grains over a resampled pair of boundaries, their loops traced again"""
  return grain2d(grains.id, grains.phaseId, grains.CSList, grains.numPixel, grains.meanRotation, grains.GOS, grains.frame, grains.prop,
                 boundary=b, innerBoundary=ib, phaseMap=grains.phaseMap)


# both boundaries are coarsened before either may drop a vertex the other still uses
def _coarsenGrains(verb, grains, *args):
  p = sharedVertices(grains)
  return withBoundary(grains, verb(grains.boundary, *args, protect=p, keepGrains=grains.id),
                      verb(grains.innerBoundary, *args, protect=p, keepGrains=grains.id))


def _refineGrains(grains, delta=None):
  """refining appends its new vertices, so the pool has to reach the second boundary before it
  is refined in turn, and the first one back afterwards"""
  if delta is None:
    delta = defaultDelta(grains.boundary)
  p = sharedVertices(grains)
  b = refineBoundary(grains.boundary, delta, protect=p)
  ib = refineBoundary(grains.innerBoundary._with(V=b.V), delta, protect=p)
  return withBoundary(grains, b._with(V=ib.V), ib)


def _reduceGrains(grains, n=2):
  """the grains thin, simplify or refine their boundary and keep their inner boundary on it"""
  return _coarsenGrains(reduceBoundary, grains, n)


def _simplifyGrains(grains, epsilon=None):
  return _coarsenGrains(simplifyBoundary, grains, epsilon)


# on a boundary the word says nothing: gB.reduce, gB.simplify, gB.refine
for _f in (reduceBoundary, simplifyBoundary, refineBoundary):
  install(grainBoundary, _f, _f.__name__.removesuffix("Boundary"))
install(grain2d, _reduceGrains, "reduceBoundary")
install(grain2d, _simplifyGrains, "simplifyBoundary")
install(grain2d, _refineGrains, "refineBoundary")
install(grain2d, sharedVertices)

__all__ = ["mustKeep", "mergeRuns", "sharedVertices"]
