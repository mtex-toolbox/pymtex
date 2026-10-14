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
# splits every segment into pieces of about equal length and keeps every vertex. A closed
# chain has no junction: the vertex the walk cuts it at is an accident of the numbering, so
# the chain is anchored at a vertex that has to stay and cut there instead. A grain list
# coarsens both of its boundaries at once, over one pool of vertices, and traces its loops
# again.
from __future__ import annotations

import numpy as np

from .boundary import grainBoundary, orderWalk
from .decomposition import NO_PIXEL
from .grains import grain2d
from .kernels import _compiledLoop
from ..geometry.install import install


# ------------------------------------------------------------------ the heads that survive

def mustKeep(gB, protect=None):
  """the heads a coarsening may not dissolve: the end of an open chain is a junction and stays
  where it is, a vertex where the pair of grains changes is a corner of both grain polygons
  even where only two segments meet there, and a protected vertex is one another boundary
  over the same vertices still walks over

    keep = mustKeep(gB)                                        # the junctions and the corners
    keep = mustKeep(gB, protect=grains.sharedVertices())       # and the vertices the inner boundary uses

  A row is the head of a segment, its vertex gB.F[:, 1]; reduce and simplify keep the rows
  marked and may dissolve the others.

  Input
    gB - grainBoundary
  Output
    keep - a boolean per segment, whether its head must stay
  Keywords
    protect - the ids of vertices that must stay as well
  See also
    grainBoundary.reduce, grainBoundary.simplify, grain2d.sharedVertices"""
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
  s, e = np.flatnonzero(gB.isChainStart), np.flatnonzero(gB.isChainEnd)
  if len(s) == 0:
    return keep
  n = e - s + 1
  few = gB.isClosed[s] & (n >= 3) & (np.add.reduceat(keep.astype(np.int64), s) < 3)
  # the vertices a third, two thirds and all the way along the chain
  keep[s[few, None] + np.rint(n[few, None] * np.array([1, 2, 3]) / 3).astype(np.int64) - 1] = True
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
  """the boundary coarsened by keeping every n-th vertex along each chain, whatever the shape;
  MATLAB's grainBoundary.reduce

    gB = gB.reduce()                           # every second vertex
    gB = gB.reduce(4)                          # every fourth vertex
    gB = gB.reduce(2, protect=vertexIds)       # these vertices kept as well
    gB = gB.reduce(2, keepGrains=grains.id)    # no grain of the list collapsed

  The vertices between the kept ones are dissolved and their segments merged, so the
  boundary stays connected and keeps its length up to the straightening of the dissolved
  corners. The junctions, the ends of the chains, and the vertices where the pair of grains
  changes always stay, so which grains touch, and where, is unchanged. A closed chain keeps
  at least three vertices. A merged segment inherits grainId, ebsdId and misrotation from
  the last segment of its run, and its ebsdId says it runs between no one pair of pixels.

  It thins a straight run and a corner alike; simplify spends the vertices where the shape
  needs them and is mostly the better choice.

  Input
    gB - grainBoundary
  Output
    gB - grainBoundary, the coarsened boundary
  Keywords
    n - keep every n-th vertex along each chain; 1 or less leaves the boundary as it is
    protect - the ids of the vertices that must stay, on top of the junctions
    keepGrains - the ids of the grains that must keep a polygon: a grain that would be left
        with fewer than three vertices is not coarsened at all; MATLAB does this for every
        grain, grain2d.reduceBoundary passes all of them
  See also
    grainBoundary.simplify, grainBoundary.refine, grain2d.reduceBoundary, smoothBoundary"""
  if gB.isempty or n <= 1:
    return gB
  starts = np.flatnonzero(gB.isChainStart)
  pos = np.arange(len(gB)) - starts[gB.chainId] + 1
  keep = mustKeep(gB, protect) | (pos % n == 0)
  return _finish(gB, keep, keepGrains)


def simplifyBoundary(gB, epsilon=None, protect=None, keepGrains=None):
  """the boundary with the vertices dropped that carry no shape, Douglas-Peucker chain by chain;
  MATLAB's grainBoundary.simplify

    gB = gB.simplify()                           # the pixel staircase straightened
    gB = gB.simplify(0.5)                        # a tolerance of 0.5 map units
    gB = gB.simplify(protect=vertexIds)          # these vertices kept as well
    gB = gB.simplify(keepGrains=grains.id)       # no grain of the list collapsed

  A vertex is dropped when leaving it out moves the boundary by less than epsilon, so many
  vertices go along a straight run and none at a corner; on a pixel grid this turns a
  staircase into the straight line it approximates. The junctions, the ends of the chains,
  always stay, so which grains touch, and where, is unchanged; a closed chain keeps at least
  three vertices. The dropped vertices' segments are merged as by reduce.

  The default tolerance is the median segment length over sqrt(2): a staircase of spacing d
  never strays further than d / sqrt(2) from the line it approximates, the worst case being a
  boundary at 45 degree, so it removes the grid and nothing else.

  Input
    gB - grainBoundary
  Output
    gB - grainBoundary, the simplified boundary
  Keywords
    epsilon - the tolerance in the units of the map, how far the boundary may move; 0 leaves
        the boundary as it is, by default the median segment length over sqrt(2)
    protect - the ids of the vertices that must stay, on top of the junctions
    keepGrains - the ids of the grains that must keep a polygon: a grain smaller than epsilon
        would lose all its vertices and is not coarsened at all; MATLAB does this for every
        grain, grain2d.simplifyBoundary passes all of them
  See also
    grainBoundary.reduce, grainBoundary.refine, grain2d.simplifyBoundary, smoothBoundary"""
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
  (_compiledLoop(_keepShapeLoop) or _keepShapeLoop)(np.ascontiguousarray(gB.V, dtype=np.float64), chainV, offset, closed, epsilon, keepV)
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


# ------------------------------------------------------------------ refining

def refineBoundary(gB, delta=None):
  """split every segment into pieces of about length delta: the old vertices stay where they
  are and the new ones lie on the old segments, so the boundary keeps its shape and its
  length and only the sampling gets finer, which gives a smoothing that follows evenly
  spaced degrees of freedom. Every piece carries the grains, the phases, the pair of
  measurements and the misorientation of the segment it lies on; MATLAB's grainBoundary.refine

    gB = gB.refine()        # pieces of half the median segment length
    gB = gB.refine(10)      # pieces of about 10 map units

  A segment is split into round(length / delta) pieces, at least one. MATLAB resamples every
  chain at equal arc length between the vertices that must stay and takes a protect list;
  here every segment is split on its own, which keeps the old vertices, and there is no
  protect.

  Input
    gB - grainBoundary
  Output
    gB - grainBoundary, the refined boundary
  Keywords
    delta - the length of the pieces in the units of the map, by default half the median
        segment length; 0 or less leaves the boundary as it is
  See also
    grainBoundary.reduce, grainBoundary.simplify, grain2d.refineBoundary, smoothBoundary"""
  if gB.isempty:
    return gB
  if delta is None:
    delta = defaultDelta(gB)
  if delta <= 0:
    return gB
  # how many pieces each segment is split into, src the segment of a piece and k its one
  # based position within it
  m = np.maximum(1, np.rint(gB.segLength / delta)).astype(np.int64)
  src = np.repeat(np.arange(len(gB)), m)
  k = np.arange(1, len(src) + 1) - np.repeat(np.cumsum(m) - m, m)
  # every piece but the last of a segment ends on a new vertex
  new = k < m[src]
  P1, P2 = gB.V[gB.F[src[new], 0]], gB.V[gB.F[src[new], 1]]
  lam = (k[new] / m[src[new]])[:, None]
  V = np.concatenate([gB.V, (P1 + lam * (P2 - P1)).astype(gB.V.dtype)])
  head = gB.F[src, 1]
  head[new] = len(gB.V) + np.arange(np.count_nonzero(new))
  tail = gB.F[src, 0]
  tail[k > 1] = head[np.flatnonzero(k > 1) - 1]
  return gB[src]._with(V=V, F=np.stack([tail, head], axis=1))


def defaultDelta(gB):
  """half the median segment"""
  return 0.0 if gB.isempty else float(np.median(gB.segLength)) / 2


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
  """the grains with their boundary and inner boundary refined, every segment split into
  pieces of about length delta; MATLAB's grain2d.refineBoundary

    grains = grains.refineBoundary()      # pieces of half the median segment length
    grains = grains.refineBoundary(10)    # pieces of about 10 map units

  The vertices stay where they are and the new ones lie on the old segments, so the grains
  keep their shape and only the sampling gets finer, which lets a following smoothBoundary
  move evenly spaced vertices. Both boundaries are refined over one pool of vertices.

  Input
    grains - grain2d
  Output
    grains - grain2d, the polygons traced again
  Keywords
    delta - the length of the pieces in the units of the map, by default half the median
        segment length of the boundary
  See also
    grainBoundary.refine, grain2d.simplifyBoundary, grain2d.reduceBoundary, smoothBoundary"""
  if delta is None:
    delta = defaultDelta(grains.boundary)
  b = refineBoundary(grains.boundary, delta)
  ib = refineBoundary(grains.innerBoundary._with(V=b.V), delta)
  return withBoundary(grains, b._with(V=ib.V), ib)


def _reduceGrains(grains, n=2):
  """the grains with every n-th vertex of their boundary kept along each chain, whatever the
  shape, the inner boundary kept on it; MATLAB's grain2d.reduceBoundary

    grains = grains.reduceBoundary()      # every second vertex
    grains = grains.reduceBoundary(4)     # every fourth vertex

  The junctions stay where they are, so which grains touch, and where, is unchanged, and the
  polygons are traced again. The vertices both boundaries use stay, and no grain collapses.
  It thins a straight run and a corner alike; simplifyBoundary spends the vertices where the
  shape needs them.

  Input
    grains - grain2d
  Output
    grains - grain2d
  Keywords
    n - keep every n-th vertex along each chain
  See also
    grainBoundary.reduce, grain2d.simplifyBoundary, grain2d.refineBoundary"""
  return _coarsenGrains(reduceBoundary, grains, n)


def _simplifyGrains(grains, epsilon=None):
  """the grains with the vertices of their boundary that carry no shape dropped, those whose
  removal moves the boundary by less than epsilon, the inner boundary kept on it; MATLAB's
  grain2d.simplifyBoundary

    grains = grains.simplifyBoundary()      # the pixel staircase straightened
    grains = grains.simplifyBoundary(50)    # a tolerance of 50 map units

  Douglas-Peucker chain by chain: a straight run collapses into one segment while a corner
  stays. The junctions stay where they are and the polygons are traced again, so the grain
  areas move by at most about epsilon times the perimeter. A grain too small for the
  tolerance is left as it is rather than collapsed.

  Input
    grains - grain2d
  Output
    grains - grain2d
  Keywords
    epsilon - the tolerance in the units of the map, by default the median segment length
        over sqrt(2), how far a staircase strays from the line it approximates
  See also
    grainBoundary.simplify, grain2d.reduceBoundary, grain2d.refineBoundary, smoothBoundary"""
  return _coarsenGrains(simplifyBoundary, grains, epsilon)


# on a boundary the word says nothing: gB.reduce, gB.simplify, gB.refine
for _f in (reduceBoundary, simplifyBoundary, refineBoundary):
  install(grainBoundary, _f, _f.__name__.removesuffix("Boundary"))
install(grain2d, _reduceGrains, "reduceBoundary")
install(grain2d, _simplifyGrains, "simplifyBoundary")
install(grain2d, _refineGrains, "refineBoundary")
install(grain2d, sharedVertices)

__all__ = ["mustKeep", "mergeRuns", "sharedVertices"]
