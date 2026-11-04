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
  keep[:-1] |= np.any(pair[1:] != pair[:-1], axis=1) & ~gB.isChainEnd[:-1]
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
  merged = runStart != surv
  eid = out.ebsdId.copy()
  eid[merged] = NO_PIXEL
  return orderWalk(out._with(F=F, ebsdId=eid, pixelDist=np.where(merged, 0.0, out.pixelDist)))


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


def simplifyBoundary(gB, epsilon=None, protect=None, keepGrains=None, margin=0.0, measuredV=None):
  """the boundary with the vertices dropped that carry no shape, Douglas-Peucker chain by chain;
  MATLAB's grainBoundary.simplify

    gB = gB.simplify()                           # the pixel staircase straightened
    gB = gB.simplify(0.5)                        # a tolerance of 0.5 map units
    gB = gB.simplify(protect=vertexIds)          # these vertices kept as well
    gB = gB.simplify(keepGrains=grains.id)       # no grain of the list collapsed
    gB = gB.simplify(0.1, margin=0.05)           # no pixel centre put on the other side

  A vertex is dropped when leaving it out moves the boundary by less than epsilon, so many
  vertices go along a straight run and none at a corner; on a pixel grid this turns a
  staircase into the straight line it approximates. The junctions, the ends of the chains,
  always stay, so which grains touch, and where, is unchanged. The dropped vertices' segments
  are merged as by reduce.

  Of several vertices at the largest distance the one nearest the middle of the piece by arc
  length is kept, then the one smallest in x, then in y; a distance within 1e-9 epsilon of
  another counts as equal. A closed chain through no junction is anchored at its smallest
  vertex and cut first at the vertex farthest from it, and keeps at least three. So the result
  depends neither on the direction a chain is walked nor on the numbering of the grains.

  With a margin a chord is also refused when it would put the centre of one of the pixels next
  to a pixel edge of the chain, see gB.pixelDist, on its other side or closer to it than the
  margin (de Berg, van Kreveld, Schirra 1998).

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
    margin - how far the pixel centres of the pixel edges stay from the boundary, on their
        side; 0, the default, checks nothing
    measuredV - the vertices (n, 2) the pixel edges were measured at, by default gB.V
  See also
    grainBoundary.reduce, grainBoundary.refine, grain2d.simplifyBoundary, smoothBoundary"""
  if gB.isempty:
    return gB
  if epsilon is None:
    epsilon = defaultEpsilon(gB)
  if epsilon <= 0:
    return gB
  keep = keepShapeChains(gB, float(epsilon), margin, measuredV) | mustKeep(gB, protect)
  return mergeRuns(gB, keepGrainPolygons(gB, keep, keepGrains))


def defaultEpsilon(gB):
  """the default tolerance of simplifyBoundary: the median segment length over sqrt(2), the
  furthest a staircase of that spacing strays from its line"""
  return 0.0 if gB.isempty else float(np.median(gB.segLength)) / np.sqrt(2)


def keepShapeChains(gB, epsilon, margin=0.0, measuredV=None):
  """the heads Douglas-Peucker keeps on every chain, keeping vertex j of a chain meaning keeping
  the head of its segment j - 1; with a margin, the pixel centres of the chain's pixel edges,
  at measuredV, constrain the chords"""
  V = np.ascontiguousarray(np.asarray(gB.V, dtype=np.float64)[:, :2])
  V0 = V if measuredV is None else np.asarray(measuredV, dtype=np.float64)[:, :2]
  starts, ends = np.flatnonzero(gB.isChainStart), np.flatnonzero(gB.isChainEnd)
  # the two pixel centres of every pixel edge, as it was measured, grouped by chain
  isPx = gB.pixelDist > 0 if margin > 0 else np.zeros(len(gB), dtype=bool)
  t = V0[gB.F[isPx, 1]] - V0[gB.F[isPx, 0]]
  off = np.stack([-t[:, 1], t[:, 0]], axis=1) / np.linalg.norm(t, axis=1)[:, None] * gB.pixelDist[isPx, None] / 2
  mid = (V0[gB.F[isPx, 0]] + V0[gB.F[isPx, 1]]) / 2
  cid = gB.chainId[isPx]
  order = np.argsort(np.concatenate([cid, cid]), kind="stable")
  C = np.ascontiguousarray(np.concatenate([mid + off, mid - off])[order])
  cptr = np.concatenate([[0], np.cumsum(2 * np.bincount(cid, minlength=len(starts)))]).astype(np.int64)
  keep = np.zeros(len(gB), dtype=np.bool_)
  _simplifyKernel()(V, gB.F.astype(np.int64), starts, ends, gB.vertexDegree().astype(np.int64), C, cptr, float(epsilon),
                    float(margin), keep)
  return keep


# the compiled simplification, its helpers compiled and bound in place of their Python forms
# first; the Python loop without numba
def _simplifyKernel():
  loop = _compiledLoop(_simplifyLoop)
  if loop is not None and not hasattr(_pick, "py_func"):
    for name in ("_pick", "_crosses", "_douglasPeucker"):
      globals()[name] = _compiledLoop(globals()[name])
  return loop or _simplifyLoop


# Douglas-Peucker on every chain. A loop through no junction is walked from its smallest vertex;
# a loop has no chord, so it is cut at the vertex farthest from its start and both halves are
# simplified, and a third vertex is given back where only the cut survives
def _simplifyLoop(V, F, starts, ends, deg, C, cptr, eps, margin, keep):
  tau = 1e-9 * eps
  for c in range(starts.shape[0]):
    a, n = starts[c], ends[c] - starts[c] + 1
    isLoop = F[a + n - 1, 1] == F[a, 0] and n > 2
    k0 = 0
    if isLoop and deg[F[a, 0]] == 2:
      for i in range(1, n):
        p, q = F[a + i, 0], F[a + k0, 0]
        if V[p, 0] < V[q, 0] or (V[p, 0] == V[q, 0] and V[p, 1] < V[q, 1]):
          k0 = i
    rows = np.empty(n, dtype=np.int64)
    P = np.empty((n + 1, 2))
    for i in range(n):
      rows[i] = a + (k0 + i) % n
      P[i, 0], P[i, 1] = V[F[rows[i], 0], 0], V[F[rows[i], 0], 1]
    P[n, 0], P[n, 1] = V[F[rows[n - 1], 1], 0], V[F[rows[n - 1], 1], 1]
    Cc = C[cptr[c]:cptr[c + 1]]
    kv = np.zeros(n + 1, dtype=np.bool_)
    d = np.empty(n + 1)
    if isLoop:
      for i in range(n + 1):
        d[i] = np.hypot(P[i, 0] - P[0, 0], P[i, 1] - P[0, 1])
      iFar = _pick(d, P, 0, n, tau)
      kv[0], kv[n], kv[iFar] = True, True, True
      _douglasPeucker(P, 0, iFar, eps, tau, Cc, margin, kv, d)
      _douglasPeucker(P, iFar, n, eps, tau, Cc, margin, kv, d)
      if np.count_nonzero(kv) < 4:
        ux, uy = P[iFar, 0] - P[0, 0], P[iFar, 1] - P[0, 1]
        L = np.hypot(ux, uy)
        for i in range(n + 1):
          d[i] = abs((P[i, 0] - P[0, 0]) * uy - (P[i, 1] - P[0, 1]) * ux) / L
        d[0], d[iFar], d[n] = -np.inf, -np.inf, -np.inf
        i = _pick(d, P, 0, n, tau)
        if d[i] > 0:
          kv[i] = True
    else:
      _douglasPeucker(P, 0, n, eps, tau, Cc, margin, kv, d)
    for i in range(n):
      keep[rows[i]] = kv[i + 1]


# the polyline P[lo..hi] with its ends kept: a piece is cut when a point lies further than eps
# from its chord, or when the chord puts a point of C on its other side or closer than margin
def _douglasPeucker(P, lo, hi, eps, tau, C, margin, keep, d):
  keep[lo], keep[hi] = True, True
  stack = np.empty((hi - lo + 2, 2), dtype=np.int64)
  stack[0, 0], stack[0, 1] = lo, hi
  top = 1
  while top > 0:
    top -= 1
    a, b = stack[top, 0], stack[top, 1]
    if b - a < 2:
      continue
    ux, uy = P[b, 0] - P[a, 0], P[b, 1] - P[a, 1]
    L = np.hypot(ux, uy)
    dMax = -np.inf
    for i in range(a + 1, b):
      rx, ry = P[i, 0] - P[a, 0], P[i, 1] - P[a, 1]
      d[i] = np.hypot(rx, ry) if L == 0 else abs(rx * uy - ry * ux) / L
      dMax = max(dMax, d[i])
    d[a], d[b] = -np.inf, -np.inf
    if dMax > eps + tau or _crosses(P, a, b, C, margin):
      i = _pick(d, P, a, b, tau)
      keep[i] = True
      stack[top, 0], stack[top, 1] = a, i
      stack[top + 1, 0], stack[top + 1, 1] = i, b
      top += 2


# the point of P[a..b] at the largest d: of equal ones, within tau, the one nearest the middle of
# the piece by arc length, then the one smallest in x, then in y
def _pick(d, P, a, b, tau):
  dMax = -np.inf
  for i in range(a, b + 1):
    dMax = max(dMax, d[i])
  s = np.zeros(b - a + 1)
  for i in range(a + 1, b + 1):
    s[i - a] = s[i - a - 1] + np.hypot(P[i, 0] - P[i - 1, 0], P[i, 1] - P[i - 1, 1])
  half = s[b - a] / 2
  mMin = np.inf
  for i in range(a, b + 1):
    if d[i] >= dMax - tau:
      mMin = min(mMin, abs(s[i - a] - half))
  best = -1
  for i in range(a, b + 1):
    if d[i] >= dMax - tau and abs(s[i - a] - half) <= mMin + tau:
      if best < 0 or P[i, 0] < P[best, 0] or (P[i, 0] == P[best, 0] and P[i, 1] < P[best, 1]):
        best = i
  return best


# whether the chord of P[a..b] puts a point of C on its other side, inside the polygon of the
# piece closed by the chord or on it, or closer to the chord than margin
def _crosses(P, a, b, C, margin):
  if C.shape[0] == 0:
    return False
  x0, x1, y0, y1 = np.inf, -np.inf, np.inf, -np.inf
  for i in range(a, b + 1):
    x0, x1 = min(x0, P[i, 0]), max(x1, P[i, 0])
    y0, y1 = min(y0, P[i, 1]), max(y1, P[i, 1])
  ux, uy = P[b, 0] - P[a, 0], P[b, 1] - P[a, 1]
  uu = max(ux * ux + uy * uy, np.finfo(np.float64).tiny)
  for k in range(C.shape[0]):
    cx, cy = C[k, 0], C[k, 1]
    if cx < x0 - margin or cx > x1 + margin or cy < y0 - margin or cy > y1 + margin:
      continue
    t = min(max(((cx - P[a, 0]) * ux + (cy - P[a, 1]) * uy) / uu, 0.0), 1.0)
    if np.hypot(cx - P[a, 0] - t * ux, cy - P[a, 1] - t * uy) < margin:
      return True
    if cx < x0 or cx > x1 or cy < y0 or cy > y1:
      continue
    # the winding number of the closed piece about the point, a point on an edge inside
    w = 0
    for i in range(a, b + 1):
      j = i + 1 if i < b else a
      qx, qy, rx, ry = P[i, 0] - cx, P[i, 1] - cy, P[j, 0] - cx, P[j, 1] - cy
      cr = qx * ry - rx * qy
      sc = max(abs(P[i, 0] + P[j, 0]) / 2, abs(P[i, 1] + P[j, 1]) / 2)
      sc = max(sc, abs(P[i, 0] + P[j, 0]) * abs(P[i, 1] + P[j, 1]) / 4)
      if abs(cr) < 3 * np.finfo(np.float64).eps * sc and qx * rx + qy * ry <= 0:
        return True
      if qy <= 0 < ry and cr > 0:
        w += 1
      elif ry <= 0 < qy and cr < 0:
        w -= 1
    if w != 0:
      return True
  return False


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
  out = gB[src]
  return out._with(V=V, F=np.stack([tail, head], axis=1), pixelDist=np.where(m[src] > 1, 0.0, out.pixelDist))


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
def _coarsenGrains(verb, grains, *args, **kw):
  p = sharedVertices(grains)
  return withBoundary(grains, verb(grains.boundary, *args, protect=p, keepGrains=grains.id, **kw),
                      verb(grains.innerBoundary, *args, protect=p, keepGrains=grains.id, **kw))


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


def _simplifyGrains(grains, epsilon=None, margin=0.0, measuredV=None):
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
    margin - how far the pixel centres of the pixel edges stay from the boundary, on their
        side; 0, the default, checks nothing
    measuredV - the vertices the pixel edges were measured at, by default the boundary's
  See also
    grainBoundary.simplify, grain2d.reduceBoundary, grain2d.refineBoundary, smoothBoundary"""
  return _coarsenGrains(simplifyBoundary, grains, epsilon, margin=margin, measuredV=measuredV)


# on a boundary the word says nothing: gB.reduce, gB.simplify, gB.refine
for _f in (reduceBoundary, simplifyBoundary, refineBoundary):
  install(grainBoundary, _f, _f.__name__.removesuffix("Boundary"))
install(grain2d, _reduceGrains, "reduceBoundary")
install(grain2d, _simplifyGrains, "simplifyBoundary")
install(grain2d, _refineGrains, "refineBoundary")
install(grain2d, sharedVertices)

__all__ = ["mustKeep", "mergeRuns", "sharedVertices"]
