# grains: the polygons the boundary encloses (MATLAB @grain2d, Julia grains.jl)
#
#   grains = calcGrains(ebsd)
#   grains.id, grains.phaseId, grains.numPixel, grains.meanOrientation, grains.GOS
#   grains.boundary, grains.innerBoundary, grains.triplePoints, grains.loops
#   grains.area, grains.perimeter, grains.centroid, grains.diameter, grains.hasHole, grains.isInclusion (polygonSet)
#   grains.neighbors(), grains.numNeighbors, grains.isBoundary, grains.checkInside(xy), grains(x, y)
#   grains[k], grains[mask], grains['Forsterite'], grains.selectByGrainId(ids), ebsd[grains]
#   grains.merge(pairs), grains.merge(gB), grains.merge(threshold=...)
#
# The boundary is in walk order with the grain of grainId[:, 0] on its left, so the
# segments around one grain, each taken in the sense that keeps the grain on the left,
# already form closed loops wound anticlockwise outside and clockwise around a hole. A
# grain is its loops, the outer one first, then one per inclusion; the flattened keyhole
# form is derived. Ids count from 1, so that 0 is a measurement without a grain, as in
# MATLAB; grains[k] is positional and selectByGrainId picks by id.
from __future__ import annotations

import numpy as np

from ..geometry._array import MtexArray
from ..geometry.kernels import backend
from ..geometry.frames import referenceFrame
from ..geometry.orientation import orientation
from ..geometry.quaternion import rotation
from ..geometry.symreduce import closestImage
from ..geometry.vector3d import vector3d
from .phaselist import phaseTable
from .polygon import polygonSet, signedArea2, _insideLoop

FIELDS = ("id", "phaseId", "CSList", "numPixel", "meanRotation", "GOS", "frame", "prop", "boundary", "innerBoundary", "loops", "phaseMap")


class grain2d(polygonSet):

  def __init__(self, id, phaseId, CSList, numPixel, meanRotation, GOS, frame, prop=None, boundary=None, innerBoundary=None, loops=None,
               phaseMap=None):
    object.__setattr__(self, "id", np.asarray(id, dtype=np.int32))
    object.__setattr__(self, "phaseMap", np.arange(len(CSList)) if phaseMap is None else np.asarray(phaseMap))
    object.__setattr__(self, "phaseId", np.asarray(phaseId, dtype=np.int8))
    object.__setattr__(self, "CSList", CSList)
    object.__setattr__(self, "numPixel", np.asarray(numPixel, dtype=np.int32))
    object.__setattr__(self, "meanRotation", meanRotation if isinstance(meanRotation, MtexArray) else rotation._fromData(meanRotation))
    object.__setattr__(self, "GOS", np.asarray(GOS))
    object.__setattr__(self, "frame", frame)
    object.__setattr__(self, "prop", {} if prop is None else dict(prop))
    object.__setattr__(self, "boundary", boundary)
    object.__setattr__(self, "innerBoundary", innerBoundary)
    if loops is None and boundary is not None:
      loops = tracePolygons(boundary, self.id)
    object.__setattr__(self, "loops", loops)

  # ------------------------------------------------------------------ the table

  @property
  def size(self):
    return len(self.id)

  def __len__(self):
    return self.size

  @property
  def shape(self):
    return (self.size,)

  @property
  def isIndexed(self):
    indexed = np.array([f.isIndexed for f in self.CSList.frames])
    return indexed[self.phaseId]

  @property
  def indexedPhaseId(self):
    present = np.unique(self.phaseId)
    return np.array([p for p in present if self.CSList.isIndexed(p)], dtype=int)

  @property
  def mineralList(self):
    return self.CSList.mineralList

  def _onePhase(self):
    ids = self.indexedPhaseId
    if len(ids) != 1:
      raise ValueError(f"the grains have {len(ids)} indexed phases, so name one: grains['phase']")
    return int(ids[0])

  # the plotting convention of the specimen frame the grains are written in
  @property
  def how2plot(self):
    from ..geometry.plottingconvention import plottingConvention
    pC = getattr(self.frame, "how2plot", None)
    return plottingConvention.default if pC is None else pC

  # the phase number of every grain as the file numbered it
  @property
  def phase(self):
    return self.phaseMap[self.phaseId]

  @property
  def CS(self):
    return self.CSList[self._onePhase()]

  @property
  def mineral(self):
    return self.CS.mineral

  # the mean orientations of the grains of the one indexed phase present
  @property
  def meanOrientation(self):
    p = self._onePhase()
    return orientation._fromRotation(self.meanRotation.data[self.phaseId == p], (self.CSList[p], self.frame))

  @property
  def grainSize(self):
    return self.numPixel

  @property
  def triplePoints(self):
    return self.boundary.triplePoints

  @property
  def V(self):
    return self.boundary.V

  @property
  def allV(self):
    return self.boundary.allV

  @property
  def plane(self):
    return self.boundary.plane

  # the normal of the plane the map lies in, as a direction of the specimen
  @property
  def N(self):
    return vector3d(self.plane.normal, frame=self.frame, isNormalized=True)

  def __getattr__(self, name):
    prop = self.__dict__.get("prop", {})
    if name in prop:
      return prop[name]
    raise AttributeError(f"the grains carry no property {name!r}" + (f"; they have " + ", ".join(repr(k) for k in prop) if prop else ""))

  def __setattr__(self, name, value):
    if name in FIELDS or name.startswith("_") or isinstance(getattr(type(self), name, None), property):
      object.__setattr__(self, name, value)
    else:
      v = value if isinstance(value, list) else np.asarray(value)
      if len(v) != self.size:
        raise ValueError(f"the property {name!r} has {len(v)} entries, the grains {self.size}")
      self.prop[name] = v

  # ------------------------------------------------------------------ the polygons

  # the grains containing the point (x, y) in map coordinates
  def __call__(self, x, y):
    return self[self.inpolygon([x, y])]

  # a grain that sits inside the hole of another one: where the grain lies, not what it touches
  @property
  def isInclusion(self):
    out = np.zeros(self.size, dtype=bool)
    V = self.V
    first = np.array([V[ls[0][0]] if ls else [np.nan, np.nan] for ls in self.loops]).reshape(-1, 2)
    for k, ls in enumerate(self.loops):
      for h in ls[1:]:
        lo, hi = V[h].min(axis=0), V[h].max(axis=0)
        cand = np.flatnonzero(~out & np.all((first >= lo) & (first <= hi), axis=1))
        cand = cand[cand != k]
        if len(cand):
          out[cand] = _insideLoop(V, h, first[cand])
    return out

  # ------------------------------------------------------------------ neighbours

  # the pairs of ids of grains sharing a boundary, both in the selection, or either with
  # full; as a symmetric sparse matrix indexed by id with matrix
  def neighbors(self, full=False, matrix=False):
    g = self.boundary.grainId
    a, b = g[:, 0], g[:, 1]
    ok = (a > 0) & (b > 0) & (a != b)
    ina, inb = np.isin(a, self.id), np.isin(b, self.id)
    ok &= (ina | inb) if full else (ina & inb)
    pairs = np.unique(np.sort(g[ok], axis=1), axis=0)
    if not matrix:
      return pairs
    from scipy.sparse import coo_matrix
    n = int(max(self.id.max(initial=0), pairs.max(initial=0)))
    return coo_matrix((np.ones(2 * len(pairs), dtype=bool), (np.concatenate([pairs[:, 0], pairs[:, 1]]) - 1,
                                                             np.concatenate([pairs[:, 1], pairs[:, 0]]) - 1)), shape=(n, n)).tocsr()

  @property
  def numNeighbors(self):
    pairs = self.neighbors(full=True)
    counts = np.bincount(pairs.reshape(-1), minlength=int(self.id.max(initial=0)) + 1)
    return counts[self.id].astype(np.int32)

  # a grain that touches the border of the map
  @property
  def isBoundary(self):
    g = self.boundary.grainId
    rim = np.unique(np.concatenate([g[g[:, 0] == 0, 1], g[g[:, 1] == 0, 0]]))
    return np.isin(self.id, rim[rim > 0])

  # the inner boundary of every grain: how many segments and how much length of the
  # subgrain boundaries lie in it
  def subBoundarySize(self, sel=None):
    ib = self.innerBoundary
    sel = np.ones(len(ib), dtype=bool) if sel is None else sel
    counts = np.bincount(ib.grainId[sel, 0], minlength=int(self.id.max(initial=0)) + 1)
    return counts[self.id].astype(np.int32)

  def subBoundaryLength(self, sel=None):
    ib = self.innerBoundary
    sel = np.ones(len(ib), dtype=bool) if sel is None else sel
    L = np.bincount(ib.grainId[sel, 0], weights=ib.segLength[sel], minlength=int(self.id.max(initial=0)) + 1)
    return L[self.id]

  # ------------------------------------------------------------------ selection

  # the unit of the map the grains were reconstructed from, as their boundary carries it
  @property
  def scanUnit(self):
    return getattr(self.boundary, "scanUnit", "um")

  # grains[x, y]: the grain the position lies in, MATLAB's grains(x, y)
  def __getitem__(self, key):
    if isinstance(key, tuple) and len(key) == 2 and all(isinstance(k, (int, float, np.integer, np.floating)) for k in key) \
       and any(isinstance(k, (float, np.floating)) for k in key):
      return self._subset(self.inpolygon([float(key[0]), float(key[1])]))
    if isinstance(key, str):
      return self._subset(self._phaseMask([key]))
    if isinstance(key, referenceFrame):
      return self._subset(self.phaseId == self.CSList.phaseId(key))
    if isinstance(key, (list, tuple)) and key and isinstance(key[0], str):
      if key[0] == "id":
        return self.selectByGrainId(key[1:])
      return self._subset(self._phaseMask(key))
    return self._subset(key)

  def _phaseMask(self, names):
    mask = np.zeros(self.size, dtype=bool)
    for name in names:
      if name.lower() == "indexed":
        mask |= self.isIndexed
      elif name.lower() == "notindexed" and "notindexed" not in [m.lower() for m in self.mineralList]:
        mask |= ~self.isIndexed
      else:
        mask |= self.phaseId == self.CSList.phaseId(name)
    return mask

  # the rows picked, with the boundaries restricted to the segments those grains touch
  def _subset(self, sel):
    ids = np.arange(self.size)[sel].reshape(-1)
    keep = self.id[ids]
    return grain2d(keep, self.phaseId[ids], self.CSList, self.numPixel[ids], self.meanRotation.data[ids], self.GOS[ids], self.frame,
                   {k: (v[ids] if isinstance(v, np.ndarray) else [v[i] for i in ids]) for k, v in self.prop.items()},
                   boundary=None if self.boundary is None else self.boundary.selectByGrainId(keep),
                   innerBoundary=None if self.innerBoundary is None else self.innerBoundary.selectByGrainId(keep),
                   loops=None if self.loops is None else [self.loops[i] for i in ids], phaseMap=self.phaseMap)

  # the positions of grain ids in this list
  def id2ind(self, ids):
    pos = {int(g): k for k, g in enumerate(self.id)}
    ids = np.atleast_1d(np.asarray(ids)).reshape(-1)
    if not all(int(i) in pos for i in ids):
      raise ValueError("a requested grain id is not in the selection")
    out = np.array([pos[int(i)] for i in ids], dtype=int)
    return out[0] if np.ndim(ids) == 1 and len(out) == 1 and np.isscalar(ids[0]) and not isinstance(ids, np.ndarray) else out

  # the grains of the phase of ori whose mean orientation lies within epsilon of it, or of
  # a fibre
  def findByOrientation(self, ori, epsilon):
    from ..geometry.fibre import fibre
    cs = ori.CS
    ids = np.flatnonzero(self.phaseId == self.CSList.phaseId(cs))
    o = orientation._fromRotation(self.meanRotation.data.reshape(-1, 4)[ids], (cs, self.frame))
    d = ori.angle(o) if isinstance(ori, fibre) else o.angle(ori)
    return self._subset(ids[np.asarray(d).reshape(-1) <= epsilon])

  # the grains named by id, in the order asked for
  def selectByGrainId(self, ids):
    ids = np.atleast_1d(np.asarray(ids)).reshape(-1)
    pos = {int(g): k for k, g in enumerate(self.id)}
    if not all(int(i) in pos for i in ids):
      raise ValueError("a requested grain id is not in the selection")
    return self._subset(np.array([pos[int(i)] for i in ids], dtype=int))

  def __iter__(self):
    return (self[k] for k in range(self.size))

  def copy(self):
    return grain2d(self.id.copy(), self.phaseId.copy(), self.CSList, self.numPixel.copy(), self.meanRotation.data.copy(), self.GOS.copy(),
                   self.frame, {k: (v.copy() if isinstance(v, np.ndarray) else list(v)) for k, v in self.prop.items()},
                   boundary=self.boundary, innerBoundary=self.innerBoundary, loops=self.loops, phaseMap=self.phaseMap)

  # ------------------------------------------------------------------ merging

  # join grains of one phase into one: by pairs of ids, by the segments of a boundary, or
  # by a misorientation threshold across the shared boundaries. The merge is a relabelling
  # and nothing else: the boundary between two grains that became one moves to the inner
  # boundary, the loops are traced again, the mean is the pixel-weighted mean of the parts.
  # The id of the parent of every grain is written into grains.parentId, the merged grains
  # carry the ids of their parts as childId
  def merge(self, pairs=None, threshold=None):
    if threshold is not None:
      pairs = self._pairsBelow(threshold)
    elif hasattr(pairs, "grainId"):
      pairs = pairs.grainId
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if len(np.unique(self.id)) != self.size:
      raise ValueError("merging requires distinct grains")
    from .segment import components
    pos = {int(g): k for k, g in enumerate(self.id)}
    a = np.array([pos.get(int(x), -1) for x in pairs[:, 0]], dtype=np.int64)
    b = np.array([pos.get(int(x), -1) for x in pairs[:, 1]], dtype=np.int64)
    ok = (a >= 0) & (b >= 0)
    ok[ok] &= self.phaseId[a[ok]] == self.phaseId[b[ok]]        # a phase boundary is never merged
    roots = components(self.size, a[ok], b[ok], backend="numpy")
    _, first, inverse = np.unique(roots, return_index=True, return_inverse=True)
    parent = (np.argsort(np.argsort(first))[inverse] + 1).astype(np.int32)
    n = len(first)
    self.parentId = self.id.copy() if n == self.size else parent
    if n == self.size:
      out = self.copy()
      out.childId = [[int(i)] for i in self.id]
      out.prop.pop("parentId", None)
      return out
    members = [np.flatnonzero(parent == p) for p in range(1, n + 1)]
    T = self.meanRotation.data.dtype
    phaseId = np.array([self.phaseId[m[0]] for m in members], dtype=np.int8)
    numPixel = np.array([self.numPixel[m].sum() for m in members], dtype=np.int32)
    meanRot = np.zeros((n, 4), dtype=T)
    gos = np.zeros(n, dtype=T)
    for k, m in enumerate(members):
      w = self.numPixel[m].astype(T)
      big = m[np.argmax(w)]
      gos[k] = np.sum(self.GOS[m] * w) / max(1, w.sum())
      if not self.CSList.isIndexed(phaseId[k]) or len(m) == 1:
        meanRot[k] = self.meanRotation.data[big]
        continue
      G = self.CSList[phaseId[k]].sym.properGroup().rotations().data.reshape(-1, 4).astype(T)
      ref = self.meanRotation.data[big]
      q = closestImage(ref, self.meanRotation.data[m], np.array([[1.0, 0, 0, 0]], dtype=T), G, False)[1]
      q = np.where((q @ ref)[:, None] < 0, -q, q) * w[:, None]
      s = q.sum(axis=0)
      meanRot[k] = s / np.linalg.norm(s)
    gB, inner = self._relabelBoundary(parent, n)
    return grain2d(np.arange(1, n + 1), phaseId, self.CSList, numPixel, meanRot, gos, self.frame, {"childId": [self.id[m].tolist() for m in members]},
                   boundary=gB, innerBoundary=inner, phaseMap=self.phaseMap)

  # the pairs of neighbours of one indexed phase whose mean orientations are within the threshold
  def _pairsBelow(self, threshold):
    pos = {int(g): k for k, g in enumerate(self.id)}
    out = []
    for idA, idB in self.neighbors():
      a, b = pos[int(idA)], pos[int(idB)]
      p = self.phaseId[a]
      if p != self.phaseId[b] or not self.CSList.isIndexed(p):
        continue
      fr = self.CSList[p]
      o1 = orientation._fromRotation(self.meanRotation.data[a], (fr, self.frame))
      o2 = orientation._fromRotation(self.meanRotation.data[b], (fr, self.frame))
      if o1.angle(o2) <= threshold:
        out.append((idA, idB))
    return np.array(out, dtype=np.int64).reshape(-1, 2)

  # the boundary of the merged grains: a segment between two grains that became one is
  # inside one now; neighbours outside the selection keep ids above the merged ones
  def _relabelBoundary(self, parent, n):
    gB, ib = self.boundary, self.innerBoundary
    labels = {int(g): int(parent[k]) for k, g in enumerate(self.id)}
    seen = np.unique(np.concatenate([gB.grainId.reshape(-1), ib.grainId.reshape(-1)]))
    external = [int(x) for x in seen if x > 0 and int(x) not in labels]
    for k, x in enumerate(external):
      labels[x] = n + k + 1
    relabel = np.vectorize(lambda x: 0 if x == 0 else labels[int(x)], otypes=[np.int32])
    gid = relabel(gB.grainId) if len(gB) else gB.grainId
    outer = gid[:, 0] != gid[:, 1]
    sub = lambda sel: gB._with(F=gB.F[sel], grainId=gid[sel], ebsdId=gB.ebsdId[sel], phaseId=gB.phaseId[sel], misrotation=gB.misrotation.data[sel])
    oldInner = ib._with(grainId=relabel(ib.grainId) if len(ib) else ib.grainId)
    return sub(outer), catBoundaries(sub(~outer), oldInner)

  # ------------------------------------------------------------------ display

  def __repr__(self):
    lines = [f"grain2d ({self.frame.char()})"]
    if self.size == 0:
      return lines[0] + "\n  no grains"
    lines.append(f"  size: {self.size}")
    ids, counts = np.unique(self.phaseId, return_counts=True)
    pixels = [int(self.numPixel[self.phaseId == p].sum()) for p in ids]
    lines.append(phaseTable(self.CSList.frames, ids, [f"{c} ({px} pixels)" for c, px in zip(counts, pixels)]).replace("Orientations", "Grains       "))
    if self.boundary is not None:
      lines.append(f"  boundary segments: {len(self.boundary)}, inner: {len(self.innerBoundary)}, triple points: {len(self.triplePoints)}")
    if self.prop:
      lines.append("  properties: " + ", ".join(sorted(self.prop)))
    # a short list shows its grains one by one, as MATLAB does
    if self.size <= 20:
      e = np.degrees(np.asarray(self.meanRotation.Euler())).reshape(3, -1)
      lines.append("  " + "".join(f"{h:>8}" for h in ("Id", "Phase", "Pixels")) + f"{'meanRotation':>20}{'GOS':>12}")
      for k in range(self.size):
        euler = f"({e[0, k]:.0f}°,{e[1, k]:.0f}°,{e[2, k]:.0f}°)"
        lines.append(f"  {self.id[k]:>8}{self.phaseId[k]:>8}{self.numPixel[k]:>8}{euler:>20}{self.GOS[k]:>12.3g}")
    return "\n".join(lines)


# ------------------------------------------------------------------ tracing

# the loops of every grain, from the segments that have it on one side: a segment is taken
# in the sense that keeps the grain on its left, so the loops come out closed and wound;
# the compiled walk over all grains at once, or the Python walk grain by grain without it
def tracePolygons(gB, ids):
  ids = np.asarray(ids, dtype=np.int64).reshape(-1)
  if backend() == "numba":
    traced = _tracePolygonsNumba(gB, ids)
    if traced is not None:
      return traced
  pos = {int(g): k for k, g in enumerate(ids)}
  edges = [[] for _ in ids]
  for k in range(len(gB)):
    g1, g2 = int(gB.grainId[k, 0]), int(gB.grainId[k, 1])
    if g1 in pos:
      edges[pos[g1]].append((gB.F[k, 0], gB.F[k, 1]))
    if g2 in pos:
      edges[pos[g2]].append((gB.F[k, 1], gB.F[k, 0]))
  return [traceLoops(e, gB.V) for e in edges]


# the directed edges of every grain sorted by grain, walked in one compiled pass; the loops
# of a grain come back with the outer one, the largest positive area, first
def _tracePolygonsNumba(gB, ids):
  from .kernels import traceLoopsNumba
  top = int(max(ids.max(initial=0), gB.grainId.max(initial=0))) + 1
  posOf = np.full(top, -1, dtype=np.int64)
  posOf[ids] = np.arange(len(ids))
  g = np.concatenate([posOf[gB.grainId[:, 0]], posOf[gB.grainId[:, 1]]])
  ea = np.concatenate([gB.F[:, 0], gB.F[:, 1]])
  eb = np.concatenate([gB.F[:, 1], gB.F[:, 0]])
  keep = g >= 0
  order = np.argsort(g[keep], kind="stable")
  ea, eb, g = ea[keep][order], eb[keep][order], g[keep][order]
  goff = np.concatenate([[0], np.cumsum(np.bincount(g, minlength=len(ids)))])
  traced = traceLoopsNumba(gB.V, ea, eb, goff)
  if traced is None:
    return None
  loopV, loff, lgrain, larea = traced
  out = [[] for _ in ids]
  for k in np.lexsort((-larea, lgrain)):
    out[lgrain[k]].append(loopV[loff[k]:loff[k + 1]])
  return out


# close the directed edges into loops; where a grain touches itself the walk leaves a
# vertex by the edge that turns most tightly clockwise, which keeps the face on the left
def traceLoops(edges, V):
  loops = []
  if not edges:
    return loops
  frm = {}
  for k, e in enumerate(edges):
    frm.setdefault(int(e[0]), []).append(k)
  used = [False] * len(edges)
  for start in range(len(edges)):
    if used[start]:
      continue
    loop = []
    k = start
    while True:
      used[k] = True
      loop.append(edges[k][0])
      cand = frm.get(int(edges[k][1]), [])
      nxt = -1
      if len(cand) == 1:
        if not used[cand[0]]:
          nxt = cand[0]
      else:
        nxt = _tightestTurn(edges, V, k, cand, used)
      if nxt < 0:
        break
      k = nxt
    if len(loop) >= 3:
      loops.append(np.array(loop, dtype=np.int32))
  # the outer loop is the one that encloses area positively; holes are negative
  loops.sort(key=lambda l: -signedArea2(V[l]))
  return loops


def _tightestTurn(edges, V, k, cand, used):
  a = V[edges[k][1]] - V[edges[k][0]]
  best, bestTurn = -1, -np.inf
  for c in cand:
    if used[c]:
      continue
    b = V[edges[c][1]] - V[edges[c][0]]
    t = np.arctan2(a[0] * b[1] - a[1] * b[0], a @ b)
    if t > bestTurn:
      bestTurn, best = t, c
  return best


# two boundaries laid end to end, on the same vertices
def catBoundaries(a, b):
  if a.V is not b.V and not np.array_equal(a.V, b.V):
    raise ValueError("the two boundaries do not share their vertices")
  return a._with(F=np.concatenate([a.F, b.F]), grainId=np.concatenate([a.grainId, b.grainId]), ebsdId=np.concatenate([a.ebsdId, b.ebsdId]),
                 phaseId=np.concatenate([a.phaseId, b.phaseId]), misrotation=np.concatenate([a.misrotation.data, b.misrotation.data]))


__all__ = ["grain2d"]
