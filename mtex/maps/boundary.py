# grain boundaries: the faces the segmentation separates, in walk order (MATLAB
# @grainBoundary, @triplePointList; Julia boundary.jl)
#
#   gB = grains.boundary, grains.innerBoundary
#   gB.F, gB.grainId, gB.ebsdId, gB.phaseId, gB.misrotation, gB.misorientation
#   gB.chainId, gB.isChainStart, gB.isClosed, gB.arcLength, gB.isJunction, gB.triplePoints
#   gB['Forsterite'], gB['Forsterite', 'Enstatite'], gB[mask], gB.hasPhase(...), gB.isTwinning(mori)
#   gB.segLength, gB.direction, gB.midPoint, gB.perimeter(), gB.curvature(), gB.calcMeanDirection()
#
# The branch ADR 0002 is kept: the segments are stored head to tail along each chain,
# oriented so that the grain in grainId[:, 0] lies to the left of the walk, and everything
# else, chains, junctions, triple points, arc length, the sign of the curvature, is
# derived from that and never stored. A chain is a maximal run of segments joined at
# vertices where exactly two meet; a vertex of any other degree is a junction and ends
# the chain. Outside the map a side carries the grain 0, the pixel -1 and the phase -1.
from __future__ import annotations

import numpy as np

from ..geometry.display import xnum2str
from ..geometry.kernels import qmul, _jit
from ..geometry.orientation import orientation
from ..geometry.quaternion import rotation
from ..geometry.units import degree
from ..geometry.vector3d import vector3d
from .decomposition import NO_PIXEL
from .phaselist import phaseTable

OUTSIDE = -1
_COMPILED = {}


def _conj(q):
  return q * np.array([1, -1, -1, -1], dtype=q.dtype)


def _cross2(a, b):
  return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


class grainBoundary:

  def __init__(self, V, F, grainId, ebsdId, phaseId, misrotation, CSList, plane, frame, scanUnit="um"):
    self.V = V
    self.scanUnit = scanUnit
    self.F = np.asarray(F, dtype=np.int32).reshape(-1, 2)
    self.grainId = np.asarray(grainId, dtype=np.int32).reshape(-1, 2)
    self.ebsdId = np.asarray(ebsdId, dtype=np.int32).reshape(-1, 2)
    self.phaseId = np.asarray(phaseId, dtype=np.int8).reshape(-1, 2)
    self.misrotation = misrotation if isinstance(misrotation, rotation) else rotation._fromData(misrotation)
    self.CSList, self.plane, self.frame = CSList, plane, frame
    # the chain structure is derived from F and V once; a boundary is never edited in place
    self._memo = {}
    self._shape = None
    # the whole network a selection was taken from, which holds its triple points
    self._root = None

  def _with(self, **kw):
    fields = dict(V=self.V, F=self.F, grainId=self.grainId, ebsdId=self.ebsdId, phaseId=self.phaseId, misrotation=self.misrotation,
                  CSList=self.CSList, plane=self.plane, frame=self.frame, scanUnit=self.scanUnit)
    fields.update(kw)
    gB = grainBoundary(**fields)
    if "V" not in kw:
      gB._root = self._root
    return gB

  def __len__(self):
    return len(self.F)

  @property
  def size(self):
    return len(self.F)

  @property
  def isempty(self):
    return len(self.F) == 0

  # ------------------------------------------------------------------ geometry

  # the vertices as framed points of the specimen, all of them and the ones in use
  @property
  def allV(self):
    return vector3d(self.plane.from2d(self.V), frame=self.frame)

  @property
  def V_used(self):
    return np.unique(self.F)

  # the midpoints in the plane of the map, the form the geometry works in
  @property
  def _mid2d(self):
    return (self.V[self.F[:, 0]] + self.V[self.F[:, 1]]) / 2

  # the midpoint of every segment, a direction of the map as MATLAB's is
  @property
  def midPoint(self):
    return vector3d(self.plane.from2d(self._mid2d), frame=self.frame)

  @property
  def segLength(self):
    return np.linalg.norm(self.V[self.F[:, 1]] - self.V[self.F[:, 0]], axis=1)

  def perimeter(self):
    return float(self.segLength.sum())

  # the direction of the walk, as a framed axis of the specimen
  @property
  def direction(self):
    d = self.plane.from2d(self.V[self.F[:, 1]]) - self.plane.from2d(self.V[self.F[:, 0]])
    return vector3d(d, frame=self.frame, antipodal=True)

  # the misorientations of the boundary, which needs one pair of phases in one order
  # throughout, with the grain exchange symmetry where both sides are one phase
  @property
  def misorientation(self):
    if self.isempty:
      raise ValueError("an empty boundary has no misorientations")
    p1, p2 = int(self.phaseId[0, 0]), int(self.phaseId[0, 1])
    if p1 < 0 or p2 < 0:
      raise ValueError("these segments lie on the border of the map")
    if not (np.all(self.phaseId[:, 0] == p1) and np.all(self.phaseId[:, 1] == p2)):
      raise ValueError("the boundary carries more than one pair of phases; select one with hasPhase")
    mori = orientation._fromRotation(self.misrotation.data, (self.CSList[p1], self.CSList[p2]), antipodal=p1 == p2)
    return mori.reshape(*self._shape) if self._shape is not None else mori

  # ------------------------------------------------------------------ chains

  def vertexDegree(self):
    if "degree" not in self._memo:
      self._memo["degree"] = np.bincount(self.F.reshape(-1), minlength=len(self.V)).astype(np.int32)
    return self._memo["degree"]

  # a vertex where any number of segments other than two meet
  @property
  def isJunction(self):
    deg = self.vertexDegree()
    return (deg != 2) & (deg > 0)

  @property
  def junctionId(self):
    return np.flatnonzero(self.isJunction)

  # the chain of every segment, from the walk order and the junctions: a chain breaks
  # where the head of one segment is not the tail of the next, and at a junction
  @property
  def chainId(self):
    if "chainId" not in self._memo:
      n = len(self)
      if n == 0:
        self._memo["chainId"] = np.zeros(0, dtype=np.int32)
      else:
        jct = self.isJunction
        brk = (self.F[1:, 0] != self.F[:-1, 1]) | jct[self.F[1:, 0]]
        self._memo["chainId"] = np.concatenate([[1], 1 + np.cumsum(brk)]).astype(np.int32)
    return self._memo["chainId"]

  @property
  def chainSize(self):
    cid = self.chainId
    return np.bincount(cid)[cid]

  @property
  def isChainStart(self):
    cid = self.chainId
    return np.concatenate([[True], cid[1:] != cid[:-1]]) if len(cid) else np.zeros(0, dtype=bool)

  @property
  def isChainEnd(self):
    cid = self.chainId
    return np.concatenate([cid[1:] != cid[:-1], [True]]) if len(cid) else np.zeros(0, dtype=bool)

  # a chain that closes onto itself, so its walk returns to where it started
  @property
  def isClosed(self):
    cid = self.chainId
    if len(cid) == 0:
      return np.zeros(0, dtype=bool)
    starts, ends = np.flatnonzero(self.isChainStart), np.flatnonzero(self.isChainEnd)
    closed = self.F[ends, 1] == self.F[starts, 0]
    return closed[cid - 1]

  # the length along the chain up to the end of each segment
  @property
  def arcLength(self):
    L = self.segLength
    cid = self.chainId
    cum = np.cumsum(L)
    starts = np.flatnonzero(self.isChainStart)
    before = np.concatenate([[0.0], cum])[starts]
    return cum - before[cid - 1]

  @property
  def chainLength(self):
    cid = self.chainId
    return np.bincount(cid, weights=self.segLength)[cid]

  # the vertices of every chain in walk order, the chains separated by -1
  @property
  def chainV(self):
    out = []
    start = self.isChainStart
    for k in range(len(self)):
      if start[k]:
        if out:
          out.append(-1)
        out.append(self.F[k, 0])
      out.append(self.F[k, 1])
    return np.array(out, dtype=np.int32)

  # the connected component each segment lies in, junctions included, and its size
  @property
  def componentId(self):
    from .segment import components
    n = len(self)
    if n == 0:
      return np.zeros(0, dtype=np.int32)
    # segments sharing a vertex are joined: every vertex links its segments to the first at it
    v = self.F.reshape(-1)
    seg = np.repeat(np.arange(n), 2)
    order = np.argsort(v, kind="stable")
    vs, ss = v[order], seg[order]
    same = vs[1:] == vs[:-1]
    roots = components(n, ss[:-1][same], ss[1:][same])
    _, first, inverse = np.unique(roots, return_index=True, return_inverse=True)
    return (np.argsort(np.argsort(first))[inverse] + 1).astype(np.int32)

  @property
  def componentSize(self):
    cid = self.componentId
    return np.bincount(cid)[cid]

  # the segment before and after each one along its chain: a closed chain wraps its two
  # ends onto each other, an open one clamps them, so a filter along the chain does not
  # spread the undefined end
  def chainNeighbors(self):
    n = len(self)
    prev, nxt = np.arange(-1, n - 1), np.arange(1, n + 1)
    hasPrev, hasNext = np.ones(n, dtype=bool), np.ones(n, dtype=bool)
    if n == 0:
      return prev, nxt, hasPrev, hasNext
    closed = self.isClosed
    for s, e in zip(np.flatnonzero(self.isChainStart), np.flatnonzero(self.isChainEnd)):
      if closed[s]:
        prev[s], nxt[e] = e, s
      else:
        hasPrev[s], hasNext[e] = False, False
        prev[s], nxt[e] = s, e
    return prev, nxt, hasPrev, hasNext

  # ------------------------------------------------------------------ triple points

  # the triple points of the network, of a selection those at a vertex of its segments, with
  # boundaryId into the whole network, as MATLAB's
  @property
  def triplePoints(self):
    if self._root is None:
      if "triplePoints" not in self._memo:
        self._memo["triplePoints"] = triplePoints(self)
      return self._memo["triplePoints"]
    tP = self._root.triplePoints
    return tP[np.isin(tP.vertexId, self.F)]

  # ------------------------------------------------------------------ subsets

  # a subset of the segments; the vertex list is shared and never rewritten
  def __getitem__(self, key):
    if isinstance(key, str) or hasattr(key, "sym"):
      return self._withFirstPhase(self[self.hasPhase(key)], key)
    if isinstance(key, tuple) and len(key) == 2 and (isinstance(key[0], str) or hasattr(key[0], "sym")):
      return self._withFirstPhase(self[self.hasPhase(*key)], key[0])
    sel = np.arange(len(self))[key]
    gB = self._with(F=self.F[sel.reshape(-1)], grainId=self.grainId[sel.reshape(-1)], ebsdId=self.ebsdId[sel.reshape(-1)],
                    phaseId=self.phaseId[sel.reshape(-1)], misrotation=self.misrotation.data[sel.reshape(-1)])
    gB._root = self if self._root is None else self._root
    # a matrix of segment ids, the three arms of triple points, keeps its shape for the misorientations
    if sel.ndim > 1:
      gB._shape = sel.shape
    return gB

  # the segments between the two phases named, in either order, or with one named on either side
  def hasPhase(self, a, b=None):
    # 'indexed' are the segments between two indexed measurements, as MATLAB reads the name;
    # 'notIndexed' those with a not indexed side or none at all, the rim of the scan
    if isinstance(a, str) and b is None and a.lower() in ("indexed", "notindexed") \
       and a.lower() not in [m.lower().replace(" ", "") for m in self.CSList.mineralList]:
      inside = np.all(self.phaseId > 0, axis=1)
      return inside if a.lower() == "indexed" else ~inside
    p = self.CSList.phaseId(a)
    if b is None:
      return (self.phaseId[:, 0] == p) | (self.phaseId[:, 1] == p)
    q = self.CSList.phaseId(b)
    return ((self.phaseId[:, 0] == p) & (self.phaseId[:, 1] == q)) | ((self.phaseId[:, 0] == q) & (self.phaseId[:, 1] == p))

  # the segments with a grain of the list on either side, walked again: a junction of the
  # whole boundary may have lost its third segment, so the chains of the subset are longer;
  # for pairs of ids the segments between the two of a pair, with the row of its pair
  def selectByGrainId(self, ids):
    ids = np.asarray(ids)
    if ids.ndim == 2 and ids.shape[1] == 2:
      M = int(max(ids.max(initial=0), self.grainId.max(initial=0))) + 1
      key = lambda g: np.minimum(g[:, 0], g[:, 1]).astype(np.int64) * M + np.maximum(g[:, 0], g[:, 1])
      pk = key(ids)
      order = np.argsort(pk)
      gB = orderWalk(self[np.isin(key(self.grainId), pk)])
      return gB, order[np.searchsorted(pk[order], key(gB.grainId))]
    return orderWalk(self[np.isin(self.grainId[:, 0], ids) | np.isin(self.grainId[:, 1], ids)])

  # a segment turned round: its two sides swap and the misorientation inverts
  def _flipSegments(self, sel):
    sel = np.asarray(sel, dtype=bool)
    F, gid, eid, pid, mis = self.F.copy(), self.grainId.copy(), self.ebsdId.copy(), self.phaseId.copy(), self.misrotation.data.copy()
    for a in (F, gid, eid, pid):
      a[sel] = a[sel][:, ::-1]
    mis[sel] = _conj(mis[sel])
    return self._with(F=F, grainId=gid, ebsdId=eid, phaseId=pid, misrotation=mis)

  # the named phase put on the first side throughout
  def _withFirstPhase(self, gB, a):
    # 'indexed' and 'notIndexed' name no phase, so no side is put first
    if isinstance(a, str) and a.lower() in ("indexed", "notindexed") \
       and a.lower() not in [m.lower().replace(" ", "") for m in self.CSList.mineralList]:
      return gB
    p = self.CSList.phaseId(a)
    return gB._flipSegments(gB.phaseId[:, 0] != p)

  # swap the two sides of the segments marked, or of all of them, and walk the chains again
  def flip(self, ind=None):
    sel = np.ones(len(self), dtype=bool) if ind is None else np.zeros(len(self), dtype=bool)
    if ind is not None:
      sel[ind] = True
    return orderWalk(self._flipSegments(sel))

  # the segments whose misorientation is the given one up to a threshold
  def isTwinning(self, mori, threshold=5 * degree):
    ind = np.zeros(len(self), dtype=bool)
    p1, p2 = self.CSList.phaseId(mori.frameA), self.CSList.phaseId(mori.frameB)
    sel = np.flatnonzero(self.hasPhase(p1, p2))
    if len(sel) == 0:
      return ind
    sub = self[sel]
    m = mori if sub.phaseId[0, 0] == p1 else mori.inv()
    ind[sel] = sub.misorientation.angle(m) < threshold
    return ind

  # ------------------------------------------------------------------ along the chain

  # the signed curvature of every segment: the Menger curvature through the midpoints of a
  # segment and its two neighbours along the chain, smoothed by a binomial filter along
  # it; positive where the boundary bulges into grainId[:, 1]
  def curvature(self, n=50):
    nF = len(self)
    T = self.V.dtype
    kappa = np.full(nF, np.nan, dtype=T)
    if nF == 0:
      return kappa
    mp = self._mid2d
    prev, nxt, hasPrev, hasNext = self.chainNeighbors()
    ok = hasPrev & hasNext
    if not ok.any():
      return kappa
    a, b, c = mp - mp[prev], mp[nxt] - mp[prev], mp - mp[nxt]
    k = np.zeros(nF, dtype=T)
    with np.errstate(invalid="ignore", divide="ignore"):
      k[ok] = (2 * _cross2(a, b) / np.sqrt(np.sum(a * a, 1) * np.sum(b * b, 1) * np.sum(c * c, 1)))[ok]
    for _ in range(n):
      k = (k[prev] + 2 * k + k[nxt]) / 4
    kappa[ok] = k[ok]
    return kappa

  # the direction of a segment averaged over 2n + 1 of them along the chain: the chord
  # across a window, clamped at the end of an open chain and shorter than a closed one
  def calcMeanDirection(self, n=1):
    nF = len(self)
    if nF == 0:
      return vector3d(np.zeros((0, 3)), frame=self.frame, antipodal=True)
    cid = self.chainId
    first, last = np.flatnonzero(self.isChainStart), np.flatnonzero(self.isChainEnd)
    closed = self.isClosed
    a, b = first[cid - 1], last[cid - 1]
    length = b - a + 1
    i = np.arange(nF)
    m = np.maximum(0, np.minimum(n, (length - 2) // 2))
    back = np.where(closed, a + np.mod(i - m - a, length), np.maximum(i - n, a))
    fwd = np.where(closed, a + np.mod(i + m - a, length), np.minimum(i + n, b))
    d = self.plane.from2d(self.V[self.F[fwd, 1]]) - self.plane.from2d(self.V[self.F[back, 0]])
    return vector3d(d, frame=self.frame, antipodal=True)

  # the angle of every segment to a direction, or to the segments of another boundary
  def angle(self, x):
    other = x.direction if isinstance(x, grainBoundary) else x
    return self.direction.angle(other, noSymmetry=True)

  # ------------------------------------------------------------------ the line and the polygon

  def _as2d(self, p):
    p = p.data if isinstance(p, vector3d) else np.asarray(p, dtype=float)
    return p if p.shape[-1] == 2 else self.plane.to2d(p)

  # where the line from p to q crosses each segment, MATLAB's [x, y] = intersect(gB, xy1, xy2):
  # the coordinates per segment, NaN where it is not crossed; segLength=True adds the
  # intercepts between successive crossings along the line
  def intersect(self, p, q, segLength=False):
    a, b = self._as2d(p).reshape(2), self._as2d(q).reshape(2)
    ab = b - a
    v1, v2 = self.V[self.F[:, 0]], self.V[self.F[:, 1]]
    d = v2 - v1
    den = _cross2(ab[None, :], d)
    with np.errstate(invalid="ignore", divide="ignore"):
      t = _cross2(v1 - a, d) / den
      s = _cross2(v1 - a, ab[None, :]) / den
    # MATLAB's tolerance, so a line through a vertex cuts both segments there
    tol = 1e-5
    hit = (den != 0) & (t >= -tol) & (t <= 1 + tol) & (s >= -tol) & (s <= 1 + tol)
    P = self.plane.from2d(a + np.where(hit, t, np.nan)[:, None] * ab)
    x, y = P[:, 0], P[:, 1]
    if not segLength:
      return x, y
    return x, y, np.diff(np.sort(t[hit])) * np.linalg.norm(ab)

  # the segments whose midpoint lies inside a polygon (N, 2), or the rectangle [x, y, w, h]
  def inpolygon(self, region):
    from .ebsd import _inPolygon
    region = np.asarray(region, dtype=float)
    mp = self._mid2d
    if region.ndim == 1:
      x, y, w, h = region
      return (x <= mp[:, 0]) & (mp[:, 0] <= x + w) & (y <= mp[:, 1]) & (mp[:, 1] <= y + h)
    return _inPolygon(mp[:, 0], mp[:, 1], self._as2d(region).reshape(-1, 2), includeEdge=False)

  # ------------------------------------------------------------------ display

  def __repr__(self):
    lines = [f"grainBoundary ({self.frame.char()})"]
    if self.isempty:
      return lines[0] + "\n  the boundary is empty"
    lines.append(f"  size: {len(self)} segments, {int(self.chainId.max())} chains")
    key = np.sort(self.phaseId, axis=1)
    pairs, inverse = np.unique(key, axis=0, return_inverse=True)
    L = np.bincount(inverse.reshape(-1), weights=self.segLength)
    counts = np.bincount(inverse.reshape(-1))
    # the side outside the map is notIndexed, as MATLAB names it
    name = lambda p: "notIndexed" if p < 0 else self.CSList[int(p)].mineral
    # the length rounded in the scan unit, as MATLAB writes it
    unit = "µm" if self.scanUnit == "um" else self.scanUnit
    rows = [f"  {c:>8}  {f'{xnum2str(l)} {unit}':>10}  {name(p[0])}  {name(p[1])}" for p, c, l in zip(pairs, counts, L)]
    lines.append("  segments      length  mineral 1  mineral 2")
    lines += rows
    return "\n".join(lines)


# ------------------------------------------------------------------ triple points

class triplePointList:

  def __init__(self, V, grainId, phaseId, boundaryId, angles, CSList, plane, frame, vertexId=None):
    self.V, self.grainId, self.phaseId, self.boundaryId, self.angles = V, grainId, phaseId, boundaryId, angles
    self.CSList, self.plane, self.frame = CSList, plane, frame
    # the index of each point into the vertex list of its boundary
    self.vertexId = np.full(len(grainId), -1, dtype=np.int64) if vertexId is None else vertexId

  def __len__(self):
    return len(self.grainId)

  @property
  def size(self):
    return len(self.grainId)

  @property
  def isempty(self):
    return len(self.grainId) == 0

  @property
  def allV(self):
    return vector3d(self.plane.from2d(self.V), frame=self.frame)

  # the points with the phases named among their three grains, a name repeated asking for that many
  def hasPhase(self, *names):
    ids = [self.CSList.phaseId(n) for n in names]
    out = np.ones(len(self), dtype=bool)
    for p in set(ids):
      out &= np.sum(self.phaseId == p, axis=1) >= ids.count(p)
    return out

  # tP('Iron bcc', 'Iron bcc', 'Iron bcc'): the points of those phases, MATLAB's call form
  def __call__(self, *names):
    return self[self.hasPhase(*names)]

  def __getitem__(self, key):
    if isinstance(key, str):
      return self[self.hasPhase(key)]
    if isinstance(key, tuple) and key and isinstance(key[0], str):
      return self[self.hasPhase(*key)]
    sel = np.arange(len(self))[key].reshape(-1)
    return triplePointList(self.V[sel], self.grainId[sel], self.phaseId[sel], self.boundaryId[sel], self.angles[sel], self.CSList,
                           self.plane, self.frame, self.vertexId[sel])

  def __repr__(self):
    if self.isempty:
      return f"triplePointList ({self.frame.char()})\n  no triple points"
    key = np.sort(self.phaseId, axis=1)
    trip, counts = np.unique(key, axis=0, return_counts=True)
    name = lambda p: "outside" if p < 0 else self.CSList[int(p)].mineral
    rows = [f"  {c:>6}  " + "  ".join(name(p) for p in t) for t, c in zip(trip, counts)]
    return f"triplePointList ({self.frame.char()})\n  size: {len(self)}\n  points  mineral 1  mineral 2  mineral 3\n" + "\n".join(rows)


# the vertices where exactly three segments of three grains meet, with the grains, the
# segments and the angles the segments enclose
def triplePoints(gB):
  T = gB.V.dtype
  deg = gB.vertexDegree()
  v = gB.F.reshape(-1)
  seg = np.repeat(np.arange(len(gB)), 2)
  order = np.argsort(v, kind="stable")
  vs, ss = v[order], seg[order]
  starts = np.searchsorted(vs, np.arange(len(gB.V) + 1))
  Vs, rows, segs, angs, vids = [], [], [], [], []
  for vert in np.flatnonzero(deg == 3):
    at = ss[starts[vert]:starts[vert + 1]]
    g = np.unique(gB.grainId[at])
    # three segments of only two grains is a cut, and a junction on the rim of the map
    # is not a point where three grains meet
    if len(g) != 3 or g.min() <= 0:
      continue
    Vs.append(gB.V[vert])
    vids.append(vert)
    rows.append(g)
    segs.append(at)
    far = np.where(gB.F[at, 0] == vert, gB.F[at, 1], gB.F[at, 0])
    o = gB.V[far] - gB.V[vert]
    th = np.sort(np.arctan2(o[:, 1], o[:, 0]))
    angs.append([th[1] - th[0], th[2] - th[1], 2 * np.pi - (th[2] - th[0])])
  n = len(Vs)
  V = np.array(Vs, dtype=T).reshape(n, 2)
  gid = np.array(rows, dtype=np.int32).reshape(n, 3)
  bid = np.array(segs, dtype=np.int32).reshape(n, 3)
  ang = np.array(angs, dtype=T).reshape(n, 3)
  # the phase of a grain is read off the segment it borders
  pid = np.full((n, 3), OUTSIDE, dtype=np.int8)
  for i in range(n):
    for c in range(3):
      hit = np.flatnonzero(gB.grainId[bid[i]] == gid[i, c])
      k, side = divmod(hit[0], 2)
      pid[i, c] = gB.phaseId[bid[i, k], side]
  return triplePointList(V, gid, pid, bid, ang, gB.CSList, gB.plane, gB.frame, np.array(vids, dtype=np.int64))


# ------------------------------------------------------------------ the walk

# decompose an edge list into maximal chains and walk each of them: the permutation that
# brings the segments into walk order and, per segment, whether the walk enters it at its
# first vertex. A chain is entered from its lowest terminal half edge, a closed chain is
# cut at its lowest vertex, and the chains follow the order their first segment is reached
def _chainOrderLoop(F, deg, offset, at, perm, entry):
  n = F.shape[0]
  used = np.zeros(n, dtype=np.bool_)
  count = 0
  # every chain that ends somewhere: start at the junctions, lowest first
  for pas in range(2):
    for start in range(n if pas == 1 else offset.shape[0] - 1):
      if pas == 0:
        v0 = start
        if deg[v0] == 2:
          continue
        lo, hi = offset[v0], offset[v0 + 1]
      else:
        if used[start]:
          continue
        v0 = min(F[start, 0], F[start, 1])
        lo, hi = 0, 1
      for a in range(lo, hi):
        k = at[a] if pas == 0 else start
        if used[k]:
          continue
        v = v0
        while True:
          used[k] = True
          perm[count] = k
          entry[k] = 1 if F[k, 0] == v else 2
          count += 1
          w = F[k, 1] if F[k, 0] == v else F[k, 0]
          if deg[w] != 2:
            break
          nxt = -1
          for b in range(offset[w], offset[w + 1]):
            j = at[b]
            if j != k and not used[j]:
              nxt = j
              break
          if nxt < 0:
            break
          k, v = nxt, w
  return count


def chainOrder(F, nV):
  F = np.asarray(F, dtype=np.int32).reshape(-1, 2)
  n = len(F)
  if n == 0:
    return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
  deg = np.bincount(F.reshape(-1), minlength=nV).astype(np.int32)
  offset = np.concatenate([[0], np.cumsum(deg)]).astype(np.int64)
  # contiguous adjacency, in increasing segment order at each vertex
  v = F.reshape(-1)
  order = np.argsort(v, kind="stable")
  at = np.repeat(np.arange(n, dtype=np.int32), 2)[order]
  perm = np.zeros(n, dtype=np.int32)
  entry = np.zeros(n, dtype=np.int32)
  loop = _compiledChainOrder()
  loop(F, deg, offset, at, perm, entry)
  return perm, entry


def _compiledChainOrder():
  nb = _jit()
  if not nb:
    return _chainOrderLoop
  if "chainOrder" not in _COMPILED:
    _COMPILED["chainOrder"] = nb.njit(cache=True)(_chainOrderLoop)
  return _COMPILED["chainOrder"]


# ------------------------------------------------------------------ construction

# the boundary of a segmentation: the faces it marks as outer, or inner, in walk order.
# The two sides are listed by phase and then by grain, so a pair of phases always appears
# in the same order and the outside of the map comes first; the site of the first grain,
# or the mirror of the other one where it is outside, says which side is left
def buildBoundary(c, seg, ebsd, inner=False):
  keep = np.flatnonzero(seg.isInner if inner else seg.isOuter)
  sites = c.sites
  T = c.V.dtype
  s1, s2 = c.cells[keep, 0], c.cells[keep, 1]
  p1, p2 = sites.pixel[s1], sites.pixel[s2]
  gid = seg.grainId
  ph = ebsd.phaseId.reshape(-1)
  g1 = np.where(p1 >= 0, gid[np.maximum(p1, 0)], 0)
  g2 = np.where(p2 >= 0, gid[np.maximum(p2, 0)], 0)
  ph1 = np.where(p1 >= 0, ph[np.maximum(p1, 0)], OUTSIDE).astype(np.int8)
  ph2 = np.where(p2 >= 0, ph[np.maximum(p2, 0)], OUTSIDE).astype(np.int8)
  swap = (ph1 > ph2) | ((ph1 == ph2) & (g1 > g2))
  s1, s2 = np.where(swap, s2, s1), np.where(swap, s1, s2)
  p1, p2 = np.where(swap, p2, p1), np.where(swap, p1, p2)
  g1, g2 = np.where(swap, g2, g1), np.where(swap, g1, g2)
  ph1, ph2 = np.where(swap, ph2, ph1), np.where(swap, ph1, ph2)
  rot = ebsd.rotations.data.reshape(-1, 4)
  both = (p1 >= 0) & (p2 >= 0)
  mori = np.full((len(keep), 4), np.nan, dtype=T)
  mori[both] = qmul(_conj(rot[p2[both]]), rot[p1[both]])
  F = c.F[keep]
  mid = (c.V[F[:, 0]] + c.V[F[:, 1]]) / 2
  leftOf = np.where((p1 >= 0)[:, None], sites.pos[s1], 2 * mid - sites.pos[s2])
  # the two measurements are named by their id, which outlives a selection, not by their place
  ids = np.asarray(ebsd.id).reshape(-1)
  e1 = np.where(p1 >= 0, ids[np.maximum(p1, 0)], NO_PIXEL)
  e2 = np.where(p2 >= 0, ids[np.maximum(p2, 0)], NO_PIXEL)
  gB = grainBoundary(c.V, F, np.stack([g1, g2], 1), np.stack([e1, e2], 1), np.stack([ph1, ph2], 1), mori, ebsd.CSList, c.plane, ebsd.frame,
                     scanUnit=ebsd.scanUnit)
  return orderWalk(gB, leftOf)


# bring the segments into walk order and orient every chain so that the grain in
# grainId[:, 0] lies to its left. With leftOf the boundary is being built and every chain
# is turned as a whole until the point of its first grain lies on the left; without it
# the boundary already holds the invariant on every segment, so a segment the walk enters
# from its head is turned round on both sides, and a chain is turned by the majority
def orderWalk(gB, leftOf=None):
  perm, entry = chainOrder(gB.F, len(gB.V))
  if len(perm) == 0:
    return gB
  head = entry[perm] == 1
  F = np.where(head[:, None], gB.F[perm], gB.F[perm][:, ::-1])
  out = gB._with(F=F, grainId=gB.grainId[perm], ebsdId=gB.ebsdId[perm], phaseId=gB.phaseId[perm], misrotation=gB.misrotation.data[perm])
  consistent = leftOf is None
  if consistent:
    out = out._flipSegments(~head)._with(F=F)
  cid = out.chainId
  starts = np.flatnonzero(out.isChainStart)
  if leftOf is None:
    kept = np.bincount(cid, weights=head)[1:]
    total = np.bincount(cid)[1:]
    flip = 2 * kept < total
  else:
    v1 = out.V[out.F[starts, 0]]
    d = out.V[out.F[starts, 1]] - v1
    flip = _cross2(d, leftOf[perm[starts]] - v1) < 0
  if not flip.any():
    return out
  return reverseChains(out, cid, flip, swap=consistent)


# reverse the chains marked, in the order of their rows and within each segment, and with
# swap on both sides of every segment turned
def reverseChains(gB, cid, flip, swap=False):
  n = len(gB)
  order = np.arange(n)
  starts = np.flatnonzero(np.concatenate([[True], cid[1:] != cid[:-1]]))
  ends = np.concatenate([starts[1:], [n]])
  for s, e in zip(starts, ends):
    if flip[cid[s] - 1]:
      order[s:e] = order[s:e][::-1]
  rows = flip[cid[order] - 1]
  F = gB.F[order]
  F[rows] = F[rows][:, ::-1]
  out = gB._with(F=F, grainId=gB.grainId[order], ebsdId=gB.ebsdId[order], phaseId=gB.phaseId[order], misrotation=gB.misrotation.data[order])
  return out._flipSegments(rows)._with(F=F) if swap else out


__all__ = ["grainBoundary", "triplePointList", "triplePoints", "chainOrder"]
