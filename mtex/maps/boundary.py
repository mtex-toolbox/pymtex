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

from typing import TYPE_CHECKING
import numpy as np

from ..geometry.frames import referenceFrame
from ..geometry.display import aligned, frameLabel, textTable, xnum2str
from ..geometry.kernels import qconj, qmul
from ..geometry.orientation import orientation
from ..geometry.quaternion import rotation
from ..geometry.units import degree
from ..geometry.vector3d import vector3d
from .decomposition import NO_PIXEL, _gather
from .grains import _pairKey
from .kernels import _compiledLoop, adjacencyNumba
from .phaselist import phaseTable
from .polygon import _cross2

OUTSIDE = -1
prange = range


# the first row of every run of equal values of a sorted chain id
def _runStarts(cid):
  return np.flatnonzero(np.concatenate([[True], cid[1:] != cid[:-1]]))


# whether a selection name is 'indexed' or 'notIndexed' rather than a phase called so, its
# spaces ignored
def _namesNoPhase(a, CSList):
  return isinstance(a, str) and a.lower() in ("indexed", "notindexed") \
    and a.lower() not in [m.lower().replace(" ", "") for m in CSList.mineralList]


class grainBoundary:
  """the grain boundary of a map: the segments F between the vertices V in walk order, each with
  its two grains, pixels, phases and misrotation, the grain in grainId[:, 0] to the left of the
  walk; chains, junctions and triple points are derived from that order

    gB = grains.boundary
    gB['Forsterite', 'Enstatite']
    gB.segLength, gB.misorientation"""

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
    """the number of segments"""
    return len(self.F)

  @property
  def isempty(self):
    """whether the boundary has no segments"""
    return len(self.F) == 0

  # ------------------------------------------------------------------ geometry

  @property
  def allV(self):
    """the vertices as framed points of the specimen, all of them and the ones in use"""
    return vector3d(self.plane.from2d(self.V), frame=self.frame)

  @property
  def V_used(self):
    """the indices of the vertices the segments use, sorted"""
    return np.unique(self.F)

  # the midpoints in the plane of the map, the form the geometry works in
  @property
  def _mid2d(self):
    return (self.V[self.F[:, 0]] + self.V[self.F[:, 1]]) / 2

  @property
  def midPoint(self):
    """the midpoint of every segment, a direction of the map as MATLAB's is"""
    return vector3d(self.plane.from2d(self._mid2d), frame=self.frame)

  @property
  def segLength(self):
    """the length of every segment"""
    return np.linalg.norm(self.V[self.F[:, 1]] - self.V[self.F[:, 0]], axis=1)

  def perimeter(self):
    """the total length of the boundary, the sum of the segment lengths"""
    return float(self.segLength.sum())

  @property
  def direction(self):
    """the direction of the walk, as a framed axis of the specimen"""
    d = self.plane.from2d(self.V[self.F[:, 1]]) - self.plane.from2d(self.V[self.F[:, 0]])
    return vector3d(d, frame=self.frame, antipodal=True)

  @property
  def misorientation(self):
    """the misorientations of the boundary, which needs one pair of phases in one order
    throughout, with the grain exchange symmetry where both sides are one phase"""
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
    """the number of segments meeting at every vertex"""
    if "degree" not in self._memo:
      self._memo["degree"] = np.bincount(self.F.reshape(-1), minlength=len(self.V)).astype(np.int32)
    return self._memo["degree"]

  @property
  def isJunction(self):
    """a vertex where any number of segments other than two meet"""
    deg = self.vertexDegree()
    return (deg != 2) & (deg > 0)

  @property
  def junctionId(self):
    """the indices of the junctions, the vertices where a number of segments other than two meet"""
    return np.flatnonzero(self.isJunction)

  @property
  def chainId(self):
    """the chain of every segment, counted from 0, from the walk order and the junctions: a chain
    breaks where the head of one segment is not the tail of the next, and at a junction (MATLAB
    counts the chains from 1)"""
    if "chainId" not in self._memo:
      n = len(self)
      if n == 0:
        self._memo["chainId"] = np.zeros(0, dtype=np.int32)
      else:
        jct = self.isJunction
        brk = (self.F[1:, 0] != self.F[:-1, 1]) | jct[self.F[1:, 0]]
        self._memo["chainId"] = np.concatenate([[0], np.cumsum(brk)]).astype(np.int32)
    return self._memo["chainId"]

  @property
  def chainSize(self):
    """the number of segments of the chain of every segment"""
    cid = self.chainId
    return np.bincount(cid)[cid]

  @property
  def isChainStart(self):
    """whether a segment is the first of its chain"""
    cid = self.chainId
    return np.concatenate([[True], cid[1:] != cid[:-1]]) if len(cid) else np.zeros(0, dtype=bool)

  @property
  def isChainEnd(self):
    """whether a segment is the last of its chain"""
    cid = self.chainId
    return np.concatenate([cid[1:] != cid[:-1], [True]]) if len(cid) else np.zeros(0, dtype=bool)

  @property
  def isClosed(self):
    """a chain that closes onto itself, so its walk returns to where it started"""
    cid = self.chainId
    if len(cid) == 0:
      return np.zeros(0, dtype=bool)
    starts, ends = np.flatnonzero(self.isChainStart), np.flatnonzero(self.isChainEnd)
    closed = self.F[ends, 1] == self.F[starts, 0]
    return closed[cid]

  @property
  def arcLength(self):
    """the length along the chain up to the end of each segment"""
    L = self.segLength
    cid = self.chainId
    cum = np.cumsum(L)
    starts = np.flatnonzero(self.isChainStart)
    before = np.concatenate([[0.0], cum])[starts]
    return cum - before[cid]

  @property
  def chainLength(self):
    """the length of the chain of every segment"""
    cid = self.chainId
    return np.bincount(cid, weights=self.segLength)[cid]

  @property
  def chainV(self):
    """the vertices of every chain in walk order, the chains separated by -1"""
    n = len(self)
    if n == 0:
      return np.zeros(0, dtype=np.int32)
    # the end of segment k after k segments, the first vertex of every chain so far and the
    # separators between them: 2 c - 1 places for the c-th chain
    start = self.isChainStart
    c = np.cumsum(start)
    out = np.empty(n + 2 * c[-1] - 1, dtype=np.int32)
    at = np.arange(n) + 2 * c - 1
    out[at] = self.F[:, 1]
    first = np.flatnonzero(start)
    out[at[first] - 1] = self.F[first, 0]
    out[at[first[1:]] - 2] = -1
    return out

  @property
  def componentId(self):
    """the connected component each segment lies in, junctions included, and its size"""
    from .segment import components, _firstAppearanceLabels
    n = len(self)
    if n == 0:
      return np.zeros(0, dtype=np.int32)
    # segments sharing a vertex are joined: every vertex links its segments to the first at it
    v = self.F.reshape(-1)
    seg = np.repeat(np.arange(n), 2)
    order = np.argsort(v, kind="stable")
    vs, ss = v[order], seg[order]
    same = vs[1:] == vs[:-1]
    return _firstAppearanceLabels(components(n, ss[:-1][same], ss[1:][same]))[0]

  @property
  def componentSize(self):
    """the number of segments of the connected component of every segment"""
    cid = self.componentId
    return np.bincount(cid)[cid]

  def chainNeighbors(self):
    """the segment before and after each one along its chain: a closed chain wraps its two
    ends onto each other, an open one clamps them, so a filter along the chain does not
    spread the undefined end"""
    n = len(self)
    prev, nxt = np.arange(-1, n - 1), np.arange(1, n + 1)
    hasPrev, hasNext = np.ones(n, dtype=bool), np.ones(n, dtype=bool)
    if n == 0:
      return prev, nxt, hasPrev, hasNext
    s, e = np.flatnonzero(self.isChainStart), np.flatnonzero(self.isChainEnd)
    closed = self.isClosed[s]
    prev[s] = np.where(closed, e, s)
    nxt[e] = np.where(closed, s, e)
    hasPrev[s[~closed]] = False
    hasNext[e[~closed]] = False
    return prev, nxt, hasPrev, hasNext

  # ------------------------------------------------------------------ triple points

  @property
  def triplePoints(self):
    """the triple points of the network, of a selection those at a vertex of its segments, with
    boundaryId into the whole network, as MATLAB's"""
    if self._root is None:
      if "triplePoints" not in self._memo:
        self._memo["triplePoints"] = triplePoints(self)
      return self._memo["triplePoints"]
    tP = self._root.triplePoints
    return tP[np.isin(tP.vertexId, self.F)]

  # ------------------------------------------------------------------ subsets

  # a subset of the segments; the vertex list is shared and never rewritten
  # the names gB['...'] and gB['...', '...'] take, which IPython completes inside the brackets
  def _ipython_key_completions_(self):
    return self.CSList.selectionNames(pairs=True)

  def __getitem__(self, key):
    if isinstance(key, (str, referenceFrame)):
      return self._withFirstPhase(self[self.hasPhase(key)], key)
    if isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], (str, referenceFrame)):
      return self._withFirstPhase(self[self.hasPhase(*key)], key[0])
    sel = np.arange(len(self))[key]
    k = sel.reshape(-1)
    gB = self._with(F=self.F[k], grainId=self.grainId[k], ebsdId=self.ebsdId[k], phaseId=self.phaseId[k], misrotation=self.misrotation.data[k])
    gB._root = self if self._root is None else self._root
    # a matrix of segment ids, the three arms of triple points, keeps its shape for the misorientations
    if sel.ndim > 1:
      gB._shape = sel.shape
    return gB

  def hasPhase(self, a, b=None):
    """the segments between the two phases named, in either order, or with one named on either
    side; MATLAB's grainBoundary.hasPhase

      ind = gB.hasPhase('Forsterite')                # a Forsterite grain on either side
      ind = gB.hasPhase('Forsterite', 'Enstatite')   # between Forsterite and Enstatite
      ind = gB.hasPhase('indexed')                   # between two indexed measurements

    A phase is named by its mineral or its phase id. A not indexed phase includes the rim of
    the scan, as MATLAB's hasPhaseId: gB.hasPhase('notIndexed') takes the segments with a not
    indexed side or with the rim on one, and gB.hasPhase('notIndexed', 'Forsterite') the rim
    of the Forsterite grains too. Without a phase of that name in the list, 'indexed' takes the
    segments between two indexed measurements and 'notIndexed' the others. gB['Forsterite']
    selects by this.

    Input
      gB - grainBoundary
      a - the first phase, a mineral name or a phase id
    Output
      ind - a boolean per segment
    Keywords
      b - the second phase; the segments with a on either side when not given
    See also
      grainBoundary.flip, grainBoundary.isTwinning"""
    # 'indexed' are the segments between two indexed measurements, as MATLAB reads the name;
    # 'notIndexed' those with a not indexed side or none at all, the rim of the scan
    if b is None and _namesNoPhase(a, self.CSList):
      inside = np.all(self.phaseId > 0, axis=1)
      return inside if a.lower() == "indexed" else ~inside
    p = self.CSList.phaseId(a)
    # the sides of a phase; a not indexed phase takes the outer rim of the scan too
    def side(p):
      on = self.phaseId == p
      return on | (self.phaseId < 0) if not self.CSList.isIndexed(p) else on
    if b is None:
      return side(p).any(axis=1)
    q = self.CSList.phaseId(b)
    if p == q:
      return np.all(self.phaseId == p, axis=1)
    sp, sq = side(p), side(q)
    return (sp[:, 0] & sq[:, 1]) | (sq[:, 0] & sp[:, 1])

  @classmethod
  def cat(cls, *parts):
    """the segments of several parts of one boundary, MATLAB's [gB1, gB2]: the parts share the
    vertices of the boundary they were taken from; a segment in more than one part is kept
    once, and the chains are walked again

      gB = grainBoundary.cat(gB1, gB2)
      gB = cat(gB['Forsterite', 'Forsterite'], gB['Forsterite', 'Enstatite'])"""
    parts = [p for p in parts if p is not None]
    first = parts[0]
    for p in parts[1:]:
      if p.V is not first.V and not (np.shape(p.V) == np.shape(first.V) and np.array_equal(p.V, first.V)):
        raise ValueError("boundaries are joined as parts of one boundary, over the same vertices")
      if p.CSList is not first.CSList or p.frame is not first.frame:
        raise ValueError("boundaries are joined as parts of one boundary, with the same phases and frame")
    F = np.concatenate([p.F for p in parts])
    # the first copy of a segment, by its two vertices in either order
    _, keep = np.unique(np.sort(F, axis=1), axis=0, return_index=True)
    keep = np.sort(keep)
    gB = first._with(F=F[keep], grainId=np.concatenate([p.grainId for p in parts])[keep],
                     ebsdId=np.concatenate([p.ebsdId for p in parts])[keep],
                     phaseId=np.concatenate([p.phaseId for p in parts])[keep],
                     misrotation=np.concatenate([p.misrotation.data.reshape(-1, 4) for p in parts])[keep])
    gB._root = first._root if first._root is not None else first
    return orderWalk(gB)

  def selectByGrainId(self, ids):
    """the segments with a grain of the list on either side, walked again: a junction of the
    whole boundary may have lost its third segment, so the chains of the subset are longer;
    for pairs of ids the segments between the two of a pair, with the row of its pair

      gBSub = gB.selectByGrainId([1, 2, 3])         # the boundaries of these grains
      gBSub, pairId = gB.selectByGrainId(pairs)      # pairs (n, 2): gBSub.grainId is pairs[pairId] up to the order"""
    ids = np.asarray(ids)
    if ids.ndim == 2 and ids.shape[1] == 2:
      M = int(max(ids.max(initial=0), self.grainId.max(initial=0))) + 1
      pk = _pairKey(ids, M)
      order = np.argsort(pk)
      gB = orderWalk(self[np.isin(_pairKey(self.grainId, M), pk)])
      return gB, order[np.searchsorted(pk[order], _pairKey(gB.grainId, M))]
    return orderWalk(self[np.isin(self.grainId[:, 0], ids) | np.isin(self.grainId[:, 1], ids)])

  # a segment turned round: its two sides swap and the misorientation inverts
  def _flipSegments(self, sel):
    sel = np.asarray(sel, dtype=bool)
    F, gid, eid, pid, mis = self.F.copy(), self.grainId.copy(), self.ebsdId.copy(), self.phaseId.copy(), self.misrotation.data.copy()
    for a in (F, gid, eid, pid):
      a[sel] = a[sel][:, ::-1]
    mis[sel] = qconj(mis[sel])
    return self._with(F=F, grainId=gid, ebsdId=eid, phaseId=pid, misrotation=mis)

  # the named phase put on the first side throughout
  def _withFirstPhase(self, gB, a):
    # 'indexed' and 'notIndexed' name no phase, so no side is put first
    if _namesNoPhase(a, self.CSList):
      return gB
    p = self.CSList.phaseId(a)
    return gB._flipSegments(gB.phaseId[:, 0] != p)

  def flip(self, ind=None):
    """swap the two sides of the segments marked, or of all of them, and walk the chains again;
    MATLAB's grainBoundary.flip

      gB = gB.flip()                                       # every segment
      gB = gB.flip(gB.grainId[:, 1] == grains[k].id)       # the grain k on the first side throughout

    The two grains, pixels and phases of a segment swap and its misorientation inverts; the
    segment is walked the other way, so the grain in grainId[:, 0] stays on its left.

    Input
      gB - grainBoundary
    Output
      gB - grainBoundary
    Keywords
      ind - the segments to flip, booleans or indices; all of them when not given
    See also
      grainBoundary.curvature, grainBoundary.hasPhase"""
    sel = np.ones(len(self), dtype=bool) if ind is None else np.zeros(len(self), dtype=bool)
    if ind is not None:
      sel[ind] = True
    return orderWalk(self._flipSegments(sel))

  def isTwinning(self, mori, threshold=5 * degree):
    """the segments whose misorientation is the given one up to a threshold; MATLAB's
    grainBoundary.isTwinning

      ind = gB.isTwinning(mori)                  # within 5 degrees of the misorientation
      ind = gB.isTwinning(mori, 2*degree)        # within 2 degrees
      ind = gB.isTwinning(CSL(3, ebsd['Iron'].CS))  # the Sigma 3 boundaries of a cubic phase

    Only the segments between the two phases of the misorientation, its frames, are compared;
    a segment with them the other way round is compared with the inverse. The angle is the
    symmetry reduced misorientation angle.

    Input
      gB - grainBoundary
      mori - the misorientation, an orientation between two crystal frames
    Output
      ind - a boolean per segment
    Keywords
      threshold - the largest misorientation angle to mori
    See also
      CSL, grainBoundary.hasPhase"""
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

  def curvature(self, n=50):
    """the signed curvature of every segment: the Menger curvature through the midpoints of a
    segment and its two neighbours along the chain, smoothed by a binomial filter along
    it; positive where the boundary bulges into grainId[:, 1]; MATLAB's grainBoundary.curvature

      kappa = gB.curvature()      # smoothed 50 times
      kappa = gB.curvature(2)     # smoothed twice

    The segments are stored in walk order with the grain grainId[:, 0] on the left, so the sign
    says into which grain the boundary bulges. To read it as convex or concave with respect to
    one grain, put that grain first throughout, gB.flip(gB.grainId[:, 1] == id). A segment at
    the end of an open chain has a neighbour on one side only and is NaN, as is a chain of one
    segment; a closed chain wraps around.

    Input
      gB - grainBoundary
    Output
      kappa - one over the radius of the fitted circle, in the units of the map, per segment
    Keywords
      n - the number of passes of the binomial filter along the chain
    See also
      grainBoundary.flip, grainBoundary.calcMeanDirection"""
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

  def calcMeanDirection(self, n=1):
    """the direction of a segment averaged over 2n + 1 of them along the chain: the chord
    across a window, clamped at the end of an open chain and shorter than a closed one;
    MATLAB's grainBoundary.calcMeanDirection

      d = gB.calcMeanDirection()     # across three segments
      d = gB.calcMeanDirection(2)    # across five

    The same as gB.direction but over a window of segments, which averages out the staircase
    of a pixel grid. The window never crosses a junction.

    Input
      gB - grainBoundary
    Output
      d - vector3d, antipodal, one per segment, not normalised
    Keywords
      n - the number of neighbouring segments on either side
    See also
      grainBoundary.curvature, grainBoundary.angle"""
    nF = len(self)
    if nF == 0:
      return vector3d(np.zeros((0, 3)), frame=self.frame, antipodal=True)
    cid = self.chainId
    first, last = np.flatnonzero(self.isChainStart), np.flatnonzero(self.isChainEnd)
    closed = self.isClosed
    a, b = first[cid], last[cid]
    length = b - a + 1
    i = np.arange(nF)
    m = np.maximum(0, np.minimum(n, (length - 2) // 2))
    back = np.where(closed, a + np.mod(i - m - a, length), np.maximum(i - n, a))
    fwd = np.where(closed, a + np.mod(i + m - a, length), np.minimum(i + n, b))
    d = self.plane.from2d(self.V[self.F[fwd, 1]]) - self.plane.from2d(self.V[self.F[back, 0]])
    return vector3d(d, frame=self.frame, antipodal=True)

  def angle(self, x):
    """the angle of every segment to a direction, or to the segments of another boundary, in
    [0, pi/2], the segments having no sense

      omega = gB.angle(vector3d.X)
      omega = angle(gB1, gB2)"""
    other = x.direction if isinstance(x, grainBoundary) else x
    return self.direction.angle(other, symmetry=False)

  # ------------------------------------------------------------------ the line and the polygon

  def _as2d(self, p):
    p = p.data if isinstance(p, vector3d) else np.asarray(p, dtype=float)
    return p if p.shape[-1] == 2 else self.plane.to2d(p)

  def intersect(self, p, q, segLength=False):
    """where the line from p to q crosses each segment, MATLAB's [x, y] = intersect(gB, xy1, xy2):
    the coordinates per segment, NaN where it is not crossed; segLength=True adds the
    intercepts between successive crossings along the line

      x, y = gB.intersect([33500, 4500], [36000, 7500])         # the crossings
      x, y, L = gB.intersect([33500, 4500], [36000, 7500], segLength=True)  # and the intercept lengths
      n = np.sum(~np.isnan(x))                                    # the number of crossings

    A line through a vertex crosses both segments there, as MATLAB's tolerance has it, and the
    intercepts count that crossing once, so none is of zero length. MATLAB returns the
    intercepts as a third output always; here segLength=True asks for them.

    Input
      gB - grainBoundary
      p, q - the end points of the line, (x, y) or a vector3d
    Output
      x, y - the coordinates of the crossing per segment, NaN where none
      L - the lengths between successive crossings along the line (segLength=True)
    Keywords
      segLength - also return the intercept lengths
    See also
      grainBoundary.inpolygon"""
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
    # the crossings along the line, the two segments of a vertex crossed there counted once
    ts = np.sort(t[hit])
    ts = ts[np.r_[True, np.diff(ts) > tol]] if len(ts) else ts
    return x, y, np.diff(ts) * np.linalg.norm(ab)

  def inpolygon(self, region):
    """the segments whose midpoint lies inside a polygon (N, 2), or the rectangle [x, y, w, h]

      ind = gB.inpolygon([xmin, ymin, dx, dy])                # by a rectangle
      ind = inpolygon(gB, [[x1, y1], [x2, y2], [x3, y3]])     # by a polygon
      gB = gB[ind]"""
    from .ebsd import _inPolygon
    region = np.asarray(region, dtype=float)
    mp = self._mid2d
    if region.ndim == 1:
      x, y, w, h = region
      return (x <= mp[:, 0]) & (mp[:, 0] <= x + w) & (y <= mp[:, 1]) & (mp[:, 1] <= y + h)
    return _inPolygon(mp[:, 0], mp[:, 1], self._as2d(region).reshape(-1, 2), includeEdge=False)

  # ------------------------------------------------------------------ display

  def __repr__(self):
    lines = [f"grainBoundary ({frameLabel(self.frame)})"]
    if self.isempty:
      return aligned(lines[0] + "\n  the boundary is empty")
    lines.append(f"  size: {len(self)} segments, {int(self.chainId.max()) + 1} chains")
    key = np.sort(self.phaseId, axis=1)
    pairs, inverse = np.unique(key, axis=0, return_inverse=True)
    L = np.bincount(inverse.reshape(-1), weights=self.segLength)
    counts = np.bincount(inverse.reshape(-1))
    # the side outside the map is notIndexed, as MATLAB names it
    def name(p):
      return "notIndexed" if p < 0 else self.CSList[int(p)].mineral
    # the length rounded in the scan unit, as MATLAB writes it
    unit = "µm" if self.scanUnit == "um" else self.scanUnit
    cells = [[str(c) for c in counts], [f"{xnum2str(l)} {unit}" for l in L], [name(p[0]) for p in pairs], [name(p[1]) for p in pairs]]
    lines.append(textTable(["segments", "length", "mineral 1", "mineral 2"], cells))
    return aligned("\n".join(lines))

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from .boundaryresampling import reduceBoundary as reduce, simplifyBoundary as simplify, refineBoundary as refine
    from .shape import characteristicShape, surfor
    from ..analysis.gbnd import calcGBND
    from ..plotting.mapplot import plotBoundary as plot, quiverMap as quiver


# ------------------------------------------------------------------ triple points

class triplePointList:
  """the triple points of a boundary: the vertices where three grains meet, each with its three
  grains, their phases, the three boundary segments and the angles between them

    tP = grains.triplePoints
    tP('Iron bcc', 'Iron bcc', 'Iron bcc')"""

  def __init__(self, V, grainId, phaseId, boundaryId, angles, CSList, plane, frame, vertexId=None):
    self.V, self.grainId, self.phaseId, self.boundaryId, self.angles = V, grainId, phaseId, boundaryId, angles
    self.CSList, self.plane, self.frame = CSList, plane, frame
    # the index of each point into the vertex list of its boundary
    self.vertexId = np.full(len(grainId), -1, dtype=np.int64) if vertexId is None else vertexId

  def __len__(self):
    return len(self.grainId)

  @property
  def size(self):
    """the number of triple points"""
    return len(self.grainId)

  @property
  def isempty(self):
    """whether the list holds no triple point"""
    return len(self.grainId) == 0

  @property
  def allV(self):
    """the positions of the triple points as framed points of the specimen"""
    return vector3d(self.plane.from2d(self.V), frame=self.frame)

  def hasPhase(self, *names):
    """the points with the phases named among their three grains, a name repeated asking for that many"""
    ids = [self.CSList.phaseId(n) for n in names]
    out = np.ones(len(self), dtype=bool)
    for p in set(ids):
      out &= np.sum(self.phaseId == p, axis=1) >= ids.count(p)
    return out

  # tP('Iron bcc', 'Iron bcc', 'Iron bcc'): the points of those phases, MATLAB's call form
  def __call__(self, *names):
    return self[self.hasPhase(*names)]

  # the names tP['...'] and tP['...', '...'] take, which IPython completes inside the brackets
  def _ipython_key_completions_(self):
    return self.CSList.selectionNames(pairs=True)

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
      return aligned(f"triplePointList ({frameLabel(self.frame)})\n  no triple points")
    key = np.sort(self.phaseId, axis=1)
    trip, counts = np.unique(key, axis=0, return_counts=True)
    def name(p):
      return "outside" if p < 0 else self.CSList[int(p)].mineral
    cells = [[str(c) for c in counts]] + [[name(t[j]) for t in trip] for j in range(3)]
    return (f"triplePointList ({frameLabel(self.frame)})\n  size: {len(self)}\n"
            + textTable(["points", "mineral 1", "mineral 2", "mineral 3"], cells))


def triplePoints(gB):
  """the vertices where exactly three segments of three grains meet, with the grains, the
  segments and the angles the segments enclose"""
  T = gB.V.dtype
  deg = gB.vertexDegree()
  v = gB.F.reshape(-1)
  seg = np.repeat(np.arange(len(gB)), 2)
  order = np.argsort(v, kind="stable")
  vs, ss = v[order], seg[order]
  starts = np.searchsorted(vs, np.arange(len(gB.V) + 1))
  # the three segments of every vertex of degree three, and the grains on their six sides
  vert = np.flatnonzero(deg == 3)
  at = ss[starts[vert][:, None] + np.arange(3)]
  G = np.sort(gB.grainId[at].reshape(len(vert), 6), axis=1)
  first = np.concatenate([np.ones((len(vert), 1), dtype=bool), G[:, 1:] != G[:, :-1]], axis=1)
  # three segments of only two grains is a cut, and a junction on the rim of the map
  # is not a point where three grains meet
  ok = (first.sum(axis=1) == 3) & (G[:, 0] > 0)
  vert, at, G, first = vert[ok], at[ok], G[ok], first[ok]
  n = len(vert)
  gid = G[first].reshape(n, 3).astype(np.int32)
  bid = at.astype(np.int32)
  V = gB.V[vert].astype(T).reshape(n, 2)
  far = np.where(gB.F[at, 0] == vert[:, None], gB.F[at, 1], gB.F[at, 0])
  o = gB.V[far] - gB.V[vert][:, None, :]
  th = np.sort(np.arctan2(o[..., 1], o[..., 0]), axis=1)
  ang = np.stack([th[:, 1] - th[:, 0], th[:, 2] - th[:, 1], 2 * np.pi - (th[:, 2] - th[:, 0])], axis=1).astype(T).reshape(n, 3)
  # the phase of a grain is read off the first side of its segments that it borders
  sides = gB.grainId[bid].reshape(n, 6)
  hit = np.argmax(sides[:, None, :] == gid[:, :, None], axis=2)
  k, side = np.divmod(hit, 2)
  pid = gB.phaseId[bid[np.arange(n)[:, None], k], side].astype(np.int8)
  return triplePointList(V, gid, pid, bid, ang, gB.CSList, gB.plane, gB.frame, vert.astype(np.int64))


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
  """the decomposition of an edge list into maximal chains, each walked end to end: the permutation
  that brings the edges into walk order, and per edge whether the walk enters it at its first
  (1) or second (2) vertex; a closed chain is cut at its lowest vertex"""
  F = np.asarray(F, dtype=np.int32).reshape(-1, 2)
  n = len(F)
  if n == 0:
    return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
  deg = np.bincount(F.reshape(-1), minlength=nV).astype(np.int32)
  offset = np.concatenate([[0], np.cumsum(deg)]).astype(np.int64)
  # contiguous adjacency, in increasing segment order at each vertex: a compiled counting
  # sort, or the stable argsort it replaces
  at = adjacencyNumba(F, offset)
  if at is None:
    at = np.repeat(np.arange(n, dtype=np.int32), 2)[np.argsort(F.reshape(-1), kind="stable")]
  perm = np.zeros(n, dtype=np.int32)
  entry = np.zeros(n, dtype=np.int32)
  (_compiledLoop(_chainOrderLoop) or _chainOrderLoop)(F, deg, offset, at, perm, entry)
  return perm, entry


# ------------------------------------------------------------------ construction

def buildBoundary(c, seg, ebsd, inner=False):
  """the boundary of a segmentation: the faces it marks as outer, or inner, in walk order.
  The two sides are listed by phase and then by grain, so a pair of phases always appears
  in the same order and the outside of the map comes first; the site of the first grain,
  or the mirror of the other one where it is outside, says which side is left"""
  keep = np.flatnonzero(seg.isInner if inner else seg.isOuter)
  sites = c.sites
  T = c.V.dtype
  # the measurements are named by their id, which outlives a selection, not by their place
  ids = np.asarray(ebsd.id).reshape(-1)
  s1, s2, p1, p2, gid, pid, eid = _sides(c.cells[keep, 0], c.cells[keep, 1], sites.pixel, seg.grainId, ebsd.phaseId.reshape(-1), ids)
  mori = _misrotations(ebsd.rotations.data.reshape(-1, 4), p1, p2, T)
  F = c.F[keep]

  # the point left of a segment: the site of its first grain, or the mirror of the second
  # site in the segment where the first side is outside
  def leftOf(r):
    mid = (c.V[F[r, 0]] + c.V[F[r, 1]]) / 2
    return np.where((p1[r] >= 0)[:, None], sites.pos[s1[r]], 2 * mid - sites.pos[s2[r]])

  idx, F = _walkOrder(F, c.V, leftOf)
  return grainBoundary(c.V, F, gid[idx], eid[idx], pid[idx], mori[idx], ebsd.CSList, c.plane, ebsd.frame, scanUnit=ebsd.scanUnit)


# the two sides of every face as the boundary lists them: the sites, the measurements, the
# grains, the phases and the measurement ids, swapped where the second side comes first
def _sides(s1, s2, pixel, grainId, phaseId, ids):
  n = len(s1)
  loop = _compiledLoop(_sidesLoop)
  if loop is not None:
    sa, sb = np.empty(n, dtype=s1.dtype), np.empty(n, dtype=s2.dtype)
    pa, pb = np.empty(n, dtype=pixel.dtype), np.empty(n, dtype=pixel.dtype)
    gid, pid, eid = np.empty((n, 2), dtype=np.int32), np.empty((n, 2), dtype=np.int8), np.empty((n, 2), dtype=np.int32)
    loop(s1, s2, pixel, grainId, phaseId, ids, OUTSIDE, NO_PIXEL, sa, sb, pa, pb, gid, pid, eid)
    return sa, sb, pa, pb, gid, pid, eid
  s = np.stack([s1, s2], axis=1)
  p = pixel[s]
  g = _gather(grainId, p, 0).astype(np.int32)
  ph = _gather(phaseId, p, OUTSIDE).astype(np.int8)
  swap = _sideSwap(ph, g)
  for a in (s, p, g, ph):
    a[swap] = a[swap, ::-1]
  return s[:, 0], s[:, 1], p[:, 0], p[:, 1], g, ph, _gather(ids, p, NO_PIXEL).astype(np.int32)


# the faces whose two sides are listed the other way round: the lower phase first, then the
# lower grain
def _sideSwap(ph, g):
  return (ph[:, 0] > ph[:, 1]) | ((ph[:, 0] == ph[:, 1]) & (g[:, 0] > g[:, 1]))


def _sidesLoop(s1, s2, pixel, grainId, phaseId, ids, outside, noPixel, sa, sb, pa, pb, gid, pid, eid):
  for k in range(s1.shape[0]):
    a, b = s1[k], s2[k]
    qa, qb = pixel[a], pixel[b]
    ga = grainId[qa] if qa >= 0 else 0
    gb = grainId[qb] if qb >= 0 else 0
    pha = phaseId[qa] if qa >= 0 else outside
    phb = phaseId[qb] if qb >= 0 else outside
    if pha > phb or (pha == phb and ga > gb):
      a, b, qa, qb, ga, gb, pha, phb = b, a, qb, qa, gb, ga, phb, pha
    sa[k], sb[k], pa[k], pb[k] = a, b, qa, qb
    gid[k, 0], gid[k, 1] = ga, gb
    pid[k, 0], pid[k, 1] = pha, phb
    eid[k, 0] = ids[qa] if qa >= 0 else noPixel
    eid[k, 1] = ids[qb] if qb >= 0 else noPixel


# the misrotation of every face with a measurement on both sides, rot2^-1 rot1, NaN elsewhere
def _misrotations(rot, p1, p2, T):
  mori = np.empty((len(p1), 4), dtype=T)
  loop = _compiledLoop(_misrotationsLoop, parallel=True)
  if loop is not None:
    loop(rot, p1, p2, mori)
    return mori
  both = (p1 >= 0) & (p2 >= 0)
  mori[:] = np.nan
  mori[both] = qmul(qconj(rot[p2[both]]), rot[p1[both]])
  return mori


# the product of the conjugate of rot[p2] with rot[p1] in qmul's arithmetic
def _misrotationsLoop(rot, p1, p2, mori):
  for k in prange(p1.shape[0]):
    if p1[k] < 0 or p2[k] < 0:
      for j in range(4):
        mori[k, j] = np.nan
      continue
    a0, a1, a2, a3 = rot[p2[k], 0], -rot[p2[k], 1], -rot[p2[k], 2], -rot[p2[k], 3]
    b0, b1, b2, b3 = rot[p1[k], 0], rot[p1[k], 1], rot[p1[k], 2], rot[p1[k], 3]
    mori[k, 0] = a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3
    mori[k, 1] = a1 * b0 + a0 * b1 - a3 * b2 + a2 * b3
    mori[k, 2] = a2 * b0 + a3 * b1 + a0 * b2 - a1 * b3
    mori[k, 3] = a3 * b0 - a2 * b1 + a1 * b2 + a0 * b3


# the rows of a boundary in walk order with its segments, every chain turned as a whole
# until the point leftOf(r) of the first grain of its first row r lies on its left: the walk
# order and the reversal of the chains compose into one permutation of the rows
def _walkOrder(F, V, leftOf):
  perm, entry = chainOrder(F, len(V))
  if len(perm) == 0:
    return perm, F
  deg = np.bincount(F.reshape(-1), minlength=len(V))
  orient = _compiledLoop(_orientLoop)
  if orient is not None:
    F1 = np.empty_like(F)
    cid = np.empty(len(F), dtype=np.int64)
    orient(F, perm, entry, deg, F1, cid)
  else:
    head = entry[perm] == 1
    F1 = np.where(head[:, None], F[perm], F[perm][:, ::-1])
    jct = (deg != 2) & (deg > 0)
    cid = np.concatenate([[0], np.cumsum((F1[1:, 0] != F1[:-1, 1]) | jct[F1[1:, 0]])])
  starts = _runStarts(cid)
  v1 = V[F1[starts, 0]]
  d = V[F1[starts, 1]] - v1
  flip = _cross2(d, leftOf(perm[starts]) - v1) < 0
  if not flip.any():
    return perm, F1
  turn = _compiledLoop(_turnLoop, parallel=True)
  if turn is not None:
    idx, F2 = np.empty_like(perm), np.empty_like(F1)
    turn(F1, perm, cid, starts, np.concatenate([starts[1:], [len(F1)]]), flip, idx, F2)
    return idx, F2
  order, rows = _reversedOrder(cid, flip)
  F2 = F1[order]
  F2[rows] = F2[rows][:, ::-1]
  return perm[order], F2


# the row order that turns the chains marked round, a row i of a flipped chain [s, e) taking the
# row s + e - 1 - i, and the rows of the flipped chains; cid numbers the chains from 0 in order
def _reversedOrder(cid, flip):
  starts = _runStarts(cid)
  ends = np.concatenate([starts[1:], [len(cid)]])
  order = np.arange(len(cid))
  rows = flip[cid]
  order[rows] = (starts + ends - 1)[cid[rows]] - order[rows]
  return order, rows


# the segments in walk order, each turned the way the walk enters it, and the chain of every
# row: a chain breaks where the head of one segment is not the tail of the next, and at a
# junction, a vertex where other than two segments meet
def _orientLoop(F, perm, entry, deg, F1, cid):
  c = -1
  for i in range(perm.shape[0]):
    k = perm[i]
    if entry[k] == 1:
      F1[i, 0], F1[i, 1] = F[k, 0], F[k, 1]
    else:
      F1[i, 0], F1[i, 1] = F[k, 1], F[k, 0]
    v = F1[i, 0]
    if i == 0 or F1[i, 0] != F1[i - 1, 1] or (deg[v] != 2 and deg[v] > 0):
      c += 1
    cid[i] = c


# the chains marked turned round: a row i of a flipped chain [s, e) takes the row
# s + e - 1 - i with its segment reversed
def _turnLoop(F1, perm, cid, starts, ends, flip, idx, F2):
  for i in prange(F1.shape[0]):
    c = cid[i]
    if flip[c]:
      j = starts[c] + ends[c] - 1 - i
      idx[i] = perm[j]
      F2[i, 0], F2[i, 1] = F1[j, 1], F1[j, 0]
    else:
      idx[i] = perm[i]
      F2[i, 0], F2[i, 1] = F1[i, 0], F1[i, 1]


def orderWalk(gB):
  """bring the segments into walk order and orient every chain so that the grain in
  grainId[:, 0] lies to its left: the boundary already holds that invariant on every
  segment, so a segment the walk enters from its head is turned round on both sides, and a
  chain is turned by the majority"""
  perm, entry = chainOrder(gB.F, len(gB.V))
  if len(perm) == 0:
    return gB
  head = entry[perm] == 1
  F = np.where(head[:, None], gB.F[perm], gB.F[perm][:, ::-1])
  out = gB._with(F=F, grainId=gB.grainId[perm], ebsdId=gB.ebsdId[perm], phaseId=gB.phaseId[perm], misrotation=gB.misrotation.data[perm])
  out = out._flipSegments(~head)._with(F=F)
  cid = out.chainId
  kept = np.bincount(cid, weights=head)
  total = np.bincount(cid)
  flip = 2 * kept < total
  if not flip.any():
    return out
  return reverseChains(out, cid, flip)


def reverseChains(gB, cid, flip):
  """reverse the chains marked, in the order of their rows and within each segment, both
  sides of every segment turned with it"""
  order, rows = _reversedOrder(cid, flip)
  F = gB.F[order]
  F[rows] = F[rows][:, ::-1]
  out = gB._with(F=F, grainId=gB.grainId[order], ebsdId=gB.ebsdId[order], phaseId=gB.phaseId[order], misrotation=gB.misrotation.data[order])
  return out._flipSegments(rows)._with(F=F)


__all__ = ["grainBoundary", "triplePointList", "triplePoints", "chainOrder"]
