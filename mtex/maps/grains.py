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

from typing import TYPE_CHECKING
import numpy as np

from ..geometry._array import MtexArray
from ..geometry.frames import referenceFrame
from ..geometry.orientation import orientation
from ..geometry.quaternion import rotation
from ..geometry.symreduce import closestImage
from ..geometry.vector3d import vector3d
from .dynprop import dynProp
from .phaselist import phaseTable
from .polygon import polygonSet, signedArea2, _flatLoops, _insideLoop
from .segment import components, _firstAppearanceLabels
from ..geometry.display import aligned, frameLabel

FIELDS = ("id", "phaseId", "CSList", "numPixel", "meanRotation", "GOS", "frame", "prop", "boundary", "innerBoundary", "loops", "phaseMap")


# the position of every id in a list of ids as a table indexed by id, -1 for an id the list does
# not hold; an id the list holds twice is found at its last place
def _positionTable(ids):
  ids = np.asarray(ids, dtype=np.int64).reshape(-1)
  pos = np.full(int(ids.max(initial=0)) + 1, -1, dtype=np.int64)
  pos[ids] = np.arange(len(ids))
  return pos


# the entries of a position table at the ids asked for, in their shape, -1 where an id lies
# outside the table
def _at(pos, ids):
  ids = np.asarray(ids).astype(np.int64)
  inside = (ids >= 0) & (ids < len(pos))
  return np.where(inside, pos[np.where(inside, ids, 0)], -1)


# one integer per unordered pair of ids below M, in 64 bits whatever the ids are stored in
def _pairKey(g, M):
  g = np.asarray(g, dtype=np.int64).reshape(-1, 2)
  return np.minimum(g[:, 0], g[:, 1]) * M + np.maximum(g[:, 0], g[:, 1])


# the positions of the ids asked for in a list of ids, in their shape, refused when one is not in it
def _idPositions(ids, query):
  out = _at(_positionTable(ids), query)
  if np.any(out < 0):
    raise ValueError("a requested grain id is not in the selection")
  return out


# the first of the members of largest weight of every group of 0 .. n - 1
def _firstLargest(weight, grp, n):
  order = np.lexsort((np.arange(len(grp)), -np.asarray(weight), grp))
  return order[np.searchsorted(grp[order], np.arange(n))]


class grain2d(polygonSet):
  """the grains of a map: a table of ids, phases, pixel counts, mean orientations and
  properties, with the boundary, the inner boundary and the polygon of every grain; MATLAB's
  grain2d

    grains = calcGrains(ebsd)                       # the grains of a map
    grains = grain2d(grains.id, grains.phaseId, grains.CSList, grains.numPixel, grains.meanRotation, grains.GOS, grains.frame, boundary=grains.boundary)  # a table from its columns
    grains['Forsterite']                            # the grains of one phase
    grains[grains.numPixel > 10]                    # the grains of a condition
    grains.selectByGrainId([1, 2, 3])               # the grains of these ids

  Grains are made by calcGrains, which segments a map; the constructor takes the columns of
  the table and the boundaries of a segmentation. The polygon of a grain is the loops of the
  boundary segments that carry its id, traced from the boundary when no loops are given, so
  grains without a boundary have no polygon and no shape. A property given in prop, or
  assigned as grains.name = values, is one value per grain and follows every selection.

  MATLAB builds grains also from vertices and polygons, grain2d(V, poly, ori, CSList); here
  grains come from a segmentation alone.

  Input
    id - integer array, the id of every grain
    phaseId - integer array, the position of every grain's phase in CSList
    CSList - phaseList, the frames of the phases
    numPixel - integer array, the number of pixels of every grain
    meanRotation - rotation, the mean orientation of every grain as a rotation from the
        crystal to the specimen
    GOS - array, the grain orientation spread of every grain in radians
    frame - specimenFrame, the frame the grains lie in
  Output
    grains - grain2d
  Keywords
    prop - dict of further properties, one value per grain, read as grains.name or
        grains.prop['name']
    boundary - grainBoundary, the segments between the grains and to the outside of the map
    innerBoundary - grainBoundary, the segments inside the grains, the low angle boundaries
        of a segmentation with two thresholds
    loops - the polygon of every grain as loops of boundary segments; traced from boundary
        when not given
    phaseMap - the phase number of the file for every entry of CSList
  See also
    calcGrains, grain2d.merge, grain2d.selectByGrainId, grainBoundary, GrainReconstruction,
    SelectingGrains, GrainSpatialPlots, ShapeParameters"""

  def __init__(self, id, phaseId, CSList, numPixel, meanRotation, GOS, frame, prop=None, boundary=None, innerBoundary=None, loops=None,
               phaseMap=None):
    self.id = np.asarray(id, dtype=np.int32)
    self.phaseMap = np.arange(len(CSList)) if phaseMap is None else np.asarray(phaseMap)
    self.phaseId = np.asarray(phaseId, dtype=np.int8)
    self.CSList = CSList
    self.numPixel = np.asarray(numPixel, dtype=np.int32)
    self.meanRotation = meanRotation if isinstance(meanRotation, MtexArray) else rotation._fromData(meanRotation)
    self.GOS = np.asarray(GOS)
    self.frame = frame
    self.prop = dynProp() if prop is None else dynProp(prop)
    self.boundary = boundary
    self.innerBoundary = innerBoundary
    self.loops = tracePolygons(boundary, self.id) if loops is None and boundary is not None else loops

  # ------------------------------------------------------------------ the table

  @property
  def size(self):
    """the number of grains"""
    return len(self.id)

  def __len__(self):
    return self.size

  @property
  def shape(self):
    """the shape of the list of grains, (size,)"""
    return (self.size,)

  @property
  def isIndexed(self):
    """whether each grain belongs to an indexed phase"""
    return self.CSList._indexedMask[self.phaseId]

  @property
  def indexedPhaseId(self):
    """the ids of the indexed phases present among the grains"""
    present = np.unique(self.phaseId)
    return np.array([p for p in present if self.CSList.isIndexed(p)], dtype=int)

  @property
  def mineralList(self):
    """the mineral names of the phases of the phase list"""
    return self.CSList.mineralList

  def _onePhase(self):
    ids = self.indexedPhaseId
    if len(ids) != 1:
      raise ValueError(f"the grains have {len(ids)} indexed phases, so name one: grains['phase']")
    return int(ids[0])

  @property
  def how2plot(self):
    """the plotting convention of the specimen frame the grains are written in"""
    from ..geometry.plottingconvention import plottingConvention
    pC = getattr(self.frame, "how2plot", None)
    return plottingConvention.default if pC is None else pC

  @property
  def phase(self):
    """the phase number of every grain as the file numbered it"""
    return self.phaseMap[self.phaseId]

  @property
  def CS(self):
    """the crystal frame of the one indexed phase of the grains, an error where there are several"""
    return self.CSList[self._onePhase()]

  def cs2phaseId(self, CS):
    """the phase id of a frame, a mineral name or a number, MATLAB's cs2phaseId"""
    return self.CSList.phaseId(CS)

  @property
  def mineral(self):
    """the mineral name of the one indexed phase of the grains"""
    return self.CS.mineral

  @property
  def meanOrientation(self):
    """the mean orientations of the grains of the one indexed phase present"""
    p = self._onePhase()
    return orientation._fromRotation(self.meanRotation.data[self.phaseId == p], (self.CSList[p], self.frame))

  @property
  def grainSize(self):
    """the number of pixels of every grain, grains.numPixel"""
    return self.numPixel

  @property
  def triplePoints(self):
    """the triple points of the grain boundary"""
    return self.boundary.triplePoints

  @property
  def V(self):
    """the vertices of the boundary in the coordinates of the map plane, (n, 2)"""
    return self.boundary.V

  @property
  def allV(self):
    """the vertices of the boundary as framed points of the specimen"""
    return self.boundary.allV

  @property
  def plane(self):
    """the plane the map lies in, a mapPlane"""
    return self.boundary.plane

  @property
  def N(self):
    """the normal of the plane the map lies in, as a direction of the specimen"""
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
    return self[self._containing([x, y])]

  @property
  def isInclusion(self):
    """a grain that sits inside the hole of another one: where the grain lies, not what it touches"""
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

  def neighbors(self, full=False, matrix=False):
    """the pairs of ids of grains sharing a boundary, both in the selection, or either with
    full; as a symmetric sparse matrix indexed by id with matrix; with other grains the pairs
    of one of these and one of those; MATLAB's grain2d.neighbors

      pairs = grains.neighbors()                  # the pairs of neighbours among the grains
      pairs = g1.neighbors(g2)                    # a grain of g1 beside a grain of g2
      pairs = grains.neighbors(full=True)         # every neighbour, inside the selection or not
      A = grains.neighbors(matrix=True)           # the adjacency matrix

    A pair is listed once, the smaller id first; the rim of the map and a grain's own inner
    boundary make no pair. With other grains the first column is the grain of these, the
    second the one of those. The adjacency matrix is a symmetric boolean scipy.sparse matrix,
    the row and column id - 1 for the grain id.

    Input
      grains - grain2d
    Output
      pairs - the grain ids, (n, 2)
      A - scipy.sparse.csr_matrix (matrix=True)
    Keywords
      full - take the neighbours outside the selection as well; another grain2d in its place
        gives the pairs between the two sets
      matrix - the result as the adjacency matrix
    See also
      grain2d.numNeighbors, grainBoundary.selectByGrainId"""
    g = self.boundary.grainId
    a, b = g[:, 0], g[:, 1]
    # g1.neighbors(g2): the pairs with the first grain in g1 and the second in g2
    if isinstance(full, grain2d):
      other = full.id
      ab = (a > 0) & (b > 0) & (a != b)
      fwd = ab & np.isin(a, self.id) & np.isin(b, other)
      bwd = ab & np.isin(b, self.id) & np.isin(a, other)
      return np.unique(np.concatenate([g[fwd], g[bwd][:, ::-1]]), axis=0)
    ok = (a > 0) & (b > 0) & (a != b)
    # the selection as a table over the ids, the pairs as one key each: the unique keys are
    # the unique sorted rows in their order
    M = int(max(g.max(initial=0), self.id.max(initial=0))) + 1
    sel = np.zeros(M, dtype=bool)
    sel[self.id] = True
    ina, inb = sel[a], sel[b]
    ok &= (ina | inb) if full else (ina & inb)
    lo, hi = np.minimum(a[ok], b[ok]).astype(np.int64), np.maximum(a[ok], b[ok]).astype(np.int64)
    key = np.unique(lo * M + hi)
    pairs = np.stack([key // M, key % M], axis=1).astype(g.dtype)
    if not matrix:
      return pairs
    from scipy.sparse import coo_matrix
    n = int(max(self.id.max(initial=0), pairs.max(initial=0)))
    return coo_matrix((np.ones(2 * len(pairs), dtype=bool), (np.concatenate([pairs[:, 0], pairs[:, 1]]) - 1,
                                                             np.concatenate([pairs[:, 1], pairs[:, 0]]) - 1)), shape=(n, n)).tocsr()

  @property
  def numNeighbors(self):
    """the number of grains every grain shares a boundary with, the neighbours outside the selection
    included"""
    pairs = self.neighbors(full=True)
    counts = np.bincount(pairs.reshape(-1), minlength=int(self.id.max(initial=0)) + 1)
    return counts[self.id].astype(np.int32)

  @property
  def isBoundary(self):
    """a grain that touches the border of the map"""
    g = self.boundary.grainId
    rim = np.unique(np.concatenate([g[g[:, 0] == 0, 1], g[g[:, 1] == 0, 0]]))
    return np.isin(self.id, rim[rim > 0])

  def subBoundarySize(self, sel=None):
    """the inner boundary of every grain: how many segments of the subgrain boundaries lie in
    it; MATLAB's grain2d.subBoundarySize

      n = grains.subBoundarySize()                                      # every inner segment
      n = grains.subBoundarySize(grains.innerBoundary.segLength > 40)   # the segments selected

    Input
      grains - grain2d
    Output
      n - the number of inner boundary segments per grain
    Keywords
      sel - a mask or the indices of the inner boundary segments counted, all when not given
    See also
      grain2d.subBoundaryLength, grain2d.neighbors"""
    ib = self.innerBoundary
    sel = np.ones(len(ib), dtype=bool) if sel is None else sel
    counts = np.bincount(ib.grainId[sel, 0], minlength=int(self.id.max(initial=0)) + 1)
    return counts[self.id].astype(np.int32)

  def subBoundaryLength(self, sel=None):
    """the length of the subgrain boundaries inside every grain, of the inner boundary segments sel
    selects if given; MATLAB's grain2d.subBoundaryLength

      L = grains.subBoundaryLength()                                      # every inner segment
      L = grains.subBoundaryLength(grains.innerBoundary.segLength > 40)   # the segments selected

    Input
      grains - grain2d
    Output
      L - the length of the inner boundary per grain, in the units of the map
    Keywords
      sel - a mask or the indices of the inner boundary segments summed, all when not given
    See also
      grain2d.subBoundarySize, grain2d.neighbors"""
    ib = self.innerBoundary
    sel = np.ones(len(ib), dtype=bool) if sel is None else sel
    L = np.bincount(ib.grainId[sel, 0], weights=ib.segLength[sel], minlength=int(self.id.max(initial=0)) + 1)
    return L[self.id]

  # ------------------------------------------------------------------ selection

  @property
  def scanUnit(self):
    """the unit of the map the grains were reconstructed from, as their boundary carries it"""
    return getattr(self.boundary, "scanUnit", "um")

  # the names grains['...'] takes, which IPython completes inside the brackets
  def _ipython_key_completions_(self):
    return self.CSList.selectionNames()

  # grains[x, y]: the grain the position lies in, MATLAB's grains(x, y)
  def __getitem__(self, key):
    if isinstance(key, tuple) and len(key) == 2 and all(isinstance(k, (int, float, np.integer, np.floating)) for k in key) \
       and any(isinstance(k, (float, np.floating)) for k in key):
      return self._subset(self._containing([float(key[0]), float(key[1])]))
    if isinstance(key, str):
      return self._subset(self._phaseMask([key]))
    if isinstance(key, referenceFrame):
      return self._subset(self.phaseId == self.CSList.phaseId(key))
    if isinstance(key, (list, tuple)) and key and isinstance(key[0], str):
      if key[0] == "id":
        return self.selectByGrainId(key[1:])
      # phase names and conditions, MATLAB's grains('Fo', grains.numPixel > 100)
      mask = self._phaseMask([k for k in key if isinstance(k, str)])
      for k in key:
        if not isinstance(k, str):
          mask &= np.asarray(k, dtype=bool).reshape(-1)
      return self._subset(mask)
    return self._subset(key)

  # grains[ind] = ori: the mean orientations of the grains named and the phase of their frame,
  # MATLAB's grains(ind).meanOrientation = ori, the boundary phases following
  def __setitem__(self, key, ori):
    ind = np.arange(self.size)[key].reshape(-1)
    q = self.meanRotation.data.reshape(-1, 4).copy()
    q[ind] = np.broadcast_to(ori.data.reshape(-1, 4), (len(ind), 4))
    ph = self.phaseId.copy()
    ph[ind] = self.CSList.phaseId(ori.CS)
    self.meanRotation = self.meanRotation._new(q)
    self.phaseId = ph
    new = self.update()
    self.boundary = new.boundary
    self.innerBoundary = new.innerBoundary

  def _phaseMask(self, names):
    return self.CSList._selectionMask(names, self.phaseId)

  # the rows picked, with the boundaries restricted to the segments those grains touch
  def _subset(self, sel):
    ids = np.arange(self.size)[sel].reshape(-1)
    keep = self.id[ids]
    return grain2d(keep, self.phaseId[ids], self.CSList, self.numPixel[ids], self.meanRotation.data[ids], self.GOS[ids], self.frame,
                   {k: (v[ids] if isinstance(v, np.ndarray) else [v[i] for i in ids]) for k, v in self.prop.items()},
                   boundary=None if self.boundary is None else self.boundary.selectByGrainId(keep),
                   innerBoundary=None if self.innerBoundary is None else self.innerBoundary.selectByGrainId(keep),
                   loops=None if self.loops is None else [self.loops[i] for i in ids], phaseMap=self.phaseMap)

  @classmethod
  def cat(cls, *parts):
    """the grains of several parts of one segmentation, MATLAB's [grains1, grains2]: the parts
    share its phases, frame and boundary vertices; a grain in more than one part is kept once,
    the boundaries are joined as grainBoundary.cat joins them

      grains = grain2d.cat(grains['Forsterite'], grains['Enstatite'])
      grains = cat(grains[:100], grains[500:])"""
    from .boundary import grainBoundary
    parts = [p for p in parts if p is not None and p.size]
    first = parts[0]
    for p in parts[1:]:
      if p.CSList is not first.CSList or p.frame is not first.frame:
        raise ValueError("grains are joined as parts of one segmentation, with the same phases and frame")
    ids = np.concatenate([p.id for p in parts])
    _, keep = np.unique(ids, return_index=True)
    keep = np.sort(keep)

    def joined(name):
      return np.concatenate([getattr(p, name) for p in parts])[keep]

    props = {}
    for k in first.prop:
      vals = [p.prop[k] for p in parts]
      if all(isinstance(v, np.ndarray) for v in vals):
        props[k] = np.concatenate(vals)[keep]
      else:
        flat = [x for v in vals for x in v]
        props[k] = [flat[i] for i in keep]

    def boundaries(name):
      bs = [getattr(p, name) for p in parts if getattr(p, name) is not None]
      return grainBoundary.cat(*bs) if bs else None

    loops = None if any(p.loops is None for p in parts) else [l for p in parts for l in p.loops]
    return grain2d(ids[keep], joined("phaseId"), first.CSList, joined("numPixel"),
                   np.concatenate([p.meanRotation.data.reshape(-1, 4) for p in parts])[keep], joined("GOS"), first.frame, props,
                   boundary=boundaries("boundary"), innerBoundary=boundaries("innerBoundary"),
                   loops=None if loops is None else [loops[i] for i in keep], phaseMap=first.phaseMap)

  def id2ind(self, ids):
    """the positions of grain ids in this list
    the positions of grain ids in the list, in the shape of the ids: one for a number, a
    pair per boundary segment for gB.grainId"""
    out = self._lookup(np.asarray(ids))
    return int(out) if np.isscalar(ids) else out

  def findByOrientation(self, ori, epsilon):
    """the grains of the phase of ori whose mean orientation lies within epsilon of it, or of
    a fibre"""
    from ..geometry.fibre import fibre
    cs = ori.CS
    ids = np.flatnonzero(self.phaseId == self.CSList.phaseId(cs))
    o = orientation._fromRotation(self.meanRotation.data.reshape(-1, 4)[ids], (cs, self.frame))
    d = ori.angle(o) if isinstance(ori, fibre) else o.angle(ori)
    return self._subset(ids[np.asarray(d).reshape(-1) <= epsilon])

  def selectByGrainId(self, ids):
    """the grains named by id, in the order asked for

      grains = grains.selectByGrainId([5, 3, 1])
      grain = grains.selectByGrainId(17)"""
    return self._subset(self._lookup(np.atleast_1d(np.asarray(ids)).reshape(-1)))

  def __iter__(self):
    return (self[k] for k in range(self.size))

  def copy(self):
    """a copy of the grains with their own table and properties, the boundaries shared"""
    return grain2d(self.id.copy(), self.phaseId.copy(), self.CSList, self.numPixel.copy(), self.meanRotation.data.copy(), self.GOS.copy(),
                   self.frame, {k: (v.copy() if isinstance(v, np.ndarray) else list(v)) for k, v in self.prop.items()},
                   boundary=self.boundary, innerBoundary=self.innerBoundary, loops=self.loops, phaseMap=self.phaseMap)

  # ------------------------------------------------------------------ merging

  def merge(self, pairs=None, threshold=None, inclusions=None, host=None, maxSize=np.inf, testRun=False):
    """the grains joined into one along given boundaries, by pairs of ids, by a label per grain,
    below a misorientation angle, or every inclusion into the grain around it; MATLAB's merge

      merged = grains.merge(grains.boundary['Forsterite', 'Forsterite'])  # every pair of grains sharing a segment of the boundary
      merged = grains.merge(np.array([[1, 2], [5, 9]]))     # the pairs of these grain ids
      merged = grains.merge(grains.neighbors(matrix=True))  # the pairs a sparse matrix over the ids marks
      merged = grains.merge(threshold=5*degree)             # neighbours of a misorientation below 5 degree
      merged = grains.merge(inclusions=True, maxSize=5)     # every inclusion of fewer than 5 pixels into its host
      merged = grains.merge(inclusions=True, host=grains.numPixel > 100)  # inclusions only into large grains
      parentId = grains.merge(threshold=5*degree, testRun=True)  # the parent ids, nothing merged

    Two grains go into one when a criterion joins them, directly or through a chain of other
    grains. Inclusions combine with pairs or a threshold. A threshold, however, replaces the
    pairs given with it, and a label per grain ignores inclusions; MATLAB applies every
    criterion given.

    Grains of two phases are merged only by an inclusion, which then takes the phase and the
    orientation of its host. A pair of two phases is passed over, and a label spanning phases
    joins the grains of each phase apart, where MATLAB joins them and gives the merged grain
    the largest phase id.

    The pixel counts of the parts add up, the mean orientation is their mean weighted by the
    pixels (MATLAB's calcMeanOrientation, always done here) and GOS their weighted mean. The
    boundary between the parts becomes inner boundary. The merged grains carry the ids of their
    parts as childId, and the grains passed get parentId, the id of the merged grain each went
    into, which MATLAB returns as a second output. MATLAB's merge at a triple point list,
    grains.merge(tP), is not provided.

    Input
      grains - grain2d
      pairs - what is merged: a grainBoundary, whose segments join the grains on both sides;
          an (n, 2) array of grain ids; a square matrix or sparse matrix M over the ids, M[i, j]
          nonzero joining the grains of ids i + 1 and j + 1; or a label per grain, the grains of
          one label joined
    Output
      merged - grain2d, the merged grains, numbered from 1
      parentId - array, with testRun: the id in the merged list of every grain
    Keywords
      threshold - join the neighbours whose mean orientations differ by less than this angle
      inclusions - True to join every inclusion, a grain surrounded by one other grain, to that
          host; a mask over the grains to join only the inclusions it marks
      host - mask over the grains of those that may take an inclusion
      maxSize - the inclusions joined are those of fewer pixels
      testRun - return the parent id of every grain instead of the merged grains, the grains
          passed left unchanged
    See also
      grain2d.isInclusion, grain2d.neighbors, calcGrains, GrainMerge, parentGrainReconstructor"""
    from .boundary import grainBoundary
    if len(np.unique(self.id)) != self.size:
      raise ValueError("merging requires distinct grains")
    pos = self._positions()
    labels = None
    if threshold is not None:
      pairs = self._pairsBelow(threshold)
    elif isinstance(pairs, grainBoundary):
      pairs = pairs.grainId
    elif hasattr(pairs, "tocoo") or (isinstance(pairs, np.ndarray) and pairs.ndim == 2 and pairs.shape[0] == pairs.shape[1] != 2):
      A = pairs.tocoo() if hasattr(pairs, "tocoo") else None
      r, c = (A.row, A.col) if A is not None else np.nonzero(pairs)
      nz = (A.data != 0) if A is not None else np.ones(len(r), dtype=bool)
      pairs = np.stack([r[nz], c[nz]], axis=1) + 1
    elif pairs is not None and np.ndim(pairs) == 1 and len(pairs) == self.size:
      # a label spanning phases joins the grains of each phase apart
      labels = np.unique(np.asarray(pairs).reshape(-1), return_inverse=True)[1].reshape(-1)
      labels = labels.astype(np.int64) * 256 + self.phaseId.astype(np.int64)
      pairs = None
    pairs = np.zeros((0, 2), dtype=np.int64) if pairs is None else np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    a, b = _at(pos, pairs[:, 0]), _at(pos, pairs[:, 1])
    ok = (a >= 0) & (b >= 0)
    ok[ok] &= self.phaseId[a[ok]] == self.phaseId[b[ok]]
    a, b = a[ok], b[ok]
    isIncl = np.zeros(self.size, dtype=bool)
    if inclusions is not None and inclusions is not False:
      hostId = self.inclusionHost
      isIncl = (hostId > 0) & (self.numPixel < maxSize)
      if not isinstance(inclusions, bool):
        isIncl &= np.asarray(inclusions, dtype=bool).reshape(-1)
      hostPos = _at(pos, hostId)
      if host is not None:
        isIncl[isIncl] &= np.asarray(host, dtype=bool).reshape(-1)[hostPos[isIncl]]
      a = np.concatenate([a, np.flatnonzero(isIncl)])
      b = np.concatenate([b, hostPos[isIncl]])
    if labels is None:
      labels = components(self.size, a, b, backend="numpy")
    parent, n = _firstAppearanceLabels(labels)
    if testRun:
      return self.id.copy() if n == self.size else parent
    self.parentId = self.id.copy() if n == self.size else parent
    if n == self.size:
      out = self.copy()
      out.childId = [[int(i)] for i in self.id]
      out.prop.pop("parentId", None)
      return out
    order = np.argsort(parent, kind="stable")
    bounds = np.searchsorted(parent[order], np.arange(1, n + 2))
    T = self.meanRotation.data.dtype
    grp = parent - 1
    w = self.numPixel.astype(T)
    size = np.bincount(grp, minlength=n)
    phaseId = self.phaseId[order[bounds[:-1]]].astype(np.int8)
    numPixel = np.bincount(grp, weights=self.numPixel, minlength=n).astype(np.int32)
    gos = (np.bincount(grp, weights=self.GOS * w, minlength=n) / np.maximum(1, np.bincount(grp, weights=w, minlength=n))).astype(T)
    big = _firstLargest(w, grp, n)
    meanRot = self.meanRotation.data[big].astype(T)
    # a group holding an inclusion takes its host, the member of the longest boundary
    hasIncl = np.zeros(n, dtype=bool)
    hasIncl[grp[isIncl]] = True
    if hasIncl.any():
      host = _firstLargest(self.boundarySize, grp, n)[hasIncl]
      phaseId[hasIncl], meanRot[hasIncl] = self.phaseId[host], self.meanRotation.data[host]
    # the others of several indexed members average the images closest to their largest member
    avg = (size > 1) & self.CSList._indexedMask[phaseId] & ~hasIncl
    for p in np.unique(phaseId[avg]):
      rows = np.flatnonzero(avg[grp] & (phaseId[grp] == p))
      G = self.CSList[p].sym.properGroup().rotations().data.reshape(-1, 4).astype(T)
      ref = self.meanRotation.data[big[grp[rows]]]
      q = closestImage(ref, self.meanRotation.data[rows], np.array([[1.0, 0, 0, 0]], dtype=T), G, False)[1]
      q = np.where(np.sum(q * ref, axis=1)[:, None] < 0, -q, q) * w[rows, None]
      sums = np.zeros((n, 4))
      np.add.at(sums, grp[rows], q)
      ks = np.unique(grp[rows])
      meanRot[ks] = sums[ks] / np.linalg.norm(sums[ks], axis=1, keepdims=True)
    gB, inner = self._relabelBoundary(parent, n)
    return grain2d(np.arange(1, n + 1), phaseId, self.CSList, numPixel, meanRot, gos, self.frame, {"childId": [ids.tolist() for ids in np.split(self.id[order], bounds[1:-1])]},
                   boundary=gB, innerBoundary=inner, phaseMap=self.phaseMap)

  # the position of every id in the list, -1 for an id that is not
  def _positions(self):
    return _positionTable(self.id)

  def _lookup(self, ids):
    return _idPositions(self.id, ids)

  @property
  def inclusionHost(self):
    """the id of the grain whose hole the grain lies in, 0 for none: every grain bordering a
    hole of another, MATLAB's second output of isInclusion"""
    out = np.zeros(self.size, dtype=np.int32)
    flat, _, lengths, owner, first = _flatLoops(self.loops)
    onHole = np.repeat(~first, lengths)
    if not onHole.any():
      return out
    # a segment on a hole of the grain beside it: both vertices among the hole vertices of that
    # grain, (grain, vertex) as one key; the last grain in the list wins where two would
    gB, pos = self.boundary, self._positions()
    NV = int(max(flat.max(initial=0), gB.F.max(initial=0))) + 1
    keys = np.unique(np.repeat(owner, lengths)[onHole] * NV + flat[onHole])
    at = _at(pos, gB.grainId)
    host = np.full(self.size, -1, dtype=np.int64)
    for side in (0, 1):
      k = at[:, side]
      on = k >= 0
      on[on] = np.isin(k[on] * NV + gB.F[on, 0], keys) & np.isin(k[on] * NV + gB.F[on, 1], keys)
      other = at[on, 1 - side]
      ok = (other >= 0) & (other != k[on])
      np.maximum.at(host, other[ok], k[on][ok])
    out[host >= 0] = self.id[host[host >= 0]]
    return out

  def update(self):
    """the grains with the phase of every boundary segment read off the grain it borders, as
    it has to be after the grains changed phase (MATLAB grains.update)"""
    out = self.copy()
    pos = self._positions()
    for name in ("boundary", "innerBoundary"):
      gB = getattr(self, name)
      if gB is None:
        continue
      p = _at(pos, gB.grainId)
      ph = np.where(p >= 0, self.phaseId[np.maximum(p, 0)], gB.phaseId).astype(np.int8)
      setattr(out, name, gB._with(phaseId=ph, CSList=self.CSList))
    return out

  # the pairs of neighbours of one indexed phase whose mean orientations are within the threshold
  def _pairsBelow(self, threshold):
    pairs = self.neighbors()
    pos = self._positions()
    a, b = pos[pairs[:, 0]], pos[pairs[:, 1]]
    keep = np.zeros(len(pairs), dtype=bool)
    q = self.meanRotation.data.reshape(-1, 4)
    for p in np.unique(self.phaseId):
      sel = np.flatnonzero((self.phaseId[a] == p) & (self.phaseId[b] == p))
      if len(sel) == 0 or not self.CSList.isIndexed(p):
        continue
      fr = self.CSList[int(p)]
      o1 = orientation._fromRotation(q[a[sel]], (fr, self.frame))
      o2 = orientation._fromRotation(q[b[sel]], (fr, self.frame))
      keep[sel] = np.asarray(o1.angle(o2)).reshape(-1) <= threshold
    return pairs[keep].astype(np.int64).reshape(-1, 2)

  # the boundary of the merged grains: a segment between two grains that became one is
  # inside one now; neighbours outside the selection keep ids above the merged ones
  def _relabelBoundary(self, parent, n):
    gB, ib = self.boundary, self.innerBoundary
    # the new label of every id as a table: the parents, the ids outside after them in order
    seen = np.unique(np.concatenate([gB.grainId.reshape(-1), ib.grainId.reshape(-1)]))
    table = np.zeros(int(max(seen.max(initial=0), self.id.max(initial=0))) + 1, dtype=np.int32)
    table[self.id] = parent
    external = seen[(seen > 0) & ~np.isin(seen, self.id)]
    table[external] = n + 1 + np.arange(len(external))
    gid = table[gB.grainId]
    outer = gid[:, 0] != gid[:, 1]

    # the segments selected, with their new grain ids
    def sub(sel):
      return gB._with(F=gB.F[sel], grainId=gid[sel], ebsdId=gB.ebsdId[sel], phaseId=gB.phaseId[sel], misrotation=gB.misrotation.data[sel])

    return sub(outer), catBoundaries(sub(~outer), ib._with(grainId=table[ib.grainId]))

  # ------------------------------------------------------------------ display

  def __repr__(self):
    lines = [f"grain2d ({frameLabel(self.frame)})"]
    if self.size == 0:
      return aligned(lines[0] + "\n  no grains")
    lines.append(f"  size: {self.size}")
    ids, counts = np.unique(self.phaseId, return_counts=True)
    pixels = [int(self.numPixel[self.phaseId == p].sum()) for p in ids]
    lines.append(phaseTable(self.CSList.frames, ids, [f"{c} ({px} pixels)" for c, px in zip(counts, pixels)], "Grains"))
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
    return aligned("\n".join(lines))

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from .boundaryresampling import (_reduceGrains as reduceBoundary, _simplifyGrains as simplifyBoundary,
                                     _refineGrains as refineBoundary, sharedVertices)
    from .smoothing import smoothBoundary
    from .shape import _hullGrains as hull, characteristicShape
    from .traces import calcTraces
    from ..plotting.mapplot import plotGrains as plot, quiverMap as quiver
    from ..plotting.grainplots import histogram


# ------------------------------------------------------------------ tracing

def tracePolygons(gB, ids):
  """the loops of every grain, from the segments that have it on one side: a segment is taken
  in the sense that keeps the grain on its left, so the loops come out closed and wound;
  the compiled walk over all grains at once, or the Python walk grain by grain without it"""
  ids = np.asarray(ids, dtype=np.int64).reshape(-1)
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
  from .kernels import edgesByGrainNumba, traceLoopsNumba
  top = int(max(ids.max(initial=0), gB.grainId.max(initial=0))) + 1
  posOf = np.full(top, -1, dtype=np.int64)
  posOf[ids] = np.arange(len(ids))
  edges = edgesByGrainNumba(gB.F, gB.grainId, posOf, len(ids))
  traced = None if edges is None else traceLoopsNumba(gB.V, *edges)
  if traced is None:
    return None
  loopV, loff, lgrain, larea = traced
  # the loops sorted by grain and within a grain by falling area, cut into the grains
  lo = loff.tolist()
  ordered = [loopV[lo[k]:lo[k + 1]] for k in np.lexsort((-larea, lgrain)).tolist()]
  bounds = np.concatenate([[0], np.cumsum(np.bincount(lgrain, minlength=len(ids)))]).tolist()
  return [ordered[a:b] for a, b in zip(bounds[:-1], bounds[1:])]


def traceLoops(edges, V):
  """close the directed edges into loops; where a grain touches itself the walk leaves a
  vertex by the edge that turns most tightly clockwise, which keeps the face on the left"""
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


def catBoundaries(a, b):
  """two boundaries laid end to end, on the same vertices"""
  if a.V is not b.V and not np.array_equal(a.V, b.V):
    raise ValueError("the two boundaries do not share their vertices")
  return a._with(F=np.concatenate([a.F, b.F]), grainId=np.concatenate([a.grainId, b.grainId]), ebsdId=np.concatenate([a.ebsdId, b.ebsdId]),
                 phaseId=np.concatenate([a.phaseId, b.phaseId]), misrotation=np.concatenate([a.misrotation.data, b.misrotation.data]))


__all__ = ["grain2d"]
