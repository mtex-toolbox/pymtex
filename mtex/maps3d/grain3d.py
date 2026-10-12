# grain3d: the grains of a volume as closed polyhedra (MATLAB grain3d, ../master)
#
#   grains.id, grains.phaseId, grains.numPixel, grains.meanOrientation, grains.GOS
#   grains.I_GF              grains x faces of grains.boundary: +1 where the face normal points
#                            out of the grain, -1 where it points in
#   grains.boundary          the grain3Boundary of the faces of the grains
#   grains.V, grains.F       the vertices and the faces of the boundary
#   grains.volume, grains.centroid, grains.surfaceArea, grains.diameter, grains.extent
#   grains.principalComponents()      the half axes of the ellipsoid of the same moments
#   grains.neighbors(), grains.numNeighbors, grains.intersected(plane), grains.slice(plane)
#   grains[k], grains[mask], grains['Iron'], grains['id', ids], grains.rotate(rot)
#   grain3d.fromFaces(V, F, I_GF, ori, CSList, phaseId)    MATLAB's grain3d(V, F, I_GF, ...)
#   grain3d.load('grains.dream3d'), grain3d.load('grains.tess'), grains.orientFaces()
#
# The volume, the centroid and the second moments are the divergence theorem over the
# oriented faces, a polygon taken as the fan of triangles from its first corner, as
# MATLAB's volume, centroid and principalComponents. A selection keeps the ids of the
# grains and of the faces, as MATLAB's subSet.
from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

from ..geometry._array import MtexArray
from ..geometry.kernels import columnMax, columnMin, qconj
from ..geometry.orientation import orientation
from ..geometry.quaternion import rotation
from ..geometry.vector3d import vector3d
from ..geometry.display import aligned, frameLabel
from ..maps.grains import _at, _idPositions, _positionTable


class grain3d:
  """the grains of a volume as closed polyhedra, given by the incidence I_GF of grains and the
  faces of a grain3Boundary: what calcGrains returns for a volume, the counterpart of grain2d

    grains = calcGrains(ebsd3)
    grain3d.fromFaces(V, F, I_GF, ori, CSList, phaseId)
    grain3d.load('grains.dream3d')"""

  def __init__(self, id, phaseId, CSList, numPixel, meanRotation, GOS, I_GF, boundary, frame, prop=None, phaseMap=None):
    self.id = np.asarray(id, dtype=np.int64).reshape(-1)
    self.phaseId = np.asarray(phaseId, dtype=np.int8).reshape(-1)
    self.CSList = CSList
    self.numPixel = np.asarray(numPixel, dtype=np.int64).reshape(-1)
    self.meanRotation = meanRotation if isinstance(meanRotation, rotation) else rotation._fromData(np.asarray(meanRotation).reshape(-1, 4))
    self.GOS = np.asarray(GOS, dtype=float).reshape(-1)
    self.I_GF = I_GF.tocsr()
    self.boundary = boundary
    self.frame = frame
    self.prop = {} if prop is None else dict(prop)
    self.phaseMap = np.arange(len(CSList)) if phaseMap is None else np.asarray(phaseMap)

  @staticmethod
  def fromFaces(V, F, I_GF, ori=None, CSList=None, phaseId=None, frame=None):
    """the grains of vertices, faces and the incidence of grains and faces, as MATLAB's
    constructor: the face lies between the grain it is +1 for and the one it is -1 for, the
    misorientation from the mean rotations, the voxels unknown"""
    from scipy.sparse import csr_matrix
    from ..geometry.frames import specimenFrame
    from ..geometry.kernels import qmul
    from ..maps.phaselist import phaseList
    from .grain3boundary import grain3Boundary
    I = csr_matrix(I_GF)
    n, m = I.shape
    cs = CSList if isinstance(CSList, phaseList) else phaseList(CSList if CSList is not None else ["notIndexed"])
    ph = np.full(n, 1 if len(cs) > 1 else 0, dtype=np.int8) if phaseId is None else np.asarray(phaseId, dtype=np.int8).reshape(-1)
    q = np.full((n, 4), np.nan) if ori is None else np.asarray(ori.data if isinstance(ori, MtexArray) else ori, dtype=float).reshape(-1, 4)
    C = I.tocoo()
    gid = np.zeros((m, 2), dtype=np.int64)
    gid[C.col[C.data > 0], 0] = C.row[C.data > 0] + 1
    gid[C.col[C.data < 0], 1] = C.row[C.data < 0] + 1
    both = np.all(gid > 0, axis=1)
    mori = np.full((m, 4), np.nan)
    mori[both] = qmul(qconj(q[gid[both, 1] - 1]), q[gid[both, 0] - 1])
    side = np.where(gid > 0, ph[np.maximum(gid, 1) - 1], -1)
    fr = specimenFrame.default if frame is None else frame
    gB = grain3Boundary(V, F, gid, None, side, mori, cs, fr)
    return grain3d(np.arange(1, n + 1), ph, cs, np.ones(n), q, np.zeros(n), I, gB, fr)

  def orientFaces(self):
    """the faces turned consistently around every grain, the normals out of it, as MATLAB's orientFaces"""
    from .orientfaces import orientFaces
    return orientFaces(self)

  @staticmethod
  def load(fname, *args, **kwargs):
    """grain3d.load('grains.dream3d'), grain3d.load('grains.tess', CS=cs)"""
    from ..io.loadgrains3d import loadGrains3d
    return loadGrains3d(fname, *args, **kwargs)

  def __len__(self):
    return len(self.id)

  @property
  def size(self):
    """the number of grains"""
    return len(self.id)

  @property
  def isIndexed(self):
    """whether the phase of every grain is indexed"""
    return self.CSList._indexedMask[self.phaseId]

  @property
  def indexedPhaseId(self):
    """the ids of the indexed phases among the grains"""
    present = np.unique(self.phaseId)
    return np.array([p for p in present if self.CSList.isIndexed(p)], dtype=int)

  @property
  def mineralList(self):
    """the names of the phases of the phase list"""
    return self.CSList.mineralList

  @property
  def CS(self):
    """the crystal frame of grains of one indexed phase"""
    ids = self.indexedPhaseId
    if len(ids) != 1:
      raise ValueError(f"the grains have {len(ids)} indexed phases, so name one: grains['phase']")
    return self.CSList[int(ids[0])]

  @property
  def color(self):
    """the colour of the phase of every grain, n x 3, as MATLAB's grains.color"""
    from ..maps.phaselist import colorTriplet, phaseColorNames
    names = phaseColorNames(self.CSList.frames)
    rgb = np.array([colorTriplet(c) if c is not None else (1.0, 1.0, 1.0) for c in names], dtype=float)
    return rgb[self.phaseId]

  @property
  def meanOrientation(self):
    """the mean orientations of the grains, which must be of one phase; assigning sets new
    orientations and recomputes the misorientations of the boundary"""
    p = np.unique(self.phaseId)
    if len(p) != 1:
      raise ValueError("the grains are of several phases; select one first")
    q = self.meanRotation.data.reshape(-1, 4)
    return orientation(rotation._fromData(q), self.CSList[int(p[0])], self.frame.stripSym())

  @meanOrientation.setter
  def meanOrientation(self, ori):
    """new orientations, one per grain, their crystal frame the phase of every grain, the
    misorientations of the boundary recomputed, as MATLAB's grains.meanOrientation = ori"""
    from ..geometry.kernels import qmul
    from ..maps.phaselist import phaseList
    q = np.broadcast_to(ori.data.reshape(-1, 4), (len(self), 4)).copy()
    frames = [f for f in self.CSList.frames if not f.isIndexed] or ["notIndexed"]
    self.CSList = phaseList(list(frames) + [ori.CS])
    p = len(self.CSList) - 1
    self.phaseId = np.full(len(self), p, dtype=np.int8)
    self.meanRotation = rotation._fromData(q)
    gB = self.boundary
    g = gB.grainId
    pos = _at(_positionTable(self.id), g)
    both = np.all(pos >= 0, axis=1)
    mori = np.full((len(gB), 4), np.nan)
    mori[both] = qmul(qconj(q[pos[both, 1]]), q[pos[both, 0]])
    gB.misrotation = rotation._fromData(mori)
    gB.phaseId = np.where(g > 0, p, -1).astype(np.int8)
    gB.CSList = self.CSList

  # ------------------------------------------------------------------ selections

  def _new(self, rows, I_GF=None, boundary=None, meanRotation=None):
    I = self.I_GF[rows] if I_GF is None else I_GF
    if boundary is None:
      cols = np.unique(I.nonzero()[1])
      I, boundary = I[:, cols], self.boundary[cols]
    prop = {k: np.asarray(v)[rows] for k, v in self.prop.items()}
    q = self.meanRotation.data.reshape(-1, 4)[rows] if meanRotation is None else meanRotation
    return grain3d(self.id[rows], self.phaseId[rows], self.CSList, self.numPixel[rows], q, self.GOS[rows], I, boundary,
                   self.frame, prop, self.phaseMap)

  def id2ind(self, ids):
    """the position of grain ids"""
    return _idPositions(self.id, ids)

  # the names grains['...'] takes, which IPython completes inside the brackets
  def _ipython_key_completions_(self):
    return self.CSList.selectionNames()

  def __getitem__(self, key):
    if isinstance(key, str):
      key = [key]
    if isinstance(key, (list, tuple)) and key and isinstance(key[0], str):
      if key[0] == "id":
        return self._new(self.id2ind(key[1] if len(key) == 2 else list(key[1:])))
      mask = np.zeros(len(self), dtype=bool)
      for name in key:
        if name.lower() == "indexed":
          mask |= self.isIndexed
        else:
          mask |= self.phaseId == self.CSList.phaseId(name)
      return self._new(np.flatnonzero(mask))
    rows = np.arange(len(self))[key]
    return self._new(np.atleast_1d(rows))

  def rotate(self, rot, keepXY=False, keepEuler=False, center=(0.0, 0.0, 0.0)):
    """the grains turned about a centre: their faces and, unless keepEuler, their orientations"""
    R = rotation(rot.data if isinstance(rot, MtexArray) else rot)
    q = self.meanRotation.data.reshape(-1, 4)
    if not keepEuler:
      from ..geometry.kernels import qmul
      q = qmul(np.broadcast_to(R.data.reshape(4), q.shape), q)
    gB = self.boundary
    if not keepXY:
      c = np.asarray(center.data if isinstance(center, vector3d) else center, dtype=float).reshape(3)
      gB = gB._copy()
      gB.allV = c + (gB.allV - c) @ R.matrix().reshape(3, 3).T
    return self._new(np.arange(len(self)), self.I_GF, gB, q)

  # MATLAB's rot * grains: the faces and the orientations turned together
  def __rmul__(self, rot):
    return self.rotate(rot) if isinstance(rot, rotation) else NotImplemented

  def triangulate(self):
    """the faces of polygonal grains as triangles, the fan from the first corner of every face"""
    gB = self.boundary
    T, face = gB._fanIndex()
    from .grain3boundary import grain3Boundary
    tri = grain3Boundary(gB.allV, T, gB.grainId[face], gB.ebsdId[face], gB.phaseId[face], gB.misrotation.data.reshape(-1, 4)[face],
                         gB.CSList, gB.frame, gB.scanUnit, gB.id[face])
    return self._new(np.arange(len(self)), self.I_GF[:, face], tri)

  # ------------------------------------------------------------------ the polyhedra

  @property
  def F(self):
    """the faces of the boundary, an (m, 3) array of triangles or a list of polygons"""
    return self.boundary.F

  @property
  def allV(self):
    """all vertices of the boundary, as vectors"""
    return vector3d(self.boundary.allV, frame=self.frame)

  @property
  def V(self):
    """the vertices the faces of the boundary use"""
    return self.boundary.V

  @property
  def numFaces(self):
    """the number of faces of every grain"""
    return np.asarray((self.I_GF != 0).sum(axis=1)).reshape(-1)

  # a quantity summed over the fan triangles of every face, then over the faces of every grain
  def _integral(self, f):
    T, face = self.boundary._fan()
    a, b, c = T[:, 0], T[:, 1], T[:, 2]
    N = np.cross(b - a, c - a)
    x = self.boundary._sumByFace(f(a, b, c, N), face)
    return self.I_GF @ x

  @property
  def volume(self):
    """the volume of every grain, by the divergence theorem over its oriented faces"""
    return self._integral(lambda a, b, c, N: np.einsum("ij,ij->i", a, N)) / 6

  def _centroid(self, vol):
    return self._integral(lambda a, b, c, N: N * ((a + b) ** 2 + (b + c) ** 2 + (c + a) ** 2)) / 48 / vol[:, None]

  @property
  def centroid(self):
    """the centroid of every grain, the centre of mass of the polyhedron"""
    return vector3d(self._centroid(self.volume), frame=self.frame)

  @property
  def surfaceArea(self):
    """the surface area of every grain, the sum of the areas of its faces"""
    return abs(self.I_GF) @ self.boundary.area

  def surface(self):
    """MATLAB's name for the surface area"""
    return self.surfaceArea

  @property
  def extent(self):
    """the box of the vertices, [xmin, xmax, ymin, ymax, zmin, zmax]"""
    P = self.boundary.allV[self.boundary.idV]
    return np.column_stack([columnMin(P), columnMax(P)]).reshape(-1)

  @property
  def midPoint(self):
    """the largest distance between two vertices of every grain, on MATLAB's rounding to 1e-4
    the centre of the box of the vertices, as MATLAB's midPoint"""
    V = self.boundary.V.data.reshape(-1, 3)
    return vector3d((columnMin(V) + columnMax(V)) / 2, frame=self.frame)

  @property
  def diameter(self):
    """the diameter of every grain, the largest distance between two of its vertices"""
    from scipy.spatial import ConvexHull
    V = np.round(1e4 * self.boundary.allV)
    I = self._vertexIncidence()
    out = np.zeros(len(self))
    for k in range(len(self)):
      P = V[I.indices[I.indptr[k]:I.indptr[k + 1]]]
      if len(P) > 100:
        P = P[ConvexHull(P).vertices]
      out[k] = np.sqrt(np.max(np.sum((P[:, None] - P[None]) ** 2, axis=-1)))
    return out / 1e4

  # grains x vertices, as a CSR matrix
  def _vertexIncidence(self):
    return ((abs(self.I_GF) > 0).astype(np.int64) @ self.boundary.I_VF.T.astype(np.int64)).tocsr() > 0

  def principalComponents(self):
    """the half axes of the ellipsoid of the same second moments and volume, the longest first"""
    vol = self.volume
    c = self._centroid(vol)

    def moments(a, b, c_, N):
      P1 = (a + b + c_) / 3
      P2, P3, P4 = (3 * a + b + c_) / 5, (a + 3 * b + c_) / 5, (a + b + 3 * c_) / 5
      I3 = -9 / 16 * P1 ** 3 + 25 / 48 * (P2 ** 3 + P3 ** 3 + P4 ** 3)
      Ixyz = -9 / 16 * P1.prod(axis=1) + 25 / 48 * (P2.prod(axis=1) + P3.prod(axis=1) + P4.prod(axis=1))
      return np.concatenate([N * I3 / 6, N * Ixyz[:, None] / 2], axis=1)

    m = self._integral(moments)
    diag = m[:, :3] - vol[:, None] * c ** 2
    off = m[:, 3:] - vol[:, None] * np.stack([c[:, 1] * c[:, 2], c[:, 0] * c[:, 2], c[:, 0] * c[:, 1]], axis=1)
    M = np.zeros((len(self), 3, 3))
    M[:, [0, 1, 2], [0, 1, 2]] = diag
    M[:, 1, 2] = M[:, 2, 1] = off[:, 0]
    M[:, 0, 2] = M[:, 2, 0] = off[:, 1]
    M[:, 0, 1] = M[:, 1, 0] = off[:, 2]
    lam, vec = np.linalg.eigh(M)
    v = np.swapaxes(vec, 1, 2) * np.sqrt(np.maximum(lam, 0))[:, :, None]
    v = v * (vol / (4 / 3 * np.pi * np.prod(np.linalg.norm(v, axis=2), axis=1)))[:, None, None] ** (1 / 3)
    return tuple(vector3d(v[:, k], frame=self.frame, antipodal=True) for k in (2, 1, 0))

  # ------------------------------------------------------------------ neighbours

  def neighbors(self, full=False):
    """the pairs of neighbouring grains, each sorted, among the grains of the set or with full
    all pairs any of them has"""
    g = self.boundary.grainId
    g = np.unique(np.sort(g[np.all(g > 0, axis=1)], axis=1), axis=0)
    if full:
      return g
    return g[np.all(np.isin(g, self.id), axis=1)]

  @property
  def numNeighbors(self):
    """the number of neighbouring grains of every grain"""
    pairs = self.neighbors(full=True)
    counts = np.bincount(pairs.reshape(-1), minlength=int(self.id.max(initial=0)) + 1)
    return counts[self.id]

  # ------------------------------------------------------------------ smoothing

  def smoothBoundary(self, *args, **kwargs):
    """the grains with their boundary surfaces smoothed, hierarchically: the triple lines along
    themselves first, then the faces; the vertices where four or more grains meet stay

      grains.smoothBoundary(10)                 a laplaceFilter of 10 iterations
      grains.smoothBoundary(taubinFilter(20))   any boundary filter
      grains.smoothBoundary(5, moveOuterBoundary=True)"""
    from .smoothing3d import _smoothBoundary3d
    return _smoothBoundary3d(self, *args, **kwargs)

  def reduceBoundary(self, *args, **kwargs):
    """the grains with their triangulated boundary coarsened, the vertices within cells of n edges
    merged

      grains.reduceBoundary(3)
      grains.reduceBoundary(2, quadric=True)"""
    from .smoothing3d import _reduceBoundary3d
    return _reduceBoundary3d(self, *args, **kwargs)

  def refineBoundary(self, *args, **kwargs):
    """the grains with their triangulated boundary refined, every triangle split into four at the
    midpoints of its edges, n times"""
    from .smoothing3d import _refineBoundary3d
    return _refineBoundary3d(self, *args, **kwargs)

  # ------------------------------------------------------------------ sections

  def intersected(self, plane, P0=None):
    """whether a plane runs through a grain: vertices of it on both sides"""
    from ..geometry.plane import plane3d
    if not isinstance(plane, plane3d):
      plane = plane3d(plane, P0 if P0 is not None else 0)
    d = np.asarray(plane.dist(vector3d(self.boundary.allV, frame=self.frame))).reshape(-1)
    I = self._vertexIncidence()
    # the largest and the smallest distance of the vertices of every grain
    rows = np.repeat(np.arange(len(self)), np.diff(I.indptr))
    hi, lo = np.full(len(self), -np.inf), np.full(len(self), np.inf)
    np.maximum.at(hi, rows, d[I.indices])
    np.minimum.at(lo, rows, d[I.indices])
    return (hi > 0) & (lo < 0)

  def slice(self, plane, P0=None):
    """the 2D grains a plane cuts from the grains, as MATLAB's slice; prop Id3d names the grain
    every section came from"""
    from .section3d import sliceGrains
    return sliceGrains(self, plane, P0)

  def __repr__(self):
    from ..maps.phaselist import phaseTable
    ids, counts = np.unique(self.phaseId, return_counts=True)
    vol = self.volume
    cells = [f"{c} (volume {vol[self.phaseId == p].sum():.6g})" for p, c in zip(ids, counts)]
    lines = [f"grain3d ({frameLabel(self.frame)})", phaseTable(self.CSList.frames, ids, cells, "Grains")]
    lines.append(f"  boundary faces: {len(self.boundary)}")
    if self.prop:
      lines.append("  properties: " + ", ".join(sorted(self.prop)))
    return aligned("\n".join(lines))

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from ..plotting.volumeplot import _plotGrains3d as plot, _quiver3dMap as quiver
    from ..plotting.grainplots import histogram


__all__ = ["grain3d"]
