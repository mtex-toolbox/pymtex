# grain3Boundary: the faces between the grains of a volume (MATLAB grain3Boundary, ../master)
#
#   gB.allV, gB.V, gB.F                 the vertices and the faces: an (m, 3) array of triangles, or
#                                       a list of polygons, each an array into allV, not closed
#   gB.id                               the number of every face, kept by a selection
#   gB.grainId, gB.ebsdId, gB.phaseId   the two sides of every face, the normal from the first
#                                       to the second; outside the volume a side carries the
#                                       grain 0, the voxel -1 and the phase -1, as in 2D
#   gB.misrotation, gB.misorientation   the rotation from the second side to the first
#   gB.N, gB.centroid, gB.area, gB.perimeter, gB.diameter     per face
#   gB.edges(), gB.nodeType()           the edges of a triangulated boundary, the grains at a vertex
#   gB.I_FG, gB.I_VF, gB.I_VG           the incidences of faces, vertices and grains
#   gB[sel], gB.flip(sel), len(gB)
#
# The polygons are kept as one index array with offsets, so that every measure is one pass
# over all corners; a triangulated boundary is the case of offsets in steps of three.
from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

from ..geometry.orientation import orientation
from ..geometry.quaternion import rotation
from ..geometry.vector3d import vector3d
from ..geometry.display import aligned


class grain3Boundary:
  """the faces between the grains of a volume, triangles or polygons over the vertices allV, with
  the grains, voxels and phases on their two sides and the misrotation across: grains.boundary"""

  def __init__(self, allV, F, grainId, ebsdId, phaseId, misrotation, CSList, frame, scanUnit="um", id=None):
    self.allV = np.asarray(allV.data if isinstance(allV, vector3d) else allV, dtype=float).reshape(-1, 3)
    if isinstance(F, tuple) and len(F) == 2 and np.ndim(F[0]) == 1 and np.ndim(F[1]) == 1:
      self._ptr, self._idx = np.asarray(F[0], dtype=np.int64), np.asarray(F[1], dtype=np.int64)
    elif isinstance(F, np.ndarray) and F.ndim == 2:
      m, k = F.shape
      self._ptr, self._idx = np.arange(0, m * k + 1, k, dtype=np.int64), np.asarray(F, dtype=np.int64).reshape(-1)
    else:
      F = [np.asarray(f, dtype=np.int64).reshape(-1) for f in F]
      self._ptr = np.concatenate([[0], np.cumsum([len(f) for f in F])]).astype(np.int64)
      self._idx = np.concatenate(F) if F else np.zeros(0, dtype=np.int64)
    m = len(self._ptr) - 1
    self.grainId = np.asarray(grainId, dtype=np.int64).reshape(-1, 2)
    self.ebsdId = np.asarray(ebsdId, dtype=np.int64).reshape(-1, 2) if ebsdId is not None else np.full((m, 2), -1, dtype=np.int64)
    self.phaseId = np.asarray(phaseId, dtype=np.int8).reshape(-1, 2)
    self.misrotation = misrotation if isinstance(misrotation, rotation) else rotation._fromData(np.asarray(misrotation).reshape(-1, 4))
    self.CSList, self.frame, self.scanUnit = CSList, frame, scanUnit
    self.id = np.arange(1, m + 1) if id is None else np.asarray(id, dtype=np.int64).reshape(-1)

  def __len__(self):
    return len(self._ptr) - 1

  @property
  def size(self):
    """the number of faces"""
    return len(self)

  # ------------------------------------------------------------------ the faces

  @property
  def numCorners(self):
    """the number of corners of every face"""
    return np.diff(self._ptr)

  @property
  def isTriangulated(self):
    """whether every face is a triangle"""
    return bool(np.all(self.numCorners == 3))

  @property
  def F(self):
    """the triangles as an (m, 3) array, a polygonal boundary as a list of index arrays"""
    if self.isTriangulated:
      return self._idx.reshape(-1, 3)
    return [self._idx[a:b] for a, b in zip(self._ptr[:-1], self._ptr[1:])]

  # the face of every corner, and the corner that follows it around its face
  def _faceOfCorner(self):
    return np.repeat(np.arange(len(self)), self.numCorners)

  def _nextCorner(self):
    k = np.arange(len(self._idx)) + 1
    k[self._ptr[1:] - 1] = self._ptr[:-1]
    return k

  # the fan triangles of every face: (k, 3, 3) corners and the face they belong to
  def _fan(self):
    if self.isTriangulated:
      return self.allV[self._idx.reshape(-1, 3)], np.arange(len(self))
    n = self.numCorners - 2
    face = np.repeat(np.arange(len(self)), n)
    start = np.repeat(self._ptr[:-1], n)
    j = np.arange(len(face)) - np.repeat(np.cumsum(n) - n, n)
    T = np.stack([self._idx[start], self._idx[start + j + 1], self._idx[start + j + 2]], axis=1)
    return self.allV[T], face

  def _sumByFace(self, x, face):
    out = np.zeros((len(self),) + x.shape[1:])
    np.add.at(out, face, x)
    return out

  # faces by position or mask, or by the phases on their sides: 'indexed' the faces between
  # two indexed grains, a mineral the faces with it on one side, two minerals those between them
  def __getitem__(self, sel):
    if isinstance(sel, str) or (isinstance(sel, tuple) and sel and isinstance(sel[0], str)):
      return self[self.hasPhase(*((sel,) if isinstance(sel, str) else sel))]
    rows = np.arange(len(self))[sel]
    rows = np.atleast_1d(rows)
    counts = self.numCorners[rows]
    ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    idx = self._idx[np.repeat(self._ptr[rows], counts) + np.arange(ptr[-1]) - np.repeat(ptr[:-1], counts)]
    return grain3Boundary(self.allV, (ptr, idx), self.grainId[rows], self.ebsdId[rows], self.phaseId[rows],
                          self.misrotation.data.reshape(-1, 4)[rows], self.CSList, self.frame, self.scanUnit, self.id[rows])

  def hasPhase(self, a, b=None):
    """the faces with a phase on one side, or between two phases, by name, frame or number"""
    if b is None and isinstance(a, str) and a.lower() in ("indexed", "notindexed") and a.lower() not in [m.lower() for m in self.CSList.mineralList]:
      inside = np.all(self.phaseId > 0, axis=1)
      return inside if a.lower() == "indexed" else ~inside
    p = self.CSList.phaseId(a)
    if b is None:
      return np.any(self.phaseId == p, axis=1)
    q = self.CSList.phaseId(b)
    return ((self.phaseId[:, 0] == p) & (self.phaseId[:, 1] == q)) | ((self.phaseId[:, 0] == q) & (self.phaseId[:, 1] == p))

  def flip(self, sel=None):
    """the two sides exchanged, of all faces or a selection, as MATLAB's flip"""
    rows = np.arange(len(self)) if sel is None else np.arange(len(self))[sel]
    gB = self[np.arange(len(self))]
    for a in (gB.grainId, gB.ebsdId, gB.phaseId):
      a[rows] = a[rows][:, ::-1]
    q = gB.misrotation.data.reshape(-1, 4)
    q[rows, 1:] *= -1
    return gB

  def flipN(self, sel=None):
    """the faces turned round: the corners in the reverse order, the normals the other way"""
    rows = np.arange(len(self)) if sel is None else np.arange(len(self))[sel]
    gB = self[np.arange(len(self))]
    for r in rows:
      a, b = gB._ptr[r], gB._ptr[r + 1]
      gB._idx[a:b] = gB._idx[a:b][::-1]
    return gB

  # ------------------------------------------------------------------ geometry

  @property
  def idV(self):
    """the indices into allV of the vertices the faces use, sorted"""
    return np.unique(self._idx)

  @property
  def V(self):
    """the vertices the faces use, as vectors"""
    return vector3d(self.allV[self.idV], frame=self.frame)

  # twice the vector area of every face: the sum over its fan triangles
  def _vectorArea(self):
    T, face = self._fan()
    return self._sumByFace(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), face)

  @property
  def N(self):
    """the unit normal of every face, NaN for a face of no area"""
    c = self._vectorArea()
    with np.errstate(invalid="ignore", divide="ignore"):
      return vector3d(c / np.linalg.norm(c, axis=1, keepdims=True), frame=self.frame)

  @property
  def area(self):
    """the area of every face"""
    return np.linalg.norm(self._vectorArea(), axis=1) / 2

  @property
  def centroid(self):
    """the mean of the corners of a triangle, the area centroid of a polygon"""
    if self.isTriangulated:
      return vector3d(self.allV[self._idx.reshape(-1, 3)].mean(axis=1), frame=self.frame)
    T, face = self._fan()
    a = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    c = self._sumByFace(a[:, None] * T.mean(axis=1), face) / self._sumByFace(a, face)[:, None]
    return vector3d(c, frame=self.frame)

  @property
  def perimeter(self):
    """the perimeter of every face, the length of its edges"""
    V = self.allV
    d = np.linalg.norm(V[self._idx[self._nextCorner()]] - V[self._idx], axis=1)
    return np.bincount(self._faceOfCorner(), weights=d, minlength=len(self))

  @property
  def diameter(self):
    """the largest distance between two corners of every face"""
    out = np.zeros(len(self))
    for k in np.unique(self.numCorners):
      rows = np.flatnonzero(self.numCorners == k)
      P = self.allV[self._idx[self._ptr[rows][:, None] + np.arange(k)]]
      out[rows] = np.sqrt(np.max(np.sum((P[:, :, None] - P[:, None, :]) ** 2, axis=-1), axis=(1, 2)))
    return out

  def edges(self):
    """the edges of a triangulated boundary, sorted pairs of vertices, and the three edges of
    every face"""
    if not self.isTriangulated:
      raise ValueError("edges needs triangular faces, see grain3d.triangulate")
    F = self._idx.reshape(-1, 3)
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    E, F2E = np.unique(E, axis=0, return_inverse=True)
    return E, F2E.reshape(3, -1).T

  def nodeType(self):
    """the number of grains meeting at every vertex, plus 10 on the surface of the volume"""
    t = np.asarray(self.I_VG.sum(axis=1)).reshape(-1)
    hull = np.zeros(len(self.allV), dtype=bool)
    outer = np.any(self.grainId == 0, axis=1)
    hull[self._idx[np.repeat(outer, self.numCorners)]] = True
    return t + 10 * hull

  # ------------------------------------------------------------------ the sides

  @property
  def isIndexed(self):
    """whether the grain on either side is indexed, (m, 2); the outside is not"""
    return self.phaseId > 0

  @property
  def misorientation(self):
    """the misorientation of every face from the second side to the first, NaN where a side is not
    indexed; the faces must lie between one pair of phases"""
    p1, p2 = self.phaseId[:, 0], self.phaseId[:, 1]
    ok = (p1 > 0) & (p2 > 0)
    if not ok.any():
      raise ValueError("no face lies between two indexed voxels")
    a, b = int(p1[ok][0]), int(p2[ok][0])
    if not (np.all(p1[ok] == a) and np.all(p2[ok] == b)):
      raise ValueError("the faces lie between different pairs of phases; select one pair first")
    q = self.misrotation.data.reshape(-1, 4).copy()
    q[~ok] = np.nan
    return orientation(rotation._fromData(q), self.CSList[a], self.CSList[b], antipodal=a == b)

  def hasGrain(self, g, h=None):
    """whether a face touches a phase, or lies between two
    whether a face touches a grain, or lies between two"""
    if h is None:
      return np.isin(self.grainId, np.atleast_1d(g)).any(axis=1)
    return ((self.grainId[:, 0] == g) & (self.grainId[:, 1] == h)) | ((self.grainId[:, 0] == h) & (self.grainId[:, 1] == g))

  # ------------------------------------------------------------------ incidences

  @property
  def I_FG(self):
    """faces x grains, the columns the grain ids"""
    from scipy.sparse import csr_matrix
    f, s = np.nonzero(self.grainId > 0)
    return csr_matrix((np.ones(len(f)), (f, self.grainId[f, s] - 1)), shape=(len(self), int(self.grainId.max(initial=0))))

  @property
  def I_VF(self):
    """the incidence of vertices and faces, a sparse boolean matrix vertices x faces"""
    from scipy.sparse import csr_matrix
    return csr_matrix((np.ones(len(self._idx), dtype=bool), (self._idx, self._faceOfCorner())), shape=(len(self.allV), len(self)))

  @property
  def I_VG(self):
    """the incidence of vertices and grains, a sparse boolean matrix vertices x grains"""
    return (self.I_VF.astype(np.int64) @ self.I_FG) > 0

  def __repr__(self):
    inner = int(np.sum(np.all(self.grainId > 0, axis=1)))
    kind = "triangles" if self.isTriangulated else "polygons"
    return aligned(f"grain3Boundary\n  faces: {len(self)} {kind} ({inner} between grains, {len(self) - inner} on the outside)\n  vertices: {len(self.idV)}")

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from ..plotting.volumeplot import plotBoundary3d as plot, quiver3dMap as quiver


__all__ = ["grain3Boundary"]
