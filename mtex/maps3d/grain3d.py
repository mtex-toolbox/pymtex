# grain3d: the grains of a volume as closed polyhedra (MATLAB grain3d, ../master)
#
#   grains.id, grains.phaseId, grains.numPixel, grains.meanOrientation, grains.GOS
#   grains.I_GF              grains x faces of grains.boundary: +1 where the face normal points
#                            out of the grain, -1 where it points in
#   grains.boundary          the grain3Boundary of all faces
#   grains.V, grains.F       the vertices and the triangles of the boundary
#   grains.volume, grains.centroid, grains.surfaceArea, grains.numFaces, grains.extent
#
# The volume and the centroid are the divergence theorem over the oriented triangles, the
# tetrahedra from the origin, as MATLAB's volume and centroid.
from __future__ import annotations

import numpy as np

from ..geometry.orientation import orientation
from ..geometry.quaternion import rotation
from ..geometry.vector3d import vector3d


class grain3d:

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

  def __len__(self):
    return len(self.id)

  @property
  def size(self):
    return len(self.id)

  @property
  def isIndexed(self):
    return np.array([self.CSList[int(p)].isIndexed for p in self.phaseId], dtype=bool)

  @property
  def meanOrientation(self):
    p = np.unique(self.phaseId)
    if len(p) != 1:
      raise ValueError("the grains are of several phases; select one first")
    q = self.meanRotation.data.reshape(-1, 4)
    from ..geometry.frames import specimenFrame
    return orientation(rotation._fromData(q), self.CSList[int(p[0])], self.frame.stripSym() if hasattr(self.frame, "stripSym") else specimenFrame.default)

  # ------------------------------------------------------------------ the polyhedra

  @property
  def F(self):
    return self.boundary.F

  @property
  def allV(self):
    return vector3d(self.boundary.allV, frame=self.frame)

  @property
  def V(self):
    return self.boundary.V

  @property
  def numFaces(self):
    return np.asarray((self.I_GF != 0).sum(axis=1)).reshape(-1)

  # six times the signed volume of the tetrahedron of every triangle and the origin
  def _tetra(self):
    A = self.boundary._corners()
    return np.einsum("ij,ij->i", A[:, 0], np.cross(A[:, 1] - A[:, 0], A[:, 2] - A[:, 0]))

  @property
  def volume(self):
    return self.I_GF @ self._tetra() / 6

  @property
  def centroid(self):
    A = self.boundary._corners()
    a, b, c = A[:, 0], A[:, 1], A[:, 2]
    N = np.cross(b - a, c - a)
    m = N * ((a + b) ** 2 + (b + c) ** 2 + (c + a) ** 2) / 48
    return vector3d((self.I_GF @ m) / self.volume[:, None], frame=self.frame)

  @property
  def surfaceArea(self):
    return np.abs(self.I_GF) @ self.boundary.area

  @property
  def extent(self):
    P = self.boundary.allV[self.boundary.idV]
    return np.array([P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max(), P[:, 2].min(), P[:, 2].max()])

  def __repr__(self):
    from ..maps.phaselist import phaseTable
    ids, counts = np.unique(self.phaseId, return_counts=True)
    lines = [f"grain3d ({self.frame.char()})", phaseTable(self.CSList.frames, ids, [str(c) for c in counts])]
    lines.append(f"  boundary faces: {len(self.boundary)}")
    return "\n".join(lines)


__all__ = ["grain3d"]
