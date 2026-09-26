# grain3Boundary: the faces between the grains of a volume (MATLAB grain3Boundary, ../master)
#
#   gB.allV, gB.V, gB.F                 the vertices and the triangles, (m, 3) indices into allV
#   gB.grainId, gB.ebsdId, gB.phaseId   the two sides of every face, the normal from the first
#                                       to the second; outside the volume a side carries the
#                                       grain 0, the voxel -1 and the phase -1, as in 2D
#   gB.misrotation, gB.misorientation   the rotation from the second side to the first
#   gB.N, gB.centroid, gB.area          per face
#   gB.I_FG, gB.I_VF, gB.I_VG           the incidences of faces, vertices and grains
#   gB[sel], len(gB)                    a subset of faces over the same vertices
from __future__ import annotations

import numpy as np

from ..geometry.orientation import orientation
from ..geometry.quaternion import rotation
from ..geometry.vector3d import vector3d


class grain3Boundary:

  def __init__(self, allV, F, grainId, ebsdId, phaseId, misrotation, CSList, frame, scanUnit="um"):
    self.allV = np.asarray(allV, dtype=float).reshape(-1, 3)
    self.F = np.asarray(F, dtype=np.int64).reshape(-1, 3)
    self.grainId = np.asarray(grainId, dtype=np.int64).reshape(-1, 2)
    self.ebsdId = np.asarray(ebsdId, dtype=np.int64).reshape(-1, 2)
    self.phaseId = np.asarray(phaseId, dtype=np.int8).reshape(-1, 2)
    self.misrotation = misrotation if isinstance(misrotation, rotation) else rotation._fromData(np.asarray(misrotation).reshape(-1, 4))
    self.CSList, self.frame, self.scanUnit = CSList, frame, scanUnit

  def __len__(self):
    return len(self.F)

  @property
  def size(self):
    return len(self.F)

  @property
  def id(self):
    return np.arange(1, len(self.F) + 1)

  def __getitem__(self, sel):
    sel = np.arange(len(self))[sel]
    return grain3Boundary(self.allV, self.F[sel], self.grainId[sel], self.ebsdId[sel], self.phaseId[sel],
                          self.misrotation.data.reshape(-1, 4)[sel], self.CSList, self.frame, self.scanUnit)

  # ------------------------------------------------------------------ geometry

  @property
  def idV(self):
    return np.unique(self.F)

  @property
  def V(self):
    return vector3d(self.allV[self.idV], frame=self.frame)

  # the corners of every triangle, (m, 3, 3)
  def _corners(self):
    return self.allV[self.F]

  def _cross(self):
    A = self._corners()
    return np.cross(A[:, 1] - A[:, 0], A[:, 2] - A[:, 0])

  @property
  def N(self):
    c = self._cross()
    return vector3d(c / np.linalg.norm(c, axis=1, keepdims=True), frame=self.frame)

  @property
  def centroid(self):
    return vector3d(self._corners().mean(axis=1), frame=self.frame)

  @property
  def area(self):
    return np.linalg.norm(self._cross(), axis=1) / 2

  # ------------------------------------------------------------------ the sides

  @property
  def misorientation(self):
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

  # whether a face touches a phase, or lies between two
  def hasPhase(self, p, q=None):
    if q is None:
      return np.any(self.phaseId == p, axis=1)
    return ((self.phaseId[:, 0] == p) & (self.phaseId[:, 1] == q)) | ((self.phaseId[:, 0] == q) & (self.phaseId[:, 1] == p))

  # ------------------------------------------------------------------ incidences

  @property
  def I_FG(self):
    from scipy.sparse import csr_matrix
    f, s = np.nonzero(self.grainId > 0)
    return csr_matrix((np.ones(len(f)), (f, self.grainId[f, s] - 1)), shape=(len(self), int(self.grainId.max(initial=0))))

  @property
  def I_VF(self):
    from scipy.sparse import csr_matrix
    f = np.repeat(np.arange(len(self)), 3)
    return csr_matrix((np.ones(len(f), dtype=bool), (self.F.reshape(-1), f)), shape=(len(self.allV), len(self)))

  @property
  def I_VG(self):
    return (self.I_VF.astype(np.int64) @ self.I_FG) > 0

  def __repr__(self):
    inner = int(np.sum(np.all(self.grainId > 0, axis=1)))
    return f"grain3Boundary\n  faces: {len(self)} ({inner} between grains, {len(self) - inner} on the outside)\n  vertices: {len(self.idV)}"


__all__ = ["grain3Boundary"]
