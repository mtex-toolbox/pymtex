# the lattice a volume is measured on: three translations d1, d2, d3 and the integer index of
# every voxel (MATLAB EBSD3square's d1, d2, d3 and the unit cell of EBSD3)
#
#   voxelLattice(origin, D, ijk)          voxel ijk[n] sits at origin + ijk[n] @ D, D's rows d1, d2, d3
#   voxelLattice.fromPositions(P)         the translations inferred, the index checked
#   voxelLattice.fromPositions(P, D)      the same with known translations
#
# The translations are the three shortest independent differences to the neighbours of a
# voxel, ordered and signed to run along x, y and z as far as they can, as MATLAB's gridify
# orders the edges of the unit cell. The index is one rounding, checked to a quarter cell.
from __future__ import annotations

import numpy as np

# a quarter cell: half of what it takes to round onto the neighbouring voxel
INDEX_TOL = 0.25
# the unit cell as MATLAB's EBSD3square lists its corners, in units of d1, d2, d3
CORNERS = 0.5 * np.array([[1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1], [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1]], dtype=float)


class voxelLattice:

  def __init__(self, origin, D, ijk):
    self.origin = np.asarray(origin, dtype=float).reshape(3)
    self.D = np.asarray(D, dtype=float).reshape(3, 3)
    self.ijk = np.asarray(ijk, dtype=np.int64).reshape(-1, 3)

  # the extent of the box of voxels the index spans
  @property
  def shape(self):
    return tuple(int(n) for n in self.ijk.max(axis=0) + 1) if len(self.ijk) else (0, 0, 0)

  @property
  def spacing(self):
    return np.linalg.norm(self.D, axis=1)

  # the corners of the voxel about the origin
  @property
  def unitCell(self):
    return CORNERS @ self.D

  def subset(self, flat):
    return voxelLattice(self.origin, self.D, self.ijk[np.asarray(flat)])

  # the positions of every voxel of the index, or of an array of indices
  def positions(self, ijk=None):
    ijk = self.ijk if ijk is None else np.asarray(ijk)
    return self.origin + ijk @ self.D

  # the real index of positions: integer on the voxel centres
  def index(self, P):
    return np.linalg.solve(self.D.T, (np.asarray(P, dtype=float).reshape(-1, 3) - self.origin).T).T

  @classmethod
  def fromPositions(cls, P, D=None):
    P = np.asarray(P, dtype=float).reshape(-1, 3)
    if len(P) < 4:
      raise ValueError("a volume needs at least four voxels")
    D = _translations(P) if D is None else np.asarray(D, dtype=float).reshape(3, 3)
    t = np.linalg.solve(D.T, (P - P[0]).T).T
    ijk = np.rint(t)
    if np.max(np.abs(t - ijk)) > INDEX_TOL:
      raise ValueError("the positions are not on one lattice of voxels")
    lo = ijk.min(axis=0)
    ijk = (ijk - lo).astype(np.int64)
    if len(np.unique(ijk @ np.array([1, 1 << 20, 1 << 40], dtype=np.int64))) != len(ijk):
      raise ValueError("two positions fall onto one voxel, so they are not on a lattice")
    return cls(P[0] + lo @ D, D, ijk)

  def __repr__(self):
    d = ", ".join(np.array2string(r, precision=4) for r in self.D)
    return f"voxelLattice {self.shape}, d1 d2 d3 = {d}"


# the three shortest independent differences from a voxel to its neighbours, ordered along
# x, y, z and pointing forward
def _translations(P):
  from scipy.spatial import cKDTree
  tree = cKDTree(P)
  # a voxel away from the border has all its neighbours; the first of a few that finds three
  for i in np.linspace(0, len(P) - 1, min(len(P), 16)).astype(int):
    _, j = tree.query(P[i], k=min(len(P), 27))
    E = P[np.atleast_1d(j)] - P[i]
    E = E[np.linalg.norm(E, axis=1) > 0]
    E = E[np.argsort(np.linalg.norm(E, axis=1), kind="stable")]
    D = _independent(E)
    if D is not None:
      return _alongAxes(D)
  raise ValueError("the positions do not span three directions")


def _independent(E):
  if not len(E):
    return None
  d = [E[0]]
  scale = np.linalg.norm(E[0])
  for e in E[1:]:
    if len(d) == 1 and np.linalg.norm(np.cross(d[0], e)) > 1e-6 * scale * np.linalg.norm(e):
      d.append(e)
    elif len(d) == 2 and abs(np.linalg.det(np.array([d[0], d[1], e]))) > 1e-6 * scale * scale * np.linalg.norm(e):
      d.append(e)
      return np.array(d)
  return None


def _alongAxes(D):
  M = np.abs(D / np.linalg.norm(D, axis=1, keepdims=True))
  order = np.zeros(3, dtype=int)
  for _ in range(3):
    r, a = np.unravel_index(np.argmax(M), M.shape)
    order[a] = r
    M[r, :] = -1
    M[:, a] = -1
  D = D[order]
  return D * np.where(np.diag(D) < 0, -1.0, 1.0)[:, None]


__all__ = ["voxelLattice"]
