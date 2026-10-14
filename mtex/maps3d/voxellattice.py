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

from ..geometry.kernels import columnMax, columnMin
from ..geometry.display import aligned
from ..maps.kernels import _compiledLoop

prange = range

# a quarter cell: half of what it takes to round onto the neighbouring voxel
INDEX_TOL = 0.25
# the unit cell as MATLAB's EBSD3square lists its corners, in units of d1, d2, d3
CORNERS = 0.5 * np.array([[1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1], [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1]], dtype=float)


# the index (i, j, k) of every voxel of a box in the order of a C array, and its position
# origin + i d1 + j d2 + k d3, written over the cores: np.indices and the product with D make
# three passes over 48 bytes a voxel, a second for the 13 million voxels of a volume
def _boxLoop(origin, D, n1, n2, n3, ijk, P):
  for i in prange(n1):
    for j in range(n2):
      for k in range(n3):
        r = (i * n2 + j) * n3 + k
        ijk[r, 0] = i
        ijk[r, 1] = j
        ijk[r, 2] = k
        for c in range(3):
          P[r, c] = (i * D[0, c] + j * D[1, c] + k * D[2, c]) + origin[c]


def boxLattice(origin, D, shape):
  """the lattice of every voxel of a box of the shape, in the order of a C array, and the
  positions of its voxels as an array of the shape"""
  origin, D = np.asarray(origin, dtype=float).reshape(3), np.asarray(D, dtype=float).reshape(3, 3)
  loop = _compiledLoop(_boxLoop, parallel=True)
  if loop is None:
    lat = voxelLattice(origin, D, np.indices(shape).reshape(3, -1).T)
    return lat, lat.positions().reshape(tuple(shape) + (3,))
  n = int(np.prod(shape))
  ijk, P = np.empty((n, 3), dtype=np.int64), np.empty((n, 3))
  loop(origin, D, shape[0], shape[1], shape[2], ijk, P)
  return voxelLattice(origin, D, ijk), P.reshape(tuple(shape) + (3,))


class voxelLattice:
  """the lattice a volume is measured on: the origin, the translations d1, d2, d3 as the rows of D
  and the integer index of every voxel, voxel n at origin + ijk[n] @ D

    voxelLattice(origin, D, ijk)
    voxelLattice.fromPositions(P)"""

  def __init__(self, origin, D, ijk):
    self.origin = np.asarray(origin, dtype=float).reshape(3)
    self.D = np.asarray(D, dtype=float).reshape(3, 3)
    self.ijk = np.asarray(ijk, dtype=np.int64).reshape(-1, 3)

  @property
  def shape(self):
    """the extent of the box of voxels the index spans"""
    return tuple(int(n) for n in columnMax(self.ijk) + 1) if len(self.ijk) else (0, 0, 0)

  @property
  def spacing(self):
    """the lengths of the three translations d1, d2, d3"""
    return np.linalg.norm(self.D, axis=1)

  @property
  def unitCell(self):
    """the corners of the voxel about the origin"""
    return CORNERS @ self.D

  def subset(self, flat):
    """the lattice of the voxels selected by a flat index or mask, origin and translations kept"""
    return voxelLattice(self.origin, self.D, self.ijk[np.asarray(flat)])

  def positions(self, ijk=None):
    """the positions of the voxels of the lattice, or of given indices

      P = ebsd3.lattice.positions()                            # every voxel of the index, (n, 3)
      P = ebsd3.lattice.positions([[1, 2, 3]])                 # the voxel (1, 2, 3)

    The position of the index (i, j, k) is origin + i d1 + j d2 + k d3. The lattice of a
    volume is ebsd3.lattice; this port's, MATLAB keeps the unit cell and d1, d2, d3 instead.

    Output
      P - (..., 3) array of positions in the scan unit
    Keywords
      ijk - (..., 3) integer indices; by default the index of the lattice, l.ijk
    See also
      voxelLattice.fromPositions, voxelLattice.index"""
    ijk = self.ijk if ijk is None else np.asarray(ijk)
    return self.origin + ijk @ self.D

  def index(self, P):
    """the real index of positions: integer on the voxel centres"""
    return np.linalg.solve(self.D.T, (np.asarray(P, dtype=float).reshape(-1, 3) - self.origin).T).T

  @classmethod
  def fromPositions(cls, P, D=None):
    """the lattice of voxels positions lie on

      l = voxelLattice.fromPositions(P)                        # the translations inferred
      l = voxelLattice.fromPositions(P, D)                     # known translations, the rows of D

    Without D the translations d1, d2, d3 are the three shortest independent differences from
    a voxel to its neighbours, ordered and signed to run along x, y and z as far as they can,
    as MATLAB's gridify orders the edges of the unit cell. The index of every position is
    rounded and must lie within a quarter cell of an integer; two positions on one voxel, or
    positions off one lattice, raise a ValueError. The origin is the corner of the box of
    indices, so every index is nonnegative.

    Input
      P - (n, 3) array of positions, at least four
    Output
      l - voxelLattice
    Keywords
      D - the three translations as the rows of a 3 x 3 array
    See also
      voxelLattice.positions, EBSD3.gridify"""
    P = np.asarray(P, dtype=float).reshape(-1, 3)
    if len(P) < 4:
      raise ValueError("a volume needs at least four voxels")
    D = _translations(P) if D is None else np.asarray(D, dtype=float).reshape(3, 3)
    t = np.linalg.solve(D.T, (P - P[0]).T).T
    ijk = np.rint(t)
    if np.max(np.abs(t - ijk)) > INDEX_TOL:
      raise ValueError("the positions are not on one lattice of voxels")
    lo = columnMin(ijk)
    ijk = (ijk - lo).astype(np.int64)
    if len(np.unique(ijk @ np.array([1, 1 << 20, 1 << 40], dtype=np.int64))) != len(ijk):
      raise ValueError("two positions fall onto one voxel, so they are not on a lattice")
    return cls(P[0] + lo @ D, D, ijk)

  def __repr__(self):
    d = ", ".join(np.array2string(r, precision=4) for r in self.D)
    return aligned(f"voxelLattice {self.shape}, d1 d2 d3 = {d}")


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
