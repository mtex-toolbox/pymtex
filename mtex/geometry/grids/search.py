# nearest neighbours in a plain list of directions or orientations (MATLAB vector3d/find,
# orientation/find): the closest point, the k closest, or all within a radius, by the
# symmetry-aware distance of the objects
#
#   ind, d = find(v, w)          the closest point of v to each w
#   ind, d = find(v, w, k)       the k closest, k an integer
#   I, D = find(v, w, eps)       all within eps, eps a float: sparse (numel(w) x numel(v))
from __future__ import annotations

import numpy as np
from scipy import sparse

from ..orientation import orientation
from ..vector3d import vector3d, asVector3d


# whether the third argument of find is a count k, not a radius
def _isCount(epsOrK):
  return isinstance(epsOrK, (int, np.integer)) and not isinstance(epsOrK, bool)


def _distances(v, w, antipodal=None):
  if isinstance(v, orientation):
    if not isinstance(w, orientation):
      w = orientation(w, v.frameA, v.frameB)
    return v.flatten()[None, :].angle(w.flatten()[:, None])
  return v.flatten()[None, :].angle(w.flatten()[:, None], antipodal=antipodal)


# orientations among many go through the quaternion tree of orientationtree.py, built over the
# list, instead of the matrix of every distance: 1000 queries among 1e5 orientations 54 ms against
# 2.2 s, and 1e4 queries among the 192 thousand of a 2.5 degree regular grid would fill 15 GB. The
# tree wins from a few dozen queries on
TREE_QUERIES, TREE_PAIRS = 32, 100_000


def _findByTree(v, w, epsOrK):
  from .orientationtree import _angle, orientationTree
  w = w if isinstance(w, orientation) else orientation(w, v.frameA, v.frameB)
  # the tree searches the images of the queries: an antipodal list makes them antipodal too
  if v.antipodal and not w.antipodal:
    w = w.copy()
    w.antipodal = True
  tree = orientationTree(v)
  if epsOrK is None:
    ind, c = tree.nearest(w.flatten())
    return ind.reshape(w.shape), _angle(c).reshape(w.shape)
  if _isCount(epsOrK):
    ind, c = tree.knn(w.flatten(), int(epsOrK))
    return ind, _angle(c)
  I, C = tree.radius(w.flatten(), float(epsOrK))
  C = C.copy()
  C.data = _angle(C.data)
  return I, C


# directions among many: a k-d tree over the points, where the Euclidean nearest is the angular
# nearest and the angle eps the chord 2 sin(eps/2); an antipodal search holds every point twice,
# as v and -v. Of a grid the tree is kept on it. Crystal directions measure by their symmetry and
# keep the matrix of every distance
class S2Tree:
  """the k-d tree of a set of directions, the nearest, k nearest and radius search as find gives them"""

  def __init__(self, xyz, antipodal=False):
    from scipy.spatial import cKDTree
    self.pts = np.ascontiguousarray(xyz, dtype=float).reshape(-1, 3)
    self.n, self.antipodal = len(self.pts), bool(antipodal)
    self.tree = cKDTree(np.vstack([self.pts, -self.pts]) if self.antipodal else self.pts)

  # the angles between points and directions, accurate for small angles
  def _angles(self, p, x):
    d = np.sum(p * x, axis=-1)
    return np.arctan2(np.linalg.norm(np.cross(p, x), axis=-1), np.abs(d) if self.antipodal else d)

  def nearest(self, xyz, k=None):
    """the index of the nearest point, or of the k nearest, and the angles"""
    from ..kernels import threads
    x = xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)
    kk = 1 if k is None else (2 * k if self.antipodal else k)
    _, ind = self.tree.query(x, k=kk, workers=threads())
    ind = np.asarray(ind).reshape(len(x), -1) % self.n
    if self.antipodal and k is not None:
      # v and -v of one point both among the 2k: the first k distinct points, in order, a
      # repeat of an earlier column sorted behind them
      repeat = np.triu(ind[:, :, None] == ind[:, None, :], 1).any(axis=1)
      ind = np.take_along_axis(ind, np.argsort(repeat, axis=1, kind="stable")[:, :k], axis=1)
    ind = ind[:, 0] if k is None else ind
    return ind, self._angles(self.pts[ind], x[:, None, :] if k is not None else x)

  def radius(self, xyz, eps):
    """the sparse incidence (directions x points) of the points within eps and the sparse angles"""
    from ..kernels import threads
    x = xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)
    hits = self.tree.query_ball_point(x, 2 * np.sin(min(eps, np.pi) / 2) * (1 + 1e-12), workers=threads())
    counts = np.array([len(h) for h in hits], dtype=np.int64)
    rows = np.repeat(np.arange(len(x)), counts)
    cols = np.concatenate(hits).astype(np.int64) % self.n if counts.sum() else np.zeros(0, dtype=np.int64)
    if self.antipodal and len(cols):
      # a point within eps by v and by -v is one hit
      pair = np.unique(rows * self.n + cols)
      rows, cols = pair // self.n, pair % self.n
    vals = self._angles(self.pts[cols], x[rows])
    keep = vals <= eps
    rows, cols, vals = rows[keep], cols[keep], vals[keep]
    shape = (len(x), self.n)
    return (sparse.csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=shape),
            sparse.csr_matrix((vals, (rows, cols)), shape=shape))


def s2Tree(grid):
  """the tree of a set of directions, kept on a grid and built on first search"""
  tree = getattr(grid, "_s2tree", None)
  if tree is None:
    tree = S2Tree(grid.data, grid.antipodal)
    try:
      grid._s2tree = tree
    except AttributeError:
      pass
  return tree


def find(v, w, epsOrK=None, antipodal=None):
  """the nearest neighbours in a plain list of directions or orientations by the symmetry-aware
  angle: the closest element of v to each w, the k closest, or all within a radius as two sparse
  matrices (len(w) x len(v)) of hits and angles

    ind, d = find(v, w)
    ind, d = find(v, w, k)              k an integer
    I, D = find(v, w, 5*degree)         a radius, a float

  directions are read up to sign where either list is antipodal, antipodal=True always, False never"""
  if isinstance(v, orientation) and isinstance(w, orientation) and not v.CS.isclose(w.CS):
    raise ValueError(f"cannot search orientations of {v.CS.char()} among orientations of {w.CS.char()}")
  if isinstance(v, orientation) and w.size >= TREE_QUERIES and v.size * w.size >= TREE_PAIRS:
    return _findByTree(v, w, epsOrK)
  if isinstance(v, vector3d) and not v.isCrystalDirection and w.size >= TREE_QUERIES and v.size * w.size >= TREE_PAIRS:
    w = asVector3d(w)
    tree = S2Tree(v.flatten().data, (v.antipodal or w.antipodal) if antipodal is None else antipodal)
    xyz = w.flatten().data.reshape(-1, 3)
    if epsOrK is None:
      ind, d = tree.nearest(xyz)
      return ind.reshape(w.shape), d.reshape(w.shape)
    if _isCount(epsOrK):
      return tree.nearest(xyz, int(epsOrK))
    return tree.radius(xyz, float(epsOrK))
  d = _distances(v, w, antipodal)
  M, N = d.shape
  if epsOrK is None:
    ind = np.argmin(d, axis=1)
    return ind.reshape(w.shape), np.take_along_axis(d, ind[:, None], axis=1)[:, 0].reshape(w.shape)
  if _isCount(epsOrK):
    k = int(epsOrK)
    order = np.argsort(d, axis=1, kind="stable")[:, :k]
    return order, np.take_along_axis(d, order, axis=1)
  eps = float(epsOrK)
  rows, cols = np.nonzero(d <= eps)
  I = sparse.csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(M, N))
  D = sparse.csr_matrix((d[rows, cols], (rows, cols)), shape=(M, N))
  return I, D


__all__ = []
