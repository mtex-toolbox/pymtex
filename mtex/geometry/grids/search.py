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
from ..vector3d import vector3d


def _distances(v, w, antipodal=False):
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
  if isinstance(epsOrK, (int, np.integer)) and not isinstance(epsOrK, bool):
    ind, c = tree.knn(w.flatten(), int(epsOrK))
    return ind, _angle(c)
  I, C = tree.radius(w.flatten(), float(epsOrK))
  C = C.copy()
  C.data = _angle(C.data)
  return I, C


def find(v, w, epsOrK=None, antipodal=False):
  """the nearest neighbours in a plain list of directions or orientations by the symmetry-aware
  angle: the closest element of v to each w, the k closest, or all within a radius as two sparse
  matrices (len(w) x len(v)) of hits and angles

    ind, d = find(v, w)
    ind, d = find(v, w, k)              k an integer
    I, D = find(v, w, 5*degree)         a radius, a float"""
  if isinstance(v, orientation) and isinstance(w, orientation) and not v.CS.isclose(w.CS):
    raise ValueError(f"cannot search orientations of {v.CS.char()} among orientations of {w.CS.char()}")
  if isinstance(v, orientation) and w.size >= TREE_QUERIES and v.size * w.size >= TREE_PAIRS:
    return _findByTree(v, w, epsOrK)
  d = _distances(v, w, antipodal)
  M, N = d.shape
  if epsOrK is None:
    ind = np.argmin(d, axis=1)
    return ind.reshape(w.shape), np.take_along_axis(d, ind[:, None], axis=1)[:, 0].reshape(w.shape)
  if isinstance(epsOrK, (int, np.integer)) and not isinstance(epsOrK, bool):
    k = int(epsOrK)
    order = np.argsort(d, axis=1, kind="stable")[:, :k]
    return order, np.take_along_axis(d, order, axis=1)
  eps = float(epsOrK)
  rows, cols = np.nonzero(d <= eps)
  I = sparse.csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(M, N))
  D = sparse.csr_matrix((d[rows, cols], (rows, cols)), shape=(M, N))
  return I, D


__all__ = []
