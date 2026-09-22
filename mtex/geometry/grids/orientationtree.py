# a k-d tree over the quaternions of an orientation grid, the nearest and radius search the
# ODF reconstruction needs (evaluating a sum of local kernels centred at the grid points).
# The grid is one fundamental region, so a query is symmetrised and every image searched; the
# union of the hits is the symmetry-reduced neighbourhood. It uses only the grid's points, so
# it serves any SO(3) grid - the ring grid or the Fibonacci grid - the same way. An
# orientation angle w maps to the Euclidean radius 2 sin(w/4) on the double cover [q, -q], so
# a ball on the tree is a misorientation ball.
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from ..symmetryops import symmetriseOrientation


def _angleFromChord(d):
  return 2 * np.arccos(np.clip(1 - d * d / 2, -1.0, 1.0))


class OrientationTree:

  def __init__(self, grid):
    self.n = grid.size
    self.res = float(getattr(grid, "resolution", 5 * np.pi / 180))
    pts = grid.data.reshape(-1, 4)
    self.pts = pts
    # the double cover: q and -q are the same rotation, so both go in and a Euclidean ball
    # catches the nearer of the two
    self.tree = cKDTree(np.vstack([pts, -pts]))

  # the symmetric images of the queries, shape (nq, S, 4); S = |L| |R| (doubled under exchange)
  def _images(self, o):
    imgs = symmetriseOrientation(o)
    return np.moveaxis(imgs.data.reshape(imgs.data.shape[0], -1, 4), 0, 1)

  # the nearest grid rotation of each query and the reduced angle, over the images. A
  # distance-bounded query, widened until every query is resolved: scipy's plain nearest
  # query prunes poorly on the 4-sphere, the bound makes it prune to the region
  def nearest(self, o):
    imgs = self._images(o)
    nq, S = imgs.shape[0], imgs.shape[1]
    flat = imgs.reshape(-1, 4)
    best = np.full(nq, np.inf)
    idx = np.zeros(nq, dtype=int)
    todo = np.ones(nq, dtype=bool)
    delta = 2.5 * self.res
    while True:
      rows = np.repeat(todo, S)
      dist, ii = self.tree.query(flat[rows], k=1, distance_upper_bound=2 * np.sin(delta / 4), workers=-1)
      dist = dist.reshape(-1, S)
      ii = (ii % self.n).reshape(-1, S)
      k = np.argmin(dist, axis=1)
      r = np.arange(dist.shape[0])
      d, g = dist[r, k], ii[r, k]
      where = np.flatnonzero(todo)
      hit = np.isfinite(d)
      best[where[hit]] = d[hit]
      idx[where[hit]] = g[hit]
      todo[where[hit]] = False
      delta *= 1.7
      if not todo.any() or delta >= np.pi:
        break
    return idx, _angleFromChord(np.where(np.isfinite(best), best, 2.0))

  # the k nearest grid rotations of each query and their reduced angles, distinct grid points
  def knn(self, o, k):
    imgs = self._images(o)
    nq, S = imgs.shape[0], imgs.shape[1]
    flat = imgs.reshape(-1, 4)
    outI = np.zeros((nq, k), dtype=int)
    outD = np.full((nq, k), np.pi)
    todo = np.ones(nq, dtype=bool)
    delta = 3.0 * self.res * np.cbrt(k)
    while True:
      rows = np.repeat(todo, S)
      dist, ii = self.tree.query(flat[rows], k=min(k, 2 * self.n),
                                 distance_upper_bound=2 * np.sin(min(delta, np.pi) / 4), workers=-1)
      dist = np.atleast_2d(dist.reshape(int(todo.sum()), -1))
      gg = (np.atleast_2d(ii.reshape(int(todo.sum()), -1)) % self.n)
      where = np.flatnonzero(todo)
      for m, q in enumerate(where):
        order = np.argsort(dist[m], kind="stable")
        seen, gi, gd = {}, [], []
        for p in order:
          dv = dist[m, p]
          if not np.isfinite(dv):
            break
          g = int(gg[m, p])
          if g in seen:
            continue
          seen[g] = True
          gi.append(g)
          gd.append(dv)
          if len(gi) == k:
            break
        if len(gi) == k:
          outI[q] = gi
          outD[q] = _angleFromChord(np.array(gd))
          todo[q] = False
      delta *= 1.7
      if not todo.any() or delta >= np.pi:
        break
    return outI, outD

  # every grid rotation within eps of each query, up to symmetry: the sparse incidence
  # (nq x n) and the sparse reduced angles, the min over the images that reach a grid point
  def radius(self, o, eps):
    r = 2 * np.sin(eps / 4)
    imgs = self._images(o)
    nq, S = imgs.shape[0], imgs.shape[1]
    flat = imgs.reshape(-1, 4)
    hits = self.tree.query_ball_point(flat, r, workers=-1)
    rows, cols, dots = [], [], []
    for m, hit in enumerate(hits):
      if not hit:
        continue
      i = m // S
      g = np.asarray(hit, dtype=int) % self.n
      d = np.abs(flat[m] @ self.pts[g].T)
      rows.append(np.full(len(g), i))
      cols.append(g)
      dots.append(d)
    if not rows:
      shape = (nq, self.n)
      return sparse.csr_matrix(shape, dtype=bool), sparse.csr_matrix(shape)
    rows, cols, dots = np.concatenate(rows), np.concatenate(cols), np.concatenate(dots)
    # keep, per (query, grid) pair, the largest dot (the smallest angle) among the images
    order = np.lexsort((-dots, cols, rows))
    rows, cols, dots = rows[order], cols[order], dots[order]
    first = np.ones(len(rows), dtype=bool)
    first[1:] = (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])
    rows, cols, dots = rows[first], cols[first], dots[first]
    ang = 2 * np.arccos(np.minimum(dots, 1.0))
    shape = (nq, self.n)
    I = sparse.csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=shape)
    D = sparse.csr_matrix((ang, (rows, cols)), shape=shape)
    return I, D


# the cached tree of a grid, built on first search
def orientationTree(grid):
  if getattr(grid, "_otree", None) is None:
    grid._otree = OrientationTree(grid)
  return grid._otree


# the MATLAB find on an orientation grid through the tree: nearest (eps None), k nearest (int),
# or all within a radius (float), returning the MATLAB shapes
def findInGrid(grid, o, epsOrK=None):
  from ..orientation import orientation
  o = o if isinstance(o, orientation) else orientation(o, grid.frameA, grid.frameB)
  tree = orientationTree(grid)
  if epsOrK is None:
    ind, d = tree.nearest(o.flatten())
    return ind.reshape(o.shape), d.reshape(o.shape)
  if isinstance(epsOrK, (int, np.integer)) and not isinstance(epsOrK, bool):
    return tree.knn(o.flatten(), int(epsOrK))
  return tree.radius(o.flatten(), float(epsOrK))


__all__ = ["OrientationTree", "orientationTree", "findInGrid"]
