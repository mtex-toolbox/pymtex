# a k-d tree over the quaternions of an orientation grid, the nearest and radius search the
# ODF reconstruction needs (evaluating a sum of local kernels centred at the grid points).
# The grid is one fundamental region, so a query is symmetrised and every image searched; the
# union of the hits is the symmetry-reduced neighbourhood. It uses only the grid's points, so
# it serves any SO(3) grid - the ring grid or the Fibonacci grid - the same way. An
# orientation angle w maps to the Euclidean radius 2 sin(w/4) on the double cover [q, -q], so
# a ball on the tree is a misorientation ball. The searches return cos(w/2), the absolute
# quaternion dot, which the kernels take and which needs no arccos.
from __future__ import annotations

import itertools

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from ..symmetryops import symmetriseOrientation


# cos(w/2) from the chord d between unit quaternions, |q - g|^2 = 2 - 2 q.g
def _cosFromChord(d):
  return np.clip(1 - d * d / 2, -1.0, 1.0)


def _angle(c):
  return 2 * np.arccos(np.minimum(c, 1.0))


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

  def nearest(self, o):
    """the nearest grid rotation of each query and cos(w/2) of the reduced angle, over the images. A
    distance-bounded query, widened until every query is resolved: scipy's plain nearest
    query prunes poorly on the 4-sphere, the bound makes it prune to the region"""
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
    return idx, _cosFromChord(np.where(np.isfinite(best), best, 2.0))

  def knn(self, o, k):
    """the k nearest grid rotations of each query and cos(w/2) of their reduced angles, distinct grid points"""
    imgs = self._images(o)
    nq, S = imgs.shape[0], imgs.shape[1]
    flat = imgs.reshape(-1, 4)
    outI = np.zeros((nq, k), dtype=int)
    outD = np.zeros((nq, k))
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
          outD[q] = _cosFromChord(np.array(gd))
          todo[q] = False
      delta *= 1.7
      if not todo.any() or delta >= np.pi:
        break
    return outI, outD

  def radius(self, o, eps):
    """every grid rotation within eps of each query, up to symmetry: the sparse incidence
    (nq x n) and the sparse cos(w/2) of the reduced angles, the max over the images that reach a grid point"""
    r = 2 * np.sin(eps / 4)
    imgs = self._images(o)
    nq, S = imgs.shape[0], imgs.shape[1]
    flat = imgs.reshape(-1, 4)
    hits = self.tree.query_ball_point(flat, r, workers=-1)
    # the hit lists flattened without a Python loop: their lengths, then their items chained
    lens = np.fromiter(map(len, hits), dtype=np.intp, count=len(hits))
    if not lens.any():
      shape = (nq, self.n)
      return sparse.csr_matrix(shape, dtype=bool), sparse.csr_matrix(shape)
    cols = np.fromiter(itertools.chain.from_iterable(hits), dtype=np.intp, count=int(lens.sum())) % self.n
    m = np.repeat(np.arange(len(hits)), lens)
    rows = m // S
    dots = np.abs(np.einsum("ij,ij->i", flat[m], self.pts[cols]))
    # keep, per (query, grid) pair, the largest dot (the smallest angle) among the images: one
    # sort on the pair's key, which also leaves the rows in the order of a CSR matrix
    key = rows.astype(np.int64) * self.n + cols
    order = np.argsort(key)
    key, dots = key[order], dots[order]
    starts = np.flatnonzero(np.concatenate([[True], key[1:] != key[:-1]]))
    dots, key = np.maximum.reduceat(dots, starts), key[starts]
    rows, cols = key // self.n, key % self.n
    indptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=nq))])
    shape = (nq, self.n)
    I = sparse.csr_matrix((np.ones(len(cols), dtype=bool), cols, indptr), shape=shape)
    D = sparse.csr_matrix((np.minimum(dots, 1.0), cols.copy(), indptr.copy()), shape=shape)
    return I, D


def orientationTree(grid):
  """the tree of a grid, cached on it and built on first search; of any other orientations a
  fresh one"""
  from .so3grid import SO3Grid
  from .superfibonacci import SO3FibonacciGrid
  if not isinstance(grid, (SO3Grid, SO3FibonacciGrid)):
    return OrientationTree(grid.flatten())
  if getattr(grid, "_otree", None) is None:
    grid._otree = OrientationTree(grid)
  return grid._otree


def findInGrid(grid, o, epsOrK=None):
  """the MATLAB find on an orientation grid through the tree: nearest (eps None), k nearest (int),
  or all within a radius (float), returning the MATLAB shapes; the nearest form gives cos(w/2)
  of the misorientation angle w as MATLAB's SO3Grid/find does, the other two the angles"""
  from ..orientation import orientation
  o = o if isinstance(o, orientation) else orientation(o, grid.frameA, grid.frameB)
  if epsOrK is None:
    ind, d = orientationTree(grid).nearest(o.flatten())
    return ind.reshape(o.shape), d.reshape(o.shape)
  if isinstance(epsOrK, (int, np.integer)) and not isinstance(epsOrK, bool):
    ind, c = orientationTree(grid).knn(o.flatten(), int(epsOrK))
    return ind, _angle(c)
  I, C = gridRadius(grid, o.flatten(), float(epsOrK))
  C.data = _angle(C.data)
  return I, C


def gridRadius(grid, o, eps):
  """every rotation of a grid or list within eps of each query, up to symmetry, as the sparse
  incidence and the sparse cos(w/2): the structured search of an equispaced grid, else the tree"""
  from .so3grid import SO3Grid
  return (grid if isinstance(grid, SO3Grid) else orientationTree(grid)).radius(o, eps)


__all__ = ["OrientationTree", "orientationTree", "findInGrid", "gridRadius"]
