# the Delaunay triangulation of directions on the sphere, the convex hull of the points
# (MATLAB geometry/@S2Triangulation)
#
#   sT = S2Triangulation(v)            the triangles over the directions v
#   sT.T                               the triangles, three vertex indices each, counterclockwise from outside
#   sT.neighbours                      the triangle opposite each of the three vertices
#   B = sT.calcBario(w)                the barycentric coordinates of w, a sparse matrix, rows w, columns vertices
#
# The directions must surround the origin, as a set covering the sphere does. A direction
# lies in the triangle whose cone holds it; its coordinates are those of the triangle's
# plane met along the ray, normalised to sum one. A direction is located by a walk over the
# neighbours, starting in the triangle with the nearest centre.
from __future__ import annotations

import numpy as np

from .vector3d import vector3d


class S2Triangulation:

  def __init__(self, vertices):
    from scipy.spatial import ConvexHull, cKDTree
    self.vertices = vertices.flatten().normalize()
    V = self.vertices.data.reshape(-1, 3)
    hull = ConvexHull(V)
    T = hull.simplices.copy()
    nb = hull.neighbors.copy()
    # counterclockwise seen from outside: a triangle with its vertices the other way round
    # swaps two of them, with their opposite neighbours
    flip = np.einsum("ij,ij->i", np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]), V[T[:, 0]]) < 0
    T[flip] = T[flip][:, [0, 2, 1]]
    nb[flip] = nb[flip][:, [0, 2, 1]]
    self.T, self.neighbours = T, nb
    mid = V[T].sum(axis=1)
    self.midPoints = vector3d(mid / np.linalg.norm(mid, axis=1, keepdims=True), frame=self.vertices.frame)
    # the coordinates of a point in the basis of a triangle's vertices
    self._inverse = np.linalg.inv(np.swapaxes(V[T], 1, 2))
    self._tree = cKDTree(self.midPoints.data.reshape(-1, 3))

  @property
  def frame(self):
    return self.vertices.frame

  def __len__(self):
    return len(self.T)

  # the triangle every direction lies in and its coordinates there, by a walk from the
  # triangle with the nearest centre towards the vertex a coordinate says is missing
  def locate(self, v):
    W = (v if isinstance(v, vector3d) else vector3d(v)).normalize().data.reshape(-1, 3)
    _, t = self._tree.query(W)
    t = np.asarray(t, dtype=np.int64)
    lam = np.einsum("nij,nj->ni", self._inverse[t], W)
    todo = np.flatnonzero(lam.min(axis=1) < -1e-12)
    for _ in range(len(self.T)):
      if todo.size == 0:
        break
      t[todo] = self.neighbours[t[todo], np.argmin(lam[todo], axis=1)]
      lam[todo] = np.einsum("nij,nj->ni", self._inverse[t[todo]], W[todo])
      todo = todo[lam[todo].min(axis=1) < -1e-12]
    lam = np.maximum(lam, 0)
    return t, lam / lam.sum(axis=1, keepdims=True)

  def calcBario(self, v):
    from scipy.sparse import csr_matrix
    t, lam = self.locate(v)
    n = len(t)
    return csr_matrix((lam.ravel(), (np.repeat(np.arange(n), 3), self.T[t].ravel())), shape=(n, self.vertices.size))

  def checkInside(self, v, t):
    W = (v if isinstance(v, vector3d) else vector3d(v)).normalize().data.reshape(-1, 3)
    return np.einsum("nij,nj->ni", self._inverse[np.asarray(t).ravel()], W).min(axis=1) >= -1e-12

  def __repr__(self):
    return f"S2Triangulation\n  vertices: {self.vertices.size}\n  triangles: {len(self.T)}"


__all__ = ["S2Triangulation"]
