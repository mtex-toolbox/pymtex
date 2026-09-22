# polygons in the plane of a map: the convex cells of the decomposition and the measures of
# a list of outlines
#
#   voronoiCell(p, nb, w)    the Voronoi cell of p among the sites p + nb, clipped to a box of half width w
#   clipHalfPlane(vs, gs, p, q, gen)   the polygon cut to the points at least as close to p as to q
#   signedArea2(vs)          twice the signed area, positive anticlockwise
#   polygonSet               area, perimeter, centroid, diameter, point in polygon over the loops
#
# The clipping is the one routine the unit cell of a lattice and the spatial decomposition
# share; it tracks which neighbour made each edge, so a cell hands back its faces with the
# site on the other side, and the NumPy form here is the oracle of the compiled loop of M2.
# The polygon set is what a list of outlines answers through its loops, the pool of 2d
# vertices they index into and the plane of the map they lie in: the grains are one such
# list and a shape2d another.
from __future__ import annotations

import numpy as np

from ..geometry.vector3d import vector3d


# the polygon (vs, gs) clipped against the half plane of the points at least as close to p
# as to q; gs[i] is the generator of the edge vs[i] -> vs[i + 1]
def clipHalfPlane(vs, gs, p, q, gen):
  n = q - p
  c = (n @ q + n @ p) / 2
  m = len(vs)
  ws, hs = [], []
  for i in range(m):
    v1, v2 = vs[i], vs[(i + 1) % m]
    d1, d2 = n @ v1 - c, n @ v2 - c
    if d1 <= 0:
      ws.append(v1)
      hs.append(gs[i])
      if d2 > 0:
        # leaving the half plane: the cut edge is q's
        ws.append(v1 + (d1 / (d1 - d2)) * (v2 - v1))
        hs.append(gen)
    elif d2 <= 0:
      # entering it again: the edge keeps its generator
      ws.append(v1 + (d1 / (d1 - d2)) * (v2 - v1))
      hs.append(gs[i])
  return ws, hs


# the square of half width w centred on p, as the starting polygon of a clip
def seedBox(p, w):
  p = np.asarray(p, dtype=float)
  return [p + d for d in (np.array([-w, -w]), np.array([w, -w]), np.array([w, w]), np.array([-w, w]))], [0, 0, 0, 0]


# the Voronoi cell of p among the sites p + nb, as an (m, 2) array of corners, anticlockwise
def voronoiCell(p, nb, w):
  p = np.asarray(p, dtype=float)
  vs, gs = seedBox(p, float(w))
  for k, d in enumerate(nb):
    vs, gs = clipHalfPlane(vs, gs, p, p + np.asarray(d, dtype=float), k + 1)
  return np.array(vs, dtype=float).reshape(-1, 2)


# twice the signed area of a closed polygon, positive when it winds anticlockwise
def signedArea2(vs):
  vs = np.asarray(vs, dtype=float)
  x, y = vs[:, 0], vs[:, 1]
  return float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


__all__ = []


def _cross2(a, b):
  return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


# whether points (m, 2) lie inside a loop, by the crossing number of a ray to the right
def _insideLoop(V, loop, q):
  P, Q = V[loop], np.roll(V[loop], -1, axis=0)
  qx, qy = q[:, 0][:, None], q[:, 1][:, None]
  crosses = (P[None, :, 1] > qy) != (Q[None, :, 1] > qy)
  with np.errstate(divide="ignore", invalid="ignore"):
    x = P[None, :, 0] + (qy - P[None, :, 1]) / (Q[None, :, 1] - P[None, :, 1]) * (Q[None, :, 0] - P[None, :, 0])
  return np.sum(crosses & (x > qx), axis=1) % 2 == 1


# ------------------------------------------------------------------ a set of polygons

# every loop of every outline as one edge list: the tails P, the heads Q, the outline each
# edge belongs to and whether it lies on the outer loop; one pass over the loops, so that
# every measure below is a bincount and not a loop over the outlines
def loopEdges(V, loopsPerOutline):
  idx, lab, outer = [], [], []
  for k, ls in enumerate(loopsPerOutline):
    for j, l in enumerate(ls):
      if len(l):
        idx.append(l)
        lab.append(np.full(len(l), k, dtype=np.int64))
        outer.append(np.full(len(l), j == 0))
  if not idx:
    return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool)
  lengths = np.array([len(l) for l in idx])
  starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
  flat = np.concatenate(idx)
  pos = np.arange(len(flat)) - np.repeat(starts, lengths)
  nxt = flat[np.repeat(starts, lengths) + (pos + 1) % np.repeat(lengths, lengths)]
  return V[flat], V[nxt], np.concatenate(lab), np.concatenate(outer)


# the measures of a list of outlines, read off loops, V, plane, frame and size of the class
# that carries them. Every measure is a sum over the loops, and the loops carry their
# winding, anticlockwise outside and clockwise around a hole, so a hole is subtracted by
# itself and nothing has to know it is there
class polygonSet:

  def _edges(self):
    return loopEdges(self.V, self.loops)

  # the loops of one outline as one vertex list, each hole stitched to the outer loop by
  # repeating its first vertex, the form a renderer or a point in polygon test wants
  def keyhole(self, k):
    ls = self.loops[k]
    if not ls:
      return np.zeros(0, dtype=np.int32)
    out = list(ls[0]) + [ls[0][0]]
    for h in ls[1:]:
      out += list(h) + [h[0], ls[0][0]]
    return np.array(out, dtype=np.int32)

  # the flattened polygon of every outline, MATLAB's poly
  @property
  def poly(self):
    return [self.keyhole(k) for k in range(self.size)]

  def _as2d(self, q):
    q = q.data if isinstance(q, vector3d) else np.asarray(q, dtype=float)
    q = q.reshape(-1, q.shape[-1]) if q.ndim > 1 else q.reshape(1, -1)
    return q if q.shape[1] == 2 else self.plane.to2d(q)

  # whether points lie inside the outlines: within the outer loop and outside the holes,
  # as a (points, outlines) matrix
  def checkInside(self, xy):
    q = self._as2d(xy)
    out = np.zeros((len(q), self.size), dtype=bool)
    V = self.V
    for k, ls in enumerate(self.loops):
      if not ls:
        continue
      inside = _insideLoop(V, ls[0], q)
      for h in ls[1:]:
        inside &= ~_insideLoop(V, h, q)
      out[:, k] = inside
    return out

  # the positions of the outlines a point lies in
  def inpolygon(self, xy):
    return np.flatnonzero(self.checkInside(xy).any(axis=0))

  # the area an outline covers, its holes taken off
  @property
  def area(self):
    P, Q, lab, _ = self._edges()
    return np.bincount(lab, _cross2(P, Q), self.size) / 2

  # the length of the outer loop of an outline, as MATLAB's default; the loops around the
  # inclusions come with perimeterWithInclusion
  @property
  def perimeter(self):
    out = np.zeros(self.size)
    for k, ls in enumerate(self.loops):
      if ls:
        l = np.asarray(ls[0])
        out[k] = np.sum(np.linalg.norm(self.V[np.roll(l, -1)] - self.V[l], axis=1))
    return out

  @property
  def perimeterWithInclusion(self):
    P, Q, lab, _ = self._edges()
    return np.bincount(lab, np.linalg.norm(Q - P, axis=1), self.size)

  @property
  def boundarySize(self):
    return np.array([sum(len(l) for l in ls) for ls in self.loops], dtype=np.int32)

  # the centroid of the area, holes taken off, as a framed point
  @property
  def centroid(self):
    P, Q, lab, _ = self._edges()
    w = _cross2(P, Q)
    a = np.bincount(lab, w, self.size)
    c = np.stack([np.bincount(lab, w * (P[:, 0] + Q[:, 0]), self.size), np.bincount(lab, w * (P[:, 1] + Q[:, 1]), self.size)], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
      out = np.where(a[:, None] != 0, c / (3 * a[:, None]), 0.0)
    return vector3d(self.plane.from2d(out), frame=self.frame)

  @property
  def equivalentRadius(self):
    return np.sqrt(np.maximum(self.area, 0) / np.pi)

  @property
  def equivalentPerimeter(self):
    return 2 * np.pi * self.equivalentRadius

  # the largest distance between two vertices of the outer loop
  @property
  def diameter(self):
    out = np.zeros(self.size)
    for k, ls in enumerate(self.loops):
      if ls:
        P = self.V[ls[0]]
        d = P[:, None, :] - P[None, :, :]
        out[k] = np.sqrt(np.max(np.sum(d * d, axis=-1)))
    return out

  @property
  def hasHole(self):
    return np.array([len(ls) > 1 for ls in self.loops], dtype=bool)
