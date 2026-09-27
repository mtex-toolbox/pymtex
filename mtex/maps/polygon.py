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

from typing import TYPE_CHECKING
import itertools

import numpy as np

from ..geometry.vector3d import vector3d


def clipHalfPlane(vs, gs, p, q, gen):
  """the polygon (vs, gs) clipped against the half plane of the points at least as close to p
  as to q; gs[i] is the generator of the edge vs[i] -> vs[i + 1]"""
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


def seedBox(p, w):
  """the square of half width w centred on p, as the starting polygon of a clip"""
  p = np.asarray(p, dtype=float)
  return [p + d for d in (np.array([-w, -w]), np.array([w, -w]), np.array([w, w]), np.array([-w, w]))], [0, 0, 0, 0]


def voronoiCell(p, nb, w):
  """the Voronoi cell of p among the sites p + nb, as an (m, 2) array of corners, anticlockwise"""
  p = np.asarray(p, dtype=float)
  vs, gs = seedBox(p, float(w))
  for k, d in enumerate(nb):
    vs, gs = clipHalfPlane(vs, gs, p, p + np.asarray(d, dtype=float), k + 1)
  return np.array(vs, dtype=float).reshape(-1, 2)


def signedArea2(vs):
  """twice the signed area of a closed polygon, positive when it winds anticlockwise"""
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

def loopEdges(V, loopsPerOutline):
  """every loop of every outline as one edge list: the tails P, the heads Q, the outline each
  edge belongs to and whether it lies on the outer loop; one pass over the loops, so that
  every measure below is a bincount and not a loop over the outlines"""
  flat, start, lengths, owner, first = _flatLoops(loopsPerOutline)
  if not len(flat):
    return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool)
  starts, size = np.repeat(start, lengths), np.repeat(lengths, lengths)
  nxt = flat[starts + (np.arange(len(flat)) - starts + 1) % size]
  return V[flat], V[nxt], np.repeat(owner, lengths), np.repeat(first, lengths)


# the loops as one vertex list: where each loop starts in it and how long it is, the outline
# it belongs to and whether it comes first there, the outer loop
def _flatLoops(loopsPerOutline):
  nPer = np.fromiter(map(len, loopsPerOutline), dtype=np.int64, count=len(loopsPerOutline))
  loops = list(itertools.chain.from_iterable(loopsPerOutline))
  lengths = np.fromiter(map(len, loops), dtype=np.int64, count=len(loops))
  owner = np.repeat(np.arange(len(nPer)), nPer)
  first = np.arange(len(loops)) == np.repeat(np.cumsum(nPer) - nPer, nPer)
  flat = np.concatenate(loops).astype(np.int64, copy=False) if lengths.any() else np.zeros(0, dtype=np.int64)
  return flat, np.cumsum(lengths) - lengths, lengths, owner, first


def loopSums(V, loopsPerOutline, size):
  """the sums over the edges of every outline, one row each: twice the area, the length of the
  outer loop, the length of all loops, the first moments times six (x, y), the second times
  twelve (y y, x x) and twenty four (x y), all about the origin"""
  from ..geometry.kernels import backend
  V = np.asarray(V, dtype=float)
  loop = None
  if backend() == "numba":
    from .kernels import grainKernel
    loop = grainKernel("loopSums")
  if loop is not None:
    flat, start, lengths, owner, first = _flatLoops(loopsPerOutline)
    out = np.zeros((size, 8))
    loop(np.ascontiguousarray(V), flat, start, lengths, owner, first, out)
    return out
  P, Q, lab, outer = loopEdges(V, loopsPerOutline)
  w = _cross2(P, Q)
  d = np.linalg.norm(Q - P, axis=1)
  (px, py), (qx, qy) = P.T, Q.T
  cols = (w, np.where(outer, d, 0.0), d, (px + qx) * w, (py + qy) * w, (py * py + py * qy + qy * qy) * w,
          (px * px + px * qx + qx * qx) * w, (px * qy + 2 * px * py + 2 * qx * qy + qx * py) * w)
  return np.stack([np.bincount(lab, c, size) for c in cols], axis=1).reshape(size, 8)


class polygonSet:
  """the measures of a list of outlines, read off loops, V, plane, frame and size of the class
  that carries them. Every measure is a sum over the loops, and the loops carry their
  winding, anticlockwise outside and clockwise around a hole, so a hole is subtracted by
  itself and nothing has to know it is there"""

  def keyhole(self, k):
    """the loops of one outline as one vertex list, each hole stitched to the outer loop by
    repeating its first vertex, the form a renderer or a point in polygon test wants"""
    ls = self.loops[k]
    if not ls:
      return np.zeros(0, dtype=np.int32)
    out = list(ls[0]) + [ls[0][0]]
    for h in ls[1:]:
      out += list(h) + [h[0], ls[0][0]]
    return np.array(out, dtype=np.int32)

  @property
  def poly(self):
    """the flattened polygon of every outline, MATLAB's poly"""
    return [self.keyhole(k) for k in range(self.size)]

  def _as2d(self, q):
    q = q.data if isinstance(q, vector3d) else np.asarray(q, dtype=float)
    q = q.reshape(-1, q.shape[-1]) if q.ndim > 1 else q.reshape(1, -1)
    return q if q.shape[1] == 2 else self.plane.to2d(q)

  def checkInside(self, xy):
    """whether points lie inside the outlines: within the outer loop and outside the holes,
    as a (points, outlines) matrix"""
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

  def findByLocation(self, xy):
    """the positions of the outlines a point lies in
    the position of the outline holding each point, -1 for a point in none"""
    hit = self.checkInside(xy)
    return np.where(hit.any(axis=1), np.argmax(hit, axis=1), -1)

  def inpolygon(self, xy):
    return np.flatnonzero(self.checkInside(xy).any(axis=0))

  @property
  def area(self):
    """the area an outline covers, its holes taken off"""
    return loopSums(self.V, self.loops, self.size)[:, 0] / 2

  @property
  def perimeter(self):
    """the length of the outer loop of an outline, as MATLAB's default; the loops around the
    inclusions come with perimeterWithInclusion"""
    return loopSums(self.V, self.loops, self.size)[:, 1]

  @property
  def perimeterWithInclusion(self):
    return loopSums(self.V, self.loops, self.size)[:, 2]

  @property
  def boundarySize(self):
    return np.array([sum(len(l) for l in ls) for ls in self.loops], dtype=np.int32)

  @property
  def centroid(self):
    """the centroid of the area, holes taken off, as a framed point"""
    S = loopSums(self.V, self.loops, self.size)
    a, c = S[:, 0], S[:, 3:5]
    with np.errstate(divide="ignore", invalid="ignore"):
      out = np.where(a[:, None] != 0, c / (3 * a[:, None]), 0.0)
    return vector3d(self.plane.from2d(out), frame=self.frame)

  @property
  def equivalentRadius(self):
    return np.sqrt(np.maximum(self.area, 0) / np.pi)

  @property
  def equivalentPerimeter(self):
    return 2 * np.pi * self.equivalentRadius

  @property
  def diameter(self):
    """the largest distance between two vertices of the outer loop"""
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

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from .shape import (principalComponents, fitEllipse, longAxis, shortAxis, aspectRatio, hullLoops, paris,
                        shapeFactor, convexity, caliper, paror, surfor)
