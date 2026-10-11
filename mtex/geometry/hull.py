# the convex hull of points in the plane (Julia hull.jl and shape.jl)
#
#   convexHull(points)         the indices of the hull of (n, 2) points, anticlockwise
#   convexHullLoops(V, loops)  the hulls of many loops of indices into V, in one pass
#
# Andrew's monotone chain: the points sorted by x, then y, the lower chain built left to
# right and the upper one right to left, a point popped whenever it would make a right turn
# or none, so collinear points and repeated points never become corners. No dependency and
# no degenerate case to handle; the chain is a numba loop with the Python loop as fallback
# and oracle, since a grain list asks for one hull per grain.
#
#   _cellAreas(u)              the areas of the spherical Voronoi cells of unit vectors
#
# The hull of points on the sphere is their Delaunay triangulation (Qhull through SciPy), and
# a cell is cut by the triangles at its point into kites: the point, the midpoints of the two
# edges and the circumcentre. Their signed areas add up to the cell, so no cell is assembled.
from __future__ import annotations

import numpy as np

from .kernels import _jit, _parallelKernel, backend

prange = range
_COMPILED = {}


def convexHull(points):
  """the indices of the convex hull of points in the plane, anticlockwise, by Andrew's monotone
  chain; collinear and repeated points never become corners"""
  P = np.ascontiguousarray(np.asarray(points, dtype=float).reshape(-1, 2))
  n = len(P)
  if n <= 2:
    return np.arange(n)
  order = np.lexsort((P[:, 1], P[:, 0])).astype(np.int64)
  out = np.empty(2 * n, dtype=np.int64)
  m = _compiledChain()(P, order, out)
  return out[:m]


# the two chains over the sorted points, written into out; returns how many corners
def _chainLoop(P, order, out):
  n = order.shape[0]
  m = 0
  for pas in range(2):
    start = m
    for t in range(n):
      i = order[t] if pas == 0 else order[n - 1 - t]
      while m - start >= 2:
        a, b = out[m - 2], out[m - 1]
        if (P[b, 0] - P[a, 0]) * (P[i, 1] - P[a, 1]) - (P[b, 1] - P[a, 1]) * (P[i, 0] - P[a, 0]) > 0:
          break
        m -= 1
      out[m] = i
      m += 1
    # the last point of a chain is the first of the other
    m -= 1
  return m


# the hulls of the loops flat[offsets[c]:offsets[c + 1]] of indices into V, each written into
# its own slot out[2 offsets[c]:2 offsets[c + 1]], as many as a hull of the loop can have, with
# its length in counts, so that the loops run in parallel: every loop sorted by x then y
# through two stable sorts, then the two chains
def _hullLoopsLoop(V, flat, offsets, out, counts):
  for c in prange(offsets.shape[0] - 1):
    s, e = offsets[c], offsets[c + 1]
    n = e - s
    at = 2 * s
    if n <= 2:
      for i in range(n):
        out[at + i] = flat[s + i]
      counts[c] = n
      continue
    idx = flat[s:e]
    ys = np.empty(n)
    for i in range(n):
      ys[i] = V[idx[i], 1]
    byY = idx[np.argsort(ys, kind="mergesort")]
    xs = np.empty(n)
    for i in range(n):
      xs[i] = V[byY[i], 0]
    order = byY[np.argsort(xs, kind="mergesort")]
    counts[c] = _chainLoop(V, order, out[at:at + 2 * n])


def convexHullLoops(V, loops):
  """the convex hulls of many loops of indices into the (n, 2) vertices V in one pass, a list of
  index arrays into V"""
  V = np.ascontiguousarray(np.asarray(V, dtype=float).reshape(-1, 2))
  lengths = np.array([len(l) for l in loops], dtype=np.int64)
  offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
  flat = np.concatenate([np.asarray(l, dtype=np.int64) for l in loops]) if len(loops) else np.zeros(0, dtype=np.int64)
  loop = _compiledLoops()
  if loop is None:
    return [np.asarray(l)[convexHull(V[np.asarray(l)])] if len(l) else np.zeros(0, dtype=np.int64) for l in loops]
  out = np.empty(2 * len(flat) + 1, dtype=np.int64)
  counts = np.empty(len(loops), dtype=np.int64)
  loop(V, flat, offsets, out, counts)
  return [out[2 * offsets[c]:2 * offsets[c] + counts[c]] for c in range(len(loops))]


def _compiledChain():
  if backend() == "numpy":
    return _chainLoop
  if "chain" not in _COMPILED:
    _COMPILED["chain"] = _jit().njit(cache=True)(_chainLoop)
  return _COMPILED["chain"]


def _compiledLoops():
  if backend() == "numpy":
    return None
  if "loops" not in _COMPILED:
    nb = _jit()
    globals()["_chainLoop"] = _compiledChain()
    globals()["prange"] = nb.prange
    _COMPILED["loops"] = _parallelKernel(_hullLoopsLoop)
  return _COMPILED["loops"]


# the signed area of the spherical triangles a, b, c, by the formula of van Oosterom and Strackee
def _triangleArea(a, b, c):
  def dot(x, y):
    return np.einsum("ij,ij->i", x, y)
  return 2 * np.arctan2(dot(a, np.cross(b, c)), 1 + dot(a, b) + dot(b, c) + dot(c, a))


# the areas of the spherical Voronoi cells of distinct unit vectors (n, 3), which sum to the
# sphere; SciPy's SphericalVoronoi(u).calculate_areas() is the oracle. Qhull raises on fewer
# than four points and on points in a plane
def _cellAreas(u):
  from scipy.spatial import ConvexHull
  u = np.asarray(u, dtype=float).reshape(-1, 3)
  T = ConvexHull(u).simplices
  # every triangle anticlockwise seen from outside, its circumcentre the outer normal
  a, b, c = u[T[:, 0]], u[T[:, 1]], u[T[:, 2]]
  o = np.cross(b - a, c - a)
  flip = np.einsum("ij,ij->i", o, a) < 0
  T = np.where(flip[:, None], T[:, [0, 2, 1]], T)
  o[flip] *= -1
  o /= np.sqrt(np.einsum("ij,ij->i", o, o))[:, None]
  P = [u[T[:, i]] for i in range(3)]
  # the midpoints of the edges opposite to each corner
  mid = []
  for i in range(3):
    m = P[(i + 1) % 3] + P[(i + 2) % 3]
    mid.append(m / np.sqrt(np.einsum("ij,ij->i", m, m))[:, None])
  area = np.zeros(len(u))
  for i in range(3):
    kite = _triangleArea(P[i], mid[(i + 2) % 3], o) + _triangleArea(P[i], o, mid[(i + 1) % 3])
    area += np.bincount(T[:, i], kite, len(u))
  return area


__all__ = ["convexHull", "convexHullLoops"]
