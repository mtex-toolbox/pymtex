# planes in space (MATLAB @plane3d; Julia plane.jl)
#
#   plane = plane3d(N, d)          the plane <N, v> = d, N normalised and d scaled with it
#   plane = plane3d(N, v0)         the plane with normal N through the point v0
#   plane = plane3d.byPoints(v)    the least squares plane through points, plane3d.fit(v)
#   plane.N, plane.d, dist(plane, v), dist(plane, plane2), plane.project(v),
#   intersect(plane, v1, v2), plane.clip(lo, hi), rot * plane, plane[k]
#
# A plane in Hesse form: the unit normal N and the signed distance d from the origin. A
# list of planes is one object over a list of normals and distances; the normal carries
# the frame, so a plane in the specimen and a lattice plane are told apart by it.
from __future__ import annotations

import itertools
from typing import TYPE_CHECKING
import numpy as np

from ._array import MtexArray
from .symreduce import _firstOfClusters
from .vector3d import vector3d, asVector3d


class plane3d:
  """a plane <N, v> = d, or a list of them

    plane = plane3d(N, d)           # <N, v> = d
    plane = plane3d(N, v0)          # through the point v0
    plane = plane3d.byPoints(v)     # fitted through the points v
  """

  __array_ufunc__ = None

  def __init__(self, N, d=0):
    if isinstance(N, plane3d):
      self.N, self.d = N.N, N.d
      return
    N = asVector3d(N)
    n = N.norm()
    self.N = N.normalize()
    if isinstance(d, vector3d):
      self.d = np.asarray(self.N.dot(d, symmetry=False), dtype=float)
    else:
      self.d = np.asarray(np.broadcast_to(np.asarray(d, dtype=float) / n, N.shape), dtype=float).copy()

  @staticmethod
  def byPoints(v):
    """the plane through points: the least squares fit through their centroid, the normal
    the smallest principal axis of the cloud"""
    v = asVector3d(v)
    xyz = v.data.reshape(-1, 3)
    xyz = xyz[np.all(np.isfinite(xyz), axis=1)]
    if len(xyz) < 3:
      raise ValueError("a plane needs at least three points")
    c = xyz.mean(axis=0)
    M = (xyz - c).T @ (xyz - c)
    n = np.linalg.eigh(M)[1][:, 0]
    return plane3d(vector3d(n, frame=v.frame), vector3d(c, frame=v.frame))

  fit = byPoints

  def _new(self, N, d):
    p = object.__new__(plane3d)
    p.N, p.d = N, d
    return p

  # ------------------------------------------------------------------ the list

  @property
  def shape(self):
    """the shape of the list of planes"""
    return self.N.shape

  @property
  def size(self):
    """the number of planes"""
    return self.N.size

  def __len__(self):
    return self.N.size

  def __getitem__(self, key):
    N, d = (self.N.reshape(-1), np.asarray(self.d).reshape(-1)) if self.N.ndim == 0 else (self.N, np.asarray(self.d))
    return self._new(N[key], d[key])

  def __iter__(self):
    return (self[k] for k in range(len(self)))

  @property
  def frame(self):
    """the frame of the planes, that of their normals"""
    return self.N.frame

  # ------------------------------------------------------------------ distances

  def dist(self, v, tol=1e-8):
    """the signed distance of points to the plane, or the distance of two parallel planes,
    NaN where the planes meet"""
    if isinstance(v, plane3d):
      lam = self.N.dot(v.N, symmetry=False)
      lam = np.where(np.abs(np.abs(lam) - 1) < tol, lam, np.nan)
      return self.d - v.d * lam
    v = asVector3d(v)
    return self.N.dot(v, symmetry=False) - self.d

  def project(self, v):
    """the orthogonal projection of points onto the plane"""
    v = asVector3d(v)
    return v - self.N * self.dist(v)

  def intersect(self, v1, v2=None, tol=1e-13):
    """where the segments from v1 to v2 cross the plane, NaN where they do not"""
    if v2 is None:
      v1, v2 = v1[..., 0], v1[..., 1]
    return lineIntersect(v1, v2, self.N, self.d, tol)

  def clip(self, lo, hi):
    """the polygon the plane cuts out of the box from lo to hi, its corners in order around
    the normal; one plane only"""
    if self.size != 1:
      raise ValueError("clip one plane, plane[k]")
    lo, hi = np.asarray(lo, dtype=float).reshape(3), np.asarray(hi, dtype=float).reshape(3)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    n, d = self.N.data.reshape(3), float(np.asarray(self.d).reshape(()))
    # the crossings of the twelve edges, each corner of the polygon once
    edges = [(i, j) for i, j in itertools.combinations(range(8), 2) if np.sum(corners[i] != corners[j]) == 1]
    pts = np.array([lineIntersect(corners[i], corners[j], n, d).data.reshape(3) for i, j in edges])
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    pts = pts[_firstOfClusters(np.linalg.norm(pts[:, None] - pts[None], axis=-1) < 1e-10)]
    if len(pts) >= 3:
      c = pts.mean(axis=0)
      u = self.N.orth().data.reshape(3)
      w = np.cross(n, u)
      pts = pts[np.argsort(np.arctan2((pts - c) @ w, (pts - c) @ u))]
    return vector3d(pts, frame=self.N.frame)

  def rotate(self, rot):
    """a rotation turns the normal, the distance stays"""
    return self._new(rot * self.N, self.d)

  def __rmul__(self, rot):
    if isinstance(rot, MtexArray):
      return self.rotate(rot)
    return NotImplemented

  # ------------------------------------------------------------------ display

  def __repr__(self):
    from .display import display, frameLabel, header, sizeString, table, takeRows
    take, pos = takeRows(self.size)
    N = self.N.data.reshape(-1, 3)[take]
    d = np.asarray(self.d, dtype=float).reshape(-1)[take]
    body = table(["x", "y", "z", "d"], [N[:, 0], N[:, 1], N[:, 2], d], digits=4, rows=pos) if self.size else None
    return display(header("plane3d", frameLabel(self.N.frame)), [("size", sizeString(self.shape) if self.size != 1 else None)], body)

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from ..plotting.shapeplot import plotPlane as plot


def lineIntersect(a, b, n, d, tol=1e-13):
  """where the segments from a to b cross the planes <n, v> = d, NaN where they do not"""
  A = asVector3d(a)
  B = asVector3d(b)
  N = asVector3d(n)
  l = B - A
  with np.errstate(divide="ignore", invalid="ignore"):
    lam = (np.asarray(d, dtype=float) - A.dot(N, symmetry=False)) / l.dot(N, symmetry=False)
  lam = np.where((lam < -tol) | (lam > 1 + tol), np.nan, lam)
  return A + l * lam


def dist(plane, v, **kw):
  """dist(plane, v), dist(plane, plane2): the signed distance to the plane"""
  return plane.dist(v, **kw)


def intersect(plane, v1, v2=None):
  """intersect(plane, v1, v2): where the segments cross the plane"""
  return plane.intersect(v1, v2)


__all__ = ["plane3d", "dist", "intersect", "lineIntersect"]
