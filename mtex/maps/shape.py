# the shape of an outline (MATLAB @grain2d/fitEllipse, principalComponents, longAxis,
# shortAxis, aspectRatio, paris, shapeFactor, caliper, paror, hull, @grainBoundary/surfor,
# characteristicShape, @shape2d; Julia shape.jl)
#
#   grains.fitEllipse(), grains.principalComponents(), grains.longAxis(), grains.shortAxis()
#   grains.aspectRatio(), grains.paris(), grains.shapeFactor(), grains.convexity(), grains.hull()
#   grains.caliper(), grains.caliper('shortest' | 'shortestPerp' | 'longest'), grains.caliper(v), grains.caliper(omega)
#   grains.paror(omega), grains.surfor(omega), gB.surfor(omega), gB.characteristicShape()
#   shape2d(V), shape2d.byRhoTheta(rho, theta), shape2d.byFV(F, V)
#
# Everything here is a property of the polygons, so it is computed from the loops and
# nothing is stored, and it is written against what a polygonSet answers: the grains are
# one and a shape2d, one closed outline in the plane of a map, another. The ellipse is the
# one with the same second area moments as the outline, which a polygon has in closed form;
# MTEX fits it to the boundary vertices weighted by the adjacent edge lengths times the
# square root of their distance from the centroid, a stand-in for the same thing. Because
# the loops carry their winding, every sum subtracts an inclusion by itself.
from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

from ..geometry.frames import specimenFrame
from ..geometry.hull import convexHullLoops
from ..geometry.vector3d import vector3d
from .boundary import OUTSIDE, grainBoundary, orderWalk
from .decomposition import NO_PIXEL
from .grains import grain2d
from .kernels import grainKernel, pixelKernel
from .lattice import mapPlane
from .polygon import loopSums, polygonSet, signedArea2, _cyclicNext, _withinRun
from ..geometry.install import install
from ..geometry.display import aligned, frameLabel


# ------------------------------------------------------------------ the moments

def polygonMoments(V, loops, size=None):
  """the moments of the polygon of every outline: the area, the first and the second moments
  about the origin, summed over its loops, as six arrays; or of one list of loops"""
  one = size is None
  S = loopSums(V, [loops] if one else loops, 1 if one else size)
  out = (S[:, 0] / 2, S[:, 3] / 6, S[:, 4] / 6, S[:, 5] / 12, S[:, 6] / 12, S[:, 7] / 24)
  return tuple(float(x[0]) for x in out) if one else out


# the half axes of the ellipse with the same second area moments as every outline, the
# longer one first, as vectors in the plane of the map
def _principal2d(self):
  ar, cx, cy, ixx, iyy, ixy = polygonMoments(self.V, self.loops, self.size)
  ok = ar > 0
  A, B = np.zeros((self.size, 2)), np.zeros((self.size, 2))
  a = ar[ok]
  x, y = cx[ok] / a, cy[ok] / a
  # the second moments about the centroid, divided by the area
  m20, m02, m11 = iyy[ok] / a - x * x, ixx[ok] / a - y * y, ixy[ok] / a - x * y
  # the eigenvalues of a symmetric two by two matrix in closed form
  tr = m20 + m02
  d = np.sqrt(np.maximum(0.0, (m20 - m02) ** 2 / 4 + m11 * m11))
  l1, l2 = tr / 2 + d, tr / 2 - d
  v1 = np.where((m11 == 0)[:, None], np.where((m20 >= m02)[:, None], [[1.0, 0.0]], [[0.0, 1.0]]), np.stack([l1 - m02, m11], axis=1))
  v1 = v1 / np.linalg.norm(v1, axis=1)[:, None]
  # an ellipse of half axes p, q has m20 = p^2 / 4
  A[ok] = 2 * np.sqrt(np.maximum(0.0, l1))[:, None] * v1
  B[ok] = 2 * np.sqrt(np.maximum(0.0, l2))[:, None] * np.stack([-v1[:, 1], v1[:, 0]], axis=1)
  return A, B


# vectors of the plane as framed axes of the specimen: an axis, since an outline has no
# sense along its long direction
def _axisOf(self, W):
  W = np.asarray(W, dtype=float).reshape(-1, 2)
  return vector3d(self.plane.from2d(W) - self.plane.o, frame=self.frame, antipodal=True)


def principalComponents(self):
  """the two half axes of the ellipse with the same second area moments as every outline, the long
  one first, as antipodal vectors of the specimen frame

    a, b = grains.principalComponents()"""
  A, B = _principal2d(self)
  return _axisOf(self, A), _axisOf(self, B)


def fitEllipse(self):
  """the ellipse of the same second moments: its centre and its two half axes"""
  return (self.centroid,) + principalComponents(self)


def longAxis(self):
  """the long half axis of the ellipse of every outline, the larger principal component

    lA = grains.longAxis()
    lA = longAxis(grains)"""
  return principalComponents(self)[0]


def shortAxis(self):
  """the short half axis of the ellipse of every outline, the smaller principal component"""
  return principalComponents(self)[1]


def aspectRatio(self):
  """the ratio of the two half axes, one for a circle

    asp = grains.aspectRatio()
    asp = aspectRatio(grains)"""
  A, B = _principal2d(self)
  a, b = np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1)
  with np.errstate(divide="ignore", invalid="ignore"):
    return np.where(b == 0, np.inf, a / b)


# ------------------------------------------------------------------ the convex hull

def hullLoops(self):
  """the hull of the outer loop of every outline, as loops of vertex indices, anticlockwise"""
  return convexHullLoops(self.V, [ls[0] if ls else np.zeros(0, dtype=np.int64) for ls in self.loops])


# the hull vertices of every outline as one flat list with the offsets of the outlines
def _flatHull(self):
  hl = hullLoops(self)
  lengths = np.array([len(h) for h in hl], dtype=np.int64)
  flat = np.concatenate(hl) if len(hl) else np.zeros(0, dtype=np.int64)
  return self.V[flat], np.concatenate([[0], np.cumsum(lengths)]), lengths


def hullPerimeter(self):
  """the perimeter of the convex hull of every outline"""
  return loopSums(self.V, [[h] for h in hullLoops(self)], self.size)[:, 2]


def hullArea(self):
  """the area of the convex hull of every outline"""
  return np.abs(loopSums(self.V, [[h] for h in hullLoops(self)], self.size)[:, 0]) / 2


# the largest minus the smallest of a value per outline over its hull vertices, zero for an
# outline without a hull
def _spread(values, offsets, lengths):
  out = np.zeros(len(lengths))
  ok = lengths > 0
  if ok.any():
    at = offsets[:-1][ok]
    out[ok] = np.maximum.reduceat(values, at) - np.minimum.reduceat(values, at)
  return out


# ------------------------------------------------------------------ the parameters

def paris(self):
  """how much the outline is indented against its convex hull, in percent"""
  p, hp = self.perimeter, hullPerimeter(self)
  with np.errstate(divide="ignore", invalid="ignore"):
    return np.where(hp == 0, 0.0, 200 * (p - hp) / hp)


def shapeFactor(self):
  """the perimeter against that of the circle of the same area, one for a circle"""
  return self.perimeter / self.equivalentPerimeter


def convexity(self):
  """the area against that of the convex hull, one for a convex outline"""
  a, ha = self.area, hullArea(self)
  with np.errstate(divide="ignore", invalid="ignore"):
    return np.where(ha == 0, 1.0, a / ha)


# the width of every outline measured across each direction of the plane: the distance between
# its two lines of support perpendicular to it, a column per direction
def _widthsAcross(self, directions):
  H, offsets, lengths = _flatHull(self)
  return np.stack([_spread(H @ (d / np.linalg.norm(d)), offsets, lengths) for d in np.asarray(directions, dtype=float).reshape(-1, 2)], axis=-1)


# the extremal widths as vectors in the plane: the width is extremal along a normal of one
# of the hull's edges, so the smallest is found over the edges, and the largest is the
# largest distance between two hull vertices
def _extremalWidths(self):
  H, offsets, lengths = _flatHull(self)
  shortest, longest = np.zeros((self.size, 2)), np.zeros((self.size, 2))
  T = len(H)
  if T == 0:
    return shortest, longest
  # one compiled loop over the outlines, the pairs below its NumPy form
  loop = pixelKernel("extremalWidths")
  if loop is not None:
    loop(np.ascontiguousarray(H[:, :2], dtype=float), np.ascontiguousarray(offsets, dtype=np.int64), shortest, longest)
    return shortest, longest
  starts = offsets[:-1]
  lab = np.repeat(np.arange(self.size), lengths)
  # the edges of every hull, cyclic within the outline, and their unit normals
  E = H[_cyclicNext(lengths)] - H
  eLen = np.linalg.norm(E, axis=1)
  nrm = np.stack([-E[:, 1], E[:, 0]], axis=1) / np.where(eLen > 0, eLen, 1)[:, None]
  # every (edge, point) pair of an outline, the pairs grouped by edge
  rep = np.repeat(lengths, lengths)
  ei = np.repeat(np.arange(T), rep)
  segStart = np.cumsum(rep) - rep
  pj = np.repeat(starts[lab], rep) + _withinRun(rep)
  proj = np.sum(H[pj] * nrm[ei], axis=1)
  width = np.maximum.reduceat(proj, segStart) - np.minimum.reduceat(proj, segStart)
  width[eLen == 0] = np.inf
  # the narrowest support of every outline is the smallest width over its edges
  order = np.lexsort((width, lab))
  k, first = np.unique(lab[order], return_index=True)
  e = order[first]
  ok = (lengths[k] >= 2) & np.isfinite(width[e])
  shortest[k[ok]] = width[e[ok], None] * nrm[e[ok]]
  # the widest is the largest distance between two hull vertices
  d = np.linalg.norm(H[pj] - H[ei], axis=1)
  order = np.lexsort((-d, lab[ei]))
  k, first = np.unique(lab[ei][order], return_index=True)
  pair = order[first]
  longest[k] = H[pj[pair]] - H[ei[pair]]
  return shortest, longest


# the direction of the plane a framed vector or an angle to the x axis of the map names
def _direction2d(self, arg):
  if isinstance(arg, vector3d):
    v = arg.data.reshape(-1)
    return np.array([v @ self.plane.e1, v @ self.plane.e2])
  return np.array([np.cos(float(arg)), np.sin(float(arg))])


def caliper(self, arg=None):
  """the caliper: with nothing, the longest width of every outline as a framed axis of that
  length; with 'shortest', the narrowest one, with 'shortestPerp' the width measured across
  the direction perpendicular to it, which is the direction an outline with flat faces is
  aligned along, with 'longest' the first again; with a direction or an angle to the x axis
  of the map, the width across it; a list of them gives a width per outline and direction

    cV = grains.caliper()                  # the longest, norm(cV) its length
    cV = grains.caliper('shortest')        # or 'longest', 'shortestPerp'
    c = grains.caliper(vector3d.X)         # the width across a direction
    c = caliper(grains, [0, 30*degree])    # the widths across angles to the x axis"""
  if arg is None or (isinstance(arg, str) and arg == "longest"):
    return _axisOf(self, _extremalWidths(self)[1])
  if isinstance(arg, str):
    s = _extremalWidths(self)[0]
    if arg == "shortest":
      return _axisOf(self, s)
    if arg == "shortestPerp":
      n = np.linalg.norm(s, axis=1)
      with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(n[:, None] > 0, np.stack([-s[:, 1], s[:, 0]], axis=1) / n[:, None], 0.0)
      H, offsets, lengths = _flatHull(self)
      w = _spread(np.sum(H * np.repeat(u, lengths, axis=0), axis=1), offsets, lengths)
      return _axisOf(self, w[:, None] * u)
    raise ValueError(f"the caliper is 'shortest', 'longest' or 'shortestPerp', not {arg!r}")
  # a list of directions or angles gives one width per outline and direction
  if isinstance(arg, vector3d):
    many, args = arg.size > 1, [arg.flatten()[k] for k in range(arg.size)]
  else:
    many, args = np.ndim(arg) > 0, np.asarray(arg, dtype=float).reshape(-1)
  W = _widthsAcross(self, [_direction2d(self, a) for a in args])
  return W if many else W[:, 0]


# ------------------------------------------------------------------ the projection functions

def paror(self, omega=None):
  """the cumulative projection length of the outlines, measured along each direction (Panozzo
  1983): the width of the convex hull seen from that direction, summed over the outlines
  and normalised by its largest value. The hulls are taken once for all the directions"""
  omega = np.linspace(0, np.pi, 181) if omega is None else np.asarray(omega, dtype=float).reshape(-1)
  D = np.stack([np.cos(omega), np.sin(omega)], axis=1)
  H, offsets, lengths = _flatHull(self)
  # one compiled pass per direction, the matrix of the projections below its NumPy form
  loop = grainKernel("projectionLength")
  if loop is not None:
    out = np.empty(len(omega))
    loop(np.ascontiguousarray(H[:, :2], dtype=float), offsets.astype(np.int64), D, out)
    return out / out.max()
  # the projections a row per direction, so that every outline's run of them is contiguous
  pr = D @ H[:, :2].T
  at = offsets[:-1][lengths > 0]
  out = np.zeros(len(omega))
  if len(at):
    out = np.sum(np.maximum.reduceat(pr, at, axis=1) - np.minimum.reduceat(pr, at, axis=1), axis=1)
  return out / out.max()


SURFOR_BLOCK = 4096


def surfor(x, omega=None):
  """the cumulative projection length of the boundary segments (Panozzo 1984): a projection
  is a dot product, the segment vector projected onto the direction, so no angle is formed
  and none has to be folded into [0, pi)"""
  gB = x.boundary if isinstance(x, grain2d) else x
  omega = np.linspace(0, np.pi, 181) if omega is None else np.asarray(omega, dtype=float).reshape(-1)
  D = np.stack([np.cos(omega), np.sin(omega)], axis=1)
  seg = gB.V[gB.F[:, 1]] - gB.V[gB.F[:, 0]]
  # the projections by blocks of segments, each a matrix product that stays in the cache
  out = np.zeros(len(D))
  for k in range(0, len(seg), SURFOR_BLOCK):
    out += np.abs(seg[k:k + SURFOR_BLOCK, :2] @ D.T).sum(axis=0)
  return out / out.max()


def characteristicShape(x, nmax=1024):
  """the characteristic shape of a set of boundary segments: every segment and its antipode,
  sorted by direction and laid end to end, the convex, centrally symmetric outline whose
  boundary length in each direction is the boundary length the map has in it. A convex
  outline of a hundred thousand segments is sampled down to nmax, which keeps its extent"""
  gB = x.boundary if isinstance(x, grain2d) else x
  if gB.isempty:
    raise ValueError("an empty boundary has no characteristic shape")
  return shape2d.byFV(gB.F, gB.V, nmax=nmax, plane=gB.plane, frame=gB.frame)


# ------------------------------------------------------------------ an outline of its own

# the plane of a map that is none: the xy plane of the specimen
def _xyPlane():
  return mapPlane(np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]))


class shape2d(polygonSet):
  """one closed outline in the plane of a map, so that every parameter above is there for a
  curve that came from no segmentation, as the characteristic shape of a boundary and a
  projection function drawn as a shape are. It is stored anticlockwise, so its area is the
  positive one"""

  def __init__(self, V, plane=None, frame=None):
    if isinstance(V, vector3d):
      frame = V.frame if frame is None else frame
      V = V.data
    V = np.asarray(V, dtype=float)
    V = V.reshape(-1, V.shape[-1])
    self.plane = _xyPlane() if plane is None else plane
    self.frame = specimenFrame.default if frame is None else frame
    if V.shape[1] == 3:
      V = self.plane.to2d(V)
    if signedArea2(V) < 0:
      V = V[::-1]
    self.V = np.ascontiguousarray(V)
    self.loops = [[np.arange(len(V), dtype=np.int32)]]

  @staticmethod
  def byRhoTheta(rho, theta, **kw):
    """an outline in polar coordinates, which is how a projection function becomes the shape
    it describes"""
    rho, theta = np.asarray(rho, dtype=float).reshape(-1), np.asarray(theta, dtype=float).reshape(-1)
    return shape2d(np.stack([rho * np.cos(theta), rho * np.sin(theta)], axis=1), **kw)

  @staticmethod
  def byFV(F, V, nmax=1024, **kw):
    """the outline of a set of segments laid end to end by direction, the characteristic shape"""
    F = np.asarray(F).reshape(-1, 2)
    V = np.asarray(V, dtype=float)
    if V.shape[1] == 3:
      pl = kw.get("plane") or _xyPlane()
      V = pl.to2d(V)
    seg = V[F[:, 1]] - V[F[:, 0]]
    seg = np.concatenate([seg, -seg])
    seg = seg[np.argsort(np.arctan2(seg[:, 1], seg[:, 0]), kind="stable")]
    P = np.cumsum(seg, axis=0)
    P -= P.mean(axis=0)
    if len(P) > nmax:
      P = P[np.round(np.linspace(0, len(P) - 1, nmax)).astype(int)]
    return shape2d(P, **kw)

  @property
  def size(self):
    """the number of outlines, one"""
    return 1

  def __len__(self):
    return 1

  @property
  def Vs(self):
    """the outline as framed points, and in polar coordinates"""
    return vector3d(self.plane.from2d(self.V), frame=self.frame)

  @property
  def allV(self):
    """the vertices of the outline as framed points, the same as Vs"""
    return self.Vs

  @property
  def rho(self):
    """the radius of every vertex in polar coordinates of the plane"""
    return np.linalg.norm(self.V, axis=1)

  @property
  def theta(self):
    """the angle of every vertex in polar coordinates of the plane, from the x axis"""
    return np.arctan2(self.V[:, 1], self.V[:, 0])

  def __getitem__(self, key):
    if len(np.arange(1)[key].reshape(-1)) != 1:
      raise IndexError("a shape2d is one outline")
    return self

  def hull(self):
    """the convex hull of the outline, as an outline"""
    return shape2d(self.V[hullLoops(self)[0]], plane=self.plane, frame=self.frame)

  def __repr__(self):
    return aligned(f"shape2d ({frameLabel(self.frame)})\n  vertices: {len(self.V)}\n  area: {self.area[0]:.4g}\n  aspect ratio: {aspectRatio(self)[0]:.4g}")

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from ..plotting.polarplot import plotShape as plot


def _hullGrains(self):
  """the grains with their convex hulls for polygons: the hull edges form the boundary, with
  nothing on the other side, and there is no inner boundary"""
  hl = hullLoops(self)
  lengths = np.array([len(h) for h in hl], dtype=np.int64)
  keep = lengths >= 2
  # every hull walked clockwise, so that the outside lies on the left as it does on the rim
  # of a map; the head of each edge is the next vertex of its loop
  flat = np.concatenate([h[::-1] for h, ok in zip(hl, keep) if ok]) if keep.any() else np.zeros(0, dtype=np.int64)
  lens = lengths[keep]
  nxt = flat[_cyclicNext(lens)]
  n = len(flat)
  gid = np.stack([np.zeros(n, dtype=np.int32), np.repeat(self.id[keep], lens)], axis=1)
  pid = np.stack([np.full(n, OUTSIDE, dtype=np.int8), np.repeat(self.phaseId[keep], lens)], axis=1)
  gB = grainBoundary(self.V, np.stack([flat, nxt], axis=1), gid, np.full((n, 2), NO_PIXEL, dtype=np.int32), pid, np.full((n, 4), np.nan),
                     self.CSList, self.plane, self.frame)
  gB = orderWalk(gB)
  return grain2d(self.id, self.phaseId, self.CSList, self.numPixel, self.meanRotation, self.GOS, self.frame, self.prop, boundary=gB,
                 innerBoundary=gB[np.zeros(0, dtype=int)], phaseMap=self.phaseMap)


for _name, _f in (("principalComponents", principalComponents), ("fitEllipse", fitEllipse), ("longAxis", longAxis), ("shortAxis", shortAxis),
                  ("aspectRatio", aspectRatio), ("hullLoops", hullLoops), ("paris", paris), ("shapeFactor", shapeFactor), ("convexity", convexity),
                  ("caliper", caliper), ("paror", paror), ("surfor", surfor)):
  install(polygonSet, _f, _name)
install(grain2d, _hullGrains, "hull")
install((grain2d, grainBoundary), characteristicShape)
install(grainBoundary, surfor)

__all__ = ["shape2d", "polygonMoments"]
