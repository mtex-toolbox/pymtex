# grids on the sphere with ring structure (MATLAB @S2Grid, equispacedS2Grid, regularS2Grid):
# a ladder of polar angles and one azimuth grid per ring. Fast search comes from that
# structure: nearest is three binary searches, an eps-cap is a theta band and per ring an
# exact azimuth window from the spherical law of cosines. The points are a vector3d, so
# everything that needs only points takes the grid as is; whatever breaks the rings -
# rotation, indexing, arithmetic - returns the plain vector3d.
from __future__ import annotations

import numpy as np

from ..frames import referenceFrame
from ..units import degree
from ..vector3d import vector3d, asVector3d
from .s1grid import S1Grid, ladder
from .gridsearch import flatS2, s2Nearest
from ..display import display, header


class S2Grid(vector3d):
  """a grid on the sphere with ring structure, a ladder of polar angles theta and one azimuth grid
  per ring, whose nearest and radius searches use that structure; built by equispacedS2Grid,
  regularS2Grid and plotS2Grid, and whatever breaks the rings returns a plain vector3d"""

  # the grid from its rings: theta a ladder of polar angles, rho one azimuth grid per ring
  def __init__(self, theta, rho, frame=None, resolution=np.nan, antipodal=False):
    if len(rho) != len(theta):
      raise ValueError("one azimuth grid per ring is needed")
    self.thetaGrid, self.rhoGrid = theta, list(rho)
    self.offset = np.concatenate([[0], np.cumsum([len(r) for r in rho])])
    th = np.concatenate([np.full(len(r), t) for t, r in zip(theta.points, rho)]) if len(rho) else np.zeros(0)
    rh = np.concatenate([r.points for r in rho]) if len(rho) else np.zeros(0)
    xyz = np.stack([np.sin(th) * np.cos(rh), np.sin(th) * np.sin(rh), np.cos(th)], axis=-1)
    vector3d.__init__(self, xyz, antipodal=antipodal, isNormalized=True, frame=frame)
    self.res = float(resolution)

  # anything derived from the grid is a plain vector3d
  def _new(self, data, **attrs):
    obj = object.__new__(vector3d)
    obj.data = data
    for name in self._carried:
      setattr(obj, name, attrs.pop(name, getattr(self, name)))
    if attrs:
      raise TypeError(f"vector3d carries no attribute {', '.join(attrs)}")
    return obj

  @property
  def resolution(self):
    """the resolution the grid was built with, NaN when none was given"""
    return self.res

  @property
  def numRings(self):
    """the number of rings, polar angles, of the grid"""
    return len(self.thetaGrid)

  def ring(self, k):
    """the indices of the points on ring k"""
    return np.arange(self.offset[k], self.offset[k + 1])

  # the flat ladders for the compiled search, or None when a ring is not an equispaced circle
  def _flat(self):
    if not hasattr(self, "_flatCache"):
      self._flatCache = flatS2(self)
    return self._flatCache

  def polar(self):
    """the polar angles of the grid points, exact from the rings"""
    th = np.concatenate([np.full(len(r), t) for t, r in zip(self.thetaGrid.points, self.rhoGrid)])
    rh = np.concatenate([r.points for r in self.rhoGrid])
    return th, rh, np.ones(self.size)

  def subGrid(self, mask, eps=None):
    """the grid points selected by a mask, or within eps of a centre; rings that empty out go"""
    if eps is not None:
      center, mask = mask, np.zeros(self.size, dtype=bool)
      mask[self.neighbourIndices(center, eps)] = True
    mask = np.asarray(mask)
    if mask.dtype != bool:
      m = np.zeros(self.size, dtype=bool)
      m[mask] = True
      mask = m
    keep = [bool(np.any(mask[self.ring(k)])) for k in range(self.numRings)]
    rho = [self.rhoGrid[k].subGrid(mask[self.ring(k)]) for k in range(self.numRings) if keep[k]]
    g = S2Grid(self.thetaGrid.subGrid(keep), rho, self.frame, self.res, self.antipodal)
    g.dispStyle = self.dispStyle
    return g

  # ------------------------------------------------------------------ search

  def _ringDot(self, k, j, theta, rho):
    tk = self.thetaGrid[k]
    return np.cos(theta) * np.cos(tk) + np.sin(theta) * np.sin(tk) * np.cos(self.rhoGrid[k].dist(self.rhoGrid[k][j], rho))

  # the nearest grid point of one direction: the closest point of the nearest ring, then
  # every ring within that distance can only improve it
  def _nearestOne(self, theta, rho):
    k = int(self.thetaGrid.nearest(theta))
    j = int(self.rhoGrid[k].nearest(rho))
    best, bd = self.offset[k] + j, self._ringDot(k, j, theta, rho)
    omega = np.arccos(np.clip(bd, -1, 1))
    for k2 in self.thetaGrid.neighbours(theta, omega):
      if k2 == k:
        continue
      j2 = int(self.rhoGrid[k2].nearest(rho))
      d2 = self._ringDot(k2, j2, theta, rho)
      if d2 > bd:
        best, bd = self.offset[k2] + j2, d2
    return int(best), bd

  def _nearestIndex(self, xyz):
    v = vector3d(xyz).normalize()
    i, di = self._nearestOne(v.theta, v.rho)
    if not self.antipodal:
      return i
    w = -v
    j, dj = self._nearestOne(w.theta, w.rho)
    return j if dj > di else i

  def nearestStepwise(self, v):
    """the nearest point without a search, for equispaced ladders: the ring is the polar angle
    over the ring step and the point the azimuth over that ring's step, both rounded
    the nearest node of a grid with uniform steps, arithmetic on every direction at once: the
    ring from the polar angle, the node on the ring from the azimuth against the ring's own
    first node, step and count"""
    v = asVector3d(v)
    xyz = np.asarray(v.data, dtype=float).reshape(-1, 3)
    r = np.linalg.norm(xyz, axis=-1)
    th = np.arccos(np.clip(xyz[:, 2] / np.where(r > 0, r, 1.0), -1.0, 1.0))
    rh = np.arctan2(xyz[:, 1], xyz[:, 0])
    k = np.asarray(self.thetaGrid.ladderIndex(th)).reshape(-1)
    if not self.hasUniformSteps:
      out = np.empty(len(k), dtype=int)
      for kk in np.unique(k):
        at = np.flatnonzero(k == kk)
        out[at] = self.offset[kk] + np.asarray(self.rhoGrid[kk].ladderIndex(rh[at])).reshape(-1)
      return out.reshape(v.shape)
    rings = self.rhoGrid
    first = np.array([r.points[0] for r in rings], dtype=float)
    step = np.array([r.step if len(r) > 1 else 1.0 for r in rings], dtype=float)
    n = np.array([len(r) for r in rings], dtype=int)
    periodic = np.array([r.periodic for r in rings], dtype=bool)
    i = np.round((rh - first[k]) / step[k]).astype(int)
    j = np.where(periodic[k], np.mod(i, n[k]), np.clip(i, 0, n[k] - 1))
    return (np.asarray(self.offset)[k] + np.where(n[k] == 1, 0, j)).reshape(v.shape)

  @property
  def hasUniformSteps(self):
    """whether the polar ladder and every ring of more than one point have uniform steps, so that the
    nearest point is found by arithmetic"""
    return self.thetaGrid.step > 0 and all(len(r) == 1 or r.step > 0 for r in self.rhoGrid)

  # the points of ring k within eps of (theta, rho): the azimuth window from
  # cos eps = cos theta cos theta_k + sin theta sin theta_k cos drho
  def _ringWindow(self, k, rho, ceps, ct, st):
    tk = self.thetaGrid[k]
    cs, ss = ct * np.cos(tk), st * np.sin(tk)
    o = self.offset[k]
    if ss < 1e-10 or (ceps - cs) / ss < -1 + 1e-12:
      return self.ring(k) if cs + ss >= ceps else np.zeros(0, dtype=int)
    if cs + ss >= ceps:
      w = np.arccos(np.clip((ceps - cs) / ss, -1, 1))
      return o + self.rhoGrid[k].neighbours(rho, w)
    return np.zeros(0, dtype=int)

  def _neighbourIndicesPolar(self, theta, rho, eps):
    ceps, ct, st = np.cos(eps), np.cos(theta), np.sin(theta)
    out = [self._ringWindow(k, rho, ceps, ct, st) for k in self.thetaGrid.neighbours(theta, eps)]
    return np.concatenate(out) if out else np.zeros(0, dtype=int)

  def neighbourIndices(self, v, eps):
    """the indices of the grid points within eps of one direction"""
    v = asVector3d(v).normalize()
    out = self._neighbourIndicesPolar(float(v.theta), float(v.rho), eps)
    if self.antipodal:
      w = -v
      out = np.concatenate([out, self._neighbourIndicesPolar(float(w.theta), float(w.rho), eps)])
    return np.unique(out)

  def find(self, v, eps=None):
    """the nearest grid point of every direction, or all within eps: an index per direction,
    or a sparse incidence matrix (directions x grid) with a sparse matrix of the angles"""
    v = asVector3d(v)
    xyz = v.data.reshape(-1, 3)
    if eps is None:
      from ..kernels import backend
      ind = s2Nearest(self, xyz, backend()).reshape(v.shape)
      d = np.arctan2(np.linalg.norm(np.cross(self.data[ind], xyz.reshape(v.shape + (3,))), axis=-1), _dots(self, self.data[ind], xyz.reshape(v.shape + (3,))))
      return ind, d
    from .search import s2Tree
    return s2Tree(self).radius(xyz, eps)

  def __repr__(self):
    lens = [len(r) for r in self.rhoGrid]
    pts = f"{lens[0]} × {len(lens)}" if lens and all(n == lens[0] for n in lens) else str(self.size)
    return display(header("S2Grid", self._headerChar()), [("points", pts), ("resolution", f"{self.res / degree:.4g}°"),
                                                          ("antipodal", "true" if self.antipodal else None)])


def _dots(g, a, b):
  d = np.sum(a * b, axis=-1)
  return np.abs(d) if g.antipodal else d


# ------------------------------------------------------------------ the equispaced grid

def equispacedS2Grid(*args, resolution=2.5 * degree, points=None, minTheta=0.0, maxTheta=None, minRho=0.0, maxRho=2 * np.pi,
                     antipodal=False, hemisphere="complete", center=True, rhoPeriod=2 * np.pi, region=None, frame=None):
  """a grid of nearly equidistant directions: rings of polar angle `resolution` apart, each
  holding sin(theta) times the equator count of azimuths, odd rings offset by half a step

    equispacedS2Grid(resolution=5*degree)
    equispacedS2Grid(points=1000)
    equispacedS2Grid(resolution=5*degree, antipodal=True)      axes on the upper hemisphere
    equispacedS2Grid(cs, resolution=5*degree)                  the fundamental sector of a frame
  """
  from ..sphericalregion import fundamentalSector, sphericalRegion
  for a in args:
    if isinstance(a, sphericalRegion):
      region = a
    elif isinstance(a, referenceFrame):
      fr = a.Laue() if antipodal else a
      region, frame = fundamentalSector(fr), a
  if region is not None and frame is None:
    frame = region.frame
  if hemisphere not in ("complete", "upper", "lower", "sphere"):
    raise ValueError("hemisphere is 'complete', 'upper' or 'lower'")
  cut = antipodal and region is None
  tmax = maxTheta if maxTheta is not None else np.pi / 2 if hemisphere == "upper" or cut else np.pi
  tmin = max(minTheta, np.pi / 2) if hemisphere == "lower" else minTheta
  dth, drh = tmax - tmin, maxRho - minRho
  res = resolution if points is None else _resolutionFor(points, tmin, tmax, drh, center)
  # the ring spacing divides the polar range
  res = 2 * tmax / max(np.round(2 * tmax / res), 1)
  ntheta = max(int(np.round(2 * dth / res + (not center))) // 2, 1)
  theta = tmin + (np.arange(ntheta) + 0.5) * res if not center else tmin + np.arange(ntheta + 1) * res
  rho = []
  for j, t in enumerate(theta):
    odd = j % 2 == 1
    if cut and abs(t - np.pi / 2) < 1e-10 and abs(drh - 2 * np.pi) < 1e-10:
      # half an equator: an axis and its opposite are one point
      pts = minRho + res * (0.5 * odd + np.arange(2 * ntheta))
      rho.append(S1Grid(pts, min=minRho, max=minRho + np.pi, periodic=True))
    else:
      steps = max(int(np.round(np.sin(t) * drh / dth * ntheta)), 1)
      pts = minRho + np.arange(steps) * (drh / steps) + odd * drh / steps / 2
      rho.append(S1Grid(pts, min=minRho, max=minRho + rhoPeriod, periodic=True))
  g = S2Grid(S1Grid(theta, min=tmin, max=tmax), rho, frame, res, antipodal)
  if region is None:
    return g
  # the sieve takes each node as the point it is
  return g.subGrid(region.checkInside(vector3d(g.data)))


# the resolution giving about n points: the area over n, refined by counting
def _resolutionFor(n, tmin, tmax, drh, center):
  area = drh * (np.cos(tmin) - np.cos(tmax))
  res = np.sqrt(area / n)

  def thetas(r):
    return tmin + (np.arange(int(np.round(2 * (tmax - tmin) / r + 1)) // 2) + 0.5) * r if not center else tmin + np.arange(int(np.round(2 * (tmax - tmin) / r)) // 2 + 1) * r

  for _ in range(4):
    ts = thetas(res)
    c = sum(max(int(np.round(np.sin(t) * drh / (tmax - tmin) * len(ts))), 1) for t in ts)
    if c == 0:
      break
    res *= np.sqrt(c / n)
  return res


# ------------------------------------------------------------------ the regular grid

def regularS2Grid(resolution=5 * degree, points=None, theta=None, rho=None, minTheta=0.0, maxTheta=None, minRho=0.0,
                  maxRho=2 * np.pi, antipodal=False, hemisphere="complete", frame=None):
  """the tensor product of a polar ladder and an azimuth ladder as an nRho x nTheta matrix
  of directions, the layout of the pole figure files"""
  tmax = maxTheta if maxTheta is not None else np.pi / 2 if hemisphere == "upper" or antipodal else np.pi
  tmin = max(minTheta, np.pi / 2) if hemisphere == "lower" else minTheta
  res = np.broadcast_to(np.asarray(resolution, dtype=float), (2,))
  nr, nt = points if points is not None else (int(np.ceil((maxRho - minRho) / res[0])), int(np.ceil((tmax - tmin) / res[1] + 1)))
  th = np.asarray(theta, dtype=float) if theta is not None else np.linspace(tmin, tmax, nt)
  rh = np.asarray(rho, dtype=float) if rho is not None else ladder(minRho, maxRho, nr, periodic=True)
  T, R = np.meshgrid(th, rh, indexing="xy")
  return vector3d.byPolar(T, R, antipodal=antipodal) if frame is None else vector3d(vector3d.byPolar(T, R, antipodal=antipodal), frame=frame)


__all__ = ["S2Grid", "equispacedS2Grid", "regularS2Grid"]
