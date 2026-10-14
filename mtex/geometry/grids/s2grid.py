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
  per ring, whose nearest and radius searches use that structure

    S2G = equispacedS2Grid(resolution=5*degree)                 # nearly equidistant directions
    S2G = equispacedS2Grid(resolution=5*degree, maxTheta=70*degree)  # a spherical cap
    S2G = S2Grid(S2G.thetaGrid, S2G.rhoGrid, resolution=5*degree)  # from its rings

  The difference to a plain list of vector3d is that the grid knows its rings, so finding the
  grid points closest to a direction, or within a radius of it, searches the ladders instead
  of every point. A grid is built by equispacedS2Grid; anything that breaks the rings, an
  index or an operation, returns a plain vector3d. regularS2Grid and plotS2Grid return a
  matrix of vector3d, as their layout is the matrix.

  Input
    theta - S1Grid, the ladder of polar angles of the rings
    rho - a list of S1Grid, the azimuths of every ring
  Output
    S2G - S2Grid
  Keywords
    frame - the frame of the directions
    resolution - the resolution the grid was built with, in radians
    antipodal - the directions are axes
  See also
    equispacedS2Grid, regularS2Grid, fibonacciS2Grid, plotS2Grid, vector3d.find, VectorGrids"""

  # the grid from its rings: theta a ladder of polar angles, rho one azimuth grid per ring
  def __init__(self, theta, rho, frame=None, resolution=np.nan, antipodal=False):
    if len(rho) != len(theta):
      raise ValueError("one azimuth grid per ring is needed")
    self.thetaGrid, self.rhoGrid = theta, list(rho)
    self.offset = np.concatenate([[0], np.cumsum([len(r) for r in rho], dtype=np.int64)])
    th, rh = self._ringPolar()
    xyz =np.stack([np.sin(th) * np.cos(rh), np.sin(th) * np.sin(rh), np.cos(th)], axis=-1)
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

  # the polar angle and the azimuth of every point, ring after ring
  def _ringPolar(self):
    th = np.repeat(np.asarray(self.thetaGrid.points, dtype=float), np.diff(self.offset))
    rh = np.concatenate([r.points for r in self.rhoGrid]) if self.rhoGrid else np.zeros(0)
    return th, rh

  def polar(self):
    """the polar angles of the grid points, exact from the rings"""
    th, rh = self._ringPolar()
    return th, rh, np.ones(self.size)

  def subGrid(self, mask, eps=None):
    """the grid points selected by a mask, or within eps of a centre; MATLAB's S2Grid.subGrid

      g = S2G.subGrid(S2G.theta < 30*degree)        # the points a condition selects
      g = S2G.subGrid(zvector, eps=20*degree)       # the points within 20 degree of a centre

    The result is again a ring grid: the rings that empty out go, the others keep the points
    selected. A mask is a boolean array over the grid points or a list of indices.

    Input
      S2G - S2Grid
      mask - a boolean array or indices of the grid points, or with eps the centre, a vector3d
    Output
      g - S2Grid
    Keywords
      eps - the radius about the centre mask within which the points are kept
    See also
      S2Grid.find, equispacedS2Grid"""
    if eps is not None:
      center, mask = mask, np.zeros(self.size, dtype=bool)
      mask[self.neighbourIndices(center, eps)] = True
    mask = np.asarray(mask)
    if mask.dtype != bool:
      m = np.zeros(self.size, dtype=bool)
      m[mask] = True
      mask = m
    counts = np.add.reduceat(mask, self.offset[:-1]) if self.size else np.zeros(self.numRings, dtype=np.int64)
    keep = counts > 0
    rho = [self.rhoGrid[k].subGrid(mask[self.ring(k)]) for k in np.flatnonzero(keep)]
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
    """the nearest grid point of every direction and its angle, or with eps the sparse matrices of
    the points within eps and their angles; MATLAB's S2Grid.find

      ind, d = S2G.find(v)              # the nearest grid point and the angle to it
      I, d = S2G.find(v, 10*degree)     # every grid point within 10 degree

    The search for the nearest point runs over the rings, a binary search in the polar angle and in
    the azimuth of each ring near it; the search within eps through a k-d tree. On an antipodal grid a direction and its antipode are the same.

    Input
      S2G - S2Grid
      v - vector3d, the directions searched for
    Output
      ind - the index of the nearest grid point, of the shape of v
      d - its angle to v; with eps a sparse matrix (directions x grid) of the angles
      I - with eps a sparse boolean matrix (directions x grid) of the grid points within eps
    Keywords
      eps - the radius of the search; by default only the nearest point is returned
    See also
      equispacedS2Grid, S2Grid.subGrid, fibonacciS2Grid.find"""
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
  """an equispaced spherical grid, nearly equidistant directions on rings of constant polar
  angle

    S2G = equispacedS2Grid(resolution=5*degree)                    # the sphere at 5 degree
    S2G = equispacedS2Grid(points=1000)                            # about 1000 points
    S2G = equispacedS2Grid(resolution=5*degree, antipodal=True)    # axes, the upper hemisphere
    S2G = equispacedS2Grid(resolution=5*degree, maxTheta=70*degree)  # a spherical cap
    S2G = equispacedS2Grid(cs, resolution=5*degree)                # the fundamental sector of a frame

  The rings lie resolution apart in the polar angle, each holding sin(theta) times the equator
  count of azimuths, the odd rings offset by half a step. With a number of points the
  resolution is chosen to give about that many. A frame restricts the grid to its
  fundamental sector, of the Laue group for axes; a sphericalRegion to that region.

  Python has no 'identified' hemisphere and no restrict2MinMax, and center is the flag of a
  point at the pole, not MATLAB's centre direction.

  Input
    args - a crystalFrame or specimenFrame, whose fundamental sector the grid covers, or a
        sphericalRegion
  Output
    S2G - S2Grid
  Keywords
    resolution - the spacing of the rings and of the points on a ring, in radians
    points - the number of points instead of the resolution
    minTheta - the smallest polar angle
    maxTheta - the largest polar angle; by default pi, or pi/2 for axes and the upper
        hemisphere
    minRho - the smallest azimuth
    maxRho - the largest azimuth
    antipodal - the directions are axes, the grid covers the upper hemisphere
    hemisphere - 'complete', 'upper' or 'lower'
    center - the first ring is the pole; False starts half a step from it, MATLAB's
        'no_center'
    rhoPeriod - the period of the azimuths, of the searches on a ring
    region - a sphericalRegion the grid is restricted to
    frame - the frame of the directions; by default that of the frame or region given
  See also
    S2Grid, regularS2Grid, fibonacciS2Grid, plotS2Grid, equispacedSO3Grid, VectorGrids"""
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
  tmin, tmax = _polarRange(minTheta, maxTheta, hemisphere, cut)
  dth, drh = tmax - tmin, maxRho - minRho
  res = resolution if points is None else _resolutionFor(points, tmin, tmax, drh, center)
  # the ring spacing divides the polar range
  res = 2 * tmax / max(np.round(2 * tmax / res), 1)
  ntheta = max(_ringCount(dth, res, center), 1)
  theta = _thetaLadder(tmin, ntheta, res, center)
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


# the polar range of a hemisphere, the upper one for axes
def _polarRange(minTheta, maxTheta, hemisphere, upper):
  if maxTheta is not None:
    tmax = maxTheta
  else:
    tmax = np.pi / 2 if hemisphere == "upper" or upper else np.pi
  tmin = max(minTheta, np.pi / 2) if hemisphere == "lower" else minTheta
  return tmin, tmax


# the number of rings a polar range of dth holds at the spacing res, and their polar angles:
# the rings at half steps, or a centred ladder from tmin with one ring more
def _ringCount(dth, res, center):
  return int(np.round(2 * dth / res + (not center))) // 2


def _thetaLadder(tmin, n, res, center):
  return tmin + np.arange(n + 1) * res if center else tmin + (np.arange(n) + 0.5) * res


# the resolution giving about n points: the area over n, refined by counting
def _resolutionFor(n, tmin, tmax, drh, center):
  area = drh * (np.cos(tmin) - np.cos(tmax))
  res = np.sqrt(area / n)
  for _ in range(4):
    ts = _thetaLadder(tmin, _ringCount(tmax - tmin, res, center), res, center)
    c = sum(max(int(np.round(np.sin(t) * drh / (tmax - tmin) * len(ts))), 1) for t in ts)
    if c == 0:
      break
    res *= np.sqrt(c / n)
  return res


# ------------------------------------------------------------------ the regular grid

def regularS2Grid(resolution=5 * degree, points=None, theta=None, rho=None, minTheta=0.0, maxTheta=None, minRho=0.0,
                  maxRho=2 * np.pi, antipodal=False, hemisphere="complete", frame=None):
  """a regular spherical grid, the tensor product of a ladder of polar angles and a ladder of
  azimuths, the layout of pole figure files

    S2G = regularS2Grid(resolution=5*degree)                       # 5 degree in both angles
    S2G = regularS2Grid(resolution=[5*degree, 2.5*degree])         # in the azimuth and the polar angle
    S2G = regularS2Grid(points=(72, 19))                           # 72 azimuths, 19 polar angles
    S2G = regularS2Grid(theta=np.arange(0, 81, 5)*degree, rho=np.arange(0, 360, 5)*degree)  # the angles given

  The grid is an nRho x nTheta matrix of vector3d, rho along the first axis and theta along
  the second. The polar angles run from minTheta to maxTheta both included, the azimuths
  from minRho to maxRho periodically, the last left out. Python has no 'identified'
  hemisphere and no restrict2MinMax.

  Output
    S2G - vector3d, an nRho x nTheta matrix
  Keywords
    resolution - the spacing in radians, one for both angles, or the pair of the azimuth and
        the polar angle
    points - (nRho, nTheta), the number of azimuths and of polar angles instead of the
        resolution
    theta - the polar angles, instead of a ladder
    rho - the azimuths, instead of a ladder
    minTheta - the smallest polar angle
    maxTheta - the largest polar angle; by default pi, or pi/2 for axes and the upper
        hemisphere
    minRho - the smallest azimuth
    maxRho - the largest azimuth, the period of the azimuths
    antipodal - the directions are axes, the grid covers the upper hemisphere
    hemisphere - 'complete', 'upper' or 'lower'
    frame - the frame of the directions
  See also
    equispacedS2Grid, plotS2Grid, S2Grid, VectorGrids"""
  tmin, tmax = _polarRange(minTheta, maxTheta, hemisphere, antipodal)
  res = np.broadcast_to(np.asarray(resolution, dtype=float), (2,))
  nr, nt = points if points is not None else (int(np.ceil((maxRho - minRho) / res[0])), int(np.ceil((tmax - tmin) / res[1] + 1)))
  th = np.asarray(theta, dtype=float) if theta is not None else np.linspace(tmin, tmax, nt)
  rh = np.asarray(rho, dtype=float) if rho is not None else ladder(minRho, maxRho, nr, periodic=True)
  T, R = np.meshgrid(th, rh, indexing="xy")
  return vector3d.byPolar(T, R, antipodal=antipodal) if frame is None else vector3d(vector3d.byPolar(T, R, antipodal=antipodal), frame=frame)


__all__ = ["S2Grid", "equispacedS2Grid", "regularS2Grid"]
