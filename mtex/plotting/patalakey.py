# the misorientation colour key of Patala, Mason and Schuh (MATLAB PatalaColorKey and its
# private colormap222, colormap23, colormap422, colormap432, colormap622, after the
# MisorientationMaps code of O. Johnson and S. Patala)
#
#   key = PatalaColorKey(cs), PatalaColorKey(cs1, cs2), PatalaColorKey(grains.boundary['Magnesium', 'Magnesium'])
#   rgb = key.orientation2color(mori)
#
# The key colours a disorientation under grain exchange symmetry, so it is always antipodal,
# and it is defined for the Laue groups m-3m, m-3, mmm, 4/mmm and 6/mmm. The Rodrigues vector
# of the disorientation, turned so that the a axis lies along x, is carried by a chain of
# piecewise maps from the fundamental zone onto a cone or a ball, whose points the HSV
# model colours. S. Patala, J. K. Mason and C. A. Schuh, Prog. Mater. Sci. 57 (2012) 1383-1425.
from __future__ import annotations

import numpy as np

from ..geometry.orientation import orientation
from .directionkeys import hsv2rgb
from .orientationkeys import orientationColorKey


# the rotation matrix of an axis and an angle, applied to rows as pts @ R
def _rot(x, y, z, w):
  a = np.array([x, y, z], dtype=float)
  a = a / np.linalg.norm(a)
  K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
  return np.eye(3) + np.sin(w) * K + (1 - np.cos(w)) * K @ K


def _safe(x):
  return x + (x == 0)


# points of a cone of unit radius and height to colours
def _hsvCone(p):
  rho = np.mod(np.arctan2(p[:, 1], p[:, 0]), 2 * np.pi) / (2 * np.pi)
  return 1 - hsv2rgb(rho, np.hypot(p[:, 0], p[:, 1]) / _safe(p[:, 2]), p[:, 2])


# points of the unit ball to colours, the upper half towards white and the lower towards black
def _hsvBall(p):
  r = np.linalg.norm(p, axis=1)
  theta = np.where(r > 0, np.arccos(np.clip(p[:, 2] / _safe(r), -1, 1)), 0.0)
  rho = np.where(r > 0, np.arctan2(p[:, 1], p[:, 0]), 0.0)
  rho = np.mod(np.mod(rho, 2 * np.pi) - 2 * np.pi / 3, 2 * np.pi) / (2 * np.pi)
  up = theta < np.pi / 2
  s = np.where(up, (2 * r / (1 + r)) * theta / np.pi * 2, 0.0)
  vval = r * (2 - theta / np.pi * 2) + 0.5 - r / 2
  s = np.where(up, s, 1 - (1 - r) / _safe(2 * vval))
  v = np.where(up, 0.5 + r / 2, vval)
  c = hsv2rgb(np.minimum(1, rho), np.minimum(1, s), np.minimum(1, v))
  return np.clip(1 - c, 0, 1)


# the cube of the fundamental zone of 222 onto a prism, a cone, a hemisphere and the ball
def _cubeToBall(p, final):
  k1 = np.sqrt(3) * p.max(axis=1) / _safe(p.sum(axis=1))
  p = p * k1[:, None]
  p = p @ _rot(1, 1, 1, np.pi / 4) @ _rot(-1, 1, 0, np.arccos(1 / np.sqrt(3))) @ _rot(0, 0, 1, -np.pi / 3)
  p[:, 2] -= 1
  # prism to cone
  phi = np.mod(np.arctan2(p[:, 1], p[:, 0]), 2 * np.pi)
  rfin = np.hypot(p[:, 0], p[:, 1]) * np.sin(np.pi / 6 + np.mod(phi, 2 * np.pi / 3)) / np.sin(np.pi / 6)
  p = np.stack([np.sqrt(0.5) * rfin * np.cos(phi), np.sqrt(0.5) * rfin * np.sin(phi), p[:, 2]], axis=1)
  # cone to hemisphere
  r = np.linalg.norm(p, axis=1)
  rad = np.hypot(p[:, 0], p[:, 1])
  p = p * ((rad - p[:, 2]) / _safe(r))[:, None]
  # hemisphere to ball, in three sectors of the azimuth
  phi1 = np.mod(np.arctan2(p[:, 1], p[:, 0]), 2 * np.pi)
  sectors = [((phi1 > 0) & (phi1 < 2 * np.pi / 3), _rot(0, 0, 1, np.pi / 3)),
             ((phi1 > 2 * np.pi / 3) & (phi1 < 4 * np.pi / 3), _rot(0, 0, 1, np.pi)),
             ((phi1 > 4 * np.pi / 3) & (phi1 < 2 * np.pi), _rot(0, 0, 1, 5 * np.pi / 3))]
  for sel, g in sectors:
    q = p[sel] @ g
    phi2, rad1 = np.arctan2(q[:, 1], q[:, 0]), np.hypot(q[:, 1], q[:, 0])
    q = np.stack([rad1 * np.cos(1.5 * phi2), rad1 * np.sin(1.5 * phi2), q[:, 2]], axis=1)
    phi4, rad4 = np.arctan2(q[:, 0], -q[:, 2]), np.hypot(q[:, 0], q[:, 2])
    q = np.stack([rad4 * np.sin(2 * phi4), q[:, 1], -rad4 * np.cos(2 * phi4)], axis=1)
    phi3, rad3 = np.arctan2(q[:, 1], q[:, 0]), np.hypot(q[:, 1], q[:, 0])
    q = np.stack([rad3 * np.cos(2 * phi3 / 3), rad3 * np.sin(2 * phi3 / 3), q[:, 2]], axis=1)
    p[sel] = q @ g.T
  return _hsvBall(p @ _rot(0, 0, 1, final))


def _colormap222(p):
  return _cubeToBall(p, np.pi / 3)


# the zone of 422 or 622 onto the cube of 222: the height stretched, the pie of the n-fold
# axis opened to a quarter and the cylinder squared
def _prismToCube(p, n):
  half = np.pi / (2 * n)
  p = np.stack([p[:, 0], p[:, 1], p[:, 2] / np.tan(half)], axis=1)
  theta = np.arctan2(p[:, 1], p[:, 0])
  rad = np.hypot(p[:, 0], p[:, 1])
  edge = np.where(theta <= half, p[:, 0], (np.cos(2 * half) * p[:, 0] + np.sin(2 * half) * p[:, 1]))
  p = p * np.stack([edge / _safe(rad)] * 2 + [np.ones(len(p))], axis=1)
  theta = np.arctan2(p[:, 1], p[:, 0]) * (n // 2)
  rad = np.hypot(p[:, 0], p[:, 1])
  p = np.stack([rad * np.cos(theta), rad * np.sin(theta), p[:, 2]], axis=1)
  m = np.maximum(p[:, 0], p[:, 1])
  f = np.hypot(p[:, 0], p[:, 1]) / _safe(m)
  return np.stack([f * p[:, 0], f * p[:, 1], p[:, 2]], axis=1)


def _colormap422(p):
  return _cubeToBall(_prismToCube(p, 4), -np.pi / 3)


def _colormap622(p):
  return _cubeToBall(_prismToCube(p, 6), -np.pi / 3)


def _colormap23(p):
  p = np.minimum(p, 1)
  p = p @ _rot(1, 1, 1, np.pi / 4) @ _rot(-1, 1, 0, np.arccos(1 / np.sqrt(3))) @ _rot(0, 0, 1, -np.pi / 3)
  x, y = p[:, 0], p[:, 1]
  t = 2 * (x ** 2 + y ** 2) ** 1.5
  t1, t2 = x + np.sqrt(3) * y, x - np.sqrt(3) * y
  t3, t4 = np.sqrt(3) * x + y, np.sqrt(3) * x - y
  p = np.stack([x * t1 * t1 * t2 / _safe(t), y * t1 * t3 * t4 / _safe(t), p[:, 2] * np.sqrt(3)], axis=1)
  return _hsvCone(p @ _rot(0, 0, 1, np.pi))


# the Rodrigues vectors of m-3m, first folded into the sector x >= y >= z >= 0 of the cube
def _colormap432(p):
  p = np.sort(np.abs(p), axis=1)[:, ::-1]
  kmax = np.sqrt(2) - 1
  theta = np.arctan2(p[:, 2], p[:, 1])
  with np.errstate(divide="ignore", invalid="ignore"):
    sel = (p[:, 0] > 1 / 3) & (theta > np.arctan((1 - 2 * p[:, 0]) / p[:, 0]))
  t2 = p[sel, 1] * (1 - p[sel, 0])
  t3 = p[sel, 0] * (p[sel, 1] + p[sel, 2]) / _safe(t2)
  p[sel, 1:] *= t3[:, None]
  p = p @ _rot(1, 0, 0, -3 * np.pi / 8)
  p[:, 0] -= kmax
  t = 1 + p[:, 1] * np.tan(np.pi / 8) / _safe(p[:, 2])
  p[:, 1:] *= t[:, None]
  p = np.stack([p[:, 0], p[:, 1] * np.cos(np.pi / 8) / np.tan(np.pi / 8), p[:, 2] - p[:, 0] / np.cos(np.pi / 8)], axis=1)
  theta = np.arctan2(-p[:, 0], p[:, 1])
  f = np.sin(theta) + np.abs(np.cos(theta))
  p = np.stack([p[:, 0] * f, p[:, 1] * f, p[:, 2]], axis=1)
  theta = np.arctan2(-p[:, 0], p[:, 1])
  rad = np.hypot(p[:, 1], p[:, 0])
  tempk = np.cos(np.pi / 8) / np.tan(np.pi / 8)
  p = np.stack([-rad * np.sin(2 * theta) / kmax, rad * np.cos(2 * theta) / kmax, p[:, 2] * tempk], axis=1)
  s = _hsvCone(p @ _rot(0, 0, 1, -np.pi / 6)) @ _rot(0, 1, 0, np.pi)
  rgb = np.stack([s[:, 1], s[:, 2] + 1, s[:, 0] + 1], axis=1)
  rgb[(rgb > 1) & (rgb - 1 <= np.finfo(float).eps)] = 1
  rgb[(rgb < 0) & (rgb >= -10 * np.finfo(float).eps)] = 0
  return rgb


COLORMAPS = {"m-3m": _colormap432, "mmm": _colormap222, "m-3": _colormap23, "4/mmm": _colormap422, "6/mmm": _colormap622}
FOLD = {"mmm": 2, "4/mmm": 4, "6/mmm": 6}


# the Rodrigues vectors of a dihedral group n22 under grain exchange into the zone the maps
# expect, whatever representative the projection chose: z >= 0 by the exchange, the azimuth
# into [0, 180° / n] by the rotations and the mirrors the twofold axes give with it
def _foldDihedral(p, n):
  p = np.where(p[:, 2:] < 0, -p, p)
  w = np.mod(np.arctan2(p[:, 1], p[:, 0]), 2 * np.pi / n)
  w = np.where(w > np.pi / n, 2 * np.pi / n - w, w)
  r = np.hypot(p[:, 0], p[:, 1])
  return np.stack([r * np.cos(w), r * np.sin(w), p[:, 2]], axis=1)


class PatalaColorKey(orientationColorKey):
  """the misorientation key of Patala, Mason and Schuh for m-3m, m-3, mmm, 4/mmm and 6/mmm

    PatalaColorKey(cs), PatalaColorKey(cs1, cs2), PatalaColorKey(gB) of a grain boundary
    rgb = key.orientation2color(mori)
  """

  def __init__(self, x, CS2=None):
    if hasattr(x, "misorientation") and not isinstance(x, orientation):
      x = x.misorientation
    super().__init__(x, CS2 if CS2 is not None or isinstance(x, orientation) else x)
    self.antipodal = True
    if self.CS1.LaueName() not in COLORMAPS:
      raise ValueError(f"the Patala colour key is not defined for the symmetry {self.CS1.sym.pointGroup}")

  def orientation2color(self, mori):
    v = mori.project2FundamentalRegion(antipodal=True).Rodrigues()
    # the zone as the maps expect it, the a axis along x
    v = v.rotate(rotationAboutZ(-self.CS1.aAxis.rho))
    p = np.array(v.data.reshape(-1, 3), dtype=float)
    laue = self.CS1.LaueName()
    rgb = COLORMAPS[laue](_foldDihedral(p, FOLD[laue]) if laue in FOLD else p)
    return rgb.reshape(mori.shape + (3,))


def rotationAboutZ(w):
  from ..geometry.quaternion import rotation
  from ..geometry.vector3d import zvector
  return rotation.byAxisAngle(zvector, w)


__all__ = ["PatalaColorKey"]
