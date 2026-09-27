# the super-Fibonacci SO(3) grid (Alexa, "Super-Fibonacci Spirals", CVPR 2022): a
# low-discrepancy spiral of unit quaternions covering SO(3), trimmed to one fundamental
# region of the frame pair. An alternative to the ring grid of equispacedSO3Grid: trivial to
# generate at any count with no ring structure, though the ring grid packs slightly more
# evenly. It carries no ladder structure, so its search is the quaternion k-d tree of
# orientationtree.py, the same one every SO(3) grid uses.
from __future__ import annotations

import numpy as np

from ..orientation import orientation
from ..quaternion import rotation
from ..units import degree

# the two irrational rates of the spiral, and the covering constant calibrated so that
# resolution is the mean nearest-neighbour spacing (119088 points give 3.62 deg on SO(3))
_PHI = np.sqrt(2.0)
_PSI = 1.533751168755204288118041
_COVER = 119088 * (3.62 * degree) ** 3


# n low-discrepancy unit quaternions on SO(3)
def _superfib(n):
  s = np.arange(n) + 0.5
  t = s / n
  d = 2 * np.pi * s
  r, R = np.sqrt(t), np.sqrt(1 - t)
  a, b = d / _PHI, d / _PSI
  return np.stack([r * np.sin(a), r * np.cos(a), R * np.sin(b), R * np.cos(b)], axis=-1)


class SO3FibonacciGrid(orientation):

  def __init__(self, rot, frameA, frameB, resolution):
    orientation.__init__(self, rot, frameA, frameB)
    self.res = float(resolution)
    self._otree = None

  # anything derived from the grid is a plain orientation list
  def _new(self, data, **attrs):
    obj = object.__new__(orientation)
    obj.data = data
    for name in self._carried:
      setattr(obj, name, attrs.pop(name, getattr(self, name)))
    if attrs:
      raise TypeError(f"orientation carries no attribute {', '.join(attrs)}")
    return obj

  @property
  def resolution(self):
    return self.res

  def subGrid(self, mask):
    return SO3FibonacciGrid(rotation._fromData(self.data[mask]), self.frameA, self.frameB, self.res)

  def find(self, o, epsOrK=None):
    """nearest (eps None), k nearest (int) or all within a radius (float), through the tree"""
    from .orientationtree import findInGrid
    return findInGrid(self, o, epsOrK)

  def __repr__(self):
    return f"SO3FibonacciGrid ({self._frameSummary()})\n  size: {self.size}\n  resolution: {self.res / degree:.4g}°"


def superfibonacciSO3Grid(CS, SS=None, resolution=5 * degree, points=None):
  """a super-Fibonacci grid of orientations covering one fundamental region of the frame
  pair: a low-discrepancy quaternion spiral trimmed to the region. `resolution` is the mean
  nearest-neighbour spacing; `points` sets the count in the region instead"""
  from ..frames import specimenFrame
  from ..orientationregion import fundamentalRegion
  SS = specimenFrame.default if SS is None else SS
  oR = fundamentalRegion(CS, SS)
  # the fraction of SO(3) the region fills, measured once, so any symmetry calibrates itself
  probe = _superfib(200000)
  accept = float(np.mean(oR.checkInside(probe)))
  if points is not None:
    nFull = int(np.round(points / max(accept, 1e-6)))
    resolution = (_COVER / nFull) ** (1 / 3)
  else:
    nFull = int(np.round(_COVER / float(resolution) ** 3))
  q = _superfib(max(nFull, 1))
  q = q[oR.checkInside(q)]
  res = (_COVER / max(nFull, 1)) ** (1 / 3)
  return SO3FibonacciGrid(rotation._fromData(q), CS, SS, res)


__all__ = ["SO3FibonacciGrid", "superfibonacciSO3Grid"]
