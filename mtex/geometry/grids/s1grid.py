# a sorted list of angles on an interval or a circle, the search primitive of every
# equispaced grid (MATLAB @S1Grid): nearest and neighbours are binary searches with a
# wrap-around branch, and on an equispaced ladder plain arithmetic
from __future__ import annotations

import numpy as np


class S1Grid:

  def __init__(self, points, min=None, max=None, periodic=False):
    p = np.sort(np.asarray(points, dtype=float).reshape(-1))
    self.points = p
    self.min = float(p[0] if min is None else min)
    self.max = float(p[-1] if max is None else max)
    self.periodic = bool(periodic)
    self.step = _uniformStep(p)

  def __len__(self):
    return len(self.points)

  def __getitem__(self, i):
    return self.points[i]

  @property
  def period(self):
    return self.max - self.min if self.periodic else 0.0

  # an angle moved into [min, min + period)
  def fold(self, x):
    return x - np.floor((x - self.min) / self.period) * self.period if self.periodic else x

  # the periodic distance of two angles
  def dist(self, x, y):
    d = np.abs(x - y)
    if not self.periodic:
      return d
    p = self.period
    d = d - np.floor(d / p) * p
    return np.minimum(d, p - d)

  # the index of the nearest point of an equispaced ladder, arithmetic; a ladder that is not
  # equispaced falls back to the search
  def ladderIndex(self, x):
    n = len(self.points)
    if n == 1:
      return np.zeros(np.shape(x), dtype=int)
    if self.step <= 0:
      return self.nearest(x)
    i = np.round((np.asarray(x, dtype=float) - self.points[0]) / self.step).astype(int)
    return np.mod(i, n) if self.periodic else np.clip(i, 0, n - 1)

  # the first index with points[i] >= x and the last with points[i] <= x, the rounding
  # tolerance widening the window
  def _firstAtLeast(self, x):
    if self.step > 0:
      return np.clip(np.ceil((x - self.points[0]) / self.step - 1e-9).astype(int), 0, len(self.points))
    return np.searchsorted(self.points, x, side="left")

  def _lastAtMost(self, x):
    if self.step > 0:
      return np.clip(np.floor((x - self.points[0]) / self.step + 1e-9).astype(int), -1, len(self.points) - 1)
    return np.searchsorted(self.points, x, side="right") - 1

  # the index of the nearest point, for any array of angles
  def nearest(self, x):
    n = len(self.points)
    if n == 0:
      raise ValueError("the grid is empty")
    x = self.fold(np.asarray(x, dtype=float))
    i = self._firstAtLeast(x)
    cands = np.stack([np.clip(i - 1, 0, n - 1), np.clip(i, 0, n - 1), np.zeros_like(i), np.full_like(i, n - 1)])
    d = self.dist(self.points[cands], x[None])
    return cands[np.argmin(d, axis=0), np.arange(x.size).reshape(x.shape)] if x.ndim else cands[np.argmin(d)]

  # the indices of the points within eps of one angle, in order
  def neighbours(self, x, eps):
    n = len(self.points)
    if not self.periodic or 2 * eps >= self.period:
      if self.periodic:
        return np.arange(n)
      lo, hi = int(self._firstAtLeast(x - eps)), int(self._lastAtMost(x + eps))
      return np.arange(lo, hi + 1)
    a, b = self.fold(x - eps), self.fold(x + eps)
    lo, hi = int(self._firstAtLeast(a)), int(self._lastAtMost(b))
    if a <= b:
      return np.arange(lo, hi + 1)
    return np.concatenate([np.arange(0, hi + 1), np.arange(max(lo, hi + 1), n)])

  # the points selected by a mask, the range kept
  def subGrid(self, mask):
    g = S1Grid.__new__(S1Grid)
    g.points = self.points[np.asarray(mask, dtype=bool)]
    g.min, g.max, g.periodic = self.min, self.max, self.periodic
    g.step = _uniformStep(g.points)
    return g

  def __repr__(self):
    return f"S1Grid\n  points: {len(self)}\n  range: {self.min * 180 / np.pi:.4g}° to {self.max * 180 / np.pi:.4g}°\n  periodic: {self.periodic}"


# the spacing of an equispaced ladder, zero when the points are not one
def _uniformStep(p):
  if len(p) < 2:
    return 0.0
  s = (p[-1] - p[0]) / (len(p) - 1)
  if s <= 0 or np.any(np.abs(np.diff(p) - s) > 1e-9 * s):
    return 0.0
  return float(s)


# a regular ladder: n points from a to b, the end excluded when periodic
def ladder(a, b, n, periodic):
  return np.linspace(a, b, n + 1)[:n] if periodic else np.linspace(a, b, n)


__all__ = ["S1Grid"]
