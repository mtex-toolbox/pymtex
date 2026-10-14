# the arithmetic the harmonic series on the circle, the sphere and SO(3) share, written once over
# each one's own sum, negation, product and pointwise map
from __future__ import annotations

import numpy as np


class harmonicArithmetic:

  def __sub__(self, g):
    return self + (-g)

  def __rsub__(self, a):
    return (-self) + a

  # a quotient of two series is the series of the pointwise quotient, at a smoothing bandwidth
  def __truediv__(self, a):
    if isinstance(a, type(self)):
      from .funops import mapFun, smoothBandwidth
      return mapFun(np.divide, self, a, bandwidth=smoothBandwidth(self, a))
    return self * (1 / a)

  def __rtruediv__(self, a):
    from .funops import mapFun, smoothBandwidth
    return mapFun(lambda x: a / x, self, bandwidth=smoothBandwidth(self))

  # an integer power multiplies the bandwidth, any other one takes a smoothing bandwidth
  def __pow__(self, p):
    from .funops import mapFun, smoothBandwidth
    L = min(p * self.bandwidth, 2048) if isinstance(p, (int, np.integer)) and p >= 0 else smoothBandwidth(self)
    return mapFun(lambda x: x ** p, self, bandwidth=L)


# the coefficient arrays of lists of functions, (*shape, dim), given the axes MATLAB's cat gives
# them: every list padded with trailing axes of length one up to the joining axis, the axis
# counted among the list axes
def _catParts(parts, axis):
  nd = max(max(p.ndim - 1 for p in parts), 1)
  if axis < 0:
    axis += nd
  if axis < 0:
    raise ValueError(f"the axis {axis} is not an axis of the lists")
  nd = max(nd, axis + 1)
  return [p.reshape(p.shape[:-1] + (1,) * (nd - p.ndim + 1) + p.shape[-1:]) for p in parts], axis


__all__ = []
