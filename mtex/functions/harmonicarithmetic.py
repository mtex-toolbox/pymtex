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


__all__ = []
