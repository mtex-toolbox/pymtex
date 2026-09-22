# S2FunHarmonic: a function on the sphere as the series of H4, on directions or on axes
# (even degrees only), in a frame or none (MATLAB @S2FunHarmonic; docs/plans/functions.md F3)
#
#   sF = S2FunHarmonic(fhat), S2FunHarmonic(fhat, cs), S2FunHarmonic(psi), S2FunHarmonic(fun, bandwidth=L)
#   sF = S2FunHarmonic.quadrature(nodes, values, bandwidth=L, weights=w)
#   sF.fhat, sF.frame, sF.bandwidth, sF.isReal, sF.antipodal
#   sF(v), sF.eval(v), sF.symmetrise(), sF.conj(), sF.mean(), sF.norm(), sF.dot(t), the linear algebra
from __future__ import annotations

import math

import numpy as np

from ..geometry.frames import referenceFrame
from ..geometry.quaternion import rotation
from ..geometry.session import SESSION, getMTEXpref
from ..geometry.vector3d import vector3d
from .harmonics import WignerD
from .kernelfunctions import S2Kernel
from .quadrature import quadratureS2Grid
from .transforms import isRealSpherical, sphericalSeries, sphericalSeriesAdjoint


def _complexType(dtype):
  return np.result_type(np.dtype(dtype), 1j)


# the conjugate symmetry of a real valued function imposed on a coefficient array
def realizeSpherical(fhat):
  L = math.isqrt(len(fhat)) - 1
  for l in range(L + 1):
    c = fhat[l * l:(l + 1) ** 2]
    m = np.arange(-l, l + 1)
    c[...] = (c + np.where(m % 2, -1.0, 1.0) * np.conj(c)[::-1]) / 2
  return fhat


# an axis valued function has even degrees only
def evenize(fhat):
  L = math.isqrt(len(fhat)) - 1
  for l in range(1, L + 1, 2):
    fhat[l * l:(l + 1) ** 2] = 0
  return fhat


# the degree of every coefficient
def _degrees(n):
  return np.floor(np.sqrt(np.arange(n))).astype(int)


# the projection onto the functions invariant under a group: an element s acts by
# f(s^-1 v), so the projection of degree l is the mean of eps_s^l D_l(r_s) with eps the
# parity of s, over all elements including the improper ones
def sphericalProjection(pg, L, dtype=np.float64):
  els = pg.rot.flatten()
  P = [np.zeros((2 * l + 1, 2 * l + 1), dtype=_complexType(dtype)) for l in range(L + 1)]
  D = WignerD(els.rot, bandwidth=L, dtype=dtype)
  improper = els.isImproper
  for l in range(L + 1):
    sign = np.where(improper & (l % 2 == 1), -1.0, 1.0)
    P[l] = np.einsum("i,ikl->kl", sign, D[l]) / els.size
  return P


class S2FunHarmonic:

  def __init__(self, fhat, frame=None, *, real=None, antipodal=None, symmetrise=True, bandwidth=None, dtype=None):
    # another representation converts itself to its harmonic form
    if not isinstance(fhat, S2FunHarmonic) and hasattr(fhat, "_harmonicS2"):
      fhat, bandwidth = fhat._harmonicS2(bandwidth), None
    if isinstance(fhat, S2FunHarmonic):
      f = fhat
      frame = f.frame if frame is None else frame
      real = f.isReal if real is None else real
      antipodal = f.antipodal if antipodal is None else antipodal
      fhat = f.truncate(bandwidth).fhat if bandwidth is not None else f.fhat
      bandwidth = None
    elif isinstance(fhat, S2Kernel):
      fhat = _kernelCoefficients(fhat, bandwidth)
      real, bandwidth = True, None
    elif callable(fhat) and not isinstance(fhat, np.ndarray):
      fhat, real = _byQuadrature(fhat, frame, bandwidth, real)
      bandwidth = None
    fhat = np.array(fhat, dtype=_complexType(np.result_type(np.asarray(fhat).dtype, np.float64) if dtype is None else dtype)).reshape(-1)
    L = math.isqrt(len(fhat)) - 1
    if (L + 1) ** 2 != len(fhat):
      raise ValueError("a coefficient vector has length (L+1)^2 for its bandwidth L")
    if bandwidth is not None and bandwidth != L:
      fhat = _cut(fhat, bandwidth)
    real = isRealSpherical(fhat) if real is None else bool(real)
    if real:
      realizeSpherical(fhat)
    # a function with a centrosymmetric frame is even whether or not it was asked to be
    antipodal = bool(antipodal) or bool(getattr(frame, "isLaue", False))
    if antipodal:
      evenize(fhat)
    self.fhat, self.frame, self.isReal, self.antipodal = fhat, frame, real, antipodal
    if symmetrise:
      self.fhat = self.symmetrise().fhat

  def _rebuild(self, fhat, **changes):
    f = object.__new__(S2FunHarmonic)
    f.fhat, f.frame, f.isReal, f.antipodal = np.asarray(fhat), self.frame, self.isReal, self.antipodal
    for k, v in changes.items():
      setattr(f, k, v)
    return f

  # from weighted samples; the weights integrate to 4 pi, a quadrature grid brings its own
  @classmethod
  def quadrature(cls, nodes, values=None, frame=None, *, bandwidth=None, weights=None, real=None, antipodal=None):
    if callable(nodes) and values is None:
      return cls(nodes, frame, bandwidth=bandwidth, real=real, antipodal=bool(antipodal))
    L = getMTEXpref("maxS2Bandwidth") if bandwidth is None else int(bandwidth)
    values = np.asarray(values)
    if isinstance(nodes, quadratureS2Grid):
      frame = nodes.CS if frame is None else frame
      fhat = sphericalSeriesAdjoint(nodes, values.ravel(), L, weights)
    else:
      v = nodes if isinstance(nodes, vector3d) else vector3d(nodes)
      frame = v.frame if frame is None else frame
      antipodal = v.antipodal if antipodal is None else antipodal
      w = np.full(v.size, 4 * np.pi / v.size) if weights is None else np.asarray(weights).ravel()
      fhat = sphericalSeriesAdjoint(v, values.ravel(), L, w)
    return cls(fhat, frame, real=(values.dtype.kind != "c") if real is None else real, antipodal=bool(antipodal))

  approximate = quadrature

  @classmethod
  def interpolate(cls, nodes, values, **kwargs):
    from .density import interpolate
    return interpolate(nodes, values, **kwargs)

  @property
  def CS(self):
    return self.frame

  @property
  def bandwidth(self):
    return math.isqrt(len(self.fhat)) - 1

  @property
  def power(self):
    return np.array([np.linalg.norm(self.fhat[l * l:(l + 1) ** 2]) for l in range(self.bandwidth + 1)])

  def truncate(self, L):
    return self if L == self.bandwidth else self._rebuild(_cut(self.fhat, L))

  # ------------------------------------------------------------------ evaluation

  # bare vectors, or framed ones in the frame of the function or one sharing its basis
  def _vectorsFor(self, x):
    v = x if isinstance(x, vector3d) else vector3d(x)
    if self.frame is not None and v.frame is not None and v.frame is not self.frame:
      try:
        SESSION.joinFrames(v.frame, self.frame)
      except ValueError:
        raise ValueError("the directions live in another frame than the function") from None
    return v

  def __call__(self, x):
    return self.eval(x)

  def eval(self, x):
    if isinstance(x, quadratureS2Grid):
      return sphericalSeries(self.fhat, x, real=self.isReal)
    v = self._vectorsFor(x)
    return sphericalSeries(self.fhat, v.normalize(), real=self.isReal)

  # ------------------------------------------------------------------ symmetry

  # the projection onto the functions invariant under the group of the frame, or under a
  # list of rotations forming a group the frame does not name; an axis valued function is even
  def symmetrise(self, rots=None):
    L = self.bandwidth
    fhat = self.fhat.copy()
    if rots is not None:
      rots = rots.flatten()
      D = WignerD(rots, bandwidth=L, dtype=self.fhat.real.dtype)
      P = [np.mean(D[l], axis=0) for l in range(L + 1)]
    elif self.frame is not None:
      P = sphericalProjection(self.frame.sym, L, self.fhat.real.dtype)
    else:
      P = None
    if P is not None:
      for l in range(L + 1):
        fhat[l * l:(l + 1) ** 2] = P[l] @ fhat[l * l:(l + 1) ** 2]
    if self.antipodal:
      evenize(fhat)
    return self._rebuild(fhat)

  # the even part, f(v) + f(-v) over two
  def even(self):
    return self._rebuild(evenize(self.fhat.copy()), antipodal=True)

  # ------------------------------------------------------------------ the algebra

  def conj(self):
    L = self.bandwidth
    out = np.empty_like(self.fhat)
    for l in range(L + 1):
      m = np.arange(-l, l + 1)
      out[l * l:(l + 1) ** 2] = np.where(m % 2, -1.0, 1.0) * np.conj(self.fhat[l * l:(l + 1) ** 2][::-1])
    return self._rebuild(out)

  def _sameFrames(self, g):
    if g.frame is not self.frame:
      raise ValueError("the functions live in different frames")

  def __add__(self, g):
    if hasattr(g, "_harmonicS2"):
      from .handles import S2FunSum
      return S2FunSum(self, g)
    if isinstance(g, S2FunHarmonic):
      self._sameFrames(g)
      L = max(self.bandwidth, g.bandwidth)
      return S2FunHarmonic(self.truncate(L).fhat + g.truncate(L).fhat, self.frame, real=self.isReal and g.isReal,
                           antipodal=self.antipodal and g.antipodal, symmetrise=False)
    if isinstance(g, (int, float, complex, np.number)):
      fhat = self.fhat.copy()
      fhat[0] += g * math.sqrt(4 * math.pi)
      return S2FunHarmonic(fhat, self.frame, real=self.isReal and np.isreal(g), antipodal=self.antipodal, symmetrise=False)
    return NotImplemented

  __radd__ = __add__

  def __neg__(self):
    return self._rebuild(-self.fhat)

  def __sub__(self, g):
    return self + (-g)

  def __rsub__(self, a):
    return (-self) + a

  def __mul__(self, a):
    if isinstance(a, S2FunHarmonic):
      from .funops import mapFun
      return mapFun(np.multiply, self, a)
    if isinstance(a, (int, float, complex, np.number)):
      return S2FunHarmonic(a * self.fhat, self.frame, real=self.isReal and np.isreal(a), antipodal=self.antipodal, symmetrise=False)
    return NotImplemented

  __rmul__ = __mul__

  def __truediv__(self, a):
    if isinstance(a, S2FunHarmonic):
      from .funops import mapFun, smoothBandwidth
      return mapFun(np.divide, self, a, bandwidth=smoothBandwidth(self, a))
    return self * (1 / a)

  def __rtruediv__(self, a):
    from .funops import mapFun, smoothBandwidth
    return mapFun(lambda x: a / x, self, bandwidth=smoothBandwidth(self))

  def __pow__(self, p):
    from .funops import mapFun, smoothBandwidth
    L = min(p * self.bandwidth, 2048) if isinstance(p, (int, np.integer)) and p >= 0 else smoothBandwidth(self)
    return mapFun(lambda x: x ** p, self, bandwidth=L)

  def __abs__(self):
    from .funops import mapFun, smoothBandwidth
    return mapFun(np.abs, self, bandwidth=smoothBandwidth(self))

  def abs(self):
    return abs(self)

  def sqrt(self):
    from .funops import mapFun, smoothBandwidth
    return mapFun(np.sqrt, self, bandwidth=smoothBandwidth(self))

  def exp(self):
    from .funops import mapFun, smoothBandwidth
    return mapFun(np.exp, self, bandwidth=smoothBandwidth(self))

  def log(self):
    from .funops import mapFun, smoothBandwidth
    return mapFun(np.log, self, bandwidth=smoothBandwidth(self))

  # ------------------------------------------------------------------ the moments

  # the mean over the sphere, int f dS / 4 pi
  def mean(self):
    x = self.fhat[0] / math.sqrt(4 * math.pi)
    return x.real if self.isReal else x

  # the integral over the sphere, int f dS, S2Fun/sum
  def sum(self):
    return 4 * math.pi * self.mean()

  # the L2 norm for the normalised surface measure, or the Sobolev norm
  def norm(self, s=None):
    if s is None:
      return float(np.linalg.norm(self.fhat)) / math.sqrt(4 * math.pi)
    return float(math.sqrt(sum((2 * l + 1) ** (2 * s) * np.sum(np.abs(self.fhat[l * l:(l + 1) ** 2]) ** 2) for l in range(self.bandwidth + 1)))) / math.sqrt(4 * math.pi)

  def dot(self, g):
    self._sameFrames(g)
    L = max(self.bandwidth, g.bandwidth)
    x = np.vdot(self.truncate(L).fhat, g.truncate(L).fhat) / (4 * math.pi)
    return x.real if self.isReal and g.isReal else x

  def isclose(self, g, rtol=1e-8, atol=0.0):
    L = max(self.bandwidth, g.bandwidth)
    a, b = self.truncate(L).fhat, g.truncate(L).fhat
    return bool(np.linalg.norm(a - b) <= atol + rtol * np.linalg.norm(b))

  # ------------------------------------------------------------------ the operations of funops

  # rot * f is the function rotated, as MATLAB writes it
  def __rmul__(self, other):
    from ..geometry.quaternion import quaternion
    if isinstance(other, quaternion):
      return self.rotate(other)
    return self.__mul__(other)

  def rotate(self, rot):
    from .funops import rotate
    return rotate(self, rot)

  def conv(self, g):
    from .funops import conv
    return conv(self, g)

  def laplace(self):
    from .funops import laplace
    return laplace(self)

  def grad(self, x=None):
    from .s2vectorfield import grad
    return grad(self, x)

  def radon(self):
    from .funops import radon
    return radon(self)

  def invRadon(self):
    from .funops import invRadon
    return invRadon(self)

  def __repr__(self):
    info = "" if self.frame is None else f" ({self.frame.char()})"
    lines = [f"S2FunHarmonic{info}", f"  bandwidth: {self.bandwidth}", f"  mean: {self.mean():.4g}"]
    if self.antipodal:
      lines.append("  antipodal: true")
    if not self.isReal:
      lines.append("  isReal: false")
    return "\n".join(lines)


def _cut(fhat, L):
  out = np.zeros((L + 1) ** 2, dtype=fhat.dtype)
  n = min(len(out), len(fhat))
  out[:n] = fhat[:n]
  return out


# a zonal kernel centred at z: fhat_l^0 = A_l sqrt(4 pi / (2l+1))
def _kernelCoefficients(psi, bandwidth):
  L = psi.bandwidth if bandwidth is None else int(bandwidth)
  A = psi.calcFourier(L)
  fhat = np.zeros((L + 1) ** 2, dtype=complex)
  for l in range(L + 1):
    fhat[l * l + l] = A[l] * math.sqrt(4 * math.pi / (2 * l + 1))
  return fhat


# a callable by exact quadrature; the callable receives the directions of the rule in the frame
def _byQuadrature(fun, frame, bandwidth, real):
  L = getMTEXpref("maxS2Bandwidth") if bandwidth is None else int(bandwidth)
  g = quadratureS2Grid(L)
  v = g.nodes()
  if frame is not None:
    v = vector3d(v, frame=frame)
  values = np.asarray(fun(v)).ravel()
  return sphericalSeriesAdjoint(g, values, L), (values.dtype.kind != "c") if real is None else real


__all__ = ["S2FunHarmonic"]
