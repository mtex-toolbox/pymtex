# S1FunHarmonic: a function on the circle as the Fourier series of H7 (MATLAB @S1FunHarmonic;
# docs/plans/functions.md F3), and the density of a list of angles; the coefficients are
# MATLAB's, f(x) = sum_k fhat_k e^{-ikx}, the transforms of transforms.py taking e^{ikx}, so
# the coefficients are reversed where they meet
#
#   sF = S1FunHarmonic(fhat), S1FunHarmonic(psi), S1FunHarmonic(fun, bandwidth=L)
#   sF = S1FunHarmonic.quadrature(x, values, bandwidth=L, weights=w)
#   sF.fhat, sF.bandwidth, sF.isReal, sF.antipodal, sF.even, sF.odd
#   sF(x), sF.diff(), sF.rotate(phi), sF.conv(psi), sF.mean(), sF.norm(), sF.max(), the linear algebra
#   calcDensity(rho, sigma=, kernel=, weights=, antipodal=)
from __future__ import annotations

import math

import numpy as np

from ..geometry.session import getMTEXpref
from ..geometry.units import degree
from .kernelfunctions import S1GaussKernel, S1Kernel
from .quadrature import quadratureS1Grid
from .transforms import fourierSeries, fourierSeriesAdjoint, isRealFourier


def _complexType(dtype):
  return np.result_type(np.dtype(dtype), 1j)


# the conjugate symmetry of a real valued function, c_{-k} = conj(c_k)
def realizeFourier(chat):
  chat[...] = (chat + np.conj(chat[::-1])) / 2
  return chat


class S1FunHarmonic:

  def __init__(self, fhat, *, real=None, bandwidth=None, dtype=None):
    if not isinstance(fhat, S1FunHarmonic) and hasattr(fhat, "_harmonicS1"):
      fhat, bandwidth = fhat._harmonicS1(bandwidth), None
    if isinstance(fhat, S1FunHarmonic):
      f = fhat
      real = f.isReal if real is None else real
      fhat = f.truncate(bandwidth).fhat if bandwidth is not None else f.fhat
      bandwidth = None
    elif isinstance(fhat, S1Kernel):
      fhat = fhat.calcFourier(bandwidth)
      real, bandwidth = True, None
    elif callable(fhat) and not isinstance(fhat, np.ndarray):
      L = getMTEXpref("maxS1Bandwidth") if bandwidth is None else int(bandwidth)
      g = quadratureS1Grid(L)
      values = np.asarray(fhat(g.nodes())).ravel()
      fhat = fourierSeriesAdjoint(g, values, L)[::-1]
      real = (values.dtype.kind != "c") if real is None else real
      bandwidth = None
    chat = np.array(fhat, dtype=_complexType(np.result_type(np.asarray(fhat).dtype, np.float64) if dtype is None else dtype)).reshape(-1)
    if len(chat) % 2 == 0:
      raise ValueError("a coefficient vector runs over k = -L..L and has odd length")
    if bandwidth is not None and bandwidth != (len(chat) - 1) // 2:
      chat = _cut(chat, bandwidth)
    real = isRealFourier(chat) if real is None else bool(real)
    if real:
      realizeFourier(chat)
    self.fhat, self.isReal = chat, real

  def _rebuild(self, chat, **changes):
    f = object.__new__(S1FunHarmonic)
    f.fhat, f.isReal = np.asarray(chat), self.isReal
    for k, v in changes.items():
      setattr(f, k, v)
    return f

  # the Fourier series through samples on the circle, as MATLAB's: equispaced samples by
  # quadrature; others merged where they coincide and fitted by least squares at the
  # bandwidth round(n / 4), regularised by lambda |k|^(2 s), with LSQR
  @classmethod
  def interpolate(cls, x, values, bandwidth=None, weights=None, regularization=1e-5, SobolevIndex=2, tol=1e-6, maxit=100):
    x = np.mod(np.asarray(x, dtype=float).reshape(-1), 2 * np.pi)
    values = np.asarray(values).reshape(len(x))
    order = np.argsort(x, kind="stable")
    x, values = x[order], values[order]
    n = len(x)
    if np.linalg.norm(x - 2 * np.pi * np.arange(n) / n) < 1e-12:
      return cls.quadrature(x, values, bandwidth=(n - 1) // 2 if bandwidth is None else bandwidth)
    w = np.full(n, 1 / n) if weights is None else np.asarray(weights, dtype=float).ravel()[order]
    # coinciding angles, within 1e-6, take the mean of their values
    key = np.round(x / 1e-6).astype(np.int64)
    key[key == round(2 * np.pi / 1e-6)] = 0
    u, first, inv = np.unique(key, return_index=True, return_inverse=True)
    count = np.bincount(inv)
    x = x[first]
    values = np.bincount(inv, weights=np.real(values)) / count + (1j * np.bincount(inv, weights=np.imag(values)) / count if np.iscomplexobj(values) else 0)
    w = w[first]
    n = len(x)
    L = min(int(np.floor(n / 4 + 0.5)), getMTEXpref("maxS1Bandwidth")) if bandwidth is None else int(bandwidth)
    from .density import _lsqr
    sw = np.sqrt(w)
    reg = math.sqrt(regularization) * np.abs(np.arange(-L, L + 1)) ** SobolevIndex
    A = lambda c: np.concatenate([sw * fourierSeries(c[::-1], x, real=False), reg * c])
    At = lambda z: fourierSeriesAdjoint(x, sw * z[:n], L, np.ones(n))[::-1] + reg * z[n:]
    b = np.concatenate([sw * values, np.zeros(2 * L + 1)]).astype(complex)
    return cls(_lsqr(A, At, b, 2 * L + 1, maxit, tol), real=not np.iscomplexobj(values))

  @classmethod
  def quadrature(cls, x, values=None, *, bandwidth=None, weights=None, real=None):
    if callable(x) and values is None:
      return cls(x, bandwidth=bandwidth, real=real)
    L = getMTEXpref("maxS1Bandwidth") if bandwidth is None else int(bandwidth)
    values = np.asarray(values)
    return cls(fourierSeriesAdjoint(x, values.ravel(), L, weights)[::-1], real=(values.dtype.kind != "c") if real is None else real)

  approximate = quadrature

  @property
  def bandwidth(self):
    return (len(self.fhat) - 1) // 2

  # f(x) = f(x + pi): the odd modes vanish
  @property
  def antipodal(self):
    return bool(np.all(self.fhat[(self.bandwidth + 1) % 2::2] == 0))

  # f(x) = f(-x): c_{-k} = c_k
  @property
  def even(self):
    return bool(np.allclose(self.fhat, self.fhat[::-1]))

  @property
  def odd(self):
    return bool(np.allclose(self.fhat, -self.fhat[::-1]))

  def truncate(self, L):
    return self if L == self.bandwidth else self._rebuild(_cut(self.fhat, L))

  def __call__(self, x):
    return self.eval(x)

  def eval(self, x):
    return fourierSeries(self.fhat[::-1], x, real=self.isReal)

  # the derivative multiplies c_k by -ik
  def diff(self):
    k = np.arange(-self.bandwidth, self.bandwidth + 1)
    return S1FunHarmonic(-1j * k * self.fhat, real=self.isReal)

  def grad(self, x=None):
    return self.diff() if x is None else self.diff()(x)

  # the rotation f(x - phi)
  def rotate(self, phi):
    k = np.arange(-self.bandwidth, self.bandwidth + 1)
    return S1FunHarmonic(np.exp(1j * k * phi) * self.fhat, real=self.isReal)

  # the convolution with a de la Vallee Poussin kernel, 2.9 degree unless a kernel or a
  # halfwidth is given
  def smooth(self, psi=None, halfwidth=2.9 * degree):
    from .kernelfunctions import S1DeLaValleePoussinKernel
    return self.conv(S1DeLaValleePoussinKernel(halfwidth=halfwidth) if psi is None else psi)

  # smoothing by a kernel is one factor per mode
  def conv(self, psi):
    if isinstance(psi, S1FunHarmonic):
      L = min(self.bandwidth, psi.bandwidth)
      return S1FunHarmonic(self.truncate(L).fhat * psi.truncate(L).fhat, real=self.isReal and psi.isReal)
    c = psi.calcFourier(self.bandwidth)
    return S1FunHarmonic(c * self.fhat, real=self.isReal)

  def conj(self):
    return self._rebuild(np.conj(self.fhat[::-1]))

  def __add__(self, g):
    if isinstance(g, S1FunHarmonic):
      L = max(self.bandwidth, g.bandwidth)
      return S1FunHarmonic(self.truncate(L).fhat + g.truncate(L).fhat, real=self.isReal and g.isReal)
    if isinstance(g, (int, float, complex, np.number)):
      chat = self.fhat.copy()
      chat[self.bandwidth] += g
      return S1FunHarmonic(chat, real=self.isReal and np.isreal(g))
    return NotImplemented

  __radd__ = __add__

  def __neg__(self):
    return self._rebuild(-self.fhat)

  def __sub__(self, g):
    return self + (-g)

  def __rsub__(self, a):
    return (-self) + a

  def __mul__(self, a):
    if isinstance(a, S1FunHarmonic):
      from .funops import mapFun
      return mapFun(np.multiply, self, a)
    if isinstance(a, (int, float, complex, np.number)):
      return S1FunHarmonic(a * self.fhat, real=self.isReal and np.isreal(a))
    return NotImplemented

  __rmul__ = __mul__

  def __truediv__(self, a):
    if isinstance(a, S1FunHarmonic):
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

  def mean(self):
    x = self.fhat[self.bandwidth]
    return x.real if self.isReal else x

  def norm(self):
    return float(np.linalg.norm(self.fhat))

  def dot(self, g):
    L = max(self.bandwidth, g.bandwidth)
    x = np.vdot(self.truncate(L).fhat, g.truncate(L).fhat)
    return x.real if self.isReal and g.isReal else x

  def isclose(self, g, rtol=1e-8, atol=0.0):
    L = max(self.bandwidth, g.bandwidth)
    a, b = self.truncate(L).fhat, g.truncate(L).fhat
    return bool(np.linalg.norm(a - b) <= atol + rtol * np.linalg.norm(b))

  # the largest and the smallest value on the circle and the angle it is taken at: a fine
  # sampling picks the cell, a ternary search settles the angle
  def _extremum(self, sign, resolution):
    n = max(8, math.ceil(2 * np.pi / resolution))
    xs = 2 * np.pi * np.arange(n) / n
    v = sign * np.real(self(xs))
    i = int(np.argmax(v))
    lo, hi = xs[i] - 2 * np.pi / n, xs[i] + 2 * np.pi / n
    for _ in range(60):
      m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
      if sign * np.real(self(m1)) > sign * np.real(self(m2)):
        hi = m2
      else:
        lo = m1
    x = (lo + hi) / 2 % (2 * np.pi)
    return float(np.real(self(x))), float(x)

  # the integral over the circle, 2 pi times the mean, MATLAB's sum
  def sum(self):
    return 2 * np.pi * self.mean()

  def max(self, resolution=0.25 * degree):
    return self._extremum(1, resolution)

  def min(self, resolution=0.25 * degree):
    return self._extremum(-1, resolution)

  def __repr__(self):
    lines = ["S1FunHarmonic", f"  bandwidth: {self.bandwidth}", f"  mean: {self.mean():.4g}"]
    if not self.isReal:
      lines.append("  isReal: false")
    return "\n".join(lines)


def _cut(chat, L):
  Lf = (len(chat) - 1) // 2
  out = np.zeros(2 * L + 1, dtype=chat.dtype)
  n = min(L, Lf)
  out[L - n:L + n + 1] = chat[Lf - n:Lf + n + 1]
  return out


# ------------------------------------------------------------------ a density on the circle


# the width of Silverman's rule, on the smaller of the standard deviation and the scaled
# interquartile range
def silvermanWidth(x):
  if len(x) < 2:
    return 10 * degree
  q1, q3 = np.quantile(x, 0.25), np.quantile(x, 0.75)
  s = min(np.std(x, ddof=1), (q3 - q1) / 1.349)
  if not s > 0:
    return 10 * degree
  return ((280 * math.sqrt(math.pi)) / 729) ** 0.2 / (2 * math.pi) * s * len(x) ** -0.2


# the density of a list of angles: the weighted point masses smoothed by a Gaussian of
# width sigma (MATLAB calcDensity on angles), normalised to a mean of one so that one is the
# uniform density; without sigma the width is Silverman's rule. Angles that are axes, rho
# and rho + pi meaning the same, keep only the even modes
def calcDensity(rho, *, sigma=None, kernel=None, weights=None, antipodal=False, bandwidth=None):
  x = np.asarray(rho, dtype=float).ravel()
  if x.size == 0:
    raise ValueError("a density needs at least one angle")
  psi = kernel if kernel is not None else S1GaussKernel(silvermanWidth(x) if sigma is None else float(sigma))
  # MATLAB's cutoff for the Gauss kernel, round(4 / sigma), where the modes fall to e^-8
  if bandwidth is None:
    L = max(1, int(np.floor(4 / psi.sigma + 0.5))) if kernel is None else psi.bandwidth
  else:
    L = int(bandwidth)
  w = np.full(x.size, 1 / x.size) if weights is None else np.asarray(weights, dtype=float).ravel() / np.sum(weights)
  f = S1FunHarmonic.quadrature(x, np.ones(x.size), bandwidth=L, weights=w).conv(psi)
  if antipodal:
    k = np.arange(-L, L + 1)
    f = S1FunHarmonic(np.where(k % 2, 0, f.fhat), real=True)
  return S1FunHarmonic(f, real=True)


__all__ = ["S1FunHarmonic", "silvermanWidth"]
