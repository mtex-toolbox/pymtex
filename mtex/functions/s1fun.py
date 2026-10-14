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

from typing import TYPE_CHECKING
import math

import numpy as np

from .harmonicarithmetic import harmonicArithmetic
from .kernels import _complexType, _groupSums
from ..geometry.session import getMTEXpref
from ..geometry.units import degree
from .kernelfunctions import S1GaussKernel, S1Kernel
from .quadrature import quadratureS1Grid
from .transforms import fourierSeries, fourierSeriesAdjoint, isRealFourier
from ..geometry.display import aligned


def realizeFourier(chat):
  """the conjugate symmetry of a real valued function, c_{-k} = conj(c_k)"""
  chat[...] = (chat + np.conj(chat[::-1])) / 2
  return chat


class S1FunHarmonic(harmonicArithmetic):
  """a function on the circle by its Fourier coefficients fhat, k = -L..L, over e^{-ikx} as
  MATLAB's

    sF = S1FunHarmonic([0.25, 0, 1, 0, 0.25])                  # from the coefficients, k = -2..2
    sF = S1FunHarmonic(lambda x: np.sin(2*x), bandwidth=16)    # a callable by quadrature
    sF = S1FunHarmonic(psi, bandwidth=32)                      # the Fourier series of an S1 kernel
    sF = S1FunHarmonic(sF, bandwidth=8)                        # another function, cut to a bandwidth

  The coefficients run from the degree -L to L, so a coefficient vector has odd length 2 L + 1.
  A callable is sampled on the equispaced quadrature grid of the bandwidth, by default the
  preference maxS1Bandwidth; an S1 kernel, an S1FunHandle or another S1FunHarmonic gives its own
  coefficients, cut or padded to the bandwidth when one is given. A real function keeps
  c_{-k} = conj(c_k), which is checked from the coefficients unless real says it.

  Input
    fhat - array of 2 L + 1 coefficients, a callable of the angle, an S1Kernel or an S1Fun
  Output
    sF - S1FunHarmonic
  Keywords
    real - whether the function takes real values only; by default read off the
        coefficients or the sampled values
    bandwidth - the largest degree L of the series
    dtype - the complex type of the coefficients, complex128 unless given
  See also
    S1FunHarmonic.quadrature, S1FunHarmonic.interpolate, S1FunHandle, S1Grid"""

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

  @classmethod
  def interpolate(cls, x, values, bandwidth=None, weights=None, regularisation=1e-5, SobolevIndex=2, tol=1e-6, maxIter=100):
    """the Fourier series through noisy samples at angles on the circle, fitted by regularised
    least squares

      sF = S1FunHarmonic.interpolate(x, values)                  # samples at the angles x
      sF = S1FunHarmonic.interpolate(x, values, bandwidth=10, regularisation=1e-3)  # a bandwidth and a stronger smoothing
      sF = S1FunHarmonic.interpolate(x, values, weights=w)       # the samples weighted

    Equispaced samples, x = 2 pi k / n, are expanded by quadrature to the bandwidth (n - 1) / 2,
    or the one given. Other angles are taken modulo 2 pi, samples at coinciding angles merged
    to their mean, and the coefficients fitted by LSQR to the weighted samples, penalised by
    regularisation times the Sobolev norm sum |k|^(2 SobolevIndex) |c_k|^2; the bandwidth is
    round(n / 4) for n distinct angles, at most the preference maxS1Bandwidth. A bandwidth
    set by hand may undersample the data.

    Input
      x - array of angles in radians
      values - array, one value per angle
    Output
      sF - S1FunHarmonic, real when the values are
    Keywords
      bandwidth - the largest degree of the series
      weights - one weight per sample, equal weights unless given
      regularisation - the factor lambda of the Sobolev penalty; 0 fits without it
      SobolevIndex - the exponent s of the Sobolev penalty |k|^(2 s)
      tol - the stopping tolerance of LSQR
      maxIter - the largest number of iterations of LSQR
    See also
      S1FunHarmonic.quadrature, S1FunHarmonic, SO3FunHarmonic.interpolate"""
    x = np.mod(np.asarray(x, dtype=float).reshape(-1), 2 * np.pi)
    values = np.asarray(values).reshape(len(x))
    order = np.argsort(x, kind="stable")
    x, values = x[order], values[order]
    n = len(x)
    if np.linalg.norm(x - 2 * np.pi * np.arange(n) / n) < 1e-12:
      return cls.quadrature(x, values, bandwidth=(n - 1) // 2 if bandwidth is None else bandwidth)
    w = np.full(n, 1 / n) if weights is None else np.asarray(weights, dtype=float).ravel()[order]
    # coinciding angles, within 1e-6, take the mean of their values
    merge = 1e-6
    key = np.round(x / merge).astype(np.int64)
    key[key == round(2 * np.pi / merge)] = 0
    _, first, inv = np.unique(key, return_index=True, return_inverse=True)
    x = x[first]
    values = _groupSums(inv, values, len(first)) / np.bincount(inv)
    w = w[first]
    n = len(x)
    L = min(int(np.floor(n / 4 + 0.5)), getMTEXpref("maxS1Bandwidth")) if bandwidth is None else int(bandwidth)
    from .density import _lsqr
    sw = np.sqrt(w)
    reg = math.sqrt(regularisation) * np.abs(np.arange(-L, L + 1)) ** SobolevIndex
    def A(c):
      return np.concatenate([sw * fourierSeries(c[::-1], x, real=False), reg * c])
    def At(z):
      return fourierSeriesAdjoint(x, sw * z[:n], L, np.ones(n))[::-1] + reg * z[n:]
    b = np.concatenate([sw * values, np.zeros(2 * L + 1)]).astype(complex)
    return cls(_lsqr(A, At, b, 2 * L + 1, maxIter, tol), real=not np.iscomplexobj(values))

  @classmethod
  def quadrature(cls, x, values=None, *, bandwidth=None, weights=None, real=None):
    """the Fourier series of a function on the circle by quadrature; MATLAB's approximate and
    quadrature

      sF = S1FunHarmonic.quadrature(lambda x: np.sin(2*x))                 # a callable on the grid of the bandwidth
      sF = S1FunHarmonic.quadrature(lambda x: np.sin(2*x), bandwidth=16)   # to another bandwidth
      sF = S1FunHarmonic.quadrature(x, values, bandwidth=16, weights=w)    # values at given angles

    A callable is sampled on the equispaced quadrature grid of the bandwidth, which makes the
    coefficients exact up to that degree; values at given angles are summed with the weights,
    the coefficient c_k = sum_j w_j f(x_j) e^{ikx_j}, which is the quadrature only when the
    weights are those of a quadrature rule for the angles (1/n for n equispaced ones). The
    bandwidth is by default the preference maxS1Bandwidth. S1FunHarmonic.approximate is the
    same function.

    Input
      x - a callable of the angle, or an array of angles in radians
    Output
      sF - S1FunHarmonic
    Keywords
      values - the values at the angles x
      bandwidth - the largest degree of the series
      weights - the quadrature weights of the angles
      real - whether the function takes real values only; by default whether the values are
          real
    See also
      S1FunHarmonic, S1FunHarmonic.interpolate, SO3FunHarmonic.quadrature"""
    if callable(x) and values is None:
      return cls(x, bandwidth=bandwidth, real=real)
    L = getMTEXpref("maxS1Bandwidth") if bandwidth is None else int(bandwidth)
    values = np.asarray(values)
    return cls(fourierSeriesAdjoint(x, values.ravel(), L, weights)[::-1], real=(values.dtype.kind != "c") if real is None else real)

  approximate = quadrature

  @property
  def bandwidth(self):
    """the bandwidth L, the largest frequency of the coefficients"""
    return (len(self.fhat) - 1) // 2

  @property
  def antipodal(self):
    """f(x) = f(x + pi): the odd modes vanish"""
    return bool(np.all(self.fhat[(self.bandwidth + 1) % 2::2] == 0))

  @property
  def even(self):
    """f(x) = f(-x): c_{-k} = c_k"""
    return bool(np.allclose(self.fhat, self.fhat[::-1]))

  @property
  def odd(self):
    """whether f(x) = -f(-x): c_{-k} = -c_k"""
    return bool(np.allclose(self.fhat, -self.fhat[::-1]))

  def truncate(self, L):
    """the function cut, or padded with zeros, to the bandwidth L"""
    return self if L == self.bandwidth else self._rebuild(_cut(self.fhat, L))

  def __call__(self, x):
    return self.eval(x)

  def eval(self, x):
    """the values at the angles x"""
    return fourierSeries(self.fhat[::-1], x, real=self.isReal)

  def diff(self):
    """the derivative multiplies c_k by -ik"""
    k = np.arange(-self.bandwidth, self.bandwidth + 1)
    return S1FunHarmonic(-1j * k * self.fhat, real=self.isReal)

  def grad(self, x=None):
    """the derivative as a function, or its values at the angles x

      dF = grad(sF)            # the derivative, an S1FunHarmonic
      d = grad(sF, [0.5, 1])  # the derivative at the angles

    The coefficient c_k of the derivative is -ik c_k, the function diff gives.

    Input
      sF - S1FunHarmonic
    Output
      dF - S1FunHarmonic, or the values at x
    Keywords
      x - angles at which to evaluate the derivative
    See also
      S1FunHarmonic.diff, S2FunHarmonic.grad"""
    return self.diff() if x is None else self.diff()(x)

  def rotate(self, phi):
    """the rotation f(x - phi)"""
    k = np.arange(-self.bandwidth, self.bandwidth + 1)
    return S1FunHarmonic(np.exp(1j * k * phi) * self.fhat, real=self.isReal)

  def smooth(self, psi=None, halfwidth=2.9 * degree):
    """the convolution with a de la Vallee Poussin kernel, 2.9 degree unless a kernel or a
    halfwidth is given; MATLAB's smooth

      sF = smooth(sF)                                          # with the de la Vallee Poussin kernel of 2.9 degree
      sF = smooth(sF, halfwidth=30*degree)                     # a wider kernel
      sF = smooth(sF, S1DeLaValleePoussinKernel(halfwidth=30*degree))  # a kernel given

    Every coefficient c_k is multiplied by the coefficient of degree k of the kernel. The
    de la Vallee Poussin kernel of halfwidth h keeps the degrees up to n = 1.2 / h unchanged and
    damps those up to 2n linearly to zero, so a narrow kernel leaves a function of low bandwidth
    as it is.

    Input
      sF - S1FunHarmonic
    Output
      sF - S1FunHarmonic, the smoothed function
    Keywords
      psi - S1Kernel, the smoothing kernel, which replaces the halfwidth
      halfwidth - the halfwidth of the de la Vallee Poussin kernel
    See also
      S1FunHarmonic.conv, S1DeLaValleePoussinKernel"""
    from .kernelfunctions import S1DeLaValleePoussinKernel
    return self.conv(S1DeLaValleePoussinKernel(halfwidth=halfwidth) if psi is None else psi)

  def conv(self, psi):
    """smoothing by a kernel is one factor per mode"""
    if isinstance(psi, S1FunHarmonic):
      L = min(self.bandwidth, psi.bandwidth)
      return S1FunHarmonic(self.truncate(L).fhat * psi.truncate(L).fhat, real=self.isReal and psi.isReal)
    c = psi.calcFourier(self.bandwidth)
    return S1FunHarmonic(c * self.fhat, real=self.isReal)

  def conj(self):
    """the complex conjugate function"""
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

  def __mul__(self, a):
    if isinstance(a, S1FunHarmonic):
      from .funops import mapFun
      return mapFun(np.multiply, self, a)
    if isinstance(a, (int, float, complex, np.number)):
      return S1FunHarmonic(a * self.fhat, real=self.isReal and np.isreal(a))
    return NotImplemented

  __rmul__ = __mul__

  def __abs__(self):
    from .funops import mapFun, smoothBandwidth
    return mapFun(np.abs, self, bandwidth=smoothBandwidth(self))

  def abs(self):
    """the pointwise absolute value, a Fourier series of a smoothing bandwidth

      sF = abs(sF)"""
    return abs(self)

  def sqrt(self, bandwidth=None):
    """the pointwise square root, a Fourier series of a smoothing bandwidth, or at the bandwidth given

      sF = sqrt(sF)                 # at twice the bandwidth, at least 32
      sF = sqrt(sF, bandwidth=64)   # at the bandwidth given

    The function is evaluated on an equispaced grid and expanded again by quadrature (mapFun).
    The square root is no trigonometric polynomial, so the expansion is cut at the bandwidth: by
    default twice that of the function, at least 32, where MATLAB takes the default bandwidth of
    S1FunHarmonic.quadrature.

    Input
      sF - S1FunHarmonic
    Output
      sF - S1FunHarmonic
    Keywords
      bandwidth - the largest frequency of the result
    See also
      mapFun, S1FunHarmonic.quadrature"""
    from .funops import mapFun, smoothBandwidth
    return mapFun(np.sqrt, self, bandwidth=smoothBandwidth(self) if bandwidth is None else bandwidth)

  def exp(self, bandwidth=None):
    """the pointwise exponential, a Fourier series of a smoothing bandwidth, or at the bandwidth given

      sF = exp(sF)                 # at twice the bandwidth, at least 32
      sF = exp(sF, bandwidth=64)   # at the bandwidth given

    The function is evaluated on an equispaced grid and expanded again by quadrature (mapFun).
    The exponential is no trigonometric polynomial, so the expansion is cut at the bandwidth: by
    default twice that of the function, at least 32, where MATLAB takes the default bandwidth of
    S1FunHarmonic.quadrature.

    Input
      sF - S1FunHarmonic
    Output
      sF - S1FunHarmonic
    Keywords
      bandwidth - the largest frequency of the result
    See also
      mapFun, S1FunHarmonic.quadrature"""
    from .funops import mapFun, smoothBandwidth
    return mapFun(np.exp, self, bandwidth=smoothBandwidth(self) if bandwidth is None else bandwidth)

  def log(self, bandwidth=None):
    """the pointwise logarithm, a Fourier series of a smoothing bandwidth, or at the bandwidth given

      sF = log(sF)                 # at twice the bandwidth, at least 32
      sF = log(sF, bandwidth=64)   # at the bandwidth given

    The function is evaluated on an equispaced grid and expanded again by quadrature (mapFun).
    The logarithm is no trigonometric polynomial, so the expansion is cut at the bandwidth: by
    default twice that of the function, at least 32, where MATLAB takes the default bandwidth of
    S1FunHarmonic.quadrature.

    Input
      sF - S1FunHarmonic
    Output
      sF - S1FunHarmonic
    Keywords
      bandwidth - the largest frequency of the result
    See also
      mapFun, S1FunHarmonic.quadrature"""
    from .funops import mapFun, smoothBandwidth
    return mapFun(np.log, self, bandwidth=smoothBandwidth(self) if bandwidth is None else bandwidth)

  def mean(self):
    """the mean over the circle, the coefficient c_0"""
    x = self.fhat[self.bandwidth]
    return x.real if self.isReal else x

  def norm(self):
    """the L2 norm for the normalised measure, the Euclidean norm of the coefficients

      t = norm(sF)"""
    return float(np.linalg.norm(self.fhat))

  def dot(self, g):
    """the inner product with g for the normalised measure, by the coefficients"""
    L = max(self.bandwidth, g.bandwidth)
    x = np.vdot(self.truncate(L).fhat, g.truncate(L).fhat)
    return x.real if self.isReal and g.isReal else x

  def isclose(self, g, rtol=1e-8, atol=0.0):
    """whether the coefficients of g agree within atol + rtol times their norm

      tf = isclose(sF1, sF2)             # equal up to a relative 1e-8
      tf = isclose(sF1, sF2, atol=1e-3)  # with an absolute tolerance

    The functions are compared at the larger of their bandwidths: the Euclidean norm of the
    difference of the coefficients, the L2 distance for the normalised measure, must not exceed
    atol + rtol |g|.

    Input
      sF1 - S1FunHarmonic
      g - S1FunHarmonic, the function to compare with
    Output
      tf - bool
    Keywords
      rtol - the tolerance relative to the norm of g
      atol - the absolute tolerance
    See also
      S1FunHarmonic.norm"""
    L = max(self.bandwidth, g.bandwidth)
    a, b = self.truncate(L).fhat, g.truncate(L).fhat
    return bool(np.linalg.norm(a - b) <= atol + rtol * np.linalg.norm(b))

  def sum(self):
    """the integral over the circle, 2 pi times the mean, MATLAB's sum

      value = sum(sF)"""
    return 2 * np.pi * self.mean()

  def max(self, *args, numLocal=1, resolution=0.25 * degree, bandwidth=None):
    """global, local and pointwise maxima of a function on the circle

      v, pos = max(sF)                           # the global maximum and the angle it is taken at
      v, pos = max(sF, resolution=1*degree)      # searched on a coarser sampling
      values, positions = max(sF, numLocal=3)    # the three largest local maxima
      sF = max(sF, 0.5)                          # the pointwise maximum with a constant
      sF = max(sF1, sF2, bandwidth=64)           # the pointwise maximum of two functions

    The function is sampled at the resolution, the discrete local maxima are refined by a
    ternary search, and the largest numLocal are returned with the values descending: a pair
    of numbers for numLocal 1, else two arrays. With a number or a second function the result
    is the pointwise maximum as a Fourier series, by quadrature at the larger bandwidth of the
    two unless bandwidth is given. MATLAB's vector valued form max(sF, [], dim) and its options
    kmax, startingNodes, tolerance and maxStepSize are not taken.

    Input
      sF, sF1, sF2 - S1FunHarmonic
    Output
      v, values - the largest values
      pos, positions - the angles, in radians, they are taken at
      sF - S1FunHarmonic, the pointwise maximum
    Keywords
      numLocal - the number of local maxima to return
      resolution - the spacing of the sampling the maxima are searched on
      bandwidth - the bandwidth of the pointwise maximum
    See also
      S1FunHarmonic.min, SO3FunHarmonic.max"""
    if args:
      return _pointwiseExtremumS1(np.maximum, self, args[0], bandwidth)
    return _localExtremaS1(self, 1, numLocal, resolution)

  def min(self, *args, numLocal=1, resolution=0.25 * degree, bandwidth=None):
    """global, local and pointwise minima of a function on the circle

      v, pos = min(sF)                           # the global minimum and the angle it is taken at
      values, positions = min(sF, numLocal=3)    # the three smallest local minima
      sF = min(sF, 0)                            # the pointwise minimum with a constant
      sF = min(sF1, sF2, bandwidth=64)           # the pointwise minimum of two functions

    The function is sampled at the resolution, the discrete local minima are refined by a
    ternary search, and the smallest numLocal are returned with the values ascending: a pair
    of numbers for numLocal 1, else two arrays. With a number or a second function the result
    is the pointwise minimum as a Fourier series, by quadrature at the larger bandwidth of the
    two unless bandwidth is given. MATLAB's vector valued form min(sF, [], dim) and its options
    kmax, startingNodes, tolerance and maxStepSize are not taken.

    Input
      sF, sF1, sF2 - S1FunHarmonic
    Output
      v, values - the smallest values
      pos, positions - the angles, in radians, they are taken at
      sF - S1FunHarmonic, the pointwise minimum
    Keywords
      numLocal - the number of local minima to return
      resolution - the spacing of the sampling the minima are searched on
      bandwidth - the bandwidth of the pointwise minimum
    See also
      S1FunHarmonic.max, SO3FunHarmonic.min"""
    if args:
      return _pointwiseExtremumS1(np.minimum, self, args[0], bandwidth)
    return _localExtremaS1(self, -1, numLocal, resolution)

  def __repr__(self):
    lines = ["S1FunHarmonic", f"  bandwidth: {self.bandwidth}", f"  mean: {self.mean():.4g}"]
    if not self.isReal:
      lines.append("  isReal: false")
    return aligned("\n".join(lines))

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from ..plotting.polarplot import plotS1 as plot


# the largest (sign 1) or the smallest (sign -1) local extrema of a function on the circle and
# the angles they are taken at: a fine sampling picks the cells of the discrete local extrema, a
# ternary search settles each angle; one pair of numbers for numLocal 1, else arrays ordered by value
def _localExtremaS1(f, sign, numLocal=1, resolution=0.25 * degree):
  n = max(8, math.ceil(2 * np.pi / resolution))
  h = 2 * np.pi / n
  xs = h * np.arange(n)
  v = sign * np.real(np.asarray(f(xs))).ravel()
  peak = (v >= np.roll(v, 1)) & (v >= np.roll(v, -1))
  # a plateau keeps its first sample
  peak &= ~((v == np.roll(v, 1)) & np.roll(peak, 1))
  idx = np.flatnonzero(peak)
  if idx.size == 0:
    idx = np.array([int(np.argmax(v))])
  lo, hi = xs[idx] - h, xs[idx] + h
  for _ in range(60):
    m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
    left = sign * np.real(np.asarray(f(m1))).ravel() > sign * np.real(np.asarray(f(m2))).ravel()
    hi, lo = np.where(left, m2, hi), np.where(left, lo, m1)
  x = (lo + hi) / 2 % (2 * np.pi)
  vals = np.real(np.asarray(f(x))).ravel()
  order = np.argsort(-sign * vals, kind="stable")[:max(1, int(numLocal))]
  if numLocal == 1:
    return float(vals[order[0]]), float(x[order[0]])
  return vals[order], x[order]


# the pointwise extremum of a function on the circle and a number or a second function, as
# MATLAB's: a Fourier series when either is one, of the larger bandwidth, else a handle
def _pointwiseExtremumS1(op, f, g, bandwidth=None):
  from .handles import S1FunHandle
  isFun = callable(g) and not np.isscalar(g)
  fun = (lambda x: op(np.real(f(x)), np.real(g(x)))) if isFun else (lambda x: op(np.real(f(x)), g))
  L = max(f.bandwidth, g.bandwidth) if isFun else f.bandwidth
  L = min(L, getMTEXpref("maxS1Bandwidth")) if bandwidth is None else int(bandwidth)
  if isinstance(f, S1FunHarmonic) or isinstance(g, S1FunHarmonic):
    return S1FunHarmonic(fun, bandwidth=L, real=True)
  return S1FunHandle(fun, bandwidth=L)


def _cut(chat, L):
  Lf = (len(chat) - 1) // 2
  out = np.zeros(2 * L + 1, dtype=chat.dtype)
  n = min(L, Lf)
  out[L - n:L + n + 1] = chat[Lf - n:Lf + n + 1]
  return out


# ------------------------------------------------------------------ a density on the circle


def silvermanWidth(x):
  """the width of Silverman's rule, on the smaller of the standard deviation and the scaled
  interquartile range"""
  if len(x) < 2:
    return 10 * degree
  q1, q3 = np.quantile(x, 0.25), np.quantile(x, 0.75)
  s = min(np.std(x, ddof=1), (q3 - q1) / 1.349)
  if not s > 0:
    return 10 * degree
  return ((280 * math.sqrt(math.pi)) / 729) ** 0.2 / (2 * math.pi) * s * len(x) ** -0.2


def calcDensity(rho, *, sigma=None, kernel=None, weights=None, antipodal=False, bandwidth=None):
  """the density of a list of angles: the weighted point masses smoothed by a Gaussian of
  width sigma (MATLAB calcDensity on angles), normalised to a mean of one so that one is the
  uniform density; without sigma the width is Silverman's rule. Angles that are axes, rho
  and rho + pi meaning the same, keep only the even modes"""
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
