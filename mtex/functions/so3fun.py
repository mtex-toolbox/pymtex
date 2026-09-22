# SO3FunHarmonic: a function on SO(3) as the Wigner series of H5, with the frames of the
# orientations it is defined on and the exchange symmetry of ADR 0008 (MATLAB
# @SO3FunHarmonic; docs/plans/functions.md F3)
#
#   SO3F = SO3FunHarmonic(fhat, CS, SS)                 from coefficients, projected onto the invariant functions
#   SO3F = SO3FunHarmonic(psi, CS, SS)                  a kernel centred at the identity
#   SO3F = SO3FunHarmonic(fun, CS, SS, bandwidth=L)      a callable by exact quadrature
#   SO3F = SO3FunHarmonic.quadrature(nodes, values, bandwidth=L, weights=w)
#   SO3F.fhat, SO3F.CS, SO3F.SS, SO3F.bandwidth, SO3F.isReal, SO3F.antipodal, SO3F.power
#   SO3F(ori), SO3F.eval(ori), SO3F.symmetrise(), SO3F.inv(), SO3F.conj(), SO3F.truncate(L)
#   SO3F.mean(), SO3F.norm(), SO3F.dot(G), the linear algebra, SO3F.withExchange()
#
# The value type is a field: a real valued function carries the conjugate symmetry of H7
# in its coefficients, and every operation keeps it; the precision follows the coefficients.
from __future__ import annotations

import math

import numpy as np

from ..geometry.frames import crystalFrame, referenceFrame, specimenFrame
from ..geometry.orientation import orientation
from ..geometry.quaternion import orthogonalTransform, quaternion
from ..geometry.session import getMTEXpref
from .harmonics import deg2dim, degreeBlock, dim2deg, invariantProjection, project
from .kernelfunctions import SO3Kernel
from .quadrature import bandwidthFor, folds, quadratureSO3Grid
from .transforms import isRealWigner, wignerSeries, wignerSeriesAdjoint


def _complexType(dtype):
  return np.result_type(np.dtype(dtype), 1j)


# the conjugate symmetry of a real valued function imposed on a coefficient array
def realizeWigner(fhat):
  for n in range(dim2deg(fhat.shape[-1]) + 1):
    F = degreeBlock(fhat, n)
    k = np.arange(-n, n + 1)
    sign = np.where((k[:, None] - k[None, :]) % 2, -1.0, 1.0)
    F[...] = (F + (sign * np.conj(F))[::-1, ::-1]) / 2
  return fhat


# f(g^-1): D_n^{kl}(g^-1) = (-1)^{k-l} D_n^{-l,-k}(g), so fhat'^{ab} = (-1)^{a-b} fhat^{-b,-a}
def invertedCoefficients(fhat):
  out = np.empty_like(fhat)
  for n in range(dim2deg(fhat.shape[-1]) + 1):
    F = degreeBlock(fhat, n)
    k = np.arange(-n, n + 1)
    sign = np.where((k[:, None] - k[None, :]) % 2, -1.0, 1.0)
    degreeBlock(out, n)[...] = sign * F[::-1, ::-1].T
  return out


# conj(f) has the coefficients (-1)^{k-l} conj(fhat^{-k,-l})
def conjugateCoefficients(fhat):
  out = np.empty_like(fhat)
  for n in range(dim2deg(fhat.shape[-1]) + 1):
    F = degreeBlock(fhat, n)
    k = np.arange(-n, n + 1)
    sign = np.where((k[:, None] - k[None, :]) % 2, -1.0, 1.0)
    degreeBlock(out, n)[...] = sign * np.conj(F[::-1, ::-1])
  return out


def _readFrames(args):
  frames = [a for a in args if isinstance(a, referenceFrame)]
  if len(frames) != len(args) or len(frames) > 2:
    raise TypeError("after the coefficients come the crystal and the specimen frame")
  CS = frames[0] if frames else crystalFrame("1")
  SS = frames[1] if len(frames) > 1 else specimenFrame.default
  return CS, SS


class SO3FunHarmonic:

  def __init__(self, fhat, *frames, real=None, antipodal=None, symmetrise=True, bandwidth=None, dtype=None):
    # another representation converts itself to its harmonic form
    if not isinstance(fhat, SO3FunHarmonic) and hasattr(fhat, "_harmonic"):
      fhat, bandwidth = fhat._harmonic(bandwidth), None
    if isinstance(fhat, SO3FunHarmonic):
      f = fhat
      CS, SS = _readFrames(frames) if frames else (f.CS, f.SS)
      real = f.isReal if real is None else real
      antipodal = f.antipodal if antipodal is None else antipodal
      fhat = f.truncate(bandwidth).fhat if bandwidth is not None else f.fhat
      bandwidth = None
    elif isinstance(fhat, SO3Kernel):
      CS, SS = _readFrames(frames)
      fhat = _kernelCoefficients(fhat, bandwidth)
      real, bandwidth = True, None
    elif callable(fhat) and not isinstance(fhat, np.ndarray):
      CS, SS = _readFrames(frames)
      fhat, real = _byQuadrature(fhat, CS, SS, bandwidth, real, antipodal)
      bandwidth = None
    else:
      CS, SS = _readFrames(frames)
    fhat = np.array(fhat, dtype=_complexType(np.result_type(np.asarray(fhat).dtype, np.float64) if dtype is None else dtype)).reshape(-1)
    L = dim2deg(len(fhat))
    if deg2dim(L + 1) != len(fhat):
      raise ValueError("a coefficient vector has length deg2dim(L+1) for its bandwidth L")
    if bandwidth is not None and bandwidth != L:
      fhat = _cut(fhat, bandwidth)
    antipodal = bool(antipodal)
    if antipodal and CS is not SS:
      raise ValueError("exchange symmetry needs the same frame on both sides")
    real = isRealWigner(fhat) if real is None else bool(real)
    if real:
      realizeWigner(fhat)
    self.fhat, self.CS, self.SS, self.isReal, self.antipodal = fhat, CS, SS, real, antipodal
    if symmetrise:
      self.fhat = self.symmetrise().fhat
    elif antipodal:
      self.fhat = (fhat + invertedCoefficients(fhat)) / 2

  # the same function with other coefficients, everything else kept
  def _rebuild(self, fhat, **changes):
    f = object.__new__(SO3FunHarmonic)
    f.fhat, f.CS, f.SS, f.isReal, f.antipodal = np.asarray(fhat), self.CS, self.SS, self.isReal, self.antipodal
    for k, v in changes.items():
      setattr(f, k, v)
    return f

  # from weighted samples at nodes, projected onto the invariant functions; a quadrature
  # grid brings its weights
  @classmethod
  def quadrature(cls, nodes, values=None, *frames, bandwidth=None, weights=None, real=None, antipodal=False):
    if callable(nodes) and values is None:
      return cls(nodes, *frames, bandwidth=bandwidth, real=real, antipodal=antipodal)
    L = getMTEXpref("maxSO3Bandwidth") if bandwidth is None else int(bandwidth)
    values = np.asarray(values)
    if isinstance(nodes, quadratureSO3Grid):
      CS, SS = (nodes.CS, nodes.SS if nodes.SS is not None else specimenFrame.default) if nodes.CS is not None else _readFrames(frames)
      fhat = wignerSeriesAdjoint(nodes, values.ravel(), L, weights)
    else:
      if isinstance(nodes, orientation):
        CS, SS = nodes.CS, nodes.SS
        antipodal = antipodal or nodes.antipodal
      else:
        CS, SS = _readFrames(frames)
      fhat = wignerSeriesAdjoint(nodes, values.ravel(), L, weights, folds=folds(CS, SS))
    return cls(fhat, CS, SS, real=(values.dtype.kind != "c") if real is None else real, antipodal=antipodal)

  approximate = quadrature

  # ------------------------------------------------------------------ properties

  @property
  def bandwidth(self):
    return dim2deg(len(self.fhat))

  # the harmonic power per degree, the norm of the coefficients of that degree
  @property
  def power(self):
    return np.array([np.linalg.norm(self.fhat[deg2dim(n):deg2dim(n + 1)]) for n in range(self.bandwidth + 1)])

  @property
  def dtype(self):
    return self.fhat.dtype

  # the coefficient vector cut or padded to a bandwidth
  def truncate(self, L):
    return self if L == self.bandwidth else self._rebuild(_cut(self.fhat, L))

  # ------------------------------------------------------------------ evaluation

  # the rotations of an argument in the frames of the function
  def _rotationsFor(self, x):
    if isinstance(x, orientation):
      if x.CS is not self.CS or x.SS is not self.SS:
        raise ValueError("the orientation lives in other frames than the function")
      return x
    if isinstance(x, orthogonalTransform):
      return x.rot
    if isinstance(x, quaternion):
      return x
    raise TypeError("a function on SO(3) is evaluated at rotations, orientations or a quadrature grid")

  def __call__(self, x):
    return self.eval(x)

  # the coefficients are invariant under the groups of the frames, so the transform may run
  # on the reduced index set of their z folds
  def eval(self, x):
    if isinstance(x, quadratureSO3Grid):
      return wignerSeries(self.fhat, x, real=self.isReal)
    r = self._rotationsFor(x)
    return wignerSeries(self.fhat, r, real=self.isReal, folds=folds(self.CS, self.SS))

  # ------------------------------------------------------------------ symmetry

  # the projection of H5 on both sides, and the exchange symmetry f(g) = f(g^-1)
  def symmetrise(self):
    L = self.bandwidth
    P, Q = invariantProjection(self.SS, L, self.fhat.real.dtype), invariantProjection(self.CS, L, self.fhat.real.dtype)
    fhat = project(self.fhat.copy(), P, Q)
    if self.antipodal:
      fhat = (fhat + invertedCoefficients(fhat)) / 2
    return self._rebuild(fhat)

  def inv(self):
    return self._rebuild(invertedCoefficients(self.fhat), CS=self.SS, SS=self.CS)

  def withExchange(self, x=True):
    if x and self.CS is not self.SS:
      raise ValueError("exchange symmetry needs the same frame on both sides")
    fhat = (self.fhat + invertedCoefficients(self.fhat)) / 2 if x else self.fhat.copy()
    return self._rebuild(fhat, antipodal=bool(x))

  # ------------------------------------------------------------------ the algebra

  def conj(self):
    return self._rebuild(conjugateCoefficients(self.fhat))

  def _sameFrames(self, g):
    if g.CS is not self.CS or g.SS is not self.SS:
      raise ValueError("the functions live in different frames")

  def __add__(self, g):
    if hasattr(g, "_harmonic") and not isinstance(g, SO3FunHarmonic):
      from .handles import SO3FunComposition
      return SO3FunComposition(self, g)
    if isinstance(g, SO3FunHarmonic):
      self._sameFrames(g)
      L = max(self.bandwidth, g.bandwidth)
      return SO3FunHarmonic(self.truncate(L).fhat + g.truncate(L).fhat, self.CS, self.SS, real=self.isReal and g.isReal,
                            antipodal=self.antipodal and g.antipodal, symmetrise=False)
    if isinstance(g, (int, float, complex, np.number)):
      fhat = self.fhat.copy()
      fhat[0] += g
      return SO3FunHarmonic(fhat, self.CS, self.SS, real=self.isReal and np.isreal(g), antipodal=self.antipodal, symmetrise=False)
    return NotImplemented

  __radd__ = __add__

  def __neg__(self):
    return self._rebuild(-self.fhat)

  def __sub__(self, g):
    return self + (-g)

  def __rsub__(self, a):
    return (-self) + a

  def __mul__(self, a):
    if isinstance(a, SO3FunHarmonic):
      from .funops import mapFun
      return mapFun(np.multiply, self, a)
    if isinstance(a, (int, float, complex, np.number)):
      return SO3FunHarmonic(a * self.fhat, self.CS, self.SS, real=self.isReal and np.isreal(a), antipodal=self.antipodal, symmetrise=False)
    return NotImplemented

  __rmul__ = __mul__

  def __truediv__(self, a):
    if isinstance(a, SO3FunHarmonic):
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

  def mean(self):
    return self.fhat[0].real if self.isReal else self.fhat[0]

  # the L2 norm, or the Sobolev norm sqrt(sum (2n+1)^{2s} |fhat_n|^2)
  def norm(self, s=None):
    if s is None:
      return float(np.linalg.norm(self.fhat))
    return float(math.sqrt(sum((2 * n + 1) ** (2 * s) * np.sum(np.abs(self.fhat[deg2dim(n):deg2dim(n + 1)]) ** 2) for n in range(self.bandwidth + 1))))

  # the inner product int conj(f) g, the first argument conjugated
  def dot(self, g):
    self._sameFrames(g)
    L = max(self.bandwidth, g.bandwidth)
    x = np.vdot(self.truncate(L).fhat, g.truncate(L).fhat)
    return x.real if self.isReal and g.isReal else x

  # the same coefficients up to a tolerance
  def isclose(self, g, rtol=1e-8, atol=0.0):
    L = max(self.bandwidth, g.bandwidth)
    a, b = self.truncate(L).fhat, g.truncate(L).fhat
    return bool(np.linalg.norm(a - b) <= atol + rtol * np.linalg.norm(b))

  # ------------------------------------------------------------------ the operations of funops

  def rotate(self, rot, side="left"):
    from .funops import rotate
    return rotate(self, rot, side)

  def conv(self, g):
    from .funops import conv
    return conv(self, g)

  def grad(self, x=None, side="left"):
    from .so3vectorfield import grad
    return grad(self, x, side)

  def laplace(self):
    from .funops import laplace
    return laplace(self)

  def pdf(self, h):
    from .funops import pdf
    return pdf(self, h)

  def ipdf(self, r):
    from .funops import ipdf
    return ipdf(self, r)

  def radon(self, h=None, r=None):
    from .funops import radon
    return radon(self, h, r)

  def calcPDF(self, h, r=None, superposition=None):
    from .density import calcPDF
    return calcPDF(self, h, r, superposition=superposition)

  @classmethod
  def interpolate(cls, nodes, values, **kwargs):
    from .density import interpolate
    return interpolate(nodes, values, **kwargs)

  # ------------------------------------------------------------------ display

  def __repr__(self):
    lines = [f"SO3FunHarmonic ({self.CS.char()} → {self.SS.char()})", f"  bandwidth: {self.bandwidth}", f"  mean: {self.mean():.4g}"]
    if self.antipodal:
      lines.append("  antipodal: true")
    if not self.isReal:
      lines.append("  isReal: false")
    return "\n".join(lines)


# a coefficient vector cut or padded to a bandwidth
def _cut(fhat, L):
  out = np.zeros(deg2dim(L + 1), dtype=fhat.dtype)
  n = min(len(out), len(fhat))
  out[:n] = fhat[:n]
  return out


# a radial kernel centred at the identity: fhat_n^{kk} = A_n / sqrt(2n+1)
def _kernelCoefficients(psi, bandwidth):
  L = psi.bandwidth if bandwidth is None else int(bandwidth)
  A = psi.calcFourier(L)
  fhat = np.zeros(deg2dim(L + 1), dtype=complex)
  for n in range(L + 1):
    degreeBlock(fhat, n)[np.arange(2 * n + 1), np.arange(2 * n + 1)] = A[n] / math.sqrt(2 * n + 1)
  return fhat


# a callable by exact quadrature on the folded rule; the callable receives the orientations
# of the rule. The projection is block diagonal in the degree and commutes with the cut, so
# the bandwidth asked for is taken before it runs
def _byQuadrature(fun, CS, SS, bandwidth, real, antipodal):
  L0 = getMTEXpref("maxSO3Bandwidth") if bandwidth is None else int(bandwidth)
  L = bandwidthFor(L0, CS, SS)
  g = quadratureSO3Grid(L, CS, SS)
  values = np.asarray(fun(g.nodes())).ravel()
  fhat = wignerSeriesAdjoint(g, values, L)[:deg2dim(L0 + 1)]
  return fhat, (values.dtype.kind != "c") if real is None else real


__all__ = ["SO3FunHarmonic"]
