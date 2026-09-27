# SO3FunHarmonic: a function on SO(3) as the Wigner series of H5, with the frames of the
# orientations it is defined on and the exchange symmetry of ADR 0008 (MATLAB
# @SO3FunHarmonic; docs/plans/functions.md F3)
#
#   SO3F = SO3FunHarmonic(fhat, CS, SS)                 from coefficients, projected onto the invariant functions
#   SO3F = SO3FunHarmonic(psi, CS, SS)                  a kernel centred at the identity
#   SO3F = SO3FunHarmonic(fun, CS, SS, bandwidth=L)      a callable by exact quadrature
#   SO3F = SO3FunHarmonic.quadrature(nodes, values, bandwidth=L, weights=w)
#   SO3F._fhat, SO3F.CS, SO3F.SS, SO3F.bandwidth, SO3F.isReal, SO3F.antipodal, SO3F.power
#   SO3F(ori), SO3F.eval(ori), SO3F.symmetrise(), SO3F.inv(), SO3F.conj(), SO3F.truncate(L)
#   SO3F.mean(), SO3F.norm(), SO3F.dot(G), the linear algebra, SO3F.withExchange()
#
# The value type is a field: a real valued function carries the conjugate symmetry of H7
# in its coefficients, and every operation keeps it; the precision follows the coefficients.
from __future__ import annotations

from typing import TYPE_CHECKING
import math

import numpy as np

from ..geometry.frames import crystalFrame, referenceFrame, specimenFrame
from ..geometry.orientation import orientation
from ..geometry.quaternion import orthogonalTransform, quaternion
from ..geometry.session import getMTEXpref
from .harmonics import deg2dim, degreeBlock, dim2deg, invariantProjection, mtexCoefficientsSO3, project
from .kernelfunctions import SO3Kernel
from .quadrature import bandwidthFor, folds, quadratureSO3Grid
from .kernels import realizeWigner
from .transforms import isRealWigner, wignerSeries, wignerSeriesAdjoint


def _complexType(dtype):
  return np.result_type(np.dtype(dtype), 1j)


def invertedCoefficients(fhat):
  """f(g^-1): D_n^{kl}(g^-1) = (-1)^{k-l} D_n^{-l,-k}(g), so fhat'^{ab} = (-1)^{a-b} fhat^{-b,-a}"""
  out = np.empty_like(fhat)
  for n in range(dim2deg(fhat.shape[-1]) + 1):
    F = degreeBlock(fhat, n)
    k = np.arange(-n, n + 1)
    sign = np.where((k[:, None] - k[None, :]) % 2, -1.0, 1.0)
    degreeBlock(out, n)[...] = sign * F[::-1, ::-1].T
  return out


def conjugateCoefficients(fhat):
  """conj(f) has the coefficients (-1)^{k-l} conj(fhat^{-k,-l})"""
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

  def __init__(self, fhat, *frames, real=None, antipodal=None, symmetrise=True, bandwidth=None, dtype=None, _internal=False):
    # coefficients given are MTEX's, those the port passes itself its own
    fromUser = not _internal
    # the quadrature scheme, 'ClenshawCurtis' or 'GaussLegendre', named among the frames
    schemes = [a for a in frames if isinstance(a, str)]
    scheme = schemes[0] if schemes else "ClenshawCurtis"
    frames = tuple(a for a in frames if not isinstance(a, str))
    # another representation converts itself to its harmonic form
    if not isinstance(fhat, SO3FunHarmonic) and hasattr(fhat, "_harmonic"):
      fhat, bandwidth = fhat._harmonic(bandwidth), None
    if isinstance(fhat, SO3FunHarmonic):
      f = fhat
      CS, SS = _readFrames(frames) if frames else (f.CS, f.SS)
      real = f.isReal if real is None else real
      antipodal = f.antipodal if antipodal is None else antipodal
      fhat = f.truncate(bandwidth)._fhat if bandwidth is not None else f._fhat
      bandwidth, fromUser = None, False
    elif isinstance(fhat, SO3Kernel):
      CS, SS = _readFrames(frames)
      fhat = _kernelCoefficients(fhat, bandwidth)
      real, bandwidth, fromUser = True, None, False
    elif callable(fhat) and not isinstance(fhat, np.ndarray):
      CS, SS = _readFrames(frames)
      fhat, real = _byQuadrature(fhat, CS, SS, bandwidth, real, antipodal, scheme)
      bandwidth, fromUser = None, False
    else:
      CS, SS = _readFrames(frames)
    # the coefficients on the last axis, the shape of a list of functions in front
    fhat = np.array(fhat, dtype=_complexType(np.result_type(np.asarray(fhat).dtype, np.float64) if dtype is None else dtype))
    fhat = fhat.reshape(1) if fhat.ndim == 0 else fhat
    # a vector ending inside a degree block is padded with zeros to its end, as MATLAB does
    L = 0
    while deg2dim(L + 1) < fhat.shape[-1]:
      L += 1
    if deg2dim(L + 1) != fhat.shape[-1]:
      fhat = _cut(fhat, L)
    if bandwidth is not None and bandwidth != L:
      fhat = _cut(fhat, bandwidth)
    if fromUser:
      fhat = mtexCoefficientsSO3(fhat)
    antipodal = bool(antipodal)
    if antipodal and CS is not SS:
      raise ValueError("exchange symmetry needs the same frame on both sides")
    real = isRealWigner(fhat) if real is None else bool(real)
    if real:
      realizeWigner(fhat)
    self._fhat, self.CS, self.SS, self._isReal, self.antipodal = fhat, CS, SS, real, antipodal
    if symmetrise:
      self._fhat = self.symmetrise()._fhat
    elif antipodal:
      self._fhat = self.withExchange()._fhat

  # the same function with other coefficients, everything else kept
  def _rebuild(self, fhat, **changes):
    f = object.__new__(SO3FunHarmonic)
    f._fhat, f.CS, f.SS, f._isReal, f.antipodal = np.asarray(fhat), self.CS, self.SS, self.isReal, self.antipodal
    for k, v in changes.items():
      setattr(f, k, v)
    return f

  # ------------------------------------------------------------------ a list of functions

  __array_ufunc__ = None

  @property
  def shape(self):
    """the shape of a list of functions, () for one"""
    return self._fhat.shape[:-1]

  @property
  def size(self):
    return int(np.prod(self.shape))

  def __len__(self):
    return self.size

  def __getitem__(self, idx):
    if not self.shape:
      raise TypeError("a single function has no elements to index")
    return self._rebuild(self._fhat[idx])

  def reshape(self, *shape):
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return self._rebuild(self._fhat.reshape(tuple(shape) + (self._fhat.shape[-1],)))

  @property
  def T(self):
    """the list transposed, a row of n functions becoming a column"""
    if not self.shape:
      return self
    if len(self.shape) == 1:
      return self._rebuild(self._fhat[:, None, :])
    return self._rebuild(np.swapaxes(self._fhat, -2, -3))

  @classmethod
  def cat(cls, *fs, axis=0):
    L = max(f.bandwidth for f in fs)
    parts = [f.truncate(L)._fhat for f in fs]
    parts = [p[None] if p.ndim == 1 else p for p in parts]
    return fs[0]._rebuild(np.concatenate(parts, axis=axis), _isReal=all(f.isReal for f in fs))

  # an operation written for one function applied to every function of the list
  def _each(self, op):
    if not self.shape:
      return op(self)
    out = [op(self._rebuild(r)) for r in self._fhat.reshape(-1, self._fhat.shape[-1])]
    if isinstance(out[0], SO3FunHarmonic):
      L = max(f.bandwidth for f in out)
      return out[0]._rebuild(np.stack([f.truncate(L)._fhat for f in out]).reshape(self.shape + (deg2dim(L + 1),)))
    return np.array(out).reshape(self.shape)

  @classmethod
  def quadrature(cls, nodes, values=None, *frames, bandwidth=None, weights=None, real=None, antipodal=False):
    """from weighted samples at nodes, projected onto the invariant functions; a quadrature
    grid brings its weights"""
    # quadrature(fun, cs, ss): a callable takes its frames right after it
    if callable(nodes) and not isinstance(nodes, quaternion) and (values is None or isinstance(values, referenceFrame)):
      frames = frames if values is None else (values, *frames)
      return cls(nodes, *frames, bandwidth=bandwidth, real=real, antipodal=antipodal)
    L = getMTEXpref("maxSO3Bandwidth") if bandwidth is None else int(bandwidth)
    values = np.asarray(values)
    if isinstance(nodes, quadratureSO3Grid):
      # a grid brings its bandwidth, and values at its distinct nodes spread onto all of them
      L = nodes.bandwidth if bandwidth is None else L
      values = nodes.fullValues(values)
      CS, SS = (nodes.CS, nodes.SS if nodes.SS is not None else specimenFrame.default) if nodes.CS is not None else _readFrames(frames)
      fhat = wignerSeriesAdjoint(nodes, values.ravel(), L, weights)
    else:
      if isinstance(nodes, orientation):
        CS, SS = nodes.CS, nodes.SS
        antipodal = antipodal or nodes.antipodal
      else:
        CS, SS = _readFrames(frames)
      fhat = wignerSeriesAdjoint(nodes, values.ravel(), L, weights, folds=folds(CS, SS))
    return cls(fhat, CS, SS, real=(values.dtype.kind != "c") if real is None else real, antipodal=antipodal, _internal=True)

  approximate = quadrature

  @staticmethod
  def example():
    """the harmonic expansion of the Dubna ODF, MTEX's example function"""
    from .so3funclass import SO3Fun
    return SO3FunHarmonic(SO3Fun.dubna())

  # ------------------------------------------------------------------ properties

  @property
  def fhat(self):
    """the coefficients in MTEX's basis, as MATLAB's SO3F.fhat reads and assigns them"""
    return mtexCoefficientsSO3(self._fhat)

  @fhat.setter
  def fhat(self, value):
    self._fhat = mtexCoefficientsSO3(np.asarray(value, dtype=self._fhat.dtype))

  @property
  def isReal(self):
    """SO3F.isReal = True makes the coefficients those of the real part, as MATLAB's does"""
    return self._isReal

  @isReal.setter
  def isReal(self, real):
    self._isReal = bool(real)
    if self._isReal and hasattr(self, "_fhat"):
      fhat = np.array(self._fhat, dtype=complex)
      for row in fhat.reshape(-1, fhat.shape[-1]):
        realizeWigner(row)
      self._fhat = fhat

  @property
  def bandwidth(self):
    return dim2deg(self._fhat.shape[-1])

  @bandwidth.setter
  def bandwidth(self, L):
    """SO3F.bandwidth = L cuts or pads the coefficients, as MATLAB's assignment does"""
    self._fhat = _cut(self._fhat, int(L))

  @property
  def power(self):
    """the harmonic power per degree, the norm of the coefficients of that degree"""
    return np.stack([np.linalg.norm(self._fhat[..., deg2dim(n):deg2dim(n + 1)], axis=-1) for n in range(self.bandwidth + 1)], axis=-1)

  @property
  def dtype(self):
    return self._fhat.dtype

  def truncate(self, L):
    """the coefficient vector cut or padded to a bandwidth"""
    return self if L == self.bandwidth else self._rebuild(_cut(self._fhat, L))

  # ------------------------------------------------------------------ evaluation

  # the rotations of an argument in the frames of the function
  def _rotationsFor(self, x):
    if isinstance(x, orientation):
      from .rbf import _sameFrame
      if not (_sameFrame(x.CS, self.CS) and _sameFrame(x.SS, self.SS)):
        raise ValueError("the orientation lives in other frames than the function")
      return x
    if isinstance(x, orthogonalTransform):
      return x.rot
    if isinstance(x, quaternion):
      return x
    raise TypeError("a function on SO(3) is evaluated at rotations, orientations or a quadrature grid")

  def __call__(self, x):
    return self.eval(x)

  def eval(self, x):
    """the coefficients are invariant under the groups of the frames, so the transform may run
    on the reduced index set of their z folds"""
    if self.shape:
      # a list of functions: the values at the arguments first, then the shape of the list
      vals = [f.eval(x) for f in (self._rebuild(r) for r in self._fhat.reshape(-1, self._fhat.shape[-1]))]
      return np.stack(vals, axis=-1).reshape(np.shape(vals[0]) + self.shape)
    if isinstance(x, quadratureSO3Grid):
      return wignerSeries(self._fhat, x, real=self.isReal)
    r = self._rotationsFor(x)
    return wignerSeries(self._fhat, r, real=self.isReal, folds=folds(self.CS, self.SS))

  # ------------------------------------------------------------------ symmetry

  def symmetrise(self):
    """the projection of H5 on both sides, and the exchange symmetry f(g) = f(g^-1)"""
    if self.shape:
      return self._each(lambda f: f.symmetrise())
    L = self.bandwidth
    P, Q = invariantProjection(self.SS, L, self._fhat.real.dtype), invariantProjection(self.CS, L, self._fhat.real.dtype)
    fhat = project(self._fhat.copy(), P, Q)
    if self.antipodal:
      fhat = (fhat + invertedCoefficients(fhat)) / 2
    return self._rebuild(fhat)

  def inv(self):
    if self.shape:
      return self._each(lambda f: f.inv())
    return self._rebuild(invertedCoefficients(self._fhat), CS=self.SS, SS=self.CS)

  def withExchange(self, x=True):
    if x and self.CS is not self.SS:
      raise ValueError("exchange symmetry needs the same frame on both sides")
    if self.shape:
      return self._each(lambda f: f.withExchange(x))
    fhat = (self._fhat + invertedCoefficients(self._fhat)) / 2 if x else self._fhat.copy()
    return self._rebuild(fhat, antipodal=bool(x))

  # ------------------------------------------------------------------ the algebra

  def conj(self):
    if self.shape:
      return self._each(lambda f: f.conj())
    return self._rebuild(conjugateCoefficients(self._fhat))

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
      return SO3FunHarmonic(self.truncate(L)._fhat + g.truncate(L)._fhat, self.CS, self.SS, real=self.isReal and g.isReal,
                            antipodal=self.antipodal and g.antipodal, symmetrise=False, _internal=True)
    if isinstance(g, (int, float, complex, np.number)):
      fhat = self._fhat.copy()
      fhat[..., 0] += g
      return SO3FunHarmonic(fhat, self.CS, self.SS, real=self.isReal and np.isreal(g), antipodal=self.antipodal, symmetrise=False, _internal=True)
    return NotImplemented

  __radd__ = __add__

  def __neg__(self):
    return self._rebuild(-self._fhat)

  def __sub__(self, g):
    return self + (-g)

  def __rsub__(self, a):
    return (-self) + a

  def __mul__(self, a):
    if isinstance(a, SO3FunHarmonic):
      from .funops import mapFun
      return mapFun(np.multiply, self, a)
    if isinstance(a, (int, float, complex, np.number)):
      return SO3FunHarmonic(a * self._fhat, self.CS, self.SS, real=self.isReal and np.isreal(a), antipodal=self.antipodal, symmetrise=False, _internal=True)
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

  def mean(self, axis=None):
    """the mean of every function; with an axis, MATLAB's mean(SO3F, d), the function that is the
    mean along that axis of the list"""
    if axis is not None:
      return self._rebuild(self._fhat.mean(axis=_leadingAxis(axis, self.shape)))
    m = self._fhat[..., 0]
    m = m.real if self.isReal else m
    return m[()] if not self.shape else m

  def sum(self, axis=None):
    """the integral over SO(3), 8 pi^2 times the mean; with an axis the sum along it of the list"""
    if axis is not None:
      return self._rebuild(self._fhat.sum(axis=_leadingAxis(axis, self.shape)))
    return 8 * np.pi ** 2 * self.mean()

  def norm(self, s=None):
    """the L2 norm, or the Sobolev norm sqrt(sum (2n+1)^{2s} |fhat_n|^2)"""
    if self.shape:
      return self._each(lambda f: f.norm(s))
    if s is None:
      return float(np.linalg.norm(self._fhat))
    return float(math.sqrt(sum((2 * n + 1) ** (2 * s) * np.sum(np.abs(self._fhat[deg2dim(n):deg2dim(n + 1)]) ** 2) for n in range(self.bandwidth + 1))))

  def dot(self, g):
    """the inner product int conj(f) g, the first argument conjugated"""
    self._sameFrames(g)
    L = max(self.bandwidth, g.bandwidth)
    x = np.vdot(self.truncate(L)._fhat, g.truncate(L)._fhat)
    return x.real if self.isReal and g.isReal else x

  def isclose(self, g, rtol=1e-8, atol=0.0):
    """the same coefficients up to a tolerance"""
    L = max(self.bandwidth, g.bandwidth)
    a, b = self.truncate(L)._fhat, g.truncate(L)._fhat
    return bool(np.linalg.norm(a - b) <= atol + rtol * np.linalg.norm(b))

  # ------------------------------------------------------------------ the operations of funops

  def rotate(self, rot, side="left"):
    from .funops import rotate
    return rotate(self, rot, side)

  def conv(self, g, *flags):
    from .funops import conv
    return conv(self, g, *flags)

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
    lines = [f"SO3FunHarmonic ({self.CS.char()} → {self.SS.char()})"]
    if self.shape:
      from ..geometry.display import sizeString
      lines.append(f"  size: {sizeString(self.shape)}")
    lines.append(f"  bandwidth: {self.bandwidth}")
    if not self.shape:
      lines.append(f"  mean: {self.mean():.4g}")
    if self.antipodal:
      lines.append("  antipodal: true")
    if not self.isReal:
      lines.append("  isReal: false")
    return "\n".join(lines)

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from typing import Any
    from .characteristics import (max_ as max, min_ as min, entropy, textureindex, discreteSample, volume, fibreVolume,
                                  calcComponents, calcMDF, calcAngleDistribution, calcAxisDistribution, calcMIndex,
                                  calcError, centerSpecimen)
    from .polefigure import calcPoleFigure
    from ..analysis.texture import calcTensor, calcLankford, calcKearnsFactor
    from ..analysis.optimalsample import optimalSample
    from ..plotting.sphericalplot import plotPDF, plotIPDF
    from ..plotting.sectionplot import plotSection
    @property
    def SRight(self) -> Any:
      ...
    @SRight.setter
    def SRight(self, value) -> None:
      ...
    @property
    def SLeft(self) -> Any:
      ...
    @SLeft.setter
    def SLeft(self, value) -> None:
      ...


# a coefficient vector cut or padded to a bandwidth
def _cut(fhat, L):
  out = np.zeros(fhat.shape[:-1] + (deg2dim(L + 1),), dtype=fhat.dtype)
  n = min(out.shape[-1], fhat.shape[-1])
  out[..., :n] = fhat[..., :n]
  return out


# MATLAB's dimension of a list counted from zero, the coefficient axis behind it
def _leadingAxis(axis, shape):
  axis = int(axis)
  return axis + len(shape) if axis < 0 else axis


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
def _byQuadrature(fun, CS, SS, bandwidth, real, antipodal, scheme="ClenshawCurtis"):
  L0 = getMTEXpref("maxSO3Bandwidth") if bandwidth is None else int(bandwidth)
  L = bandwidthFor(L0, CS, SS)
  g = quadratureSO3Grid(L, scheme, CS, SS)
  values = np.asarray(fun(g.nodes()))
  n = len(g)
  # a function of several values, one row per node: a list of functions, one per column
  shape = values.shape[1:] if values.size != n else ()
  cols = values.reshape(n, -1)
  fhat = np.stack([wignerSeriesAdjoint(g, cols[:, k], L)[:deg2dim(L0 + 1)] for k in range(cols.shape[1])]).reshape(shape + (deg2dim(L0 + 1),))
  return fhat, (values.dtype.kind != "c") if real is None else real


__all__ = ["SO3FunHarmonic"]
