# S2FunHarmonic: a function on the sphere as the series of H4, on directions or on axes
# (even degrees only), in a frame or none (MATLAB @S2FunHarmonic; docs/plans/functions.md F3)
#
#   sF = S2FunHarmonic(fhat), S2FunHarmonic(fhat, cs), S2FunHarmonic(psi), S2FunHarmonic(fun, bandwidth=L)
#   sF = S2FunHarmonic.quadrature(nodes, values, bandwidth=L, weights=w)
#   sF._fhat, sF.frame, sF.bandwidth, sF.isReal, sF.antipodal
#   sF(v), sF.eval(v), sF.symmetrise(), sF.conj(), sF.mean(), sF.norm(), sF.dot(t), the linear algebra
#
# A list of functions is one container, as in MATLAB: the coefficients sit on the last axis
# of fhat and the leading axes are the shape of the list, S2FunHarmonic(np.eye(10)) being
# ten functions; sF[1:3], sF.reshape(3, -1), sF.T and cat index and arrange them, sF(v) is
# v.shape + sF.shape, and the moments are one per function.
from __future__ import annotations

import math

import numpy as np

from ..geometry.frames import referenceFrame
from ..geometry.quaternion import rotation
from ..geometry.session import SESSION, getMTEXpref
from ..geometry.vector3d import vector3d
from .harmonics import _WignerD, _projectBlock, invariantProjection, mtexCoefficientsS2
from .kernelfunctions import S2Kernel
from .quadrature import quadratureS2Grid
from .kernels import realizeSpherical
from .transforms import isRealSpherical, sphericalSeries, sphericalSeriesAdjoint


# the bandwidth a callable or scattered values are expanded to when none is asked for, what
# MATLAB's S2FunHarmonic.quadrature gives, within the preference maxS2Bandwidth
QUADRATURE_BANDWIDTH = 250


def quadratureBandwidth():
  return min(QUADRATURE_BANDWIDTH, getMTEXpref("maxS2Bandwidth"))


def _complexType(dtype):
  return np.result_type(np.dtype(dtype), 1j)


# an axis valued function has even degrees only
def evenize(fhat):
  L = math.isqrt(fhat.shape[-1]) - 1
  for l in range(1, L + 1, 2):
    fhat[..., l * l:(l + 1) ** 2] = 0
  return fhat


# whether every function of a coefficient array is real valued
def _isReal(fhat):
  return all(isRealSpherical(r) for r in fhat.reshape(-1, fhat.shape[-1]))


# the degree of every coefficient
def _degrees(n):
  return np.floor(np.sqrt(np.arange(n))).astype(int)


# the projection onto the functions invariant under a group, in place: an element s acts by
# f(s^-1 v) with its parity eps_s on the degree, so degree l projects by the mean of
# eps_s^l D_l(r_s). On the even degrees that is the projection P_K onto the rotation parts K
# of all elements; on the odd ones the proper elements H count +1 and the improper ones -1,
# which sums to 2|H|/|G| P_H - P_K. Both are the held projections of H5
def projectSpherical(fhat, pg):
  L = math.isqrt(fhat.shape[-1]) - 1
  dtype = fhat.real.dtype
  K = invariantProjection(pg, L, dtype)
  H = None if pg.isProper else invariantProjection(pg, L, dtype, proper=True)
  c = 2 * pg.numProper() / pg.numSym()
  rows = fhat.reshape(-1, fhat.shape[-1]) if fhat.flags.c_contiguous else np.ascontiguousarray(fhat).reshape(-1, fhat.shape[-1])
  for l in range(L + 1):
    B = rows[:, l * l:(l + 1) ** 2]
    b = _projectBlock(B.copy(), None, K, l)
    B[...] = b if H is None or l % 2 == 0 else c * _projectBlock(B.copy(), None, H, l) - b
  if not fhat.flags.c_contiguous:
    fhat[...] = rows.reshape(fhat.shape)
  return fhat


class S2FunHarmonic:
  # NumPy leaves the arithmetic with a function to the function, not reading it as a sequence
  __array_ufunc__ = None

  def __init__(self, fhat, frame=None, *, real=None, antipodal=None, symmetrise=True, bandwidth=None, dtype=None,
               _internal=False):
    # coefficients given are MTEX's, those the port passes itself its own
    fromUser = not _internal
    # another representation converts itself to its harmonic form
    if not isinstance(fhat, S2FunHarmonic) and hasattr(fhat, "_harmonicS2"):
      fhat, bandwidth = fhat._harmonicS2(bandwidth), None
    if isinstance(fhat, S2FunHarmonic):
      f = fhat
      frame = f.frame if frame is None else frame
      real = f.isReal if real is None else real
      antipodal = f.antipodal if antipodal is None else antipodal
      fhat = f.truncate(bandwidth)._fhat if bandwidth is not None else f._fhat
      bandwidth, fromUser = None, False
    elif isinstance(fhat, S2Kernel):
      fhat = _kernelCoefficients(fhat, bandwidth)
      real, bandwidth, fromUser = True, None, False
    elif callable(fhat) and not isinstance(fhat, np.ndarray):
      fhat, real = _byQuadrature(fhat, frame, bandwidth, real)
      bandwidth, fromUser = None, False
    fhat = np.array(fhat, dtype=_complexType(np.result_type(np.asarray(fhat).dtype, np.float64) if dtype is None else dtype))
    fhat = fhat.reshape(1) if fhat.ndim == 0 else fhat
    # a count between two full degrees is padded with zeros to the next, as MATLAB does
    L = math.isqrt(fhat.shape[-1] - 1) if fhat.shape[-1] > 1 else 0
    if (L + 1) ** 2 != fhat.shape[-1]:
      fhat = _cut(fhat, L)
    if bandwidth is not None and bandwidth != L:
      fhat = _cut(fhat, bandwidth)
    if fromUser:
      fhat = mtexCoefficientsS2(fhat)
    real = _isReal(fhat) if real is None else bool(real)
    if real:
      realizeSpherical(fhat)
    # a function with a centrosymmetric frame is even whether or not it was asked to be
    antipodal = bool(antipodal) or bool(getattr(frame, "isLaue", False))
    if antipodal:
      evenize(fhat)
    self._fhat, self.frame, self.isReal, self.antipodal = fhat, frame, real, antipodal
    if symmetrise:
      self._fhat = self.symmetrise()._fhat

  def _rebuild(self, fhat, **changes):
    f = object.__new__(S2FunHarmonic)
    f._fhat, f.frame, f.isReal, f.antipodal = np.asarray(fhat), self.frame, self.isReal, self.antipodal
    for k, v in changes.items():
      setattr(f, k, v)
    return f

  # from weighted samples; the weights integrate to 4 pi, a quadrature grid brings its own
  @classmethod
  def quadrature(cls, nodes, values=None, frame=None, *, bandwidth=None, weights=None, real=None, antipodal=None):
    # quadrature(fun, cs): a callable takes the frame second
    if callable(nodes) and not isinstance(nodes, vector3d) and (values is None or isinstance(values, referenceFrame)):
      return cls(nodes, values if frame is None else frame, bandwidth=bandwidth, real=real, antipodal=bool(antipodal))
    values = np.asarray(values)
    if isinstance(nodes, quadratureS2Grid):
      L = nodes.bandwidth if bandwidth is None else int(bandwidth)
      frame = nodes.frame if frame is None else frame
      n, adjoint = len(nodes), lambda x: sphericalSeriesAdjoint(nodes, x, L, weights)
    else:
      L = quadratureBandwidth() if bandwidth is None else int(bandwidth)
      v = nodes if isinstance(nodes, vector3d) else vector3d(nodes)
      frame = v.frame if frame is None else frame
      antipodal = v.antipodal if antipodal is None else antipodal
      w = np.full(v.size, 4 * np.pi / v.size) if weights is None else np.asarray(weights).ravel()
      n, adjoint = v.size, lambda x: sphericalSeriesAdjoint(v, x, L, w)
    # values for a list of functions: the nodes first, then the shape of the list
    fhat = _perColumn(adjoint, values, n)
    return cls(fhat, frame, real=(values.dtype.kind != "c") if real is None else real, antipodal=bool(antipodal), _internal=True)

  approximate = quadrature

  # S2FunHarmonic.adjointNFSFT(v, values, bandwidth=32): the adjoint transform, the
  # coefficients sum_j w_j y_j conj(Y(v_j)) with unit weights unless given, a grid's own on a grid
  @classmethod
  def adjointNFSFT(cls, nodes, values, *, bandwidth=128, weights=None):
    values = np.asarray(values)
    if isinstance(nodes, quadratureS2Grid):
      L, w, v = nodes.bandwidth, nodes.weights if weights is None else weights, nodes
    else:
      v = (nodes if isinstance(nodes, vector3d) else vector3d(nodes)).flatten()
      L, w = int(bandwidth), np.ones(v.size) if weights is None else np.broadcast_to(np.asarray(weights, dtype=float).ravel(), (v.size,))
    fhat = _perColumn(lambda x: sphericalSeriesAdjoint(v, x, L, w), np.nan_to_num(values), len(v))
    return cls(fhat, v.frame, real=values.dtype.kind != "c", _internal=True)

  adjoint = adjointNFSFT

  # S2FunHarmonic.unimodal(v, halfwidth=w): MATLAB's other name of S2Fun.unimodal
  @staticmethod
  def unimodal(*args, **kwargs):
    from .s2funclass import S2Fun
    return S2Fun.unimodal(*args, **kwargs)

  @classmethod
  def interpolate(cls, nodes, values, **kwargs):
    from .density import interpolate
    return interpolate(nodes, values, **kwargs)

  @property
  def CS(self):
    return self.frame

  # the plotting convention of the frame, or the session's
  @property
  def how2plot(self):
    from ..geometry.plottingconvention import plottingConvention
    pC = getattr(self.frame, "how2plot", None)
    return plottingConvention(pC if pC is not None else plottingConvention.default)

  # the coefficients in MTEX's basis, as MATLAB's sF.fhat reads and assigns them
  @property
  def fhat(self):
    return mtexCoefficientsS2(self._fhat)

  @fhat.setter
  def fhat(self, value):
    self._fhat = mtexCoefficientsS2(np.asarray(value, dtype=self._fhat.dtype))

  @property
  def bandwidth(self):
    return math.isqrt(self._fhat.shape[-1]) - 1

  # sF.bandwidth = L cuts or pads the coefficients, as MATLAB's assignment does
  @bandwidth.setter
  def bandwidth(self, L):
    self._fhat = _cut(self._fhat, int(L))

  # the shape of a list of functions, () for one
  @property
  def shape(self):
    return self._fhat.shape[:-1]

  @property
  def size(self):
    return int(np.prod(self.shape))

  def __len__(self):
    return self.size

  def __getitem__(self, idx):
    if not self.shape:
      raise TypeError("a single function has no elements to index")
    fhat = self._fhat[idx]
    return self._rebuild(fhat if fhat.ndim else fhat.reshape(1))

  def reshape(self, *shape):
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return self._rebuild(self._fhat.reshape(tuple(shape) + (self._fhat.shape[-1],)))

  # the list transposed, a row of n functions becoming a column, and MATLAB's ' as sF.conj().T
  @property
  def T(self):
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
    f0 = fs[0]
    return f0._rebuild(np.concatenate(parts, axis=axis), isReal=all(f.isReal for f in fs), antipodal=all(f.antipodal for f in fs))

  @property
  def power(self):
    return np.stack([np.linalg.norm(self._fhat[..., l * l:(l + 1) ** 2], axis=-1) for l in range(self.bandwidth + 1)], axis=-1)

  # the coefficients up to degree L; without L the trailing degrees whose power, damped by
  # (l+1)^2, stays below 1e-8 of the largest are dropped, MATLAB's truncate
  def truncate(self, L=None):
    if L is None:
      l = _degrees(self._fhat.shape[-1])
      w = np.abs(self._fhat / (l + 1) ** 2) ** 2
      per = np.sqrt(np.stack([np.sum(w[..., l == k], axis=-1) for k in range(self.bandwidth + 1)], axis=-1))
      per = per.reshape(-1, self.bandwidth + 1).max(axis=0)
      above = np.flatnonzero(per > per.max() * 1e-8)
      L = int(above[-1]) if above.size else 0
    return self if L == self.bandwidth else self._rebuild(_cut(self._fhat, L))

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
    v = x if isinstance(x, quadratureS2Grid) else self._vectorsFor(x).normalize()
    if not self.shape:
      return sphericalSeries(self._fhat, v, real=self.isReal)
    # a list of functions: the values at the nodes first, then the shape of the list
    vals = [sphericalSeries(r, v, real=self.isReal) for r in self._fhat.reshape(-1, self._fhat.shape[-1])]
    return np.stack(vals, axis=-1).reshape(np.shape(vals[0]) + self.shape)

  # ------------------------------------------------------------------ symmetry

  # the projection onto the functions invariant under the group of the frame, or under a
  # list of rotations forming a group the frame does not name; an axis valued function is even
  def symmetrise(self, rots=None):
    # symmetrise(sF, frame): the function averaged over the group of the frame, in that frame
    if isinstance(rots, referenceFrame):
      return S2FunHarmonic(self._fhat, rots, real=self.isReal, antipodal=self.antipodal, _internal=True)
    L = self.bandwidth
    fhat = self._fhat.copy()
    if rots is not None:
      rots = rots.flatten()
      D = _WignerD(rots, bandwidth=L, dtype=self._fhat.real.dtype)
      for l in range(L + 1):
        fhat[..., l * l:(l + 1) ** 2] = fhat[..., l * l:(l + 1) ** 2] @ np.mean(D[l], axis=0).T
    elif self.frame is not None:
      projectSpherical(fhat, self.frame.sym)
    if self.antipodal:
      evenize(fhat)
    return self._rebuild(fhat)

  # the even part, f(v) + f(-v) over two
  def even(self):
    return self._rebuild(evenize(self._fhat.copy()), antipodal=True)

  # ------------------------------------------------------------------ the algebra

  def conj(self):
    L = self.bandwidth
    out = np.empty_like(self._fhat)
    for l in range(L + 1):
      m = np.arange(-l, l + 1)
      out[..., l * l:(l + 1) ** 2] = np.where(m % 2, -1.0, 1.0) * np.conj(self._fhat[..., l * l:(l + 1) ** 2][..., ::-1])
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
      return S2FunHarmonic(self.truncate(L)._fhat + g.truncate(L)._fhat, self.frame, real=self.isReal and g.isReal,
                           antipodal=self.antipodal and g.antipodal, symmetrise=False, _internal=True)
    if isinstance(g, (int, float, complex, np.number)):
      fhat = self._fhat.copy()
      fhat[..., 0] += g * math.sqrt(4 * math.pi)
      return S2FunHarmonic(fhat, self.frame, real=self.isReal and np.isreal(g), antipodal=self.antipodal, symmetrise=False, _internal=True)
    return NotImplemented

  __radd__ = __add__

  def __neg__(self):
    return self._rebuild(-self._fhat)

  def __sub__(self, g):
    return self + (-g)

  def __rsub__(self, a):
    return (-self) + a

  def __mul__(self, a):
    if isinstance(a, S2FunHarmonic):
      from .funops import mapFun
      return mapFun(np.multiply, self, a)
    if isinstance(a, (int, float, complex, np.number)):
      return S2FunHarmonic(a * self._fhat, self.frame, real=self.isReal and np.isreal(a), antipodal=self.antipodal, symmetrise=False, _internal=True)
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

  # |f| as a handle, evaluated when asked, as MATLAB's abs of a spherical function
  def __abs__(self):
    from .handles import S2FunHandle
    return S2FunHandle(lambda v: np.abs(self.eval(v)), self.frame, bandwidth=self.bandwidth, antipodal=self.antipodal,
                       symmetrise=False)

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

  # the mean over the sphere, int f dS / 4 pi, one per function; with an axis the mean
  # function along that axis of a list
  def mean(self, axis=None):
    if axis is not None:
      return self._rebuild(np.mean(self._fhat, axis=_leadingAxis(axis, self.shape)))
    x = self._fhat[..., 0] / math.sqrt(4 * math.pi)
    x = x.real if self.isReal else x
    return x[()] if x.ndim == 0 else x

  # the integral over the sphere, int f dS, S2Fun/sum; with an axis the sum of the functions
  # along that axis of a list
  def sum(self, axis=None):
    if axis is not None:
      return self._rebuild(np.sum(self._fhat, axis=_leadingAxis(axis, self.shape)))
    return 4 * math.pi * self.mean()

  # the L2 norm for the normalised surface measure, or the Sobolev norm, one per function
  def norm(self, s=None):
    if s is None:
      n = np.linalg.norm(self._fhat, axis=-1) / math.sqrt(4 * math.pi)
    else:
      n = np.sqrt(sum((2 * l + 1) ** (2 * s) * np.sum(np.abs(self._fhat[..., l * l:(l + 1) ** 2]) ** 2, axis=-1)
                      for l in range(self.bandwidth + 1))) / math.sqrt(4 * math.pi)
    return float(n) if np.ndim(n) == 0 else n

  def dot(self, g):
    self._sameFrames(g)
    L = max(self.bandwidth, g.bandwidth)
    x = np.sum(np.conj(self.truncate(L)._fhat) * g.truncate(L)._fhat, axis=-1) / (4 * math.pi)
    x = x.real if self.isReal and g.isReal else x
    return x[()] if np.ndim(x) == 0 else x

  # the pointwise minimum or maximum of the functions of a list along an axis, MATLAB's
  # min(sF, [], d), by quadrature; min(sF, axis=d) and max reach it through characteristics
  def _reduce(self, op, axis):
    ax = _leadingAxis(axis, self.shape)
    fun = lambda v: op(self.eval(v), axis=np.ndim(v.data) - 1 + ax)
    return S2FunHarmonic(fun, self.frame, bandwidth=self.bandwidth, real=self.isReal, antipodal=self.antipodal)

  def isclose(self, g, rtol=1e-8, atol=0.0):
    L = max(self.bandwidth, g.bandwidth)
    a, b = self.truncate(L)._fhat, g.truncate(L)._fhat
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

  def radon(self, delta=None):
    from .funops import radon
    return radon(self, delta)

  def invRadon(self):
    from .funops import invRadon
    return invRadon(self)

  def __repr__(self):
    info = "" if self.frame is None else f" ({self.frame.char()})"
    lines = [f"S2FunHarmonic{info}"]
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


def _cut(fhat, L):
  out = np.zeros(fhat.shape[:-1] + ((L + 1) ** 2,), dtype=fhat.dtype)
  n = min(out.shape[-1], fhat.shape[-1])
  out[..., :n] = fhat[..., :n]
  return out


# the axis of the list an axis argument names, negative ones counted from the end
def _leadingAxis(axis, shape):
  if not shape:
    raise ValueError("a single function has no axis to reduce")
  return int(axis) % len(shape)


# the adjoint transform column by column of values shaped (nodes, list...), the list shape
# in front of the coefficients
def _perColumn(adjoint, values, n):
  values = np.asarray(values)
  if values.size == n:
    return adjoint(values.ravel())
  cols = values.reshape(n, -1)
  return np.stack([adjoint(cols[:, k]) for k in range(cols.shape[1])]).reshape(values.shape[1:] + (-1,))


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
  L = quadratureBandwidth() if bandwidth is None else int(bandwidth)
  g = quadratureS2Grid(L)
  v = g.nodes()
  if frame is not None:
    v = vector3d(v, frame=frame)
  values = np.asarray(fun(v))
  return _perColumn(lambda x: sphericalSeriesAdjoint(g, x, L), values, len(g)), (values.dtype.kind != "c") if real is None else real


__all__ = ["S2FunHarmonic"]
