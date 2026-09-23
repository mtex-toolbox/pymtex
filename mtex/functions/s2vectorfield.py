# vector fields on the sphere: the harmonic field, the surface gradient, the handle field
# and the local polar basis (MATLAB @S2VectorFieldHarmonic, @S2VectorFieldHandle;
# docs/plans/functions.md F7)
#
#   vF = S2VectorFieldHarmonic(fhat, cs), S2VectorFieldHarmonic(fun, bandwidth=), S2VectorFieldHarmonic(v, values)
#   vF(v)  the vectors at the directions;  grad(sF), grad(sF, v)
#   dot(vF, x), cross(vF, x), f * vF, abs(vF), normalize(vF), rotate(vF, rot), mean, norm
#
# The surface gradient is grad f = (L f) x xi with L_i f_l = i J_i f_l; the frame's group
# acts by v(s xi) = s v(xi) for every element including the improper ones.
from __future__ import annotations

import math

import numpy as np

from ..geometry.frames import referenceFrame
from ..geometry.orientation import orientation
from ..geometry.quaternion import rotation
from ..geometry.session import SESSION, getMTEXpref
from ..geometry.tangent import spinTensor
from ..geometry.vector3d import vector3d
from .funops import _rotatedFrame
from .harmonics import mtexCoefficientsS2, _WignerD
from .quadrature import quadratureS2Grid
from .s2fun import S2FunHarmonic, realizeSpherical
from .so3vectorfield import _projectSphericalRows, equivariantProjection
from .transforms import DIRECT_SUM_NODES, S2FourierPlan, sphericalPlan, sphericalSeries, sphericalSeriesAdjoint


def _seriesColumns(fhat, v):
  L = math.isqrt(fhat.shape[0]) - 1
  d = vector3d(v.data.reshape(-1, 3))
  if d.size < DIRECT_SUM_NODES or L == 0:
    return np.stack([np.real(sphericalSeries(fhat[:, i], d, real=True)) for i in range(fhat.shape[1])], axis=-1)
  p = S2FourierPlan(d, L, real=True)
  return np.stack([p(fhat[:, i]) for i in range(fhat.shape[1])], axis=-1)


def _adjointColumns(nodes, values, L, w):
  if isinstance(nodes, quadratureS2Grid):
    p = sphericalPlan(nodes, L, False)
    return np.stack([p.adjoint(values[:, i], w) for i in range(values.shape[1])], axis=-1)
  d = vector3d(nodes.data.reshape(-1, 3))
  if d.size < DIRECT_SUM_NODES:
    return np.stack([sphericalSeriesAdjoint(d, values[:, i], L, w) for i in range(values.shape[1])], axis=-1)
  p = S2FourierPlan(d, L)
  return np.stack([p.adjoint(values[:, i], w) for i in range(values.shape[1])], axis=-1)


def _values(x):
  return (x.data if isinstance(x, vector3d) else np.asarray(x, dtype=float)).reshape(-1, 3)


# bare vectors, or framed ones in the frame of the field or one sharing its basis
def _vectorsFor(f, x):
  v = x if isinstance(x, vector3d) else vector3d(x)
  if f.frame is not None and v.frame is not None and v.frame is not f.frame:
    try:
      SESSION.joinFrames(v.frame, f.frame)
    except ValueError:
      raise ValueError("the directions live in another frame than the field") from None
  return v


class S2VectorField:

  @property
  def CS(self):
    return self.frame

  def __call__(self, x):
    return self.eval(x)

  def _framed(self, v):
    return v if self.frame is None else vector3d(v.data, frame=self.frame)

  def _sameFrames(self, w):
    if w.frame is not self.frame:
      raise ValueError("the fields live in different frames")

  def dot(self, x):
    if isinstance(x, S2VectorField):
      return mapField(np.multiply, self, x, reduce=True, invariant=True)
    return S2VectorFieldHarmonic(self).dot(x)

  def cross(self, w):
    if isinstance(w, S2VectorField):
      return mapField(np.cross, self, w)
    return S2VectorFieldHarmonic(self).cross(w)

  def abs(self):
    return mapField(lambda a: np.linalg.norm(a, axis=-1), self, bandwidth=max(2 * self.bandwidth, 32), invariant=True)

  def __abs__(self):
    return self.abs()

  def normalize(self):
    return mapField(lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True), self, bandwidth=max(2 * self.bandwidth, 32))

  def angle(self, w):
    return mapField(lambda a, b: np.arccos(np.clip(np.sum(a * b, axis=-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1), -1, 1)), self, w,
                    bandwidth=max(2 * max(self.bandwidth, w.bandwidth), 32), invariant=True)

  # the length at every direction, a function on the sphere, as MATLAB's norm
  def norm(self):
    return self.abs()

  def mean(self):
    return S2VectorFieldHarmonic(self).mean()

  def __neg__(self):
    return -1 * self

  def __sub__(self, w):
    return self + (-1 * w)

  def __radd__(self, w):
    return self + w

  def __truediv__(self, a):
    return (1 / a) * self

  def __mul__(self, a):
    if isinstance(a, (int, float, np.number)):
      return self._scaled(a)
    return mapField(lambda x, s: s[:, None] * x, self, a)

  __rmul__ = __mul__


class S2VectorFieldHarmonic(S2VectorField):

  def __init__(self, fhat, *args, bandwidth=None, weights=None, symmetrise=True):
    frames = [a for a in args if isinstance(a, referenceFrame)]
    frame = frames[0] if frames else None
    if isinstance(fhat, S2VectorField):
      v = fhat
      frame = v.frame
      if isinstance(v, S2VectorFieldHarmonic):
        fhat, symmetrise = (v.truncate(bandwidth)._fhat if bandwidth is not None else v._fhat), False
      else:
        fhat, symmetrise = v._harmonicField(bandwidth)._fhat, False
    elif isinstance(fhat, vector3d) and args and not isinstance(args[0], referenceFrame):
      pts, values = fhat, args[0]
      frame = pts.frame if frame is None else frame
      L = getMTEXpref("maxS2Bandwidth") if bandwidth is None else int(bandwidth)
      d = vector3d(pts.data.reshape(-1, 3)).normalize()
      w = np.full(d.size, 4 * np.pi / d.size) if weights is None else np.asarray(weights, dtype=float).ravel()
      fhat = _adjointColumns(d, _values(values), L, w)
    elif isinstance(fhat, (list, tuple)) and fhat and isinstance(fhat[0], S2FunHarmonic):
      # the x, y and z components, or x and y with z zero
      if len(fhat) not in (2, 3):
        raise ValueError("two or three component functions are needed")
      if len(fhat) == 2:
        fhat = [*fhat, 0 * fhat[0]]
      L = max(f.bandwidth for f in fhat)
      fhat = np.stack([f.truncate(L)._fhat for f in fhat], axis=-1)
    elif callable(fhat) and not isinstance(fhat, np.ndarray):
      L = getMTEXpref("maxS2Bandwidth") if bandwidth is None else int(bandwidth)
      g = quadratureS2Grid(L)
      v = g.nodes()
      values = _values(fhat(v if frame is None else vector3d(v, frame=frame)))
      fhat = _adjointColumns(g, values, L, g.weights)
    fhat = np.array(fhat, dtype=complex)
    if fhat.ndim != 2 or fhat.shape[1] != 3:
      raise ValueError("a vector field has three components")
    if math.isqrt(fhat.shape[0]) ** 2 != fhat.shape[0]:
      raise ValueError("a coefficient matrix has (L+1)^2 rows for its bandwidth L")
    for i in range(3):
      realizeSpherical(fhat[:, i])
    self._fhat, self.frame = fhat, frame
    if symmetrise:
      self._fhat = self.symmetrise()._fhat

  def _rebuild(self, fhat, **changes):
    v = object.__new__(S2VectorFieldHarmonic)
    v._fhat, v.frame = np.asarray(fhat), self.frame
    for k, x in changes.items():
      setattr(v, k, x)
    return v

  # the coefficients of the components in MTEX's basis, one column per component
  @property
  def fhat(self):
    return mtexCoefficientsS2(np.moveaxis(self._fhat, 0, -1)).swapaxes(-1, 0) if self._fhat.ndim > 1 else mtexCoefficientsS2(self._fhat)

  @property
  def bandwidth(self):
    return math.isqrt(self._fhat.shape[0]) - 1

  # the field of the outer normals, v at v, exact at bandwidth one
  @classmethod
  def normal(cls):
    return cls(lambda v: v, bandwidth=2)

  def truncate(self, L):
    if L == self.bandwidth:
      return self
    fhat = np.zeros(((L + 1) ** 2, 3), dtype=self._fhat.dtype)
    n = min(len(fhat), len(self._fhat))
    fhat[:n] = self._fhat[:n]
    return self._rebuild(fhat)

  # S2VectorFieldHarmonic.quadrature(f, bandwidth=L), from a callable or from nodes and
  # values with their weights, MATLAB's name for the constructor
  @classmethod
  def quadrature(cls, f, *args, bandwidth=None, weights=None):
    return cls(f, *args, bandwidth=bandwidth, weights=weights)

  # the least squares fit to vectors at directions
  @classmethod
  def interpolate(cls, nodes, values, **kwargs):
    return interpolateField(nodes, values, **kwargs)

  # the Cartesian components as functions
  @property
  def x(self):
    return self.component(0)

  @property
  def y(self):
    return self.component(1)

  @property
  def z(self):
    return self.component(2)

  # one component as a function without the group
  def component(self, i):
    return S2FunHarmonic(self._fhat[:, i], None if self.frame is None else self.frame.stripSym(), real=True, symmetrise=False, _internal=True)

  def eval(self, x):
    v = _vectorsFor(self, x).normalize()
    return vector3d(_seriesColumns(self._fhat, v).reshape(*v.shape, 3))

  # the projection onto the equivariant fields of the frame, v(s xi) = s v(xi) for the
  # elements of the group including the improper ones
  def symmetrise(self):
    if self.frame is None or self.frame.sym.numSym() == 1:
      return self
    L = self.bandwidth
    els = self.frame.sym.rot.flatten()
    actions = [els[i].matrix() for i in range(els.size)]
    Ds = []
    for i in range(els.size):
      D = _WignerD(els[i].rot, bandwidth=L)
      Ds.append([(-1 if els[i].isImproper and l % 2 else 1) * D[l] for l in range(L + 1)])
    return self._rebuild(_projectSphericalRows(self._fhat.copy(), equivariantProjection(actions, Ds)))

  def __add__(self, w):
    if isinstance(w, vector3d):
      fhat = self._fhat.copy()
      fhat[0] += math.sqrt(4 * math.pi) * w.data.reshape(3)
      return self._rebuild(fhat)
    if not isinstance(w, S2VectorField):
      return NotImplemented
    self._sameFrames(w)
    w = S2VectorFieldHarmonic(w)
    L = max(self.bandwidth, w.bandwidth)
    return self._rebuild(self.truncate(L)._fhat + w.truncate(L)._fhat)

  def _scaled(self, a):
    return self._rebuild(a * self._fhat)

  def mean(self):
    return vector3d(np.real(self._fhat[0]) / math.sqrt(4 * math.pi))

  def isclose(self, w, rtol=1e-8):
    w = S2VectorFieldHarmonic(w)
    L = max(self.bandwidth, w.bandwidth)
    a, b = self.truncate(L)._fhat, w.truncate(L)._fhat
    return bool(np.linalg.norm(a - b) <= rtol * np.linalg.norm(b))

  # the inner product with a constant vector, exact, a function without the group
  def dot(self, x):
    if isinstance(x, S2VectorField):
      return S2VectorField.dot(self, x)
    x = x if isinstance(x, vector3d) else vector3d(x)
    if x.frame is not None and self.frame is not None and x.frame is not self.frame:
      raise ValueError("the vector lives in another frame than the field")
    return S2FunHarmonic(self._fhat @ x.data.reshape(3), None if self.frame is None else self.frame.stripSym(), real=True, symmetrise=False, _internal=True)

  # F x x = -[x]_x F, on the rows of the coefficients F [x]_x
  def cross(self, x):
    if isinstance(x, S2VectorField):
      return S2VectorField.cross(self, x)
    x = x if isinstance(x, vector3d) else vector3d(x)
    return self._rebuild(self._fhat @ spinTensor(x.data.reshape(3)))

  # v'(xi) = r v(r^-1 xi): the components rotate and mix
  def rotate(self, r):
    if isinstance(r, orientation):
      if self.frame is not None and r.CS is not self.frame:
        raise ValueError("the orientation maps out of the frame of the field")
      return self.rotate(rotation(r))._rebuild(None, frame=r.SS) if False else _reframe(self.rotate(rotation(r)), r.SS)
    rot = rotation(r.data.reshape(4))
    L = self.bandwidth
    D = _WignerD(rot, bandwidth=L)
    fhat = self._fhat.copy()
    for l in range(L + 1):
      fhat[l * l:(l + 1) ** 2] = D[l] @ self._fhat[l * l:(l + 1) ** 2]
    fhat = fhat @ rot.matrix().T
    return self._rebuild(fhat, frame=None if self.frame is None else _rotatedFrame(self.frame, rot))

  # the field of the antipode, v(-xi)
  def antipode(self):
    l = np.floor(np.sqrt(np.arange(len(self._fhat)))).astype(int)
    return self._rebuild(np.where(l % 2, -1.0, 1.0)[:, None] * self._fhat)

  def __repr__(self):
    info = "" if self.frame is None else f" ({self.frame.char()})"
    return f"S2VectorFieldHarmonic{info}\n  bandwidth: {self.bandwidth}"


def _reframe(v, frame):
  return v._rebuild(v._fhat, frame=frame)


# ------------------------------------------------------------------ the surface gradient


# the ladder coefficients of the angular derivatives over a coefficient vector: at the
# position of (l, m) the factor of f(l, m+1) and of f(l, m-1), zero at the ends of a degree
_LADDERS = {}


def _ladders(L):
  if L not in _LADDERS:
    l = np.repeat(np.arange(L + 1), 2 * np.arange(L + 1) + 1)
    m = np.concatenate([np.arange(-k, k + 1) for k in range(L + 1)])
    up = np.where(m < l, np.sqrt(np.maximum(l * (l + 1) - m * (m + 1), 0)) / 2, 0.0)
    down = np.where(m > -l, np.sqrt(np.maximum(l * (l + 1) - (m - 1) * m, 0)) / 2, 0.0)
    _LADDERS[L] = (m, up, down)
  return _LADDERS[L]


# L_i f, the derivative under the rotation about e_i, on the coefficients: the ladder
# operators as shifts of the coefficient vector, the degrees never mixing
def angularDerivatives(fhat):
  L = math.isqrt(len(fhat)) - 1
  m, up, down = _ladders(L)
  f = np.asarray(fhat)
  next_ = np.zeros_like(f)
  prev = np.zeros_like(f)
  next_[:-1] = f[1:]
  prev[1:] = f[:-1]
  out = np.empty((len(fhat), 3), dtype=np.result_type(f.dtype, 1j))
  out[:, 0] = 1j * (down * prev + up * next_)
  out[:, 1] = down * prev - up * next_
  out[:, 2] = 1j * m * f
  return out


# the surface gradient (L f) x xi, at directions and as a field of the next bandwidth
def grad(f, x=None):
  if not f.isReal:
    raise ValueError("the gradient of a complex valued function is not a real vector field")
  if x is None:
    if f.bandwidth == 0:
      return S2VectorFieldHarmonic(np.zeros((1, 3), dtype=complex), f.frame, symmetrise=False)
    return S2VectorFieldHarmonic(lambda v: grad(f, vector3d(v.data)), f.frame, bandwidth=f.bandwidth + 1, symmetrise=False)
  v = f._vectorsFor(x).normalize()
  Lf = _seriesColumns(angularDerivatives(f._fhat), v)
  return vector3d(np.cross(Lf, v.data.reshape(-1, 3)).reshape(*v.shape, 3))


# ------------------------------------------------------------------ pointwise operations by quadrature


def mapField(op, v, *args, bandwidth=None, invariant=False, reduce=False):
  others = [a for a in args if not isinstance(a, (int, float, np.number))]
  L = min(v.bandwidth + sum(a.bandwidth for a in others), getMTEXpref("maxS2Bandwidth")) if bandwidth is None else int(bandwidth)
  g = quadratureS2Grid(L)
  xs = v._framed(g.nodes())

  def evaluate(a):
    if isinstance(a, S2VectorField):
      v._sameFrames(a)
      return _values(a(xs))
    if isinstance(a, (int, float, np.number)):
      return a
    if a.frame is not v.frame:
      raise ValueError("the fields live in different frames")
    return np.real(a(xs)).ravel()

  values = op(evaluate(v), *(evaluate(a) for a in args))
  if reduce:
    values = np.sum(values, axis=-1)
  if values.ndim == 2:
    return S2VectorFieldHarmonic(_adjointColumns(g, values, L, g.weights), v.frame, symmetrise=False)
  fr = v.frame if invariant or v.frame is None else v.frame.stripSym()
  return S2FunHarmonic(sphericalSeriesAdjoint(g, values, L), fr, real=True, symmetrise=False, _internal=True)


# ------------------------------------------------------------------ the handle and the polar basis


class S2VectorFieldHandle(S2VectorField):

  def __init__(self, fun, frame=None, *, bandwidth=None):
    if isinstance(fun, S2VectorField):
      v = fun
      fun, frame, bandwidth = (lambda x: v(x)), v.frame, v.bandwidth
    self.fun, self.frame = fun, frame
    self.bandwidth = getMTEXpref("maxS2Bandwidth") if bandwidth is None else int(bandwidth)

  def eval(self, x):
    v = _vectorsFor(self, x).normalize()
    return vector3d(_values(self.fun(self._framed(v))).reshape(*v.shape, 3))

  def _harmonicField(self, bandwidth):
    return S2VectorFieldHarmonic(lambda x: self(x), self.frame, bandwidth=self.bandwidth if bandwidth is None else bandwidth)

  @property
  def fhat(self):
    return S2VectorFieldHarmonic(self)._fhat

  def _scaled(self, a):
    return S2VectorFieldHandle(lambda x: a * _values(self.fun(x)), self.frame, bandwidth=self.bandwidth)

  def __add__(self, w):
    return S2VectorFieldHarmonic(self) + w

  def rotate(self, r):
    rot = rotation(r.data.reshape(4))
    return S2VectorFieldHandle(lambda x: (rot * self(rot.inv() * vector3d(x.data))).data, None if self.frame is None else _rotatedFrame(self.frame, rot), bandwidth=self.bandwidth)

  def __repr__(self):
    info = "" if self.frame is None else f" ({self.frame.char()})"
    return f"S2VectorFieldHandle{info}\n  bandwidth hint: {self.bandwidth}"


# the unit vectors of the polar coordinates at a direction: along the polar angle and
# along the azimuth
def thetaVector(v):
  v = v if isinstance(v, vector3d) else vector3d(v)
  t, r = v.theta, v.rho
  return vector3d(np.cos(r) * np.cos(t), np.sin(r) * np.cos(t), -np.sin(t), frame=v.frame)


def rhoVector(v):
  v = v if isinstance(v, vector3d) else vector3d(v)
  r = v.rho
  return vector3d(-np.sin(r), np.cos(r), np.zeros_like(r), frame=v.frame)


# the vectors at directions by least squares componentwise
def interpolateField(v, values, **kwargs):
  from .density import interpolate
  d = vector3d(v.data.reshape(-1, 3), frame=v.frame).normalize()
  vals = _values(values)
  fs = [interpolate(vector3d(d.data), vals[:, i], **kwargs) for i in range(3)]
  return S2VectorFieldHarmonic(fs, v.frame)


__all__ = ["S2VectorFieldHarmonic", "S2VectorFieldHandle", "thetaVector", "rhoVector", "angularDerivatives"]
