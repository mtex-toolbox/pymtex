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

from typing import TYPE_CHECKING
import math

import numpy as np

from ..geometry.frames import referenceFrame
from ..geometry.orientation import orientation
from ..geometry.quaternion import rotation
from ..geometry.session import SESSION, getMTEXpref
from ..geometry.tangent import spinTensor
from ..geometry.vector3d import vector3d, asVector3d
from .funops import _rotatedFrame
from .harmonics import _s2Degrees, mtexCoefficientsS2, _WignerD
from .quadrature import quadratureS2Grid
from .s2fun import S2FunHarmonic, realizeSpherical
from .so3vectorfield import _projectSphericalRows, equivariantProjection
from .transforms import DIRECT_SUM_NODES, S2FourierPlan, sphericalPlan, sphericalSeries, sphericalSeriesAdjoint
from ..geometry.display import aligned, frameLabel


# the values of every column at the directions, one plan for all
def _seriesColumns(fhat, v):
  L = math.isqrt(fhat.shape[0]) - 1
  d = vector3d(v.data.reshape(-1, 3))
  if d.size < DIRECT_SUM_NODES or L == 0:
    def series(c):
      return np.real(sphericalSeries(c, d, real=True))
  else:
    series = S2FourierPlan(d, L, real=True)
  return np.stack([series(fhat[:, i]) for i in range(fhat.shape[1])], axis=-1)


# the adjoint of every column of values, one plan for all; on a quadrature grid its FFT
def _adjointColumns(nodes, values, L, w):
  if isinstance(nodes, quadratureS2Grid):
    adjoint = sphericalPlan(nodes, L, False).adjoint
  elif nodes.size < DIRECT_SUM_NODES:
    d = vector3d(nodes.data.reshape(-1, 3))

    def adjoint(y, w):
      return sphericalSeriesAdjoint(d, y, L, w)
  else:
    adjoint = S2FourierPlan(vector3d(nodes.data.reshape(-1, 3)), L).adjoint
  return np.stack([adjoint(values[:, i], w) for i in range(values.shape[1])], axis=-1)


# the Wigner matrices D_0 .. D_L of the rotation part of every element of a group, the odd
# degrees of an improper element negated: how it acts on a component from the left
def _parityWignerD(els, L):
  out = []
  for i in range(els.size):
    D = _WignerD(els[i].rot, bandwidth=L)
    out.append([(-1 if els[i].isImproper and l % 2 else 1) * D[l] for l in range(L + 1)])
  return out


def _values(x):
  return (x.data if isinstance(x, vector3d) else np.asarray(x, dtype=float)).reshape(-1, 3)


# bare vectors, or framed ones in the frame of the field or one sharing its basis
def _vectorsFor(f, x):
  v = asVector3d(x)
  if f.frame is not None and v.frame is not None and v.frame is not f.frame:
    try:
      SESSION.joinFrames(v.frame, f.frame)
    except ValueError:
      raise ValueError("the directions live in another frame than the field") from None
  return v


class S2VectorField:
  """the base of the vector fields on the sphere: evaluation by calling, the pointwise operations by
  quadrature, the arithmetic with numbers, functions and other fields"""

  @property
  def CS(self):
    """the frame of the field, MATLAB's vF.CS"""
    return self.frame

  def __call__(self, x):
    return self.eval(x)

  def _framed(self, v):
    return v if self.frame is None else vector3d(v.data, frame=self.frame)

  def _sameFrames(self, w):
    if w.frame is not self.frame:
      raise ValueError("the fields live in different frames")

  def dot(self, x):
    """the inner product with another field, pointwise a function on the sphere, or with a
    constant vector

      dot(vF, wF)
      dot(vF, vector3d.Z)"""
    if isinstance(x, S2VectorField):
      return mapField(np.multiply, self, x, reduce=True, invariant=True)
    return S2VectorFieldHarmonic(self).dot(x)

  def cross(self, w):
    """the cross product with another field, pointwise a field, or with a constant vector"""
    if isinstance(w, S2VectorField):
      return mapField(np.cross, self, w)
    return S2VectorFieldHarmonic(self).cross(w)

  def abs(self):
    """the length of the field at every direction, a function on the sphere"""
    return mapField(lambda a: np.linalg.norm(a, axis=-1), self, bandwidth=max(2 * self.bandwidth, 32), invariant=True)

  def __abs__(self):
    return self.abs()

  def normalize(self):
    """the field scaled to unit length at every direction"""
    return mapField(lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True), self, bandwidth=max(2 * self.bandwidth, 32))

  def angle(self, w):
    """the angle between two fields at every direction, or between the field and a constant
    vector, a function on the sphere

      f = angle(sVF1, sVF2)
      f = angle(sVF, vector3d.X)"""
    def between(a, b):
      return np.arccos(np.clip(np.sum(a * b, axis=-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1), -1, 1))
    if isinstance(w, S2VectorField):
      return mapField(between, self, w, bandwidth=max(2 * max(self.bandwidth, w.bandwidth), 32), invariant=True)
    x = asVector3d(w)
    if x.frame is not None and self.frame is not None and x.frame is not self.frame:
      raise ValueError("the vector lives in another frame than the field")
    b = x.data.reshape(3)
    return mapField(lambda a: between(a, b), self, bandwidth=max(2 * self.bandwidth, 32), invariant=True)

  def norm(self):
    """the length at every direction, a function on the sphere, as MATLAB's norm

      sF = norm(sVF)"""
    return self.abs()

  def mean(self):
    """the mean vector of the field over the sphere"""
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
    return mapField(np.multiply, self, a)

  __rmul__ = __mul__

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from ..plotting.sphericalplot import plot, quiver
    from ..plotting.spaceplot import plot3d


class S2VectorFieldHarmonic(S2VectorField):
  """a vector field on the sphere by the spherical harmonic expansions of its three Cartesian
  components

    vF = S2VectorFieldHarmonic(lambda v: v.cross(vector3d.Z), bandwidth=16)  # a callable of the directions
    vF = S2VectorFieldHarmonic(v, w, weights=wq)                    # vectors at directions, by quadrature
    vF = S2VectorFieldHarmonic([sFx, sFy, sFz])                     # the component functions
    vF = S2VectorFieldHarmonic(fhat, cs)                            # the coefficients, (L+1)^2 x 3
    vF = S2VectorFieldHarmonic(sF.grad())                           # the expansion of another field
    vF = S2VectorFieldHarmonic.quadrature(lambda v: v.cross(vector3d.Z))  # MATLAB's form

  A callable is evaluated on a quadrature grid of the bandwidth and expanded; vectors w at
  directions v are expanded with the quadrature weights given, equal ones by default, which
  suits directions spread evenly; two component functions leave z zero. The field is
  projected onto the fields equivariant under the group of the frame, which a crystal frame
  asks for, unless symmetrise=False. vF.x, vF.y, vF.z are the components, vF.bandwidth the
  largest degree.

  Input
    fun - callable taking a vector3d and returning vector3d, or an S2VectorField
    v - vector3d, the directions of given vectors
    w - vector3d, the vectors at them
    sFx, sFy, sFz - S2FunHarmonic, the components
    fhat - complex array of (L+1)^2 x 3 coefficients
    cs - crystalFrame or specimenFrame of the directions
  Output
    vF - S2VectorFieldHarmonic
  Keywords
    bandwidth - the largest degree of the expansion, by default the preference
        'defaultS2Bandwidth'; truncates a field given
    weights - the quadrature weights of the directions v
    symmetrise - project onto the fields equivariant under the group of the frame
  See also
    S2VectorField, S2FunHarmonic, S2AxisFieldHarmonic, S2FunVectorField"""

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
      L = getMTEXpref("defaultS2Bandwidth") if bandwidth is None else int(bandwidth)
      d = vector3d(pts.data.reshape(-1, 3)).normalize()
      w = np.full(d.size, 4 * np.pi / max(d.size, 1)) if weights is None else np.asarray(weights, dtype=float).ravel()
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
      L = getMTEXpref("defaultS2Bandwidth") if bandwidth is None else int(bandwidth)
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

  @property
  def fhat(self):
    """the coefficients of the components in MTEX's basis, one column per component"""
    return mtexCoefficientsS2(self._fhat.T).T

  @property
  def bandwidth(self):
    """the harmonic degree up to which the components are expanded"""
    return math.isqrt(self._fhat.shape[0]) - 1

  @classmethod
  def normal(cls):
    """the field of the outer normals, v at v, exact at bandwidth one"""
    return cls(lambda v: v, bandwidth=2)

  def truncate(self, L):
    """the field cut off, or padded with zeros, at the bandwidth L"""
    if L == self.bandwidth:
      return self
    fhat = np.zeros(((L + 1) ** 2, 3), dtype=self._fhat.dtype)
    n = min(len(fhat), len(self._fhat))
    fhat[:n] = self._fhat[:n]
    return self._rebuild(fhat)

  @classmethod
  def quadrature(cls, f, *args, bandwidth=None, weights=None):
    """S2VectorFieldHarmonic.quadrature(f, bandwidth=L), from a callable or from nodes and
    values with their weights, MATLAB's name for the constructor

      vF = S2VectorFieldHarmonic.quadrature(lambda v: v.cross(zvector), bandwidth=8)  # of a function returning vectors
      vF = S2VectorFieldHarmonic.quadrature(v, w, bandwidth=8, weights=wq)            # of vectors w at directions v

    A function is approximated by quadrature on a grid of the bandwidth, componentwise in x, y
    and z; vectors at scattered directions by the quadrature sum with the weights given, which
    is exact only for the weights of a quadrature rule; S2VectorFieldHarmonic.interpolate fits
    them by least squares.

    Input
      f - a function of vector3d returning vectors, S2VectorField, or vector3d of directions
      args - the vectors at the directions, and the frame of the field
    Output
      vF - S2VectorFieldHarmonic
    Keywords
      bandwidth - the largest harmonic degree; by default the preference 'defaultS2Bandwidth',
          for a field the field's
      weights - the quadrature weights of vectors at scattered directions, equal ones of sum
          4 pi by default
    See also
      S2VectorFieldHarmonic, S2VectorFieldHarmonic.interpolate"""
    return cls(f, *args, bandwidth=bandwidth, weights=weights)

  @classmethod
  def interpolate(cls, nodes, values, **kwargs):
    """the least squares fit to vectors at directions"""
    return interpolateField(nodes, values, **kwargs)

  @property
  def x(self):
    """the Cartesian components as functions"""
    return self.component(0)

  @property
  def y(self):
    """the y component as a function on the sphere"""
    return self.component(1)

  @property
  def z(self):
    """the z component as a function on the sphere"""
    return self.component(2)

  def component(self, i):
    """one component as a function without the group"""
    return S2FunHarmonic(self._fhat[:, i], None if self.frame is None else self.frame.stripSym(), real=True, symmetrise=False, _internal=True)

  def eval(self, x):
    """the vectors of the field at the directions x, vF(x)"""
    v = _vectorsFor(self, x).normalize()
    return vector3d(_seriesColumns(self._fhat, v).reshape(*v.shape, 3))

  def symmetrise(self):
    """the projection onto the equivariant fields of the frame, v(s xi) = s v(xi) for the
    elements of the group including the improper ones"""
    if self.frame is None or self.frame.sym.numSym() == 1:
      return self
    L = self.bandwidth
    els = self.frame.sym.rot.flatten()
    actions = [els[i].matrix() for i in range(els.size)]
    return self._rebuild(_projectSphericalRows(self._fhat.copy(), equivariantProjection(actions, _parityWignerD(els, L))))

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
    """the mean vector of the field over the sphere, read off the coefficients of degree zero"""
    return vector3d(np.real(self._fhat[0]) / math.sqrt(4 * math.pi))

  def isclose(self, w, rtol=1e-8):
    """whether two fields agree in their coefficients up to the relative tolerance rtol

      tf = isclose(vF1, vF2)             # equal up to a relative 1e-8
      tf = isclose(vF1, vF2, rtol=1e-2)  # up to a relative 1e-2

    The fields are compared at the larger of their bandwidths: the norm of the difference of
    the coefficients of the three components must not exceed rtol times the norm of those of
    w; a field of another kind is expanded first.

    Input
      vF1 - S2VectorFieldHarmonic
      w - S2VectorField, the field to compare with
    Output
      tf - bool
    Keywords
      rtol - the tolerance relative to the norm of w
    See also
      S2FunHarmonic.isclose"""
    w = S2VectorFieldHarmonic(w)
    L = max(self.bandwidth, w.bandwidth)
    a, b = self.truncate(L)._fhat, w.truncate(L)._fhat
    return bool(np.linalg.norm(a - b) <= rtol * np.linalg.norm(b))

  def dot(self, x):
    """the inner product with a constant vector, exact, a function without the group"""
    if isinstance(x, S2VectorField):
      return S2VectorField.dot(self, x)
    x = asVector3d(x)
    if x.frame is not None and self.frame is not None and x.frame is not self.frame:
      raise ValueError("the vector lives in another frame than the field")
    return S2FunHarmonic(self._fhat @ x.data.reshape(3), None if self.frame is None else self.frame.stripSym(), real=True, symmetrise=False, _internal=True)

  def cross(self, x):
    """F x x = -[x]_x F, on the rows of the coefficients F [x]_x"""
    if isinstance(x, S2VectorField):
      return S2VectorField.cross(self, x)
    x = asVector3d(x)
    return self._rebuild(self._fhat @ spinTensor(x.data.reshape(3)))

  def rotate(self, r):
    """v'(xi) = r v(r^-1 xi): the components rotate and mix"""
    if isinstance(r, orientation):
      if self.frame is not None and r.CS is not self.frame:
        raise ValueError("the orientation maps out of the frame of the field")
      return _reframe(self.rotate(rotation(r)), r.SS)
    rot = rotation(r.data.reshape(4))
    L = self.bandwidth
    D = _WignerD(rot, bandwidth=L)
    fhat = self._fhat.copy()
    for l in range(L + 1):
      fhat[l * l:(l + 1) ** 2] = D[l] @ self._fhat[l * l:(l + 1) ** 2]
    fhat = fhat @ rot.matrix().T
    return self._rebuild(fhat, frame=None if self.frame is None else _rotatedFrame(self.frame, rot))

  def antipode(self):
    """the field of the antipode, v(-xi)"""
    l = _s2Degrees(self.bandwidth)
    return self._rebuild(np.where(l % 2, -1.0, 1.0)[:, None] * self._fhat)

  def __repr__(self):
    info = "" if self.frame is None else f" ({frameLabel(self.frame)})"
    return aligned(f"S2VectorFieldHarmonic{info}\n  bandwidth: {self.bandwidth}")


def _reframe(v, frame):
  return v._rebuild(v._fhat, frame=frame)


# ------------------------------------------------------------------ the surface gradient


# the ladder coefficients of the angular derivatives over a coefficient vector: at the
# position of (l, m) the factor of f(l, m+1) and of f(l, m-1), zero at the ends of a degree
_LADDERS = {}


def _ladders(L):
  if L not in _LADDERS:
    l = _s2Degrees(L)
    m = np.arange((L + 1) ** 2) - l * l - l
    up = np.where(m < l, np.sqrt(np.maximum(l * (l + 1) - m * (m + 1), 0)) / 2, 0.0)
    down = np.where(m > -l, np.sqrt(np.maximum(l * (l + 1) - (m - 1) * m, 0)) / 2, 0.0)
    _LADDERS[L] = (m, up, down)
  return _LADDERS[L]


def angularDerivatives(fhat):
  """L_i f, the derivative under the rotation about e_i, on the coefficients: the ladder
  operators as shifts of the coefficient vector, the degrees never mixing"""
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


def grad(f, x=None):
  """the surface gradient (L f) x xi, at directions and as a field of the next bandwidth"""
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
  """a pointwise operation on fields, functions and numbers by quadrature at the bandwidth, a field
  where op returns vectors and a function on the sphere where it returns numbers; a field's
  vectors are an (n, 3) array, a function's values an (n, 1) column broadcasting against them"""
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
    return np.real(a(xs)).reshape(-1, 1)

  values = op(evaluate(v), *(evaluate(a) for a in args))
  if reduce:
    values = np.sum(values, axis=-1)
  # three columns are a field, a single value per node a function
  if values.ndim == 2 and values.shape[-1] == 3:
    return S2VectorFieldHarmonic(_adjointColumns(g, values, L, g.weights), v.frame, symmetrise=False)
  values = np.broadcast_to(values, (len(xs),) + np.shape(values)[1:]).reshape(len(xs))
  fr = v.frame if invariant or v.frame is None else v.frame.stripSym()
  return S2FunHarmonic(sphericalSeriesAdjoint(g, values, L), fr, real=True, symmetrise=False, _internal=True)


# ------------------------------------------------------------------ the handle and the polar basis


class S2VectorFieldHandle(S2VectorField):
  """a vector field on the sphere given by a callable of the directions, the bandwidth a hint for
  its harmonic expansion

    vF = S2VectorFieldHandle(lambda v: v.cross(vector3d.Z))

  Input
    fun - callable receiving unit vector3d directions and returning one vector per direction
    frame - referenceFrame of the query directions, or None for frame-free directions
    bandwidth - harmonic degree used when approximating the callable; default defaultS2Bandwidth preference"""

  def __init__(self, fun, frame=None, *, bandwidth=None):
    if isinstance(fun, S2VectorField):
      v = fun
      fun, frame, bandwidth = (lambda x: v(x)), v.frame, v.bandwidth
    self.fun, self.frame = fun, frame
    self.bandwidth = getMTEXpref("defaultS2Bandwidth") if bandwidth is None else int(bandwidth)

  def eval(self, x):
    """the vectors the callable gives at the directions x, vF(x)"""
    v = _vectorsFor(self, x).normalize()
    return vector3d(_values(self.fun(self._framed(v))).reshape(*v.shape, 3))

  def _harmonicField(self, bandwidth):
    return S2VectorFieldHarmonic(lambda x: self(x), self.frame, bandwidth=self.bandwidth if bandwidth is None else bandwidth)

  @property
  def fhat(self):
    """the coefficients of the harmonic expansion of the field at its bandwidth hint"""
    return S2VectorFieldHarmonic(self)._fhat

  def _scaled(self, a):
    return S2VectorFieldHandle(lambda x: a * _values(self.fun(x)), self.frame, bandwidth=self.bandwidth)

  def __add__(self, w):
    return S2VectorFieldHarmonic(self) + w

  def rotate(self, r):
    """the field rotated by r, v'(xi) = r v(r^-1 xi)"""
    rot = rotation(r.data.reshape(4))
    return S2VectorFieldHandle(lambda x: (rot * self(rot.inv() * vector3d(x.data))).data, None if self.frame is None else _rotatedFrame(self.frame, rot), bandwidth=self.bandwidth)

  def __repr__(self):
    info = "" if self.frame is None else f" ({frameLabel(self.frame)})"
    return aligned(f"S2VectorFieldHandle{info}\n  bandwidth hint: {self.bandwidth}")


def thetaVector(v):
  """the unit vectors of the polar coordinates at a direction: along the polar angle and
  along the azimuth"""
  v = asVector3d(v)
  t, r = v.theta, v.rho
  return vector3d(np.cos(r) * np.cos(t), np.sin(r) * np.cos(t), -np.sin(t), frame=v.frame)


def rhoVector(v):
  """the unit vector along the azimuth at a direction, the second vector of the polar basis"""
  v = asVector3d(v)
  r = v.rho
  return vector3d(-np.sin(r), np.cos(r), np.zeros_like(r), frame=v.frame)


def interpolateField(v, values, **kwargs):
  """the vectors at directions by least squares componentwise"""
  from .density import interpolate
  d = vector3d(v.data.reshape(-1, 3), frame=v.frame).normalize()
  vals = _values(values)
  fs = [interpolate(vector3d(d.data), vals[:, i], **kwargs) for i in range(3)]
  return S2VectorFieldHarmonic(fs, v.frame)


__all__ = ["S2VectorFieldHarmonic", "S2VectorFieldHandle", "thetaVector", "rhoVector", "angularDerivatives"]
