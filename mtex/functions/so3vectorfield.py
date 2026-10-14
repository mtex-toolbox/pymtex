# vector fields on SO(3): tangent vectors of one side of ADR 0010 at every orientation
# (MATLAB @SO3VectorFieldHarmonic, @SO3VectorFieldHandle, @SO3VectorFieldRBF;
# docs/plans/functions.md F7)
#
#   SO3VF = SO3VectorFieldHarmonic(fhat, CS, SS, 'right')      three coefficient columns, the components
#   SO3VF = SO3VectorFieldHarmonic(fun | t | field, CS, SS, bandwidth=)
#   SO3VF(ori)                                                  the tangent vectors at ori
#   SO3VF.left(), SO3VF.right(), component(SO3VF, i), curl(SO3VF), div(SO3VF), antiderivative(SO3VF)
#   dot(SO3VF, x), cross(SO3VF, W), f * SO3VF, abs(SO3VF), normalize(SO3VF), rotate, inv
#
# side: left components live in the frame the rotation maps into, right components in the
# frame it maps from. One coefficient derivative serves every differential operator,
# differentiate applying -i J_i^T on the rows (left) or the columns (right) of each degree
# block: grad f = (X_i f)_i, div F = sum_i X_i F_i, curl F = rot F +- F, antiderivative
# F = Laplace^-1 div F. A left field is invariant under the crystal group and equivariant
# under the specimen one, F(s R) = s F(R); its components are functions on the frames with
# the group of the tangent side stripped.
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..geometry.kernels import _qmulNumpy as _qmul, qconj
from ..geometry.frames import referenceFrame
from ..geometry.grids.so3grid import equispacedSO3Grid
from ..geometry.orientation import orientation
from ..geometry.quaternion import quaternion, rotation
from ..geometry.session import getMTEXpref
from ..geometry.symreduce import reductionGroups
from ..geometry.tangent import SO3TangentVector
from ..geometry.units import degree
from ..geometry.vector3d import vector3d, xvector, yvector, zvector, asVector3d
from ..geometry.display import framesLabel
from .funops import _rotatedCoefficients, _rotatedFrame
from .harmonics import _so3Degrees, _so3PerDegree, mtexCoefficientsSO3, _WignerD, deg2dim, degreeBlock, dim2deg, invariantProjection, projectLeft, projectRight
from .kernelfunctions import SO3DeLaValleePoussinKernel
from .quadrature import bandwidthFor, quadratureSO3Grid
from .rbf import _framesAmong, _rotationsFor, supportCosine
from .so3fun import SO3FunHarmonic, invertedCoefficients, realizeWigner
from .transforms import DIRECT_SUM_NODES, SO3FourierPlan, wignerPlan, wignerSeries, wignerSeriesAdjoint
from ..geometry.display import aligned
from ..geometry.framepair import installFramePair

SIDES = ("left", "right")


def _side(side):
  if side not in SIDES:
    raise ValueError(f"the tangent space is 'left' or 'right', not {side!r}")
  return side


def _other(side):
  return "right" if side == "left" else "left"


def componentFrames(CS, SS, side="left"):
  """the frames the components are functions on: the group on the tangent side is stripped

    cA, cB = componentFrames(cs, ss)                 # of a field in the left tangent space
    cA, cB = componentFrames(cs, ss, side='right')   # of one in the right tangent space

  A field in the left tangent space returns vectors in the specimen frame, which turn with a
  specimen symmetry, so its three components are invariant only under the crystal symmetry;
  in the right tangent space the other way round. The port's own, no MATLAB counterpart.

  Input
    CS, SS - the crystal and the specimen frame of the field
  Output
    cA, cB - the crystal and the specimen frame of the components
  Keywords
    side - 'left' or 'right', the tangent space
  See also
    SO3VectorFieldHarmonic"""
  return (CS, SS.stripSym()) if side == "left" else (CS.stripSym(), SS)


# the group a field is equivariant under, as (n, 4) quaternions
def _equivariantGroup(v):
  fr = v.SS if v.tangentSpace == "left" else v.CS
  return fr.sym.properGroup().rotations().data.reshape(-1, 4)


# ------------------------------------------------------------------ the coefficient derivative and the column transforms


def differentiate(fhat, side="left"):
  """the three derivatives of a coefficient vector, on the rows for the left side and on the
  columns for the right; the degree is preserved

    D = differentiate(SO3F._fhat)                  # the left derivatives, one column each
    D = differentiate(SO3F._fhat, side='right')    # the right derivatives

  The three columns are the coefficients of the derivatives along the x, y and z axes of the
  tangent space, together the coefficient matrix of the gradient field. The coefficients are
  in the port's internal convention, those of SO3F._fhat, not MATLAB's fhat. MATLAB has no
  counterpart.

  Input
    fhat - ndarray, deg2dim(L + 1) Wigner coefficients
  Output
    D - ndarray of shape (deg2dim(L + 1), 3)
  Keywords
    side - 'left' or 'right', the tangent space
  See also
    SO3FunHarmonic.grad, div, curl"""
  side = _side(side)
  fhat = np.asarray(fhat)
  L = dim2deg(len(fhat))
  out = np.empty((len(fhat), 3), dtype=fhat.dtype)
  for n in range(L + 1):
    F = degreeBlock(fhat, n)
    m = np.arange(-n, n)
    c = np.sqrt(n * (n + 1) - m * (m + 1)) / 2
    idx = np.arange(2 * n)
    Jx = np.zeros((2 * n + 1, 2 * n + 1), dtype=fhat.dtype)
    Jy = np.zeros_like(Jx)
    Jx[idx + 1, idx] = -1j * c
    Jx[idx, idx + 1] = -1j * c
    Jy[idx + 1, idx] = c
    Jy[idx, idx + 1] = -c
    Jz = np.diag(-1j * np.arange(-n, n + 1)).astype(fhat.dtype)
    for i, J in enumerate((Jx, Jy, Jz)):
      degreeBlock(out[:, i], n)[...] = J @ F if side == "left" else F @ J
  return out


# the values of every column at the rotations, one plan for all
def _seriesColumns(fhat, rots):
  L = dim2deg(fhat.shape[0])
  q = rotation(rots.data.reshape(-1, 4))
  if q.size < DIRECT_SUM_NODES or L == 0:
    def series(c):
      return np.real(wignerSeries(c, q, real=True))
  else:
    series = SO3FourierPlan(q, L, real=True)
  return np.stack([series(fhat[:, i]) for i in range(fhat.shape[1])], axis=-1)


# the adjoint of every column of values, one plan for all; on a quadrature grid its FFT
def _adjointColumns(nodes, values, L, w):
  if isinstance(nodes, quadratureSO3Grid):
    adjoint = wignerPlan(nodes, L, True).adjoint
  elif nodes.size < DIRECT_SUM_NODES:
    q = rotation(nodes.data.reshape(-1, 4))

    def adjoint(y, w):
      return wignerSeriesAdjoint(q, y, L, w)
  else:
    adjoint = SO3FourierPlan(rotation(nodes.data.reshape(-1, 4)), L, real=True).adjoint
  return np.stack([adjoint(values[:, i], w) for i in range(values.shape[1])], axis=-1)


# the values a callable returned, as (n, 3) components on the side asked for
def _tangentValues(x, side):
  if isinstance(x, SO3TangentVector):
    x = x.left() if side == "left" else x.right()
    return x.data.reshape(-1, 3)
  return (x.data if isinstance(x, vector3d) else np.asarray(x, dtype=float)).reshape(-1, 3)


def equivariantProjection(actions, Ds):
  """the projection of stacked components onto the equivariant fields: for every degree the
  mean over the group of M(s) kron D(s), M the matrix acting on the components and D the
  matrix acting on one component from the left; on the columns the component acts by
  D(s)^T from the right, which is the transpose of the same Kronecker product"""
  L = len(Ds[0]) - 1
  k = actions[0].shape[0]
  P = [np.zeros((k * (2 * n + 1), k * (2 * n + 1)), dtype=complex) for n in range(L + 1)]
  for M, D in zip(actions, Ds):
    for n in range(L + 1):
      P[n] += np.kron(M, D[n])
  return [M / len(actions) for M in P]


# the degree blocks of the k components stacked, on the rows (axis 0) projected from the left
# by P_n, on the columns (axis 1) from the right by P_n^T, and written back in place
def _projectStacked(fhat, P, axis):
  k = fhat.shape[1]
  for n in range(len(P)):
    stacked = np.concatenate([degreeBlock(fhat[:, i], n) for i in range(k)], axis=axis)
    stacked = P[n] @ stacked if axis == 0 else stacked @ P[n].T
    for i, block in enumerate(np.split(stacked, k, axis=axis)):
      degreeBlock(fhat[:, i], n)[...] = block
  return fhat


def _projectSphericalRows(fhat, P):
  k = fhat.shape[1]
  for l in range(len(P)):
    block = fhat[l * l:(l + 1) ** 2, :]
    fhat[l * l:(l + 1) ** 2, :] = (P[l] @ block.T.ravel()).reshape(k, 2 * l + 1).T
  return fhat


def _readSide(args):
  sides = [a for a in args if isinstance(a, str)]
  if len(sides) > 1:
    raise TypeError("one tangent space")
  return _side(sides[0]) if sides else "left", [a for a in args if not isinstance(a, str)]


# ------------------------------------------------------------------ what every field on SO(3) shares


class SO3VectorField:
  """the base of the vector fields on SO(3): a tangent vector at every orientation, in the left or
  the right tangent space (tangentSpace), with the algebra, the pointwise operations and the
  differential operators every field shares; the subclasses implement eval"""

  @property
  def side(self):
    """the tangent space the values are returned in, 'left' or 'right', the same as tangentSpace"""
    return self.tangentSpace

  def componentFrames(self):
    """the frames the components are functions on: the field's frames with the group on the tangent
    side stripped"""
    return componentFrames(self.CS, self.SS, self.tangentSpace)

  def __call__(self, x):
    return self.eval(x)

  def _rotationsFor(self, x):
    return _rotationsFor(self, x)

  def _tangent(self, values, x):
    r = self._rotationsFor(x)
    return SO3TangentVector(vector3d(values.reshape(*r.shape, 3)), x if isinstance(x, (orientation, quaternion)) else r, self.tangentSpace)

  def left(self):
    """the field with its values in the left tangent space, as a harmonic field"""
    return self.toSide("left")

  def right(self):
    """the field with its values in the right tangent space, as a harmonic field"""
    return self.toSide("right")

  def toSide(self, side):
    """the other side: F_L(R) = R F_R(R), a product with a degree one function, exact by
    quadrature at the next bandwidth"""
    side = _side(side)
    if side == self.tangentSpace and isinstance(self, SO3VectorFieldHarmonic):
      return self
    return SO3VectorFieldHarmonic(lambda o: self(o), self.CS, self.SS, side, bandwidth=self.bandwidth + (1 if isinstance(self, SO3VectorFieldHarmonic) else 0))

  def _sameFrames(self, w):
    if w.CS is not self.CS or w.SS is not self.SS:
      raise ValueError("the fields live in different frames")

  def dot(self, x):
    """the pointwise inner product: with another field a function on SO(3), with a constant vector a
    function on the component frames"""
    if isinstance(x, SO3VectorField):
      return mapField(np.multiply, self, x, reduce=True, invariant=True)
    return SO3VectorFieldHarmonic(self).dot(x)

  def cross(self, w):
    """the pointwise cross product with another field, a field of the same side"""
    return mapField(np.cross, self, w)

  def abs(self):
    """the length of the vectors, a function on SO(3) by quadrature at twice the bandwidth"""
    return mapField(lambda a: np.linalg.norm(a, axis=-1), self, bandwidth=max(2 * self.bandwidth, 32), invariant=True)

  def __abs__(self):
    return self.abs()

  def normalize(self):
    """the field of unit vectors in the directions of the field, by quadrature at twice the bandwidth"""
    return mapField(lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True), self, bandwidth=max(2 * self.bandwidth, 32))

  def norm(self):
    """the length of the vectors as a function on SO(3), the same as abs

      SO3F = norm(SO3VF)"""
    return self.abs()

  def curl(self, x=None):
    """the curl, a field of the same side; with x its values at the orientations x

      C = SO3VF.curl()        # the curl, an SO3VectorFieldHarmonic
      c = SO3VF.curl(ori)     # its values at orientations

    The field is taken in its harmonic form; the curl of a gradient vanishes. MATLAB's curl of a
    field that is not harmonic returns a handle, here always the harmonic field.

    Input
      SO3VF - SO3VectorField
      x - orientation, where to evaluate the curl
    Output
      C - SO3VectorFieldHarmonic
      c - SO3TangentVector, the values at x
    See also
      div, SO3VectorField.antiderivative, SO3FunHarmonic.grad"""
    return curl(self, x)

  def div(self, x=None):
    """the divergence, a function on SO(3); with x its values at the orientations x

      D = SO3VF.div()         # the divergence, an SO3FunHarmonic
      d = SO3VF.div(ori)      # its values at orientations

    The field is taken in its harmonic form; the divergence of a gradient is the Laplacian of
    the function.

    Input
      SO3VF - SO3VectorField
      x - orientation, where to evaluate the divergence
    Output
      D - SO3FunHarmonic
      d - array, the values at x
    See also
      curl, SO3VectorField.antiderivative, SO3FunHarmonic.grad"""
    return div(self, x)

  def antiderivative(self, x=None, **kwargs):
    """the function of mean zero whose gradient the field is; raises unless the field is a gradient,
    or conservative=True says it is; with x its values at the orientations x

      F = SO3VF.antiderivative()                    # the function, an SO3FunHarmonic
      f = SO3VF.antiderivative(ori)                 # its values at orientations
      F = SO3VF.antiderivative(conservative=True)   # without checking the field is a gradient
      F = SO3VF.antiderivative(tol=1e-3)            # checked with another tolerance

    The keywords conservative and tol go to antiderivative: the function is the inverse
    Laplacian of the divergence, and unless conservative=True the gradient of the result must
    give the field back within tol (1e-4) relative to its norm, else a ValueError is raised.

    Input
      SO3VF - SO3VectorField, converted to its harmonic form
      x - orientation, where to evaluate the antiderivative
    Output
      F - SO3FunHarmonic
      f - array, the values at x
    See also
      antiderivative, SO3FunHarmonic.grad, SO3VectorField.curl"""
    return antiderivative(self, x, **kwargs)

  def __neg__(self):
    return -1 * self

  def __sub__(self, w):
    return self + (-1 * w)

  def __truediv__(self, a):
    return (1 / a) * self

  def __radd__(self, w):
    return self + w

  def __mul__(self, a):
    if isinstance(a, (int, float, np.number)):
      return self._scaled(a)
    return mapField(np.multiply, self, a)

  __rmul__ = __mul__


# ------------------------------------------------------------------ the harmonic field


class SO3VectorFieldHarmonic(SO3VectorField):
  """a vector field on SO(3) whose three components are series of Wigner functions, in the left or
  the right tangent space

    SO3VF = SO3VectorFieldHarmonic(fhat, cs, ss, 'left')        # from a coefficient matrix
    SO3VF = SO3VectorFieldHarmonic(F, 'right')                  # another field in the right tangent space
    SO3VF = SO3VectorFieldHarmonic(F, bandwidth=8)              # another field cut to a bandwidth
    SO3VF = SO3VectorFieldHarmonic(t, bandwidth=8)              # through tangent vectors t at orientations
    SO3VF = SO3VectorFieldHarmonic([F1, F2, F3])                # from its three component functions
    SO3VF = SO3VectorFieldHarmonic(lambda ori: F(ori), cs, ss, bandwidth=8)  # a callable by quadrature

  A field in the left tangent space returns its tangent vectors in the specimen frame, one
  in the right tangent space in the crystal frame; 'left' unless named. Another field given
  is converted to the side named.

  The field is built from:

  - a coefficient matrix: the components as columns of MTEX's coefficients, made equivariant
  under the frames unless symmetrise=False.
  - tangent vectors t at reference orientations: summed with the weights against the Wigner
  functions, a quadrature only for the weights of a quadrature rule.
  - a callable of orientations: sampled on a quadrature grid of the bandwidth, by default the
  preference defaultSO3Bandwidth.
  - three component functions: their coefficients, cut to the largest bandwidth; a Python
  list takes its frames from cs and ss, else the trivial frames.

  Input
    fhat - (deg2dim(L + 1), 3) array, an SO3VectorField, an SO3TangentVector, three
        SO3FunHarmonic or a callable of orientations
    cs, ss - crystalFrame and specimenFrame
  Output
    SO3VF - SO3VectorFieldHarmonic
  Keywords
    bandwidth - the largest degree of the components
    weights - the weights of the tangent vectors, equal unless given
    symmetrise - make a coefficient matrix equivariant under the frames' symmetries
  See also
    SO3VectorFieldRBF, grad, antiderivative, mapField, SO3TangentVector, SO3FunVectorField"""

  def __init__(self, fhat, *args, bandwidth=None, weights=None, symmetrise=True):
    side, rest = _readSide(args)
    frames = [a for a in rest if isinstance(a, referenceFrame)]
    CS, SS = _framesAmong(frames)
    # values at orientations are tangent vectors there
    if isinstance(fhat, orientation) and rest and not isinstance(rest[0], referenceFrame):
      fhat = SO3TangentVector(vector3d(_tangentValues(rest[0], side).reshape(*fhat.shape, 3)), fhat, side)
    if isinstance(fhat, SO3VectorField):
      v = fhat
      CS, SS = v.CS, v.SS
      side = side if any(isinstance(a, str) for a in args) else v.tangentSpace
      if isinstance(v, SO3VectorFieldHarmonic) and side == v.tangentSpace:
        fhat = v.truncate(bandwidth)._fhat if bandwidth is not None else v._fhat
      else:
        fhat = v._harmonicField(side, bandwidth)._fhat
      symmetrise = False
    elif isinstance(fhat, SO3TangentVector):
      t = fhat
      ref = t.oriRef
      if not isinstance(ref, orientation):
        ref = orientation._fromRotation(ref.data, (CS, SS))
      CS, SS, side = ref.CS, ref.SS, t.side
      L = getMTEXpref("defaultSO3Bandwidth") if bandwidth is None else int(bandwidth)
      w = np.full(ref.size, 1 / ref.size) if weights is None else np.asarray(weights, dtype=float).ravel()
      fhat = _adjointColumns(rotation(ref).flatten(), t.data.reshape(-1, 3), L, w)
    elif isinstance(fhat, SO3FunHarmonic) and fhat.shape == (3,):
      # a list of three functions: their components, in their frames
      if not frames:
        CS, SS = fhat.CS, fhat.SS
      fhat = np.moveaxis(fhat._fhat, 0, -1)
    elif isinstance(fhat, (list, tuple)) and fhat and isinstance(fhat[0], SO3FunHarmonic):
      if len(fhat) != 3:
        raise ValueError("three component functions are needed")
      L = max(f.bandwidth for f in fhat)
      fhat = np.stack([f.truncate(L)._fhat for f in fhat], axis=-1)
    elif callable(fhat) and not isinstance(fhat, np.ndarray):
      fhat, cut = self._byQuadrature(fhat, CS, SS, side, bandwidth)
      self._fhat, self.CS, self.SS, self.tangentSpace = fhat, CS, SS, side
      self._fhat = self.symmetrise()._fhat
      self._fhat = self.truncate(cut)._fhat
      return
    fhat = np.array(fhat, dtype=complex)
    if fhat.ndim != 2 or fhat.shape[1] != 3:
      raise ValueError("a vector field has three components")
    if deg2dim(dim2deg(fhat.shape[0]) + 1) != fhat.shape[0]:
      raise ValueError("a coefficient matrix has deg2dim(L+1) rows for its bandwidth L")
    for i in range(3):
      realizeWigner(fhat[:, i])
    self._fhat, self.CS, self.SS, self.tangentSpace = fhat, CS, SS, side
    if symmetrise:
      self._fhat = self.symmetrise()._fhat

  @classmethod
  def quadrature(cls, fun, *args, bandwidth=None, tol=1e-10):
    """MATLAB's SO3VectorFieldHarmonic.quadrature(fun, cs): the field of a callable, cut to
    the last degree that carries more than rounding, as a polynomial field is

      SO3VF = SO3VectorFieldHarmonic.quadrature(lambda ori: ori * xvector, cs, ss)  # a callable, cut to its degree
      SO3VF = SO3VectorFieldHarmonic.quadrature(F.eval, cs, ss, bandwidth=20)    # at a bandwidth of its own
      SO3VF = SO3VectorFieldHarmonic.quadrature(t, bandwidth=8)   # through tangent vectors t at orientations

    The field is the SO3VectorFieldHarmonic of fun and the further arguments, a callable taking
    orientations and returning vectors or tangent vectors evaluated by exact quadrature, another
    field, or tangent vectors at orientations; then every degree above the last whose
    coefficients exceed tol times the largest is dropped. A callable is in the left tangent space
    unless 'right' is passed after the frames.

    Input
      fun - callable, SO3VectorField or SO3TangentVector
      cs, ss - the crystal and the specimen frame, for a callable
    Output
      SO3VF - SO3VectorFieldHarmonic
    Keywords
      bandwidth - the bandwidth of the quadrature; the preference 'defaultSO3Bandwidth' (64) by
          default
      tol - the coefficients below tol times the largest count as rounding when the degree is cut
    See also
      SO3VectorFieldHarmonic, SO3FunHarmonic.quadrature, interpolate"""
    F = cls(fun, *args, bandwidth=bandwidth)
    size = np.abs(F._fhat).max(axis=1)
    big = np.flatnonzero(size > tol * max(size.max(), np.finfo(float).tiny))
    L = 0
    while big.size and deg2dim(L + 1) <= big[-1]:
      L += 1
    return F.truncate(L)

  @classmethod
  def approximate(cls, fun, *args, **kwargs):
    """MATLAB's approximate: the quadrature of a callable or of values at orientations"""
    return cls.quadrature(fun, *args, **kwargs)

  # from a callable by exact quadrature on the rule of the component frames, then the
  # projection, which the folded rule needs; the callable receives the orientations of
  # the rule in the frames of the field and returns vectors or tangent vectors
  @staticmethod
  def _byQuadrature(fun, CS, SS, side, bandwidth):
    L0 = getMTEXpref("defaultSO3Bandwidth") if bandwidth is None else int(bandwidth)
    cA, cB = componentFrames(CS, SS, side)
    L = bandwidthFor(L0, cA, cB)
    g = quadratureSO3Grid(L, cA, cB)
    o = orientation._fromRotation(g.nodes().data, (CS, SS))
    values = _tangentValues(fun(o), side)
    return _adjointColumns(g, values, L, g.weights), L0

  def _harmonicField(self, side, bandwidth):
    return self.toSide(side).truncate(self.bandwidth if bandwidth is None else bandwidth)

  def _rebuild(self, fhat, **changes):
    v = object.__new__(SO3VectorFieldHarmonic)
    v._fhat, v.CS, v.SS, v.tangentSpace = np.asarray(fhat), self.CS, self.SS, self.tangentSpace
    for k, x in changes.items():
      setattr(v, k, x)
    return v

  @property
  def fhat(self):
    """the coefficients of the components in MTEX's basis, one column per component"""
    return mtexCoefficientsSO3(self._fhat.T).T

  @property
  def bandwidth(self):
    """the bandwidth, the largest degree of the component expansions"""
    return dim2deg(self._fhat.shape[0])

  def truncate(self, L):
    """the field with its coefficients cut or padded to the bandwidth L"""
    if L == self.bandwidth:
      return self
    fhat = np.zeros((deg2dim(L + 1), 3), dtype=self._fhat.dtype)
    n = min(len(fhat), len(self._fhat))
    fhat[:n] = self._fhat[:n]
    return self._rebuild(fhat)

  def component(self, i):
    """one component as a function on the component frames"""
    cA, cB = self.componentFrames()
    return SO3FunHarmonic(self._fhat[:, i], cA, cB, real=True, symmetrise=False, _internal=True)

  def eval(self, x):
    """the tangent vectors of the field at orientations"""
    r = self._rotationsFor(x)
    return self._tangent(_seriesColumns(self._fhat, r), x)

  def symmetrise(self):
    """the projection onto the equivariant fields of the frames: the invariant side one
    component at a time, the equivariant side F -> s F(s^-1 R) on the left, F -> s F(R s)
    on the right, by the Kronecker products"""
    L = self.bandwidth
    fhat = self._fhat.copy()
    if self.tangentSpace == "left":
      Q = invariantProjection(self.CS, L)
      for i in range(3):
        projectRight(fhat[:, i], Q)
    else:
      P = invariantProjection(self.SS, L)
      for i in range(3):
        projectLeft(fhat[:, i], P)
    rots = _equivariantGroup(self)
    if len(rots) == 1:
      return self._rebuild(fhat)
    actions = [rotation(q).matrix() for q in rots]
    if self.tangentSpace == "left":
      Ds = [[np.conj(D) for D in _WignerD(rotation(q), bandwidth=L)] for q in rots]
      _projectStacked(fhat, equivariantProjection(actions, Ds), 0)
    else:
      Ds = [_WignerD(rotation(q), bandwidth=L) for q in rots]
      _projectStacked(fhat, equivariantProjection(actions, Ds), 1)
    return self._rebuild(fhat)

  # ---- the linear algebra

  def __add__(self, w):
    if not isinstance(w, SO3VectorField):
      return NotImplemented
    self._sameFrames(w)
    w = SO3VectorFieldHarmonic(w, self.tangentSpace)
    L = max(self.bandwidth, w.bandwidth)
    return self._rebuild(self.truncate(L)._fhat + w.truncate(L)._fhat)

  def _scaled(self, a):
    return self._rebuild(a * self._fhat)

  def mean(self):
    """the mean over SO(3), and the length of the vectors as a function, MATLAB's norm"""
    return vector3d(np.real(self._fhat[0]))

  def norm(self):
    """the length of the vectors as a function on SO(3), the same as abs

      SO3F = norm(SO3VF)"""
    return self.abs()

  def isclose(self, w, rtol=1e-8):
    """whether two fields have the same coefficients on one side up to the relative tolerance rtol

      SO3VF.isclose(SO3VF2)              # equal up to the relative tolerance 1e-8
      SO3VF.isclose(SO3VF2, rtol=1e-3)   # up to another one

    The second field is taken in the harmonic form and the tangent space of the first; the
    coefficients, padded to the larger bandwidth, agree when the norm of their difference is at
    most rtol times the norm of those of the second.

    Input
      SO3VF - SO3VectorFieldHarmonic
      w - SO3VectorField
    Output
      tf - bool
    Keywords
      rtol - the tolerance relative to the norm of w
    See also
      SO3FunHarmonic.isclose"""
    w = SO3VectorFieldHarmonic(w, self.tangentSpace)
    L = max(self.bandwidth, w.bandwidth)
    a, b = self.truncate(L)._fhat, w.truncate(L)._fhat
    return bool(np.linalg.norm(a - b) <= rtol * np.linalg.norm(b))

  def dot(self, x):
    """the inner product with a constant vector, exact, a function on the component frames"""
    if isinstance(x, SO3VectorField):
      return SO3VectorField.dot(self, x)
    x = asVector3d(x)
    # a vector of the frame of the other side takes the components of that side
    own, other = (self.SS, self.CS) if self.tangentSpace == "left" else (self.CS, self.SS)
    if x.frame is not None and x.frame is other and other is not own:
      return self.toSide(_other(self.tangentSpace)).dot(x)
    if x.frame is not None and x.frame is not own:
      raise ValueError("the vector lives in another frame than the components")
    cA, cB = self.componentFrames()
    return SO3FunHarmonic(self._fhat @ x.data.reshape(3), cA, cB, real=True, symmetrise=False, _internal=True)

  # ---- rotation and inversion

  def rotate(self, r, side="left"):
    """F'(R) = q F(q^-1 R) for a rotation of the specimen acting on a left field, F(q^-1 R) on
    a right one; F'(R) = F(R q^-1) and q^-1 F(R q^-1) for a rotation of the crystal

      SO3VF = rotate(SO3VF, rot)                 # turned in the specimen frame
      SO3VF = SO3VF.rotate(rot, side='right')    # turned in the crystal frame

    The coefficients of the three components are turned as those of a function, and where the
    rotation acts on the tangent space of the field the components are combined linearly as the
    vectors turn. A symmetry the rotation does not map onto itself is dropped with a warning. An
    orientation turns the field into the other frame of the orientation, as SO3FunHarmonic.rotate.

    Input
      SO3VF - SO3VectorFieldHarmonic
      rot - rotation or orientation, one
    Output
      SO3VF - SO3VectorFieldHarmonic
    Keywords
      side - 'left', the specimen side, or 'right', the crystal side; MATLAB's flag 'right'
    See also
      SO3VectorFieldHandle.rotate, SO3VectorFieldRBF.rotate, SO3FunHarmonic.rotate"""
    side = _side(side)
    if isinstance(r, orientation):
      o = r
      if side == "left":
        if o.CS is not self.SS:
          raise ValueError("a left rotation maps out of the specimen frame of the field")
        return _reframe(self.rotate(rotation(o), side), self.CS, o.SS)
      if o.SS is not self.CS:
        raise ValueError("a right rotation maps into the crystal frame of the field")
      return _reframe(self.rotate(rotation(o), side), o.CS, self.SS)
    rot = rotation(r.data.reshape(4))
    fhat = np.stack([_rotatedCoefficients(self._fhat[:, i], rot, side) for i in range(3)], axis=-1)
    M = _coordinateRotation(rot, self.tangentSpace, side)
    if M is not None:
      fhat = fhat @ M.T
    CS = _rotatedFrame(self.CS, rot) if side == "right" else self.CS
    SS = _rotatedFrame(self.SS, rot) if side == "left" else self.SS
    return self._rebuild(fhat, CS=CS, SS=SS)

  def inv(self):
    """the push-forward under R -> R^-1: a left vector v at R becomes the right vector -v at
    R^-1, so inv(grad f) = grad(inv f); the frames swap with the side"""
    fhat = np.stack([-invertedCoefficients(self._fhat[:, i]) for i in range(3)], axis=-1)
    return self._rebuild(fhat, CS=self.SS, SS=self.CS, tangentSpace=_other(self.tangentSpace))

  def __repr__(self):
    return aligned(f"SO3VectorFieldHarmonic ({framesLabel(self.CS, self.SS)})\n  tangent space: {self.tangentSpace}\n  bandwidth: {self.bandwidth}")

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from typing import Any
    @property
    def frameA(self) -> Any:
      ...
    @frameA.setter
    def frameA(self, value) -> None:
      ...
    @property
    def frameB(self) -> Any:
      ...
    @frameB.setter
    def frameB(self, value) -> None:
      ...


def _reframe(v, CS, SS):
  return v._rebuild(v._fhat, CS=CS, SS=SS) if isinstance(v, SO3VectorFieldHarmonic) else v._reframe(CS, SS)


def _coordinateRotation(r, storage, side):
  if storage != side:
    return None
  return r.matrix() if side == "left" else r.inv().matrix()


# ------------------------------------------------------------------ the differential operators


# accumulate only the three derivatives contributing to the divergence
def _divergenceCoefficients(fhat, side):
  out = np.zeros(fhat.shape[0], dtype=fhat.dtype)
  for i in range(3):
    out += differentiate(fhat[:, i], side)[:, i]
  return out


def div(v, x=None):
  """the divergence, invariant under both groups

    D = div(SO3VF)          # the divergence, an SO3FunHarmonic
    d = div(SO3VF, ori)     # its values at orientations

  The sum of the derivatives of the three components along their own axes of the tangent space,
  of the field in its harmonic form; of a gradient it is the Laplacian of the function.

  Input
    v - SO3VectorField
    x - orientation, where to evaluate the divergence
  Output
    D - SO3FunHarmonic
    d - array, the values at x
  See also
    curl, antiderivative, SO3FunHarmonic.grad"""
  v = SO3VectorFieldHarmonic(v)
  f = SO3FunHarmonic(_divergenceCoefficients(v._fhat, v.tangentSpace), v.CS, v.SS, real=True, symmetrise=False, _internal=True)
  return f if x is None else f(x)


def curl(v, x=None):
  """the curl, the dual of the exterior derivative, a field of the same side

    C = curl(SO3VF)         # the curl, an SO3VectorFieldHarmonic
    c = curl(SO3VF, ori)    # its values at orientations

  Of the field in its harmonic form, in its tangent space; the curl of a gradient vanishes.

  Input
    v - SO3VectorField
    x - orientation, where to evaluate the curl
  Output
    C - SO3VectorFieldHarmonic
    c - SO3TangentVector, the values at x
  See also
    div, antiderivative, SO3FunHarmonic.grad"""
  v = SO3VectorFieldHarmonic(v)
  G = [differentiate(v._fhat[:, k], v.tangentSpace) for k in range(3)]
  rot = np.stack([G[2][:, 1] - G[1][:, 2], G[0][:, 2] - G[2][:, 0], G[1][:, 0] - G[0][:, 1]], axis=-1)
  c = v._rebuild(rot + v._fhat if v.tangentSpace == "left" else rot - v._fhat)
  return c if x is None else c(x)


def antiderivative(v, x=None, conservative=False, tol=1e-4):
  """the antiderivative of a gradient field, the function of mean zero whose gradient it is

    f = antiderivative(SO3VF)                         # the function, an SO3FunHarmonic
    values = antiderivative(SO3VF, x)                 # its values at orientations x
    f = antiderivative(SO3VF, conservative=True)      # without checking the field is a gradient

  The function is the inverse Laplacian of the divergence of the field, defined up to a
  constant, which is chosen to give mean zero. A field that is no gradient is refused unless
  conservative says it is one: the gradient of the result must give the field back within
  tol relative to its norm.

  Input
    SO3VF - SO3VectorField, converted to its harmonic form
    x - orientation, where to evaluate the antiderivative
  Output
    f - SO3FunHarmonic
    values - array, the values at x
  Keywords
    conservative - the field is known to be a gradient, so it is not checked
    tol - the relative tolerance of the check
  See also
    grad, div, curl, SO3VectorFieldHarmonic, SO3FunVectorField"""
  v = SO3VectorFieldHarmonic(v)
  d = div(v)
  n = _so3Degrees(d.bandwidth)
  fhat = np.zeros_like(d._fhat)
  fhat[1:] = d._fhat[1:] / -(n * (n + 1))[1:]
  f = d._rebuild(fhat)
  if not conservative and np.linalg.norm(differentiate(f._fhat, v.tangentSpace) - v._fhat) > tol * max(np.linalg.norm(v._fhat), np.finfo(float).eps):
    raise ValueError("the field is not conservative: it is not the gradient of a function")
  return f if x is None else f(x)


def grad(f, x=None, side="left", delta=0.05 * degree):
  """the gradient of a harmonic function as a harmonic field, of any other function as the
  central difference handle at 0.05 degree"""
  if isinstance(x, str):
    x, side = None, x
  side = _side(side)
  if isinstance(f, SO3FunHarmonic):
    if not f.isReal:
      raise ValueError("the gradient of a complex valued function is not a real vector field")
    field = SO3VectorFieldHarmonic(differentiate(f._fhat, side), f.CS, f.SS, side, symmetrise=False)
    return field if x is None else field(x)
  if x is not None and hasattr(f, "grad"):
    return f.grad(x, side)
  steps = [rotation.byAxisAngle(e, delta / 2) for e in (xvector, yvector, zvector)]

  def fun(o):
    r = rotation(o.data.reshape(-1, 4))
    g = np.zeros((r.size, 3))
    for i, s in enumerate(steps):
      plus = orientation._fromRotation((s * r if side == "left" else r * s).data, (f.CS, f.SS), f.antipodal)
      minus = orientation._fromRotation((s.inv() * r if side == "left" else r * s.inv()).data, (f.CS, f.SS), f.antipodal)
      g[:, i] = (np.real(f(plus)).ravel() - np.real(f(minus)).ravel()) / delta
    return g

  field = SO3VectorFieldHandle(fun, f.CS, f.SS, side, bandwidth=f.bandwidth)
  return field if x is None else field(x)


def component(v, i):
  """component(v, i) is the i-th component of the field v, a function on its component frames"""
  return SO3VectorFieldHarmonic(v).component(i)


# ------------------------------------------------------------------ pointwise operations
# by quadrature on the component grid of the first field, projected afterwards: the
# arguments are vector fields brought to its side, scalar functions on its frames and
# numbers; a vector result is a field of that side, a scalar one (reduce=True sums the
# products over the components) a function on the component frames, on the full frames
# when told it is invariant


def mapField(op, v, *args, bandwidth=None, invariant=False, reduce=False):
  """a pointwise operation on vector fields on SO(3), as a field or a function

    u = mapField(np.cross, v, w)                                  # the cross product of two fields
    f = mapField(np.multiply, v, w, reduce=True, invariant=True)  # the inner product, a function on the full frames
    u = mapField(np.multiply, v, F, bandwidth=8)                  # a field times a scalar function
    u = mapField(np.multiply, v, 2)                               # a field times a number

  The arguments are evaluated on a quadrature grid, op is applied to the arrays of values and
  the result is expanded again. op sees a field as an (n, 3) array of tangent vectors in the
  tangent space of v, a function as an (n, 1) column, which broadcasts against them, and a
  number as itself.

  An (n, 3) result is a field in the tangent space of v, one value per node a function. A
  function lives on the component frames of v, the frames with the symmetry on the tangent
  side stripped, unless invariant=True puts it on the full frames of the field.

  The bandwidth of the result is the sum of those of the operands, at most the preference
  maxSO3Bandwidth.

  Input
    op - a callable of NumPy arrays
    v - SO3VectorField
    args - further SO3VectorField, SO3Fun or numbers, on the frames of v
  Output
    u - SO3VectorFieldHarmonic
    f - SO3FunHarmonic
  Keywords
    bandwidth - the bandwidth of the result
    invariant - put a scalar result on the frames of the field rather than its component
        frames
    reduce - sum the result over the three components before it is expanded
  See also
    SO3VectorFieldHarmonic, grad, antiderivative, SO3FunVectorField"""
  others = [a for a in args if not isinstance(a, (int, float, np.number))]
  L0 = min(v.bandwidth + sum(a.bandwidth for a in others), getMTEXpref("maxSO3Bandwidth")) if bandwidth is None else int(bandwidth)
  cA, cB = v.componentFrames()
  L = bandwidthFor(L0, cA, cB)
  g = quadratureSO3Grid(L, cA, cB)
  os = orientation._fromRotation(g.nodes().data, (v.CS, v.SS))

  def evaluate(a):
    if isinstance(a, SO3VectorField):
      v._sameFrames(a)
      return _tangentValues(a(os), v.tangentSpace)
    if isinstance(a, (int, float, np.number)):
      return a
    return np.real(a(os)).reshape(-1, 1)

  values = op(evaluate(v), *(evaluate(a) for a in args))
  if reduce:
    values = np.sum(values, axis=-1)
  # three columns are a field, a single value per node a function
  if values.ndim == 2 and values.shape[-1] == 3:
    return SO3VectorFieldHarmonic(_adjointColumns(g, values, L, g.weights), v.CS, v.SS, v.tangentSpace).truncate(L0)
  values = np.broadcast_to(values, (len(os),) + np.shape(values)[1:]).reshape(len(os))
  fhat = wignerSeriesAdjoint(g, values, L)
  f = SO3FunHarmonic(fhat, v.CS, v.SS, real=True, _internal=True) if invariant else SO3FunHarmonic(fhat, cA, cB, real=True, _internal=True)
  return f.truncate(L0)


# ------------------------------------------------------------------ the handle


class SO3VectorFieldHandle(SO3VectorField):
  """a vector field on SO(3) given by a callable returning vectors or tangent vectors at
  orientations, in the left or right tangent space; bandwidth is the hint for its harmonic form

    SO3VectorFieldHandle(lambda ori: ..., cs, ss, 'left')"""

  def __init__(self, fun, *args, bandwidth=None):
    side, rest = _readSide(args)
    if isinstance(fun, SO3VectorField):
      v = fun
      fun, rest, side, bandwidth = (lambda o: v(o)), [v.CS, v.SS], v.tangentSpace, v.bandwidth
    self.CS, self.SS = _framesAmong([a for a in rest if isinstance(a, referenceFrame)])
    self.fun, self.tangentSpace = fun, side
    self.bandwidth = getMTEXpref("defaultSO3Bandwidth") if bandwidth is None else int(bandwidth)

  def eval(self, x):
    """the tangent vectors of the field at orientations, by the callable"""
    r = self._rotationsFor(x)
    o = r if isinstance(r, orientation) else orientation._fromRotation(r.data, (self.CS, self.SS))
    return self._tangent(_tangentValues(self.fun(o), self.tangentSpace), x)

  def _harmonicField(self, side, bandwidth):
    return SO3VectorFieldHarmonic(lambda o: self(o), self.CS, self.SS, side, bandwidth=self.bandwidth if bandwidth is None else bandwidth)

  # the coefficients of the harmonic form at the bandwidth hint
  @property
  def _fhat(self):
    return SO3VectorFieldHarmonic(self)._fhat

  def _scaled(self, a):
    return SO3VectorFieldHandle(lambda o: a * _tangentValues(self.fun(o), self.tangentSpace), self.CS, self.SS, self.tangentSpace, bandwidth=self.bandwidth)

  def rotate(self, r, side="left"):
    """the field turned by a rotation, from the left or with side='right' from the right, again a
    handle; an orientation turns its harmonic form

      SO3VF = rotate(SO3VF, rot)                 # turned in the specimen frame
      SO3VF = SO3VF.rotate(rot, side='right')    # turned in the crystal frame

    The new handle evaluates the old one at the orientations turned back and turns the vectors
    it returns where the rotation acts on their tangent space.

    Input
      SO3VF - SO3VectorFieldHandle
      rot - rotation, or an orientation, one
    Output
      SO3VF - SO3VectorFieldHandle, or SO3VectorFieldHarmonic for an orientation
    Keywords
      side - 'left', the specimen side, or 'right', the crystal side; MATLAB's flag 'right'
    See also
      SO3VectorField.rotate, SO3VectorFieldRBF.rotate"""
    side = _side(side)
    if isinstance(r, orientation):
      return SO3VectorFieldHarmonic(self).rotate(r, side)
    rot = rotation(r.data.reshape(4))
    M = _coordinateRotation(rot, self.tangentSpace, side)
    q = rot.inv()

    def fun(o):
      moved = orientation._fromRotation((q * rotation(o) if side == "left" else rotation(o) * q).data, (self.CS, self.SS))
      values = _tangentValues(self.fun(moved), self.tangentSpace)
      return values if M is None else values @ M.T

    CS = _rotatedFrame(self.CS, rot) if side == "right" else self.CS
    SS = _rotatedFrame(self.SS, rot) if side == "left" else self.SS
    return SO3VectorFieldHandle(fun, CS, SS, self.tangentSpace, bandwidth=self.bandwidth)

  def _reframe(self, CS, SS):
    return SO3VectorFieldHandle(self.fun, CS, SS, self.tangentSpace, bandwidth=self.bandwidth)

  def __add__(self, w):
    return SO3VectorFieldHarmonic(self) + w

  def __repr__(self):
    return aligned(f"SO3VectorFieldHandle ({framesLabel(self.CS, self.SS)})\n  tangent space: {self.tangentSpace}\n  bandwidth hint: {self.bandwidth}")

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from typing import Any
    @property
    def frameA(self) -> Any:
      ...
    @frameA.setter
    def frameA(self, value) -> None:
      ...
    @property
    def frameB(self) -> Any:
      ...
    @frameB.setter
    def frameB(self, value) -> None:
      ...


# ------------------------------------------------------------------ the radial basis field
# F(R) = sum_j mean_{s,t} (s w_j) K(R, s c_j t) for a left field, the right group of the
# frames on t and the left one on s with the weight turned along; on the right the weight
# turns by t^-1 instead. Exactly equivariant.


class SO3VectorFieldRBF(SO3VectorField):
  """a vector field on SO(3) as a sum of radial kernels at centre orientations with weight vectors,
  exactly equivariant

    SO3VF = SO3VectorFieldRBF(center, weights, psi)                 # from centres, weight vectors and a kernel
    SO3VF = SO3VectorFieldRBF(center, weights, None, 'right')      # in the right tangent space, the default kernel
    SO3VF = SO3VectorFieldRBF(t, kernel=psi)                       # fitted through tangent vectors t
    SO3VF = SO3VectorFieldRBF(F, resolution=10*degree)             # another field fitted on a grid

  The field is the sum over the centres of the kernel at the misorientation to the centre,
  times the centre's weight vector. It is summed over the symmetric equivalents, so that it
  is equivariant under the frames.

  Through tangent vectors t, an SO3TangentVector at reference orientations, the weights solve
  the dense linear system of the kernel between the references, which suits hundreds of
  centres. Another field is sampled on an equispaced grid of the resolution and fitted the
  same way, with a kernel of halfwidth the resolution unless one is given.

  MATLAB stores the three components as SO3FunRBF; here the weights are one vector per centre.

  Input
    center - orientation, the centres; or an SO3TangentVector or SO3VectorField to fit
    weights - (n, 3) array, one weight vector per centre
    psi - SO3Kernel, a de la Vallee Poussin kernel of halfwidth 10 degree unless given
  Output
    SO3VF - SO3VectorFieldRBF
  Keywords
    kernel - the kernel of a fit
    resolution - the grid spacing of the fit of another field, 5 degree unless given
  See also
    SO3VectorFieldHarmonic, SO3FunRBF, SO3TangentVector, SO3FunVectorField"""

  def __init__(self, center, weights=None, psi=None, *args, kernel=None, resolution=None):
    side, _ = _readSide(args)
    if isinstance(center, SO3TangentVector):
      self._fit(center, kernel)
      return
    if isinstance(center, SO3VectorField):
      v = center
      res = 5 * degree if resolution is None else resolution
      psi = kernel if kernel is not None else SO3DeLaValleePoussinKernel(halfwidth=res)
      g = equispacedSO3Grid(v.CS, v.SS, resolution=res)
      self._fit(v(orientation._fromRotation(g.data, (v.CS, v.SS))), psi)
      return
    self.center = center.flatten()
    self.weights = np.asarray(weights, dtype=float).reshape(-1, 3)
    if len(self.weights) != self.center.size:
      raise ValueError("one weight vector per centre")
    self.psi = SO3DeLaValleePoussinKernel(halfwidth=10 * degree) if psi is None else psi
    self.tangentSpace = side

  # the field through given tangent vectors: the weights solve the dense linear system of
  # the kernel between the references, a 3x3 block per pair summed over the equivalents
  # of the centre with the turned identity, so this is for hundreds of centres
  def _fit(self, t, kernel):
    ref = t.oriRef
    if not isinstance(ref, orientation) or ref.size != t.size:
      raise ValueError("a field needs one reference orientation per tangent vector")
    psi = SO3DeLaValleePoussinKernel(halfwidth=10 * degree) if kernel is None else kernel
    center = orientation._fromRotation(ref.data.reshape(-1, 4), (ref.CS, ref.SS))
    cs = center.data.reshape(-1, 4)
    n = len(cs)
    L, R = reductionGroups(center)
    cut = supportCosine(psi)
    A = np.zeros((3 * n, 3 * n))
    for l in L:
      for r in R:
        cj = _qmul(_qmul(l[None], cs), r[None])
        M = (rotation(l).matrix() if t.side == "left" else rotation(qconj(r)).matrix()) / (len(L) * len(R))
        d = np.abs(cs @ cj.T)
        k = psi(np.minimum(d, 1.0)) * (d > cut)
        A += np.kron(k, M)
    w = np.linalg.lstsq(A, t.data.reshape(-1), rcond=None)[0]
    self.center, self.weights, self.psi, self.tangentSpace = center, w.reshape(-1, 3), psi, t.side

  @property
  def CS(self):
    """the crystal frame of the field, that of the centres"""
    return self.center.CS

  @property
  def SS(self):
    """the specimen frame of the field, that of the centres"""
    return self.center.SS

  @property
  def bandwidth(self):
    """the bandwidth of the kernel"""
    return self.psi.bandwidth

  # the equivalents of the centres with their turned weights
  def _expandedCentres(self):
    L, R = reductionGroups(self.center)
    c = self.center.data.reshape(-1, 4)
    rots, w = [], []
    for l in L:
      Ml = rotation(l).matrix()
      for r in R:
        Mr = rotation(qconj(r)).matrix()
        rots.append(_qmul(_qmul(l[None], c), r[None]))
        w.append(self.weights @ (Ml.T if self.tangentSpace == "left" else Mr.T))
    n = len(L) * len(R)
    return np.concatenate(rots), np.concatenate(w) / n

  def eval(self, x):
    """the tangent vectors of the field at orientations, the kernel sums over the symmetric
    equivalents of the centres"""
    r = self._rotationsFor(x)
    q = r.data.reshape(-1, 4)
    cs, w = self._expandedCentres()
    cut = supportCosine(self.psi)
    out = np.zeros((len(q), 3))
    for start in range(0, len(q), 1024):
      D = np.abs(q[start:start + 1024] @ cs.T)
      K = self.psi(np.minimum(D, 1.0)) * (D > cut)
      out[start:start + 1024] = K @ w
    return self._tangent(out, x)

  # the harmonic form: the point measure of the centres convolved with the kernel, then the
  # equivariant projection does the averaging over the groups
  def _harmonicField(self, side, bandwidth):
    L = min(self.psi.bandwidth, getMTEXpref("maxSO3Bandwidth")) if bandwidth is None else int(bandwidth)
    fhat = _adjointColumns(rotation(self.center), self.weights, L, np.ones(self.center.size))
    fhat *= _so3PerDegree(self.psi.calcFourier(L) / (2 * np.arange(L + 1) + 1))[:, None]
    return SO3VectorFieldHarmonic(fhat, self.CS, self.SS, self.tangentSpace).toSide(side)

  # the coefficients of the harmonic form at the bandwidth of the kernel
  @property
  def _fhat(self):
    return SO3VectorFieldHarmonic(self)._fhat

  def _scaled(self, a):
    return SO3VectorFieldRBF(self.center, a * self.weights, self.psi, self.tangentSpace)

  def __add__(self, w):
    if isinstance(w, SO3VectorFieldRBF) and w.tangentSpace == self.tangentSpace and w.psi == self.psi:
      self._sameFrames(w)
      center = orientation._fromRotation(np.concatenate([self.center.data, w.center.data]), (self.CS, self.SS))
      return SO3VectorFieldRBF(center, np.concatenate([self.weights, w.weights]), self.psi, self.tangentSpace)
    return SO3VectorFieldHarmonic(self) + w

  def rotate(self, r, side="left"):
    """the field turned by a rotation, from the left or with side='right' from the right, the centres
    and weights moved; an orientation turns its harmonic form

      SO3VF = rotate(SO3VF, rot)                 # turned in the specimen frame
      SO3VF = SO3VF.rotate(rot, side='right')    # turned in the crystal frame

    The centres are multiplied by the rotation, from the left or the right, and the weight
    vectors turned where the rotation acts on the tangent space, so the result is again a sum of
    kernels. A symmetry the rotation does not map onto itself is dropped with a warning.

    Input
      SO3VF - SO3VectorFieldRBF
      rot - rotation, or an orientation, one
    Output
      SO3VF - SO3VectorFieldRBF, or SO3VectorFieldHarmonic for an orientation
    Keywords
      side - 'left', the specimen side, or 'right', the crystal side; MATLAB's flag 'right'
    See also
      SO3VectorField.rotate, SO3VectorFieldHandle.rotate"""
    side = _side(side)
    if isinstance(r, orientation):
      return SO3VectorFieldHarmonic(self).rotate(r, side)
    rot = rotation(r.data.reshape(4))
    M = _coordinateRotation(rot, self.tangentSpace, side)
    rots = rot * rotation(self.center) if side == "left" else rotation(self.center) * rot
    CS = _rotatedFrame(self.CS, rot) if side == "right" else self.CS
    SS = _rotatedFrame(self.SS, rot) if side == "left" else self.SS
    center = orientation._fromRotation(rots.data, (CS, SS))
    return SO3VectorFieldRBF(center, self.weights if M is None else self.weights @ M.T, self.psi, self.tangentSpace)

  def _reframe(self, CS, SS):
    return SO3VectorFieldRBF(orientation._fromRotation(self.center.data, (CS, SS)), self.weights, self.psi, self.tangentSpace)

  def __repr__(self):
    return aligned(f"SO3VectorFieldRBF ({framesLabel(self.CS, self.SS)})\n  tangent space: {self.tangentSpace}\n  kernel: {self.psi}\n  center: {self.center.size} orientations")

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from typing import Any
    @property
    def frameA(self) -> Any:
      ...
    @frameA.setter
    def frameA(self, value) -> None:
      ...
    @property
    def frameB(self) -> Any:
      ...
    @frameB.setter
    def frameB(self, value) -> None:
      ...


def interpolateField(t, **kwargs):
  """the tangent vectors of a field at orientations by least squares componentwise on the
  component frames"""
  from .density import interpolate
  ref = t.oriRef
  if not isinstance(ref, orientation):
    raise ValueError("a field needs one reference orientation per tangent vector")
  cA, cB = componentFrames(ref.CS, ref.SS, t.side)
  os = orientation._fromRotation(ref.data, (cA, cB))
  fs = [interpolate(os, t.data.reshape(-1, 3)[:, i], **kwargs) for i in range(3)]
  return SO3VectorFieldHarmonic(fs, ref.CS, ref.SS, t.side)


# the two frames under their names of the Julia port, frameA and frameB
installFramePair((SO3VectorFieldHarmonic, SO3VectorFieldHandle, SO3VectorFieldRBF))

__all__ = ["SO3VectorFieldHarmonic", "SO3VectorFieldHandle", "SO3VectorFieldRBF", "componentFrames", "component", "curl",
           "div", "antiderivative", "mapField", "differentiate", "equivariantProjection"]
