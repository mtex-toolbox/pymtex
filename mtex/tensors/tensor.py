# tensors of any rank in a reference frame (MATLAB TensorAnalysis/@tensor and the classes
# derived from it; docs/plans/functions.md P4c)
#
#   T = tensor(M, cs, rank=2)                  the coefficients M with the list dimensions in front
#   C = stiffnessTensor(V, cs, density=3.3)    a 6 x 6 Voigt matrix, or the 3 x 3 x 3 x 3 array
#   T.M, T.rank, T.frame, T.CS, Voigt(C), Kelvin(C)
#   EinsteinSum(C, [1, 2, -1, -2], eps, [-1, -2]), colon(C, eps), dyad(a, b)
#   rotate(T, r), ori * T, symmetrise(T), checkSymmetry(T), directionalMagnitude(T, v)
#   eig(T), svd(T), polar(T), expm(T), logm(T), sqrtm(T)
#   tensor.eye(rank=4), tensor.rand(5, rank=2), tensor.leviCivita, mean(Ts)
#
# A tensor of rank R is an element of (R^3)^(x R) written in a reference frame; a rotation
# acts on every index. A tensor turned by anything but an element of the group its frame
# claims is written in the frame without the group. What MTEX splits into a class per
# physical quantity is a subclass here fixing the rank, the Voigt factors of its notation
# and the class its inverse takes; a contraction returns a plain tensor, the physical
# products name their class themselves.
from __future__ import annotations

from typing import TYPE_CHECKING
import itertools
import math

import numpy as np

from ..geometry._array import MtexArray
from ..geometry.display import display, frameLabel, header, numString, sizeString, table
from ..geometry.frames import crystalFrame, referenceFrame, specimenFrame
from ..geometry.orientation import orientation
from ..geometry.products import dropGroup, fitFrame
from ..geometry.quaternion import orthogonalTransform, quaternion, rotation
from ..geometry.vector3d import classproperty, vector3d
from .kernels import turnBlocks

# the pair index of the Voigt notation: 11 22 33 23 13 12
VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
LETTERS = "abcdefghijklmnopqrstuvwxyz"


# ------------------------------------------------------------------ the Voigt and Kelvin notation

def _voigtWeights(double):
  return np.array([1, 1, 1, 2, 2, 2], dtype=float) if double else np.ones(6)


KELVIN_WEIGHTS = np.array([1, 1, 1, math.sqrt(2), math.sqrt(2), math.sqrt(2)])


# the 6 x 6 matrix of a rank four, the 3 x 6 of a rank three, the six components of a rank
# two, the factors of the off diagonal entries given per side
def _toVoigt(M, rank, w):
  a = np.array([p[0] for p in VOIGT])
  b = np.array([p[1] for p in VOIGT])
  if rank == 4:
    return M[..., a[:, None], b[:, None], a[None, :], b[None, :]] * w[:, None] * w[None, :]
  if rank == 3:
    return M[..., :, a, b] * w
  if rank == 2:
    return M[..., a, b] * w
  raise ValueError("the Voigt notation is for tensors of rank two, three and four")


# the tensor of a Voigt matrix, the factors taken out again and the pair symmetry filled in,
# as one gather through the Voigt index of every pair
def _fromVoigt(V, rank, w):
  V = np.asarray(V, dtype=float)
  pair = np.zeros((3, 3), dtype=int)
  for i, (a, b) in enumerate(VOIGT):
    pair[a, b] = pair[b, a] = i
  if rank == 4:
    I, J = pair.reshape(9, 1), pair.reshape(1, 9)
    M = (V / (w[:, None] * w[None, :]))[..., I, J]
  elif rank == 3:
    M = (V / w)[..., :, pair.reshape(9)]
  else:
    M = (V / w)[..., pair.reshape(9)]
  return M.reshape(M.shape[:-2 if rank == 4 or rank == 3 else -1] + (3,) * rank)


def Voigt(T):
  """the Voigt matrix of a tensor: 6 x 6 for rank four, 3 x 6 for rank three, six components
  for rank two, the off diagonal entries doubled where the class prescribes it"""
  return _toVoigt(T.M, T.rank, _voigtWeights(T.doubleConvention))


def Kelvin(T):
  """the Kelvin matrix, the factor sqrt(2) on every off diagonal side"""
  return _toVoigt(T.M, T.rank, KELVIN_WEIGHTS)


# ------------------------------------------------------------------ the operands of a contraction

# the coefficient array of an operand and, for a tensor, its frame
def _operand(x):
  if isinstance(x, tensor):
    return x.M, x.rank, x.frame
  if isinstance(x, vector3d):
    return x.data, 1, x.frame
  if isinstance(x, quaternion):
    return x.matrix(), 2, None
  a = np.asarray(x, dtype=float)
  return a, a.ndim, None


def _readArgs(args):
  frames = [a for a in args if isinstance(a, referenceFrame)]
  rest = [a for a in args if not isinstance(a, referenceFrame)]
  return (frames[0] if frames else None), rest


# ------------------------------------------------------------------ the container

class tensor(MtexArray):
  """a tensor of rank R in a reference frame, or a list of them

    tensor(M, cs, rank=2)                     the coefficients, the list dimensions in front
    tensor(v), tensor(rot)                    a vector as a rank one, a rotation as a rank two
    tensor(V, rank=4)                         a 6 x 6 Voigt matrix, doubleConvention=True for the compliance factors
    T.M, T.rank, T.frame, T.CS, T.isSymmetric
  """
  _carried = ("rank", "_ncoord", "_frame", "doubleConvention", "density", "unit", "name")
  _rank = None
  _double = False
  _symmetricRank2 = False

  def __init__(self, M, *args, rank=None, doubleConvention=None, frame=None, density=None, unit=None, name=None, Kelvin=False,
               dtype=None):
    fr, rest = _readArgs(args)
    frame = fr if frame is None else frame
    for a in rest:
      if a is None:
        continue
      if isinstance(a, str) and a.lower() == "doubleconvention":
        doubleConvention = True
      else:
        raise TypeError(f"tensor does not understand the argument {a!r}")
    if isinstance(M, tensor):
      rank = M.rank if rank is None else rank
      if self._rank is not None and rank != self._rank:
        raise ValueError(f"a {type(self).__name__} has rank {self._rank}, not {rank}")
      data, frame = M.data.copy(), M.frame if frame is None else frame
      doubleConvention = M.doubleConvention if doubleConvention is None and type(self) is tensor else doubleConvention
      density, unit, name = M.density if density is None else density, M.unit if unit is None else unit, M.name if name is None else name
    elif isinstance(M, vector3d):
      data, rank = M.data.copy(), 1
      frame = M.frame if frame is None else frame
    elif isinstance(M, quaternion):
      data, rank = M.matrix().reshape(M.shape + (9,)), 2
    else:
      A = np.asarray(M, dtype=float if dtype is None else dtype)
      rank = self._rank if rank is None else int(rank)
      double = self._double if doubleConvention is None else bool(doubleConvention)
      w = KELVIN_WEIGHTS if Kelvin else _voigtWeights(double)
      if A.ndim >= 2 and A.shape[-2:] == (6, 6) and rank in (None, 4):
        A, rank = _fromVoigt(A, 4, w), 4
      elif A.ndim >= 2 and A.shape[-2:] == (3, 6) and rank in (None, 3):
        A, rank = _fromVoigt(A, 3, w), 3
      elif rank is None:
        rank = 0
        while rank < min(A.ndim, 4) and A.shape[A.ndim - 1 - rank] == 3:
          rank += 1
      if A.ndim < rank or any(s != 3 for s in A.shape[A.ndim - rank:]):
        raise ValueError(f"a tensor of rank {rank} has {rank} indices of dimension 3, not the shape {A.shape}")
      data = A.reshape(A.shape[:A.ndim - rank] + (3 ** rank,))
    self.rank = int(rank)
    self._ncoord = 3 ** self.rank
    super().__init__(data)
    self._frame = specimenFrame.default if frame is None else frame
    self.doubleConvention = self._double if doubleConvention is None else bool(doubleConvention)
    self.density, self.unit, self.name = density, unit, name

  # a tensor of another class with these coefficients, in the frame of this one
  def _as(self, cls, data, frame=None):
    obj = object.__new__(cls)
    obj.data = data
    obj.rank = int(round(math.log(data.shape[-1], 3))) if data.shape[-1] > 1 else 0
    obj._ncoord = data.shape[-1]
    obj._frame = self._frame if frame is None else frame
    obj.doubleConvention = cls._double if cls is not tensor else self.doubleConvention
    obj.density, obj.unit, obj.name = self.density, self.unit, self.name
    return obj

  # ------------------------------------------------------------------ accessors

  @property
  def M(self):
    """the coefficients as an array of shape (..., 3, ..., 3), the rank's indices last"""
    return self.data.reshape(self.shape + (3,) * self.rank)

  @M.setter
  def M(self, value):
    """sets the coefficients from an array of shape (..., 3, ..., 3), the shape of the list kept"""
    self.data = np.asarray(value, dtype=float).reshape(self.shape + (self._ncoord,))

  @property
  def frame(self):
    """the frame the tensor is written in, a crystal frame or the specimen frame"""
    return self._frame

  @frame.setter
  def frame(self, fr):
    """sets the frame the tensor is written in, the coefficients kept"""
    if not isinstance(fr, referenceFrame):
      raise TypeError("the frame of a tensor has to be a referenceFrame")
    self._frame = fr

  @property
  def CS(self):
    """the frame the tensor is written in, MATLAB's T.CS"""
    return self._frame

  @CS.setter
  def CS(self, fr):
    """sets the frame the tensor is written in, the coefficients kept"""
    self.frame = fr

  @property
  def isSymmetric(self):
    """whether every tensor of rank two or four equals its transpose, up to a relative 1e-6"""
    if self.rank not in (2, 4):
      raise ValueError("symmetry is checked on tensors of rank two and four")
    return bool(np.all(np.linalg.norm((self - self.transpose()).data, axis=-1) / (1e-12 + np.linalg.norm(self.data, axis=-1)) < 1e-6))

  @property
  def isSkewSymmetric(self):
    """whether every tensor equals minus its transpose, up to a relative 1e-6"""
    return bool(np.all(np.linalg.norm((self + self.transpose()).data, axis=-1) / (1e-12 + np.linalg.norm(self.data, axis=-1)) < 1e-6))

  def __repr__(self):
    # a crystal tensor is headed by its frame, a specimen one by the convention it is drawn in, as a vector is
    d = self.density
    lines = display(header(type(self).__name__, frameLabel(self._frame)), [
      ("name", self.name),
      ("density", None if d is None else numString(d, digits=5) if np.ndim(d) == 0 else ", ".join(numString(x, digits=5) for x in np.ravel(d))),
      ("unit", self.unit), ("rank", f"{self.rank} ({sizeString((3,) * self.rank)})" if self.rank else "0"), ("size", sizeString(self.shape) if self.ndim else None),
      ("doubleConvention", "true" if type(self) is tensor and self.doubleConvention else None)]).split("\n")
    # a single tensor shows its coefficients: a number, a column, a matrix, the compact or the Voigt matrix
    if self.ndim == 0:
      A = self.M.reshape(1, 1) if self.rank == 0 else self.M.reshape(3, 1) if self.rank == 1 else self.M if self.rank == 2 else Voigt(self)
      finite = np.abs(A[np.isfinite(A)])
      m = float(finite.max()) if finite.size else 0.0
      p = 0 if m == 0 else math.floor(math.log10(m))
      scale = p if (p >= 3 or p <= -2) else 0
      title = {3: "tensor in compact matrix form", 4: "tensor in Voigt matrix representation"}.get(self.rank)
      if title or scale:
        lines.append("  " + ((title + ": " if title else "") + f"*10^{scale}" if scale else title))
      S = A / 10.0 ** scale
      lines.append(table([""] * S.shape[1], [S[:, j] for j in range(S.shape[1])], decimals=4).split("\n", 1)[1])
    return "\n".join(lines)

  # ------------------------------------------------------------------ the algebra

  def _sameFrame(self, other):
    a, b = self._frame, other._frame
    if a is not b and not (type(a) is type(b) and a.isAligned(b)):
      raise ValueError("the tensors are written in different frames")

  def _rebuild(self, data, cls=None):
    return self._as(type(self) if cls is None else cls, data)

  def __add__(self, other):
    if isinstance(other, tensor):
      self._sameFrame(other)
      if other.rank != self.rank:
        raise ValueError("tensors of different rank cannot be added")
      cls = type(self) if type(self) is type(other) or type(other) is tensor else type(other) if type(self) is tensor else tensor
      return self._as(cls, self.data + other.data)
    return self._rebuild(self.data + np.asarray(other, dtype=float)[..., None])

  __radd__ = __add__

  def __sub__(self, other):
    return self + (-other)

  def __rsub__(self, other):
    return (-self) + other

  def __neg__(self):
    return self._rebuild(-self.data)

  # a number scales, a rank two tensor times a rank two is the matrix product, times a
  # vector its image, a rank four times a rank two the double contraction
  def __mul__(self, other):
    if isinstance(other, tensor):
      self._sameFrame(other)
      if self.rank == 2 and other.rank == 2:
        return self._as(tensor, _flatten(np.einsum("...ij,...jk->...ik", self.M, other.M), 2))
      if self.rank == 2 and other.rank == 1:
        return self._as(tensor, np.einsum("...ij,...j->...i", self.M, other.M))
      if self.rank == 4 and other.rank == 2:
        return self._contractedProduct(other)
      raise ValueError("for products between tensors use EinsteinSum")
    if isinstance(other, vector3d):
      if self.rank != 2:
        raise ValueError("a tensor times a vector needs rank two")
      if other.frame is not None and other.frame is not self._frame and not other.frame.isAligned(self._frame):
        raise ValueError("the vector is written in another frame than the tensor")
      return vector3d(np.einsum("...ij,...j->...i", self.M, other.data), frame=self._frame)
    if isinstance(other, MtexArray) or hasattr(other, "rotate"):
      return NotImplemented
    return self._rebuild(self.data * np.asarray(other, dtype=float)[..., None])

  def __rmul__(self, other):
    if isinstance(other, (quaternion, orthogonalTransform)):
      return self.rotate(other)
    if isinstance(other, MtexArray):
      return NotImplemented
    return self._rebuild(self.data * np.asarray(other, dtype=float)[..., None])

  # the double contraction with its class: a stiffness on a strain is a stress
  def _contractedProduct(self, other):
    return colon(self, other)

  def times(self, other):
    """MATLAB's .*: the product of corresponding coefficients, the rank kept"""
    if isinstance(other, tensor):
      self._sameFrame(other)
      if other.rank != self.rank:
        raise ValueError("the componentwise product is of tensors of one rank")
      return self._rebuild(self.data * other.data)
    return self._rebuild(self.data * np.asarray(other, dtype=float)[..., None])

  def __truediv__(self, a):
    if isinstance(a, tensor):
      raise ValueError("a tensor is divided by a number")
    return self._rebuild(self.data / np.asarray(a, dtype=float)[..., None])

  def __eq__(self, other):
    return isinstance(other, tensor) and other.rank == self.rank and other._frame is self._frame and np.array_equal(self.data, other.data)

  def __ne__(self, other):
    return not self == other

  def isclose(self, other, rtol=1e-8, atol=0.0):
    """whether two tensors of one rank and shape agree coefficientwise within rtol and atol"""
    return isinstance(other, tensor) and other.rank == self.rank and self.data.shape == other.data.shape and bool(np.allclose(self.data, other.data, rtol=rtol, atol=atol))

  def norm(self):
    """the Frobenius norm of every tensor, over all its coefficients"""
    return np.linalg.norm(self.data, axis=-1)

  def normalize(self):
    """the tensors divided by their Frobenius norms"""
    return self / self.norm()

  def trace(self):
    """the trace of a rank two tensor, or of the Voigt matrix of a rank four one"""
    if self.rank == 2:
      return np.einsum("...ii->...", self.M)
    if self.rank == 4:
      return np.einsum("...ii->...", Voigt(self))
    raise ValueError("the trace is of a tensor of rank two or four")

  def det(self):
    """the determinant of a rank two tensor"""
    if self.rank != 2:
      raise ValueError("the determinant is of a tensor of rank two")
    return np.linalg.det(self.M)

  def transpose(self):
    """the transpose swaps the indices of a rank two and the pairs of a rank four"""
    if self.rank == 2:
      return self._rebuild(_flatten(np.swapaxes(self.M, -1, -2), 2))
    if self.rank == 4:
      return self._rebuild(_flatten(np.moveaxis(self.M, (-4, -3, -2, -1), (-2, -1, -4, -3)), 4))
    if self.rank <= 1:
      return self
    raise ValueError("the transpose is of a tensor of rank two or four")

  @property
  def T(self):
    """the transpose, the indices of a rank two and the index pairs of a rank four swapped"""
    return self.transpose()

  ctranspose = transpose

  def sym(self):
    """the symmetric and the antisymmetric part, the mean over the permutations of the indices"""
    if self.rank <= 1:
      return self
    M = self.M
    lead = M.ndim - self.rank
    out = np.zeros_like(M)
    perms = list(itertools.permutations(range(self.rank)))
    for p in perms:
      out += np.transpose(M, tuple(range(lead)) + tuple(lead + k for k in p))
    return self._rebuild(_flatten(out / len(perms), self.rank))

  def antiSym(self):
    """the antisymmetric part (T - T^t) / 2 of a rank two tensor"""
    if self.rank != 2:
      raise ValueError("the antisymmetric part is of a tensor of rank two")
    return self._rebuild(_flatten((self.M - np.swapaxes(self.M, -1, -2)) / 2, 2))

  def inv(self):
    """the inverse: rank two by the matrix, rank four through the Voigt matrix, which swaps the
    convention and with it the class, a stiffness inverting to a compliance"""
    cls = self._inverseClass()
    if self.rank == 2:
      out = self._as(cls, _flatten(np.linalg.inv(self.M), 2))
    elif self.rank == 4:
      V = np.linalg.inv(Voigt(self))
      double = cls._double if cls is not tensor else not self.doubleConvention
      out = self._as(cls, _flatten(_fromVoigt(V, 4, _voigtWeights(double)), 4))
      out.doubleConvention = double
    else:
      raise ValueError("the inverse is of a tensor of rank two or four")
    out.unit = _inverseUnit(self.unit)
    return out

  def _inverseClass(self):
    return type(self)

  def eig(self):
    """the eigenvalues in ascending order and the eigenvectors as axes of the frame; a rank
    four tensor is diagonalised in its Voigt matrix"""
    if self.rank == 2:
      M = self.M
      if self.isSymmetric:
        vals, vecs = np.linalg.eigh(M)
      else:
        vals, vecs = np.linalg.eig(M)
        order = np.argsort(np.real(vals), axis=-1)
        vals = np.take_along_axis(vals, order, axis=-1)
        vecs = np.take_along_axis(vecs, order[..., None, :], axis=-1)
      v = vector3d(np.moveaxis(vecs, -1, -2), antipodal=True, frame=self._frame)
      v.dispStyle = "xyz"
      return vals, v
    if self.rank == 4:
      vals, vecs = np.linalg.eigh(Voigt(self))
      return vals, vecs
    raise ValueError("eigenvalues are of a tensor of rank two or four")

  def eigvals(self):
    """the eigenvalues in ascending order, of a rank two tensor or of the Voigt matrix of a rank four"""
    return self.eig()[0]

  # the matrix functions: a rank two tensor as its matrix, a rank four one as its Kelvin
  # matrix, whose product is the double contraction, the class kept
  def _matrixFunction(self, f):
    if self.rank == 2:
      return self._rebuild(_flatten(_perMatrix(f, self.M), 2))
    if self.rank == 4:
      return self._rebuild(_flatten(_fromVoigt(_perMatrix(f, Kelvin(self)), 4, KELVIN_WEIGHTS), 4))
    raise ValueError("matrix functions are of tensors of rank two or four")

  def expm(self):
    """the matrix exponential of a rank two tensor, or of a rank four one through its Kelvin matrix,
    the class kept"""
    from scipy.linalg import expm
    return self._matrixFunction(expm)

  def logm(self):
    """the matrix logarithm of a rank two tensor, or of a rank four one through its Kelvin matrix, the
    class kept"""
    from scipy.linalg import logm
    out = self._matrixFunction(logm)
    out.name = None if self.name is None else f"log {self.name}"
    return out

  def sqrtm(self):
    """the matrix square root of a rank two tensor, or of a rank four one through its Kelvin matrix,
    the class kept"""
    from scipy.linalg import sqrtm
    return self._matrixFunction(sqrtm)

  def svd(self):
    """the singular value decomposition: of a rank two tensor the left singular vectors, the
    singular values from the largest and the right singular vectors, T v_k = s_k u_k; of a
    rank four one the rotation of the higher order SVD and the core, the tensor written in
    the basis of its mode singular vectors, and no V"""
    if self.rank == 2:
      U, s, Vt = np.linalg.svd(self.M)
      u = vector3d(np.swapaxes(U, -1, -2), frame=self._frame)
      v = vector3d(Vt, frame=self._frame)
      u.dispStyle = v.dispStyle = "xyz"
      return u, s, v
    if self.rank == 4:
      U = np.linalg.svd(self.M.reshape(self.shape + (3, 27)))[0]
      U[..., :, 2] *= np.sign(np.linalg.det(U))[..., None]
      return rotation.byMatrix(U), self._rebuild(_applyMatrix(self.M, np.swapaxes(U, -1, -2), 4)), None
    raise ValueError("the singular value decomposition is of tensors of rank two or four")

  def polar(self):
    """the polar decomposition T = R SR = SL R of a rank two tensor, R orthogonal, SL and SR
    symmetric"""
    if self.rank != 2:
      raise ValueError("the polar decomposition is of tensors of rank two")
    U, s, Vt = np.linalg.svd(self.M)
    V = np.swapaxes(Vt, -1, -2)
    R = U @ Vt
    SL = (U * s[..., None, :]) @ np.swapaxes(U, -1, -2)
    SR = (V * s[..., None, :]) @ Vt
    return tuple(self._as(tensor, _flatten(A, 2)) for A in (R, SL, SR))

  # ------------------------------------------------------------------ the group action

  def rotate(self, r):
    """the tensor turned by a rotation, an orthogonal transform or an orientation; the frame
    keeps its group only under the group's own elements, and an orientation carries the
    tensor into its specimen frame, without the specimen group"""
    if isinstance(r, orientation):
      t = fitFrame(self._frame, r.CS) if isinstance(self._frame, crystalFrame) else None
      q = r if t is None else rotation(r.data) * t
      return self._as(type(self), _applyMatrix(self.M, rotation(q.data).matrix(), self.rank), r.SS.stripSym())
    if isinstance(r, orthogonalTransform):
      fr = self._frame if all(x in self._frame for x in r.flatten()) else self._frame.stripSym()
      return self._as(type(self), _applyMatrix(self.M, r.matrix(), self.rank), fr)
    if isinstance(r, quaternion):
      return self._as(type(self), _applyMatrix(self.M, rotation(r.data).matrix(), self.rank), dropGroup(rotation(r.data), self._frame))
    Q = np.asarray(r, dtype=float)
    if Q.shape[-2:] != (3, 3):
      raise TypeError("rotate takes a rotation, an orientation or a 3 x 3 matrix")
    return self._as(type(self), _applyMatrix(self.M, Q, self.rank), self._frame.stripSym())

  def transformReferenceFrame(self, fr):
    """the same coefficients read in another frame, the numbers changed by the rigid part of
    the transition between the two bases"""
    ok, M = self._frame.isCompatible(fr)
    if not ok:
      raise ValueError("the two frames are not related by a rigid change of coordinates")
    U, _, Vt = np.linalg.svd(M)
    return self._as(type(self), _applyMatrix(self.M, U @ Vt, self.rank), fr)

  def symmetrise(self):
    """the mean over the group of the frame, and whether the tensor is invariant under it"""
    Q = self._frame.rot.matrix()
    if len(Q) <= 1:
      return self
    out = np.zeros(self.data.shape)
    for q in Q:
      out += _applyMatrix(self.M, q, self.rank)
    return self._rebuild(out / len(Q))

  def checkSymmetry(self, atol=1e-6):
    """whether the tensor is invariant under every element of the group of its frame, up to atol
    relative to the largest norm"""
    scale = atol * max(float(np.max(self.norm())) if self.size else 0.0, 1e-12)
    return all(np.allclose(_applyMatrix(self.M, q, self.rank), self.data, atol=scale) for q in self._frame.rot.matrix())

  # ------------------------------------------------------------------ magnitudes

  def directionalMagnitude(self, v=None):
    """the tensor contracted with the unit direction on every index; without a direction the
    function on the sphere, a polynomial of the rank's degree"""
    if v is None:
      from ..functions.s2fun import S2FunHarmonic
      return S2FunHarmonic(lambda x: self.directionalMagnitude(x), self._frame, bandwidth=max(self.rank, 1), real=True,
                           antipodal=self.rank % 2 == 0)
    v = v if isinstance(v, vector3d) else vector3d(v)
    if v.frame is not None and v.frame is not self._frame and not v.frame.isAligned(self._frame):
      raise ValueError("the direction is written in another frame than the tensor")
    u = v.normalize().data
    R = self.rank
    if R == 0:
      return np.broadcast_to(self.M, np.broadcast_shapes(self.shape, v.shape)).copy()
    a, b = (R + 1) // 2, R // 2
    M2 = self.M.reshape(self.shape + (3 ** a, 3 ** b))
    Ua = _vecPower(u, a)
    W = Ua @ M2 if self.ndim == 0 else (Ua[..., None, :] @ M2)[..., 0, :]
    return np.sum(W * _vecPower(u, b), axis=-1) if b else W[..., 0]

  # ------------------------------------------------------------------ lists

  def mean(self, *args, weights=None, **kwargs):
    """the mean over the list, weighted or not; with an ODF the texture average"""
    if args and not isinstance(args[0], str):
      from ..analysis.texture import calcTensor
      return calcTensor(args[0], self, *args[1:], weights=weights, **kwargs)
    # MATLAB's other outputs: the inverse of the mean inverse, the Hill mean, all three
    which = next((a for a in args if a in ("Reuss", "Hill", "all")), None)
    if which is not None:
      V, R = self.mean(weights=weights), self.inv().mean(weights=weights).inv()
      return {"Reuss": R, "Hill": (V + R) / 2, "all": (V, R, (V + R) / 2)}[which]
    if self.ndim == 0:
      return self
    flat = self.data.reshape(-1, self._ncoord)
    if weights is None:
      out = flat.mean(axis=0)
    else:
      w = np.asarray(weights, dtype=float).ravel()
      if w.size != flat.shape[0]:
        raise ValueError("one weight per tensor")
      out = (w[:, None] * flat).sum(axis=0) / np.sum(w)
    t = self._as(type(self), out)
    if self.density is not None and np.ndim(self.density):
      d = np.asarray(self.density, dtype=float).ravel()
      t.density = float(d.mean() if weights is None else (np.asarray(weights, dtype=float).ravel() * d).sum() / np.sum(weights))
    return t

  def sum(self, axis=None):
    """the sum of all tensors, or along one axis of the array, MATLAB's sum(T, 2)"""
    if axis is None:
      return self._as(type(self), self.data.reshape(-1, self._ncoord).sum(axis=0))
    return self._as(type(self), self.data.sum(axis=axis % self.ndim))

  @classmethod
  def cat(cls, *arrays, axis=0):
    """the list of tensors of different densities keeps one density per tensor, as MATLAB's
    [C1, C2] does, which is what a weighted mean averages"""
    dens = [a.density for a in arrays]
    if all(d is None for d in dens) or all(d is not None and np.array_equal(d, dens[0]) for d in dens):
      return super().cat(*arrays, axis=axis)
    if any(d is None for d in dens):
      raise ValueError("cannot combine tensors with and without a density")
    parts = [a if a.ndim > 0 else a.reshape(1) for a in arrays]
    bare = []
    for a in parts:
      b = a._new(a.data)
      b.density = None
      bare.append(b)
    out = super().cat(*bare, axis=axis)
    out.density = np.concatenate([np.broadcast_to(np.asarray(a.density, dtype=float), a.shape) for a in parts], axis=axis)
    return out

  # ------------------------------------------------------------------ factories

  @classmethod
  def eye(cls, *args, rank=None, **kwargs):
    """the identity tensor, rank two the unit matrix, rank four the symmetric identity on rank two
    tensors; the rank of the class unless rank= is given

      tensor.eye(cs)
      stiffnessTensor.eye()
      tensor.eye(rank=4)"""
    fr, rest = _readArgs(args)
    rank = (cls._rank or 2) if rank is None else int(rank)
    if rank == 2:
      return cls(np.eye(3), fr, **kwargs)
    if rank == 4:
      i, j, k, l = np.indices((3, 3, 3, 3))
      M = ((i == k) & (j == l)) / 2 + ((i == l) & (j == k)) / 2
      return cls(M.astype(float), fr, rank=4, **kwargs)
    raise ValueError("the identity is of rank two or four")

  @classmethod
  def ones(cls, *args, rank=None, **kwargs):
    """a tensor of ones of the rank of the class, or of rank=, the list shape given by the numbers

      tensor.ones(cs, 5)"""
    fr, rest = _readArgs(args)
    rank = (cls._rank or 2) if rank is None else int(rank)
    return cls(np.ones(tuple(int(n) for n in rest) + (3,) * rank), fr, rank=rank, **kwargs)

  @classmethod
  def zeros(cls, *args, rank=None, **kwargs):
    """a tensor of zeros of the rank of the class, or of rank=, the list shape given by the numbers

      tensor.zeros(cs, 5)"""
    fr, rest = _readArgs(args)
    rank = (cls._rank or 2) if rank is None else int(rank)
    return cls(np.zeros(tuple(int(n) for n in rest) + (3,) * rank), fr, rank=rank, **kwargs)

  @classmethod
  def rand(cls, *args, rank=None, rng=None, **kwargs):
    """a random tensor with the symmetries its class and frame ask for"""
    rng = np.random.default_rng() if rng is None else rng
    fr, rest = _readArgs(args)
    rank = (cls._rank or 2) if rank is None else int(rank)
    shape = tuple(int(n) for n in rest)
    T = cls(rng.standard_normal(shape + (3,) * rank), fr, rank=rank, **kwargs)
    if cls._symmetricRank2 or cls._rank == 4:
      T = T.sym().symmetrise()
    if cls._rank == 4:
      T = _positiveDefinite(T)
    return T

  @classproperty
  def leviCivita(cls):
    """the Levi-Civita symbol, the rank three tensor of the signs of the permutations of (1, 2, 3)"""
    i, j, k = np.indices((3, 3, 3))
    return tensor(((i - j) * (j - k) * (k - i) / 2).astype(float), rank=3)

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from ..io.loadtensor import _load
    from ..plotting.sphericalplot import plot
    load = classmethod(_load)


# a function of one square matrix over the matrices of a stack, a result real up to
# rounding returned real
def _perMatrix(f, A):
  n = A.shape[-1]
  flat = A.reshape(-1, n, n)
  out = np.stack([np.asarray(f(a)) for a in flat]) if len(flat) else np.zeros_like(flat)
  if np.iscomplexobj(out) and np.all(np.abs(out.imag) <= 1e-10 * max(1.0, float(np.max(np.abs(out.real), initial=0)))):
    out = out.real
  return out.reshape(A.shape)


# the trailing rank axes of a coefficient array joined into the one axis the container holds
def _flatten(M, rank):
  return M.reshape(M.shape[:M.ndim - rank] + (3 ** rank,))


# a rank four tensor shifted until its Voigt matrix is positive definite
def _positiveDefinite(T):
  V = Voigt(T)
  V = (V + np.swapaxes(V, -1, -2)) / 2
  lam = np.min(np.linalg.eigvalsh(V))
  if lam <= 0:
    V = V + (1e-3 + abs(lam)) * np.eye(6)
  cls = type(T)
  return cls(V, T.frame, **({} if cls is not tensor else {"rank": 4, "doubleConvention": T.doubleConvention})).symmetrise()


# Q acting on every index of M: out_{i...} = Q_{i i'} ... M_{i'...}, over broadcast leading
# dims; the compiled turn index by index where either side is one tensor or matrix or both are
# as many, else the indices split into two groups, each turned by a Kronecker power of Q, so
# that rank four is two products of 9 x 9 matrices
def _applyMatrix(M, Q, rank):
  Q = np.asarray(Q, dtype=float)
  if rank == 0:
    return _flatten(M, rank)
  lead, qlead = M.shape[:M.ndim - rank], Q.shape[:-2]
  shape = np.broadcast_shapes(lead, qlead)
  n, nm, nq = math.prod(shape), math.prod(lead), math.prod(qlead)
  if n > 1 and nm in (1, n) and nq in (1, n) and (nm == 1 or lead == shape) and (nq == 1 or qlead == shape):
    out = turnBlocks(M.reshape(nm, 3 ** rank), [rank], [0, 3 ** rank], Q.reshape(nq, 3, 3))
    if out is not None:
      return out.reshape(shape + (3 ** rank,))
  a, b = (rank + 1) // 2, rank // 2
  lead = M.shape[:M.ndim - rank]
  M2 = M.reshape(lead + (3 ** a, 3 ** b))
  out = _kronPower(Q, a) @ M2
  if b:
    out = out @ np.swapaxes(_kronPower(Q, b), -1, -2)
  return out.reshape(out.shape[:-2] + (3 ** rank,))


# Q x ... x Q, n times, with the row-major order of the flattened indices
def _kronPower(Q, n):
  K = Q
  for _ in range(n - 1):
    K = (K[..., :, None, :, None] * Q[..., None, :, None, :]).reshape(Q.shape[:-2] + (3 * K.shape[-2], 3 * K.shape[-1]))
  return K


# the vectors u x ... x u, n times, flattened in row-major order
def _vecPower(u, n):
  U = u
  for _ in range(n - 1):
    U = (U[..., :, None] * u[..., None, :]).reshape(u.shape[:-1] + (3 * U.shape[-1],))
  return U


# ------------------------------------------------------------------ the Einstein sum

def EinsteinSum(*args):
  """the contraction of tensors, vectors, rotations or arrays over the indices they share

    EinsteinSum(C, [1, 2, -1, -2], eps, [-1, -2])     sigma_ij = C_ijkl eps_kl
    EinsteinSum(a, [1], b, [2])                        the dyad a_i b_j
    EinsteinSum(T, [-1, -2, -1, -2])                   a label twice in one operand is a trace

  Every operand carries one label per index: a positive one is free and orders the result,
  a negative one is summed over. Lists of tensors combine elementwise over their leading
  dimensions. The result keeps the frame of the first tensor and is a plain tensor, or a
  number when no index is free.
  """
  if len(args) % 2:
    raise ValueError("every operand needs an index vector")
  ops, ids = args[0::2], [list(np.atleast_1d(i).astype(int)) for i in args[1::2]]
  arrays, subs, frame = [], [], None
  labels = {}
  for op, idx in zip(ops, ids):
    A, rank, fr = _operand(op)
    if fr is not None and frame is None:
      frame = fr
    if isinstance(op, (tensor, vector3d, quaternion)):
      lead = A.ndim - rank
    else:
      lead, rank = 0, A.ndim
    if len(idx) != rank:
      raise ValueError(f"{len(idx)} labels for an operand of {rank} indices")
    for k, l in enumerate(idx):
      if l == 0:
        raise ValueError("a label is a positive or a negative integer")
      n = A.shape[lead + k]
      if labels.setdefault(l, n) != n:
        raise ValueError(f"the index {l} is used with dimensions {n} and {labels[l]}")
    arrays.append(A)
    subs.append(("..." if lead or isinstance(op, (tensor, vector3d, quaternion)) else "") + "".join(_letter(l) for l in idx))
  free = sorted(l for l in labels if l > 0)
  out = "..." + "".join(_letter(l) for l in free)
  result = np.einsum(",".join(subs) + "->" + out, *arrays)
  if not free:
    return result if result.ndim else float(result)
  if any(labels[l] != 3 for l in free):
    return result
  t = tensor(result, rank=len(free))
  if frame is not None:
    t._frame = frame
  return t


def _letter(l):
  return LETTERS[abs(l) - 1 + (13 if l < 0 else 0)]


def colon(A, B):
  """the double contraction A : B, the last two indices of A against the first two of B"""
  out = EinsteinSum(A, list(range(1, A.rank - 1)) + [-1, -2], B, [-1, -2] + [A.rank - 2 + k for k in range(1, B.rank - 1)])
  if isinstance(out, tensor):
    cls = A._productClass(B) if hasattr(A, "_productClass") else None
    if cls is not None:
      out = A._as(cls, out.data)
      out.density, out.unit = None, None
  return out


doubleContract = colon


def dyad(a, b=None, n=None):
  """the outer product of tensors or vectors"""
  if b is None and n is None:
    return a if isinstance(a, tensor) else tensor(a)
  if isinstance(b, (int, np.integer)) and n is None:
    if b == 1:
      return dyad(a)
    args = []
    r = _operand(a)[1]
    for k in range(b):
      args += [a, list(range(k * r + 1, (k + 1) * r + 1))]
    return EinsteinSum(*args)
  ra, rb = _operand(a)[1], _operand(b)[1]
  return EinsteinSum(a, list(range(1, ra + 1)), b, list(range(ra + 1, ra + rb + 1)))


# the unit of the inverse, as MATLAB writes it: GPa to 1 / GPa, 1/GPa to GPa, m/s to s / m
def _inverseUnit(unit):
  if unit is None:
    return None
  if "/" not in unit:
    return "1 / " + unit.strip()
  a, b = unit.split("/", 1)
  a = a.replace("1", "").strip()
  return b.strip() if not a else f"{b.strip()} / {a}"


# ------------------------------------------------------------------ the physical tensors

class stiffnessTensor(tensor):
  """the elastic stiffness C, sigma = C : eps; a 6 x 6 Voigt matrix in GPa or the full array"""
  _rank, _double = 4, False

  def __init__(self, M, *args, **kwargs):
    kwargs.setdefault("unit", "GPa")
    super().__init__(M, *args, **kwargs)

  def _inverseClass(self):
    return complianceTensor

  def _productClass(self, other):
    return stressTensor if isinstance(other, strainTensor) else None

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from typing import Any
    from ..analysis.elasticity import (velocity, energyVector, YoungsModulus, linearCompressibility,
                                       volumeCompressibility, shearModulus, PoissonRatio, bulkModulus,
                                       HashinShtrikmanModulus, hashinShtrikmanBounds)
    from ..plotting.tensorplot import plotSeismicVelocities
    def ChristoffelTensor(self, *args, **kwargs) -> Any:
      """ChristoffelTensor(C, n): E_ik = C_ijkl n_j n_l of a stiffness and a direction"""


class complianceTensor(tensor):
  """the elastic compliance S = inv(C), its Voigt matrix carrying the factors of the double convention"""
  _rank, _double = 4, True

  def __init__(self, M, *args, **kwargs):
    kwargs.setdefault("unit", "1/GPa")
    super().__init__(M, *args, **kwargs)

  def _inverseClass(self):
    return stiffnessTensor

  def _productClass(self, other):
    return strainTensor if isinstance(other, stressTensor) else None

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from typing import Any
    from ..analysis.elasticity import (velocity, energyVector, YoungsModulus, linearCompressibility,
                                       volumeCompressibility, shearModulus, PoissonRatio, bulkModulus,
                                       HashinShtrikmanModulus, hashinShtrikmanBounds)
    def ChristoffelTensor(self, *args, **kwargs) -> Any:
      """ChristoffelTensor(C, n): E_ik = C_ijkl n_j n_l of a stiffness and a direction"""


class stressTensor(tensor):
  """a rank two symmetric stress tensor; stressTensor.uniaxial(v, sigma) the uniaxial tension along v

    sigma = stressTensor(M)
    sigma = stressTensor.uniaxial(vector3d.X)"""
  _rank, _symmetricRank2 = 2, True

  @staticmethod
  def uniaxial(v, sigma=1.0):
    """the stress of a uniaxial tension along v"""
    v = v if isinstance(v, vector3d) else vector3d(v)
    u = v.normalize().data
    return stressTensor(sigma * u[..., :, None] * u[..., None, :], v.frame if v.frame is not None else specimenFrame.default)

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from ..analysis.mechanics import meanStress, equivalentStress, calcShearStress, deviatoric as deviatoricStress


class strainTensor(tensor):
  """a rank two symmetric strain tensor, in the double convention of the Voigt notation

    eps = strainTensor(M)
    eps = strainTensor(M, cs)"""
  _rank, _double, _symmetricRank2 = 2, True, True

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from ..analysis.mechanics import (deviatoric as deviatoricStrain, equivalentStrain, octahedralStrain, axisRatios,
                                      maxStrain)
    from ..analysis.taylor import calcTaylor


class strainRateTensor(tensor):
  """a rank two symmetric strain rate tensor, the symmetric part of a velocity gradient, in the double
  convention of the Voigt notation"""
  _rank, _double, _symmetricRank2 = 2, True, True


class deformationGradientTensor(tensor):
  """a deformation gradient tensor F, the rank two tensor carrying a line segment dX of the undeformed
  body into dx = F dX of the deformed one

    F = deformationGradientTensor(M)
    F = deformationGradientTensor.uniaxial(d, rate)"""
  _rank = 2

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from ..analysis.mechanics import (_uniaxialDeformation, _simpleShearDeformation, _pureShearDeformation,
                                      CauchyStrain, GreenStrain)
    uniaxial = staticmethod(_uniaxialDeformation)
    simpleShear = staticmethod(_simpleShearDeformation)
    pureShear = staticmethod(_pureShearDeformation)


class velocityGradientTensor(tensor):
  """a velocity gradient tensor L, the rank two tensor of the spatial derivatives of the velocity; its
  symmetric part the strain rate, its skew part the spin

    L = velocityGradientTensor(M)
    L = velocityGradientTensor.uniaxial(d, e)
    L = velocityGradientTensor.simpleShear(d, n, e)"""
  _rank = 2

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from ..analysis.mechanics import (_uniaxialFlow, _simpleShearFlow, strainRate, spin, angularVelocityVector,
                                      rotationRate, vorticity)
    uniaxial = staticmethod(_uniaxialFlow)
    simpleShear = staticmethod(_simpleShearFlow)


class spinTensor(tensor):
  """spinTensor(M), or spinTensor(v) the skew matrix of a rotation vector, in its frame"""
  _rank = 2

  def __init__(self, M, *args, **kwargs):
    if isinstance(M, vector3d) or (not isinstance(M, tensor) and np.ndim(M) >= 1 and np.shape(M)[-1] == 3 and (np.ndim(M) == 1 or np.shape(M)[-2] != 3)):
      v = M if isinstance(M, vector3d) else vector3d(np.asarray(M, dtype=float))
      x, y, z = v.data[..., 0], v.data[..., 1], v.data[..., 2]
      zero = np.zeros_like(x)
      M = np.stack([np.stack([zero, -z, y], axis=-1), np.stack([z, zero, -x], axis=-1), np.stack([-y, x, zero], axis=-1)], axis=-2)
      if v.frame is not None and not args and "frame" not in kwargs:
        args = (v.frame,)
    super().__init__(M, *args, **kwargs)

  def exp(self, ref=None, side=None):
    """exp(S, ref, side): the rotation the axial vector points to at the reference"""
    return vector3d(self).exp(ref, side)

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from ..analysis.mechanics import spinVector as vector


class curvatureTensor(tensor):
  """the lattice curvature tensor, the rank two tensor of the spatial derivatives of the orientation
  field, what the dislocation density tensor is derived from"""
  _rank = 2

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from ..analysis.mechanics import dislocationDensity


class dislocationDensityTensor(tensor):
  """the dislocation density or Nye tensor, the rank two tensor derived from the curvature tensor,
  whose fit against dislocation systems gives the geometrically necessary densities

    alpha = dislocationDensityTensor(M, cs)"""
  _rank = 2

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from ..analysis.mechanics import curvature


class ChristoffelTensor(tensor):
  """ChristoffelTensor(C, n): E_ik = C_ijkl n_j n_l of a stiffness and a direction"""
  _rank, _symmetricRank2 = 2, True

  def __init__(self, M, *args, **kwargs):
    if isinstance(M, (stiffnessTensor, complianceTensor)):
      from ..analysis.elasticity import _christoffel
      M, fr, density = _christoffel(M, args[0])
      args, kwargs = (fr,), dict(kwargs, density=density)
    super().__init__(M, *args, **kwargs)


class refractiveIndexTensor(tensor):
  """the rank two symmetric refractive index tensor, whose anisotropy makes a crystal birefringent

    rI = refractiveIndexTensor(M, cs)"""
  _rank, _symmetricRank2 = 2, True

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from ..analysis.optics import refractiveIndex, birefringence, opticalAxis, spectralTransmission


class SchmidTensor(tensor):
  """SchmidTensor(n, b) of a plane normal and a slip direction, SchmidTensor(sS) of slip systems"""
  _rank = 2

  def __init__(self, M, *args, **kwargs):
    if isinstance(M, vector3d) or hasattr(M, "CRSS"):
      from ..analysis.mechanics import _schmidTensor
      M, fr = _schmidTensor(M, args[0] if args else None)
      args = (fr,)
    super().__init__(M, *args, **kwargs)


# ------------------------------------------------------------------ the verbs as functions

def rotate(T, r):
  """rotate(T, r) is T.rotate(r): the tensor turned by a rotation, an orientation or a matrix"""
  return T.rotate(r)


def symmetrise(T):
  """symmetrise(T) is T.symmetrise(): the mean of the tensor over the group of its frame"""
  return T.symmetrise()


def checkSymmetry(T, atol=1e-6):
  """checkSymmetry(T) is T.checkSymmetry(): whether the tensor is invariant under the group of its
  frame"""
  return T.checkSymmetry(atol)


def directionalMagnitude(T, v=None):
  """directionalMagnitude(T, v) is T.directionalMagnitude(v): the tensor contracted with the direction
  on every index, without v the function on the sphere"""
  return T.directionalMagnitude(v)


def transformReferenceFrame(T, fr):
  """transformReferenceFrame(T, fr) is T.transformReferenceFrame(fr): the tensor read in another
  frame"""
  return T.transformReferenceFrame(fr)


def sym(T):
  """sym(T) is T.sym(): the symmetric part, the mean over the permutations of the indices"""
  return T.sym()


def antiSym(T):
  """antiSym(T) is T.antiSym(): the antisymmetric part of a rank two tensor"""
  return T.antiSym()


def times(A, B):
  """times(A, B) is A.times(B): the product of corresponding coefficients, MATLAB's A .* B"""
  return A.times(B)


def trace(T):
  """trace(T) is T.trace(): the trace of a rank two tensor, or of the Voigt matrix of a rank four"""
  return T.trace()


def eig(T):
  """eig(T) is T.eig(): the eigenvalues in ascending order and the eigenvectors"""
  return T.eig()


def expm(T):
  """expm(T) is T.expm(): the matrix exponential"""
  return T.expm()


def logm(T):
  """logm(T) is T.logm(): the matrix logarithm"""
  return T.logm()


def sqrtm(T):
  """sqrtm(T) is T.sqrtm(): the matrix square root"""
  return T.sqrtm()


def svd(T):
  """svd(T) is T.svd(): the singular value decomposition, u, s, v of a rank two tensor"""
  return T.svd()


__all__ = ["tensor", "stiffnessTensor", "complianceTensor", "stressTensor", "strainTensor", "strainRateTensor",
           "deformationGradientTensor", "velocityGradientTensor", "spinTensor", "curvatureTensor", "dislocationDensityTensor",
           "ChristoffelTensor", "refractiveIndexTensor", "SchmidTensor", "EinsteinSum", "colon", "doubleContract", "dyad",
           "Voigt", "Kelvin", "times", "checkSymmetry", "directionalMagnitude", "sym", "antiSym", "trace", "eig", "expm", "logm",
           "sqrtm", "svd"]
