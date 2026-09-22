# tensors of any rank in a reference frame (MATLAB TensorAnalysis/@tensor and the classes
# derived from it; docs/plans/functions.md P4c)
#
#   T = tensor(M, cs, rank=2)                  the coefficients M with the list dimensions in front
#   C = stiffnessTensor(V, cs, density=3.3)    a 6 x 6 Voigt matrix, or the 3 x 3 x 3 x 3 array
#   T.M, T.rank, T.frame, T.CS, Voigt(C), Kelvin(C)
#   EinsteinSum(C, [1, 2, -1, -2], eps, [-1, -2]), colon(C, eps), dyad(a, b)
#   rotate(T, r), ori * T, symmetrise(T), checkSymmetry(T), directionalMagnitude(T, v)
#   tensor.eye(rank=4), tensor.rand(5, rank=2), tensor.leviCivita, mean(Ts)
#
# A tensor of rank R is an element of (R^3)^(x R) written in a reference frame; a rotation
# acts on every index. A tensor turned by anything but an element of the group its frame
# claims is written in the frame without the group. What MTEX splits into a class per
# physical quantity is a subclass here fixing the rank, the Voigt factors of its notation
# and the class its inverse takes; a contraction returns a plain tensor, the physical
# products name their class themselves.
from __future__ import annotations

import itertools
import math

import numpy as np

from ..geometry._array import MtexArray
from ..geometry.display import table
from ..geometry.frames import crystalFrame, referenceFrame, specimenFrame
from ..geometry.orientation import orientation
from ..geometry.products import dropGroup, fitFrame
from ..geometry.quaternion import orthogonalTransform, quaternion, rotation
from ..geometry.vector3d import classproperty, vector3d

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


# the tensor of a Voigt matrix, the factors taken out again and the pair symmetry filled in
def _fromVoigt(V, rank, w):
  V = np.asarray(V, dtype=float)
  lead = V.shape[:-2] if rank > 2 else V.shape[:-1]
  M = np.zeros(lead + (3,) * rank)
  if rank == 4:
    for i, (a, b) in enumerate(VOIGT):
      for j, (c, d) in enumerate(VOIGT):
        v = V[..., i, j] / (w[i] * w[j])
        for p, q in ((a, b), (b, a)):
          for r, s in ((c, d), (d, c)):
            M[..., p, q, r, s] = v
  elif rank == 3:
    for j, (a, b) in enumerate(VOIGT):
      v = V[..., :, j] / w[j]
      M[..., :, a, b] = v
      M[..., :, b, a] = v
  else:
    for i, (a, b) in enumerate(VOIGT):
      M[..., a, b] = V[..., i] / w[i]
      M[..., b, a] = V[..., i] / w[i]
  return M


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
    return self.data.reshape(self.shape + (3,) * self.rank)

  @M.setter
  def M(self, value):
    self.data = np.asarray(value, dtype=float).reshape(self.shape + (self._ncoord,))

  @property
  def frame(self):
    return self._frame

  @frame.setter
  def frame(self, fr):
    if not isinstance(fr, referenceFrame):
      raise TypeError("the frame of a tensor has to be a referenceFrame")
    self._frame = fr

  @property
  def CS(self):
    return self._frame

  @CS.setter
  def CS(self, fr):
    self.frame = fr

  @property
  def isSymmetric(self):
    if self.rank not in (2, 4):
      raise ValueError("symmetry is checked on tensors of rank two and four")
    return bool(np.all(np.linalg.norm((self - self.transpose()).data, axis=-1) / (1e-12 + np.linalg.norm(self.data, axis=-1)) < 1e-6))

  @property
  def isSkewSymmetric(self):
    return bool(np.all(np.linalg.norm((self + self.transpose()).data, axis=-1) / (1e-12 + np.linalg.norm(self.data, axis=-1)) < 1e-6))

  def __repr__(self):
    lines = [f"{type(self).__name__} ({self._frame.char()})"]
    if self.density is not None:
      lines.append(f"  density: {self.density}")
    if self.unit is not None:
      lines.append(f"  unit: {self.unit}")
    size = " x ".join(str(n) for n in self.shape) if self.ndim else ""
    lines.append(f"  rank: {self.rank} ({' x '.join(['3'] * self.rank) or '1'})" + (f", size: {size}" if size else ""))
    if self.ndim == 0 and self.rank in (2, 3, 4):
      A = self.M if self.rank == 2 else Voigt(self)
      finite = np.abs(A[np.isfinite(A)])
      m = float(finite.max()) if finite.size else 0.0
      p = 0 if m == 0 else math.floor(math.log10(m))
      scale = p if (p >= 3 or p <= -2) else 0
      lines.append("")
      lines.append(("tensor in matrix representation" if self.rank == 2 else "tensor in Voigt matrix representation") + (f": *10^{scale}" if scale else ""))
      S = A / 10.0 ** scale
      lines.append(table([""] * S.shape[1], [S[:, j] for j in range(S.shape[1])], decimals=4))
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

  def __truediv__(self, a):
    if isinstance(a, tensor):
      raise ValueError("a tensor is divided by a number")
    return self._rebuild(self.data / np.asarray(a, dtype=float)[..., None])

  def __eq__(self, other):
    return isinstance(other, tensor) and other.rank == self.rank and other._frame is self._frame and np.array_equal(self.data, other.data)

  def __ne__(self, other):
    return not self == other

  def isclose(self, other, rtol=1e-8, atol=0.0):
    return isinstance(other, tensor) and other.rank == self.rank and self.data.shape == other.data.shape and bool(np.allclose(self.data, other.data, rtol=rtol, atol=atol))

  def norm(self):
    return np.linalg.norm(self.data, axis=-1)

  def normalize(self):
    return self / self.norm()

  def trace(self):
    if self.rank == 2:
      return np.einsum("...ii->...", self.M)
    if self.rank == 4:
      return np.einsum("...ii->...", Voigt(self))
    raise ValueError("the trace is of a tensor of rank two or four")

  def det(self):
    if self.rank != 2:
      raise ValueError("the determinant is of a tensor of rank two")
    return np.linalg.det(self.M)

  # the transpose swaps the indices of a rank two and the pairs of a rank four
  def transpose(self):
    if self.rank == 2:
      return self._rebuild(_flatten(np.swapaxes(self.M, -1, -2), 2))
    if self.rank == 4:
      return self._rebuild(_flatten(np.moveaxis(self.M, (-4, -3, -2, -1), (-2, -1, -4, -3)), 4))
    if self.rank <= 1:
      return self
    raise ValueError("the transpose is of a tensor of rank two or four")

  @property
  def T(self):
    return self.transpose()

  ctranspose = transpose

  # the symmetric and the antisymmetric part, the mean over the permutations of the indices
  def sym(self):
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
    if self.rank != 2:
      raise ValueError("the antisymmetric part is of a tensor of rank two")
    return self._rebuild(_flatten((self.M - np.swapaxes(self.M, -1, -2)) / 2, 2))

  # the inverse: rank two by the matrix, rank four through the Voigt matrix, which swaps the
  # convention and with it the class, a stiffness inverting to a compliance
  def inv(self):
    cls = self._inverseClass()
    if self.rank == 2:
      return self._as(cls, _flatten(np.linalg.inv(self.M), 2))
    if self.rank == 4:
      V = np.linalg.inv(Voigt(self))
      double = cls._double if cls is not tensor else not self.doubleConvention
      out = self._as(cls, _flatten(_fromVoigt(V, 4, _voigtWeights(double)), 4))
      out.doubleConvention = double
      return out
    raise ValueError("the inverse is of a tensor of rank two or four")

  def _inverseClass(self):
    return type(self)

  # the eigenvalues in ascending order and the eigenvectors as axes of the frame; a rank
  # four tensor is diagonalised in its Voigt matrix
  def eig(self):
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
      return vals, v
    if self.rank == 4:
      vals, vecs = np.linalg.eigh(Voigt(self))
      return vals, vecs
    raise ValueError("eigenvalues are of a tensor of rank two or four")

  def eigvals(self):
    return self.eig()[0]

  # ------------------------------------------------------------------ the group action

  # the tensor turned by a rotation, an orthogonal transform or an orientation; the frame
  # keeps its group only under the group's own elements, and an orientation carries the
  # tensor into its specimen frame, without the specimen group
  def rotate(self, r):
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

  # the same coefficients read in another frame, the numbers changed by the rigid part of
  # the transition between the two bases
  def transformReferenceFrame(self, fr):
    ok, M = self._frame.isCompatible(fr)
    if not ok:
      raise ValueError("the two frames are not related by a rigid change of coordinates")
    U, _, Vt = np.linalg.svd(M)
    return self._as(type(self), _applyMatrix(self.M, U @ Vt, self.rank), fr)

  # the mean over the group of the frame, and whether the tensor is invariant under it
  def symmetrise(self):
    Q = self._frame.rot.matrix()
    if len(Q) <= 1:
      return self
    out = np.zeros(self.data.shape)
    for q in Q:
      out += _applyMatrix(self.M, q, self.rank)
    return self._rebuild(out / len(Q))

  def checkSymmetry(self, atol=1e-6):
    scale = atol * max(float(np.max(self.norm())) if self.size else 0.0, 1e-12)
    return all(np.allclose(_applyMatrix(self.M, q, self.rank), self.data, atol=scale) for q in self._frame.rot.matrix())

  # ------------------------------------------------------------------ magnitudes

  # the tensor contracted with the unit direction on every index; without a direction the
  # function on the sphere, a polynomial of the rank's degree
  def directionalMagnitude(self, v=None):
    if v is None:
      from ..functions.s2fun import S2FunHarmonic
      return S2FunHarmonic(lambda x: self.directionalMagnitude(x), self._frame, bandwidth=max(self.rank, 1), real=True,
                           antipodal=self.rank % 2 == 0)
    v = v if isinstance(v, vector3d) else vector3d(v)
    if v.frame is not None and v.frame is not self._frame and not v.frame.isAligned(self._frame):
      raise ValueError("the direction is written in another frame than the tensor")
    u = v.normalize().data
    R = self.rank
    idx = LETTERS[:R]
    lead = "" if self.ndim == 0 else "..."
    vlead = "" if v.ndim == 0 else "..."
    subs = lead + idx + "".join("," + vlead + c for c in idx) + "->..."
    return np.einsum(subs, self.M, *([u] * R))

  # ------------------------------------------------------------------ lists

  # the mean over the list, weighted or not; with an ODF the texture average
  def mean(self, *args, weights=None, **kwargs):
    if args and not isinstance(args[0], str):
      from ..analysis.texture import calcTensor
      return calcTensor(args[0], self, *args[1:], weights=weights, **kwargs)
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

  def sum(self):
    return self._as(type(self), self.data.reshape(-1, self._ncoord).sum(axis=0))

  # ------------------------------------------------------------------ factories

  @classmethod
  def eye(cls, *args, rank=None, **kwargs):
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
    fr, rest = _readArgs(args)
    rank = (cls._rank or 2) if rank is None else int(rank)
    return cls(np.ones(tuple(int(n) for n in rest) + (3,) * rank), fr, rank=rank, **kwargs)

  @classmethod
  def zeros(cls, *args, rank=None, **kwargs):
    fr, rest = _readArgs(args)
    rank = (cls._rank or 2) if rank is None else int(rank)
    return cls(np.zeros(tuple(int(n) for n in rest) + (3,) * rank), fr, rank=rank, **kwargs)

  # a random tensor with the symmetries its class and frame ask for
  @classmethod
  def rand(cls, *args, rank=None, rng=None, **kwargs):
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
    i, j, k = np.indices((3, 3, 3))
    return tensor(((i - j) * (j - k) * (k - i) / 2).astype(float), rank=3)


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


# Q acting on every index of M: out_{i...} = Q_{i i'} ... M_{i'...}, over broadcast leading dims
def _applyMatrix(M, Q, rank):
  Q = np.asarray(Q, dtype=float)
  out = M
  for d in range(rank):
    out = _modeProduct(out, Q, rank, d)
  return _flatten(out, rank)


# the product of Q with the d-th index of M
def _modeProduct(M, Q, rank, d):
  axis = M.ndim - rank + d
  X = np.moveaxis(M, axis, -1)
  lead = np.broadcast_shapes(X.shape[:X.ndim - rank], Q.shape[:-2])
  Xb = np.broadcast_to(X, lead + X.shape[X.ndim - rank:])
  Qb = np.broadcast_to(Q, lead + (3, 3))
  # the tensor indices other than d sit between the list dims and the last axis
  other = Xb.shape[len(lead):-1]
  Xr = Xb.reshape(lead + (int(np.prod(other)),) + (3,)) if other else Xb.reshape(lead + (1, 3))
  Y = np.einsum("...ab,...nb->...na", Qb, Xr)
  Y = Y.reshape(lead + other + (3,))
  return np.moveaxis(Y, -1, Y.ndim - rank + d)


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


# the double contraction A : B, the last two indices of A against the first two of B
def colon(A, B):
  out = EinsteinSum(A, list(range(1, A.rank - 1)) + [-1, -2], B, [-1, -2] + [A.rank - 2 + k for k in range(1, B.rank - 1)])
  if isinstance(out, tensor):
    cls = A._productClass(B) if hasattr(A, "_productClass") else None
    if cls is not None:
      out = A._as(cls, out.data)
  return out


doubleContract = colon


# the outer product of tensors or vectors
def dyad(a, b=None, n=None):
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


class complianceTensor(tensor):
  """the elastic compliance S = inv(C), its Voigt matrix carrying the factors of the double convention"""
  _rank, _double = 4, True

  def _inverseClass(self):
    return stiffnessTensor

  def _productClass(self, other):
    return strainTensor if isinstance(other, stressTensor) else None


class stressTensor(tensor):
  _rank, _symmetricRank2 = 2, True

  # the stress of a uniaxial tension along v
  @staticmethod
  def uniaxial(v, sigma=1.0):
    v = v if isinstance(v, vector3d) else vector3d(v)
    u = v.normalize().data
    return stressTensor(sigma * u[..., :, None] * u[..., None, :], v.frame if v.frame is not None else specimenFrame.default)


class strainTensor(tensor):
  _rank, _double, _symmetricRank2 = 2, True, True


class strainRateTensor(tensor):
  _rank, _double, _symmetricRank2 = 2, True, True


class deformationGradientTensor(tensor):
  _rank = 2


class velocityGradientTensor(tensor):
  _rank = 2


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

  # exp(S, ref, side): the rotation the axial vector points to at the reference
  def exp(self, ref=None, side=None):
    return vector3d(self).exp(ref, side)


class curvatureTensor(tensor):
  _rank = 2


class dislocationDensityTensor(tensor):
  _rank = 2


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
  _rank, _symmetricRank2 = 2, True


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
  return T.rotate(r)


def symmetrise(T):
  return T.symmetrise()


def checkSymmetry(T, atol=1e-6):
  return T.checkSymmetry(atol)


def directionalMagnitude(T, v=None):
  return T.directionalMagnitude(v)


def transformReferenceFrame(T, fr):
  return T.transformReferenceFrame(fr)


def sym(T):
  return T.sym()


def antiSym(T):
  return T.antiSym()


def trace(T):
  return T.trace()


def eig(T):
  return T.eig()


__all__ = ["tensor", "stiffnessTensor", "complianceTensor", "stressTensor", "strainTensor", "strainRateTensor",
           "deformationGradientTensor", "velocityGradientTensor", "spinTensor", "curvatureTensor", "dislocationDensityTensor",
           "ChristoffelTensor", "refractiveIndexTensor", "SchmidTensor", "EinsteinSum", "colon", "doubleContract", "dyad",
           "Voigt", "Kelvin", "checkSymmetry", "directionalMagnitude", "sym", "antiSym", "trace", "eig"]
