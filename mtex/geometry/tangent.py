# the tangent spaces of SO(3) (Julia ADR 0010, MATLAB @SO3TangentVector)
#
#   SO3TangentVector(v, oriRef)             a rotation vector at the reference, on the left: exp(v) * oriRef
#   SO3TangentVector(v, oriRef, 'right')    on the right: oriRef * exp(v)
#   SO3TangentVector(S, oriRef, side)       from skew matrices of shape (..., 3, 3)
#   log(ori, oriRef), log(ori, oriRef, 'right')   the vector pointing from oriRef to ori
#   exp(t)                                  the rotation or orientation t points to
#   left(t), right(t)                       the other side, by the adjoint action of the reference
#   spinTensor(t)                           the skew matrix of the vector
#   geodesic(a, b, s)                       exp(s * log(b, a)), the shortest path from a to b
#
# A tangent vector stores its components, its side and its reference, nothing else: a
# left vector is written in the frame the reference maps into, a right vector in the
# frame it maps from, and both symmetries live on the reference. The reference is one
# rotation shared by every element or one per element. log of orientations reduces the
# pair by symmetry first, so its norm is angle(ori, oriRef).
from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

from ._array import MtexArray
from .kernels import qmul, qrot
from .orientation import orientation
from .quaternion import orthogonalTransform, quaternion, rotation
from .representations import axis2quat, isnull
from .session import SESSION
from .vector3d import vector3d
from .display import aligned

SIDES = ("left", "right")
REFERENCE_TOLERANCE = 1e-8


def _side(side):
  if side not in SIDES:
    raise ValueError(f"the side of a tangent space is 'left' or 'right', not {side!r}")
  return side


def _conj(q):
  return q * np.array([1.0, -1.0, -1.0, -1.0])


def rotationVector(q):
  """the rotation vector of a rotation at the identity, through atan2 as the angle is"""
  q = np.asarray(q, dtype=float)
  v = q[..., 1:]
  n = np.linalg.norm(v, axis=-1)
  zero = isnull(n)
  s = np.where(q[..., 0] < 0, -1.0, 1.0)
  f = np.where(zero, 0.0, 2 * np.arctan2(n, np.abs(q[..., 0])) / np.where(zero, 1.0, n) * s)
  return f[..., None] * v


def rotationOf(v):
  """the rotation of a rotation vector at the identity"""
  v = np.asarray(v, dtype=float)
  return axis2quat(v, np.linalg.norm(v, axis=-1))


def spinTensor(v):
  """the skew matrix of a vector, and the vector of a skew matrix, spinTensor(S) reading the
  axial vector of matrices of shape (..., 3, 3)"""
  if isinstance(v, vector3d):
    v = v.data
  else:
    v = np.asarray(v, dtype=float)
    if v.ndim >= 2 and v.shape[-2:] == (3, 3):
      return vector3d(_axialVector(v))
  x, y, z = v[..., 0], v[..., 1], v[..., 2]
  o = np.zeros_like(x)
  return np.stack([np.stack([o, -z, y], -1), np.stack([z, o, -x], -1), np.stack([-y, x, o], -1)], -2)


def _axialVector(S):
  S = 0.5 * (S - np.swapaxes(S, -1, -2))
  return np.stack([S[..., 2, 1], S[..., 0, 2], S[..., 1, 0]], axis=-1)


# the reference rebuilt around another rotation, its frames kept
def _withRotation(ref, data):
  if isinstance(ref, orientation):
    return orientation._fromRotation(data, (ref.frameA, ref.frameB), ref.antipodal)
  return rotation._fromData(data)


def _frameOf(ref, side):
  if not isinstance(ref, orientation):
    return None
  return ref.frameB if side == "left" else ref.frameA


class SO3TangentVector(vector3d):
  """a tangent vector of SO(3), a rotation vector v at a reference rotation or orientation, on the
  left, exp(v) * oriRef, or on the right, oriRef * exp(v); from vectors or skew matrices

    SO3TangentVector(v, oriRef)
    SO3TangentVector(v, oriRef, 'right')
    SO3TangentVector(S, oriRef, side)      from skew matrices of shape (..., 3, 3)"""
  _carried = ("antipodal", "isNormalized", "dispStyle", "side", "oriRef")

  def __init__(self, v, oriRef, side="left"):
    side = _side(side)
    oriRef = _reference(oriRef)
    data = _components(v)
    MtexArray.__init__(self, data)
    self.side = side
    self.oriRef = _fitShape(oriRef, self.shape)
    self.antipodal = False
    self.isNormalized = False
    self.dispStyle = _styleFor(_frameOf(self.oriRef, side))
    # a framed vector must agree with the frame the tangent space is written in
    if isinstance(v, vector3d) and v.frame is not None:
      SESSION.joinFrames(v.frame, self._frame)

  # ------------------------------------------------------------------ the frame, read through to the reference

  @property
  def _frame(self):
    return _frameOf(self.oriRef, self.side)

  @property
  def frame(self):
    """the frame the components are written in: the specimen frame of the reference on the left, its
    crystal frame on the right"""
    return self._frame

  @frame.setter
  def frame(self, fr):
    """refuses: the frame of a tangent vector follows from its reference and side"""
    raise AttributeError("the frame of a tangent vector is the specimen frame (left) or the crystal frame (right) of its reference")

  @property
  def CS(self):
    """the crystal frame of the reference, None for a rotation"""
    return self.oriRef.CS if isinstance(self.oriRef, orientation) else None

  @property
  def SS(self):
    """the specimen frame of the reference, None for a rotation"""
    return self.oriRef.SS if isinstance(self.oriRef, orientation) else None

  # ------------------------------------------------------------------ derived objects

  # the same tangent vector with other components, or a plain framed vector when the result
  # is no longer one: a vector3d operation hands in the joined frame, and a shape that no
  # longer fits a reference per element has lost its base points
  def _new(self, data, **attrs):
    if "_frame" in attrs or (self.oriRef.ndim and data.shape[:-1] != self.oriRef.shape):
      frame = attrs.pop("_frame", self._frame)
      obj = object.__new__(vector3d)
      obj.data = data
      obj.antipodal = attrs.get("antipodal", False)
      obj.isNormalized = attrs.get("isNormalized", False)
      obj._frame = frame
      obj.dispStyle = attrs.get("dispStyle", self.dispStyle if frame is self._frame else "xyz")
      return obj
    return super()._new(data, **attrs)

  def _rebuild(self, data, oriRef=None, side=None):
    obj = object.__new__(type(self))
    obj.data = data
    obj.antipodal = False
    obj.isNormalized = False
    obj.side = self.side if side is None else side
    obj.oriRef = self.oriRef if oriRef is None else oriRef
    obj.dispStyle = _styleFor(obj._frame)
    return obj

  def copy(self):
    """a copy of the tangent vectors, components and references"""
    return self._rebuild(self.data.copy(), self.oriRef.copy())

  def __getitem__(self, idx):
    idx = idx if isinstance(idx, tuple) else (idx,)
    ref = self.oriRef[idx] if self.oriRef.ndim else self.oriRef
    return self._rebuild(self.data[idx + (slice(None),)], ref)

  def __setitem__(self, idx, value):
    idx = idx if isinstance(idx, tuple) else (idx,)
    value = self._fit(value)
    self.data[idx + (slice(None),)] = value.data
    if self.oriRef.ndim:
      self.oriRef[idx] = value.oriRef

  def reshape(self, *shape):
    """the tangent vectors in another shape, a reference per element reshaped with them"""
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    data = self.data.reshape(tuple(shape) + (3,))
    ref = self.oriRef.reshape(data.shape[:-1]) if self.oriRef.ndim else self.oriRef
    return self._rebuild(data, ref)

  @property
  def T(self):
    """the transposed array of tangent vectors, a reference per element transposed with them"""
    axes = tuple(range(self.ndim))[::-1] + (self.ndim,)
    return self._rebuild(self.data.transpose(axes), self.oriRef.T if self.oriRef.ndim else self.oriRef)

  @classmethod
  def cat(cls, *arrays, axis=0):
    """tangent vectors concatenate on one side; the references concatenate with them unless
    every list shares the same one"""
    first = arrays[0]
    arrays = [first._fit(a, sameReference=False) for a in arrays]
    arrays = [a if a.ndim > 0 else a.reshape(1) for a in arrays]
    axis = axis if axis >= 0 else axis - 1
    data = np.concatenate([a.data for a in arrays], axis=axis)
    refs = [a.oriRef for a in arrays]
    if all(r.ndim == 0 for r in refs) and all(_sameReference(refs[0], r) for r in refs[1:]):
      return first._rebuild(data, refs[0])
    refs = [_fitShape(r, a.shape) for r, a in zip(refs, arrays)]
    return first._rebuild(data, type(refs[0]).cat(*refs, axis=axis))

  # ------------------------------------------------------------------ the two sides

  def left(self):
    """the vectors on the left side, turned by the reference where they are on the right"""
    if self.side == "left":
      return self
    return self._rebuild(qrot(self.oriRef.data, self.data), side="left")

  def right(self):
    """the vectors on the right side, turned by the inverse reference where they are on the left"""
    if self.side == "right":
      return self
    return self._rebuild(qrot(_conj(self.oriRef.data), self.data), side="right")

  def toSide(self, side):
    """the vectors on the side asked for, 'left' or 'right'"""
    return self.left() if _side(side) == "left" else self.right()

  # the other operand brought to this reference and side
  def _fit(self, other, sameReference=True):
    if not isinstance(other, SO3TangentVector):
      raise TypeError("a tangent vector combines only with tangent vectors at the same reference; take vector3d(t) for the plain vector")
    if sameReference and not np.all(_sameReference(self.oriRef, other.oriRef)):
      raise ValueError("tangent vectors at different references cannot be combined")
    return other.toSide(self.side)

  # ------------------------------------------------------------------ exp, spin tensor

  def exp(self):
    """the rotation or orientation the vector points to, with the frames of the reference"""
    q = rotationOf(self.data)
    ref = self.oriRef.data
    return _withRotation(self.oriRef, qmul(q, ref) if self.side == "left" else qmul(ref, q))

  def spinTensor(self):
    """the skew matrix as the tensor of the tensors layer, in the frame of the side"""
    from ..tensors.tensor import spinTensor as spinTensorClass
    return spinTensorClass(vector3d(self.data, frame=self.frame))

  # ------------------------------------------------------------------ algebra at one reference

  def __add__(self, other):
    return self._rebuild(self.data + self._fit(other).data)

  def __radd__(self, other):
    return self.__add__(other)

  def __sub__(self, other):
    return self._rebuild(self.data - self._fit(other).data)

  def __rsub__(self, other):
    return self._fit(other).__sub__(self)

  def __neg__(self):
    return self._rebuild(-self.data)

  # a number scales; a rotation on the right turns the crystal, so it moves a right vector with its reference
  def __mul__(self, other):
    if isinstance(other, orientation) or isinstance(other, orthogonalTransform):
      return NotImplemented
    if isinstance(other, quaternion):
      if self.side != "right":
        raise TypeError("a rotation acts on a left tangent vector from the left and on a right one from the right; use left(t) or right(t)")
      return self._rebuild(qrot(_conj(other.data), self.data), self.oriRef * other)
    if isinstance(other, MtexArray):
      raise TypeError("tangent vectors are scaled by numbers and summed with tangent vectors; there is no other product")
    return self._rebuild(self.data * np.asarray(other, dtype=float)[..., None])

  def __rmul__(self, other):
    if isinstance(other, MtexArray):
      raise TypeError("tangent vectors are scaled by numbers and summed with tangent vectors; there is no other product")
    return self._rebuild(self.data * np.asarray(other, dtype=float)[..., None])

  def __truediv__(self, other):
    if isinstance(other, MtexArray):
      raise TypeError("a tangent vector is divided by numbers only")
    return self._rebuild(self.data / np.asarray(other, dtype=float)[..., None])

  # a rotation on the left turns the specimen, so it moves a left vector with its reference
  def _rotated(self, q):
    if self.side != "left":
      raise TypeError("a rotation acts on a left tangent vector from the left and on a right one from the right; use left(t) or right(t)")
    return self._rebuild(qrot(q, self.data), rotation._fromData(q) * self.oriRef)

  def dot(self, other, **kwargs):
    """the inner product with tangent vectors at the same reference, brought to this side"""
    return np.sum(self.data * self._fit(other).data, axis=-1)

  def cross(self, other):
    """the cross product with tangent vectors at the same reference, brought to this side"""
    return self._rebuild(np.cross(self.data, self._fit(other).data))

  def angle(self, other=None, **kwargs):
    """the angle of the rotation the vector describes, or the angle between two vectors"""
    if other is None:
      return self.norm()
    other = self._fit(other)
    d = np.sum(self.data * other.data, axis=-1)
    c = np.linalg.norm(np.cross(self.data, other.data), axis=-1)
    return np.arctan2(c, d)

  # equal vectors at the same reference, up to a tolerance in the components
  def __eq__(self, other, tol=1e-6):
    if not isinstance(other, SO3TangentVector):
      return NotImplemented
    same = _sameReference(self.oriRef, other.oriRef)
    other = other.toSide(self.side)
    return np.logical_and(same, np.linalg.norm(self.data - other.data, axis=-1) < tol)

  def __ne__(self, other):
    eq = self.__eq__(other)
    return NotImplemented if eq is NotImplemented else np.logical_not(eq)

  # sums and means keep the class at one reference, which every element has to share
  def _oneReference(self):
    ref = self.oriRef
    if ref.ndim == 0:
      return ref
    if not np.all(_sameReference(ref.flatten()[0], ref)):
      raise ValueError("the tangent vectors sit at different references")
    return ref.flatten()[0]

  def sum(self, axis=None, **kwargs):
    """the sum of tangent vectors at one shared reference, over all or along an axis"""
    ref = self._oneReference()
    axis = tuple(range(self.ndim)) if axis is None else axis
    return self._rebuild(np.sum(self.data, axis=axis), ref)

  def mean(self, weights=None, axis=None, **kwargs):
    """the mean of tangent vectors at one shared reference, weighted when weights are given"""
    ref = self._oneReference()
    axis = tuple(range(self.ndim)) if axis is None else axis
    if weights is None:
      return self._rebuild(np.nanmean(self.data, axis=axis), ref)
    w = np.asarray(weights, dtype=float)
    return self._rebuild(np.sum(self.data * w[..., None], axis=axis) / np.sum(w, axis=axis), ref)

  def unique(self, tol=1e-6, return_index=False, return_inverse=False, **kwargs):
    """the distinct tangent vectors at one shared reference up to a tolerance, in order of first
    appearance; with return_index and return_inverse the index maps as well"""
    ref = self._oneReference()
    out = vector3d(self.data).unique(tol=tol, stable=True, return_index=True, return_inverse=True)
    u = self._rebuild(out[0].data, ref)
    out = (u,) + ((out[1],) if return_index else ()) + ((out[2],) if return_inverse else ())
    return out if len(out) > 1 else u

  # ------------------------------------------------------------------ display

  # the components in x, y, z of the frame of the side, as MATLAB prints a tangent vector
  def __repr__(self):
    style, self.dispStyle = self.dispStyle, "xyz"
    try:
      lines = vector3d.__repr__(self).split("\n")
    finally:
      self.dispStyle = style
    lines[0] = f"SO3TangentVector ({self.side}, {self._headerChar()})"
    ref = self.oriRef
    what = ("misorientation" if ref.isMisorientation else "orientation") + f" ({ref._frameSummary()})" if isinstance(ref, orientation) else "rotation"
    lines.insert(1 + (self.size != 1), f"  reference: {what}" + ("" if ref.ndim == 0 else ", one per element"))
    return aligned("\n".join(lines))

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from ..plotting.spaceplot import quiver3d


# ------------------------------------------------------------------ construction helpers

def _reference(ref):
  if isinstance(ref, rotation):
    return ref
  if isinstance(ref, (quaternion, orthogonalTransform)):
    return rotation(ref)
  raise TypeError("a tangent vector needs the reference rotation or orientation its tangent space is located at: SO3TangentVector(v, oriRef, side)")


# one reference shared by all vectors, or one per vector; a list of another size is broadcast
def _fitShape(ref, shape):
  if ref.ndim == 0 or ref.shape == shape:
    return ref
  try:
    data = np.broadcast_to(ref.data, shape + (4,)).copy()
  except ValueError:
    raise ValueError("the sizes of the tangent vectors and their reference orientations do not match") from None
  return ref._new(data)


def _components(v):
  if isinstance(v, vector3d):
    return v.data.copy()
  # a spin tensor, the tensor class or plain skew matrices
  v = np.asarray(v.M if hasattr(v, "M") and getattr(v, "rank", None) == 2 else v, dtype=float)
  if v.ndim >= 2 and v.shape[-2:] == (3, 3):
    return _axialVector(v)
  if v.ndim >= 1 and v.shape[-1] == 3:
    return v
  raise ValueError(f"a tangent vector takes a vector3d, coordinates of shape (..., 3) or skew matrices of shape (..., 3, 3), got {v.shape}")


def _styleFor(frame):
  from .frames import crystalFrame
  if not isinstance(frame, crystalFrame):
    return "xyz"
  return "UVTW" if frame.lattice.isTriHex else "uvw"


# two references are the same when their frames are identical handles and their rotations agree
def _sameReference(a, b):
  if isinstance(a, orientation) != isinstance(b, orientation):
    return False
  if isinstance(a, orientation) and (a.frameA is not b.frameA or a.frameB is not b.frameB):
    return False
  return rotation._fromData(a.data).angle(rotation._fromData(b.data)) < REFERENCE_TOLERANCE


# ------------------------------------------------------------------ log and exp

# the side may stand in the place of the reference, log(q, 'right')
def _refAndSide(ref, side):
  if isinstance(ref, str):
    ref, side = None, ref
  return ref, _side(side)


def logRotation(q, ref=None, side="left"):
  """rotations carry no symmetry: the vector of the quotient on the chosen side"""
  ref, side = _refAndSide(ref, side)
  if ref is None:
    return SO3TangentVector(vector3d(rotationVector(q.data)), rotation.id, side)
  ref = _reference(ref)
  m = qmul(q.data, _conj(ref.data)) if side == "left" else qmul(_conj(ref.data), q.data)
  return SO3TangentVector(vector3d(rotationVector(m)), ref, side)


def logOrientation(o, ref=None, side="left", symmetry=True):
  """orientations: the pair is reduced by symmetry first, as angle does it, so that the norm
  is angle(ori, ref) and the vector does not depend on the representatives"""
  ref, side = _refAndSide(ref, side)
  if ref is None:
    ref = orientation.id(o.frameA, o.frameB)
  elif isinstance(ref, orientation):
    frames = (SESSION.joinFrames(o.frameA, ref.frameA), SESSION.joinFrames(o.frameB, ref.frameB))
    if frames[0] is not ref.frameA or frames[1] is not ref.frameB:
      ref = orientation._fromRotation(ref.data, frames, ref.antipodal)
  else:
    ref = orientation._fromRotation(_reference(ref).data, (o.frameA, o.frameB))
  q = o.data if not symmetry else o.projectIntoFundamentalRegion(ref).data
  m = qmul(q, _conj(ref.data)) if side == "left" else qmul(_conj(ref.data), q)
  return SO3TangentVector(vector3d(rotationVector(m)), ref, side)


def expVector(v, ref=None, side=None):
  """a plain rotation vector: at the identity a rotation, in a crystal frame the misorientation
  of that frame with itself; with a reference the tangent vector at it, on the right when
  the vector is a crystal direction"""
  from .frames import crystalFrame
  if isinstance(v, SO3TangentVector):
    return v.exp()
  crystal = isinstance(v.frame, crystalFrame)
  if ref is None:
    q = rotationOf(v.data)
    return orientation._fromRotation(q, (v.frame, v.frame)) if crystal else rotation._fromData(q)
  if side is None:
    side = "right" if crystal else "left"
  return SO3TangentVector(v, ref, side).exp()


def geodesic(a, b, s):
  """the geodesic from a to b: exp(s * log(b, a)), s in [0, 1]; a NaN endpoint yields the other"""
  out = (np.asarray(s, dtype=float) * a._coerce(b).log(a)).exp()
  aNaN, bNaN = np.isnan(a.data).any(axis=-1)[..., None], np.isnan(b.data).any(axis=-1)[..., None]
  return out._new(np.where(aNaN, b.data, np.where(bNaN, a.data, out.data)))


def left(t):
  """left(t) is t.left(): the tangent vectors on the left side"""
  return t.left()


def right(t):
  """right(t) is t.right(): the tangent vectors on the right side"""
  return t.right()


__all__ = ["SO3TangentVector", "spinTensor", "left", "right", "geodesic"]
