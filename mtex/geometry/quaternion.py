# quaternions, rotations and orthogonal transforms (Julia ADR 0001, 0003)
#
#   quaternion(a, b, c, d)      the algebra: any norm, q and -q differ
#   rotation(a, b, c, d)        SO(3): a unit quaternion, equal up to sign, always proper
#   orthogonalTransform(rot, improper)   O(3): a rotation and a parity; point group
#                               elements, reflections, the inversion
#
# The coefficients sit on the trailing axis; the orthogonal transform keeps its parity as
# a fifth coordinate, +1 or -1, so indexing, cat and reshape treat it like any other
# container. Products do not renormalise; construction from user input does.
from __future__ import annotations

from typing import TYPE_CHECKING, Any
import numpy as np

from ._array import MtexArray
from .display import display, sizeString, table, takeRows
from .representations import (EULER_LABELS, axis2quat, eulerConvention, euler2quat, eulerTriple, fitKernel, isnull,
                              map1Kernel, map2Kernel, mat2quat, quat2euler, quat2mat, quat2rodrigues,
                              quatAngle, quatAxis, rodrigues2quat)
from .homochoric import cubo2homo, homo2cubo, homochoric2quat, quat2homochoric
from .kernels import qconj, qmul, qrot
from .units import degree
from .vector3d import classproperty, vector3d
from .display import aligned

if TYPE_CHECKING:
  from .vocabulary import EulerConventionName


def _readCoefficients(cls, args, n):
  if len(args) == n:
    return np.stack(np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in args)), axis=-1), None
  if len(args) == 1:
    a = args[0]
    if isinstance(a, MtexArray):
      return None, a
    a = np.asarray(a, dtype=float)
    if a.ndim >= 1 and a.shape[-1] == n:
      return a, None
    if a.ndim == 2 and a.shape[0] == n:
      return a.T, None
    raise ValueError(f"{cls.__name__} needs coefficients of shape (..., {n}) or ({n}, N), got {a.shape}")
  if len(args) == 0:
    return np.zeros((0, n)), None
  raise TypeError(f"{cls.__name__} takes {n} coefficients or one array of them")


class quaternion(MtexArray):
  """quaternions a + b i + c j + d k, the coefficients along the last axis; q * q2 is the product,
  q * v rotates a vector

    quaternion(a, b, c, d)
    quaternion(abcd)                     an array of shape (..., 4) or (4, N)"""
  _ncoord = 4

  def __init__(self, *args, dtype=None):
    data, obj = _readCoefficients(quaternion, args, 4)
    if obj is not None:
      if isinstance(obj, orthogonalTransform):
        data = obj.data[..., :4].copy()
      elif isinstance(obj, quaternion):
        data = obj.data.copy()
      else:
        raise TypeError(f"cannot build a quaternion from a {type(obj).__name__}")
    super().__init__(data, dtype)

  # ------------------------------------------------------------------ constructors

  @classproperty
  def id(cls):
    """the identity, 1 + 0i + 0j + 0k (MATLAB's id and eye)"""
    return cls(1.0, 0.0, 0.0, 0.0)

  @classmethod
  def nan(cls, *shape):
    """quaternions of NaN coefficients, of the shape given"""
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return cls(np.full(tuple(shape) + (4,), np.nan))

  @classmethod
  def rand(cls, *shape, rng=None, maxAngle=None):
    """uniformly distributed rotations, the Haar measure; with maxAngle restricted to the ball
    of that rotational angle about the identity, the angle by inverting its distribution
    (w - sin w) / pi, the axis uniform"""
    rng = np.random.default_rng() if rng is None else rng
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    if maxAngle is not None and maxAngle < np.pi:
      w = np.linspace(0, maxAngle, 4001)
      cdf = w - np.sin(w)
      omega = np.interp(rng.random(shape) * cdf[-1], cdf, w)
      axis = rng.standard_normal(tuple(shape) + (3,))
      axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
      return cls(np.concatenate([np.cos(omega / 2)[..., None], np.sin(omega / 2)[..., None] * axis], axis=-1))
    alpha = 2 * np.pi * rng.random(shape)
    beta = np.arccos(2 * rng.random(shape) - 1)
    gamma = 2 * np.pi * rng.random(shape)
    return cls(euler2quat(alpha, beta, gamma, "Matthies"))

  # ------------------------------------------------------------------ coefficients

  @property
  def a(self):
    """the real part a of the quaternions"""
    return self.data[..., 0]

  @a.setter
  def a(self, value):
    """sets the real part a of the quaternions"""
    self.data[..., 0] = value

  @property
  def b(self):
    """the coefficient b of i"""
    return self.data[..., 1]

  @b.setter
  def b(self, value):
    """sets the coefficient b of i"""
    self.data[..., 1] = value

  @property
  def c(self):
    """the coefficient c of j"""
    return self.data[..., 2]

  @c.setter
  def c(self, value):
    """sets the coefficient c of j"""
    self.data[..., 2] = value

  @property
  def d(self):
    """the coefficient d of k"""
    return self.data[..., 3]

  @d.setter
  def d(self, value):
    """sets the coefficient d of k"""
    self.data[..., 3] = value

  @property
  def abcd(self):
    """the coefficients a, b, c, d as one array of shape (..., 4)"""
    return self.data

  # ------------------------------------------------------------------ the algebra

  # q1 * q2 is the product, q * v rotates, q * s scales
  def __mul__(self, other):
    if isinstance(other, orthogonalTransform):
      return orthogonalTransform(self, False) * other
    if isinstance(other, quaternion):
      return self._product(other)
    if isinstance(other, vector3d):
      return other._rotated(self.data)
    if isinstance(other, MtexArray) or hasattr(other, "rotate"):
      return NotImplemented
    return quaternion(self.data * np.asarray(other, dtype=float)[..., None])

  def __rmul__(self, other):
    if isinstance(other, MtexArray) or hasattr(other, "rotate"):
      return NotImplemented
    return quaternion(self.data * np.asarray(other, dtype=float)[..., None])

  # the class of a product follows the left factor unless the right one is more special
  def _product(self, other):
    out = other if isinstance(other, rotation) and not isinstance(self, rotation) else self
    return out._new(qmul(self.data, other.data))

  def __truediv__(self, other):
    return quaternion(self.data / np.asarray(other, dtype=float)[..., None])

  def __add__(self, other):
    return quaternion(self.data + self._coerce(other).data)

  def __sub__(self, other):
    return quaternion(self.data - self._coerce(other).data)

  def __neg__(self):
    return quaternion(-self.data)

  def _coerce(self, other):
    if isinstance(other, quaternion):
      return other
    raise TypeError(f"expected a quaternion, got {type(other).__name__}")

  def conj(self):
    """the conjugate quaternion a - b i - c j - d k, of a rotation its inverse"""
    return self._new(qconj(self.data))

  def inv(self):
    """the inverse quaternion, the conjugate over the squared norm"""
    return quaternion(self.conj().data / np.sum(self.data * self.data, axis=-1)[..., None])

  def norm(self):
    """the norm of the quaternions, sqrt(a^2 + b^2 + c^2 + d^2)"""
    return np.linalg.norm(self.data, axis=-1)

  def normalize(self):
    """the quaternions divided by their norm"""
    return self._new(self.data / self.norm()[..., None])

  def dot(self, other, antipodal=True):
    """the inner product of the coefficients"""
    return np.sum(self.data * self._coerce(other).data, axis=-1)

  def angle(self, other=None):
    """the rotational angle in [0, pi], or the angle between two, both from atan2 forms that
    keep their digits near 0 and pi"""
    if other is None:
      return quatAngle(self.data)
    d = np.sum(self.data * other.data[..., :4], axis=-1)
    s = np.where(d < 0, -1.0, 1.0)[..., None]
    h = np.linalg.norm(self.data - s * other.data[..., :4], axis=-1)
    return 4 * np.arcsin(np.minimum(h / 2, 1.0))

  def axis(self):
    """the rotational axis, a unit vector"""
    return vector3d(quatAxis(self.data), isNormalized=True)

  # two quaternions are equal when their coefficients are; isclose compares within a tolerance
  def __eq__(self, other):
    if not isinstance(other, quaternion):
      return NotImplemented
    return np.all(self.data[..., :4] == other.data[..., :4], axis=-1)

  def isclose(self, other, tol=5e-4):
    """equal within the angular tolerance tol, MATLAB's eq"""
    return self.dot(other) > np.cos(tol)

  def __ne__(self, other):
    return ~(self == other)

  def isnan(self):
    """whether a quaternion has a NaN coefficient"""
    return np.any(np.isnan(self.data), axis=-1)

  # ------------------------------------------------------------------ representations

  def Euler(self, convention: EulerConventionName | Any = None):
    """the Euler angles (alpha, beta, gamma) in the convention named, the session's by default

      phi1, Phi, phi2 = rot.Euler()
      alpha, beta, gamma = rot.Euler('ZYZ')"""
    return quat2euler(self.data, convention)

  @property
  def phi1(self):
    """the first Bunge Euler angle phi1"""
    return quat2euler(self.data, "Bunge")[0]

  @property
  def Phi(self):
    """the second Bunge Euler angle Phi"""
    return quat2euler(self.data, "Bunge")[1]

  @property
  def phi2(self):
    """the third Bunge Euler angle phi2"""
    return quat2euler(self.data, "Bunge")[2]

  def matrix(self):
    """the rotation matrices, of shape (..., 3, 3)"""
    return quat2mat(self.data)

  def Rodrigues(self):
    """the Rodrigues vectors, the axes scaled by tan(omega/2)"""
    return vector3d(quat2rodrigues(self.data))

  def homochoric(self):
    """the homochoric vectors, the axes scaled by (3/4 (omega - sin omega))^(1/3)"""
    return vector3d(quat2homochoric(self.data))

  def cubochoric(self):
    """the cubochoric vectors, the homochoric ball mapped onto the cube"""
    return vector3d(homo2cubo(quat2homochoric(self.data)))

  def log(self, ref=None, side="left"):
    """the tangent vector pointing from the reference, the identity by default, to this rotation"""
    from .tangent import logRotation
    return logRotation(self, ref, side)

  def exp(self, v, side=None):
    """the rotation a tangent vector at this reference points to, exp(ori, v) of MATLAB"""
    from .tangent import SO3TangentVector
    # a tangent vector carries its reference, which is this rotation
    if isinstance(v, SO3TangentVector):
      return v.exp()
    return v.exp(self, side)

  def geodesic(self, other, s):
    """the point at s in [0, 1] on the shortest path from this rotation to another"""
    from .tangent import geodesic
    return geodesic(self, other, s)

  # ------------------------------------------------------------------ lists

  # the disjoint list up to a tolerance in the coefficients
  def _uniqueKey(self, antipodal):
    abcd = self.data.reshape(-1, 4)
    if antipodal:
      a, b, c, d = abcd.T
      return np.stack([a * a, b * b, c * c, d * d, a * b, a * c, a * d, b * c, b * d, c * d], axis=-1)
    return abcd

  def unique(self, tol=1e-3, antipodal=False, stable=False, return_index=False, return_inverse=False):
    """the quaternions with those equal up to the tolerance in the coefficients removed; antipodal=True
    takes q and -q as one; optionally the indices of the kept ones and the index of each into them"""
    from .vector3d import _uniqueRows
    iv, iu = _uniqueRows(self._uniqueKey(antipodal), tol, stable)
    u = self._new(self.data.reshape(-1, self._ncoord)[iv])
    out = (u,) + ((iv,) if return_index else ()) + ((iu,) if return_inverse else ())
    return out if len(out) > 1 else u

  def interp(self, values=None, *flags, **kwargs):
    """interp(ori, values, 'harmonic' | 'bingham' | 'density'): a function on SO(3) through
    values at the rotations, by the function spaces above this layer, MATLAB's rotation/interp"""
    from ..functions.rbfinterpolation import interpSO3
    return interpSO3(self, values, *flags, **kwargs)

  def mean(self, weights=None, robust=False):
    """the mean, the principal eigenvector of the weighted moment matrix; NaN entries left out,
    robust=True drops the outliers and averages again"""
    q = self.data.reshape(-1, 4)
    w = np.ones(q.shape[0]) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
    # an entry that is not a number, a site without an orientation, does not enter
    finite = np.all(np.isfinite(q), axis=-1)
    if not np.all(finite):
      q, w = q[finite], w[finite]
    if q.shape[0] == 0:
      return self._new(np.full(4, np.nan))
    T = (q * w[:, None]).T @ q
    vals, vecs = np.linalg.eigh(T)
    m = self._new(vecs[:, -1])
    if robust and q.shape[0] > 4:
      omega = m.angle(self._new(q))
      keep = omega <= np.quantile(omega, 0.8) * 2.5
      if not np.all(keep):
        return self._new(q[keep]).mean(weights=None if weights is None else w[keep], robust=True)
    return m

  def std(self, weights=None, robust=False):
    """the mean angular deviation from the mean, quaternion/std; for orientations the angle is
    the symmetry-reduced one, as the mean is"""
    if self.size == 0:
      return float("nan")
    m = self.mean(weights=weights, robust=robust)
    return float(np.sqrt(np.nanmean(m.angle(self) ** 2)))

  # ------------------------------------------------------------------ display

  # one rotation as MATLAB's char writes it, its Euler angles to three digits
  def _eulerLine(self):
    e = np.stack(quat2euler(self.data.reshape(-1, 4)[:1], eulerConvention()), axis=-1)[0] / degree
    return "(" + ",".join(f"{0.0 if abs(x) < 1e-8 else x:.3g}°" for x in e) + ")"

  def _eulerTable(self):
    """the Euler angles of the rows a display shows, under the name of their convention"""
    if not self.size:
      return None
    take, pos = takeRows(self.size)
    convention = eulerConvention()
    e = np.stack(quat2euler(self.data.reshape(-1, 4)[take], convention), axis=-1) / degree
    return f"  {convention} Euler angles in degree\n" + table(EULER_LABELS[convention], e.T, tiny=1e-8, rows=pos)

  def _sizeField(self):
    return ("size", sizeString(self.shape) if self.size != 1 else None)

  def __repr__(self):
    take, pos = takeRows(self.size)
    body = table(["a", "b", "c", "d"], self.data.reshape(-1, 4)[take].T, digits=4, tiny=1e-13, rows=pos) if self.size else None
    return display(type(self).__name__, [self._sizeField()], body)


# the elements of a point group, or of the frame carrying one, as MATLAB's rotation(cs)
def _groupElements(x):
  x = getattr(x, "sym", x)
  return x.rot if hasattr(x, "rot") and hasattr(x, "pointGroup") else None


class rotation(quaternion):
  """rotations, unit quaternions with q and -q one rotation; numbers are normalised, a point group
  or a frame gives its elements, an improper orthogonalTransform is refused

    rotation(a, b, c, d)
    rotation(q)
    rotation(cs)"""

  # a rotation from numbers normalises; from an orthogonal transform only when proper
  def __init__(self, *args, dtype=None):
    if len(args) == 1 and _groupElements(args[0]) is not None:
      args = (_groupElements(args[0]),)
    data, obj = _readCoefficients(rotation, args, 4)
    if obj is not None:
      if isinstance(obj, orthogonalTransform):
        if np.any(obj.isImproper):
          raise ValueError("an improper orthogonalTransform is not a rotation")
        data = obj.data[..., :4].copy()
      elif isinstance(obj, rotation):
        data = obj.data.copy()
      elif isinstance(obj, quaternion):
        data = obj.data / obj.norm()[..., None]
      else:
        raise TypeError(f"cannot build a rotation from a {type(obj).__name__}")
    elif data.size:
      data = data / np.linalg.norm(data, axis=-1, keepdims=True)
    MtexArray.__init__(self, data, dtype)

  # ------------------------------------------------------------------ constructors

  @classproperty
  def inversion(cls):
    """the inversion, the orthogonalTransform -I"""
    return orthogonalTransform(cls.id, True)

  @classmethod
  def byEuler(cls, a, b=None, c=None, convention: EulerConventionName | Any = None):
    """the rotations of the Euler angles, Bunge's ZXZ convention unless a convention is named

      rotation.byEuler(phi1, Phi, phi2)
      rotation.byEuler(alpha, beta, gamma, 'ZYZ')"""
    a, b, c, rest = eulerTriple(a, b, c, () if convention is None else (convention,))
    return cls._fromData(euler2quat(a, b, c, rest[0] if rest else None))

  @classmethod
  def byAxisAngle(cls, v, omega):
    """the rotations by the angle omega about the axis v"""
    return cls._fromData(axis2quat(_xyz(v), omega))

  @classmethod
  def byMatrix(cls, M):
    """a proper orthogonal matrix, or a list of them on the leading axes"""
    M = np.asarray(M, dtype=float)
    if np.any(np.linalg.det(M) < 0):
      raise ValueError("the matrix has determinant -1; use orthogonalTransform.byMatrix")
    return cls._fromData(mat2quat(M))

  @classmethod
  def byRodrigues(cls, v):
    """the rotations of Rodrigues vectors, the axes scaled by tan(omega/2)"""
    return cls._fromData(rodrigues2quat(_xyz(v)))

  @classmethod
  def byHomochoric(cls, v):
    """the rotations of homochoric vectors"""
    return cls._fromData(homochoric2quat(_xyz(v)))

  @classmethod
  def byCubochoric(cls, v):
    """the rotations of cubochoric vectors"""
    return cls._fromData(homochoric2quat(cubo2homo(_xyz(v))))

  @classmethod
  def map(cls, u1, v1, u2=None, v2=None):
    """the smallest rotation taking u1 to v1, or the one taking u1 to v1 and u2 to v2"""
    if u2 is None:
      return cls._fromData(map1Kernel(_xyz(u1), _xyz(v1)))
    return cls._fromData(map2Kernel(_xyz(u1), _xyz(v1), _xyz(u2), _xyz(v2)))

  @classmethod
  def fit(cls, u, v, weights=None, method="horn"):
    """the rotation best mapping the list u onto the list v, by Horn's quaternion method or
    Kabsch's matrix method"""
    return cls._fromData(fitKernel(_xyz(u), _xyz(v), weights, method))

  @classmethod
  def _fromData(cls, data):
    obj = object.__new__(cls)
    obj.data = np.asarray(data, dtype=float)
    return obj

  # ------------------------------------------------------------------ the algebra

  def __mul__(self, other):
    if isinstance(other, orthogonalTransform):
      return orthogonalTransform(self, False) * other
    if isinstance(other, quaternion):
      return self._product(other)
    if isinstance(other, vector3d):
      return other._rotated(self.data)
    if isinstance(other, MtexArray) or hasattr(other, "rotate"):
      return NotImplemented
    return quaternion(self.data * np.asarray(other, dtype=float)[..., None])

  # -r is the rotation followed by the inversion
  def __neg__(self):
    return orthogonalTransform(self, True)

  @classmethod
  def cat(cls, *arrays, axis=0):
    """a list joining rotations with an improper transform is an orthogonalTransform"""
    if any(isinstance(a, orthogonalTransform) for a in arrays):
      return orthogonalTransform.cat(*(a if isinstance(a, orthogonalTransform) else orthogonalTransform(a) for a in arrays), axis=axis)
    return super().cat(*arrays, axis=axis)

  def inv(self):
    """the inverse rotation, the conjugate quaternion"""
    return self.conj()

  def dot(self, other, antipodal=True):
    """the inner product up to sign, since q and -q are one rotation"""
    d = np.sum(self.data * self._coerce(other).data, axis=-1)
    if isinstance(other, orthogonalTransform):
      d = d * ~other.isImproper
    return d if not antipodal else np.abs(d)

  def _coerce(self, other):
    if isinstance(other, quaternion):
      return other
    if isinstance(other, orthogonalTransform):
      return other
    raise TypeError(f"expected a rotation, got {type(other).__name__}")

  # two rotations are equal when their quaternions are, q and -q being one rotation
  def __eq__(self, other):
    if isinstance(other, orthogonalTransform):
      return orthogonalTransform(self, False) == other
    if not isinstance(other, quaternion):
      return NotImplemented
    a, b = self.data[..., :4], other.data[..., :4]
    return np.all(a == b, axis=-1) | np.all(a == -b, axis=-1)

  def isclose(self, other, tol=5e-4):
    """equal within the angular tolerance tol, MATLAB's eq"""
    if isinstance(other, orthogonalTransform):
      return orthogonalTransform(self, False).isclose(other, tol)
    return self.dot(other) > np.cos(tol)

  def __ne__(self, other):
    return ~(self == other)

  def _uniqueKey(self, antipodal):
    return quaternion._uniqueKey(self, True)

  # ------------------------------------------------------------------ display

  def __repr__(self):
    return display(type(self).__name__, [self._sizeField()], self._eulerTable())

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from ..functions.discrepancy import discrepancy
    from ..plotting.spaceplot import plot3d, scatterOrientations as scatter


class orthogonalTransform(MtexArray):
  """orthogonal transforms, a rotation and a parity as a fifth coordinate, -1 for an improper one,
  the rotation followed by the inversion

    orthogonalTransform(rot)
    orthogonalTransform(rot, True)       the improper transform -rot
    orthogonalTransform(cs)              the elements of a point group"""
  _ncoord = 5

  def __init__(self, rot=None, improper=False, dtype=None):
    if rot is None:
      data = np.zeros((0, 5))
    elif _groupElements(rot) is not None:
      data = _groupElements(rot).data.copy()
    elif isinstance(rot, orthogonalTransform):
      data = rot.data.copy()
    elif isinstance(rot, quaternion):
      q = rot.data if isinstance(rot, rotation) else rot.data / rot.norm()[..., None]
      parity = np.where(np.broadcast_to(np.asarray(improper, dtype=bool), q.shape[:-1]), -1.0, 1.0)
      data = np.concatenate([q, parity[..., None]], axis=-1)
    else:
      data = np.asarray(rot, dtype=float)
      if data.shape[-1] == 4:
        data = np.concatenate([data, np.ones(data.shape[:-1] + (1,))], axis=-1)
    super().__init__(data, dtype)

  @classproperty
  def id(cls):
    """the identity transform"""
    return cls(rotation.id, False)

  @classproperty
  def inversion(cls):
    """the inversion -I"""
    return cls(rotation.id, True)

  @classmethod
  def byMatrix(cls, M):
    """an orthogonal matrix of either sign"""
    M = np.asarray(M, dtype=float)
    improper = np.linalg.det(M) < 0
    q = mat2quat(np.where(improper[..., None, None], -M, M))
    return cls(rotation._fromData(q), improper)

  @classmethod
  def byEuler(cls, a, b=None, c=None, convention: EulerConventionName | Any = None):
    """the proper transforms of the Euler angles, Bunge's ZXZ convention unless one is named"""
    return cls(rotation.byEuler(a, b, c, convention), False)

  @classmethod
  def byAxisAngle(cls, v, omega):
    """the proper transforms rotating by omega about the axis v"""
    return cls(rotation.byAxisAngle(v, omega), False)

  # ------------------------------------------------------------------ parts

  @property
  def rot(self):
    """the rotational part, a rotation"""
    return rotation._fromData(self.data[..., :4])

  @property
  def parity(self):
    """the parity, 1 for a proper and -1 for an improper transform"""
    return self.data[..., 4]

  @property
  def isImproper(self):
    """whether the transform is improper, its determinant -1"""
    return self.data[..., 4] < 0

  @property
  def i(self):
    """whether the transform is improper, MATLAB's rot.i"""
    return self.isImproper

  @property
  def a(self):
    """the coefficient a of the rotational part"""
    return self.data[..., 0]

  @property
  def b(self):
    """the coefficient b of the rotational part"""
    return self.data[..., 1]

  @property
  def c(self):
    """the coefficient c of the rotational part"""
    return self.data[..., 2]

  @property
  def d(self):
    """the coefficient d of the rotational part"""
    return self.data[..., 3]

  # ------------------------------------------------------------------ the algebra

  def _other(self, other):
    if isinstance(other, orthogonalTransform):
      return other.data[..., :4], other.parity
    if isinstance(other, rotation):
      return other.data, np.ones(other.shape)
    return None, None

  def __mul__(self, other):
    from .orientation import orientation
    if isinstance(other, orientation):
      raise TypeError("orthogonalTransform * orientation is not defined; orientations are proper")
    q, p = self._other(other)
    if q is not None:
      return self._new(np.concatenate([qmul(self.data[..., :4], q), (self.parity * p)[..., None]], axis=-1))
    if isinstance(other, vector3d):
      return other._new(qrot(self.data[..., :4], other.data) * self.parity[..., None])
    return NotImplemented

  def __rmul__(self, other):
    if isinstance(other, rotation):
      return orthogonalTransform(other, False) * self
    return NotImplemented

  def __neg__(self):
    return self._new(self.data * [1, 1, 1, 1, -1])

  def inv(self):
    """the inverse transform, the inverse rotation with the same parity"""
    return self._new(self.data * [1, -1, -1, -1, 1])

  def matrix(self):
    """the orthogonal matrices, the rotation matrix times the parity"""
    return self.parity[..., None, None] * quat2mat(self.data[..., :4])

  def angle(self, other=None):
    """the rotational angle of the rotational part, or the angle between the rotational parts of two"""
    return self.rot.angle(None if other is None else other.rot if isinstance(other, orthogonalTransform) else other)

  def axis(self):
    """the rotational axis of the rotational part"""
    return self.rot.axis()

  def Euler(self, convention: EulerConventionName | Any = None):
    """the Euler angles of the rotational part, in the convention named or the session's"""
    return quat2euler(self.data[..., :4], convention)

  def dot(self, other, antipodal=True):
    """the inner product of the rotational parts up to sign, zero between transforms of different
    parity; antipodal=False keeps the sign"""
    q, p = self._other(other)
    d = np.sum(self.data[..., :4] * q, axis=-1) * (self.parity == p)
    return d if not antipodal else np.abs(d)

  # two transforms are equal when their quaternions are, up to sign, and their parities
  def __eq__(self, other):
    q, p = self._other(other)
    if q is None:
      return NotImplemented
    a = self.data[..., :4]
    return ((np.all(a == q, axis=-1) | np.all(a == -q, axis=-1)) & (self.parity == p))

  def isclose(self, other, tol=5e-4):
    """equal within the angular tolerance tol and of one parity, MATLAB's eq"""
    return self.dot(other) > np.cos(tol)

  def __ne__(self, other):
    return ~(self == other)

  def unique(self, tol=1e-3, stable=False, return_index=False, return_inverse=False):
    """the transforms with those equal up to the tolerance removed, a rotation and its negative one;
    optionally the indices of the kept ones and the index of each into them"""
    from .vector3d import _uniqueRows
    key = np.concatenate([self.rot._uniqueKey(True), self.parity.reshape(-1, 1)], axis=-1)
    iv, iu = _uniqueRows(key, tol, stable)
    u = self._new(self.data.reshape(-1, 5)[iv])
    out = (u,) + ((iv,) if return_index else ()) + ((iu,) if return_inverse else ())
    return out if len(out) > 1 else u

  def __repr__(self):
    lines = [type(self).__name__]
    if self.size != 1:
      lines.append(f"  size: {sizeString(self.shape)}")
    if 0 < self.size < 20:
      convention = eulerConvention()
      e = np.stack(quat2euler(self.data.reshape(-1, 5)[:, :4], convention), axis=-1) / degree
      lines.append(f"  {convention} Euler angles in degree")
      lines.append(table(EULER_LABELS[convention] + ("Inv.",), [e[:, 0], e[:, 1], e[:, 2], self.isImproper.reshape(-1).astype(int)], tiny=1e-8))
    return aligned("\n".join(lines))


def reflection(n):
  """a reflection at the plane with normal n"""
  return -rotation.byAxisAngle(n, np.pi)


# the coordinates of a vector3d or of a plain triple
def _xyz(v):
  return v.data if isinstance(v, vector3d) else np.asarray(v, dtype=float)


__all__ = ["quaternion", "rotation", "orthogonalTransform", "reflection"]
