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

from ._array import MtexArray, numpyReduction
from .display import aligned, display, sizeString, table, takeRows
from .representations import (EULER_LABELS, axis2quat, chordAngle, eulerConvention, euler2quat, eulerTriple, fitKernel,
                              isnull, map1Kernel, map2Kernel, mat2quat, quat2euler, quat2mat, quat2rodrigues,
                              quatAngle, quatAxis, rodrigues2quat)
from .homochoric import cubo2homo, homo2cubo, homochoric2quat, quat2homochoric
from .kernels import qconj, qmul, qrot
from .units import degree
from .vector3d import _uniqueOutput, _uniqueRows, _xyz, classproperty, vector3d

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
  """quaternions stored as `a + b*i + c*j + d*k`, with one coefficient tuple per element.

    q = quaternion(a, b, c, d)
    q = quaternion(abcd)

  Separate coefficient arrays broadcast together. A coordinate array has shape (..., 4),
  or (4, n) for a column-oriented list. Construction preserves the norm and sign;
  q and -q are distinct quaternions. Use `rotation` for unit quaternions representing
  proper rotations, where opposite signs represent the same rotation.

  Quaternion multiplication composes products in the usual order. For unit quaternions,
  `q*v` applies the rotation to a vector3d. Indexing and reshaping preserve the container.

  Keywords
    dtype - floating-point type of the stored coefficients
  Properties
    a, b, c, d - real and imaginary coefficients as arrays of the container shape
    data - numeric coefficient array with a trailing axis of length four
    shape - shape of the quaternion array, excluding the four coefficients
  See also
    rotation, orthogonalTransform, quaternion.normalize"""
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
    """construct quaternions whose four coefficients are NaN.

      q = quaternion.nan(10)
      rot = rotation.nan(2, 3)

    Sizes may be separate integers or one tuple; no size gives one scalar element.
    The class on which the method is called determines the result type. Missing rotations
    can be detected with isnan and are omitted by rotation mean calculations."""
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return cls(np.full(tuple(shape) + (4,), np.nan))

  @classmethod
  def rand(cls, *shape, rng=None, maxAngle=None):
    """uniformly distributed rotations, the Haar measure; with maxAngle restricted to the ball
    of that rotational angle about the identity, the angle by inverting its distribution
    (w - sin w) / pi, the axis uniform; rng= a NumPy random generator

      rot = rotation.rand()                            # one
      rot = rotation.rand(100)                         # a list of 100
      rot = rotation.rand(10, 3)                       # a 10 x 3 array
      rot = rotation.rand(100, maxAngle=10*degree)     # within 10 degree of the identity
      rot = rotation.rand(100, rng=np.random.default_rng(0))  # reproducible

    Input
      shape - the sizes of the array, one tuple or several integers
    Output
      rot - quaternions of the class it is called on, rotation or quaternion
    Keywords
      rng - the NumPy random generator, a fresh one by default
      maxAngle - the largest rotational angle about the identity
    See also
      orientation.rand, vector3d.rand, rotation.id"""
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
    """return the quaternion conjugate `a - b*i - c*j - d*k`.

      conjugate = q.conj()

    The real coefficient is unchanged and the three imaginary coefficients change sign.
    For a unit quaternion this is the inverse rotation; for a general quaternion use
    `quaternion.inv` to include division by its squared norm."""
    return self._new(qconj(self.data))

  def inv(self):
    """return the multiplicative inverse of each quaternion.

      inverse = q.inv()

    The inverse is `conj(q)/norm(q)**2`, so `q*inverse` is the identity for nonzero q.
    The result is a quaternion, with the same array shape. Zero quaternions have no
    inverse and produce nonfinite coefficients.

    See also
      quaternion.conj, rotation.inv"""
    return quaternion(self.conj().data / np.sum(self.data * self.data, axis=-1)[..., None])

  def norm(self):
    """the norm of the quaternions, sqrt(a^2 + b^2 + c^2 + d^2)

      n = norm(q)"""
    return np.linalg.norm(self.data, axis=-1)

  def normalize(self):
    """divide each quaternion by its Euclidean coefficient norm.

      unit = q.normalize()

    The result keeps the input class and shape. A zero quaternion produces NaN coefficients.
    Normalisation changes the norm, without choosing between the signs q and -q.

    See also
      quaternion.norm, rotation"""
    return self._new(self.data / self.norm()[..., None])

  def dot(self, other, antipodal=False):
    """the inner product of the coefficients

      d = dot(q1, q2)                               # a b + ... + d d of the coefficients
      d = dot(q1, q2, antipodal=True)               # its absolute value

    The plain inner product of the quaternions as vectors of R^4, signed: q and -q give
    opposite values. rotation.dot takes its absolute value by default.

    Input
      q1, q2 - quaternion
    Output
      d - the inner products
    Keywords
      antipodal - take the absolute value, q and -q read as one
    See also
      rotation.dot, quaternion.angle"""
    d = np.sum(self.data * self._coerce(other).data[..., :4], axis=-1)
    return np.abs(d) if antipodal else d

  def angle(self, other=None):
    """the rotational angle in [0, pi], or the angle between two, both from atan2 forms that
    keep their digits near 0 and pi

      omega = angle(q)                              # the rotational angle
      omega = angle(q1, q2)                         # the angle of q1^-1 q2, between the two

    Input
      q - quaternion
      other - quaternion, the second one; the angle is that of the rotation from one to the
          other
    Output
      omega - the angles in radians
    See also
      quaternion.axis, quaternion.dot, orientation.angle"""
    if other is None:
      return quatAngle(self.data)
    return chordAngle(self.data, other.data[..., :4])

  def axis(self):
    """return a unit vector along the rotation axis of each quaternion.

      a = q.axis()

    The result is a vector3d with the same shape as q. The axis of the identity rotation
    is not geometrically unique; use `quaternion.angle` when only the rotation magnitude
    is needed.

    See also
      rotation.byAxisAngle, quaternion.angle"""
    return vector3d(quatAxis(self.data), isNormalized=True)

  # two quaternions are equal when their coefficients are; isclose compares within a tolerance
  def __eq__(self, other):
    if not isinstance(other, quaternion):
      return NotImplemented
    return np.all(self.data[..., :4] == other.data[..., :4], axis=-1)

  def isclose(self, other, tol=5e-4):
    """equal within the angular tolerance tol, MATLAB's eq

      b = q1.isclose(q2)                            # equal up to 5e-4
      b = q1.isclose(q2, tol=1e-2)                  # a coarser tolerance

    True where the inner product of the coefficients exceeds cos(tol), as MATLAB's eq; for
    unit quaternions the angle between them is below 2 tol. A quaternion and its negative
    are not close; rotation.isclose takes them as one. The operator == compares exactly.

    Input
      q1, q2 - quaternion
    Output
      b - boolean array
    Keywords
      tol - the tolerance, an angle
    See also
      rotation.isclose, quaternion.dot"""
    return self.dot(other) > np.cos(tol)

  def __ne__(self, other):
    return ~(self == other)

  def isnan(self):
    """whether a quaternion has a NaN coefficient"""
    return np.any(np.isnan(self.data), axis=-1)

  # ------------------------------------------------------------------ representations

  def Euler(self, convention: EulerConventionName | Any = None):
    """the Euler angles (alpha, beta, gamma) in the convention named, the session's by default

      phi1, Phi, phi2 = rot.Euler()                 # the session's convention, Bunge by default
      alpha, beta, gamma = rot.Euler('ZYZ')         # Matthies' convention

    The conventions are 'Bunge' or 'ZXZ', 'Matthies', 'ZYZ' or 'ABG', 'Roe', 'Kocks',
    'Canova' and 'nfft'; the default is the preference 'EulerAngleConvention'. Where the
    second angle is 0 only the sum of the first and third is determined, and the first takes
    it.

    Input
      rot - quaternion
    Output
      alpha, beta, gamma - three arrays of angles in radians in the shape of rot
    Keywords
      convention - the name of the convention
    See also
      rotation.byEuler, quaternion.Rodrigues, quaternion.matrix"""
    return quat2euler(self.data, convention)

  @property
  def phi1(self):
    """first Bunge Euler angle phi1 in radians, returned in the shape of the rotations."""
    return quat2euler(self.data, "Bunge")[0]

  @property
  def Phi(self):
    """second Bunge Euler angle Phi in radians, returned in the shape of the rotations."""
    return quat2euler(self.data, "Bunge")[1]

  @property
  def phi2(self):
    """third Bunge Euler angle phi2 in radians, returned in the shape of the rotations."""
    return quat2euler(self.data, "Bunge")[2]

  def matrix(self):
    """return the numeric rotation matrix of each unit quaternion.

      M = rot.matrix()
      rotated = M @ vector

    The output shape is rot.shape + (3, 3). Matrices act on Cartesian column vectors;
    `rot*v` is the vector3d form of the same action. General quaternions should be
    normalised before interpreting the matrix as a proper rotation.

    See also
      rotation.byMatrix, quaternion.Euler"""
    return quat2mat(self.data)

  def Rodrigues(self):
    """express each rotation as an axis scaled by tan(omega/2).

      r = rot.Rodrigues()

    The result is a vector3d of dimensionless Rodrigues coordinates. The identity maps
    to the origin; half-turns lie at infinity, so this representation is singular at pi.

    See also
      rotation.byRodrigues, quaternion.homochoric"""
    return vector3d(quat2rodrigues(self.data))

  def homochoric(self):
    """express rotations as equal-volume coordinates in a ball.

      h = rot.homochoric()

    The vector points along the rotation axis and has length
    `(3*(omega - sin(omega))/4)**(1/3)`, with omega in radians. The identity maps to zero;
    rotations through pi reach the ball radius `(3*pi/4)**(1/3)`.

    See also
      rotation.byHomochoric, quaternion.cubochoric"""
    return vector3d(quat2homochoric(self.data))

  def cubochoric(self):
    """express rotations as equal-volume coordinates in a cube.

      c = rot.cubochoric()

    The homochoric ball is mapped to a cube centred at zero. The result is a vector3d
    whose coordinates are dimensionless, with the same array shape as rot.

    See also
      rotation.byCubochoric, quaternion.homochoric"""
    return vector3d(homo2cubo(quat2homochoric(self.data)))

  def log(self, ref=None, side="left"):
    """the tangent vector pointing from the reference, the identity by default, to this rotation

      t = log(rot)                                  # the rotation vector at the identity
      t = log(rot, rotRef)                          # the tangent vector at a reference
      t = log(rot, rotRef, side='right')            # in the frame of the reference's crystal
      t = log(rot, 'right')                         # the side alone, at the identity

    The logarithmic map: the vector, of length the rotational angle, whose exponential at the
    reference gives the rotation. On the left side the vector is that of rot * inv(rotRef),
    in the specimen's coordinates; on the right that of inv(rotRef) * rot, in the crystal's.
    orientation.log reduces the pair by the symmetry first.

    Input
      rot - rotation
      ref - rotation, the reference, or 'left' or 'right' for the side alone
    Output
      t - SO3TangentVector at the reference
    Keywords
      side - 'left' or 'right', the side the tangent vector is taken on
    See also
      quaternion.exp, SO3TangentVector, geodesic"""
    from .tangent import logRotation
    return logRotation(self, ref, side)

  def exp(self, v, side=None):
    """the rotation a tangent vector at this reference points to, exp(ori, v) of MATLAB

      rot = exp(rotRef, v)                          # v a rotation vector in the specimen's coordinates
      ori = exp(oriRef, Miller(...))                # a crystal direction, taken on the right
      ori = exp(oriRef, v, side='right')            # v in the crystal's coordinates

    The exponential map: the rotation exp(v) * ref on the left side, ref * exp(v) on the
    right, so that log(exp(ref, v), ref) is v again. An SO3TangentVector carries its own
    reference and side, and this reference is ignored. The vectors and the references must
    have one shape; they do not broadcast.

    Input
      ref - rotation or orientation, the reference
      v - vector3d, a rotation vector, or an SO3TangentVector
    Output
      rot - rotation or orientation of the frames of the reference
    Keywords
      side - 'left' or 'right'; by default right for a vector in a crystal frame, left
          otherwise
    See also
      quaternion.log, vector3d.exp, SO3TangentVector"""
    from .tangent import SO3TangentVector
    # a tangent vector carries its reference, which is this rotation
    if isinstance(v, SO3TangentVector):
      return v.exp()
    return v.exp(self, side)

  def geodesic(self, other, s):
    """interpolate along the shortest rotational path between two rotations.

      middle = rot1.geodesic(rot2, 0.5)

    Parameter s=0 gives the first endpoint and s=1 the second. The interpolation uses
    rotation angle rather than a linear average of quaternion coefficients; orientations
    also account for their crystal and specimen symmetries.

    Input
      other - second rotation or orientation
      s - interpolation parameter, usually in [0, 1]
    See also
      geodesic, quaternion.log, quaternion.exp"""
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
    """the disjoint list of quaternions, those equal up to a tolerance in the coefficients taken
    as one

      u = unique(q)                                           # sorted, one of each
      u = unique(q, tol=0.01)                                 # a coarser tolerance
      u = unique(q, stable=True)                              # in the order of first appearance
      u, iq, iu = unique(q, return_index=True, return_inverse=True)  # the index maps

    Two quaternions are one when their coefficients agree to tol after rounding to multiples
    of it, so two close quaternions on either side of a rounding step stay apart. A quaternion
    and its negative, one rotation, are kept apart unless antipodal is set; orientation.unique
    takes them as one.

    Input
      q - quaternion
    Output
      u - quaternion, a one-dimensional list
      iq - indices with u = q.flatten()[iq]
      iu - indices with q.flatten() = u[iu]
    Keywords
      tol - the tolerance in the coefficients
      antipodal - take q and -q as one
      stable - keep the order of first appearance instead of sorting
      return_index - also return iq
      return_inverse - also return iu
    See also
      orientation.unique, vector3d.unique"""
    iv, iu = _uniqueRows(self._uniqueKey(antipodal), tol, stable)
    return _uniqueOutput(self._new(self.data.reshape(-1, self._ncoord)[iv]), iv, iu, return_index, return_inverse)

  def interp(self, values=None, *flags, **kwargs):
    """interp(ori, values, 'harmonic' | 'bingham'): a function on SO(3) through
    values at the rotations, by the function spaces above this layer, MATLAB's rotation/interp

      SO3F = ori.interp(values)                     # a sum of kernels about the nodes
      SO3F = ori.interp(values, halfwidth=10*degree, density=True)  # a density by a given kernel
      SO3F = ori.interp(values, 'harmonic', bandwidth=16)  # in harmonic form
      SO3F = ori.interp(values, 'bingham')          # a Bingham distribution

    The method is the flag, and every keyword goes on to it; interpSO3 lists them.

    Input
      ori - rotation or orientation, the nodes
      values - the values at the nodes, or vectors for a vector field
      flags - 'harmonic' or 'bingham'
    Output
      SO3F - SO3FunRBF, SO3FunHarmonic, SO3FunBingham or a vector field
    See also
      interpSO3, interpolateRBF, interpolate, calcBinghamODF"""
    from ..functions.rbfinterpolation import interpSO3
    return interpSO3(self, values, *flags, **kwargs)

  def mean(self, weights=None, robust=False, return_inertia=False, axis=None, dtype=None, out=None, keepdims=False, **kwargs):
    """the mean of a list of quaternions, with the principal moments and axes of inertia

      m = mean(q)                                  # the mean
      m = mean(q, weights=w)                       # a weighted mean
      m = mean(q, robust=True)                     # outliers left out
      m, lam, V = mean(q, return_inertia=True)     # the moments and axes of inertia too

    The mean is the principal eigenvector of the weighted moment matrix sum w q q^T / sum w,
    so q and -q count alike. Entries that are not a number, sites without an orientation, are
    left out. A robust mean drops the quaternions further from the mean than 2.5 times the 80%
    quantile of the angles and averages again, as long as more than four remain.

    The mean is taken over the whole list; axis is refused, index the list for the mean of a
    part. orientation.mean reduces by the symmetry first.

    Input
      q - quaternion, rotation or orientation
    Output
      m - the mean, of the class of q
      lam - the principal moments of inertia, the eigenvalues of the moment matrix in
          ascending order
      V - quaternion, the four principal axes of inertia, the eigenvectors
    Keywords
      weights - one weight per quaternion
      robust - leave out the outliers
      return_inertia - also return lam and V
      axis - NumPy's reduction axis, refused
      dtype - NumPy's argument, refused
      out - NumPy's argument, refused
      keepdims - keep the reduced axes with length one, as np.mean does
    See also
      quaternion.std, orientation.mean"""
    numpyReduction("mean", dtype, out, kwargs, "rotations")
    if axis is not None:
      raise ValueError("the mean of rotations is taken over the whole list; index the list for the mean of a part")
    if keepdims:
      m = self.mean(weights=weights, robust=robust, return_inertia=return_inertia)
      return (m[0].reshape((1,) * self.ndim),) + m[1:] if return_inertia else m.reshape((1,) * self.ndim)
    q = self.data.reshape(-1, 4)
    w = np.ones(q.shape[0]) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
    # an entry that is not a number, a site without an orientation, does not enter
    finite = np.all(np.isfinite(q), axis=-1)
    if not np.all(finite):
      q, w = q[finite], w[finite]
    if q.shape[0] == 0:
      m = self._new(np.full(4, np.nan))
      return (m, np.array([0.0, 0, 0, 1]), quaternion(np.full((4, 4), np.nan))) if return_inertia else m
    T = (q * w[:, None]).T @ q / np.sum(w)
    vals, vecs = np.linalg.eigh(T)
    m = self._new(vecs[:, -1])
    if robust and q.shape[0] > 4:
      omega = m.angle(self._new(q))
      keep = omega <= np.quantile(omega, 0.8) * 2.5
      if not np.all(keep):
        return self._new(q[keep]).mean(weights=None if weights is None else w[keep], robust=True,
                                       return_inertia=return_inertia)
    return (m, vals, quaternion(vecs.T)) if return_inertia else m

  def std(self, weights=None, robust=False, axis=None, dtype=None, out=None, keepdims=False, ddof=0, **kwargs):
    """the mean angular deviation of a list of rotations from their mean

      sigma = std(rot)                     # the deviation in radians
      sigma = std(rot, robust=True)        # from the robust mean
      sigma = std(rot, weights=w)          # from the weighted mean

    The root mean square of the rotational angles between the mean and every rotation; for
    orientations the angle is the symmetry reduced one, as the mean is. The weights enter the
    mean only, not the average of the angles.

    Input
      rot - quaternion, rotation or orientation
    Output
      sigma - the deviation in radians
    Keywords
      weights - one weight per rotation, for the mean
      robust - deviation from the robust mean, outliers left out of the mean
      axis - NumPy's reduction axis, refused: the deviation is over the whole list
      dtype - NumPy's argument, refused
      out - NumPy's argument, refused
      keepdims - keep the reduced axes with length one, as np.std does
      ddof - NumPy's degrees of freedom, refused
    See also
      quaternion.mean, orientation.mean"""
    numpyReduction("std", dtype, out, kwargs, "rotations")
    if axis is not None:
      raise ValueError("the std of rotations is taken over the whole list; index the list for the std of a part")
    if ddof:
      raise ValueError("the std of rotations is the mean angular deviation; it takes no ddof")
    if keepdims:
      return np.reshape(self.std(weights=weights, robust=robust), (1,) * self.ndim)
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
  """proper rotations represented by unit quaternions, with opposite signs identified.

    rot = rotation(a, b, c, d)
    rot = rotation(q)
    rot = rotation(cs)

  Numeric coefficients and general quaternions are normalised on construction. A point
  group or reference frame supplies its proper rotational parts. Improper orthogonal
  transforms cannot be converted to a rotation; use `orthogonalTransform` to retain parity.

  For vector3d data, `rot*v` rotates each vector. The product `rot1*rot2` applies rot2 first,
  then rot1. Array shapes broadcast, while the trailing four coefficients are internal
  to each element.

  Keywords
    dtype - floating-point type of the stored coefficients
  Properties
    a, b, c, d - unit quaternion coefficients, with opposite signs representing the same rotation
    data - coefficient array of shape rot.shape + (4,)
    phi1, Phi, phi2 - Bunge Euler angles in radians
    shape - shape of the rotation array
  See also
    rotation.byEuler, rotation.byAxisAngle, rotation.byMatrix, quaternion"""

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
    """rotations defined by Euler angles

      rot = rotation.byEuler(phi1, Phi, phi2)              # Bunge's convention
      rot = rotation.byEuler(alpha, beta, gamma, 'ZYZ')    # Matthies' convention
      rot = rotation.byEuler(phi1, Phi, phi2, 'Kocks')     # Kocks' convention
      rot = rotation.byEuler([phi1, Phi, phi2])            # the angles as one triple, or an array of triples

    The angles are in radians, arrays of them broadcast against each other. Without a
    convention the preference EulerAngleConvention decides, Bunge's ZXZ by default.

    Input
      a - the first angle, phi1 or alpha; or an array whose last axis holds the three angles
      b - the second angle, Phi or beta; the convention when a holds the triples
      c - the third angle, phi2 or gamma
    Output
      rot - rotation
    Keywords
      convention - 'Bunge' or 'ZXZ', 'Matthies', 'ZYZ' or 'ABG', 'Roe', 'Kocks', 'Canova',
          'nfft'
    See also
      rotation.Euler, orientation.byEuler, rotation.byAxisAngle, rotation.byMatrix,
      RotationRepresentations"""
    a, b, c, rest = eulerTriple(a, b, c, () if convention is None else (convention,))
    return cls._fromData(euler2quat(a, b, c, rest[0] if rest else None))

  @classmethod
  def byAxisAngle(cls, v, omega):
    """construct right-handed rotations about given axes.

      rot = rotation.byAxisAngle(v, omega)
      rot = rotation.byAxisAngle(vector3d.Z, 30*degree)
      rot = rotation.byAxisAngle(axis, angles)

    Angles are in radians and broadcast with the axes. Each nonzero axis is normalised
    internally, so its length does not change the rotation angle.

    Input
      v - vector3d or Cartesian coordinates of shape (..., 3)
      omega - rotation angles in radians
    Output
      rot - rotation container with the broadcast shape

    Example
    Rotate the x direction by 90 degrees about z, then apply the inverse rotation.
    The axis sets the direction of rotation; its length is immaterial.

      import numpy as np
      from mtex import rotation, vector3d, degree, inv

      rot = rotation.byAxisAngle(vector3d.Z, 90*degree)
      v = vector3d.X
      turned = rot * v
      recovered = inv(rot) * turned
      print('Rotated x:', np.round(turned.data, 6))
      print('Recovered x:', np.round(recovered.data, 6))

    This is an active, right-handed rotation: positive x moves toward positive y.

    See also
      rotation.axis, rotation.angle, rotation.byEuler"""
    return cls._fromData(axis2quat(_xyz(v), omega))

  @classmethod
  def byMatrix(cls, M):
    """construct rotations from proper orthogonal matrices.

      rot = rotation.byMatrix(M)

    Input
      M - numeric array of shape (..., 3, 3), with orthogonal columns and determinant +1
    Output
      rot - rotation with shape M.shape[:-2]

    Matrices of negative determinant are refused. To retain a reflection or inversion,
    use orthogonalTransform.byMatrix. This constructor assumes the supplied matrices
    are orthogonal; it is not a least-squares fit to arbitrary matrices.

    See also
      rotation.matrix, orthogonalTransform.byMatrix, rotation.fit"""
    M = np.asarray(M, dtype=float)
    if np.any(np.linalg.det(M) < 0):
      raise ValueError("the matrix has determinant -1; use orthogonalTransform.byMatrix")
    return cls._fromData(mat2quat(M))

  @classmethod
  def byRodrigues(cls, v):
    """construct rotations from dimensionless Rodrigues vectors.

      rot = rotation.byRodrigues(r)

    The vector direction is the rotation axis and its length is tan(omega/2).
    Input may be vector3d or Cartesian triples with shape (..., 3).

    See also
      rotation.Rodrigues, rotation.byAxisAngle"""
    return cls._fromData(rodrigues2quat(_xyz(v)))

  @classmethod
  def byHomochoric(cls, v):
    """construct rotations from coordinates in the homochoric ball.

      rot = rotation.byHomochoric(h)

    Input may be vector3d or dimensionless Cartesian triples. A vector of length r
    represents a rotation through omega satisfying `r**3 = 3*(omega - sin(omega))/4`.
    Use coordinates inside the ball of radius `(3*pi/4)**(1/3)`.

    See also
      rotation.homochoric, rotation.byCubochoric"""
    return cls._fromData(homochoric2quat(_xyz(v)))

  @classmethod
  def byCubochoric(cls, v):
    """construct rotations from equal-volume cubochoric coordinates.

      rot = rotation.byCubochoric(c)

    Input may be vector3d or Cartesian triples in the cubochoric cube. Coordinates
    are first mapped to the homochoric ball and then converted to rotations.

    See also
      rotation.cubochoric, rotation.byHomochoric"""
    return cls._fromData(homochoric2quat(cubo2homo(_xyz(v))))

  @classmethod
  def map(cls, u1, v1, u2=None, v2=None):
    """the smallest rotation taking u1 to v1, or the one taking u1 to v1 and u2 to v2

      rot = rotation.map(u1, v1)                    # the smallest rotation with rot * u1 || v1
      rot = rotation.map(u1, v1, u2, v2)            # u1 onto v1 and u2 onto v2

    With two pairs the angle between u1 and u2 must be that between v1 and v2, up to 1e-3,
    and u1, u2 must not be colinear; an error is raised otherwise. The lengths of the vectors
    do not matter.

    Input
      u1, v1 - vector3d
      u2, v2 - vector3d, a second pair
    Output
      rot - rotation
    See also
      rotation.fit, rotation.byAxisAngle, rotation.byMatrix"""
    if u2 is None:
      return cls._fromData(map1Kernel(_xyz(u1), _xyz(v1)))
    return cls._fromData(map2Kernel(_xyz(u1), _xyz(v1), _xyz(u2), _xyz(v2)))

  @classmethod
  def fit(cls, u, v, weights=None, method="horn"):
    """the rotation best mapping the list u onto the list v, by Horn's quaternion method or
    Kabsch's matrix method

      rot = rotation.fit(u, v)                      # rot * u is nearest v
      rot = rotation.fit(u, v, weights=w)           # some pairs counting more
      rot = rotation.fit(u, v, method='kabsch')     # by the singular value decomposition

    The rotation maximising the weighted sum of <rot u_i, v_i> over the unit vectors: Horn's
    method takes the largest eigenvector of a 4 x 4 matrix, Kabsch's the singular value
    decomposition of the cross covariance; both give the same rotation. MATLAB's call is
    rotation.fit(r, l) with l = rot * r, the same order. MATLAB's handling of antipodal
    vectors is not ported: the vectors are taken with their sign.

    Input
      u, v - vector3d, lists of one length
    Output
      rot - rotation
    Keywords
      weights - one weight per pair
      method - 'horn' or 'kabsch'
    References
      W. Kabsch, A solution for the best rotation to relate two sets of vectors, Acta Cryst.
          A32 (1976) 922
      B. K. P. Horn, Closed-form solution of absolute orientation using unit quaternions,
          Journal of the Optical Society of America A 4 (1987) 629
    See also
      rotation.map, rotation.byMatrix"""
    return cls._fromData(fitKernel(_xyz(u), _xyz(v), weights, method))

  @classmethod
  def _fromData(cls, data):
    obj = object.__new__(cls)
    obj.data = np.asarray(data, dtype=float)
    return obj

  # ------------------------------------------------------------------ the algebra

  # -r is the rotation followed by the inversion
  def __neg__(self):
    return orthogonalTransform(self, True)

  @classmethod
  def cat(cls, *arrays, axis=0):
    """the list of the operands, MATLAB's [r1, r2]; joining rotations with an improper transform
    gives an orthogonalTransform, orientations stay orientations between the same frames

      rot = cat(rot1, rot2)            # one after the other
      rot = cat(rot1, rot2, axis=1)    # side by side

    Orientations must share their frames.

    Input
      arrays - the rotations, orientations or orthogonal transforms
    Output
      rot - the joined list, of the class of the operands
    Keywords
      axis - the axis along which they are joined, as NumPy's concatenate
    See also
      vector3d.cat"""
    if any(isinstance(a, orthogonalTransform) for a in arrays):
      return orthogonalTransform.cat(*(a if isinstance(a, orthogonalTransform) else orthogonalTransform(a) for a in arrays), axis=axis)
    return super().cat(*arrays, axis=axis)

  def inv(self):
    """return the rotation that reverses each input rotation.

      inverse = rot.inv()

    For unit quaternions, inversion is quaternion conjugation. If `v2=rot*v1`, then
    `inverse*v2=v1`, up to numerical precision. Array shape is preserved.

    See also
      quaternion.conj, orientation.inv"""
    return self.conj()

  def dot(self, other, antipodal=True):
    """the inner product up to sign, since q and -q are one rotation

      d = dot(rot1, rot2)                           # |cos(omega/2)| of the rotation between them
      d = dot(rot1, rot2, antipodal=False)          # with the sign of the quaternions

    The absolute value is cos(omega / 2) of the angle omega between the rotations; the
    product with an improper transform is zero. orientation.dot reduces by the symmetry.

    Input
      rot1, rot2 - rotation
    Output
      d - the inner products
    Keywords
      antipodal - take the absolute value, q and -q being one rotation
    See also
      quaternion.dot, rotation.angle, orientation.dot"""
    d = np.sum(self.data * self._coerce(other).data[..., :4], axis=-1)
    if isinstance(other, orthogonalTransform):
      d = d * ~other.isImproper
    return d if not antipodal else np.abs(d)

  def _coerce(self, other):
    if isinstance(other, (quaternion, orthogonalTransform)):
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
    """equal within the angular tolerance tol, MATLAB's eq

      b = rot1.isclose(rot2)                        # equal up to 5e-4
      b = rot1.isclose(rot2, tol=1*degree)          # a coarser tolerance

    True where |cos(omega / 2)| of the angle omega between them exceeds cos(tol), as
    MATLAB's eq, so where omega is below 2 tol; q and -q are one rotation. An improper
    transform is never close to a rotation. The operator == compares exactly.

    Input
      rot1, rot2 - rotation
    Output
      b - boolean array
    Keywords
      tol - the tolerance, an angle
    See also
      quaternion.isclose, rotation.dot, orientation.isclose"""
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
    orthogonalTransform(rot, True)  # the improper transform -rot
    orthogonalTransform(cs)  # the elements of a point group

  Input
    rot - rotation, quaternion coordinates, or a referenceFrame or symmetry supplying its elements; None selects identity
    improper - whether to apply inversion after the rotation; default False
    dtype - floating-point type of the stored coordinates"""
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
    """proper orthogonal transforms defined by Euler angles

      T = orthogonalTransform.byEuler(phi1, Phi, phi2)              # Bunge's convention
      T = orthogonalTransform.byEuler(alpha, beta, gamma, 'ZYZ')    # Matthies' convention

    The rotations of rotation.byEuler as orthogonal transforms of parity one, to be combined
    with improper ones. The angles are in radians; without a convention the preference
    EulerAngleConvention decides, Bunge's ZXZ by default.

    Input
      a - the first angle, phi1 or alpha; or an array whose last axis holds the three angles
      b - the second angle, Phi or beta; the convention when a holds the triples
      c - the third angle, phi2 or gamma
    Output
      T - orthogonalTransform, proper
    Keywords
      convention - 'Bunge' or 'ZXZ', 'Matthies', 'ZYZ' or 'ABG', 'Roe', 'Kocks', 'Canova',
          'nfft'
    See also
      rotation.byEuler, orthogonalTransform.byMatrix, RotationRepresentations"""
    return cls(rotation.byEuler(a, b, c, convention), False)

  @classmethod
  def byAxisAngle(cls, v, omega):
    """the proper transforms rotating by omega about the axis v

      T = orthogonalTransform.byAxisAngle(v, omega)"""
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
    """the rotational angle of the rotational part, or the angle between the rotational parts of two

      omega = angle(T)                              # the angle of the rotational part
      omega = angle(T1, T2)                         # between the rotational parts

    The parity is not looked at: a transform and its negative have the angle 0 between them.

    Input
      T - orthogonalTransform
      other - orthogonalTransform or rotation, the second one
    Output
      omega - the angles in radians
    See also
      orthogonalTransform.dot, quaternion.angle"""
    return self.rot.angle(None if other is None else other.rot if isinstance(other, orthogonalTransform) else other)

  def axis(self):
    """the rotational axis of the rotational part

      v = axis(T)"""
    return self.rot.axis()

  def Euler(self, convention: EulerConventionName | Any = None):
    """the Euler angles of the rotational part, in the convention named or the session's

      phi1, Phi, phi2 = T.Euler()                   # the session's convention
      alpha, beta, gamma = T.Euler('ZYZ')           # Matthies' convention

    Input
      T - orthogonalTransform
    Output
      alpha, beta, gamma - three arrays of angles in radians
    Keywords
      convention - the name of the convention, as quaternion.Euler
    See also
      quaternion.Euler"""
    return quat2euler(self.data[..., :4], convention)

  def dot(self, other, antipodal=True):
    """the inner product of the rotational parts up to sign, zero between transforms of different
    parity; antipodal=False keeps the sign

      d = dot(T1, T2)                               # |cos(omega/2)|, 0 for different parity
      d = dot(T1, T2, antipodal=False)              # with the sign of the quaternions

    Input
      T1 - orthogonalTransform
      T2 - orthogonalTransform or rotation
    Output
      d - the inner products
    Keywords
      antipodal - take the absolute value
    See also
      orthogonalTransform.isclose, rotation.dot"""
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
    """equal within the angular tolerance tol and of one parity, MATLAB's eq

      b = T1.isclose(T2)                            # equal up to 5e-4
      b = T1.isclose(T2, tol=1*degree)              # a coarser tolerance

    As rotation.isclose, the angle between the rotational parts below 2 tol, and the parities
    equal.

    Input
      T1 - orthogonalTransform
      T2 - orthogonalTransform or rotation
    Output
      b - boolean array
    Keywords
      tol - the tolerance, an angle
    See also
      orthogonalTransform.dot, rotation.isclose"""
    return self.dot(other) > np.cos(tol)

  def __ne__(self, other):
    return ~(self == other)

  def unique(self, tol=1e-3, stable=False, return_index=False, return_inverse=False):
    """the disjoint list of orthogonal transforms, those equal up to a tolerance taken as one

      u = unique(T)                                           # sorted, one of each
      u = unique(T, stable=True)                              # in the order of first appearance
      u, iT, iu = unique(T, return_index=True, return_inverse=True)  # the index maps

    Two transforms are one when their parities agree and their rotational parts agree to tol,
    a rotation and its negative counted as one.

    Input
      T - orthogonalTransform
    Output
      u - orthogonalTransform, a one-dimensional list
      iT - indices with u = T.flatten()[iT]
      iu - indices with T.flatten() = u[iu]
    Keywords
      tol - the tolerance in the quaternion coefficients
      stable - keep the order of first appearance instead of sorting
      return_index - also return iT
      return_inverse - also return iu
    See also
      quaternion.unique, orientation.unique"""
    key = np.concatenate([self.rot._uniqueKey(True), self.parity.reshape(-1, 1)], axis=-1)
    iv, iu = _uniqueRows(key, tol, stable)
    return _uniqueOutput(self._new(self.data.reshape(-1, 5)[iv]), iv, iu, return_index, return_inverse)

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


__all__ = ["quaternion", "rotation", "orthogonalTransform", "reflection"]
