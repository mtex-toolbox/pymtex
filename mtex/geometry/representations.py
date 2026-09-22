# the representations of a rotation as kernels on coefficient arrays: Euler angles in the
# five conventions, axis and angle, matrices, Rodrigues vectors, pairs of directions
#
# A rotation is stored as the quaternion (a, b, c, d) on the trailing axis; every kernel
# here maps between that array and a representation and knows nothing of the classes.
# Every Euler convention is stated as an active product of elementary rotations about the
# fixed axes, right factor first - docs/design/semantics.md S9 to S14.
from __future__ import annotations

import numpy as np

from .kernels import eulerQuat
from .session import getMTEXpref

# a small number is zero, MATLAB isnull
def isnull(x, tol=1e-10):
  return np.abs(x) <= tol


# ------------------------------------------------------------------ Euler angles

# the names a convention is called by, and the angle names of each
EULER_CONVENTIONS = {
  "bunge": "Bunge", "zxz": "Bunge",
  "matthies": "Matthies", "zyz": "Matthies", "abg": "Matthies", "nfft": "Matthies",
  "roe": "Roe", "kocks": "Kocks", "canova": "Canova",
}
EULER_LABELS = {
  "Bunge": ("phi1", "Phi", "phi2"),
  "Matthies": ("alpha", "beta", "gamma"),
  "Roe": ("Psi", "Theta", "Phi"),
  "Kocks": ("Psi", "Theta", "phi"),
  "Canova": ("omega", "Theta", "phi"),
}


def eulerConvention(name=None):
  name = getMTEXpref("EulerAngleConvention") if name is None else name
  try:
    return EULER_CONVENTIONS[str(name).lower()]
  except KeyError:
    raise ValueError(f"unknown Euler angle convention '{name}'") from None


# the other conventions are substitutions on Matthies, geometry_tools/euler2quat
def toMatthies(convention, a, b, c):
  pi = np.pi
  if convention in ("Matthies", "Roe"):
    return a, b, c
  if convention == "Bunge":
    return a - pi / 2, b, c + pi / 2
  if convention == "Kocks":
    return a, b, pi - c
  return pi / 2 - a, b, 3 * pi / 2 - c


# Rz(alpha) Ry(beta) Rz(gamma) as a quaternion; a long list goes through the kernel
def matthies2quat(alpha, beta, gamma):
  alpha, beta, gamma = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (alpha, beta, gamma)))
  if alpha.size >= 1024:
    q = eulerQuat(alpha, beta, gamma)
    if q is not None:
      return q
  hP, hp, hm = beta / 2, (alpha + gamma) / 2, (alpha - gamma) / 2
  cP, sP = np.cos(hP), np.sin(hP)
  return np.stack([cP * np.cos(hp), -sP * np.sin(hm), sP * np.cos(hm), cP * np.sin(hp)], axis=-1)


# byEuler(E, ...) with E an (..., 3) array of triples, as MATLAB's byEuler([phi1 Phi phi2]):
# the three angles and the arguments after them
def eulerTriple(a, b, c, rest):
  isAngle = lambda x: isinstance(x, (int, float, np.number, np.ndarray, list, tuple))
  if b is not None and isAngle(b):
    return a, b, c, rest
  E = np.asarray(a, dtype=float)
  if E.shape[-1:] != (3,):
    raise ValueError("byEuler takes three angles or an array of triples")
  return E[..., 0], E[..., 1], E[..., 2], tuple(x for x in (b, c) if x is not None) + tuple(rest)


def euler2quat(a, b, c, convention=None):
  return matthies2quat(*toMatthies(eulerConvention(convention), a, b, c))


# quaternion to Euler angles, quaternion/Euler: the first and third angle in [0, 2pi),
# the second in [0, pi]; at beta == 0 only the sum of the outer two is determined, and
# it goes into the first angle of the target convention, the third being zero
def quat2euler(q, convention=None):
  convention = eulerConvention(convention)
  pi = np.pi
  a, b, c, d = np.moveaxis(q, -1, 0)
  at1, at2 = np.arctan2(d, a), np.arctan2(b, c)
  beta = 2 * np.arctan2(np.sqrt(b * b + c * c), np.sqrt(a * a + d * d))
  alpha, gamma = np.mod(at1 - at2, 2 * pi), np.mod(at1 + at2, 2 * pi)
  if convention == "Bunge":
    alpha, gamma = np.mod(alpha + pi / 2, 2 * pi), np.mod(gamma - pi / 2, 2 * pi)
  elif convention == "Kocks":
    gamma = np.mod(pi - gamma, 2 * pi)
  elif convention == "Canova":
    alpha, gamma = np.mod(pi / 2 - alpha, 2 * pi), np.mod(3 * pi / 2 - gamma, 2 * pi)
  # the degenerate case: a pure turn about z by omega, written with a vanishing third angle
  ind = isnull(beta)
  if np.any(ind):
    omega = 2 * np.arctan2(d, a)
    if convention == "Kocks":
      omega = omega - pi
    elif convention == "Canova":
      omega = -omega
    alpha = np.where(ind, np.mod(omega, 2 * pi), alpha)
    beta = np.where(ind, 0.0, beta)
    gamma = np.where(ind, 0.0, gamma)
  return alpha, beta, gamma


# ------------------------------------------------------------------ axis and angle

# the identity has no axis; a turn by a multiple of 2pi about nothing is the identity,
# any other turn about nothing is an error rather than a quaternion of NaNs
def axis2quat(xyz, omega):
  xyz = np.asarray(xyz, dtype=float)
  omega = np.asarray(omega, dtype=float)
  n = np.linalg.norm(xyz, axis=-1)
  zero = isnull(n)
  if np.any(zero & ~isnull(np.mod(np.broadcast_to(omega, np.broadcast_shapes(omega.shape, n.shape)), 2 * np.pi))):
    raise ValueError("a rotation by a nonzero angle needs an axis")
  unit = xyz / np.where(zero, 1.0, n)[..., None]
  s = np.sin(omega / 2)[..., None]
  return np.concatenate([np.cos(omega / 2)[..., None] * np.ones_like(unit[..., :1]), s * unit], axis=-1)


# the rotational angle in [0, pi], through atan2 for full accuracy near 0 and pi
def quatAngle(q):
  return 2 * np.arctan2(np.linalg.norm(q[..., 1:], axis=-1), np.abs(q[..., 0]))


# the rotational axis, a unit vector; x for the identity
def quatAxis(q):
  s = np.where(q[..., 0] < 0, -1.0, 1.0)[..., None]
  v = s * q[..., 1:]
  n = np.linalg.norm(v, axis=-1)
  zero = isnull(n)
  v = np.where(zero[..., None], [1.0, 0.0, 0.0], v / np.where(zero, 1.0, n)[..., None])
  return v


# ------------------------------------------------------------------ Rodrigues

def quat2rodrigues(q):
  a = np.where(q[..., 0] == 0, 1e-100, q[..., 0])
  return q[..., 1:] / a[..., None]


def rodrigues2quat(xyz):
  xyz = np.asarray(xyz, dtype=float)
  return axis2quat(xyz, 2 * np.arctan(np.linalg.norm(xyz, axis=-1)))


# ------------------------------------------------------------------ matrices

def quat2mat(q):
  a, b, c, d = np.moveaxis(q, -1, 0)
  M = np.empty(q.shape[:-1] + (3, 3), dtype=q.dtype)
  M[..., 0, 0] = 2 * b * b + 2 * a * a - 1
  M[..., 0, 1] = 2 * b * c - 2 * a * d
  M[..., 0, 2] = 2 * b * d + 2 * a * c
  M[..., 1, 0] = 2 * c * b + 2 * a * d
  M[..., 1, 1] = 2 * c * c + 2 * a * a - 1
  M[..., 1, 2] = 2 * c * d - 2 * a * b
  M[..., 2, 0] = 2 * d * b - 2 * a * c
  M[..., 2, 1] = 2 * d * c + 2 * a * b
  M[..., 2, 2] = 2 * d * d + 2 * a * a - 1
  return M


# direction cosine matrix to quaternion, mat2quat: the largest coefficient is taken from
# the diagonal for accuracy; the matrices must be proper, the caller handles the sign
def mat2quat(M):
  M = np.asarray(M, dtype=float)
  t = np.stack([M[..., 0, 0], M[..., 1, 1], M[..., 2, 2]], axis=-1)
  absQ = np.sqrt(np.maximum(0, np.stack([
    1 + t[..., 0] + t[..., 1] + t[..., 2],
    1 - t[..., 0] - t[..., 1] + t[..., 2],
    1 + t[..., 0] - t[..., 1] - t[..., 2],
    1 - t[..., 0] + t[..., 1] - t[..., 2]], axis=-1))) / 2
  k = np.argmax(absQ, axis=-1)
  q = np.take_along_axis(absQ, k[..., None], axis=-1)[..., 0]
  q4 = 4 * q
  m = lambda i, j: M[..., i, j]
  cand = np.stack([
    np.stack([q, (m(1, 2) - m(2, 1)) / q4, (m(2, 0) - m(0, 2)) / q4, (m(0, 1) - m(1, 0)) / q4], axis=-1),
    np.stack([(m(0, 1) - m(1, 0)) / q4, (m(2, 0) + m(0, 2)) / q4, (m(2, 1) + m(1, 2)) / q4, q], axis=-1),
    np.stack([(m(1, 2) - m(2, 1)) / q4, q, (m(0, 1) + m(1, 0)) / q4, (m(2, 0) + m(0, 2)) / q4], axis=-1),
    np.stack([(m(2, 0) - m(0, 2)) / q4, (m(0, 1) + m(1, 0)) / q4, q, (m(1, 2) + m(2, 1)) / q4], axis=-1),
  ], axis=-2)
  out = np.take_along_axis(cand, k[..., None, None], axis=-2)[..., 0, :]
  # the formulas above give the inverse rotation
  out[..., 1:] *= -1
  return out


# ------------------------------------------------------------------ pairs of directions

def _unit(v):
  v = np.asarray(v, dtype=float)
  return v / np.linalg.norm(v, axis=-1, keepdims=True)


# an arbitrary orthogonal direction, vector3d/orth: (x,y,z) -> (-y,x,0), (0,y,z) -> (1,0,0)
def orthKernel(xyz):
  xyz = np.asarray(xyz, dtype=float)
  y = np.where(isnull(xyz[..., 0]), -1.0, xyz[..., 1])
  return _unit(np.stack([-y, xyz[..., 0], np.zeros_like(y)], axis=-1))


# the smallest rotation with r * u parallel to v, hr2quat
def map1Kernel(u, v):
  u, v = _unit(u), _unit(v)
  n = np.cross(u, v)
  nn = np.linalg.norm(n, axis=-1)
  ax = np.where(isnull(nn)[..., None], orthKernel(u), n / np.where(isnull(nn), 1.0, nn)[..., None])
  omega = np.arctan2(nn, np.sum(u * v, axis=-1))
  return axis2quat(ax, omega)


# the rotation with r * u1 parallel to v1 and r * u2 parallel to v2, rotation/map
def map2Kernel(u1, v1, u2, v2):
  u1, v1, u2, v2 = (_unit(x) for x in (u1, v1, u2, v2))
  ang = lambda a, b: np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a * b, axis=-1))
  delta = np.abs(ang(u1, u2) - ang(v1, v2))
  if np.any(delta >= 1e-3):
    raise ValueError(f"inconsistent pairs of vectors: the angles between u1, u2 and v1, v2 differ by {np.max(delta) * 180 / np.pi:.3g}°")
  if np.any(np.abs(np.sum(u1 * u2, axis=-1)) > 1 - np.finfo(float).eps):
    raise ValueError("the two directions must not be colinear")
  u3, v3 = _unit(np.cross(u1, u2)), _unit(np.cross(v1, v2))
  u2t, v2t = _unit(np.cross(u3, u1)), _unit(np.cross(v3, v1))
  M = v1[..., :, None] * u1[..., None, :] + v2t[..., :, None] * u2t[..., None, :] + v3[..., :, None] * u3[..., None, :]
  return mat2quat(M)


# the rotation best mapping the list u onto the list v, rotation/fit: Horn's quaternion
# method, or Kabsch's singular value decomposition of the cross covariance
def fitKernel(u, v, weights=None, method="horn"):
  u, v = _unit(u).reshape(-1, 3), _unit(v).reshape(-1, 3)
  if weights is not None:
    u = u * np.asarray(weights, dtype=float).reshape(-1, 1)
  M = u.T @ v
  if method.lower() == "kabsch":
    U, _, Vt = np.linalg.svd(M)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return mat2quat(Vt.T @ np.diag([1.0, 1.0, d]) @ U.T)
  if method.lower() != "horn":
    raise ValueError("the fit method is 'horn' or 'kabsch'")
  MS, MA = M + M.T, M - M.T
  N = np.array([
    [np.trace(M), MA[1, 2], MA[2, 0], MA[0, 1]],
    [MA[1, 2], M[0, 0] - M[1, 1] - M[2, 2], MS[0, 1], MS[0, 2]],
    [MA[2, 0], MS[0, 1], M[1, 1] - M[0, 0] - M[2, 2], MS[1, 2]],
    [MA[0, 1], MS[0, 2], MS[1, 2], M[2, 2] - M[0, 0] - M[1, 1]]])
  w, V = np.linalg.eigh(N)
  return V[:, -1]


__all__ = []
