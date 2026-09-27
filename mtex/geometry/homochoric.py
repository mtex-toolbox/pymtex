# homochoric and cubochoric coordinates of a rotation, as kernels on coefficient arrays
#
# The homochoric vector is rho(omega) n with rho^3 = 3/4 (omega - sin omega), the
# equal-volume ball of radius (3pi/4)^(1/3); its inverse is a Newton iteration on omega,
# rotation/byHomochoric. The cube is reached from the ball through the six pyramids of
# cuboRegionId, homo2cubo and cubo2homo (Rosca et al. 2014).
from __future__ import annotations

import numpy as np

from .representations import axis2quat, isnull


def homochoricRadius():
  """the radius (3 pi / 4)^(1/3) of the homochoric ball, the image of the rotations of angle pi"""
  return (3 * np.pi / 4) ** (1 / 3)


def omegaMinusSin(omega):
  """omega - sin omega, with the series for small omega where the difference cancels"""
  w2 = omega * omega
  series = omega ** 3 / 6 * (1 - w2 / 20 * (1 - w2 / 42 * (1 - w2 / 72)))
  return np.where(omega < 0.1, series, omega - np.sin(omega))


def quat2homochoric(q):
  """quaternion/homochoric"""
  a, v = q[..., 0], q[..., 1:]
  n = np.linalg.norm(v, axis=-1)
  zero = isnull(n)
  omega = 2 * np.arctan2(n, np.abs(a))
  f = np.where(a < 0, -1.0, 1.0) / np.where(zero, 1.0, n) * np.cbrt(0.75 * omegaMinusSin(omega))
  return np.where(zero[..., None], 0.0, f[..., None] * v)


def homochoric2quat(xyz):
  """rotation/byHomochoric: Newton on 3/4 (omega - sin omega) = rho^3"""
  xyz = np.asarray(xyz, dtype=float)
  rho = np.linalg.norm(xyz, axis=-1)
  omega = 2 * rho
  for _ in range(20):
    f = 0.75 * omegaMinusSin(omega) - rho ** 3
    df = 1.5 * np.sin(omega / 2) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
      step = f / df
    step = np.where(np.isfinite(step), step, 0.0)
    omega = np.clip(omega - step, 0, np.pi)
    if np.all(np.abs(step) <= np.finfo(float).eps * omega):
      break
  axis = np.where((rho == 0)[..., None], [0.0, 0.0, 1.0], xyz)
  return axis2quat(axis, omega)


def cuboRegionId(v):
  """the pyramid of the cube a point lies in, cuboRegionId, and the permutations of each"""
  x, y, z = np.abs(v[..., 0]), np.abs(v[..., 1]), np.abs(v[..., 2])
  id = np.zeros(x.shape, dtype=int)
  id = np.where((x >= z) & (z >= y), 1, id)
  id = np.where((y >= x) & (x >= z), 2, id)
  id = np.where((y >= z) & (z >= x), 3, id)
  id = np.where((z >= x) & (x >= y), 4, id)
  id = np.where((z >= y) & (y >= x), 5, id)
  return id


CUBO_PERM = np.array([[1, 2, 0], [2, 1, 0], [0, 2, 1], [2, 0, 1], [0, 1, 2], [1, 0, 2]])
CUBO_IPERM = np.array([[2, 0, 1], [2, 1, 0], [0, 2, 1], [1, 2, 0], [0, 1, 2], [1, 0, 2]])


def _permute(v, perm):
  return np.take_along_axis(v, perm, axis=-1)


def homo2cubo(h):
  """homo2cubo"""
  h = np.asarray(h, dtype=float)
  id = cuboRegionId(h)
  w = _permute(h, CUBO_PERM[id])
  wx, wy, wz = w[..., 0], w[..., 1], w[..., 2]
  rho = np.linalg.norm(w, axis=-1)
  safeX = np.where(wx == 0, 1.0, wx)
  safeRho = np.where(rho == 0, 1.0, rho)
  D = np.sqrt(2 / (1 + np.abs(wz) / safeRho))
  E = np.sqrt(2 * wx * wx + wy * wy)
  F = np.sqrt(np.abs(wx) + E)
  G = np.sign(wx)
  H = np.sqrt(np.pi / 12)
  I = (6 / np.pi) ** (1 / 6)
  K = D * np.sqrt(E) * F * I
  c = np.stack([K * G * H, K / H * (G * np.arctan(wy / safeX) - np.arctan(wy / np.where(E == 0, 1.0, E))), np.sign(wz) * rho / I ** 2], axis=-1)
  c = np.where((wx == 0)[..., None], w / (6 / np.pi) ** (1 / 3), c)
  return _permute(c, CUBO_IPERM[id])


def cubo2homo(c):
  """cubo2homo"""
  c = np.asarray(c, dtype=float)
  id = cuboRegionId(c)
  X = _permute(c, CUBO_PERM[id])
  x, y, z = X[..., 0], X[..., 1], X[..., 2]
  C = (6 / np.pi) ** (1 / 3)
  safeX = np.where(x == 0, 1.0, x)
  safeZ = np.where(z == 0, 1.0, z)
  cosyx = np.sqrt(2) * np.cos(np.pi / 12 * y / safeX)
  sinyx = np.sqrt(2) * np.sin(np.pi / 12 * y / safeX)
  A = 2 * cosyx - 3
  B = 2 - cosyx
  D = np.sqrt(np.maximum(A * (x * x - z * z) + z * z, 0))
  E = C * x / np.abs(safeZ) * D / B
  h = np.stack([E * (cosyx - 1), E * sinyx, C * (z + x * x / safeZ * A / B)], axis=-1)
  h = np.where((x == 0)[..., None], C * X, h)
  return _permute(h, CUBO_IPERM[id])


__all__ = ["homochoricRadius", "cubo2homo", "homo2cubo"]
