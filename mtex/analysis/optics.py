# optics: the refractive index, birefringence, the optical axes and the colour of a
# birefringent plate (MATLAB TensorAnalysis/@refractiveIndexTensor; docs/plans/functions.md P4d)
#
#   refractiveIndex(rI, v)
#   deltaN, pFast, pSlow = birefringence(rI, vprop)      the index difference and the polarisation axes
#   deltaN, pFast, pSlow = birefringence(rI)             as a function on the sphere and two axis fields
#   opticalAxis(rI)                                       the one or two directions of equal indices
#   spectralTransmission(rI, vprop, thickness, tau=, phi=, polarizationDirection=)
#   spectralTransmission(rI, thickness)                   the three colour channels on the sphere
from __future__ import annotations

import math

import numpy as np

from ..geometry.vector3d import vector3d, asVector3d
from ..tensors.tensor import _axesIn, _inFrameOf, refractiveIndexTensor
from ..geometry.install import install

# CIE XYZ to Adobe RGB
ADOBE_RGB = np.array([[2.04414, -0.5649, -0.3447], [-0.9693, 1.8760, 0.0416], [0.0134, -0.1184, 1.0154]])


# the unit directions in the frame of the tensor, as coordinates
def _unit(rI, v):
  return _inFrameOf(rI, v).normalize().data


def refractiveIndex(rI, v):
  """the refractive index in the direction v, n_ij v_i v_j"""
  return rI.directionalMagnitude(vector3d(_unit(rI, v)))


# the two indices and polarisations of light travelling along directions: the tensor
# restricted to the plane normal to each, over an (n, 3) array; a list of tensors pairs
# with the directions, one of either side broadcast
def _birefringence(rI, U):
  M = rI.M.reshape(-1, 3, 3)
  n = max(len(U), len(M))
  U = np.broadcast_to(U, (n, 3))
  # real tensors at many directions in one compiled pass
  if n >= 64 and M.dtype.kind == "f":
    from .kernels import birefringenceKernel
    out = birefringenceKernel(U, M)
    if out is not None:
      return out
  p1 = vector3d(U).orth().normalize().data
  p2 = np.cross(U, p1)
  p2 = p2 / np.linalg.norm(p2, axis=-1, keepdims=True)
  # the tensor on the two axes, M p1 and M p2 as two products of n rows: one tensor by one
  # matrix product, a list of them row by row
  Mp1, Mp2 = (p1 @ M[0].T, p2 @ M[0].T) if len(M) == 1 else (np.einsum("nij,nj->ni", M, p1), np.einsum("nij,nj->ni", M, p2))
  # the eigen decomposition of the symmetric 2 x 2 in closed form: the difference of the
  # indices without cancellation, the axes by the angle of the larger one
  a, c = np.einsum("ni,ni->n", p1, Mp1), np.einsum("ni,ni->n", p2, Mp2)
  b = (np.einsum("ni,ni->n", p1, Mp2) + np.einsum("ni,ni->n", p2, Mp1)) / 2
  delta = 2 * np.hypot((a - c) / 2, b)
  mid = (a + c) / 2
  theta = np.arctan2(2 * b, a - c)[:, None] / 2
  co, si = np.cos(theta), np.sin(theta)
  slow = co * p1 + si * p2
  fast = -si * p1 + co * p2
  return delta, fast, slow, mid - delta / 2, mid + delta / 2


def birefringence(rI, vprop=None):
  """the birefringence for light along vprop: the index difference and the fast and the
  slow polarisation axis; at a list of directions arrays and axis lists; without a
  direction the function on the sphere and two axis fields"""
  if vprop is None:
    from ..functions.handles import S2FunHandle
    from ..functions.s2axisfield import S2AxisFieldHandle
    def component(k):
      return lambda v: birefringence(rI, vector3d(v.data))[k]
    def axis(k):
      return lambda v: component(k)(v).data
    delta = S2FunHandle(component(0), rI.frame, antipodal=True, bandwidth=32, symmetrise=False)
    return delta, S2AxisFieldHandle(axis(1), rI.frame, bandwidth=32), S2AxisFieldHandle(axis(2), rI.frame, bandwidth=32)
  v = asVector3d(vprop)
  delta, fast, slow, _, _ = _birefringence(rI, _unit(rI, v).reshape(-1, 3))
  shape = np.broadcast_shapes(v.shape, rI.shape)
  return delta.reshape(shape), _axesIn(fast.reshape(shape + (3,)), rI.frame), _axesIn(slow.reshape(shape + (3,)), rI.frame)


def opticalAxis(rI, tol=1e-8):
  """the optical axes of a refractive index tensor: the directions along which the two
  indices coincide, one for a uniaxial and two for a biaxial crystal"""
  n, V = rI.eig()
  e = V.data
  if abs(n[0] - n[2]) < tol:
    return _axesIn(np.zeros((0, 3)), rI.frame)
  if abs(n[0] - n[1]) < tol:
    return _axesIn(e[2], rI.frame)
  if abs(n[1] - n[2]) < tol:
    return _axesIn(e[0], rI.frame)
  a = math.sqrt((n[2] - n[1]) / (n[2] - n[0]))
  b = math.sqrt((n[1] - n[0]) / (n[2] - n[0]))
  return _axesIn(np.stack([-a * e[2] + b * e[0], a * e[2] + b * e[0]]), rI.frame)


def spectralTransmission(rI, vprop, thickness=None, *, polarizationDirection=None, tau=45 * np.pi / 180, phi=np.pi / 2, bandwidth=32):
  """the colour of white light passing a birefringent plate between a polariser and an analyser,
  as a thin section shows it under the microscope

    rgb = rI.spectralTransmission(vprop, thickness)                    # polariser at 45 degree to the fast wave
    rgb = rI.spectralTransmission(vprop, thickness, polarizationDirection=p)  # a polariser direction
    rgb = rI.spectralTransmission(vprop, thickness, tau=30*degree, phi=90*degree)  # the angles given
    r, g, b = rI.spectralTransmission(thickness, bandwidth=64)         # functions of the propagation direction

  The light propagates along vprop through a plate of the given thickness; the two waves the
  refractive index tensor allows along it leave with a retardation, the birefringence times
  the thickness, and the analyser lets through what their interference leaves of every
  wavelength. The spectrum is weighted with the CIE 1931 colour matching functions and given
  as Adobe RGB. A list of directions gives one colour per direction. With the thickness alone
  the three channels are S2FunHarmonic of the propagation direction, which plot draws as
  the interference colours of every direction.

  Input
    rI - refractiveIndexTensor
    vprop - vector3d or Miller, the propagation direction of the light, in the frame of rI
    thickness - the thickness of the plate in nanometres (30 micrometre = 30000)
  Output
    rgb - (..., 3) array, the colour of every direction
    r, g, b - S2FunHarmonic, the channels as functions of the propagation direction
  Keywords
    polarizationDirection - vector3d, the direction of the polariser; the angle between it and
        the fast wave replaces tau
    tau - the angle between the polariser and the vibration direction of the fast wave
    phi - the angle between the polariser and the analyser; 90 degree for crossed polarisers
    bandwidth - the bandwidth of the three functions when no direction is given
  See also
    refractiveIndexTensor.birefringence, refractiveIndexTensor.refractiveIndex,
    refractiveIndexTensor.opticalAxis, refractiveIndexTensor, BirefringenceDemo"""
  from .data.cie1931 import CIE1931, WAVELENGTHS
  if thickness is None:
    from ..functions.quadrature import quadratureS2Grid
    from ..functions.s2fun import S2FunHarmonic
    grid = quadratureS2Grid(bandwidth)
    rgb = spectralTransmission(rI, vector3d(grid.nodes().data), vprop, polarizationDirection=polarizationDirection, tau=tau, phi=phi)
    return tuple(S2FunHarmonic.quadrature(grid, rgb[:, k], rI.frame, bandwidth=bandwidth, real=True, antipodal=True) for k in range(3))
  v = asVector3d(vprop)
  U = _unit(rI, v).reshape(-1, 3)
  delta, fast, _, _, _ = _birefringence(rI, U)
  if polarizationDirection is None:
    t = np.full(len(fast), float(tau))
  else:
    p = asVector3d(polarizationDirection)
    P = np.broadcast_to(p.normalize().data.reshape(-1, 3), fast.shape)
    t = np.arccos(np.minimum(1.0, np.abs(np.sum(P * fast, axis=-1))))
  d = delta * thickness
  # the transmitted spectrum of every direction, weighed with the colour matching functions
  s = np.cos(phi) ** 2 - np.sin(2 * (t - phi))[:, None] * np.sin(2 * t)[:, None] * np.sin(np.pi * d[:, None] / WAVELENGTHS) ** 2
  rgb = (s @ CIE1931) @ ADOBE_RGB.T / 100
  return rgb.reshape(np.broadcast_shapes(v.shape, rI.shape) + (3,))


def _install():
  for f in (refractiveIndex, birefringence, opticalAxis, spectralTransmission):
    install(refractiveIndexTensor, f)


_install()

__all__ = ["refractiveIndex", "birefringence", "opticalAxis", "spectralTransmission"]
