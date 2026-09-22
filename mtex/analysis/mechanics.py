# stress and strain, deformation and its rate, the Schmid tensor, curvature and the
# dislocation density (MATLAB TensorAnalysis/@stressTensor, @strainTensor,
# @deformationGradientTensor, @velocityGradientTensor, @spinTensor, @curvatureTensor,
# @dislocationDensityTensor, SchmidTensor.m; docs/plans/functions.md P4d)
#
#   meanStress(sigma), deviatoricStress(sigma), equivalentStress(sigma), equivalentStrain(eps)
#   SchmidTensor(n, b), SchmidTensor(sS), calcShearStress(sigma, sS), SchmidFactor(sS, sigma)
#   deformationGradientTensor.uniaxial(d), .simpleShear(d, n, gamma), .pureShear(e, c, lam)
#   velocityGradientTensor.uniaxial(d, q), .simpleShear(d, n, e); strainRateTensor(L), spinTensor(L)
#   dislocationDensity(kappa), curvature(alpha)                                    Nye's relation
from __future__ import annotations

import math

import numpy as np

from ..geometry.frames import specimenFrame
from ..geometry.quaternion import rotation
from ..geometry.slipsystem import slipSystem
from ..geometry.vector3d import vector3d
from ..tensors.tensor import (EinsteinSum, SchmidTensor, colon, curvatureTensor, deformationGradientTensor,
                     dislocationDensityTensor, spinTensor, strainRateTensor, strainTensor, stressTensor, tensor,
                     velocityGradientTensor)


def _frameOf(v):
  return v.frame if isinstance(v, vector3d) and v.frame is not None else specimenFrame.default


def _unit(v):
  v = v if isinstance(v, vector3d) else vector3d(v)
  return v.normalize().data


# ------------------------------------------------------------------ stress and strain

def meanStress(sigma):
  return sigma.trace() / 3


def deviatoric(T):
  """the deviatoric part, the tensor less a third of its trace on the diagonal"""
  return T._rebuild(T.data - np.asarray(T.trace())[..., None] / 3 * np.eye(3).ravel())


deviatoricStress = deviatoric
deviatoricStrain = deviatoric


def equivalentStress(sigma):
  s = deviatoric(sigma)
  return np.sqrt(1.5 * np.sum(s.data * s.data, axis=-1))


def equivalentStrain(eps):
  e = deviatoric(eps)
  return np.sqrt(2 / 3 * np.sum(e.data * e.data, axis=-1))


# the octahedral shear and normal strain, from the principal stretches
def octahedralStrain(eps):
  q = (np.sqrt(1 + 2 * eps.eigvals()) - 1)[..., ::-1]
  oG = 2 / 3 * np.sqrt((q[..., 0] - q[..., 1]) ** 2 + (q[..., 1] - q[..., 2]) ** 2 + (q[..., 2] - q[..., 0]) ** 2)
  return oG, q.sum(axis=-1) / 3


# the differences of the principal values, largest first
def axisRatios(T):
  e = T.eigvals()[..., ::-1]
  return e[..., 0] - e[..., 1], e[..., 1] - e[..., 2], e[..., 2] - e[..., 0]


# the magnitude of a strain is its largest principal value, which the Taylor factor is
# measured against; norm stays the Frobenius norm on every class
def maxStrain(eps):
  return np.max(np.abs(eps.eigvals()), axis=-1)


# ------------------------------------------------------------------ the Schmid tensor

# the Schmid tensor b (x) n of a slip plane normal and a slip direction, in their frame; of a
# slip system its symmetric part, one per system; SchmidTensor(n, b) is the class constructor
def _schmidTensor(n, b=None):
  if isinstance(n, slipSystem):
    return n.SchmidTensor(), n.frame if n.frame is not None else specimenFrame.default
  fr = _frameOf(n) if b is None or not isinstance(b, vector3d) or b.frame is None else b.frame
  nn, bb = _unit(n), _unit(b)
  return bb[..., :, None] * nn[..., None, :], fr


def calcShearStress(sigma, sS):
  """the shear stress the stress tensor resolves on each slip system, and the index of the
  system carrying the largest one"""
  tau = np.einsum("...ij,...ij->...", SchmidTensor(sS).M, sigma.M[..., None, :, :] if sigma.ndim else sigma.M)
  return tau, np.argmax(np.abs(tau), axis=-1)


def SchmidFactor(sS, sigma=None, relative=False):
  """the Schmid factor of slip systems under a stress tensor: the shear the stress resolves
  on the system over the difference of the extreme principal stresses, so that a uniaxial
  tension gives the direction cosines; without a load the functions of the tension
  direction on the sphere, one per system"""
  if sigma is None:
    from ..functions.s2fun import S2FunHarmonic
    fr = sS.frame.stripSym() if sS.frame is not None else None
    return [S2FunHarmonic(lambda v, k=k: sS[k].SchmidFactor(vector3d(v.data)), fr, bandwidth=2, real=True) for k in range(len(sS))]
  if isinstance(sigma, vector3d):
    return sS.SchmidFactor(sigma, relative=relative)
  if not isinstance(sigma, stressTensor):
    raise TypeError("the Schmid factor is of a stressTensor or a tension direction")
  lam = sigma.eigvals()
  spread = lam[..., -1] - lam[..., 0]
  P = SchmidTensor(sS).M
  if sigma.ndim == 0:
    m = np.einsum("kij,ij->k", P, sigma.M) / spread
  else:
    m = np.einsum("kij,...ij->...k", P, sigma.M) / spread[..., None]
  if relative:
    m = m / sS.CRSS
  return m


# ------------------------------------------------------------------ deformation and its rate

def _rotationFromTo(a, b, c=None, d=None):
  if c is None:
    return rotation.map(vector3d(a), vector3d(b))
  return rotation.map(vector3d(a), vector3d(b), vector3d(c), vector3d(d))


def _uniaxialDeformation(d, rate=2.0):
  """the deformation gradient of a uniaxial stretch by rate along d, the volume kept"""
  rot = rotation.map(vector3d(1, 0, 0), vector3d(_unit(d)))
  F = deformationGradientTensor(np.diag([rate, 1 / math.sqrt(rate), 1 / math.sqrt(rate)]), _frameOf(d))
  return F.rotate(rot)


def _simpleShearDeformation(d, n, gamma=45 * np.pi / 180):
  """the deformation gradient of a simple shear along d on the plane n by the angle gamma"""
  M = math.tan(gamma) * np.outer(_unit(d), _unit(n)) + np.eye(3)
  return deformationGradientTensor(M, _frameOf(d))


def _pureShearDeformation(ext, comp, lam=2.0):
  """the deformation gradient of a pure shear, stretched by lam along ext and shortened along comp"""
  rot = rotation.map(vector3d(1, 0, 0), vector3d(_unit(ext)), vector3d(0, 0, 1), vector3d(_unit(comp)))
  F = deformationGradientTensor(np.diag([lam, 1.0, 1 / lam]), _frameOf(ext))
  return F.rotate(rot)


def CauchyStrain(F):
  return strainTensor((F.sym().M - np.eye(3)), F.frame)


def GreenStrain(F):
  return strainTensor(((F.transpose() * F).M - np.eye(3)) / 2, F.frame)


def _uniaxialFlow(d=None, q=0.5):
  """the velocity gradient of a uniaxial flow along d with the ratio q of the transverse rates"""
  d = vector3d(0, 0, 1) if d is None else d
  rot = rotation.map(vector3d(1, 0, 0), vector3d(_unit(d)))
  L = velocityGradientTensor(np.diag([1.0, -q, q - 1]), _frameOf(d))
  return L.rotate(rot)


def _simpleShearFlow(d, n, e=1.0):
  """the velocity gradient of a simple shear along d on the plane n with the rate e"""
  return velocityGradientTensor(2 * e * np.outer(_unit(d), _unit(n)), _frameOf(d))


# the symmetric part of a velocity gradient is the strain rate, the antisymmetric one the
# spin, whose axial vector is the angular velocity
def strainRate(L):
  return strainRateTensor(L.sym().M, L.frame)


def spin(L):
  return spinTensor(L.antiSym().M, L.frame)


def angularVelocityVector(L):
  W = L.antiSym().M
  return vector3d(np.stack([W[..., 2, 1], W[..., 0, 2], W[..., 1, 0]], axis=-1), frame=L.frame)


def strainRateMagnitude(L):
  return 0.5 * np.max(np.abs(L.sym().eigvals()), axis=-1) if L.isSymmetric else 0.5 * np.max(np.linalg.svd(L.M, compute_uv=False), axis=-1)


def rotationRate(L):
  return angularVelocityVector(L).norm()


def vorticity(L):
  return rotationRate(L) / strainRateMagnitude(L)


# the axial vector of a spin tensor
def spinVector(W):
  return angularVelocityVector(W)


def spinTensorOf(v, frame=None):
  return spinTensor(v, frame=frame) if frame is not None else spinTensor(v)


# ------------------------------------------------------------------ curvature and the dislocation density, Nye's relation

# a diagonal entry of the density is minus the sum of the other two diagonal entries of the
# curvature, so the entry a map cannot measure drops out of alpha_33
def dislocationDensity(kappa):
  K = kappa.M
  M = np.swapaxes(K, -1, -2).copy()
  for i in range(3):
    M[..., i, i] = -sum(K[..., j, j] for j in range(3) if j != i)
  return dislocationDensityTensor(M, kappa.frame)


def curvature(alpha):
  A = alpha.M
  return curvatureTensor(np.swapaxes(A, -1, -2) - 0.5 * np.trace(A, axis1=-2, axis2=-1)[..., None, None] * np.eye(3), alpha.frame)


# ------------------------------------------------------------------ installation

def _install():
  stressTensor.meanStress = lambda self: meanStress(self)
  stressTensor.deviatoricStress = lambda self: deviatoric(self)
  stressTensor.equivalentStress = lambda self: equivalentStress(self)
  stressTensor.calcShearStress = lambda self, sS: calcShearStress(self, sS)
  strainTensor.deviatoricStrain = lambda self: deviatoric(self)
  strainTensor.equivalentStrain = lambda self: equivalentStrain(self)
  strainTensor.octahedralStrain = lambda self: octahedralStrain(self)
  strainTensor.axisRatios = lambda self: axisRatios(self)
  strainTensor.maxStrain = lambda self: maxStrain(self)
  deformationGradientTensor.uniaxial = staticmethod(_uniaxialDeformation)
  deformationGradientTensor.simpleShear = staticmethod(_simpleShearDeformation)
  deformationGradientTensor.pureShear = staticmethod(_pureShearDeformation)
  deformationGradientTensor.CauchyStrain = lambda self: CauchyStrain(self)
  deformationGradientTensor.GreenStrain = lambda self: GreenStrain(self)
  velocityGradientTensor.uniaxial = staticmethod(_uniaxialFlow)
  velocityGradientTensor.simpleShear = staticmethod(_simpleShearFlow)
  velocityGradientTensor.strainRate = lambda self: strainRate(self)
  velocityGradientTensor.spin = lambda self: spin(self)
  velocityGradientTensor.angularVelocityVector = lambda self: angularVelocityVector(self)
  velocityGradientTensor.rotationRate = lambda self: rotationRate(self)
  velocityGradientTensor.vorticity = lambda self: vorticity(self)
  spinTensor.vector = lambda self: spinVector(self)
  curvatureTensor.dislocationDensity = lambda self: dislocationDensity(self)
  dislocationDensityTensor.curvature = lambda self: curvature(self)
  # the Schmid factor of the geometry takes directions; a stress tensor and the function
  # form come from here
  direct = slipSystem.SchmidFactor

  def schmidFactor(self, sigma=None, relative=False):
    if sigma is None or isinstance(sigma, stressTensor):
      return SchmidFactor(self, sigma, relative)
    return direct(self, sigma, relative=relative)

  slipSystem.SchmidFactor = schmidFactor


_install()

__all__ = ["meanStress", "deviatoric", "deviatoricStress", "deviatoricStrain", "equivalentStress", "equivalentStrain",
           "octahedralStrain", "axisRatios", "maxStrain", "calcShearStress", "SchmidFactor", "CauchyStrain",
           "GreenStrain", "strainRate", "spin", "angularVelocityVector", "rotationRate", "vorticity", "spinTensorOf",
           "dislocationDensity"]
