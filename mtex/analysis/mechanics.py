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
from ..geometry.vector3d import vector3d, asVector3d
from ..tensors.tensor import (EinsteinSum, SchmidTensor, colon, curvatureTensor, deformationGradientTensor,
                     dislocationDensityTensor, spinTensor, strainRateTensor, strainTensor, stressTensor, tensor,
                     velocityGradientTensor)
from ..geometry.install import install


def _frameOf(v):
  return v.frame if isinstance(v, vector3d) and v.frame is not None else specimenFrame.default


def _unit(v):
  v = asVector3d(v)
  return v.normalize().data


# the axial vectors (W_32, W_13, W_21) of the matrices W, over their leading dimensions
def _axial(W):
  return np.stack([W[..., 2, 1], W[..., 0, 2], W[..., 1, 0]], axis=-1)


# ------------------------------------------------------------------ stress and strain

def meanStress(sigma):
  """the mean stress, a third of the trace of the stress tensor"""
  return sigma.trace() / 3


def deviatoric(T):
  """the deviatoric part, the tensor less a third of its trace on the diagonal"""
  return T._rebuild(T.data - np.asarray(T.trace())[..., None] / 3 * np.eye(3).ravel())


deviatoricStress = deviatoric
deviatoricStrain = deviatoric


def equivalentStress(sigma):
  """the von Mises equivalent stress, sqrt(3/2 s:s) of the deviatoric stress s"""
  s = deviatoric(sigma)
  return np.sqrt(1.5 * np.sum(s.data * s.data, axis=-1))


def equivalentStrain(eps):
  """the von Mises equivalent strain, sqrt(2/3 e:e) of the deviatoric strain e"""
  e = deviatoric(eps)
  return np.sqrt(2 / 3 * np.sum(e.data * e.data, axis=-1))


def octahedralStrain(eps):
  """the octahedral shear and normal strain, from the principal stretches"""
  q = (np.sqrt(1 + 2 * eps.eigvals()) - 1)[..., ::-1]
  oG = 2 / 3 * np.sqrt((q[..., 0] - q[..., 1]) ** 2 + (q[..., 1] - q[..., 2]) ** 2 + (q[..., 2] - q[..., 0]) ** 2)
  return oG, q.sum(axis=-1) / 3


def axisRatios(T):
  """the differences of the principal values, largest first"""
  e = T.eigvals()[..., ::-1]
  return e[..., 0] - e[..., 1], e[..., 1] - e[..., 2], e[..., 2] - e[..., 0]


def maxStrain(eps):
  """the magnitude of a strain is its largest principal value, which the Taylor factor is
  measured against; norm stays the Frobenius norm on every class"""
  return np.max(np.abs(eps.eigvals()), axis=-1)


# ------------------------------------------------------------------ the Schmid tensor

# the Schmid tensor b (x) n of a slip plane normal and a slip direction, in their frame, one per
# slip system; SchmidTensor(n, b) is the class constructor
def _schmidTensor(n, b=None):
  if isinstance(n, slipSystem):
    return n._dyads(), n.frame if n.frame is not None else specimenFrame.default
  fr = b.frame if isinstance(b, vector3d) and b.frame is not None else _frameOf(n)
  nn, bb = _unit(n), _unit(b)
  return bb[..., :, None] * nn[..., None, :], fr


def calcShearStress(sigma, sS):
  """the shear stress the stress tensor resolves on each slip system, and the index of the
  system carrying the largest one"""
  tau = np.einsum("...ij,...ij->...", SchmidTensor(sS).M, sigma.M[..., None, :, :] if sigma.ndim else sigma.M)
  return tau, np.argmax(np.abs(tau), axis=-1)


def SchmidFactor(sS, sigma=None, relative=False):
  """the Schmid factor of slip systems under a stress tensor or a uniaxial tension; MATLAB's
  SchmidFactor

    sF = SchmidFactor(sS)                         # a function of the tension direction
    tau = SchmidFactor(sS, v)                     # under a uniaxial tension along v
    tau = SchmidFactor(sS, sigma)                 # under a stress tensor
    tau = SchmidFactor(sS, sigma, relative=True)  # over the critical resolved shear stress

  Under a stress tensor the shear it resolves on the system over the difference of the
  extreme principal stresses, so that a uniaxial tension gives the direction cosines
  cos(phi) cos(lambda) of a tension direction. One stress on a list of systems gives one
  factor per system, a list of stresses or directions on a list of systems a (loads,
  systems) matrix. Without a load the function of the tension direction on the sphere, one
  function for one system and a vector valued one for several, antipodal as a tension along
  r and along -r are one load. The systems and the load must be given in the same frame:
  both in crystal coordinates, or ori * sS and the stress as measured.

  Input
    sS - slipSystem
    sigma - stressTensor, or vector3d or Miller, the tension directions
  Output
    tau - the Schmid factors, or an S2FunHarmonic of the tension direction
  Keywords
    relative - divide by the critical resolved shear stress sS.CRSS of every system
  See also
    SchmidTensor, calcShearStress, stressTensor.uniaxial"""
  if sigma is None:
    from ..functions.s2fun import S2FunHarmonic
    fr = sS.frame.stripSym() if sS.frame is not None else None
    def values(v):
      m = sS.SchmidFactor(vector3d(v.data.reshape(-1, 3))).reshape(-1, len(sS))
      return m[:, 0] if len(sS) == 1 else m
    return S2FunHarmonic(values, fr, bandwidth=2, real=True, antipodal=True)
  if isinstance(sigma, vector3d):
    return sS.SchmidFactor(sigma, relative=relative)
  if not isinstance(sigma, stressTensor):
    raise TypeError("the Schmid factor is of a stressTensor or a tension direction")
  lam = sigma.eigvals()
  spread = lam[..., -1] - lam[..., 0]
  P = SchmidTensor(sS).M
  # one stress on every system, a list of stresses on a list of systems as a (stresses,
  # systems) matrix, or systems and stresses of one shape entry by entry
  if sigma.ndim == 0:
    m = np.einsum("...ij,ij->...", P, sigma.M) / spread
  elif P.ndim == 3:
    m = np.einsum("kij,...ij->...k", P, sigma.M) / spread[..., None]
  else:
    m = np.einsum("...ij,...ij->...", P, sigma.M) / spread
  if relative:
    m = m / sS.CRSS
  # one system under one load is one number
  return float(np.ravel(m)[0]) if np.size(m) == 1 and sigma.ndim == 0 else m


# ------------------------------------------------------------------ deformation and its rate

def _uniaxialDeformation(d, rate=2.0):
  """the deformation gradient of a uniaxial stretch along d, the volume kept; MATLAB's
  deformationGradientTensor.uniaxial

    F = deformationGradientTensor.uniaxial(d)       # stretched to twice its length along d
    F = deformationGradientTensor.uniaxial(d, 3)    # by the factor 3

  The directions perpendicular to d are shortened by one over the square root of rate; a
  rate below one compresses. The tensor is in the frame of d.

  Input
    d - vector3d or Miller, the direction of the stretch
  Output
    F - deformationGradientTensor
  Keywords
    rate - the stretch along d
  See also
    deformationGradientTensor.pureShear, deformationGradientTensor.simpleShear,
    velocityGradientTensor.uniaxial"""
  rot = rotation.map(vector3d(1, 0, 0), vector3d(_unit(d)))
  F = deformationGradientTensor(np.diag([rate, 1 / math.sqrt(rate), 1 / math.sqrt(rate)]), _frameOf(d))
  return F.rotate(rot)


def _simpleShearDeformation(d, n, gamma=45 * np.pi / 180):
  """the deformation gradient of a simple shear along d on the plane n by the angle gamma,
  I + tan(gamma) d n^T; MATLAB's deformationGradientTensor.simpleShear

    F = deformationGradientTensor.simpleShear(d, n)               # sheared by 45 degree
    F = deformationGradientTensor.simpleShear(d, n, 30*degree)    # by 30 degree

  Input
    d - vector3d or Miller, the shear direction
    n - vector3d or Miller, the normal of the shear plane
  Output
    F - deformationGradientTensor
  Keywords
    gamma - the shear angle in radians
  See also
    deformationGradientTensor.pureShear, velocityGradientTensor.simpleShear"""
  M = math.tan(gamma) * np.outer(_unit(d), _unit(n)) + np.eye(3)
  return deformationGradientTensor(M, _frameOf(d))


def _pureShearDeformation(ext, comp, lam=2.0):
  """the deformation gradient of a pure shear, stretched by lam along ext and shortened by
  1 / lam along comp; MATLAB's deformationGradientTensor.pureShear

    F = deformationGradientTensor.pureShear(ext, comp)         # stretched to twice its length
    F = deformationGradientTensor.pureShear(ext, comp, lam=3)  # by the factor 3

  The direction perpendicular to both keeps its length, so does the volume.

  Input
    ext - vector3d or Miller, the direction of extension
    comp - vector3d or Miller, the direction of compression, perpendicular to ext
  Output
    F - deformationGradientTensor
  Keywords
    lam - the stretch ratio, MATLAB's lambda
  See also
    deformationGradientTensor.uniaxial, deformationGradientTensor.simpleShear"""
  rot = rotation.map(vector3d(1, 0, 0), vector3d(_unit(ext)), vector3d(0, 0, 1), vector3d(_unit(comp)))
  F = deformationGradientTensor(np.diag([lam, 1.0, 1 / lam]), _frameOf(ext))
  return F.rotate(rot)


def CauchyStrain(F):
  """the Cauchy (small) strain of a deformation gradient, its symmetric part less the identity"""
  return strainTensor((F.sym().M - np.eye(3)), F.frame)


def GreenStrain(F):
  """the Green-Lagrange strain of a deformation gradient, (F^T F - I) / 2"""
  return strainTensor(((F.transpose() * F).M - np.eye(3)) / 2, F.frame)


def _uniaxialFlow(d=None, q=0.5):
  """the velocity gradient of a uniaxial tension along d, of unit rate along it; MATLAB's
  velocityGradientTensor.uniaxial

    L = velocityGradientTensor.uniaxial()               # axisymmetric tension along z
    L = velocityGradientTensor.uniaxial(d)              # along d
    L = velocityGradientTensor.uniaxial(d, q=0)         # plane strain, no flow along the second axis
    L = -velocityGradientTensor.uniaxial(d)             # uniaxial compression

  The two directions perpendicular to d contract at the rates q and 1 - q, so that the volume
  is kept; q = 0.5 is axisymmetric. A strain rate r multiplies the tensor, r * L.

  Input
    d - vector3d or Miller, the tension direction, z when left out
  Output
    L - velocityGradientTensor
  Keywords
    q - the share of the second transverse direction in the contraction, 0 to 1
  See also
    velocityGradientTensor.simpleShear, deformationGradientTensor.uniaxial"""
  d = vector3d(0, 0, 1) if d is None else d
  rot = rotation.map(vector3d(1, 0, 0), vector3d(_unit(d)))
  L = velocityGradientTensor(np.diag([1.0, -q, q - 1]), _frameOf(d))
  return L.rotate(rot)


def _simpleShearFlow(d, n, e=1.0):
  """the velocity gradient of a simple shear along d on the plane n, 2 e d n^T; MATLAB's
  velocityGradientTensor.simpleShear

    L = velocityGradientTensor.simpleShear(d, n)          # of the strain rate one
    L = velocityGradientTensor.simpleShear(d, n, e=1e-3)  # of the strain rate 1e-3

  Input
    d - vector3d or Miller, the shear direction
    n - vector3d or Miller, the normal of the shear plane
  Output
    L - velocityGradientTensor
  Keywords
    e - the strain rate
  See also
    velocityGradientTensor.uniaxial, deformationGradientTensor.simpleShear"""
  return velocityGradientTensor(2 * e * np.outer(_unit(d), _unit(n)), _frameOf(d))


def strainRate(L):
  """the symmetric part of a velocity gradient is the strain rate, the antisymmetric one the
  spin, whose axial vector is the angular velocity"""
  return strainRateTensor(L.sym().M, L.frame)


def spin(L):
  """the spin tensor of a velocity gradient, its antisymmetric part"""
  return spinTensor(L.antiSym().M, L.frame)


def angularVelocityVector(L):
  """the angular velocity of a velocity gradient, the axial vector of its antisymmetric part"""
  W = L.antiSym().M
  return vector3d(_axial(W), frame=L.frame)


def strainRateMagnitude(L):
  """the magnitude of the strain rate of a velocity gradient, half its largest absolute principal
  value, or half its largest singular value when the tensor is not symmetric"""
  return 0.5 * np.max(np.abs(L.sym().eigvals()), axis=-1) if L.isSymmetric else 0.5 * np.max(np.linalg.svd(L.M, compute_uv=False), axis=-1)


def rotationRate(L):
  """the rotation rate of a velocity gradient, the length of its angular velocity vector"""
  return angularVelocityVector(L).norm()


def vorticity(L):
  """the vorticity of a velocity gradient, its rotation rate over the magnitude of its strain rate"""
  return rotationRate(L) / strainRateMagnitude(L)


def spinVector(W):
  """the axial vector of a spin tensor"""
  return angularVelocityVector(W)


def spinTensorOf(v, frame=None):
  """the spin tensor of an axial vector v, the antisymmetric tensor W with W u = v x u

    W = spinTensorOf(v)                  # in the frame of v
    W = spinTensorOf(a, frame=cs)        # of the coordinates a in the frame cs

  Input
    v - vector3d, or an array of (..., 3) coordinates
  Output
    W - spinTensor
  Keywords
    frame - the frame of the tensor, for coordinates without one
  See also
    spinTensor, spinTensor.vector, spin"""
  return spinTensor(v, frame=frame) if frame is not None else spinTensor(v)


# ------------------------------------------------------------------ curvature and the dislocation density, Nye's relation

def dislocationDensity(kappa):
  """a diagonal entry of the density is minus the sum of the other two diagonal entries of the
  curvature, so the entry a map cannot measure drops out of alpha_33"""
  K = kappa.M
  M = np.swapaxes(K, -1, -2).copy()
  for i in range(3):
    M[..., i, i] = -sum(K[..., j, j] for j in range(3) if j != i)
  return dislocationDensityTensor(M, kappa.frame, unit=kappa.unit)


def curvature(alpha):
  """the curvature tensor of a dislocation density tensor by Nye's relation, kappa = alpha^T - tr(alpha) I / 2"""
  A = alpha.M
  return curvatureTensor(np.swapaxes(A, -1, -2) - 0.5 * np.trace(A, axis1=-2, axis2=-1)[..., None, None] * np.eye(3), alpha.frame,
                         unit=alpha.unit)


# ------------------------------------------------------------------ installation

def _install():
  for f in (meanStress, equivalentStress, calcShearStress):
    install(stressTensor, f)
  install(stressTensor, deviatoric, "deviatoricStress")
  install(strainTensor, deviatoric, "deviatoricStrain")
  for f in (equivalentStrain, octahedralStrain, axisRatios, maxStrain):
    install(strainTensor, f)
  install(deformationGradientTensor, staticmethod(_uniaxialDeformation), "uniaxial")
  install(deformationGradientTensor, staticmethod(_simpleShearDeformation), "simpleShear")
  install(deformationGradientTensor, staticmethod(_pureShearDeformation), "pureShear")
  install(deformationGradientTensor, CauchyStrain)
  install(deformationGradientTensor, GreenStrain)
  install(velocityGradientTensor, staticmethod(_uniaxialFlow), "uniaxial")
  install(velocityGradientTensor, staticmethod(_simpleShearFlow), "simpleShear")
  for f in (strainRate, spin, angularVelocityVector, rotationRate, vorticity):
    install(velocityGradientTensor, f)
  install(spinTensor, spinVector, "vector")
  install(curvatureTensor, dislocationDensity)
  install(dislocationDensityTensor, curvature)
  # the Schmid factor of the geometry takes directions; a stress tensor and the function
  # form come from here
  direct = slipSystem.SchmidFactor

  def schmidFactor(self, sigma=None, relative=False):
    """the Schmid factor of the slip systems; MATLAB's SchmidFactor

      tau = sS.SchmidFactor(v)                      # under a uniaxial tension along v
      tau = sS.SchmidFactor(sigma)                  # under a stress tensor
      tau = sS.SchmidFactor(sigma, relative=True)   # over the critical resolved shear stress
      sF = sS.SchmidFactor()                        # a function of the tension direction

    For a tension direction cos(phi) cos(lambda), the cosines of its angles to the slip
    direction and to the plane normal; under a stress tensor the shear it resolves on the
    system over the difference of the extreme principal stresses; without a load the
    function of the tension direction on the sphere. A list of loads on a list of systems
    gives a (loads, systems) matrix. The systems and the load must be given in the same
    frame: both in crystal coordinates, or ori * sS and the stress as measured.

    Input
      sS - slipSystem
      sigma - stressTensor, or vector3d or Miller, the tension directions
    Output
      tau - the Schmid factors, or an S2FunHarmonic of the tension direction
    Keywords
      relative - divide by the critical resolved shear stress sS.CRSS of every system
    See also
      SchmidFactor, SchmidTensor, calcShearStress"""
    if sigma is None or isinstance(sigma, stressTensor):
      return SchmidFactor(self, sigma, relative)
    return direct(self, sigma, relative=relative)

  install(slipSystem, schmidFactor, "SchmidFactor")


_install()

__all__ = ["meanStress", "deviatoric", "deviatoricStress", "deviatoricStrain", "equivalentStress", "equivalentStrain",
           "octahedralStrain", "axisRatios", "maxStrain", "calcShearStress", "SchmidFactor", "CauchyStrain",
           "GreenStrain", "strainRate", "spin", "angularVelocityVector", "rotationRate", "vorticity", "spinTensorOf",
           "dislocationDensity"]
