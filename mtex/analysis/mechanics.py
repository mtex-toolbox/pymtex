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
  """compute the mean normal stress, trace(sigma) / 3

    pressureComponent = meanStress(sigma)

  Return one scalar per stress tensor, in the same units as sigma. This uses the sign of
  the tensor's normal components; a positive tensile stress gives a positive mean stress.
  The complementary deviatoric stress has zero trace.

  See also
    deviatoricStress, equivalentStress"""
  return sigma.trace() / 3


def deviatoric(T):
  """remove the isotropic part of a rank-two stress or strain tensor

    s = deviatoric(sigma)
    e = deviatoric(eps)

  Subtract trace(T) / 3 times the identity from each tensor. The result has zero trace
  and retains the input class, frame and array shape. The names deviatoricStress and
  deviatoricStrain perform the same operation.

  See also
    meanStress, equivalentStress, equivalentStrain"""
  return T._rebuild(T.data - np.asarray(T.trace())[..., None] / 3 * np.eye(3).ravel())


deviatoricStress = deviatoric
deviatoricStrain = deviatoric


def equivalentStress(sigma):
  """compute the von Mises equivalent stress

    value = equivalentStress(sigma)

  For deviatoric stress `s = sigma - trace(sigma)*I/3`, return `sqrt(3/2 * sum(s*s))`,
  summing all Cartesian coefficients. The result is nonnegative, with one value per tensor
  in the stress units. Hydrostatic stress gives zero; uniaxial stress gives its magnitude.

  See also
    deviatoricStress, meanStress, equivalentStrain"""
  s = deviatoric(sigma)
  return np.sqrt(1.5 * np.sum(s.data * s.data, axis=-1))


def equivalentStrain(eps):
  """compute the von Mises equivalent strain

    value = equivalentStrain(eps)

  For deviatoric strain `e = eps - trace(eps)*I/3`, return `sqrt(2/3 * sum(e*e))`,
  summing all Cartesian coefficients. Return one nonnegative scalar per strain tensor.
  The tensor frame and the units used to express strain do not change this calculation.

  See also
    deviatoricStrain, equivalentStress"""
  e = deviatoric(eps)
  return np.sqrt(2 / 3 * np.sum(e.data * e.data, axis=-1))


def octahedralStrain(eps):
  """compute octahedral shear and normal strain from Green–Lagrange strain

    shear, normal = octahedralStrain(eps)

  Convert principal Green–Lagrange strains E_i to engineering stretches `q_i = sqrt(1+2*E_i)-1`.
  The normal strain is mean(q_i); the shear strain is 2/3 times the square root of the sum
  of squared pairwise differences. Values below E_i = -1/2 have no real stretch and give NaN.
  Return one shear value and one normal value per tensor.

  See also
    GreenStrain, equivalentStrain"""
  q = (np.sqrt(1 + 2 * eps.eigvals()) - 1)[..., ::-1]
  oG = 2 / 3 * np.sqrt((q[..., 0] - q[..., 1]) ** 2 + (q[..., 1] - q[..., 2]) ** 2 + (q[..., 2] - q[..., 0]) ** 2)
  return oG, q.sum(axis=-1) / 3


def axisRatios(T):
  """return signed differences between the principal values of a rank-two tensor

    d12, d23, d31 = axisRatios(T)

  With principal values e1 >= e2 >= e3, the outputs are e1-e2, e2-e3 and e3-e1.
  Each output has the tensor-array shape and the same units as the coefficients. Despite
  the name, these are differences rather than quotients.

  See also
    tensor.eig, maxStrain"""
  e = T.eigvals()[..., ::-1]
  return e[..., 0] - e[..., 1], e[..., 1] - e[..., 2], e[..., 2] - e[..., 0]


def maxStrain(eps):
  """return the largest absolute principal strain of each tensor

    magnitude = maxStrain(eps)

  The largest magnitude among the three eigenvalues measures the strongest extension or
  contraction. This is the strain scale used by the Taylor-factor calculation. The result
  has one value per tensor; tensor.norm instead uses all Cartesian coefficients.

  See also
    equivalentStrain, calcTaylor, tensor.eigvals"""
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
  """resolve a stress tensor on slip systems and identify the most highly stressed system

    tau, index = calcShearStress(sigma, sS)

  Compute b dot sigma dot n using unit slip directions b and plane normals n. The resolved
  shear stress is signed and has the same units as sigma. A single stress applied to a list
  of systems gives one value per system; an array of stresses adds a final system axis.
  Express stress and systems in the same frame, for example with `ori * sS` in specimen coordinates.

  Input
    sigma - stressTensor
    sS - slipSystem list
  Output
    tau - signed resolved shear stresses
    index - zero-based index of the system with largest absolute shear stress, along the
        last axis of tau
  See also
    SchmidFactor, slipSystem.rotate, SchmidTensor"""
  tau = np.einsum("...ij,...ij->...", SchmidTensor(sS).M, sigma.M[..., None, :, :] if sigma.ndim else sigma.M)
  return tau, np.argmax(np.abs(tau), axis=-1)


def SchmidFactor(sS, sigma=None, relative=False):
  """the Schmid factor of slip systems under a stress tensor or a uniaxial tension; MATLAB's
  SchmidFactor

    sF = SchmidFactor(sS)                         # a function of the tension direction
    tau = SchmidFactor(sS, v)                     # under a uniaxial tension along v
    tau = SchmidFactor(sS, sigma)                 # under a stress tensor
    tau = SchmidFactor(sS, sigma, relative=True)  # over the critical resolved shear stress

  Under a tension direction r the Schmid factor is cos(phi) cos(lambda), the product of the
  cosines between r and the slip direction and between r and the plane normal. Under a stress
  tensor it is the resolved shear stress divided by the difference of the largest and the
  smallest principal stress, so a uniaxial tension gives the same number and the factor
  lies between -0.5 and 0.5.

  One load on a list of systems gives one factor per system, a list of loads a (loads,
  systems) matrix. Without a load the result is the function of the tension direction on
  the sphere, antipodal, vector valued for several systems; relative=True divides it by the
  CRSS as well.

  The systems and the load must be given in the same frame: both in crystal coordinates, or
  ori * sS and the stress as measured.

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
      m = sS.SchmidFactor(vector3d(v.data.reshape(-1, 3)), relative=relative).reshape(-1, len(sS))
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
  """compute the small-strain approximation from a deformation gradient

    eps = CauchyStrain(F)

  Return (F + F.transpose())/2 - I as a strainTensor in the frame of F. This linearised
  strain is appropriate for small displacement gradients; finite rigid rotations can produce
  nonzero values. Use GreenStrain for a finite-strain measure invariant under rigid rotation.

  See also
    GreenStrain, deformationGradientTensor"""
  return strainTensor((F.sym().M - np.eye(3)), F.frame)


def GreenStrain(F):
  """compute the Green–Lagrange finite strain of a deformation gradient

    eps = GreenStrain(F)

  Return `(F.transpose() * F - I)/2` as a strainTensor in the frame of F. This measures strain
  relative to the undeformed configuration and gives zero for a rigid rotation. Leading
  array dimensions are retained.

  See also
    CauchyStrain, deformationGradientTensor"""
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
  """extract the symmetric strain-rate tensor from a velocity gradient

    D = strainRate(L)

  Return (L + L.transpose())/2 as a strainRateTensor in the same frame and array shape.
  Its units are those of L, normally inverse time. Together with spin(L), it reconstructs L.

  See also
    spin, angularVelocityVector, velocityGradientTensor"""
  return strainRateTensor(L.sym().M, L.frame)


def spin(L):
  """extract the antisymmetric spin tensor from a velocity gradient

    W = spin(L)

  Return (L - L.transpose())/2 as a spinTensor in the same frame and array shape.
  Its axial vector is the angular velocity, and W times a vector is its cross product
  with that angular velocity. Together with strainRate(L), it reconstructs L.

  See also
    strainRate, angularVelocityVector, spinTensor"""
  return spinTensor(L.antiSym().M, L.frame)


def angularVelocityVector(L):
  """return the angular velocity represented by the antisymmetric part of a velocity gradient

    omega = angularVelocityVector(L)

  For W = (L - L.transpose())/2, the vector is (W[2,1], W[0,2], W[1,0]).
  The output is vector3d in the tensor frame, one vector per tensor, in radians per unit time
  when L has inverse-time units. Its length is rotationRate(L).

  See also
    spin, rotationRate, spinTensorOf"""
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
  """convert lattice curvature to the Nye dislocation density tensor

    alpha = dislocationDensity(kappa)

  Use `alpha = kappa.transpose() - trace(kappa)*I`. In particular, each diagonal component
  of alpha is minus the sum of the other two diagonal components of kappa. This can retain
  a measurable alpha[2,2] even when a surface map cannot determine kappa[2,2].

  Return dislocationDensityTensor in the input frame, preserving its unit and array shape.
  The coefficients describe net dislocation content; fitting individual dislocation-system
  densities is a separate step performed by fitDislocationSystems or calcGND.

  See also
    curvature, curvatureTensor, calcGND"""
  K = kappa.M
  M = np.swapaxes(K, -1, -2).copy()
  for i in range(3):
    M[..., i, i] = -sum(K[..., j, j] for j in range(3) if j != i)
  return dislocationDensityTensor(M, kappa.frame, unit=kappa.unit)


def curvature(alpha):
  """recover lattice curvature from a Nye dislocation density tensor

    kappa = curvature(alpha)

  Use `kappa = alpha.transpose() - trace(alpha)*I/2`, the inverse of dislocationDensity.
  The result is curvatureTensor in the same frame, with the input unit and array shape.
  Missing coefficients propagate through the relation.

  See also
    dislocationDensity, curvatureTensor, calcGND"""
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
