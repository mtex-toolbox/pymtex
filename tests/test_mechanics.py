# stress and strain, the Schmid tensor and factor, the deformation and flow factories,
# Nye's relation (docs/plans/functions.md P4d: T6, T7 and X7 after the Julia tests)
import numpy as np
import pytest

from mtex import (Miller, SchmidTensor, calcShearStress, crystalFrame, deformationGradientTensor, degree, dislocationDensityTensor,
                  equivalentStrain, equivalentStress, inv, meanStress, orientation, rotation, slipSystem, specimenFrame, spinTensor,
                  strainRateTensor, strainTensor, stressTensor, curvatureTensor, vector3d, velocityGradientTensor, xvector, yvector,
                  zvector)
from mtex.analysis.mechanics import (CauchyStrain, GreenStrain, SchmidFactor, angularVelocityVector, axisRatios, curvature,
                                      deviatoric, dislocationDensity, maxStrain, octahedralStrain, spin, spinTensorOf, strainRate,
                                      vorticity)

rng = np.random.default_rng(23)


def test_stressMeasures_ofADiagonalStress():
  sigma = stressTensor(np.diag([3.0, 1, 2]))
  assert np.isclose(meanStress(sigma), 2) and np.isclose(deviatoric(sigma).trace(), 0)
  assert np.isclose(equivalentStress(sigma), np.sqrt(3))
  assert np.isclose(equivalentStress(stressTensor.uniaxial(zvector, 5)), 5)
  assert np.allclose(sigma.deviatoricStress().M, deviatoric(sigma).M)


def test_strainMeasures_ofADiagonalStrain():
  eps = strainTensor(np.diag([1.0, -0.4, -0.6]))
  assert np.isclose(maxStrain(eps), 1) and np.isclose(equivalentStrain(eps), np.sqrt(2 / 3 * (1 + 0.16 + 0.36)))
  assert np.allclose(axisRatios(eps), [1.4, 0.2, -1.6])
  oG, oN = octahedralStrain(strainTensor(np.zeros((3, 3))))
  assert np.isclose(oG, 0) and np.isclose(oN, 0)


def test_stressTensor_uniaxialAlongAMillerIsInItsFrame():
  # a uniaxial stress along a direction, in its frame, and its negative
  cs = crystalFrame("m-3m")
  s = stressTensor.uniaxial(Miller(0, 0, 1, cs))
  assert s.frame is cs and np.allclose(s.M, np.diag([0, 0, 1.0]))
  assert np.allclose((-s).M, -s.M)


def fcc():
  cs = crystalFrame("m-3m")
  return cs, slipSystem.fcc(cs), Miller(0, 0, 1, cs)


def test_SchmidTensor_ofAPlaneAndADirectionAndOfASystem():
  # the Schmid tensor of a plane and a direction, and of a system
  cs, sS, _ = fcc()
  P = SchmidTensor(Miller(1, 1, 1, cs), Miller(1, -1, 0, cs, "uvw"))
  assert isinstance(P, SchmidTensor) and P.frame is cs and np.allclose(P.M, np.outer([1, -1, 0], [1, 1, 1]) / np.sqrt(6))
  Ps = SchmidTensor(sS)
  assert Ps.shape == (1,) and np.allclose(Ps.M[0], sS.SchmidTensor().M)


def test_SchmidFactor_ofAStressIsTheDirectionCosines():
  # the stress route reproduces the direction cosines, free of the scale and with the sign
  _, sS, r = fcc()
  m = sS.SchmidFactor(r)
  sigma = stressTensor.uniaxial(r)
  assert np.isclose(sS.SchmidFactor(sigma), m) and np.isclose(sS.SchmidFactor(3 * sigma), m) and np.isclose(sS.SchmidFactor(-sigma), -m)
  assert np.isclose(abs(m), 1 / np.sqrt(6))
  assert np.allclose(SchmidFactor(sS, sigma), sS.SchmidFactor(sigma))


def test_SchmidFactor_ofTheCubeOrientationInTensionIsAtMostOneOverSqrt6():
  # all systems: the largest factor of the cube orientation in tension is 1/sqrt(6)
  _, sS, r = fcc()
  sigma = stressTensor.uniaxial(r)
  sSall = sS.symmetrise(antipodal=True)
  SF = sSall.SchmidFactor(sigma)
  assert SF.shape == (len(sSall),) and np.isclose(np.max(np.abs(SF)), 1 / np.sqrt(6))
  tau, active = calcShearStress(sigma, sSall)
  assert np.allclose(tau, SF) and np.isclose(abs(tau[active]), np.max(np.abs(tau)))


def test_SchmidFactor_asAFunctionIsAPolynomialOfDegreeTwo():
  # the function of the tension direction is exact, a polynomial of degree two
  _, sS, r = fcc()
  m = sS.SchmidFactor(r)
  F = sS.SchmidFactor()
  assert F.shape == () and F.bandwidth == 2 and np.isclose(np.real(F(r)), m, atol=1e-10)
  assert F.frame.numSym() == 1
  # several systems give one vector valued function
  sSall = sS.symmetrise(antipodal=True)
  Fall = sSall.SchmidFactor()
  assert Fall.shape == (len(sSall),) and np.allclose(np.real(Fall(r)).ravel(), sSall.SchmidFactor(r), atol=1e-10)


def test_SchmidFactor_ofTheTurnedSystemIsThatOfTheTurnedStress():
  # the system turned into the specimen and the stress turned into the crystal agree
  cs, sS, _ = fcc()
  o = orientation.byEuler(20 * degree, 35 * degree, 10 * degree, cs)
  sz = stressTensor.uniaxial(zvector)
  assert np.isclose((o * sS).SchmidFactor(sz), sS.SchmidFactor(inv(o) * sz))


def test_SchmidFactor_ofAListOfStressesIsAMatrix():
  # a list of stresses gives a matrix of factors, one row per orientation
  cs, sS, _ = fcc()
  sSall = sS.symmetrise(antipodal=True)
  sz = stressTensor.uniaxial(zvector)
  os = orientation.rand(6, cs, rng=rng)
  local = inv(os) * sz
  assert local.shape == (6,) and sSall.SchmidFactor(local).shape == (6, len(sSall))
  assert np.allclose(sSall.SchmidFactor(local)[2], sSall.SchmidFactor(inv(os[2]) * sz))


def test_deformationGradientTensor_ofTheStandardDeformations():
  # the deformation gradients of the standard deformations, in the frame of their directions
  F = deformationGradientTensor.uniaxial(zvector, 2)
  assert isinstance(F, deformationGradientTensor) and np.allclose(F.M, np.diag([1 / np.sqrt(2), 1 / np.sqrt(2), 2]))
  assert np.isclose(F.det(), 1) and F.frame is specimenFrame.default
  Fs = deformationGradientTensor.simpleShear(xvector, zvector, 30 * degree)
  assert np.allclose(Fs.M, np.eye(3) + np.tan(30 * degree) * np.outer([1, 0, 0], [0, 0, 1]))
  Fp = deformationGradientTensor.pureShear(xvector, zvector, 2)
  assert np.allclose(Fp.M, np.diag([2, 1, 0.5]))


def test_GreenAndCauchyStrain_ofADeformationGradient():
  F = deformationGradientTensor.uniaxial(zvector, 2)
  Fs = deformationGradientTensor.simpleShear(xvector, zvector, 30 * degree)
  assert isinstance(GreenStrain(F), strainTensor) and np.allclose(GreenStrain(F).M, (F.M.T @ F.M - np.eye(3)) / 2)
  assert np.allclose(CauchyStrain(Fs).M, (Fs.M + Fs.M.T) / 2 - np.eye(3)) and np.allclose(F.GreenStrain().M, GreenStrain(F).M)


def test_velocityGradient_splitsIntoStrainRateAndSpin():
  # the flows, their strain rate, spin and vorticity
  L = velocityGradientTensor.simpleShear(xvector, zvector, 1)
  assert np.allclose(L.M, 2 * np.outer([1, 0, 0], [0, 0, 1]))
  assert isinstance(strainRate(L), strainRateTensor) and np.allclose(strainRate(L).M, (L.M + L.M.T) / 2)
  assert isinstance(spin(L), spinTensor) and np.allclose(spin(L).M, (L.M - L.M.T) / 2)
  assert np.allclose(angularVelocityVector(L).data, [0, 1, 0]) and np.isclose(vorticity(L), 1)
  Lu = velocityGradientTensor.uniaxial(zvector)
  assert np.allclose(Lu.M, np.diag([-0.5, -0.5, 1])) and np.isclose(L.rotationRate(), 1) and np.isclose(Lu.rotationRate(), 0)


def test_spinTensorOf_aVectorIsSkewAndGivesItBack():
  W = spinTensorOf(vector3d(1, 2, 3))
  assert W.isSkewSymmetric and np.allclose(W.vector().data, [1, 2, 3])


def test_dislocationDensity_isNyesRelationBothWays():
  # Nye's relation between the curvature and the dislocation density, both ways
  kappa = curvatureTensor(rng.standard_normal((3, 3)))
  alpha = kappa.dislocationDensity()
  assert isinstance(alpha, dislocationDensityTensor)
  off = ([0, 1, 2, 0, 1], [1, 0, 1, 2, 2])
  assert np.allclose(alpha.M[off], kappa.M.T[off])
  assert np.allclose(np.diag(alpha.M), -np.array([kappa.M[1, 1] + kappa.M[2, 2], kappa.M[0, 0] + kappa.M[2, 2], kappa.M[0, 0] + kappa.M[1, 1]]))
  back = alpha.curvature()
  assert isinstance(back, curvatureTensor) and np.allclose(back.M, kappa.M)
  assert np.allclose(curvature(dislocationDensity(kappa)).M, kappa.M)


def test_dislocationDensity_ofAListOfCurvatures():
  # a list of curvature tensors, as a map delivers them
  ks = curvatureTensor(rng.standard_normal((7, 3, 3)))
  assert ks.dislocationDensity().shape == (7,) and np.allclose(ks.dislocationDensity()[3].M, curvatureTensor(ks.M[3]).dislocationDensity().M)
