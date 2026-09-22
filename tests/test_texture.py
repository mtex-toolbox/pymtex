# the texture average of a tensor, the Lankford coefficients, the Kearns factor, the shear
# banding function and the transport of a texture (docs/plans/functions.md P4e: X1, X4, X5,
# X6 after the Julia tests; check_calcLankford pinned against MATLAB)
import numpy as np
import pytest

from mtex import (Miller, SO3FunHandle, SO3FunHarmonic, SO3FunSBF, SO3VectorFieldHandle, Voigt, bulkModulus, calcKearnsFactor,
                  calcLankford, calcTensor, crystalFrame, degree, doEulerStep, mean, orientation, pdf, rotate, rotation,
                  slipSystem, specimenFrame, stiffnessTensor, strainRateTensor, uniformODF, unimodalODF, vector3d, xvector,
                  yvector, zvector)
from mtex.analysis.elasticity import isotropicShearModulus

rng = np.random.default_rng(30)
CUBIC = np.array([[231.0, 134, 134, 0, 0, 0], [134, 231, 134, 0, 0, 0], [134, 134, 231, 0, 0, 0], [0, 0, 0, 116, 0, 0],
                  [0, 0, 0, 0, 116, 0], [0, 0, 0, 0, 0, 116]])


def test_X1_the_texture_average():
  cs, ss = crystalFrame("m-3m"), specimenFrame.default
  C = stiffnessTensor(CUBIC, cs)
  # the uniform ODF gives the isotropic bounds, and those are the Voigt and Reuss averages
  G, K = isotropicShearModulus(C), bulkModulus(C, "all")
  CV, CR, CH = calcTensor(uniformODF(cs, ss), C, "all")
  assert np.isclose(Voigt(CV)[0, 0], K.voigt + 4 * G.voigt / 3, atol=1e-4) and np.isclose(Voigt(CV)[3, 3], G.voigt, atol=1e-4)
  assert np.isclose(Voigt(CR)[0, 0], K.reuss + 4 * G.reuss / 3, atol=1e-4)
  assert np.allclose(Voigt(CH), (Voigt(CV) + Voigt(CR)) / 2) and CV.frame is ss and isinstance(CV, stiffnessTensor)
  assert calcTensor(uniformODF(cs, ss), C).isclose(CV) and calcTensor(uniformODF(cs, ss), C, "Hill").isclose(CH)
  assert C.mean(uniformODF(cs, ss)).isclose(CV) and uniformODF(cs, ss).calcTensor(C, "Reuss").isclose(CR)
  # a single orientation averages to the rotated tensor, a sharp ODF nearly so
  o = orientation.rand(cs, ss, rng=rng)
  assert calcTensor(o, C).isclose(rotate(C, o))
  assert (calcTensor(unimodalODF(o, halfwidth=2 * degree), C) - rotate(C, o)).norm() < 0.01 * C.norm()
  # weights select, and every scheme agrees on one orientation
  os = orientation(rotation.cat(rotation(o.data), rotation.rand(rng=rng)), cs, ss)
  for scheme in ("Voigt", "Reuss", "Hill"):
    assert calcTensor(os, C, scheme, weights=[1, 0]).isclose(rotate(C, o), atol=1e-9)
  assert o.calcTensor(C).isclose(rotate(C, o))
  with pytest.raises(ValueError):
    calcTensor(os, C, "Nobody")
  with pytest.raises(ValueError):
    calcTensor(os, C, weights=[1])
  with pytest.raises(ValueError):
    calcTensor(os, C, weights=[0, 0])
  with pytest.raises(ValueError):
    calcTensor(os, C, weights=[-1, 2])


def test_X4_the_Lankford_coefficients():
  # check_calcLankford, with the work curve pinned against MATLAB
  cs = crystalFrame("6/mmm", [3.2, 3.2, 5.2])
  sS = slipSystem.cat(slipSystem.basal(cs, 1), slipSystem.prismatic2A(cs, 4), slipSystem.pyramidalCA(cs, 7))
  assert len(sS.symmetrise()) == 36
  o = orientation.byEuler(30 * degree, 45 * degree, 0, cs, specimenFrame.default)
  rho = np.linspace(0, 1, 11)
  R, M, minM = calcLankford(o, sS, theta=[0.0], rho=rho)
  assert np.allclose(R, [1.5]) and M.shape == (11, 1) and np.isclose(minM[0], np.min(M))
  assert np.allclose(M.ravel(), [11.4259, 11.118, 10.9041, 10.6902, 10.5489, 10.5247, 10.5208, 10.6085, 10.6963, 10.784, 10.8717], atol=1e-3)
  # the sheet directions are normalised, and they have to be orthogonal
  assert np.allclose(calcLankford(o, sS, theta=[0.0], rho=rho, RD=vector3d(2, 0, 0), ND=vector3d(0, 0, 3))[0], R)
  with pytest.raises(ValueError):
    calcLankford(o, sS, theta=[0.0], rho=rho, RD=xvector, ND=vector3d(1, 0, 1))
  # three angles give the standard averages, weighted or not
  Rv = calcLankford(o, sS, theta=np.array([0, 45 * degree, 90 * degree]), rho=rho)[0]
  assert Rv.shape == (3,) and np.all(np.isfinite(Rv))
  os = orientation(rotation.cat(rotation(o.data), rotation.rand(rng=rng)), cs, specimenFrame.default)
  assert np.allclose(calcLankford(os, sS, theta=[0.0], rho=rho, weights=[1, 0])[1], M)
  # of an ODF: a sharp unimodal one gives the coefficient of its centre
  Ro = calcLankford(unimodalODF(o, halfwidth=5 * degree), sS, theta=[0.0], rho=[0.4, 0.6, 0.8], bandwidth=4)[0]
  assert Ro.shape == (1,) and np.isfinite(Ro[0])


def test_X5_the_Kearns_factor():
  cs, ss = crystalFrame("6/mmm", [3.2, 3.2, 5.2]), specimenFrame.default
  h = Miller(0, 0, 0, 1, cs)
  # the uniform ODF spreads the c axes evenly, so every direction gets a third
  assert np.isclose(calcKearnsFactor(uniformODF(cs, ss), h), 1 / 3)
  # a c axis fibre along z has nearly all of it there, and the three axes share one
  odf = unimodalODF(orientation.id(cs, ss), halfwidth=10 * degree)
  assert calcKearnsFactor(odf, h) > 0.9
  assert np.isclose(sum(calcKearnsFactor(odf, h, v) for v in (xvector, yvector, zvector)), 1)
  assert np.isclose(calcKearnsFactor(pdf(odf, h), zvector), calcKearnsFactor(odf, h)) and np.isclose(odf.calcKearnsFactor(h), calcKearnsFactor(odf, h))


def test_X6_the_shear_banding_function_and_the_Euler_step():
  cs = crystalFrame("222", [4.779, 10.277, 5.995], mineral="olivine")
  ss = specimenFrame.default
  sS = slipSystem(Miller(1, 0, 0, cs, "uvw"), Miller(0, 1, 0, cs))
  E = 0.3 * strainRateTensor(np.array([[1.0, 0, 0], [0, 0, 0], [0, 0, -1]]))
  f = SO3FunSBF(sS, E)
  assert isinstance(f, SO3FunHandle) and f.CS is cs and f.SS is ss
  os = orientation.rand(20, cs, ss, rng=rng)
  assert np.all(np.real(f(os)) > 0) and np.isclose(mean(SO3FunHarmonic(f, bandwidth=16)), 1, atol=1e-6)
  with pytest.raises(ValueError):
    SO3FunSBF(sS.symmetrise(), E)
  # a constant rotation velocity moves the orientations by its angle, and the density with them
  omega = 20 * degree
  spin = SO3VectorFieldHandle(lambda o: vector3d(np.tile([0.0, 0.0, omega], (o.size, 1)).reshape(*o.shape, 3)), cs, ss, "left")
  moved = doEulerStep(spin, os, 4)
  assert moved.shape == os.shape and np.allclose(moved.angle(os, noSymmetry=True), omega, atol=1e-9)
  assert np.allclose(rotation(moved.data).data, (rotation.byAxisAngle(zvector, omega) * rotation(os.data)).data, atol=1e-9)
  odf = SO3FunHarmonic(unimodalODF(orientation.rand(cs, ss, rng=rng), halfwidth=20 * degree), bandwidth=12)
  turned = rotate(odf, rotation.byAxisAngle(zvector, omega))
  stepped = doEulerStep(spin, odf, 20)
  assert stepped.bandwidth == 12 and np.isclose(mean(stepped), 1, atol=1e-6)
  test = orientation.rand(200, cs, ss, rng=rng)
  assert np.linalg.norm(np.real(stepped(test)) - np.real(turned(test))) < 0.15 * np.linalg.norm(np.real(turned(test)))
