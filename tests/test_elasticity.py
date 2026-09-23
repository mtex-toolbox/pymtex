# elasticity: Hooke's law, the Christoffel tensor and the wave velocities, the moduli
# against the closed forms of an isotropic material, the Hashin-Shtrikman bounds
# (docs/plans/functions.md P4d: T4, T5 and P4g after the Julia tests)
import numpy as np
import pytest

from mtex import (ChristoffelTensor, HashinShtrikmanModulus, Miller, PoissonRatio, Voigt, YoungsModulus, bulkModulus,
                  complianceTensor, crystalFrame, degree, energyVector, hashinShtrikmanBounds, inv, linearCompressibility,
                  shearModulus, stiffnessTensor, strainTensor, stressTensor, vector3d, velocity, volumeCompressibility, xvector,
                  yvector, zvector)
from mtex.analysis.elasticity import isotropicShearModulus

rng = np.random.default_rng(22)
OLIVINE = np.array([[320.5, 68.15, 71.6, 0, 0, 0], [68.15, 196.5, 76.8, 0, 0, 0], [71.6, 76.8, 233.5, 0, 0, 0], [0, 0, 0, 64, 0, 0],
                    [0, 0, 0, 0, 77, 0], [0, 0, 0, 0, 0, 78.7]])
CUBIC = np.array([[231.0, 134, 134, 0, 0, 0], [134, 231, 134, 0, 0, 0], [134, 134, 231, 0, 0, 0], [0, 0, 0, 116, 0, 0],
                  [0, 0, 0, 0, 116, 0], [0, 0, 0, 0, 0, 116]])


def isotropic(lam, mu, fr=None):
  V = np.array([[lam + 2 * mu, lam, lam, 0, 0, 0], [lam, lam + 2 * mu, lam, 0, 0, 0], [lam, lam, lam + 2 * mu, 0, 0, 0],
                [0, 0, 0, mu, 0, 0], [0, 0, 0, 0, mu, 0], [0, 0, 0, 0, 0, mu]], dtype=float)
  return stiffnessTensor(V, fr)


def test_T4_Hookes_law_and_the_moduli_of_an_isotropic_material():
  lam, mu = 60.0, 40.0
  cs = crystalFrame("m-3m")
  C = isotropic(lam, mu, cs)
  E = mu * (3 * lam + 2 * mu) / (lam + mu)
  nu = lam / (2 * (lam + mu))
  K = lam + 2 * mu / 3
  # Hooke's law with the classes it produces, and the energy
  eps = strainTensor(np.diag([1.0, 1.1, 0.9]), cs)
  sigma = C * eps
  assert isinstance(sigma, stressTensor) and np.allclose(sigma.M, lam * 3 * np.eye(3) + 2 * mu * eps.M)
  S = inv(C)
  assert isinstance(S * sigma, strainTensor) and (S * sigma).isclose(eps, atol=1e-9)
  # every modulus against its closed form, from the stiffness and from the compliance
  for T in (C, S):
    assert np.isclose(YoungsModulus(T, vector3d.rand(rng=rng)), E)
    assert np.isclose(PoissonRatio(T, xvector, yvector), nu) and np.isclose(PoissonRatio(T), nu)
    assert np.isclose(shearModulus(T, xvector, yvector), mu) and np.isclose(shearModulus(T), mu)
    assert np.isclose(bulkModulus(T), K) and np.isclose(linearCompressibility(T, zvector), 1 / (3 * K))
    assert np.isclose(volumeCompressibility(T), 1 / K)
  b = bulkModulus(C, "all")
  assert np.isclose(b.voigt, K) and np.isclose(b.reuss, K) and np.isclose(bulkModulus(C, "Hill"), K)
  g = shearModulus(C, "all")
  assert np.isclose(g.voigt, mu) and np.isclose(g.reuss, mu) and np.isclose(shearModulus(C, "Reuss"), mu)
  with pytest.raises(ValueError):
    bulkModulus(C, "Nobody")
  # the function forms, evaluated at directions
  assert np.isclose(np.real(YoungsModulus(C)(vector3d.rand(rng=rng))), E)
  assert np.isclose(np.real(linearCompressibility(C)(xvector)), 1 / (3 * K))
  assert np.isclose(np.real(PoissonRatio(C, zvector)(xvector)), nu)
  assert np.isclose(np.real(shearModulus(C, zvector)(xvector)), mu)
  # the methods are the same calls
  assert np.isclose(C.YoungsModulus(xvector), E) and np.isclose(C.bulkModulus(), K) and np.isclose(S.shearModulus(), mu)


def test_T5_the_waves():
  lam, mu = 60.0, 40.0
  cs = crystalFrame("m-3m")
  C = isotropic(lam, mu, cs)
  rho = 2.0
  vp, vs1, vs2, pp, ps1, ps2 = velocity(C, zvector, density=rho)
  assert np.isclose(vp, np.sqrt((lam + 2 * mu) / rho)) and np.isclose(vs1, np.sqrt(mu / rho)) and np.isclose(vs2, vs1)
  assert pp.antipodal and np.isclose(abs(pp.data[2]), 1) and np.isclose(abs(ps1.data[2]), 0)
  # the Christoffel tensor of a direction, and the density stored on the tensor
  T = ChristoffelTensor(C, xvector)
  assert isinstance(T, ChristoffelTensor) and np.allclose(T.M, np.diag([lam + 2 * mu, mu, mu]))
  Co = stiffnessTensor(OLIVINE, crystalFrame("mmm", [4.7646, 10.2296, 5.9942], mineral="Olivine"), density=3.355)
  v = velocity(Co, xvector)
  assert np.isclose(v.vp, np.sqrt(320.5 / 3.355)) and v.vs1 >= v.vs2
  assert np.isclose(v.vp, Co.velocity(xvector).vp)
  # a list of directions, and the fields on the sphere
  xyz = vector3d.cat(xvector, yvector, zvector)
  assert np.allclose(velocity(Co, xyz).vp, np.sqrt(np.array([320.5, 196.5, 233.5]) / 3.355))
  vpF, vs1F, vs2F, ppF, ps1F, ps2F = velocity(Co)
  assert np.isclose(np.real(vpF(xvector)), v.vp) and ppF(xvector).antipodal and np.isclose(abs(ppF(xvector).data[0]), 1)
  # the energy vector projects onto the propagation direction as the phase speed
  n = vector3d.byPolar(45 * degree, 20 * degree)
  vpn, _, _, ppn, _, _ = velocity(Co, n)
  Ve = energyVector(Co, n, vpn, ppn)
  assert np.isclose(Ve.dot(n), vpn) and Ve.norm() >= vpn
  VeF = energyVector(Co, None, vpF, ppF)
  assert np.allclose(VeF(n).data, Ve.data)
  with pytest.raises(ValueError):
    velocity(C, xvector, density=-1)
  # MATLAB's density as the third argument, and the harmonic expansions of bandwidth 48
  assert np.isclose(velocity(Co, xvector, 2.0).vp, velocity(Co, xvector, density=2.0).vp)
  h = velocity(Co, 'harmonic')
  assert h.vp.bandwidth == 48 and h.pp.bandwidth == 48
  assert np.isclose(h.vp.eval(xvector), v.vp, atol=1e-6) and np.isclose(h.vs2.eval(n), velocity(Co, n).vs2, atol=1e-4)
  assert abs(np.dot(h.pp.eval(n).data.ravel(), ppn.data.ravel())) > 1 - 1e-6


def test_P4g_the_Hashin_Shtrikman_bounds():
  cs = crystalFrame("m-3m")
  C = stiffnessTensor(CUBIC, cs)
  bounds = hashinShtrikmanBounds(C)
  # the bulk modulus of a cubic crystal is exact, the shear bounds bracket C44 and (C11 - C12)/2
  K = (231 + 2 * 134) / 3
  assert np.isclose(bounds.lower.K, K, atol=1e-6) and np.isclose(bounds.upper.K, K, atol=1e-6)
  assert (231 - 134) / 2 < bounds.lower.G < bounds.upper.G < 116 and abs(bounds.lower.K - bounds.upper.K) < 1e-9
  G = isotropicShearModulus(C)
  assert G.reuss < bounds.lower.G and bounds.upper.G < G.voigt
  # the mesh of references MTEX evaluates gives the same extremes within its resolution
  K0 = np.linspace(1, 300, 120)[:, None]
  G0 = np.linspace(1, 300, 120)[None, :]
  khs, ghs, definite = HashinShtrikmanModulus(C, K0, G0)
  assert khs.shape == (120, 120) and set(np.unique(definite)) <= {-1, 0, 1}
  assert np.isclose(np.nanmax(ghs[definite == 1]), bounds.lower.G, atol=0.5)
  assert np.isclose(np.nanmin(ghs[definite == -1]), bounds.upper.G, atol=0.5)
  assert np.all(np.isnan(khs[definite == 0]))
  k1, g1, d1 = HashinShtrikmanModulus(C, 50.0, 30.0)
  assert d1 == 1 and k1 <= K + 1e-9 and g1 < bounds.lower.G
  # an isotropic tensor has coinciding bounds
  Ci = isotropic(60, 40, cs)
  bi = hashinShtrikmanBounds(Ci)
  assert np.isclose(bi.lower.G, 40, atol=1e-3) and np.isclose(bi.upper.G, 40, atol=1e-3) and np.isclose(bi.lower.K, 60 + 80 / 3, atol=1e-3)
