# elasticity: Hooke's law, the Christoffel tensor and the wave velocities, the moduli
# against the closed forms of an isotropic material, the Hashin-Shtrikman bounds
# (docs/plans/functions.md P4d: T4, T5 and P4g after the Julia tests)
import numpy as np
import pytest

from mtex import (ChristoffelTensor, HashinShtrikmanModulus, Miller, PoissonRatio, Voigt, YoungsModulus, bulkModulus,
                  complianceTensor, crystalFrame, degree, energyVector, hashinShtrikmanBounds, inv, linearCompressibility, mean,
                  shearModulus, stiffnessTensor, strainTensor, stressTensor, vector3d, velocity, volumeCompressibility, xvector,
                  uniformODF, yvector, zvector)
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


LAM, MU = 60.0, 40.0
E_ISO = MU * (3 * LAM + 2 * MU) / (LAM + MU)
NU_ISO = LAM / (2 * (LAM + MU))
K_ISO = LAM + 2 * MU / 3


def isotropicCubic():
  return isotropic(LAM, MU, crystalFrame("m-3m"))


def olivine():
  return stiffnessTensor(OLIVINE, crystalFrame("mmm", [4.7646, 10.2296, 5.9942], mineral="Olivine"), density=3.355)


def test_HookesLaw_mapsStrainToStressAndBack():
  # Hooke's law with the classes it produces, and the energy
  cs = crystalFrame("m-3m")
  C = isotropic(LAM, MU, cs)
  eps = strainTensor(np.diag([1.0, 1.1, 0.9]), cs)
  sigma = C * eps
  assert isinstance(sigma, stressTensor) and np.allclose(sigma.M, LAM * 3 * np.eye(3) + 2 * MU * eps.M)
  S = inv(C)
  assert isinstance(S * sigma, strainTensor) and (S * sigma).isclose(eps, atol=1e-9)


@pytest.mark.parametrize("compliance", [False, True], ids=["stiffness", "compliance"])
def test_moduli_ofAnIsotropicMaterialAreTheClosedForms(compliance):
  # every modulus against its closed form, from the stiffness and from the compliance
  C = isotropicCubic()
  T = inv(C) if compliance else C
  assert np.isclose(YoungsModulus(T, vector3d.rand(rng=rng)), E_ISO)
  assert np.isclose(PoissonRatio(T, xvector, yvector), NU_ISO) and np.isclose(PoissonRatio(T), NU_ISO)
  assert np.isclose(shearModulus(T, xvector, yvector), MU) and np.isclose(shearModulus(T), MU)
  assert np.isclose(bulkModulus(T), K_ISO) and np.isclose(linearCompressibility(T, zvector), 1 / (3 * K_ISO))
  assert np.isclose(volumeCompressibility(T), 1 / K_ISO)


def test_bulkAndShearModulus_giveTheVoigtReussAndHillAverages():
  C = isotropicCubic()
  b = bulkModulus(C, "all")
  assert np.isclose(b.voigt, K_ISO) and np.isclose(b.reuss, K_ISO) and np.isclose(bulkModulus(C, "Hill"), K_ISO)
  g = shearModulus(C, "all")
  assert np.isclose(g.voigt, MU) and np.isclose(g.reuss, MU) and np.isclose(shearModulus(C, "Reuss"), MU)
  with pytest.raises(ValueError):
    bulkModulus(C, "Nobody")


def test_moduli_asFunctionsOnTheSphere():
  # the function forms, evaluated at directions
  C = isotropicCubic()
  assert np.isclose(np.real(YoungsModulus(C)(vector3d.rand(rng=rng))), E_ISO)
  assert np.isclose(np.real(linearCompressibility(C)(xvector)), 1 / (3 * K_ISO))
  assert np.isclose(np.real(PoissonRatio(C, zvector)(xvector)), NU_ISO)
  assert np.isclose(np.real(shearModulus(C, zvector)(xvector)), MU)


def test_moduli_areMethodsToo():
  # the methods are the same calls
  C = isotropicCubic()
  assert np.isclose(C.YoungsModulus(xvector), E_ISO) and np.isclose(C.bulkModulus(), K_ISO) and np.isclose(inv(C).shearModulus(), MU)


def test_velocity_ofAnIsotropicMaterialIsTheClosedForm():
  C = isotropicCubic()
  rho = 2.0
  vp, vs1, vs2, pp, ps1, ps2 = velocity(C, zvector, density=rho)
  assert np.isclose(vp, np.sqrt((LAM + 2 * MU) / rho)) and np.isclose(vs1, np.sqrt(MU / rho)) and np.isclose(vs2, vs1)
  assert pp.antipodal and np.isclose(abs(pp.data[2]), 1) and np.isclose(abs(ps1.data[2]), 0)
  with pytest.raises(ValueError):
    velocity(C, xvector, density=-1)


def test_ChristoffelTensor_ofADirection():
  # the Christoffel tensor of a direction
  T = ChristoffelTensor(isotropicCubic(), xvector)
  assert isinstance(T, ChristoffelTensor) and np.allclose(T.M, np.diag([LAM + 2 * MU, MU, MU]))


def test_velocity_takesTheDensityStoredOnTheTensor():
  # the density stored on the tensor
  Co = olivine()
  v = velocity(Co, xvector)
  assert np.isclose(v.vp, np.sqrt(320.5 / 3.355)) and v.vs1 >= v.vs2
  assert np.isclose(v.vp, Co.velocity(xvector).vp)


def test_velocity_atAListOfDirectionsAndOnTheSphere():
  # a list of directions, and the fields on the sphere
  Co = olivine()
  v = velocity(Co, xvector)
  xyz = vector3d.cat(xvector, yvector, zvector)
  assert np.allclose(velocity(Co, xyz).vp, np.sqrt(np.array([320.5, 196.5, 233.5]) / 3.355))
  vpF, vs1F, vs2F, ppF, ps1F, ps2F = velocity(Co)
  assert np.isclose(np.real(vpF(xvector)), v.vp) and ppF(xvector).antipodal and np.isclose(abs(ppF(xvector).data[0]), 1)


def test_energyVector_projectsOntoThePropagationDirectionAsThePhaseSpeed():
  # the energy vector projects onto the propagation direction as the phase speed
  Co = olivine()
  n = vector3d.byPolar(45 * degree, 20 * degree)
  vpn, _, _, ppn, _, _ = velocity(Co, n)
  Ve = energyVector(Co, n, vpn, ppn)
  assert np.isclose(Ve.dot(n), vpn) and Ve.norm() >= vpn
  vpF, _, _, ppF, _, _ = velocity(Co)
  VeF = energyVector(Co, None, vpF, ppF)
  assert np.allclose(VeF(n).data, Ve.data)


def test_velocity_takesTheDensityAsThirdArgument():
  # MATLAB's density as the third argument
  Co = olivine()
  assert np.isclose(velocity(Co, xvector, 2.0).vp, velocity(Co, xvector, density=2.0).vp)


def test_velocity_harmonicExpandsToBandwidth48():
  # the harmonic expansions of bandwidth 48
  Co = olivine()
  n = vector3d.byPolar(45 * degree, 20 * degree)
  _, _, _, ppn, _, _ = velocity(Co, n)
  h = velocity(Co, harmonic=True)
  assert h.vp.bandwidth == 48 and h.pp.bandwidth == 48
  assert np.isclose(h.vp.eval(xvector), velocity(Co, xvector).vp, atol=1e-6) and np.isclose(h.vs2.eval(n), velocity(Co, n).vs2, atol=1e-4)
  assert abs(np.dot(h.pp.eval(n).data.ravel(), ppn.data.ravel())) > 1 - 1e-6


def test_eigh3_compiledAgreesWithNumpy():
  from mtex import getMTEXpref, setMTEXpref
  from mtex.tensors.kernels import eigh3
  # random matrices, diagonal ones and the double and triple eigenvalues of isotropic ones
  E = rng.standard_normal((500, 3, 3))
  E = np.concatenate([E + np.swapaxes(E, 1, 2), np.eye(3) * rng.standard_normal((50, 1, 3)),
                      np.broadcast_to(np.diag([1.0, 1, 2]), (3, 3, 3)), np.zeros((2, 3, 3)), np.eye(3)[None]])
  E = E.reshape(2, -1, 3, 3)
  before = getMTEXpref("kernels", "numba")
  try:
    setMTEXpref("kernels", "numba")
    lam, V = eigh3(E)
    setMTEXpref("kernels", "numpy")
    lam0, _ = eigh3(E)
  finally:
    setMTEXpref("kernels", before)
  assert lam.shape == (2, 278, 3) and V.shape == (2, 278, 3, 3)
  assert np.allclose(lam, lam0, atol=1e-13) and np.all(np.diff(lam, axis=-1) >= 0)
  assert np.allclose(np.swapaxes(V, -1, -2) @ V, np.eye(3), atol=1e-13)
  assert np.allclose(E @ V, V * lam[..., None, :], atol=1e-12)


def test_hashinShtrikmanBounds_ofACubicCrystal():
  C = stiffnessTensor(CUBIC, crystalFrame("m-3m"))
  bounds = hashinShtrikmanBounds(C)
  # the bulk modulus of a cubic crystal is exact, the shear bounds bracket C44 and (C11 - C12)/2
  K = (231 + 2 * 134) / 3
  assert np.isclose(bounds.lower.K, K, atol=1e-6) and np.isclose(bounds.upper.K, K, atol=1e-6)
  assert (231 - 134) / 2 < bounds.lower.G < bounds.upper.G < 116 and abs(bounds.lower.K - bounds.upper.K) < 1e-9
  G = isotropicShearModulus(C)
  assert G.reuss < bounds.lower.G and bounds.upper.G < G.voigt


def test_HashinShtrikmanModulus_onAMeshReachesTheBounds():
  # the mesh of references MTEX evaluates gives the same extremes within its resolution
  C = stiffnessTensor(CUBIC, crystalFrame("m-3m"))
  bounds = hashinShtrikmanBounds(C)
  K = (231 + 2 * 134) / 3
  K0 = np.linspace(1, 300, 120)[:, None]
  G0 = np.linspace(1, 300, 120)[None, :]
  khs, ghs, definite = HashinShtrikmanModulus(C, K0, G0)
  assert khs.shape == (120, 120) and set(np.unique(definite)) <= {-1, 0, 1}
  assert np.isclose(np.nanmax(ghs[definite == 1]), bounds.lower.G, atol=0.5)
  assert np.isclose(np.nanmin(ghs[definite == -1]), bounds.upper.G, atol=0.5)
  assert np.all(np.isnan(khs[definite == 0]))
  k1, g1, d1 = HashinShtrikmanModulus(C, 50.0, 30.0)
  assert d1 == 1 and k1 <= K + 1e-9 and g1 < bounds.lower.G


def test_HashinShtrikmanModulus_onAMeshIsTheEstimateOfEachReference():
  # the batched estimate equals the one of every reference on its own
  C = stiffnessTensor(CUBIC, crystalFrame("m-3m"))
  K0, G0 = np.meshgrid(np.linspace(1, 300, 7), np.linspace(1, 300, 5))
  khs, ghs, definite = HashinShtrikmanModulus(C, K0, G0)
  single = [HashinShtrikmanModulus(C, k, g) for k, g in zip(K0.ravel(), G0.ravel())]
  assert np.allclose(np.array(single), np.stack([khs.ravel(), ghs.ravel(), definite.ravel()], axis=1), rtol=1e-12, equal_nan=True)


def test_HashinShtrikmanModulus_ofAnIndefiniteReferenceIsNaN():
  # a reference whose residual is indefinite gives no estimate
  k, g, d = HashinShtrikmanModulus(stiffnessTensor(CUBIC, crystalFrame("m-3m")), 1.0, 300.0)
  assert d == 0 and np.isnan(k) and np.isnan(g)


def test_hashinShtrikmanBounds_ofAnIsotropicTensorCoincide():
  # an isotropic tensor has coinciding bounds
  bi = hashinShtrikmanBounds(isotropic(60, 40, crystalFrame("m-3m")))
  assert np.isclose(bi.lower.G, 40, atol=1e-3) and np.isclose(bi.upper.G, 40, atol=1e-3) and np.isclose(bi.lower.K, 60 + 80 / 3, atol=1e-3)


def test_PoissonRatio_isotropicIsThatOfTheMeanCompliance():
  # the isotropic ratio is the ratio of the compliance averaged over all orientations
  S = inv(olivine())
  Siso = mean(S, uniformODF(S.CS))
  assert np.isclose(PoissonRatio(S), PoissonRatio(Siso, xvector, yvector))
