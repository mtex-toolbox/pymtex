# the discrepancies on the sphere and on SO(3): the analytic cases of MATLAB's
# check_optimalSample (scaling, symmetry, antipodal measures, atomic jumps, spectral
# weights, the legacy kernel objective), their counterparts on SO(3), and numbers of
# MATLAB develop on inputs both build from the same formulas
import math

import numpy as np
import pytest

from mtex import (S2FunHandle, S2FunHarmonic, SO3FunHarmonic, crystalFrame, deg2dim, discrepancy, orientation, quaternion,
                  rotation, rotationalDiscrepancy, sphericalDiscrepancy, vector3d, xvector, zvector)

rng = np.random.default_rng(23)


def uniformS2():
  return S2FunHarmonic([1 / math.sqrt(4 * np.pi)])


def directed(n):
  return vector3d(rng.standard_normal((n, 3))).normalize()


def rotations(n):
  return rotation(quaternion(rng.standard_normal((n, 4))))


# the restricted distance kernel by its Legendre expansion, the negative constant left out
def restrictedKernel(t, L):
  prev, cur, k = np.ones_like(t), t, np.zeros_like(t)
  for l in range(1, L + 1):
    k = k + 4 / ((2 * l - 1) * (2 * l + 3)) * cur
    prev, cur = cur, ((2 * l + 1) * t * cur - l * prev) / (l + 1)
  return k


# ------------------------------------------------------------------ the sphere, D_2 and L2


def test_s2_D2_ofAnAtomAgainstTheUniformDensity():
  u, z = uniformS2(), zvector
  assert abs(discrepancy(u, z) - math.sqrt(1 / 3)) < 1e-12
  assert abs(discrepancy(u, z, squared=True) - 1 / 3) < 1e-12
  assert abs(discrepancy(z, -z) - 1) < 1e-12
  assert abs(discrepancy(z, xvector, squared=True) - math.sqrt(2) / 2) < 1e-12
  assert discrepancy(z, z) == 0


def test_s2_D2_keepsTheAtomsAtEveryBandwidth():
  z, x = zvector, xvector
  assert abs(discrepancy(z, x, bandwidth=0) - discrepancy(z, x, bandwidth=8)) < 1e-12
  assert abs(discrepancy(uniformS2(), z, bandwidth=0) - math.sqrt(1 / 3)) < 1e-12


def test_s2_antipodalPointsAreHalfTheirMassAtEitherSign():
  u, z = uniformS2(), zvector
  a = vector3d(z.data, antipodal=True)
  assert abs(discrepancy(u, a, squared=True) - 1 / 12) < 1e-12
  assert abs(discrepancy(u, a) - discrepancy(u, vector3d(np.array([[0, 0, 1], [0, 0, -1.]])))) < 1e-12


def test_s2_scalesWithTheMass():
  u, z = uniformS2(), zvector
  assert abs(discrepancy(3 * u, z) - 3 * discrepancy(u, z)) < 1e-12


def test_s2_D2_andL2_ofADipole():
  # f - u = alpha z: the cap integral is pi alpha m.z (1 - t^2)
  alpha, u = 0.03, uniformS2()
  f = S2FunHarmonic([1 / math.sqrt(4 * np.pi), 0, alpha * math.sqrt(4 * np.pi / 3), 0])
  assert abs(discrepancy(f, u, squared=True) - 16 * np.pi ** 2 * alpha ** 2 / 45) < 1e-12
  assert abs(discrepancy(f, u, metric="L2") - alpha * math.sqrt(4 * np.pi / 3)) < 1e-12
  assert abs(discrepancy(f, u, metric="L2", degreeWeights=[0, 4]) - 2 * discrepancy(f, u, metric="L2")) < 1e-12
  assert abs(discrepancy(f, u, metric="E_L", degreeWeights=lambda l: 1 / (1 + l * (l + 1)))
             - discrepancy(f, u, metric="L2") / math.sqrt(3)) < 1e-12
  # the mixed D_2 against an independent continuous - discrete calculation
  expected = 1 / 3 + 16 * np.pi ** 2 * alpha ** 2 / 45 - 8 * np.pi * alpha / 15
  assert abs(discrepancy(f, zvector, squared=True) - expected) < 1e-12
  assert abs(discrepancy(zvector, f) - discrepancy(f, zvector)) < 1e-12


def test_s2_D2_isTheChordalEnergyOfWeightedPoints():
  v, w = directed(8), directed(6)
  c, e = np.arange(1, 9.) / 36, np.arange(1, 7.) / 21
  actual = discrepancy(v, w, weights=c, weights2=e, squared=True)
  chord = lambda a, b: np.linalg.norm(a.data[:, None, :] - b.data[None, :, :], axis=-1)
  expected = (2 * c @ chord(v, w) @ e - c @ chord(v, v) @ c - e @ chord(w, w) @ e) / 4
  assert abs(actual - expected) < 1e-12
  r = rotations(1)
  assert abs(actual - discrepancy(r * v, r * w, weights=c, weights2=e, squared=True)) < 1e-12
  assert abs(actual - discrepancy(w, v, weights=e, weights2=c, squared=True)) < 1e-12
  # the truncated kernel converges to the full point energy from below, without its factor four
  k4 = discrepancy(v, w, weights=c, weights2=e, metric="kernel", bandwidth=4) / 4
  k16 = discrepancy(v, w, weights=c, weights2=e, metric="kernel", bandwidth=16) / 4
  assert k4 < k16 < actual


def test_s2_aDifferenceInMassIsMeasured():
  u = uniformS2()
  assert abs(discrepancy(2 * u, u) - discrepancy(u, 0 * u)) < 1e-12
  assert discrepancy(2 * u, u, metric="L2") > 0
  with pytest.raises(ValueError):
    discrepancy(2 * u, u, metric="kernel")


def test_s2_rejectsBadOptions():
  f, v = uniformS2(), directed(5)
  with pytest.raises(ValueError):
    discrepancy(f, v, metric="L2", degreeWeights=[1, -1])
  with pytest.raises(ValueError):
    discrepancy(v, v, metric="D_cap", numHeights=1)
  with pytest.raises(ValueError):
    discrepancy(v, v, metric="D_ball")
  with pytest.raises(ValueError):
    discrepancy(v, v, bandwidth=-1)
  with pytest.raises(ValueError):
    discrepancy(v, v, weights=np.zeros(5))
  with pytest.raises(TypeError):
    discrepancy(v, rotations(3))


# ------------------------------------------------------------------ the sphere, D_cap


def test_s2_cap_ofTheDipoleAndOfAtoms():
  alpha, u, z = 0.03, uniformS2(), zvector
  f = S2FunHarmonic([1 / math.sqrt(4 * np.pi), 0, alpha * math.sqrt(4 * np.pi / 3), 0])
  cap = dict(metric="D_cap", numCenters=17, numHeights=17)
  d, info = discrepancy(f, u, info=True, **cap)
  assert abs(d - np.pi * alpha) < 1e-12 and not info.isExact
  assert abs(abs(info.signedDifference) - d) < 1e-15
  assert abs(discrepancy(u, z, **cap) - 1) < 1e-12
  assert abs(discrepancy(z, -z, **cap) - 1) < 1e-12
  assert discrepancy(z, z, **cap) == 0
  assert abs(discrepancy(u, vector3d(z.data, antipodal=True), **cap) - 0.5) < 1e-12
  assert abs(discrepancy(3 * f, 3 * u, **cap) - 3 * d) < 1e-12
  assert abs(discrepancy(f, u, squared=True, **cap) - d ** 2) < 1e-12
  assert abs(discrepancy(u, f, **cap) - d) < 1e-12
  assert discrepancy(f, u, metric="D_cap", centers=xvector) < 1e-12


def test_s2_cap_tiedAtomsMoveTogether():
  # interleaving + and - weights must not create a discrepancy between equal measures
  z, x = zvector.data, xvector.data
  a, b = vector3d(np.stack([z, z, x])), vector3d(np.stack([x, z]))
  assert discrepancy(a, b, metric="D_cap", numCenters=17, numHeights=17, weights=[1, 1, 2], weights2=[1, 1]) < 1e-12


def test_s2_cap_isSymmetricInItsArguments():
  v, w = directed(8), directed(6)
  c, e = np.arange(1, 9.), np.arange(1, 7.)
  cap = dict(metric="D_cap", numCenters=65, numHeights=33)
  assert abs(discrepancy(v, w, weights=c, weights2=e, **cap) - discrepancy(w, v, weights=e, weights2=c, **cap)) < 1e-12


# ------------------------------------------------------------------ the sphere, the legacy kernel


def test_s2_kernel_isTheSpatialRestrictedDistanceKernel():
  L = 6
  v, u = directed(12), directed(7)
  c, d = np.arange(1, 13.), np.arange(7, 0, -1.)
  c, d = c / c.sum(), d / d.sum()
  opts = dict(bandwidth=L, weights=c, weights2=d, metric="kernel")
  actual = discrepancy(v, u, **opts)
  dot = lambda a, b: a.data @ b.data.T
  expected = c @ restrictedKernel(dot(v, v), L) @ c + d @ restrictedKernel(dot(u, u), L) @ d \
    - 2 * c @ restrictedKernel(dot(v, u), L) @ d
  assert abs(actual - expected) < 1e-10
  assert abs(actual - discrepancy(u, v, weights=d, weights2=c, bandwidth=L, metric="kernel")) < 1e-12
  assert abs(actual - discrepancy(v, u, weights=10 * c, weights2=3 * d, bandwidth=L, metric="kernel")) < 1e-12
  assert discrepancy(v, v, bandwidth=L, metric="kernel") < 1e-20
  assert discrepancy(v, u, bandwidth=0, metric="kernel") == 0
  r = rotations(1)
  assert abs(actual - discrepancy(r * v, r * u, **opts)) < 1e-10
  # the same through the adjoint transforms, as functions
  mu = S2FunHarmonic.adjointNFSFT(v, c, bandwidth=L)
  nu = S2FunHarmonic.adjointNFSFT(u, d, bandwidth=L)
  assert abs(actual - discrepancy(mu, nu, bandwidth=L, metric="kernel")) < 1e-10
  assert abs(actual - discrepancy(mu, u, weights=d, bandwidth=L, metric="kernel")) < 1e-10
  assert abs(actual - discrepancy(u, mu, weights=d, bandwidth=L, metric="kernel")) < 1e-10


def test_s2_kernel_ofTwoFunctions():
  f = S2FunHarmonic([1 / math.sqrt(4 * np.pi), 0, 0.03, 0])
  gh = np.zeros(25, dtype=complex)
  gh[:4] = f.fhat
  gh[20] = 0.02
  g = S2FunHarmonic(gh)
  expected = 16 * np.pi / (7 * 9 * 11) * 0.02 ** 2
  assert abs(discrepancy(f, g, metric="kernel") - expected) < 1e-14
  assert abs(discrepancy(g, f, metric="kernel") - expected) < 1e-14
  assert discrepancy(f, g, metric="kernel", bandwidth=1) < 1e-20
  assert abs(discrepancy(3 * f, 3 * g, metric="kernel") - 9 * expected) < 1e-13
  h = S2FunHandle(lambda x: f.eval(x))
  assert abs(discrepancy(h, g, metric="kernel", bandwidth=4) - expected) < 1e-10
  # an even target does not hide the odd part of the other function
  u = uniformS2()
  assert abs(discrepancy(u, f, metric="kernel") - 16 * np.pi / 15 * 0.03 ** 2) < 1e-14
  # four times the squared D_2 of the bandlimited difference
  assert abs(discrepancy(f, g, metric="kernel") - 4 * discrepancy(f, g, squared=True)) < 1e-14
  with pytest.raises(ValueError):
    discrepancy(f, 2 * g, metric="kernel")


def test_s2_kernel_antipodalPointsAreTheirCopies():
  L = 6
  v, u = directed(12), directed(7)
  c, d = np.arange(1, 13.) / 78, np.arange(7, 0, -1.) / 28
  va, ua = vector3d(v.data, antipodal=True), vector3d(u.data, antipodal=True)
  axial = discrepancy(va, ua, weights=c, weights2=d, bandwidth=L, metric="kernel")
  copies = discrepancy(vector3d(np.concatenate([v.data, -v.data])), vector3d(np.concatenate([u.data, -u.data])),
                       weights=np.concatenate([c, c]) / 2, weights2=np.concatenate([d, d]) / 2, bandwidth=L,
                       metric="kernel")
  assert abs(axial - copies) < 1e-10


# ------------------------------------------------------------------ SO(3)


def test_so3_D2_ofAtoms():
  u = SO3FunHarmonic([1 / (8 * np.pi ** 2)])
  one = rotation(quaternion(np.array([1.0, 0, 0, 0])))
  half = rotation(quaternion(np.array([0.0, 0, 0, 1])))
  assert abs(discrepancy(u, one, squared=True) - 8 / (3 * np.pi ** 2)) < 1e-12
  assert abs(discrepancy(3 * u, one) - 3 * discrepancy(u, one)) < 1e-12
  # sin(w/2) = 1 for the half turn: D_2^2 = 2 / pi
  assert abs(discrepancy(one, half, squared=True) - 2 / np.pi) < 1e-12
  assert discrepancy(one, one) == 0
  assert abs(discrepancy(one, half, metric="D_ball", numCenters=17, numRadii=17) - 1) < 1e-12


def test_so3_anOrientationIsItsOrbit():
  cs = crystalFrame("222")
  o = orientation.rand(3, cs, rng=rng)
  orbit = rotation(o.symmetrise().flatten())
  c = np.arange(1, 4.)
  w = np.tile(c / c.sum() / 4, 4)
  assert discrepancy(o, orbit, weights=c, weights2=w, squared=True) < 1e-12
  assert discrepancy(o, orbit, weights=c, weights2=w, metric="L2", bandwidth=8) < 1e-10


def test_so3_isInvariantAndSymmetric():
  r1, r2 = rotations(8), rotations(6)
  c, e = np.arange(1, 9.), np.arange(1, 7.)
  g = rotations(1)
  for m in ("D_2", "L2"):
    d = discrepancy(r1, r2, weights=c, weights2=e, metric=m, bandwidth=6)
    assert abs(d - discrepancy(g * r1, g * r2, weights=c, weights2=e, metric=m, bandwidth=6)) < 1e-10
    assert abs(d - discrepancy(r2, r1, weights=e, weights2=c, metric=m, bandwidth=6)) < 1e-12
  ball = dict(metric="D_ball", numCenters=65, numRadii=33)
  assert abs(discrepancy(r1, r2, weights=c, weights2=e, **ball) - discrepancy(r2, r1, weights=e, weights2=c, **ball)) < 1e-12


def test_so3_kernel_isTheScaledD2OfTheBandlimitedDifference():
  F, G = formulaF(), formulaG()
  assert abs(discrepancy(F, G, metric="kernel") - 2 * math.sqrt(2) * np.pi * discrepancy(F, G, squared=True)) < 1e-10
  with pytest.raises(ValueError):
    discrepancy(F, 2 * G, metric="kernel")


def test_so3_info():
  F, r = formulaF(), rotations(5)
  d, info = rotationalDiscrepancy(F, r, metric="D_ball", numCenters=17, numRadii=17, info=True)
  assert abs(abs(info.signedDifference) - d) < 1e-15 and not info.isExact
  assert 0 <= info.radius <= np.pi and isinstance(info.center, rotation)
  d, info = sphericalDiscrepancy(uniformS2(), zvector, info=True)
  assert info.isExact and info.bandwidth == 0
  with pytest.raises(ValueError):
    rotationalDiscrepancy(F, r, metric="kernel")


# ------------------------------------------------------------------ MATLAB develop


# the inputs, built in MATLAB develop from the same formulas
def formulaS2():
  n = np.arange(25)
  f = np.cos(n) + 1j * np.sin(2 * n + 1)
  f[0] = 2
  n = np.arange(16)
  g = np.sin(n + 1) + 1j * np.cos(3 * n)
  g[0] = 2
  k = np.arange(1, 9)
  v = vector3d(np.stack([np.sin(k), np.cos(2 * k), np.sin(3 * k) + 0.5], -1)).normalize()
  k = np.arange(1, 7)
  w = vector3d(np.stack([np.cos(k), np.sin(2 * k) + 0.3, np.cos(5 * k)], -1)).normalize()
  centers = vector3d(np.array([[1, 0, 0, 1, 1, -1, 0.3], [0, 1, 0, 1, -1, 0.2, -0.7], [0, 0, 1, 1, 0.5, 0.4, 0.6]]).T).normalize()
  return S2FunHarmonic(f, real=True), S2FunHarmonic(g, real=True), v, w, centers


def formulaF():
  n = np.arange(deg2dim(4))
  F = np.cos(n) / 10 + 1j * np.sin(3 * n + 1) / 10
  F[0] = 1
  return SO3FunHarmonic(F, real=True)


def formulaG():
  n = np.arange(deg2dim(3))
  G = np.sin(2 * n + 1) / 10 + 1j * np.cos(n) / 10
  G[0] = 1
  return SO3FunHarmonic(G, real=True)


def formulaSO3():
  def unit(a):
    a = np.stack(a, -1)
    return rotation(quaternion(a / np.linalg.norm(a, axis=1, keepdims=True)))
  k = np.arange(1, 9)
  r1 = unit([np.cos(k), np.sin(2 * k), np.cos(3 * k), np.sin(5 * k)])
  k = np.arange(1, 7)
  r2 = unit([np.sin(k) + 0.2, np.cos(2 * k), np.sin(3 * k), np.cos(7 * k)])
  cq = np.array([[1, 0, 0, 0], [math.cos(.3), math.sin(.3), 0, 0], [math.cos(.6), 0, math.sin(.6), 0], [.5, .5, .5, .5],
                 [math.cos(1.2), 0, 0, math.sin(1.2)]])
  return r1, r2, rotation(quaternion(cq))


C8, C6 = np.arange(1, 9.), np.arange(1, 7.)

S2_CASES = {
  "D2 f v": (lambda f, g, v, w, m: discrepancy(f, v, weights=C8), 1.819140473614316),
  "D2 v f": (lambda f, g, v, w, m: discrepancy(v, f, weights=C8), 1.8191404736143162),
  "D2 f g": (lambda f, g, v, w, m: discrepancy(f, g), 0.9602985175476666),
  "D2 f antipodal v": (lambda f, g, v, w, m: discrepancy(f, vector3d(v.data, antipodal=True)), 1.2937439563516995),
  "L2 f v": (lambda f, g, v, w, m: discrepancy(f, v, metric="L2", weights=C8), 5.113327724386873),
  "E_L f v heat": (lambda f, g, v, w, m: discrepancy(f, v, metric="E_L", degreeWeights=lambda l: np.exp(-0.1 * l * (l + 1)),
                                                     weights=C8), 3.016906249697651),
  "cap f v": (lambda f, g, v, w, m: discrepancy(f, v, metric="D_cap", centers=m, numHeights=33, weights=C8), 3.7270988153464306),
  "cap f g": (lambda f, g, v, w, m: discrepancy(f, g, metric="D_cap", numCenters=65, numHeights=33), 1.8516515470684594),
  "kernel f v": (lambda f, g, v, w, m: discrepancy(f, v, metric="kernel", weights=C8), 11.656227541544128),
  "kernel f g": (lambda f, g, v, w, m: discrepancy(f, g, metric="kernel"), 3.688692971216983),
  "D2 v w": (lambda f, g, v, w, m: discrepancy(v, w, weights=C8, weights2=C6), 0.20490066302432267),
  "L2 v w": (lambda f, g, v, w, m: discrepancy(v, w, metric="L2", bandwidth=6, weights=C8, weights2=C6), 1.1531680468732217),
  "cap v w": (lambda f, g, v, w, m: discrepancy(v, w, metric="D_cap", numCenters=65, numHeights=33, weights=C8,
                                                weights2=C6), 0.5952380952380952),
  "kernel v w": (lambda f, g, v, w, m: discrepancy(v, w, metric="kernel", bandwidth=6, weights=C8, weights2=C6),
                 0.11412396066678636),
}


# MATLAB's NFSFT has a relative error near 1e-12, its fast Fibonacci grid is the port's
@pytest.mark.parametrize("case", list(S2_CASES))
def test_s2_matchesMATLAB(case):
  run, ref = S2_CASES[case]
  assert abs(run(*formulaS2()) - ref) <= 1e-10 * abs(ref)


SO3_CASES = {
  "D2 F r": (lambda F, G, r1, r2, m: discrepancy(F, r1, weights=C8), 15.502874468739751, 1e-7),
  "D2 F G": (lambda F, G, r1, r2, m: discrepancy(F, G), 3.7883357769661123, 1e-12),
  "L2 F G": (lambda F, G, r1, r2, m: discrepancy(F, G, metric="L2"), 6.555363416890756, 1e-12),
  "L2 F r": (lambda F, G, r1, r2, m: discrepancy(F, r1, metric="L2", weights=C8), 31.83350679791513, 1e-7),
  "ball F r": (lambda F, G, r1, r2, m: discrepancy(F, r1, metric="D_ball", centers=m, numRadii=33, weights=C8),
               35.99007714271397, 1e-12),
  "ball F G": (lambda F, G, r1, r2, m: discrepancy(F, G, metric="D_ball", centers=m, numRadii=33), 7.831987228259666, 1e-12),
  "kernel F G": (lambda F, G, r1, r2, m: discrepancy(F, G, metric="kernel"), 127.52396198082089, 1e-12),
  "kernel F r": (lambda F, G, r1, r2, m: discrepancy(F, r1, metric="kernel", weights=C8), 1693.2216351817162, 1e-5),
  "D2 r r": (lambda F, G, r1, r2, m: discrepancy(r1, r2, weights=C8, weights2=C6), 0.30136109386185694, 1e-12),
  "ball r r": (lambda F, G, r1, r2, m: discrepancy(r1, r2, metric="D_ball", centers=m, numRadii=33, weights=C8, weights2=C6),
               0.5119047619047619, 1e-12),
  "D2 F orbit": (lambda F, G, r1, r2, m: discrepancy(F, orientation(r1, crystalFrame("432")), weights=C8),
                 2.435381366286409, 1e-7),
  "ball F orbit": (lambda F, G, r1, r2, m: discrepancy(F, orientation(r1, crystalFrame("432")), metric="D_ball", centers=m,
                                                       numRadii=33, weights=C8), 7.198221334051475, 1e-12),
}


# the moments of atoms are exact sums here and MATLAB's NFSOFT adjoints in MATLAB, which are good
# to 3e-8 (SO3FunHarmonic.adjoint) and 3e-6 (adjointNFSOFT, the one the kernel metric takes)
@pytest.mark.parametrize("case", list(SO3_CASES))
def test_so3_matchesMATLAB(case):
  run, ref, tol = SO3_CASES[case]
  assert abs(run(formulaF(), formulaG(), *formulaSO3()) - ref) <= tol * abs(ref)
