# the halfwidth selection from the degree energies (docs/plans/halfwidth.md, H0 to H2)
import numpy as np
import pytest

from mtex import (Miller, S2DeLaValleePoussinKernel, SO3DeLaValleePoussinKernel, calcKernel, crystalFrame, degree,
                  discreteSample, orientation, rotation, specimenFrame, unimodalODF, vector3d)
from mtex.functions.halfwidth import (_characters, _legendre, candidateHalfwidths, degreeEnergies, familyMultipliers,
                                      kernelMultipliers, lowerBoundHalfwidth, noiseFloorHalfwidth, riskCurve,
                                      selectHalfwidth, ucvHalfwidth)

rng = np.random.default_rng(5)


def _bumps(n, kappa=200.0):
  # directions about z from the de la Vallee Poussin law, cos = 2 u^(1/(kappa+1)) - 1, and a uniform half
  t = 2 * rng.random(n) ** (1 / (kappa + 1)) - 1
  phi = rng.uniform(0, 2 * np.pi, n)
  s = np.sqrt(1 - t * t)
  x = np.stack([s * np.cos(phi), s * np.sin(phi), t], axis=1)
  u = rng.standard_normal((n, 3))
  u /= np.linalg.norm(u, axis=1, keepdims=True)
  half = rng.random(n) < 0.5
  x[half] = u[half]
  return x


def test_the_energies_are_the_pairwise_sums_on_the_sphere():
  x = _bumps(200)
  w = rng.random(200)
  for weights in (None, w):
    E = degreeEnergies(vector3d(x), 10, weights=weights, rng=rng)
    ww = np.ones(200) / 200 if weights is None else w / w.sum()
    P = _legendre(x @ x.T, 10)
    direct = np.array([(2 * l + 1) * ww @ P[l] @ ww for l in range(11)])
    assert np.allclose(E.energy, direct, atol=1e-9)
    assert np.isclose(E.s2, np.sum(ww ** 2)) and E.nFolds == 8


def test_the_cross_fitted_energies_of_a_uniform_sample_vanish_within_their_error():
  # the cross-fitted energies are unbiased: for a uniform sample they vanish within their error
  u = rng.standard_normal((20000, 3))
  E = degreeEnergies(vector3d(u), 8, rng=rng)
  assert np.all(np.abs(E.A[1:]) < 5 * E.se[1:] + 1e-12)


def test_symmetric_directions_are_projected_onto_the_group():
  cs = crystalFrame("m-3m")
  h = Miller(vector3d(rng.standard_normal((150, 3))), cs)
  E = degreeEnergies(h, 8, rng=rng)
  x = E._sample
  M = cs.sym.rot.matrix().reshape(-1, 3, 3)
  # the projected energy: n^-2 sum_ij (2l+1) mean_s P_l(x_i . s x_j)
  P = _legendre(np.clip(np.einsum("ni,gij,mj->gnm", x, M, x), -1, 1), 8)
  direct = np.array([(2 * l + 1) * P[l].mean(axis=0).sum() / 150 ** 2 for l in range(9)])
  assert np.allclose(E.energy, direct, atol=1e-9)
  # the cubic group leaves nothing in the degrees 1, 2, 3 and 5
  assert np.all(np.abs(E.energy[[1, 2, 3, 5]]) < 1e-12)
  # the traces are the diagonal of the projected kernel
  c = E._traces(8)
  assert np.allclose(c, np.stack([(2 * l + 1) * P[l].mean(axis=0).diagonal() for l in range(9)]))


def test_axial_directions_leave_only_the_even_degrees():
  a = degreeEnergies(vector3d(_bumps(300), antipodal=True), 6, rng=rng)
  assert np.all(a.energy[1::2] == 0) and np.allclose(a.dims(6)[0], (2 * np.arange(7) + 1) * (np.arange(7) % 2 == 0))


@pytest.mark.parametrize("cs, ss, antipodal", [("-1", "1", False), ("m-3m", "1", False), ("6/mmm", "222", False),
                                               ("m-3m", None, True)])
def test_the_energies_are_the_character_sums_on_SO3(cs, ss, antipodal):
  CS = crystalFrame(cs)
  SS = CS if ss is None else specimenFrame(ss)
  o = orientation(rotation.rand(120, rng=rng), CS, SS, antipodal=antipodal)
  E = degreeEnergies(o, 5, rng=rng)
  x = o.data.reshape(-1, 4)
  U = _characters(np.stack([np.abs(x @ y.T) for y in E._symmetricCopies(x)]), 5)
  direct = np.array([(2 * l + 1) * U[l].mean(axis=0).sum() / 120 ** 2 for l in range(6)])
  assert np.allclose(E.energy, direct, atol=1e-9)
  E._sample = x
  assert np.allclose(E._traces(5), np.stack([(2 * l + 1) * U[l].mean(axis=0).diagonal() for l in range(6)]))


def test_the_multipliers_of_the_family():
  for space, fam in (("S2", S2DeLaValleePoussinKernel), ("SO3", SO3DeLaValleePoussinKernel)):
    b = familyMultipliers(space, 10 * degree)
    assert b[0] == 1 and np.all(np.diff(b) <= 0) and b[-1] >= 1e-3
    assert np.allclose(b, kernelMultipliers(fam(halfwidth=10 * degree), len(b) - 1))
    # every multiplier falls as the halfwidth grows, all the conservative argument needs
    b2 = familyMultipliers(space, 12 * degree)
    assert np.all(b2 <= b[:len(b2)] + 1e-15)


def test_lower_energies_select_a_wider_kernel():
  E = degreeEnergies(vector3d(_bumps(3000)), 32, rng=rng)
  H = candidateHalfwidths("S2", 2 * degree, 40 * degree, 30)
  A = np.maximum(E.A, 0)
  wide = H[np.argmin(riskCurve(E, H, 0.5 * A))]
  assert wide >= H[np.argmin(riskCurve(E, H, A))]


# the exact risk of the model: half uniform, half the kernel of kappa 200, for 4000 directions
def _modelRisk(h, L=256):
  from mtex.functions.kernelfunctions import _dlvpRule
  a = _dlvpRule(200.0, L) / (2 * np.arange(L + 1) + 1)
  A = 0.25 * a * a * (2 * np.arange(L + 1) + 1)
  b = familyMultipliers("S2", h)
  k = min(len(b), L + 1)
  l = np.arange(k)
  return np.sum(((1 - b[:k]) ** 2 * A[:k] + b[:k] ** 2 * (2 * l + 1 - A[:k]) / 4000)[1:]) + np.sum(A[k:])


# a sample of 4000 directions from the model, its energies to degree 32 and the candidate halfwidths
@pytest.fixture(scope="module")
def sphereSample():
  r0 = np.random.default_rng(6)
  x = _bumps(4000)
  H = candidateHalfwidths("S2", 1 * degree, 60 * degree, 60)
  return x, H, degreeEnergies(vector3d(x), 32, rng=r0)


def test_ucv_and_the_lower_bound_come_near_the_risk_of_the_model(sphereSample):
  x, H, E0 = sphereSample
  r = np.array([_modelRisk(h) for h in H])
  hO = H[np.argmin(r)]
  E1 = degreeEnergies(vector3d(x), 256, rng=rng)
  hU, h2 = ucvHalfwidth(E1, H), lowerBoundHalfwidth(E0, H)
  assert _modelRisk(hU) < 1.3 * r.min() and h2 >= hO and _modelRisk(h2) < 1.5 * r.min()


def test_the_noise_floor_falls_as_the_sample_grows(sphereSample):
  x, H, E0 = sphereSample
  f1 = noiseFloorHalfwidth(E0, H)
  f2 = noiseFloorHalfwidth(degreeEnergies(vector3d(_bumps(40000)), 32, rng=rng), H)
  assert f2 <= f1


def test_the_conservative_choice_is_the_larger_of_the_lower_bound_and_the_noise_floor(sphereSample):
  x, H, E0 = sphereSample
  assert selectHalfwidth(E0, "conservative", H) == max(lowerBoundHalfwidth(E0, H), noiseFloorHalfwidth(E0, H))
  with pytest.raises(ValueError):
    selectHalfwidth(E0, "nothing")


def test_calcKernel_of_directions():
  v = vector3d(_bumps(2000))
  psi = calcKernel(v, rng=rng)
  assert isinstance(psi, S2DeLaValleePoussinKernel) and 1 * degree <= psi.halfwidth() <= 60 * degree
  assert isinstance(calcKernel(v, "UCV", rng=rng), S2DeLaValleePoussinKernel)
  with pytest.raises(ValueError):
    calcKernel(v, "klcv")


def test_calcKernel_of_orientations():
  cs = crystalFrame("m-3m")
  ori = discreteSample(unimodalODF(orientation(rotation.rand(rng=rng), cs), halfwidth=10 * degree), 3000, rng=rng)
  for m in ("conservative", "UCV"):
    psi = calcKernel(ori, m, rng=rng)
    assert isinstance(psi, SO3DeLaValleePoussinKernel) and 2 * degree <= psi.halfwidth() <= 40 * degree


# the pixels of a map: forty grains of fifty pixels each, spread by half a degree, and their grain ids
def _pixels():
  cs = crystalFrame("m-3m")
  g = orientation(rotation.rand(40, rng=rng), cs)
  gid = np.repeat(np.arange(40), 50)
  pix = orientation(rotation(g.data[gid]) * rotation.byAxisAngle(vector3d.rand(2000, rng=rng), 0.5 * degree), cs)
  pix.notIID = True
  return pix, gid


def test_pixels_of_a_map_need_their_grains():
  pix, gid = _pixels()
  for m in ("conservative", "UCV", "klcv", "magicRule"):
    with pytest.raises(ValueError, match="independent"):
      calcKernel(pix, m, rng=rng)


def test_pixels_folded_by_grains_have_the_variance_of_the_grains():
  pix, gid = _pixels()
  # folded by grains, the variance is that of forty grains, not of two thousand pixels
  E = degreeEnergies(pix, 16, groups=gid, rng=rng)
  assert E.grouped and E.nUnits == 40 and np.isclose(E.neff, 40)


def test_pixels_folded_by_grains_select_a_wider_kernel_than_independent_ones():
  pix, gid = _pixels()
  iid = pix.copy()
  iid.notIID = False
  H = candidateHalfwidths("SO3")
  assert calcKernel(pix, groups=gid, method="conservative", rng=rng).halfwidth() > lowerBoundHalfwidth(degreeEnergies(iid, 16, rng=rng), H)


def test_one_transform_gives_the_U_statistic():
  # without folds the energies are the U-statistic of all pairs, symmetric copies included
  cs = crystalFrame("m-3m")
  o = orientation(rotation.rand(150, rng=rng), cs)
  w = rng.random(150)
  E = degreeEnergies(o, 5, folds=1, weights=w, rng=rng)
  x = o.data.reshape(-1, 4)
  U = _characters(np.stack([np.abs(x @ y.T) for y in E._symmetricCopies(x)]), 5)
  ww = w / w.sum()
  K = np.stack([(2 * l + 1) * U[l].mean(axis=0) for l in range(6)])
  direct = np.array([(ww @ K[l] @ ww - ww ** 2 @ np.diag(K[l])) / (1 - np.sum(ww ** 2)) for l in range(6)])
  assert np.allclose(E.A[1:], direct[1:], atol=1e-9) and np.all(np.isnan(E.se[1:]))


def test_the_lower_bounds_need_the_errors_and_grouped_points_need_folds():
  cs = crystalFrame("m-3m")
  o = orientation(rotation.rand(150, rng=rng), cs)
  E = degreeEnergies(o, 5, folds=1, weights=rng.random(150), rng=rng)
  # the lower bounds need the errors, grouped points need folds
  with pytest.raises(ValueError, match="folds"):
    lowerBoundHalfwidth(E)
  with pytest.raises(ValueError, match="folds"):
    degreeEnergies(o, 5, folds=1, groups=np.repeat(np.arange(15), 10), rng=rng)


def test_ucv_asks_for_a_higher_bandwidth_at_its_edge():
  from mtex.functions.halfwidth import ucvAtEdge
  cs = crystalFrame("m-3m")
  ori = discreteSample(unimodalODF(orientation(rotation.rand(rng=rng), cs), halfwidth=5 * degree), 20000, rng=rng)
  H = candidateHalfwidths("SO3")
  # a sharp sample pushes UCV to the narrowest halfwidth bandwidth 16 resolves, not so at 128
  assert ucvAtEdge(degreeEnergies(ori, 16, folds=1, rng=rng), H)
  assert not ucvAtEdge(degreeEnergies(ori, 128, folds=1, rng=rng), H)
  assert calcKernel(ori).halfwidth() < 3.9 * degree


def test_degreeEnergies_isTheClassOfItsSpace():
  from mtex.functions.halfwidth import S2DegreeEnergies, SO3DegreeEnergies, degreeEnergies
  cs = crystalFrame("432")
  E = degreeEnergies(orientation.rand(50, cs), 4, folds=1, rng=np.random.default_rng(0))
  F = degreeEnergies(vector3d.rand(50), 4, folds=1, rng=np.random.default_rng(0))
  assert isinstance(E, SO3DegreeEnergies) and isinstance(F, S2DegreeEnergies) and F.space == "S2"
