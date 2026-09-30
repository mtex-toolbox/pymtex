# the coincidence site lattices of any lattice: the tables of the literature, sigma counted, the search
import numpy as np
import pytest

from mtex import CSL, Miller, angle, axis, crystalFrame, degree, symmetrise, vector3d
from mtex.geometry.csl import _rationalBasis, _shell


def inFamily(o, CS, *hkl):
  fam = symmetrise(Miller(*hkl, CS, "uvw"), unique=True)
  return np.min(angle(vector3d(axis(o), antipodal=True), fam, noSymmetry=True)) < 1e-8


# the index of L n ML in L, counted over the residues modulo sigma
def countedSigma(M, sigma):
  g = np.stack(np.meshgrid(*[np.arange(sigma)] * 3, indexing="ij"), -1).reshape(-1, 3)
  y = g @ np.linalg.inv(M).T
  return sigma ** 3 // np.sum(np.all(np.abs(y - np.rint(y)) < 1e-6, axis=1))


def test_the_cubic_classes_are_grimmers():
  cs = crystalFrame("m-3m", mineral="iron")
  s3 = CSL(3, cs)
  assert s3.size == 1 and abs(s3.angle() - 60 * degree) < 1e-10 and s3.antipodal and inFamily(s3, cs, 1, 1, 1)
  assert abs(CSL(5, cs).angle() - 2 * np.arctan(1 / 3)) < 1e-10 and inFamily(CSL(5, cs), cs, 1, 0, 0)
  assert abs(CSL(9, cs).angle() - 38.94 * degree) < 0.01 * degree and inFamily(CSL(9, cs), cs, 1, 1, 0)
  # the classes of one sigma ascend in angle: 13a, 13b and 49a, 49b, 49c
  table = {7: [38.21], 11: [50.48], 13: [22.62, 27.80], 25: [16.26, 51.68], 29: [43.60, 46.40], 49: [43.57, 43.57, 49.23]}
  for s, angles in table.items():
    assert np.allclose(np.atleast_1d(np.asarray(CSL(s, cs).angle())) / degree, angles, atol=0.01)


def test_the_hexagonal_classes_of_the_ideal_ratio_are_bonnets():
  hx = crystalFrame("6/mmm", [1, 1, np.sqrt(8 / 3)])
  table = {7: 21.79, 10: 78.46, 11: 62.96, 13: 27.80, 14: 44.42, 17: 86.63, 18: 70.53, 19: 13.17}
  for s, a in table.items():
    assert np.allclose(np.asarray(CSL(s, hx).angle()), [a * degree], atol=0.01 * degree)
  # 17a about <2-1-10> next to the {10-12} twin, 11 about <10-10> next to the {11-22} twin
  assert inFamily(CSL(7, hx), hx, 0, 0, 0, 1) and inFamily(CSL(11, hx), hx, 1, 0, -1, 0)
  assert inFamily(CSL(17, hx), hx, 2, -1, -1, 0) and CSL(3, hx).size == 0


def test_a_list_of_sigmas_ascends_in_sigma_then_angle():
  cs = crystalFrame("m-3m")
  o = CSL([13, 3, 5], cs)
  assert o.size == 4 and np.allclose(np.asarray(o.angle()) / degree, [60, 36.87, 22.62, 27.80], atol=0.01)


@pytest.mark.parametrize("sym, abc", [("m-3m", [1, 1, 1]), ("6/mmm", [1, 1, np.sqrt(8 / 3)]), ("4/mmm", [1, 1, np.sqrt(2)]),
                                      ("mmm", [1, np.sqrt(2), np.sqrt(3)]), ("-3m", [1, 1, np.sqrt(3)]), ("2/m", [1, 1.5, 2]),
                                      ("R-3m", [1, 1, np.sqrt(6)]), ("I4/mmm", [1, 1, np.sqrt(3)]), ("Fmmm", [1, np.sqrt(2), np.sqrt(3)])])
def test_every_misorientation_has_the_sigma_it_is_found_for(sym, abc):
  # sigma is the index in the Bravais lattice, counted in a primitive cell of it
  cs = crystalFrame(sym, abc)
  A = cs.primitiveAxes().data.T
  for s in range(1, 12):
    o = CSL(s, cs)
    for k in range(o.size):
      M = np.linalg.solve(A, o[k].matrix().reshape(3, 3) @ A)
      assert np.allclose(s * M, np.rint(s * M), atol=1e-8) and countedSigma(M, s) == s


def test_a_real_ratio_has_the_basal_coincidences_exactly_and_the_others_near():
  ti = crystalFrame("6/mmm", [2.95, 2.95, 4.68], mineral="titanium")
  exact = CSL(range(1, 20), ti)
  assert exact.size == 4 and all(inFamily(exact[k], ti, 0, 0, 0, 1) for k in range(1, 4))
  # within a misfit of one percent c^2/a^2 = 2.517 is 5/2
  B = _rationalBasis(ti.axes.data.T, 0.01)
  G = B.T @ B
  assert np.allclose(G / G[0, 0], [[1, -0.5, 0], [-0.5, 1, 0], [0, 0, 2.5]])
  assert CSL(range(1, 20), ti, delta=0.01).size > exact.size


def test_a_sigma_without_coincidences_is_empty_and_no_sigma_below_one():
  cs = crystalFrame("m-3m")
  assert CSL(4, cs).size == 0
  with pytest.raises(ValueError):
    CSL(0, cs)


def test_the_shell_is_the_brute_force_one():
  for cs, r, delta in ((crystalFrame("m-3m"), 13 * np.sqrt(2), 1e-9), (crystalFrame("6/mmm", [1, 1, 1.63]), 7 * 1.63, 0.02)):
    A = cs.axes.data.T
    b = np.ceil(r * (1 + delta) * np.linalg.norm(np.linalg.inv(A), axis=1)).astype(int) + 1
    g = np.stack(np.meshgrid(*[np.arange(-x, x + 1) for x in b], indexing="ij"), -1).reshape(-1, 3)
    brute = g[np.abs(np.linalg.norm(g @ A.T, axis=1) - r) <= delta * r + 1e-9]
    assert set(map(tuple, _shell(A, r, delta))) == set(map(tuple, brute))


def test_the_cubic_P_I_and_F_lattices_share_their_coincidences():
  classes = {s: [np.round(np.atleast_1d(np.asarray(CSL(range(1, 30, 2), crystalFrame(s, [4, 4, 4])).angle())), 10)]
             for s in ("Pm-3m", "Im-3m", "Fm-3m")}
  assert np.array_equal(classes["Pm-3m"], classes["Im-3m"]) and np.array_equal(classes["Pm-3m"], classes["Fm-3m"])


def test_a_rhombohedral_lattice_has_other_coincidences_than_the_hexagonal_cell():
  hP, hR = crystalFrame("P-3m1", [1, 1, np.sqrt(6)]), crystalFrame("R-3m", [1, 1, np.sqrt(6)])
  angles = lambda o: np.atleast_1d(np.asarray(o.angle()))
  assert not np.array_equal(angles(CSL(7, hP)), angles(CSL(7, hR)))
  # the letter given overrides the centring of the frame
  assert np.allclose(angles(CSL(range(1, 10), hP, "R")), angles(CSL(range(1, 10), hR)))


def test_a_centring_the_point_group_does_not_keep_is_refused():
  with pytest.raises(ValueError):
    CSL(3, crystalFrame("6/mmm"), "I")
