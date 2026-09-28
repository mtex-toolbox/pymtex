# symmetrise, multiplicity, dot, angle, unique, mean under the group; grain exchange
# (the symmetrise parts of check_Miller, check_antipodalDistance)
import numpy as np
import pytest
from numpy import pi

import mtex
from mtex import (orientation, rotation, crystalFrame, specimenFrame, Miller, vector3d, degree,
                  symmetrise, multiplicity, angle, dot, unique, mean, inv, cat, axis)
from conftest import close

rng = np.random.default_rng(0)


def bruteAngle(w, v):
  # the smallest angle from w to any equivalent of v, or of its inverse under exchange
  return np.array([[np.min(w[i].angle(symmetrise(v[j]), noSymmetry=True)) for j in range(v.size)] for i in range(w.size)])


@pytest.mark.parametrize("s", ["321", "m-3m", "1", "622"])
def test_grain_exchange_symmetry(s):
  cs = crystalFrame(s)
  v = orientation(rotation.rand(6, rng=rng), cs, cs, antipodal=True)
  w = orientation(rotation.rand(4, rng=rng), cs, cs, antipodal=True)
  bf = bruteAngle(w, v)
  d = angle(w[:, None], v[None, :])
  assert d.shape == (4, 6) and np.max(np.abs(d - bf)) < 1e-8
  v0, w0 = orientation(v, antipodal=False), orientation(w, antipodal=False)
  v0.antipodal, w0.antipodal = False, False
  assert np.max(np.abs(angle(w0[:, None], v[None, :]) - d)) < 1e-8
  assert np.max(np.abs(angle(w[:, None], v0[None, :]) - d)) < 1e-8
  assert np.max(np.abs(angle(w0[:, None], v0[None, :]) - bruteAngle(w0, v0))) < 1e-8
  assert symmetrise(v[0]).size == 2 * cs.numProper() ** 2 and symmetrise(v0[0]).size == cs.numProper() ** 2
  assert np.max(angle(v, v)) < 1e-6
  assert close(angle(v), angle(inv(v)))


def test_vectorised_forms_agree_with_scalar():
  cs = crystalFrame("m-3m")
  os = orientation(rotation.rand(20, rng=rng), cs)
  ref = orientation(rotation.rand(rng=rng), cs)
  assert close(angle(os, ref), [angle(o, ref) for o in os])
  assert close(angle(ref, os), angle(os, ref))
  assert close(angle(os, os), np.zeros(20), tol=1e-6)
  assert close(dot(os, ref, noSymmetry=True), np.abs(np.sum(os.data * ref.data, axis=-1)))
  assert close(dot(os, ref), [dot(o, ref) for o in os]) and np.all(dot(os, ref) >= dot(os, ref, noSymmetry=True) - 1e-12)
  assert close(angle(os), [angle(o) for o in os]) and np.all(angle(os) <= angle(os, noSymmetry=True) + 1e-12)
  assert close(2 * np.arccos(np.minimum(dot(os, ref), 1)), angle(os, ref), tol=1e-6)
  # a list of references projects every element to the equivalent closest to its own one
  refs = orientation(rotation.rand(20, rng=rng), cs, crystalFrame("6/mmm"))
  mos = orientation(rotation.rand(20, rng=rng), cs, crystalFrame("6/mmm"))
  proj = mos.project2FundamentalRegion(refs)
  assert close(proj.data, np.concatenate([mos[k].project2FundamentalRegion(refs[k]).data[None] for k in range(20)]))
  assert close(angle(proj, refs, noSymmetry=True), angle(mos, refs))


def test_symmetrise_multiplicity_unique_mean_of_directions():
  cs = crystalFrame("6/mmm", [3, 3, 5])
  csC = crystalFrame("m-3m")
  for fr, expected in ((cs, None), (csC, [6, 12, 8, 48])):
    for k, m in enumerate(([1, 0, 0], [1, 1, 0], [1, 1, 1], [3, 2, 1])):
      h = Miller(*m, fr)
      n = symmetrise(h, unique=True).size
      assert multiplicity(h) == n and fr.numSym() % n == 0 and symmetrise(h).shape == (fr.numSym(),)
      if expected is not None:
        assert n == expected[k]
  assert symmetrise(Miller([[1, 0, 0], [1, 1, 0]], csC)).shape == (48, 2)
  assert multiplicity(Miller([[1, 0, 0], [1, 1, 0]], csC)).tolist() == [6, 12]
  # equivalent directions compare equal, and unique keeps one of each family
  assert Miller(1, 0, 0, csC) == Miller(0, 1, 0, csC) and Miller(1, 0, 0, csC) != Miller(1, 1, 0, csC)
  assert angle(Miller(1, 0, 0, csC), Miller(0, -1, 0, csC)) < 1e-10
  assert close(angle(Miller(1, 0, 0, csC), Miller(0, -1, 0, csC), noSymmetry=True), pi / 2)
  assert close(dot(Miller(1, 1, 1, csC), Miller(-1, 1, 1, csC)), 3) and close(dot(Miller(1, 1, 1, csC), Miller(-1, 1, 1, csC), noSymmetry=True), 1)
  assert close(dot(Miller(1, 1, 1, csC), Miller(-1, 1, 1, csC), min=True), -3)
  assert dot(Miller(1, 1, 1, csC), Miller(-1, 1, 1, csC), all=True).shape == (48,)
  assert unique(cat(symmetrise(Miller(1, 1, 0, csC)), symmetrise(Miller(1, 0, 0, csC)))).size == 2
  u, iv, iu = unique(symmetrise(Miller(1, 1, 0, csC)), return_index=True, return_inverse=True)
  assert u.size == 1 and iv.tolist() == [0] and np.all(iu == 0)
  # the group-free sibling stands for itself
  assert symmetrise(Miller("(100)", csC)).size == 1 and Miller("(100)", csC) != Miller("(0-10)", csC)
  # a plane normal with no front and back: the antipodal orbit, and axes
  qz = crystalFrame("321", [4.913, 4.913, 5.405], mineral="Quartz")
  r = Miller(1, 0, -1, 1, qz)
  assert multiplicity(r) == 6 and multiplicity(r, antipodal=True) == 3 and not r.antipodal
  a = vector3d(r, antipodal=True)
  assert multiplicity(a) == 3 and a.frame is qz and a.dispStyle == "hkil"
  assert symmetrise(r, antipodal=True).size == 12 and symmetrise(Miller(1, 0, 0, csC), antipodal=True).size == 48
  assert multiplicity(Miller(1, 0, 0, csC)) == 6 and multiplicity(vector3d(Miller(1, 0, 0, csC), antipodal=True)) == 3
  # a specimen frame's group does not enter vector comparison
  ss = specimenFrame("222")
  v1, v2 = vector3d(1, 0, 0, frame=ss), vector3d(-1, 0, 0, frame=ss)
  assert v1 != v2 and close(angle(v1, v2), pi) and close(dot(v1, v2), -1)
  # the mean of crystal directions is taken over the closest equivalents
  h = Miller(1, 1, 1, csC)
  spread = symmetrise(cat(h, Miller(1.05, 1, 0.95, csC), Miller(0.95, 1.05, 1, csC)))
  m = mean(spread.flatten())
  assert angle(m, h) < 3 * degree and m.frame is csC
  assert Miller(1, 0, 0, csC) != Miller(1, 0, 0, crystalFrame("6/mmm", [3, 3, 5]))


def test_orientation_unique_and_mean():
  csC = crystalFrame("m-3m")
  o = orientation(rotation.rand(rng=rng), csC)
  eq = symmetrise(o)
  assert unique(eq).size == 1 and unique(cat(eq, orientation(rotation.rand(rng=rng), csC))).size == 2
  assert unique(eq, noSymmetry=True).size == 24
  u, iv, iu = unique(cat(o, o, orientation(rotation.rand(rng=rng), csC)), return_index=True, return_inverse=True)
  assert u.size == 2 and iv.tolist() == [0, 2] and iu.tolist() == [0, 0, 1]
  small = rotation.byAxisAngle(vector3d.rand(200, rng=rng), 0.1 * rng.random(200))
  G = csC.properGroup().rotations()
  cloud = orientation(o * small * G[rng.integers(0, 24, 200)], csC)
  assert angle(mean(cloud), o) < 2 * degree and mean(cloud).CS is csC
  assert angle(mean(cloud, weights=np.ones(200)), o) < 2 * degree
  assert angle(mean(cat(cloud, orientation(rotation.rand(rng=rng), csC)), robust=True), o) < 3 * degree
  with pytest.raises(ValueError):
    mean(orientation(csC))


def test_axis():
  q = crystalFrame("321", [4.9, 4.9, 5.4], mineral="Quartz")
  fe = crystalFrame("m-3m", [2.87, 2.87, 2.87], mineral="Ferrite")
  ss = specimenFrame("222")
  rots = rotation.rand(6, rng=rng)
  mori = orientation(rots, q, q, antipodal=True)
  plain = orientation(rots, q, q)
  ori = orientation(rots, q, ss)
  # the axis is that of the representative in the fundamental region, not of the value given
  assert np.all(axis(mori) == axis(mori.project2FundamentalRegion(), noSymmetry=True))
  assert np.any(~(axis(mori) == axis(mori, noSymmetry=True)))
  assert np.all(axis(mori, noSymmetry=True) == rots.axis())
  # only the rotations both frames share reduce it, and grain exchange takes the sign
  assert axis(mori).frame.sym == q.sym.properGroup().Laue() and axis(mori).frame.isLaue
  assert axis(plain).frame.sym == q.sym.properGroup() and not axis(plain).frame.isLaue
  assert axis(ori).frame.numSym() == q.sym.disjoint(ss.sym).properGroup().numSym()
  pb = orientation(rots, q, fe)
  assert axis(pb).frame.numSym() < min(q.numProper(), fe.numProper())
  assert np.allclose(axis(mori).frame.abc, q.abc) and axis(mori).frame.mineral == q.mineral
  assert np.all(axis(mori)[2] == axis(mori[2]))
  # the axis between two orientations lives in the specimen frame
  cs = crystalFrame("432")
  o1, o2 = orientation.rand(cs, rng=rng), orientation.rand(cs, rng=rng)
  a = axis(o1, o2)
  assert a.frame is o2.SS and a.antipodal
  assert axis(orientation.rand(5, cs, rng=rng), orientation.rand(5, cs, rng=rng)).shape == (5,)
  with pytest.raises(ValueError):
    axis(o1, orientation.rand(cs, specimenFrame.rolling, rng=rng))


def test_comparison_needs_fitting_frames():
  cs = crystalFrame("m-3m")
  o = orientation.rand(cs, rng=rng)
  with pytest.raises(ValueError):
    angle(o, orientation.rand(cs, specimenFrame.rolling, rng=rng))
  with pytest.raises(ValueError):
    angle(o, inv(o))
  assert angle(o, orientation.rand(crystalFrame("m-3m", mineral="other"), rng=rng)) <= pi
  assert not (o == orientation(o, crystalFrame("432")))


def test_the_axis_of_a_pair_turns_the_second_onto_the_first():
  cs = crystalFrame('432')
  rng = np.random.default_rng(8)
  o1, o2 = orientation.rand(500, cs, rng=rng), orientation.rand(500, cs, rng=rng)
  R = rotation.byAxisAngle(vector3d(axis(o1, o2).data), angle(o1, o2))
  assert np.max(angle(R * o2, o1)) < 1e-9


@pytest.mark.parametrize("backend", ["numba", "numpy"])
@pytest.mark.parametrize("s, antipodal", [("m-3m", False), ("432", False), ("6/mmm", False), ("-3m", True), ("2/m", True)])
def test_folded_directions_lie_in_the_sector_exactly(backend, s, antipodal):
  # an image within the tolerance of the rim must not win over the equivalent one inside
  cs = crystalFrame(s)
  x = rng.standard_normal((20000, 3))
  m = Miller(vector3d(x / np.linalg.norm(x, axis=1, keepdims=True)), cs, antipodal=antipodal)
  sR = mtex.fundamentalSector(cs, antipodal=antipodal)
  mtex.setMTEXpref("kernels", backend)
  try:
    p = mtex.project2FundamentalRegion(m).data.reshape(-1, 3)
  finally:
    mtex.setMTEXpref("kernels", "numba")
  margin = np.min(p @ np.asarray(sR.N).T - sR.alpha, axis=1)
  if sR.antipodal:
    margin = np.maximum(margin, np.min(-p @ np.asarray(sR.N).T - sR.alpha, axis=1))
  assert margin.min() > -1e-12
  assert np.max(np.abs(angle(Miller(vector3d(p), cs), m, noSymmetry=False))) < 1e-8
