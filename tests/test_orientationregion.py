# the fundamental region of a pair of frames and the projection into it (check_fundamentalRegion)
import numpy as np
import pytest
from numpy import pi

from mtex import orientation, rotation, crystalFrame, specimenFrame, fundamentalRegion, angle, degree, vector3d, zvector
from mtex.geometry.orientationregion import eulerRegion
from mtex.geometry.symreduce import reductionGroups

rng = np.random.default_rng(0)
LAUE = ["-1", "2/m", "mmm", "4/m", "4/mmm", "-3", "-3m", "6/m", "6/mmm", "m-3", "m-3m"]


@pytest.mark.parametrize("pair", [(l, l) for l in LAUE] + [("m-3m", "-3m"), ("6/mmm", "2/m"), ("4/mmm", "-1"), ("mmm", "m-3")])
def test_projection_reproduces_the_disorientation_angle(pair):
  s1, s2 = pair
  cs1, cs2 = crystalFrame(s1, mineral="A"), crystalFrame(s2, mineral="B")
  N = 300
  q1, q2 = rotation.rand(N, rng=rng), rotation.rand(N, rng=rng)
  o1, o2 = orientation(q1, cs1), orientation(q2, cs2)
  mori = orientation(q1.inv() * q2, cs2, cs1)
  omega = mori.projectIntoFundamentalRegion().angle(symmetry=False)
  omega2 = angle(o1, o2)
  assert np.max(np.abs(omega - omega2)) < 1e-10
  plain = (q1.inv() * q2).angle()
  assert np.all(omega <= plain + 1e-10)
  if cs1.numProper() * cs2.numProper() > 1:
    assert np.any(omega < plain - 1e-6)
  assert np.all(fundamentalRegion(cs2, cs1).checkInside(mori.projectIntoFundamentalRegion()))
  assert np.all(mori.fundamentalRegion().checkInside(mori.projectIntoFundamentalRegion()))


@pytest.mark.parametrize("pair", [("m-3m", "1"), ("mmm", "1"), ("6/mmm", "222"), ("2/m", "1"), ("m-3m", "m-3m"), ("-3m", "-3m")])
def test_the_region_holds_one_orientation_per_coset(pair):
  a, b = pair
  q = rotation.rand(100000, rng=np.random.default_rng(5))
  cs = crystalFrame(a)
  ss = crystalFrame(b, [3, 3, 3], mineral="other") if b == a else specimenFrame(b)
  oR = fundamentalRegion(cs, ss)
  L, R = reductionGroups(orientation(rotation(), cs, ss))
  frac = np.sum(oR.checkInside(q, tol=0)) / q.size
  assert abs(frac * len(L) * len(R) - 1) < 0.1
  assert np.all(oR.checkInside(orientation(q[:2000], cs, ss).projectIntoFundamentalRegion()))


def test_max_angle_and_axis_sector():
  cs = crystalFrame("m-3m")
  oR = fundamentalRegion(cs)
  assert abs(oR.maxAngle() - 62.8 * degree) < 1.5 * degree
  assert abs(oR.maxAngle(vector3d(1, 1, 1)) - 60 * degree) < 1e-10 and abs(oR.maxAngle(zvector) - 45 * degree) < 1e-10
  assert abs(fundamentalRegion(crystalFrame("1")).maxAngle() - pi) < 1e-12
  sR = cs.fundamentalSector(angle=60 * degree)
  assert len(sR.N) > len(cs.fundamentalSector().N) and np.all(np.isfinite(sR.alpha))
  assert cs.fundamentalRegion(cs).frameA is cs and cs.sym.fundamentalRegion(cs.sym).N.shape[1] == 4
  assert fundamentalRegion(cs, antipodal=True).antipodal and "faces" in repr(oR)


def test_euler_region():
  assert np.allclose(eulerRegion(crystalFrame("1")), (2 * pi, pi, 2 * pi))
  assert np.allclose(eulerRegion(crystalFrame("m-3m")), (2 * pi, pi / 2, pi / 2))
  assert np.allclose(eulerRegion(crystalFrame("6/mmm")), (2 * pi, pi / 2, pi / 3))
  # the two-folds of both sides fold phi1 about zero, so it is the one halved
  assert np.allclose(eulerRegion(crystalFrame("mmm"), specimenFrame("222")), (pi / 2, pi / 2, pi))
  assert np.allclose(eulerRegion(crystalFrame("-3")), (2 * pi, pi, 2 * pi / 3))


def test_grain_exchange_halves_the_region():
  for g in ('mmm', 'm-3m', '-1', '6/mmm'):
    cs = crystalFrame(g)
    r = orientation.rand(40000, cs, cs, rng=np.random.default_rng(3))
    full = np.mean(fundamentalRegion(cs, cs).checkInside(r))
    half = np.mean(fundamentalRegion(cs, cs, antipodal=True).checkInside(r))
    assert abs(half / full - 0.5) < 0.05


# a group turned against its axes, which has no conventional sector and takes the Dirichlet domain
def turned(g):
  from mtex.geometry.kernels import qmul
  r = rotation.byAxisAngle(vector3d(1, 2, 3), 37 * degree).data.reshape(4)
  q = crystalFrame(g).sym.rot.data.reshape(-1, 5)[:, :4]
  return crystalFrame(rotation._fromData(np.array([qmul(qmul(r, x), r * [1, -1, -1, -1]) for x in q])), intern=False)


# -3 has a conventional sector no centre's Dirichlet domain is; with and without exchange
@pytest.mark.parametrize("antipodal", [False, True])
@pytest.mark.parametrize("group", ["-1", "mmm", "4/m", "-3", "-3m", "6/mmm", "m-3m", "turned 432", "turned 32"])
def test_the_projection_lands_in_the_region(group, antipodal):
  cs = turned(group.split()[1]) if group.startswith("turned") else crystalFrame(group)
  m = orientation.rand(2000, cs, cs, rng=np.random.default_rng(4))
  m.antipodal = antipodal
  p = m.projectIntoFundamentalRegion()
  assert np.all(fundamentalRegion(p).checkInside(p)) and np.max(angle(p, m)) < 1e-9
