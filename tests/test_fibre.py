# fibres: the Julia RotationFibre page, the MATLAB @fibre methods, and the fit
import numpy as np
import pytest
from numpy import pi

from mtex import (fibre, orientation, rotation, crystalFrame, specimenFrame, vector3d, Miller, zvector, degree,
                  angle, symmetrise, multiplicity, projectIntoFundamentalRegion, cat)
from conftest import close


def rng():
  return np.random.default_rng(0)


# the Goss and brass components of 432 and the segment between them
def gossBrass():
  cs = crystalFrame("432")
  oA = orientation.goss(cs)
  oB = projectIntoFundamentalRegion(orientation.brass(cs), oA)
  return cs, oA, oB, fibre(oA, oB)


def test_fibre_segmentBetweenTwoComponentsCarriesTheirFrames():
  cs, oA, oB, f = gossBrass()
  assert f.CS is cs and f.SS is oA.SS and f.h.frame is cs and f.r.frame is oA.SS and f.shape == ()
  assert f.h.round().isclose(Miller(0, 1, 1, cs, "uvw")) and f.r.isclose(zvector)


def test_fibre_segmentEndsMapHToR():
  # every orientation on the fibre maps h to r, the ends exactly
  _, oA, oB, f = gossBrass()
  assert angle(oA * f.h.round(), f.r) < 1e-10 and angle(oB * f.h.round(), f.r) < 1e-10


def test_arcAngle_ofASegmentIsTheAngleOfItsEnds():
  _, oA, oB, f = gossBrass()
  assert not f.isFull and close(f.arcAngle(), angle(oA, oB)) and close(f.arcAngle(), 35 * degree)


def test_orientation_ofASegmentRunsFromEndToEndOnTheFibre():
  cs, oA, oB, f = gossBrass()
  os = f.orientation(50)
  assert os.shape == (50,) and os.CS is cs and angle(os[0], oA) < 1e-10 and angle(os[-1], oB) < 1e-10
  assert np.max(angle(os * f.h, f.r)) < 1e-10


def test_fibre_fullKeepsThePairAndDropsTheEnds():
  cs, oA, oB, f = gossBrass()
  full = fibre(f.h, f.r)
  assert full.isFull and close(full.arcAngle(), 2 * pi) and full.CS is cs and full.SS is f.SS
  assert fibre(oA, oB, full=True).isFull and fibre(oA, f.h).isFull and fibre(oA, f.h).isclose(full)


def test_orientation_ofAFullFibreClosesTheCircle():
  _, oA, _, f = gossBrass()
  full = fibre(f.h, f.r)
  os = full.orientation(400)
  assert np.max(angle(os * f.h, f.r)) < 1e-10 and angle(os[0], oA) < 1e-10 and angle(os[-1], oA) < 1e-10
  assert type(full.rotation(10)) is rotation


def test_fibre_endsHaveToShareTheirFrames():
  _, oA, oB, _ = gossBrass()
  with pytest.raises(ValueError):
    fibre(oA, orientation(oB, crystalFrame("m-3m")))
  with pytest.raises(ValueError):
    fibre(oA, Miller(1, 0, 0, crystalFrame("622"), "uvw"))


def test_fibre_coincidingEndsTurnAboutC():
  cs, oA, _, _ = gossBrass()
  assert fibre(oA, oA).h.isclose(Miller(0, 0, 1, cs, "uvw"))


def test_angle_vanishesAtTheComponentsOfAFullFibre():
  cs, oA, oB, _ = gossBrass()
  full = fibre(oA, oB, full=True)
  assert angle(full, oA) < 1e-10 and angle(full, orientation.brass(cs)) < 1e-10


def test_angle_ofTheCubeToTheFibreIs45Degrees():
  # <011> to z is 45 degrees for the cube orientation, however it is written
  cs, oA, oB, _ = gossBrass()
  full = fibre(oA, oB, full=True)
  assert close(angle(full, orientation.cube(cs)), 45 * degree) and close(angle(orientation.cube(cs), full), 45 * degree)
  assert close(angle(full, orientation(cs.properGroup().rotations()[5], cs)), 45 * degree)
  assert angle(full, orientation.cube(cs), symmetry=False) > 45 * degree - 1e-10


# a random fibre of 432 and 222 and 20 random orientations
def randomFibre():
  cs, ss = crystalFrame("432"), specimenFrame("222")
  g = rng()
  f = fibre.rand(cs, ss, rng=g)
  ori = orientation.rand(20, cs, ss, rng=g)
  return cs, ss, f, ori


def test_angle_toAFibreIsSymmetricUnderBothGroups():
  cs, ss, f, ori = randomFibre()
  d = angle(f, ori)
  assert d.shape == (20,)
  for s in ss.properGroup().rotations():
    assert np.max(np.abs(angle(f, orientation(s * rotation(ori), cs, ss)) - d)) < 1e-10
  for g in cs.properGroup().rotations():
    assert np.max(np.abs(angle(f, orientation(rotation(ori) * g, cs, ss)) - d)) < 1e-10


# crystal and specimen point groups with and without the inversion and improper elements
POINT_GROUP_PAIRS = [("m-3m", "1"), ("432", "mmm"), ("mm2", "222"), ("-43m", "1"), ("-1", "2/m")]


@pytest.mark.parametrize("pg", POINT_GROUP_PAIRS)
def test_angle_vanishesWhereAnImageOfHUnderThePointGroupIsAnImageOfR(pg):
  # the improper elements count: in a centrosymmetric crystal -h is an equivalent of h
  cs, ss = crystalFrame(pg[0]), specimenFrame(pg[1])
  h, r = Miller(1, 2, 4, cs, "uvw"), vector3d(1, 2, 3)
  f = fibre(h, r, ss)
  hs, rs = cs.rot * vector3d(h.data), ss.rot * r
  ori = orientation.map(hs[:, None], rs[None, :])
  assert np.max(angle(f, orientation(rotation(ori), cs, ss))) < 1e-7


def test_angle_vanishesAlongTheFibre():
  _, _, f, _ = randomFibre()
  assert np.max(angle(f, f.orientation(100))) < 1e-10


def test_angle_ofASmallStepOffTheFibreToFullPrecision():
  cs, ss, f, _ = randomFibre()
  step = rotation.byAxisAngle(f.r.perp(), 1e-6)
  assert close(angle(f, orientation(step * rotation(f.o1), cs, ss)), 1e-6, tol=1e-12)


def test_angle_fibreToFibre():
  cs, _, f, _ = randomFibre()
  assert angle(f, f) < 1e-8 and angle(fibre.alpha(cs), fibre.gamma(cs)) > 5 * degree


def test_angle_nearestLiesOnTheFibreAtTheAngle():
  # the orientation on the fibre nearest ori is omega away from it, for specimen symmetry and for two crystal frames
  cs, ss, f, ori = randomFibre()
  csB = crystalFrame("m-3m", mineral="B")
  fM = fibre.rand(cs, csB, rng=rng())
  for f, ori in ((f, ori), (fM, orientation.rand(20, cs, csB, rng=rng()))):
    omega, oriF = angle(f, ori, return_nearest=True)
    assert np.max(angle(f, oriF)) < 1e-10 and np.max(np.abs(angle(oriF, ori) - omega)) < 1e-10


def test_angle_nearestOfAnOrientationOnTheFibreIsItself():
  _, _, f, _ = randomFibre()
  ori = f.orientation(10)
  _, oriF = angle(f, ori, return_nearest=True)
  assert np.max(angle(oriF, ori)) < 1e-7


def test_angle_refusesAListOfFibres():
  cs, _, _, ori = randomFibre()
  with pytest.raises(ValueError):
    angle(fibre.rand(3, cs, rng=rng()), ori)


@pytest.mark.parametrize("name, h, r", [("alpha", (1, 1, 0), (1, 0, 0)), ("gamma", (1, 1, 1), (0, 0, 1)), ("eta", (1, 0, 0), (1, 0, 0)),
                                         ("epsilon", (1, 1, 0), (0, 1, 0)), ("tau", (1, 1, 0), (0, 1, 0))])
def test_fibre_namedFibresTurnAboutTheirCrystalDirectionAlongTheirSpecimenAxis(name, h, r):
  # the alpha fibre is <110> || RD, gamma <111> || ND, eta <100> || RD, epsilon and tau <110> || TD
  cs = crystalFrame("432")
  f = getattr(fibre, name)(cs)
  hs = symmetrise(Miller(*h, cs, "uvw"), unique=True).normalize().data.reshape(-1, 3)
  assert np.max(np.abs(hs @ f.h.normalize().data.reshape(3))) > 1 - 1e-9 and abs(f.r.normalize().data.reshape(3) @ r) > 1 - 1e-9 and not f.isFull


@pytest.mark.parametrize("name", ("alpha", "beta", "gamma", "epsilon", "eta", "tau"))
def test_fibre_namedFibreIsASegmentThroughItsEnds(name):
  cs = crystalFrame("432")
  f = getattr(fibre, name)(cs, specimenFrame("222"))
  assert f.CS is cs and f.SS is specimenFrame("222") and not f.isFull
  assert getattr(fibre, name)(cs, full=True).isFull
  assert angle(f, f.o1) < 1e-10 and angle(f, f.o2) < 1e-10


def test_fibre_theta():
  cs = crystalFrame("432")
  th = fibre.theta(cs)
  assert th.isFull and th.h.isclose(Miller(1, 0, 0, cs, "uvw")) and th.r.isclose(zvector)
  assert fibre.theta().CS.pointGroup == cs.pointGroup


def test_symmetrise_ofAFibreAreAllEqualToIt():
  th = fibre.theta(crystalFrame("432"), specimenFrame("222"))
  all_ = symmetrise(th)
  assert all_.shape == (24 * 4,) and np.all(all_.isclose(th))


def test_symmetrise_uniqueIdentifiesTheAntipodalPair():
  # <100> has six members, z two under 222, and (h, r) is the same fibre as (-h, -r)
  th = fibre.theta(crystalFrame("432"), specimenFrame("222"))
  u = symmetrise(th, unique=True)
  assert u.size == 6 and multiplicity(th) == 6 and np.all(u.isclose(th))


def test_multiplicity_ofAGenericFibreIsTheFullOrbit():
  # a generic direction has no antipode in its orbit, so nothing is identified
  cs, ss = crystalFrame("432"), specimenFrame("222")
  g = rng()
  assert multiplicity(fibre.theta(cs)) == 6 and multiplicity(fibre.rand(cs, ss, rng=g)) == 24 * 4
  assert multiplicity(fibre.rand(crystalFrame("1"), rng=g)) == 1


def test_fibre_equalityIsUpToSymmetryAndASegmentIsNotTheFullFibre():
  cs, ss = crystalFrame("432"), specimenFrame("222")
  a = fibre.alpha(cs)
  assert a == a and a != fibre.alpha(cs, full=True) and a != fibre.gamma(cs)
  assert not (a == fibre.alpha(cs, ss))


def test_fibre_equalityIdentifiesTheAntipodes():
  th = fibre.theta(crystalFrame("432"), specimenFrame("222"))
  assert fibre(th.h, -th.r).isclose(th) and fibre(-th.h, -th.r).isclose(th)


@pytest.mark.parametrize("pg", POINT_GROUP_PAIRS)
def test_symmetrise_ofAnArcLiesOnTheFibreAndKeepsItsLength(pg):
  cs, ss = crystalFrame(pg[0]), specimenFrame(pg[1])
  h = Miller(1, 2, 4, cs, "uvw")
  o1 = orientation.byEuler(10 * degree, 20 * degree, 30 * degree, cs, ss)
  arc = fibre(o1, o1 * orientation.byAxisAngle(h, 40 * degree))
  full = fibre(o1, h)
  S = symmetrise(arc).flatten()
  assert S.size == cs.rot.size * ss.rot.size
  assert np.max(angle(full, S.o1)) < 1e-7 and np.max(angle(full, S.o2)) < 1e-7
  assert np.allclose(S.arcAngle(), 40 * degree)


@pytest.mark.parametrize("pg, n", [(("m-3m", "1"), 48), (("m-3m", "mmm"), 192), (("432", "mmm"), 192), (("mm2", "222"), 16)])
def test_multiplicity_countsThePairsOfImagesUnderThePointGroupsUpToTheCommonFlip(pg, n):
  cs, ss = crystalFrame(pg[0]), specimenFrame(pg[1])
  assert multiplicity(fibre(Miller(1, 2, 4, cs, "uvw"), vector3d(1, 2, 3), ss)) == n


def test_symmetrise_uniqueRefusesAList():
  with pytest.raises(ValueError):
    symmetrise(fibre.rand(2, crystalFrame("432"), rng=rng()), unique=True)


def test_fibre_listShapeIndexingAndIteration():
  cs, ss = crystalFrame("432"), specimenFrame("222")
  f = fibre.rand(3, cs, ss, rng=rng())
  assert f.shape == (3,) and f.size == 3 and len(f) == 3 and f.CS is cs and f.SS is ss
  assert f[1].shape == () and f[1] == f[1] and f[1:].shape == (2,)
  assert all(g.shape == () for g in f)
  assert cat(f, f[0]).shape == (4,) and f.reshape(3, 1).shape == (3, 1)


def test_orientation_ofAListOfFibres():
  f = fibre.rand(3, crystalFrame("432"), specimenFrame("222"), rng=rng())
  assert f.orientation(10).shape == (10, 3) and np.all(f.isFull) and f.arcAngle().shape == (3,)
  ori = f.orientation(10)
  assert np.max(angle(ori * f.h, f.r)) < 1e-10


@pytest.mark.parametrize("full", [False, True])
def test_rotation_returnsTheAnglesItTurnedEachFibreBy(full):
  # rot = o1 * rot(h, omega) for every point of every fibre
  cs, ss = crystalFrame("432"), specimenFrame("222")
  o1, o2 = orientation.rand(3, cs, ss, rng=rng()), orientation.rand(3, cs, ss, rng=rng())
  f = fibre(o1, Miller(1, 1, 1, cs, "uvw")) if full else fibre(o1, projectIntoFundamentalRegion(o2, o1))
  rot, omega = f.rotation(20, return_angles=True)
  ref = rotation(f.o1) * rotation.byAxisAngle(vector3d(f.h.normalize().data), omega)
  assert omega.shape == rot.shape == (20, 3) and close(omega[-1], f.arcAngle()) and np.max(angle(rot, ref)) < 1e-10


def test_fibre_endsAndADirectionBroadcast():
  cs = crystalFrame("432")
  o = orientation.rand(4, cs, rng=rng())
  g = fibre(o, Miller(1, 1, 1, cs, "uvw"))
  assert g.shape == (4,) and g.h.shape == (4,) and np.all(g.isFull)


def test_fibre_randOfAShape():
  assert fibre.rand(2, 3, crystalFrame("432"), rng=rng()).shape == (2, 3)


# 500 orientations scattered by 2 degrees about a random fibre, as sampled
def sampledFibre(g):
  cs, ss = crystalFrame("432"), specimenFrame("222")
  f = fibre.rand(cs, ss, rng=g)
  ori = orientation(f.orientation(500) * rotation.byAxisAngle(vector3d.rand(500, rng=g), 2 * degree), cs, ss)
  return cs, ss, f, ori


def test_fit_recoversAFibreFromOneCircle():
  # the representatives as sampled lie on one circle, which the raw tensor sees
  cs, ss, f, ori = sampledFibre(rng())
  fRec, lam, delta = fibre.fit(ori, diagnostics=True, symmetry=False)
  assert fRec.isFull and fRec.CS is cs and fRec.SS is ss and angle(fRec, f) < 1 * degree and delta < 2 * degree
  assert lam.shape == (4,) and close(np.sum(lam), 1) and lam[-2] > 100 * lam[-3]


def test_fit_putsRInTheUpperHemisphere():
  # the fit fixes the sign of (h, r) by r in the upper hemisphere, whatever the sample
  g = rng()
  _, _, _, ori = sampledFibre(g)
  for k in range(4):
    sub = ori[g.permutation(500)[:200]]
    r = fibre.fit(sub, symmetry=False).r.data.reshape(3)
    assert r[2] > 0 or (abs(r[2]) < 1e-9 and r[1] > 0)


# a concentrated cloud about a 30 degree segment, scattered over the crystal group; with a
# specimen symmetry on top the fundamental region is too small for a 30 degree arc to count as
# concentrated, so the cloud has none, as measured orientations do
def scatteredCloud():
  g = rng()
  cs = crystalFrame("432")
  f = fibre.rand(cs, rng=g)
  seg = fibre(f.o1, f.orientation(13)[1])
  ori = orientation(seg.orientation(500) * rotation.byAxisAngle(vector3d.rand(500, rng=g), 2 * degree), cs)
  ori = orientation(rotation(ori) * cs.properGroup().rotations()[g.integers(24, size=500)], cs)
  return cs, f, ori


def test_fit_gathersAScatteredCloudAtItsMeanFirst():
  _, f, ori = scatteredCloud()
  fRec, lam, delta = fibre.fit(ori, diagnostics=True)
  assert angle(fRec, f) < 2 * degree and abs(delta - np.mean(angle(f, ori))) < 0.2 * degree


def test_fit_withoutDiagnosticsGivesTheFibre():
  _, _, ori = scatteredCloud()
  fRec, _, _ = fibre.fit(ori, diagnostics=True)
  assert type(fibre.fit(ori)) is fibre and fibre.fit(ori) == fRec


def test_fit_dropsNaNOrientations():
  cs, _, ori = scatteredCloud()
  fRec = fibre.fit(ori)
  o = orientation.cat(ori, orientation.nan(cs))
  assert angle(fibre.fit(o), fRec) < 1e-6


@pytest.mark.parametrize("signs", [(1, 1), (-1, 1), (1, -1), (-1, -1)])
def test_fit_hemisphere_does_not_depend_on_eigenvector_signs(monkeypatch, signs):
  cs = crystalFrame("1")
  f = fibre(Miller(1, 2, 3, cs), vector3d(0.2, 0.3, -1).normalize())
  ori = f.orientation(100)
  eigh = np.linalg.eigh

  def signed(A):
    vals, vecs = eigh(A)
    vecs[:, -2:] *= signs
    return vals, vecs

  monkeypatch.setattr(np.linalg, "eigh", signed)
  fitted = fibre.fit(ori, symmetry=False)
  assert fitted.r.z > 0 and angle(fitted, f) < 1e-7


def test_fibre_reprOfASegment():
  s = repr(fibre.alpha(crystalFrame("432")))
  assert s.startswith("fibre (432 → y↓→x)") and "h || r :" in s and "o1 → o2:" in s


def test_fibre_reprOfAFullFibreAndAList():
  cs = crystalFrame("432")
  assert "o1 → o2" not in repr(fibre.theta(cs)) and "size: 3" in repr(fibre.rand(3, cs, rng=rng()))
  assert "[100] || (0,0,1)" in repr(fibre.theta(cs))
