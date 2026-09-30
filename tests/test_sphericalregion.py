# spherical regions, fundamental and conventional sectors
import numpy as np
import pytest
from numpy import pi

from mtex import (sphericalRegion, fundamentalSector, conventionalSector, projectIntoRegion, projectIntoFundamentalRegion,
                  crystalFrame, symmetry, rotation, vector3d, xvector, zvector, degree, angle, cat)
from mtex.geometry.kernels import qmul
from mtex.geometry.sphericalregion import sectorCenter, dirichletSector, tilesSphere

LAUE_AND_MORE = ["1", "-1", "2/m", "mmm", "4/m", "4/mmm", "-3", "-3m", "6/m", "6/mmm", "m-3", "m-3m", "321", "-42m", "23"]
CONVENTIONAL = ["1", "-1", "2", "m", "2/m", "222", "mm2", "mmm", "4", "-4", "4/m", "422", "4mm", "-42m", "4/mmm",
                "3", "-3", "32", "3m", "-3m", "6", "-6", "6/m", "622", "6mm", "-6m2", "6/mmm", "23", "m-3", "432", "-43m", "m-3m"]
GROUPS = ["1", "-1", "2", "m", "2/m", "222", "mm2", "mmm", "4", "-4", "4/m", "422", "4mm", "-42m", "-4m2", "4/mmm", "3", "-3", "321",
          "312", "3m1", "31m", "-31m", "-3m1", "6", "-6", "6/m", "622", "6mm", "-6m2", "-62m", "6/mmm", "23", "m-3", "432", "-43m", "m-3m"]


# 2000 random directions of a frame and their projection into its fundamental sector
def projected(s):
  cs = crystalFrame(s)
  vs = vector3d(vector3d.rand(2000, rng=np.random.default_rng(5)), frame=cs)
  return cs, vs, projectIntoFundamentalRegion(vs)


@pytest.mark.parametrize("s", LAUE_AND_MORE)
def test_projectIntoFundamentalRegion_keepsTheDirectionUpToSymmetry(s):
  _, vs, p = projected(s)
  assert np.all(angle(p[:200], vs[:200]) < 1e-10)


@pytest.mark.parametrize("s", LAUE_AND_MORE)
def test_projectIntoFundamentalRegion_landsInTheSector(s):
  cs, _, p = projected(s)
  assert np.all(fundamentalSector(cs).checkInside(p))


@pytest.mark.parametrize("s", LAUE_AND_MORE)
def test_fundamentalSector_volumeIsOneOverTheOrder(s):
  cs = crystalFrame(s)
  assert abs(fundamentalSector(cs).volume() * cs.numSym() - 1) < 0.05


@pytest.mark.parametrize("s", LAUE_AND_MORE)
def test_projectIntoFundamentalRegion_isIdempotent(s):
  _, _, p = projected(s)
  assert np.max(angle(p, projectIntoFundamentalRegion(p), symmetry=False)) < 1e-10


@pytest.mark.parametrize("s", LAUE_AND_MORE)
def test_fundamentalSector_carriesTheFrame(s):
  cs = crystalFrame(s)
  sR = fundamentalSector(cs)
  assert sR.frame is cs and cs.fundamentalSector() == sR


def test_fundamentalSector_cubicIsTheStandardTriangle():
  sR = fundamentalSector(crystalFrame("m-3m"))
  assert len(sR.N) == 3
  for v in ([0, 0, 1], [1, 0, 1], [1, 1, 1]):
    assert sR.checkInside(vector3d(*v).normalize())
  assert not sR.checkInside(vector3d(-1, 0, 0)) and not sR.checkInside(vector3d(1, 1, 0.5).normalize())
  assert sR.checkInside(vector3d([[0, 0, 1], [-1, 0, 0]])).tolist() == [True, False]


def test_fundamentalSector_hexagonalIsATriangleAtZ():
  sH = fundamentalSector(crystalFrame("6/mmm", [3, 3, 5]))
  assert len(sH.N) == 3 and sH.checkInside(zvector)


def test_fundamentalSector_ofAFivefoldHasAFifth():
  pg = symmetry(rotation.byAxisAngle(zvector, 2 * pi / 5))
  assert abs(fundamentalSector(pg).volume() * 5 - 1) < 0.05


def test_checkInside_anAxisCountsOnBothSides():
  # an axis counts on both sides, and the Laue sibling halves the sector
  sR = fundamentalSector(crystalFrame("m-3m"))
  a = vector3d(0, 0, -1, antipodal=True)
  assert sR.checkInside(a) and not sR.checkInside(vector3d(0, 0, -1))


def test_fundamentalSector_antipodalCountsBothEnds():
  # an antipodal region counts both ends, so the sector of the Laue group covers 2/48 of the sphere
  assert abs(fundamentalSector(crystalFrame("432"), antipodal=True).volume() * 24 - 1) < 0.05


def test_fundamentalSector_triclinicHalfSphereAboutACentre():
  tric = crystalFrame("-1", [2, 3, 4], np.array([80, 95, 110]) * degree)
  assert fundamentalSector(tric, center=xvector).N.shape == (1, 3)
  assert np.allclose(fundamentalSector(tric, center=xvector).N[0], [1, 0, 0])


def test_vertices_ofTheStandardTriangle():
  V = fundamentalSector(crystalFrame("m-3m")).vertices()
  assert V.size == 3
  for v in ([0, 0, 1], [1, 0, 1], [1, 1, 1]):
    assert np.any(angle(V, vector3d(*v)) < 1e-8)


def test_thetaInterval_agreesWithCheckInside():
  sR = fundamentalSector(crystalFrame("m-3m"))
  for rho in 2 * pi * np.random.default_rng(5).random(20):
    iv = sR.thetaInterval(rho)
    # off the poles: the pole is a corner of the triangle, a single point of every meridian
    theta = np.linspace(0.5 * degree, pi - 0.5 * degree, 91)
    inside = sR.checkInside(vector3d.byPolar(theta, rho), tol=0)
    expected = np.zeros(91, dtype=bool) if iv is None else (iv[0] - 1e-9 <= theta) & (theta <= iv[1] + 1e-9)
    assert np.all(inside == expected)


def test_boundary_arcsLieInTheRegionAndEndAtVertices():
  sR = fundamentalSector(crystalFrame("m-3m"))
  V = sR.vertices()
  arcs = sR.boundary(resolution=2 * degree)
  assert len(arcs) == 3 and all(np.all(sR.checkInside(a, tol=1e-8)) for a in arcs)
  assert all(np.any(angle(V, a[0]) < 1e-8) and np.any(angle(V, a[-1]) < 1e-8) for a in arcs)


def test_sphericalRegion_hemisphereHasOneCircleAsBoundary():
  h = sphericalRegion(zvector)
  assert h.vertices().isempty and len(h.boundary()) == 1 and h.boundary()[0].size > 300
  assert h.thetaInterval(1.0) == (0.0, pi / 2)


def test_sphericalRegion_cap():
  cap = sphericalRegion(zvector, np.cos(30 * degree))
  assert abs(cap.thetaInterval(2.0)[1] - 30 * degree) < 1e-12 and np.all(np.abs(cap.boundary()[0].theta - 30 * degree) < 1e-8)


def test_sphericalRegion_namedRegionsAndRestrict2Upper():
  h = sphericalRegion(zvector)
  assert sphericalRegion("upper") == h and sphericalRegion().checkInside(vector3d.rand(5, rng=np.random.default_rng(5))).all()
  assert sphericalRegion().restrictToUpper() == h and sphericalRegion("lower").restrictToUpper().volume() < 1e-3


def test_sphericalRegion_byVertices():
  assert sphericalRegion.byVertices(vector3d([[0, 0, 1], [1, 0, 0], [0, 1, 0]])).checkInside(vector3d(1, 1, 1))


def test_sphericalRegion_repr():
  assert repr(fundamentalSector(crystalFrame("m-3m"))).startswith("sphericalRegion")


@pytest.mark.parametrize("name", CONVENTIONAL)
def test_conventionalSector_volumeIsOneOverTheOrder(name):
  fr = crystalFrame(name)
  assert abs(conventionalSector(fr).volume(resolution=0.5 * degree) * fr.numSym() - 1) < 0.03


@pytest.mark.parametrize("name", CONVENTIONAL)
def test_projectIntoRegion_landsInTheConventionalSector(name):
  fr = crystalFrame(name)
  sR = conventionalSector(fr)
  vs = vector3d(vector3d.rand(20, rng=np.random.default_rng(len(name))), frame=fr)
  assert np.all(sR.checkInside(projectIntoRegion(vs, sR)))


@pytest.mark.parametrize("name", CONVENTIONAL)
def test_conventionalSector_asAMethodOfTheFrame(name):
  fr = crystalFrame(name)
  assert fr.conventionalSector() == conventionalSector(fr)


samples = vector3d.rand(100, rng=np.random.default_rng(427))
turn = rotation.byAxisAngle(vector3d(1, 2, 3), 37 * degree)


# the triclinic and monoclinic Laue frames, standard and turned against the specimen
def lowFrames():
  for symbol, angles in (("-1", [80, 95, 110]), ("2/m11", [105, 90, 90]), ("12/m1", [90, 110, 90]), ("112/m", [90, 90, 115])):
    for r in (None, turn):
      yield crystalFrame(symbol, [2, 3, 4], np.array(angles) * degree, rotAxes=r, intern=False)


def test_sectors_ofLowSymmetryFramesHaveTheVolume():
  for fr in lowFrames():
    for sector in (conventionalSector(fr), fundamentalSector(fr)):
      assert abs(sector.volume(resolution=2 * degree) * fr.numSym() - 1) < 0.03


def test_projectIntoRegion_ofLowSymmetryFramesLandsInTheSector():
  for fr in lowFrames():
    for sector in (conventionalSector(fr), fundamentalSector(fr)):
      assert np.all(sector.checkInside(projectIntoRegion(vector3d(samples, frame=fr), sector)))


def test_projectIntoFundamentalRegion_ofLowSymmetryFramesLandsInTheSector():
  for fr in lowFrames():
    vs = vector3d(samples, frame=fr)
    assert np.all(fundamentalSector(fr).checkInside(projectIntoFundamentalRegion(vs)))


@pytest.mark.parametrize("name, angles", (("211", [105, 90, 90]), ("121", [90, 110, 90]), ("112", [90, 90, 115]),
                                          ("m11", [105, 90, 90]), ("1m1", [90, 110, 90]), ("11m", [90, 90, 115])))
def test_conventionalSector_ofATurnedMonoclinicFrameTilesTheSphere(name, angles):
  fr = crystalFrame(name, [2, 3, 4], np.array(angles) * degree, rotAxes=turn, intern=False)
  sector = conventionalSector(fr)
  assert abs(sector.volume(resolution=2 * degree) * fr.numSym() - 1) < 0.03
  els = fr.rot
  for v in samples[:20]:
    assert np.sum(sector.checkInside(els * v, tol=1e-9)) == 1


def test_sectorCenter_ofTheCubicTriangle():
  assert sectorCenter(crystalFrame("m-3m")) == vector3d(1, 1, 1).normalize() + vector3d(1, 0, 1).normalize() + zvector


# the conventional sector of a frame's group, or of its Laue group, in the frame's plotting convention
def conventionalOfFrame(CS, antipodal):
  grp = CS.sym.Laue() if antipodal else CS.sym
  return grp, conventionalSector(grp, up=CS.how2plot.outOfScreen, east=CS.how2plot.east)


@pytest.mark.parametrize("antipodal", [False, True])
@pytest.mark.parametrize("s", GROUPS)
def test_conventionalSector_ofACrystalFrameTilesTheSphere(s, antipodal):
  grp, conv = conventionalOfFrame(crystalFrame(s), antipodal)
  assert tilesSphere(conv, grp), s


@pytest.mark.parametrize("antipodal", [False, True])
@pytest.mark.parametrize("s", GROUPS)
def test_fundamentalSector_ofACrystalFrameIsTheConventionalOne(s, antipodal):
  cs = crystalFrame(s)
  _, conv = conventionalOfFrame(cs, antipodal)
  assert fundamentalSector(cs, antipodal=antipodal).N.shape == conv.N.shape and np.allclose(fundamentalSector(cs, antipodal=antipodal).N, conv.N)


def test_projectIntoFundamentalRegion_calciteBelowTheEquatorGoesToMATLABsEquivalent():
  # a calcite direction just below the equator goes to MATLAB's equivalent above it, azimuth -22 degrees,
  # not the one at -38 degrees below it, which a Dirichlet domain about an off-centre point keeps
  cs = crystalFrame("-3m", [5, 5, 17], mineral="calcite")
  p = projectIntoFundamentalRegion(vector3d(1.11272303, -0.86934213, -0.07804949, frame=cs))
  assert np.allclose(p.data, [1.309234, -0.528975, 0.078049], atol=1e-6)


# proper groups, standard and turned against their axes, each with and without the inversion
def turnedGroups():
  r = rotation.byAxisAngle(vector3d(1, 2, 3), 37 * degree).data.reshape(4)
  for s in ("222", "3", "32", "422", "622", "23", "432"):
    q = crystalFrame(s).sym.rot.data.reshape(-1, 5)[:, :4]
    for turned in (False, True):
      pg = symmetry(rotation._fromData(np.array([qmul(qmul(r, x), r * [1, -1, -1, -1]) for x in q]) if turned else q))
      for antipodal in (False, True):
        grp = pg.Laue() if antipodal else pg
        yield pg, grp, turned, antipodal


def test_conventionalSector_ofATurnedGroupDoesNotTile():
  # the rules of the conventional sector read the standard axes, so a turned group falls back
  for _, grp, turned, _ in turnedGroups():
    assert tilesSphere(conventionalSector(grp), grp) != turned


def test_fundamentalSector_ofATurnedGroupIsTheDirichletSector():
  for pg, grp, turned, antipodal in turnedGroups():
    if turned:
      assert fundamentalSector(pg, antipodal=antipodal) == dirichletSector(grp)


def test_projectIntoRegion_landsInTheSectorOfAnyGroup():
  vs = vector3d(np.random.default_rng(5).normal(size=(3000, 3)))
  for pg, grp, _, antipodal in turnedGroups():
    sR = fundamentalSector(pg, antipodal=antipodal)
    assert np.all(sR.checkInside(projectIntoRegion(vs, sR, grp)))


def test_an_icosahedral_group_from_rounded_axes_has_its_sector():
  # the threefold axis at the rounded angle 37.377 degrees closes the group only to about
  # 1e-5, and its sector is still the kite of the proper group, the triangle of the Laue one
  rot5 = rotation.byAxisAngle(zvector, 72 * degree)
  rot3 = rotation.byAxisAngle(vector3d.byPolar(37.377 * degree, 0), 120 * degree)
  for els, faces in ((cat(rot5, rot3), 4), (cat(rot5, rot3, rotation.inversion), 3)):
    cs = crystalFrame.byElements(els)
    sR = conventionalSector(cs)
    assert len(sR.N) == faces and tilesSphere(sR, cs.sym) and sR.checkInside(sectorCenter(cs.sym))
