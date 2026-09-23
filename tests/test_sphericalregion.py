# spherical regions, fundamental and conventional sectors
import numpy as np
import pytest
from numpy import pi

from mtex import (sphericalRegion, fundamentalSector, conventionalSector, projectIntoRegion, project2FundamentalRegion,
                  crystalFrame, symmetry, rotation, vector3d, xvector, zvector, Miller, degree, angle, symmetrise, unique)
from mtex.geometry.sphericalregion import sectorCenter

rng = np.random.default_rng(5)


@pytest.mark.parametrize("s", ["1", "-1", "2/m", "mmm", "4/m", "4/mmm", "-3", "-3m", "6/m", "6/mmm", "m-3", "m-3m", "321", "-42m", "23"])
def test_fundamental_sectors(s):
  cs = crystalFrame(s)
  sR = fundamentalSector(cs)
  vs = vector3d(vector3d.rand(2000, rng=rng), frame=cs)
  p = project2FundamentalRegion(vs)
  assert np.all(angle(p[:200], vs[:200]) < 1e-10)
  assert np.all(sR.checkInside(p))
  assert abs(sR.volume() * cs.numSym() - 1) < 0.05
  assert np.max(angle(p, project2FundamentalRegion(p), noSymmetry=True)) < 1e-10
  assert sR.frame is cs and cs.fundamentalSector() == sR


def test_the_standard_triangle_and_the_wedge():
  cs = crystalFrame("m-3m")
  sR = fundamentalSector(cs)
  assert len(sR.N) == 3
  for v in ([0, 0, 1], [1, 0, 1], [1, 1, 1]):
    assert sR.checkInside(vector3d(*v).normalize())
  assert not sR.checkInside(vector3d(-1, 0, 0)) and not sR.checkInside(vector3d(1, 1, 0.5).normalize())
  assert sR.checkInside(vector3d([[0, 0, 1], [-1, 0, 0]])).tolist() == [True, False]
  cH = crystalFrame("6/mmm", [3, 3, 5])
  sH = fundamentalSector(cH)
  assert len(sH.N) == 3 and sH.checkInside(zvector)
  assert unique(symmetrise(Miller(1, 2, 3, cs))).size == 1
  pg = symmetry(rotation.byAxisAngle(zvector, 2 * pi / 5))
  assert abs(fundamentalSector(pg).volume() * 5 - 1) < 0.05
  # an axis counts on both sides, and the Laue sibling halves the sector
  a = vector3d(0, 0, -1, antipodal=True)
  assert sR.checkInside(a) and not sR.checkInside(vector3d(0, 0, -1))
  # an antipodal region counts both ends, so the sector of the Laue group covers 2/48 of the sphere
  assert abs(fundamentalSector(crystalFrame("432"), antipodal=True).volume() * 24 - 1) < 0.05
  assert fundamentalSector(crystalFrame("-1", [2, 3, 4], np.array([80, 95, 110]) * degree), center=xvector).N.shape == (1, 3)
  assert np.allclose(fundamentalSector(crystalFrame("-1", [2, 3, 4], np.array([80, 95, 110]) * degree), center=xvector).N[0], [1, 0, 0])


def test_the_shape_of_a_region():
  cs = crystalFrame("m-3m")
  sR = fundamentalSector(cs)
  V = sR.vertices()
  assert V.size == 3
  for v in ([0, 0, 1], [1, 0, 1], [1, 1, 1]):
    assert np.any(angle(V, vector3d(*v)) < 1e-8)
  for rho in 2 * pi * rng.random(20):
    iv = sR.thetaInterval(rho)
    # off the poles: the pole is a corner of the triangle, a single point of every meridian
    theta = np.linspace(0.5 * degree, pi - 0.5 * degree, 91)
    inside = sR.checkInside(vector3d.byPolar(theta, rho), tolerance=0)
    expected = np.zeros(91, dtype=bool) if iv is None else (iv[0] - 1e-9 <= theta) & (theta <= iv[1] + 1e-9)
    assert np.all(inside == expected)
  arcs = sR.boundary(resolution=2 * degree)
  assert len(arcs) == 3 and all(np.all(sR.checkInside(a, tolerance=1e-8)) for a in arcs)
  assert all(np.any(angle(V, a[0]) < 1e-8) and np.any(angle(V, a[-1]) < 1e-8) for a in arcs)
  h = sphericalRegion(zvector)
  assert h.vertices().isempty and len(h.boundary()) == 1 and h.boundary()[0].size > 300
  assert h.thetaInterval(1.0) == (0.0, pi / 2)
  cap = sphericalRegion(zvector, np.cos(30 * degree))
  assert abs(cap.thetaInterval(2.0)[1] - 30 * degree) < 1e-12 and np.all(np.abs(cap.boundary()[0].theta - 30 * degree) < 1e-8)
  assert sphericalRegion("upper") == h and sphericalRegion().checkInside(vector3d.rand(5, rng=rng)).all()
  assert sphericalRegion().restrict2Upper() == h and sphericalRegion("lower").restrict2Upper().volume() < 1e-3
  assert sphericalRegion.byVertices(vector3d([[0, 0, 1], [1, 0, 0], [0, 1, 0]])).checkInside(vector3d(1, 1, 1))
  assert repr(sR).startswith("sphericalRegion")


@pytest.mark.parametrize("name", ["1", "-1", "2", "m", "2/m", "222", "mm2", "mmm", "4", "-4", "4/m", "422", "4mm", "-42m", "4/mmm",
  "3", "-3", "32", "3m", "-3m", "6", "-6", "6/m", "622", "6mm", "-6m2", "6/mmm", "23", "m-3", "432", "-43m", "m-3m"])
def test_the_conventional_sectors(name):
  fr = crystalFrame(name)
  sR = conventionalSector(fr)
  assert abs(sR.volume(resolution=0.5 * degree) * fr.numSym() - 1) < 0.03
  vs = vector3d(vector3d.rand(20, rng=np.random.default_rng(len(name))), frame=fr)
  assert np.all(sR.checkInside(projectIntoRegion(vs, sR)))
  assert fr.conventionalSector() == sR


def test_crystal_sectors_follow_the_plotting_convention():
  samples = vector3d.rand(100, rng=np.random.default_rng(427))
  turn = rotation.byAxisAngle(vector3d(1, 2, 3), 37 * degree)
  for symbol, angles in (("-1", [80, 95, 110]), ("2/m11", [105, 90, 90]), ("12/m1", [90, 110, 90]), ("112/m", [90, 90, 115])):
    for r in (None, turn):
      fr = crystalFrame(symbol, [2, 3, 4], np.array(angles) * degree, rotAxes=r, noIntern=True)
      for sector in (conventionalSector(fr), fundamentalSector(fr)):
        assert abs(sector.volume(resolution=2 * degree) * fr.numSym() - 1) < 0.03
        assert np.all(sector.checkInside(projectIntoRegion(vector3d(samples, frame=fr), sector)))
      vs = vector3d(samples, frame=fr)
      assert np.all(fundamentalSector(fr).checkInside(project2FundamentalRegion(vs)))
  for name, angles in (("211", [105, 90, 90]), ("121", [90, 110, 90]), ("112", [90, 90, 115]), ("m11", [105, 90, 90]), ("1m1", [90, 110, 90]), ("11m", [90, 90, 115])):
    fr = crystalFrame(name, [2, 3, 4], np.array(angles) * degree, rotAxes=turn, noIntern=True)
    sector = conventionalSector(fr)
    assert abs(sector.volume(resolution=2 * degree) * fr.numSym() - 1) < 0.03
    els = fr.rot
    for v in samples[:20]:
      assert np.sum(sector.checkInside(els * v, tolerance=1e-9)) == 1
  assert sectorCenter(crystalFrame("m-3m")) == vector3d(1, 1, 1).normalize() + vector3d(1, 0, 1).normalize() + zvector


GROUPS = ["1", "-1", "2", "m", "2/m", "222", "mm2", "mmm", "4", "-4", "4/m", "422", "4mm", "-42m", "-4m2", "4/mmm", "3", "-3", "321",
          "312", "3m1", "31m", "-31m", "-3m1", "6", "-6", "6/m", "622", "6mm", "-6m2", "-62m", "6/mmm", "23", "m-3", "432", "-43m", "m-3m"]


def test_the_fundamental_sector_of_a_crystal_frame_is_the_conventional_one():
  from mtex.geometry.sphericalregion import tilesSphere
  for s in GROUPS:
    cs = crystalFrame(s)
    for antipodal in (False, True):
      grp = cs.sym.Laue() if antipodal else cs.sym
      conv = conventionalSector(grp, up=cs.how2plot.outOfScreen, east=cs.how2plot.east)
      assert tilesSphere(conv, grp), s
      assert fundamentalSector(cs, antipodal=antipodal).N.shape == conv.N.shape and np.allclose(fundamentalSector(cs, antipodal=antipodal).N, conv.N)
  # a calcite direction just below the equator goes to MATLAB's equivalent above it, azimuth -22 degrees,
  # not the one at -38 degrees below it, which a Dirichlet domain about an off-centre point keeps
  cs = crystalFrame("-3m", [5, 5, 17], mineral="calcite")
  p = project2FundamentalRegion(vector3d(1.11272303, -0.86934213, -0.07804949, frame=cs))
  assert np.allclose(p.data, [1.309234, -0.528975, 0.078049], atol=1e-6)


def test_a_turned_group_takes_the_dirichlet_domain():
  from mtex.geometry.kernels import qmul
  from mtex.geometry.sphericalregion import dirichletSector, tilesSphere
  r = rotation.byAxisAngle(vector3d(1, 2, 3), 37 * degree).data.reshape(4)
  vs = vector3d(rng.normal(size=(3000, 3)))
  for s in ("222", "3", "32", "422", "622", "23", "432"):
    q = crystalFrame(s).sym.rot.data.reshape(-1, 5)[:, :4]
    for turned in (False, True):
      pg = symmetry(rotation._fromData(np.array([qmul(qmul(r, x), r * [1, -1, -1, -1]) for x in q]) if turned else q))
      for antipodal in (False, True):
        grp = pg.Laue() if antipodal else pg
        sR = fundamentalSector(pg, antipodal=antipodal)
        # the rules of the conventional sector read the standard axes, so a turned group falls back
        assert tilesSphere(conventionalSector(grp), grp) != turned
        if turned:
          assert sR == dirichletSector(grp)
        assert np.all(sR.checkInside(projectIntoRegion(vs, sR, grp)))


def test_an_icosahedral_group_from_rounded_axes_has_its_sector():
  # the threefold axis at the rounded angle 37.377 degrees closes the group only to about
  # 1e-5, and its sector is still the kite of the proper group, the triangle of the Laue one
  from mtex import cat
  rot5 = rotation.byAxisAngle(zvector, 72 * degree)
  rot3 = rotation.byAxisAngle(vector3d.byPolar(37.377 * degree, 0), 120 * degree)
  for els, faces in ((cat(rot5, rot3), 4), (cat(rot5, rot3, rotation.inversion), 3)):
    cs = crystalFrame.byElements(els)
    from mtex.geometry.sphericalregion import tilesSphere
    sR = conventionalSector(cs)
    assert len(sR.N) == faces and tilesSphere(sR, cs.sym) and sR.checkInside(sectorCenter(cs.sym))
