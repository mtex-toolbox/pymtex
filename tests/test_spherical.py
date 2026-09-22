# the geometry of spherical plots (docs/plans/plotting.md S1): projections and their
# inverses, the charts a region needs, the conforming grid, boundary and labels; the numbers
# of ../mtexJulia/test/spherical.jl P1 to P4
import numpy as np
import pytest
from numpy import pi

from mtex import (S2FunHarmonic, conventionalSector, crystalFrame, degree, fundamentalSector, plottingConvention,
                  rotation, specimenFrame, sphericalRegion, vector3d, xvector, zvector)
from mtex.plotting.spherical import (eangleProjection, eareaProjection, edistProjection, gnomonicProjection, isLower, isUpper,
                                     makeSphericalProjection, millerLabel, orthographicProjection, plainProjection, plotRegion,
                                     projectionClass, rhoRange, sphericalProjection)

rng = np.random.default_rng(91)


def sameDirection(u, v, tol=1e-10):
  a, b = u.normalize().data.reshape(3), v.normalize().data.reshape(3)
  return np.linalg.norm(a - b) < tol


def test_P1_projections_and_their_inverses():
  for cls in (eareaProjection, eangleProjection, edistProjection, orthographicProjection, gnomonicProjection):
    for theta in (0.0, 0.3, 1.0, 1.5):
      assert abs(cls.polarAngle(cls.radius(theta)) - theta) < 1e-12
    assert cls.radius(0.0) == 0
  # the equal area projection preserves area: a cap of angle theta has the area of the disc
  theta = 40 * degree
  assert np.isclose(pi * eareaProjection.radius(theta) ** 2, 2 * pi * (1 - np.cos(theta)))
  # the stereographic projection is conformal: small circles stay circles
  c = makeSphericalProjection(sphericalRegion(), projection="stereo")[0]
  n = vector3d(0.3, 0.4, 0.5).normalize()
  o = n.orth()
  pts = [c.projectUnchecked(rotation.byAxisAngle(n, t) * o * 0.2 + n * np.sqrt(1 - 0.04)) for t in np.linspace(0, 2 * pi, 12)]
  A = np.array([[x, y, 1.0] for x, y in pts])
  d, e, f = np.linalg.lstsq(A, [-(x ** 2 + y ** 2) for x, y in pts], rcond=None)[0]
  rs = [np.hypot(x + d / 2, y + e / 2) for x, y in pts]
  assert max(rs) - min(rs) < 1e-6 * np.mean(rs)
  assert projectionClass("earea") is eareaProjection and projectionClass("eangle") is eangleProjection
  assert projectionClass("schmidt") is eareaProjection and projectionClass("plain") is plainProjection
  with pytest.raises(ValueError):
    projectionClass("mercator")
  assert isinstance(makeSphericalProjection(projection="stereo")[0], eangleProjection)


def test_P2_charts_and_the_hemispheres():
  ss = specimenFrame("222")
  pC = ss.how2plot
  up = pC.outOfScreen
  sphere = sphericalRegion(frame=ss)
  cl = makeSphericalProjection(sphere)
  assert len(cl) == 2 and cl[0].upper and not cl[1].upper and not cl[0].antipodal
  assert cl[0].frame is ss and cl[0].pC == pC
  # every direction lands on exactly one chart, and comes back from it
  for v in vector3d.rand(20, rng=rng):
    xy = [c.project(v) for c in cl]
    hits = [not np.isnan(x) for x, _ in xy]
    assert sum(hits) == 1
    k = hits.index(True)
    assert sameDirection(cl[k].iproject(*xy[k]), v)
  # the upper chart holds the directions out of the screen
  v = vector3d(0.3, 0.2, 0.9)
  assert not np.isnan(cl[0 if v.dot(up) > 0 else 1].project(v)[0])
  # one chart for a hemisphere or an antipodal region, both on request
  assert len(makeSphericalProjection(sphere, upper=True)) == 1
  assert len(makeSphericalProjection(sphere, lower=True)) == 1 and not makeSphericalProjection(sphere, lower=True)[0].upper
  assert len(makeSphericalProjection(sphere, antipodal=True)) == 1
  assert len(makeSphericalProjection(sphere, antipodal=True, complete=True)) == 2
  ca = makeSphericalProjection(sphere, antipodal=True)[0]
  assert np.allclose(ca.project(v), ca.project(-v))
  assert len(makeSphericalProjection(sphere.restrict2Upper(pC))) == 1
  assert isUpper(sphere.restrict2Upper(pC), pC) and not isLower(sphere.restrict2Upper(pC), pC)
  assert not isUpper(sphere, pC) and not isLower(sphere, pC)
  assert len(makeSphericalProjection(sphere, projection="plain")) == 1
  with pytest.raises(ValueError):
    makeSphericalProjection(sphere, upper=True, lower=True)
  # a list of axes is drawn on one chart, a list of directions on two
  assert len(makeSphericalProjection(vector3d.rand(5, antipodal=True, rng=rng))) == 1
  assert len(makeSphericalProjection(vector3d.rand(5, rng=rng))) == 2
  # a sector of a crystal frame is one chart, and its region is the sector
  cs = crystalFrame("m-3m")
  sR = conventionalSector(cs)
  cc = makeSphericalProjection(sR)
  assert len(cc) == 1 and np.allclose(cc[0].sR.N, sR.restrict2Upper(cs.how2plot).N)
  # a symmetric function is drawn on its sector, a frame-free one on the sphere
  f = S2FunHarmonic(rng.standard_normal(16) + 1j * rng.standard_normal(16), cs, real=True)
  assert np.allclose(makeSphericalProjection(f)[0].sR.N, sR.restrict2Upper(cs.how2plot).N)
  g = S2FunHarmonic(rng.standard_normal(16) + 1j * rng.standard_normal(16), real=True)
  assert len(makeSphericalProjection(g)) == 2 and len(makeSphericalProjection(g)[0].sR.N) == 1
  assert np.allclose(plotRegion(cs, fundamentalRegion=True).N, sR.N)
  assert len(plotRegion(cs).N) == 0 and len(plotRegion(None, fundamentalRegion=True).N) == 0
  vs = vector3d(vector3d.rand(3, rng=rng), frame=cs)
  assert np.allclose(makeSphericalProjection(vs, fundamentalRegion=True)[0].sR.N, cc[0].sR.N)


def test_P3_the_grid_the_boundary_and_the_labels():
  cs = crystalFrame("m-3m")
  sR = conventionalSector(cs)
  c = makeSphericalProjection(sR)[0]
  # the azimuth range of the cubic sector is the 45 degree wedge
  lo, hi = rhoRange(c.chartRegion())
  assert abs(hi - lo - 45 * degree) < 1e-6
  assert rhoRange(sphericalRegion()) == (0.0, 2 * pi)
  assert rhoRange(sphericalRegion().restrict2Upper(zvector)) == (0.0, 2 * pi)
  # the grid lies inside the sector, its first and last rows on the boundary
  V, X, Y = c.grid(2 * degree)
  assert V.shape[1] >= 20 and V.shape[0] % 2 == 1 and V.frame is cs
  assert np.all(sR.checkInside(V))
  assert np.all(np.isfinite(X)) and np.all(np.isfinite(Y)) and X.shape == V.shape
  assert np.allclose(np.stack(c.projectUnchecked(V)), np.stack((X, Y)))
  # the boundary of the sector: three arcs, all in the plane
  b = c.boundary()
  assert len(b) == 3 and all(np.all(np.isfinite(x)) for x, _ in b)
  # the corners are labelled with Miller indices
  assert {s for _, s in c.labels()} == {"[001]", "[101]", "[111]"}
  assert millerLabel(vector3d(1, 1, 0), cs) == "[110]" and millerLabel(vector3d(-1, 0, 0), cs, utf8=False) == "[-100]"
  # a specimen frame labels its axes where they are visible
  ss = specimenFrame("1")
  cl = makeSphericalProjection(sphericalRegion(frame=ss))
  assert {s for _, s in cl[0].labels()} <= {"X", "Y", "Z"} and len(cl[0].labels()) + len(cl[1].labels()) >= 3
  named = specimenFrame.rolling
  assert {s for _, s in makeSphericalProjection(sphericalRegion(frame=named))[0].labels()} <= set(named.axesNames)
  # the whole hemisphere: the horizon is its boundary, the grid closes around
  ch = cl[0]
  assert len(ch.boundary()) == 1
  V, X, Y = ch.grid(10 * degree)
  assert V.shape[1] == 37 and sameDirection(V[-1, 0], V[-1, -1])
  assert abs(np.max(np.hypot(X, Y)) - eareaProjection.radius(pi / 2)) < 1e-6
  xlo, xhi, ylo, yhi = ch.bounds()
  assert xlo < -1.4 and xhi > 1.4 and ylo < -1.4 and yhi > 1.4
  # grid lines are clipped to the chart
  gl = ch.gridLines()
  assert len(gl) == 11 and np.any(np.isnan(gl[-1][0]))
  # the plain projection: a rectangle in degrees
  cp = makeSphericalProjection(sphericalRegion(), projection="plain")[0]
  assert np.allclose(cp.project(vector3d.byPolar(90 * degree, 45 * degree)), (45.0, 90.0))
  assert sameDirection(cp.iproject(45.0, 90.0), vector3d.byPolar(90 * degree, 45 * degree))
  V, X, Y = cp.grid(10 * degree)
  assert V.shape == (19, 37) and np.isclose(X[0, -1], 360) and np.isclose(Y[-1, 0], 180)
  assert cp.bounds() == (0.0, 360.0, 0.0, 180.0)
  # the sector of a point group that is not Laue, for axes
  a = vector3d(vector3d.rand(3, antipodal=True, rng=rng), frame=crystalFrame("432"))
  pC = a.how2plot
  expected = conventionalSector(crystalFrame("432").sym.Laue(), up=pC.outOfScreen, east=pC.east)
  assert np.allclose(makeSphericalProjection(a, fundamentalRegion=True)[0].sR.N[1:], expected.N) or \
    np.allclose(makeSphericalProjection(a, fundamentalRegion=True)[0].sR.N, expected.N)
  # the trigonal sector reaches into both hemispheres and gets two charts
  assert len(makeSphericalProjection(fundamentalSector(crystalFrame("-3m")))) == 2
  # a chart says what it is
  assert "earea" in repr(c) and "upper" in repr(c)


def test_P4_a_per_plot_convention_selects_a_drawable_sector():
  cs = crystalFrame("m-3m")
  f = S2FunHarmonic(rng.standard_normal(36) + 1j * rng.standard_normal(36), cs, real=True)
  original = cs.how2plot
  for pc in (plottingConvention("y↑→x"), plottingConvention("y↓→x")):
    region = plotRegion(cs, fundamentalRegion=True, how2plot=pc)
    charts = makeSphericalProjection(region, how2plot=pc, antipodal=True)
    assert len(charts) == 1 and charts[0].pC == pc
    V, X, Y = charts[0].grid(10 * degree)
    assert V.size > 20 and np.all(np.isfinite(X)) and np.all(np.isfinite(Y))
    # the chart is antipodal, so the grid lies in the sector or in its antipode
    assert np.all(region.checkInside(V, tolerance=1e-8) | region.checkInside(-V, tolerance=1e-8))
  assert cs.how2plot == original
  # the projection of a direction respects the convention: x east means x at (r, 0)
  c = makeSphericalProjection(sphericalRegion(), how2plot=plottingConvention("y↑→x"))[0]
  x, y = c.project(xvector)
  assert x > 0 and abs(y) < 1e-12
  # a projection made from a frame carries its convention
  p = sphericalProjection(sphericalRegion(frame=specimenFrame.rolling))
  assert p.pC == specimenFrame.rolling.how2plot
