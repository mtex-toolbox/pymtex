# M6: the shape of a grain, the Julia H2 to H10
import numpy as np
import pytest

from mtex import (calcGrains, degree, vector3d, xvector, yvector, shape2d, fitEllipse, longAxis, aspectRatio, caliper, paror, surfor,
                  characteristicShape, hull, paris, shapeFactor, convexity, principalComponents)
from mtex.maps.shape import hullArea, hullPerimeter, outerPerimeter, polygonMoments, _extremalWidths, hullLoops
from conftest import close
from test_grains import mapOf, SQ5


# the grains of an n x n map whose membership is a function of the pixel, and the index of
# the smallest grain
def shapeGrains(n, f):
  g = calcGrains(mapOf(SQ5, n, lambda i, j: f(i, j).astype(int)), angle=5 * degree)
  return g, int(np.argmin(g.numPixel))


def rect():
  return shapeGrains(30, lambda i, j: (5 <= i) & (i < 25) & (10 <= j) & (j < 20))


def unit(v):
  return v.data.reshape(-1) / np.linalg.norm(v.data)


def test_H2_the_ellipse_of_a_rectangle_is_its_closed_form():
  # a w x h rectangle has the same second moments as the ellipse with half axes w / sqrt(3)
  # and h / sqrt(3), whatever MTEX's boundary weighting gives
  g, k = rect()
  a, b = g.principalComponents()
  assert close(g.area[k], 200) and close(a.norm()[k], 20 / np.sqrt(3)) and close(b.norm()[k], 10 / np.sqrt(3))
  assert close(g.aspectRatio()[k], 2) and close(aspectRatio(g)[k], 2)
  assert close(abs(unit(a[k]) @ [1, 0, 0]), 1) and close(abs(unit(b[k]) @ [0, 1, 0]), 1)
  # the long axis is an axis, not a direction: it has no sense, and it is framed
  la = g.longAxis()
  assert la.antipodal and la.frame is g.frame and close(la.norm()[k], 20 / np.sqrt(3)) and longAxis(g).norm()[k] == la.norm()[k]
  assert close(g.shortAxis().norm()[k], 10 / np.sqrt(3))
  c, aa, bb = g.fitEllipse()
  assert close(c[k].data, g.centroid[k].data) and np.array_equal(aa.data, a.data) and np.array_equal(bb.data, b.data)
  assert len(fitEllipse(g)) == 3 and len(principalComponents(g)) == 2
  # the moments of the rectangle about the origin: the polygon runs from cell edge to cell edge
  ar, cx, cy, ixx, iyy, ixy = polygonMoments(g.V, g.loops[k])
  assert close(ar, 200) and close(cx / ar, 14.5) and close(cy / ar, 14.5) and close(iyy / ar - 14.5 ** 2, 400 / 12)


def test_H3_a_square_is_isotropic():
  g, k = shapeGrains(30, lambda i, j: (10 <= i) & (i < 20) & (10 <= j) & (j < 20))
  a, b = g.principalComponents()
  assert close(a.norm()[k], 10 / np.sqrt(3)) and close(b.norm()[k], 10 / np.sqrt(3))
  assert close(g.aspectRatio()[k], 1) and close(g.convexity()[k], 1) and close(g.paris()[k], 0)
  assert close(convexity(g)[k], 1) and close(paris(g)[k], 0)
  # the perimeter against the circle of the same area
  assert close(g.shapeFactor()[k], 40 / (2 * np.pi * np.sqrt(100 / np.pi))) and close(shapeFactor(g), g.shapeFactor())


def test_H4_the_caliper_measures_the_width():
  g, k = rect()
  assert close(g.caliper(0.0)[k], 20) and close(g.caliper(np.pi / 2)[k], 10)       # across x, across y
  assert close(g.caliper(xvector)[k], 20) and close(caliper(g, yvector)[k], 10)
  lo, hi = g.caliper("shortest").norm(), g.caliper().norm()
  assert close(lo[k], 10) and close(hi[k], np.sqrt(20 ** 2 + 10 ** 2))              # the narrowest support, the longest diagonal
  assert lo[k] <= g.caliper(0.3)[k] <= hi[k] and np.array_equal(g.caliper("longest").data, g.caliper().data)
  # the extremal widths as axes: the narrowest across y, the widest along the diagonal, and
  # the width across the normal of the narrowest one, which is the length of the rectangle
  short, long, perp = g.caliper("shortest"), g.caliper("longest"), g.caliper("shortestPerp")
  assert short.antipodal and close(abs(unit(short[k]) @ [0, 1, 0]), 1) and close(abs(unit(long[k]) @ unit(vector3d(2, 1, 0))), 1)
  assert close(abs(unit(perp[k]) @ [1, 0, 0]), 1) and close(perp.norm()[k], 20)
  with pytest.raises(ValueError):
    g.caliper("widest")


def test_H5_a_non_convex_grain_is_indented_against_its_hull():
  # an L shape: a 20 x 20 block with its upper right quarter given away
  isL = lambda i, j: (5 <= i) & (i < 25) & (5 <= j) & (j < 25) & ~((15 <= i) & (i < 25) & (15 <= j) & (j < 25))
  g = calcGrains(mapOf(SQ5, 30, lambda i, j: isL(i, j).astype(int)), angle=5 * degree)
  k = int(np.flatnonzero(g.numPixel == 300)[0])
  assert close(g.area[k], 300) and close(hullArea(g)[k], 400 - 50)                  # the hull cuts the notch
  assert close(g.convexity()[k], 300 / 350) and g.paris()[k] > 0 and outerPerimeter(g)[k] > hullPerimeter(g)[k]
  assert close(g.paris()[k], 200 * (outerPerimeter(g)[k] - hullPerimeter(g)[k]) / hullPerimeter(g)[k])
  assert len(g.hullLoops()[k]) == 5                                                 # the L hulls to a pentagon
  # the hulled grains: convex polygons on a boundary with nothing on the other side
  h = g.hull()
  assert close(h.area[k], 350) and close(h.area, hullArea(g)) and np.all(h.convexity() > 1 - 1e-12) and np.array_equal(h.id, g.id)
  assert h.boundary.grainId.min() == 0 and h.innerBoundary.isempty and np.all(h.boundary.grainId.max(axis=1) > 0)
  assert close(h.perimeter, hullPerimeter(g)) and close(hull(g).area, h.area)


def test_H6_a_hole_is_taken_off_every_moment():
  # a square grain with a square hole: the moments of the ring, not of the disc
  outer = lambda i, j: (5 <= i) & (i < 25) & (5 <= j) & (j < 25)
  inner = lambda i, j: (12 <= i) & (i < 18) & (12 <= j) & (j < 18)
  g = calcGrains(mapOf(SQ5, 30, lambda i, j: np.where(inner(i, j), 2, np.where(outer(i, j), 1, 0))), angle=5 * degree)
  k = int(np.flatnonzero(g.numPixel == 400 - 36)[0])
  assert g.hasHole[k] and close(g.area[k], 400 - 36)
  # the ring is centred where the square is, and stays isotropic
  assert close(g.centroid[k].data, [14.5, 14.5, 0]) and close(g.aspectRatio()[k], 1, 1e-9)
  # its hull is the outer square, so the hole makes it inconvex by area
  assert close(hullArea(g)[k], 400) and close(g.convexity()[k], (400 - 36) / 400)


def test_H7_the_projection_function_of_a_rectangle():
  # the width of a w x h rectangle seen from omega is w|cos| + h|sin|, and the projection
  # function is that normalised by its largest value
  g, k = rect()
  omega = np.linspace(0, np.pi, 181)
  p = g[[k]].paror(omega)
  ref = 20 * np.abs(np.cos(omega)) + 10 * np.abs(np.sin(omega))
  assert close(p, ref / ref.max()) and close(p[0] / p[90], 2) and close(p.max(), 1)
  assert len(paror(g)) == 181


def test_H8_the_projection_function_of_its_boundary():
  # 40 units of boundary run along x and 20 along y, so the segments project to 40|cos| + 20|sin|
  g, k = rect()
  omega = np.linspace(0, np.pi, 181)
  l = g[[k]].surfor(omega)
  ref = 40 * np.abs(np.cos(omega)) + 20 * np.abs(np.sin(omega))
  assert close(l, ref / ref.max()) and close(l[0] / l[90], 2)
  assert np.array_equal(g[[k]].boundary.surfor(omega), l) and np.array_equal(surfor(g[[k]], omega), l)


def test_H9_the_characteristic_shape_of_a_rectangle():
  # every segment and its antipode laid end to end by direction: 40 units of boundary along
  # x and 20 along y give the 40 x 20 rectangle
  g, k = rect()
  cs = g[[k]].boundary.characteristicShape()
  assert isinstance(cs, shape2d) and len(cs) == 1
  assert close(cs.area[0], 800) and close(cs.aspectRatio()[0], 2) and close(cs.centroid[0].data, 0, 1e-9) and close(cs.perimeter[0], 120)
  # it is convex and centrally symmetric, whatever the outline was
  assert close(cs.convexity()[0], 1) and close(abs(unit(cs.longAxis()[0]) @ [1, 0, 0]), 1, 1e-9)
  assert close(characteristicShape(g[[k]]).area[0], 800) and abs(g.characteristicShape(nmax=64).area[0] / characteristicShape(g).area[0] - 1) < 0.01
  with pytest.raises(ValueError):
    g.innerBoundary.characteristicShape()


def test_H10_an_outline_of_its_own():
  # a shape given in polar coordinates answers every parameter a grain does
  theta = np.linspace(0, 2 * np.pi, 721)[:-1]
  circle = shape2d.byRhoTheta(np.full(len(theta), 3.0), theta)
  assert abs(circle.area[0] - 9 * np.pi) < 1e-4 * 9 * np.pi and close(circle.aspectRatio()[0], 1, 1e-6)
  assert abs(circle.equivalentRadius[0] - 3) < 1e-3 and abs(circle.shapeFactor()[0] - 1) < 1e-4
  assert close(circle.rho, 3) and close(circle.theta, theta - 2 * np.pi * (theta > np.pi)) and circle.Vs.frame is circle.frame
  # a circle projects to its diameter in every direction
  p = circle.paror()
  assert p.max() - p.min() < 1e-4 and close(circle.caliper(0.7)[0], 6, 1e-3) and close(circle.hull().area, circle.area, 1e-9)
  # the winding is fixed on construction, so the area is the positive one
  P = np.stack([np.cos(theta), np.sin(theta)], axis=1)
  assert shape2d(P[::-1]).area[0] > 0 and shape2d(vector3d(P[:, 0], P[:, 1], 0)).area[0] > 0
  assert "vertices" in repr(circle) and circle[0] is circle and circle.checkInside([0, 0]).all() and not circle.checkInside([4, 0]).any()
  with pytest.raises(IndexError):
    circle[[0, 0]]


def test_the_extremal_widths_over_all_grains_are_those_of_each():
  # random grains from Voronoi seeds, their widths against the search edge by edge and
  # pair by pair of one hull at a time
  rng = np.random.default_rng(7)
  seeds = rng.uniform(0, 40, (30, 2))
  i, j = np.meshgrid(np.arange(40), np.arange(40), indexing="xy")
  P = np.stack([i.reshape(-1), j.reshape(-1)], 1).astype(float)
  lab = np.argmin(np.linalg.norm(P[:, None, :] - seeds[None, :, :], axis=-1), axis=1)
  g = calcGrains(mapOf(SQ5, 40, lambda i, j: lab), angle=5 * degree)
  short, long = _extremalWidths(g)
  for k, h in enumerate(hullLoops(g)):
    H = g.V[h]
    E = np.roll(H, -1, axis=0) - H
    n = np.stack([-E[:, 1], E[:, 0]], axis=1) / np.linalg.norm(E, axis=1)[:, None]
    w = (H @ n.T).max(axis=0) - (H @ n.T).min(axis=0)
    assert close(np.linalg.norm(short[k]), w.min(), 1e-9)
    d = np.linalg.norm(H[:, None, :] - H[None, :, :], axis=-1)
    assert close(np.linalg.norm(long[k]), d.max(), 1e-9)
    assert close(abs(short[k] @ n[np.argmin(w)]), w.min(), 1e-9)
