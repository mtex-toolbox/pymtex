# G7: planes in space, the Julia plane.jl numbers; MATLAB has no tests for them
import numpy as np
import pytest

from mtex import crystalFrame, degree, dist, intersect, plane3d, rotation, specimenFrame, vector3d, zvector
from mtex.geometry.plane import lineIntersect


def test_a_plane_its_distances_and_intersections():
  p = plane3d(vector3d(0, 0, 2), 4)
  assert np.allclose(p.N.data, [0, 0, 1]) and p.d == 2 and p.size == 1 and len(p) == 1 and p.shape == ()
  assert dist(p, vector3d(1, 1, 5)) == 3 and np.allclose(p.project(vector3d(1, 1, 5)).data, [1, 1, 2])
  assert np.allclose(intersect(p, vector3d(0, 0, 0), vector3d(0, 0, 4)).data, [0, 0, 2])
  assert np.all(np.isnan(p.intersect(vector3d(0, 0, 0), vector3d(0, 0, 1)).data))
  assert np.all(np.isnan(lineIntersect([0, 0, 0], [1, 0, 0], [0, 0, 1], 2).data))
  # parallel planes have a distance, meeting ones none
  assert dist(p, plane3d(vector3d(0, 0, 1), 5)) == -3 and np.isnan(dist(p, plane3d(vector3d(1, 0, 0), 5)))
  assert dist(p, plane3d(vector3d(0, 0, -1), -5)) == -3
  # through a point
  q = plane3d(vector3d(1, 1, 1), vector3d(1, 0, 0))
  assert abs(q.dist(vector3d(1, 0, 0))) < 1e-12 and abs(q.dist([0, 1, 0])) < 1e-12 and np.isclose(q.d, 1 / np.sqrt(3))
  # the fit through noisy points of a plane
  rng = np.random.default_rng(1)
  x, y = np.meshgrid(np.arange(-1, 1.01, 0.25), np.arange(-1, 1.01, 0.25))
  pts = vector3d(np.stack([x.ravel(), y.ravel(), 0.5 * x.ravel() - 0.2 * y.ravel() + 1], axis=1) + 1e-3 * rng.standard_normal((x.size, 3)))
  pf = plane3d.byPoints(pts)
  assert pf.N.angle(vector3d(-0.5, 0.2, 1), antipodal=True) < 1e-2 and np.max(np.abs(pf.dist(pts))) < 1e-2
  assert plane3d.fit(pts).d == pf.d
  with pytest.raises(ValueError):
    plane3d.byPoints(vector3d(np.zeros((2, 3))))
  # the polygon cut from the unit box
  c = plane3d(vector3d(1, 1, 1), vector3d(0.5, 0.5, 0.5))
  poly = c.clip([0, 0, 0], [1, 1, 1])
  assert poly.size == 6 and np.max(np.abs(c.dist(poly))) < 1e-12
  assert plane3d(zvector, 0.5).clip([0, 0, 0], [1, 1, 1]).size == 4 and plane3d(zvector, 2).clip([0, 0, 0], [1, 1, 1]).size == 0
  # the intersection of segments given as pairs
  seg = vector3d(np.array([[[0, 0, 0], [0, 0, 4]], [[1, 0, 0], [1, 0, 1]]], dtype=float))
  v = p.intersect(seg)
  assert v.shape == (2,) and np.allclose(v[0].data, [0, 0, 2]) and np.all(np.isnan(v[1].data))


def test_lists_frames_rotation_and_display():
  ss = specimenFrame("222")
  N = vector3d(np.array([[0, 0, 1], [1, 0, 0], [0, 2, 0]], dtype=float), frame=ss)
  P = plane3d(N, [1, 2, 3])
  assert P.size == 3 and P.shape == (3,) and np.allclose(P.d, [1, 2, 1.5]) and P.frame is ss
  assert np.allclose(P.dist(vector3d(1, 1, 1, frame=ss)), [0, -1, -0.5])
  assert P[1].d == 2 and P[1:].size == 2 and len(list(P)) == 3 and P[np.array([True, False, True])].size == 2
  assert plane3d(N).d.shape == (3,) and np.all(plane3d(N).d == 0)
  assert plane3d(P) is not P and plane3d(P).N is P.N
  # a rotation turns the normals and keeps the distances
  r = rotation.byAxisAngle(zvector, 90 * degree)
  R = r * P
  assert np.allclose(R.N[1].data, [0, 1, 0], atol=1e-12) and np.array_equal(R.d, P.d) and R.frame is ss
  assert P.rotate(r).d[0] == 1
  with pytest.raises(ValueError):
    P.clip([0, 0, 0], [1, 1, 1])
  # a lattice plane carries its crystal frame
  cs = crystalFrame("m-3m")
  L = plane3d(vector3d(1, 1, 1, frame=cs), 0.0)
  assert L.frame is cs
  s = repr(P)
  assert s.startswith("plane3d") and "size: 3" in s and "d" in s and repr(plane3d(zvector, 1)).count("\n") == 2


def test_a_plane_drawn_in_space():
  matplotlib = pytest.importorskip("matplotlib")
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d.art3d import Poly3DCollection
  from mtex import crystalShape, gcm, hold, plot
  cs = crystalFrame("m-3m")
  try:
    plot(crystalShape.cube(cs), faceAlpha=0.3)
    hold(True)
    h = plot(plane3d(vector3d(1, 1, 1), 0), faceColor=[0.75, 0.75, 0.75], edgeColor=None)
    hold(False)
    ax = gcm().children[-1]
    assert isinstance(h, Poly3DCollection) and len(h.get_facecolor()) == 7 and ax.get_xlim()[1] < 1
    h = plot(plane3d(vector3d(np.eye(3)), [0.2, 0.3, 0.4]), faceColor=np.eye(3))
    assert len(h.get_facecolor()) == 3
    plane3d(zvector, 0).plot()
  finally:
    hold(False)
    plt.close("all")
