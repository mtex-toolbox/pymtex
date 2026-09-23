# the drawing of plots in space (docs/plans/plotting.md S5), on the Agg backend
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection  # noqa: E402

from mtex import (S2AxisFieldHarmonic, S2FunHarmonic, arrow3d, crystalFrame, hold, mtexColorMap, mtexColorbar, degree, discreteSample, fundamentalRegion, gcm, grad, orientation,
                  plot, plot3d, plottingConvention, quiver3, rotation, scatter, scatter3d, specimenFrame, unimodalODF, vector3d, zvector)

rng = np.random.default_rng(9)


@pytest.fixture(autouse=True)
def closeFigures():
  yield
  plt.close("all")


def axesOf():
  return gcm().children[-1]


def test_directions_functions_and_fields_on_the_sphere():
  ss = specimenFrame("1")
  vs = vector3d(vector3d.rand(30, rng=rng), frame=ss)
  h = plot3d(vs)
  ax = axesOf()
  assert ax.name == "3d" and hasattr(ax, "_mtexSpacePlot") and len(h.get_offsets()) == 30
  # the sphere, the grid and the axes labels are there
  assert any(isinstance(c, Poly3DCollection) for c in ax.collections) and len(ax.lines) == 17
  assert {t.get_text() for t in ax.texts} == {"X", "Y", "Z"}
  # axes are drawn at both ends, values through the colour map with a bar
  h = scatter3d(vector3d.rand(10, antipodal=True, rng=rng), rng.random(10), colorbar=True)
  assert len(h.get_offsets()) == 20 and len(gcm().fig.axes) == 2
  vs.plot3d(label=[str(k) for k in range(30)], noLabel=True)
  assert len(axesOf().texts) == 30
  # a function colours the sphere at its faces, radial pushes it out
  s = S2FunHarmonic(rng.standard_normal(49) + 1j * rng.standard_normal(49), real=True) + 3
  h = plot3d(s, colorbar=True)
  assert isinstance(h, Poly3DCollection) and h.get_array().size == len(h.get_paths()) if hasattr(h, "get_paths") else True
  h = s.plot3d(radial=True, resolution=10 * degree)
  assert isinstance(h, Poly3DCollection) and axesOf().get_xlim()[1] > 1.3
  # a field as arrows, an axis field as segments
  h = plot3d(grad(s))
  assert isinstance(h, Line3DCollection)
  A = S2AxisFieldHarmonic(lambda v: vector3d(v.cross(zvector).data, antipodal=True), bandwidth=12)
  h = A.plot3d()
  assert isinstance(h, Line3DCollection)
  # quiver3 of a field on the sphere draws onto a held sphere without a second one
  plot3d(s)
  n = len(axesOf().collections)
  hold(True)
  h = quiver3(A, color="white")
  hold(False)
  assert isinstance(h, Line3DCollection) and len(axesOf().collections) == n + 1


def test_orientation_space():
  cs, ss = crystalFrame("m-3m"), specimenFrame("222")
  oR = fundamentalRegion(cs, ss)
  h = plot(oR)
  ax = axesOf()
  assert isinstance(h, Poly3DCollection) and ax.get_xlabel() == "x / °"
  # the axes are fitted to the region
  P = np.concatenate(oR.edges()) / degree
  assert ax.get_xlim()[1] >= P[:, 0].max() and ax.get_xlim()[0] <= P[:, 0].min() and len(ax.lines) > 10
  # without the parallels and meridians only the edges of the region are lines; without the
  # surface only they are drawn
  oR.plot(lines=False)
  assert len(axesOf().lines) == len(oR.edges()) > 0
  hs = plot(oR, noSurface=True, boundaryColor='r')
  assert len(hs) == len(oR.edges()) and not axesOf().collections
  # a misorientation region has walls from the identity
  mR = fundamentalRegion(cs, cs)
  plot(mR)
  assert len([c for c in axesOf().collections if isinstance(c, Poly3DCollection)]) > 1
  # orientations as points in the region, in degrees, coloured by a value
  odf = unimodalODF(orientation.byEuler(30 * degree, 50 * degree, 10 * degree, cs, ss), halfwidth=10 * degree)
  ori = discreteSample(odf, 200, rng=rng)
  # an orientation scatters in the Euler region by default, as MATLAB does; axis angle on request
  h = scatter(ori, axisAngle=True)
  ax = axesOf()
  P = np.stack([np.asarray(a) for a in h._offsets3d], axis=1)
  assert P.shape == (200, 3) and np.all(np.linalg.norm(P, axis=1) <= 63) and ax.get_xlabel() == "x / °"
  assert any(isinstance(c, Poly3DCollection) for c in ax.collections)
  h = ori.scatter(np.degrees(ori.angle()), colorbar=True, region=False, points=50)
  assert len(h._offsets3d[0]) == 50 and not any(isinstance(c, Poly3DCollection) for c in axesOf().collections)
  # Euler and Rodrigues coordinates
  h = scatter(ori, Euler=True)
  ax = axesOf()
  P = np.stack([np.asarray(a) for a in h._offsets3d], axis=1)
  assert ax.get_xlabel() == "φ1 / °" and np.all(P[:, 1] <= 90 + 1e-6) and np.all(P[:, 2] <= 90 + 1e-6)
  h = scatter(ori, Rodrigues=True)
  assert axesOf().get_xlabel() == "r1"
  # the representatives as stored, and a single orientation
  h = scatter(ori, ignoreFundamentalRegion=True)
  assert len(h._offsets3d[0]) == 200
  plot3d(ori[0])
  assert len(axesOf().collections) >= 1


def screenOf(ax, p):
  from mpl_toolkits.mplot3d import proj3d
  ax.figure.canvas.draw()
  M = ax.get_proj()
  return np.array(proj3d.proj_transform(*p, M)[:2]) - np.array(proj3d.proj_transform(0, 0, 0, M)[:2])


@pytest.mark.parametrize("convention, east, north", [("y↑→x", (1, 0, 0), (0, 1, 0)), ("x↑←y", (0, -1, 0), (1, 0, 0)), ("z↑→x", (1, 0, 0), (0, 0, 1))])
def test_arrows_in_fresh_axes_look_down_on_the_screen_plane(convention, east, north):
  # MATLAB's new axes show the plane of the plotting convention, east to the right
  pC = plottingConvention.default
  try:
    plottingConvention.default(convention)
    arrow3d(vector3d(1, 1, 1), label="v")
    ax = axesOf()
    e, n = screenOf(ax, east), screenOf(ax, north)
    assert e[0] > 0 and abs(e[1]) < 1e-3 * abs(e[0]) and n[1] > 0 and abs(n[0]) < 1e-3 * abs(n[1])
    # a held plot keeps the camera, the limits take in the labels
    hold(True)
    arrow3d(4 * vector3d.X, label="u")
    hold(False)
    assert ax.get_xlim()[1] >= 1.25 * 4 - 1e-9
  finally:
    pC.makeDefault()


def test_odf_isosurfaces_in_euler_and_axis_angle_space():
  cs = crystalFrame("321")
  center = orientation.byEuler(10 * degree, 20 * degree, 0, cs)
  odf = unimodalODF(center, halfwidth=10 * degree)
  hs = plot3d(odf, 'Bunge')
  ax = axesOf()
  assert len(hs) > 3 and ax.get_xlabel() == "φ1 / °"
  # the colour map and the bar reach the isosurfaces, which take their colours from it
  ax.figure.canvas.draw()
  before = hs[-1].get_facecolor()[0].copy()
  mtexColorMap('LaboTeX')
  ax.figure.canvas.draw()
  assert not np.allclose(hs[-1].get_facecolor()[0], before)
  assert len(mtexColorbar(title='mrd')) == 1
  # in axis angle space the highest surface wraps the representative of the centre
  hs = plot3d(odf, 'axisAngle')
  assert axesOf().get_xlabel() == "x / °"
  peak = center.project2FundamentalRegion()
  r = peak.axis().data.reshape(3) * peak.angle() / degree
  T = np.asarray(hs[-1]._faces).reshape(-1, 3)
  assert np.all(np.linalg.norm(T - r, axis=1) < 15)


def test_scatter_of_misorientations_fits_the_region():
  # a new scatter with its region looks at the region and the points, as MATLAB fits its axes
  cs = crystalFrame("6/mmm")
  m = orientation.rand(50, cs, cs, rng=np.random.default_rng(0))
  scatter(m)
  ax = gcm().children[-1]
  (x0, x1), (z0, z1) = ax.get_xlim(), ax.get_zlim()
  assert x0 > -20 and x1 < 110 and z0 > -30 and z1 < 50
