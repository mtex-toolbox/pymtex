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


# 30 random directions in a specimen without symmetry
def randomDirections():
  return vector3d(vector3d.rand(30, rng=np.random.default_rng(9)), frame=specimenFrame("1"))


# a smooth positive function on the sphere
def positiveFunction():
  g = np.random.default_rng(10)
  return S2FunHarmonic(g.standard_normal(49) + 1j * g.standard_normal(49), real=True) + 3


# the axis field about z
def axisField():
  return S2AxisFieldHarmonic(lambda v: vector3d(v.cross(zvector).data, antipodal=True), bandwidth=12)


def test_plot3d_ofDirectionsDrawsTheSphereTheGridAndTheAxesLabels():
  h = plot3d(randomDirections())
  ax = axesOf()
  assert ax.name == "3d" and hasattr(ax, "_mtexSpacePlot") and len(h.get_offsets()) == 30
  # the sphere, the grid and the axes labels are there
  assert any(isinstance(c, Poly3DCollection) for c in ax.collections) and len(ax.lines) == 17
  assert {t.get_text() for t in ax.texts} == {"X", "Y", "Z"}


def test_scatter3d_ofAxesDrawsBothEndsColouredWithABar():
  # axes are drawn at both ends, values through the colour map with a bar
  h = scatter3d(vector3d.rand(10, antipodal=True, rng=rng), rng.random(10), colorbar=True)
  assert len(h.get_offsets()) == 20 and len(gcm().fig.axes) == 2


def test_plot3d_writesALabelPerDirection():
  randomDirections().plot3d(label=[str(k) for k in range(30)], noLabel=True)
  assert len(axesOf().texts) == 30


def test_plot3d_ofAFunctionColoursTheSphereAtItsFaces():
  h = plot3d(positiveFunction(), colorbar=True)
  assert isinstance(h, Poly3DCollection) and h.get_array().size == len(h.get_paths())


def test_plot3d_radialPushesTheSphereOut():
  h = positiveFunction().plot3d(radial=True, resolution=10 * degree)
  assert isinstance(h, Poly3DCollection) and axesOf().get_xlim()[1] > 1.3


def test_plot3d_ofAVectorFieldIsArrows():
  h = plot3d(grad(positiveFunction()))
  assert isinstance(h, Line3DCollection)


def test_plot3d_ofAnAxisFieldIsSegments():
  h = axisField().plot3d()
  assert isinstance(h, Line3DCollection)


def test_quiver3_ofASphereFieldDrawsOntoAHeldSphereWithoutASecondOne():
  plot3d(positiveFunction())
  n = len(axesOf().collections)
  hold(True)
  h = quiver3(axisField(), color="white")
  hold(False)
  assert isinstance(h, Line3DCollection) and len(axesOf().collections) == n + 1


# the cubic orientation region against an orthorhombic specimen
def cubicRegion():
  return fundamentalRegion(crystalFrame("m-3m"), specimenFrame("222"))


# 200 orientations about one peak
@pytest.fixture(scope="module")
def sample():
  cs, ss = crystalFrame("m-3m"), specimenFrame("222")
  odf = unimodalODF(orientation.byEuler(30 * degree, 50 * degree, 10 * degree, cs, ss), halfwidth=10 * degree)
  return discreteSample(odf, 200, rng=np.random.default_rng(11))


def test_plot_ofARegionFitsTheAxesToIt():
  oR = cubicRegion()
  h = plot(oR)
  ax = axesOf()
  assert isinstance(h, Poly3DCollection) and ax.get_xlabel() == "x / °"
  # the axes are fitted to the region
  P = np.concatenate(oR.edges()) / degree
  assert ax.get_xlim()[1] >= P[:, 0].max() and ax.get_xlim()[0] <= P[:, 0].min() and len(ax.lines) > 10


def test_plot_ofARegionWithoutLinesDrawsOnlyItsEdges():
  oR = cubicRegion()
  # without the parallels and meridians only the edges of the region are lines
  oR.plot(lines=False)
  assert len(axesOf().lines) == len(oR.edges()) > 0


def test_plot_ofARegionWithoutSurfaceDrawsOnlyItsEdges():
  oR = cubicRegion()
  hs = plot(oR, noSurface=True, boundaryColor='r')
  assert len(hs) == len(oR.edges()) and not axesOf().collections


def test_plot_ofAMisorientationRegionHasWallsFromTheIdentity():
  cs = crystalFrame("m-3m")
  plot(fundamentalRegion(cs, cs))
  assert len([c for c in axesOf().collections if isinstance(c, Poly3DCollection)]) > 1


def test_scatter_ofOrientationsInAxisAngleLiesInTheRegionInDegrees(sample):
  # an orientation scatters in the Euler region by default, as MATLAB does; axis angle on request
  h = scatter(sample, axisAngle=True)
  ax = axesOf()
  P = np.stack([np.asarray(a) for a in h._offsets3d], axis=1)
  assert P.shape == (200, 3) and np.all(np.linalg.norm(P, axis=1) <= 63) and ax.get_xlabel() == "x / °"
  assert any(isinstance(c, Poly3DCollection) for c in ax.collections)


def test_scatter_ofOrientationsColouredByAValueThinnedWithoutRegion(sample):
  h = sample.scatter(np.degrees(sample.angle()), colorbar=True, region=False, points=50)
  assert len(h._offsets3d[0]) == 50 and not any(isinstance(c, Poly3DCollection) for c in axesOf().collections)


def test_scatter_ofOrientationsInEulerLiesInTheEulerBox(sample):
  h = scatter(sample, Euler=True)
  ax = axesOf()
  P = np.stack([np.asarray(a) for a in h._offsets3d], axis=1)
  assert ax.get_xlabel() == "φ1 / °" and np.all(P[:, 1] <= 90 + 1e-6) and np.all(P[:, 2] <= 90 + 1e-6)


def test_scatter_ofOrientationsInRodrigues(sample):
  scatter(sample, Rodrigues=True)
  assert axesOf().get_xlabel() == "r1"


def test_scatter_ignoreFundamentalRegionDrawsTheRepresentativesAsStored(sample):
  h = scatter(sample, ignoreFundamentalRegion=True)
  assert len(h._offsets3d[0]) == 200


def test_plot3d_ofASingleOrientation(sample):
  plot3d(sample[0])
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
  finally:
    pC.makeDefault()


@pytest.mark.parametrize("convention", ["y↑→x", "x↑←y", "z↑→x"])
def test_arrow3d_heldKeepsTheCameraAndTheLimitsTakeInTheLabels(convention):
  pC = plottingConvention.default
  try:
    plottingConvention.default(convention)
    arrow3d(vector3d(1, 1, 1), label="v")
    ax = axesOf()
    hold(True)
    arrow3d(4 * vector3d.X, label="u")
    hold(False)
    assert ax.get_xlim()[1] >= 1.25 * 4 - 1e-9
  finally:
    pC.makeDefault()


# a peak of a trigonal ODF off the origin
def trigonalPeak():
  center = orientation.byEuler(10 * degree, 20 * degree, 0, crystalFrame("321"))
  return center, unimodalODF(center, halfwidth=10 * degree)


def test_plot3d_ofAnOdfInEulerSpaceTakesTheColourMapAndTheBar():
  _, odf = trigonalPeak()
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


def test_plot3d_ofAnOdfInAxisAngleSpaceWrapsTheCentreInTheHighestSurface():
  center, odf = trigonalPeak()
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


# a field on SO(3): one arrow per grid point of the fundamental region, pointing where the
# field moves the orientation, the longest nine tenths of the grid spacing
def test_quiver3_of_an_SO3_vector_field():
  import matplotlib.pyplot as plt
  from mtex import SO3VectorFieldHandle, crystalFrame, degree, quiver3
  cs = crystalFrame("432")
  VF = SO3VectorFieldHandle(lambda m: m.axis() * m.angle(), cs, cs)
  h = quiver3(VF, 'axisAngle', resolution=10 * degree)
  # matplotlib keeps three segments per arrow, the shaft from the tip to the tail first
  segs = np.asarray(h._segments3d)[0::3]
  assert len(segs) > 10
  # axis times angle points away from the identity: every arrow runs outwards
  start, end = segs[:, 1, :], segs[:, 0, :]
  far = np.linalg.norm(start, axis=1) > 1
  assert np.all(np.linalg.norm(end[far], axis=1) > np.linalg.norm(start[far], axis=1))
  plt.close("all")


def test_held_orientation_sets_take_the_colour_order_as_open_circles():
  from mtex.plotting.colormaps import MTEX_COLORS
  cs = crystalFrame("-3m")
  odf = unimodalODF(orientation.id(cs))
  a, b = discreteSample(odf, 20, rng=rng), discreteSample(odf, 20, rng=rng)
  h1 = scatter(a, 'axisAngle')
  hold(True)
  h2 = scatter(b)
  hold(False)
  assert np.allclose(h1.get_edgecolor()[0][:3], MTEX_COLORS[0]) and np.allclose(h2.get_edgecolor()[0][:3], MTEX_COLORS[1])
  assert h1.get_facecolor().size == 0 or np.all(h1.get_facecolor()[:, 3] == 0)


def test_annotate_goesIntoTheSpaceTheFigureShowsInItsOwnColour():
  from mtex import annotate
  cs = crystalFrame("-3m")
  scatter(discreteSample(unimodalODF(orientation.id(cs)), 20, rng=rng), 'axisAngle')
  h3 = annotate(orientation.id(cs), MarkerFaceColor='r')
  assert np.allclose(h3.get_facecolor()[0][:3], (1, 0, 0))
