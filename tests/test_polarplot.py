# the polar plots (docs/plans/plotting.md, the Grains chapter), on the Agg backend
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mtex import (S1FunHarmonic, S2FunHarmonic, S2VectorFieldHarmonic, calcDensity, characteristicShape, crystalFrame, degree, gcm,
                  histogram, hold, legend, mtexColorbar, mtexColorMap, newMtexFigure, nextAxis, plot, plotSection, plottingConvention,
                  quiverSection, setView, specimenFrame, vector3d)
from mtex.plotting.polarplot import polarAxes


@pytest.fixture(autouse=True)
def closeFigures():
  yield
  plt.close("all")


def axesOf():
  return gcm().children[-1]


# 300 axes in the plane normal to z around 70 degrees
def axesAround70(rng):
  rho = rng.normal(70 * degree, 15 * degree, 300)
  return vector3d.byPolar(np.full(300, np.pi / 2), rho, antipodal=True)


def test_histogram_ofAxesCountsAtAzimuthAndOpposite():
  rng = np.random.default_rng(3)
  v = axesAround70(rng)
  h = histogram(v, 36, weights=rng.random(300))
  ax = axesOf()
  assert ax.name == "polar" and len(h) == 36
  # an axis counts at its azimuth and opposite, the bins sum to one
  heights = np.array([b.get_height() for b in h])
  assert np.isclose(heights.sum(), 1) and np.isclose(heights[6:8].sum(), heights[24:26].sum())


def test_setView_followsTheConvention():
  histogram(axesAround70(np.random.default_rng(3)), 36)
  ax = axesOf()
  # the view follows the convention: y up and x right is counterclockwise from the east
  setView(plottingConvention('y↑→x'), ax)
  assert ax.get_theta_direction() == 1 and ax.get_theta_offset() == 0
  setView(plottingConvention('y↓→x'), ax)
  assert ax.get_theta_direction() == -1


def test_histogram_turnsThePendingAxesOfALayoutPolar():
  v = axesAround70(np.random.default_rng(3))
  # the pending axes of a layout turns polar
  newMtexFigure(layout=[1, 2])
  histogram(v)
  nextAxis()
  histogram(v, 18)
  assert len(gcm().children) == 2 and all(a.name == "polar" for a in gcm().children)
  assert polarAxes(gcm().children[0]) is gcm().children[0]


# a density on the circle of axes around 70 degrees
def circleDensity():
  rng = np.random.default_rng(4)
  rho = np.mod(rng.normal(70 * degree, 15 * degree, 300), np.pi)
  return calcDensity(rho, weights=np.ones(300), periodic=True, antipodal=True, sigma=5 * degree)


# a function on the sphere peaking at x
def peakAtX():
  return S2FunHarmonic(lambda v: 1 + v.x ** 2, bandwidth=8)


def test_plot_ofS1FunIsPolarAndHolds():
  f = circleDensity()
  assert isinstance(f, S1FunHarmonic)
  h = plot(f, lineWidth=2)
  assert axesOf().name == "polar" and h.get_linewidth() == 2
  hold(True)
  plot(f, displayName="again")
  hold(False)
  assert len(axesOf().lines) == 2
  plot(f, polar=False)
  assert axesOf().name != "polar"


def test_plotSection_ofS2FunPeaksWhereItShould():
  s = peakAtX()
  # a function on the sphere along the great circle normal to z peaks where it should
  h = plotSection(s, vector3d.Z, lineWidth=3)
  r = np.hypot(h.get_xdata(), h.get_ydata())
  assert axesOf().name != "polar" and np.isclose(r.max(), 2, atol=0.05) and np.isclose(r.min(), 1, atol=0.05)
  # seen down z with x east, the peaks lie left and right
  x, y = h.get_xdata(), h.get_ydata()
  assert np.isclose(abs(x).max(), 2, atol=0.05) and np.isclose(abs(y[np.argmin(abs(x))]), 1, atol=0.05)
  h = plotSection(s, vector3d.Z, polarplot=True)
  assert axesOf().name == "polar" and np.isclose(h.get_ydata().max(), 2, atol=0.05)


def test_plotSection_colorInterpColoursTheLineByItsValues():
  # MATLAB's color 'interp' colours the line by its values, which the colour bar reads
  h = plotSection(peakAtX(), vector3d.Z, color="interp", lineWidth=5)
  assert np.isclose(h.get_array().max(), 2, atol=0.05) and h.get_linewidth()[0] == 5
  assert gcm()._mappables()[-1][1] is h


def test_S1FunHarmonic_interpolate_fitsRepeatedEndPointAtQuarterBandwidth():
  # samples with the end point repeated are fitted by least squares at a quarter of their number, as MATLAB does
  omega = np.linspace(0, 2 * np.pi, 361)
  g = S1FunHarmonic.interpolate(omega, 0.85 + 0.15 * np.cos(2 * (omega - 71 * degree)))
  assert g.bandwidth == 90 and np.isclose(np.degrees(g.max()[1]) % 180, 71, atol=0.3) and np.isclose(np.degrees(g.min()[1]) % 180, 161, atol=0.3)


# two rings with arrows at the outer and axes at the inner one, in the first of two panels
def ringsWithArrows():
  s1 = S2FunHarmonic(lambda v: 2 + v.x ** 2, bandwidth=8)
  s2 = S2FunHarmonic(lambda v: 1 + 0 * v.x, bandwidth=2)
  newMtexFigure(layout=[1, 2])
  a = plotSection(s1, vector3d.Z, color="interp", displayName="outer")
  hold(True)
  b = plotSection(s2, vector3d.Z, color="interp", displayName="inner")
  # the normals as arrows at the outer ring, of MATLAB's autoscaled length
  q = quiverSection(s1, S2VectorFieldHarmonic.normal(), vector3d.Z, color="k", autoScaleFactor=0.5, displayName="n")
  # an axis field as segments both ways, the reverse half left out of the legend
  axes = lambda v: vector3d(v.data, antipodal=True)
  qa = quiverSection(s2, axes, vector3d.Z, color="m", displayName="p")
  hold(False)
  return s1, a, b, q, qa


def test_quiverSection_axisFieldIsSegmentsBothWays():
  _, _, _, q, qa = ringsWithArrows()
  assert len(q) == 1 and len(qa) == 2 and qa[0].get_label() == "_nolegend_"


def test_quiverSection_arrowsSitOnTheRingPointingOutAtAutoscaledLength():
  _, _, _, q, _ = ringsWithArrows()
  # arrow tails sit on the outer ring and point outwards
  X, U = np.stack([q[0].X, q[0].Y], axis=1), np.stack([q[0].U, q[0].V], axis=1)
  r = np.hypot(X[:, 0], X[:, 1])
  assert np.all((r > 1.99) & (r < 3.01)) and np.all(np.sum(X * U, axis=1) > 0)
  spacing = np.sqrt(np.sum(((X.max(axis=0) - X.min(axis=0)) / 6) ** 2))
  assert np.isclose(np.hypot(U[:, 0], U[:, 1]).max(), 0.5 * spacing, rtol=0.05)


def test_plotSection_ringsShareOneColourRangeAndMap():
  _, a, b, _, _ = ringsWithArrows()
  # the rings share one colour range and follow the colour map of the figure
  assert np.allclose(a.get_clim(), b.get_clim()) and np.isclose(a.get_clim()[0], 1, atol=0.01) and np.isclose(a.get_clim()[1], 3, atol=0.05)
  mtexColorMap("blue2red")
  assert a.get_cmap().name == b.get_cmap().name


def test_legend_namesRingsAndArrows():
  ringsWithArrows()
  leg = legend("outer", "inner", "n", "p")
  assert [t.get_text() for t in leg.get_texts()] == ["outer", "inner", "n", "p"] and leg.legend_handles[3].get_label() != "_nolegend_"


def test_mtexColorbar_oneTitlePerPanel():
  s1 = ringsWithArrows()[0]
  # one colour bar title per panel
  nextAxis()
  plotSection(s1, vector3d.Z, color="interp")
  bars = mtexColorbar(title=["km/s", "s/km"])
  assert [cb.ax.get_xlabel() or cb.ax.get_ylabel() for cb in bars] == ["km/s", "s/km"]


def test_a_characteristic_shape():
  from mtex import EBSD, calcGrains, phaseList, rotation, zvector
  fr = crystalFrame("m-3m", mineral="cubes")
  nx, ny = 12, 6
  i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
  i, j = i.reshape(-1, order="F"), j.reshape(-1, order="F")
  pos = np.stack([i, j, np.zeros(nx * ny)], axis=-1).astype(float)
  rot = rotation.byAxisAngle(zvector, (i // 4) * 30 * degree)
  g = calcGrains(EBSD(pos, rot, np.full(nx * ny, 1, dtype=np.int8), phaseList(fr)), angle=5 * degree)
  s = characteristicShape(g.boundary)
  h = plot(s, lineWidth=2, displayName="cubes")
  assert axesOf().name == "polar" and h.get_label() == "cubes"
  plot(s, normalize=True, plain=True)
  assert len(axesOf().get_yticks()) == 0


def test_rose_diagrams_go_into_the_subplots_the_user_made():
  # MATLAB's figure; subplot(1,2,1); histogram(v): the rose takes the polar axes prepared for it
  from mtex import histogram, vector3d
  v = vector3d.rand(200, rng=np.random.default_rng(0))
  f = plt.figure()
  a1 = plt.subplot(1, 2, 1, projection="polar")
  histogram(v, 36)
  a2 = plt.subplot(1, 2, 2, projection="polar")
  histogram(v, 36)
  assert plt.gcf() is f and len(f.axes) == 2 and len(a1.patches) == 36 and len(a2.patches) == 36
