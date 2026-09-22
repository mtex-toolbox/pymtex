# the polar plots (docs/plans/plotting.md, the Grains chapter), on the Agg backend
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mtex import (S1FunHarmonic, S2FunHarmonic, calcDensity, characteristicShape, crystalFrame, degree, gcm, histogram, hold, plot, plotSection,
                  plottingConvention, setView, specimenFrame, vector3d)
from mtex.plotting.polarplot import polarAxes


@pytest.fixture(autouse=True)
def closeFigures():
  yield
  plt.close("all")


def axesOf():
  return gcm().children[-1]


def test_rose_diagrams_and_the_view():
  rng = np.random.default_rng(3)
  rho = rng.normal(70 * degree, 15 * degree, 300)
  v = vector3d.byPolar(np.full(300, np.pi / 2), rho, antipodal=True)
  h = histogram(v, 36, weights=rng.random(300))
  ax = axesOf()
  assert ax.name == "polar" and len(h) == 36
  # an axis counts at its azimuth and opposite, the bins sum to one
  heights = np.array([b.get_height() for b in h])
  assert np.isclose(heights.sum(), 1) and np.isclose(heights[6:8].sum(), heights[24:26].sum())
  # the view follows the convention: y up and x right is counterclockwise from the east
  setView(plottingConvention('y↑→x'), ax)
  assert ax.get_theta_direction() == 1 and ax.get_theta_offset() == 0
  setView(plottingConvention('y↓→x'), ax)
  assert ax.get_theta_direction() == -1
  # the pending axes of a layout turns polar
  from mtex import newMtexFigure, nextAxis
  newMtexFigure(layout=[1, 2])
  histogram(v)
  nextAxis()
  histogram(v, 18)
  assert len(gcm().children) == 2 and all(a.name == "polar" for a in gcm().children)
  assert polarAxes(gcm().children[0]) is gcm().children[0]


def test_functions_on_the_circle_and_sections_of_the_sphere():
  rng = np.random.default_rng(4)
  rho = np.mod(rng.normal(70 * degree, 15 * degree, 300), np.pi)
  f = calcDensity(rho, weights=np.ones(300), periodic=True, antipodal=True, sigma=5 * degree)
  assert isinstance(f, S1FunHarmonic)
  h = plot(f, lineWidth=2)
  assert axesOf().name == "polar" and h.get_linewidth() == 2
  hold(True)
  plot(f, DisplayName="again")
  hold(False)
  assert len(axesOf().lines) == 2
  plot(f, notPolar=True)
  assert axesOf().name != "polar"
  # a function on the sphere along the great circle normal to z peaks where it should
  s = S2FunHarmonic(lambda v: 1 + v.x ** 2, bandwidth=8)
  h = plotSection(s, vector3d.Z, lineWidth=3)
  theta, r = h.get_xdata(), h.get_ydata()
  assert axesOf().name == "polar" and np.isclose(r.max(), 2, atol=0.05) and np.isclose(r.min(), 1, atol=0.05)
  # the interpolation through samples resolves them and no more
  omega = np.linspace(0, 2 * np.pi, 361)
  g = S1FunHarmonic.interpolate(omega, 0.85 + 0.15 * np.cos(2 * (omega - 71 * degree)))
  assert g.bandwidth == 179 and np.isclose(np.degrees(g.max()[1]) % 180, 71, atol=0.3) and np.isclose(np.degrees(g.min()[1]) % 180, 161, atol=0.3)


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
  h = plot(s, lineWidth=2, DisplayName="cubes")
  assert axesOf().name == "polar" and h.get_label() == "cubes"
  plot(s, normalize=True, plain=True)
  assert len(axesOf().get_yticks()) == 0
