# the histograms of grains and the ellipses on a map (the Grains chapter), on the Agg backend
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from mtex import EBSD, calcGrains, crystalFrame, degree, gcm, histogram, hold, phaseList, plot, plotEllipse, rotation, zvector


@pytest.fixture(autouse=True)
def closeFigures():
  yield
  plt.close("all")


def grid():
  a, b = crystalFrame("m-3m", mineral="A"), crystalFrame("6/mmm", [1, 1, 1.6], mineral="B")
  nx, ny = 12, 8
  i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
  i, j = i.reshape(-1, order="F"), j.reshape(-1, order="F")
  pos = np.stack([i, j, np.zeros(nx * ny)], axis=-1).astype(float)
  rot = rotation.byAxisAngle(zvector, (i // 4 + 3 * (j // 4)) * 20 * degree)
  phase = np.where(j < 4, 1, 2).astype(np.int8)
  return calcGrains(EBSD(pos, rot, phase, phaseList(crystalFrame("1"), a, b)), angle=5 * degree)


def test_grouped_histogram_ofGrainsIsAreaWeightedPerPhase():
  hs = histogram(grid(), grouped=True)
  ax = gcm().children[-1]
  assert len(hs) == 2 and ax.get_ylabel() == "relative area (%)"


def test_grouped_histogram_ofGrainsAddsUpToTheWholeArea():
  hs = histogram(grid(), grouped=True)
  # the bars of both phases add up to the whole analysed area
  total = sum(b.get_height() for h in hs for b in h)
  assert np.isclose(total, 100)


def test_histogram_ofAPropertyAddsUpToTheWholeArea():
  g = grid()
  hs = histogram(g, g.numPixel, 5)
  assert len(hs) == 2 and np.isclose(sum(b.get_height() for h in hs for b in h), 100)


def test_histogram_takesTheNumberOfBins():
  hs = grid().histogram(bins=4, grouped=True)
  assert len(hs[0]) == 3


# the grains of the grid drawn with their ellipses on top
def _ellipses():
  g = grid()
  plot(g)
  hold(True)
  c, a, b = g.fitEllipse()
  h = plotEllipse(c, a, b, lineColor="w", lineWidth=2)
  hold(False)
  return g, h


def test_plotEllipse_drawsOneEllipsePerGrain():
  g, h = _ellipses()
  assert isinstance(h, LineCollection) and len(h.get_paths()) == g.size


def test_plotEllipse_ofASquareGrainHasTheAreaOfTheGrain():
  g, h = _ellipses()
  # the ellipse of a square grain has the area of the grain
  P = h.get_paths()[0].vertices
  area = 0.5 * abs(np.sum(P[:-1, 0] * P[1:, 1] - P[1:, 0] * P[:-1, 1]))
  assert np.isclose(area, g.area[0], rtol=0.05)
