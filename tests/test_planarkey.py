# the colour key of two properties (MATLAB planarColorKey), on the Agg backend
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mtex import degree, gcm, planarColorKey, plot


@pytest.fixture(autouse=True)
def closeFigures():
  yield
  plt.close("all")


# the twilight key over a property of period pi and a second one from 1 to 3
def _twilight():
  pK = planarColorKey("twilight")
  pK.periode = np.pi
  pK.range2 = [1, 3]
  return pK


def test_planarColorKey_givesAColourPerPair():
  c = _twilight().property2color([0, 0.5, 1, 1.5, 1.5], [1, 1.5, 2, 3, 5])
  assert c.shape == (5, 3) and np.all((c >= 0) & (c <= 1))


def test_planarColorKey_fadesToWhiteAtTheLowEndOfTheSecondProperty():
  c = _twilight().property2color([0, 0.5, 1, 1.5, 1.5], [1, 1.5, 2, 3, 5])
  # the low end of the second property fades to white, the high end is the full hue
  assert np.allclose(c[0], 1) and not np.allclose(c[3], 1) and np.allclose(c[3], c[4])


def test_planarColorKey_periodicPropertyWraps():
  pK = _twilight()
  # a periodic property wraps: the angle and the angle plus the period share the colour
  a, b = pK.property2color([20 * degree, 20 * degree + np.pi], [3, 3])
  assert np.allclose(a, b)


def test_planarColorKey_valuesBeyondATurnAreDegrees():
  pK = _twilight()
  a, _ = pK.property2color([20 * degree, 20 * degree + np.pi], [3, 3])
  # degrees are taken as such once the values exceed a turn
  d, = pK.property2color([20], [3])
  assert np.allclose(a, d)


def test_planarColorKey_blackShadingOverATableOfColours():
  # black shading and a table of colours
  pB = planarColorKey(np.array([[1, 0, 0], [0, 0, 1.0]]), shading="black", range1=(0, 1), range2=(0, 1))
  assert np.allclose(pB.property2color([0, 1], [1, 1]), [[1, 0, 0], [0, 0, 1]]) and np.allclose(pB.property2color([0.5], [0]), 0)


def test_planarColorKey_refusesPropertiesOfDifferentLengths():
  with pytest.raises(ValueError):
    _twilight().property2color([1, 2], [1])


def test_planarColorKey_reprNamesTheKey():
  assert "planarColorKey" in repr(_twilight())


def test_plot_ofAPlanarColorKeyIsAnImageWithoutTicks():
  pK = planarColorKey("twilight")
  pK.periode = np.pi
  h = plot(pK)
  ax = gcm().children[-1]
  assert h.get_array().shape == (128, 256, 3) and len(ax.get_xticks()) == 0


def test_plot_ofAPlanarColorKeyOverValuesLabelsItsAxes():
  pK = planarColorKey("twilight")
  pK.periode = np.pi
  pK.label1, pK.label2 = "angle", "ratio"
  plot(pK, np.linspace(0, 180, 20), np.linspace(1, 3, 20))
  ax = gcm().children[-1]
  assert ax.get_xlabel() == "angle" and ax.get_ylabel() == "ratio" and [t.get_text() for t in ax.get_xticklabels()][-1] == "180"
  pK.plot(np.linspace(0, 1, 9), np.linspace(0, 2, 9))
