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


def test_hue_by_one_property_and_saturation_by_the_other():
  pK = planarColorKey("twilight")
  pK.periode = np.pi
  pK.range2 = [1, 3]
  c = pK.property2color([0, 0.5, 1, 1.5, 1.5], [1, 1.5, 2, 3, 5])
  assert c.shape == (5, 3) and np.all((c >= 0) & (c <= 1))
  # the low end of the second property fades to white, the high end is the full hue
  assert np.allclose(c[0], 1) and not np.allclose(c[3], 1) and np.allclose(c[3], c[4])
  # a periodic property wraps: the angle and the angle plus the period share the colour
  a, b = pK.property2color([20 * degree, 20 * degree + np.pi], [3, 3])
  assert np.allclose(a, b)
  # degrees are taken as such once the values exceed a turn
  d, = pK.property2color([20], [3])
  assert np.allclose(a, d)
  # black shading and a table of colours
  pB = planarColorKey(np.array([[1, 0, 0], [0, 0, 1.0]]), shading="black", range1=(0, 1), range2=(0, 1))
  assert np.allclose(pB.property2color([0, 1], [1, 1]), [[1, 0, 0], [0, 0, 1]]) and np.allclose(pB.property2color([0.5], [0]), 0)
  with pytest.raises(ValueError):
    pK.property2color([1, 2], [1])
  assert "planarColorKey" in repr(pK)


def test_the_key_as_an_image():
  pK = planarColorKey("twilight")
  pK.periode = np.pi
  pK.label1, pK.label2 = "angle", "ratio"
  h = plot(pK)
  ax = gcm().children[-1]
  assert h.get_array().shape == (128, 256, 3) and len(ax.get_xticks()) == 0
  h = plot(pK, np.linspace(0, 180, 20), np.linspace(1, 3, 20))
  ax = gcm().children[-1]
  assert ax.get_xlabel() == "angle" and ax.get_ylabel() == "ratio" and [t.get_text() for t in ax.get_xticklabels()][-1] == "180"
  pK.plot(np.linspace(0, 1, 9), np.linspace(0, 2, 9))
