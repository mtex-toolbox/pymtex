# MTEX's colour maps as stops, and the contour levels
import numpy as np
import pytest

from mtex.plotting.colormaps import colormapColors, colormapStops, contourLevels, mtexColorMap, mtexColormap


def test_the_stops_and_the_colours():
  colors, pos = colormapStops("WhiteJet")
  assert colors[0] == (1.0, 1.0, 1.0) and colors[-1] == (0.5, 0.0, 0.0) and np.isclose(pos[-1], 1)
  assert colormapStops("nothing") is None
  C = colormapColors("whitejet", 5)
  assert C.shape == (5, 3) and np.allclose(C[0], 1) and np.allclose(C[-1], (0.5, 0, 0))
  assert np.allclose(colormapColors("LaboTeX", 3)[1], (1.0, 0.4, 0.4))
  assert np.allclose(colormapColors("blue2red", 3)[1], 1)
  # MATLAB's blue2red runs through pure blue and pure red between its dark ends and white
  assert np.allclose(colormapColors("blue2red", 101)[16], [0, 0, 1]) and np.allclose(colormapColors("blue2red", 101)[84], [1, 0, 0])
  with pytest.raises(ValueError):
    colormapColors("nothing")


def test_the_matplotlib_maps():
  pytest.importorskip("matplotlib")
  cmap = mtexColormap("WhiteJet")
  assert cmap.name == "WhiteJet" and np.allclose(cmap(0.0)[:3], 1) and np.allclose(cmap(1.0)[:3], (0.5, 0, 0))
  assert mtexColormap("whitejet").name == "WhiteJet" and mtexColormap(cmap) is cmap
  assert mtexColormap("viridis").name == "viridis"
  assert mtexColorMap("blue2red").name == "blue2red"
  # the preference defaultColorMap is the map when none is named, a colour matrix or a named map function as MATLAB passes it
  from mtex import blue2redColorMap, getMTEXpref, setMTEXpref
  old = getMTEXpref("defaultColorMap")
  try:
    assert mtexColormap().name == "WhiteJet" or old is not None
    setMTEXpref("defaultColorMap", blue2redColorMap)
    assert np.allclose(mtexColormap()(0.0)[:3], blue2redColorMap()[0])
    setMTEXpref("defaultColorMap", blue2redColorMap())
    assert np.allclose(mtexColormap()(1.0)[:3], blue2redColorMap()[-1])
  finally:
    setMTEXpref("defaultColorMap", old)


def test_contour_levels():
  assert np.allclose(contourLevels(0, 1, 3), [0, 0.5, 1])
  assert np.allclose(contourLevels(1, 100, 3, log=True), [1, 10, 100])
  lv = contourLevels(2, 2, 3)
  assert len(lv) == 3 and lv[-1] > lv[0]


def test_vega20():
  from mtex import vega20
  C = vega20(24)
  assert C.shape == (24, 3) and np.allclose(C[20], C[0]) and np.allclose(C[0], np.array([31, 119, 180]) / 255)


def test_a_plain_figure_takes_the_map():
  plt = pytest.importorskip("matplotlib.pyplot")
  plt.close("all")
  plt.figure()
  x, y = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
  filled = plt.contourf(x, y, x + y)
  lines = plt.contour(x, y, x + y, colors="k")
  mtexColorMap("LaboTeX")
  assert filled.get_cmap().name == mtexColormap("LaboTeX").name
  assert np.allclose(lines.get_edgecolor()[0][:3], 0)
  plt.close("all")
