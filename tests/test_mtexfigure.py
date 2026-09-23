# the figure a plot goes into: the layout, the hold state, the colour bars
import numpy as np
import pytest

from mtex.plotting.mtexfigure import layoutFor


def test_the_layout_fills_a_row():
  assert layoutFor(1) == (1, 1) and layoutFor(2) == (1, 2) and layoutFor(3) == (1, 3)
  assert layoutFor(4) == (2, 2) and layoutFor(6) == (2, 3) and layoutFor(8) == (2, 4) and layoutFor(9) == (3, 3)
  assert layoutFor(5) == (2, 3) and layoutFor(7) == (2, 4) and layoutFor(12) == (3, 4)


def test_the_figure_and_its_axes():
  matplotlib = pytest.importorskip("matplotlib")
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from mtex.plotting.mtexfigure import gcm, hold, mtexColorbar, mtexFigure, newMtexFigure, setColorRange
  fig = newMtexFigure(3)
  assert gcm() is fig and (fig.nrows, fig.ncols) == (1, 3) and fig.children == []
  axes = [fig.nextAxis() for _ in range(3)]
  assert fig.children == axes and fig.currentAxes is axes[-1]
  # a fourth axes re-lays the grid
  ax4 = fig.nextAxis()
  assert (fig.nrows, fig.ncols) == (2, 2) and ax4.get_subplotspec().num1 == 3
  # hold on the current figure
  assert not fig.hold
  hold("on")
  assert fig.hold
  hold(False)
  assert not fig.hold
  # colour bars read the plots on the axes: none without a mappable, one per axes with
  # different ranges, one for all with equal ranges
  assert mtexColorbar() == []
  for k, ax in enumerate(axes[:2]):
    m = ax.pcolormesh(np.arange(4.0).reshape(2, 2) + k)
    ax._mtexSphericalPlot = type("sP", (), {"mappable": m})()
  assert len(fig.colorbar()) == 2
  setColorRange("equal")
  assert fig.colorRange == (0.0, 4.0) and all(ax._mtexSphericalPlot.mappable.get_clim() == (0.0, 4.0) for ax in axes[:2])
  # the range redraws the two bars as one, and a further call toggles them away and back
  assert len(fig.colorbars) == 1
  assert fig.colorbar() == [] and fig.colorbars == []
  assert len(fig.colorbar()) == 1
  fig.setColorRange((1, 2))
  assert fig.colorRange == (1, 2)
  with pytest.raises(ValueError):
    fig.setColorRange("same")
  fig.setColormap(matplotlib.colormaps["viridis"])
  assert axes[0]._mtexSphericalPlot.mappable.get_cmap().name == "viridis"
  fig.drawNow()
  other = mtexFigure(figSize=(2, 2))
  assert gcm() is other and other.fig.get_size_inches().tolist() == [2, 2]
  plt.close("all")


def test_xlabel_on_a_plain_axes_and_on_a_chart():
  pytest.importorskip("matplotlib")
  import matplotlib
  matplotlib.use("Agg")
  from mtex import histogram, plot, vector3d, xlabel
  # a histogram keeps its axis, so the label is the axis label under the ticks
  import matplotlib.pyplot as plt
  histogram(np.linspace(0, 90, 50), 10)
  xlabel("angle")
  assert plt.gca().get_xlabel() == "angle"
  # a spherical chart has no axis, so the label is a text under it
  plot(vector3d(1, 0, 0), upper=True)
  t = xlabel("chart")
  assert t.axes.get_xlabel() == "" and t.get_text() == "chart"


def test_colorbar_of_explicit_colours_shows_the_map_and_range_set():
  pytest.importorskip("matplotlib")
  import matplotlib
  matplotlib.use("Agg")
  from mtex import mtexColorbar, mtexColorMap, plot, setColorRange, vector3d
  # RGB colours carry no values, so the bar shows the map and range set after the plot, as MATLAB
  plot(vector3d(np.eye(3)), np.eye(3), upper=True)
  mtexColorMap("blue2red")
  setColorRange([0, 90])
  cb = mtexColorbar(title="angle")
  assert len(cb) == 1 and cb[0].mappable.get_clim() == (0, 90) and cb[0].ax.get_ylabel() == "angle"


def test_a_named_size_keeps_room_for_the_labels():
  from mtex.plotting.mtexfigure import figureSize
  assert figureSize("small", (3.2, 3.2)) == (2.5, 2.5)
  assert figureSize("large", (3.2, 3.2)) == (1.3 * 3.2, 1.3 * 3.2)
  assert figureSize("small", (7.0, 2.0)) == (0.6 * 7.0, 2.0)
