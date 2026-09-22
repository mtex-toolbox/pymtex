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
