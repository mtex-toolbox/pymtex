# the figure a plot goes into: the layout, the hold state, the colour bars
import numpy as np
import pytest

from mtex.plotting.mtexfigure import layoutFor, newMtexFigure


def test_layoutFor_fillsARowFirst():
  assert layoutFor(1) == (1, 1) and layoutFor(2) == (1, 2) and layoutFor(3) == (1, 3)
  assert layoutFor(4) == (2, 2) and layoutFor(6) == (2, 3) and layoutFor(8) == (2, 4) and layoutFor(9) == (3, 3)
  assert layoutFor(5) == (2, 3) and layoutFor(7) == (2, 4) and layoutFor(12) == (3, 4)


# the Agg backend of matplotlib, the test skipped without it
def agg():
  matplotlib = pytest.importorskip("matplotlib")
  matplotlib.use("Agg")
  return matplotlib


@pytest.fixture(autouse=True)
def closeFigures():
  yield
  try:
    import matplotlib.pyplot as plt
  except ImportError:
    return
  plt.close("all")


# a figure of three axes, the first two holding a mesh of values 0..3 and 1..4
def twoMeshes():
  agg()
  from mtex.plotting.mtexfigure import newMtexFigure
  fig = newMtexFigure(3)
  axes = [fig.nextAxis() for _ in range(3)]
  for k, ax in enumerate(axes[:2]):
    m = ax.pcolormesh(np.arange(4.0).reshape(2, 2) + k)
    ax._mtexSphericalPlot = type("sP", (), {"mappable": m})()
  return fig, axes


def test_newMtexFigure_isTheCurrentFigureAndNextAxisFillsItsRow():
  agg()
  from mtex.plotting.mtexfigure import gcm
  fig = newMtexFigure(3)
  assert gcm() is fig and (fig.nrows, fig.ncols) == (1, 3) and fig.children == []
  axes = [fig.nextAxis() for _ in range(3)]
  assert fig.children == axes and fig.currentAxes is axes[-1]


def test_nextAxis_aFourthAxesReLaysTheGrid():
  agg()
  fig = newMtexFigure(3)
  for _ in range(3):
    fig.nextAxis()
  ax4 = fig.nextAxis()
  assert (fig.nrows, fig.ncols) == (2, 2) and ax4.get_subplotspec().num1 == 3


def test_hold_switchesTheCurrentFigure():
  agg()
  from mtex.plotting.mtexfigure import hold
  fig = newMtexFigure(3)
  assert not fig.hold
  hold("on")
  assert fig.hold
  hold(False)
  assert not fig.hold


def test_mtexColorbar_withoutAMappableIsNone():
  agg()
  from mtex.plotting.mtexfigure import mtexColorbar
  fig = newMtexFigure(3)
  for _ in range(3):
    fig.nextAxis()
  assert mtexColorbar() == []


def test_colorbar_isOnePerAxesOfDifferentRanges():
  fig, _ = twoMeshes()
  assert len(fig.colorbar()) == 2


def test_setColorRange_equalSetsOneRangeAndRedrawsTheBarsAsOne():
  from mtex.plotting.mtexfigure import setColorRange
  fig, axes = twoMeshes()
  fig.colorbar()
  setColorRange("equal")
  assert fig.colorRange == (0.0, 4.0) and all(ax._mtexSphericalPlot.mappable.get_clim() == (0.0, 4.0) for ax in axes[:2])
  # the range redraws the two bars as one
  assert len(fig.colorbars) == 1


def test_colorbar_aFurtherCallTogglesTheBarsAwayAndBack():
  from mtex.plotting.mtexfigure import setColorRange
  fig, _ = twoMeshes()
  fig.colorbar()
  setColorRange("equal")
  assert fig.colorbar() == [] and fig.colorbars == []
  assert len(fig.colorbar()) == 1


def test_setColorRange_takesAPair():
  fig, _ = twoMeshes()
  fig.setColorRange((1, 2))
  assert fig.colorRange == (1, 2)


def test_setColorRange_refusesAnUnknownName():
  fig, _ = twoMeshes()
  with pytest.raises(ValueError):
    fig.setColorRange("same")


def test_setColormap_colourMapsTheMappables():
  matplotlib = agg()
  fig, axes = twoMeshes()
  fig.setColormap(matplotlib.colormaps["viridis"])
  assert axes[0]._mtexSphericalPlot.mappable.get_cmap().name == "viridis"
  fig.drawNow()


def test_mtexFigure_ofASizeBecomesTheCurrentFigure():
  agg()
  from mtex.plotting.mtexfigure import gcm, mtexFigure
  other = mtexFigure(figSize=(2, 2))
  assert gcm() is other and other.fig.get_size_inches().tolist() == [2, 2]


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


# a figure of a random picture saved at zlib level 1, the default, and at level 6
def savedTwice(tmp_path):
  fig = newMtexFigure()
  fig.currentAxes.imshow(np.random.default_rng(0).random((200, 300, 3)))
  fig.savefig(tmp_path / "fast.png")
  fig.savefig(tmp_path / "slow.png", pil_kwargs={"compress_level": 6})
  return fig


def test_savefig_writesTheSamePixelsAtAnyLevelAndIntoABuffer(tmp_path):
  import io
  from PIL import Image
  fig = savedTwice(tmp_path)
  buf = io.BytesIO()
  fig.savefig(buf)
  fast, slow = (np.asarray(Image.open(tmp_path / f)) for f in ("fast.png", "slow.png"))
  assert np.array_equal(fast, slow) and np.array_equal(np.asarray(Image.open(buf)), fast)


def test_savefig_compressesAtLevelOneByDefaultAndKeepsACallersLevel(tmp_path):
  savedTwice(tmp_path)
  # zlib level 1 by default, a caller's own level kept
  assert (tmp_path / "fast.png").stat().st_size > (tmp_path / "slow.png").stat().st_size


def test_saveFigure_crops_and_writes_the_format_of_the_extension(tmp_path):
  matplotlib = pytest.importorskip("matplotlib")
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from PIL import Image
  from mtex import plot, saveFigure, vector3d
  plot(vector3d.rand(20, rng=np.random.default_rng(0)), upper=True)
  saveFigure(tmp_path / "v.png")
  saveFigure(tmp_path / "v.pdf")
  fig = plt.gcf()
  w, h = Image.open(tmp_path / "v.png").size
  # cropped to the chart, at 1.5 times the screen resolution
  assert w < 1.5 * fig.dpi * fig.get_size_inches()[0] + 1 and (tmp_path / "v.pdf").stat().st_size > 0
  plt.close("all")


# a figure of two smooth plots of 50 directions, coloured hot and cool
def hotAndCool():
  agg()
  from mtex import mtexColorMap, newMtexFigure, nextAxis, plot, vector3d
  v = vector3d.rand(50, rng=np.random.default_rng(1))
  fig = newMtexFigure()
  for cm in ("hot", "cool"):
    nextAxis()
    plot(v, smooth=True, upper=True)
    # mtexColorMap(ax, name) colours only that axes; fig.gca is the axes drawn into last
    mtexColorMap(fig.gca, cm)
  return fig


def test_mtexColorMap_ofAnAxesColoursOnlyThatAxes():
  fig = hotAndCool()
  maps = [ax._mtexSphericalPlot.mappable.get_cmap().name for ax in fig.children]
  assert maps == ["hot", "cool"]


def test_mtexColorbar_multipleGivesABarPerAxesAlthoughTheRangesAgree():
  from mtex import mtexColorbar
  hotAndCool()
  assert len(mtexColorbar('multiple')) == 2


def test_drawNow_figSizeResizesToANamedSize():
  fig = hotAndCool()
  fig.drawNow(figSize='small')
  assert fig.fig.get_size_inches()[0] < 6.4


def test_legend_aSecondReplacesTheFirst():
  agg()
  import matplotlib.pyplot as plt
  from mtex import legend, plot, vector3d
  plot(vector3d.X, upper=True, displayName='x')
  legend('show')
  legend('show', location='eastoutside', title='T')
  f = plt.gcf()
  assert sum(ax.get_legend() is not None for ax in f.axes) + len(f.legends) == 1


def test_nextAxis_leavesTheAxesItMakesOnTheWayBlank():
  agg()
  from mtex import gcm, newMtexFigure, nextAxis, plot, vector3d
  newMtexFigure(layout=[1, 3])
  plot(vector3d.X, upper=True)
  nextAxis(0, 2)
  assert not gcm().children[1].axison


def test_marginx_sets_the_space_around_the_axes():
  matplotlib = pytest.importorskip("matplotlib")
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from mtex.plotting.mtexfigure import mtexFigure
  fig = mtexFigure(2, marginx=48)
  assert np.isclose(fig.fig.get_layout_engine().get()["w_pad"], 0.5)
  plt.close("all")


def test_nextAxis_ofAnOccupiedCellRepaintsItAndHoldOverlays():
  # MATLAB #2615: nextAxis(k) picks the cell the next plot paints, the layout kept
  agg()
  import matplotlib.pyplot as plt
  from mtex import hold, newMtexFigure, nextAxis, plot, vector3d
  rng = np.random.default_rng(2)
  # the markers are made when the figure is drawn
  def points(ax):
    ax.figure.canvas.draw()
    return sum(len(c.get_offsets()) for c in ax.collections)
  # directions of the upper hemisphere, all of them drawn: z points into the screen in the
  # default convention, so they are those of negative z
  up = lambda n: vector3d(rng.normal(size=n), rng.normal(size=n), -np.abs(rng.normal(size=n))).normalize()
  fig = newMtexFigure()
  plot(up(30), upper=True)
  nextAxis()
  plot(up(20), upper=True)
  nextAxis(0)
  plot(up(5), upper=True)
  assert len(fig.children) == 2 and points(fig.children[0]) == 5 and points(fig.children[1]) == 20
  # the repainted cell draws its own frame again
  assert fig.children[0]._mtexSphericalPlot is not None
  # hold overlays what the cell holds
  nextAxis(1)
  hold(True)
  plot(up(3), upper=True)
  hold(False)
  assert points(fig.children[1]) == 23
  plt.close("all")


def test_mtexColorbar_givesABarPerColormapAlthoughTheRangesAgree():
  # MATLAB #2614: a shared bar needs a shared colour map
  import matplotlib.pyplot as plt
  from mtex import mtexColorbar, setColorRange
  fig = hotAndCool()
  setColorRange("equal")
  assert len(mtexColorbar()) == 2
  plt.close("all")


def test_freshAxes_takesTheFirstAxesOfAFreshFigure():
  matplotlib = pytest.importorskip("matplotlib")
  matplotlib.use("Agg")
  from mtex.plotting.mtexfigure import _freshAxes
  fig = newMtexFigure(2)
  assert _freshAxes() == (fig, fig.children[0])


def test_freshAxes_makesANewFigureOfTheDefaultSizeOtherwise():
  matplotlib = pytest.importorskip("matplotlib")
  matplotlib.use("Agg")
  from mtex.plotting.mtexfigure import _freshAxes
  newMtexFigure(1).nextAxis()
  fig, ax = _freshAxes(default=(4.5, 3.2))
  assert tuple(fig.fig.get_size_inches()) == (4.5, 3.2) and fig.children == [ax]


@pytest.mark.parametrize("inner, outer", [(10, 40), (30, 15), (None, 25), (20, None), (8, 8)])
def test_the_plot_spacings_are_the_gap_between_the_axes_and_the_margin_around_them(inner, outer):
  # in pixels at 96 to the inch, as MATLAB's innerPlotSpacing and outerPlotSpacing
  import matplotlib.pyplot as plt
  agg()
  from mtex import mtexFigure
  f = mtexFigure(layout=(2, 2), figSize=(6, 5), innerPlotSpacing=inner, outerPlotSpacing=outer)
  axes = [f.nextAxis() for _ in range(4)]
  for ax in axes:
    ax.axis("off")
  f.fig.set_dpi(96)
  f.fig.canvas.draw()
  W, H = f.fig.get_size_inches() * 96
  a, b, c = (ax.get_position() for ax in axes[:3])
  if inner is not None:
    assert np.isclose((b.x0 - a.x1) * W, inner) and np.isclose((a.y0 - c.y1) * H, inner)
  if outer is not None:
    margin = max(outer, (inner or 0) / 2)
    assert np.allclose([a.x0 * W, (1 - b.x1) * W, (1 - a.y1) * H, c.y0 * H], margin)
  plt.close("all")
