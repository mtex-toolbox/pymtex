# the drawing of EBSD and grain maps (docs/plans/plotting.md S4), on the Agg backend
import os

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mtex import (EBSD, calcGrains, crystalFrame, degree, gcm, hold, ipfHSVKey, loadEBSD, phaseList, plot, quiver, rotation,
                  vector3d, xvector, zvector, mtexColorMap, mtexColorbar, setColorRange)
from mtex.plotting.mapplot import mapValues

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "ebsd", "Forsterite_cut.ctf")
CS = crystalFrame("m-3m", mineral="raster")


@pytest.fixture(autouse=True)
def closeFigures():
  yield
  plt.close("all")


@pytest.fixture(scope="module")
def forsterite():
  ebsd = loadEBSD(FIXTURE)
  return ebsd, calcGrains(ebsd["indexed"], angle=10 * degree)


def mapOf(ebsd):
  return gcm().children[-1]._mtexMapPlot


def gridMap(nx, ny, h):
  i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
  i, j = i.reshape(-1, order="F"), j.reshape(-1, order="F")
  pos = np.stack([h * i, h * j, np.zeros(nx * ny)], axis=-1)
  return EBSD(pos, rotation.byAxisAngle(zvector, (i + nx * j) * 3 * degree), np.ones(nx * ny, dtype=np.int8),
              phaseList(crystalFrame("1"), CS), prop={"bc": np.arange(nx * ny, dtype=float)})


def test_plot_ofAPhaseMapIsOnePolyCollectionWithItsLegend(forsterite):
  ebsd, _ = forsterite
  h = plot(ebsd, legend=True)
  fig = gcm()
  assert len(fig.children) == 1 and isinstance(h, matplotlib.collections.PolyCollection)
  assert fig.children[0].get_legend() is not None


# the layout of the session's convention flips y, and the scale bar with its triad is drawn
def test_plot_ofAMapFlipsYAndDrawsTheScaleBar(forsterite):
  ebsd, _ = forsterite
  plot(ebsd)
  mP = mapOf(ebsd)
  assert np.array_equal(mP.layout, [[1, 0], [0, -1]]) and len(mP.scaleBarArtists) > 5


# a value per measurement through the colour map, a colour bar, and the picture backend
def test_plot_ofValuesThroughImagescIsAPictureWithAColourBar(forsterite):
  ebsd, _ = forsterite
  h = plot(ebsd, ebsd.bc, colorbar=True, backend="imagesc")
  assert isinstance(h, matplotlib.image.AxesImage) and len(gcm().colorbars) == 1
  assert mapOf(ebsd).mappable is h


# orientations through the inverse pole figure key of their phase, and the key legend
def test_plot_ofOrientationsDrawsTheKeyLegend(forsterite):
  ebsd, _ = forsterite
  h = plot(ebsd, "orientation", legend=True)
  assert isinstance(h, matplotlib.collections.PolyCollection) and len(mapOf(ebsd).ax.child_axes) >= 1
  fo = ebsd["Forsterite"]
  plot(fo, fo.orientations)
  plot(fo, ipfHSVKey(fo.CS, ipfDirection=xvector))


# rgb per measurement, one colour, and the points backend
def test_plot_throughTheScatterBackendIsPointsWithoutAScaleBar(forsterite):
  ebsd, _ = forsterite
  plot(ebsd, np.random.default_rng(1).random((ebsd.size, 3)))
  h = plot(ebsd, faceColor="black", backend="scatter", micronbar=False)
  assert isinstance(h, matplotlib.collections.PathCollection) and not mapOf(ebsd).scaleBarArtists
  # the method is the function
  ebsd.plot(ebsd.bc)


def test_plot_refusesValuesOfTheWrongLengthAndAnUnknownBackend(forsterite):
  ebsd, _ = forsterite
  with pytest.raises(ValueError):
    plot(ebsd, np.zeros(5))
  with pytest.raises(ValueError):
    plot(ebsd, backend="volume")


def test_plot_throughImagescIsOnePixelPerCellWithTitleAndCoordinates():
  ebsd = gridMap(6, 4, 0.5)
  h = plot(ebsd, "bc", backend="imagesc", coordinates=True, title="grid")
  mP = mapOf(ebsd)
  assert isinstance(h, matplotlib.image.AxesImage) and h.get_array().shape == (4, 6)
  assert mP.ax.get_title() == "grid" and mP.ax.get_xlabel().startswith("X") and mP.ax.get_ylabel().startswith("-Y")


# the map under another convention is turned, not resampled
def test_plot_underAnotherConventionTurnsThePicture():
  ebsd = gridMap(6, 4, 0.5)
  h = plot(ebsd, "bc", backend="imagesc", how2plot="x↑→y")
  assert h.get_array().shape == (6, 4) and np.array_equal(mapOf(ebsd).layout, [[0, 1], [1, 0]])


# the cells with their edges, and one cell per measurement
def test_plot_throughThePatchBackendIsOneCellPerMeasurement():
  ebsd = gridMap(6, 4, 0.5)
  h = plot(ebsd, backend="patch", edgeColor="k")
  assert len(h.get_paths()) == 24


# cells without outlines are stroked in their own colour, which closes the seams between
# them; translucent cells are not, since their strokes would show
def test_plot_closesTheSeamsOfOpaqueCellsWithAStrokeInTheirOwnColour():
  ebsd = gridMap(6, 4, 0.5)
  bare, faint = plot(ebsd, backend="patch"), plot(ebsd, backend="patch", alpha=0.5)
  assert bare.get_linewidth()[0] > 0 and np.allclose(bare.get_edgecolor(), bare.get_facecolor())
  assert faint.get_linewidth()[0] == 0


def test_mapValues_byNameAndByArrayAgree():
  ebsd = gridMap(6, 4, 0.5)
  v1, s1 = mapValues(ebsd, "bc")
  v2, s2 = mapValues(ebsd, ebsd.bc)
  assert s1 and s2 and np.array_equal(v1, v2)


def test_mapValues_ofNothingAreThePhaseColours():
  v3, s3 = mapValues(gridMap(6, 4, 0.5), None)
  assert not s3 and v3.shape == (24, 3)


def test_plot_ofGrainsIsAPathPerGrainWithBoundaryAndScaleBar(forsterite):
  ebsd, g = forsterite
  h = plot(g)
  assert isinstance(h, matplotlib.collections.PathCollection) and len(h.get_paths()) == g.size
  mP = mapOf(g)
  assert len(mP.ax.lines) >= 2                      # the boundary and the scale bar


def test_plot_ofGrainValuesTakesAColourBar(forsterite):
  ebsd, g = forsterite
  h = plot(g, g.GOS, colorbar=True, boundary=False)
  assert mapOf(g).mappable is h and len(gcm().colorbars) == 1
  plot(g["Forsterite"], g["Forsterite"].meanOrientation)


# the boundary drawn on a held map widens it and keeps the bar
def test_plot_ofABoundaryOnAHeldMapWidensIt(forsterite):
  ebsd, g = forsterite
  plot(g["Forsterite"])
  mP = mapOf(g)
  before = mP.extent
  hold(True)
  h = plot(g.boundary, lineWidth=1.5, lineColor="r")
  hold(False)
  assert mapOf(g) is mP and isinstance(h, matplotlib.lines.Line2D)
  assert mP.extent[1] - mP.extent[0] >= before[1] - before[0]


# a value per segment, and a colour per segment
def test_plot_ofBoundaryValuesOrColoursIsALineCollection(forsterite):
  ebsd, g = forsterite
  gB = g.boundary[g.boundary.hasPhase("Forsterite", "Forsterite")]
  angles = gB.misorientation.angle()
  plot(ebsd, ebsd.bc, colormap="gray")
  hold(True)
  h = plot(gB, angles, lineWidth=2, colorbar=True)
  hold(False)
  assert h is None and len(gcm().colorbars) == 1 and isinstance(mapOf(gB).mappable, matplotlib.collections.LineCollection)
  h = plot(gB, np.random.default_rng(2).random((len(gB), 3)))
  assert isinstance(h, matplotlib.collections.LineCollection)
  with pytest.raises(ValueError):
    plot(gB, np.zeros(2))


# arrows: an axis per grain centred on it, a direction per measurement
def test_quiver_drawsAnAxisPerGrainAndADirectionPerMeasurement(forsterite):
  ebsd, g = forsterite
  plot(g)
  hold(True)
  h = quiver(g, g.longAxis(), color="w")
  hold(False)
  assert isinstance(h, matplotlib.collections.LineCollection) and len(h.get_segments()) <= g.size
  plot(ebsd)
  hold(True)
  h = quiver(ebsd, vector3d(np.ones((ebsd.size, 3))))
  hold(False)
  assert isinstance(h, matplotlib.quiver.Quiver)
  with pytest.raises(ValueError):
    quiver(g, vector3d(np.ones((3, 3))))


# two quivers without a colour take the first two of the colour order, as MATLAB's
def test_quiver_withoutAColourTakesTheColourOrder(forsterite):
  ebsd, g = forsterite
  plot(g)
  hold(True)
  h1 = quiver(g, g.longAxis(), displayName="long axis")
  h2 = quiver(g, g.longAxis(), displayName="again")
  hold(False)
  c1, c2 = (matplotlib.colors.to_hex(h.get_edgecolor()[0] if len(h.get_edgecolor()) else h.get_color()) for h in (h1, h2))
  assert c1 != c2 and c1 == matplotlib.colors.to_hex("C0") and h1.get_label() == "long axis"


# no pixel of the unindexed phase is painted, and none of them is opaque black
def test_plot_throughImagescPaintsNoUnindexedPixel(forsterite):
  ebsd = forsterite[0]
  ids = np.asarray(ebsd.phaseId).reshape(-1)
  assert np.any(ids == 0) and not ebsd.CSList.isIndexed(0)
  h = plot(ebsd, "orientation", backend="imagesc")
  img = h.get_array()
  assert img.dtype == np.uint8
  opaqueBlack = (img[..., 3] == 255) & np.all(img[..., :3] == 0, axis=-1)
  assert not np.any(opaqueBlack)


# the colours of the pixels alone, in the picture's order, are the colours of the map at
# those measurements
def test_mapValues_atThePixelsAreTheColoursOfTheMapThere(forsterite):
  from mtex.plotting.maps import mapRaster
  ebsd = forsterite[0]
  (_, _), (_, _), ind = mapRaster(ebsd, maxPixels=20_000)
  at = np.where(ind >= 0, ind, 0).reshape(-1)
  narrow, _ = mapValues(ebsd, "orientation", at=at)
  whole, _ = mapValues(ebsd, "orientation")
  assert narrow.shape == (at.size, 3)
  assert np.array_equal(np.nan_to_num(narrow, nan=-1.0), np.nan_to_num(whole[at], nan=-1.0))


# the picture and the limits span the measurements, and no pixel shows a padding cell
def test_plot_ofAGridShowsNoPaddingInThePictureOrTheLimits():
  from mtex.plotting.maps import mapBox, mapRaster
  from test_gridify import turnedMap
  g = turnedMap().gridify()
  box = mapBox(g)
  assert np.allclose([box[:, 0].min(), box[:, 0].max(), box[:, 1].min(), box[:, 1].max()], [0, 20, -10, 0])
  (_, _), (_, _), ind = mapRaster(g, maxPixels=20_000)
  assert not g.isPadding.reshape(-1)[ind[ind >= 0]].any()


# the cells and the points leave the padding and the colourless sites out
@pytest.mark.parametrize("backend, count", [("patch", lambda h: len(h.get_paths())), ("scatter", lambda h: len(h.get_offsets()))])
def test_plot_ofAGridLeavesPaddingAndColourlessSitesOut(backend, count):
  from mtex import ipfColorKey
  from test_gridify import turnedMap
  g = turnedMap().gridify()
  rgb = ipfColorKey(g).orientation2color(g.orientations)
  assert np.isnan(rgb[g.isPadding]).all()
  measured = g.size - g.isPadding.sum()
  plt.figure()
  assert count(plot(g, rgb, backend=backend)) == measured - 1
  assert count(plot(g, backend=backend)) == measured


# a boundary quiver is MATLAB's quiver3 of v and -v from every midpoint, the longest half
# 0.9 f times the spacing of the points
def test_quiver_ofBoundariesIsANeedleThroughEveryMidpoint(forsterite):
  ebsd, g = forsterite
  gB = g.boundary["Forsterite", "Forsterite"]
  plot(g)
  hold(True)
  v = vector3d(np.tile([1.0, 0, 0], (len(gB), 1)))
  h = quiver(gB, v, autoScaleFactor=0.4)
  hold(False)
  segs = np.array(h.get_segments())
  c = gB.midPoint.data.reshape(-1, 3)[:, :2]
  k = np.sqrt(2 * len(gB))
  spacing = np.sqrt(3) * np.hypot(np.ptp(c[:, 0]) / k, np.ptp(c[:, 1]) / k)
  # the half towards the viewer on top, starting at the midpoint; the other half beneath
  assert len(segs) == len(gB) and np.allclose(np.abs(segs[:, 0]), np.abs(c))
  assert np.allclose(np.linalg.norm(segs[:, 1] - segs[:, 0], axis=-1), 0.4 * 0.9 * spacing)
  assert h.get_zorder() > 1 and min(x.get_zorder() for x in gcm().children[-1].collections) < 1


# translucent fades a grain by its colour: pale grains let the map below show
def test_plot_translucentFadesAGrainByItsColour(forsterite):
  ebsd, g = forsterite
  h = plot(g, translucent=1)
  rgb = np.asarray(h.get_facecolors())[:, :3]
  assert np.allclose(h.get_alpha(), 1 - rgb.min(axis=1))


# legend('off') removes the legend of the phase map
def test_legend_offRemovesTheLegendOfThePhaseMap(forsterite):
  from mtex import legend
  ebsd, g = forsterite
  plot(g)
  assert gcm().children[-1].get_legend() is not None
  legend("off")
  assert gcm().children[-1].get_legend() is None


def test_triple_points_join_the_legend_and_none_leave_the_limits(forsterite):
  ebsd, g = forsterite
  plot(g)
  hold(True)
  plot(g.triplePoints, color="b")
  mP = mapOf(g)
  n = len(mP.ax.get_legend().get_texts())
  limits = mP.ax.get_xlim(), mP.ax.get_ylim()
  plot(g.triplePoints[:3], displayName="three", color="r")
  plot(g.triplePoints[np.zeros(len(g.triplePoints), dtype=bool)], color="m")
  hold(False)
  assert [t.get_text() for t in mP.ax.get_legend().get_texts()][-1] == "three" and len(mP.ax.get_legend().get_texts()) == n + 1
  assert (mP.ax.get_xlim(), mP.ax.get_ylim()) == limits


def test_an_infinite_value_takes_the_end_of_the_colour_map():
  ebsd = gridMap(4, 3, 1.0)
  v = np.arange(ebsd.size, dtype=float)
  v[0], v[1], v[2] = -np.inf, np.inf, np.nan
  plot(ebsd, v, backend="imagesc")
  img = gcm().children[-1].images[0].get_array()
  assert np.ma.count_masked(img) == 1


def test_grains_of_one_face_colour_name_no_phases(forsterite):
  ebsd, g = forsterite
  plot(g, faceColor=[0.8, 0.8, 0.8])
  assert mapOf(g).ax.get_legend() is None
  plot(g[0], faceColor=[0.2, 0.8, 0.8], displayName="largest grain")
  assert [t.get_text() for t in mapOf(g[0]).ax.get_legend().get_texts()] == ["largest grain"]


def test_a_colour_map_and_range_reach_every_boundary_coloured(forsterite):
  ebsd, g = forsterite
  gB = g.boundary
  plot(gB[:50], np.arange(50.0))
  hold(True)
  plot(gB[50:100], np.arange(50.0) + 100)
  hold(False)
  mtexColorMap("blue2red")
  setColorRange([-1, 1])
  ms = mapOf(g).mappables
  assert len(ms) == 2 and all(m.get_clim() == (-1, 1) and m.get_cmap().name == ms[0].get_cmap().name for m in ms)
  assert len(mtexColorbar()) == 1


def test_a_fresh_figure_takes_the_shape_of_its_map():
  from mtex import newMtexFigure
  ebsd = gridMap(40, 10, 1.0)
  newMtexFigure(figSize="large")
  plot(ebsd, ebsd.bc)
  w, h = gcm().fig.get_size_inches()
  assert w > 2 * h and np.isclose(w, 1.3 * 7.0)


def test_the_outline_of_grains_takes_their_face_colour(forsterite):
  # MATLAB passes FaceColor on to the boundary, so a grain map of one colour has no outline
  import matplotlib.colors as mc
  ebsd, g = forsterite
  plot(g, faceColor=[0.9, 0.9, 0.9])
  line = gcm().children[-1].lines[0]
  assert np.allclose(mc.to_rgb(line.get_color()), [0.9, 0.9, 0.9])
  plot(g, faceColor=[0.9, 0.9, 0.9], lineColor="r")
  assert mc.to_hex(gcm().children[-1].lines[0].get_color()) == "#ff0000"
  plot(g, faceColor="none")
  assert mc.to_hex(gcm().children[-1].lines[0].get_color()) == "#000000"


def test_a_thin_segment_shows_its_alpha_twice(forsterite):
  # MATLAB draws a thin segment there and back, so alpha a shows as 1 - (1 - a)^2
  ebsd, g = forsterite
  gB = g.boundary
  a = np.linspace(0, 1, len(gB))
  h = plot(gB, edgeAlpha=a, lineWidth=1.5)
  assert np.allclose(h.get_colors()[:, 3], 1 - (1 - a) ** 2)
  h = plot(gB, edgeAlpha=a, lineWidth=4)
  assert np.allclose(h.get_colors()[:, 3], a)


# a zoom takes the scale bar into the corner of the new view, as MATLAB redraws its bar
def test_the_scale_bar_follows_a_zoom():
  plt.close("all")
  e = gridMap(40, 30, 1.0)
  plot(e, e.orientations)
  ax = plt.gca()
  bar = lambda: [a for a in ax.patches if a.get_zorder() == 10][0]
  x0 = bar().get_x()
  ax.set_xlim(20, 30)
  ax.set_ylim(10, 18)
  box = bar()
  assert box.get_x() != x0 and 20 <= box.get_x() <= 30 and 10 <= box.get_y() <= 18
  assert len([a for a in ax.patches if a.get_zorder() == 10]) == 1
  plt.close("all")


# an image is drawn where it sits, under the convention of its frame: an image of its own
# row 0 at the top
def test_plot_ofAnImageDrawsItOnTheSpecimenRowZeroOnTop():
  from mtex import mapImage
  img = np.arange(12.0).reshape(3, 4)
  mg = mapImage(img, dxy=0.5, origin=vector3d(1, 2, 0), name="bse")
  h = plot(mg)
  ax = h.axes
  assert np.allclose(h.get_extent(), (0.75, 2.75, -3.25, -1.75))
  # row 0 at the top of the screen: the array is drawn flipped, origin lower
  assert np.array_equal(np.asarray(h.get_array())[::-1], img)
  assert ax.get_xlim() == (0.75, 2.75)


# a list of images, one axes each, titled by name
def test_plot_ofAnImageListIsOneAxesPerImageTitledByName():
  from mtex import mapImage, mapImageList
  mg = mapImage(np.arange(12.0).reshape(3, 4), dxy=0.5, origin=vector3d(1, 2, 0), name="bse")
  hs = plot(mapImageList([mg, mapImage(np.random.rand(3, 4, 3), name="rgb")]), layout=[1, 2])
  assert len(hs) == 2 and [a.get_title() for a in gcm().children] == ["bse", "rgb"]
  assert hs[1].get_array().shape == (3, 4, 4)


def test_plot_ofAnImageWithEdgeKeepsItsShape():
  from mtex import mapImage
  mg = mapImage(np.arange(12.0).reshape(3, 4), dxy=0.5, origin=vector3d(1, 2, 0), name="bse")
  e = plot(mg, edge=True)
  assert e.get_array().shape == (3, 4)


# the key of a group without the inversion whose sector reaches into both hemispheres takes
# two insets in the legend, the upper over the lower; a Laue group's one
def test_the_key_legend_draws_both_hemispheres():
  from mtex.plotting.mapplot import keyLegend
  pl = phaseList(crystalFrame("mm2", mineral="polar"), crystalFrame("m-3m", mineral="cubic"))
  i, j = np.meshgrid(np.arange(10), np.arange(10))
  pos = np.column_stack([i.ravel(), j.ravel(), np.zeros(100)])
  e = EBSD(pos, rotation.rand(100), np.where(i.ravel() < 5, 1, 2).astype(np.int8), pl)
  plot(e, "orientation")
  axes = keyLegend(plt.gcf().axes[0]._mtexMapPlot, e)
  assert len(axes) == 3 and axes[0].get_title() == "polar" and axes[2].get_title() == "cubic"
  upper, lower = axes[0]._mtexSphericalPlot, axes[1]._mtexSphericalPlot
  assert upper.proj.upper and not lower.proj.upper
  assert axes[0].get_position().y0 > axes[1].get_position().y1


def test_an_ipf_direction_alone_colours_by_orientation(forsterite):
  # MATLAB's plot(ebsd, 'ipfDirection', d): the orientations along d, not the phases
  ebsd, grains = forsterite
  rgbOri, _ = mapValues(ebsd, "orientation", ipfDirection=xvector)
  rgbDir, scalar = mapValues(ebsd, None, ipfDirection=xvector)
  rgbPhase, _ = mapValues(ebsd, None)
  assert not scalar and np.allclose(rgbDir, rgbOri, equal_nan=True) and not np.allclose(rgbDir, rgbPhase, equal_nan=True)
  assert np.allclose(mapValues(grains, None, ipfDirection=xvector)[0], mapValues(grains, "orientation", ipfDirection=xvector)[0],
                     equal_nan=True)
  # the map carries no phase legend then
  plot(ebsd, ipfDirection=xvector)
  assert gcm().children[-1].get_legend() is None
  # a face colour still wins over the direction
  rgbFace, _ = mapValues(ebsd, None, faceColor="red", ipfDirection=xvector)
  assert np.allclose(rgbFace, [1, 0, 0])


def test_matlab_ipf_flags_point_to_the_keyword(forsterite):
  ebsd, _ = forsterite
  with pytest.raises(ValueError, match="ipfDirection=zvector"):
    plot(ebsd, "ipfz")


def test_grainPaths_isEveryGrainWithItsLoopsClosed(forsterite):
  from mtex.plotting.mapplot import _grainPaths
  from mtex.plotting.maps import grainPolygons
  _, grains = forsterite
  plot(grains, micronbar=False)
  paths, at = _grainPaths(grains, mapOf(grains))
  k = max(range(len(at)), key=lambda i: len(grains.loops[at[i]]))
  rings = grainPolygons(grains, at[k])
  assert len(paths[k].vertices) == sum(len(r) + 1 for r in rings) and list(paths[k].codes).count(1) == len(rings)


def test_grainPaths_ofNoGrainIsEmpty(forsterite):
  from mtex.plotting.mapplot import _grainPaths
  _, grains = forsterite
  plot(grains, micronbar=False)
  assert _grainPaths(grains[[]], mapOf(grains)) == ([], [])


def test_needles_takeOneSegmentInThePlaneAndTwoOutOfIt(forsterite):
  _, grains = forsterite
  g = grains[:2]
  h = plot(g, vector3d([[1, 0, 0], [0, 0, 1]]), micronbar=False)
  assert len(h.get_segments()) == 3 and h.get_colors()[0][0] == 0.5


def test_mapPlot_outOfScreen_turnsWithAFlippingLayout(forsterite):
  ebsd, _ = forsterite
  plot(ebsd, micronbar=False)
  mP = mapOf(ebsd)
  assert np.allclose(mP.outOfScreen, mP.plane.normal * np.sign(np.linalg.det(mP.layout)))


def test_onOff_readsMATLABsSwitches():
  from mtex.plotting.mapplot import _onOff
  assert (_onOff("on"), _onOff("off"), _onOff(None)) == (True, False, None)


@pytest.mark.parametrize("values", [False, True])
@pytest.mark.parametrize("smooth", [False, True])
def test_plot_ofABoundaryWithSmoothRoundsItsJoins(forsterite, values, smooth):
  _, grains = forsterite
  gB = grains.boundary
  h = plot(gB, np.arange(len(gB)) if values else None, lineWidth=4, smooth=smooth)
  join = h.get_joinstyle() if values else h.get_solid_joinstyle()
  assert (join == "round") == smooth


@pytest.mark.parametrize("name", ["area", "perimeter", "equivalentRadius", "diameter", "numNeighbors", "aspectRatio",
                                  "shapeFactor", "paris", "convexity"])
def test_mapValues_ofTheNameOfAGrainMeasureAreItsValues(forsterite, name):
  _, grains = forsterite
  value = getattr(grains, name)
  values, scalar = mapValues(grains, name)
  assert scalar and np.allclose(values, np.asarray(value() if callable(value) else value, dtype=float).reshape(-1), equal_nan=True)
  # and the plot colours the grains by them
  assert plot(grains, name).get_array() is not None


def test_mapValues_ofAMethodThatIsNoMeasureIsAColourName(forsterite):
  _, grains = forsterite
  with pytest.raises(ValueError):
    mapValues(grains, "copy")
