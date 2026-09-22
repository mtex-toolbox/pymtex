# the drawing of EBSD and grain maps (docs/plans/plotting.md S4), on the Agg backend
import os

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mtex import (EBSD, calcGrains, crystalFrame, degree, gcm, hold, ipfHSVKey, loadEBSD, phaseList, plot, quiver, rotation,
                  vector3d, xvector, zvector)
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


def test_an_ebsd_map_by_phase_value_and_orientation(forsterite):
  ebsd, _ = forsterite
  h = plot(ebsd, legend=True)
  fig = gcm()
  mP = mapOf(ebsd)
  assert len(fig.children) == 1 and isinstance(h, matplotlib.collections.PolyCollection)
  # the layout of the session's convention flips y, and the scale bar with its triad is drawn
  assert np.array_equal(mP.layout, [[1, 0], [0, -1]]) and len(mP.scaleBarArtists) > 5
  assert fig.children[0].get_legend() is not None
  # a value per measurement through the colour map, a colour bar, and the picture backend
  h = plot(ebsd, ebsd.bc, colorbar=True, backend="imagesc")
  assert isinstance(h, matplotlib.image.AxesImage) and len(gcm().colorbars) == 1
  assert mapOf(ebsd).mappable is h
  # orientations through the inverse pole figure key of their phase, and the key legend
  h = plot(ebsd, "orientation", legend=True)
  assert isinstance(h, matplotlib.collections.PolyCollection) and len(mapOf(ebsd).ax.child_axes) >= 1
  fo = ebsd["Forsterite"]
  plot(fo, fo.orientations)
  plot(fo, ipfHSVKey(fo.CS, ipfDirection=xvector))
  # rgb per measurement, one colour, and the points backend
  plot(ebsd, np.random.default_rng(1).random((ebsd.size, 3)))
  h = plot(ebsd, faceColor="black", backend="scatter", micronbar=False)
  assert isinstance(h, matplotlib.collections.PathCollection) and not hasattr(mapOf(ebsd), "scaleBarArtists")
  with pytest.raises(ValueError):
    plot(ebsd, np.zeros(5))
  with pytest.raises(ValueError):
    plot(ebsd, backend="volume")
  # the method is the function
  ebsd.plot(ebsd.bc)


def test_the_picture_the_cells_and_the_layout():
  ebsd = gridMap(6, 4, 0.5)
  h = plot(ebsd, "bc", backend="imagesc", coordinates=True, title="grid")
  mP = mapOf(ebsd)
  assert isinstance(h, matplotlib.image.AxesImage) and h.get_array().shape == (4, 6)
  assert mP.ax.get_title() == "grid" and mP.ax.get_xlabel().startswith("X") and mP.ax.get_ylabel().startswith("-Y")
  # the map under another convention is turned, not resampled
  h = plot(ebsd, "bc", backend="imagesc", how2plot="x↑→y")
  assert h.get_array().shape == (6, 4) and np.array_equal(mapOf(ebsd).layout, [[0, 1], [1, 0]])
  # the cells with their edges, and one cell per measurement
  h = plot(ebsd, backend="patch", edgeColor="k")
  assert len(h.get_paths()) == 24
  # values by name and by array agree
  v1, s1 = mapValues(ebsd, "bc")
  v2, s2 = mapValues(ebsd, ebsd.bc)
  assert s1 and s2 and np.array_equal(v1, v2)
  v3, s3 = mapValues(ebsd, None)
  assert not s3 and v3.shape == (24, 3)


def test_grains_boundaries_and_arrows(forsterite):
  ebsd, g = forsterite
  h = plot(g)
  assert isinstance(h, matplotlib.collections.PathCollection) and len(h.get_paths()) == g.size
  mP = mapOf(g)
  assert len(mP.ax.lines) >= 2                      # the boundary and the scale bar
  h = plot(g, g.GOS, colorbar=True, noBoundary=True)
  assert mapOf(g).mappable is h and len(gcm().colorbars) == 1
  plot(g["Forsterite"], g["Forsterite"].meanOrientation)
  # the boundary drawn on a held map widens it and keeps the bar
  plot(g["Forsterite"])
  mP = mapOf(g)
  before = mP.extent
  hold(True)
  h = plot(g.boundary, lineWidth=1.5, lineColor="r")
  hold(False)
  assert mapOf(g) is mP and isinstance(h, matplotlib.lines.Line2D)
  assert mP.extent[1] - mP.extent[0] >= before[1] - before[0]
  # a value per segment, and a colour per segment
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


def test_an_unindexed_measurement_is_drawn_as_nothing_and_the_picture_colours_what_it_shows(forsterite):
  import numpy as np
  from mtex import plot
  from mtex.plotting.mapplot import mapValues
  ebsd = forsterite[0]
  ids = np.asarray(ebsd.phaseId).reshape(-1)
  assert np.any(ids == 0) and not ebsd.CSList.isIndexed(0)
  h = plot(ebsd, "orientation", backend="imagesc")
  img = h.get_array()
  assert img.dtype == np.uint8
  # no pixel of the unindexed phase is painted, and none of them is opaque black
  opaqueBlack = (img[..., 3] == 255) & np.all(img[..., :3] == 0, axis=-1)
  assert not np.any(opaqueBlack)
  # the colours of the pixels alone, in the picture's order, are the colours of the map at
  # those measurements
  from mtex.plotting.maps import mapRaster
  (_, _), (_, _), ind = mapRaster(ebsd, maxPixels=20_000)
  at = np.where(ind >= 0, ind, 0).reshape(-1)
  narrow, _ = mapValues(ebsd, "orientation", at=at)
  whole, _ = mapValues(ebsd, "orientation")
  assert narrow.shape == (at.size, 3)
  assert np.array_equal(np.nan_to_num(narrow, nan=-1.0), np.nan_to_num(whole[at], nan=-1.0))


def test_the_padding_of_a_grid_is_not_drawn():
  from mtex import ipfColorKey
  from mtex.plotting.maps import mapBox, mapRaster
  from test_gridify import turnedMap
  g = turnedMap().gridify()
  # the picture and the limits span the measurements, and no pixel shows a padding cell
  box = mapBox(g)
  assert np.allclose([box[:, 0].min(), box[:, 0].max(), box[:, 1].min(), box[:, 1].max()], [0, 20, -10, 0])
  (_, _), (_, _), ind = mapRaster(g, maxPixels=20_000)
  assert not g.isPadding.reshape(-1)[ind[ind >= 0]].any()
  # the cells and the points leave the padding and the colourless sites out
  rgb = ipfColorKey(g).orientation2color(g.orientations)
  assert np.isnan(rgb[g.isPadding]).all()
  measured = g.size - g.isPadding.sum()
  for backend, count in (("patch", lambda h: len(h.get_paths())), ("scatter", lambda h: len(h.get_offsets()))):
    plt.figure()
    assert count(plot(g, rgb, backend=backend)) == measured - 1
    assert count(plot(g, backend=backend)) == measured
