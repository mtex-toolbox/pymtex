# volumeplot: plot and quiver of 3D grains, boundaries and volumes into a figure holding a
# scene, hold and setCamera on it, maps of other planes lifted into space, the HTML export
import os

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from mtex import (EBSD3square, calcGrains, crystalFrame, degree, exportHTML, gcm, hold, phaseList, plane3d, plot, plottingConvention, quiver,
                  rotation, setCamera, vector3d)
from mtex.plotting.render3d import hasPyVista
from mtex.plotting.scene3d import picture3d
from mtex.plotting.volumeplot import gcs

pytestmark = pytest.mark.skipif(not hasPyVista(), reason="PyVista, the plot3d extra")


def volume():
  r = np.zeros((4, 3, 2, 4))
  r[..., 0] = 1
  r[2:] = rotation.byAxisAngle(vector3d(1, 0, 0), 30 * degree).data
  return EBSD3square(rotation(r), np.ones((4, 3, 2), dtype=np.int8), phaseList([crystalFrame("m-3m", mineral="Iron")]), (1.0, 1.0, 1.0))


@pytest.fixture(autouse=True)
def fresh():
  import matplotlib.pyplot as plt
  plt.close("all")
  yield
  hold(False)
  plt.close("all")


def test_plot_of_grains_is_a_figure_holding_a_scene():
  g = calcGrains(volume(), angle=10 * degree)
  fig = plot(g, g.meanOrientation, edgeAlpha=0.1, micronbar="off")
  assert fig is gcm() and len(gcs().items) == 1
  assert fig.fig._mtexScene is gcs() and fig.children[0].images
  # hold adds to the scene, quiver draws shafts and heads
  hold(True)
  quiver(g, g.centroid * 0 + vector3d(0, 0, 1))
  plot(g.boundary, faceColor="red")
  hold(False)
  assert len(gcs().items) == 4
  v = fig.fig._mtexSceneVersion
  setCamera(plottingConvention(vector3d(0, 0, 1), vector3d(1, 0, 0)))
  assert fig.fig._mtexSceneVersion == v + 1 and np.allclose(gcs().camera()[0], [0, 0, 1])
  # a new plot without hold is a new figure
  plot(g)
  assert gcm() is not fig


def test_plot_of_a_volume_shows_its_slices():
  e = volume()
  plot(e, e.orientations)
  s = gcs().items[0]
  assert s.visible.shape == (4, 3, 2)


def test_maps_of_other_planes_go_into_space():
  e = volume()
  c = vector3d(1.5, 1, 0.5)
  plot(e.slice(plane3d(vector3d.Z, c)), micronbar="off")
  assert gcs() is None
  hold(True)
  plot(e.slice(plane3d(vector3d.X, c)), micronbar="off")
  assert gcs() is not None and len(gcs().items) == 2
  # a map in a plane the scene shows already is drawn into that picture
  s = e.slice(plane3d(vector3d.X, c))
  plot(s, s.orientations)
  hold(False)
  pics = [it for it in gcs().items if isinstance(it, picture3d)]
  assert len(pics) == 2 and len(pics[1].sources) == 2
  # the picture of the x slice lies in the plane x = 1.5 and spans the volume in y
  P = pics[1].corners
  assert np.allclose(P[:, 0], P[0, 0]) and abs(P[0, 0] - 1.5) <= 0.5
  assert P[:, 1].min() <= -0.5 + 1e-6 and P[:, 1].max() >= 2.5 - 1e-6


def test_setCamera_turns_a_map_into_a_scene():
  g = calcGrains(volume(), angle=10 * degree)
  g2 = g.slice(plane3d(vector3d(1, 1, 1), vector3d(1.5, 1, 0.5)))
  plot(g2, g2.meanOrientation, micronbar="off")
  assert gcs() is None
  setCamera(plottingConvention.default3D)
  assert gcs() is not None and isinstance(gcs().items[0], picture3d)
  n = np.cross(*(gcs().items[0].corners[1:3] - gcs().items[0].corners[0]))
  assert np.allclose(np.abs(n / np.linalg.norm(n)), 1 / np.sqrt(3))


def test_exportHTML_writes_the_scene(tmp_path):
  g = calcGrains(volume(), angle=10 * degree)
  plot(g)
  f = exportHTML(str(tmp_path / "g.html"))
  assert os.path.getsize(f) > 1000 and "three" in open(f).read()


def test_the_histogram_of_3d_grains_is_weighted_by_volume():
  from mtex import histogram
  g = calcGrains(volume(), angle=10 * degree)
  hs = histogram(g, g.volume, [0, 10, 20])
  ax = hs[0][0].axes
  # two grains of 12 voxels each, both in the second bin
  assert np.allclose([b.get_height() for b in hs[0]], [0, 100]) and ax.get_ylabel() == "relative volume (%)"
