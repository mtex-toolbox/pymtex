# render3d: a scene drawn by PyVista offscreen, on matplotlib's 3D axes, and as the data of an
# interactive three.js page
import base64
import json
import zlib

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from mtex.plotting.render3d import _payload, drawMatplotlib, hasPyVista, renderImage, sceneHTML
from mtex.plotting.scene3d import arrows3d, mesh3d, picture3d, scene3d, slices3d


# a red unit square in the xy plane seen from above, and a slice and a picture beside it
def square():
  s = scene3d()
  s.add(mesh3d([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [0, 4], [0, 1, 2, 3], (1, 0, 0), edgeAlpha=0.5))
  return s


def inflate(b64, dtype):
  return np.frombuffer(zlib.decompress(base64.b64decode(b64)), dtype=dtype)


def test_the_page_data_decode_to_the_scene():
  s = square()
  s.add(slices3d(np.zeros((2, 3, 1, 3)), np.ones((2, 3, 1)), np.zeros(3), np.eye(3), at=(None, None, 0)))
  s.add(picture3d(np.ones((4, 5, 4)), [[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]))
  for it in arrows3d([[0, 0, 0]], [[0, 0, 1]]):
    s.add(it)
  d = _payload(s)
  kinds = [it["kind"] for it in d["items"]]
  assert kinds == ["mesh", "slice", "picture", "lines", "mesh"]
  m = d["items"][0]
  V = inflate(m["V"], np.float32).reshape(-1, 3)
  T = inflate(m["T"], np.uint32).reshape(-1, 3)
  C = inflate(m["C"], np.uint8).reshape(-1, 3)
  assert len(T) == 2 and np.all(C == [255, 0, 0]) and len(inflate(m["E"], np.uint32)) == 8
  # the vertices about the centre of the scene
  lo, hi = s.bounds()
  assert np.allclose(V + (lo + hi) / 2, [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], atol=1e-6)
  sl = d["items"][1]
  assert (sl["w"], sl["h"]) == (2, 3) and len(inflate(sl["rgba"], np.uint8)) == 2 * 3 * 4
  assert np.isclose(np.linalg.norm(d["out"]), 1)


def test_the_page_is_standalone_but_for_three():
  html = sceneHTML(square(), title="square")
  assert html.startswith("<!doctype html>") and "<title>square</title>" in html
  assert "cdn.jsdelivr.net/npm/three@" in html and "TrackballControls" in html
  data = html.split('<script id="scene" type="application/json">')[1].split("</script>")[0]
  assert json.loads(data)["items"][0]["kind"] == "mesh"


def test_matplotlib_draws_the_scene():
  import matplotlib.pyplot as plt
  fig = plt.figure()
  ax = fig.add_subplot(projection="3d")
  drawMatplotlib(square(), ax)
  assert len(ax.collections) == 2 and np.allclose(ax.get_xlim(), (0, 1))
  plt.close(fig)


@pytest.mark.skipif(not hasPyVista(), reason="PyVista, the plot3d extra")
def test_pyvista_renders_the_square_unlit():
  img = renderImage(square(), (200, 200))
  assert img.shape == (200, 200, 3)
  # the face in its own colour, unlit, and white around it
  red = np.all(img == [255, 0, 0], axis=-1)
  assert red.sum() > 1000 and np.all(img[0, 0] == 255)


def test_sliceGrid_spansTheRectangleOfTheSlice():
  from mtex.plotting.render3d import _sliceGrid
  P = _sliceGrid(np.array([[0, 0, 0], [2, 0, 0], [0, 3, 0], [2, 3, 0]], dtype=float), 2, 3)
  assert P.shape == (3, 4, 3) and P[-1, -1].tolist() == [2, 3, 0]
