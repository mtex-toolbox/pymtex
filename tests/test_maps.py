# the geometry of a map plot (docs/plans/plotting.md S4): the raster, the cells, the scale bar
# and the frame triad, with the numbers of ../mtexJulia/test/maps.jl P10 and S4b
import os

import numpy as np
import pytest

from mtex import EBSD, crystalFrame, degree, loadEBSD, phaseList, plottingConvention, rotation, vector3d, xvector, yvector, zvector
from mtex.plotting.maps import (boundaryPolyline, boundarySegments, frameOnScreen, mapCells, mapExtent, mapLayout, mapPoints,
                                mapRaster, phasePalette, scaleBar, switchUnit, triadGeometry)

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "ebsd", "Forsterite_cut.ctf")
CS = crystalFrame("m-3m", mineral="raster")


def gridMap(nx, ny, h, A=None):
  i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
  i, j = i.reshape(-1, order="F"), j.reshape(-1, order="F")
  xy = np.stack([h * i, h * j], axis=-1) if A is None else np.stack([i, j], axis=-1) @ np.asarray(A).T
  pos = np.concatenate([xy, np.zeros((len(xy), 1))], axis=1)
  rot = rotation.byAxisAngle(zvector, (i + nx * j) * 3 * degree)
  return EBSD(pos, rot, np.ones(nx * ny, dtype=np.int8), phaseList(crystalFrame("1"), CS))


def test_P10_the_scale_bar():
  assert scaleBar(360.0, "um") == (25.0, "25 µm")
  assert scaleBar(36000.0, "um") == (5000.0, "5 mm")
  assert scaleBar(3.6, "um") == (0.5, "500 nm")
  assert scaleBar(1200.0, "nm") == (125.0, "125 nm")
  for w in (1.0, 2.7, 13.0, 55.0, 480.0, 9300.0, 1.2e5):
    length, _ = scaleBar(w, "um")
    assert 0.02 * w <= length <= 0.25 * w
  for x, u in ((2500.0, "um"), (0.004, "mm"), (7.0, "nm")):
    y, _, f = switchUnit(x, u)
    assert np.isclose(y, x * f) and (1 <= y < 1000 or y == x)
  ebsd = loadEBSD(FIXTURE)
  e = mapExtent(ebsd)
  length, label = scaleBar(ebsd)
  assert 0.02 * (e[1] - e[0]) <= length <= 0.25 * (e[1] - e[0]) and "m" in label


def test_S4b_a_map_on_a_lattice_is_a_picture():
  nx, ny, h = 5, 4, 0.5
  ebsd = gridMap(nx, ny, h)
  r = mapRaster(ebsd)
  assert r is not None
  (xlo, xhi), (ylo, yhi), index = r
  # the picture spans the cells, half a cell beyond the outermost centres, one pixel per cell
  assert np.allclose((xlo, xhi), (-h / 2, h * (nx - 1) + h / 2)) and np.allclose((ylo, yhi), (-h / 2, h * (ny - 1) + h / 2))
  assert index.shape == (ny, nx)
  # every pixel names the measurement that sits there, in the map's own order
  assert np.array_equal(index, np.arange(nx * ny).reshape(ny, nx))
  # a hole is a pixel no measurement covers, and it says so with -1
  holed = ebsd[np.arange(nx * ny) != 6]
  ih = mapRaster(holed)[2]
  assert np.sum(ih < 0) == 1 and len(np.unique(ih)) == nx * ny
  # the cap keeps the shape of the map
  small = mapRaster(ebsd, maxPixels=6)[2]
  assert small.size <= 6 and small.shape[1] >= small.shape[0]
  # a hexagonal lattice lands on the same rectangle, a pixel painted for the cell it lies in
  A = np.array([[1.0, 0.5], [0.0, np.sqrt(3) / 2]])
  hebsd = gridMap(5, 4, 1.0, A)
  hr = mapRaster(hebsd)
  assert hr is not None and hebsd.lattice.shape == "hex"
  (hx, hy, hindex) = hr
  assert len(np.unique(hindex[hindex >= 0])) == 20
  H = hebsd.lattice.plane.to2d(hebsd.pos.data.reshape(-1, 3))
  ny_, nx_ = hindex.shape
  px = hx[0] + (np.arange(nx_) + 0.5) * (hx[1] - hx[0]) / nx_
  py = hy[0] + (np.arange(ny_) + 0.5) * (hy[1] - hy[0]) / ny_
  for j in range(0, ny_, 3):
    for i in range(0, nx_, 3):
      if hindex[j, i] < 0:
        continue
      q = np.array([px[i], py[j]])
      d = np.linalg.norm(H - q, axis=1)
      assert np.linalg.norm(H[hindex[j, i]] - q) <= d.min() + 1e-9
  assert 0.8 < np.sum(hindex >= 0) * 0.25 ** 2 / (20 * np.sqrt(3) / 2) < 1.2
  # a lattice the stage drifted over is not a picture, and its cells tile
  dn, dd = 10, 0.1
  i, j = np.meshgrid(np.arange(dn), np.arange(dn), indexing="ij")
  i, j = i.reshape(-1, order="F"), j.reshape(-1, order="F")
  dpos = np.stack([i * (1 + dd * j / dn), j * (1 + dd * i / dn), np.zeros(dn * dn)], axis=-1)
  debsd = EBSD(dpos, rotation.id.data * np.ones((dn * dn, 1)), np.ones(dn * dn, dtype=np.int8), phaseList(crystalFrame("1"), CS))
  assert debsd.lattice.shape == "square" and mapRaster(debsd) is None
  corners = mapCells(debsd)
  assert corners.shape == (dn * dn, 4, 2)
  assert len(np.unique(np.round(corners.reshape(-1, 2), 8), axis=0)) == (dn + 1) ** 2
  P = debsd.lattice.plane.to2d(dpos)
  node = (P[3 + dn * 4] + P[4 + dn * 4] + P[3 + dn * 5] + P[4 + dn * 5]) / 4
  assert np.min(np.linalg.norm(corners.reshape(-1, 2) - node, axis=1)) < 1e-10
  # a rigid grid keeps the ideal cell
  rc = mapCells(ebsd)
  assert np.all((np.abs(np.mod(rc[..., 0] + h / 2, h)) < 1e-12) | (np.abs(np.mod(rc[..., 0], h)) < 1e-12))
  # a map that is not on a lattice has no picture
  one = EBSD(np.zeros((1, 3)), rotation.id.data.reshape(1, 4), np.ones(1, dtype=np.int8), phaseList(crystalFrame("1"), CS))
  assert mapRaster(one) is None
  with pytest.raises(ValueError):
    mapCells(one)


def test_S4b_the_picture_under_a_layout():
  nx, ny, h = 5, 4, 0.5
  ebsd = gridMap(nx, ny, h)
  index = mapRaster(ebsd)[2]
  flip = mapLayout(plottingConvention(-zvector, xvector))   # y down
  swap = mapLayout(plottingConvention(-zvector, yvector))   # x up
  assert np.array_equal(flip, [[1, 0], [0, -1]]) and np.array_equal(swap, [[0, 1], [1, 0]])
  assert np.array_equal(mapRaster(ebsd, layout=flip)[2], index[::-1])
  assert np.array_equal(mapRaster(ebsd, layout=swap)[2], index.T)
  sx, sy, _ = mapRaster(ebsd, layout=swap)
  assert np.allclose(sx, (-h / 2, h * (ny - 1) + h / 2)) and np.allclose(sy, (-h / 2, h * (nx - 1) + h / 2))
  e, es = mapExtent(ebsd), mapExtent(ebsd, layout=swap)
  assert es == (e[2], e[3], e[0], e[1])
  assert np.allclose(mapCells(ebsd, layout=flip)[..., 1], -mapCells(ebsd)[..., 1])
  assert np.allclose(mapPoints(ebsd, layout=flip), mapPoints(ebsd) * [1, -1])
  # the default layout is the session's convention, y down under ij
  assert np.array_equal(mapLayout(), [[1, 0], [0, -1]])
  # the frame on the screen: swapping the axes swaps the arrows, flipping the map puts the
  # viewer on the other side of the plane
  pl = ebsd.lattice.plane
  assert frameOnScreen(pl, swap)[0] == (0.0, 1.0, 0.0) and frameOnScreen(pl, swap)[1] == (1.0, 0.0, 0.0)
  assert np.isclose(frameOnScreen(pl, swap)[2][2], -1) and np.isclose(frameOnScreen(pl, flip)[2][2], -1)
  assert np.isclose(frameOnScreen(pl)[2][2], 1)


def test_the_frame_triad():
  arrows, symbol, texts, bbox = triadGeometry([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)], ["x", "y", "z"])
  assert len(arrows) == 2 and all(len(a) == 7 for a in arrows)
  assert len(symbol) and not np.any(np.isnan(symbol[:, 0]))
  lab = {s: q for q, s in texts}
  assert lab["x"][0] > 0 and abs(lab["x"][1]) < 1e-9 and lab["y"][1] > 0 and abs(lab["y"][0]) < 1e-9
  assert np.all(lab["z"] == 0) and bbox[0] < 0 < bbox[1] and bbox[2] < 0 < bbox[3]
  # seen from -z the symbol gets four spokes round its label, and a tilted axis a shorter arrow
  _, symbol2, _, _ = triadGeometry([(1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)], ["x", "y", "z"])
  assert np.sum(np.isnan(symbol2[:, 0])) == 4
  assert np.all(np.isnan(symbol2[:, 0]) | (np.hypot(symbol2[:, 0], symbol2[:, 1]) > 0.25))
  arrows3, _, texts3, _ = triadGeometry([(0.5, 0.0, np.sqrt(0.75)), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0)], ["x", "y"])
  tip = lambda a: np.max(np.hypot(a[:, 0], a[:, 1]))
  assert tip(arrows3[0]) < tip(arrows3[1]) and len(texts3) == 2


def test_boundaries_grains_and_the_palette():
  from mtex import calcGrains
  ebsd = loadEBSD(FIXTURE)
  g = calcGrains(ebsd["indexed"], angle=10 * degree)
  gB = g.boundary
  poly = boundaryPolyline(gB)
  assert poly.shape[1] == 2 and np.sum(np.isnan(poly[:, 0])) == np.sum(gB.isChainStart) - 1
  assert boundarySegments(gB).shape == (len(gB), 2, 2)
  e = mapExtent(g)
  eb = mapExtent(gB)
  assert e[1] > e[0] and eb[1] > eb[0]
  palette = phasePalette(ebsd.CSList)
  assert len(palette) == len(ebsd.CSList) and all(c is not None for c in palette)


def test_the_picture_of_an_aligned_lattice_is_the_nearest_site_of_every_pixel():
  import numpy as np
  from mtex import crystalFrame, orientation
  from mtex.maps.ebsd import EBSD
  from mtex.plotting.maps import mapRaster
  nx, ny = 7, 5
  i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
  pos = np.stack([0.5 * i.ravel() + 2.0, 0.5 * j.ravel() - 1.0, np.zeros(nx * ny)], axis=-1)
  cs = crystalFrame("432")
  ebsd = EBSD(pos, orientation.rand(nx * ny, cs), np.ones(nx * ny, dtype=int), ["notIndexed", cs])
  (xlo, xhi), (ylo, yhi), index = mapRaster(ebsd)
  assert index.shape == (ny, nx)
  px = np.stack(np.meshgrid(xlo + (np.arange(nx) + 0.5) * (xhi - xlo) / nx, ylo + (np.arange(ny) + 0.5) * (yhi - ylo) / ny), axis=-1).reshape(-1, 2)
  d = np.linalg.norm(px[:, None, :] - pos[None, :, :2], axis=-1)
  assert np.array_equal(index.reshape(-1), np.argmin(d, axis=1))
