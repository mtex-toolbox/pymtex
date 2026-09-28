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


def test_scaleBar_takesJuliasLengthsAndLabels():
  assert scaleBar(360.0, "um") == (25.0, "25 µm")
  assert scaleBar(36000.0, "um") == (5000.0, "5 mm")
  assert scaleBar(3.6, "um") == (0.5, "500 nm")
  assert scaleBar(1200.0, "nm") == (125.0, "125 nm")


def test_scaleBar_liesBetweenATwentiethAndAQuarterOfTheWidth():
  for w in (1.0, 2.7, 13.0, 55.0, 480.0, 9300.0, 1.2e5):
    length, _ = scaleBar(w, "um")
    assert 0.02 * w <= length <= 0.25 * w


def test_switchUnit_bringsTheValueBetweenOneAndAThousand():
  for x, u in ((2500.0, "um"), (0.004, "mm"), (7.0, "nm")):
    y, _, f = switchUnit(x, u)
    assert np.isclose(y, x * f) and (1 <= y < 1000 or y == x)


def test_scaleBar_ofAMapFitsItsExtent():
  ebsd = loadEBSD(FIXTURE)
  e = mapExtent(ebsd)
  length, label = scaleBar(ebsd)
  assert 0.02 * (e[1] - e[0]) <= length <= 0.25 * (e[1] - e[0]) and "m" in label


def test_mapRaster_spansTheCellsWithOnePixelPerCell():
  nx, ny, h = 5, 4, 0.5
  r = mapRaster(gridMap(nx, ny, h))
  assert r is not None
  (xlo, xhi), (ylo, yhi), index = r
  # the picture spans the cells, half a cell beyond the outermost centres, one pixel per cell
  assert np.allclose((xlo, xhi), (-h / 2, h * (nx - 1) + h / 2)) and np.allclose((ylo, yhi), (-h / 2, h * (ny - 1) + h / 2))
  assert index.shape == (ny, nx)


def test_mapRaster_pixelNamesTheMeasurementThereInTheMapsOrder():
  nx, ny = 5, 4
  index = mapRaster(gridMap(nx, ny, 0.5))[2]
  assert np.array_equal(index, np.arange(nx * ny).reshape(ny, nx))


def test_mapRaster_holeIsMinusOne():
  nx, ny = 5, 4
  holed = gridMap(nx, ny, 0.5)[np.arange(nx * ny) != 6]
  ih = mapRaster(holed)[2]
  assert np.sum(ih < 0) == 1 and len(np.unique(ih)) == nx * ny


def test_mapRaster_capOnPixelsKeepsTheShape():
  small = mapRaster(gridMap(5, 4, 0.5), maxPixels=6)[2]
  assert small.size <= 6 and small.shape[1] >= small.shape[0]


# a hexagonal lattice lands on the same rectangle, a pixel painted for the cell it lies in
def test_mapRaster_hexagonalLatticePaintsEachPixelByTheCellItLiesIn():
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
def test_mapCells_ofADriftedLatticeTileWhereNoPictureIs():
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


def test_mapCells_ofARigidGridKeepTheIdealCell():
  h = 0.5
  rc = mapCells(gridMap(5, 4, h))
  assert np.all((np.abs(np.mod(rc[..., 0] + h / 2, h)) < 1e-12) | (np.abs(np.mod(rc[..., 0], h)) < 1e-12))


def test_mapRaster_mapNotOnALatticeHasNoPictureAndNoCells():
  one = EBSD(np.zeros((1, 3)), rotation.id.data.reshape(1, 4), np.ones(1, dtype=np.int8), phaseList(crystalFrame("1"), CS))
  assert mapRaster(one) is None
  with pytest.raises(ValueError):
    mapCells(one)


# the layout that puts y down
def flipLayout():
  return mapLayout(plottingConvention(-zvector, xvector))


# the layout that puts x up
def swapLayout():
  return mapLayout(plottingConvention(-zvector, yvector))


def test_mapLayout_ofAConventionFlipsOrSwaps():
  assert np.array_equal(flipLayout(), [[1, 0], [0, -1]]) and np.array_equal(swapLayout(), [[0, 1], [1, 0]])


def test_mapRaster_underALayoutIsTheFlippedOrTransposedPicture():
  nx, ny, h = 5, 4, 0.5
  ebsd = gridMap(nx, ny, h)
  index = mapRaster(ebsd)[2]
  assert np.array_equal(mapRaster(ebsd, layout=flipLayout())[2], index[::-1])
  assert np.array_equal(mapRaster(ebsd, layout=swapLayout())[2], index.T)
  sx, sy, _ = mapRaster(ebsd, layout=swapLayout())
  assert np.allclose(sx, (-h / 2, h * (ny - 1) + h / 2)) and np.allclose(sy, (-h / 2, h * (nx - 1) + h / 2))


def test_mapExtent_underASwapSwapsTheAxes():
  ebsd = gridMap(5, 4, 0.5)
  e, es = mapExtent(ebsd), mapExtent(ebsd, layout=swapLayout())
  assert es == (e[2], e[3], e[0], e[1])


def test_mapCells_underAFlipNegateY():
  ebsd = gridMap(5, 4, 0.5)
  assert np.allclose(mapCells(ebsd, layout=flipLayout())[..., 1], -mapCells(ebsd)[..., 1])


def test_mapPoints_underAFlipNegateY():
  ebsd = gridMap(5, 4, 0.5)
  assert np.allclose(mapPoints(ebsd, layout=flipLayout()), mapPoints(ebsd) * [1, -1])


# the default layout is the session's convention, y down under ij
def test_mapLayout_defaultIsYDown():
  assert np.array_equal(mapLayout(), [[1, 0], [0, -1]])


# the frame on the screen: swapping the axes swaps the arrows, flipping the map puts the
# viewer on the other side of the plane
def test_frameOnScreen_followsTheLayout():
  pl = gridMap(5, 4, 0.5).lattice.plane
  swap, flip = swapLayout(), flipLayout()
  assert frameOnScreen(pl, swap)[0] == (0.0, 1.0, 0.0) and frameOnScreen(pl, swap)[1] == (1.0, 0.0, 0.0)
  assert np.isclose(frameOnScreen(pl, swap)[2][2], -1) and np.isclose(frameOnScreen(pl, flip)[2][2], -1)
  assert np.isclose(frameOnScreen(pl)[2][2], 1)


def test_triadGeometry_drawsTwoArrowsWithTheirLabelsOnTheAxes():
  arrows, symbol, texts, bbox = triadGeometry([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)], ["x", "y", "z"])
  assert len(arrows) == 2 and all(len(a) == 7 for a in arrows)
  assert len(symbol) and not np.any(np.isnan(symbol[:, 0]))
  lab = {s: q for q, s in texts}
  assert lab["x"][0] > 0 and abs(lab["x"][1]) < 1e-9 and lab["y"][1] > 0 and abs(lab["y"][0]) < 1e-9
  assert np.all(lab["z"] == 0) and bbox[0] < 0 < bbox[1] and bbox[2] < 0 < bbox[3]


# seen from -z the symbol gets four spokes round its label
def test_triadGeometry_seenFromMinusZTheSymbolHasFourSpokes():
  _, symbol2, _, _ = triadGeometry([(1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)], ["x", "y", "z"])
  assert np.sum(np.isnan(symbol2[:, 0])) == 4
  assert np.all(np.isnan(symbol2[:, 0]) | (np.hypot(symbol2[:, 0], symbol2[:, 1]) > 0.25))


def test_triadGeometry_tiltedAxisGetsAShorterArrow():
  arrows3, _, texts3, _ = triadGeometry([(0.5, 0.0, np.sqrt(0.75)), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0)], ["x", "y"])
  tip = lambda a: np.max(np.hypot(a[:, 0], a[:, 1]))
  assert tip(arrows3[0]) < tip(arrows3[1]) and len(texts3) == 2


@pytest.fixture(scope="module")
def forsteriteGrains():
  from mtex import calcGrains
  ebsd = loadEBSD(FIXTURE)
  return ebsd, calcGrains(ebsd["indexed"], angle=10 * degree)


def test_boundaryPolyline_breaksOnceBetweenChains(forsteriteGrains):
  gB = forsteriteGrains[1].boundary
  poly = boundaryPolyline(gB)
  assert poly.shape[1] == 2 and np.sum(np.isnan(poly[:, 0])) == np.sum(gB.isChainStart) - 1


def test_boundarySegments_isOneSegmentPerBoundary(forsteriteGrains):
  gB = forsteriteGrains[1].boundary
  assert boundarySegments(gB).shape == (len(gB), 2, 2)


def test_mapExtent_ofGrainsAndBoundariesIsNotEmpty(forsteriteGrains):
  g = forsteriteGrains[1]
  e = mapExtent(g)
  eb = mapExtent(g.boundary)
  assert e[1] > e[0] and eb[1] > eb[0]


def test_phasePalette_hasAColourForEveryPhase(forsteriteGrains):
  ebsd = forsteriteGrains[0]
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


# the compiled lookup of the picture against the array route, on square, hexagonal, holed,
# padded and turned maps
def test_the_compiled_picture_is_the_numpy_one():
  from mtex import gridify, setMTEXpref
  A = np.array([[1.0, 0.5], [0.0, np.sqrt(3) / 2]])
  hexa = gridMap(9, 7, 1.0, A)
  maps = [gridMap(6, 5, 0.5), hexa, hexa[np.arange(63) % 5 != 2], gridify(hexa[np.arange(63) != 20]), gridMap(6, 5, 0.5).rotate(0.3)]
  layouts = [None, np.array([[0.0, 1.0], [-1.0, 0.0]])]
  compiled = [mapRaster(e, layout=L) for e in maps for L in layouts]
  setMTEXpref("kernels", "numpy")
  arrays = [mapRaster(e, layout=L) for e in maps for L in layouts]
  for a, b in zip(compiled, arrays):
    assert (a is None) == (b is None)
    if a is not None:
      assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1]) and np.array_equal(a[2], b[2])
