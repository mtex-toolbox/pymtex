# M1: a map put on its grid, the Julia E5
import os

import numpy as np
import pytest

from mtex import EBSD, gridify, rotation, phaseList, crystalFrame, vector3d
from conftest import close
from test_ebsd import sampleMap


def test_E5_gridify():
  ebsd = sampleMap()
  g = gridify(ebsd)
  assert g.isGridded and g.shape == (12, 12) and len(g) == len(ebsd) and g.mad.shape == (12, 12)
  assert g.isIndexed.sum() == ebsd.isIndexed.sum() and gridify(g) is g
  # a complete scan shares its arrays with the list
  assert np.shares_memory(g.rotations.data, ebsd.rotations.data) and np.shares_memory(g.mad, ebsd.mad)
  # the grid addresses scan positions in NumPy's order, row i and column j, and a block is a map again
  assert len(g[3, 4]) == 1 and g[3, 4].x[0] == 4 and g[3, 4].y[0] == 3
  assert len(g[2:4, 3:6]) == 6 and g[2:4, 3:6].shape == (2, 3) and g[2:4, 3:6].lattice.ij.shape == (6, 2)
  assert g.pos.data[5, 7, 0] == 7 and g.pos.data[5, 7, 1] == 5
  assert np.array_equal(g.lattice.ij[:, 0].reshape(12, 12), np.tile(np.arange(12), (12, 1)))
  assert g.flatten().shape == (144,) and np.array_equal(g.flatten().mad, ebsd.mad)
  # a map with holes keeps its shape and fills the gaps with not indexed cells at their ideal positions
  holed = ebsd[[k for k in range(144) if k % 7 != 6]]
  gh = gridify(holed)
  assert gh.shape == (12, 12) and gh.isIndexed.sum() == holed.isIndexed.sum() and len(gh) == 144
  assert np.isnan(gh.rotations.data[0, 6]).all() and gh.phaseId[0, 6] == 0 and gh.mad[0, 6] == 0
  assert close(gh.pos.data[0, 6], [6, 0, 0]) and np.array_equal(gh.pos.data, g.pos.data)
  # the coordinates in the plane are the specimen's own, wherever the scan starts
  away = EBSD(ebsd.pos.data + [10.0, 20.0, 0.0], ebsd.rotations, ebsd.phaseId, ebsd.CSList)
  assert close(away.lattice.plane.o, [0, 0, 0]) and close(away.lattice.plane.to2d([13.0, 24.0, 0.0]), [13, 24])
  assert close(away.lattice.position([0, 0]), [10, 20, 0])


def test_a_hexagonal_scan_is_gridded_as_its_rows():
  # every second row shifted by half a cell, five by four cells, no padding, the rows in
  # the order they were scanned, so the arrays are shared
  h = 0.5
  hpos = np.array([[h * (i + (r % 2) / 2), h * r * np.sqrt(3) / 2, 0.0] for r in range(4) for i in range(5)])
  pl = phaseList(crystalFrame("m-3m", mineral="test"))
  hexa = EBSD(hpos, rotation.byAxisAngle(vector3d.Z, np.zeros(20)), np.ones(20, dtype=int), pl)
  assert hexa.lattice.shape == "hex"
  hg = gridify(hexa)
  assert hg.shape == (4, 5) and np.shares_memory(hg.rotations.data, hexa.rotations.data)
  assert np.array_equal(hg.pos.data.reshape(-1, 3), hpos)
  assert np.max(np.abs(hg.lattice.position(hg.lattice.ij) - hpos)) < 1e-12
  # a row one cell short is padded at its end and nowhere else
  short = hexa[[k for k in range(20) if k != 9]]
  sg = gridify(short)
  assert sg.shape == (4, 5) and sg.isIndexed.sum() == 19 and close(sg.pos.data[1, 4], hpos[9])
  # a rotated square scan is gridded along its own rows, without padding
  ebsd = sampleMap().rotate(0.4)
  g = gridify(ebsd)
  assert g.shape == (12, 12) and len(g) == 144 and g.isIndexed.sum() == 139


def test_gridify_refuses_what_it_cannot_grid():
  ebsd = sampleMap()
  n = 24
  line = EBSD(np.stack([np.arange(n) * 0.3, np.zeros(n), np.zeros(n)], axis=-1), rotation.rand(n), np.ones(n, dtype=int), ebsd.CSList)
  with pytest.raises(ValueError):
    gridify(line)
  # two measurements in one cell are not silently dropped
  pos = ebsd.pos.data.copy()
  pos[5] = pos[6] + [0.01, 0, 0]
  twice = EBSD(pos, ebsd.rotations, ebsd.phaseId, ebsd.CSList)
  assert twice.lattice is None or pytest.raises(ValueError, gridify, twice)


# the positions of mtexdata('sharp') in small: rows of step 2 along x, each shifted by half a
# step's quarter against the one below, cut to an axis aligned box, so the lattice is
# turned by 45 degrees against the box and its grid is closed by padding
def turnedMap():
  pts = [(2 * a + 0.5 * b, 0.5 * b - 10) for b in range(41) for a in range(-12, 12)]
  pts = np.array([p for p in pts if 0 <= p[0] <= 20 and -10 <= p[1] <= 0])
  n = len(pts)
  ph = np.ones(n, dtype=np.int8)
  ph[5] = 0
  pos = np.column_stack([pts, np.zeros(n)])
  return EBSD(pos, rotation.byAxisAngle(vector3d(0, 0, 1), np.linspace(0, 0.1, n)), ph, phaseList(crystalFrame("m-3m", mineral="test")),
              {"mad": np.ones(n)})


def test_the_padding_of_a_turned_grid_is_no_measurement():
  lst = turnedMap()
  g = gridify(lst)
  pad = g.isPadding
  assert g.isGridded and pad.sum() == g.size - len(lst) and pad.sum() > len(lst) / 2
  assert not lst.isPadding.any() and pad.shape == g.shape
  # padding is not indexed, yet neither counted, selected nor measured as notIndexed
  assert np.all(g.phaseId[pad] == 0) and np.isnan(g.rotations.data[pad]).all()
  assert len(g["notIndexed"]) == 1 and len(g["indexed"]) == len(lst) - 1 and len(g["test"]) == len(lst) - 1
  assert close(g.extent(), [0, 20, -10, 0]) and close(g.extent(), lst.extent())
  assert "0      1 (" in repr(g) and "[0 → 20] × [-10 → 0]" in repr(g)
  # the mark travels with the map through selections, copies and moves
  assert g.flatten().isPadding.sum() == pad.sum() and g.copy().isPadding.sum() == pad.sum()
  assert np.array_equal(g[2:5, 3:9].isPadding, pad[2:5, 3:9]) and g.shift([1.0, 0, 0]).isPadding.sum() == pad.sum()
  assert gridify(g.flatten()).isPadding.sum() == pad.sum()


# MATLAB's second output: the flat index of every measurement in the grid
def test_gridify_returns_where_every_measurement_went():
  ebsd = sampleMap()
  holed = ebsd[[k for k in range(144) if k % 7 != 6]]
  g, newId = gridify(holed, return_index=True)
  assert newId.shape == (len(holed),) and len(np.unique(newId)) == len(holed)
  assert np.array_equal(g.pos.data.reshape(-1, 3)[newId], holed.pos.data.reshape(-1, 3))
  assert np.array_equal(g.rotations.data.reshape(-1, 4)[newId], holed.rotations.data.reshape(-1, 4))
  g2, same = gridify(g, return_index=True)
  assert g2 is g and np.array_equal(same, np.arange(len(g)))


def square(s):
  return vector3d(s * np.array([[-1, -1, 0], [-1, 1, 0], [1, 1, 0], [1, -1, 0]], float))


# gridify(ebsd, 'unitCell', uC): the lattice of another cell over the extent, every site the
# nearest measurement within the reach of the map's own cell, nothing interpolated
def test_a_map_is_resampled_on_another_cell():
  ebsd = sampleMap()
  # its own cell gives the map back
  g = gridify(ebsd, unitCell=square(0.5))
  assert g.shape == (12, 12) and np.array_equal(g.rotations.data, gridify(ebsd).rotations.data)
  # a cell of half the size: every site copies a measurement, the nearest one
  h, newId = gridify(ebsd, unitCell=square(0.25), return_index=True)
  assert h.shape == (23, 23) and h.isGridded and h.lattice.shape == "square" and close(h.dPos, 0.5)
  P, Q = ebsd.pos.data.reshape(-1, 3), h.pos.data.reshape(-1, 3)
  d = np.linalg.norm(Q[:, None, :2] - P[None, :, :2], axis=-1)
  nearest = np.argmin(d, axis=1)
  assert np.array_equal(h.rotations.data.reshape(-1, 4), ebsd.rotations.data.reshape(-1, 4)[nearest])
  assert np.array_equal(h.mad.reshape(-1), ebsd.mad.reshape(-1)[nearest])
  # every measurement sits on a site of the finer grid
  assert close(Q[newId], P)
  # a cell turned by 45 degrees reaches beyond the map, where the sites are not indexed
  t = gridify(ebsd, unitCell=rotation.byAxisAngle(vector3d(0, 0, 1), np.pi / 4) * square(0.5))
  far = np.linalg.norm(t.pos.data.reshape(-1, 3)[:, None, :2] - P[None, :, :2], axis=-1).min(axis=1) > np.sqrt(0.5) + 1e-9
  assert np.all(np.isnan(t.rotations.data.reshape(-1, 4)[far])) and np.all(t.phaseId.reshape(-1)[far] == 0) and far.any()
  assert np.all(np.isnan(t.mad.reshape(-1)[far]))
  with pytest.raises(ValueError):
    gridify(ebsd, unitCell=vector3d(np.array([[1.0, 0, 0], [0.5, 0.87, 0], [-0.5, 0.87, 0], [-1, 0, 0], [-0.5, -0.87, 0], [0.5, -0.87, 0]])))


# gridify(ebsd, gL): the arrays reordered by a transpose and flips, nothing resampled
def test_a_map_is_stored_in_the_layout_asked_for():
  from mtex import gridLayout, xvector, yvector
  g = gridify(sampleMap())
  assert g.layout == gridLayout.columnMajor() and g.layout.name == "columnMajor" and sampleMap().layout is None
  turned = gridify(g, gridLayout(-xvector, yvector))
  assert np.array_equal(turned.mad, np.rot90(g.mad)) and turned.layout == gridLayout(-xvector, yvector)
  assert np.array_equal(turned.pos.data.reshape(-1, 3)[np.argsort(turned.id.reshape(-1))], g.pos.data.reshape(-1, 3)[np.argsort(g.id.reshape(-1))])
  back = gridify(turned, g.layout)
  assert np.array_equal(back.mad, g.mad) and np.array_equal(back.rotations.data, g.rotations.data)
  assert np.array_equal(gridify(g, "rowMajor").mad, g.mad.T) and gridify(g, "rowMajor").layout.name == "rowMajor"
  # the neighbours come from the lattice and not from the order of the arrays
  assert np.array_equal(np.sort(turned.lattice.ij, axis=0), np.sort(g.lattice.ij, axis=0))
  lst, newId = sampleMap(), None
  t, newId = gridify(lst, gridLayout(-xvector, yvector), return_index=True)
  assert np.array_equal(t.pos.data.reshape(-1, 3)[newId], lst.pos.data.reshape(-1, 3))
  with pytest.raises(ValueError):
    gridify(g, "diagonal")


LOCAL = os.path.join(os.path.dirname(__file__), "..", "..", "frameSymmetry", "data")


# MATLAB's gridify(ebsd, 'unitCell', rotate(2*ebsd.unitCell, 45*degree)) on the twins map of
# doc/EBSDAnalysis/EBSDInter.m, read through bench/matlab/probe_resample.m
@pytest.mark.skipif(not os.path.isdir(os.path.join(LOCAL, "EBSD")), reason="the MTEX data folder is not here")
def test_the_resampled_twins_map_as_MATLAB():
  from mtex import degree, mtexdata, rotate
  ebsd = mtexdata("twins", root=LOCAL, download=False, silent=True)
  g = gridify(ebsd, unitCell=rotate(2 * ebsd.unitCell, 45 * degree))
  assert g.shape == (108, 109) and g.isIndexed.sum() == 5713
  x, y = g.pos.data[..., 0], g.pos.data[..., 1]
  assert abs(x.min() + 20.7889) < 1e-4 and abs(x.max() - 70.4278) < 1e-4 and abs(y.min() + 25.0316) < 1e-4 and abs(y.max() - 66.1852) < 1e-4
