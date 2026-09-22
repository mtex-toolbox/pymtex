# M1: a map put on its grid, the Julia E5
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
  assert "0      1 (" in repr(g) and "X x Y: [0 → 20] x [-10 → 0]" in repr(g)
  # the mark travels with the map through selections, copies and moves
  assert g.flatten().isPadding.sum() == pad.sum() and g.copy().isPadding.sum() == pad.sum()
  assert np.array_equal(g[2:5, 3:9].isPadding, pad[2:5, 3:9]) and g.shift([1.0, 0, 0]).isPadding.sum() == pad.sum()
  assert gridify(g.flatten()).isPadding.sum() == pad.sum()
