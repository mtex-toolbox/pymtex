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
