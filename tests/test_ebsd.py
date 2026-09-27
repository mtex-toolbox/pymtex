# M1: the EBSD container, the Julia E1 to E6 and the parts of check_ebsd and
# check_ebsdGrid that need no import, plotting or grains
import numpy as np
import pytest

from mtex import (EBSD, phaseList, lattice, latticeBasis, crystalFrame, specimenFrame, orientation, rotation, vector3d,
                  zvector, degree, angle, extent, shift, transform, gridify, findByLocation, findByOrientation,
                  inpolygon, fibre, rotate)
from conftest import close


def sampleMap(n=12, phases=1, dtype=np.float64, A=np.eye(2)):
  cs = crystalFrame("m-3m", mineral="test")
  cs2 = crystalFrame("321", mineral="other")
  i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")
  i, j = i.reshape(-1), j.reshape(-1)
  pos = np.stack([A[0, 0] * i + A[0, 1] * j, A[1, 0] * i + A[1, 1] * j, np.zeros(n * n)], axis=-1).astype(dtype)
  rot = rotation.byAxisAngle(zvector, (np.arange(n * n) * 7 * degree).astype(dtype))
  ph = np.full(n * n, 1, dtype=np.int8)
  ph[:5] = 0
  if phases == 2:
    ph[-21:] = 2
  pl = phaseList(cs, cs2) if phases == 2 else phaseList(cs)
  return EBSD(pos, rot, ph, pl, {"mad": np.arange(1.0, n * n + 1)})


def test_E1_the_table():
  ebsd = sampleMap()
  assert len(ebsd) == 144 and ebsd.shape == (144,) and ebsd.size == 144 and not ebsd.isGridded
  assert ebsd.isIndexed.sum() == 139 and ebsd.frame is specimenFrame.default
  assert len(ebsd.CSList) == 2 and list(ebsd.indexedPhaseId) == [1] and ebsd.mineralList == ["notIndexed", "test"]
  assert np.array_equal(ebsd.mad, np.arange(1.0, 145)) and np.array_equal(ebsd.prop["mad"], ebsd.mad)
  with pytest.raises(AttributeError):
    ebsd.bc
  ebsd.bc = np.arange(144)
  assert ebsd.bc[7] == 7 and "bc" in ebsd.prop
  with pytest.raises(ValueError):
    ebsd.ci = np.arange(10)
  # a multi channel property keeps its columns
  ebsd.fs = np.arange(144 * 5).reshape(144, 5)
  assert ebsd.fs.shape == (144, 5) and ebsd[ebsd.mad > 100].fs.shape == (44, 5)
  assert np.array_equal(ebsd.id, np.arange(144)) and np.array_equal(ebsd.phase, ebsd.phaseId) and np.array_equal(ebsd[5:9].id, [5, 6, 7, 8])
  assert ebsd.x[13] == 1 and ebsd.y[13] == 1 and ebsd.xyz.shape == (144, 3)
  assert ebsd.CS is ebsd.CSList[1] and ebsd.dtype == np.float64
  # the arguments have to agree
  with pytest.raises(ValueError):
    EBSD(ebsd.pos, ebsd.rotations, ebsd.phaseId[:10], ebsd.CSList)
  with pytest.raises(ValueError):
    EBSD(ebsd.pos, ebsd.rotations, ebsd.phaseId + 5, ebsd.CSList)
  # the MATLAB list with a named not indexed phase
  e = EBSD(ebsd.pos, ebsd.rotations, ebsd.phaseId, ["notIndexed", crystalFrame("m-3m", mineral="test")])
  assert e.CSList.mineralList == ["notIndexed", "test"]


def test_E2_selection_returns_a_map():
  ebsd = sampleMap(phases=2)
  assert len(ebsd["test"]) == 118 and len(ebsd["other"]) == 21 and len(ebsd["notIndexed"]) == 5
  assert len(ebsd["indexed"]) == 139 and len(ebsd[ebsd.CSList[2]]) == 21
  with pytest.raises(ValueError):
    ebsd["quartz"]
  # a single measurement is a map of one, with its phases and frame intact
  one = ebsd[7]
  assert isinstance(one, EBSD) and len(one) == 1 and one.frame is ebsd.frame and len(one.CSList) == 3
  # a mask selects, and the properties follow
  sel = ebsd.mad > 100
  assert len(ebsd[sel]) == sel.sum() and np.array_equal(ebsd[sel].mad, ebsd.mad[sel])
  assert ebsd.CSList["test"] is ebsd.CSList[1] and ebsd.CSList["NOTINDEXED"] is ebsd.CSList[0]
  with pytest.raises(ValueError):
    ebsd.CSList["missing"]
  selected = ebsd[["other", "test", "other"]]
  assert np.array_equal(selected.mad, ebsd.mad[ebsd.isIndexed])
  assert ebsd[np.zeros(144, dtype=bool)].isempty and len(ebsd[1:10]) == 9 and len(ebsd[[3, 5]]) == 2
  with pytest.raises(ValueError):
    ebsd[["test", "missing"]]
  # ebsd[i, j] means the pixel of a grid and is refused on a list
  with pytest.raises(IndexError):
    ebsd[2, 3]
  assert len(ebsd["xy", 2, 3]) == 1 and ebsd["xy", 2, 3].x[0] == 2 and ebsd["xy", 2, 3].y[0] == 3
  assert len(ebsd["id", 3]) == 1 and ebsd["id", 3].mad[0] == 4


def test_E2a_regions_include_their_edges():
  ebsd = sampleMap()
  mask = inpolygon(ebsd, [2, 3, 4, 2])
  assert mask.sum() == 15
  assert np.array_equal(ebsd[mask].mad, [x + 12 * y + 1 for y in range(3, 6) for x in range(2, 7)])
  polygon = np.array([[2, 3], [6, 3], [6, 5], [2, 5]])
  assert np.array_equal(inpolygon(ebsd, polygon), mask)
  assert np.array_equal(inpolygon(ebsd, polygon[::-1]), mask)
  assert np.array_equal(inpolygon(ebsd, np.vstack([polygon, polygon[:1]])), mask)
  # an L shape: the notch is outside, and both inner edges belong to the selection
  assert inpolygon(ebsd, [[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3]]).sum() == 12
  assert inpolygon(ebsd, [2, 3, 0, 0]).sum() == 1
  grid = gridify(ebsd)
  assert inpolygon(grid, [2, 3, 4, 2]).shape == grid.shape and inpolygon(grid, polygon).sum() == 15
  for bad in ([0, 0, -1, 2], [0, np.nan, 1, 2], np.zeros((2, 2))):
    with pytest.raises(ValueError):
      inpolygon(ebsd, bad)


def test_E3_orientations_are_framed_by_their_phase():
  ebsd = sampleMap(phases=2)
  o = ebsd["test"].orientations
  assert len(o) == 118 and o.CS is ebsd.CSList[1] and o.SS is ebsd.frame
  # with more than one indexed phase the map cannot answer without being asked which
  with pytest.raises(ValueError):
    ebsd.orientations
  with pytest.raises(ValueError):
    ebsd.CS
  assert len(ebsd["other"].orientations) == 21 and ebsd["other"].CS.mineral == "other"


def test_E4_the_map_carries_its_lattice():
  ebsd = sampleMap()
  assert ebsd.lattice.shape == "square" and ebsd.dPos == 1 and ebsd.unitCell.shape == (4,) and ebsd.unitCell.frame is ebsd.frame
  assert ebsd.N == zvector
  hexa = sampleMap(A=np.array([[1.0, 0.5], [0.0, np.sqrt(3) / 2]]))
  assert hexa.lattice.shape == "hex" and hexa.unitCell.shape == (6,)
  # a lattice survives a rigid motion of the map
  r = ebsd.rotate(rotation.byAxisAngle(zvector, 30 * degree))
  assert r.lattice.shape == "square" and close(r.lattice.dxy, ebsd.lattice.dxy)
  s = shift(ebsd, vector3d(3.0, -2.0, 0.0))
  assert close(extent(s), extent(ebsd) + [3, 3, -2, -2]) and close(s.lattice.dxy, ebsd.lattice.dxy)
  for parent in (ebsd, hexa, r):
    for ids in ([1], [1, 2, 3], [1, 3, 5]):
      selected = parent[ids]
      assert selected.lattice.shape == parent.lattice.shape and np.array_equal(selected.lattice.A, parent.lattice.A)
      assert close(selected.lattice.position(selected.lattice.ij), parent.pos.data[ids], tol=1e-9)
  assert ebsd[[1, 1]].lattice is None
  # a unit cell given to the constructor is the basis the lattice is built on
  e = EBSD(ebsd.pos, ebsd.rotations, ebsd.phaseId, ebsd.CSList, unitCell=ebsd.unitCell)
  assert np.array_equal(e.lattice.ij, ebsd.lattice.ij)
  # a scan line is not a lattice
  n = 24
  line = EBSD(np.stack([np.arange(n) * 0.3, np.zeros(n), np.zeros(n)], axis=-1), rotation.rand(n), np.ones(n, dtype=int), [crystalFrame("m-3m")])
  assert line.lattice is None and np.isnan(line.dPos)
  with pytest.raises(ValueError):
    line.unitCell


# moving a gridded map never reorders its arrays: the positions move and the layout turns with
# them, where MATLAB stores a rotated map anew in its default layout (decided 2026-09-26)
def test_E4b_moving_a_grid_keeps_its_memory_order():
  from mtex import gridLayout, xvector, yvector
  grid = gridify(sampleMap())
  assert grid.layout == gridLayout.columnMajor()
  for moved, layout in ((rotate(grid, 90 * degree), gridLayout(-xvector, yvector)), (shift(grid, vector3d(3, -1, 0)), gridLayout.columnMajor()),
                        (transform(grid, lambda p: vector3d(-p.data)), gridLayout(-yvector, -xvector))):
    assert moved.shape == grid.shape and moved.layout == layout
    assert np.array_equal(moved.phaseId, grid.phaseId)
    assert np.array_equal(moved.mad, grid.mad) and np.array_equal(moved.id, grid.id)
  # a new order is asked for explicitly
  assert gridify(rotate(grid, 90 * degree), gridLayout.columnMajor()).shape == grid.shape[::-1]


def test_E4a_rotate_positions_and_orientations_separately():
  ebsd = sampleMap()
  r = rotation.byAxisAngle(zvector, 30 * degree)
  positions, rotations = ebsd.pos.data.copy(), ebsd.rotations.data.copy()
  both = ebsd.rotate(r)
  onlyPositions = ebsd.rotate(r, keepEuler=True)
  onlyOrientations = ebsd.rotate(r, keepXY=True)
  assert close(onlyPositions.pos.data, both.pos.data) and np.array_equal(onlyPositions.rotations.data, rotations)
  assert np.array_equal(onlyOrientations.pos.data, positions) and close(onlyOrientations.rotations.data, both.rotations.data)
  assert close(both.pos.data, (r * ebsd.pos).data) and close(both.rotations.data, (r * ebsd.rotations).data)
  assert np.array_equal(ebsd.pos.data, positions) and np.array_equal(ebsd.rotations.data, rotations)
  assert np.array_equal(onlyPositions.mad, ebsd.mad) and np.array_equal(onlyOrientations.mad, ebsd.mad)
  # a number turns about z, as MATLAB reads it
  assert close(ebsd.rotate(30 * degree).pos.data, both.pos.data)
  # the function form dispatches to the method, MATLAB's flags as strings
  assert close(rotate(ebsd, r).pos.data, both.pos.data)
  # MATLAB's rot * ebsd turns both
  assert close((r * ebsd).pos.data, both.pos.data) and close((r * ebsd).rotations.data, both.rotations.data)
  assert np.array_equal(rotate(ebsd, r, 'keepEuler').rotations.data, rotations)
  with pytest.raises(ValueError):
    ebsd.rotate(r, keepXY=True, keepEuler=True)


def test_transform_keeps_the_lattice_consistent():
  # check_ebsdGrid: a translation, a scale and a shear leave the lattice index alone
  ebsd = sampleMap()
  A0 = ebsd.lattice.A
  t = transform(ebsd, lambda p: p + vector3d(3, -2, 0))
  assert close(t.lattice.A, A0, tol=1e-12) and np.array_equal(t.lattice.ij, ebsd.lattice.ij)
  s = transform(ebsd, lambda p: 2.5 * p)
  assert close(s.lattice.A, 2.5 * A0, tol=1e-12) and close(s.dPos, 2.5)
  g = 0.3
  sh = transform(ebsd, lambda p: vector3d(p.x + g * p.y, p.y, p.z))
  assert close(sh.lattice.A, np.array([[1, g], [0, 1]]) @ A0, tol=1e-12)
  # the sheared cell has six neighbours, which is what the shape counts
  assert np.array_equal(sh.lattice.ij, ebsd.lattice.ij) and sh.lattice.shape == "hex"
  # a trapezoidal drift keeps the index too
  d = transform(ebsd, lambda p: vector3d(p.x * (1 + 0.05 * p.y / 12), p.y * (1 + 0.05 * p.x / 12), p.z))
  assert np.array_equal(d.lattice.ij, ebsd.lattice.ij)
  # the unit cell read back from the map is the cell of the lattice
  B, stencil, dxy = latticeBasis(sh.unitCell)
  assert close(abs(np.linalg.det(B)), abs(np.linalg.det(sh.lattice.A)), tol=1e-9)


def test_E6_finding_measurements():
  ebsd = sampleMap()
  assert findByLocation(ebsd, vector3d(3.0, 4.0, 0.0)).id[0] == 3 + 4 * 12
  assert len(findByLocation(ebsd, [3.0, 4.0], radius=1.1)) == 5
  assert len(findByLocation(ebsd, [[3.0, 4.0], [5.0, 5.0]])) == 2
  o = ebsd["test"].orientations[0]
  near = findByOrientation(ebsd, o, 3 * degree)
  assert len(near) > 0 and np.all(angle(near.orientations, o) <= 3 * degree) and near.isIndexed.all()
  f = fibre(o, o.CS.cAxis if hasattr(o.CS, "cAxis") else vector3d(0, 0, 1, o.CS))
  assert len(findByOrientation(ebsd, f, 1 * degree)) == 139


def test_copy_cat_and_display():
  ebsd = sampleMap()
  c = ebsd.copy()
  c.mad[:] = 0
  assert ebsd.mad[0] == 1 and c.CSList is ebsd.CSList
  both = EBSD.cat(ebsd[:10], ebsd[10:20])
  assert len(both) == 20 and np.array_equal(both.mad, ebsd.mad[:20]) and both.lattice.shape == "square"
  with pytest.raises(ValueError):
    EBSD.cat(ebsd[:10], EBSD(ebsd.pos, ebsd.rotations, ebsd.phaseId, ebsd.CSList))
  s = repr(ebsd)
  assert s.startswith("EBSD (") and "size: 144" in s and "test" in s and "properties    : mad" in s and "square lattice" in s
  assert "the map is empty" in repr(ebsd[np.zeros(144, dtype=bool)])
  # a float32 map keeps its precision
  e32 = sampleMap(dtype=np.float32)
  assert e32.dtype == np.float32 and e32.pos.data.dtype == np.float32 and e32.lattice.A.dtype == np.float32
  # nan positions are dropped
  pos = ebsd.pos.data.copy()
  pos[3] = np.nan
  assert len(EBSD(pos, ebsd.rotations, ebsd.phaseId, ebsd.CSList, ebsd.prop)) == 143


def test_the_measurements_of_grains_leave_out_the_padding():
  from test_calcgrains import squareBlockGrid
  from mtex import calcGrains
  oris = rotation.byAxisAngle(zvector, np.arange(1, 5) * 20 * degree)
  base = squareBlockGrid(12, 2, oris)
  # one missing measurement inside a grain, padded back by gridify
  hole = np.zeros(base.shape, dtype=bool)
  hole[4, 4] = True
  e = gridify(base[~hole])
  assert np.array_equal(e.isPadding, hole)
  g = calcGrains(e, angle=5 * degree)
  # the closing gives the hole the grain around it, the selection keeps the measurements
  assert e.grainId[4, 4] > 0
  sel = e[g["id", int(e.grainId[4, 4])]]
  assert not sel.isPadding.any() and len(sel) == np.count_nonzero((e.grainId == e.grainId[4, 4]) & ~e.isPadding)


def test_the_phase_shares_are_of_the_measurements_not_the_padding():
  from test_calcgrains import squareBlockGrid
  base = squareBlockGrid(12, 2, rotation.byAxisAngle(zvector, np.arange(1, 5) * 20 * degree))
  hole = np.zeros(base.shape, dtype=bool)
  hole[6:18, 6:18] = True
  e = gridify(base[~hole])
  n = np.count_nonzero(e.isIndexed & ~e.isPadding)
  assert e.isPadding.sum() == hole.sum() and f"{n} ({float(f'{100 * n / (e.size - hole.sum()):.2g}'):g}%)" in repr(e)


def test_the_orientations_of_a_map_are_not_independent():
  ebsd = sampleMap()
  o = ebsd.orientations
  assert o.notIID and o[3:9].notIID and ebsd[10:20].orientations.notIID
