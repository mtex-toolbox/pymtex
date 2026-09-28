import os
import shutil
import warnings

import numpy as np
import pytest

import mtex
from mtex import EBSD, crystalFrame, degree, loadEBSD, rotation, setMTEXpref
from mtex.io.load import (ebsdFromColumns, phaseColumn, phaseShift, placedPhases, regularPoly, tryGridify,
                          tslPointGroup, unmeasuredRows)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "ebsd")


# ------------------------------------------------------------------ the phase numbering

def test_phaseShift_reads_the_numbering_off_the_values():
  assert phaseShift([0, 1, 2, 1], 2) == 0        # one based, zero not indexed
  assert phaseShift([0, 1, 0], 2) == 1           # zero based
  assert phaseShift([1, 1, 2], 2) == 0
  assert phaseShift([0, 0, 0], 1) == 1           # a single phase written as zero
  assert phaseShift([], 2) == 0


def test_phaseShift_ignores_negative_values():
  assert phaseShift([-1, 0, 0], 1) == 1
  assert phaseShift([-1, 1, 2, 3, 4], 4) == 0


def test_phaseColumn_shifts_and_bounds():
  ids = phaseColumn([0, 1, 2, 7, -1, 1.2], 0, 3)
  assert ids.dtype == np.int8
  assert list(ids) == [0, 1, 2, 0, 0, 1]
  assert list(phaseColumn([0, 1, 255], 1, 3)) == [1, 2, 0]


def test_tslPointGroup():
  assert tslPointGroup(43) == "m-3m"
  assert tslPointGroup("62") == "6/mmm"
  assert tslPointGroup(2) == "2/m"
  assert tslPointGroup("x") == "-1"
  assert tslPointGroup(62, 126) == "6/mmm"
  assert tslPointGroup(43, "130") == "-43m"
  assert tslPointGroup(22, "Orthorhombic (D2h) [mmm]") == "mmm"
  # a point group of another Laue class is not believed
  assert tslPointGroup(62, 131) == "6/mmm"
  assert tslPointGroup(43, "999") == "m-3m"


def test_placedPhases_places_by_number_and_fills_the_gaps():
  a, b = crystalFrame("m-3m", mineral="a"), crystalFrame("mmm", mineral="b")
  pl = placedPhases({3: b, 1: a})
  assert len(pl) == 4
  assert not pl.frames[0].isIndexed
  assert pl.frames[1] is a and pl.frames[3] is b
  assert not pl.frames[2].isIndexed and pl.frames[2].mineral == "phase 2"
  assert len(placedPhases({})) == 1


def test_unmeasuredRows():
  props = {"iq": np.array([0, 1, 0]), "ci": np.array([0, 0, 0])}
  z = np.array([0, 0, 0.0])
  mask = unmeasuredRows(z, z, np.array([0, 0, 0.5]), props)
  assert list(mask) == [True, False, False]
  assert unmeasuredRows(z, z, z, {"sem": z}) is None
  assert unmeasuredRows(np.array([1, 1, 1.0]), z, z, props) is None


# ------------------------------------------------------------------ the map

def grid(nx=3, ny=2, step=1.0):
  x, y = np.meshgrid(np.arange(nx) * step, np.arange(ny) * step)
  return x.ravel(), y.ravel()


def test_ebsdFromColumns_builds_the_map():
  x, y = grid()
  cs = [crystalFrame("m-3m", mineral="Fe"), crystalFrame("mmm", mineral="Ol")]
  phi1, Phi, phi2 = np.full(6, 10 * degree), np.full(6, 20 * degree), np.full(6, 30 * degree)
  e = ebsdFromColumns(x, y, phi1, Phi, phi2, [1, 2, 0, 1, 1, 2], cs, {"mad": np.arange(6.0)})
  assert len(e) == 6
  assert list(e.phaseId) == [1, 2, 0, 1, 1, 2]
  assert list(e.phase) == [1, 2, 0, 1, 1, 2]
  assert np.isnan(e.rotations.data[2]).all()
  assert mtex.angle(e[0].rotations, rotation.byEuler(10 * degree, 20 * degree, 30 * degree)) < 1e-12
  assert e.mad[3] == 3
  assert e.lattice.shape == "square"


def test_ebsdFromColumns_applies_the_correction_once():
  x, y = grid()
  cs = [crystalFrame("m-3m", mineral="Fe")]
  r = rotation.byAxisAngle(mtex.vector3d(1, -1, 0), np.pi)
  raw = rotation.byEuler(10 * degree, 20 * degree, 30 * degree)
  e = ebsdFromColumns(x, y, np.full(6, 10 * degree), np.full(6, 20 * degree), np.full(6, 30 * degree), np.ones(6), cs, {},
                      eulerCorrection=r)
  assert mtex.angle(e[0].rotations, r * raw) < 1e-12


def test_ebsdFromColumns_zero_based_and_unindexed():
  x, y = grid()
  cs = [crystalFrame("m-3m", mineral="Fe"), crystalFrame("mmm", mineral="Ol")]
  z = np.zeros(6)
  e = ebsdFromColumns(x, y, z, z, z, [0, 1, 0, 1, 255, 0], cs, {}, shift=1, unindexed=[False, False, True, False, False, False])
  assert list(e.phaseId) == [1, 2, 0, 2, 0, 1]
  assert list(e.phaseMap) == [-1, 0, 1]
  e = ebsdFromColumns(x, y, z, z, z, [0, 1, 0, 1, 2, 0], cs, {}, shift=0)
  assert list(e.phaseId) == [0, 1, 0, 1, 2, 0]
  assert list(e.phaseMap) == [0, 1, 2]


def test_ebsdFromColumns_conventions():
  x, y = grid()
  cs = [crystalFrame("m-3m", mineral="Fe")]
  a = np.full(6, 0.1)
  e = ebsdFromColumns(x, y, a, 2 * a, 3 * a, np.ones(6), cs, {}, convention="ABG")
  assert mtex.angle(e[0].rotations, rotation.byEuler(0.1, 0.2, 0.3, "Matthies")) < 1e-12


def test_ebsdFromColumns_takes_the_unit_cell_hint():
  x, y = grid(step=0.5)
  e = ebsdFromColumns(x, y, np.zeros(6), np.zeros(6), np.zeros(6), np.ones(6), [crystalFrame("m-3m")], {},
                      unitCell=regularPoly(4, [0.5, 0.5]))
  assert e.lattice.shape == "square"
  assert abs(e.dPos - 0.5) < 1e-12


def test_regularPoly():
  sq = regularPoly(4, [1, 2])
  assert np.allclose(np.sort(np.abs(sq[:, 0])), 0.5) and np.allclose(np.abs(sq[:, 1]), 1)
  hx = regularPoly(6, 1)
  assert abs(hx[:, 0].max() - hx[:, 0].min() - 1) < 1e-12
  assert np.allclose(hx[:, 2], 0)


# ------------------------------------------------------------------ the grid

def test_tryGridify_puts_a_complete_scan_on_its_grid():
  x, y = grid(4, 3)
  e = ebsdFromColumns(x, y, np.zeros(12), np.zeros(12), np.zeros(12), np.ones(12), [crystalFrame("m-3m")], {})
  with warnings.catch_warnings():
    warnings.simplefilter("error")
    g = tryGridify(e)
  assert g.shape == (3, 4)
  assert tryGridify(g) is g


def test_tryGridify_keeps_a_list_with_a_stray_position():
  x, y = grid(4, 3)
  x = np.append(x, 1000.0)
  y = np.append(y, 1000.0)
  e = ebsdFromColumns(x, y, np.zeros(13), np.zeros(13), np.zeros(13), np.ones(13), [crystalFrame("m-3m")], {})
  with pytest.warns(UserWarning, match="not form a grid|not on a lattice"):
    g = tryGridify(e)
  assert g.ndim == 1


def test_tryGridify_keeps_a_list_on_a_collision():
  x, y = grid(4, 3)
  x = np.append(x, 1.0)
  y = np.append(y, 1.0)
  e = ebsdFromColumns(x, y, np.zeros(13), np.zeros(13), np.zeros(13), np.ones(13), [crystalFrame("m-3m")], {})
  with pytest.warns(UserWarning):
    g = tryGridify(e)
  assert g.ndim == 1 and len(g) == 13


# ------------------------------------------------------------------ the entry point

def test_loadEBSD_dispatches_on_the_extension():
  e = loadEBSD(os.path.join(FIXTURES, "ferrite_cut.ang"))
  assert e.ndim == 2
  e = loadEBSD(os.path.join(FIXTURES, "Forsterite_cut.ctf"), noGrid=True)
  assert e.ndim == 1 and len(e) == 1464


def test_loadEBSD_interface_names_the_format(tmp_path):
  p = tmp_path / "renamed.dat"
  shutil.copy(os.path.join(FIXTURES, "ferrite_cut.ang"), p)
  e = loadEBSD(p, interface="ang")
  assert e.CSList.mineralList[1] == "Ferrite"
  e = loadEBSD(str(p), interface=".ang", noGrid=True)
  assert len(e) == 467


def test_loadEBSD_a_phase_list_replaces_the_phases_of_a_vendor_file():
  # an import script states its own frames; they replace the file's, the not indexed one first
  e = loadEBSD(os.path.join(FIXTURES, "ferrite_cut.ang"), crystalFrame("m-3m", mineral="Iron bcc"))
  assert e.CSList.mineralList[1] == "Iron bcc" and not e.CSList.isIndexed(0)
  with pytest.raises(ValueError, match="names"):
    loadEBSD(os.path.join(FIXTURES, "ferrite_cut.ang"), [crystalFrame("m-3m"), crystalFrame("6/mmm")])


def test_loadEBSD_a_table_needs_phases():
  with pytest.raises(ValueError, match="crystal frame"):
    loadEBSD(os.path.join(FIXTURES, "generic_cut.txt"))


def test_loadEBSD_a_list_of_files():
  maps = loadEBSD([os.path.join(FIXTURES, "ferrite_cut.ang"), os.path.join(FIXTURES, "olivine_cut.ang")], noGrid=True)
  assert [len(m) for m in maps] == [467, 700]


def test_EBSD_load_is_the_same_verb():
  e = EBSD.load(os.path.join(FIXTURES, "ferrite_cut.ang"), noGrid=True)
  assert len(e) == 467 and e.ndim == 1


def test_gridifyOnImport_preference():
  setMTEXpref("gridifyOnImport", False)
  e = loadEBSD(os.path.join(FIXTURES, "ferrite_cut.ang"))
  assert e.ndim == 1


def test_the_numpy_backend_reads_the_same_map():
  a = loadEBSD(os.path.join(FIXTURES, "Forsterite_cut.ctf"))
  setMTEXpref("kernels", "numpy")
  b = loadEBSD(os.path.join(FIXTURES, "Forsterite_cut.ctf"))
  assert a.shape == b.shape
  assert np.array_equal(a.pos.data, b.pos.data)
  assert np.array_equal(a.rotations.data, b.rotations.data, equal_nan=True)
  assert np.array_equal(a.phaseId, b.phaseId) and np.array_equal(a.mad, b.mad)


# a hexagonal scan of rows of 7 and 6 points that stopped four points into its eighth row
def stoppedHexScan(x0=0.0, dtype=np.float64):
  h = 0.5
  rows = [[(x0 + h * (i + (r % 2) / 2), h * r * np.sqrt(3) / 2) for i in range(7 - r % 2)] for r in range(8)]
  xy = np.array([p for row in rows for p in row][:-3], dtype=dtype)
  return xy.astype(np.float64), h


def test_a_stopped_hexagonal_scan_is_read_off_its_rows():
  from mtex.io.load import gridLattice, scanGridFromPositions
  xy, h = stoppedHexScan()
  sg = scanGridFromPositions(xy[:, 0], xy[:, 1])
  assert sg[:2] == (7, 8) and sg[4:] == (True, 6)
  pos = np.column_stack([xy, np.zeros(len(xy))])
  a = gridLattice(pos, *sg)
  setMTEXpref("kernels", "numpy")
  b = gridLattice(pos, *sg)
  assert a is not None and np.array_equal(a.ij, b.ij) and np.allclose(a.A, b.A) and np.allclose(a.origin, b.origin)
  assert a.ij.min() == 0 and np.max(np.abs(a.position(a.ij)[:, :2] - xy)) < 1e-12


def test_single_precision_positions_far_out_keep_the_header_steps():
  from mtex.io.load import gridLattice, scanGridFromPositions
  xy, h = stoppedHexScan(x0=4000.0, dtype=np.float32)
  sg = scanGridFromPositions(xy[:, 0], xy[:, 1], dx=h, dy=h * np.sqrt(3) / 2)
  l = gridLattice(np.column_stack([xy, np.zeros(len(xy))]), *sg)
  assert np.array_equal(l.A, [[h, h / 2], [0.0, h * np.sqrt(3) / 2]])


# the grid written while the file is read is the list gridded afterwards, for every format
@pytest.mark.parametrize("name", ["copper_cut.osc", "ferrite_cut.ang", "Forsterite_cut.ctf", "generic_cut.txt", "martensite_cut.cpr",
                                  "olivine_cut.ang", "h5/bruker_cut.h5", "h5/edax_cut.h5", "h5/oxford_cut.h5", "h5/oli_cut.h5oina"])
def test_the_grid_built_while_reading_is_the_list_gridded(name):
  from mtex import gridify
  from mtex.io import load as L
  if name.startswith("h5/"):
    pytest.importorskip("h5py")
  fname = os.path.join(FIXTURES, name)
  kw = {"CS": ["notIndexed", crystalFrame("m-3m", mineral="Fe"), crystalFrame("m-3m", mineral="Mg")]} if name.endswith(".txt") else {}
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    a = loadEBSD(fname, **kw)
    b = loadEBSD(fname, noGrid=True, **kw)
  if not a.isGridded:
    return
  b = gridify(b)
  assert a.shape == b.shape and a.CSList.mineralList == b.CSList.mineralList
  assert np.array_equal(a.pos.data, b.pos.data) and np.array_equal(a.rotations.data, b.rotations.data, equal_nan=True)
  assert np.array_equal(a.phaseId, b.phaseId) and np.array_equal(a.id, b.id) and np.array_equal(a.isPadding, b.isPadding)
  assert set(a.prop) == set(b.prop) and all(np.array_equal(a.prop[k], b.prop[k], equal_nan=True) and a.prop[k].dtype == b.prop[k].dtype for k in a.prop)
  assert np.array_equal(a.lattice.ij, b.lattice.ij) and np.array_equal(a.lattice.A, b.lattice.A) and np.array_equal(a.lattice.origin, b.lattice.origin)
