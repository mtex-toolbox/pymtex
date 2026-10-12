import os

import numpy as np
import pytest

import mtex
from mtex import degree, loadEBSD, rotation, vector3d, yvector
from mtex.io.h5 import (H5Scope, alignmentList, eulerColumns, h5PhaseShift, h5Resolve, hdf5Configs, _loadEBSD_h5,
                        matrixRotations, spaceGroupNumber)

h5py = pytest.importorskip("h5py")

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "ebsd", "h5")
CORPUS = os.environ.get("MTEX_TEST_H5", "")


def fixture(name):
  return os.path.join(FIXTURES, name)


def firstEuler(e):
  q = e.rotations.data.reshape(-1, 4)
  k = int(np.flatnonzero(np.isfinite(q[:, 0]))[0])
  return k, np.array(rotation(q[k]).Euler()) / degree


def counts(e):
  return [int((e.phaseId == k).sum()) for k in range(len(e.CSList))]


# ------------------------------------------------------------------ the fixtures, one per vendor layout

def test_edax_oh5():
  e = loadEBSD(fixture("kamila_cut.oh5"))
  assert e.shape == (2, 305) and e.meta["vendor"] == "EDAX"
  assert counts(e) == [0, 610]
  assert e.CSList.mineralList[1] == "Ti (Titanium, alpha)" and e.CS.pointGroup == "6/mmm"
  assert np.allclose(firstEuler(e)[1], [114.06853843, 175.36763636, 71.61957914], atol=1e-6)
  assert e.lattice.shape == "square" and e.dPos == 4.0
  assert set(e.prop) >= {"ci", "iq", "fit", "semsignal"}
  assert e.meta["header"]["Grid Type"] == "SqrGrid" and e.meta["header"]["nColumns"] == 305


def test_edax_oh5_counts_a_single_phase_written_as_zero_from_zero():
  e = loadEBSD(fixture("kamila_cut.oh5"))
  # one phase written as zero throughout is counted from zero, as in an .ang
  assert e.meta["h5"]["phase"]["shift"] == 1 and len(e.meta["h5"]["euler"]["paths"]) == 3


def test_edax_oh5_takes_the_alignment_the_file_states():
  e = loadEBSD(fixture("kamila_cut.oh5"))
  # X||a, the EDAX alignment the file states as "a || x, c* || z"
  assert abs(vector3d(e.CS.axes[0]).angle(mtex.xvector)) < 1e-12


def test_edax_h5_list_with_unplaced_pixels():
  e = loadEBSD(fixture("edax_cut.h5"), grid=False)
  assert len(e) == 547 and e.meta["unplaced"] == 1
  assert counts(e) == [0, 547] and e.CS.pointGroup == "mmm"
  assert np.allclose(firstEuler(e)[1], [304.96433441, 57.7159131, 100.76364115], atol=1e-6)


def test_edax_h5_grids_on_a_hexagonal_lattice():
  g = loadEBSD(fixture("edax_cut.h5"))
  assert g.shape == (2, 274) and g.lattice.shape == "hex"


def test_edaxh5_table_of_matrices():
  e = loadEBSD(fixture("kamila_cut.edaxh5"))
  assert e.shape == (2, 305) and e.meta["vendor"] == "EDAXh5"
  assert counts(e) == [0, 610]
  assert np.allclose(firstEuler(e)[1], [114.06853843, 175.36763636, 71.61957914], atol=1e-6)
  assert set(e.prop) >= {"ci", "iq", "fit", "custom"}


def test_edaxh5_is_the_same_map_as_the_oh5():
  e = loadEBSD(fixture("kamila_cut.edaxh5"))
  # the same map as the .oh5, pixel for pixel
  o = loadEBSD(fixture("kamila_cut.oh5"))
  assert mtex.angle(e.rotations.flatten()[:50], o.rotations.flatten()[:50]).max() < 1e-6
  assert np.array_equal(e.pos.data, o.pos.data)


def test_oxford_h5oina_reads_its_processed_data_set():
  e = loadEBSD(fixture("oli_cut.h5oina"))
  assert e.shape == (2, 186) and e.meta["vendor"] == "Oxford"
  assert e.meta["dataSets"] == ["/1/Data Processing", "/1/EBSD"] and e.meta["dataSet"] == "/1/Data Processing"
  assert counts(e) == [3, 369] and e.CSList.mineralList[1] == "Forsterite" and e.CS.pointGroup == "mmm"
  assert np.allclose(firstEuler(e)[1], [240.13360381, 146.45168516, 335.24092444], atol=1e-6)
  assert e.meta["surface"] == (0.0, 0.0, 0.0)
  assert set(e.prop) == {"bands", "error", "meanangulardeviation"}


def test_oxford_h5oina_data_set_is_chosen_by_name_or_number():
  raw = loadEBSD(fixture("oli_cut.h5oina"), dataSet="EBSD", grid=False)
  assert raw.meta["dataSet"] == "/1/EBSD" and len(raw) == 372
  byNumber = loadEBSD(fixture("oli_cut.h5oina"), dataSet=1, grid=False)
  assert byNumber.meta["dataSet"] == "/1/EBSD"


def test_oxford_h5oina_an_unknown_data_set_is_an_error():
  with pytest.raises(ValueError, match="numbered 0 to 1"):
    loadEBSD(fixture("oli_cut.h5oina"), dataSet=2)
  with pytest.raises(ValueError, match="no data set"):
    loadEBSD(fixture("oli_cut.h5oina"), dataSet="Mars")


def test_oxford_h5():
  e = loadEBSD(fixture("oxford_cut.h5"))
  assert e.shape == (2, 430)
  assert counts(e) == [61, 2, 796, 1]
  assert e.CSList.mineralList == ["notIndexed", "CoCr", "Co", "Cr"]
  assert [f.pointGroup for f in e.CSList.frames[1:]] == ["4/mmm", "m-3m", "m-3m"]
  assert np.allclose(firstEuler(e)[1], [118.18436011, 133.42662369, 215.5493861], atol=1e-6)


def test_oxford_h5_keeps_the_eds_columns_but_not_the_positions_as_properties():
  e = loadEBSD(fixture("oxford_cut.h5"))
  assert "windowintegralcoka1" in e.prop and "bandcontrast" in e.prop
  assert "x" not in e.prop and "y" not in e.prop


def test_thermofisher_stack_layout():
  e = loadEBSD(fixture("thermofisher_cut.hdf5"))
  assert e.shape == (2, 31) and e.meta["vendor"] == "ThermoFisher"
  assert counts(e) == [0, 56, 0, 6]
  assert e.CSList.mineralList == ["notIndexed", "Titanium", "Titanium-beta", "Vanadium"]
  assert [f.pointGroup for f in e.CSList.frames[1:]] == ["6/mmm", "m-3m", "m-3m"]
  # the file's (149.04, 142.57, 24.80) after the half turn about y between the pixel and the
  # Euler x axes, MATLAB's map correction
  assert np.allclose(firstEuler(e)[1], [210.96269226, 37.42823792, 204.79652023], atol=1e-6)
  # the file writes metres and says nothing about it, as MTEX reads it too
  assert abs(e.dPos - 1.5e-7) < 1e-12


def test_emsphinx_zero_based_phases():
  e = loadEBSD(fixture("emsphinx_cut.h5"))
  assert e.shape == (2, 955) and e.meta["vendor"] == "EMSphInx"
  assert counts(e) == [0, 0, 1910, 0]
  assert [f.pointGroup for f in e.CSList.frames[1:]] == ["m-3m", "m-3m", "6/mmm"]
  assert np.allclose(firstEuler(e)[1], [251.63514628, 34.73104247, 124.36628874], atol=1e-6)
  assert e.meta["h5"]["phase"]["zeroBased"] and e.meta["h5"]["phase"]["shift"] == 1
  assert set(e.prop) == {"iq", "metric"}


def test_bruker_direct_positions():
  e = loadEBSD(fixture("bruker_cut.h5"), grid=False)
  assert len(e) == 2000 and e.meta["vendor"] == "Bruker"
  assert counts(e) == [225, 1689, 86]
  assert e.CSList.mineralList == ["notIndexed", "Zirconium - alpha", "Zirconium"]
  assert [f.pointGroup for f in e.CSList.frames[1:]] == ["6/mmm", "m-3m"]
  k, euler = firstEuler(e)
  assert k == 0 and np.allclose(euler, [294.0380859375, 159.52241516, 127.85329056], atol=1e-6)
  assert abs(e.pos.x[0] - 142.6921844) < 1e-6


def test_bruker_takes_the_space_group_number_and_a_cell_in_one_row():
  e = loadEBSD(fixture("bruker_cut.h5"), grid=False)
  # the space group number, not the symbol beside it, and a cell written as one row of six
  assert np.allclose(e.CSList[2].abc, e.CSList[2].abc[0])
  assert e.meta["h5"]["cs"][0]["cell"]


# ------------------------------------------------------------------ the corrections

# the correction each vendor layout applies: Bruker's half turn about y for id five, the EDAX
# setting the file states (the half turn about (1,-1,0)), EMSphInx's the same, ThermoFisher the
# half turn about y between its pixel and its Euler x axes (xTalView manual, MATLAB 9a51196b2)
@pytest.mark.parametrize("name, correction", [
  ("bruker_cut.h5", rotation.byAxisAngle(vector3d(0, 1, 0), np.pi)),
  ("kamila_cut.oh5", rotation.byEuler(135 * degree, 180 * degree, 225 * degree)),
  ("emsphinx_cut.h5", rotation.byEuler(135 * degree, 180 * degree, 225 * degree)),
  ("thermofisher_cut.hdf5", rotation.byEuler(90 * degree, 180 * degree, 270 * degree)),
])
def test_the_vendor_correction(name, correction):
  a = loadEBSD(fixture(name), grid=False)
  b = loadEBSD(fixture(name), eulerCorrection=rotation.id, grid=False)
  k = np.flatnonzero(a.isIndexed)[:20]
  assert mtex.angle(a.rotations[k], correction * b.rotations[k]).max() < 1e-12


def test_oxford_tilted_surface_is_undone_with_a_warning(tmp_path):
  import shutil
  p = tmp_path / "tilt.h5oina"
  shutil.copy(fixture("oli_cut.h5oina"), p)
  with h5py.File(p, "r+") as f:
    f["1/EBSD/Header/Scanning Rotation Angle"][()] = np.pi
    f["1/EBSD/Header/Specimen Orientation Euler"][()] = np.array([0, 90, 0], dtype=np.float32)
  with pytest.warns(UserWarning, match="acquisition surface"):
    e = loadEBSD(p, grid=False)
  raw = loadEBSD(p, eulerCorrection=rotation.id, grid=False)
  expected = rotation.byAxisAngle(yvector, np.pi) * rotation.byEuler(0, 90 * degree, 0).inv() * raw.rotations[0]
  assert mtex.angle(e.rotations[0], expected) < 1e-6   # the file holds its angle as float32
  assert e.meta["surface"] == (0.0, 90.0, 0.0)


def test_oxford_scan_rotation_the_file_states_is_applied():
  raw = loadEBSD(fixture("oli_cut.h5oina"), eulerCorrection=rotation.id, grid=False)
  # the scan rotation the file states, in radians since it is below a full turn, on top of the flip about x
  with h5py.File(fixture("oli_cut.h5oina")) as f:
    stated = float(np.asarray(f["1/EBSD/Header/Scanning Rotation Angle"][()]).reshape(-1)[0])
  plain = loadEBSD(fixture("oli_cut.h5oina"), grid=False)
  assert mtex.angle(plain.rotations[0], rotation.byAxisAngle(vector3d(1, 0, 0), np.pi) * rotation.byAxisAngle(vector3d(0, 0, 1), stated) * raw.rotations[0]) < 1e-6


def test_an_unknown_vendor_is_an_error(tmp_path):
  p = tmp_path / "nobody.h5"
  with h5py.File(p, "w") as f:
    f["Manufacturer"] = np.array([b"Nobody"])
    f["Scan/EBSD/Data/Phi1"] = np.zeros(4)
  with pytest.raises(ValueError, match="names no manufacturer"):
    loadEBSD(p)


# ------------------------------------------------------------------ the resolver

def test_h5Resolve_modes():
  paths = ["/1", "/1/EBSD", "/1/EBSD/Data", "/1/EBSD/Data/Phase", "/1/EBSD/Header", "/1/EBSD/Header/X Cells", "/1/EDS/Data/Phase"]
  scope = H5Scope("/1/EBSD", "/1/EBSD", "/1")
  assert h5Resolve(paths, {"mode": "absolute", "value": "/1/EBSD/Data/Phase"}, scope) == ["/1/EBSD/Data/Phase"]
  assert h5Resolve(paths, {"mode": "search_set", "value": "^/Data/Phase$"}, scope) == ["/1/EBSD/Data/Phase"]
  assert h5Resolve(paths, {"mode": "search_free", "value": "/Data/Phase$"}, scope) == ["/1/EBSD/Data/Phase", "/1/EDS/Data/Phase"]
  assert h5Resolve(paths, {"mode": "search_root", "value": "x cells$"}, scope) == ["/1/EBSD/Header/X Cells"]
  assert h5Resolve(paths, {"data": 1}, scope) == []
  assert h5Resolve(paths, "x", scope) == []


def test_the_configs_are_loaded():
  c = hdf5Configs()
  assert set(c) == {"Bruker", "EDAX", "EDAXh5", "EMSphInx", "Oxford", "ThermoFisher"}
  assert c["Oxford"]["ebsd"]["acquisition_surface"]["mode"] == "search_free"


def test_eulerColumns_shapes():
  m = np.arange(12.0).reshape(4, 3)
  first, cols = eulerColumns(m)
  assert not first and np.array_equal(cols[0], m[:, 0])
  first, cols = eulerColumns(m.T)
  assert first and np.array_equal(cols[2], m[:, 2])
  first, cols = eulerColumns(np.arange(24.0).reshape(2, 4, 3))
  assert not first and cols[0].shape == (8,)
  with pytest.raises(ValueError):
    eulerColumns(np.zeros((5, 4)))


def test_h5PhaseShift_reads_the_numbering_off_the_values():
  assert h5PhaseShift(np.array([0, 1, -1]), 2) == 1
  assert h5PhaseShift(np.array([0, 1, 2]), 2) == 0
  assert h5PhaseShift(np.array([0, 1]), 2) == 1


def test_spaceGroupNumber_takes_a_number_not_a_symbol():
  assert spaceGroupNumber(np.array([225.0])) == 225 and spaceGroupNumber("F m#ovl3m") is None
  assert spaceGroupNumber(None) is None and spaceGroupNumber(194) == 194


def test_alignmentList_reads_the_stated_alignment():
  assert alignmentList("a || x, c* || z") == ["X||a", "Z||c*"]
  assert alignmentList("X||a*, Z||c") == ["X||a*", "Z||c"]
  assert alignmentList("EDAX") == ["EDAX"] and alignmentList("") == []


def test_matrixRotations_are_column_major_and_zeros_are_none():
  r = rotation.byEuler(0.3, 0.4, 0.5)
  M = np.array(r.matrix()).reshape(3, 3)
  flat = M.T.reshape(-1)          # written column by column
  rots, none = matrixRotations(np.stack([flat, np.zeros(9)]))
  assert mtex.angle(rots[0], r) < 1e-12 and list(none) == [False, True]


# ------------------------------------------------------------------ the corpus, pinned to the Julia reader

@pytest.mark.skipif(not os.path.isdir(CORPUS), reason="the HDF5 corpus is not here")
@pytest.mark.parametrize("name, n, cnt, euler, prop, mean", [
  ("h5_edax.h5", 56615, [0, 56615], [304.96433441, 57.7159131, 100.76364115], "ci", 0.30648656782842953),
  ("EDAX/KamilaTi.oh5", 70760, [0, 70760], [114.06853843, 175.36763636, 71.61957914], "iq", 32563.99760105614),
  ("Oxford/oli_test.h5oina", 30132, [423, 29709], [240.13360381, 146.45168516, 335.24092444], "bands", 11.343355900703571),
  ("Oxford/oxford.h5", 138890, [5299, 610, 132875, 106], [118.18436011, 133.42662369, 215.5493861], "bandcontrast", 120.74588523291814),
  ("ThermoFisher/Example_dataset_Duplex_steel.tfs.hdf5", 310, [0, 251, 1, 58], [149.03730774, 142.57176208, 24.79652023], None, None),
  ("EMSphinx/Christos Steel Specimen 1 Site 2 Map Data 3_BW88.h5", 485140, [0, 2148, 478052, 4940], [251.63514628, 34.73104247, 124.36628874], "iq", 0.4180047965378519),
  ("Bruker/BCWQ10min2.h5", 666000, [91646, 548007, 26347], [294.0380859375, 159.52241516, 127.85329056], "mad", 0.9073786979317385),
  ("EDAX/KamilaTi.edaxh5", 70760, [0, 70760], [114.06853843, 175.36763636, 71.61957914], "fit", 1.2239011302977851),
])
def test_corpus_matches_julia(name, n, cnt, euler, prop, mean):
  e = _loadEBSD_h5(os.path.join(CORPUS, name))
  assert len(e) == n and counts(e) == cnt
  k, eu = firstEuler(e)
  assert k == 0 and np.allclose(eu, euler, atol=1e-6)
  if prop is not None:
    assert abs(e.prop[prop].astype(float).mean() - mean) < 1e-9 * max(1, abs(mean))


# the pictures a file holds are indexed at the load and read when one is looked at: a
# vendor writes a dozen electron images beside the map, and the map is not made of them
def picturesFile(tmp_path):
  path = tmp_path / "pictures.h5"
  with h5py.File(path, "w") as f:
    # a plane, a stack of one channel, a flat run the header counts, and a colour run
    f["/a/Header/X Cells"] = np.array([40], dtype=np.int32)
    f["/a/Header/Y Cells"] = np.array([30], dtype=np.int32)
    f["/a/Data/plane"] = np.arange(30 * 40, dtype=np.uint8).reshape(30, 40)
    f["/a/Data/stack"] = np.zeros((1, 30, 40), dtype=np.uint16)
    f["/a/Data/flat"] = np.arange(30 * 40, dtype=np.int16)
    f["/a/Data/colour"] = np.zeros((30 * 40, 3), dtype=np.uint8)
    f["/a/Data/tooSmall"] = np.zeros((4, 4), dtype=np.uint8)
    f["/a/Data/notANumber"] = np.array([b"x" * 4096])
  return path


def pictureIndex(path):
  from mtex.io.h5 import h5ImageIndex, h5Paths
  with h5py.File(path, "r") as f:
    paths, groups = h5Paths(f)
    return h5ImageIndex(f, paths, groups)


def test_h5ImageIndex_finds_planes_stacks_and_runs_of_the_header_shape(tmp_path):
  index = pictureIndex(picturesFile(tmp_path))
  found = {e["name"].split("/")[-1]: tuple(e["shape"]) for e in index}
  assert found == {"plane": (30, 40), "stack": (30, 40), "flat": (30, 40), "colour": (30, 40, 3)}


def test_h5ImageIndex_takes_the_maps_own_columns_for_no_pictures(tmp_path):
  from mtex.io.h5 import h5ImageIndex, h5Paths
  path = picturesFile(tmp_path)
  # and the map's own columns, which are flat runs of the same shape, are no pictures
  with h5py.File(path, "r") as f:
    paths, groups = h5Paths(f)
    assert not [e for e in h5ImageIndex(f, paths, groups, n=30 * 40)]


def test_h5Image_reads_a_flat_run_and_a_colour_run_in_the_picture_shape(tmp_path):
  from mtex.io.h5 import h5Image
  path = picturesFile(tmp_path)
  index = pictureIndex(path)
  entry = next(e for e in index if e["name"].endswith("flat"))
  a = h5Image(path, entry)
  assert a.shape == (30, 40) and a[0, 1] == 1
  colour = h5Image(path, next(e for e in index if e["name"].endswith("colour")))
  assert colour.shape == (30, 40, 3) and colour.dtype == np.uint8


OXFORD_PICTURES = os.path.join(CORPUS, "Oxford", "AEVJ1_anon.h5oina")
BRUKER_PICTURES = os.path.join(CORPUS, "h5_bruker.h5")


@pytest.mark.skipif(not os.path.isfile(OXFORD_PICTURES), reason="the Oxford file of the vendor corpus is not on this machine")
def test_pictures_of_an_oxford_file_are_indexed_not_read():
  ebsd = loadEBSD(OXFORD_PICTURES, verbose=False)
  names = [e["name"] for e in ebsd.meta["images"]]
  assert len(names) == 12 and any("Electron Image/Data/SE" in n for n in names)
  assert all(tuple(e["shape"])[:2] == (1536, 2048) or tuple(e["shape"])[:2] == (1539, 2048) for e in ebsd.meta["images"])
  # the pixels are not in the map: the index says where they are
  assert all(isinstance(e, dict) and "path" in e for e in ebsd.meta["images"])


@pytest.mark.skipif(not os.path.isfile(BRUKER_PICTURES), reason="the Bruker file of the vendor corpus is not on this machine")
def test_pictures_of_a_bruker_file_are_indexed():
  ebsd = loadEBSD(BRUKER_PICTURES, verbose=False)
  assert any(e["name"].endswith("SEM Image") for e in ebsd.meta["images"])


def test_eulerColumns_reads_a_three_by_three_data_set_as_columns():
  first, (a, _, _) = eulerColumns(np.arange(9.0).reshape(3, 3))
  assert not first and np.array_equal(a, [0, 3, 6])


def test_separate_lists_the_entries_by_name_without_the_type():
  from mtex.io.h5 import _separate
  assert _separate({"type": "separate", "b": 2, "a": 1}) == [1, 2]


def test_eulerEntry_is_empty_for_a_table_of_matrices():
  from mtex.io.h5 import _eulerEntry
  assert _eulerEntry({"type": "byMatrix"}) == {} and _eulerEntry({"type": "euler_stack", "euler_stack": {"phi": 1}}) == {"phi": 1}
