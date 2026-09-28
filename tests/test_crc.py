import os
import shutil

import numpy as np
import pytest

import mtex
from mtex import degree, loadEBSD, rotation
from mtex.io.crc import cprPhases, crcNames, loadEBSD_crc, readIni

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "ebsd")
DATA = os.path.join(os.path.dirname(__file__), "..", "..", "frameSymmetry", "data", "EBSD")
CUT = os.path.join(FIXTURES, "martensite_cut.cpr")


def firstIndexedEuler(e):
  k = int(np.flatnonzero(e.isIndexed)[0])
  return k, np.array(e.rotations[k].Euler()) / degree


def test_the_cut_reads_on_its_grid():
  e = loadEBSD(CUT)
  assert e.shape == (60, 707)
  assert e.lattice.shape == "square" and e.dPos == 0.5


def test_the_pair_reads_the_same_by_either_name():
  e = loadEBSD(CUT)
  f = loadEBSD(CUT.replace(".cpr", ".crc"))
  assert f.shape == e.shape and np.array_equal(e.phaseId, f.phaseId)


def test_the_cut_matches_the_julia_reader_in_its_phases():
  e = loadEBSD(CUT, noGrid=True)
  assert len(e) == 42420
  assert [int((e.phaseId == k).sum()) for k in range(3)] == [12238, 30182, 0]
  assert e.CSList.mineralList == ["notIndexed", "Iron bcc (old)", "Iron fcc"]
  assert [f.pointGroup for f in e.CSList.frames[1:]] == ["m-3m", "m-3m"]
  assert list(e.phaseMap) == [0, 1, 2]


def test_the_cut_matches_the_julia_reader_in_its_orientations_and_positions():
  e = loadEBSD(CUT, noGrid=True)
  k, euler = firstIndexedEuler(e)
  assert k == 1 and np.allclose(euler, [332.489218806, 44.301505127, 64.471699199], atol=1e-6)
  assert np.allclose(e.pos.data[1], [0.5, 0, 0])


def test_the_cut_matches_the_julia_reader_in_its_properties():
  e = loadEBSD(CUT, noGrid=True)
  assert sorted(e.prop) == ["bands", "bc", "bs", "error", "mad", "reliabilityindex"]
  assert abs(e.mad.mean() - 0.4562887121366055) < 1e-12
  assert abs(e.bc.mean() - 90.66895332390382) < 1e-10


def test_the_header_is_kept():
  e = loadEBSD(CUT, noGrid=True)
  h = e.meta["header"]
  assert h["job"]["xcells"] == 707 and h["job"]["griddistx"] == 0.5
  assert h["phase1"]["structurename"] == "Iron bcc (old)" and h["phase1"]["spacegroup"] == 229
  assert h["fields"]["count"] == 9


def test_the_acquisition_surface_is_kept():
  e = loadEBSD(CUT, noGrid=True)
  assert e.meta["surface"] == (0.0, 0.0, 0.0)


def test_the_correction_is_half_a_turn_about_z():
  a = loadEBSD(CUT, noGrid=True)
  b = loadEBSD(CUT, eulerCorrection=rotation.id, noGrid=True)
  r = rotation.byEuler(np.pi, 0, 0)
  assert mtex.angle(a.rotations[1], r * b.rotations[1]) < 1e-12


def test_readIni_reads_the_sections_as_numbers():
  ini = readIni(CUT)
  assert ini["general"]["version"] == 5.0
  assert ini["acquisitionsurface"]["euler1"] == 0.0
  assert "" not in ini


def test_crcNames_names_the_edx_windows_and_the_unknown_fields():
  ini = {"edxwindows": {"count": 2.0, "window1": "Fe Ka", "window2": "Mn Ka"}}
  assert crcNames(ini, [3, 4, 5, 101, 102, 103, 13]) == ["phi1", "phi", "phi2", "feka", "mnka", "unknown103", "unknown13"]


def test_a_phase_without_a_lattice_is_not_indexed():
  ini = {"phases": {"count": 2.0}, "phase1": {"structurename": "Fe", "a": 2.87, "b": 2.87, "c": 2.87, "alpha": 90.0, "beta": 90.0, "gamma": 90.0, "lauegroup": 11.0},
         "phase2": {"structurename": "x"}}
  with pytest.warns(UserWarning, match="does not describe phase 2"):
    pl = cprPhases(ini, "x.cpr")
  assert len(pl) == 3 and pl[1].pointGroup == "m-3m" and not pl.frames[2].isIndexed


def test_cprPhases_takes_the_laue_group_as_a_symbol():
  ini = {"phases": {"count": 1.0}, "phase1": {"structurename": "Fe", "a": 2.87, "b": 2.87, "c": 2.87, "alpha": 90.0, "beta": 90.0, "gamma": 90.0, "lauegroup": "m-3m"}}
  assert cprPhases(ini, "x.cpr")[1].pointGroup == "m-3m"


def test_half_a_pair_is_an_error(tmp_path):
  shutil.copy(CUT, tmp_path / "half.cpr")
  with pytest.raises(ValueError, match="no .crc"):
    loadEBSD(tmp_path / "half.cpr")
  shutil.copy(CUT.replace(".cpr", ".crc"), tmp_path / "other.crc")
  with pytest.raises(ValueError, match="no .cpr"):
    loadEBSD(tmp_path / "other.crc")


def test_a_truncated_crc_is_an_error(tmp_path):
  shutil.copy(CUT, tmp_path / "short.cpr")
  with open(CUT.replace(".cpr", ".crc"), "rb") as f:
    data = f.read()
  (tmp_path / "short.crc").write_bytes(data[:-100])
  with pytest.raises(ValueError, match="records of 25 bytes"):
    loadEBSD(tmp_path / "short.cpr")


def test_explicit_positions_are_read_from_the_record(tmp_path):
  # a record with x and y written, the same map with its positions stated
  with open(CUT) as f:
    cpr = f.read()
  cpr = cpr.replace("Field1=3\nField2=4\nField3=5\nField4=6\nField5=7\nField6=8\nField7=10\nField8=11\nField9=12\nCount=9",
                    "Field1=1\nField2=2\nField3=3\nField4=4\nField5=5\nCount=5")
  assert "Count=5" in cpr
  e = loadEBSD(CUT, noGrid=True)
  rec = np.zeros(len(e), dtype=[("p", "i1"), ("x", "<f4"), ("y", "<f4"), ("a", "<f4"), ("b", "<f4"), ("c", "<f4")])
  rec["p"] = e.phaseId
  rec["x"], rec["y"] = e.pos.x * 2, e.pos.y * 2
  raw = loadEBSD(CUT, eulerCorrection=rotation.id, noGrid=True)
  rec["a"], rec["b"], rec["c"] = (np.nan_to_num(v) for v in raw.rotations.Euler())
  (tmp_path / "xy.cpr").write_text(cpr)
  rec.tofile(tmp_path / "xy.crc")
  f = loadEBSD(tmp_path / "xy.cpr")
  assert f.shape == (60, 707) and abs(f.dPos - 1.0) < 1e-9
  assert np.array_equal(f.phaseId, e.gridify().phaseId)
  assert f.prop == {}


@pytest.mark.skipif(not os.path.exists(os.path.join(DATA, "martensite.cpr")), reason="the MTEX data folder is not here")
def test_full_martensite_matches_julia():
  e = loadEBSD_crc(os.path.join(DATA, "martensite.cpr"))
  assert len(e) == 343602
  assert [int((e.phaseId == k).sum()) for k in range(3)] == [92415, 251187, 0]
  assert abs(e.mad.mean() - 0.47057671893357395) < 1e-12
  assert loadEBSD(os.path.join(DATA, "martensite.cpr")).shape == (486, 707)


@pytest.mark.skipif(not os.path.exists(os.path.join(DATA, "EDXLMDTi64.cpr")), reason="the MTEX data folder is not here")
def test_full_ti64_matches_julia():
  e = loadEBSD(os.path.join(DATA, "EDXLMDTi64.crc"))
  assert e.shape == (384, 512)
  assert [int((e.phaseId == k).sum()) for k in range(3)] == [10449, 437, 185722]
  assert [f.pointGroup for f in e.CSList.frames[1:]] == ["m-3m", "6/mmm"]
  k, euler = firstIndexedEuler(e.flatten())
  assert k == 1 and np.allclose(euler, [318.21199242, 111.74850644, 10.43622511], atol=1e-6)
  assert e.meta["surface"] == (-90.0, 0.0, 0.0)
  assert abs(e.bc.mean() - 144.2910919189453) < 1e-9
