import os

import numpy as np
import pytest

import mtex
from mtex import crystalFrame, degree, loadEBSD, rotation
from mtex.io.generic import loadEBSD_generic, tableColumns

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "ebsd")
DATA = os.path.join(os.path.dirname(__file__), "..", "..", "frameSymmetry", "data", "EBSD")
GENERIC = os.path.join(FIXTURES, "generic_cut.txt")


def phases():
  return ["notIndexed", crystalFrame("m-3m", mineral="Fe"), crystalFrame("m-3m", mineral="Mg")]


def firstEuler(e):
  q = e.rotations.data.reshape(-1, 4)
  return np.array(rotation(q[np.isfinite(q[:, 0])][0]).Euler()) / degree


def test_the_columns_are_read_off_the_header_line():
  e = loadEBSD(GENERIC, phases(), grid=False)
  assert len(e) == 900
  assert e.CSList.mineralList == ["notIndexed", "Fe", "Mg"]
  assert set(e.prop) == {"index", "mad", "bc", "bs", "bands", "error", "reliabilityindex"}
  assert np.allclose(firstEuler(e), [234.12, 20.846, 25.195], atol=1e-6)
  assert e.mad[1] == 0.6522
  assert list(e.phaseMap) == [0, 1, 2]


def test_the_map_grids():
  e = loadEBSD(GENERIC, phases())
  assert e.shape == (3, 425)


def test_CS_as_one_frame_or_a_phaseList():
  cs = crystalFrame("m-3m", mineral="Fe")
  e = loadEBSD(GENERIC, cs, grid=False)
  assert e.CSList.mineralList == ["notIndexed", "Fe"]
  assert int((e.phaseId == 1).sum()) == int((loadEBSD(GENERIC, phases(), grid=False).phaseId == 1).sum())
  e = loadEBSD(GENERIC, mtex.phaseList(phases()), grid=False)
  assert len(e.CSList) == 3


def test_columnNames_given(tmp_path):
  p = tmp_path / "t.txt"
  p.write_text("0 0 10 20 30 1 0.5\n1 0 10 20 30 2 0.7\n0 1 10 20 30 0 0.1\n1 1 10 20 30 1 0.2\n")
  e = loadEBSD(p, phases(), columnNames=["x", "y", "Euler 1", "Euler 2", "Euler 3", "phase", "mad"], grid=False)
  assert len(e) == 4
  assert set(e.prop) == {"mad"}
  assert list(e.phaseId) == [1, 2, 0, 1]
  assert mtex.angle(e[0].rotations, rotation.byEuler(10 * degree, 20 * degree, 30 * degree)) < 1e-12


def test_columns_select_and_reorder(tmp_path):
  p = tmp_path / "t.txt"
  p.write_text("9 1 0 0 10 20 30\n9 1 1 0 10 20 30\n")
  e = loadEBSD(p, phases(), columnNames=["phase", "x", "y", "phi1", "Phi", "phi2"], columns=[1, 2, 3, 4, 5, 6], grid=False)
  assert list(e.pos.x) == [0, 1]
  assert list(e.phaseId) == [1, 1]


def test_radians_and_the_convention(tmp_path):
  p = tmp_path / "t.txt"
  p.write_text("x y alpha beta gamma\n0 0 0.1 0.2 0.3\n1 0 0.1 0.2 0.3\n")
  e = loadEBSD(p, crystalFrame("m-3m"), "ABG", radians=True, grid=False)
  assert mtex.angle(e[0].rotations, rotation.byEuler(0.1, 0.2, 0.3, "Matthies")) < 1e-12
  e = loadEBSD(p, crystalFrame("m-3m"), grid=False)
  assert mtex.angle(e[0].rotations, rotation.byEuler(0.1 * degree, 0.2 * degree, 0.3 * degree)) < 1e-12


def test_no_phase_column_is_one_phase(tmp_path):
  p = tmp_path / "t.txt"
  p.write_text("x,y,phi1,Phi,phi2\n0,0,1,2,3\n1,0,1,2,3\n")
  e = loadEBSD(p, phases(), delimiter=",", grid=False)
  assert list(e.phaseId) == [1, 1]
  assert list(e.phase) == [1, 1]


def test_header_count_and_delimiter(tmp_path):
  p = tmp_path / "t.txt"
  p.write_text("a b c\nd e f\n0;0;1;2;3;1\n1;0;1;2;3;1\n")
  e = loadEBSD(p, phases(), header=2, delimiter=";", columnNames=["x", "y", "Euler1", "Euler2", "Euler3", "phase"], grid=False)
  assert len(e) == 2


def test_a_file_that_names_nothing_asks_for_columnNames(tmp_path):
  p = tmp_path / "t.txt"
  p.write_text("0 0 1 2 3 1\n1 0 1 2 3 1\n")
  with pytest.raises(ValueError, match="columnNames"):
    loadEBSD(p, phases())
  p.write_text("a b c d e f\n0 0 1 2 3 1\n")
  with pytest.raises(ValueError, match="columnNames"):
    loadEBSD(p, phases())


def test_the_names_must_cover_the_map(tmp_path):
  p = tmp_path / "t.txt"
  p.write_text("x y phi1 phase\n0 0 1 1\n")
  with pytest.raises(ValueError, match="Euler angles"):
    loadEBSD(p, phases())
  with pytest.raises(ValueError, match="names given"):
    loadEBSD(p, phases(), columnNames=["x", "y", "phi1", "Phi", "phi2", "phase"])


def test_no_numbers_no_phases_bad_argument(tmp_path):
  p = tmp_path / "t.txt"
  p.write_text("x y\nno numbers\n")
  with pytest.raises(ValueError, match="row of numbers"):
    loadEBSD(p, phases())
  with pytest.raises(ValueError, match="crystal frame"):
    loadEBSD_generic(GENERIC, None)
  with pytest.raises(TypeError):
    loadEBSD_generic(GENERIC, phases(), 3)


def test_tableColumns():
  assert tableColumns([b"note", b"Index\tPhase\tX\tY\tEuler1\tEuler2\tEuler3\tMAD"]) == \
    ["index", "phase", "x", "y", "phi1", "Phi", "phi2", "mad"]
  assert tableColumns([b"a b c"]) is None
  assert tableColumns([]) is None


@pytest.mark.skipif(not os.path.exists(os.path.join(DATA, "85_829grad_07_09_06.txt")), reason="the MTEX data folder is not here")
def test_full_generic_matches_mtex():
  e = loadEBSD(os.path.join(DATA, "85_829grad_07_09_06.txt"), phases(), grid=False)
  assert len(e) == 60775
  assert [int((e.phaseId == k).sum()) for k in range(3)] == [11411, 48184, 1180]
  assert np.allclose(firstEuler(e), [234.12, 20.846, 25.195], atol=1e-6)
  assert int(np.flatnonzero(e.phaseId == 2)[0]) == 141
