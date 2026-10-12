import os

import numpy as np
import pytest

import mtex
from mtex import degree, loadEBSD, rotation
from mtex.io.ctf import ctfHeader, _loadEBSD_ctf, oxfordCorrection

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "ebsd")
DATA = os.path.join(os.path.dirname(__file__), "..", "..", "frameSymmetry", "data", "EBSD")
FORSTERITE = os.path.join(FIXTURES, "Forsterite_cut.ctf")


def firstEuler(e):
  q = e.rotations.data.reshape(-1, 4)
  return np.array(rotation(q[np.isfinite(q[:, 0])][0]).Euler()) / degree


def test_forsterite_cut_reads_on_its_grid():
  e = loadEBSD(FORSTERITE)
  assert e.shape == (2, 732)
  assert e.lattice.shape == "square" and abs(e.dPos - 50) < 1e-9


def test_forsterite_cut_reads_its_phases():
  e = loadEBSD(FORSTERITE)
  assert e.CSList.mineralList == ["notIndexed", "Forsterite", "Enstatite", "Diopside", "Silicon"]
  assert [f.pointGroup for f in e.CSList.frames[1:]] == ["mmm", "mmm", "12/m1", "m-3m"]
  assert [int((e.phaseId == k).sum()) for k in range(5)] == [301, 1029, 113, 21, 0]
  assert list(e.phaseMap) == [0, 1, 2, 3, 4]
  assert np.allclose(e.CSList[3].abg / degree, [90, 105.63, 90])


def test_forsterite_first_orientation_is_mtex_s():
  e = loadEBSD(FORSTERITE, grid=False)
  assert np.allclose(firstEuler(e), [264.675, 137.65, 113.67], atol=1e-3)


def test_the_columns_are_named_by_the_file():
  e = loadEBSD(FORSTERITE, grid=False)
  assert set(e.prop) == {"bands", "error", "mad", "bc", "bs"}
  assert e.bc[0] == 138 and e.bs[0] == 255 and e.mad[0] == 0.1


def test_the_header_is_kept_without_its_phases():
  e = loadEBSD(FORSTERITE, grid=False)
  h = e.meta["header"]
  assert h["XCells"] == 732 and h["XStep"] == 50 and h["JobMode"] == "Grid" and h["Mag"] == 35
  assert "Phases" not in h


def test_the_acquisition_surface_is_kept():
  e = loadEBSD(FORSTERITE, grid=False)
  assert e.meta["surface"] == (0.0, 0.0, 0.0)


def test_eulerCorrection_identity_gives_the_angles_as_written():
  e = loadEBSD(FORSTERITE, eulerCorrection=rotation.id, grid=False)
  assert np.allclose(firstEuler(e), [84.675, 137.65, 113.67], atol=1e-3)


def test_decimal_comma(tmp_path):
  with open(FORSTERITE, "rb") as f:
    data = f.read()
  head, block = data.split(b"Phase\tX\tY", 1)
  p = tmp_path / "comma.ctf"
  p.write_bytes(head + b"Phase\tX\tY" + block.replace(b".", b","))
  a = loadEBSD(p, grid=False)
  b = loadEBSD(FORSTERITE, grid=False)
  assert np.array_equal(a.pos.data, b.pos.data)
  assert np.array_equal(a.mad, b.mad)


def test_a_tilted_surface_is_undone_with_a_warning(tmp_path):
  with open(FORSTERITE, "rb") as f:
    data = f.read()
  p = tmp_path / "tilt.ctf"
  p.write_bytes(data.replace(b"AcqE2\t0", b"AcqE2\t90"))
  with pytest.warns(UserWarning, match="acquisition surface"):
    e = loadEBSD(p, grid=False)
  raw = loadEBSD(FORSTERITE, eulerCorrection=rotation.id, grid=False)
  expected = oxfordCorrection((0.0, 90.0, 0.0)) * raw.rotations[0]
  assert mtex.angle(e.rotations[0], expected) < 1e-12
  assert mtex.angle(e.rotations[0], loadEBSD(FORSTERITE, grid=False).rotations[0]) > 1e-3
  assert e.meta["surface"] == (0.0, 90.0, 0.0)


def test_oxfordCorrection_isHalfATurnAboutZ_whateverTheFirstSurfaceAngle():
  assert mtex.angle(oxfordCorrection(None), rotation.byEuler(np.pi, 0, 0)) < 1e-12
  assert mtex.angle(oxfordCorrection((30.0, 0.0, 0.0)), rotation.byEuler(np.pi, 0, 0)) < 1e-12


def test_oxfordCorrection_undoesTheSurfaceTilt():
  r = oxfordCorrection((0.0, 0.0, 45.0))
  assert mtex.angle(r, rotation.byEuler(np.pi, 0, 0) * rotation.byEuler(0, 0, 45 * degree).inv()) < 1e-12


def test_ctfHeader_drops_the_note_of_an_odd_line():
  h = ctfHeader(["Channel Text File", "XCells\t3", "note\tMag\t35\tKV\t17"])
  assert h == {"XCells": 3.0, "Mag": 35.0, "KV": 17.0}


def test_not_a_channel_file_is_an_error(tmp_path):
  p = tmp_path / "x.ctf"
  p.write_text("hello\n")
  with pytest.raises(ValueError, match="Channel Text File"):
    loadEBSD(p)


def test_a_channel_file_without_phases_is_an_error(tmp_path):
  p = tmp_path / "x.ctf"
  p.write_text("Channel Text File\nXCells\t2\n")
  with pytest.raises(ValueError, match="declares no phases"):
    loadEBSD(p)


def test_a_phase_without_a_laue_class_falls_back_to_the_space_group(tmp_path):
  p = tmp_path / "sg.ctf"
  p.write_text("Channel Text File\nXStep\t1\nYStep\t1\nPhases\t2\n3;3;3\t90;90;90\tA\t\t225\n3;3;3\t90;90;90\tB\tx\t\n"
               "Phase\tX\tY\tEuler1\tEuler2\tEuler3\n1\t0\t0\t1\t2\t3\n2\t1\t0\t1\t2\t3\n")
  e = loadEBSD(p, grid=False)
  assert [f.pointGroup for f in e.CSList.frames[1:]] == ["m-3m", "-1"]
  assert set(e.prop) == set()


@pytest.mark.skipif(not os.path.exists(os.path.join(DATA, "Forsterite.ctf")), reason="the MTEX data folder is not here")
def test_full_forsterite_matches_mtex():
  e = _loadEBSD_ctf(os.path.join(DATA, "Forsterite.ctf"))
  assert len(e) == 245952
  assert [int((e.phaseId == k).sum()) for k in range(5)] == [58485, 152345, 26058, 9064, 0]
  assert np.allclose(firstEuler(e), [264.675, 137.65, 113.67], atol=1e-3)
  assert loadEBSD(os.path.join(DATA, "Forsterite.ctf")).shape == (336, 732)


def test_surfaceTilt_is_none_for_a_surface_without_tilt():
  from mtex.io.ctf import _surfaceTilt
  assert _surfaceTilt(None) is None and _surfaceTilt((30.0, 0.0, 0.0)) is None


def test_surfaceTilt_undoes_the_second_and_third_angle():
  from mtex.io.ctf import _surfaceTilt
  assert np.allclose(_surfaceTilt((0.0, 90.0, 0.0)).data, rotation.byEuler(0.0, np.pi / 2, 0.0, "Bunge").inv().data)


def test_ctfHead_reads_header_phases_surface_and_the_column_line():
  from mtex.io.ctf import _ctfHead
  data = (b"Channel Text File\r\nXStep\t2\r\nAcqE1\t0\r\nAcqE2\t10\r\nAcqE3\t0\r\nPhases\t1\r\n"
          b"4.756;10.207;5.98\t90;90;90\tForsterite\t3\t0\r\nPhase\tX\tY\r\n")
  header, phases, acq, offset = _ctfHead(data, "f.ctf")
  assert (header["XStep"], phases.mineralList, acq, data[offset:]) == (2.0, ["notIndexed", "Forsterite"], (0.0, 10.0, 0.0), b"Phase\tX\tY\r\n")
