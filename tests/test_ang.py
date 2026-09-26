import os

import numpy as np
import pytest

import mtex
from mtex import crystalFrame, degree, loadEBSD, rotation, vector3d
from mtex.io.ang import angHeader, angIsEMSphInx, angPhases, eulerCorrectionRotation, loadEBSD_ang

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "ebsd")
DATA = os.path.join(os.path.dirname(__file__), "..", "..", "frameSymmetry", "data", "EBSD")
FERRITE = os.path.join(FIXTURES, "ferrite_cut.ang")
OLIVINE = os.path.join(FIXTURES, "olivine_cut.ang")


def firstEuler(e):
  q = e.rotations.data.reshape(-1, 4)
  return np.array(rotation(q[np.isfinite(q[:, 0])][0]).Euler()) / degree


def test_ferrite_cut_reads_on_its_hexagonal_grid():
  e = loadEBSD(FERRITE)
  assert e.shape == (2, 234)
  assert e.lattice.shape == "hex"
  assert abs(e.lattice.dxy - 0.3) < 1e-5 and abs(e.dPos - 0.3 / np.sqrt(3)) < 1e-5
  assert e.CSList.mineralList == ["notIndexed", "Ferrite"]
  assert e.CS.pointGroup == "m-3m"
  assert list(e.phaseMap) == [-1, 0]
  assert int((e.phaseId == 1).sum()) == 467
  assert e.scanUnit == "um"


def test_ferrite_first_orientation_is_mtex_s():
  e = loadEBSD(FERRITE, noGrid=True)
  assert np.allclose(firstEuler(e), [27.6983, 113.4183, 92.6414], atol=1e-3)


def test_ferrite_header_and_properties():
  e = loadEBSD(FERRITE, noGrid=True)
  h = e.meta["header"]
  assert h["GRID"] == "HexGrid" and h["XSTEP"] == 0.3 and h["NROWS"] == 270
  assert "MaterialName" not in h and "hklFamilies" not in h
  # the two columns after the phase carry MATLAB's names
  assert set(e.prop) == {"iq", "ci", "sem_signal", "fit"}
  assert e.iq.shape == (467,)


def test_setting_zero_leaves_the_angles_as_written():
  e = loadEBSD(FERRITE, setting=0, noGrid=True)
  with open(FERRITE) as f:
    row = next(l for l in f if not l.startswith("#")).split()
  assert np.allclose(firstEuler(e), np.array(row[:3], dtype=float) / degree, atol=1e-6)


def test_eulerCorrection_overrides_the_setting():
  a = loadEBSD(FERRITE, noGrid=True)
  b = loadEBSD(FERRITE, eulerCorrection=eulerCorrectionRotation(2), noGrid=True)
  c = loadEBSD(FERRITE, setting=3, noGrid=True)
  assert mtex.angle(a.rotations[:5], b.rotations[:5]).max() < 1e-12
  assert mtex.angle(a.rotations[:5], c.rotations[:5]).max() > 1e-3


def test_eulerCorrectionRotation():
  assert mtex.angle(eulerCorrectionRotation(2), rotation.byAxisAngle(vector3d(1, -1, 0), np.pi)) < 1e-12
  assert mtex.angle(eulerCorrectionRotation(0), rotation.id) < 1e-12
  with pytest.raises(ValueError):
    eulerCorrectionRotation(5)


def test_olivine_phases_are_placed_by_the_number_they_declare():
  e = loadEBSD(OLIVINE, noGrid=True)
  assert e.CSList.mineralList == ["notIndexed", "olivine", "Dolomite", "Enstatite", "Chalcopyrite"]
  assert [f.pointGroup for f in e.CSList.frames[1:]] == ["mmm", "-3", "mmm", "4/mmm"]
  assert [int((e.phaseId == k).sum()) for k in range(5)] == [0, 652, 11, 28, 9]
  assert list(e.phaseMap) == [0, 1, 2, 3, 4]
  assert np.allclose(e.CSList[4].abc, [5.24, 5.24, 10.3])


def test_olivine_grids_with_padding():
  e = loadEBSD(OLIVINE)
  assert e.shape == (4, 223)
  assert int((e.phaseId == 0).sum()) == 4 * 223 - 700
  # the display leaves the padding out of the rows, its shares are of all cells to two
  # significant digits, as MATLAB writes them
  assert "652 (73%)" in repr(e) and "notIndexed" not in repr(e)


def test_the_edax_alignment_is_x_parallel_a():
  e = loadEBSD(OLIVINE, noGrid=True)
  cs = e.CSList[2]
  assert cs.pointGroup == "-3"
  assert abs(vector3d(cs.axes[0]).angle(mtex.xvector)) < 1e-12


def test_columnNames_rename_the_columns():
  e = loadEBSD(FERRITE, columnNames=["phi1", "Phi", "phi2", "x", "y", "iq", "ci", "phase", "sem", "fit"], noGrid=True)
  assert set(e.prop) == {"iq", "ci", "sem", "fit"}
  with pytest.raises(ValueError, match="column names"):
    loadEBSD(FERRITE, columnNames=["a"] * 20)


def test_too_few_columns(tmp_path):
  p = tmp_path / "short.ang"
  p.write_text("# Phase 1\n# MaterialName Fe\n# Symmetry 43\n1 2 3 4 5\n")
  with pytest.raises(ValueError, match="eight columns"):
    loadEBSD(p)


def test_a_negative_confidence_index_is_not_indexed(tmp_path):
  p = tmp_path / "ci.ang"
  p.write_text("# Phase 1\n# MaterialName Fe\n# Symmetry 43\n# LatticeConstants 2.87 2.87 2.87 90 90 90\n# NumberFamilies 1\n# VERSION 3\n"
               "0 0 0 0 0 5 0.5 1\n0 0 0 1 0 5 -1 1\n0 0 0 0 1 5 0.5 1\n0 0 0 1 1 5 0.5 1\n")
  e = loadEBSD(p, noGrid=True)
  assert list(e.phaseId) == [1, 0, 1, 1]
  assert e.meta["header"] == {"VERSION": 3.0}


EMSPHINX = ("# Phase 1\n# MaterialName \n# Symmetry 43\n# LatticeConstants 2.87 2.87 2.87 90 90 90\n# NumberFamilies 0\n"
            "# Phase 2\n# MaterialName \n# Symmetry 62\n# LatticeConstants 3 3 5 90 90 120\n# NumberFamilies 0\n"
            "0.1 0.2 0.3 0 0 5 0.5 0\n0.1 0.2 0.3 1 0 5 0.5 1\n0.1 0.2 0.3 0 1 5 0.5 255\n0 0 0 1 1 0 0 0\n")


def test_an_emsphinx_file_is_fingerprinted(tmp_path):
  p = tmp_path / "em.ang"
  p.write_text(EMSPHINX)
  e = loadEBSD(p, noGrid=True)
  assert list(e.phaseId) == [1, 2, 0, 0]
  assert list(e.phaseMap) == [-1, 0, 1]
  assert e.CSList.mineralList[1:] == ["phase 1", "phase 2"]
  # the flags overrule the guess
  e = loadEBSD(p, EDAX=True, noGrid=True)
  assert list(e.phaseId) == [0, 1, 0, 0]
  e = loadEBSD(p.with_name("edax.ang").write_text(EMSPHINX.replace("# NumberFamilies 0", "# NumberFamilies 2")) and p.with_name("edax.ang"),
               EMSphInx=True, noGrid=True)
  assert list(e.phaseId) == [1, 2, 0, 0]


def test_angIsEMSphInx_needs_all_three_marks():
  header = ["# Phase 1", "# MaterialName ", "# NumberFamilies 0"]
  _, names, families = angPhases(header)
  assert angIsEMSphInx(header, names, families)
  assert not angIsEMSphInx(header + ["# VERSION 3"], names, families)
  assert not angIsEMSphInx(header, ["Fe"], families)
  assert not angIsEMSphInx([], [], [])


def test_angHeader_leaves_out_the_phase_blocks():
  header = ["# x-star 0.5", "# Phase 1", "# MaterialName Fe", "# Symmetry 43", "# hklFamilies 1 1 1", "# GRID: SqrGrid", "# XSTEP: 2"]
  assert angHeader(header) == {"x_star": 0.5, "GRID": "SqrGrid", "XSTEP": 2.0}


def test_a_header_without_phase_numbers(tmp_path):
  p = tmp_path / "old.ang"
  p.write_text("# MaterialName Fe\n# Symmetry 43\n# MaterialName Mg\n# Symmetry 62\n0 0 0 0 0 5 0.5 1\n0 0 0 1 0 5 0.5 2\n")
  e = loadEBSD(p, noGrid=True)
  assert e.CSList.mineralList == ["notIndexed", "Fe", "Mg"]
  assert [f.pointGroup for f in e.CSList.frames[1:]] == ["m-3m", "6/mmm"]


@pytest.mark.skipif(not os.path.exists(os.path.join(DATA, "ferrite.ang")), reason="the MTEX data folder is not here")
def test_full_ferrite_matches_mtex():
  e = loadEBSD_ang(os.path.join(DATA, "ferrite.ang"))
  assert len(e) == 63045
  assert int(e.isIndexed.sum()) == 63044
  assert np.allclose(firstEuler(e), [27.6983, 113.4183, 92.6414], atol=1e-3)
  g = loadEBSD(os.path.join(DATA, "ferrite.ang"))
  assert g.shape == (270, 234)
  # the lattice read off the header, its steps as the header states them
  assert g.lattice.shape == "hex"
  assert g.lattice.A[0, 0] == 0.3 and g.lattice.A[1, 1] == 0.259808


@pytest.mark.skipif(not os.path.exists(os.path.join(DATA, "olivineopticalmap.ang")), reason="the MTEX data folder is not here")
def test_full_olivine_matches_mtex():
  e = loadEBSD(os.path.join(DATA, "olivineopticalmap.ang"), noGrid=True)
  assert len(e) == 49729
  assert [int((e.phaseId == k).sum()) for k in range(5)] == [0, 44953, 1370, 2311, 1095]
