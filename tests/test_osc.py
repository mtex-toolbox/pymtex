import os

import numpy as np
import pytest

import mtex
from mtex import degree, loadEBSD, rotation
from mtex.io.ang import eulerCorrectionRotation
from mtex.io.osc import OSC_HEADER_CLOSE, OSC_HEADER_OPEN, loadEBSD_osc, oscPhaseAt, oscPhases

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "ebsd")
DATA = os.path.join(os.path.dirname(__file__), "..", "..", "frameSymmetry", "data", "EBSD")
CUT = os.path.join(FIXTURES, "copper_cut.osc")


def test_the_cut_matches_the_julia_reader():
  e = loadEBSD(CUT, noGrid=True)
  assert len(e) == 1880
  assert e.CSList.mineralList == ["notIndexed", "Copper"]
  assert e.CS.pointGroup == "m-3m" and np.allclose(e.CS.abc, 3.61)
  assert int(e.isIndexed.sum()) == 1880
  assert np.allclose(np.array(e.rotations[0].Euler()) / degree, [235.93265066, 95.83422214, 134.19629699], atol=1e-6)
  assert np.allclose(e.pos.data[-1], [507.5, 64.9519043, 0], atol=1e-5)
  assert sorted(e.prop) == ["confidenceindex", "extra1", "extra2", "extra3", "extra4", "fit", "imagequality", "semsignal"]
  assert abs(e.imagequality.mean() - 57680.502202460106) < 1e-6
  assert abs(e.confidenceindex.mean() - 0.5997622281253496) < 1e-12
  assert e.meta["header"]["xstep"] == 5.0 and abs(e.meta["header"]["ystep"] - 4.330127239227295) < 1e-12
  assert e.meta["header"]["nColumns"] == 118 and e.meta["header"]["nRows"] == 16 and e.meta["header"]["operator"] == "Administrator"


def test_the_cut_reads_on_its_hexagonal_grid_from_the_header():
  e = loadEBSD(CUT)
  assert e.shape == (16, 119)
  assert e.lattice.shape == "hex"
  # the cut is not whole rows, so the lattice is inferred from the positions here
  assert abs(e.lattice.A[0, 0] - 5.0) < 1e-6
  assert int(e.isIndexed.sum()) == 1880


def test_the_setting_and_the_correction():
  a = loadEBSD(CUT, noGrid=True)
  b = loadEBSD(CUT, setting=0, noGrid=True)
  assert mtex.angle(a.rotations[0], eulerCorrectionRotation(2) * b.rotations[0]) < 1e-12
  c = loadEBSD(CUT, eulerCorrection=rotation.id, noGrid=True)
  assert mtex.angle(b.rotations[0], c.rotations[0]) < 1e-12


def test_a_phase_record_is_recognised_by_being_one():
  with open(CUT, "rb") as f:
    b = f.read()
  first, close = b.find(OSC_HEADER_OPEN) + 8, b.find(OSC_HEADER_CLOSE)
  pl = oscPhases(b, first, close)
  assert pl.mineralList == ["notIndexed", "Copper"]
  # a name without a cell is not a record
  fake = b"Copperoid".ljust(256, b"\x00") + (43).to_bytes(4, "little") + b"\x00" * 28
  assert oscPhaseAt(fake, 0) is None
  cell = np.array([3.6, 3.6, 3.6, 90, 90, 90], dtype="<f4").tobytes()
  rec = b"Copperoid".ljust(256, b"\x00") + (43).to_bytes(4, "little") + cell + b"\x00" * 4
  fr = oscPhaseAt(rec, 0)
  assert fr.mineral == "Copperoid" and fr.pointGroup == "m-3m"
  bad = b"Copperoid".ljust(256, b"\x00") + (43).to_bytes(4, "little") + np.array([3.6, 3.6, 3.6, 90, 200, 90], dtype="<f4").tobytes() + b"\x00" * 4
  assert oscPhaseAt(bad, 0) is None
  # two records in one block, with a tail of reflections between them
  block = rec + b"\x01\x02" * 50 + rec.replace(b"Copperoid", b"Silverish")
  assert oscPhases(block, 0, len(block)).mineralList == ["notIndexed", "Copperoid", "Silverish"]


def test_short_and_markerless_files_are_errors(tmp_path):
  p = tmp_path / "x.osc"
  p.write_bytes(b"\x00" * 10)
  with pytest.raises(ValueError, match="too short"):
    loadEBSD(p)
  p.write_bytes(b"\x00" * 16 + (3).to_bytes(4, "little") * 3 + b"\x00" * 100)
  with pytest.raises(ValueError, match="no header marker"):
    loadEBSD(p)


def test_a_negative_confidence_index_is_not_indexed(tmp_path):
  with open(CUT, "rb") as f:
    b = bytearray(f.read())
  e = loadEBSD(CUT, noGrid=True)
  # the confidence index of the first record sits at column 7
  from mtex.io.osc import oscColumnCount, oscDataStart
  start, xstep, _ = oscDataStart(bytes(b), b.find(OSC_HEADER_CLOSE), 1880)
  m = oscColumnCount(bytes(b), start, xstep)
  b[start + 4 * 6:start + 4 * 7] = np.array([-1], dtype="<f4").tobytes()
  p = tmp_path / "ci.osc"
  p.write_bytes(bytes(b))
  f = loadEBSD(p, noGrid=True)
  assert int(f.isIndexed.sum()) == 1879 and not f.isIndexed[0]
  assert m == 14


@pytest.mark.skipif(not os.path.exists(os.path.join(DATA, "copper.osc")), reason="the MTEX data folder is not here")
def test_full_copper_matches_julia():
  e = loadEBSD_osc(os.path.join(DATA, "copper.osc"))
  assert len(e) == 16116 and int(e.isIndexed.sum()) == 16116
  assert abs(e.imagequality.mean() - 53823.96678839158) < 1e-6
  g = loadEBSD(os.path.join(DATA, "copper.osc"))
  assert g.shape == (136, 119) and g.lattice.A[0, 0] == 5.0
