import glob
import os

import numpy as np
import pytest

import mtex
from mtex import crystalFrame, degree, export, loadEBSD, rotation, vector3d
from mtex.io.exportebsd import channelCode, exportPhaseNumbers, gridStep, h5PhaseColumn, scanOf, tslCode
from mtex.io.kernels import formatTableNumba, formatTableNumpy

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "ebsd")


def same(a, b, tol=1e-5):
  q = np.isfinite(a.rotations.data[..., 0]) & np.isfinite(b.rotations.data[..., 0])
  assert np.array_equal(np.isfinite(a.rotations.data[..., 0]), np.isfinite(b.rotations.data[..., 0]))
  assert np.array_equal(a.phaseId, b.phaseId)
  assert np.allclose(a.pos.data, b.pos.data, atol=1e-5)
  assert mtex.angle(a.rotations[q], b.rotations[q]).max() < tol


# ------------------------------------------------------------------ the text formats

def test_ctf_round_trip_is_bit_for_bit(tmp_path):
  e = loadEBSD(os.path.join(FIXTURES, "Forsterite_cut.ctf"))
  export(e, tmp_path / "rt.ctf")
  b = loadEBSD(tmp_path / "rt.ctf")
  assert np.array_equal(e.pos.data, b.pos.data)
  same(e, b, 1e-12)
  assert sorted(b.prop) == ["bands", "bc", "bs", "error", "mad"]
  for k in b.prop:
    assert np.array_equal(e.prop[k], b.prop[k])
  assert b.CSList.mineralList == ["notIndexed", "Forsterite", "Enstatite", "Diopside"]
  assert [f.pointGroup for f in b.CSList.frames[1:]] == ["mmm", "mmm", "12/m1"]
  assert b.meta["header"]["XStep"] == 50 and b.meta["header"]["XCells"] == 732 and b.meta["surface"] == (0.0, 0.0, 0.0)


def test_ang_round_trip_to_its_five_decimals(tmp_path):
  e = loadEBSD(os.path.join(FIXTURES, "ferrite_cut.ang"))
  e.export(tmp_path / "rt.ang")
  b = loadEBSD(tmp_path / "rt.ang")
  assert b.shape == e.shape and b.lattice.shape == "hex"
  same(e, b, 1e-5)
  assert b.meta["header"]["XSTEP"] == 0.3 and b.meta["header"]["GRID"] == "HexGrid" and b.meta["header"]["NCOLS_EVEN"] == 233
  assert np.array_equal(e.iq, b.iq) and np.array_equal(e.ci, b.ci)
  assert b.meta["header"]["x_star"] == e.meta["header"]["x_star"]


def test_across_formats_and_only_the_phases_present_are_declared(tmp_path):
  e = loadEBSD(os.path.join(FIXTURES, "Forsterite_cut.ctf"))
  export(e, tmp_path / "x.ang")
  b = loadEBSD(tmp_path / "x.ang")
  # Silicon is declared by the .ctf and never measured, so the .ang does not declare it
  assert b.CSList.mineralList == ["notIndexed", "Forsterite", "Enstatite", "Diopside"]
  same(e, b, 2e-5)
  assert np.array_equal(b.iq, e.bc) and np.array_equal(b.ci, e.bs)
  export(b, tmp_path / "y.ctf")
  c = loadEBSD(tmp_path / "y.ctf")
  same(e, c, 2e-5)


def test_the_correction_is_undone(tmp_path):
  e = loadEBSD(os.path.join(FIXTURES, "Forsterite_cut.ctf"), noGrid=True)
  export(e, tmp_path / "a.ctf", eulerCorrection=rotation.id)
  raw = loadEBSD(tmp_path / "a.ctf", eulerCorrection=rotation.id, noGrid=True)
  same(e, raw, 1e-5)
  export(e, tmp_path / "b.ang", setting=0)
  raw = loadEBSD(tmp_path / "b.ang", setting=0, noGrid=True)
  same(e, raw, 2e-5)


def test_a_tilted_surface_goes_back_into_the_header(tmp_path):
  with open(os.path.join(FIXTURES, "Forsterite_cut.ctf"), "rb") as f:
    data = f.read()
  (tmp_path / "tilt.ctf").write_bytes(data.replace(b"AcqE2\t0", b"AcqE2\t90"))
  with pytest.warns(UserWarning):
    e = loadEBSD(tmp_path / "tilt.ctf", noGrid=True)
  export(e, tmp_path / "back.ctf")
  with pytest.warns(UserWarning):
    b = loadEBSD(tmp_path / "back.ctf", noGrid=True)
  assert b.meta["surface"] == (0.0, 90.0, 0.0)
  same(e, b, 1e-5)


def test_the_table(tmp_path):
  e = loadEBSD(os.path.join(FIXTURES, "Forsterite_cut.ctf"))
  export(e, tmp_path / "t.txt")
  with open(tmp_path / "t.txt") as f:
    head = f.readline().split()
  assert head[:6] == ["x", "y", "phi1", "Phi", "phi2", "phase"] and "mad" in head
  present = [e.CSList.frames[k] for k in np.unique(e.phaseId) if k > 0]
  b = loadEBSD(tmp_path / "t.txt", present)
  assert b.shape == e.shape
  assert np.array_equal(e.isIndexed, b.isIndexed) and np.array_equal(e.mad, b.mad)
  q = e.isIndexed
  assert mtex.angle(e.rotations[q], b.rotations[q]).max() < 2e-6   # the floor of the angle itself
  export(e, tmp_path / "abg.txt", "ABG", radians=True)
  with open(tmp_path / "abg.txt") as f:
    assert f.readline().split()[2:5] == ["alpha", "beta", "gamma"]
  b = loadEBSD(tmp_path / "abg.txt", present, "ABG", radians=True)
  assert mtex.angle(e.rotations[q], b.rotations[q]).max() < 2e-6   # the floor of the angle itself


def test_no_writer_for_the_binary_pair(tmp_path):
  e = loadEBSD(os.path.join(FIXTURES, "ferrite_cut.ang"))
  with pytest.raises(NotImplementedError):
    export(e, tmp_path / "x.cpr")


def test_codes_numbers_and_the_scan():
  assert tslCode(crystalFrame("m-3m")) == 43 and tslCode(crystalFrame("432")) == 43 and tslCode(crystalFrame("-3")) == 3
  assert channelCode(crystalFrame("6mm")) == 9 and channelCode(crystalFrame("2/m", [1, 2, 3], [90, 100, 90])) == 2
  e = loadEBSD(os.path.join(FIXTURES, "olivine_cut.ang"))
  numbers, indexed = exportPhaseNumbers(e)
  assert indexed == [1, 2, 3, 4] and numbers.max() == 4 and numbers.min() == 0
  assert gridStep(np.array([0, 4, 8, 8, 12.0])) == 4 and gridStep(np.array([1.0])) == 1
  stepX, stepY, nOdd, nEven, nRows, hex = scanOf(e)
  assert (stepX, stepY, nOdd, nEven, nRows, hex) == (4.0, 4.0, 223, 223, 4, False)


def test_the_two_formatters_agree():
  rng = np.random.default_rng(2)
  c = np.column_stack([rng.uniform(-400, 400, 50), rng.integers(-3, 300, 50), rng.uniform(0, 1e6, 50), rng.uniform(-1, 1, 50)])
  c[3, 3] = np.nan
  c[4, 3] = -0.00004
  c[5, 3] = 0.0
  a = formatTableNumba(c, [5, -1, 1, 4], b"\t", True)
  b = formatTableNumpy(c, [5, -1, 1, 4], b"\t", True)
  assert a is None or a == b
  a = formatTableNumba(c, [0, 2, -1, 6])
  b = formatTableNumpy(c, [0, 2, -1, 6])
  assert a is None or a == b


# ------------------------------------------------------------------ the HDF5 write-back

@pytest.mark.parametrize("name", sorted(os.path.basename(f) for f in glob.glob(os.path.join(FIXTURES, "h5", "*"))))
def test_h5_round_trip_with_a_changed_map(tmp_path, name):
  pytest.importorskip("h5py")
  e = loadEBSD(os.path.join(FIXTURES, "h5", name), noGrid=True)
  m = e.rotate(rotation.byAxisAngle(vector3d(1, 2, 3), 0.3), keepXY=True)
  key = next(iter(m.prop)) if m.prop else None
  if key is not None:
    m.prop[key] = (np.arange(len(m)) % 100).astype(float)   # within the range of a byte column
  m.prop["gained"] = np.full(len(m), 2.5)
  m.CSList.frames[int(m.CSList.indexedPhaseId[0])].mineral = "Renamed"
  out = tmp_path / name
  export(m, out)
  b = loadEBSD(out, noGrid=True)
  assert len(b) == len(m)
  same(m, b, 1e-6)
  # a fixed length name field cuts a longer name short, with a warning
  got = b.CSList.mineralList[int(m.CSList.indexedPhaseId[0])]
  assert got and "Renamed".startswith(got)
  if key is not None:
    assert np.allclose(b.prop[key], np.arange(len(m)) % 100)
  from mtex.io.h5 import hdf5Configs
  if name.endswith(".edaxh5"):
    assert "gained" not in b.prop     # one record per measurement leaves no room beside it
  elif "additions" in hdf5Configs()[b.meta["vendor"]]:
    assert np.allclose(b.prop["gained"], 2.5)
  # everything else the file held is still there
  import h5py
  with h5py.File(os.path.join(FIXTURES, "h5", name)) as f, h5py.File(out) as g:
    before, after = [], []
    f.visititems(lambda n, o: before.append(n))
    g.visititems(lambda n, o: after.append(n))
    assert set(before) <= set(after)


def test_h5_export_needs_the_record_and_the_rows(tmp_path):
  pytest.importorskip("h5py")
  e = loadEBSD(os.path.join(FIXTURES, "ferrite_cut.ang"))
  with pytest.raises(ValueError, match="no record"):
    export(e, tmp_path / "x.h5")
  h = loadEBSD(os.path.join(FIXTURES, "h5", "kamila_cut.oh5"), noGrid=True)
  with pytest.raises(ValueError, match="lost or gained"):
    export(h[:100], tmp_path / "x.oh5")
  with pytest.raises(ValueError, match="itself"):
    export(h, h.meta["h5"]["file"])
  with pytest.raises(ValueError, match="not there"):
    export(h, tmp_path / "x.oh5", reference=tmp_path / "missing.oh5")
  # another file of the same layout as the reference
  export(h, tmp_path / "y.oh5", reference=os.path.join(FIXTURES, "h5", "kamila_cut.oh5"))
  same(h, loadEBSD(tmp_path / "y.oh5", noGrid=True), 1e-6)


def test_h5PhaseColumn():
  ids = np.array([0, 1, 2, 1])
  assert list(h5PhaseColumn(ids, np.dtype("int32"), False, 0)) == [0, 1, 2, 1]
  assert list(h5PhaseColumn(ids, np.dtype("uint8"), True, 1)) == [255, 0, 1, 0]
  assert list(h5PhaseColumn(ids, np.dtype("int8"), True, 1)) == [-1, 0, 1, 0]
