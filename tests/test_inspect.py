import os
import time

import numpy as np
import pytest

from mtex import inspectEBSD, loadEBSD, rotation, yvector
from mtex.io.inspect import EBSDFileInfo, ebsdFiles, isEBSDFile

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "ebsd")
DATA = os.path.join(os.path.dirname(__file__), "..", "..", "frameSymmetry", "data", "EBSD")


def fixture(name):
  return os.path.join(FIXTURES, name)


# the description agrees with the load on the phases, the properties, the step and the correction
@pytest.mark.parametrize("name", ["ferrite_cut.ang", "olivine_cut.ang", "Forsterite_cut.ctf", "martensite_cut.cpr", "copper_cut.osc"])
def test_text_and_binary_formats(name):
  info = inspectEBSD(fixture(name))
  ebsd = loadEBSD(fixture(name))
  assert isinstance(info, EBSDFileInfo)
  assert [f.name for f in info.phases.frames] == [f.name for f in ebsd.CSList.frames]
  assert sorted(info.properties) == sorted(ebsd.prop)
  assert info.step is not None and info.grid is not None
  assert float(ebsd.EulerCorrection.angle(info.eulerCorrection)) < 1e-12
  assert info.size == os.path.getsize(fixture(name))
  assert info.date is not None


def test_acom_ang():
  info = inspectEBSD(fixture("acom_cut.ang"))
  ebsd = loadEBSD(fixture("acom_cut.ang"))
  assert info.vendor == "ACOM"
  assert [f.name for f in info.phases.frames] == [f.name for f in ebsd.CSList.frames]
  assert float(ebsd.EulerCorrection.angle(info.eulerCorrection)) < 1e-12


def test_hexagonal_ang():
  info = inspectEBSD(fixture("ferrite_cut.ang"))
  assert info.hex and info.grid == (234, 270) and info.points == 63045
  assert info.vendor == "EDAX" and info.eulerAxesNames == ("A1", "A2", "A3")
  assert dict(info.describe())["Euler correction"] == "180° about (1,-1,0)"


def test_ctf():
  info = inspectEBSD(fixture("Forsterite_cut.ctf"))
  assert info.vendor == "Oxford" and info.format == "ctf"
  assert info.grid == (732, 336) and info.step == (50.0, 50.0) and info.points == 245952
  assert info.extent == (731 * 50.0, 335 * 50.0)
  rows = dict(info.describe())
  assert rows["grid"] == "square grid 732 x 336, d = 50 µm"
  assert rows["extent"] == "36.55 mm x 16.75 mm"
  assert rows["phases"] == "Forsterite, Enstatite, Diopside, Silicon"
  assert rows["Euler correction"] == "180° about y"
  assert info.eulerAxesNames == ("X1", "Y1", "Z1")
  assert float(info.eulerCorrection.angle(rotation.byAxisAngle(yvector, np.pi))) < 1e-12


def test_cpr_by_either_name():
  a, b = inspectEBSD(fixture("martensite_cut.cpr")), inspectEBSD(fixture("martensite_cut.crc"))
  assert a.points == b.points == 42420 and a.grid == (707, 60)
  assert sorted(a.properties) == ["bands", "bc", "bs", "error", "mad", "reliabilityindex"]


def test_osc():
  info = inspectEBSD(fixture("copper_cut.osc"))
  assert info.hex and info.grid == (118, 16) and info.points == 1880
  assert [f.name for f in info.phases.frames if f.isIndexed] == ["Copper"]


# the HDF5 files: the data sets, the properties and the correction as the loader reads them
@pytest.mark.parametrize("name", ["bruker_cut.h5", "edax_cut.h5", "emsphinx_cut.h5", "kamila_cut.edaxh5", "kamila_cut.oh5",
                                  "oli_cut.h5oina", "oxford_cut.h5", "thermofisher_cut.hdf5"])
def test_hdf5(name):
  pytest.importorskip("h5py")
  info = inspectEBSD(fixture(os.path.join("h5", name)))
  ebsd = loadEBSD(fixture(os.path.join("h5", name)), verbose=False)
  assert [f.name for f in info.phases.frames] == [f.name for f in ebsd.CSList.frames]
  assert sorted(info.properties) == sorted(ebsd.prop)
  assert ["/" + s for s in info.dataSets] == ebsd.meta["dataSets"]
  assert float(ebsd.EulerCorrection.angle(info.eulerCorrection)) < 1e-12
  assert info.vendor == ebsd.meta["vendor"]


def test_data_sets_and_dates():
  pytest.importorskip("h5py")
  info = inspectEBSD(fixture("h5/emsphinx_cut.h5"))
  assert info.dataSets == ["Scan 1/EBSD", "Scan 2/EBSD", "Scan 3/EBSD", "Scan 4/EBSD"]
  assert dict(info.describe())["data sets"] == "4"
  info = inspectEBSD(fixture("h5/oxford_cut.h5"))
  assert info.date.year == 2018


def test_folder_listing():
  names = ebsdFiles(FIXTURES)
  assert "martensite_cut.cpr" in names and "martensite_cut.crc" not in names
  assert "generic_cut.txt" not in names and "ferrite_cut.ang" in names
  assert isEBSDFile("a.H5OINA") and not isEBSDFile("a.txt")


# a header is not a map: the real files are described in a fraction of their load
@pytest.mark.skipif(not os.path.isdir(DATA), reason="the MATLAB data folder is not beside the repository")
def test_real_files_fast():
  for name, points in (("Forsterite.ctf", 245952), ("martensite.cpr", 343602), ("copper.osc", 16116), ("olivineopticalmap.ang", 49729)):
    t = time.perf_counter()
    info = inspectEBSD(os.path.join(DATA, name))
    assert time.perf_counter() - t < 0.5
    assert info.points == points


# ------------------------------------------------------------------ volumes

@pytest.mark.parametrize("kind", ["dream3d", "xnovo"])
def test_a_volume_is_described_as_EBSD3_load_reads_it(volumeFile, kind):
  from mtex import EBSD3
  path = volumeFile(kind)
  info, e = inspectEBSD(path), EBSD3.load(path)
  assert info.is3d and info.grid == e.shape and info.points == e.size and np.allclose(info.step, e.lattice.spacing)
  assert sorted(info.properties) == sorted(e.prop) and [f.name for f in info.phases.frames] == [f.name for f in e.CSList.frames]
  assert [f.pointGroup for f in info.phases.frames] == [f.pointGroup for f in e.CSList.frames]
  assert info.unit == {"um": "µm"}.get(e.scanUnit, e.scanUnit) and float(info.eulerCorrection.angle()) == 0
  assert np.allclose(info.extent, e.extent().reshape(3, 2) @ [-1, 1])


def test_a_volume_says_its_grid_and_its_grains(volumeFile):
  rows = dict(inspectEBSD(volumeFile()).describe())
  assert rows["vendor"] == "DREAM.3D" and rows["grid"] == "voxel grid 5 x 4 x 3, dx = 0.5, dy = 0.25, dz = 2 µm"
  assert rows["extent"] == "2 µm x 0.75 µm x 4 µm" and rows["grains"] == "feature ids" and rows["points"] == "60"
  rows = dict(inspectEBSD(volumeFile("xnovo")).describe())
  assert rows["vendor"] == "Xnovo" and rows["grid"].endswith("mm") and rows["extent"] == "20 µm x 15 µm x 20 µm" and rows["grains"] == "grain ids"


def test_a_map_is_not_a_volume():
  assert not inspectEBSD(fixture("Forsterite_cut.ctf")).is3d and not any(inspectEBSD(os.path.join(FIXTURES, "h5", n)).is3d for n in os.listdir(os.path.join(FIXTURES, "h5")))


def test_a_dream3d_file_without_voxels_is_refused(tmp_path):
  import h5py
  with h5py.File(tmp_path / "mesh.dream3d", "w") as f:
    f.create_dataset("DataStructure/Geometry/SharedVertexList", data=np.zeros((3, 3)))
  with pytest.raises(ValueError, match="voxel data"):
    inspectEBSD(tmp_path / "mesh.dream3d")
