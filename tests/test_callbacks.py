import os

import numpy as np
import pytest

from mtex import plottingConvention, rotation, zvector
from mtex.geometry.plottingconvention import AXIS_CONVENTIONS
from mtex.wizard import callbacks as cb
from mtex.wizard.state import WizardState

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "ebsd")


@pytest.fixture
def st():
  plottingConvention.ij.makeDefault()
  s = WizardState(FIXTURES)
  cb.setFolder(s, FIXTURES)
  return s


def test_browser(st):
  names = [r.name for r in st.entries]
  assert names[0] == "h5" and st.entries[0].isDir and "martensite_cut.cpr" in names and "martensite_cut.crc" not in names
  cb.toggleFolder(st, 0)
  assert st.entries[0].open and st.entries[1].depth == 1 and st.entries[1].name.endswith(".h5")
  n = len(st.entries)
  cb.toggleFolder(st, 0)
  assert not st.entries[0].open and len(st.entries) < n
  crumbs = cb.breadcrumbs(st)
  assert crumbs[-1][1] == FIXTURES and crumbs[0][1] == os.sep


def test_describe_sets_the_two_frames(st):
  cb.selectEntry(st, os.path.join(FIXTURES, "Forsterite_cut.ctf"))
  assert st.info is not None and st.status[0] == "idle" and not st.hasMap
  assert AXIS_CONVENTIONS[st.mapConvention] == "y↓→x" and st.eulerAxesNames == ("X1", "Y1", "Z1")
  # the correction the two frames compose is the vendor's
  assert float(cb.correctionOf(st).angle(st.info.eulerCorrection)) < 1e-12
  cb.selectEntry(st, os.path.join(FIXTURES, "ferrite_cut.ang"))
  assert float(cb.correctionOf(st).angle(st.info.eulerCorrection)) < 1e-12 and st.eulerAxesNames == ("A1", "A2", "A3")
  # a file that is no map fails in the status line and leaves the state usable
  cb.selectEntry(st, os.path.join(FIXTURES, "generic_cut.txt"))
  assert st.status[0] == "failed" and st.info is None


def test_import_and_views(st):
  seen = []
  st.on("status", lambda: seen.append(st.status[0]))
  cb.openEntry(st, os.path.join(FIXTURES, "Forsterite_cut.ctf"))
  assert seen == ["idle", "loading", "loaded"] and st.hasMap and st.loadSeconds is not None
  assert st.tabs[:4] == ["phase", "IPF X", "IPF Y", "IPF Z"] and st.tabs[-1] == "pole figures"
  assert "bc" in st.tabs and st.currentTab == "IPF Z"
  assert st.drawn == {0: True, 1: True, 2: True, 3: True, 4: True} and st.ebsd.lattice is not None
  assert cb.dominantPhase(st) == 1
  cb.pickTab(st, 0)
  assert st.currentTab == "phase"
  cb.setMiller(st, 0, "110")
  assert st.millers[0] == "110"
  assert len(st.metaRows) and cb.visibleMeta(st)
  # the load on a worker: the window's shape of the call
  ran = []

  def worker(load, commit):
    ran.append("load")
    commit(load())

  cb.openEntry(st, os.path.join(FIXTURES, "ferrite_cut.ang"), worker=worker)
  assert ran == ["load"] and st.status[0] == "loaded" and st.ebsd.size > 0


def test_euler_convention_in_place(st):
  cb.openEntry(st, os.path.join(FIXTURES, "Forsterite_cut.ctf"))
  q = lambda: st.ebsd.rotations.data.reshape(-1, 4)[10].copy()
  before, k = q(), st.eulerConvention
  other = (k + 3) % 8
  cb.setEulerConvention(st, other)
  assert float(st.ebsd.EulerCorrection.angle(cb.correctionOf(st))) < 1e-12
  assert not (np.allclose(q(), before) or np.allclose(q(), -before))
  cb.setEulerConvention(st, k)
  assert np.allclose(q(), before, atol=1e-12) or np.allclose(q(), -before, atol=1e-12)
  # the map convention only re-views: the session default follows it
  cb.setMapConvention(st, 0)
  assert plottingConvention.default == plottingConvention("y↑→x")
  plottingConvention.ij.makeDefault()


def test_phase_edits(st):
  cb.openEntry(st, os.path.join(FIXTURES, "Forsterite_cut.ctf"))
  cb.editPhase(st, 1, "a", 4.8)
  assert abs(st.ebsd.CSList.frames[1].abc[0] - 4.8) < 1e-9
  cb.editPhase(st, 3, "symmetry", "m-3m")
  f = st.ebsd.CSList.frames[3]
  assert f.pointGroup == "m-3m" and np.allclose(f.abc, f.abc[0]) and cb.freeCells(f) == {"a": True, "b": False, "c": False, "alpha": False, "beta": False, "gamma": False}
  cb.editPhase(st, 2, "color", "red")
  assert st.ebsd.CSList.frames[2].color == "red"
  cb.editPhase(st, 2, "alignment", "X||c, Z||a*")
  assert cb.closestAlignment(st.ebsd.CSList.frames[2]) == ["X||c", "Z||a*"]
  cb.editPhase(st, 4, "mineral", "Si")
  assert st.ebsd.CSList.frames[4].name == "Si"
  cb.editPhase(st, 1, "plot", False)
  assert st.drawn[1] is False and cb.dominantPhase(st) != 1
  assert cb.snapLattice("6/mmm", [3, 4, 5], [80, 80, 80], "a") == ([3, 3, 5], [90.0, 90.0, 120.0])
  assert cb.snapLattice("2/m", [3, 4, 5], [90, 100, 90], "a") == ([3, 4, 5], [90.0, 100.0, 90.0])


def test_outputs(st, tmp_path):
  cb.openEntry(st, os.path.join(FIXTURES, "Forsterite_cut.ctf"))
  text = cb.scriptOf(st)
  assert "EBSD.load(fname)" in text and "csList" not in text
  cb.editPhase(st, 1, "mineral", "Fo")
  assert "csList" in cb.scriptOf(st) and "mineral='Fo'" in cb.scriptOf(st)
  assert os.path.basename(cb.defaultScriptPath(st)) == "import_Forsterite_cut.py"
  path = cb.writeScript(st, os.path.join(tmp_path, "mine.py"))
  assert os.path.isfile(path) and open(path).read() == cb.scriptOf(st) and st.status[1].startswith("written")


def test_data_sets(st):
  pytest.importorskip("h5py")
  cb.openEntry(st, os.path.join(FIXTURES, "h5", "emsphinx_cut.h5"))
  assert len(st.info.dataSets) == 4 and st.ebsd.meta["dataSet"] == "/Scan 1/EBSD"
  cb.setDataSet(st, 2)
  assert st.ebsd.meta["dataSet"] == "/Scan 3/EBSD" and "dataSet=2" in cb.scriptOf(st)


# the pictures a file holds: the ones its loader indexed and the ones that came in the
# header, each named in the tree, and picking one there shows it in the view for pictures
def test_pictures(st):
  cb.openEntry(st, os.path.join(FIXTURES, "Forsterite_cut.ctf"))
  assert st.images == [] and "images" not in st.tabs
  # a map whose metadata holds a picture and whose file indexes two more
  ebsd = st.ebsd
  ebsd.meta = dict(ebsd.meta, header=dict(ebsd.meta.get("header", {}), **{"SEM Image": np.zeros((1, 40, 60), dtype=np.uint16)}),
                   images=[{"name": "/1/Electron Image/Data/SE/one", "path": "/x", "shape": (30, 40), "dtype": "int16"},
                           {"name": "/1/Layered Image/2/Data/Color", "path": "/y", "shape": (30, 40, 3), "dtype": "uint8"}])
  cb.commitMap(st, ebsd, 0.0)
  assert [name for name, _ in st.images] == ["SE/one", "2/Color", "header/SEM Image"]
  assert "images" in st.tabs
  # the indexed ones stand in a branch of their own, the read one on its own row
  rows = {r[1]: r for r in st.metaRows if r[4] is not None}
  assert set(rows) == {"SE/one", "2/Color", "SEM Image"}
  assert rows["SEM Image"][2] == "picture 60 x 40" and rows["2/Color"][2] == "picture 40 x 30 colour"
  # picking a picture shows it and moves to the pictures
  i = next(i for i, r in enumerate(st.metaRows) if r[1] == "SEM Image")
  cb.pickMeta(st, i)
  assert st.currentTab == "images" and st.images[st.image][0] == "header/SEM Image"
  # the one that was read comes back as it is; the indexed ones would be read from the file
  assert cb.pictureOf(st, st.image).shape == (40, 60)
  assert cb.asPicture(np.zeros((3, 40, 60), dtype=np.uint8)).shape == (40, 60, 3)
  assert cb.asPicture(np.zeros((4, 4))) is None and cb.asPicture("not an array") is None


def test_parent_and_letter_rows():
  st = WizardState(FIXTURES)
  cb.setFolder(st, FIXTURES)
  k = next(i for i, r in enumerate(st.entries) if r.isDir)
  cb.toggleFolder(st, k)
  assert cb.parentRow(st, k + 1) == k and cb.parentRow(st, k) is None
  names = [r.name.lower() for r in st.entries]
  f = [j for j, n in enumerate(names) if n.startswith("f")]
  assert len(f) == 2 and cb.letterRow(st, None, "F") == f[0] and cb.letterRow(st, f[0], "f") == f[1] and cb.letterRow(st, f[1], "f") == f[0]
  assert cb.letterRow(st, 0, "#") is None


# two maps clicked one after the other, the first read the slower: only the second is taken
def test_only_the_latest_import_is_taken():
  st = WizardState(FIXTURES)
  jobs = []
  worker = lambda load, commit: jobs.append((load, commit))
  cb.openEntry(st, os.path.join(FIXTURES, "Forsterite_cut.ctf"), worker)
  cb.openEntry(st, os.path.join(FIXTURES, "olivine_cut.ang"), worker)
  (loadA, commitA), (loadB, commitB) = jobs
  commitB(loadB())
  commitA(loadA())
  assert st.status[0] == "loaded" and st.status[1].startswith("olivine_cut.ang") and st.selected.endswith("olivine_cut.ang")
  assert st.ebsd.size == loadB()[0].size
