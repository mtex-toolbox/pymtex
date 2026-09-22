import os
import shutil

import numpy as np
import pytest

import mtex
from mtex import mtexdata, mtexdatafile, setMTEXpref
from mtex.io.sampledata import DATASETS, SOURCES, mtexDataPath, testGrainsMap

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "ebsd")
LOCAL = os.path.join(os.path.dirname(__file__), "..", "..", "frameSymmetry", "data")


def test_the_table_lists_every_data_set():
  t = mtexdata()
  names = list(t)
  assert "forsterite" in names and "dubna" in names and len(t) == len(DATASETS)
  text = repr(t)
  assert "available sample data" in text and "forsterite" in text and "and 6 more" in text
  for d in DATASETS:
    if d["artifact"]:
      assert d["artifact"] in SOURCES and d["files"]


def test_an_unknown_name_is_an_error():
  with pytest.raises(ValueError, match="mtexdata\\(\\) lists"):
    mtexdata("mars")


def test_the_data_path_preference_and_environment(monkeypatch, tmp_path):
  setMTEXpref("mtexDataPath", str(tmp_path))
  assert mtexDataPath() == str(tmp_path)
  setMTEXpref("mtexDataPath", None)
  monkeypatch.setenv("MTEX_DATA", str(tmp_path / "env"))
  assert mtexDataPath() == str(tmp_path / "env")
  monkeypatch.delenv("MTEX_DATA")
  assert mtexDataPath().endswith(os.path.join(".mtex", "data"))


def test_a_local_folder_serves_the_files_by_either_layout(tmp_path):
  os.makedirs(tmp_path / "EBSD")
  shutil.copy(os.path.join(FIXTURES, "ferrite_cut.ang"), tmp_path / "EBSD" / "ferrite.ang")
  assert mtexdatafile("ferrite", root=tmp_path) == str(tmp_path / "EBSD" / "ferrite.ang")
  e = mtexdata("Ferrite", root=tmp_path, download=False)
  assert e.CSList.mineralList == ["notIndexed", "Ferrite"] and len(e) == 468
  # the EBSD folder itself
  assert mtexdatafile("ferrite", root=tmp_path / "EBSD") == str(tmp_path / "EBSD" / "ferrite.ang")
  with pytest.raises(FileNotFoundError, match="not under"):
    mtexdatafile("copper", root=tmp_path, download=False)


def test_a_binary_pair_is_fetched_whole_and_read_by_its_first_file(tmp_path):
  os.makedirs(tmp_path / "EBSD")
  for ext in ("cpr", "crc"):
    shutil.copy(os.path.join(FIXTURES, "martensite_cut." + ext), tmp_path / "EBSD" / ("martensite." + ext))
  files = mtexdatafile("martensite", root=tmp_path, download=False)
  assert [os.path.basename(f) for f in files] == ["martensite.cpr", "martensite.crc"]
  e = mtexdata("martensite", root=tmp_path, download=False)
  assert e.CSList[1].pointGroup == "432" and e.CSList[2].pointGroup == "432"


def test_a_download_is_checked_against_its_digest(monkeypatch, tmp_path):
  import mtex.io.sampledata as sd
  source = SOURCES["ferrite"][0]

  def fake(url, path):
    assert url == source["url"]
    shutil.copy(os.path.join(FIXTURES, "ferrite_cut.ang"), path)

  monkeypatch.setattr(sd.urllib.request, "urlretrieve", fake)
  with pytest.raises(OSError, match="digest"):
    mtexdatafile("ferrite", root=tmp_path)
  assert not os.path.exists(tmp_path / "EBSD" / "ferrite.ang")
  monkeypatch.setitem(source, "sha256", sd._sha256(os.path.join(FIXTURES, "ferrite_cut.ang")))
  assert mtexdatafile("ferrite", root=tmp_path) == str(tmp_path / "EBSD" / "ferrite.ang")
  assert os.path.exists(tmp_path / "EBSD" / "ferrite.ang")


def test_testgrains_needs_no_file():
  e = testGrainsMap()
  assert e.shape == (120, 160) and int((e.phaseId == 0).sum()) > 0
  grains = mtexdata("testgrains")
  assert len(grains) == 15


def test_what_waits_on_P4_says_so():
  with pytest.raises(NotImplementedError, match="P4"):
    mtexdata("vpsc", download=False)


@pytest.mark.skipif(not os.path.isdir(os.path.join(LOCAL, "EBSD")), reason="the MTEX data folder is not here")
@pytest.mark.parametrize("name, shape, minerals", [
  ("aachen", (143, 425), ["notIndexed", "Fe", "Mg"]),
  ("sharp", (301, 151), ["notIndexed", "calcite"]),
  ("small", (3721,), ["notIndexed", "Forsterite", "Enstatite", "Diopside", "Silicon"]),
  ("csl", (301, 512), ["notIndexed", "iron"]),
  ("mylonite", (100, 301), ["notIndexed", "Andesina", "Quartz", "Biotite", "Orthoclase"]),
  ("epidote", (38, 1330), ["notIndexed", "Glaucophane", "Epidote", "Pyrope", "omphacite"]),
  ("forsterite", (336, 732), ["notIndexed", "Forsterite", "Enstatite", "Diopside", "Silicon"]),
  ("olivine", (223, 223), ["notIndexed", "olivine", "Dolomite", "Enstatite", "Chalcopyrite"]),
  ("twins", (137, 167), ["notIndexed", "Magnesium"]),
  ("copper", (136, 119), ["notIndexed", "Copper"]),
  ("single", (101, 101), ["notIndexed", "Al"]),
  ("alu", (38, 113), ["notIndexed", "Al"]),
  ("titanium", (97, 84), ["notIndexed", "Titanium (Alpha)"]),
  ("ferrite", (270, 234), ["notIndexed", "Ferrite"]),
  ("dc06", (101, 51), ["notIndexed", "Iron (Alpha)"]),
  ("martensite", (486, 707), ["notIndexed", "Iron bcc (old)", "Iron fcc"]),
  ("emsland", (450, 600), ["notIndexed", "Ferrite, bcc", "Austenite"]),
  ("alphaBetaTitanium", (384, 512), ["notIndexed", "Ti (BETA)", "Ti (alpha)"]),
  ("EMSphinx", (508, 955), ["notIndexed", "Iron bcc", "Iron fcc", "Cobalt"]),
])
def test_every_recipe_on_the_local_data(name, shape, minerals):
  e = mtexdata(name, root=LOCAL, download=False)
  assert e.shape == shape and e.CSList.mineralList == minerals


@pytest.mark.skipif(not os.path.isdir(os.path.join(LOCAL, "EBSD")), reason="the MTEX data folder is not here")
def test_the_recipes_apply_what_the_pages_do():
  e = mtexdata("csl", root=LOCAL, download=False)
  assert int((~e.isIndexed).sum()) == 5 and "filephase" in e.prop
  e = mtexdata("epidote", root=LOCAL, download=False)
  assert not np.isin(e.phase, [3, 4])[e.isIndexed].any()
  e = mtexdata("martensite", root=LOCAL, download=False)
  assert e.CSList[1].pointGroup == "432"
  a = mtexdata("forsterite", root=LOCAL, download=False)
  s = mtexdata("small", root=LOCAL, download=False)
  assert s.pos.x.min() >= 33000 and s.pos.x.max() <= 36000 and s.pos.y.min() >= 4500 and s.pos.y.max() <= 7500
  raw = mtex.loadEBSD(os.path.join(LOCAL, "EBSD", "Forsterite.ctf"), eulerCorrection=mtex.rotation.id)
  k = np.flatnonzero(a.isIndexed.ravel())[0]
  assert mtex.angle(a.rotations.flatten()[k], mtex.rotation.byAxisAngle(mtex.vector3d(1, 0, 0), np.pi) * raw.rotations.flatten()[k]) < 1e-12


@pytest.mark.skipif(not os.path.isdir(os.path.join(LOCAL, "PoleFigure")), reason="the MTEX data folder is not here")
def test_the_pole_figure_sets_read_with_their_recipes():
  from mtex import degree
  pf = mtexdata("dubna", root=LOCAL, download=False)
  assert pf.numPF == 7 and pf.CS.mineral == "Quartz" and pf.CS.pointGroup == "321"
  assert pf.allH[2].size == 2 and np.allclose(pf.c[2], [0.52, 1.23]) and pf.allI[0].shape == (72, 19)
  assert np.isclose(pf.resolution, 5 * degree) and np.isclose(pf.max()[1], 89.8)
  g = mtexdata("geesthacht", root=LOCAL, download=False)
  assert g.numPF == 4 and np.allclose(g.h.hkl, [[1, 0, 4], [1, 0, 4], [1, 1, 0], [1, 1, 0]])
  p = mtexdata("ptx", root=LOCAL, download=False)
  assert p.numPF == 3 and p.CS.pointGroup == "mmm" and np.allclose(p.h.hkl, [[1, 0, 4], [1, 1, 0], [2, 0, 2]])
  s = mtexdata("steel", root=LOCAL, download=False)
  assert s.numPF == 3 and np.allclose(s.h.hkl, [[1, 1, 0], [2, 0, 0], [2, 1, 1]])
  # the steel specimen is turned by 90 degree about z
  raw = mtexdata("steel", root=LOCAL, download=False)
  assert np.allclose(s.allR[0].data, raw.allR[0].data)
  z = mtexdata("zncuti", root=LOCAL, download=False)
  assert z.numPF == 8 and z.CS.mineral == "Zn"
  assert mtexdata("aachenexp", root=LOCAL, download=False).numPF == 4
