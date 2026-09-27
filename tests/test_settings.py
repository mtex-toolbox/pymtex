import json
import os
import subprocess
import sys

import pytest

import mtex
from mtex import getMTEXpref, loadSettings, plottingConvention, setMTEXpref, settingsFile, vector3d


def test_the_file_sits_in_the_configuration_folder(monkeypatch, tmp_path):
  monkeypatch.delenv("MTEX_SETTINGS", raising=False)
  monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
  if sys.platform != "win32":
    assert settingsFile() == os.path.join(str(tmp_path), "mtex", "mtex_settings.json")
  # MTEX_SETTINGS names another file, and when empty no file at all
  monkeypatch.setenv("MTEX_SETTINGS", str(tmp_path / "lab.json"))
  assert settingsFile() == str(tmp_path / "lab.json")
  monkeypatch.setenv("MTEX_SETTINGS", "")
  assert settingsFile() is None and loadSettings() == {}


def test_the_file_sets_the_default_convention_and_preferences(tmp_path):
  f = tmp_path / "mtex_settings.json"
  f.write_text('{"plottingConvention": "y↑→x", "displayRows": 40}', encoding="utf-8")
  assert plottingConvention.default.char() == "y↓→x"
  assert loadSettings(str(f)) == {"plottingConvention": "y↑→x", "displayRows": 40}
  assert plottingConvention.default.char() == "y↑→x" and getMTEXpref("displayRows") == 40
  # every vector in the default frame follows, its display included
  assert repr(vector3d(1, 0, 0)).startswith("vector3d (y↑→x)")
  # reset() is the fresh installation again
  mtex.reset()
  assert plottingConvention.default.char() == "y↓→x" and getMTEXpref("displayRows") == 20


def test_saving_writes_the_file_and_keeps_the_other_settings(monkeypatch, tmp_path):
  f = tmp_path / "sub" / "mtex_settings.json"
  monkeypatch.setenv("MTEX_SETTINGS", str(f))
  setMTEXpref("displayRows", 40, save=True)
  plottingConvention.default("y↑→x", save=True)
  assert json.loads(f.read_text(encoding="utf-8")) == {"displayRows": 40, "plottingConvention": "y↑→x"}
  # a convention without a pictogram is stored by its directions and read back
  pC = plottingConvention(vector3d(1, 1, 1), vector3d(1, -1, 0))
  pC.makeDefault(save=True)
  mtex.reset()
  loadSettings()
  assert plottingConvention.default.outOfScreen.angle(pC.outOfScreen) < 1e-12
  assert plottingConvention.default.east.angle(pC.east) < 1e-12 and getMTEXpref("displayRows") == 40
  # a value None removes a setting
  setMTEXpref("displayRows", None, save=True)
  assert "displayRows" not in json.loads(f.read_text(encoding="utf-8"))


def test_without_a_file_nothing_is_saved(monkeypatch, tmp_path):
  monkeypatch.setenv("MTEX_SETTINGS", "")
  setMTEXpref("displayRows", 30, save=True)
  assert getMTEXpref("displayRows") == 30 and not list(tmp_path.iterdir())


def test_a_broken_file_or_setting_warns_and_leaves_mtex_usable(tmp_path):
  f = tmp_path / "broken.json"
  f.write_text("{not json", encoding="utf-8")
  with pytest.warns(UserWarning, match="could not be read"):
    assert loadSettings(str(f)) == {}
  f.write_text('{"plottingConvention": "nonsense", "displayRows": 7}', encoding="utf-8")
  with pytest.warns(UserWarning, match="plottingConvention"):
    loadSettings(str(f))
  assert getMTEXpref("displayRows") == 7 and plottingConvention.default.char() == "y↓→x"


def test_import_reads_the_users_file(tmp_path):
  f = tmp_path / "mtex_settings.json"
  f.write_text('{"plottingConvention": "y↑→x"}', encoding="utf-8")
  env = dict(os.environ, MTEX_SETTINGS=str(f))
  out = subprocess.run([sys.executable, "-c", "import mtex; print(mtex.plottingConvention.default.char())"],
                       env=env, capture_output=True, text=True, check=True).stdout
  assert out.strip() == "y↑→x"
