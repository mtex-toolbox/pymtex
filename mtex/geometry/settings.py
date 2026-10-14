# the user's settings, what MATLAB keeps in mtex_settings.m: a JSON file of preferences read when
# mtex is imported, so that a choice holds in every session. The file is data only; mtex writes it
# when a setting is saved, and a user may edit it by hand
#
#   {
#     "plottingConvention": "y↑→x",
#     "displayRows": 40
#   }
#
# A fresh installation has no such file and keeps the defaults. The environment variable
# MTEX_SETTINGS names another file, or with an empty value reads and writes none, as the tests
# and the doc build do
import json
import os
import sys
import warnings

SETTINGS_NAME = "mtex_settings.json"


def settingsFile():
  """the path of the user's settings file, whether or not it exists: MTEX_SETTINGS when set, else
  mtex_settings.json in the user's configuration folder, %APPDATA%/mtex on Windows and
  $XDG_CONFIG_HOME/mtex or ~/.config/mtex elsewhere; None when MTEX_SETTINGS is empty"""
  given = os.environ.get("MTEX_SETTINGS")
  if given is not None:
    return os.path.abspath(os.path.expanduser(given)) if given.strip() else None
  if sys.platform == "win32" and os.environ.get("APPDATA"):
    root = os.environ["APPDATA"]
  else:
    root = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
  return os.path.join(root, "mtex", SETTINGS_NAME)


def readSettings(path=None):
  """the saved settings as a dict, empty when there is no file; a file that is no JSON object is
  reported as a warning and read as empty"""
  path = settingsFile() if path is None else os.path.abspath(os.path.expanduser(path))
  if path is None or not os.path.isfile(path):
    return {}
  try:
    with open(path, encoding="utf-8") as f:
      data = json.load(f)
  except (OSError, ValueError) as e:
    warnings.warn(f"the settings file {path} could not be read and is left out: {e}", stacklevel=2)
    return {}
  if not isinstance(data, dict):
    warnings.warn(f"the settings file {path} holds no JSON object and is left out", stacklevel=2)
    return {}
  return data


def loadSettings(path=None):
  """applies the saved settings to the session: the key plottingConvention sets the default
  convention, every other key the preference of that name. Returns the settings applied

    loadSettings()                                # the user's file again, after reset() say
    loadSettings('~/lab/mtex_settings.json')      # another file

  A setting that cannot be applied is left out with a warning; a missing file applies
  nothing. The threads and huge pages are applied anew afterwards.

  Input
    path - the settings file, by default the user's: MTEX_SETTINGS when set, else
        mtex_settings.json in the user's configuration folder
  Output
    data - dict of the settings read
  See also
    setMTEXpref, getMTEXpref, plottingConvention.makeDefault"""
  from .session import SESSION
  data = readSettings(path)
  for name, value in data.items():
    try:
      if name == "plottingConvention":
        conventionFromSetting(value).makeDefault()
      else:
        SESSION.prefs[name] = value
    except Exception as e:
      warnings.warn(f"the setting {name} = {value!r} is left out: {e}", stacklevel=2)
  from .session import applyHugePages
  applyHugePages()
  from .kernels import applyThreads
  applyThreads()
  return data


def saveSetting(name, value):
  """stores one setting in the user's file, the others kept; a value None removes it. The file and
  its folder are created when missing; nothing is written when MTEX_SETTINGS is empty"""
  path = settingsFile()
  if path is None:
    return None
  data = readSettings(path)
  if value is None:
    data.pop(name, None)
  else:
    data[name] = value
  os.makedirs(os.path.dirname(path), exist_ok=True)
  # written beside and moved over, so that an interrupted write leaves the old file
  tmp = path + ".tmp"
  with open(tmp, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
    f.write("\n")
  os.replace(tmp, path)
  return path


def conventionSetting(pC):
  """a plotting convention as the settings file stores it: its pictogram 'y↑→x' when it has one,
  else its out of screen and east directions"""
  if pC.isPictogram:
    return pC.char(utf8=True)
  return {"outOfScreen": [float(x) for x in pC.outOfScreen.data.reshape(3)], "east": [float(x) for x in pC.east.data.reshape(3)]}


def conventionFromSetting(value):
  """the plotting convention a stored setting describes"""
  from .plottingconvention import plottingConvention
  from .vector3d import vector3d
  if isinstance(value, dict):
    return plottingConvention(vector3d(*value["outOfScreen"]), vector3d(*value["east"]))
  return plottingConvention(value)


__all__ = ["settingsFile", "loadSettings"]
