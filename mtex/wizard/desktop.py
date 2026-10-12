# the import wizard known to a Linux desktop: a task manager under Wayland finds the icon of a
# window not on the window but in the desktop entry named by its app id, so the wizard runs as
# mtex-wizard and writes that entry and its icon into the user's data folder
# (~/.local/share/applications and icons/hicolor/scalable/apps), hidden from the menus
import os
import shutil
import subprocess
import sys

APP_ID = "mtex-wizard"


def dataHome():
  """the user's data folder of the XDG base directories"""
  return os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")


def desktopEntry(icon):
  """the text of the desktop entry: the wizard of this Python, hidden from the menus"""
  return "\n".join([
    "[Desktop Entry]",
    "Type=Application",
    "Name=MTEX import wizard",
    "Comment=Import EBSD maps with MTEX for Python",
    f"Exec={sys.executable} -m mtex.wizard %f",
    f"Icon={icon}",
    f"StartupWMClass={APP_ID}",
    "Categories=Science;Education;",
    "NoDisplay=true",
    "",
  ])


# the content of a file, bytes in mode "rb", text in mode "r"
def _read(path, mode):
  with open(path, mode, encoding=None if "b" in mode else "utf-8") as fh:
    return fh.read()


def installDesktopEntry(root=None):
  """writes the desktop entry and the icon where they are missing or out of date and returns
  the path of the entry; nothing off Linux"""
  if not sys.platform.startswith("linux"):
    return None
  root = root or dataHome()
  source = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.svg")
  icon = os.path.join(root, "icons", "hicolor", "scalable", "apps", APP_ID + ".svg")
  entry = os.path.join(root, "applications", APP_ID + ".desktop")
  try:
    changed = False
    if not os.path.isfile(icon) or _read(icon, "rb") != _read(source, "rb"):
      os.makedirs(os.path.dirname(icon), exist_ok=True)
      shutil.copyfile(source, icon)
      changed = True
    text = desktopEntry(icon)
    if not os.path.isfile(entry) or _read(entry, "r") != text:
      os.makedirs(os.path.dirname(entry), exist_ok=True)
      with open(entry, "w", encoding="utf-8") as fh:
        fh.write(text)
      changed = True
  except OSError:
    return None
  # KDE reads the entries through a cache; rebuilt in the background it knows the new one at once
  cache = shutil.which("kbuildsycoca6") or shutil.which("kbuildsycoca5")
  if changed and cache and root == dataHome():
    subprocess.Popen([cache], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  return entry


__all__ = ["installDesktopEntry", "APP_ID"]
