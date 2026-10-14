"""the import wizard: a window over the loaders (docs/plans/wizard.md)

  importWizard()              browse the current folder
  importWizard('/data/ebsd')  a folder, or a file to open on start
  mtex-wizard [path]          the same from a terminal, or python -m mtex.wizard [path]

The core - the state, the callbacks, the script - needs no window and is what the tests
drive; the window needs PySide6, the `wizard` extra, and is imported only when asked for."""
import os
import sys

from .state import WizardState

# the system libraries Qt loads on Linux, which pip cannot install, each with its Debian package
_QT_LIBS = {"libGL.so.1": "libgl1", "libEGL.so.1": "libegl1", "libxkbcommon.so.0": "libxkbcommon0",
            "libfontconfig.so.1": "libfontconfig1", "libglib-2.0.so.0": "libglib2.0-0", "libdbus-1.so.3": "libdbus-1-3"}
_PLATFORM_LIBS = {
  "xcb": {"libX11-xcb.so.1": "libx11-xcb1", "libxkbcommon-x11.so.0": "libxkbcommon-x11-0", "libxcb-cursor.so.0": "libxcb-cursor0",
          "libxcb-icccm.so.4": "libxcb-icccm4", "libxcb-image.so.0": "libxcb-image0", "libxcb-keysyms.so.1": "libxcb-keysyms1",
          "libxcb-randr.so.0": "libxcb-randr0", "libxcb-render-util.so.0": "libxcb-render-util0", "libxcb-shape.so.0": "libxcb-shape0",
          "libxcb-shm.so.0": "libxcb-shm0", "libxcb-sync.so.1": "libxcb-sync1", "libxcb-xfixes.so.0": "libxcb-xfixes0",
          "libxcb-xkb.so.1": "libxcb-xkb1", "libxcb-glx.so.0": "libxcb-glx0"},
  "wayland": {"libwayland-client.so.0": "libwayland-client0", "libwayland-cursor.so.0": "libwayland-cursor0"},
}


def _checkQtLibraries():
  """raises with the system libraries the window misses and the command that installs them; Qt
  itself ends the process when its platform plugin does not load"""
  if not sys.platform.startswith("linux"):
    return
  import ctypes
  platform = os.environ.get("QT_QPA_PLATFORM") or ("wayland" if os.environ.get("XDG_SESSION_TYPE") == "wayland" else "xcb")
  libs = _QT_LIBS | next((v for k, v in _PLATFORM_LIBS.items() if platform.startswith(k)), {})

  def absent(lib):
    try:
      ctypes.CDLL(lib)
    except OSError:
      return True

  missing = {lib: pkg for lib, pkg in libs.items() if absent(lib)}
  if missing:
    raise ImportError(f"the import wizard needs system libraries that pip does not install: {', '.join(missing)}\n"
                      f"on Debian and Ubuntu: sudo apt install {' '.join(missing.values())}")


def importWizard(path=None, **kwargs):
  """the import wizard: a window that browses the files of a folder, shows what a file holds,
  lets the phases, the conventions and the reference frames be chosen and writes the Python of
  the import; MATLAB's import_wizard

    ebsd = importWizard()                             # browse the current folder
    ebsd = importWizard('/data/ebsd')                 # a folder, or a file to open on start
    w = importWizard(show=False, exec=False)          # the window, without its event loop

  The window blocks until it is closed and returns the map imported, unless a Qt event loop
  runs already, as under IPython's %gui qt, where it returns the window. From a terminal,
  mtex-wizard [path] or python -m mtex.wizard [path] opens the same. It needs PySide6, the
  wizard extra (pip install mtex[wizard]), and on Linux the system libraries of Qt, which it
  names when they are missing. The keywords go to runWizard: show=False keeps the window
  hidden, exec=False returns it without running the event loop.

  Input
    path - a folder to browse or a file to open, by default the current folder
  Output
    ebsd - the map imported, None when nothing was; or the window
  See also
    loadEBSD, inspectEBSD, buildImportWizardScript"""
  _checkQtLibraries()
  try:
    from .window import runWizard
  except ImportError as e:
    raise ImportError("the import wizard needs PySide6 and matplotlib: pip install mtex[wizard]") from e
  return runWizard(path, **kwargs)


__all__ = ["importWizard", "WizardState"]
