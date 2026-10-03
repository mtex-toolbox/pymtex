"""the import wizard: a window over the loaders (docs/plans/wizard.md)

  importWizard()              browse the current folder
  importWizard('/data/ebsd')  a folder, or a file to open on start
  mtex-wizard [path]          the same from a terminal, or python -m mtex.wizard [path]

The core - the state, the callbacks, the script - needs no window and is what the tests
drive; the window needs PySide6, the `wizard` extra, and is imported only when asked for."""
from .state import WizardState


def importWizard(path=None, **kwargs):
  """the import wizard's window; returns the map it imported when the window is closed"""
  try:
    from .window import runWizard
  except ImportError as e:
    raise ImportError("the import wizard needs PySide6 and matplotlib: pip install mtex[wizard]") from e
  return runWizard(path, **kwargs)


__all__ = ["importWizard", "WizardState"]
