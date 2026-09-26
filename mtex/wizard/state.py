# everything the import wizard shows, as one state the window binds to (Julia state.jl;
# docs/plans/wizard.md section 2)
#
# The GUI is a view of the load's keyword arguments: the state is those arguments, the map
# they produced and what the views derive from it. A callback changes fields and announces
# them through changed(); the window connects hooks by field name and the tests connect
# nothing, so the whole wizard is driven with no window open.
from __future__ import annotations

import os
from dataclasses import dataclass

from ..geometry.plottingconvention import AXIS_CONVENTIONS


# one row of the browser: a folder or a map, at its depth below the folder being looked at
@dataclass
class FileRow:
  name: str
  path: str
  depth: int = 0
  isDir: bool = False
  open: bool = False
  size: int = 0


class WizardState:

  def __init__(self, folder=None):
    # the browser
    self.folder = os.path.abspath(folder or os.getcwd())
    self.entries = []
    self.selected = None
    self.info = None
    self.status = ("idle", "click a map to import it, a folder to look inside")
    self.dataSet = 0
    # the map, the phases the file declared, and which phases are drawn
    self.ebsd = None
    self.headerPhases = None
    self.drawn = {}
    self.loadSeconds = None
    self.imports = 0           # the imports started, so that only the latest is committed
    # the two reference frames as entries of AXIS_CONVENTIONS, and the axis names of each
    self.mapConvention = 0
    self.eulerConvention = 0
    self.mapAxesNames = ("x", "y", "z")
    self.eulerAxesNames = ("x", "y", "z")
    # the views
    self.tabs = []
    self.tabKinds = []
    self.tab = 0
    self.millers = ["100", "010", "001"]
    self.densities = {}
    # the phase whose inverse pole figure key the key panel shows
    self.keyPhase = None
    self.metaRows = []
    self.metaOpen = set()
    self.images = []
    self.image = 0
    self.imageCache = {}
    # the outputs
    self.variable = "ebsd"
    self._hooks = {}

  # ------------------------------------------------------------------ the hooks

  # the window's binding of a field: fn() runs after every change announced for it
  def on(self, name, fn):
    self._hooks.setdefault(name, []).append(fn)

  def changed(self, *names):
    for n in names:
      for fn in self._hooks.get(n, ()):
        fn()

  # ------------------------------------------------------------------ what the state says

  @property
  def conventions(self):
    return AXIS_CONVENTIONS

  @property
  def currentTab(self):
    return self.tabs[self.tab] if 0 <= self.tab < len(self.tabs) else None

  @property
  def hasMap(self):
    return self.ebsd is not None

  def __repr__(self):
    lines = [f"importWizard state in {self.folder}", f"  status: {self.status[0]} - {self.status[1]}"]
    if self.info is not None:
      lines.append(f"  file: {self.info.name}")
    if self.ebsd is not None:
      lines.append(f"  map: {self.ebsd.size} points, tab {self.currentTab!r}")
      lines.append(f"  map coordinates {AXIS_CONVENTIONS[self.mapConvention]}, Euler coordinates {AXIS_CONVENTIONS[self.eulerConvention]}")
    return "\n".join(lines)


__all__ = ["WizardState", "FileRow"]
