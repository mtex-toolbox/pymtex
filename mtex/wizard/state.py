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
from ..geometry.display import aligned


@dataclass
class FileRow:
  """one row of the browser: a folder or a map, at its depth below the folder being looked at"""
  name: str
  path: str
  depth: int = 0
  isDir: bool = False
  open: bool = False
  size: int = 0


class WizardState:
  """everything the import wizard shows: the browser, the map and its load arguments, the frames
  and the views; callbacks change it and announce the fields through changed()"""

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
    # the map the views draw: the map itself, or of a volume the layer sectionIndex across the
    # axis sectionAxis (0, 1, 2 for x, y, z)
    self.section = None
    self.sectionAxis = 2
    self.sectionIndex = 0
    self.imports = 0           # the imports started, so that only the latest is committed
    self._phaseCounts = None   # (the phase ids counted, their counts), kept by phaseCounts
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

  def on(self, name, fn):
    """the window's binding of a field: fn() runs after every change announced for it"""
    self._hooks.setdefault(name, []).append(fn)

  def changed(self, *names):
    """the hooks of the named fields run, the announcement of a change"""
    for n in names:
      for fn in self._hooks.get(n, ()):
        fn()

  # ------------------------------------------------------------------ what the state says

  @property
  def conventions(self):
    """the axis conventions the map and the Euler frames are chosen from"""
    return AXIS_CONVENTIONS

  @property
  def currentTab(self):
    """the view of the tab shown, None when there is none"""
    return self.tabs[self.tab] if 0 <= self.tab < len(self.tabs) else None

  @property
  def currentKind(self):
    """the kind of the view of the tab shown, None when there is none"""
    return self.tabKinds[self.tab] if 0 <= self.tab < len(self.tabKinds) else None

  @property
  def hasMap(self):
    """whether a map has been imported"""
    return self.ebsd is not None

  @property
  def isVolume(self):
    """whether the map imported is a volume, shown one section at a time"""
    return self.ebsd is not None and self.ebsd.ndim == 3

  def __repr__(self):
    lines = [f"importWizard state in {self.folder}", f"  status: {self.status[0]} - {self.status[1]}"]
    if self.info is not None:
      lines.append(f"  file: {self.info.name}")
    if self.ebsd is not None:
      lines.append(f"  map: {self.ebsd.size} points, tab {self.currentTab!r}")
      if self.isVolume:
        lines.append(f"  section: layer {self.sectionIndex} of {self.ebsd.shape[self.sectionAxis]} across {'xyz'[self.sectionAxis]}")
      lines.append(f"  map coordinates {AXIS_CONVENTIONS[self.mapConvention]}, Euler coordinates {AXIS_CONVENTIONS[self.eulerConvention]}")
    return aligned("\n".join(lines))


__all__ = ["WizardState", "FileRow"]
