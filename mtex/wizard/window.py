# the import wizard's window (docs/plans/wizard.md W2): a splitter with the browser side
# left and the analysis side right; the load on a worker thread, committed on a signal; the
# views drawn on one matplotlib canvas through the raster of the map
#
#   runWizard(path)            the window, returning the map when it is closed
#   grabWindow(window, png)    the window as a picture, which needs no display offscreen
from __future__ import annotations

import os
import sys
import traceback

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtGui import QColor, QFont, QIcon
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QMainWindow, QSizePolicy, QSplitter, QTabBar, QVBoxLayout,
                               QWidget)

from . import callbacks as cb
from .browser import BrowserSide, framed
from .desktop import APP_ID, installDesktopEntry
from .phases import PhaseTable
from .pictogram import CoordinatePanel
from .scriptview import ScriptView
from .state import WizardState
from .views import KeyPanel, MillerBar, ViewCanvas

FONT_SIZE = 10
BROWSER_WIDTH = 360   # the browser side on opening: the longest file name and the file table fit
FONT_SMALL = 9
TAB_COLORS = {"phase": "#1f4e79", "ipf": "#2e7d32", "property": "#333333", "polefigures": "#6a1b9a", "images": "#e65100"}


# ------------------------------------------------------------------ the worker

class LoadWorker(QObject):
  """a load on its own thread, the result or the exception handed back through a signal"""
  done = Signal(object)

  def __init__(self, load):
    super().__init__()
    self.load = load

  @Slot()
  def run(self):
    """runs the load and emits its result, or the exception it raised"""
    try:
      self.done.emit(self.load())
    except Exception as e:
      self.done.emit(e)


class LoadBridge(QObject):
  """the receiver on the main thread: a signal connected to a plain callable runs on the
  emitting thread, so the commit goes through a slot of an object living here"""

  def __init__(self, window, commit, thread, worker):
    super().__init__(window)
    self.window = window
    self.commit = commit
    self.thread = thread
    self.worker = worker

  @Slot(object)
  def finish(self, result):
    """stops the thread and commits the result on the main thread, a failure shown as the status;
    the job's closures are let go then, since they hold the map it was for"""
    self.thread.quit()
    try:
      self.commit(result)
    except Exception as e:
      traceback.print_exc()
      self.window.st.status = ("failed", f"{e}")
      self.window.st.changed("status")
    finally:
      self.commit = None
      self.worker.load = None

  @Slot()
  def forget(self):
    """the job let go by the window once its thread has finished"""
    self.window.jobs = [job for job in self.window.jobs if job[2] is not self]


class WizardWindow(QMainWindow):
  """the import wizard's window over a WizardState: the file browser on the left, the frames, the
  phase table, the script and the key on the right above the tabs of the views"""

  def __init__(self, st, app=None):
    super().__init__()
    self.st = st
    self.app = app or QApplication.instance()
    self.setWindowTitle("MTEX import wizard")
    self.setWindowIcon(wizardIcon())
    self.resize(1600, 900)
    self.small = QFont(self.font())
    self.small.setPointSize(FONT_SMALL)
    # the loads under way, (thread, worker, bridge), kept alive while their thread runs
    self.jobs = []
    split =QSplitter(Qt.Horizontal)
    self.setCentralWidget(split)
    # the browser side
    self.browser = BrowserSide(st, self.small, worker=self.worker)
    split.addWidget(self.browser)
    # the analysis side: the two frames, the phase table, the script of the import and the key of
    # one phase in one row, the views under them
    right = QWidget()
    rlay = QVBoxLayout(right)
    rlay.setContentsMargins(4, 4, 4, 4)
    rlay.setSpacing(4)
    top = QHBoxLayout()
    top.setSpacing(3)
    self.coordinates = CoordinatePanel(st)
    self.phases = PhaseTable(st, self.small)
    self.key = KeyPanel(st)
    self.script = ScriptView(st)
    top.addWidget(self.coordinates)
    top.addWidget(framed(self.phases), 0)
    top.addWidget(framed(self.script), 1)
    top.addWidget(framed(self.key), 0)
    rlay.addLayout(top)
    self.tabs = QTabBar()
    self.tabs.setExpanding(False)
    self.tabs.setDocumentMode(True)
    self.tabs.currentChanged.connect(self.onTab)
    rlay.addWidget(self.tabs)
    self.millers = MillerBar(st)
    rlay.addWidget(self.millers)
    self.canvas = ViewCanvas(st, worker=self.worker)
    self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    rlay.addWidget(self.canvas, 1)
    split.addWidget(right)
    split.setStretchFactor(0, 0)
    split.setStretchFactor(1, 1)
    split.setSizes([BROWSER_WIDTH, 1600 - BROWSER_WIDTH])
    st.on("tabs", self.refreshTabs)
    st.on("tab", self.showTab)
    self.refreshTabs()
    # the window opens with the keys on the list of the files, to walk it with the arrows
    self.browser.files.setFocus()

  # ------------------------------------------------------------------ the load

  def worker(self, load, commit):
    """the window's shape of a load: the reading on a thread, the commit on this one"""
    thread = QThread(self)
    w = LoadWorker(load)
    w.moveToThread(thread)
    bridge = LoadBridge(self, commit, thread, w)
    thread.started.connect(w.run)
    w.done.connect(bridge.finish, Qt.QueuedConnection)
    thread.finished.connect(bridge.forget, Qt.QueuedConnection)
    self.jobs.append((thread, w, bridge))
    thread.start()

  # ------------------------------------------------------------------ the strip

  def refreshTabs(self):
    """rebuilds the tab strip from the state's tabs, hidden while there is only one"""
    self.tabs.blockSignals(True)
    while self.tabs.count():
      self.tabs.removeTab(0)
    for name, kind in zip(self.st.tabs, self.st.tabKinds):
      k = self.tabs.addTab(name)
      self.tabs.setTabTextColor(k, QColor(TAB_COLORS.get(kind, "#333")))
    self.tabs.setVisible(self.tabs.count() > 1)
    self.tabs.setCurrentIndex(self.st.tab)
    self.tabs.blockSignals(False)
    self.showTab()

  def onTab(self, k):
    """hands a tab chosen in the strip to the state"""
    if k != self.st.tab:
      cb.pickTab(self.st, k)

  def showTab(self):
    """lets the strip follow the state's tab and redraws the view"""
    if self.tabs.currentIndex() != self.st.tab:
      self.tabs.blockSignals(True)
      self.tabs.setCurrentIndex(self.st.tab)
      self.tabs.blockSignals(False)
    self.canvas.scheduleDraw()

  def closeEvent(self, event):
    """a window closed with a load still running waits for it rather than tearing the thread
    down; the table is silenced first, since it announces every cell as it is destroyed"""
    self.phases.blockSignals(True)
    for thread, _, _ in list(self.jobs):
      thread.quit()
      thread.wait(10000)
    super().closeEvent(event)


# ------------------------------------------------------------------ running it

def wizardIcon():
  """the wizard's icon: a grain map in orientation colours with an import arrow"""
  return QIcon(os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.svg"))


def application():
  """the running QApplication, created on first call, with the wizard's font"""
  app = QApplication.instance()
  if app is None:
    # the app id names the desktop entry a Wayland task manager takes the icon from; an offscreen
    # run has no desktop to tell
    QApplication.setDesktopFileName(APP_ID)
    if "offscreen" not in os.environ.get("QT_QPA_PLATFORM", ""):
      installDesktopEntry()
    app = QApplication(sys.argv[:1])
    app.setWindowIcon(wizardIcon())
  app.setFont(QFont(app.font().family(), FONT_SIZE))
  return app


def runWizard(path=None, show=True, exec=None):
  """the window; blocks until it is closed unless an event loop already runs, and returns the map"""
  app = application()
  folder = os.path.abspath(path or os.getcwd())
  file = None
  if os.path.isfile(folder):
    file, folder = folder, os.path.dirname(folder)
  st = WizardState(folder)
  cb.setFolder(st, folder)
  w = WizardWindow(st, app)
  if show and exec is not False and "offscreen" not in os.environ.get("QT_QPA_PLATFORM", ""):
    w.showMaximized()
  elif show:
    w.show()
  if file is not None:
    cb.openEntry(st, file, w.worker)
  if exec is None:
    exec = not _loopRunning()
  if exec:
    app.exec()
    return st.ebsd
  return w


# whether a Qt event loop already runs, as under IPython's %gui qt
def _loopRunning():
  try:
    from IPython import get_ipython
    ip = get_ipython()
    return ip is not None and ip.active_eventloop in ("qt", "qt6")
  except Exception:
    return False


def grabWindow(window, png):
  """the whole window as a picture, at the size it has; wait for the load first"""
  window.grab().save(png)
  return png


__all__ = ["runWizard", "WizardWindow", "grabWindow", "application"]
