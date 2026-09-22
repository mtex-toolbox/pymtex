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
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QMainWindow, QSizePolicy, QSplitter, QTabBar, QVBoxLayout,
                               QWidget)

from . import callbacks as cb
from .browser import BrowserSide, framed
from .phases import PhaseTable
from .pictogram import CoordinatePanel
from .state import WizardState
from .views import MetaTree, MillerBar, ViewCanvas

FONT_SIZE = 10
BROWSER_WIDTH = 360   # the browser side on opening: the longest file name and the file table fit
FONT_SMALL = 9
TAB_COLORS = {"phase": "#1f4e79", "ipf": "#2e7d32", "property": "#333333", "polefigures": "#6a1b9a", "images": "#e65100", "metadata": "#333333"}


# ------------------------------------------------------------------ the worker

# a load on its own thread, the result or the exception handed back through a signal
class LoadWorker(QObject):
  done = Signal(object)

  def __init__(self, load):
    super().__init__()
    self.load = load

  @Slot()
  def run(self):
    try:
      self.done.emit(self.load())
    except Exception as e:
      self.done.emit(e)


# the receiver on the main thread: a signal connected to a plain callable runs on the
# emitting thread, so the commit goes through a slot of an object living here
class LoadBridge(QObject):

  def __init__(self, window, commit, thread):
    super().__init__(window)
    self.window = window
    self.commit = commit
    self.thread = thread

  @Slot(object)
  def finish(self, result):
    self.thread.quit()
    try:
      self.commit(result)
    except Exception as e:
      traceback.print_exc()
      self.window.st.status = ("failed", f"{e}")
      self.window.st.changed("status")


class WizardWindow(QMainWindow):

  def __init__(self, st, app=None):
    super().__init__()
    self.st = st
    self.app = app or QApplication.instance()
    self.setWindowTitle("MTEX import wizard")
    self.resize(1600, 900)
    self.small = QFont(self.font())
    self.small.setPointSize(FONT_SMALL)
    self.threads = []
    split = QSplitter(Qt.Horizontal)
    self.setCentralWidget(split)
    # the browser side
    self.browser = BrowserSide(st, self.small, worker=self.worker, clipboard=self.app.clipboard() if self.app else None)
    split.addWidget(self.browser)
    # the analysis side: the two frames, the phase table and the metadata in one row, the views under them
    right = QWidget()
    rlay = QVBoxLayout(right)
    rlay.setContentsMargins(4, 4, 4, 4)
    rlay.setSpacing(4)
    top = QHBoxLayout()
    top.setSpacing(3)
    self.coordinates = CoordinatePanel(st)
    self.phases = PhaseTable(st, self.small)
    self.meta = MetaTree(st, self.small)
    top.addWidget(self.coordinates)
    top.addWidget(framed(self.phases), 0)
    top.addWidget(framed(self.meta), 1)
    rlay.addLayout(top)
    self.tabs = QTabBar()
    self.tabs.setExpanding(False)
    self.tabs.setDocumentMode(True)
    self.tabs.currentChanged.connect(self.onTab)
    rlay.addWidget(self.tabs)
    self.millers = MillerBar(st)
    rlay.addWidget(self.millers)
    self.canvas = ViewCanvas(st)
    self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    rlay.addWidget(self.canvas, 1)
    split.addWidget(right)
    split.setStretchFactor(0, 0)
    split.setStretchFactor(1, 1)
    split.setSizes([BROWSER_WIDTH, 1600 - BROWSER_WIDTH])
    st.on("tabs", self.refreshTabs)
    st.on("tab", self.showTab)
    self.refreshTabs()

  # ------------------------------------------------------------------ the load

  # the window's shape of a load: the reading on a thread, the commit on this one
  def worker(self, load, commit):
    thread = QThread(self)
    w = LoadWorker(load)
    w.moveToThread(thread)
    bridge = LoadBridge(self, commit, thread)
    thread.started.connect(w.run)
    w.done.connect(bridge.finish, Qt.QueuedConnection)
    thread.finished.connect(lambda: self.threads.remove(thread) if thread in self.threads else None)
    self.threads.append(thread)
    self._keep = getattr(self, "_keep", []) + [w, bridge]
    thread.start()

  # ------------------------------------------------------------------ the strip

  def refreshTabs(self):
    self.tabs.blockSignals(True)
    while self.tabs.count():
      self.tabs.removeTab(0)
    for name, kind in zip(self.st.tabs, self.st.tabKinds):
      k = self.tabs.addTab(name)
      from PySide6.QtGui import QColor
      self.tabs.setTabTextColor(k, QColor(TAB_COLORS.get(kind, "#333")))
    self.tabs.setVisible(self.tabs.count() > 1)
    self.tabs.setCurrentIndex(self.st.tab)
    self.tabs.blockSignals(False)
    self.showTab()

  def onTab(self, k):
    if k != self.st.tab:
      cb.pickTab(self.st, k)

  def showTab(self):
    if self.tabs.currentIndex() != self.st.tab:
      self.tabs.blockSignals(True)
      self.tabs.setCurrentIndex(self.st.tab)
      self.tabs.blockSignals(False)
    self.canvas.scheduleDraw()

  # a window closed with a load still running waits for it rather than tearing the thread
  # down; the table is silenced first, since it announces every cell as it is destroyed
  def closeEvent(self, event):
    self.phases.blockSignals(True)
    for t in list(self.threads):
      t.quit()
      t.wait(10000)
    super().closeEvent(event)


# ------------------------------------------------------------------ running it

def application():
  app = QApplication.instance()
  if app is None:
    app = QApplication(sys.argv[:1])
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
  if show:
    w.showMaximized() if exec is not False and "offscreen" not in os.environ.get("QT_QPA_PLATFORM", "") else w.show()
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


# the whole window as a picture, at the size it has; wait for the load first
def grabWindow(window, png):
  window.grab().save(png)
  return png


__all__ = ["runWizard", "WizardWindow", "grabWindow", "application"]
