# W0 of docs/plans/wizard.md: the five item shell on PySide6 - a splitter, a tree over the
# file system, an editable table cell, a tab strip and a map on a matplotlib canvas
# recoloured on a click - with the load on a worker thread. Measures a recolour through the
# raster and whether the window stays live during the load, and grabs the window offscreen
# into bench/out/wizard_spike.png. Usage:
#   .venv/bin/python bench/wizard_spike.py            offscreen, the picture and the numbers
#   .venv/bin/python bench/wizard_spike.py --show     on the screen
import os
import sys
import time

if "--show" not in sys.argv:
  os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib
import matplotlib.figure
matplotlib.use("QtAgg")
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from PySide6.QtCore import QDir, QObject, QThread, Qt, Signal
from PySide6.QtWidgets import (QApplication, QFileSystemModel, QLabel, QMainWindow, QSplitter, QTabBar, QTableWidget,
                               QTableWidgetItem, QTreeView, QVBoxLayout, QWidget)

from mtex import loadEBSD, mtexFigure, plot
from mtex.plotting.mapplot import mapValues
from mtex.plotting.maps import mapRaster

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataDir = os.path.join(root, "..", "frameSymmetry", "data", "EBSD")
outDir = os.path.join(root, "bench", "out")


# the load on a worker thread, the map handed back through a signal
class Loader(QObject):
  done = Signal(object, float)

  def __init__(self, path):
    super().__init__()
    self.path = path

  def run(self):
    t = time.perf_counter()
    ebsd = loadEBSD(self.path)
    self.done.emit(ebsd, time.perf_counter() - t)


class Spike(QMainWindow):
  def __init__(self):
    super().__init__()
    self.setWindowTitle("wizard spike")
    self.resize(1600, 900)
    split = QSplitter(Qt.Horizontal)
    self.setCentralWidget(split)
    # left: the tree over the data folder and an editable table
    left = QWidget()
    lay = QVBoxLayout(left)
    self.model = QFileSystemModel()
    self.model.setNameFilters(["*.ang", "*.ctf", "*.cpr", "*.osc", "*.h5", "*.h5oina", "*.oh5"])
    self.model.setNameFilterDisables(False)
    self.model.setRootPath(dataDir)
    self.tree = QTreeView()
    self.tree.setModel(self.model)
    self.tree.setRootIndex(self.model.index(dataDir))
    for k in (1, 2, 3):
      self.tree.hideColumn(k)
    self.tree.doubleClicked.connect(self.onOpen)
    lay.addWidget(self.tree)
    self.table = QTableWidget(2, 4)
    self.table.setHorizontalHeaderLabels(["mineral", "a", "b", "c"])
    for i, row in enumerate([("Forsterite", "4.756", "10.207", "5.98"), ("Enstatite", "18.22", "8.81", "5.18")]):
      for j, v in enumerate(row):
        self.table.setItem(i, j, QTableWidgetItem(v))
    self.table.cellChanged.connect(lambda i, j: self.status.setText(f"edited {i},{j}: {self.table.item(i, j).text()}"))
    lay.addWidget(self.table)
    self.status = QLabel("double click a map")
    lay.addWidget(self.status)
    split.addWidget(left)
    # right: the tab strip over one canvas
    right = QWidget()
    rlay = QVBoxLayout(right)
    self.tabs = QTabBar()
    for name in ("phase", "IPF X", "IPF Y", "IPF Z", "bc", "mad"):
      self.tabs.addTab(name)
    self.tabs.currentChanged.connect(self.onTab)
    rlay.addWidget(self.tabs)
    self.canvas = FigureCanvasQTAgg(matplotlib.figure.Figure(layout="constrained"))
    rlay.addWidget(NavigationToolbar2QT(self.canvas, right))
    rlay.addWidget(self.canvas)
    split.addWidget(right)
    split.setSizes([450, 1150])
    self.ebsd, self.image, self.index, self.ticks = None, None, None, 0

  def onOpen(self, idx):
    path = self.model.filePath(idx)
    if os.path.isdir(path):
      return
    self.status.setText(f"loading {os.path.basename(path)} ...")
    self.thread = QThread()
    self.loader = Loader(path)
    self.loader.moveToThread(self.thread)
    self.thread.started.connect(self.loader.run)
    self.loader.done.connect(self.onLoaded)
    self.loader.done.connect(self.thread.quit)
    self.thread.start()

  # the map drawn once through the package's own plot into the canvas's figure
  def onLoaded(self, ebsd, seconds):
    self.ebsd = ebsd
    self.canvas.figure.clear()
    mtexFigure(1, figure=self.canvas.figure)
    t = time.perf_counter()
    self.image = plot(ebsd, "phase")
    r = mapRaster(ebsd)
    self.index = None if r is None else r[2]
    self.canvas.draw()
    self.status.setText(f"loaded {ebsd.size} points in {seconds:.2f} s, drawn in {time.perf_counter() - t:.2f} s")

  # a change of tab rewrites the picture's colours and nothing else
  def onTab(self, k):
    if self.ebsd is None or self.index is None:
      return
    name = self.tabs.tabText(k)
    t = time.perf_counter()
    color = {"phase": "phase", "IPF X": "orientation", "IPF Y": "orientation", "IPF Z": "orientation"}.get(name, name)
    direction = {"IPF X": "x", "IPF Y": "y", "IPF Z": "z"}.get(name)
    from mtex import xvector, yvector, zvector
    d = {"x": xvector, "y": yvector, "z": zvector, None: None}[direction]
    values, scalar = mapValues(self.ebsd, color, ipfDirection=d)
    t1 = time.perf_counter()
    covered = self.index >= 0
    if scalar:
      img = np.full(self.index.shape, np.nan)
      img[covered] = values[self.index[covered]]
      lo, hi = np.nanmin(img), np.nanmax(img)
      rgb = matplotlib.colormaps["viridis"]((img - lo) / max(hi - lo, 1e-12))
      rgb[~covered, 3] = 0
    else:
      rgb = np.zeros(self.index.shape + (4,))
      rgb[covered, :3] = values[self.index[covered]]
      rgb[covered, 3] = 1
    self.image.set_data(rgb)
    self.canvas.draw()
    self.status.setText(f"{name}: colours {1e3 * (t1 - t):.0f} ms, picture {1e3 * (time.perf_counter() - t1):.0f} ms")


if __name__ == "__main__":
  app = QApplication(sys.argv)
  w = Spike()
  w.show()
  if "--show" in sys.argv:
    sys.exit(app.exec())
  # drive it: open the Forsterite map, count the event loop ticks while it loads
  os.makedirs(outDir, exist_ok=True)
  path = os.path.join(dataDir, "Forsterite.ctf")
  w.onOpen(w.model.index(path))
  ticks, t0 = 0, time.perf_counter()
  while w.ebsd is None:
    app.processEvents()
    ticks += 1
    time.sleep(0.001)
  print(f"{w.status.text()}; {ticks} event loop ticks while loading, {time.perf_counter() - t0:.2f} s wall")
  print(f"raster {w.index.shape} of {w.ebsd.size} points")
  for k in range(w.tabs.count()):
    w.tabs.setCurrentIndex(k)
    app.processEvents()
    print(w.status.text())
  # the second pass is what a recolour costs once the keys are known
  for k in (0, 3, 4):
    w.tabs.setCurrentIndex(k)
    app.processEvents()
    print("again", w.status.text())
  w.table.item(0, 1).setText("4.75")
  app.processEvents()
  print(w.status.text())
  png = os.path.join(outDir, "wizard_spike.png")
  w.grab().save(png)
  print("grabbed", png, w.size().width(), "x", w.size().height())
