# the import wizard opened offscreen, driven and grabbed as a picture (docs/plans/wizard.md
# W2 and W5): the Forsterite map read through the worker thread, every tab drawn, the
# window written to bench/out/wizard-<tab>.png, and the layout measured. Usage:
#   .venv/bin/python bench/wizard.py            offscreen, the pictures and the numbers
#   .venv/bin/python bench/wizard.py --show     on the screen
import os
import sys
import time

if "--show" not in sys.argv:
  os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib
matplotlib.use("QtAgg")

from PySide6.QtGui import QCursor

from mtex.wizard import callbacks as cb
from mtex.wizard.window import application, grabWindow, runWizard

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataDir = os.path.join(root, "..", "frameSymmetry", "data", "EBSD")
outDir = os.path.join(root, "bench", "out")


def pump(app, until, timeout=120):
  t = time.perf_counter()
  while not until() and time.perf_counter() - t < timeout:
    app.processEvents()
    time.sleep(0.002)
  return time.perf_counter() - t


if __name__ == "__main__":
  app = application()
  if "--show" in sys.argv:
    runWizard(dataDir)
    sys.exit(0)
  os.makedirs(outDir, exist_ok=True)
  # the offscreen pointer rests in the window's corner, on the folder button, which it would raise
  QCursor.setPos(-100, -100)
  w = runWizard(dataDir, exec=False)
  st = w.st
  app.processEvents()
  print("window", w.size().width(), "x", w.size().height(), "browser", w.browser.width(), "table", w.phases.width(), "key", w.key.width())
  cb.openEntry(st, os.path.join(dataDir, "Forsterite.ctf"), w.worker)
  seconds = pump(app, lambda: st.status[0] in ("loaded", "failed"))
  print(st.status, f"window live for {seconds:.2f} s")
  for name in st.tabs:
    t = [time.perf_counter()]
    cb.pickTab(st, st.tabs.index(name))
    t.append(time.perf_counter())
    app.processEvents()
    t.append(time.perf_counter())
    w.canvas.draw()
    t.append(time.perf_counter())
    png = os.path.join(outDir, f"wizard-{name.replace(' ', '_')}.png")
    grabWindow(w, png)
    t.append(time.perf_counter())
    steps = " ".join(f"{1e3 * (b - a):5.0f}" for a, b in zip(t, t[1:]))
    print(f"{name:14s} {1e3 * (t[-1] - t[0]):6.0f} ms  (tab, events, draw, grab: {steps})  {os.path.basename(png)}")
  cb.pickTab(st, st.tabs.index("IPF Z"))
  app.processEvents()
  w.canvas.draw()
  grabWindow(w, os.path.join(outDir, "wizard.png"))
  # the two frames, and the eight the chooser offers as pictures: a popup is its own window
  # and is not in the grab of this one
  from mtex.wizard.pictogram import ConventionChooser
  w.coordinates.grab().save(os.path.join(outDir, "wizard-frames.png"))
  chooser = ConventionChooser(w.coordinates.euler, st.eulerConvention, w.coordinates.euler.names, w.coordinates.euler.ink)
  body = chooser.actions()[0].defaultWidget()
  body.resize(body.sizeHint())
  body.show()
  app.processEvents()
  body.grab().save(os.path.join(outDir, "wizard-conventions.png"))
  print("frames", w.coordinates.size().toTuple(), "chooser", body.size().toTuple())
  # the widths the columns give their content
  widths = [w.phases.columnWidth(k) for k in range(w.phases.columnCount())]
  print("phase columns", widths, "rows", w.phases.rowCount(), "row height", w.phases.rowHeight(0) if w.phases.rowCount() else None)
  fonts = sorted({w.font().pointSize(), w.small.pointSize()})
  print("point sizes", fonts)
  w.close()
