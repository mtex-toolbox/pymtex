import os
import time

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import matplotlib

matplotlib.use("QtAgg")

from mtex.wizard import callbacks as cb
from mtex.wizard.window import application, grabWindow, runWizard

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "ebsd")


def pump(app, until, timeout=60):
  t = time.perf_counter()
  while not until() and time.perf_counter() - t < timeout:
    app.processEvents()
    time.sleep(0.002)
  app.processEvents()


# the window over the fixtures: a load through the worker thread, every tab drawn, a grab
def test_window_offscreen(tmp_path):
  app = application()
  w = runWizard(FIXTURES, exec=False)
  st = w.st
  assert w.isVisible() and st.folder == os.path.abspath(FIXTURES) and not st.hasMap
  cb.openEntry(st, os.path.join(FIXTURES, "Forsterite_cut.ctf"), w.worker)
  pump(app, lambda: st.status[0] in ("loaded", "failed"))
  assert st.status[0] == "loaded", st.status
  assert w.tabs.count() == len(st.tabs) and w.tabs.currentIndex() == st.tab and st.currentTab == "IPF Z"
  assert w.phases.rowCount() == 5 and len(w.browser.info.fileRows()) >= 3
  assert w.browser.status.text().startswith("Forsterite_cut.ctf")
  for k in range(len(st.tabs)):
    cb.pickTab(st, k)
    app.processEvents()
    w.canvas.draw()
    assert st.status[0] == "loaded", (st.tabs[k], st.status)
  assert w.millers.isVisible() is False or st.tabKinds[st.tab] == "polefigures"
  png = grabWindow(w, str(tmp_path / "wizard.png"))
  assert os.path.getsize(png) > 10000
  # a load that fails is reported in the band and leaves the window usable
  cb.openEntry(st, os.path.join(FIXTURES, "generic_cut.txt"), w.worker)
  pump(app, lambda: st.status[0] == "failed")
  assert st.status[0] == "failed" and st.hasMap
  w.close()
  app.processEvents()


# the file given to the wizard is opened on start
def test_open_file_on_start():
  app = application()
  w = runWizard(os.path.join(FIXTURES, "olivine_cut.ang"), exec=False)
  pump(app, lambda: w.st.status[0] in ("loaded", "failed"))
  assert w.st.status[0] == "loaded" and w.st.folder == os.path.abspath(FIXTURES)
  w.close()
  app.processEvents()
