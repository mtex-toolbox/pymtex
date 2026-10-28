import os
import time

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from mtex.wizard import callbacks as cb
from mtex.wizard.window import application, grabWindow, runWizard

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "ebsd")


def pump(app, until, timeout=60):
  t = time.perf_counter()
  while not until() and time.perf_counter() - t < timeout:
    app.processEvents()
    time.sleep(0.002)
  app.processEvents()


# a window closed while a load runs waits for it, however long, rather than tearing its thread
# down, and builds no colour grid it had not begun
def test_a_window_closed_during_a_load_waits_for_it():
  app = application()
  w = runWizard(FIXTURES, exec=False)
  finished = []
  w.worker(lambda: (time.sleep(0.5), finished.append(True)), lambda result: None)
  w.close()
  assert finished == [True] and all(thread.isFinished() for thread, _, _ in w.jobs) and cb.COLOURS_STOP.is_set()
  app.processEvents()


# the window over the fixtures: a load through the worker thread shown in every panel, a grab
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
  assert w.millers.isHidden()
  cb.pickTab(st, st.tabs.index("pole figures"))
  app.processEvents()
  assert w.tabs.currentIndex() == st.tab and not w.millers.isHidden()
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


# the window opens with the keyboard focus on the list of the files, not on the folder button
def test_the_file_list_has_the_focus_on_start():
  application()
  w = runWizard(FIXTURES, exec=False)
  assert w.focusWidget() is w.browser.files


def test_the_window_has_the_wizard_icon():
  application()
  w = runWizard(FIXTURES, show=False, exec=False)
  icon = w.windowIcon()
  assert not icon.isNull()
  assert not icon.pixmap(32, 32).toImage().isNull()
  w.close()


# the arrow keys read the file the cursor stops on, and the map shown is that file's
def test_walking_with_the_arrow_keys_reads_the_file():
  from PySide6.QtCore import Qt
  from PySide6.QtTest import QTest
  app = application()
  w = runWizard(FIXTURES, exec=False)
  files = w.browser.files
  names = [r.path for r in w.st.entries]
  maps = [i for i, r in enumerate(w.st.entries) if not r.isDir and r.path.endswith((".ang", ".ctf"))][:2]
  files.goTo(maps[0])
  pump(app, lambda: w.st.status[0] in ("loaded", "failed") and w.st.selected == names[maps[0]])
  files.goTo(maps[1] - 1)
  QTest.keyClick(files, Qt.Key_Down)
  pump(app, lambda: w.st.status[0] in ("loaded", "failed") and w.st.selected == names[maps[1]])
  assert w.st.status[0] == "loaded"
  assert os.path.basename(names[maps[1]]) in w.st.status[1]
  lay = w.browser.layout()
  order = [lay.itemAt(k).widget() for k in range(lay.count())]
  assert order.index(w.browser.status) == order.index(files.parentWidget()) + 1
  w.close()
  app.processEvents()


# a map read before is let go when the next one is committed: the finished jobs keep no closure
def test_the_maps_read_before_are_let_go():
  import gc
  import weakref
  app = application()
  w = runWizard(FIXTURES, exec=False)
  read = []
  for name in ("Forsterite_cut.ctf", "ferrite_cut.ang", "olivine_cut.ang"):
    cb.openEntry(w.st, os.path.join(FIXTURES, name), w.worker)
    pump(app, lambda: w.st.status[0] in ("loaded", "failed") and os.path.basename(w.st.selected) == name)
    read.append(weakref.ref(w.st.ebsd))
  pump(app, lambda: not w.jobs, timeout=30)
  gc.collect()
  assert read[0]() is None and read[1]() is None and read[2]() is w.st.ebsd
  w.close()
  app.processEvents()


# a closed window is destroyed by the deferred deletes of the GUI thread, never left to the
# garbage collector, which would free it on whichever thread allocates
def test_a_closed_window_is_destroyed_on_the_gui_thread():
  import shiboken6
  from PySide6.QtCore import QCoreApplication, QEvent
  application()
  w = runWizard(FIXTURES, exec=False)
  w.close()
  QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
  assert not shiboken6.isValid(w)


# the bars under the canvas give way on the pole figure tab: the size the pole figures are drawn
# ahead at on any tab is the size the tab shows them at, so the picture drawn ahead is the one shown
@pytest.mark.parametrize("kind", ["map", "volume"])
def test_the_pole_figures_are_drawn_ahead_at_the_size_shown(volumeFile, kind):
  app = application()
  w = runWizard(FIXTURES, exec=False)
  st = w.st
  cb.openEntry(st, volumeFile() if kind == "volume" else os.path.join(FIXTURES, "Forsterite_cut.ctf"), w.worker)
  pump(app, lambda: st.status[0] in ("loaded", "failed"))
  assert st.status[0] == "loaded" and st.isVolume == (kind == "volume") and w.section.isHidden() != st.isVolume
  ahead = w.canvas.poleFigureSize()
  cb.pickTab(st, st.tabs.index("pole figures"))
  pump(app, lambda: False, timeout=0.3)
  assert w.section.isHidden() and w.canvas.poleFigureSize() == ahead
  w.close()
  app.processEvents()
