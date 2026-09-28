import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QFont, QKeyEvent

from mtex.wizard import callbacks as cb
from mtex.wizard.browser import CRUMBS, CRUMB_WIDTH, BrowserSide, STATUS_COLORS
from mtex.wizard.state import WizardState
from mtex.wizard.window import application

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "ebsd")


def test_browser_side():
  application()
  st = WizardState(FIXTURES)
  cb.setFolder(st, FIXTURES)
  side = BrowserSide(st, QFont())
  files = side.files
  assert files.model_.rowCount() == len(st.entries) and st.entries[0].isDir
  # the crumbs end in the folder, the band starts idle, the data sets are hidden for a file of one
  assert side.breadcrumb.lay.count() > 2
  assert "idle" in STATUS_COLORS and STATUS_COLORS["idle"] in side.status.styleSheet()
  # a click on the folder row unfolds it; a click on a map reads it
  files.onClick(files.model_.index(0, 0))
  assert st.entries[0].open and st.entries[1].depth == 1
  k = [r.name for r in st.entries].index("Forsterite_cut.ctf")
  files.onClick(files.model_.index(k, 0))
  assert st.status[0] == "loaded" and STATUS_COLORS["loaded"] in side.status.styleSheet()
  # the file's own rows the window shows nowhere else, at the top, the metadata of the map after them
  rows = dict(side.info.fileRows())
  assert rows["vendor"] == "Oxford" and "grid" in rows and "extent" in rows
  assert not {"file", "format", "size", "points", "phases", "properties", "data sets"} & set(rows)
  tops = [side.info.topLevelItem(k).text(0) for k in range(side.info.topLevelItemCount())]
  assert tops[:len(rows)] == list(rows) and tops[len(rows)] == "header"
  assert not side.dataSetPanel.isVisibleTo(side)
  # the cursor moved to a file reads it once it stops there, not at once
  k = [r.name for r in st.entries].index("olivine_cut.ang")
  files.setCurrentIndex(files.model_.index(k, 0))
  assert files.walkTimer.isActive() and st.info.name == "Forsterite_cut.ctf"
  files.walkTimer.stop()
  files.readCurrent()
  assert st.info.name == "olivine_cut.ang" and st.status[0] == "loaded" and st.selected == st.entries[k].path


def test_data_set_list():
  pytest.importorskip("h5py")
  application()
  st = WizardState(FIXTURES)
  cb.setFolder(st, FIXTURES)
  side = BrowserSide(st, QFont())
  cb.openEntry(st, os.path.join(FIXTURES, "h5", "emsphinx_cut.h5"))
  assert side.dataSets.count() == 4 and side.dataSetPanel.isVisibleTo(side) and side.dataSets.currentRow() == 0
  side.dataSets.itemClicked.emit(side.dataSets.item(2))
  assert st.dataSet == 2 and st.ebsd.meta["dataSet"] == "/Scan 3/EBSD"


# the path is the end of the path: the panel's width is the browser's, not the folder's depth
def test_breadcrumb_is_compact(tmp_path):
  application()
  deep = tmp_path
  for name in ("one", "two", "three", "four", "five"):
    deep = deep / name
  deep.mkdir(parents=True)
  st = WizardState(str(deep))
  cb.setFolder(st, str(deep))
  side = BrowserSide(st, QFont())
  lay = side.breadcrumb.lay
  texts = [lay.itemAt(i).widget().text() for i in range(lay.count() - 1)]
  assert texts[0] == "…" and texts[-1] == "five"
  assert [t for t in texts if t != "/"] == ["…", "three", "four", "five"]
  # the folders left out are in the ellipsis menu, from the root down, and still reach their folder
  menu = lay.itemAt(0).widget().menu()
  assert menu.actions()[0].text() == os.sep
  # neither depth nor a long name widens the row beyond the crumbs it draws
  bound = CRUMBS * CRUMB_WIDTH + 80
  assert side.breadcrumb.minimumSizeHint().width() < bound
  deeper = deep / ("verylongfoldername" * 3)
  deeper.mkdir()
  cb.setFolder(st, str(deeper))
  texts = [lay.itemAt(i).widget().text() for i in range(lay.count() - 1)]
  assert texts[-1].endswith("…") or "…" in texts[-1]
  assert side.breadcrumb.minimumSizeHint().width() < bound
  assert side.minimumSizeHint().width() < 300


# Return walks into a folder, as a double click does, and reads a map
def test_return_opens(tmp_path):
  application()
  st = WizardState(FIXTURES)
  cb.setFolder(st, FIXTURES)
  side = BrowserSide(st, QFont())
  files = side.files
  files.setCurrentIndex(files.model_.index(0, 0))
  assert st.entries[0].isDir
  inside = st.entries[0].path
  files.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Return, Qt.NoModifier))
  assert st.folder == inside and not st.entries[0].isDir
  # and a double click on a folder does the same from the folder above
  cb.setFolder(st, FIXTURES)
  files.onDoubleClick(files.model_.index(0, 0))
  assert st.folder == inside
  # Return on a map reads it
  cb.setFolder(st, FIXTURES)
  k = [r.name for r in st.entries].index("Forsterite_cut.ctf")
  files.setCurrentIndex(files.model_.index(k, 0))
  files.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Return, Qt.NoModifier))
  assert st.status[0] == "loaded"


# a double click walks into a folder, though its first click has unfolded it and so reset the
# list; a single click still only unfolds, and an open folder folded by the first click is walked into too
def test_double_click_walks_into_a_folder():
  from PySide6.QtTest import QTest
  app = application()
  st = WizardState(FIXTURES)
  cb.setFolder(st, FIXTURES)
  side = BrowserSide(st, QFont())
  files = side.files
  files.resize(300, 600)
  files.show()
  app.processEvents()
  k = next(i for i, r in enumerate(st.entries) if r.isDir)
  name = st.entries[k].name
  center = lambda: files.visualRect(files.model_.index(k, 0)).center()
  QTest.mouseClick(files.viewport(), Qt.LeftButton, Qt.NoModifier, center())
  assert st.entries[k].open and st.folder == os.path.abspath(FIXTURES)
  QTest.mouseDClick(files.viewport(), Qt.LeftButton, Qt.NoModifier, center())
  app.processEvents()
  assert st.folder == os.path.join(os.path.abspath(FIXTURES), name)
  # the release that ends it acts on nothing of the new folder
  QTest.mouseRelease(files.viewport(), Qt.LeftButton, Qt.NoModifier, center())
  assert not files.swallowRelease and not any(r.open for r in st.entries)


# the keys: Right unfolds a folder and steps into it, Left steps out and folds it, a letter goes
# to the first entry starting with it and again to the next, Backspace walks up onto the folder
def test_keyboard_navigation():
  from PySide6.QtTest import QTest
  app = application()
  st = WizardState(FIXTURES)
  cb.setFolder(st, os.path.join(FIXTURES, "h5"))
  side = BrowserSide(st, QFont())
  files = side.files
  files.show()
  app.processEvents()
  # up to the fixtures, onto the folder the list came out of
  QTest.keyClick(files, Qt.Key_Backspace)
  k = files.currentIndex().row()
  assert st.folder == os.path.abspath(FIXTURES) and st.entries[k].name == "h5" and st.entries[k].isDir
  QTest.keyClick(files, Qt.Key_Right)
  assert st.entries[k].open and files.currentIndex().row() == k
  QTest.keyClick(files, Qt.Key_Right)
  assert files.currentIndex().row() == k + 1 and st.entries[k + 1].depth == 1
  QTest.keyClick(files, Qt.Key_Left)
  assert files.currentIndex().row() == k and st.entries[k].open
  QTest.keyClick(files, Qt.Key_Left)
  assert not st.entries[k].open and files.currentIndex().row() == k
  # a letter: the first name starting with it, the same letter again the next
  QTest.keyClick(files, "f")
  first = files.currentIndex().row()
  files.readCurrent()
  assert st.entries[first].name.lower().startswith("f") and st.selected == st.entries[first].path
  QTest.keyClick(files, "F")
  second = files.currentIndex().row()
  assert second > first and st.entries[second].name.lower().startswith("f")
  QTest.keyClick(files, "f")
  assert files.currentIndex().row() == first
  QTest.keyClick(files, "o")
  assert st.entries[files.currentIndex().row()].name.startswith("olivine")


# a double click on a file reads it when it is not the one selected - two quick clicks near one
# another come as one - and does not read again the file its first click has just read
def test_double_click_on_a_file():
  from PySide6.QtTest import QTest
  app = application()
  st = WizardState(FIXTURES)
  cb.setFolder(st, FIXTURES)
  side = BrowserSide(st, QFont())
  files = side.files
  files.resize(300, 600)
  files.show()
  app.processEvents()
  started = []
  worker = lambda load, commit: started.append(load)
  files.worker = worker
  k = [r.name for r in st.entries].index("Forsterite_cut.ctf")
  center = files.visualRect(files.model_.index(k, 0)).center()
  QTest.mouseDClick(files.viewport(), Qt.LeftButton, Qt.NoModifier, center)
  assert st.selected.endswith("Forsterite_cut.ctf") and len(started) == 1 and files.currentIndex().row() == k
  QTest.mouseDClick(files.viewport(), Qt.LeftButton, Qt.NoModifier, center)
  assert len(started) == 1
