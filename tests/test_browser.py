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
  assert side.info.rowCount() > 5 and side.info.item(0, 1).text() == "Forsterite_cut.ctf"
  assert not side.dataSetPanel.isVisibleTo(side)
  # the arrow keys describe the current row without reading it
  k = [r.name for r in st.entries].index("olivine_cut.ang")
  files.setCurrentIndex(files.model_.index(k, 0))
  assert st.info.name == "olivine_cut.ang" and st.ebsd.CSList.frames[1].name == "Forsterite"
  # the exports: the variable name and the load call
  side.exports.name.setText("fo")
  side.exports.name.editingFinished.emit()
  assert st.variable == "fo"
  assert cb.copyCall(st).startswith("fo = EBSD.load(")


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
