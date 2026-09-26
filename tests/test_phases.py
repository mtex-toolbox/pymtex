import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from mtex.wizard import callbacks as cb
from mtex.wizard.phases import PhaseTable, WIDTHS
from mtex.wizard.state import WizardState
from mtex.wizard.window import application

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "ebsd")


@pytest.fixture
def loaded():
  application()
  st = WizardState(FIXTURES)
  cb.openEntry(st, os.path.join(FIXTURES, "Forsterite_cut.ctf"))
  return st


def test_table_rows_and_edits(loaded):
  st = loaded
  table = PhaseTable(st, QFont())
  assert table.rowCount() == 5 and table.item(0, 2).text() == "notIndexed" and table.item(1, 2).text() == "Forsterite"
  assert table.item(1, 0).checkState() == Qt.Checked and table.item(1, 6).text() == "4.756"
  # a cubic lattice greys b and c; the not indexed row is fixed throughout
  assert not table.item(4, 7).flags() & Qt.ItemIsEditable and table.item(4, 6).flags() & Qt.ItemIsEditable
  assert not table.item(0, 6).flags() & Qt.ItemIsEditable
  # an edit through the table reaches the state and comes back into the row
  table.item(1, 6).setText("4.8")
  assert abs(st.ebsd.CSList.frames[1].abc[0] - 4.8) < 1e-9 and table.item(1, 6).text() == "4.8"
  table.item(1, 0).setCheckState(Qt.Unchecked)
  assert st.drawn[1] is False
  table.item(2, 2).setText("En")
  assert st.ebsd.CSList.frames[2].name == "En"
  # a value that is no number is reported, not applied
  table.item(1, 7).setText("x")
  assert st.status[0] == "failed" and table.item(1, 7).text() != "x"
  assert table.width() >= sum(WIDTHS)




# every phase has a plot box, the one that was never indexed among them, and it is centred
def test_plot_column(loaded):
  st = loaded
  table = PhaseTable(st, QFont())
  assert st.drawn == {0: True, 1: True, 2: True, 3: True, 4: True}
  for row in range(table.rowCount()):
    item = table.item(row, 0)
    assert item.flags() & Qt.ItemIsUserCheckable and item.checkState() == Qt.Checked
  # the box the delegate draws is in the middle of its cell, not against its left edge
  from PySide6.QtWidgets import QStyleOptionViewItem
  delegate = table.itemDelegateForColumn(0)
  option = QStyleOptionViewItem()
  option.rect = table.visualRect(table.model().index(0, 0))
  box = delegate.boxRect(option)
  assert abs(box.center().x() - option.rect.center().x()) <= 1
  assert box.left() > option.rect.left() + 6
  # a click in the cell ticks it off, and the phase then reaches the drawing as hidden
  _click(table, 0)
  assert st.drawn[0] is False and table.item(0, 0).checkState() == Qt.Unchecked
  _click(table, 0)
  assert st.drawn[0] is True


def _click(table, row, column=0):
  from PySide6.QtCore import QEvent
  from PySide6.QtGui import QMouseEvent
  from PySide6.QtWidgets import QApplication
  r = table.visualRect(table.model().index(row, column))
  for kind in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease):
    QApplication.instance().sendEvent(table.viewport(), QMouseEvent(kind, r.center(), table.viewport().mapToGlobal(r.center()),
                                                                    Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))


def test_a_lattice_parameter_left_undefined_is_shown_as_it_is():
  from mtex.wizard.phases import _fmt
  assert _fmt(float("nan")) == "nan" and _fmt(4.756) == "4.756" and _fmt(90.0) == "90"


# a click on a row shows its phase's key, whose name then stands out; the row never indexed has none
def test_a_click_picks_the_key(loaded):
  st = loaded
  table = PhaseTable(st, QFont())
  assert table.item(1, 2).font().bold() and st.keyPhase == 1
  table.onClick(2, 3)
  assert st.keyPhase == 2 and table.item(2, 2).font().bold() and not table.item(1, 2).font().bold()
  table.onClick(0, 3)
  assert st.keyPhase == 2
