# the phase table of the import wizard (docs/plans/wizard.md W3)
#
# The table is inline editable: a cell is its value until it is clicked. Its columns are
# each as wide as the longest thing they can hold, the numeric ones right aligned, and the
# cells the lattice forces are greyed background and all. The two reference frames beside
# it are pictogram.py.
from __future__ import annotations

import numpy as np

from PySide6.QtCore import QEvent, QRect, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QFrame, QHeaderView, QStyle,
                               QStyleOptionButton, QStyleOptionViewItem, QStyledItemDelegate, QTableWidget,
                               QTableWidgetItem)

from ..geometry.frames import alignmentSetups, closestAlignment
from ..geometry.symmetry import symmetry
from ..geometry.units import degree
from ..maps.phaselist import CSS_255, phaseColorNames
from . import callbacks as cb

HEADINGS = ["Plot", "Phase", "mineral", "pixels", "%", "symmetry", "a", "b", "c", "α", "β", "γ", "alignment"]
WIDTHS = [36, 44, 118, 64, 48, 74, 56, 56, 56, 49, 49, 49, 108]
NUMERIC = {"pixels", "percent", "a", "b", "c", "alpha", "beta", "gamma"}
GREY = QColor("#e4e4e4")
ROW_HEIGHT = 28
FRAME = "#c8c8c8"


# ------------------------------------------------------------------ the phase table

# the check box centred in its cell: an item draws its indicator against the left edge
# whatever the item's alignment says, so the cell is drawn without one and the box is put
# in the middle of it, where a column three characters wide wants it
class CheckDelegate(QStyledItemDelegate):

  def paint(self, painter, option, index):
    opt = QStyleOptionViewItem(option)
    self.initStyleOption(opt, index)
    opt.features &= ~QStyleOptionViewItem.HasCheckIndicator
    opt.text = ""
    style = opt.widget.style() if opt.widget else QApplication.style()
    style.drawControl(QStyle.CE_ItemViewItem, opt, painter, opt.widget)
    box = QStyleOptionButton()
    box.state = QStyle.State_Enabled | (QStyle.State_On if _checked(index) else QStyle.State_Off)
    box.rect = self.boxRect(option, style)
    style.drawControl(QStyle.CE_CheckBox, box, painter)

  # where the box sits: the style's own indicator, centred in the cell
  def boxRect(self, option, style=None):
    style = style or QApplication.style()
    size = style.subElementRect(QStyle.SE_CheckBoxIndicator, QStyleOptionButton(), None).size()
    c = option.rect.center()
    return QRect(c.x() - size.width() // 2 + 1, c.y() - size.height() // 2 + 1, size.width(), size.height())

  # a click anywhere in the cell ticks it, as does the space bar
  def editorEvent(self, event, model, option, index):
    if not index.flags() & Qt.ItemIsUserCheckable:
      return False
    toggled = event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton \
        or event.type() == QEvent.KeyPress and event.key() in (Qt.Key_Space, Qt.Key_Select)
    if not toggled:
      return event.type() in (QEvent.MouseButtonPress, QEvent.MouseButtonDblClick)
    model.setData(index, Qt.Unchecked if _checked(index) else Qt.Checked, Qt.CheckStateRole)
    return True


# whether a cell is ticked; the role comes back as a number where the flag is an enum
def _checked(index):
  v = index.data(Qt.CheckStateRole)
  return v is not None and v == Qt.Checked.value


# a dropdown in a cell: the point groups, or the twelve setups
class ChoiceDelegate(QStyledItemDelegate):

  def __init__(self, choices):
    super().__init__()
    self.choices = choices

  def createEditor(self, parent, option, index):
    box = QComboBox(parent)
    box.addItems(self.choices)
    box.setEditable(False)
    return box

  def setEditorData(self, editor, index):
    k = editor.findText(index.data(Qt.EditRole) or "")
    editor.setCurrentIndex(max(k, 0))
    editor.showPopup()

  def setModelData(self, editor, model, index):
    model.setData(index, editor.currentText(), Qt.EditRole)


class PhaseTable(QTableWidget):
  """the phases of the map, one row each, every cell but the counts editable"""

  def __init__(self, st, small):
    super().__init__(0, len(HEADINGS))
    self.st = st
    self.small = small
    self.filling = 0
    self.setHorizontalHeaderLabels(HEADINGS)
    self.verticalHeader().hide()
    self.setFrameShape(QFrame.NoFrame)
    self.setAlternatingRowColors(True)
    self.setSelectionMode(QAbstractItemView.NoSelection)
    self.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed | QAbstractItemView.AnyKeyPressed)
    self.horizontalHeader().setSectionResizeMode(QHeaderView.Fixed)
    self.horizontalHeader().setStretchLastSection(False)
    self.horizontalHeader().setHighlightSections(False)
    for k, w in enumerate(WIDTHS):
      self.setColumnWidth(k, w)
    self.groups = [g[0] for g in symmetry.pointGroups]
    self.setItemDelegateForColumn(0, CheckDelegate())
    self.setItemDelegateForColumn(5, ChoiceDelegate(self.groups))
    self.setItemDelegateForColumn(12, ChoiceDelegate([", ".join(s) for s in alignmentSetups()]))
    self.cellChanged.connect(self.onEdit)
    self.cellClicked.connect(self.onClick)
    self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    self.verticalHeader().setDefaultSectionSize(ROW_HEIGHT)
    self.setFixedWidth(sum(WIDTHS) + self.verticalScrollBar().sizeHint().width() + 4)
    for name in ("ebsd", "phases", "drawn"):
      st.on(name, self.refresh)
    self.refresh()

  # the rows: every phase of the map, the not indexed one first as MTEX lists it
  def rows(self):
    if self.st.ebsd is None:
      return []
    counts = cb.phaseCounts(self.st)
    return [(k, f, counts.get(k, 0)) for k, f in enumerate(self.st.ebsd.CSList.frames) if f.isIndexed or counts.get(k, 0)]

  # the table filled with its signals blocked, so that no cell set here reads as an edit
  def refresh(self):
    self.filling += 1
    self.blockSignals(True)
    try:
      self._fill()
    finally:
      self.blockSignals(False)
      self.filling -= 1

  def _fill(self):
    rows = self.rows()
    self.setRowCount(len(rows))
    total = max(sum(c for _, _, c in rows), 1)
    colors = phaseColorNames(self.st.ebsd.CSList.frames) if self.st.ebsd is not None else []
    for i, (k, f, count) in enumerate(rows):
      free = cb.freeCells(f)
      # every phase is drawn or not, the one that was never indexed among them
      plot = QTableWidgetItem()
      plot.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
      plot.setCheckState(Qt.Checked if self.st.drawn.get(k, False) else Qt.Unchecked)
      plot.setTextAlignment(Qt.AlignCenter)
      self.setItem(i, 0, plot)
      phase = self._fixed(str(k), center=True)
      c = colors[k] if k < len(colors) else None
      if c is not None:
        rgb = _rgb(c)
        phase.setBackground(QColor(*rgb))
        phase.setForeground(QColor("black") if 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2] > 140 else QColor("white"))
        phase.setToolTip("click to choose the colour")
      self.setItem(i, 1, phase)
      self.setItem(i, 2, self._cell(f.name))
      self.setItem(i, 3, self._fixed(str(count), right=True))
      self.setItem(i, 4, self._fixed(f"{100 * count / total:.1f}", right=True))
      if f.isIndexed:
        self.setItem(i, 5, self._cell(f.pointGroup + " ▾"))
        for j, (v, key) in enumerate(zip(f.abc, ("a", "b", "c"))):
          self.setItem(i, 6 + j, self._cell(_fmt(v), right=True, grey=not free[key]))
        for j, (v, key) in enumerate(zip(f.abg / degree, ("alpha", "beta", "gamma"))):
          self.setItem(i, 9 + j, self._cell(_fmt(v), right=True, grey=not free[key]))
        self.setItem(i, 12, self._cell(", ".join(closestAlignment(f)) + " ▾"))
      else:
        for j in range(5, 13):
          self.setItem(i, j, self._fixed("", grey=True))

  def _cell(self, text, right=False, grey=False):
    item = QTableWidgetItem(text)
    if right:
      item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
    if grey:
      item.setBackground(GREY)
      item.setFlags(Qt.ItemIsEnabled)
    return item

  def _fixed(self, text, right=False, center=False, grey=False):
    item = QTableWidgetItem(text)
    item.setFlags(Qt.ItemIsEnabled)
    if right:
      item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
    if center:
      item.setTextAlignment(Qt.AlignCenter)
    if grey:
      item.setBackground(GREY)
    return item

  # a cell the user changed; the table emits one per cell as it is torn down with the
  # window, when the map behind it may already be gone, and those are not edits
  def onEdit(self, i, j):
    if self.filling or self.st.ebsd is None:
      return
    try:
      rows = self.rows()
    except AttributeError:
      return
    if not 0 <= i < len(rows) or self.item(i, j) is None:
      return
    k = rows[i][0]
    column = cb.PHASE_COLUMNS[j]
    if column in ("phase", "pixels", "percent"):
      return
    text = self.item(i, j).text().replace("▾", "").strip()
    try:
      if column == "plot":
        cb.editPhase(self.st, k, "plot", self.item(i, 0).checkState() == Qt.Checked)
      elif column in ("mineral", "symmetry"):
        cb.editPhase(self.st, k, column, text)
      elif column == "alignment":
        cb.editPhase(self.st, k, column, text)
      elif column in NUMERIC:
        cb.editPhase(self.st, k, column, float(text))
    except Exception as e:
      self.st.status = ("failed", f"{column}: {e}")
      self.st.changed("status")
      self.refresh()

  # a click opens the cell's editor; the phase cell is the swatch, the id and the target for the colour dialog in one
  def onClick(self, i, j):
    if j != 1:
      item = self.item(i, j)
      if item is not None and item.flags() & Qt.ItemIsEditable:
        self.editItem(item)
      return
    rows = self.rows()
    if not 0 <= i < len(rows):
      return
    from PySide6.QtWidgets import QColorDialog
    k, f, _ = rows[i]
    colors = phaseColorNames(self.st.ebsd.CSList.frames)
    start = QColor(*_rgb(colors[k])) if colors[k] is not None else QColor("white")
    c = QColorDialog.getColor(start, self, f"the colour of {f.name}")
    if c.isValid():
      cb.editPhase(self.st, k, "color", (c.redF(), c.greenF(), c.blueF()))


def _fmt(v):
  return f"{float(v):.4g}" if abs(v - round(v)) > 1e-9 else str(int(round(v)))


# a colour name or triple as 0 to 255
def _rgb(c):
  if isinstance(c, str):
    if c.lower() in CSS_255:
      return CSS_255[c.lower()]
    q = QColor(c)
    return (q.red(), q.green(), q.blue())
  a = np.asarray(c, dtype=float).reshape(-1)[:3]
  return tuple(int(round(255 * x)) for x in a)


__all__ = ["PhaseTable", "CheckDelegate"]
