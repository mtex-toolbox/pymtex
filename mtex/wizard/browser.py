# the browser side of the import wizard: the path, the tree, the status band, the data
# sets, the file information with the metadata of the map (docs/plans/wizard.md W2)
#
# Every control reads the state and writes back through one callback; nothing here holds
# data of its own.
from __future__ import annotations

import os

from PySide6.QtCore import QAbstractListModel, QModelIndex, QPointF, QRect, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QIcon, QPainter, QPainterPath, QPalette, QPen, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QFileDialog, QFrame, QHBoxLayout, QLabel, QListView, QListWidget,
                               QMenu, QSizePolicy, QStyle, QStyledItemDelegate, QToolButton, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)

from . import callbacks as cb

FRAME = "#c8c8c8"
CRUMBS = 3          # the folders of the path drawn, the rest behind the ellipsis
CRUMB_WIDTH = 110   # the widest a folder name is drawn before it is elided
CRUMB_GREY = "#888" # the separators of the path and the folder before it
STATUS_COLORS = {"idle": "#dbe9f6", "loading": "#fbe5b6", "loaded": "#d8efd3", "failed": "#f4c7c3"}
ROW_MARGIN = 3


def framed(widget):
  """a panel: a widget behind the grey frame the whole window is drawn in. It carries no
  heading of its own - what each panel holds is said by what is in it, the file's own rows,
  the columns of the phase table, the names of the data sets - and four headings over four
  panels is a row of labels to read before the window can be."""
  box = QFrame()
  box.setFrameShape(QFrame.NoFrame)
  box.setStyleSheet(f"QFrame#panel {{ border: 1px solid {FRAME}; background: white; }}")
  box.setObjectName("panel")
  lay = QVBoxLayout(box)
  lay.setContentsMargins(1, 1, 1, 1)
  lay.setSpacing(0)
  lay.addWidget(widget)
  return box


# ------------------------------------------------------------------ the path

def folderIcon(side, ratio=1.0):
  """a folder drawn as its outline in the grey of the separators, so that it reads as part of
  the path and not as a coloured picture of the desktop's theme before it"""
  pix = QPixmap(round(side * ratio), round(side * ratio))
  pix.setDevicePixelRatio(ratio)
  pix.fill(Qt.transparent)
  painter = QPainter(pix)
  painter.setRenderHint(QPainter.Antialiasing)
  pen = QPen(QColor(CRUMB_GREY), max(1.0, side / 13))
  pen.setJoinStyle(Qt.RoundJoin)
  painter.setPen(pen)
  # the tab on the upper left, the body below it, in units of the side
  outline = [(0.10, 0.80), (0.10, 0.20), (0.40, 0.20), (0.50, 0.33), (0.90, 0.33), (0.90, 0.80)]
  path = QPainterPath(QPointF(*outline[0]) * side)
  for p in outline[1:]:
    path.lineTo(QPointF(*p) * side)
  path.closeSubpath()
  painter.drawPath(path)
  painter.end()
  return QIcon(pix)


class Breadcrumb(QWidget):
  """the folder as the end of its path, every folder on it a target

  Only the last CRUMBS folders are drawn, and a long name is elided, so that the panel's
  width is the browser's and not the depth of whatever folder is open; the ones left out
  are behind the leading ellipsis, which lists them from the root down.
  """

  def __init__(self, st):
    super().__init__()
    self.st = st
    # the row is given whatever width the panel has rather than claiming its contents'
    self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
    row = QHBoxLayout(self)
    row.setContentsMargins(4, 2, 4, 2)
    row.setSpacing(2)
    self.browse = self._browse()
    row.addWidget(self.browse)
    # the crumbs are a row of their own, rebuilt with every folder, the button before them stays
    self.lay = QHBoxLayout()
    self.lay.setContentsMargins(0, 0, 0, 0)
    self.lay.setSpacing(0)
    row.addLayout(self.lay, 1)
    st.on("folder", self.refresh)
    self.refresh()

  def chooseFolder(self):
    """the folder picked in the dialog of the system, which reaches other drives and network
    shares; MATLAB's BrowseFolderButton of the import wizard"""
    folder = QFileDialog.getExistingDirectory(self.window(), "Select the data folder", self.st.folder)
    if folder:
      cb.setFolder(self.st, folder)

  # the button of the dialog: a folder in outline, as high as a crumb and raised as one
  def _browse(self):
    b = QToolButton()
    b.setAutoRaise(True)
    b.setCursor(Qt.PointingHandCursor)
    b.setToolTip("Choose a folder")
    side = QFontMetrics(b.font()).height()
    b.setIcon(folderIcon(side, self.devicePixelRatioF()))
    b.setIconSize(QSize(side, side))
    b.clicked.connect(self.chooseFolder)
    return b

  def refresh(self):
    """the path redrawn from the folder of the state"""
    while self.lay.count():
      item = self.lay.takeAt(0)
      if item.widget() is not None:
        item.widget().deleteLater()
    crumbs = cb.breadcrumbs(self.st)
    hidden, shown = crumbs[:-CRUMBS], crumbs[-CRUMBS:]
    if hidden:
      self.lay.addWidget(self._ellipsis(hidden))
      self.lay.addWidget(self._separator())
    for k, (name, path) in enumerate(shown):
      if k:
        self.lay.addWidget(self._separator())
      self.lay.addWidget(self._crumb(name, path, last=k == len(shown) - 1))
    self.lay.addStretch(1)

  def _crumb(self, name, path, last=False):
    b = QToolButton()
    b.setAutoRaise(True)
    b.setCursor(Qt.PointingHandCursor)
    b.setToolTip(path)
    b.clicked.connect(lambda _=False, p=path: cb.setFolder(self.st, p))
    if last:
      f = b.font()
      f.setBold(True)
      b.setFont(f)
    b.setText(QFontMetrics(b.font()).elidedText(name, Qt.ElideMiddle, CRUMB_WIDTH))
    return b

  # what is left of the path, from the root down
  def _ellipsis(self, hidden):
    b = QToolButton()
    b.setText("…")
    b.setAutoRaise(True)
    b.setCursor(Qt.PointingHandCursor)
    b.setToolTip(hidden[-1][1])
    b.setPopupMode(QToolButton.InstantPopup)
    menu = QMenu(b)
    for name, path in hidden:
      menu.addAction(name or path).triggered.connect(lambda _=False, p=path: cb.setFolder(self.st, p))
    b.setMenu(menu)
    return b

  def _separator(self):
    sep = QLabel("/")
    sep.setStyleSheet(f"color: {CRUMB_GREY}")
    return sep


# ------------------------------------------------------------------ the status band

class StatusBand(QLabel):
  """what to do and how the load went, coloured by the state"""

  def __init__(self, st):
    super().__init__()
    self.st = st
    self.setContentsMargins(8, 4, 8, 4)
    self.setWordWrap(True)
    self.setMinimumHeight(26)
    st.on("status", self.refresh)
    self.refresh()

  def refresh(self):
    """the band redrawn from the status of the state, its text and the colour of its kind"""
    kind, text = self.st.status
    self.setText(text)
    self.setStyleSheet(f"background: {STATUS_COLORS.get(kind, STATUS_COLORS['idle'])}; border: 1px solid {FRAME};")


# ------------------------------------------------------------------ the tree

class EntriesModel(QAbstractListModel):
  """the rows of the browser, the entries of the state, as a Qt list model"""

  def __init__(self, st):
    super().__init__()
    self.st = st

  def rowCount(self, parent=QModelIndex()):
    """the number of entries"""
    return 0 if parent.isValid() else len(self.st.entries)

  def data(self, index, role=Qt.DisplayRole):
    """the name of an entry for display, the entry itself for the user role"""
    if not index.isValid():
      return None
    row = self.st.entries[index.row()]
    if role == Qt.DisplayRole:
      return row.name
    if role == Qt.UserRole:
      return row
    return None

  def refresh(self):
    """the model reset, so that the view reads the entries anew"""
    self.beginResetModel()
    self.endResetModel()


class EntryDelegate(QStyledItemDelegate):
  """a row: an indent by its depth, a chevron for a folder, the name, the size in the small grey"""

  def __init__(self, small):
    super().__init__()
    self.small = small

  def paint(self, painter, option, index):
    """a row drawn: the indent, the chevron of a folder, the elided name and the size"""
    row = index.data(Qt.UserRole)
    painter.save()
    if option.state & QStyle.State_Selected:
      painter.fillRect(option.rect, option.palette.color(QPalette.Highlight))
      painter.setPen(option.palette.color(QPalette.HighlightedText))
    else:
      painter.setPen(option.palette.color(QPalette.Text))
    r = option.rect
    x = r.left() + 6 + 12 * row.depth
    fm = QFontMetrics(option.font)
    if row.isDir:
      painter.drawText(QRect(x, r.top(), 12, r.height()), Qt.AlignVCenter, "▾" if row.open else "▸")
    x += 14
    sizeText = _kB(row.size) if not row.isDir else ""
    wSize = fm.horizontalAdvance(sizeText) + 8 if sizeText else 0
    painter.drawText(QRect(x, r.top(), r.width() - x - wSize - 4, r.height()), Qt.AlignVCenter, fm.elidedText(row.name, Qt.ElideMiddle, r.width() - x - wSize - 4))
    if sizeText:
      painter.setFont(self.small)
      if not option.state & QStyle.State_Selected:
        painter.setPen(QColor("#777"))
      painter.drawText(QRect(r.right() - wSize, r.top(), wSize - 4, r.height()), Qt.AlignVCenter | Qt.AlignRight, sizeText)
    painter.restore()

  def sizeHint(self, option, index):
    """the height of a row, a line of text with its margins"""
    return QSize(option.rect.width(), QFontMetrics(option.font).height() + 2 * ROW_MARGIN)


def _kB(n):
  if n >= 1 << 30:
    return f"{n / (1 << 30):.1f} GB"
  if n >= 1 << 20:
    return f"{n / (1 << 20):.1f} MB"
  return f"{n / (1 << 10):.0f} kB"


# the pause after the cursor stops on a file before it is read, so that walking over a folder
# with the arrow keys reads only the file it ends on
WALK_DELAY_MS = 300


class FileList(QListView):
  """the folder as a tree: a click unfolds a folder or reads a map, a double click walks in"""

  def __init__(self, st, small, worker=None):
    super().__init__()
    self.st = st
    self.worker = worker
    self.walkTimer = QTimer(self)
    self.walkTimer.setSingleShot(True)
    self.walkTimer.setInterval(WALK_DELAY_MS)
    self.walkTimer.timeout.connect(self.readCurrent)
    self.model_ = EntriesModel(st)
    self.setModel(self.model_)
    self.setItemDelegate(EntryDelegate(small))
    self.setFrameShape(QFrame.NoFrame)
    self.setSelectionMode(QAbstractItemView.SingleSelection)
    self.clicked.connect(self.onClick)
    self.swallowRelease = False
    st.on("entries", self.model_.refresh)

  def onClick(self, index):
    """a click on a row: a folder unfolded or folded, a file read"""
    row = self.st.entries[index.row()]
    if row.isDir:
      cb.toggleFolder(self.st, index.row())
    else:
      cb.openEntry(self.st, row.path, self.worker)

  def mouseDoubleClickEvent(self, event):
    """the double click read off the row under the cursor, not off Qt's doubleClicked: the first
    click of it has unfolded the folder, which resets the model, and Qt then no longer finds
    the row pressed and drops the signal. Unfolding adds rows below the folder only, so the
    row under the cursor is still the folder; the release that ends the double click is
    swallowed, since the rows under it are the new folder's"""
    index = self.indexAt(event.position().toPoint())
    if event.button() == Qt.LeftButton and index.isValid():
      self.swallowRelease = True
      event.accept()
      self.onDoubleClick(index)
      return
    super().mouseDoubleClickEvent(event)

  def mousePressEvent(self, event):
    """a platform that ends a double click without a release leaves nothing to swallow"""
    self.swallowRelease = False
    super().mousePressEvent(event)

  def mouseReleaseEvent(self, event):
    """the release that ends a double click swallowed, any other passed on"""
    if self.swallowRelease:
      self.swallowRelease = False
      event.accept()
      return
    super().mouseReleaseEvent(event)

  def onDoubleClick(self, index):
    """a folder is walked into; a file is read unless it is the one selected already, which the
    first click of the double click has read - two quick clicks near one another come as a
    double click too, and the second must not be lost"""
    row = self.st.entries[index.row()]
    if row.isDir:
      cb.setFolder(self.st, row.path)
    elif self.st.selected != row.path:
      self.setCurrentIndex(index)
      cb.openEntry(self.st, row.path, self.worker)

  def currentChanged(self, current, previous):
    """the arrow keys read the file the cursor stops on; the one it leaves is not drawn"""
    super().currentChanged(current, previous)
    if current.isValid() and current.row() < len(self.st.entries):
      row = self.st.entries[current.row()]
      if not row.isDir and self.st.selected != row.path:
        cb.leaveEntry(self.st)
    self.walkTimer.start()

  def readCurrent(self):
    """the file under the cursor read, unless it is the one read already"""
    index = self.currentIndex()
    if index.isValid() and index.row() < len(self.st.entries):
      row = self.st.entries[index.row()]
      if not row.isDir and self.st.selected != row.path:
        cb.openEntry(self.st, row.path, self.worker)

  def keyPressEvent(self, event):
    """Return walks into a folder and reads a map, as a double click does; Right unfolds a folder
    where it stands and on an open one steps to its first entry, Left folds it or steps to the
    folder a row was unfolded from; Backspace walks up, onto the folder it came out of; a
    letter goes to the first entry starting with it, the same letter again to the next"""
    key, index = event.key(), self.currentIndex()
    i = index.row() if index.isValid() else None
    row = self.st.entries[i] if i is not None else None
    if key in (Qt.Key_Return, Qt.Key_Enter) and row is not None:
      if row.isDir:
        cb.setFolder(self.st, row.path)
      else:
        cb.openEntry(self.st, row.path, self.worker)
      return
    if key == Qt.Key_Backspace:
      left = self.st.folder
      cb.setFolder(self.st, os.path.dirname(left))
      self.goTo(next((j for j, r in enumerate(self.st.entries) if r.path == left), 0))
      return
    if key == Qt.Key_Right and row is not None and row.isDir:
      if not row.open:
        cb.toggleFolder(self.st, i)
        self.goTo(i)
      elif i + 1 < len(self.st.entries) and self.st.entries[i + 1].depth > row.depth:
        self.goTo(i + 1)
      return
    if key == Qt.Key_Left and row is not None:
      if row.isDir and row.open:
        cb.toggleFolder(self.st, i)
        self.goTo(i)
      else:
        parent = cb.parentRow(self.st, i)
        if parent is not None:
          self.goTo(parent)
      return
    text = event.text()
    if len(text) == 1 and text.isprintable() and not text.isspace() and not event.modifiers() & (Qt.ControlModifier | Qt.AltModifier):
      j = cb.letterRow(self.st, i, text)
      if j is not None:
        self.goTo(j)
      return
    super().keyPressEvent(event)

  def goTo(self, i):
    """the current row, shown; the list is refilled on every fold, which forgets it"""
    if 0 <= i < self.model_.rowCount():
      index = self.model_.index(i, 0)
      self.setCurrentIndex(index)
      self.scrollTo(index)


# ------------------------------------------------------------------ the data sets

class DataSetList(QListWidget):
  """the maps a project file holds; picking one is the import. Not there for a file of one."""

  def __init__(self, st, worker=None):
    super().__init__()
    self.st = st
    self.worker = worker
    self.setFrameShape(QFrame.NoFrame)
    self.itemClicked.connect(lambda item: cb.setDataSet(self.st, self.row(item), self.worker))
    st.on("info", self.refresh)
    st.on("dataSet", self.mark)
    self.refresh()

  def refresh(self):
    """the list refilled with the data sets of the file; its panel is shown by BrowserSide for
    more than one"""
    self.clear()
    sets = self.st.info.dataSets if self.st.info is not None else []
    for s in sets:
      self.addItem(s)
    self.setMaximumHeight(22 * max(len(sets), 1) + 6)
    self.mark()

  def mark(self):
    """the data set of the state selected in the list"""
    if 0 <= self.st.dataSet < self.count():
      self.setCurrentRow(self.st.dataSet)


# ------------------------------------------------------------------ the file information

PICTURE_INK = "#1f4e79"   # a leaf of the metadata that is a picture, and can be looked at


class InfoTree(QTreeWidget):
  """what the file says about itself, then the metadata of the map, in one tree

  The file's own description stands at the top, the few rows of it the window shows nowhere
  else; the metadata follows as the map carries it. A picture is a leaf like any other and
  says so; clicking it shows it, which is what the images view is for.
  """

  def __init__(self, st, small):
    super().__init__()
    self.st = st
    self.small = small
    self.setColumnCount(2)
    self.setHeaderHidden(True)
    self.setFrameShape(QFrame.NoFrame)
    self.setSelectionMode(QAbstractItemView.SingleSelection)
    self.setIndentation(12)
    self.header().setStretchLastSection(True)
    self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    self.itemClicked.connect(self.onClick)
    self.itemExpanded.connect(lambda item: self.remember(item, True))
    self.itemCollapsed.connect(lambda item: self.remember(item, False))
    for name in ("info", "meta"):
      st.on(name, self.refresh)
    st.on("image", self.mark)
    self.refresh()

  def refresh(self):
    """the tree refilled: the rows of the file, then the metadata of the map, branches opened as
    they were"""
    self.blockSignals(True)
    self.clear()
    for k, v in cb.infoRows(self.st):
      item = QTreeWidgetItem(self.invisibleRootItem(), [k, v])
      item.setData(0, Qt.UserRole, "file")
      item.setForeground(0, QColor("#666"))
      item.setFont(0, self.small)
      item.setToolTip(1, v)
    stack = [(-1, self.invisibleRootItem())]
    for i, (depth, key, value, branch, picture) in enumerate(self.st.metaRows):
      while stack and stack[-1][0] >= depth:
        stack.pop()
      parent = stack[-1][1] if stack else self.invisibleRootItem()
      item = QTreeWidgetItem(parent, [key, value])
      item.setData(0, Qt.UserRole, i)
      if branch:
        stack.append((depth, item))
        item.setExpanded(i in self.st.metaOpen)
      elif picture is not None:
        item.setForeground(1, QColor(PICTURE_INK))
        item.setToolTip(1, "click to look at it")
    self.resizeColumnToContents(0)
    self.setColumnWidth(0, min(self.columnWidth(0) + 8, 180))
    self.blockSignals(False)
    self.mark()

  def remember(self, item, isOpen):
    """a branch opened or shut stays so when the tree is filled again"""
    i = item.data(0, Qt.UserRole)
    if isinstance(i, int):
      (self.st.metaOpen.add if isOpen else self.st.metaOpen.discard)(i)

  def mark(self):
    """the picture being looked at is the one selected in the tree"""
    for item in self.walk():
      i = item.data(0, Qt.UserRole)
      row = self.st.metaRows[i] if isinstance(i, int) and i < len(self.st.metaRows) else None
      if row is not None and row[4] is not None and row[4] == self.st.image:
        self.setCurrentItem(item)

  def walk(self, parent=None):
    """every item below a parent, the whole tree by default, depth first"""
    parent = parent or self.invisibleRootItem()
    for k in range(parent.childCount()):
      child = parent.child(k)
      yield child
      yield from self.walk(child)

  def fileRows(self):
    """the rows of the file itself, name against value"""
    items = (self.topLevelItem(k) for k in range(self.topLevelItemCount()))
    return [(item.text(0), item.text(1)) for item in items if item.data(0, Qt.UserRole) == "file"]

  def onClick(self, item, column):
    """a click on a metadata row: the row picked, a picture shown"""
    i = item.data(0, Qt.UserRole)
    if isinstance(i, int):
      cb.pickMeta(self.st, i)


# ------------------------------------------------------------------ the side

class BrowserSide(QWidget):
  """the whole browser side, top to bottom"""

  def __init__(self, st, small, worker=None):
    super().__init__()
    lay = QVBoxLayout(self)
    lay.setContentsMargins(4, 4, 4, 4)
    lay.setSpacing(4)
    self.breadcrumb = Breadcrumb(st)
    self.status = StatusBand(st)
    self.files = FileList(st, small, worker)
    self.dataSets = DataSetList(st, worker)
    self.info = InfoTree(st, small)
    lay.addWidget(self.breadcrumb)
    lay.addWidget(framed(self.files), 3)
    lay.addWidget(self.status)
    self.dataSetPanel = framed(self.dataSets)
    lay.addWidget(self.dataSetPanel)
    lay.addWidget(framed(self.info), 3)
    st.on("info", self.showDataSets)
    self.showDataSets()

  def showDataSets(self):
    """the panel of the data sets shown only when the file holds more than one"""
    sets = self.dataSets.st.info.dataSets if self.dataSets.st.info is not None else []
    self.dataSetPanel.setVisible(len(sets) > 1)


__all__ = ["BrowserSide", "framed", "FRAME"]
