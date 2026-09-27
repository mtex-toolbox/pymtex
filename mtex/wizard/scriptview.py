# the script of the import as it stands, beside the phase table: what the wizard's clicks come
# to in code, rewritten with every change that alters it, read only and ready to copy
from __future__ import annotations

import re

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFontDatabase, QGuiApplication, QSyntaxHighlighter, QTextCharFormat
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QPlainTextEdit, QToolButton, QWidget

from . import callbacks as cb

# what the script is written from: the map, its phases, the two frames, the phases drawn and the
# data set
WATCHED = ("ebsd", "phases", "mapConvention", "eulerConvention", "drawn", "dataSet")
COMMENT_INK = "#7a7a7a"
STRING_INK = "#2e7d32"
PLACEHOLDER = "the script of the import appears here once a map is read"
MIN_WIDTH = 240   # narrower than that a line of the script cannot be read, and a note says so


def buttonStyle(palette):
  """the buttons filled in the colour of a selected row of the file list, the palette's highlight,
  green while they say they did their work"""
  from PySide6.QtGui import QPalette
  fill, ink = palette.color(QPalette.Highlight), palette.color(QPalette.HighlightedText)
  return f"""
QToolButton {{ background: {fill.name()}; color: {ink.name()}; border: none; border-radius: 3px; padding: 1px 7px; font-weight: bold; }}
QToolButton:hover {{ background: {fill.lighter(115).name()}; }}
QToolButton:pressed {{ background: {fill.darker(125).name()}; }}
QToolButton:disabled {{ background: #c8c8c8; color: #f4f4f4; }}
QToolButton[done="true"] {{ background: #2e7d32; color: white; }}
"""


class PythonInk(QSyntaxHighlighter):
  """the comments grey, the strings green, as an editor shows Python"""

  STRING = re.compile(r"'[^'\n]*'|\"[^\"\n]*\"")

  def __init__(self, document):
    super().__init__(document)
    self.comment, self.string = QTextCharFormat(), QTextCharFormat()
    self.comment.setForeground(QColor(COMMENT_INK))
    self.string.setForeground(QColor(STRING_INK))

  def highlightBlock(self, text):
    strings = [m.span() for m in self.STRING.finditer(text)]
    for a, b in strings:
      self.setFormat(a, b - a, self.string)
    # a hash inside a string is no comment
    for m in re.finditer("#", text):
      if not any(a <= m.start() < b for a, b in strings):
        self.setFormat(m.start(), len(text) - m.start(), self.comment)
        break


class ScriptView(QWidget):
  """the import script, in a small fixed-width font, with a save and a copy button in its corner"""

  def __init__(self, st):
    super().__init__()
    self.st = st
    lay = QGridLayout(self)
    lay.setContentsMargins(0, 0, 0, 0)
    self.text = QPlainTextEdit()
    self.text.setReadOnly(True)
    self.text.setFrameShape(QFrame.NoFrame)
    self.text.setLineWrapMode(QPlainTextEdit.NoWrap)
    self.text.setPlaceholderText(PLACEHOLDER)
    font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
    font.setPointSizeF(max(7.0, self.font().pointSizeF() - 1))
    self.text.setFont(font)
    self.ink = PythonInk(self.text.document())
    # the buttons are children of the text, which keeps them over it whatever the platform
    # stacks, filled so that they stand out and cover the end of the first line cleanly
    self.save = self._button("save…", "write the script to a file", self.saveScript)
    self.copy = self._button("copy", "copy the script to the clipboard", self.copyScript)
    self.narrow = QLabel("widen the window to see the import script")
    self.narrow.setWordWrap(True)
    self.narrow.setAlignment(Qt.AlignCenter)
    self.narrow.setStyleSheet(f"color: {COMMENT_INK};")
    lay.addWidget(self.text, 0, 0)
    lay.addWidget(self.narrow, 0, 0)
    self.setMinimumWidth(0)
    self.fit()
    for name in WATCHED:
      st.on(name, self.refresh)
    self.refresh()

  def _button(self, label, tip, action):
    b = QToolButton(self.text)
    b.setText(label)
    b.setToolTip(tip)
    b.setStyleSheet(buttonStyle(self.palette()))
    b.setCursor(Qt.PointingHandCursor)
    b.clicked.connect(action)
    b.label = label
    return b

  def refresh(self):
    """the text replaced only when the script changed, so that a scrolled view stays where it is"""
    script = cb.scriptOf(self.st)
    if script != self.text.toPlainText():
      bar = self.text.verticalScrollBar().value()
      self.text.setPlainText(script)
      self.text.verticalScrollBar().setValue(bar)
    self.copy.setEnabled(bool(script))
    self.save.setEnabled(bool(script))

  def fit(self):
    """the script where there is room to read it, the note where there is not; the buttons in the
    top right corner of the text, clear of its scroll bar"""
    wide = self.width() >= MIN_WIDTH
    self.text.setVisible(wide)
    self.narrow.setVisible(not wide)
    bar = self.text.verticalScrollBar()
    x = self.text.width() - (bar.width() if bar.isVisible() else 0) - 4
    for b in (self.copy, self.save):
      b.adjustSize()
      x -= b.width()
      b.move(x, 3)
      x -= 4

  def resizeEvent(self, event):
    super().resizeEvent(event)
    self.fit()

  def changeEvent(self, event):
    """a changed theme recolours the buttons with it"""
    from PySide6.QtCore import QEvent
    super().changeEvent(event)
    if event.type() == QEvent.PaletteChange and hasattr(self, "copy"):
      for b in (self.save, self.copy):
        b.setStyleSheet(buttonStyle(self.palette()))

  def showEvent(self, event):
    super().showEvent(event)
    self.fit()

  def confirm(self, button, text):
    """a button says for a moment that it did what it was asked, where the eye is"""
    self.mark(button, text, True)
    QTimer.singleShot(1500, lambda: self.mark(button, button.label, False))

  def mark(self, button, text, done):
    button.setText(text)
    button.setProperty("done", done)
    button.style().unpolish(button)
    button.style().polish(button)
    button.adjustSize()
    self.fit()

  def copyScript(self):
    script = self.text.toPlainText()
    if script:
      QGuiApplication.clipboard().setText(script)
      self.st.status = ("loaded", "the import script is on the clipboard")
      self.st.changed("status")
      self.confirm(self.copy, "copied ✓")

  def saveScript(self):
    from PySide6.QtWidgets import QFileDialog
    path, _ = QFileDialog.getSaveFileName(self, "save the import script", cb.defaultScriptPath(self.st), "Python (*.py)")
    if path and cb.writeScript(self.st, path):
      self.confirm(self.save, "saved ✓")


__all__ = ["ScriptView"]
