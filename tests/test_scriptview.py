import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from mtex.wizard import callbacks as cb
from mtex.wizard.scriptview import ScriptView
from mtex.wizard.state import WizardState
from mtex.wizard.window import application

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "ebsd")


# the script beside the phase table is the one the wizard writes, rewritten with every change that
# alters it; copy puts it on the clipboard and says so on the button
def test_the_script_follows_the_import():
  app = application()
  st = WizardState(FIXTURES)
  view = ScriptView(st)
  view.resize(600, 200)
  view.show()
  assert view.text.toPlainText() == "" and not view.copy.isEnabled()
  cb.openEntry(st, os.path.join(FIXTURES, "Forsterite_cut.ctf"))
  assert view.text.toPlainText() == cb.scriptOf(st) and "EBSD.load" in view.text.toPlainText()
  k = next(i for i, f in enumerate(st.ebsd.CSList.frames) if f.isIndexed)
  cb.editPhase(st, k, "symmetry", "222")
  assert "crystalFrame('222'" in view.text.toPlainText()
  # the buttons are over the text, in its top right corner, in the colour of a selected row
  from PySide6.QtGui import QPalette
  assert view.palette().color(QPalette.Highlight).name() in view.copy.styleSheet()
  assert view.copy.parent() is view.text and view.copy.x() > view.text.width() / 2 and view.save.x() < view.copy.x()
  view.copy.click()
  assert app.clipboard().text() == view.text.toPlainText() and view.copy.text() == "copied ✓" and view.copy.property("done")
  # back to its label, and to its blue, a moment later
  view.mark(view.copy, view.copy.label, False)
  assert view.copy.text() == "copy" and not view.copy.property("done")


# save writes the script where the dialog says, and says so on the button
def test_save(tmp_path, monkeypatch):
  from PySide6.QtWidgets import QFileDialog
  application()
  st = WizardState(FIXTURES)
  cb.openEntry(st, os.path.join(FIXTURES, "Forsterite_cut.ctf"))
  view = ScriptView(st)
  target = str(tmp_path / "import.py")
  monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (target, "")))
  view.save.click()
  assert open(target).read() == cb.scriptOf(st) and view.save.text() == "saved ✓"


# a box too narrow to read a line of the script says so instead
def test_a_narrow_box_says_so():
  application()
  st = WizardState(FIXTURES)
  cb.openEntry(st, os.path.join(FIXTURES, "Forsterite_cut.ctf"))
  view = ScriptView(st)
  view.resize(120, 200)
  view.show()
  assert view.narrow.isVisible() and not view.text.isVisible()
  view.resize(500, 200)
  assert view.text.isVisible() and not view.narrow.isVisible()
