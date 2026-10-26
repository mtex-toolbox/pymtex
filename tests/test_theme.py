import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QWidget

from mtex.wizard.theme import DARK, LIGHT, applyScheme, colours
from mtex.wizard.window import application, runWizard

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "ebsd")


# both tables name the same colours
def test_theme_tablesNameTheSameColours():
  assert all(LIGHT.keys() == DARK.keys() and (not isinstance(v, dict) or v.keys() == DARK[k].keys())
             for k, v in LIGHT.items())


# the tab of the file's pictures has the ink of the pictures in the tree of its metadata, and no other tab has it
@pytest.mark.parametrize("table", [LIGHT, DARK], ids=["light", "dark"])
def test_theme_imagesTabHasThePictureInk(table):
  inks = list(table["tabInk"].values())
  assert table["tabInk"]["images"] == table["pictureInk"] and inks.count(table["pictureInk"]) == 1


# the window follows the desktop's scheme, every widget of it, the item views included, drawn in
# its colours
@pytest.mark.parametrize("dark", [False, True], ids=["light", "dark"])
def test_theme_windowFollowsTheDesktop(dark):
  app = application()
  app.setPalette(QPalette(QColor("#2b2b2b" if dark else "#efefef"), QColor("#2b2b2b" if dark else "#efefef")))
  w = runWizard(FIXTURES, exec=False)
  bases = [x.palette().color(QPalette.Base).lightness() for x in [w, *w.findChildren(QWidget)] if x.isEnabled()]
  w.close()
  applyScheme(app, dark=False)
  assert all((b < 128) == dark for b in bases)


# a scheme asked for wins over the desktop's
def test_theme_schemeAskedForWins():
  app = application()
  app.setPalette(QPalette(QColor("#efefef"), QColor("#efefef")))
  application("dark")
  dark = colours()["dark"]
  applyScheme(app, dark=False)
  assert dark
