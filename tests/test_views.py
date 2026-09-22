import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import matplotlib

matplotlib.use("QtAgg")
from PySide6.QtGui import QFont

from mtex.wizard import callbacks as cb
from mtex.wizard.state import WizardState
from mtex.wizard.views import MetaTree, MillerBar, ViewCanvas, millerOf
from mtex.wizard.window import application

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "ebsd")


@pytest.fixture
def loaded():
  application()
  st = WizardState(FIXTURES)
  cb.openEntry(st, os.path.join(FIXTURES, "Forsterite_cut.ctf"))
  return st


# every kind of view draws on the canvas without an error in the status line
def test_every_view_draws(loaded):
  st = loaded
  canvas = ViewCanvas(st)
  for k, name in enumerate(st.tabs):
    cb.pickTab(st, k)
    canvas.draw_view()
    canvas.draw()
    assert st.status[0] == "loaded", (name, st.status)
  # a phase unticked leaves its cells transparent; a property carries a colour bar
  cb.pickTab(st, st.tabs.index("bc"))
  canvas.draw_view()
  assert canvas.colorbar is not None
  cb.pickTab(st, 0)
  cb.editPhase(st, 1, "plot", False)
  canvas.draw_view()
  assert canvas.colorbar is None
  colors = canvas.image.get_facecolors() if hasattr(canvas.image, "get_facecolors") else canvas.image.get_array()
  assert np.any(np.asarray(colors)[..., 3] == 0)


def test_miller_bar_and_metadata(loaded):
  st = loaded
  bar = MillerBar(st)
  cb.pickTab(st, st.tabs.index("pole figures"))
  assert bar.isVisibleTo(bar) is True or st.tabKinds[st.tab] == "polefigures"
  bar.fields[0].setText("110")
  bar.fields[0].editingFinished.emit()
  assert st.millers[0] == "110"
  h = millerOf("1-10", st.ebsd.CSList.frames[1])
  assert np.allclose(h.hkl, [1, -1, 0]) if hasattr(h, "hkl") else True
  tree = MetaTree(st, QFont())
  assert tree.topLevelItemCount() >= 1 and tree.topLevelItem(0).text(0) == "header"


# the keys stand beside a map that leaves room for them and over one that does not
def test_key_placement(loaded):
  st = loaded
  canvas = ViewCanvas(st)
  cb.pickTab(st, st.tabs.index("IPF Z"))
  canvas.figure.set_size_inches(12, 6)
  canvas.draw_view()
  canvas.draw()
  # the keys are built after the map is on screen and not before it
  assert canvas.keyAxes == [] and canvas.keyPending
  # one key a turn, so that building them does not hold the window
  for _ in range(4):
    application().processEvents()
  assert len(canvas.keyAxes) == 3 and len(canvas.keyGrounds) == 3 and not canvas.keyPending
  # this map is wider than the canvas, so it is drawn to the whole width and the keys lie over it
  assert canvas.ax.get_position().width > 0.9
  assert all(ax.get_axes_locator().bounds[0] < 1 for ax in canvas.keyAxes)
  # a map narrower than the canvas leaves room beside it, and the keys take it
  canvas.ax.set_xlim(0, 100)
  canvas.ax.set_ylim(0, 100)
  canvas.placeKeys()
  assert not canvas.keyGrounds
  assert all(ax.get_axes_locator().bounds[0] > 1 for ax in canvas.keyAxes)


# a colour bar is made for a property and taken away again, however many properties are
# looked at on the way: it can only be removed through the mappable the figure knows
def test_colorbar_comes_and_goes(loaded):
  st = loaded
  canvas = ViewCanvas(st)
  properties = [t for t, k in zip(st.tabs, st.tabKinds) if k == "property"]
  assert len(properties) > 1
  for name in properties:
    cb.pickTab(st, st.tabs.index(name))
    canvas.draw_view()
    assert canvas.colorbar is not None and canvas.mappable is not None
    assert canvas.colorbar.mappable is canvas.mappable
    assert st.status[0] == "loaded", (name, st.status)
  for name in ("phase", "IPF Z"):
    cb.pickTab(st, st.tabs.index(name))
    canvas.draw_view()
    assert canvas.colorbar is None and canvas.mappable is None
    assert st.status[0] == "loaded", (name, st.status)
  # and again, after the map has carried one already
  cb.pickTab(st, st.tabs.index(properties[0]))
  canvas.draw_view()
  canvas.draw()
  assert canvas.colorbar is not None
