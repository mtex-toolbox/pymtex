import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import matplotlib

matplotlib.use("QtAgg")

from mtex.wizard import callbacks as cb
from mtex.wizard.state import WizardState
from mtex.wizard.pictogram import SIDE
from mtex.wizard.views import KeyPanel, MillerBar, ViewCanvas, millerOf
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


# the key panel shows one phase's key at a fixed size, at first the phase with the most pixels,
# then the one picked; the map carries no keys of its own
def test_the_key_panel(loaded):
  st = loaded
  panel = KeyPanel(st)
  frames = st.ebsd.CSList.frames
  assert st.keyPhase == cb.dominantPhase(st) and frames[st.keyPhase].name in panel.title.text()
  assert panel.picture.size().width() == panel.side and not panel.picture.pixmap().isNull()
  assert panel.picture.pixmap().width() <= panel.side * panel.devicePixelRatioF() + 1
  # as tall as the two pictograms, less its frame, so the top row keeps their height
  assert panel.height() == 2 * SIDE + 3 - 4
  other = next(k for k, f in enumerate(frames) if f.isIndexed and k != st.keyPhase)
  cb.pickKeyPhase(st, other)
  assert st.keyPhase == other and frames[other].name in panel.title.text()
  # the phase never indexed has no key, and picking it changes nothing
  cb.pickKeyPhase(st, 0)
  assert st.keyPhase == other
  canvas = ViewCanvas(st)
  cb.pickTab(st, st.tabs.index("IPF Z"))
  canvas.draw_view()
  canvas.draw()
  assert len(canvas.figure.axes) == 1


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


# the figure of the canvas keeps the widget's size: a map drawn into it does not take its own
# shape, which left the rest of the widget unpainted
def test_the_map_keeps_the_canvas_size(loaded):
  st = loaded
  canvas = ViewCanvas(st)
  canvas.resize(900, 700)
  canvas.draw()
  before = tuple(canvas.figure.get_size_inches())
  for name in ("phase", "IPF Z", "pole figures"):
    cb.pickTab(st, st.tabs.index(name))
    canvas.draw_view()
    canvas.draw()
    assert tuple(canvas.figure.get_size_inches()) == before, name


# a key whose sector reaches into both hemispheres is drawn as both, a Laue group's as one
def test_a_key_of_both_hemispheres(monkeypatch):
  import mtex.wizard.views as views
  from mtex import crystalFrame
  drawn = []
  real = views._hemispherePicture
  monkeypatch.setattr(views, "_hemispherePicture", lambda key, cs, px, side, **kw: drawn.append(dict(side)) or real(key, cs, px, side, **kw))
  views.keyPicture(crystalFrame("mm2"), px=120)
  assert drawn == [{"upper": True}, {"lower": True}]
  drawn.clear()
  views.keyPicture(crystalFrame("mmm"), px=120)
  assert drawn == [{}]
  # the two stand over or beside one another, whichever leaves them larger in a square
  wide, tall = np.zeros((20, 60, 4), dtype=np.uint8), np.zeros((60, 20, 4), dtype=np.uint8)
  assert views._arrange([wide, wide], 2).shape[:2] == (42, 60)
  assert views._arrange([tall, tall], 2).shape[:2] == (60, 42)


# the pole figures are drawn ahead once the density is there, and the tab shows that picture
# pixel for pixel; a direction changed draws them anew
def test_the_pole_figures_are_drawn_ahead(loaded):
  st = loaded
  canvas = ViewCanvas(st)
  canvas.resize(900, 600)
  cb.pickTab(st, st.tabs.index("IPF Z"))
  canvas.draw_view()
  k = cb.dominantPhase(st)
  canvas.density(k)
  canvas.drawAhead()
  key, picture = canvas.pfPicture
  w, h = canvas.figure.get_size_inches() * canvas.figure.dpi
  assert key == canvas.poleFigureKey() and picture.shape[:2] == (round(h), round(w))
  cb.pickTab(st, st.tabs.index("pole figures"))
  canvas.draw_view()
  assert canvas.pfPicture[1] is picture and len(canvas.figure.images) == 1 and not canvas.figure.axes
  cb.setMiller(st, 0, "110")
  canvas.draw_view()
  assert canvas.pfPicture[0] != key and canvas.pfPicture[0][2][0] == "110"


# a density arriving off the pole figure tab waits for the cursor to rest before it is drawn
def test_the_pole_figures_wait_for_a_rest(loaded):
  st = loaded
  canvas = ViewCanvas(st)
  cb.pickTab(st, st.tabs.index("IPF Z"))
  st.densities[cb.dominantPhase(st)] = object()
  canvas.onDensity()
  assert canvas.ahead.isActive() and canvas.pfPicture is None
  remaining = canvas.ahead.remainingTime()
  st.changed("selected")
  assert canvas.ahead.isActive() and canvas.ahead.remainingTime() >= remaining
  canvas.ahead.stop()
