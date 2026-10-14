import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from mtex.wizard import callbacks as cb
from mtex.wizard.state import WizardState
from mtex.wizard.pictogram import SIDE
from mtex.wizard.views import KeyPanel, MillerBar, SectionBar, ViewCanvas, millerOf
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


# the Miller bar stands on the pole figure tab alone, and a direction typed into it reaches the state
def test_miller_bar_shows_on_the_pole_figures_and_commits(loaded):
  st = loaded
  bar = MillerBar(st)
  cb.pickTab(st, st.tabs.index("IPF Z"))
  assert bar.isHidden()
  cb.pickTab(st, st.tabs.index("pole figures"))
  assert not bar.isHidden()
  bar.fields[0].setText("110")
  bar.fields[0].editingFinished.emit()
  assert st.millers[0] == "110"


# a direction is typed with its minus signs inside, and text that is none falls back to 100
def test_millerOf_reads_the_typed_direction(loaded):
  cs = loaded.ebsd.CSList.frames[1]
  assert np.allclose(millerOf("1-10", cs).hkl, [1, -1, 0])
  assert np.allclose(millerOf("x", cs).hkl, [1, 0, 0])


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
  monkeypatch.setattr(views, "_hemispherePicture", lambda key, CS, px, side, **kw: drawn.append(dict(side)) or real(key, CS, px, side, **kw))
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


def test_millerOf_reads_a_direction_typed_with_blanks(loaded):
  assert np.allclose(millerOf("1 -1 0", loaded.ebsd.CSList.frames[1]).hkl, [1, -1, 0])


# what a canvas shows of a tab: where its opaque cells or pixels are, their colours as bytes,
# and the range of its colour bar
def _shown(st, canvas, name):
  cb.pickTab(st, st.tabs.index(name))
  canvas.draw_view()
  h = canvas.image
  if hasattr(h, "get_paths"):
    h.update_scalarmappable()
    rgba = (np.asarray(h.get_facecolors()) * 255 + 0.5).astype(np.uint8)
    where = np.array([q.vertices[0] for q in h.get_paths()])
  else:
    rgba = h.to_rgba(h.get_array(), bytes=True).reshape(-1, 4)
    where = np.arange(len(rgba))
  opaque = rgba[:, 3] > 0
  bar = None if canvas.mappable is None else (canvas.mappable.norm.vmin, canvas.mappable.norm.vmax)
  return where[opaque], rgba[opaque], bar


# a tab reached from another one recolours the picture, the fast path of drawing it anew: both
# show the same, on a square map, a hexagonal one with its padding, and one drawn as a picture,
# whether the picture was first drawn by a number or by colours
@pytest.mark.parametrize("fname, cap", [("Forsterite_cut.ctf", None), ("ferrite_cut.ang", None), ("ferrite_cut.ang", 0)])
@pytest.mark.parametrize("first", [0, -1])
def test_a_recoloured_tab_shows_what_a_fresh_one_draws(fname, cap, first, monkeypatch):
  from mtex.plotting import mapplot
  application()
  if cap is not None:
    monkeypatch.setattr(mapplot, "CELLS_MAX", cap)
  st = WizardState(FIXTURES)
  cb.openEntry(st, os.path.join(FIXTURES, fname))
  maps = [t for t, k in zip(st.tabs, st.tabKinds) if k in ("phase", "ipf", "property")]
  recoloured = ViewCanvas(st)
  _shown(st, recoloured, maps[first])
  for name in maps:
    again, fresh = _shown(st, recoloured, name), _shown(st, ViewCanvas(st), name)
    assert all(np.array_equal(a, b) for a, b in zip(again[:2], fresh[:2])) and again[2] == fresh[2], name


# ------------------------------------------------------------------ volumes

@pytest.fixture
def volume(volumeFile):
  application()
  st = WizardState(FIXTURES)
  cb.openEntry(st, volumeFile())
  return st


def test_every_view_of_a_volume_draws_its_section(volume):
  st = volume
  canvas = ViewCanvas(st)
  for k, name in enumerate(st.tabs):
    cb.pickTab(st, k)
    canvas.draw_view()
    canvas.draw()
    assert st.status[0] == "loaded", (name, st.status)
  cb.pickTab(st, 0)
  canvas.draw_view()
  assert canvas.drawnFor[0] is st.section


# the colours of the picture a canvas shows, whether it holds numbers with their range or colours
def pictureOf(canvas):
  a = canvas.image.get_array()
  return canvas.image.to_rgba(a, bytes=True) if np.ndim(a) == 2 else np.asarray(a)


@pytest.mark.parametrize("tab", ["phase", "IPF Z", "grainId", "ConfidenceIndex"])
def test_a_moved_section_shows_what_its_own_drawing_shows(volume, tab):
  # the picture after the slider moved is the picture of a canvas that opens on that layer, where
  # it lies and with its range; the layout around a colour bar depends on the bars before it
  st = volume
  cb.pickTab(st, st.tabs.index(tab))
  canvas = ViewCanvas(st)
  canvas.draw_view()
  for axis, index in ((None, 2), (0, None), (0, 4), (1, 0)):
    cb.setSection(st, axis, index)
    canvas.draw_view()
    fresh = ViewCanvas(st)
    fresh.draw_view()
    assert canvas.drawnFor[0] is st.section and np.array_equal(pictureOf(canvas), pictureOf(fresh)), (axis, index)
    assert canvas.image.get_extent() == fresh.image.get_extent()
    assert (canvas.colorbar is None) == (fresh.colorbar is None)
    if canvas.colorbar is not None:
      assert (canvas.mappable.norm.vmin, canvas.mappable.norm.vmax) == (fresh.mappable.norm.vmin, fresh.mappable.norm.vmax)


def test_a_map_stands_where_a_fresh_canvas_draws_it(volume):
  # whatever tabs came before, the map of a tab stands where a canvas opening on it draws it, and a
  # map without a colour bar stands in the middle; a colour bar once anchored it at its right side.
  # The canvas is wide, as in the window, so that the map does not fill it
  st = volume
  canvas = ViewCanvas(st)

  def place(c):
    c.figure.set_size_inches(12, 4)
    c.draw_view()
    c.draw()
    a = c.ax.get_position()
    return a.x0, a.x1

  for name in ("IPF Z", "ConfidenceIndex", "pole figures", "ConfidenceIndex", "IPF Z", "grainId", "phase", "pole figures", "IPF X"):
    cb.pickTab(st, st.tabs.index(name))
    if name == "pole figures":
      canvas.draw_view()
      continue
    fresh = ViewCanvas(st)
    assert place(canvas) == pytest.approx(place(fresh), abs=2e-3), name
    if canvas.colorbar is None:
      assert sum(place(canvas)) / 2 == pytest.approx(0.5, abs=1e-3), name


def test_a_turned_volume_recolours_the_section_drawn(volume):
  # the Euler correction leaves the layer where it is and gives it the colours of the turned orientations
  st = volume
  cb.pickTab(st, st.tabs.index("IPF X"))
  canvas = ViewCanvas(st)
  canvas.draw_view()
  before, image = pictureOf(canvas).copy(), canvas.image
  cb.setEulerConvention(st, (st.eulerConvention + 2) % len(st.conventions))
  canvas.draw_view()
  fresh = ViewCanvas(st)
  fresh.draw_view()
  assert canvas.image is image and not np.array_equal(pictureOf(canvas), before) and np.array_equal(pictureOf(canvas), pictureOf(fresh))


def test_the_section_bar_follows_and_moves_the_section(volume, loaded):
  st = volume
  bar = SectionBar(st)
  assert not bar.isHidden() and bar.slider.maximum() == 2 and bar.slider.value() == 1 and bar.axes.checkedId() == 2
  assert bar.label.text() == "layer 2 of 3 at z = 2 µm"
  bar.slider.setValue(2)
  assert st.sectionIndex == 2 and bar.label.text().startswith("layer 3 of 3")
  # the pole figures are of the whole volume, and the bar gives way on their tab
  cb.pickTab(st, st.tabs.index("pole figures"))
  assert bar.isHidden()
  cb.pickTab(st, st.tabs.index("IPF Z"))
  assert not bar.isHidden()
  bar.axes.button(0).click()
  assert (st.sectionAxis, st.sectionIndex) == (0, 2) and bar.slider.maximum() == 4 and bar.slider.value() == 2
  assert bar.label.text() == "layer 3 of 5 at x = 1 µm"
  # a map has no section to choose
  assert SectionBar(loaded).isHidden()


def test_the_Euler_correction_turns_a_volume_and_its_section(volume):
  from mtex import EBSD3
  st = volume
  before = st.ebsd.rotations.data.copy()
  cb.setEulerConvention(st, (st.eulerConvention + 1) % len(st.conventions))
  d = cb.correctionOf(st)
  assert isinstance(st.ebsd, EBSD3) and st.ebsd.lattice is not None and st.ebsd.shape == (5, 4, 3) and float(d.angle()) > 1
  assert np.allclose(st.ebsd.rotations.data, (d * type(d)(before)).data) and np.allclose(st.ebsd.EulerCorrection.data, d.data)
  assert np.array_equal(st.section.rotations.data, st.ebsd.rotations.data[:, :, 1])
  text = cb.scriptOf(st)
  assert "ebsd = EBSD3.load(fname)\nebsd = ebsd.rotate(eulerCorrection, keepXY=True)" in text
  scope = {}
  exec(text.split("# a look at")[0], scope)
  assert np.allclose(scope["ebsd"].rotations.data, st.ebsd.rotations.data)
