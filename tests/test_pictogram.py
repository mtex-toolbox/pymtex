import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QRectF
from PySide6.QtGui import QImage, QPainter

from mtex import plottingConvention
from mtex.geometry.plottingconvention import AXIS_CONVENTIONS
from mtex.wizard import callbacks as cb
from mtex.wizard.pictogram import (CHOICE, SIDE, ConventionButton, ConventionChooser, CoordinatePanel, PictogramButton,
                                   conventionArrows, drawPictogram)
from mtex.wizard.state import WizardState
from mtex.wizard.window import application

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "ebsd")


@pytest.fixture
def loaded():
  application()
  st = WizardState(FIXTURES)
  cb.openEntry(st, os.path.join(FIXTURES, "Forsterite_cut.ctf"))
  return st


# where each reference axis lies on the screen: its components along east, north and out
def test_convention_arrows():
  a = conventionArrows(plottingConvention("y↑→x"))
  assert np.allclose(a, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], atol=1e-9)
  a = conventionArrows(plottingConvention("x←↓y"))
  assert np.allclose(a[0], [-1, 0, 0], atol=1e-9) and np.allclose(a[1], [0, -1, 0], atol=1e-9)
  # the axis out of the screen is the one with the third component near one
  for s in AXIS_CONVENTIONS:
    a = np.asarray(conventionArrows(plottingConvention(s)))
    assert np.sum(np.abs(a[:, 2]) > 0.9) == 1
    assert np.allclose(a @ a.T, np.eye(3), atol=1e-9)


# every convention draws ink, and the drawing is centred on that ink rather than on its
# crossing, which sits wherever the two arrows leave it
def test_pictogram_draws_centred():
  application()
  for s in AXIS_CONVENTIONS:
    for names in (("x", "y", "z"), ("X1", "Y1", "Z1"), ("RD", "TD", "ND")):
      ink = _ink(plottingConvention(s), names, "map")
      assert len(ink) > 200, (s, names)
      box = [(ink[:, k].min() + ink[:, k].max()) / 2 for k in (0, 1)]
      assert abs(box[0] - SIDE / 2) < 8 and abs(box[1] - SIDE / 2) < 8, (s, names, box)
      # scaled to fit: the ink fills the square in the direction that binds
      assert max(np.ptp(ink[:, 0]), np.ptp(ink[:, 1])) > 0.8 * SIDE, (s, names)
  # the caption is drawn beside the frame, so it adds ink and moves none
  assert len(_ink(plottingConvention("y↑→x"), ("x", "y", "z"), "map")) > len(_ink(plottingConvention("y↑→x"), ("x", "y", "z")))


# every name stands as far off its arrow as every other, the one at the head and the one in
# the crossing, measured on the ink: a lowercase letter below a head no farther than beside it
def test_pictogram_spacesTheNamesEquallyOffTheArrows():
  application()
  n = 600
  for s in AXIS_CONVENTIONS:
    for names in (("x", "y", "z"), ("X1", "Y1", "Z1"), ("RD", "TD", "ND")):
      gaps = _gaps(_inked(plottingConvention(s), names, n))
      assert len(gaps) == 4 and max(gaps) - min(gaps) <= 0.03 * max(gaps) + 1, (s, names, gaps)


# the distances from either end of each arrow to the ink box of the nearest letter beyond it
# that the arrow's head, widened across, would meet
def _gaps(ink):
  from scipy import ndimage
  lab, _ = ndimage.label(ink)
  boxes = [(sl[0].start, sl[0].stop, sl[1].start, sl[1].stop) for sl in ndimage.find_objects(lab)]
  # an arrow is a long thin piece, a figure one only thin
  isArrow = [max(r1 - r0, c1 - c0) >= max(1.8 * min(r1 - r0, c1 - c0), ink.shape[0] / 5) for r0, r1, c0, c1 in boxes]
  out = []
  for (r0, r1, c0, c1), arrow in zip(boxes, isArrow):
    if not arrow:
      continue
    across, along = ((r0, r1), (c0, c1)) if c1 - c0 > r1 - r0 else ((c0, c1), (r0, r1))
    k = (0, 2) if c1 - c0 > r1 - r0 else (2, 0)
    letters = [(b[k[1]], b[k[1] + 1]) for b, a in zip(boxes, isArrow)
               if not a and b[k[0]] < across[1] and b[k[0] + 1] > across[0]]
    out.append(min(along[0] - e for s0, e in letters if e <= along[0]))
    out.append(min(s0 - along[1] for s0, e in letters if s0 >= along[1]))
  return out


# a pictogram drawn on a square of side n, as a mask of its dark pixels
def _inked(pC, names, n):
  img = QImage(n, n, QImage.Format_ARGB32)
  img.fill(0xFFFFFFFF)
  p = QPainter(img)
  drawPictogram(p, QRectF(0, 0, n, n), pC, names, "#000")
  p.end()
  return np.frombuffer(img.constBits(), dtype=np.uint8).reshape(n, n, 4)[..., :3].sum(axis=2) < 3 * 128


# the pixels a pictogram inks, as row and column pairs
def _ink(pC, names, caption=""):
  img = QImage(SIDE, SIDE, QImage.Format_ARGB32)
  img.fill(0xFFFFFFFF)
  p = QPainter(img)
  drawPictogram(p, QRectF(0, 0, SIDE, SIDE), pC, names, "#333", caption)
  p.end()
  a = np.frombuffer(img.constBits(), dtype=np.uint8).reshape(SIDE, SIDE, 4)[..., :3]
  return np.argwhere(a.sum(axis=2) < 3 * 254)


# the names come from the frame, so an Oxford file's Euler frame reads X1, Y1, Z1
def test_button_and_chooser(loaded):
  st = loaded
  panel = CoordinatePanel(st)
  assert panel.map.names == ("x", "y", "z") and panel.euler.names == ("X1", "Y1", "Z1")
  assert panel.euler.ink != panel.map.ink and panel.map.caption == "map"
  assert panel.map.size().width() == SIDE and panel.map.size().height() == SIDE
  panel.map.update()
  assert not panel.map.grab().isNull()
  # the chooser holds the eight as pictures, the one in force marked
  chooser = ConventionChooser(panel.euler, st.eulerConvention, panel.euler.names, panel.euler.ink)
  assert len(chooser.buttons) == 8 and sum(b.isChecked() for b in chooser.buttons) == 1
  assert chooser.buttons[st.eulerConvention].isChecked()
  assert chooser.buttons[0].size().width() == CHOICE and not chooser.buttons[0].grab().isNull()
  assert "↑" in chooser.buttons[0].toolTip() or "→" in chooser.buttons[0].toolTip()
  # picking one closes the chooser and reaches the state
  before = st.eulerConvention
  other = (before + 3) % 8
  chooser.buttons[other].click()
  assert chooser.picked == other
  panel.euler.choose(chooser.picked)
  assert st.eulerConvention == other and float(st.ebsd.EulerCorrection.angle(cb.correctionOf(st))) < 1e-12
  panel.euler.choose(before)
