# the two reference frames of the import wizard, each drawn as the frame it stands for, and
# the chooser that offers the eight as pictures (docs/plans/wizard.md W3)
#
# A convention is a picture, so the button is the picture and the chooser is eight of them:
# a frame is read at a glance where a pictogram string is spelled out. The axis that leaves
# the screen is named in the middle of the crossing - no ring, no cross, no line under the
# letters - and the two that lie in the screen are arrows pointing where they point.
from __future__ import annotations

from math import atan2, cos, sin

import numpy as np

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (QAbstractButton, QGridLayout, QMenu, QSizePolicy, QVBoxLayout, QWidget, QWidgetAction)

from ..geometry.plottingconvention import AXIS_CONVENTIONS, conventionNames, plottingConvention
from ..geometry.vector3d import xvector, yvector, zvector
from . import callbacks as cb

# the drawing in units of its own, scaled as a whole afterwards: a font taken as a fraction
# of the side and an arrow as another fraction do not keep their proportions together
LEN, HEAD, LABEL, GAP, WEIGHT, PAD = 46.0, 14.0, 18.0, 8.0, 4.5, 3.0
SIDE = 104
CHOICE = 96
FRAME = "#c8c8c8"
MAP_INK = "#333333"
EULER_INK = "#c62828"


# where each reference axis lies on the screen under a convention: its components along
# east, north and out of the screen, so that the window draws the frame rather than
# spelling it. An axis with a component near one out of the screen leaves it.
def conventionArrows(pC):
  e, n, o = pC.east, pC.north, pC.outOfScreen
  return [[float(ax.dot(d)) for d in (e, n, o)] for ax in (xvector, yvector, zvector)]


# ------------------------------------------------------------------ the drawing

# the frame as the map draws it in its corner. The picture is laid out from its crossing
# and then centred on its ink and not on that crossing, which sits wherever the two arrows
# leave it: 'x' and 'RD' are different widths, so an origin centred picture is not.
def drawPictogram(p, rect, pC, names=("x", "y", "z"), ink=MAP_INK, caption=""):
  axes = conventionArrows(pC)
  bold = QFont(p.font())
  bold.setPixelSize(int(LABEL))
  bold.setBold(True)
  light = QFont(bold)
  light.setBold(False)
  fmBold, fmLight = QFontMetricsF(bold), QFontMetricsF(light)
  # which axis leaves the screen, and which way the other two lean
  out, sx, sy = None, 0.0, 0.0
  for k, a in enumerate(axes):
    if abs(a[2]) > 0.9:
      out = k
      continue
    sx, sy = sx + a[0], sy + a[1]
  outName = names[out] if out is not None else ""
  # every arrow, with the place its name stands off the head along it; screen y grows down
  arrows = []
  for j, a in enumerate(axes):
    if j == out:
      continue
    dx, dy = a[0], -a[1]
    tx, ty = dx * LEN, dy * LEN
    w = fmBold.horizontalAdvance(names[j])
    arrows.append((dx, dy, tx, ty, names[j], tx + dx * (GAP + w / 2), ty + dy * (GAP + LABEL / 2), w))
  # the shafts start clear of whatever the name in the crossing needs
  hub = max(fmBold.horizontalAdvance(outName) / 2 if outName else 0.0, LABEL / 2) + 4
  # the caption stands in the quadrant the arrows leave free, read after them and not before
  capWidth = fmLight.horizontalAdvance(caption) if caption else 0.0
  cx = (1 if sx >= 0 else -1) * LEN * 0.78
  cy = -(1 if sy >= 0 else -1) * LEN * 0.78
  lo, hi = [1e9, 1e9], [-1e9, -1e9]

  def see(x, y, hw, hh):
    lo[0], hi[0] = min(lo[0], x - hw), max(hi[0], x + hw)
    lo[1], hi[1] = min(lo[1], y - hh), max(hi[1], y + hh)

  see(0, 0, hub, LABEL / 2)
  for dx, dy, tx, ty, name, lx, ly, w in arrows:
    see(tx, ty, HEAD / 2, HEAD / 2)
    see(lx, ly, w / 2, LABEL / 2)
  if caption:
    see(cx, cy, capWidth / 2, LABEL / 2)
  scale = min((rect.width() - 2 * PAD) / max(hi[0] - lo[0], 1e-9), (rect.height() - 2 * PAD) / max(hi[1] - lo[1], 1e-9))
  p.save()
  p.setRenderHint(QPainter.Antialiasing)
  p.translate(rect.center().x() - scale * (lo[0] + hi[0]) / 2, rect.center().y() - scale * (lo[1] + hi[1]) / 2)
  p.scale(scale, scale)
  colour = QColor(ink)
  p.setPen(QPen(colour, WEIGHT, Qt.SolidLine, Qt.FlatCap, Qt.RoundJoin))
  p.setBrush(colour)
  p.setFont(bold)
  for dx, dy, tx, ty, name, lx, ly, w in arrows:
    p.drawLine(QPointF(dx * hub, dy * hub), QPointF(tx - 0.9 * HEAD * dx, ty - 0.9 * HEAD * dy))
    a = atan2(dy, dx)
    p.setPen(Qt.NoPen)
    p.drawPolygon(QPolygonF([QPointF(tx, ty),
                             QPointF(tx - HEAD * cos(a - 0.36), ty - HEAD * sin(a - 0.36)),
                             QPointF(tx - HEAD * cos(a + 0.36), ty - HEAD * sin(a + 0.36))]))
    p.setPen(QPen(colour, WEIGHT, Qt.SolidLine, Qt.FlatCap, Qt.RoundJoin))
    _centred(p, lx, ly, name)
  if outName:
    _centred(p, 0, 0, outName)
  if caption:
    p.setFont(light)
    p.setOpacity(0.6)
    _centred(p, cx, cy, caption)
    p.setOpacity(1.0)
  p.restore()


def _centred(p, x, y, text):
  p.drawText(QRectF(x - 4 * LABEL, y - LABEL, 8 * LABEL, 2 * LABEL), Qt.AlignCenter, text)


# ------------------------------------------------------------------ the chooser

# one of the eight, as a picture that can be picked
class PictogramButton(QAbstractButton):

  def __init__(self, pC, names, ink, side=CHOICE):
    super().__init__()
    self.pC = pC
    self.names = names
    self.ink = ink
    self.setCheckable(True)
    self.setFixedSize(side, side)
    self.setCursor(Qt.PointingHandCursor)

  def paintEvent(self, event):
    p = QPainter(self)
    ground = "#e8eef6" if self.isChecked() else ("#f2f2f2" if self.underMouse() else "white")
    p.fillRect(self.rect(), QColor(ground))
    p.setPen(QPen(QColor("#8a8a8a" if self.isChecked() or self.underMouse() else FRAME), 1))
    p.drawRect(self.rect().adjusted(0, 0, -1, -1))
    drawPictogram(p, QRectF(self.rect()).adjusted(3, 3, -3, -3), self.pC, self.names, self.ink)
    p.end()

  def enterEvent(self, event):
    self.update()

  def leaveEvent(self, event):
    self.update()


# the eight as pictures, four to a row, the one in force marked
class ConventionChooser(QMenu):

  def __init__(self, parent, index, names, ink):
    super().__init__(parent)
    self.picked = None
    body = QWidget()
    grid = QGridLayout(body)
    grid.setContentsMargins(4, 4, 4, 4)
    grid.setSpacing(3)
    self.buttons = []
    for k, s in enumerate(AXIS_CONVENTIONS):
      b = PictogramButton(plottingConvention(s), names, ink)
      b.setChecked(k == index)
      b.setToolTip(conventionNames(names)[k])
      b.clicked.connect(lambda _=False, k=k: self.pick(k))
      grid.addWidget(b, k // 4, k % 4)
      self.buttons.append(b)
    action = QWidgetAction(self)
    action.setDefaultWidget(body)
    self.addAction(action)

  def pick(self, k):
    self.picked = k
    self.close()


# ------------------------------------------------------------------ the two frames

class ConventionButton(QWidget):
  """one reference frame, as a square drawing the frame it stands for

  A framed panel and not a raised button: the rest of the window is panels behind a grey
  frame, and a raised button in the middle of them reads as a control strip that wandered
  in. It still says it can be clicked, the ground lightening under the pointer.
  """
  chosen = Signal(int)

  def __init__(self, st, which):
    super().__init__()
    self.st = st
    self.which = which
    self.setFixedSize(SIDE, SIDE)
    self.setCursor(Qt.PointingHandCursor)
    self.hover = False

  @property
  def index(self):
    return self.st.mapConvention if self.which == "map" else self.st.eulerConvention

  @property
  def names(self):
    return self.st.mapAxesNames if self.which == "map" else self.st.eulerAxesNames

  @property
  def ink(self):
    # the two frames are told apart by colour: the map in ink, the Euler frame in the red
    # the pole figures carry its axes in
    return EULER_INK if self.which == "euler" else MAP_INK

  @property
  def caption(self):
    return "map" if self.which == "map" else "Euler"

  def enterEvent(self, event):
    self.hover = True
    self.update()

  def leaveEvent(self, event):
    self.hover = False
    self.update()

  def mousePressEvent(self, event):
    self.openChooser(event.globalPosition().toPoint())

  def openChooser(self, at=None):
    menu = ConventionChooser(self, self.index, self.names, self.ink)
    menu.exec(at if at is not None else self.mapToGlobal(self.rect().bottomLeft()))
    if menu.picked is not None:
      self.choose(menu.picked)
    return menu

  def choose(self, k):
    if self.which == "map":
      cb.setMapConvention(self.st, k)
    else:
      cb.setEulerConvention(self.st, k)
    self.update()

  def paintEvent(self, event):
    p = QPainter(self)
    p.fillRect(self.rect(), QColor("#f2f2f2" if self.hover else "transparent"))
    p.setPen(QPen(QColor("#8a8a8a" if self.hover else FRAME), 1))
    p.drawRect(self.rect().adjusted(0, 0, -1, -1))
    pC = plottingConvention(AXIS_CONVENTIONS[self.index])
    self.setToolTip(f"{self.caption} coordinates: {conventionNames(self.names)[self.index]}")
    drawPictogram(p, QRectF(self.rect()).adjusted(2, 2, -2, -2), pC, self.names, self.ink, self.caption)
    p.end()


class CoordinatePanel(QWidget):
  """the two reference frames, stacked: the map above, the Euler coordinates below"""

  def __init__(self, st):
    super().__init__()
    lay = QVBoxLayout(self)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(3)
    self.map = ConventionButton(st, "map")
    self.euler = ConventionButton(st, "euler")
    lay.addWidget(self.map)
    lay.addWidget(self.euler)
    lay.addStretch(1)
    self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
    for name in ("mapConvention", "eulerConvention", "info", "ebsd"):
      st.on(name, self.refresh)

  def refresh(self):
    self.map.update()
    self.euler.update()


__all__ = ["CoordinatePanel", "ConventionButton", "ConventionChooser", "PictogramButton", "drawPictogram",
           "conventionArrows"]
