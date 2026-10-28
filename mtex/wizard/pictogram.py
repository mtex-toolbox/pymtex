# the two reference frames of the import wizard, each drawn as the frame it stands for, and
# the chooser that offers the eight as pictures (docs/plans/wizard.md W3)
#
# A convention is a picture, so the button is the picture and the chooser is eight of them:
# a frame is read at a glance where a pictogram string is spelled out. It is the frame a map
# draws in its scale bar box, triadGeometry: the axis that leaves the screen is named in the
# middle of the crossing - no ring, no cross, no line under the letters - and the two that lie
# in the screen are arrows pointing where they point.
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (QAbstractButton, QGridLayout, QMenu, QSizePolicy, QVBoxLayout, QWidget, QWidgetAction)

from ..geometry.plottingconvention import AXIS_CONVENTIONS, conventionNames, plottingConvention
from ..geometry.vector3d import xvector, yvector, zvector
from ..plotting.maps import triadGeometry
from . import callbacks as cb
from .theme import colours

# the drawing in pixels of a name LABEL high, scaled as a whole afterwards: a font taken as a
# fraction of the side and an arrow as another fraction do not keep their proportions together
LABEL, PAD = 18.0, 6.0
SIDE = 104
CHOICE = 96


def conventionArrows(pC):
  """where each reference axis lies on the screen under a convention: its components along
  east, north and out of the screen, so that the window draws the frame rather than
  spelling it. An axis with a component near one out of the screen leaves it."""
  e, n, o = pC.east, pC.north, pC.outOfScreen
  return [[float(ax.dot(d)) for d in (e, n, o)] for ax in (xvector, yvector, zvector)]


# ------------------------------------------------------------------ the drawing

def drawPictogram(p, rect, pC, names=("x", "y", "z"), ink=None, caption=""):
  """the frame as the map draws it in its corner, triadGeometry. The picture is laid out from
  its crossing and then centred on its ink and not on that crossing, which sits wherever the
  two arrows leave it: 'x' and 'RD' are different widths, so an origin centred picture is not."""
  axes = conventionArrows(pC)
  bold = QFont(p.font())
  bold.setPixelSize(int(LABEL))
  bold.setBold(True)
  light = QFont(bold)
  light.setBold(False)
  # the geometry runs upwards, the painter downwards
  glyphs = {s: _glyphs(bold, s) for s in names if s}
  arrows, texts, bbox = triadGeometry(axes, list(names), LABEL / 0.8, box=lambda s: _upwards(glyphs[s]))
  lo, hi = [bbox[0], -bbox[3]], [bbox[1], -bbox[2]]
  # the caption stands in the quadrant the arrows leave free, read after them and not before
  sx = sum(a[0] for a in axes if abs(a[2]) <= 0.9)
  sy = sum(a[1] for a in axes if abs(a[2]) <= 0.9)
  cx, cy = (1 if sx >= 0 else -1) * 2 * LABEL, -(1 if sy >= 0 else -1) * 2 * LABEL
  if caption:
    capGlyphs = _glyphs(light, caption)
    c, ox, oy, _ = capGlyphs
    lo = [min(lo[0], cx - ox + c.left()), min(lo[1], cy - oy + c.top())]
    hi = [max(hi[0], cx - ox + c.right()), max(hi[1], cy - oy + c.bottom())]
  scale = min((rect.width() - 2 * PAD) / max(hi[0] - lo[0], 1e-9), (rect.height() - 2 * PAD) / max(hi[1] - lo[1], 1e-9))
  p.save()
  p.setRenderHint(QPainter.Antialiasing)
  p.translate(rect.center().x() - scale * (lo[0] + hi[0]) / 2, rect.center().y() - scale * (lo[1] + hi[1]) / 2)
  p.scale(scale, scale)
  p.setPen(Qt.NoPen)
  p.setBrush(QColor(ink or colours()["mapInk"]))
  for a in arrows:
    p.drawPolygon(QPolygonF([QPointF(x, -y) for x, y in a]))
  for q, s in texts:
    _centred(p, q[0], -q[1], glyphs[s])
  if caption:
    p.setOpacity(0.6)
    _centred(p, cx, cy, capGlyphs)
    p.setOpacity(1.0)
  p.restore()


# a text as its outline about its origin on the baseline, the ink of that outline, and the
# point it is centred by: the middle of its ink across, and of its ink above the baseline up
# and down, so that a descender reaches into the gap below without lifting the letter off the
# shaft. The outline is what is drawn, so the gaps measured on it are the gaps seen
def _glyphs(font, text):
  path = QPainterPath()
  path.addText(0, 0, font, text)
  ink = path.boundingRect()
  return ink, ink.center().x(), (ink.top() + min(ink.bottom(), 0.0)) / 2, path


# the ink of a text about its centre as triadGeometry takes it, (left, right, bottom, top) upwards
def _upwards(g):
  ink, ox, oy, _ = g
  return ink.left() - ox, ink.right() - ox, oy - ink.bottom(), oy - ink.top()


# a text drawn with its centre at (x, y), in the brush
def _centred(p, x, y, g):
  ink, ox, oy, path = g
  p.drawPath(path.translated(x - ox, y - oy))


# ------------------------------------------------------------------ the chooser

class PictogramButton(QAbstractButton):
  """one of the eight, as a picture that can be picked"""

  def __init__(self, pC, names, ink, side=CHOICE):
    super().__init__()
    self.pC = pC
    self.names = names
    self.ink = ink
    self.setCheckable(True)
    self.setFixedSize(side, side)
    self.setCursor(Qt.PointingHandCursor)

  def paintEvent(self, event):
    """draws the pictogram on a ground marking it checked or under the pointer"""
    p = QPainter(self)
    c = colours()
    ground = c["checked"] if self.isChecked() else (c["hover"] if self.underMouse() else c["ground"])
    p.fillRect(self.rect(), QColor(ground))
    p.setPen(QPen(QColor(c["hoverFrame"] if self.isChecked() or self.underMouse() else c["frame"]), 1))
    p.drawRect(self.rect().adjusted(0, 0, -1, -1))
    drawPictogram(p, QRectF(self.rect()).adjusted(3, 3, -3, -3), self.pC, self.names, self.ink)
    p.end()

  def enterEvent(self, event):
    """repaints as the pointer enters"""
    self.update()

  def leaveEvent(self, event):
    """repaints as the pointer leaves"""
    self.update()


class ConventionChooser(QMenu):
  """the eight as pictures, four to a row, the one in force marked"""

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
    """the convention k taken as the one picked, and the menu closed"""
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
    """the index of the convention in force for the frame of the button, map or Euler"""
    return self.st.mapConvention if self.which == "map" else self.st.eulerConvention

  @property
  def names(self):
    """the names of the axes of the frame of the button"""
    return self.st.mapAxesNames if self.which == "map" else self.st.eulerAxesNames

  @property
  def ink(self):
    """the colour the frame is drawn in: the map in ink, the Euler frame in red"""
    # the two frames are told apart by colour: the map in ink, the Euler frame in the red
    # the pole figures carry its axes in
    return colours()["eulerInk" if self.which == "euler" else "mapInk"]

  @property
  def caption(self):
    """the caption of the button, 'map' or 'Euler'"""
    return "map" if self.which == "map" else "Euler"

  def enterEvent(self, event):
    """the ground lightened as the pointer enters"""
    self.hover = True
    self.update()

  def leaveEvent(self, event):
    """the ground restored as the pointer leaves"""
    self.hover = False
    self.update()

  def mousePressEvent(self, event):
    """opens the chooser at the pointer"""
    self.openChooser(event.globalPosition().toPoint())

  def openChooser(self, at=None):
    """the chooser of the eight conventions opened at a point, below the button by default; the one
    picked is taken, the menu returned"""
    menu = ConventionChooser(self, self.index, self.names, self.ink)
    menu.exec(at if at is not None else self.mapToGlobal(self.rect().bottomLeft()))
    if menu.picked is not None:
      self.choose(menu.picked)
    return menu

  def choose(self, k):
    """the convention k set for the frame of the button in the state"""
    if self.which == "map":
      cb.setMapConvention(self.st, k)
    else:
      cb.setEulerConvention(self.st, k)
    self.update()

  def paintEvent(self, event):
    """draws the pictogram of the convention in force, with its tooltip"""
    p = QPainter(self)
    c = colours()
    p.fillRect(self.rect(), QColor(c["hover"] if self.hover else "transparent"))
    p.setPen(QPen(QColor(c["hoverFrame"] if self.hover else c["frame"]), 1))
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
    """repaints both frames after the state changed"""
    self.map.update()
    self.euler.update()


__all__ = ["CoordinatePanel", "ConventionButton", "ConventionChooser", "PictogramButton", "drawPictogram",
           "conventionArrows"]
