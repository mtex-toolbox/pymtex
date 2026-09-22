# how a reference frame is laid out on screen (MATLAB plotting/plottingConvention.m)
#
# A value type holding one rotation: screen coordinates to reference coordinates, so that
# east = rot * x, north = rot * y, outOfScreen = rot * z. Frames carry one as their
# default; data never does (MATLAB ADR 0003).
#
#   plottingConvention(outOfScreen, east)
#   plottingConvention('y↑→x')      each axis is followed or preceded by the screen
#                                   direction it points to; 'y^->x' in ASCII
#   plottingConvention.ij           x east, y south, z into the screen - SEM images
#   plottingConvention.default3D    a tilted view for three dimensional plots
#   plottingConvention.edax(2)      the EDAX/TSL frames 1 to 4
#   plottingConvention.default      the session default, which makeDefault sets
from __future__ import annotations

import numpy as np

from .quaternion import rotation
from .session import SESSION, getMTEXpref
from .vector3d import classproperty, vector3d, xvector, yvector, zvector

ARROWS_UTF8 = ("←", "→", "↑", "↓", "⊗", "⊙")
ARROWS_ASCII = ("<-", "->", "^", "v", "(x)", "(.)")


class plottingConvention:

  def __init__(self, outOfScreen=None, east=None):
    if isinstance(outOfScreen, plottingConvention):
      self.rot = outOfScreen.rot
    elif isinstance(outOfScreen, str):
      self.rot = _parseConvention(outOfScreen)
    elif isinstance(outOfScreen, rotation):
      self.rot = outOfScreen
    elif outOfScreen is None:
      self.rot = rotation.id
    elif east is None:
      self.rot = rotation.map(zvector, outOfScreen)
    else:
      self.rot = rotation.map(zvector, outOfScreen, xvector, east)

  # ------------------------------------------------------------------ the named conventions

  @classproperty
  def ij(cls):
    return cls(rotation(0.0, 1.0, 0.0, 0.0))

  @classproperty
  def default3D(cls):
    return cls(vector3d(-10, -5, 2), vector3d(1, -2, 0))

  @classmethod
  def edax(cls, setting=2):
    if not 1 <= setting <= 4:
      raise ValueError("EDAX settings are 1 to 4")
    return cls(zvector, (yvector, -yvector, xvector, -xvector)[setting - 1])

  # the session default is the convention of the default specimen frame, so that everything
  # holding that frame follows a change of it
  @classproperty
  def default(cls):
    return _sessionDefault(_defaultFrame().how2plot)

  def makeDefault(self):
    _defaultFrame().how2plot = plottingConvention(self)

  # ------------------------------------------------------------------ the screen directions

  @property
  def east(self):
    return self.rot * xvector

  @property
  def west(self):
    return -self.east

  @property
  def north(self):
    return self.rot * yvector

  @property
  def south(self):
    return -self.north

  @property
  def outOfScreen(self):
    return self.rot * zvector

  @property
  def intoScreen(self):
    return -self.outOfScreen

  # moving one screen direction is the least rotation that does it
  @east.setter
  def east(self, v):
    self.rot = rotation.map(self.east, v) * self.rot

  @west.setter
  def west(self, v):
    self.east = -v

  @north.setter
  def north(self, v):
    self.rot = rotation.map(self.north, v) * self.rot

  @south.setter
  def south(self, v):
    self.north = -v

  @outOfScreen.setter
  def outOfScreen(self, v):
    self.rot = rotation.map(self.outOfScreen, v) * self.rot

  @intoScreen.setter
  def intoScreen(self, v):
    self.outOfScreen = -v

  # two conventions are the same when they lay the frame out the same way
  def __eq__(self, other):
    return isinstance(other, plottingConvention) and bool(self.rot.angle(other.rot) < 1e-6)

  def __ne__(self, other):
    return not self == other

  __hash__ = None

  # ------------------------------------------------------------------ pictograms

  # 'y↑→x' when both screen axes are reference axes, the three names run together otherwise
  def char(self, utf8=None, names=("x", "y", "z")):
    iN, up = _axisOf(self.north)
    iE, right = _axisOf(self.east)
    if iN is None or iE is None:
      return "".join(names)
    return _arrowPictogram(names[iN], up, names[iE], right, utf8)

  @property
  def isPictogram(self):
    return self.char() != "xyz"

  def __str__(self):
    return self.char()

  def __repr__(self):
    if self.isPictogram:
      return f"plottingConvention ({self.char()})"
    fmt = lambda v: "(" + ",".join(f"{x:.3g}" for x in v.data) + ")"
    return f"plottingConvention\n  outOfScreen: {fmt(self.outOfScreen)}\n  north: {fmt(self.north)}\n  east: {fmt(self.east)}"


# which reference axis, and with which sign, a screen direction is
def _axisOf(v):
  for k, e in enumerate((xvector, yvector, zvector)):
    d = v.dot(e)
    if abs(d) > 1 - 1e-6:
      return k, bool(d > 0)
  return None, None


def _arrows(utf8):
  utf8 = getMTEXpref("UTF8Output") if utf8 is None else utf8
  return ARROWS_UTF8 if utf8 else ARROWS_ASCII


# the axis running north with its arrow up or down, then the one running east
def _arrowPictogram(nameN, up, nameE, right, utf8=None):
  a = _arrows(utf8)
  return nameN + a[2 if up else 3] + a[1] + nameE if right else nameE + a[0] + a[2 if up else 3] + nameN


# 'y↑→x' -> the rotation, str2rot
def _parseConvention(s):
  for old, new in (("(x)", "⊗"), ("(.)", "⊙"), ("->", "→"), ("<-", "←"), ("^", "↑"), ("v", "↓"), ("V", "↓")):
    s = s.replace(old, new)
  text = "".join(ch for ch in s if not ch.isspace())
  refDir = (xvector, yvector, zvector)
  screenDir = (-xvector, xvector, yvector, -yvector, -zvector, zvector)
  tokens = []
  sign = 1
  for ch in text:
    if ch == "-":
      sign = -sign
      continue
    if ch in "xyz":
      tokens.append((True, sign * refDir["xyz".index(ch)]))
      sign = 1
      continue
    if ch not in "←→↑↓⊗⊙":
      raise ValueError(f"cannot interpret '{s}' as a plotting convention - unknown symbol '{ch}'; use x, y, z and the arrows ←→↑↓⊗⊙ (or <- -> ^ v (x) (.)), e.g. 'y↑→x'")
    tokens.append((False, screenDir["←→↑↓⊗⊙".index(ch)]))
  pairs = []
  open_ = None
  for t in tokens:
    if open_ is None:
      open_ = t
    elif open_[0] != t[0]:
      screen, ref = (t[1], open_[1]) if open_[0] else (open_[1], t[1])
      pairs.append((screen, ref))
      open_ = None
    else:
      open_ = False
      break
  if open_ is not None or not pairs:
    raise ValueError(f"cannot interpret '{s}' as a plotting convention - every axis needs an arrow next to it, e.g. 'y↑→x'")
  if len(pairs) > 2:
    raise ValueError(f"cannot interpret '{s}' as a plotting convention - at most two axes may be given")
  if len(pairs) == 1:
    return rotation.map(pairs[0][0], pairs[0][1])
  if abs(pairs[0][0].dot(pairs[1][0])) > 1e-6 or abs(pairs[0][1].dot(pairs[1][1])) > 1e-6:
    raise ValueError(f"cannot interpret '{s}' as a plotting convention - the two axes and the two screen directions have to be perpendicular")
  return rotation.map(pairs[0][0], pairs[0][1], pairs[1][0], pairs[1][1])


# ------------------------------------------------------------------ the axis aligned conventions

# the eight conventions with both screen axes along reference axes, the map and the Euler
# coordinates the import wizard offers; each vendor's Euler correction is one of them
AXIS_CONVENTIONS = ("y↑→x", "y↓→x", "x↑→y", "x↓→y", "x←↑y", "x←↓y", "y←↑x", "y←↓x")


# the eight written in the axis names of a frame, 'Y1↑→X1' for an Oxford file
def conventionNames(names=("x", "y", "z")):
  return [plottingConvention(s).char(utf8=True, names=tuple(names)) for s in AXIS_CONVENTIONS]


# the entry of AXIS_CONVENTIONS nearest a rotation, and how far it is from it in radians
def closestConvention(rot):
  rot = rot.rot if isinstance(rot, plottingConvention) else rot
  d = np.array([float(rot.angle(plottingConvention(s).rot)) for s in AXIS_CONVENTIONS])
  k = int(np.argmin(d))
  return k, d[k]


# ------------------------------------------------------------------ the session default

# what plottingConvention.default returns: the session default, and a call on it with a
# convention sets it, plottingConvention.default('y↑→x'), as MATLAB reads it
class _sessionDefault(plottingConvention):

  def __call__(self, outOfScreen, east=None):
    plottingConvention(outOfScreen, east).makeDefault()

  def __repr__(self):
    return plottingConvention.__repr__(plottingConvention(self))


# the frame that supplies the session default convention, minted by frames.py on first use
def _defaultFrame():
  if SESSION.defaultFrame is None:
    SESSION.defaultFrame = SESSION.mintDefaultFrame()
  if SESSION.defaultFrame.how2plot is None:
    SESSION.defaultFrame.how2plot = plottingConvention.ij
  return SESSION.defaultFrame


def _setDefault(**directions):
  pC = plottingConvention(plottingConvention.default)
  for name, v in directions.items():
    setattr(pC, name, v)
  pC.makeDefault()


def plotx2east():
  _setDefault(east=xvector)


def plotx2north():
  _setDefault(north=xvector)


def plotx2west():
  _setDefault(west=xvector)


def plotx2south():
  _setDefault(south=xvector)


def plotzOutOfPlane():
  _setDefault(outOfScreen=zvector)


def plotzIntoPlane():
  _setDefault(intoScreen=zvector)


__all__ = ["AXIS_CONVENTIONS", "conventionNames", "closestConvention", "plottingConvention", "plotx2east", "plotx2north", "plotx2west", "plotx2south", "plotzOutOfPlane", "plotzIntoPlane"]
