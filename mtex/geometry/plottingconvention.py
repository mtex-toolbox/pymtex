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

from typing import TYPE_CHECKING, Any
import numpy as np

from .quaternion import rotation
from .session import SESSION, getMTEXpref
from .vector3d import classproperty, vector3d, xvector, yvector, zvector
from .display import aligned

if TYPE_CHECKING:
  from .vocabulary import ConventionName

ARROWS_UTF8 = ("←", "→", "↑", "↓", "⊗", "⊙")
ARROWS_ASCII = ("<-", "->", "^", "v", "(x)", "(.)")


class plottingConvention:
  """how a reference frame is laid out on screen: one rotation taking the screen axes to the
  reference axes, east = rot * x, north = rot * y, outOfScreen = rot * z

    pC = plottingConvention(outOfScreen, east)
    pC = plottingConvention('y↑→x')      each axis next to the screen direction it points to
    pC = plottingConvention.ij          x east, y south, z into the screen"""

  def __init__(self, outOfScreen: ConventionName | Any = None, east=None):
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
    """the convention of images: x east, y south, z into the screen"""
    return cls(rotation(0.0, 1.0, 0.0, 0.0))

  @classproperty
  def default3D(cls):
    """a tilted view for three dimensional plots"""
    return cls(vector3d(-10, -5, 2), vector3d(1, -2, 0))

  @classmethod
  def edax(cls, setting=2):
    """the EDAX/TSL frame of the given setting, 1 to 4, z out of the screen"""
    if not 1 <= setting <= 4:
      raise ValueError("EDAX settings are 1 to 4")
    return cls(zvector, (yvector, -yvector, xvector, -xvector)[setting - 1])

  @classproperty
  def default(cls):
    """the session default is the convention of the default specimen frame, so that everything
    holding that frame follows a change of it; plottingConvention.default('y↑→x') sets it,
    plottingConvention.default(north=xvector) changes one direction and keeps the others
    (MATLAB's plotx2east, plotx2north, plotx2west, plotx2south, plotzOutOfPlane and plotzIntoPlane)"""
    return _sessionDefault(_defaultFrame().how2plot)

  def makeDefault(self, save=False):
    """makes this convention the session default, that of the default specimen frame; with
    save=True also the default of every later session, stored in the user's settings file

      plottingConvention('y↑→x').makeDefault()
      plottingConvention.default('y↑→x', save=True)"""
    _defaultFrame().how2plot = plottingConvention(self)
    if save:
      from .settings import conventionSetting, saveSetting
      saveSetting("plottingConvention", conventionSetting(self))

  # ------------------------------------------------------------------ the screen directions

  @property
  def east(self):
    """the reference direction pointing east on screen"""
    return self.rot * xvector

  @property
  def west(self):
    """the reference direction pointing west on screen"""
    return -self.east

  @property
  def north(self):
    """the reference direction pointing north on screen"""
    return self.rot * yvector

  @property
  def south(self):
    """the reference direction pointing south on screen"""
    return -self.north

  @property
  def outOfScreen(self):
    """the reference direction pointing out of the screen"""
    return self.rot * zvector

  @property
  def intoScreen(self):
    """the reference direction pointing into the screen"""
    return -self.outOfScreen

  @east.setter
  def east(self, v):
    """moving one screen direction is the least rotation that does it"""
    self.rot = rotation.map(self.east, v) * self.rot

  @west.setter
  def west(self, v):
    """sets the direction pointing west, the least rotation that moves it there"""
    self.east = -v

  @north.setter
  def north(self, v):
    """sets the direction pointing north, the least rotation that moves it there"""
    self.rot = rotation.map(self.north, v) * self.rot

  @south.setter
  def south(self, v):
    """sets the direction pointing south, the least rotation that moves it there"""
    self.north = -v

  @outOfScreen.setter
  def outOfScreen(self, v):
    """sets the direction pointing out of the screen, the least rotation that moves it there"""
    self.rot = rotation.map(self.outOfScreen, v) * self.rot

  @intoScreen.setter
  def intoScreen(self, v):
    """sets the direction pointing into the screen, the least rotation that moves it there"""
    self.outOfScreen = -v

  # two conventions are the same when they lay the frame out the same way
  def __eq__(self, other):
    return isinstance(other, plottingConvention) and bool(self.rot.angle(other.rot) < 1e-6)

  def __ne__(self, other):
    return not self == other

  __hash__ = None

  # ------------------------------------------------------------------ pictograms

  def char(self, utf8=None, names=("x", "y", "z")):
    """'y↑→x' when both screen axes are reference axes, the three names run together otherwise"""
    iN, up = _axisOf(self.north)
    iE, right = _axisOf(self.east)
    if iN is None or iE is None:
      return "".join(names)
    return _arrowPictogram(names[iN], up, names[iE], right, utf8)

  @property
  def isPictogram(self):
    """whether both screen axes lie along reference axes, so that the convention reads as 'y↑→x'"""
    return self.char() != "xyz"

  def __str__(self):
    return self.char()

  def __repr__(self):
    if self.isPictogram:
      return f"plottingConvention ({self.char()})"
    from .display import display, numString
    def fmt(v):
      return "(" + ", ".join(numString(x) for x in v.data) + ")"
    return display("plottingConvention", [("outOfScreen", fmt(self.outOfScreen)), ("north", fmt(self.north)), ("east", fmt(self.east))])

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from ..plotting.polarplot import setView


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


def conventionNames(names=("x", "y", "z")):
  """the eight written in the axis names of a frame, 'Y1↑→X1' for an Oxford file"""
  return [plottingConvention(s).char(utf8=True, names=tuple(names)) for s in AXIS_CONVENTIONS]


def closestConvention(rot):
  """the entry of AXIS_CONVENTIONS nearest a rotation, and how far it is from it in radians"""
  rot = rot.rot if isinstance(rot, plottingConvention) else rot
  d = np.array([float(rot.angle(plottingConvention(s).rot)) for s in AXIS_CONVENTIONS])
  k = int(np.argmin(d))
  return k, d[k]


# ------------------------------------------------------------------ the session default

# what plottingConvention.default returns: the session default, and a call on it with a
# convention sets it, plottingConvention.default('y↑→x'), as MATLAB reads it
class _sessionDefault(plottingConvention):

  # default('y↑→x'), default(zvector, xvector): a convention made the default; default(north=
  # xvector), default(intoScreen=zvector): the default with those directions changed, the rest
  # kept (MATLAB's plotx2north, plotzIntoPlane)
  def __call__(self, outOfScreen: ConventionName | Any = None, east=None, save=False, **directions):
    if isinstance(outOfScreen, (str, plottingConvention)) or (outOfScreen is not None and east is not None):
      plottingConvention(outOfScreen, east).makeDefault(save=save)
      return
    pC = plottingConvention(self)
    given = dict(directions, **{k: v for k, v in (("outOfScreen", outOfScreen), ("east", east)) if v is not None})
    for name, v in given.items():
      setattr(pC, name, v)
    pC.makeDefault(save=save)

  def __repr__(self):
    return aligned(plottingConvention.__repr__(plottingConvention(self)))


# the frame that supplies the session default convention, minted by frames.py on first use
def _defaultFrame():
  if SESSION.defaultFrame is None:
    SESSION.defaultFrame = SESSION.mintDefaultFrame()
  if SESSION.defaultFrame.how2plot is None:
    SESSION.defaultFrame.how2plot = plottingConvention.ij
  return SESSION.defaultFrame


__all__ = ["AXIS_CONVENTIONS", "conventionNames", "closestConvention", "plottingConvention"]
