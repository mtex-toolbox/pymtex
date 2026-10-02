# the values of an ODF along a fibre, against the angle along it (MATLAB @SO3Fun/plotFibre)
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from ..geometry.units import degree
from .keywords import lineKeywords
from .mtexfigure import gcm, mtexFigure

if TYPE_CHECKING:
  from typing import Unpack

  from .keywords import PlotKeywords


def plotFibre(odf, f, points=1000, lineWidth=1.5, **kw: Unpack[PlotKeywords]):
  """plotFibre(odf, f): the ODF along the fibre f, as a line over the angle along the fibre; a
  list of functions one line each"""
  fig = gcm()
  if fig is not None and fig.hold and fig.children:
    ax = fig.children[-1]
  else:
    ax = mtexFigure(figSize=kw.pop("figSize", None)).nextAxis()
  o = f.orientation(points)
  omega = np.linspace(0, 2 * np.pi, o.size)
  # a fibre of other frames is followed by its rotations, as MATLAB does
  if getattr(o, "CS", None) is not odf.CS or getattr(o, "SS", None) is not odf.SS:
    from ..geometry.quaternion import rotation
    o = rotation(o.data)
  h = ax.plot(omega / degree, np.real(odf.eval(o)), linewidth=lineWidth, **lineKeywords(kw))
  ax.set_xlim(0, 360)
  ax.set_xlabel("angle along the fibre in degree")
  ax.set_ylabel("mrd")
  return h


__all__ = ["plotFibre"]
