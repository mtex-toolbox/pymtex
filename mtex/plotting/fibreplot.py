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
  """the ODF along the fibre f, as a line over the angle along the fibre; MATLAB's plotFibre

    odf = SantaFe()
    f = fibre.gamma(odf.CS, odf.SS)
    plotFibre(odf, f)                                     # the values along the gamma fibre
    plotFibre(odf, f, points=200, lineWidth=3, color='r')  # fewer points, a thick red line

  The fibre is sampled at equally spaced orientations and the ODF evaluated there, the real
  part drawn against the angle along the fibre from 0 to 360 degree, in mrd. A held figure
  takes the line into its last axes, so that several functions can be compared along one
  fibre. A fibre of other frames than the function's is followed by its rotations, as MATLAB
  does. The other keywords are those of a line plot. MATLAB returns the values and the angles;
  the port returns the line.

  Input
    odf - SO3Fun
    f - fibre
  Output
    h - the matplotlib lines drawn
  Keywords
    points - the number of orientations along the fibre, MATLAB's resolution
    lineWidth - the width of the line
  See also
    fibre, plotSection, eval"""
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
