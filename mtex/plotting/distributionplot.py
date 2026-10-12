# the distributions of the misorientation angle and axis (MATLAB plotAngleDistribution,
# plotAxisDistribution)
#
#   plotAngleDistribution(mori)            a histogram of the disorientation angles in percent
#   plotAngleDistribution(cs1, cs2)        the uniform distribution, a line
#   plotAngleDistribution(mdf)             the distribution of a function, a line
#   plotAxisDistribution(mori, contourf=True, halfwidth=5*degree, weights=w)
#   plotAxisDistribution(ori1, ori2)       the axes in specimen coordinates
#   plotAxisDistribution(cs1, cs2, antipodal=True), plotAxisDistribution(mdf)
#
# A line drawn into a held histogram is scaled from multiples of the uniform density to the
# percent of a bin, so the two can be compared, as MATLAB does.
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from ..geometry.frames import referenceFrame
from ..geometry.orientation import orientation
from ..geometry.orientationregion import fundamentalRegion
from ..geometry.units import degree
from .keywords import lineKeywords
from .mtexfigure import figureSize, gcm, mtexFigure

if TYPE_CHECKING:
  from typing import Unpack

  from .keywords import PlotKeywords


# the axes a distribution goes into: the current ones when held, else a new figure
def _lineAxes(figSize):
  fig = gcm()
  if fig is not None and fig.hold and fig.children:
    return fig.children[-1], False
  if fig is not None and fig.pending is not None:
    return fig.takeAxis(), True
  fig = mtexFigure(figSize=figureSize(figSize, (6.4, 4.2)))
  return fig.nextAxis(), True


def plotAngleDistribution(obj, *args, figSize=None, resolution=5 * degree, numBins=None, percent=False, displayName=None,
                          **kw: Unpack[PlotKeywords]):
  """plotAngleDistribution(mori), plotAngleDistribution(cs1, cs2), plotAngleDistribution(mdf)"""
  from ..functions.characteristics import calcAngleDistribution
  kw = lineKeywords(kw)
  ax, isNew = _lineAxes(figSize)
  bars = getattr(ax, "_mtexAngleBins", None)
  # bars and lines take the colours of MTEX's order one after the other, as in MATLAB
  if "color" not in kw:
    from .colormaps import MTEX_COLORS
    k = getattr(ax, "_mtexColorIndex", 0)
    kw["color"] = tuple(MTEX_COLORS[k % len(MTEX_COLORS)])
    ax._mtexColorIndex = k + 1
  if isinstance(obj, orientation):
    maxOmega = fundamentalRegion(obj.CS, obj.SS).maxAngle()
    omega = obj.angle().ravel()
    if bars is None:
      if np.max(omega) < maxOmega / 2:
        maxOmega = float(np.max(omega))
      n = max(15, round(maxOmega / resolution)) if numBins is None else int(numBins)
      bins = np.linspace(-np.finfo(float).eps, maxOmega + 0.01, n)
    else:
      bins = bars
    d, _ = np.histogram(omega, bins)
    mid = (bins[:-1] + bins[1:]) / 2
    label = displayName if displayName is not None else f"{obj.CS.mineral}-{obj.SS.mineral}"
    kw.setdefault("edgecolor", "k")
    kw.setdefault("linewidth", 0.5)
    h = ax.bar(mid / degree, 100 * d / np.sum(d), width=(bins[1] - bins[0]) / degree, label=label, **kw)
    ax._mtexAngleBins = bins
    ax.set_xlim(0, bins[-1] / degree)
    unit = "%"
  else:
    if isinstance(obj, referenceFrame):
      frames = [obj] + [a for a in args if isinstance(a, referenceFrame)][:1]
      density, omega = calcAngleDistribution(*frames)
    else:
      density, omega = calcAngleDistribution(obj)
    # in a histogram a density of one is the percent of one bin
    if bars is not None:
      factor, unit = 100 / (len(bars) - 1), "%"
    else:
      factor, unit = (100, "%") if percent else (1, "mrd")
    kw.setdefault("linewidth", 2)
    h = ax.plot(omega / degree, factor * np.maximum(0, density), label=displayName, **kw)[0]
  if isNew:
    ax.set_xlabel("Misorientation angle (degrees)")
    ax.set_ylabel(f"Frequency ({unit})")
  return h


def plotAxisDistribution(obj, *args, halfwidth=None, weights=None, antipodal=False, **kw: Unpack[PlotKeywords]):
  """plotAxisDistribution(mori), plotAxisDistribution(ori1, ori2), plotAxisDistribution(cs1, cs2),
  plotAxisDistribution(mdf): the rotation axes on the sector of the symmetry the frames share"""
  from ..functions.characteristics import calcAxisDistribution
  from ..functions.density import calcDensity
  from .sphericalplot import FILL_FLAGS, plot
  if isinstance(obj, orientation):
    other = args[0] if args and isinstance(args[0], orientation) else None
    v = obj.axis(other) if other is not None else obj.axis()
    v.antipodal = bool(antipodal or obj.antipodal)
    if any(kw.get(f) for f in FILL_FLAGS):
      # the axes of a list of misorientations are smoothed as MATLAB smooths them, by 10 degrees
      f = calcDensity(v, weights=weights, halfwidth=10 * degree if halfwidth is None else halfwidth)
      return plot(f, fundamentalRegion=True, **kw)
    return plot(v, fundamentalRegion=True, **kw)
  if isinstance(obj, referenceFrame):
    cs2 = args[0] if args and isinstance(args[0], referenceFrame) else None
    f = calcAxisDistribution(obj, cs2) if cs2 is not None else calcAxisDistribution(obj)
  else:
    f = calcAxisDistribution(obj)
  return plot(f, fundamentalRegion=True, **kw)


__all__ = ["plotAngleDistribution", "plotAxisDistribution"]
