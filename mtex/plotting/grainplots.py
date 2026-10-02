# histograms of grains and ellipses on a map (MATLAB @grain2d/hist, @grain2d/histogram,
# plotEllipse)
#
#   histogram(grains), histogram(grains, prop, bins)   the relative area per phase, overlaid
#   histogram(grains, grouped=True)                     the bars of the phases side by side
#   plotEllipse(c, a, b, lineColor=, lineWidth=)   the fitted ellipses drawn into the held map
#
# A grain histogram is area weighted: every bar is the share of the analysed area the
# grains of one phase in that class hold, in percent, so many small grains do not dominate
# it. matplotlib is imported inside the functions.
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from ..geometry.session import getMTEXpref
from ..geometry.vector3d import vector3d
from .maps import phaseColor, phasePalette
from .mtexfigure import gcm, mtexFigure

if TYPE_CHECKING:
  from typing import Unpack

  from .keywords import PlotKeywords


# the bins and the share of the area of every phase in each, one column per indexed phase
# the measure a grain is weighted by: its volume in 3D, its area in a map
def _measure(grains):
  from ..maps3d.grain3d import grain3d
  return ("volume", grains.volume) if isinstance(grains, grain3d) else ("area", grains.area)


def _areaShares(grains, prop, bins):
  area = np.asarray(_measure(grains)[1], dtype=float)
  prop = area if prop is None else np.asarray(prop, dtype=float).reshape(-1)
  if np.isscalar(bins):
    bins = np.linspace(0, np.max(prop) + np.finfo(float).eps, int(bins))
  bins = np.asarray(bins, dtype=float)
  ids = grains.indexedPhaseId
  shares = np.zeros((len(bins) - 1, len(ids)))
  total = np.sum(area)
  for k, p in enumerate(ids):
    sel = grains.phaseId == p
    # the inner edges decide the bin, so the largest value lands in the last one
    binId = np.digitize(prop[sel], bins[1:-1])
    ok = (prop[sel] >= bins[0]) & (prop[sel] <= bins[-1])
    shares[:, k] = np.bincount(binId[ok], area[sel][ok], len(bins) - 1) / total
  return bins, ids, shares


def _axes(parent=None):
  if parent is not None:
    return parent
  fig = gcm()
  if fig is not None and fig.hold and fig.children:
    return fig.children[-1]
  if fig is not None and fig.pending is not None:
    return fig.takeAxis()
  if fig is not None and not fig.children:
    return fig.nextAxis()
  return mtexFigure(1, figSize=(5.5, 4.0)).nextAxis()


def histogram(x, *args, **kw: Unpack[PlotKeywords]):
  """histogram(grains), histogram(grains, prop, bins): the area weighted distribution of a
  grain property as one histogram per indexed phase, drawn over each other, or with
  grouped=True side by side (MATLAB's hist); histogram(v, numBin, weights=w): the rose
  diagram of directions"""
  if isinstance(x, vector3d):
    from .polarplot import histogram as rose
    return rose(x, *args, **kw)
  # a plain list of numbers is an ordinary histogram, as MATLAB's histogram(x, n) is, its bins
  # chosen from the data unless given, as MATLAB's automatic binning
  if isinstance(x, np.ndarray) or (isinstance(x, (list, tuple)) and not hasattr(x, "loops")):
    bins = args[0] if args else kw.pop("bins", "auto")
    ax = _axes(kw.pop("parent", None))
    return ax.hist(np.asarray(x, dtype=float).reshape(-1), bins=bins, **kw)
  return _grainHistogram(x, *args, **kw)


def _grainHistogram(grains, prop=None, bins=15, grouped=False, **kw):
  bins, ids, shares = _areaShares(grains, prop, bins)
  ax = _axes(kw.pop("parent", None))
  palette = phasePalette(grains.CSList)
  hs = []
  # grouped: the bars of the phases side by side in each class, else overlaid
  width = np.diff(bins) / (len(ids) + 0.5)
  centre = 0.5 * (bins[:-1] + bins[1:])
  alpha = kw.pop("alpha", 1.0 if grouped else 0.6)
  for k, p in enumerate(ids):
    if grouped:
      hs.append(ax.bar(centre + (k - (len(ids) - 1) / 2) * width, 100 * shares[:, k], width=width, color=palette[int(p)],
                       alpha=alpha, label=grains.CSList[int(p)].mineral, **kw))
    else:
      hs.append(ax.bar(bins[:-1], 100 * shares[:, k], width=np.diff(bins), align="edge", color=palette[int(p)], alpha=alpha,
                       edgecolor=palette[int(p)], label=grains.CSList[int(p)].mineral, **kw))
  ax.set_xlim(bins[0], bins[-1])
  size = getMTEXpref("FontSize", 13)
  measure = _measure(grains)[0]
  ax.set_ylabel(f"relative {measure} (%)", fontsize=size)
  if prop is None:
    ax.set_xlabel(f"grain {measure}", fontsize=size)
  ax.legend()
  return hs


def plotEllipse(c, a, b, lineColor="k", lineWidth=1.0, displayName=None, **kw: Unpack[PlotKeywords]):
  """plotEllipse(c, a, b): the ellipses with the centres c and the half axes a and b drawn
  into the held map"""
  from matplotlib.collections import LineCollection
  from .mapplot import _target
  c, a, b = (v if isinstance(v, vector3d) else vector3d(v) for v in (c, a, b))
  phi = np.linspace(0, 2 * np.pi, 100)
  C, A, B = (v.data.reshape(-1, 3) for v in (c, a, b))
  P = C[:, None, :] + A[:, None, :] * np.cos(phi)[None, :, None] + B[:, None, :] * np.sin(phi)[None, :, None]
  fig = gcm()
  parent = kw.pop("parent", None)
  if parent is not None and hasattr(parent, "_mtexMapPlot"):
    mP = parent._mtexMapPlot
  elif fig is not None and fig.children and hasattr(fig.children[-1], "_mtexMapPlot"):
    mP = fig.children[-1]._mtexMapPlot
  else:
    raise ValueError("plotEllipse draws into a map: plot the grains first and hold")
  flat = [mP.plane.to2d(p) @ mP.layout.T for p in P]
  colors = lineColor if isinstance(lineColor, str) else np.asarray(lineColor, dtype=float)
  h = LineCollection(flat, colors=colors, linewidths=lineWidth, zorder=5, label=displayName, **kw)
  mP.ax.add_collection(h)
  return h


__all__ = ["histogram", "plotEllipse"]
