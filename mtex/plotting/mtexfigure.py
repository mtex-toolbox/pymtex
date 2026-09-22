# the figure a plot goes into: a thin layout over a matplotlib figure (MATLAB @mtexFigure,
# newMtexFigure, gcm, mtexColorbar)
#
# MATLAB keeps one mtexFigure per figure window and lets plot commands ask for the next
# axes; here the same: mtexFigure holds the figure, its axes in the order they were made,
# the hold state, and lays n axes out in a grid slightly wider than square. matplotlib is
# imported inside the methods, so that import mtex costs nothing.
from __future__ import annotations

import math

import numpy as np

_current = None


# MATLAB's named figure sizes scale the default size of a figure
FIGURE_SIZES = {"tiny": 0.4, "small": 0.6, "medium": 0.8, "normal": 1.0, "large": 1.3, "huge": 1.6}


def figureSize(figSize, default):
  if figSize is None:
    return default
  if isinstance(figSize, str):
    if figSize.lower() not in FIGURE_SIZES:
      raise ValueError("figSize is 'tiny', 'small', 'medium', 'normal', 'large' or 'huge', or a pair of inches")
    s = FIGURE_SIZES[figSize.lower()]
    return (s * default[0], s * default[1])
  return figSize


class mtexFigure:
  """a figure of MTEX plots

    mtexFigure()              a new figure
    mtexFigure(n)             a new figure laid out for n axes
    fig.nextAxis()            the next axes of the layout
    fig.children              the axes made so far
    fig.colorbar()            a colour bar per axes, or one when the ranges are equal
  """

  def __init__(self, n=1, figSize=None, hold=False, figure=None, columns=None, layout=None, innerPlotSpacing=None, outerPlotSpacing=None):
    global _current
    import matplotlib.pyplot as plt
    self.nrows, self.ncols = (int(layout[0]), int(layout[1])) if layout is not None else layoutFor(n, columns)
    figSize = figureSize(figSize, (min(3.2 * self.ncols, 16), min(3.2 * self.nrows, 12)))
    self.fig = plt.figure(figsize=figSize, layout="constrained") if figure is None else figure
    self.children = []
    self.hold = bool(hold)
    self.colorbars = []
    self.colorRange = None
    self.colormap = None
    # the axes nextAxis made that no plot has drawn into yet
    self.pending = None
    _current = self

  # ------------------------------------------------------------------ the axes

  def nextAxis(self):
    k = len(self.children)
    if k >= self.nrows * self.ncols:
      rows, cols = self.nrows, self.ncols
      self.nrows, self.ncols = layoutFor(k + 1)
      # the figure grows with its layout, so the panels keep their size
      w, h = self.fig.get_size_inches()
      self.fig.set_size_inches(min(w * self.ncols / cols, 16), min(h * self.nrows / rows, 12))
      gs = self.fig.add_gridspec(self.nrows, self.ncols)
      for i, ax in enumerate(self.children):
        ax.set_subplotspec(gs[i])
      ax = self.fig.add_subplot(gs[k])
      self.children.append(ax)
      return ax
    ax = self.fig.add_subplot(self.nrows, self.ncols, k + 1)
    self.children.append(ax)
    return ax

  # the axes the next plot draws into: the pending one, else a new one
  def takeAxis(self):
    ax, self.pending = self.pending, None
    return ax if ax is not None else self.nextAxis()

  @property
  def currentAxes(self):
    return self.children[-1] if self.children else self.nextAxis()

  # the drawings on the axes, the objects a colour bar reads
  def _mappables(self):
    out = []
    for ax in self.children:
      for holder in ("_mtexSphericalPlot", "_mtexMapPlot"):
        m = getattr(getattr(ax, holder, None), "mappable", None)
        if m is not None:
          out.append((ax, m))
    return out

  # ------------------------------------------------------------------ colour

  # one colour range over every axes ('equal'), or one given as (lo, hi)
  def setColorRange(self, colorRange="equal"):
    ms = self._mappables()
    if not ms:
      return
    if isinstance(colorRange, str):
      if colorRange != "equal":
        raise ValueError("colorRange is 'equal' or (lo, hi)")
      lo = min(m.get_clim()[0] for _, m in ms)
      hi = max(m.get_clim()[1] for _, m in ms)
    else:
      lo, hi = colorRange
    for ax, m in ms:
      refill = getattr(getattr(ax, "_mtexSphericalPlot", None), "refill", None)
      if refill is None:
        m.set_clim(lo, hi)
      else:
        refill(colorRange=(lo, hi))
    self.colorRange = (lo, hi)
    # the bars follow the range: one for all once the ranges agree, as MATLAB redraws them
    if self.colorbars:
      labels = [cb.ax.get_ylabel() for cb in self.colorbars]
      for cb in self.colorbars:
        cb.remove()
      self.colorbars = []
      self.colorbar(label=labels[0] if labels else "")

  # a filled contour plot cuts its bands out of the map, so it is drawn again
  def setColormap(self, cmap):
    self.colormap = cmap
    for ax, m in self._mappables():
      refill = getattr(getattr(ax, "_mtexSphericalPlot", None), "refill", None)
      if refill is None:
        m.set_cmap(cmap)
      else:
        refill(colormap=cmap)

  # mtexColorbar: a bar per axes, or one for all when they share a range; a figure that
  # has bars loses them instead, as MATLAB's toggle does
  def colorbar(self, label="", **kwargs):
    if self.colorbars:
      for cb in self.colorbars:
        cb.remove()
      self.colorbars = []
      return []
    ms = self._mappables()
    if not ms:
      return []
    clims = {tuple(np.round(m.get_clim(), 12)) for _, m in ms}
    if len(clims) == 1 and len(ms) > 1:
      cb = self.fig.colorbar(ms[-1][1], ax=[ax for ax, _ in ms], shrink=0.7, label=label, **kwargs)
      self.colorbars.append(cb)
      return [cb]
    bars = [self.fig.colorbar(m, ax=ax, shrink=0.7, label=label, **kwargs) for ax, m in ms]
    self.colorbars += bars
    return bars

  # ------------------------------------------------------------------ pass through

  def drawNow(self):
    self.fig.canvas.draw_idle()

  def savefig(self, *args, **kwargs):
    return self.fig.savefig(*args, **kwargs)

  def show(self):
    import matplotlib.pyplot as plt
    plt.show()


# the grid n axes are laid out in: the divisor of n nearest to the width of a slightly wide
# picture, or the rounded width when no divisor comes close, so that a row is filled; or
# the given number of columns
def layoutFor(n, columns=None):
  n = max(int(n), 1)
  if columns:
    return math.ceil(n / int(columns)), int(columns)
  target = math.sqrt(1.5 * n)
  divisors = [d for d in range(1, n + 1) if n % d == 0]
  ncols = min(divisors, key=lambda d: (abs(d - target), -d))
  if ncols > 2 * target or ncols < target / 2:
    ncols = math.ceil(target)
  return math.ceil(n / ncols), ncols


def newMtexFigure(n=1, **kwargs):
  return mtexFigure(n, **kwargs)


# nextAxis: the next plot goes into the next axes of the current figure
def nextAxis(k=None, fig=None):
  fig = gcm() if fig is None else fig
  if fig is None:
    fig = mtexFigure()
  # nextAxis(k) makes the axes k, counted from zero, the one the next plot draws into
  fig.pending = fig.children[int(k)] if k is not None else fig.nextAxis()
  return fig.pending


# mtexTitle('text'): the title of the current axes
def mtexTitle(text, fig=None, **kwargs):
  fig = gcm() if fig is None else fig
  if fig is not None and fig.children:
    fig.children[-1].set_title(text, **kwargs)


# the current mtexFigure, None before the first plot
def gcm():
  # a figure closed with plt.close is no longer current; a figure pyplot does not manage,
  # one embedded in a window, has no number and stays current
  if _current is not None:
    import matplotlib.pyplot as plt
    number = getattr(_current.fig, "number", None)
    if number is not None and number not in plt.get_fignums():
      return None
  return _current


# hold on: the next plot goes into the current axes
def hold(on=True):
  fig = gcm()
  if fig is None:
    # a plain figure the user opened is adopted, else a new one
    import matplotlib.pyplot as plt
    fig = mtexFigure(figure=plt.gcf()) if plt.get_fignums() else mtexFigure()
  fig.hold = on if isinstance(on, bool) else str(on).lower() == "on"
  return fig


def mtexColorbar(fig=None, title=None, **kwargs):
  if title is not None:
    kwargs["label"] = title
  fig = gcm() if fig is None else fig
  return [] if fig is None else fig.colorbar(**kwargs)


# logAxis(ax, xLimits, yLimits): both axes logarithmic, a decade the same length on each
def logAxis(ax=None, xLimits=None, yLimits=None):
  import matplotlib.pyplot as plt
  ax = plt.gca() if ax is None else ax
  ax.set_xscale("log")
  ax.set_yscale("log")
  if xLimits is not None:
    ax.set_xlim(*xLimits)
  if yLimits is not None:
    ax.set_ylim(*yLimits)
  ax.set_aspect("equal")
  return ax


def setColorRange(colorRange="equal", fig=None):
  fig = gcm() if fig is None else fig
  if fig is not None:
    fig.setColorRange(colorRange)


# MATLAB's compass names for where a legend sits
LEGEND_LOCATIONS = {"north": "upper center", "south": "lower center", "east": "center right", "west": "center left",
                    "northeast": "upper right", "northwest": "upper left", "southeast": "lower right",
                    "southwest": "lower left", "best": "best"}


# legend(): one legend for the figure, from every artist a DisplayName named, on the last
# chart, or beside the charts for MATLAB's outside locations (MATLAB legend('show'))
def legend(location="northeast", numColumns=1, fontSize=None, **kw):
  import matplotlib.pyplot as plt

  from ..geometry.session import getMTEXpref
  fig = plt.gcf()
  seen, handles, labels = set(), [], []
  for ax in fig.axes:
    for h, l in zip(*ax.get_legend_handles_labels()):
      if l not in seen:
        seen.add(l)
        handles.append(h)
        labels.append(l)
  kw.setdefault("ncols", int(numColumns))
  kw.setdefault("fontsize", getMTEXpref("FontSize", 13) if fontSize is None else fontSize)
  where = str(kw.pop("location", location)).lower()
  if where.endswith("outside"):
    return _sameSize(fig.legend(handles, labels, loc="outside " + LEGEND_LOCATIONS.get(where[:-7], "lower center"), **kw))
  ax = fig.axes[-1] if fig.axes else plt.gca()
  return _sameSize(ax.legend(handles, labels, loc=LEGEND_LOCATIONS.get(where, where), **kw))


# every entry of a legend gets the same marker, whatever size the markers it stands for have
def _sameSize(leg, size=36.0):
  for h in leg.legend_handles:
    if hasattr(h, "set_sizes"):
      h.set_sizes([size])
  return leg


__all__ = ["logAxis", "mtexFigure", "newMtexFigure", "gcm", "hold", "legend", "nextAxis", "mtexTitle", "mtexColorbar",
           "setColorRange"]
