# the figure a plot goes into: a thin layout over a matplotlib figure (MATLAB @mtexFigure,
# newMtexFigure, gcm, mtexColorbar)
#
# MATLAB keeps one mtexFigure per figure window and lets plot commands ask for the next
# axes; here the same: mtexFigure holds the figure, its axes in the order they were made,
# the hold state, and lays n axes out in a grid slightly wider than square. matplotlib is
# imported inside the methods, so that import mtex costs nothing.
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from ..geometry.display import display, header
from .keywords import refuseVariants

if TYPE_CHECKING:
  from .vocabulary import FigureSizeName

_current = None


# MATLAB's named figure sizes scale the default size of a figure
FIGURE_SIZES = {"tiny": 0.4, "small": 0.6, "medium": 0.8, "normal": 1.0, "large": 1.3, "huge": 1.6}


def figureSize(figSize, default):
  """the size of a figure in inches: the default, scaled by one of MATLAB's names 'tiny', 'small',
  'medium', 'normal', 'large', 'huge', or the pair given"""
  if figSize is None:
    return default
  if isinstance(figSize, str):
    if figSize.lower() not in FIGURE_SIZES:
      raise ValueError("figSize is 'tiny', 'small', 'medium', 'normal', 'large' or 'huge', or a pair of inches")
    s = FIGURE_SIZES[figSize.lower()]
    # a small figure keeps room for its labels and colour bar, whose fonts do not shrink
    return (max(s * default[0], min(2.5, default[0])), max(s * default[1], min(2.5, default[1])))
  return figSize


# the holders of the plots of an axes, one per kind of plot
PLOT_HOLDERS = ("_mtexSphericalPlot", "_mtexMapPlot", "_mtexSpacePlot", "_mtexLinePlot", "_mtexScenePlot")


# the redraw of a plot whose colours are cut from the map, a filled contour or isosurfaces
def _refillOf(ax):
  for holder in ("_mtexSphericalPlot", "_mtexSpacePlot"):
    refill = getattr(getattr(ax, holder, None), "refill", None)
    if refill is not None:
      return refill
  return None


# matplotlib's constrained layout keeping a margin around the axes, in inches: the pad around
# each axes counts towards it, the rest is the inset of the layout's rectangle, renewed on every
# layout since the figure grows with its axes
def _marginLayout(margin, **pads):
  from matplotlib.layout_engine import ConstrainedLayoutEngine

  class marginLayout(ConstrainedLayoutEngine):
    def execute(self, fig):
      w, h = fig.get_size_inches()
      p = self.get()
      dx, dy = max(margin - p["w_pad"], 0.0) / w, max(margin - p["h_pad"], 0.0) / h
      self.set(rect=(dx, dy, 1 - 2 * dx, 1 - 2 * dy))
      return super().execute(fig)

  return marginLayout(**pads)


class mtexFigure:
  """a figure of MTEX plots: equally sized axes in a grid, with colour bars and legends that
  follow them

    fig = mtexFigure()                               # a new figure
    fig = mtexFigure(4, figSize='large')             # laid out for 4 axes, large
    fig = mtexFigure(layout=(1, 3))                  # one row of three axes
    fig = mtexFigure(6, columns=2)                   # 6 axes in two columns
    ax = fig.nextAxis()                              # the next axes of the layout
    fig.children                                     # the axes made so far
    fig.colorbar()                                   # a colour bar per axes, or one for all

  The figure becomes the current one, which gcm returns and into which the plots of MTEX
  draw; newMtexFigure makes one the same way. Without layout or columns the n axes are laid
  out in the grid nearest to a slightly wide picture. The layout grows when more axes are
  asked for than it holds, and the figure grows with it, so that the axes keep their size.
  The spacing between the axes is matplotlib's constrained layout, which makes room for
  titles, labels and colour bars; innerPlotSpacing sets the gap between neighbouring axes and
  outerPlotSpacing the margin around them all, at least half that gap, in pixels at 96 to
  the inch, as MATLAB's.

  Input
    n - the number of axes the layout is made for
  Output
    fig - mtexFigure, with fig.fig the matplotlib figure and fig.children its axes
  Keywords
    figSize - 'tiny', 'small', 'medium', 'normal', 'large', 'huge', or (width, height) in
        inches
    hold - start with hold on: the next plot draws into the last axes
    figure - a matplotlib figure to adopt, with its axes, instead of a new one; one embedded
        in a window keeps its size
    columns - the number of columns of the layout
    layout - (rows, columns) of the layout
    innerPlotSpacing - the gap between neighbouring axes, in pixels at 96 to the inch
    outerPlotSpacing - the margin around the axes, in pixels at 96 to the inch
    marginx - the space left and right of every axes, in pixels at 96 to the inch
    marginy - the space above and below every axes, in pixels at 96 to the inch
  See also
    newMtexFigure, nextAxis, gcm, hold, mtexColorbar, Multiplot"""

  def __init__(self, n=1, figSize: FigureSizeName | Any = None, hold=False, figure=None, columns=None, layout=None, innerPlotSpacing=None, outerPlotSpacing=None,
               marginx=None, marginy=None):
    global _current
    import matplotlib.pyplot as plt
    self.nrows, self.ncols = (int(layout[0]), int(layout[1])) if layout is not None else layoutFor(n, columns)
    # the size asked for, so that a map drawn first can shape the figure after itself
    self.figSize = figSize
    figSize = figureSize(figSize, (min(3.2 * self.ncols, 16), min(3.2 * self.nrows, 12)))
    self.fig = plt.figure(figsize=figSize, layout="constrained") if figure is None else figure
    # a figure pyplot does not manage is embedded in a window, whose size is the widget's:
    # no plot resizes it
    self.fixedSize = figure is not None and getattr(figure.canvas, "manager", None) is None
    # MATLAB's marginx and marginy: the space around each axes in pixels, 96 to the inch;
    # innerPlotSpacing the gap between two axes, which is twice that space
    pads = {name: float(m) / 96 for name, m in (("w_pad", marginx), ("h_pad", marginy)) if m is not None}
    if innerPlotSpacing is not None:
      pads = {"w_pad": float(innerPlotSpacing) / 192, "h_pad": float(innerPlotSpacing) / 192, "wspace": 0.0, "hspace": 0.0} | pads
    if figure is None and outerPlotSpacing is not None:
      self.fig.set_layout_engine(_marginLayout(float(outerPlotSpacing) / 96, **pads))
    elif figure is None and pads:
      self.fig.get_layout_engine().set(**pads)
    # a figure adopted keeps its axes, the current one current
    self.children = [] if figure is None else list(figure.axes)
    self.hold = bool(hold)
    self.colorbars = []
    self.colorRange = None
    self.colormap = None
    # the axes nextAxis made that no plot has drawn into yet, and the one drawn into last
    self.pending = None
    self.current = figure.gca() if figure is not None and figure.axes else None
    # the sections of an ODF section plot, which later plots draw into
    self.sections = None
    _current = self

  # ------------------------------------------------------------------ the axes

  # the axes drawn into last, which nextAxis need not have made last; None in an empty figure
  def _axesDrawnInto(self):
    if not self.children:
      return None
    return self.current if self.current in self.children else self.children[-1]

  def nextAxis(self):
    """a new axes in the next cell of the layout, the layout and the figure grown when it is full

      ax = mtexFig.nextAxis()"""
    k = len(self.children)
    if k >= self.nrows * self.ncols:
      rows, cols = self.nrows, self.ncols
      self.nrows, self.ncols = layoutFor(k + 1)
      # the figure grows with its layout, so the panels keep their size
      if not self.fixedSize:
        w, h = self.fig.get_size_inches()
        self.fig.set_size_inches(min(w * self.ncols / cols, 16), min(h * self.nrows / rows, 12))
      gs = self.fig.add_gridspec(self.nrows, self.ncols)
      for i, ax in enumerate(self.children):
        ax.set_subplotspec(gs[i])
      ax = self.fig.add_subplot(gs[k])
    else:
      ax = self.fig.add_subplot(self.nrows, self.ncols, k + 1)
    self.children.append(ax)
    self.current = ax
    return ax

  def takeAxis(self, keep=False):
    """the axes the next plot draws into: the pending one, else a new one; keep=True leaves a
    pending one as it is, for an annotation onto the plot it holds"""
    ax, self.pending = self.pending, None
    if ax is not None:
      # an axes nextAxis left blank shows its frame again
      if getattr(ax, "_mtexBlank", False):
        ax.set_axis_on()
        ax._mtexBlank = False
      # one drawn into is emptied for the next plot unless hold keeps it, as MATLAB's
      # nextAxis(k) picks the cell the next plot paints (#2615); the records of the old plot
      # go with it, so that the new one draws its own frame, grid and labels
      elif not self.hold and not keep:
        ax.cla()
        # a repaint the count of artists may not show, for whoever snapshots the figure
        self.fig._mtexRepaints = getattr(self.fig, "_mtexRepaints", 0) + 1
        for holder in PLOT_HOLDERS:
          if hasattr(ax, holder):
            delattr(ax, holder)
      self.current = ax
    return ax if ax is not None else self.nextAxis()

  @property
  def gca(self):
    """MATLAB's mtexFig.gca: the axes drawn into last"""
    ax = self._axesDrawnInto()
    return ax if ax is not None else self.nextAxis()

  @property
  def currentAxes(self):
    """the last axes of the figure, a new one when there is none"""
    return self.children[-1] if self.children else self.nextAxis()

  # the drawings on the axes, the objects a colour bar reads
  def _mappables(self):
    out = []
    for ax in self.children:
      for holder in PLOT_HOLDERS:
        plot = getattr(ax, holder, None)
        m = getattr(plot, "mappable", None)
        if m is not None:
          out += [(ax, h) for h in getattr(plot, "mappables", None) or [m]]
    return out

  # ------------------------------------------------------------------ colour

  def setColorRange(self, colorRange="equal"):
    """one colour range over every axes ('equal'), each axes its own ('tight'), or one given
    as (lo, hi)

      mtexFig.setColorRange()             # one range over every axes
      mtexFig.setColorRange('tight')      # each axes the range of its own data
      mtexFig.setColorRange((0, 5))       # the range given"""
    ms = self._mappables()
    if isinstance(colorRange, str) and colorRange == "tight":
      return self._tightColorRange(ms)
    # a figure of explicit colours keeps the range for the bar mtexColorbar draws
    if not ms:
      if not isinstance(colorRange, str):
        self.colorRange = tuple(colorRange)
      return
    if isinstance(colorRange, str):
      if colorRange != "equal":
        raise ValueError("colorRange is 'equal' or (lo, hi)")
      lo = min(m.get_clim()[0] for _, m in ms)
      hi = max(m.get_clim()[1] for _, m in ms)
    else:
      lo, hi = colorRange
    for ax, m in ms:
      refill = _refillOf(ax)
      if refill is None:
        m.set_clim(lo, hi)
      else:
        refill(colorRange=(lo, hi))
    self.colorRange = (lo, hi)
    self._redrawColorbars()

  # every axes back to the range of its own data
  def _tightColorRange(self, ms):
    for ax, m in ms:
      refill = _refillOf(ax)
      sP = getattr(ax, "_mtexSphericalPlot", None)
      if refill is not None and sP is not None and np.isfinite(sP.minData):
        refill(colorRange=(sP.minData, sP.maxData))
      elif refill is None:
        m.autoscale()
    self.colorRange = None
    self._redrawColorbars()

  # the bars follow the range: one for all once the ranges agree, as MATLAB redraws them
  def _redrawColorbars(self):
    if self.colorbars:
      label = self.colorbars[0].ax.get_ylabel()
      self._removeBars()
      self.colorbar(label=label)

  # the colour bars of the figure taken away
  def _removeBars(self):
    for cb in self.colorbars:
      _removeBar(cb)
    self.colorbars = []

  # one bar for the axes given over a range in the colour map of the figure
  def _rangeBar(self, colorRange, axes, label, **kwargs):
    import matplotlib.cm
    from matplotlib.colors import Normalize
    from .colormaps import mtexColormap
    sm = matplotlib.cm.ScalarMappable(norm=Normalize(*colorRange), cmap=mtexColormap(self.colormap))
    cb = self.fig.colorbar(sm, ax=axes, shrink=0.7, label=label, **kwargs)
    self.colorbars.append(cb)
    return [cb]

  def setColorScale(self, scale="linear"):
    """MATLAB's set(gca, 'ColorScale', 'log'): the colours spread over the decades of the range"""
    from matplotlib.colors import LogNorm, Normalize
    for ax, m in self._mappables():
      lo, hi = m.get_clim()
      if scale == "log":
        m.set_norm(LogNorm(lo if lo > 0 else hi * 1e-6, hi))
      else:
        m.set_norm(Normalize(lo, hi))
    self._redrawColorbars()

  def setColormap(self, cmap, ax=None):
    """a filled contour plot cuts its bands out of the map, so it is drawn again"""
    # one axes given keeps the map of the figure for the others
    if ax is None:
      self.colormap = cmap
    for a, m in self._mappables():
      if ax is not None and a is not ax:
        continue
      refill = _refillOf(a)
      if refill is None:
        m.set_cmap(cmap)
      else:
        refill(colormap=cmap)

  def colorbar(self, label="", multiple=False, **kwargs):
    """mtexColorbar: a bar per axes, or one for all when they share a range and a colour map; a
    figure that has bars loses them instead, as MATLAB's toggle does"""
    if self.colorbars:
      self._removeBars()
      return []
    ms = self._mappables()
    # a figure of explicit colours, as a boundary coloured by num2rgb, shows the map and the
    # range set for it, as MATLAB's bar reads them off the axes
    if not ms:
      if self.colorRange is None or not self.children:
        return []
      return self._rangeBar(self.colorRange, self.children, label, **kwargs)
    # a bar per axes, of the last drawing coloured by values on it
    ms = list({id(ax): (ax, m) for ax, m in ms}.values())
    # MATLAB's title list: one title per colour bar, in the order of the axes
    if isinstance(label, (list, tuple)):
      bars = [self.fig.colorbar(_barOf(m), ax=ax, shrink=0.7, label=l, **kwargs) for (ax, m), l in zip(ms, label)]
      self.colorbars += bars
      return bars
    # a shared bar needs a shared colour map as well as a shared range, else it describes only
    # one of the axes (MATLAB #2614)
    oneMap = len({m.get_cmap().name for _, m in ms}) == 1
    # one bar for the common range setColorRange gave, whose contour plots keep their own levels
    common = self.colorRange
    if common is not None and not multiple and oneMap and len(ms) > 1 and all(getattr(m, "levels", None) is not None for _, m in ms):
      return self._rangeBar(common, [ax for ax, _ in ms], label, **kwargs)
    clims = {tuple(np.round(m.get_clim(), 12)) for _, m in ms}
    if len(clims) == 1 and len(ms) > 1 and not multiple and oneMap:
      cb = self.fig.colorbar(_barOf(ms[-1][1]), ax=[ax for ax, _ in ms], shrink=0.7, label=label, **kwargs)
      self.colorbars.append(cb)
      return [cb]
    bars = [self.fig.colorbar(_barOf(m), ax=ax, shrink=0.7, label=label, **kwargs) for ax, m in ms]
    self.colorbars += bars
    return bars

  # ------------------------------------------------------------------ pass through

  def drawNow(self, figSize=None):
    """drawNow(figSize='normal'): the figure resized to a named size of its layout, then drawn"""
    if figSize is not None:
      self.fig.set_size_inches(*figureSize(figSize, (min(3.2 * self.ncols, 16), min(3.2 * self.nrows, 12))))
    self.fig.canvas.draw_idle()

  def savefig(self, *args, **kwargs):
    """a PNG is written at zlib level 1, not Pillow's 6: the same pixels, a file about a tenth
    larger, in half the time a large map took, most of which was the compression"""
    import os
    import matplotlib
    name = args[0] if args else kwargs.get("fname")
    ext = os.path.splitext(os.fspath(name))[1][1:] if isinstance(name, (str, os.PathLike)) else ""
    fmt = kwargs.get("format") or ext or matplotlib.rcParams["savefig.format"]
    if fmt.lower() == "png":
      kwargs["pil_kwargs"] = {"compress_level": 1, **(kwargs.get("pil_kwargs") or {})}
    return self.fig.savefig(*args, **kwargs)

  def show(self):
    """shows the figures, matplotlib's show"""
    import matplotlib.pyplot as plt
    plt.show()

  def __repr__(self):
    return display(header("mtexFigure", f"{self.nrows} × {self.ncols}"), [("hold", "on" if self.hold else None)])


# what a colour bar shows of a drawing: filled bands their map over the range, continuously
def _barOf(m):
  bar = getattr(m, "_mtexBar", None)
  if bar is None:
    return m
  import matplotlib.cm
  return matplotlib.cm.ScalarMappable(norm=bar[1], cmap=bar[0])


# a colour bar goes with its axes, also when the drawing it described was redrawn and is
# gone, where matplotlib fails only after removing it, resetting the gone drawing's axes
def _removeBar(cb):
  try:
    cb.remove()
  except AttributeError:
    if getattr(cb.mappable, "axes", None) is not None:
      raise


def layoutFor(n, columns=None):
  """the grid n axes are laid out in: the divisor of n nearest to the width of a slightly wide
  picture, or the rounded width when no divisor comes close, so that a row is filled; or
  the given number of columns"""
  n = max(int(n), 1)
  if columns:
    return math.ceil(n / int(columns)), int(columns)
  target = math.sqrt(1.5 * n)
  divisors = [d for d in range(1, n + 1) if n % d == 0]
  ncols = min(divisors, key=lambda d: (abs(d - target), -d))
  if ncols > 2 * target or ncols < target / 2:
    ncols = math.ceil(target)
  return math.ceil(n / ncols), ncols


# the axes a plot of its own goes into, with its figure: the one nextAxis prepared, the first
# of a fresh newMtexFigure, else that of a new figure of the size asked for or the default
def _freshAxes(figSize=None, default=(5.5, 5.5)):
  fig = gcm()
  if fig is not None and fig.pending is not None:
    return fig, fig.takeAxis()
  if fig is not None and not fig.children and not fig.hold:
    return fig, fig.nextAxis()
  fig = mtexFigure(1, figSize=figureSize(figSize, default))
  return fig, fig.nextAxis()


def newMtexFigure(n=1, **kwargs):
  """a new mtexFigure laid out for n axes; it is always new, so MATLAB's second output, whether
  it is, has no counterpart

    newMtexFigure(4, figSize='large')"""
  return mtexFigure(n, **kwargs)


def nextAxis(k=None, j=None, fig=None):
  """the axes the next plot goes into: the next of the current figure, or the one given by
  its number or its row and column

    nextAxis()                                       # the next axes
    nextAxis(2)                                      # the third axes
    nextAxis(1, 0)                                   # the axes in the second row and the first column
    ax = nextAxis(fig=mtexFig)                       # the next axes of a figure given

  The axes are counted from 0, where MATLAB counts them from 1. An axes made by nextAxis stays
  blank, without a frame, until a plot draws into it, as MATLAB leaves an unused cell; the
  axes before k that do not exist yet are made blank too. Without a current figure a new one
  is made.

  Input
    k - the number of the axes, counted from 0, or the row with j
    j - the column, counted from 0
  Output
    ax - the matplotlib axes the next plot takes
  Keywords
    fig - the mtexFigure; the current one by default
  See also
    mtexFigure, newMtexFigure, Multiplot"""
  fig = gcm() if fig is None else fig
  if fig is None:
    fig = mtexFigure()
  if j is not None:
    k = int(k) * fig.ncols + int(j)
  # an axes prepared stays blank until a plot takes it, as MATLAB leaves an unused cell
  if k is None:
    fig.pending = fig.nextAxis()
    fig.pending.set_axis_off()
    fig.pending._mtexBlank = True
  # the axes up to k that do not exist yet are made, blank
  else:
    while len(fig.children) <= int(k):
      ax = fig.nextAxis()
      ax.set_axis_off()
      ax._mtexBlank = True
    fig.pending = fig.children[int(k)]
  return fig.pending


def mtexTitle(*args, fig=None, **kwargs):
  """mtexTitle('text'): the title of the current axes, mtexTitle(ax, 'text') that of the axes given"""
  if "fontSize" in refuseVariants(kwargs):
    kwargs["fontsize"] = kwargs.pop("fontSize")
  # mtexTitle(ax, 'text'): the title of the axes named
  if len(args) == 2:
    ax, text = args
    return ax.set_title(text, **kwargs)
  text = args[0]
  import matplotlib.pyplot as plt
  # an axes the user made with plt.subplot, in a plain figure or in this one, takes it
  m = gcm() if fig is None else fig
  if fig is None and plt.get_fignums() and plt.gcf().axes and (m is None or plt.gcf() is not m.fig or plt.gca() not in m.children):
    return plt.gca().set_title(text, **kwargs)
  fig = m
  if fig is not None and fig.children:
    fig._axesDrawnInto().set_title(text, **kwargs)


def ylabel(*args, fig=None, **kwargs):
  """ylabel(ax, 'text') or ylabel('text'): a label left of the axes, written beside a
  spherical chart, which has no axis lines to carry matplotlib's own label"""
  ax, text = args if len(args) == 2 else (None, args[0])
  if ax is None:
    ax = _labelAxes(gcm() if fig is None else fig)
  if ax is None:
    import matplotlib.pyplot as plt
    return plt.ylabel(text, **kwargs)
  if ax.axison:
    return ax.set_ylabel(text, **kwargs)
  return ax.text(-0.02, 0.5, text, transform=ax.transAxes, ha="right", va="center", rotation=90, **kwargs)


def xlabel(text, fig=None, **kwargs):
  """xlabel('text'): a label under the current axes, which a spherical chart draws without
  axis lines, so matplotlib's own label would not show"""
  ax = _labelAxes(gcm() if fig is None else fig)
  if ax is None:
    import matplotlib.pyplot as plt
    return plt.xlabel(text, **kwargs)
  if ax.axison:
    return ax.set_xlabel(text, **kwargs)
  return ax.text(0.5, -0.02, text, transform=ax.transAxes, ha="center", va="top", **kwargs)


# the axes a label goes to: the one nextAxis prepared, else the current one, else the last
def _labelAxes(fig):
  if fig is None:
    return None
  return fig.pending or fig.current or (fig.children[-1] if fig.children else None)


def gcm():
  """the current mtexFigure, None before the first plot

    mtexFig = gcm()"""
  # a figure closed with plt.close is no longer current; a figure pyplot does not manage,
  # one embedded in a window, has no number and stays current
  if _current is not None:
    import matplotlib.pyplot as plt
    number = getattr(_current.fig, "number", None)
    if number is not None and number not in plt.get_fignums():
      return None
  return _current


def hold(on=True):
  """hold on: the next plot goes into the current axes

    hold(True)               # the next plots go into the current axes
    hold(False)              # each plot takes its axes for itself again
    mtexFig = hold('on')     # MATLAB's words, the figure held returned"""
  import matplotlib.pyplot as plt
  fig = gcm()
  # a plain figure the user opened after the last MTEX figure is adopted, else a new one
  if plt.get_fignums() and (fig is None or plt.gcf() is not fig.fig):
    fig = mtexFigure(figure=plt.gcf())
  elif fig is None:
    fig = mtexFigure()
  fig.hold = on if isinstance(on, bool) else str(on).lower() == "on"
  return fig


# mtexColorbar('southoutside'): MATLAB's locations beside the axes
COLORBAR_LOCATIONS = {"southoutside": "bottom", "northoutside": "top", "eastoutside": "right", "westoutside": "left",
                      "south": "bottom", "north": "top", "east": "right", "west": "left"}


def mtexColorbar(fig=None, title=None, **kwargs):
  """a colour bar per axes of the current figure, or one when the ranges agree; toggled off when
  the figure has bars; a MATLAB location as the first argument

    mtexColorbar()
    mtexColorbar('southoutside', title='mrd')
    mtexColorbar('multiple')"""
  if isinstance(fig, str):
    kwargs.setdefault("location", fig)
    fig = None
  # mtexColorbar('multiple'): a bar per axes, even where the ranges agree
  if str(kwargs.get("location", "")).lower() == "multiple":
    kwargs.pop("location")
    kwargs["multiple"] = True
  if "location" in kwargs:
    kwargs["location"] = COLORBAR_LOCATIONS.get(str(kwargs["location"]).lower(), kwargs["location"])
  if title is not None:
    kwargs["label"] = title
  fig = gcm() if fig is None else fig
  return [] if fig is None else fig.colorbar(**kwargs)


def saveFigure(fname, fig=None, *args, **kwargs):
  """saveFigure('name.png'): the current figure written cropped to its content, the format by the
  extension; a bitmap at 1.5 times the screen resolution, a map at 2.5 times and without
  smoothing between its cells, so that no colour is invented between measurements
  (MATLAB saveFigure); 'crop' and 'pdf' of MATLAB's TeX tools need nothing here"""
  import os
  import matplotlib.pyplot as plt
  m = gcm() if fig is None else fig
  figure = m.fig if isinstance(m, mtexFigure) else (m if m is not None else plt.gcf())
  ext = os.path.splitext(os.fspath(fname))[1][1:].lower()
  if ext not in ("pdf", "eps", "ps", "svg"):
    isMap = any(hasattr(ax, "_mtexMapPlot") for ax in figure.axes)
    kwargs.setdefault("dpi", (2.5 if isMap else 1.5) * figure.dpi)
    if isMap:
      for ax in figure.axes:
        for im in ax.images:
          im.set_interpolation("nearest")
  kwargs.setdefault("bbox_inches", "tight")
  kwargs.setdefault("pad_inches", 0.02)
  if ext == "png":
    kwargs.setdefault("pil_kwargs", {"compress_level": 1})
  figure.savefig(fname, **kwargs)


def logAxis(ax=None, xLimits=None, yLimits=None):
  """both axes of a plot logarithmic, a decade of the same length on either

    logAxis()                                        # the current axes
    logAxis(ax, (1, 100), (1, 1000))                 # the axes given, with their limits

  Input
    ax - the matplotlib axes; the current one by default
    xLimits - (lo, hi) of the x axis; by default the limits it has
    yLimits - (lo, hi) of the y axis; by default the limits it has
  Output
    ax - the axes
  See also
    plotAngleDistribution"""
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


def setColorRange(colorRange: Literal["equal", "tight"] | Any = "equal", fig=None):
  """one colour range over every axes of the current figure ('equal'), each its own ('tight') or
  the one given as (lo, hi)

    setColorRange('equal')
    setColorRange((0, 4))"""
  fig = gcm() if fig is None else fig
  if fig is not None:
    fig.setColorRange(colorRange)


def setColorScale(scale="linear", fig=None):
  """setColorScale('log'): logarithmic colours on every plot of the figure, 'linear' back"""
  fig = gcm() if fig is None else fig
  if fig is not None:
    fig.setColorScale(scale)


# MATLAB's compass names for where a legend sits
LEGEND_LOCATIONS = {"north": "upper center", "south": "lower center", "east": "center right", "west": "center left",
                    "northeast": "upper right", "northwest": "upper left", "southeast": "lower right",
                    "southwest": "lower left", "best": "best"}


def legend(*names, location="northeast", numColumns=1, fontSize=None, **kw):
  """a legend: of every artist a displayName named, or naming the curves of the current axes

    legend()                                         # a legend of every displayName of the figure
    legend('first', 'second')                        # names the curves in the order they were drawn
    legend(location='southwest', numColumns=2)       # another place, two columns
    legend(location='northoutside')                  # beside the charts, outside the axes
    legend('off')                                    # removes the legends

  Without names the legend collects every artist of the figure that a displayName (a
  label) named, each name once, MATLAB's legend('show'); with names it names the bars,
  curves and markers of the current axes in the order they were drawn, as MATLAB's legend
  with labels does. A new legend replaces those the figure has. It goes into the last axes,
  or for a location ending in 'outside' beside the charts, the figure widened when it would
  not fit. Every entry gets a marker of the same size. Other keywords go to matplotlib's
  legend.

  Input
    names - str, the names of the curves; 'off' removes the legends
  Output
    leg - the matplotlib legend, None for 'off'
  Keywords
    location - MATLAB's compass names, 'north', 'northeast', ..., 'best', or with 'outside'
        appended to place it beside the charts
    numColumns - the number of columns of the legend
    fontSize - the font size; by default the preference FontSize
  See also
    Legends, Annotations"""
  import matplotlib.pyplot as plt

  from ..geometry.session import getMTEXpref
  fig = plt.gcf()
  # legend('off') removes the legends of the axes; a new legend replaces the ones the figure
  # has, inside an axes or beside the charts
  for ax in fig.axes:
    if ax.get_legend() is not None:
      ax.get_legend().remove()
  if names == ("off",):
    return None
  names = [n for n in names if n != "show"]
  for leg in list(fig.legends):
    leg.remove()
  if names:
    # the MTEX figure's last chart, unless a plain figure opened after it is the current one
    m = gcm()
    ax = m.children[-1] if m is not None and m.children and m.fig is fig else plt.gca()
    handles, labels = _drawnInOrder(ax)[:len(names)], list(names)
  else:
    seen, handles, labels = set(), [], []
    for ax in fig.axes:
      for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in seen:
          seen.add(l)
          handles.append(h)
          labels.append(l)
  refuseVariants(kw)
  kw.setdefault("ncols", int(numColumns))
  kw.setdefault("fontsize", getMTEXpref("FontSize", 13) if fontSize is None else fontSize)
  where = str(kw.pop("location", location)).lower()
  if where.endswith("outside"):
    leg = _sameSize(fig.legend(handles, labels, loc="outside " + LEGEND_LOCATIONS.get(where[:-7], "lower center"), **kw))
    # a legend outside the charts widens the figure it would not fit
    width = leg.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    w, h = fig.get_size_inches()
    if width + 0.2 > w:
      fig.set_size_inches(width + 0.2, h)
    return leg
  ax = fig.axes[-1] if fig.axes else plt.gca()
  leg = _sameSize(ax.legend(handles, labels, loc=LEGEND_LOCATIONS.get(where, where), **kw))
  # above the labels of a chart, which sit on its rim in white boxes
  leg.set_zorder(100)
  return leg


# the bars, lines and markers of an axes in the order they were drawn, a set of bars as one
def _drawnInOrder(ax):
  order = {id(a): k for k, a in enumerate(ax.get_children())}
  items = [(order.get(id(c.patches[0]), 0), c) for c in ax.containers if len(c.patches)]
  # what is drawn as _nolegend_, as MATLAB's IconDisplayStyle off, takes no name
  items += [(order.get(id(a), 0), a) for a in list(ax.lines) + list(ax.collections) if a.get_label() != "_nolegend_"]
  return [a for _, a in sorted(items, key=lambda t: t[0])]


# every entry of a legend gets the same marker, whatever size the markers it stands for have
def _sameSize(leg, size=36.0):
  for h in leg.legend_handles:
    if hasattr(h, "set_sizes"):
      h.set_sizes([size])
  return leg


def firstOfList(f):
  """a list of functions is drawn by its first one, with MATLAB's warning"""
  if getattr(f, "shape", ()):
    import warnings
    warnings.warn("You try to plot a vector valued function. Plot the desired components manually. "
                  "In the following the first component is plotted.")
    return f[(0,) * len(f.shape)]
  return f


__all__ = ["saveFigure", "logAxis", "mtexFigure", "newMtexFigure", "gcm", "hold", "legend", "nextAxis", "mtexTitle", "xlabel", "ylabel", "mtexColorbar",
           "setColorRange", "setColorScale"]
