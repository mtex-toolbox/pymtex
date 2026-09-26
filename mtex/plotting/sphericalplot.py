# the matplotlib drawing of spherical plots: the axes with its boundary, grid and labels,
# markers, filled functions, arrows, outlines, pole figures (MATLAB @sphericalPlot,
# @vector3d/plot, scatter, text, quiver, @S2Fun/plot, contourf, pcolor, @SO3Fun/plotPDF,
# plotIPDF; docs/plans/plotting.md S1)
#
# Every function here asks spherical.py for the geometry - which charts, the projected
# points, the grid, the boundary, the labels - and hands matplotlib the plane coordinates.
# matplotlib is imported inside the functions, so that import mtex costs nothing. One
# sphericalPlot per axes owns the projection, the boundary, the grid and the labels, and
# is kept on the axes so that a second plot into it reuses them.
from __future__ import annotations

import re

import numpy as np

from ..geometry.frames import crystalFrame
from ..geometry.grids.s2grid import equispacedS2Grid
from ..geometry.session import getMTEXpref
from ..geometry.crystalshape import crystalShape
from ..geometry.orientation import orientation
from ..geometry.plane import plane3d
from ..geometry.sphericalregion import projectIntoRegion, sphericalRegion
from ..geometry.units import degree
from ..geometry.vector3d import vector3d, xvector, yvector, zvector
from .colormaps import contourLevels, mtexColormap
from .mtexfigure import gcm, mtexFigure
from .spherical import hasSymmetry, makeSphericalProjection, plainProjection

PROJECTION_KEYS = ("projection", "upper", "lower", "complete", "antipodal", "how2plot", "fundamentalRegion")
AXES_KEYS = ("grid", "grid_res", "noLabel", "fontSize")
WHERE_KEYS = ("parent", "add2all", "figSize", "layout", "marginx", "marginy")
MARKER_COLOR = (0.0, 0.447, 0.741)
BOUNDARY_COLOR = (0.35, 0.35, 0.35)
GRID_COLOR = (0.8, 0.8, 0.8)


# a negative index as Miller.char writes it, the digit followed by a combining overline, set
# as mathtext's bar, which matplotlib places over the digit; the combining character it
# leaves beside it. A run of them is one formula, and a text that is mathtext already stays.
_BARRED = re.compile("((?:.\u0305)+)")


def plotText(s):
  s = str(s)
  if "\u0305" not in s or "$" in s:
    return s
  return _BARRED.sub(lambda m: "$" + "".join(r"\bar{%s}" % c for c in m.group(0)[::2]) + "$", s)


class sphericalPlot:
  """the spherical plot of one axes: its projection, boundary, grid and labels

    sP = sphericalPlot(ax, proj)
    sP.proj, sP.sR, sP.antipodal, sP.bounds, sP.minData, sP.maxData, sP.mappable
  """

  def __init__(self, ax, proj, grid=False, grid_res=30 * degree, noLabel=False, fontSize=None, title=None):
    self.ax, self.proj = ax, proj
    self.fontSize = getMTEXpref("FontSize", 13) if fontSize is None else fontSize
    self.minData, self.maxData = np.nan, np.nan
    self.mappable = None
    self.fill, self.refill = [], None
    self.grid, self.labels, self.boundary = [], [], []
    ax._mtexSphericalPlot = self
    ax.set_aspect("equal")
    if proj.isPlain:
      ax.set_xlabel("ρ", fontsize=self.fontSize)
      ax.set_ylabel("θ", fontsize=self.fontSize)
      ax.invert_yaxis()
    else:
      ax.set_axis_off()
    outside = isinstance(proj.frame, crystalFrame) and proj.sR.vertices().size > 0
    margin = 0.0 if proj.isPlain else 0.2 if outside else 0.1
    self.bounds = proj.bounds(margin)
    xlo, xhi, ylo, yhi = self.bounds
    ax.set_xlim(xlo, xhi)
    ax.set_ylim((yhi, ylo) if proj.isPlain else (ylo, yhi))
    # the grid lies over the data, as MATLAB's, under the boundary and the markers
    if grid:
      for x, y in proj.gridLines(grid_res):
        self.grid += ax.plot(x, y, color=GRID_COLOR, linewidth=0.5, zorder=4)
    for x, y in proj.boundary():
      self.boundary += ax.plot(x, y, color=BOUNDARY_COLOR, linewidth=1.5, zorder=5)
    if not noLabel:
      self._drawLabels(outside)
    if title is not None:
      ax.set_title(plotText(title), fontsize=self.fontSize)

  # a crystal sector hangs its labels at the corners, centred over a corner in the upper half
  # and under one in the lower, so that they take no width beside the sector; a corner at half
  # height takes its label beside it, away from the centre. Any other chart writes its axes
  # names on the rim in a white box
  def _drawLabels(self, outside):
    xlo, xhi, ylo, yhi = self.bounds
    cx, cy = (xlo + xhi) / 2, (ylo + yhi) / 2
    for v, name in self.proj.labels():
      x, y = (float(c) for c in self.proj.projectUnchecked(v))
      if outside and not self.proj.isPlain:
        if abs(y - cy) > 0.05 * (yhi - ylo):
          up = y > cy
          xytext, ha, va = (0, 5 if up else -5), "center", "bottom" if up else "top"
        else:
          right = x > cx
          xytext, ha, va = (6 if right else -6, 0), "left" if right else "right", "center"
        self.labels.append(self.ax.annotate(plotText(name), (x, y), xytext=xytext, textcoords="offset points",
                                            ha=ha, va=va, fontsize=self.fontSize, zorder=8))
      else:
        self.labels.append(self.ax.text(x, y, plotText(name), ha="center", va="center", fontsize=self.fontSize, zorder=8,
                                        bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none")))

  @property
  def sR(self):
    return self.proj.sR

  @property
  def antipodal(self):
    return self.proj.antipodal

  def updateMinMax(self, values):
    vals = np.asarray(values, dtype=float).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size:
      self.minData = np.nanmin([self.minData, vals.min()])
      self.maxData = np.nanmax([self.maxData, vals.max()])

  # the data range in the corners, MATLAB's TL and BL annotations
  def annotateMinMax(self):
    if np.isnan(self.minData):
      return
    for y, name, val in ((1, "Max:", self.maxData), (0, "Min:", self.minData)):
      self.ax.text(0, y, f"{name}\n{val:.3g}", transform=self.ax.transAxes, ha="left", va="top" if y else "bottom",
                   fontsize=self.fontSize - 1, zorder=8)

  def hemisphereLabel(self):
    self.ax.text(1, 1, "upper" if self.proj.upper else "lower", transform=self.ax.transAxes, ha="right", va="top",
                 fontsize=self.fontSize - 1, zorder=8)


# ------------------------------------------------------------------ where a plot goes

def _split(kw):
  # MATLAB's 'silent' says nothing here, and its line keys are spelt lineWidth, lineColor
  kw.pop("silent", None)
  for old, new in (("LineWidth", "lineWidth"), ("LineColor", "lineColor"), ("LineStyle", "lineStyle"), ("linewidth", "lineWidth"),
                   ("linecolor", "lineColor"), ("linestyle", "lineStyle"), ("FontSize", "fontSize")):
    if old in kw:
      kw[new] = kw.pop(old)
  projKw = {k: kw.pop(k) for k in PROJECTION_KEYS if k in kw}
  axesKw = {k: kw.pop(k) for k in AXES_KEYS if k in kw}
  whereKw = {k: kw.pop(k) for k in WHERE_KEYS if k in kw}
  return projKw, axesKw, whereKw


# the spherical plots a command draws into: the axes named by parent, every axes of the
# current figure for add2all, the current axes when it is held, else a new figure with one
# axes per chart the data need
def _targets(v, projKw, axesKw, whereKw, titles=None):
  parent = whereKw.get("parent")
  if parent is not None:
    axes = list(parent) if isinstance(parent, (list, tuple)) else [parent]
    return [_plotOf(ax, v, projKw, axesKw) for ax in axes]
  fig = gcm()
  if whereKw.get("add2all") and fig is not None:
    sPs = [ax._mtexSphericalPlot for ax in fig.children if hasattr(ax, "_mtexSphericalPlot")]
    if sPs:
      return sPs
  # a held plot draws into the axes nextAxis named, else into the current one
  if fig is not None and fig.hold and fig.pending is not None and hasattr(fig.pending, "_mtexSphericalPlot"):
    return [fig.takeAxis()._mtexSphericalPlot]
  # a blank axes nextAxis prepared takes a held plot too
  if fig is not None and fig.hold and fig.pending is None and fig.current is not None and fig.current is not fig.children[-1] and \
     hasattr(fig.current, "_mtexSphericalPlot"):
    return [fig.current._mtexSphericalPlot]
  if fig is not None and fig.hold and fig.pending is None and fig.children and hasattr(fig.children[-1], "_mtexSphericalPlot"):
    # the two hemispheres of one sphere take a held plot together, each clipping its own
    sPs = [ax._mtexSphericalPlot for ax in fig.children if hasattr(ax, "_mtexSphericalPlot")]
    if len(sPs) == 2 and {bool(getattr(sPs[0].proj, "upper", False)), bool(getattr(sPs[1].proj, "upper", False))} == {True, False} and len(fig.children) == 2:
      return sPs
    return [fig.children[-1]._mtexSphericalPlot]
  projs = makeSphericalProjection(v, **projKw)
  # the axes nextAxis prepared takes the plot, a fresh figure of newMtexFigure its charts; else
  # a new figure, laid out as asked
  if fig is not None and fig.pending is not None and len(projs) == 1:
    return [_plotOf(fig.takeAxis(), v, projKw, axesKw)]
  # the two hemispheres go into the axes prepared and the one after it
  if fig is not None and fig.pending is not None and len(projs) == 2:
    sPs = [sphericalPlot(fig.takeAxis() if i == 0 else fig.nextAxis(), p, **axesKw) for i, p in enumerate(projs)]
    for sP in sPs:
      sP.hemisphereLabel()
    return sPs
  if fig is not None and not fig.children and not fig.hold:
    sPs = [sphericalPlot(fig.nextAxis(), p, **axesKw, title=None if titles is None else titles[i]) for i, p in enumerate(projs)]
    if len(projs) == 2:
      for sP in sPs:
        sP.hemisphereLabel()
    return sPs
  fig = mtexFigure(len(projs), figSize=whereKw.get("figSize"), layout=whereKw.get("layout"), marginx=whereKw.get("marginx"),
                    marginy=whereKw.get("marginy"))
  sPs = [sphericalPlot(fig.nextAxis(), p, **axesKw, title=None if titles is None else titles[i]) for i, p in enumerate(projs)]
  if len(projs) == 2:
    for sP in sPs:
      sP.hemisphereLabel()
  return sPs


# the charts drawn into the axes named by parent, one each, as MATLAB's subplot and
# 'parent' arrange independent plots; the figure of those axes becomes the current one
def intoParents(parent, charts, axesKw):
  axes = list(parent) if isinstance(parent, (list, tuple)) else [parent]
  fig = gcm()
  if fig is None or fig.fig is not axes[0].figure:
    fig = mtexFigure(figure=axes[0].figure)
  sPs = []
  for ax, (p, t) in zip(axes, charts):
    sP = getattr(ax, "_mtexSphericalPlot", None)
    sPs.append(sP if sP is not None else sphericalPlot(ax, p, **axesKw, title=t))
    if ax not in fig.children:
      fig.children.append(ax)
  return fig, sPs


def _plotOf(ax, v, projKw, axesKw):
  sP = getattr(ax, "_mtexSphericalPlot", None)
  if sP is None:
    sP = sphericalPlot(ax, makeSphericalProjection(v, **projKw)[0], **axesKw)
    fig = gcm()
    # an axes of the figure joins its children; an inset drawn into another axes does not
    if fig is not None and ax not in fig.children and ax in fig.fig.axes:
      fig.children.append(ax)
  return sP


# ------------------------------------------------------------------ markers

MARKER_NAMES = {"square": "s", "diamond": "D", "circle": "o", "triangle": "^", "pentagram": "p", "hexagram": "h", "plus": "+",
                "cross": "x", "point": ".", "asterisk": "*"}
MARKER_KEYS = {"MarkerSize": "size", "Marker": "marker", "MarkerFaceColor": "color", "MarkerColor": "color",
               "MarkerEdgeAlpha": "alpha", "MarkerFaceAlpha": "alpha", "MarkerAlpha": "alpha", "alpha": "alpha",
               "MarkerEdgeColor": "edgecolor", "color": "color", "markersize": "size", "marker": "marker",
               "markerEdgeColor": "edgecolor", "markerFaceColor": "color", "markerColor": "color", "markeredgecolor": "edgecolor",
               "markerfacecolor": "color", "markerSize": "size", "Markersize": "size"}


def _markers(sP, v, data=None, label=None, rgb=None, colormap=None, colorRange=None, zorder=6, **kw):
  # 'all' asks for every point, and 'filled' for a filled marker, which is what is drawn anyway
  kw.pop("all", None)
  kw.pop("filled", None)
  # labeled writes each direction's own name, backgroundColor boxes the text, textAboveMarker lifts it
  if kw.pop("labeled", False) and label is None:
    label = [_nameOf(v.flatten()[i]) for i in range(v.size)]
  background = kw.pop("backgroundColor", kw.pop("backgroundcolor", kw.pop("BackgroundColor", None)))
  # the text goes below the marker unless textAboveMarker, MATLAB's textBelowMarker being that default
  above = kw.pop("textAboveMarker", False) and not kw.pop("textBelowMarker", False)
  kw.pop("textBelowMarker", None)
  # MATLAB's textColor colours the labels; beside a MarkerFaceColor, color is the text's too
  textColor = kw.pop("textColor", kw.pop("fontColor", kw.pop("Color", None)))
  labelFontSize = kw.pop("labelFontSize", None)
  if textColor is None and label is not None and "color" in kw and ("MarkerFaceColor" in kw or "MarkerColor" in kw):
    textColor = kw.pop("color")
  # the markers shrink with the spacing of many directions, as MATLAB's do; the spacing is
  # that of as many directions spread evenly
  sizeGiven = any(key in kw for key, name in MARKER_KEYS.items() if name == "size")
  size = getMTEXpref("markerSize", 10)
  if not sizeGiven and v.size > 4:
    size = min(size, 50 * max(np.sqrt(4 * np.pi / v.size / (1 + v.antipodal)), 0.5 * degree))
  opts = {"size": size, "marker": "o", "color": MARKER_COLOR, "edgecolor": "k"}
  # an open marker, MarkerFaceColor 'none', still takes the colour of the order, on its edge
  faceNone = any(name == "color" and isinstance(kw.get(key), str) and kw[key].lower() == "none" for key, name in MARKER_KEYS.items())
  edgeGiven = any(key in kw for key, name in MARKER_KEYS.items() if name == "edgecolor")
  ownColor = any(key in kw for key, name in MARKER_KEYS.items() if name == "color") and not faceNone
  for key, name in MARKER_KEYS.items():
    if key in kw:
      opts[name] = kw.pop(key)
  # every further set of markers in one chart takes the next colour of the order, as MATLAB's
  # held plots do; text alone, markers of size zero, takes none
  if not ownColor and data is None and rgb is None and opts["size"]:
    from .colormaps import MTEX_COLORS
    k = getattr(sP, "markerSets", 0)
    opts["color"] = tuple(MTEX_COLORS[k % len(MTEX_COLORS)])
    sP.markerSets = k + 1
  if faceNone:
    if not edgeGiven:
      opts["edgecolor"] = opts["color"]
    opts["color"] = "none"
  # a marker is edged in its own colour unless told otherwise, as MATLAB's scatter does
  elif not edgeGiven:
    opts["edgecolor"] = "face"
  # MATLAB's marker names; a line width is the width of the marker edge
  opts["marker"] = MARKER_NAMES.get(str(opts["marker"]).lower(), opts["marker"])
  edgeWidth = kw.pop("lineWidth", kw.pop("linewidth", kw.pop("LineWidth", None)))
  # MATLAB's DisplayName names the markers in a legend
  legendName = kw.pop("DisplayName", kw.pop("displayName", None))
  # an axis on the rim of a hemisphere chart is marked at both ends; values stay one per point
  if v.antipodal and v.size and data is None and rgb is None:
    rim = np.abs((sP.proj.pC.rot.inv() * v).z.reshape(-1)) < 1e-6
    if np.any(rim):
      xyz = v.data.reshape(-1, 3)
      v = v._new(np.concatenate([xyz, -xyz[rim]]), antipodal=False)
      data = None if data is None else np.concatenate([np.asarray(data).reshape(-1), np.asarray(data).reshape(-1)[rim]])
      rgb = None if rgb is None else np.concatenate([np.asarray(rgb, dtype=float).reshape(-1, 3), np.asarray(rgb, dtype=float).reshape(-1, 3)[rim]])
      if label is not None and not isinstance(label, str):
        texts = list(np.asarray(label, dtype=object).ravel())
        label = texts + [t for t, r in zip(texts, rim) if r]
  x, y = sP.proj.project(v)
  x, y = np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()
  keep = np.isfinite(x)
  if not np.any(keep):
    return None
  ax = sP.ax
  size = opts["size"]
  # one size for all, a diameter, or one per marker, an area as MATLAB's scatter reads it;
  # the edge thins with the marker, so a small marker still shows its face; an open marker is
  # all edge and keeps it
  sizes = np.asarray(size, dtype=float)
  area = sizes ** 2
  if sizes.ndim > 0 and sizes.size > 1:
    area = sizes.reshape(-1)[keep] if sizes.size == len(x) else sizes.reshape(-1)
    sizes = np.sqrt(area)
  small = float(np.min(sizes)) if sizes.size else float(sizes)
  common = dict(marker=opts["marker"], s=area, edgecolors=opts["edgecolor"] if small >= 2 or faceNone else "none",
                linewidths=0.5 * min(1.0, max(small, 3.0 if faceNone else 0.0) / 6) if edgeWidth is None else edgeWidth, zorder=zorder,
                alpha=opts.get("alpha"), label=legendName, **kw)
  if data is not None:
    vals = np.asarray(data, dtype=float).ravel()[keep]
    sP.updateMinMax(vals)
    lo, hi = (np.nanmin(vals), np.nanmax(vals)) if colorRange is None else colorRange
    # a marker coloured by its value is edged in the same colour
    if not edgeGiven:
      common["edgecolors"] = "face"
    h = ax.scatter(x[keep], y[keep], c=vals, cmap=mtexColormap(colormap), vmin=lo, vmax=hi, **common)
    sP.mappable = h
  elif rgb is not None:
    h = ax.scatter(x[keep], y[keep], c=np.asarray(rgb, dtype=float).reshape(-1, 3)[keep], **common)
  else:
    color = opts["color"]
    # a colour per marker is thinned with the markers; one RGB triple is a single colour,
    # never a value per marker
    perMarker = len(np.shape(color)) == 2 and len(color) == len(x)
    perMarker = perMarker or (isinstance(color, (list, tuple)) and len(color) == len(x) and all(isinstance(c, str) for c in color))
    if perMarker:
      color = [color[i] for i in np.flatnonzero(keep)]
    if isinstance(color, str) and color.lower() == "none":
      h = ax.scatter(x[keep], y[keep], facecolors="none", **common)
    elif perMarker:
      h = ax.scatter(x[keep], y[keep], c=color, **common)
    else:
      h = ax.scatter(x[keep], y[keep], color=color, **common)
  if label is not None:
    texts = [label] * len(x) if isinstance(label, str) else [str(t) for t in np.asarray(label, dtype=object).ravel()]
    # MATLAB's tex interpreter: v_1 and x^2 are set as sub- and superscripts
    texts = [f"${t}$" if ("_" in t or "^" in t) and "$" not in t and " " not in t else plotText(t) for t in texts]
    bbox = None if background is None else dict(facecolor=background, edgecolor="none", pad=1)
    # text alone, on markers of size zero, stands on its point rather than below it
    alone = np.ndim(size) == 0 and float(size) == 0
    for i in np.flatnonzero(keep):
      ax.annotate(texts[i], (x[i], y[i]), xytext=(0, 0 if alone else size / 2 + 1 if above else -size / 2 - 1), textcoords="offset points",
                  ha="center", va="center" if alone else "bottom" if above else "top",
                  fontsize=sP.fontSize - 2 if labelFontSize is None else labelFontSize, zorder=8, bbox=bbox, color=textColor or "k")
  return h


# the name a direction is labelled with: its indices in a crystal, x, y, z and their negatives
# in a specimen, else its coordinates
def _nameOf(w):
  if w.isCrystalDirection:
    return w.char(family=False)
  d = w.normalize().data.reshape(3)
  for name, e in (("x", [1, 0, 0]), ("y", [0, 1, 0]), ("z", [0, 0, 1])):
    if np.allclose(d, e, atol=1e-6):
      return name
    if np.allclose(d, -np.array(e, dtype=float), atol=1e-6):
      return "-" + name
  return str(w)


# ------------------------------------------------------------------ filled functions

# a function's values on the chart grid, the frame of the function put on the directions
def _valuesOn(f, V):
  fr = getattr(f, "frame", None)
  framed = vector3d(V.data, frame=fr, isNormalized=True) if fr is not None else vector3d(V.data, isNormalized=True)
  return np.real(np.asarray(f(framed), dtype=complex)).reshape(V.shape)


# the columns of the grid between its blank ones
def _blocks(X):
  blank = ~np.all(np.isfinite(X), axis=0)
  edges = np.flatnonzero(np.diff(np.concatenate([[True], blank, [True]]).astype(int)))
  return [(edges[k], edges[k + 1]) for k in range(0, len(edges), 2) if edges[k + 1] - edges[k] >= 2]


def _filled(sP, X, Y, values, style="pcolor", contours=10, colormap=None, colorRange=None, log=False,
            lineColor=None, lineWidth=0.5, contourLabels=False, zorder=2, **kw):
  from matplotlib.colors import LogNorm, Normalize
  # MATLAB's 'logarithmic' colour scale
  log = kw.pop("logarithmic", log)
  # MATLAB's 'ShowText', 'on' labels the contour lines
  showText = kw.pop("ShowText", kw.pop("showText", None))
  if showText is not None:
    contourLabels = showText.lower() == "on" if isinstance(showText, str) else bool(showText)
  # MATLAB's faceAlpha is the transparency of the surface
  for name in ("faceAlpha", "FaceAlpha"):
    if name in kw:
      kw["alpha"] = kw.pop(name)
  # the colour map is fixed when the surface is drawn, a later preference leaves it alone
  colormap = mtexColormap(colormap)
  # the bands of a contour plot are cut for one range and one colour map, so a change of
  # either draws it again
  again = dict(style=style, contours=contours, log=log, lineColor=lineColor, lineWidth=lineWidth,
               contourLabels=contourLabels, zorder=zorder, **kw)
  # a line contour is an overlay and leaves the surface under it; a surface replaces the
  # surface that is there, so that a new colour range draws it again
  slot = "lines" if style == "contour" else "fill"
  for artist in getattr(sP, slot, []):
    artist.remove()
  setattr(sP, slot, [])
  bucket = getattr(sP, slot)
  sP.updateMinMax(values)
  lo, hi = (sP.minData, sP.maxData) if colorRange is None else colorRange
  if not hi > lo:
    hi = lo + max(abs(lo), 1.0) * 1e-6

  levels = contourLevels(lo, hi, contours, log=log) if np.isscalar(contours) else np.asarray(contours, dtype=float)

  # a new colour range or map recolours the bands and keeps their levels, as MATLAB's
  # setColorRange moves the colour limits of a contour plot and not its contours
  def refill(colorRange=(lo, hi), colormap=colormap):
    return _filled(sP, X, Y, values, colorRange=colorRange, colormap=colormap, **dict(again, contours=levels))

  sP.refill = refill if style in ("contourf", "contour") else None
  cmap = mtexColormap(colormap)
  norm = LogNorm(lo, hi) if log and lo > 0 else Normalize(lo, hi)
  ax = sP.ax
  h = None
  for j0, j1 in _blocks(X):
    Xb, Yb, Cb = X[:, j0:j1], Y[:, j0:j1], values[:, j0:j1]
    if style == "pcolor":
      h = ax.pcolormesh(Xb, Yb, np.ma.masked_invalid(Cb), cmap=cmap, norm=norm, shading="gouraud", zorder=zorder, **kw)
      bucket.append(h)
    elif style == "contourf":
      # a band takes the colour of its lower level, as MATLAB's contourf does, so the lowest
      # band of a map that starts white stays white; below the lowest level nothing is filled
      from matplotlib.colors import BoundaryNorm, ListedColormap
      bands = ListedColormap([(1.0, 1.0, 1.0, 0.0)] + [cmap(norm(l)) for l in levels[:-1]] + [cmap(1.0)], name=cmap.name)
      h = ax.contourf(Xb, Yb, np.ma.masked_invalid(Cb), levels=levels, cmap=bands, norm=BoundaryNorm(levels, bands.N, extend="both"),
                      extend="both", zorder=zorder, **kw)
      # the colour bar shows the map over the range continuously, as MATLAB's does
      h._mtexBar = (cmap, norm)
      bucket.append(h)
    if style in ("contourf", "contour"):
      colors = lineColor if lineColor is not None else "k" if style == "contourf" else None
      cs = ax.contour(Xb, Yb, np.ma.masked_invalid(Cb), levels=levels, colors=colors, linewidths=lineWidth, zorder=zorder + 1,
                      linestyles="solid", **({} if colors is not None else dict(cmap=cmap, norm=norm)))
      bucket.append(cs)
      if contourLabels:
        ax.clabel(cs, fontsize=sP.fontSize - 3)
      h = h if h is not None else cs
  sP.mappable = h if h is not None else sP.mappable
  # a surface drawn in blocks is recoloured block by block
  if style == "pcolor" and len(bucket) > 1:
    sP.mappables = list(bucket)
  elif h is not None and hasattr(sP, "mappables"):
    del sP.mappables
  return h


def _drawFunction(sP, f, style="pcolor", resolution=1 * degree, **kw):
  V, X, Y = sP.proj.grid(resolution)
  if V.size == 0:
    return None
  return _filled(sP, X, Y, _valuesOn(f, V), style=style, **kw)


# ------------------------------------------------------------------ arrows

# arrows in the plane: the tangent vector w at v carried through the projection, the
# longest one arrowSize times the grid spacing, all of that length with normalize
def _arrows(sP, v, w, resolution=10 * degree, arrowSize=0.5, normalize=False, color="k", axisField=False, zorder=6, lineWidth=1.0, **kw):
  from matplotlib.collections import LineCollection
  eps = 1e-4
  x0, y0 = sP.proj.project(v)
  x1, y1 = sP.proj.project(v.normalize() + eps * w)
  x2, y2 = sP.proj.project(v.normalize() - eps * w)
  x0, y0, x1, y1, x2, y2 = (np.asarray(a, dtype=float).ravel() for a in (x0, y0, x1, y1, x2, y2))
  # a step over the rim of the chart lands far away or nowhere, so the shorter one-sided
  # difference is the derivative
  forward = np.hypot(x1 - x0, y1 - y0)
  backward = np.hypot(x0 - x2, y0 - y2)
  useBackward = ~np.isfinite(forward) | (np.isfinite(backward) & (backward < forward))
  x1, y1 = np.where(useBackward, 2 * x0 - x2, x1), np.where(useBackward, 2 * y0 - y2, y1)
  keep = np.isfinite(x0) & np.isfinite(x1)
  if not np.any(keep):
    return None
  dx, dy = (x1 - x0)[keep] / eps, (y1 - y0)[keep] / eps
  length = np.hypot(dx, dy)
  dx, dy = dx / np.where(length > 0, length, 1.0), dy / np.where(length > 0, length, 1.0)
  # as MATLAB's quiver: the tangential part at its length, the longest vector of the whole
  # field half the grid spacing, so a direction almost normal to the sphere draws almost nothing
  unit = resolution / degree if sP.proj.isPlain else resolution
  if normalize:
    size = np.ones(len(dx))
  else:
    V = v.normalize().data.reshape(-1, 3)
    W = np.broadcast_to(np.asarray(w.data if isinstance(w, vector3d) else w, dtype=float).reshape(-1, 3), V.shape)
    full = np.linalg.norm(W, axis=1)
    size = np.linalg.norm(W - np.sum(W * V, axis=1, keepdims=True) * V, axis=1)[keep]
    size = size / (np.max(full) if np.max(full) > 0 else 1.0)
  dx, dy = dx * size * arrowSize * unit, dy * size * arrowSize * unit
  x0, y0 = x0[keep], y0[keep]
  if axisField:
    segments = np.stack([np.stack([x0 - dx / 2, y0 - dy / 2], axis=1), np.stack([x0 + dx / 2, y0 + dy / 2], axis=1)], axis=1)
    h = LineCollection(segments, colors=color, linewidths=lineWidth, zorder=zorder, **kw)
    sP.ax.add_collection(h)
    return h
  return sP.ax.quiver(x0, y0, dx, dy, angles="xy", scale_units="xy", scale=1, color=color, width=0.004 * lineWidth, zorder=zorder, **kw)


# ------------------------------------------------------------------ the commands

# what kind of thing a plot command was given
def _isFunction(f):
  return callable(f) and not isinstance(f, vector3d) and hasattr(f, "frame") and not _isField(f)


def _isField(f):
  return type(f).__name__.startswith(("S2VectorField", "S2AxisField"))


def _mapKind(x):
  from .mapplot import _kind
  return _kind(x)


def _isAxisField(f):
  return type(f).__name__.startswith("S2AxisField")


def _isSO3Fun(f):
  return hasattr(f, "pdf") and hasattr(f, "CS") and hasattr(f, "SS")


FILL_FLAGS = ("contourf", "contour", "pcolor", "smooth")


# the style flag of a filled plot, popped from the keywords; MATLAB's 'contourf', n or
# 'contourf', levels also gives the number of contours or the levels themselves
def fillFlag(kw, default=None):
  for f in FILL_FLAGS:
    x = kw.pop(f, None)
    if x is None or (isinstance(x, (bool, np.bool_)) and not x):
      continue
    if not isinstance(x, (bool, np.bool_, str)):
      kw.setdefault("contours", int(x) if np.isscalar(x) else np.asarray(x, dtype=float))
    return f
  return default


def plot(v, *args, **kw):
  """plot(v), plot(v, data), plot(v, rgb), plot(v, w), plot(sF), plot(vF), plot(sR)

  Directions as markers; with data coloured by it, with rgb in those colours, with a second
  list of vectors as arrows; a function on the sphere as pcolor; a vector or axis field as
  arrows; a spherical region as its outline. Flags contourf, contour, pcolor, smooth draw
  the density of the directions or the interpolant of their data.
  """
  if isinstance(v, sphericalRegion):
    return plotRegion(v, **kw)
  from .volumeplot import gcs, is3dMap, plot3dMap, plotPoints3d
  if is3dMap(v):
    return plot3dMap(v, *args, **kw)
  # positions plotted into a held 3D scene are markers there
  if isinstance(v, vector3d) and not args and gcm() is not None and gcm().hold and gcs() is not None:
    return plotPoints3d(v, **kw)
  from ..geometry.frames import referenceFrame
  if isinstance(v, referenceFrame):
    return plotSymmetry(v, *args, **kw)
  from ..functions.kernelfunctions import S2Kernel, SO3Kernel
  from .orientationkeys import orientationColorKey
  if isinstance(v, S2Kernel):
    return plotKernel(v, **kw)
  if isinstance(v, SO3Kernel):
    from .kernelplot import plotSO3Kernel
    return plotSO3Kernel(v, **kw)
  from .directionkeys import directionColorKey
  if isinstance(v, directionColorKey):
    return v.plot(*args, **kw)
  if isinstance(v, orientationColorKey):
    if hasattr(v, "plot"):
      return v.plot(*args, **kw)
    from .sectionplot import plotKeySections
    return plotKeySections(v, *args, **kw)
  # an ODF draws its sections, or its values along a fibre
  if _isSO3Fun(v):
    from ..geometry.fibre import fibre
    if args and isinstance(args[0], fibre):
      from .fibreplot import plotFibre
      return plotFibre(v, *args, **kw)
    from .sectionplot import plotSection
    return plotSection(v, *args, **kw)
  if hasattr(v, "allH") and hasattr(v, "allR"):
    from .polefigureplot import plotPoleFigure
    return plotPoleFigure(v, *args, **kw)
  # three functions as the colour channels of one surface
  from .tensorplot import isRGB, plotRGB
  if isRGB(v):
    kw.pop("rgb", None)
    return plotRGB(v, *[a for a in args if not (isinstance(a, str) and a == "rgb")], **kw)
  # a tensor is drawn as its directional magnitude
  if hasattr(v, "rank") and hasattr(v, "M"):
    return plot(v.directionalMagnitude(), *args, **kw)
  # plot(ori) on a held section plot: the orientations into their sections
  if hasattr(v, "frameA") and hasattr(v, "frameB") and getattr(gcm(), "sections", None) is not None and (gcm().hold or kw.get("add2all")):
    from .sectionplot import plotSection
    kw.pop("add2all", None)
    return plotSection(v, gcm().sections, onto=True, **kw)
  # plot(ori, add2all=True) on pole figures: the poles of the orientations on every one
  if hasattr(v, "frameA") and hasattr(v, "frameB") and kw.get("add2all") and gcm() is not None and \
      any(getattr(getattr(ax, "_mtexSphericalPlot", None), "h", None) is not None for ax in gcm().children):
    # the markers shrink with the number of poles, as those of plotPDF(ori, h) do
    from .poles import dynamicMarkerSize
    if not any(k in kw for k, name in MARKER_KEYS.items() if name == "size"):
      kw["MarkerSize"] = dynamicMarkerSize(v.size * max(v.CS.numSym() // 2, 1))
    return _annotatePoles(v, **kw)
  # plot(f, add2all=True) on pole figures: the poles of the fibre as curves on every one
  from ..geometry.fibre import fibre
  if isinstance(v, fibre) and kw.get("add2all") and gcm() is not None and \
      any(getattr(getattr(ax, "_mtexSphericalPlot", None), "h", None) is not None for ax in gcm().children):
    return _fibrePoles(v, **kw)
  # the geometry of a section plot: its charts with the reference fields
  from .sections import ODFSections
  if isinstance(v, ODFSections):
    from .sectionplot import plotSections
    return plotSections(v, *args, **kw)
  # orientations, fibres and their region live in space, not on the sphere; a fibre goes
  # into the sections of the current figure when it has some
  from ..geometry.fibre import fibre
  if isinstance(v, fibre) and getattr(gcm(), "sections", None) is not None:
    from .sectionplot import plotFibreSection
    return plotFibreSection(v, *args, **kw)
  from ..geometry.quaternion import quaternion
  if isinstance(v, (fibre, quaternion)) or (hasattr(v, "frameA") and hasattr(v, "frameB")):
    from .spaceplot import plot3d
    return plot3d(v, *args, **kw)
  # a function on the circle and a characteristic shape are polar lines
  from ..functions.handles import S1FunHandle
  from ..functions.s1fun import S1FunHarmonic
  from ..maps.shape import shape2d
  if isinstance(v, (S1FunHarmonic, S1FunHandle)):
    from .polarplot import plotS1
    return plotS1(v, *args, **kw)
  if isinstance(v, shape2d):
    from .polarplot import plotShape
    return plotShape(v, *args, **kw)
  # an image on the specimen, or a sequence of them
  from ..maps.mapimage import mapImage
  if isinstance(v, mapImage) or (isinstance(v, (list, tuple)) and v and all(isinstance(m, mapImage) for m in v)):
    from .mapplot import plotMapImage
    kw.update({a: True for a in args if isinstance(a, str)})
    return plotMapImage(v, **kw)
  kind = _mapKind(v)
  if kind is not None:
    from . import mapplot
    f = {"ebsd": mapplot.plotEBSD, "grains": mapplot.plotGrains, "boundary": mapplot.plotBoundary,
         "triplePoints": mapplot.plotTriplePoints}[kind]
    return f(v, *args, **mapplot.mapAliases(kw, grains=kind == "grains"))
  if _isField(v):
    return quiver(v, *args, **kw)
  # a vector field on SO(3) as arrows in the sections of orientation space
  if type(v).__name__.startswith("SO3VectorField"):
    from .sectionplot import quiverSectionField
    return quiverSectionField(v, *args, **kw)
  if _isFunction(v):
    # MATLAB's positional flags, plot(sF, 'complete', 'smooth', 'upper', 'lower')
    kw.update({a: True for a in args if isinstance(a, str)})
    style = fillFlag(kw, "pcolor")
    # a list of functions: one panel for each, each on its own colour range
    if getattr(v, "shape", ()):
      fs = [v[k] for k in np.ndindex(v.shape)]
      kw.setdefault("colorRange", "individual")
      return _plotRow(fs, [None] * len(fs), style="pcolor" if style == "smooth" else style, **kw)
    return _plotFunction(v, style="pcolor" if style == "smooth" else style, **kw)
  # a key of two properties as its image
  if hasattr(v, "property2color"):
    from .planarkey import plotPlanarKey
    return plotPlanarKey(v, *args, **kw)
  # a plane in space, clipped by the axes
  if isinstance(v, plane3d):
    from .shapeplot import plotPlane
    return plotPlane(v, *args, **kw)
  # a crystal shape, possibly behind its position: plot(x, y, cS)
  if any(isinstance(a, crystalShape) for a in (v,) + args):
    from .shapeplot import plotShape
    return plotShape(v, *args, **kw)
  v = v if isinstance(v, vector3d) else vector3d(v)
  # MATLAB's positional flags, plot(v, 'contourf', 'fundamentalRegion')
  for a in [a for a in args if isinstance(a, str)]:
    kw[a] = True
  args = tuple(a for a in args if not isinstance(a, str))
  if kw.pop("plane", False):
    return _planeCircles(v, **kw)
  if args and isinstance(args[0], vector3d):
    return quiver(v, args[0], **kw)
  style = fillFlag(kw, None)
  data = args[0] if args else None
  if style is not None and data is not None and getattr(v, "_plotGrid", None) is not None:
    return _plotOnGrid(v, data, style="pcolor" if style == "smooth" else style, **kw)
  if style is not None:
    return _plotVectorDensity(v, data, style="pcolor" if style == "smooth" else style, **kw)
  return scatter(v, data, **kw)


# values on the directions of plotS2Grid, drawn where they lie on the charts of its hemisphere
def _plotOnGrid(v, data, style="pcolor", colorbar=False, colorRange=None, **kw):
  for key, value in v._plotGrid.items():
    kw.setdefault(key, value)
  projKw, axesKw, whereKw = _split(kw)
  values = np.asarray(data, dtype=float).reshape(v.shape)
  sPs = _targets(v, projKw, axesKw, whereKw)
  if colorRange is None:
    colorRange = _rangeOf([values])
  hs = []
  for sP in sPs:
    x, y = sP.proj.project(v)
    hs.append(_filled(sP, np.asarray(x, dtype=float), np.asarray(y, dtype=float), values, style=style, colorRange=colorRange, **kw))
  if colorbar:
    gcm().colorbar()
  return hs


# circle(v, radius): the small circle of opening angle radius about v, on the axes nextAxis
# chose or the last one, on every axes with add2all (MATLAB plotting/circle.m)
def circle(v, radius=np.pi / 2, add2all=False, lineColor="k", lineWidth=1.5, EdgeAlpha=1.0, parent=None, **kw):
  vs = v if isinstance(v, vector3d) else vector3d(v)
  t = np.linspace(0, 2 * np.pi, 361)
  rings = []
  for k in range(vs.size):
    n = vs.flatten()[k].normalize()
    u = n.orth()
    w = n.cross(u).normalize()
    rings.append(np.cos(radius) * n.data.reshape(1, 3) + np.sin(radius) * (np.cos(t)[:, None] * u.data.reshape(1, 3) + np.sin(t)[:, None] * w.data.reshape(1, 3)))
  return _curves(rings, vs.frame, vs.antipodal, add2all, lineColor, lineWidth, EdgeAlpha, parent, "circle", **kw)


def ellipse(rot, a, b, add2all=True, lineColor="k", lineWidth=1.5, EdgeAlpha=1.0, parent=None, **kw):
  """ellipse(rot, a, b): the ellipse of semi-axes a and b, in radians, about rot * z, its
  axes along rot * x and rot * y, onto the spherical plots of the figure"""
  from ..geometry.quaternion import rotation
  rots = rot.flatten() if rot.size > 1 else [rot]
  a, b = np.broadcast_to(np.asarray(a, dtype=float).ravel(), (len(rots),)), np.broadcast_to(np.asarray(b, dtype=float).ravel(), (len(rots),))
  rho = np.linspace(0, 2 * np.pi, 361)
  rings = []
  for r, ai, bi in zip(rots, a, b):
    # the polar distance of the ellipse from its centre, then turned into place
    theta = ai * bi / np.sqrt((ai * np.cos(rho)) ** 2 + (bi * np.sin(rho)) ** 2)
    c = rotation(r.data.reshape(4)) * vector3d.byPolar(theta, rho)
    rings.append(c.data.reshape(-1, 3))
  return _curves(rings, None, False, add2all, lineColor, lineWidth, EdgeAlpha, parent, "ellipse", **kw)


# rings of directions as lines on the spherical plots: the one drawn last, every one with
# add2all, or those named
def _curves(rings, frame, antipodal, add2all, lineColor, lineWidth, EdgeAlpha, parent, name, **kw):
  lineColor = kw.pop("color", kw.pop("linecolor", kw.pop("LineColor", kw.pop("lineColor", lineColor))))
  lineWidth = kw.pop("linewidth", kw.pop("LineWidth", lineWidth))
  kw.pop("doNotDraw", None)
  # MATLAB's DisplayName names the first curve on each axes in a legend
  legendName = kw.pop("DisplayName", kw.pop("displayName", None))
  fig = gcm()
  if parent is not None:
    axes = list(parent) if isinstance(parent, (list, tuple)) else [parent]
  elif fig is None or not fig.children:
    raise ValueError(f"{name} draws onto a spherical plot; plot the directions first")
  elif add2all:
    axes = [ax for ax in fig.children if hasattr(ax, "_mtexSphericalPlot")]
  else:
    ax = fig.takeAxis() if fig.pending is not None else fig.children[-1]
    axes = [ax]
  hs = []
  for ring in rings:
    for ax in axes:
      sP = ax._mtexSphericalPlot
      x, y = sP.proj.project(vector3d(ring, frame=frame, antipodal=antipodal))
      x, y = np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()
      # a curve leaving the chart, or folded to its antipode, is broken where it jumps
      jump = np.flatnonzero(np.hypot(np.diff(x), np.diff(y)) > 0.25 * np.nanmax(np.hypot(x, y)))
      x, y = np.insert(x, jump + 1, np.nan), np.insert(y, jump + 1, np.nan)
      named = legendName is not None and ring is rings[0]
      hs.append(ax.plot(x, y, color=lineColor, linewidth=lineWidth, alpha=EdgeAlpha, zorder=7, label=legendName if named else "_nolegend_", **kw))
  return hs


# plot(psi): a spherical kernel over the angle in degrees, into the held axes or a new figure
def plotKernel(psi, maxAngle=np.pi, symmetric=False, **kw):
  """plot(psi): a kernel on the sphere against the angle, over -180 to 180 degree with symmetric"""
  from .kernelplot import _lineAxes, _lineKw
  ax = _lineAxes(kw)
  kw = _lineKw(kw)
  kw.setdefault("linewidth", 1.5)
  omega = np.linspace(-maxAngle if symmetric else 0, maxAngle, 1441 if symmetric else 721)
  h = ax.plot(omega / degree, np.real(psi.eval(np.cos(omega))), **kw)
  ax.set_xlim(-maxAngle / degree if symmetric else 0, maxAngle / degree)
  ax.set_xlabel("angle in degree")
  return h


def scatter(v, data=None, *args, minmax=False, symmetrised=False, colorbar=False, **kw):
  """scatter(v), scatter(v, data), scatter(v, label=texts): directions as markers; scatter(ori)
  puts orientations into axis angle space; scatter(x, y, sz, c, 'filled') of two coordinate
  lists marks points on the current map"""
  from ..geometry._array import MtexArray
  from ..geometry.quaternion import quaternion
  if not isinstance(v, MtexArray) and data is not None and not isinstance(data, MtexArray) and np.ndim(v) == 1 \
     and np.shape(data) == np.shape(v):
    from .mapplot import scatterXY
    return scatterXY(v, data, *args, **kw)
  if isinstance(v, quaternion):
    from .spaceplot import scatterOrientations
    return scatterOrientations(v, data, **kw)
  v = v if isinstance(v, vector3d) else vector3d(v)
  projKw, axesKw, whereKw = _split(kw)
  # beside labels MATLAB's FontSize is the size of their text, not of the chart's
  if ("label" in kw or kw.get("labeled")) and "fontSize" in axesKw:
    kw["labelFontSize"] = axesKw.pop("fontSize")
  if symmetrised and v.isCrystalDirection:
    # every distinct equivalent once, so that no label is written twice in one place
    s = v.symmetrise(unique=data is None and v.size == 1)
    v, data = s.flatten(), None if data is None else np.tile(np.asarray(data).ravel(), s.shape[0])
  sPs = _targets(v, projKw, axesKw, whereKw)
  rgb = None
  if data is not None:
    arr = np.asarray(data)
    if arr.ndim == 2 and arr.shape[-1] == 3 and arr.shape[0] == v.size:
      rgb, data = arr, None
  hs = []
  for sP in sPs:
    w = projectIntoRegion(v, sP.sR) if projKw.get("fundamentalRegion") and hasSymmetry(v.frame) else v
    # an antipodal chart shows an equivalent inside it rather than the antipode of one
    if symmetrised and sP.antipodal and data is None and rgb is None and w.size > 1:
      w = _insideFirst(w, sP.sR)
    h = _markers(sP, w, data, rgb=rgb, **kw)
    hs.append(h)
    if minmax:
      sP.annotateMinMax()
  if colorbar and data is not None:
    gcm().colorbar()
  return hs


# the directions without those whose antipode is among them and inside the region
def _insideFirst(v, sR):
  xyz = v.normalize().data.reshape(-1, 3)
  inside = np.asarray(sR.checkInside(v.flatten())).reshape(-1)
  opposite = (xyz @ xyz.T) < -1 + 1e-6
  drop = ~inside & np.any(opposite & inside[None, :], axis=1)
  return v.flatten()[np.flatnonzero(~drop)]


def text(v, labels=None, *more, **kw):
  """text(v, labels): text at directions, text(v, labeled=True) each direction's own name;
  text(ebsd, labels): labels on the map; text(x, y, s) and text(x, y, z, s): text at a point
  of the current axes, as MATLAB's text"""
  if labels is None and not kw.get("labeled"):
    raise TypeError("text needs the labels or labeled=True")
  if more or isinstance(labels, (int, float, np.number)):
    return _textAt((v, labels) + more[:-1], more[-1], **kw)
  if _mapKind(v) is not None:
    from .mapplot import textMap
    return textMap(v, labels, **kw)
  kw.setdefault("MarkerSize", 0)
  # text is added to the charts of the current figure, as MATLAB's text adds to its axes,
  # or to the one nextAxis chose in a held figure
  fig = gcm()
  chosen = fig is not None and fig.hold and (fig.pending is not None or (fig.current is not None and bool(fig.children)
                                                                         and fig.current is not fig.children[-1]))
  if not chosen:
    kw.setdefault("add2all", True)
  return scatter(v, label=labels, **kw)


# a string at coordinates of the current axes, in space when the axes are 3D
def _textAt(xyz, s, **kw):
  import matplotlib.pyplot as plt
  fig = gcm()
  ax = fig.children[-1] if fig is not None and fig.children else plt.gca()
  # MATLAB's property names, capitalised or not
  names = {"fontweight": "fontweight", "fontsize": "fontsize", "color": "color", "horizontalalignment": "ha",
           "verticalalignment": "va", "backgroundcolor": "backgroundcolor"}
  kw = {names.get(k.lower(), k): val for k, val in kw.items()}
  return ax.text(*[float(c) for c in xyz], s, **kw)


# the great circles normal to directions, as lines through the charts
def _planeCircles(v, lineStyle="-", lineWidth=1.0, lineColor="k", **kw):
  from .space import smallCircle
  v = v if isinstance(v, vector3d) else vector3d(v)
  projKw, axesKw, whereKw = _split(kw)
  whereKw.setdefault("add2all", True)
  sPs = _targets(v, projKw, axesKw, whereKw)
  hs = []
  for n in (v.flatten() if v.ndim > 0 else [v]):
    circle = vector3d(smallCircle(n.data.reshape(3), np.pi / 2), frame=v.frame, isNormalized=True)
    for sP in sPs:
      x, y = sP.proj.project(circle)
      x, y = np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()
      hs += sP.ax.plot(x, y, linestyle=lineStyle, linewidth=lineWidth, color=lineColor, zorder=5)
  return hs


# plot(cs): the symmetry elements of a point group as the stereographic symbols, a polygon
# of n corners for an n-fold rotation axis, a lens for a twofold, a hollow one for a
# rotoinversion axis, a great circle for a mirror plane, a hollow circle for the inversion
# (MATLAB @referenceFrame/plot)
def plotHKL(cs, complete=False, **kw):
  """plotHKL(cs): the low index directions <100>, <110>, <111> of a crystal with all their
  symmetric equivalents, labelled, on the upper hemisphere unless complete (MATLAB
  crystalFrame/plotHKL)"""
  from ..geometry.miller import Miller
  hkl = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1), (-1, -1, 0), (-1, -1, 1))
  ms = Miller(np.array(hkl, dtype=float), cs).unique()
  opts = dict(symmetrised=True, labeled=True, MarkerEdgeColor="k", grid=True, backgroundColor="w", noLabel=True)
  if not complete:
    opts["upper"] = True
  opts.update(kw)
  # in MATLAB's order: <111>, <110>, <100> take the colours of the order
  ms = ms[::-1]
  hs = scatter(ms[0], **opts)
  # the other directions into the charts the first one made
  axes = [h.axes for h in hs if h is not None]
  opts.pop("figSize", None)
  for k in range(1, ms.size):
    hs += scatter(ms[k], parent=axes, **opts)
  return hs


def plotSymmetry(fr, symbolSize=None, mirrorLineWidth=2.5, **kw):
  from matplotlib.patches import Ellipse, Polygon
  from ..geometry.quaternion import orthogonalTransform
  ops = orthogonalTransform(fr).flatten()
  improper = ops.isImproper
  axes = ops.axis().data.reshape(-1, 3)
  omega = np.round(ops.angle() / degree)
  # the distinct axes up to sign, z up
  uaxes, ids = [], np.zeros(len(axes), dtype=int)
  for k, a in enumerate(axes):
    for j, u in enumerate(uaxes):
      if abs(abs(float(a @ u)) - 1) < 0.005:
        ids[k] = j
        break
    else:
      ids[k] = len(uaxes)
      uaxes.append(a if a[2] >= 0 else -a)
  projKw, axesKw, whereKw = _split(kw)
  projKw.update(upper=True, complete=True)
  axesKw.setdefault("noLabel", True)
  anchor = vector3d(0, 0, 1, frame=fr)
  sP = _targets(anchor, projKw, axesKw, whereKw)[0]
  ax = sP.ax
  d = max(ax.get_xlim()[1] - ax.get_xlim()[0], ax.get_ylim()[1] - ax.get_ylim()[0])
  s = 0.15 * (d / 3 if symbolSize is None else symbolSize)
  # the mirror planes as great circles
  mir = improper & (np.abs(ops.angle() - np.pi) < 1e-4)
  for a in axes[mir]:
    circle(vector3d(a, frame=fr), lineWidth=mirrorLineWidth, lineColor="k")

  def npoly(x, y, n, size, angle, **st):
    if n == 2:
      ax.add_patch(Ellipse((x, y), 0.8 * size, 2 * size, angle=angle / degree, zorder=7, **st))
    else:
      w = angle + np.pi + np.linspace(0, 2 * np.pi, n + 1)[:-1]
      ax.add_patch(Polygon(np.column_stack([x + size * np.cos(w), y + size * np.sin(w)]), closed=True, zorder=7, **st))

  for i, u in enumerate(uaxes):
    sel = ids == i
    properOrders = omega[sel & ~improper & (omega > 0)]
    improperOrders = omega[sel & improper & (omega > 0)]
    multiP = int(round(360 / properOrders.min())) if properOrders.size else None
    multiI = int(round(360 / improperOrders.min())) if improperOrders.size else None
    if multiP is None and multiI is None:
      continue
    multi = max(m for m in (multiP, multiI) if m is not None)
    for n in (u, -u):
      x, y = sP.proj.project(vector3d(n, frame=fr, isNormalized=True))
      x, y = float(np.asarray(x).ravel()[0]), float(np.asarray(y).ravel()[0])
      if not (np.isfinite(x) and np.isfinite(y)):
        continue
      rho = np.pi / 2 if np.hypot(x, y) < 1e-3 else np.arctan2(y, x)
      if multi > 2 or multiP is not None:
        npoly(x, y, multi, 1.1 * s, rho, facecolor="k", edgecolor="w", linewidth=1)
      if multiI is not None and (multiP is None or multiI > multiP):
        npoly(x, y, multi, 0.9 * s, rho, facecolor="w", edgecolor="none")
        if multiP is not None:
          npoly(x, y, multiP, 0.9 * s, rho, facecolor="k", edgecolor="none")
  if fr.isLaue:
    x, y = sP.proj.project(anchor)
    ax.add_patch(Ellipse((float(np.asarray(x).ravel()[0]), float(np.asarray(y).ravel()[0])), 0.8 * s, 0.8 * s, facecolor="w", edgecolor="k",
                         linewidth=2, zorder=8))
  # room for the symbols on the rim
  for lim, setter in ((ax.get_xlim(), ax.set_xlim), (ax.get_ylim(), ax.set_ylim)):
    c, h = 0.5 * (lim[0] + lim[1]), 0.5 * (lim[1] - lim[0])
    setter(c - 1.1 * h, c + 1.1 * h)
  return sP


# annotate: markers on every axes of the current figure, black by default
def annotate(v, *args, **kw):
  kw.setdefault("MarkerFaceColor", "k")
  kw.setdefault("add2all", True)
  # an orientation is marked in the sections of the current figure when it has some, or by
  # its poles on the pole figures the figure holds
  if hasattr(v, "frameA") and hasattr(v, "frameB"):
    if getattr(gcm(), "sections", None) is not None:
      from .sectionplot import annotateSection
      return annotateSection(v, *args, **kw)
    if gcm() is not None and any(getattr(getattr(ax, "_mtexSphericalPlot", None), "h", None) is not None or
                                 getattr(getattr(ax, "_mtexSphericalPlot", None), "r", None) is not None for ax in gcm().children):
      return _annotatePoles(v, **kw)
    # into the orientation space the figure shows, in its coordinates
    fig = gcm()
    if fig is not None and fig.children and hasattr(fig.children[-1], "_mtexSpacePlot"):
      from .spaceplot import scatterOrientations
      kw.pop("add2all", None)
      held, fig.hold = fig.hold, True
      try:
        return scatterOrientations(v, *args, **kw)
      finally:
        fig.hold = held
  # the great circle normal to a direction, drawn as a line on every axes
  if kw.pop("plane", False):
    return _planeCircles(v, **kw)
  if isinstance(v, sphericalRegion):
    return plotRegion(v, **kw)
  from ..functions.kernelfunctions import S2Kernel, SO3Kernel
  from .orientationkeys import orientationColorKey
  if isinstance(v, S2Kernel):
    return plotKernel(v, **kw)
  if isinstance(v, SO3Kernel):
    from .kernelplot import plotSO3Kernel
    return plotSO3Kernel(v, **kw)
  from .directionkeys import directionColorKey
  if isinstance(v, directionColorKey):
    return v.plot(*args, **kw)
  if isinstance(v, orientationColorKey):
    if hasattr(v, "plot"):
      return v.plot(*args, **kw)
    from .sectionplot import plotKeySections
    return plotKeySections(v, *args, **kw)
  # an ODF draws its sections, or its values along a fibre
  if _isSO3Fun(v):
    from ..geometry.fibre import fibre
    if args and isinstance(args[0], fibre):
      from .fibreplot import plotFibre
      return plotFibre(v, *args, **kw)
    from .sectionplot import plotSection
    return plotSection(v, *args, **kw)
  return plot(v, **kw)


# the density of directions, weighted on request, axial with MATLAB's 'antipodal'; or the
# interpolant of values given at them
def _plotVectorDensity(v, data=None, style="contourf", halfwidth=5 * degree, weights=None, **kw):
  from ..functions.density import calcDensity, interpolate
  if data is None:
    f = calcDensity(v, halfwidth=halfwidth, weights=weights, antipodal=bool(kw.get("antipodal", False)))
  else:
    f = interpolate(v, np.asarray(data, dtype=float).ravel())
  return _plotFunction(f, style=style, **kw)


# one colour range over the charts of one function, unless one is given
def _plotFunction(f, style="pcolor", resolution=1 * degree, minmax=False, colorbar=False, colorRange=None, **kw):
  projKw, axesKw, whereKw = _split(kw)
  fr = getattr(f, "frame", None)
  if projKw.get("complete", False):
    projKw.setdefault("antipodal", bool(getattr(f, "antipodal", False)))
    sPs = _targets(sphericalRegion(frame=fr), dict(projKw, symmetric=hasSymmetry(fr)), axesKw, whereKw)
  else:
    sPs = _targets(f, projKw, axesKw, whereKw)
  grids = [sP.proj.grid(resolution) for sP in sPs]
  values = [_valuesOn(f, V) if V.size else np.zeros((0, 0)) for V, _, _ in grids]
  if colorRange is None:
    colorRange = _levelRange(kw) or _rangeOf(values)
  hs = [_filled(sP, X, Y, vals, style=style, colorRange=colorRange, **kw) if vals.size else None
        for sP, (_, X, Y), vals in zip(sPs, grids, values)]
  for sP in sPs:
    if minmax:
      sP.annotateMinMax()
  if colorbar:
    gcm().colorbar()
  return hs


# the range of contour levels given explicitly, None for a number of them
def _levelRange(kw):
  levels = kw.get("contours")
  if levels is None or np.isscalar(levels):
    return None
  levels = np.asarray(levels, dtype=float)
  return (float(levels.min()), float(levels.max()))


def _rangeOf(values):
  finite = np.concatenate([np.ravel(v)[np.isfinite(np.ravel(v))] for v in values]) if values else np.zeros(0)
  return (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)


def contourf(x, *args, **kw):
  """contourf(sF), contourf(v), contourf(v, data): filled contours"""
  return plot(x, *args, contourf=True, **kw)


def contour(x, *args, **kw):
  """contour(sF), contour(v), contour(v, data): contour lines"""
  return plot(x, *args, contour=True, **kw)


def pcolor(x, *args, **kw):
  """pcolor(sF), pcolor(v), pcolor(v, data): a coloured surface"""
  return plot(x, *args, pcolor=True, **kw)


smooth = pcolor


def quiver(v, w=None, resolution=10 * degree, **kw):
  """quiver(v, w), quiver(vF), quiver(ori, w): arrows at directions, a vector or axis field,
  or arrows at orientations in the sections of the current figure"""
  from .volumeplot import is3dMap, quiver3dMap
  if is3dMap(v):
    return quiver3dMap(v, w, **kw)
  if _mapKind(v) is not None:
    from .mapplot import quiverMap
    return quiverMap(v, w, **kw)
  if hasattr(v, "frameA") and hasattr(v, "frameB"):
    from .sectionplot import quiverSection
    return quiverSection(v, w, **kw)
  projKw, axesKw, whereKw = _split(kw)
  if w is None:
    f = v
    # the values of an axis field are axes, its argument a direction: both hemispheres
    projKw.setdefault("antipodal", False)
    sPs = _targets(f, projKw, axesKw, whereKw)
    hs = []
    # the arrows sit on an equispaced grid, so that they do not crowd the pole
    g = equispacedS2Grid(resolution=resolution)
    for sP in sPs:
      V = g[sP.sR.checkInside(g)]
      framed = vector3d(V.data, frame=getattr(f, "frame", None), isNormalized=True)
      W = f(framed)
      hs.append(_arrows(sP, V.flatten(), vector3d(W.data.reshape(-1, 3)), resolution=resolution, axisField=_isAxisField(f), **kw))
    return hs
  v = v if isinstance(v, vector3d) else vector3d(v)
  w = w if isinstance(w, vector3d) else vector3d(w)
  sPs = _targets(v, projKw, axesKw, whereKw)
  return [_arrows(sP, v.flatten(), w.flatten(), resolution=resolution, axisField=w.antipodal, **kw) for sP in sPs]


def plotRegion(sR, color="k", linewidth=1.0, **kw):
  """plot(sR): the outline of a spherical region"""
  linewidth = kw.pop("lineWidth", kw.pop("LineWidth", linewidth))
  color = kw.pop("lineColor", kw.pop("LineColor", kw.pop("linecolor", color)))
  projKw, axesKw, whereKw = _split(kw)
  sPs = _targets(sR, projKw, axesKw, whereKw)
  hs = []
  for sP in sPs:
    for arc in sR.boundary(0.5 * degree):
      x, y = sP.proj.project(arc)
      hs += sP.ax.plot(x, y, color=color, linewidth=linewidth, zorder=8, **kw)
  return hs


# ------------------------------------------------------------------ colour keys

# a direction key drawn as the sector it colours, an inverse pole figure key as its direction
# key: the key evaluated on the chart grid, as an rgb surface
def plotKey(key, resolution=0.5 * degree, fundamentalRegion=True, **kw):
  key = getattr(key, "dirMap", key)
  projKw, axesKw, whereKw = _split(kw)
  # complete colours the whole sphere, every direction by its equivalent in the sector
  if projKw.get("complete"):
    fundamentalRegion = False
  region = key.keySector if fundamentalRegion else sphericalRegion(frame=key.sR.frame)
  sPs = _targets(region, projKw, axesKw, whereKw)
  hs = []
  for sP in sPs:
    V, X, Y = sP.proj.grid(resolution)
    if V.size == 0:
      continue
    rgb = np.clip(key.direction2color(V), 0, 1)
    hs += [sP.ax.pcolormesh(Xb, Yb, np.ma.masked_invalid(np.ones(Xb.shape)), color=rgb[:, j0:j1].reshape(-1, 3), shading="flat",
                            zorder=2) if False else _rgbMesh(sP.ax, X[:, j0:j1], Y[:, j0:j1], rgb[:, j0:j1]) for j0, j1 in _blocks(X)]
  return hs


# an rgb surface on the curvilinear grid: gouraud shading needs one scalar per node, so the
# colours go into the mesh as its face colours over the nodes
def _rgbMesh(ax, X, Y, rgb):
  from matplotlib.collections import QuadMesh
  coords = np.stack([X, Y], axis=-1)
  mesh = QuadMesh(coords, shading="gouraud", zorder=2)
  mesh.set_array(None)
  mesh.set_facecolors(np.concatenate([rgb.reshape(-1, 3), np.ones((rgb.shape[0] * rgb.shape[1], 1))], axis=1))
  ax.add_collection(mesh, autolim=False)
  return mesh


# ------------------------------------------------------------------ pole figures

def _asList(h):
  if isinstance(h, vector3d):
    return [h[i] for i in range(h.size)] if h.size > 1 else [h]
  return [x if isinstance(x, vector3d) else vector3d(x) for x in h]


# the name of a specimen direction: an axis name of its frame, else its coordinates
def _directionTitle(r):
  fr = r.frame
  # a frame-free direction is named by the axes of the session's specimen frame
  from ..geometry.frames import specimenFrame
  names = tuple((specimenFrame.default if fr is None else fr).axesNames)
  u = r.normalize().data.reshape(3)
  for k, e in enumerate((xvector, yvector, zvector)):
    d = u @ e.data
    if abs(d) > 1 - 1e-6:
      return names[k] if d > 0 else "-" + names[k]
  return "(" + ",".join(f"{c:.2g}" for c in u) + ")"


# a row of functions on the sphere with one colour range: the pole figures and the inverse
# pole figures of an ODF
def _plotRow(fs, titles, style=None, resolution=1 * degree, colorRange="equal", colorbar=False, minmax=False,
             noTitle=False, **kw):
  flag = fillFlag(kw, None)
  style = ("pcolor" if flag in (None, "smooth") else flag) if style is None else style
  projKw, axesKw, whereKw = _split(kw)
  complete = projKw.pop("complete", False)
  charts, halves = [], []
  for f, title in zip(fs, titles):
    if complete:
      v = sphericalRegion(frame=getattr(f, "frame", None))
      pk = dict(projKw, complete=True, antipodal=projKw.get("antipodal", bool(getattr(f, "antipodal", False))),
                symmetric=hasSymmetry(v.frame))
    else:
      v, pk = f, projKw
    # a sphere drawn as two hemispheres is titled once and each half says which it is
    projs = makeSphericalProjection(v, **pk)
    for i, p in enumerate(projs):
      charts.append((f, p, None if noTitle or i else title))
      halves.append(len(projs) == 2)
  # a held figure of as many charts takes the surface on top of what it has; the axes
  # nextAxis prepared takes the first chart and the axes after it the others, as a row of
  # a newMtexFigure layout continues; a fresh figure of newMtexFigure the charts; else a
  # new figure, laid out as asked
  fig = gcm()
  if whereKw.get("parent") is not None:
    fig, sPs = intoParents(whereKw["parent"], [(p, t) for _, p, t in charts], axesKw)
  elif fig is not None and fig.hold and len(fig.children) == len(charts) and all(hasattr(ax, "_mtexSphericalPlot") for ax in fig.children):
    sPs = [ax._mtexSphericalPlot for ax in fig.children]
  elif fig is not None and fig.pending is not None:
    sPs = [sphericalPlot(fig.takeAxis() if i == 0 else fig.nextAxis(), p, **axesKw, title=t) for i, (_, p, t) in enumerate(charts)]
  elif fig is not None and not fig.children and not fig.hold:
    sPs = [sphericalPlot(fig.nextAxis(), p, **axesKw, title=t) for _, p, t in charts]
  else:
    fig = mtexFigure(len(charts), figSize=whereKw.get("figSize"), layout=whereKw.get("layout"), marginx=whereKw.get("marginx"),
                      marginy=whereKw.get("marginy"))
    sPs = [sphericalPlot(fig.nextAxis(), p, **axesKw, title=t) for _, p, t in charts]
  if not (fig is not None and fig.hold):
    for sP, half in zip(sPs, halves):
      if half:
        sP.hemisphereLabel()
  grids = [sP.proj.grid(resolution) for sP in sPs]
  values = [_valuesOn(f, V) for (f, _, _), (V, _, _) in zip(charts, grids)]
  # one range over the row, or one per function where the panels are not comparable
  # explicit contour levels span the colours, as MATLAB's contourf with levels
  if isinstance(colorRange, str) and _levelRange(kw) is not None:
    colorRange = _levelRange(kw)
  if isinstance(colorRange, str) and colorRange == "individual":
    own = {}
    for (f, _, _), vals in zip(charts, values):
      own.setdefault(id(f), []).append(vals)
    ranges = [_rangeOf(own[id(f)]) for f, _, _ in charts]
  else:
    if isinstance(colorRange, str) and colorRange == "equal":
      colorRange = _rangeOf(values)
    ranges = [colorRange] * len(charts)
  hs = [_filled(sP, X, Y, vals, style=style, colorRange=cr, **kw) for sP, (_, X, Y), vals, cr in zip(sPs, grids, values, ranges)]
  if colorRange is not None and not isinstance(colorRange, str):
    fig.colorRange = tuple(colorRange)
  for sP in sPs:
    if minmax:
      sP.annotateMinMax()
  if colorbar:
    fig.colorbar()
  return hs


def plotPDF(odf, h, *args, superposition=None, **kw):
  """plotPDF(odf, h): the pole figures of an ODF, one axes per crystal direction"""
  # MATLAB's positional flags, plotPDF(odf, h, 'antipodal', 'contourf'), and a convention
  kw.update({a: True for a in args if isinstance(a, str)})
  from ..geometry.plottingconvention import plottingConvention
  for a in args:
    if isinstance(a, plottingConvention):
      kw["how2plot"] = a
  hs = _asList(h)
  # plotPDF(odf, h, '3d'): the pole figure on the sphere in space
  if kw.pop("3d", False):
    from .spaceplot import plotFunction3d
    kw.pop("noTitle", None)
    return [plotFunction3d(odf.pdf(x), **kw) for x in hs]
  if superposition is not None:
    from ..functions.density import calcPDF
    # one superposed pole figure per entry when the coefficients come as a list of arrays,
    # plotPDF(odf, pf.allH, superposition=pf.c), else one of all the directions
    cs = superposition if isinstance(superposition, (list, tuple)) else [superposition]
    if isinstance(h, (list, tuple)) and len(cs) == len(h) and all(np.size(c) == x.size for c, x in zip(cs, hs)):
      fs = [calcPDF(odf, x, superposition=c) for x, c in zip(hs, cs)]
      titles = [x.char() for x in hs]
      poles = hs
    else:
      fs = [calcPDF(odf, hs, superposition=superposition)]
      titles = ["".join(x.char() for x in hs)]
      poles = [hs]
  else:
    from ..functions.density import calcPDF
    fs = [odf.pdf(x) if x.size == 1 else calcPDF(odf, x) for x in hs]
    # an antipodal pole figure superposes both hemispheres, as a measurement does
    if kw.get("antipodal"):
      fs = [f.even() if not getattr(f, "antipodal", True) and hasattr(f, "even") else f for f in fs]
    titles = [x.char() for x in hs]
    poles = hs
  # every pole figure on its own range, as MATLAB's; colorRange='equal' shares one
  kw.setdefault("colorRange", "individual")
  out = _plotRow(fs, titles, **kw)
  tagPoleAxes(poles, len(out))
  return out


# the pole figure axes remember their crystal direction, so that annotate(ori) can mark
# the poles of an orientation on them; the last charts of the current figure, in order
def tagPoleAxes(hs, numCharts):
  fig = gcm()
  if fig is None or numCharts == 0 or len(hs) == 0:
    return
  axes = fig.children[-numCharts:]
  per = max(numCharts // len(hs), 1)
  for i, ax in enumerate(axes):
    sP = getattr(ax, "_mtexSphericalPlot", None)
    if sP is not None:
      sP.h = hs[min(i // per, len(hs) - 1)]


# annotate(ori) on pole figures: the poles of the orientation on every axes that has a
# crystal direction
def _annotatePoles(ori, **kw):
  from .poles import poles
  fig = gcm()
  kw.pop("add2all", None)
  kw.setdefault("MarkerSize", 8)
  hs = []
  for ax in fig.children:
    sP = getattr(ax, "_mtexSphericalPlot", None)
    # an inverse pole figure: the crystal direction along its specimen direction, in the sector
    if sP is not None and getattr(sP, "r", None) is not None:
      from ..geometry.sphericalregion import projectIntoRegion
      from .poles import inversePoles
      w = inversePoles(ori.flatten(), sP.r, antipodal=bool(getattr(sP.sR, "antipodal", False))).flatten()
      hs.append(_markers(sP, projectIntoRegion(w, sP.sR), **dict(kw)))
      continue
    if sP is None or getattr(sP, "h", None) is None:
      continue
    directions = sP.h if isinstance(sP.h, (list, tuple)) else [sP.h] if sP.h.ndim == 0 else [sP.h[i] for i in range(sP.h.size)]
    for h in directions:
      p = poles(ori.flatten(), h, antipodal=bool(getattr(sP.sR, "antipodal", False)) or bool(h.antipodal))
      hs.append(_markers(sP, p.flatten(), **dict(kw)))
  return hs


# the poles of a fibre on the pole figures of the current figure: one curve per equivalent
# of each pole figure's crystal direction (MATLAB fibre/plot, orientation/line)
def _fibrePoles(f, points=1000, lineColor="k", lineWidth=1.5, **kw):
  from .poles import poles
  kw.pop("add2all", None)
  label = kw.pop("DisplayName", kw.pop("displayName", None))
  ori = f.orientation(points).flatten()
  hs = []
  for ax in gcm().children:
    sP = getattr(ax, "_mtexSphericalPlot", None)
    if sP is None or getattr(sP, "h", None) is None:
      continue
    directions = sP.h if isinstance(sP.h, (list, tuple)) else [sP.h] if sP.h.ndim == 0 else [sP.h[i] for i in range(sP.h.size)]
    for h in directions:
      p = poles(ori, h, antipodal=bool(getattr(sP.sR, "antipodal", False)) or bool(h.antipodal))
      xyz = p.data.reshape(ori.size, -1, 3)
      rings = [xyz[:, j] for j in range(xyz.shape[1])]
      # the legend names the fibre once per axes
      for k, h1 in enumerate(_curves(rings, None, p.antipodal, False, lineColor, lineWidth, 1.0, ax, "fibre", **kw)):
        h1[0].set_label(label if (label is not None and k == 0) else "_nolegend_")
        hs.append(h1)
  return hs


def plotIPDF(odf, r, *args, **kw):
  """plotIPDF(odf, r): the inverse pole figures of an ODF, one axes per specimen direction"""
  rs = _asList(r)
  fs = [odf.ipdf(x) for x in rs]
  out = _plotRow(fs, [_directionTitle(x) for x in rs], **kw)
  # the axes remember their specimen direction, so that annotate(ori) can mark it
  fig = gcm()
  if fig is not None and len(out) == len(rs):
    for ax, x in zip(fig.children[-len(rs):], rs):
      if hasattr(ax, "_mtexSphericalPlot"):
        ax._mtexSphericalPlot.r = x
  return out


__all__ = ["plotHKL", "plotSymmetry", "sphericalPlot", "plot", "scatter", "text", "annotate", "contourf", "contour", "pcolor", "quiver", "plotRegion", "plotKernel", "circle", "ellipse",
           "plotPDF", "plotIPDF", "plotKey"]
