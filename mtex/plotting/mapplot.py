# the matplotlib drawing of EBSD and grain maps (MATLAB @EBSD/plot, @grain2d/plot,
# @grainBoundary/plot, @mapPlot, scaleBar; docs/plans/plotting.md S4)
#
# maps.py says what to draw: the picture of a rigid lattice, the cells of a drifted one,
# the points of a map with no lattice, the grains as polygons, the boundary as chains or as
# segments, the scale bar with the frame triad. This file draws it into one axes per map,
# under the layout of the plotting convention, and keeps a mapPlot on the axes so that a
# boundary drawn afterwards lands on the map it belongs to.
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from ..geometry.arguments import refuseFlags
from ..geometry.crystalshape import crystalShape
from ..geometry.orientation import orientation
from ..geometry.plottingconvention import plottingConvention
from ..geometry.session import getMTEXpref
from ..geometry.units import degree
from ..geometry.vector3d import vector3d, zvector, asVector3d
from ..maps.boundary import grainBoundary, triplePointList
from ..maps.ebsd import EBSD
from ..maps.grains import grain2d
from .colormaps import _colorLimits, mtexColormap
from .maps import (CELLS_MAX, boundaryPolyline, boundarySegments, frameOnScreen, mapCells, mapExtent, mapLayout, mapPoints,
                   mapRaster, phasePalette, planeOf, scaleBar, triadGeometry, unitLabel)
from .keywords import refuseVariants
from .mtexfigure import gcm, mtexFigure
from .orientationkeys import ipfColorKey, ipfColors, orientationColorKey, _rotationsOf
from ..geometry.display import display, frameLabel, header

if TYPE_CHECKING:
  from typing import Unpack

  from .keywords import PlotKeywords

MAP_KEYS = ("how2plot", "coordinates", "title", "region", "parent", "figSize", "unit")


class mapPlot:
  """the map plot of one axes: the axes, the layout of the plane on the screen, the unit

    mP = mapPlot(ax, x)
    mP.layout, mP.extent, mP.unit, mP.mappable
  """

  def __init__(self, ax, x, how2plot=None, coordinates=False, title=None, region=None, unit=None):
    self.ax = ax
    self.layout = mapLayout(how2plot)
    self.extent = mapExtent(x, self.layout)
    self.unit = getattr(x, "scanUnit", "um") if unit is None else unit
    self.plane = planeOf(x)
    self.frame = x.frame
    self.mappables = []
    # the scale bar, its arguments for a redraw, and whether it follows a zoom
    self.scaleBarArtists, self.scaleBarArgs, self._followsView = [], None, False
    # whether a held plot drew into this map after the first
    self.reused = False
    ax._mtexMapPlot = self
    ax.set_aspect("equal")
    # a region is given in the coordinates of the map and drawn under its layout
    if region is not None:
      c = np.array([[region[0], region[2]], [region[1], region[2]], [region[0], region[3]], [region[1], region[3]]], dtype=float) @ self.layout.T
      region = (c[:, 0].min(), c[:, 0].max(), c[:, 1].min(), c[:, 1].max())
    xlo, xhi, ylo, yhi = self.extent if region is None else region
    # the limits are the map's, and nothing drawn into it later widens them on its own
    ax.set_autoscale_on(False)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    if coordinates:
      right, up = self.screenAxisNames()
      u = unitLabel(self.unit)
      ax.set_xlabel(f"{right} / {u}")
      ax.set_ylabel(f"{up} / {u}")
    else:
      ax.set_xticks([])
      ax.set_yticks([])
    if title is not None:
      ax.set_title(title, fontsize=getMTEXpref("FontSize", 13))

  @property
  def mappable(self):
    """every drawing coloured by values, so that a colour map or range set afterwards reaches
    all of them, as MATLAB's does; the mappable is the last one, the colour bar's"""
    return self.mappables[-1] if self.mappables else None

  @mappable.setter
  def mappable(self, h):
    """adds a drawing coloured by values to the list the colour map and range reach; None adds nothing"""
    if h is not None:
      self.mappables.append(h)

  def screenAxisNames(self):
    """which reference axis runs to the right and which up under the layout, by name and sign"""
    names = tuple(self.frame.axesNames) if self.frame is not None else ("X", "Y", "Z")
    e1, e2 = self.plane.e1, self.plane.e2
    axes = [np.array([a @ e1, a @ e2]) for a in np.eye(3)]
    out = []
    for row in range(2):
      best = ""
      for k, a in enumerate(axes):
        s = self.layout[row] @ a
        if abs(s) > 1 - 1e-6:
          best = ("-" if s < 0 else "") + names[k]
      out.append(best)
    return tuple(out)

  def onScreen(self, xy):
    """screen coordinates of points of the map plane, the layout applied to them"""
    return np.asarray(xy, dtype=float) @ self.layout.T

  @property
  def outOfScreen(self):
    """the normal of the map plane that points at the viewer, who looks from the other side
    when the layout flips the plane"""
    return self.plane.normal * (-1 if np.linalg.det(self.layout) < 0 else 1)

  def extend(self, x):
    """more data drawn into the map widen it, as hold on does in MATLAB, and the scale bar
    follows the new limits"""
    xlo, xhi, ylo, yhi = mapExtent(x, self.layout)
    e = self.extent
    new = (min(e[0], xlo), max(e[1], xhi), min(e[2], ylo), max(e[3], yhi))
    if new == e:
      return
    self.extent = new
    self.ax.set_xlim(new[0], new[1])
    self.ax.set_ylim(new[2], new[3])
    self._redrawScaleBar()

  def _redrawScaleBar(self):
    if self.scaleBarArtists:
      for a in self.scaleBarArtists:
        a.remove()
      self.scaleBarArtists = []
      scaleBarOn(self, None, **self.scaleBarArgs)

  def followView(self):
    """a zoom, by xlim and ylim or in a window, takes the scale bar into the new view, as MATLAB
    redraws its bar in the corner of what is shown"""
    if self._followsView:
      return
    self._followsView = True

    def moved(ax):
      (xlo, xhi), (ylo, yhi) = ax.get_xlim(), ax.get_ylim()
      view = (min(xlo, xhi), max(xlo, xhi), min(ylo, yhi), max(ylo, yhi))
      if view != tuple(self.extent) and self.scaleBarArtists:
        self.extent = view
        self._redrawScaleBar()
    self.ax.callbacks.connect("xlim_changed", moved)
    self.ax.callbacks.connect("ylim_changed", moved)

  def __repr__(self):
    return display(header("mapPlot", frameLabel(self.frame)), [("unit", self.unit)])


# ------------------------------------------------------------------ where a map goes

# what kind of map x is, None for anything else; an EBSD3 volume is an EBSD map
MAP_KINDS = ((EBSD, "ebsd"), (grain2d, "grains"), (grainBoundary, "boundary"), (triplePointList, "triplePoints"))


def _kind(x):
  return next((kind for cls, kind in MAP_KINDS if isinstance(x, cls)), None)


# the size of a figure of one map: seven inches along its longer side, scaled by a named size
def _mapFigSize(x, mapKw, figSize):
  from .mtexfigure import figureSize
  xlo, xhi, ylo, yhi = mapExtent(x, mapLayout(mapKw.get("how2plot")))
  w, h = max(xhi - xlo, 1e-12), max(yhi - ylo, 1e-12)
  return figureSize(figSize, (7.0, min(max(7.0 * h / w, 2.5), 9.0)) if w >= h else (max(7.0 * w / h, 3.0), 7.0))


# the map plot a command draws into: the parent named, the current axes when it is held
# and holds a map, else a new figure sized after the map
def _target(x, mapKw):
  plotKw = {k: v for k, v in mapKw.items() if k not in ("parent", "figSize")}
  parent = mapKw.get("parent")
  if parent is not None:
    mP = getattr(parent, "_mtexMapPlot", None)
    return mP if mP is not None else mapPlot(parent, x, **plotKw)
  fig = gcm()
  if fig is not None and fig.hold and fig.children and hasattr(fig.children[-1], "_mtexMapPlot"):
    mP = fig.children[-1]._mtexMapPlot
    mP.extend(x)
    mP.reused = True
    return mP
  # the axes nextAxis prepared takes the map, a fresh figure of newMtexFigure its first axes
  if fig is not None and fig.pending is not None:
    return mapPlot(fig.takeAxis(), x, **plotKw)
  # a fresh figure of one panel takes the shape of the map at the size it was asked for
  if fig is not None and not fig.children and not fig.hold:
    if fig.nrows * fig.ncols == 1 and not fig.fixedSize:
      fig.fig.set_size_inches(*_mapFigSize(x, mapKw, getattr(fig, "figSize", None)))
    return mapPlot(fig.nextAxis(), x, **plotKw)
  fig = mtexFigure(1, figSize=_mapFigSize(x, mapKw, mapKw.get("figSize")))
  return mapPlot(fig.nextAxis(), x, **plotKw)


def _splitMap(kw):
  return {k: kw.pop(k) for k in MAP_KEYS if k in kw}


def mapAliases(kw, grains=False):
  """the style keywords of a map plot, checked for their spelling; MATLAB's transparent is
  the face alpha of a map, and the grains keep translucent, MATLAB's alpha that follows the colour"""
  refuseVariants(kw)
  if grains and "transparent" in kw:
    kw["translucent"] = kw.pop("transparent")
  if not grains:
    for old in ("translucent", "transparent"):
      if old in kw:
        kw["faceAlpha"] = kw.pop(old)
  return kw


# ------------------------------------------------------------------ what is coloured

def colorRequest(color, faceColor=None, ipfDirection=None):
  """the colouring a plot asks for: an ipfDirection alone asks for the orientations in the
  inverse pole figure key along it, as MATLAB's plot(ebsd, 'ipfDirection', d); MATLAB's flags
  'ipfx', 'ipfy', 'ipfz' are spelled ipfDirection=xvector, ... here"""
  if isinstance(color, str) and color.lower() in ("ipfx", "ipfy", "ipfz"):
    raise ValueError(f"'{color}' is written plot(..., ipfDirection={color[-1].lower()}vector)")
  if color is None and faceColor is None and ipfDirection is not None:
    return "orientation"
  return color


def mapValues(x, color, faceColor=None, ipfDirection=None, at=None):
  """the colour of every measurement or grain, and whether it is a scalar to be mapped:
  the phase palette, an orientation list through the inverse pole figure key of its
  phase, an orientation colour key, the name of a property, one value or one colour per
  element, or one colour for all
  the value or the colour of every element, or of the elements `at` alone: a picture shows
  fewer measurements than a large map holds, and the colour of one it never shows is work
  thrown away. A route that yields numbers ignores `at` and stays whole, since the colour
  range is the map's and not the picture's; a route that yields colours returns one per
  entry of `at`."""
  from matplotlib.colors import to_rgb
  color = colorRequest(color, faceColor, ipfDirection)
  n = x.size
  ids = np.asarray(x.phaseId).reshape(-1)
  sel = slice(None) if at is None else np.asarray(at).reshape(-1)
  m = n if at is None else len(sel)
  if faceColor is not None and color is None:
    return np.tile(to_rgb(faceColor), (m, 1)), False
  if _isPhase(color):
    rgb = np.array([to_rgb(c) for c in phasePalette(x.CSList)])
    return rgb[ids[sel]], False
  if _isName(color, "orientation"):
    rgb = ipfColors(x, ipfDirection=zvector if ipfDirection is None else ipfDirection, at=at).reshape(-1, 3)
    return _withUnindexed(x, rgb, at), False
  if isinstance(color, orientation):
    if color.size != n:
      raise ValueError("one orientation per measurement")
    # the inverse pole figure key of the phase, looking along the direction asked for
    key = ipfColorKey(color)
    if ipfDirection is not None:
      key.ipfDirection = ipfDirection
    o = color if at is None else color.flatten()[sel]
    # a site without an orientation is white
    rgb = key.orientation2color(o).reshape(-1, 3)
    has = np.all(np.isfinite(o.data.reshape(-1, o.data.shape[-1])), axis=-1) & np.all(np.isfinite(rgb), axis=-1)
    return np.where(has[:, None], np.clip(rgb, 0, 1), 1.0), False
  if isinstance(color, orientationColorKey):
    direction = color.ipfDirection if isinstance(color, ipfColorKey) else zvector
    rgb = ipfColors(x, ipfDirection=direction, at=at).reshape(-1, 3)
    q = _rotationsOf(x).data.reshape(-1, 4)[sel]
    for i in x.CSList.indexedPhaseId:
      fr = x.CSList[i]
      same = fr is color.frameA or fr.Laue() is color.frameA.Laue()
      sub = np.flatnonzero(ids[sel] == i)
      if same and len(sub):
        o = orientation._fromRotation(q[sub], (fr, x.frame))
        c = color.orientation2color(o)
        rgb[sub] = np.where(np.all(np.isfinite(c), axis=-1)[:, None], np.clip(c, 0, 1), 1.0)
      elif not isinstance(color, ipfColorKey) and len(sub):
        rgb[sub] = 1.0
    return _withUnindexed(x, rgb, at), False
  if isinstance(color, str):
    if color in x.prop:
      return np.asarray(x.prop[color], dtype=float).reshape(-1), True
    return np.tile(to_rgb(color), (m, 1)), False
  arr = np.asarray(color)
  # colours in the shape of the map are one per measurement too
  if arr.ndim >= 2 and arr.shape[-1] == 3 and arr.size == 3 * n:
    return np.clip(arr.astype(float).reshape(-1, 3)[sel], 0, 1), False
  if arr.size == n:
    return arr.astype(float).reshape(-1), True
  if arr.shape == (3,):
    return np.tile(arr.astype(float), (m, 1)), False
  raise ValueError(f"one value or one colour per element is needed, {n} of them")


# an inverse pole figure map has no colour for a phase that was not indexed, so it is white
# unless its frame carries a colour of its own, black being what most people give it
def _withUnindexed(x, rgb, at=None):
  from matplotlib.colors import to_rgb
  ids = np.asarray(x.phaseId).reshape(-1)
  if at is not None:
    ids = ids[np.asarray(at).reshape(-1)]
  for i in range(len(x.CSList)):
    c = x.CSList[i].color
    if not x.CSList.isIndexed(i) and c is not None:
      rgb[ids == i] = to_rgb(c)
  return rgb


def _norm(values, colorRange):
  from matplotlib.colors import Normalize
  return Normalize(*_colorLimits(values, colorRange))


# whether a colour request is the name given, a string compared only as a string
def _isName(color, name):
  return isinstance(color, str) and color == name


# whether a colour request asks for the phase palette, the default
def _isPhase(color):
  return color is None or _isName(color, "phase")


# ------------------------------------------------------------------ the EBSD map

# a map plotted into a held 3D scene, or into a held map of another plane, goes into space;
# every other map is recorded on its axes, so that setCamera can turn the axes into a scene
def _liftable(f):
  import functools

  @functools.wraps(f)
  def g(x, *args, **kw):
    from . import volumeplot as vp
    if kw.get("parent") is None:
      out = vp.mapInto3d(f, x, args, kw)
      if out is not None:
        return out
    out = f(x, *args, **kw)
    first = out[0] if isinstance(out, list) and out else out
    ax = getattr(first, "axes", None)
    if ax is None and kw.get("parent") is None and gcm() is not None and gcm().children:
      ax = gcm().children[-1]
    if kw.get("parent") is None:
      vp.recordMap(ax, f, x, args, kw)
    return out
  return g


@_liftable
def plotEBSD(ebsd, color=None, faceColor=None, backend=None, colormap="viridis", colorRange=None, colorbar=False, micronbar=None,
             refFrame=True, location=None, ipfDirection=None, maxPixels=None, edgeColor=None, lineWidth=0.5, alpha=1.0, legend=None, displayName=None,
             markerSize=4, **kw: Unpack[PlotKeywords]):
  """plot(ebsd), plot(ebsd, values), plot(ebsd, ori), plot(ebsd, rgb), plot(ebsd, faceColor='black')

  A spatial EBSD map coloured by phase, by a value per measurement, by orientations through
  the inverse pole figure key of their phase, by an orientation colour key, or in one
  colour. backend is 'imagesc' for the picture a rigid lattice is, 'patch' for one cell per
  measurement, 'scatter' for the measurements alone; by default the first the map allows.
  """
  color = colorRequest(color, faceColor, ipfDirection)
  # plot(ebsd, cS): the crystal shape at every measurement of its phase
  if isinstance(color, crystalShape):
    from .shapeplot import plotShapesOnMap
    return plotShapesOnMap(ebsd, color, faceColor=faceColor, faceAlpha=kw.pop("faceAlpha", alpha), edgeColor=kw.pop("edgeColor", edgeColor or "k"),
                           lineWidth=lineWidth, colormap=colormap, micronbar=micronbar, refFrame=refFrame, displayName=displayName, **kw)
  location = kw.pop("Location", location)
  alpha = kw.pop("faceAlpha", alpha)
  # one transparency for all or one per measurement, a site without a value opaque to nothing
  a = np.nan_to_num(np.asarray(alpha, dtype=float), nan=0.0)
  alpha = float(a) if a.ndim == 0 else np.clip(a.reshape(-1), 0, 1)
  mapKw = _splitMap(kw)
  mP = _target(ebsd, mapKw)
  ax = mP.ax
  # faceColor='none' outlines the cells
  hollow = isinstance(faceColor, str) and faceColor.lower() == "none"
  cmap = mtexColormap(colormap)
  style = {None: "auto", "imagesc": "raster", "surf": "cells", "patch": "cells", "scatter": "points", "points": "points",
           "raster": "raster", "cells": "cells", "auto": "auto"}.get(backend)
  if style is None:
    raise ValueError("backend is 'imagesc', 'patch', 'surf' or 'scatter'")
  if style == "auto" and ebsd.lattice is not None and (edgeColor is not None or ebsd.size < CELLS_MAX):
    style = "cells"
  # the picture holds twice the pixels of the axes on each side unless asked otherwise, since
  # every draw, pan and zoom resamples the whole of it
  r = mapRaster(ebsd, maxPixels=_pixelCap(ax) if maxPixels is None else maxPixels, layout=mP.layout) if style in ("auto", "raster") else None
  # every pixel shows one measurement, so a map larger than its picture is coloured where it
  # is seen and nowhere else; a route that yields numbers is coloured whole, its range being
  # the map's
  pick = None if r is None else np.where(r[2] >= 0, r[2], 0)
  at = pick.reshape(-1) if pick is not None and pick.size < ebsd.size else None
  values, scalar = mapValues(ebsd, color, None if hollow else faceColor, ipfDirection, at=at)
  norm = _norm(values, colorRange) if scalar else None
  if r is not None:
    (xlo, xhi), (ylo, yhi), index = r
    covered = index >= 0
    if scalar:
      img = np.full(index.shape, np.nan)
      img[covered] = values[index[covered]]
      # one transparency for all or one per measurement, laid out as the picture is
      alphaImg = alpha if np.ndim(alpha) == 0 else np.where(covered, alpha[np.clip(index, 0, None)], 0.0)
      # a NaN is drawn as nothing, an infinite value at the end of the colour map, as MATLAB
      img[np.isneginf(img)], img[np.isposinf(img)] = norm.vmin - 1, norm.vmax + 1
      h = ax.imshow(np.ma.masked_invalid(img), extent=(xlo, xhi, ylo, yhi), origin="lower", interpolation="nearest",
                    cmap=cmap, norm=norm, alpha=alphaImg, zorder=1, **kw)
    else:
      # the colours of the pixels as bytes, the map's own colours never assembled
      rgb = values.reshape(index.shape + (3,)) if at is not None else values[pick]
      img = np.zeros(index.shape + (4,), dtype=np.uint8)
      img[..., :3] = (np.nan_to_num(np.clip(rgb, 0, 1)) * 255 + 0.5).astype(np.uint8)
      # one transparency for all or one per measurement; a measurement whose colour is no
      # colour, an unindexed one, is drawn as nothing at all
      shown = covered & np.all(np.isfinite(rgb), axis=-1)
      opacity = int(alpha * 255 + 0.5) if np.ndim(alpha) == 0 else (alpha[pick] * 255 + 0.5).astype(np.uint8)
      img[..., 3] = np.where(shown, opacity, 0)
      h = ax.imshow(img, extent=(xlo, xhi, ylo, yhi), origin="lower", interpolation="nearest", zorder=1, **kw)
  elif style == "raster":
    raise ValueError("the map is not a rigid lattice, so it has no picture; backend='patch' draws its cells")
  elif ebsd.lattice is not None and style != "points":
    from matplotlib.collections import PolyCollection
    corners = mapCells(ebsd, layout=mP.layout)
    keep = _drawn(ebsd, values, scalar)
    if keep is not None:
      corners, values, alpha = corners[keep], values[keep], _alphaOf(alpha, keep)
    h = PolyCollection(corners, closed=True, linewidths=lineWidth if edgeColor is not None else 0,
                       edgecolors=edgeColor if edgeColor is not None else "face", alpha=alpha, zorder=1, **kw)
    if scalar:
      h.set_array(values)
      h.set_cmap(cmap)
      h.set_norm(norm)
    else:
      h.set_facecolors("none" if hollow else values)
    ax.add_collection(h)
  elif style == "cells":
    raise ValueError("the map is not on a lattice, so it has no cells; backend='scatter' draws its measurements")
  else:
    P = mapPoints(ebsd, layout=mP.layout)
    keep = _drawn(ebsd, values, scalar)
    if keep is not None:
      P, values, alpha = P[keep], values[keep], _alphaOf(alpha, keep)
    mapped = dict(cmap=cmap, norm=norm) if scalar else {}
    h = ax.scatter(P[:, 0], P[:, 1], c=values, s=markerSize ** 2, linewidths=0, alpha=alpha, zorder=1, **mapped, **kw)
  if displayName is not None:
    h.set_label(displayName)
  if scalar:
    mP.mappable = h
    if colorbar:
      gcm().colorbar(label=color if isinstance(color, str) else "")
  _finish(mP, ebsd, micronbar, refFrame, legend, color, location, ipfDirection)
  return h


# the measurements a cell or a point is drawn for, None for all: the padding of a grid is no
# measurement, and a colour that is no colour, of a site without an orientation, is nothing
# drawn, as in the picture
def _drawn(ebsd, values, scalar):
  keep = np.ones(ebsd.size, dtype=bool) if ebsd._pad is None else ~ebsd._pad.reshape(-1)
  if not scalar:
    keep &= np.all(np.isfinite(values), axis=-1)
  return None if keep.all() else keep


# the transparency of the measurements kept, one for all or one each
def _alphaOf(alpha, keep):
  return alpha if np.ndim(alpha) == 0 else alpha[keep]


# twice the pixels of the axes on each side, at the figure's dpi
def _pixelCap(ax):
  fig = ax.figure
  pos = ax.get_position()
  w, h = fig.get_figwidth() * fig.dpi * pos.width, fig.get_figheight() * fig.dpi * pos.height
  return int(max(4 * w * h, 250_000))


def _finish(mP, x, micronbar, refFrame, legend, color, location=None, ipfDirection=None):
  # MATLAB's 'micronbar', 'off' and 'refFrame', 'off'; a plot into a held map adds no second scale bar
  micronbar, refFrame = _onOff(micronbar), _onOff(refFrame)
  if micronbar is None:
    micronbar = getMTEXpref("showMicronBar", True) and not mP.reused
  if micronbar and not mP.scaleBarArtists:
    show = bool(_onOff(getMTEXpref("showRefFrame", True)))
    scaleBarOn(mP, x, location=location or "sw", refFrame=refFrame if show else False)
    mP.followView()
  # a phase map carries its legend unless told not to, as MATLAB draws it
  if legend is None:
    legend = _isPhase(color)
  if not legend:
    return
  if _isPhase(color):
    phaseLegend(mP, x)
  elif isinstance(color, ipfColorKey):
    keyLegend(mP, x, ipfDirection=color.ipfDirection)
  elif _isName(color, "orientation"):
    keyLegend(mP, x, ipfDirection=zvector if ipfDirection is None else ipfDirection)


# MATLAB's 'on' and 'off' as a truth value, anything else as it is
def _onOff(v):
  return v.lower() == "on" if isinstance(v, str) else v


# ------------------------------------------------------------------ the grains and the boundary

@_liftable
def plotGrains(grains, color=None, faceColor=None, colormap="viridis", colorRange=None, colorbar=False, micronbar=None, refFrame=True,
               location=None, ipfDirection=None, boundary=True, lineWidth=1.0, lineColor=None, alpha=1.0, legend=None, displayName=None,
               translucent=None, **kw: Unpack[PlotKeywords]):
  """plot(grains), plot(grains, property), plot(grains, grains.meanOrientation)

  The grains drawn as the polygons the boundary encloses, coloured by phase, by a value per
  grain, by their mean orientations or an orientation key, or in one colour, with the
  boundary on top unless noBoundary. translucent=s fades a grain by its colour, as MATLAB:
  s (1 - min(r, g, b)) for a colour, s v / max(v) for a value, so pale grains let the map
  below show.
  """
  alpha = kw.pop("faceAlpha", alpha)
  color = colorRequest(color, faceColor, ipfDirection)
  # plot(grains, v): an axis per grain as a compass needle at its centroid
  if isinstance(color, vector3d):
    return needlesOnMap(grains, color, micronbar=micronbar, refFrame=refFrame, **kw)
  # plot(grains, cS): the crystal shape of every grain into the map
  if isinstance(color, crystalShape):
    from .shapeplot import plotShapesOnMap
    return plotShapesOnMap(grains, color, faceColor=faceColor, faceAlpha=alpha, edgeColor=kw.pop("edgeColor", lineColor or "k"),
                           lineWidth=lineWidth, colormap=colormap, micronbar=micronbar, refFrame=refFrame, displayName=displayName, **kw)
  location = kw.pop("Location", location)
  mapKw = _splitMap(kw)
  mP = _target(grains, mapKw)
  ax = mP.ax
  values, scalar = mapValues(grains, color, faceColor, ipfDirection)
  cmap = mtexColormap(colormap)
  # every grain is one path with its holes, all of them one collection
  paths, at = _grainPaths(grains, mP)
  if not paths:
    return None
  from matplotlib.collections import PathCollection
  h = PathCollection(paths, linewidths=0, alpha=alpha, zorder=1, **kw)
  if scalar:
    h.set_array(values[at])
    h.set_cmap(cmap)
    h.set_norm(_norm(values, colorRange))
    mP.mappable = h
  else:
    h.set_facecolors(values[at])
    h.set_edgecolors("none")
  if translucent is not None:
    h.set_alpha(_translucency(values[at], scalar, float(translucent)))
  ax.add_collection(h)
  if displayName is not None:
    h.set_label(displayName)
  if boundary and grains.boundary is not None:
    # the boundary takes the line colour, else the face colour, else black, as MATLAB passes
    # its options on; faces of no colour keep a black outline
    if lineColor is None:
      lineColor = faceColor if faceColor is not None and not (isinstance(faceColor, str) and faceColor.lower() == "none") else "k"
    _boundaryLines(mP, grains.boundary, lineColor, lineWidth, zorder=3)
  if scalar and colorbar:
    gcm().colorbar(label=color if isinstance(color, str) else "")
  # grains of one face colour name no phases, as in MATLAB, but the name they were given
  named = legend is None and faceColor is not None
  if named:
    legend = False
  _finish(mP, grains, micronbar, refFrame, legend, color, location, ipfDirection)
  if named and displayName is not None:
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch
    mP.ax.legend(handles=[Patch(facecolor=to_rgb(faceColor), edgecolor="k", label=displayName)], loc="upper right")
  return h


# MATLAB's translucent alpha of a face: s (1 - min(rgb)) for colours, s v / max(v) for values
def _translucency(values, scalar, s):
  v = np.asarray(values, dtype=float)
  if scalar:
    return np.clip(s * v / np.nanmax(v), 0, 1)
  return np.clip(s * (1 - np.min(v[:, :3], axis=1)), 0, 1)


# every grain as one path on the screen, its outer loop first and its holes after it, each
# loop closed; a grain whose outer loop has fewer than three corners has no face. Returns
# the paths and the indices of the grains they draw
def _grainPaths(grains, mP):
  from matplotlib.path import Path
  V = mP.onScreen(grains.V)
  paths, at = [], []
  for k in range(grains.size):
    loops = grains.loops[k]
    if not loops or len(loops[0]) < 3:
      continue
    ends = np.cumsum([len(loop) + 1 for loop in loops])
    verts = V[np.concatenate([np.append(loop, loop[0]) for loop in loops])]
    codes = np.full(ends[-1], Path.LINETO, dtype=Path.code_type)
    codes[np.concatenate([[0], ends[:-1]])] = Path.MOVETO
    codes[ends - 1] = Path.CLOSEPOLY
    paths.append(Path(verts, codes))
    at.append(k)
  return paths, at


def needlesOnMap(grains, v, micronbar=None, refFrame=True, lineWidth=2.0, **kw: Unpack[PlotKeywords]):
  """an axis per grain as MATLAB draws it: a needle through the centroid, grey when it lies
  in the plane of the map, its half pointing out of the screen black and the other white
  when it does not, half a diameter long"""
  from matplotlib.collections import LineCollection
  mapKw = _splitMap(kw)
  mP = _target(grains, mapKw)
  d = np.asarray(v.normalize().data, dtype=float).reshape(-1, 3)
  c = grains.centroid.data.reshape(-1, 3)
  length = np.asarray(grains.diameter).reshape(-1, 1) / 4
  n = mP.outOfScreen
  out = d @ n
  inPlane = d - out[:, None] * n
  norm = np.linalg.norm(inPlane, axis=1, keepdims=True)
  u = np.where(norm > 1e-12, inPlane / np.where(norm > 1e-12, norm, 1), 0.0)
  P0, PA, PB = (mP.plane.to2d(x) @ mP.layout.T for x in (c, c - length * u, c + length * u))
  # two segments per grain, grey from end to end where the needle lies in the plane, else
  # its half out of the screen black and the other white from the centroid
  flat, up = (np.abs(out) < 0.1)[:, None], (out > 0)[:, None]
  black, white, grey = np.array([0.0, 0, 0, 1]), np.array([1.0, 1, 1, 1]), np.array([0.5, 0.5, 0.5, 1])
  segs = np.stack([np.stack([np.where(flat, PA, P0), PB], axis=1), np.stack([P0, PA], axis=1)], axis=1).reshape(-1, 2, 2)
  colors = np.stack([np.where(flat, grey, np.where(up, black, white)), np.where(up, white, black)], axis=1).reshape(-1, 4)
  used = np.stack([np.ones(len(c), dtype=bool), ~flat[:, 0]], axis=1).reshape(-1)
  h = LineCollection(segs[used], colors=colors[used], linewidths=lineWidth, zorder=6, path_effects=None)
  mP.ax.add_collection(h)
  mP.ax.scatter(*P0.T, s=lineWidth ** 2 * 3, color="k", zorder=7)
  _finish(mP, grains, micronbar, refFrame, False, None)
  return h


def _boundaryLines(mP, gB, color, lineWidth, zorder=3, **kw):
  P = mP.onScreen(boundaryPolyline(gB))
  h, = mP.ax.plot(P[:, 0], P[:, 1], color=color, linewidth=lineWidth, solid_joinstyle="miter", snap=True, zorder=zorder, **kw)
  return h


@_liftable
def plotBoundary(gB, color=None, lineColor="k", lineWidth=1.0, colormap="viridis", colorRange=None, colorbar=False, edgeAlpha=1.0,
                 micronbar=None, refFrame=True, displayName=None, **kw: Unpack[PlotKeywords]):
  """plot(gB), plot(gB, values), plot(gB, rgb), plot(gB, lineColor='r', lineWidth=2)

  The grain boundary segments: as joined chains in one colour, or segment by segment with
  a value or a colour per segment, angle(gB.misorientation) say.
  """
  from matplotlib.collections import LineCollection
  # a boundary is lines: MATLAB's edge colour is their colour
  lineColor = kw.pop("edgeColor", lineColor)
  mapKw = _splitMap(kw)
  mP = _target(gB, mapKw)
  ax = mP.ax
  n = len(gB)
  if color is None:
    if np.ndim(edgeAlpha) > 0:
      # one transparency per segment: the segments as a collection in their own RGBA colours
      from matplotlib.colors import to_rgba
      V = boundarySegments(gB)
      segs = mP.onScreen(V.reshape(-1, V.shape[-1])).reshape(-1, 2, 2)
      rgba = np.tile(to_rgba(lineColor), (len(segs), 1))
      a = np.clip(np.asarray(edgeAlpha, dtype=float).reshape(-1), 0, 1)
      # MATLAB draws thin segments as the outlines of two-vertex faces, there and back, so a
      # segment of alpha a shows as 1 - (1 - a)^2
      rgba[:, 3] = 1 - (1 - a) ** 2 if lineWidth <= 3 else a
      h = LineCollection(segs, colors=rgba, linewidths=lineWidth, snap=True, zorder=3, capstyle="round", **kw)
      mP.ax.add_collection(h)
    else:
      h = _boundaryLines(mP, gB, lineColor, lineWidth, alpha=edgeAlpha, **kw)
  else:
    arr = np.asarray(color)
    segs = mP.onScreen(boundarySegments(gB).reshape(-1, 2)).reshape(n, 2, 2)
    # round caps join the segments of a staircase into one line, as MATLAB draws it
    kw.setdefault("capstyle", "round")
    if arr.ndim == 2 and arr.shape == (n, 3):
      h = LineCollection(segs, colors=np.clip(arr, 0, 1), linewidths=lineWidth, alpha=edgeAlpha, snap=True, zorder=3, **kw)
    elif arr.size == n:
      h = LineCollection(segs, linewidths=lineWidth, alpha=edgeAlpha, snap=True, zorder=3, cmap=mtexColormap(colormap),
                         norm=_norm(arr.astype(float).reshape(-1), colorRange), **kw)
      h.set_array(arr.astype(float).reshape(-1))
      mP.mappable = h
      if colorbar:
        ax.add_collection(h)
        gcm().colorbar()
        h = None
    else:
      raise ValueError(f"one value or one colour per segment is needed, {n} of them")
    if h is not None:
      ax.add_collection(h)
  if displayName is not None and h is not None:
    h.set_label(displayName)
  # a boundary carries no phase legend of its own
  _finish(mP, gB, micronbar, refFrame, False, None)
  return h



def plotTriplePoints(tP, color=None, markerFaceColor=None, markerEdgeColor=None, markerSize=6, lineWidth=1.0, colormap="viridis",
                     colorRange=None, colorbar=False, micronbar=None, refFrame=True, displayName=None, **kw: Unpack[PlotKeywords]):
  """plot(tP), plot(tP, values): the triple points of a boundary network as circles, filled
  by a value where one is given"""
  mapKw = _splitMap(kw)
  mP = _target(tP, mapKw)
  P = mP.onScreen(np.asarray(tP.V).reshape(-1, 2))
  # the circles are edged in the colour asked for, a colour name given as the data, else black
  if markerEdgeColor is not None:
    edge = markerEdgeColor
  else:
    edge = color if isinstance(color, str) else "k"
  size = float(markerSize) ** 2
  # no points draw nothing, an empty scatter would stretch the limits to the origin
  if len(P) == 0:
    h = None
  elif color is None or isinstance(color, str):
    face = markerFaceColor if markerFaceColor is not None else "none"
    h = mP.ax.scatter(P[:, 0], P[:, 1], s=size, facecolors=face, edgecolors=edge, linewidths=lineWidth, zorder=6,
                      label=displayName, **kw)
  else:
    arr = np.asarray(color, dtype=float).reshape(-1)
    h = mP.ax.scatter(P[:, 0], P[:, 1], s=size, c=arr, cmap=mtexColormap(colormap), norm=_norm(arr, colorRange),
                      edgecolors=edge, linewidths=lineWidth, zorder=6, label=displayName, **kw)
    mP.mappable = h
    if colorbar:
      gcm().colorbar()
  _finish(mP, tP, micronbar, refFrame, False, None)
  if h is not None and displayName is not None:
    _joinLegend(mP, h)
  return h


# the map a line or points in its coordinates go into: the current one, else a plain axes
def _currentMap():
  fig = gcm()
  if fig is not None and fig.children and hasattr(fig.children[-1], "_mtexMapPlot"):
    return fig.children[-1]._mtexMapPlot
  return None


def _onMap(x, y):
  xy = np.stack([np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()], axis=-1)
  mP = _currentMap()
  if mP is None:
    import matplotlib.pyplot as plt
    return plt.gca(), xy
  return mP.ax, mP.onScreen(xy)


def line(x, y, z=None, **kw: Unpack[PlotKeywords]):
  """line(x, y, lineStyle=':', lineWidth=4, color='w'): a polyline in the coordinates of the
  current map, drawn under its layout, MATLAB's line"""
  # a line in space goes into a 3D scene
  if z is not None:
    from .volumeplot import line3
    return line3(x, y, z, **kw)
  kw = {k.lower(): v for k, v in refuseVariants(kw).items()}
  ax, P = _onMap(x, y)
  return ax.plot(P[:, 0], P[:, 1], linestyle=kw.pop("linestyle", "-"), linewidth=kw.pop("linewidth", 0.5),
                 color=kw.pop("color", "k"), zorder=7, **kw)[0]


def scatterXY(x, y, *args, filled=False, **kw: Unpack[PlotKeywords]):
  """scatter(x, y, sz, c, filled=True): markers at the coordinates of the current map, MATLAB's
  scatter of two coordinate lists; a string names the colour; NaN positions are left out"""
  if "filled" in args:
    refuseFlags(("filled",), "scatter")
  sizes = next((a for a in args if not isinstance(a, str)), 36)
  names = [a for a in args if isinstance(a, str)]
  color = kw.pop("color", next(iter(names), "b"))
  ax, P = _onMap(x, y)
  keep = np.all(np.isfinite(P), axis=1)
  return ax.scatter(P[keep, 0], P[keep, 1], s=sizes, c=color if filled else "none", edgecolors=color, zorder=7, **kw)


# an artist named into a map with a legend joins it, as MATLAB's legend grows
def _joinLegend(mP, h):
  leg = mP.ax.get_legend()
  if leg is None:
    return
  handles, labels = list(leg.legend_handles), [t.get_text() for t in leg.get_texts()]
  loc = leg._loc
  mP.ax.legend(handles=handles + [h], labels=labels + [h.get_label()], loc=loc)


# ------------------------------------------------------------------ arrows on a map

def textMap(x, labels, fontSize=None, color="k", **kw: Unpack[PlotKeywords]):
  """text(ebsd, labels): a label at every measurement or grain of the current map, or a new one"""
  mP = _currentMap() or _target(x, _splitMap(kw))
  if isinstance(x, grain2d):
    P = mP.plane.to2d(x.centroid.data.reshape(-1, 3)) @ np.asarray(mP.layout, dtype=float).T
  else:
    P = mapPoints(x, mP.layout)
  labels = [labels] * len(P) if isinstance(labels, str) else [str(t) for t in np.asarray(labels, dtype=object).ravel()]
  size = getMTEXpref("FontSize", 13) - 4 if fontSize is None else fontSize
  return [mP.ax.text(px, py, t, ha="center", va="center", fontsize=size, color=color, zorder=8, clip_on=True, **kw) for (px, py), t in zip(P, labels)]


def quiverMap(x, v, scale=None, color=None, lineWidth=None, scaling=True, displayName=None, autoScaleFactor=None, projectIntoPlane=False,
              antipodal=False, **kw: Unpack[PlotKeywords]):
  """quiver(grains, v), quiver(ebsd, v), quiver(gB, v): an arrow per grain at its centroid,
  per measurement, per segment at its midpoint; an axis is centred, a direction starts there.
  By default an arrow is a fifth of the grain's diameter, one and a half spacings of a map,
  three median segment lengths of a boundary; scale draws it scale times its in-plane length.
  """
  from matplotlib.collections import LineCollection
  refuseVariants(kw)
  mapKw = _splitMap(kw)
  mP = _target(x, mapKw)
  v = asVector3d(v)
  raw = v.data.reshape(-1, 3)
  kind = _kind(x)
  if raw.shape[0] != x.size:
    raise ValueError("one vector per element is needed")
  pl = mP.plane
  if kind == "grains":
    c, default = x.centroid, np.asarray(x.diameter).reshape(-1) / 5
  elif kind == "ebsd":
    c = pl.to2d(x.pos.data.reshape(-1, 3))
    default = np.full(len(raw), 1.5 * (x.lattice.dxy if x.lattice is not None else 1.0))
  else:
    c = x.midPoint
    default = np.full(len(raw), 3 * float(np.median(x.segLength)))
  # autoScaleFactor stretches the automatic length, as MATLAB's option does
  if autoScaleFactor is not None:
    default = default * float(autoScaleFactor)
  c = pl.to2d(c.data.reshape(-1, 3)) if isinstance(c, vector3d) else np.asarray(c, dtype=float).reshape(-1, 2)
  if kind == "boundary" and scaling and scale is None:
    return _quiverBoundary(mP, c, raw, 0.15 if autoScaleFactor is None else float(autoScaleFactor), color or "k",
                           2.0 if lineWidth is None else lineWidth, displayName, **kw)
  if kind == "grains" and scaling and scale is None:
    # without a colour the arrows of the grains take the next of the axes' colour order, as MATLAB's
    color = mP.ax._get_lines.get_next_color() if color is None else color
    return _quiverGrains(mP, c, raw, default, v.antipodal or antipodal, color, 2.0 if lineWidth is None else lineWidth, displayName, projectIntoPlane, **kw)
  lineWidth = 1.0 if lineWidth is None else lineWidth
  color = "k" if color is None else color
  u = np.stack([raw @ pl.e1, raw @ pl.e2], axis=-1)
  n = np.linalg.norm(u, axis=-1)
  ok = (n > 0) & np.all(np.isfinite(raw), axis=-1)
  if not scaling:
    length = n[ok]
  else:
    length = default[ok] if scale is None else scale * n[ok]
  u = u[ok] / n[ok][:, None] * length[:, None]
  c = c[ok]
  # an axis, or MATLAB's 'antipodal', is a line centred on its point
  if v.antipodal or antipodal:
    segs = np.stack([mP.onScreen(c - u / 2), mP.onScreen(c + u / 2)], axis=1)
    h = LineCollection(segs, colors=color, linewidths=lineWidth, zorder=4, **kw)
    mP.ax.add_collection(h)
    return h
  p, d = mP.onScreen(c), u @ mP.layout.T
  return mP.ax.quiver(p[:, 0], p[:, 1], d[:, 0], d[:, 1], angles="xy", scale_units="xy", scale=1, color=color, width=0.003, label=displayName,
                      zorder=4, **kw)


# the needles on a boundary as MATLAB draws them: quiver3 of v and -v from every midpoint,
# without heads, autoscaled as quiver3d does, f 0.9 times the mean spacing of the points
# for the longest vector in space, so that a steep one is short on the map
def _quiverBoundary(mP, c, raw, f, color, lineWidth, displayName, **kw):
  from matplotlib.collections import LineCollection
  pl = mP.plane
  ok = np.all(np.isfinite(raw), axis=-1)
  raw, c = raw[ok], c[ok]
  nrm = np.linalg.norm(raw, axis=-1)
  if not len(raw) or nrm.max() == 0:
    return None
  # quiver's spacing of 2n points in a list: the extent over the square root of their
  # number; the needles of MATLAB's figures are sqrt(3) of that, measured on them
  k = np.sqrt(2 * len(raw))
  spacing = np.sqrt(3) * np.hypot(np.ptp(c[:, 0]) / k, np.ptp(c[:, 1]) / k)
  d = raw * (f * 0.9 * spacing / nrm.max())
  # the half towards the viewer lies on top, the half into the map beneath the grains, as
  # MATLAB's three dimensional arrows are hidden by the faces they pierce
  d = d * np.where(d @ mP.outOfScreen < 0, -1, 1)[:, None]
  u = np.stack([d @ pl.e1, d @ pl.e2], axis=-1)
  up = np.stack([mP.onScreen(c), mP.onScreen(c + u)], axis=1)
  down = np.stack([mP.onScreen(c), mP.onScreen(c - u)], axis=1)
  mP.ax.add_collection(LineCollection(down, colors=color, linewidths=lineWidth, zorder=0.5, **kw))
  h = LineCollection(up, colors=color, linewidths=lineWidth, zorder=4, label=displayName, **kw)
  mP.ax.add_collection(h)
  return h


# the arrows on grains as MATLAB draws them: a fifth of the diameter along the direction in
# space, seen on the map, so that a steep direction is short; an axis both ways from the
# centroid, a direction pointing into the map ending at the centroid; a dot at the centroid
def _quiverGrains(mP, c, raw, length, antipodal, color, lineWidth, displayName, projectIntoPlane, **kw):
  from matplotlib.collections import LineCollection
  pl = mP.plane
  nrm = np.linalg.norm(raw, axis=-1)
  ok = (nrm > 0) & np.all(np.isfinite(raw), axis=-1)
  d = raw[ok] / nrm[ok][:, None] * length[ok][:, None]
  if projectIntoPlane:
    d = d - (d @ pl.normal)[:, None] * pl.normal
  u, c = np.stack([d @ pl.e1, d @ pl.e2], axis=-1), c[ok]
  if antipodal:
    segs = np.stack([mP.onScreen(c - u), mP.onScreen(c + u)], axis=1)
    h = LineCollection(segs, colors=color, linewidths=lineWidth, zorder=4, label=displayName, **kw)
    mP.ax.add_collection(h)
  else:
    start = c - u * (d @ mP.outOfScreen < 0)[:, None]
    p, q = mP.onScreen(start), u @ mP.layout.T
    h = mP.ax.quiver(p[:, 0], p[:, 1], q[:, 0], q[:, 1], angles="xy", scale_units="xy", scale=1, color=color, width=0.0015 * lineWidth,
                     label=displayName, zorder=4, **kw)
  mP.ax.scatter(*mP.onScreen(c).T, s=4 * lineWidth ** 2, color=color, zorder=5)
  return h


# ------------------------------------------------------------------ the scale bar and the legends

def scaleBarOn(mP, x=None, location="sw", fraction=0.1, refFrame=True, color="k", background="w", alpha=0.75, length=None):
  """a scale bar in one corner of a map: a round length near a tenth of its width, over the
  length it stands for, and above it the axes of the specimen frame as they lie in the picture"""
  from matplotlib.patches import Polygon, Rectangle
  ax = mP.ax
  mP.scaleBarArgs = dict(location=location, fraction=fraction, refFrame=refFrame, color=color, background=background, alpha=alpha,
                         length=length)
  artists = []
  xlo, xhi, ylo, yhi = mP.extent
  if length is None:
    barLength, label = scaleBar(xhi - xlo, mP.unit, fraction)
  else:
    barLength, label = float(length), f"{length:g} {unitLabel(mP.unit)}"
  if not barLength > 0:
    return None
  w, h = xhi - xlo, yhi - ylo
  t = 0.035 * min(w, h)
  gap = t / 4
  names = tuple(mP.frame.axesNames) if mP.frame is not None else ("X", "Y", "Z")
  triad = triadGeometry(frameOnScreen(mP.plane, mP.layout), list(names), t) if refFrame else None
  tw, th = (0.0, 0.0) if triad is None else (triad[3][1] - triad[3][0], triad[3][3] - triad[3][2])
  boxw = max(barLength, 0.6 * t * len(label), tw) + 2 * gap
  boxh = 3.5 * gap + t + th if refFrame else 3 * gap + t
  boxx = xhi - gap - boxw if location in ("se", "ne") else xlo + gap
  boxy = yhi - gap - boxh if location in ("nw", "ne") else ylo + gap
  # the text is the unit of the box: its size follows the map on the screen
  fontsize = _pointsPerUnit(ax) * 0.9 * t
  artists.append(ax.add_patch(Rectangle((boxx, boxy), boxw, boxh, facecolor=background, alpha=alpha, edgecolor=color, linewidth=0.8,
                                        zorder=10)))
  x0 = boxx + (boxw - barLength) / 2
  artists += ax.plot([x0, x0 + barLength], [boxy + 1.5 * gap] * 2, color=color, linewidth=max(1.0, 0.12 * t * _pointsPerUnit(ax)),
                     zorder=11, solid_capstyle="butt")
  artists.append(ax.text(boxx + boxw / 2, boxy + 2.5 * gap, label, color=color, fontsize=fontsize, ha="center", va="bottom", zorder=11))
  if triad is not None:
    arrows, symbol, texts, bbox = triad
    ou = boxx + boxw / 2 - (bbox[0] + bbox[1]) / 2
    ov = boxy + 2.5 * gap + t - bbox[2]
    for a in arrows:
      artists.append(ax.add_patch(Polygon(a + (ou, ov), closed=True, facecolor=color, edgecolor="none", zorder=11)))
    if len(symbol):
      artists += ax.plot(symbol[:, 0] + ou, symbol[:, 1] + ov, color=color, linewidth=1.0, zorder=11)
    for q, s in texts:
      artists.append(ax.text(q[0] + ou, q[1] + ov, s, color=color, fontsize=0.85 * fontsize, fontweight="bold", ha="center",
                             va="center", zorder=12))
  mP.scaleBarArtists = artists
  return label


# the points of text a data unit takes, from the axes size in inches and its limits
def _pointsPerUnit(ax):
  fig = ax.figure
  bbox = ax.get_position()
  width = bbox.width * fig.get_figwidth() * 72
  xlo, xhi = ax.get_xlim()
  return width / max(abs(xhi - xlo), 1e-12)


def _hasColor(c):
  return c is not None and not (isinstance(c, np.ndarray) and np.any(np.isnan(c)))


def phaseLegend(mP, x, loc="upper right"):
  """the legend of the phases present in a map or grains, one patch per phase in its colour,
  a not indexed phase only when it was given a colour of its own"""
  from matplotlib.patches import Patch
  from matplotlib.colors import to_rgb
  pl = x.CSList
  ids = sorted(set(np.asarray(x.phaseId).reshape(-1).tolist()))
  # a not indexed phase enters the legend once it was given a colour of its own, as in MATLAB
  ids = [i for i in ids if 0 <= i < len(pl) and (pl.isIndexed(i) or _hasColor(pl[i].color))]
  palette = phasePalette(pl)
  handles = [Patch(facecolor=to_rgb(palette[i]), label=pl[i].mineral or pl[i].name or f"phase {i}") for i in ids]
  return mP.ax.legend(handles=handles, loc=loc)


def keyResolution(px):
  """the step a key of that many pixels is meshed at: about two pixels to a quad over a sector
  some ninety degrees across, never finer than a key drawn for a page nor coarser than the
  eye would catch"""
  return float(np.clip(180 * degree / max(px, 1.0), 0.5 * degree, 4 * degree))


def keyLegend(mP, x, ipfDirection=None, width=0.28, resolution=None):
  """the inverse pole figure key of every indexed phase present, drawn in insets beside the map,
  a sector reaching into both hemispheres as two; returns the inset axes"""
  from .directionkeys import HSVDirectionKey
  from .spherical import makeSphericalProjection
  from .sphericalplot import plotKey
  pl = x.CSList
  present = set(np.asarray(x.phaseId).reshape(-1).tolist())
  ids = [i for i in pl.indexedPhaseId if i in present]
  axes = []
  step = 1 / max(len(ids), 1)
  fontSize = max(6, min(10, 22 * step))
  # an inset a hundred pixels wide is meshed far coarser than a key drawn for a page
  fig = mP.ax.figure
  mesh = resolution if resolution is not None else keyResolution(fig.get_figwidth() * fig.dpi * mP.ax.get_position().width * width)
  for row, i in enumerate(ids):
    key = HSVDirectionKey(pl[i])
    # a sector reaching into both hemispheres, of a group without the inversion, takes two
    # insets in its row, the upper hemisphere over the lower
    n = len(makeSphericalProjection(key.keySector, how2plot=pl[i].how2plot))
    # the two keep a gap for the labels that stand below the upper and above the lower
    h, gap = (0.7 * step, 0.0) if n == 1 else (0.3 * step, 0.2 * step)
    for j in range(n):
      inset = mP.ax.inset_axes([1.08, 1 - (row + 0.9) * step - (n - 1) * 0.05 * step + (n - 1 - j) * (h + gap), width, h])
      side = {} if n == 1 else {"upper": True} if j == 0 else {"lower": True}
      plotKey(key, resolution=mesh, parent=inset, how2plot=pl[i].how2plot, fontSize=fontSize, **side)
      if n == 2:
        inset._mtexSphericalPlot.hemisphereLabel()
      if j == 0:
        inset.set_title(pl[i].mineral or pl[i].name, fontsize=fontSize + 1)
      axes.append(inset)
  return axes


# ------------------------------------------------------------------ images on the specimen

# the outline of an image, what a map plot needs to place it: its corners a half pixel out,
# the plane it lies in, its frame and unit
def _imageOutline(mg):
  from types import SimpleNamespace
  from ..maps.lattice import mapPlane
  r, c = mg.gridSize()
  C = mg._origin + np.array([-0.5, -0.5, r - 0.5, r - 0.5])[:, None] * mg._d1 + np.array([-0.5, c - 0.5, -0.5, c - 0.5])[:, None] * mg._d2
  return SimpleNamespace(V=C[:, :2], F=np.array([[0, 1], [1, 3], [3, 2], [2, 0]]), plane=mapPlane(C), frame=mg.frame, scanUnit=mg.scanUnit)


def plotMapImage(mg, channel=None, edge=False, micronbar=None, refFrame=True, colormap=None, layout=None, figSize=None, **kw: Unpack[PlotKeywords]):
  """plot(mg), plot(imgList), plot(imgList, edge=True), plot(mg, channel=1)

  An image drawn where it sits on the specimen, in the screen orientation of its frame; a list
  of images one axes each, titled by their names. One channel is grey through the colour map,
  three are drawn as RGB; edge draws the edge transform a registration matches on, its
  contrast stretched over the 2nd to 98th percentile.
  """
  from ..maps.mapimage import edgeMap, mapImage
  from .mtexfigure import newMtexFigure, nextAxis
  kw = mapAliases(kw)
  if isinstance(mg, (list, tuple)):
    fig = newMtexFigure(layout=layout, figSize=figSize) if layout is not None else newMtexFigure(len(mg), figSize=figSize)
    h = []
    for m in mg:
      nextAxis(fig=fig)
      h.append(plotMapImage(m, channel=channel, edge=edge, micronbar=micronbar, refFrame=refFrame, colormap=colormap,
                            title=m.name if m.name else None, **kw))
    return h
  if not isinstance(mg, mapImage):
    raise TypeError("plotMapImage draws a mapImage or a list of them")
  mapKw = _splitMap(kw)
  mapKw.setdefault("how2plot", mg.how2plot)
  x = _imageOutline(mg)
  mP = _target(x, mapKw)
  ax = mP.ax
  d = edgeMap(mg) if edge else np.array(mg.img, dtype=float)
  if d.ndim == 2:
    d = d[..., None]
  if channel is not None:
    d = d[..., np.atleast_1d(channel)]
  if d.shape[2] not in (1, 3):
    raise ValueError(f"a {d.shape[2]} channel image is not a picture on its own; name one with channel=k, or take three for RGB")
  # the rows and the columns of the array on the screen
  M = mP.layout
  s1, s2 = M @ mg._d1[:2], M @ mg._d2[:2]
  P = mg._posData()[..., :2] @ M.T
  aligned = min(abs(s1[0]), abs(s1[1])) < 1e-9 * np.linalg.norm(s1) and min(abs(s2[0]), abs(s2[1])) < 1e-9 * np.linalg.norm(s2)
  if d.shape[2] == 3:
    f = d[np.isfinite(d)]
    d = (d - f.min()) / (f.max() - f.min()) if f.size and f.max() > f.min() else d
  if edge:
    f = d[np.isfinite(d)]
    lo, hi = np.percentile(f, [2, 98], method="hazen") if f.size else (0, 1)
  else:
    lo, hi = np.nanmin(d), np.nanmax(d)
  cmap = mtexColormap(colormap if colormap is not None else "gray")
  from matplotlib.colors import Normalize
  norm = Normalize(lo, hi) if d.shape[2] == 1 and hi > lo else None
  if aligned:
    img = d
    if abs(s1[0]) > abs(s1[1]):
      img, s1, s2 = img.transpose(1, 0, 2), s2, s1
    if s1[1] < 0:
      img = img[::-1]
    if s2[0] < 0:
      img = img[:, ::-1]
    hx, hy = abs(s2[0]) / 2, abs(s1[1]) / 2
    ext = (P[..., 0].min() - hx, P[..., 0].max() + hx, P[..., 1].min() - hy, P[..., 1].max() + hy)
    if img.shape[2] == 1:
      h = ax.imshow(np.ma.masked_invalid(img[..., 0]), extent=ext, origin="lower", interpolation="nearest", cmap=cmap, norm=norm, zorder=1, **kw)
    else:
      rgba = np.concatenate([np.nan_to_num(np.clip(img, 0, 1)), np.isfinite(img).all(axis=2, keepdims=True).astype(float)], axis=2)
      h = ax.imshow(rgba, extent=ext, origin="lower", interpolation="nearest", zorder=1, **kw)
  else:
    # turned against the screen: the pixels as quadrilaterals
    r, c = mg.gridSize()
    i, j = np.meshgrid(np.arange(r + 1) - 0.5, np.arange(c + 1) - 0.5, indexing="ij")
    Q = (mg._origin[:2] + i[..., None] * mg._d1[:2] + j[..., None] * mg._d2[:2]) @ M.T
    if d.shape[2] == 1:
      h = ax.pcolormesh(Q[..., 0], Q[..., 1], np.ma.masked_invalid(d[..., 0]), cmap=cmap, norm=norm, zorder=1, **kw)
    else:
      h = ax.pcolormesh(Q[..., 0], Q[..., 1], np.nan_to_num(np.clip(d, 0, 1)), zorder=1, **kw)
  if d.shape[2] == 1:
    mP.mappable = h
  _finish(mP, x, micronbar, refFrame, False, "image")
  return h


__all__ = ["line", "scatterXY", "mapPlot", "plotEBSD", "plotGrains", "plotBoundary", "plotTriplePoints", "quiverMap", "scaleBarOn", "phaseLegend", "keyLegend",
           "keyResolution", "mapValues", "mapAliases"]
