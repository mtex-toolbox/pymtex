# the matplotlib drawing of EBSD and grain maps (MATLAB @EBSD/plot, @grain2d/plot,
# @grainBoundary/plot, @mapPlot, scaleBar; docs/plans/plotting.md S4)
#
# maps.py says what to draw: the picture of a rigid lattice, the cells of a drifted one,
# the points of a map with no lattice, the grains as polygons, the boundary as chains or as
# segments, the scale bar with the frame triad. This file draws it into one axes per map,
# under the layout of the plotting convention, and keeps a mapPlot on the axes so that a
# boundary drawn afterwards lands on the map it belongs to.
from __future__ import annotations

import numpy as np

from ..geometry.crystalshape import crystalShape
from ..geometry.orientation import orientation
from ..geometry.plottingconvention import plottingConvention
from ..geometry.session import getMTEXpref
from ..geometry.units import degree
from ..geometry.vector3d import vector3d, zvector
from .colormaps import mtexColormap
from .maps import (CELLS_MAX, boundaryPolyline, boundarySegments, frameOnScreen, grainPolygons, mapCells, mapExtent, mapLayout,
                   mapPoints, mapRaster, phasePalette, planeOf, scaleBar, triadGeometry)
from .mtexfigure import gcm, mtexFigure
from .orientationkeys import ipfColorKey, ipfColors, orientationColorKey

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
    self.mappable = None
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
      u = "µm" if self.unit == "um" else self.unit
      ax.set_xlabel(f"{right} / {u}")
      ax.set_ylabel(f"{up} / {u}")
    else:
      ax.set_xticks([])
      ax.set_yticks([])
    if title is not None:
      ax.set_title(title, fontsize=getMTEXpref("FontSize", 13))

  # which reference axis runs to the right and which up under the layout, by name and sign
  def screenAxisNames(self):
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
    return np.asarray(xy, dtype=float) @ self.layout.T

  # more data drawn into the map widen it, as hold on does in MATLAB, and the scale bar
  # follows the new limits
  def extend(self, x):
    xlo, xhi, ylo, yhi = mapExtent(x, self.layout)
    e = self.extent
    new = (min(e[0], xlo), max(e[1], xhi), min(e[2], ylo), max(e[3], yhi))
    if new == e:
      return
    self.extent = new
    self.ax.set_xlim(new[0], new[1])
    self.ax.set_ylim(new[2], new[3])
    if getattr(self, "scaleBarArtists", None):
      for a in self.scaleBarArtists:
        a.remove()
      args = self.scaleBarArgs
      self.scaleBarArtists = []
      scaleBarOn(self, None, **args)


# ------------------------------------------------------------------ where a map goes

def _kind(x):
  if hasattr(x, "lattice") and hasattr(x, "rotations"):
    return "ebsd"
  if hasattr(x, "loops"):
    return "grains"
  if hasattr(x, "F") and hasattr(x, "misrotation"):
    return "boundary"
  if hasattr(x, "boundaryId") and hasattr(x, "angles"):
    return "triplePoints"
  return None


# the map plot a command draws into: the parent named, the current axes when it is held
# and holds a map, else a new figure sized after the map
def _target(x, mapKw):
  parent = mapKw.get("parent")
  if parent is not None:
    mP = getattr(parent, "_mtexMapPlot", None)
    return mP if mP is not None else mapPlot(parent, x, **{k: v for k, v in mapKw.items() if k not in ("parent", "figSize")})
  fig = gcm()
  if fig is not None and fig.hold and fig.children and hasattr(fig.children[-1], "_mtexMapPlot"):
    mP = fig.children[-1]._mtexMapPlot
    mP.extend(x)
    mP.reused = True
    return mP
  # the axes nextAxis prepared takes the map, a fresh figure of newMtexFigure its first axes
  if fig is not None and fig.pending is not None:
    return mapPlot(fig.takeAxis(), x, **{k: v for k, v in mapKw.items() if k not in ("parent", "figSize")})
  if fig is not None and not fig.children and not fig.hold:
    return mapPlot(fig.nextAxis(), x, **{k: v for k, v in mapKw.items() if k not in ("parent", "figSize")})
  layout = mapLayout(mapKw.get("how2plot"))
  xlo, xhi, ylo, yhi = mapExtent(x, layout)
  w, h = max(xhi - xlo, 1e-12), max(yhi - ylo, 1e-12)
  from .mtexfigure import figureSize
  figSize = figureSize(mapKw.get("figSize"), (7.0, min(max(7.0 * h / w, 2.5), 9.0)) if w >= h else (max(7.0 * w / h, 3.0), 7.0))
  fig = mtexFigure(1, figSize=figSize)
  return mapPlot(fig.nextAxis(), x, **{k: v for k, v in mapKw.items() if k not in ("parent", "figSize")})


def _splitMap(kw):
  return {k: kw.pop(k) for k in MAP_KEYS if k in kw}


# MATLAB's option names that the map plots know under another spelling
def mapAliases(kw):
  for old, new in (("translucent", "faceAlpha"), ("displayName", "DisplayName"), ("lineColour", "lineColor"),
                   ("markerSize", "MarkerSize"), ("markersize", "MarkerSize"), ("linewidth", "lineWidth"),
                   ("linecolor", "lineColor"), ("LineWidth", "lineWidth"), ("LineColor", "lineColor")):
    if old in kw:
      kw[new] = kw.pop(old)
  return kw


# ------------------------------------------------------------------ what is coloured

# the colour of every measurement or grain, and whether it is a scalar to be mapped:
# the phase palette, an orientation list through the inverse pole figure key of its
# phase, an orientation colour key, the name of a property, one value or one colour per
# element, or one colour for all
# the value or the colour of every element, or of the elements `at` alone: a picture shows
# fewer measurements than a large map holds, and the colour of one it never shows is work
# thrown away. A route that yields numbers ignores `at` and stays whole, since the colour
# range is the map's and not the picture's; a route that yields colours returns one per
# entry of `at`.
def mapValues(x, color, faceColor=None, ipfDirection=None, at=None):
  n = x.size
  ids = np.asarray(x.phaseId).reshape(-1)
  sel = slice(None) if at is None else np.asarray(at).reshape(-1)
  m = n if at is None else len(sel)
  if faceColor is not None and color is None:
    from matplotlib.colors import to_rgb
    return np.tile(to_rgb(faceColor), (m, 1)), False
  if color is None or (isinstance(color, str) and color == "phase"):
    palette = phasePalette(x.CSList)
    from matplotlib.colors import to_rgb
    rgb = np.array([to_rgb(c) for c in palette])
    return rgb[ids[sel]], False
  if isinstance(color, str) and color == "orientation":
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
    rots = x.meanRotation if hasattr(x, "meanRotation") else x.rotations
    q = rots.data.reshape(-1, 4)[sel]
    for i in x.CSList.indexedPhaseId:
      fr = x.CSList[i]
      same = fr is color.CS1 or fr.Laue() is color.CS1.Laue()
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
    from matplotlib.colors import to_rgb
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
  finite = values[np.isfinite(values)]
  lo, hi = (finite.min(), finite.max()) if colorRange is None and finite.size else (0.0, 1.0) if colorRange is None else colorRange
  if not hi > lo:
    hi = lo + max(abs(lo), 1.0) * 1e-6
  return Normalize(lo, hi)


# ------------------------------------------------------------------ the EBSD map

def plotEBSD(ebsd, color=None, faceColor=None, backend=None, colormap="viridis", colorRange=None, colorbar=False, micronbar=None,
             refFrame=True, location=None, ipfDirection=None, maxPixels=None, edgeColor=None, lineWidth=0.5, alpha=1.0, legend=None, DisplayName=None,
             MarkerSize=4, **kw):
  """plot(ebsd), plot(ebsd, values), plot(ebsd, ori), plot(ebsd, rgb), plot(ebsd, faceColor='black')

  A spatial EBSD map coloured by phase, by a value per measurement, by orientations through
  the inverse pole figure key of their phase, by an orientation colour key, or in one
  colour. backend is 'imagesc' for the picture a rigid lattice is, 'patch' for one cell per
  measurement, 'scatter' for the measurements alone; by default the first the map allows.
  """
  # plot(ebsd, cS): the crystal shape at every measurement of its phase
  if isinstance(color, crystalShape):
    from .shapeplot import plotShapesOnMap
    return plotShapesOnMap(ebsd, color, faceColor=faceColor, faceAlpha=kw.pop("faceAlpha", alpha), edgeColor=kw.pop("edgeColor", edgeColor or "k"),
                           lineWidth=lineWidth, colormap=colormap, micronbar=micronbar, refFrame=refFrame, DisplayName=DisplayName, **kw)
  location = kw.pop("Location", location)
  alpha = kw.pop("FaceAlpha", kw.pop("faceAlpha", alpha))
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
      a = np.nan_to_num(np.asarray(alpha, dtype=float), nan=0.0)
      alphaImg = alpha if a.ndim == 0 else np.where(covered, a.reshape(-1)[np.clip(index, 0, None)], 0.0)
      h = ax.imshow(np.ma.masked_invalid(img), extent=(xlo, xhi, ylo, yhi), origin="lower", interpolation="nearest",
                    cmap=cmap, norm=norm, alpha=alphaImg, zorder=1, **kw)
    else:
      # the colours of the pixels as bytes, the map's own colours never assembled
      rgb = values.reshape(index.shape + (3,)) if at is not None else values[pick]
      img = np.zeros(index.shape + (4,), dtype=np.uint8)
      img[..., :3] = (np.nan_to_num(np.clip(rgb, 0, 1)) * 255 + 0.5).astype(np.uint8)
      # one transparency for all or one per measurement; a measurement whose colour is no
      # colour, an unindexed one, is drawn as nothing at all
      a = np.nan_to_num(np.asarray(alpha, dtype=float), nan=0.0)
      shown = covered & np.all(np.isfinite(rgb), axis=-1)
      img[..., 3] = np.where(shown, int(a * 255 + 0.5) if a.ndim == 0 else (a.reshape(-1)[pick] * 255 + 0.5).astype(np.uint8), 0)
      h = ax.imshow(img, extent=(xlo, xhi, ylo, yhi), origin="lower", interpolation="nearest", zorder=1, **kw)
  elif style == "raster":
    raise ValueError("the map is not a rigid lattice, so it has no picture; backend='patch' draws its cells")
  elif ebsd.lattice is not None and style != "points":
    from matplotlib.collections import PolyCollection
    corners = mapCells(ebsd, layout=mP.layout)
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
    if scalar:
      h = ax.scatter(P[:, 0], P[:, 1], c=values, cmap=cmap, norm=norm, s=MarkerSize ** 2, linewidths=0, alpha=alpha, zorder=1, **kw)
    else:
      h = ax.scatter(P[:, 0], P[:, 1], c=values, s=MarkerSize ** 2, linewidths=0, alpha=alpha, zorder=1, **kw)
  if DisplayName is not None:
    h.set_label(DisplayName)
  if scalar:
    mP.mappable = h
    if colorbar:
      gcm().colorbar(label=color if isinstance(color, str) else "")
  _finish(mP, ebsd, micronbar, refFrame, legend, color, location)
  return h


# twice the pixels of the axes on each side, at the figure's dpi
def _pixelCap(ax):
  fig = ax.figure
  pos = ax.get_position()
  w, h = fig.get_figwidth() * fig.dpi * pos.width, fig.get_figheight() * fig.dpi * pos.height
  return int(max(4 * w * h, 250_000))


def _finish(mP, x, micronbar, refFrame, legend, color, location=None):
  # a plot into a held map adds no second scale bar
  if micronbar is None:
    micronbar = getMTEXpref("showMicronBar", True) and not getattr(mP, "reused", False)
  if micronbar and not getattr(mP, "scaleBarArtists", None):
    scaleBarOn(mP, x, location=location or "sw", refFrame=refFrame if getMTEXpref("showRefFrame", True) else False)
  # a phase map carries its legend unless told not to, as MATLAB draws it
  if legend is None:
    legend = color is None or (isinstance(color, str) and color == "phase")
  if legend:
    if color is None or (isinstance(color, str) and color == "phase"):
      phaseLegend(mP, x)
    elif (isinstance(color, str) and color == "orientation") or isinstance(color, ipfColorKey):
      keyLegend(mP, x, ipfDirection=color.ipfDirection if isinstance(color, ipfColorKey) else zvector)


# ------------------------------------------------------------------ the grains and the boundary

def plotGrains(grains, color=None, faceColor=None, colormap="viridis", colorRange=None, colorbar=False, micronbar=None, refFrame=True,
               location=None, ipfDirection=None, noBoundary=False, lineWidth=1.0, lineColor="k", alpha=1.0, legend=None, DisplayName=None, **kw):
  """plot(grains), plot(grains, property), plot(grains, grains.meanOrientation)

  The grains drawn as the polygons the boundary encloses, coloured by phase, by a value per
  grain, by their mean orientations or an orientation key, or in one colour, with the
  boundary on top unless noBoundary.
  """
  # MATLAB's capitalised spellings
  alpha = kw.pop("FaceAlpha", kw.pop("faceAlpha", alpha))
  lineWidth = kw.pop("LineWidth", lineWidth)

  from matplotlib.path import Path
  # plot(grains, v): an axis per grain as a compass needle at its centroid
  if isinstance(color, vector3d):
    return needlesOnMap(grains, color, micronbar=micronbar, refFrame=refFrame, **kw)
  # plot(grains, cS): the crystal shape of every grain into the map
  if isinstance(color, crystalShape):
    from .shapeplot import plotShapesOnMap
    return plotShapesOnMap(grains, color, faceColor=faceColor, faceAlpha=kw.pop("faceAlpha", alpha), edgeColor=kw.pop("edgeColor", lineColor),
                           lineWidth=lineWidth, colormap=colormap, micronbar=micronbar, refFrame=refFrame, DisplayName=DisplayName, **kw)
  location = kw.pop("Location", location)
  mapKw = _splitMap(kw)
  mP = _target(grains, mapKw)
  ax = mP.ax
  values, scalar = mapValues(grains, color, faceColor, ipfDirection)
  cmap = mtexColormap(colormap)
  # every grain is one path with its holes, all of them one collection
  paths, at = [], []
  for k in range(grains.size):
    rings = grainPolygons(grains, k)
    if not rings or len(rings[0]) < 3:
      continue
    verts, codes = [], []
    for ring in rings:
      ring = mP.onScreen(ring)
      verts += list(ring) + [ring[0]]
      codes += [Path.MOVETO] + [Path.LINETO] * (len(ring) - 1) + [Path.CLOSEPOLY]
    paths.append(Path(np.array(verts), codes))
    at.append(k)
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
  ax.add_collection(h)
  if DisplayName is not None:
    h.set_label(DisplayName)
  if not noBoundary and grains.boundary is not None:
    _boundaryLines(mP, grains.boundary, lineColor, lineWidth, zorder=3)
  if scalar and colorbar:
    gcm().colorbar(label=color if isinstance(color, str) else "")
  _finish(mP, grains, micronbar, refFrame, legend, color, location)
  return h


# an axis per grain as MATLAB draws it: a needle through the centroid, grey when it lies
# in the plane of the map, its half pointing out of the screen black and the other white
# when it does not, half a diameter long
def needlesOnMap(grains, v, micronbar=None, refFrame=True, lineWidth=2.0, **kw):
  from matplotlib.collections import LineCollection
  mapKw = _splitMap(kw)
  mP = _target(grains, mapKw)
  d = np.asarray(v.normalize().data, dtype=float).reshape(-1, 3)
  c = grains.centroid.data.reshape(-1, 3)
  length = np.asarray(grains.diameter).reshape(-1) / 4
  n = mP.plane.normal * (-1 if np.linalg.det(mP.layout) < 0 else 1)
  out = d @ n
  inPlane = d - out[:, None] * n
  norm = np.linalg.norm(inPlane, axis=1)
  u = np.where(norm[:, None] > 1e-12, inPlane / np.where(norm[:, None] > 1e-12, norm[:, None], 1), 0.0)
  segs, colors = [], []
  for k in range(len(c)):
    a, b = c[k] - length[k] * u[k], c[k] + length[k] * u[k]
    p0, pa, pb = (mP.plane.to2d(x) @ mP.layout.T for x in (c[k], a, b))
    if abs(out[k]) < 0.1:
      segs.append([pa, pb])
      colors.append((0.5, 0.5, 0.5, 1))
    else:
      segs += [[p0, pb], [p0, pa]]
      colors += [(0, 0, 0, 1) if out[k] > 0 else (1, 1, 1, 1), (1, 1, 1, 1) if out[k] > 0 else (0, 0, 0, 1)]
  h = LineCollection(segs, colors=colors, linewidths=lineWidth, zorder=6, path_effects=None)
  mP.ax.add_collection(h)
  mP.ax.scatter(*((mP.plane.to2d(c) @ mP.layout.T).T), s=lineWidth ** 2 * 3, color="k", zorder=7)
  _finish(mP, grains, micronbar, refFrame, False, None)
  return h


def _boundaryLines(mP, gB, color, lineWidth, zorder=3, **kw):
  P = mP.onScreen(boundaryPolyline(gB))
  h, = mP.ax.plot(P[:, 0], P[:, 1], color=color, linewidth=lineWidth, solid_joinstyle="miter", snap=True, zorder=zorder, **kw)
  return h


def plotBoundary(gB, color=None, lineColor="k", lineWidth=1.0, colormap="viridis", colorRange=None, colorbar=False, edgeAlpha=1.0,
                 micronbar=None, refFrame=True, DisplayName=None, **kw):
  """plot(gB), plot(gB, values), plot(gB, rgb), plot(gB, lineColor='r', lineWidth=2)

  The grain boundary segments: as joined chains in one colour, or segment by segment with
  a value or a colour per segment, angle(gB.misorientation) say.
  """
  from matplotlib.collections import LineCollection
  mapKw = _splitMap(kw)
  mP = _target(gB, mapKw)
  ax = mP.ax
  n = len(gB)
  if color is None:
    if np.ndim(edgeAlpha) > 0:
      # one transparency per segment: the segments as a collection in their own RGBA colours
      from matplotlib.collections import LineCollection
      from matplotlib.colors import to_rgba
      V = boundarySegments(gB)
      segs = mP.onScreen(V.reshape(-1, V.shape[-1])).reshape(-1, 2, 2)
      rgba = np.tile(to_rgba(lineColor), (len(segs), 1))
      rgba[:, 3] = np.clip(np.asarray(edgeAlpha, dtype=float).reshape(-1), 0, 1)
      h = LineCollection(segs, colors=rgba, linewidths=lineWidth, snap=True, zorder=3, **kw)
      mP.ax.add_collection(h)
    else:
      h = _boundaryLines(mP, gB, lineColor, lineWidth, alpha=edgeAlpha, **kw)
  else:
    arr = np.asarray(color)
    segs = mP.onScreen(boundarySegments(gB).reshape(-1, 2)).reshape(n, 2, 2)
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
  if DisplayName is not None and h is not None:
    h.set_label(DisplayName)
  # a boundary carries no phase legend of its own
  _finish(mP, gB, micronbar, refFrame, False, None)
  return h



def plotTriplePoints(tP, color=None, markerColor=None, markerEdgeColor=None, MarkerSize=6, lineWidth=1.0, colormap="viridis",
                     colorRange=None, colorbar=False, micronbar=None, refFrame=True, DisplayName=None, **kw):
  """plot(tP), plot(tP, values): the triple points of a boundary network as circles, filled
  by a value where one is given"""
  mapKw = _splitMap(kw)
  mP = _target(tP, mapKw)
  P = mP.onScreen(np.asarray(tP.V).reshape(-1, 2))
  edge = markerEdgeColor if markerEdgeColor is not None else (color if color is not None and isinstance(color, str) else "k")
  size = float(MarkerSize) ** 2
  if color is None or isinstance(color, str):
    face = markerColor if markerColor is not None else "none"
    h = mP.ax.scatter(P[:, 0], P[:, 1], s=size, facecolors=face, edgecolors=edge, linewidths=lineWidth, zorder=6,
                      label=DisplayName, **kw)
  else:
    arr = np.asarray(color, dtype=float).reshape(-1)
    h = mP.ax.scatter(P[:, 0], P[:, 1], s=size, c=arr, cmap=mtexColormap(colormap), norm=_norm(arr, colorRange),
                      edgecolors=edge, linewidths=lineWidth, zorder=6, label=DisplayName, **kw)
    mP.mappable = h
    if colorbar:
      gcm().colorbar()
  _finish(mP, tP, micronbar, refFrame, False, None)
  return h


# ------------------------------------------------------------------ arrows on a map

# text(ebsd, labels): a label at every measurement or grain of the current map, or a new one
def textMap(x, labels, fontSize=None, color="k", **kw):
  fig = gcm()
  if fig is not None and fig.children and hasattr(fig.children[-1], "_mtexMapPlot"):
    mP = fig.children[-1]._mtexMapPlot
  else:
    mP = _target(x, _splitMap(kw))
  P = mapPoints(x, mP.layout) if not hasattr(x, "numPixel") else mP.plane.to2d(x.centroid.data.reshape(-1, 3)) @ np.asarray(mP.layout, dtype=float).T
  labels = [labels] * len(P) if isinstance(labels, str) else [str(t) for t in np.asarray(labels, dtype=object).ravel()]
  size = getMTEXpref("FontSize", 13) - 4 if fontSize is None else fontSize
  return [mP.ax.text(px, py, t, ha="center", va="center", fontsize=size, color=color, zorder=8, clip_on=True, **kw) for (px, py), t in zip(P, labels)]


def quiverMap(x, v, scale=None, color="k", lineWidth=1.0, noScaling=False, DisplayName=None, autoScaleFactor=None, **kw):
  """quiver(grains, v), quiver(ebsd, v), quiver(gB, v): an arrow per grain at its centroid,
  per measurement, per segment at its midpoint; an axis is centred, a direction starts there.
  By default an arrow is a fifth of the grain's diameter, one and a half spacings of a map,
  three median segment lengths of a boundary; scale draws it scale times its in-plane length.
  """
  from matplotlib.collections import LineCollection
  mapKw = _splitMap(kw)
  mP = _target(x, mapKw)
  v = v if isinstance(v, vector3d) else vector3d(v)
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
  u = np.stack([raw @ pl.e1, raw @ pl.e2], axis=-1)
  n = np.linalg.norm(u, axis=-1)
  ok = (n > 0) & np.all(np.isfinite(raw), axis=-1)
  length = n[ok] if noScaling else default[ok] if scale is None else scale * n[ok]
  u = u[ok] / n[ok][:, None] * length[:, None]
  c = c[ok]
  if v.antipodal:
    segs = np.stack([mP.onScreen(c - u / 2), mP.onScreen(c + u / 2)], axis=1)
    h = LineCollection(segs, colors=color, linewidths=lineWidth, zorder=4, **kw)
    mP.ax.add_collection(h)
    return h
  p, d = mP.onScreen(c), u @ mP.layout.T
  return mP.ax.quiver(p[:, 0], p[:, 1], d[:, 0], d[:, 1], angles="xy", scale_units="xy", scale=1, color=color, width=0.003, label=DisplayName,
                      zorder=4, **kw)


# ------------------------------------------------------------------ the scale bar and the legends

# a scale bar in one corner of a map: a round length near a tenth of its width, over the
# length it stands for, and above it the axes of the specimen frame as they lie in the picture
def scaleBarOn(mP, x=None, location="sw", fraction=0.1, refFrame=True, color="k", background="w", alpha=0.75, Length=None):
  from matplotlib.patches import Polygon, Rectangle
  ax = mP.ax
  mP.scaleBarArgs = dict(location=location, fraction=fraction, refFrame=refFrame, color=color, background=background, alpha=alpha,
                         Length=Length)
  artists = []
  xlo, xhi, ylo, yhi = mP.extent
  if Length is None:
    length, label = scaleBar(xhi - xlo, mP.unit, fraction)
  else:
    length, label = float(Length), f"{Length:g} {'µm' if mP.unit == 'um' else mP.unit}"
  if not length > 0:
    return None
  w, h = xhi - xlo, yhi - ylo
  t = 0.035 * min(w, h)
  gap = t / 4
  names = tuple(mP.frame.axesNames) if mP.frame is not None else ("X", "Y", "Z")
  triad = triadGeometry(frameOnScreen(mP.plane, mP.layout), list(names), t) if refFrame else None
  tw, th = (0.0, 0.0) if triad is None else (triad[3][1] - triad[3][0], triad[3][3] - triad[3][2])
  boxw = max(length, 0.6 * t * len(label), tw) + 2 * gap
  boxh = 3.5 * gap + t + th if refFrame else 3 * gap + t
  boxx = xhi - gap - boxw if location in ("se", "ne") else xlo + gap
  boxy = yhi - gap - boxh if location in ("nw", "ne") else ylo + gap
  # the text is the unit of the box: its size follows the map on the screen
  fontsize = _pointsPerUnit(ax) * 0.9 * t
  artists.append(ax.add_patch(Rectangle((boxx, boxy), boxw, boxh, facecolor=background, alpha=alpha, edgecolor=color, linewidth=0.8,
                                        zorder=10)))
  x0 = boxx + (boxw - length) / 2
  artists += ax.plot([x0, x0 + length], [boxy + 1.5 * gap] * 2, color=color, linewidth=max(1.0, 0.12 * t * _pointsPerUnit(ax)),
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
  from matplotlib.patches import Patch
  from matplotlib.colors import to_rgb
  pl = x.CSList
  ids = sorted(set(np.asarray(x.phaseId).reshape(-1).tolist()))
  # a not indexed phase enters the legend once it was given a colour of its own, as in MATLAB
  ids = [i for i in ids if 0 <= i < len(pl) and (pl.isIndexed(i) or _hasColor(pl[i].color))]
  palette = phasePalette(pl)
  handles = [Patch(facecolor=to_rgb(palette[i]), label=pl[i].mineral or pl[i].name or f"phase {i}") for i in ids]
  return mP.ax.legend(handles=handles, loc=loc)


# the inverse pole figure key of every indexed phase present, drawn beside the map
# the step a key of that many pixels is meshed at: about two pixels to a quad over a sector
# some ninety degrees across, never finer than a key drawn for a page nor coarser than the
# eye would catch
def keyResolution(px):
  return float(np.clip(180 * degree / max(px, 1.0), 0.5 * degree, 4 * degree))


def keyLegend(mP, x, ipfDirection=None, width=0.28, resolution=None):
  from .directionkeys import HSVDirectionKey
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
    inset = mP.ax.inset_axes([1.08, 1 - (row + 0.9) * step, width, 0.7 * step])
    key = HSVDirectionKey(pl[i])
    plotKey(key, resolution=mesh, parent=inset, how2plot=pl[i].how2plot, fontSize=fontSize)
    inset.set_title(pl[i].mineral or pl[i].name, fontsize=fontSize + 1)
    axes.append(inset)
  return axes


__all__ = ["mapPlot", "plotEBSD", "plotGrains", "plotBoundary", "plotTriplePoints", "quiverMap", "scaleBarOn", "phaseLegend", "keyLegend",
           "keyResolution", "mapValues", "mapAliases"]
