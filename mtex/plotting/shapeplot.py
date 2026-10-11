# the drawing of crystal shapes (MATLAB @crystalShape/plot, plotInnerFace, plotInnerDirection,
# plotSlipSystem, @grain2d/plot with a crystalShape; docs/plans/plotting.md S5)
#
# A shape stands in a 3D axes as the polygons of its faces. Drawn into a held map it is
# seen along the normal of the map, its front faces as polygons on the screen; space.py
# says which faces those are and where they lie. matplotlib is imported inside the
# functions.
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from ..geometry.kernels import columnMax, columnMin
from ..geometry.crystalshape import crystalShape
from ..geometry.session import getMTEXpref
from ..geometry.slipsystem import slipSystem
from ..geometry.vector3d import vector3d, asVector3d
from .colormaps import mtexColormap
from .keywords import refuseVariants
from .mtexfigure import gcm
from .space import shapeOutlines, shapesOnMap
from .spaceplot import _spaceAxes

if TYPE_CHECKING:
  from typing import Unpack

  from .keywords import PlotKeywords

# MTEX's colour of a crystal without a mineral colour, LightSkyBlue
SHAPE_COLOR = (0.53, 0.81, 0.98)
# the colours of the forms under colored, matplotlib's own cycle
FORM_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf")


# ------------------------------------------------------------------ where a shape goes

# the held map plot, or a 3D axes: the parent named, the current axes when it is held, else
# a new figure
def _target(cS, kw):
  axesKw = {k: kw.pop(k) for k in ("figSize", "parent", "azimuth", "elevation", "title") if k in kw}
  parent = axesKw.get("parent")
  if parent is not None and hasattr(parent, "_mtexMapPlot"):
    return "map", parent._mtexMapPlot
  if parent is not None and hasattr(parent, "_mtexSpacePlot"):
    return "space", parent._mtexSpacePlot
  if parent is None:
    fig = gcm()
    if fig is not None and fig.hold and fig.children and hasattr(fig.children[-1], "_mtexMapPlot"):
      return "map", fig.children[-1]._mtexMapPlot
  return "space", _spaceAxes(cS.frame, **axesKw)


# the axes fitted around what is drawn, at equal scale, widened when it held something
def _fit(sP, P, held):
  P = np.asarray(P, dtype=float).reshape(-1, 3)
  lo, hi = columnMin(P), columnMax(P)
  pad = 0.05 * max(float(np.max(hi - lo)), 1e-12)
  lo, hi = lo - pad, hi + pad
  if held:
    cur = np.array([sP.ax.get_xlim(), sP.ax.get_ylim(), sP.ax.get_zlim()])
    lo, hi = np.minimum(lo, cur[:, 0]), np.maximum(hi, cur[:, 1])
  sP.ax.set_xlim(lo[0], hi[0])
  sP.ax.set_ylim(lo[1], hi[1])
  sP.ax.set_zlim(lo[2], hi[2])
  sP.ax.set_box_aspect(tuple(hi - lo))


# polygons in space into a 3D axes, or through the plane of a map onto its screen. In
# space every shape of the axes joins one collection, since matplotlib orders its
# collections by depth as wholes and only the faces within one against each other; the
# alpha goes into the colours for that
def _draw(kind, target, polys, colors, edgeColor, lineWidth, alpha, label=None, zorder=4):
  from matplotlib.colors import to_rgba
  if kind in ("map", "chart"):
    from matplotlib.collections import PolyCollection
    # on a chart the polygons come in the coordinates of the axes already
    flat = [target.plane.to2d(p) @ target.layout.T for p in polys] if kind == "map" else [np.asarray(p, dtype=float) for p in polys]
    h = PolyCollection(flat, facecolors=colors, edgecolors=edgeColor, linewidths=lineWidth, alpha=alpha, zorder=zorder + 2)
    target.ax.add_collection(h)
    if label is not None:
      h.set_label(label)
    return h
  from mpl_toolkits.mplot3d.art3d import Poly3DCollection
  colors = np.asarray(colors, dtype=float).reshape(-1, 4).copy()
  colors[:, 3] *= alpha
  edges = np.broadcast_to(to_rgba(edgeColor) if edgeColor is not None else (0, 0, 0, 0), (len(polys), 4))
  widths = np.full(len(polys), lineWidth)
  polys = [np.asarray(p, dtype=float) for p in polys]
  h = getattr(target, "shapes", None)
  if h is None or h.axes is not target.ax:
    held = len(target.ax.collections) > 0
    h = Poly3DCollection(polys, facecolors=colors, edgecolors=edges, linewidths=widths)
    target.ax.add_collection3d(h)
    target.shapes = h
    target.shapeParts = (polys, colors, edges, widths)
    if label is not None:
      h.set_label(label)
  else:
    held = True
    P, C, E, W = target.shapeParts
    P, C, E, W = P + polys, np.concatenate([C, colors]), np.concatenate([E, edges]), np.concatenate([W, widths])
    target.shapeParts = (P, C, E, W)
    h.set_verts(P)
    h.set_facecolor(C)
    h.set_edgecolor(E)
    h.set_linewidths(W)
  _fit(target, np.concatenate(polys), held)
  return h


# ------------------------------------------------------------------ what colour a face is

def faceColors(cS, faceColor=None, colored=None, colormap="viridis"):
  """the colour of every face of every shape, an (n, nF, 4) array: the colour of the mineral
  or LightSkyBlue, one colour, one per face, one per shape, a value per face through the
  colour map, or with colored a colour per form or the direction key of the normals"""
  from matplotlib.colors import Normalize, to_rgba
  n, nF = len(cS), cS.numFaces
  if colored is not None and colored is not False:
    if colored is True:
      forms = np.array([to_rgba(FORM_COLORS[k % len(FORM_COLORS)]) for k in range(len(cS.N))])
    else:
      key = colored.dirMap if hasattr(colored, "dirMap") else colored
      rgb = np.asarray(key.direction2color(cS.N), dtype=float).reshape(-1, 3)
      forms = np.concatenate([rgb, np.ones((len(rgb), 1))], axis=1)
    return np.broadcast_to(forms[cS._form], (n, nF, 4)).copy()
  if faceColor is None:
    c = cS.CS.color
    faceColor = SHAPE_COLOR if c is None else c
  if isinstance(faceColor, str) or (isinstance(faceColor, (tuple, list)) and len(faceColor) in (3, 4) and np.isscalar(faceColor[0])):
    return np.broadcast_to(to_rgba(faceColor), (n, nF, 4)).copy()
  fc = np.asarray(faceColor, dtype=float)
  if fc.ndim == 1:
    if len(fc) != nF:
      raise ValueError(f"a value per face has {nF} entries, got {len(fc)}")
    lo, hi = float(np.nanmin(fc)), float(np.nanmax(fc))
    rgba = mtexColormap(colormap)(Normalize(lo, hi if hi > lo else lo + 1)(fc))
    return np.broadcast_to(rgba, (n, nF, 4)).copy()
  if fc.ndim == 2 and fc.shape[1] in (3, 4):
    rgba = np.concatenate([fc, np.ones((len(fc), 1))], axis=1) if fc.shape[1] == 3 else fc
    if len(fc) == nF:
      return np.broadcast_to(rgba, (n, nF, 4)).copy()
    if len(fc) == n:
      return np.broadcast_to(rgba[:, None, :], (n, nF, 4)).copy()
  raise ValueError(f"faceColor is one colour, one per face ({nF}) or one per shape ({n})")


# the polygon legend of the forms, MATLAB's colored legend
def _formLegend(ax, cS, colors):
  from matplotlib.patches import Patch
  handles = []
  for k in range(len(cS.N)):
    at = np.nonzero(cS._form == k)[0]
    if len(at):
      handles.append(Patch(facecolor=colors[0, at[0]], edgecolor="k", label=cS.N[k].round().char()))
  ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=getMTEXpref("FontSize", 13) - 2)


# ------------------------------------------------------------------ the shape

# plot(x, y, cS), plot(x, y, z, cS), plot(xy, cS): the positions before the shape
def _parse(args):
  args = list(args)
  at = next(k for k, a in enumerate(args) if isinstance(a, crystalShape))
  cS, pos, rest = args[at], args[:at], args[at + 1:]
  if pos:
    xyz = np.asarray(pos[0], dtype=float)
    if len(pos) > 1:
      cols = [np.asarray(p, dtype=float).reshape(-1) for p in pos]
      xyz = np.stack(np.broadcast_arrays(*cols), axis=-1)
    cS = cS + xyz
  return cS, rest


def plotShape(*args, faceColor=None, faceAlpha=None, edgeColor="k", lineWidth=0.8, colored=None, colormap="viridis",
              displayName=None, **kw: Unpack[PlotKeywords]):
  """plot(cS), plot(cS, faceColor='red', faceAlpha=0.5), plot(cS, colored=True), plot(cS, colored=key),
  plot(x, y, cS), plot(x, y, z, cS), plot(xy, cS), plot(cS, N), plot(cS, sS)

  A crystal shape as the polygons of its faces in a 3D axes, or seen along the normal of a
  held map; colored gives every form its own colour with a legend, or the colours of the
  direction key of the normals. With a plane normal the section through the centre, with
  a slip system its plane and its direction inside the shape.
  """
  refuseVariants(kw)
  cS, rest = _parse(args)
  if rest and isinstance(rest[0], slipSystem):
    # a slip plane is translucent unless told otherwise, as MATLAB's plotSlipSystem
    return plotSlipSystem(cS, rest[0], faceColor=faceColor if faceColor is not None else "red",
                          faceAlpha=0.5 if faceAlpha is None else faceAlpha, edgeColor=edgeColor,
                          lineWidth=lineWidth, displayName=displayName, **kw)
  faceAlpha = 1.0 if faceAlpha is None else faceAlpha
  if rest and isinstance(rest[0], vector3d):
    return plotInnerFace(cS, rest[0], faceColor=faceColor if faceColor is not None else "blue", faceAlpha=faceAlpha, edgeColor=edgeColor,
                         lineWidth=lineWidth, displayName=displayName, **kw)
  colors = faceColors(cS, faceColor, colored, colormap)
  kind, target = _target(cS, kw)
  V = cS.V.data.reshape(len(cS), cS.numVertices, 3)
  if kind == "map":
    polys, shape, face = shapeOutlines(cS, target.plane, target.layout)
    polys = [target.plane.from2d(p @ np.linalg.inv(target.layout).T) for p in polys]
    colors = colors[shape, face]
  else:
    polys, colors = _facePolygons(cS, V), colors.reshape(-1, 4)
  h = _draw(kind, target, polys, colors, edgeColor, lineWidth, faceAlpha, displayName)
  if colored is not None and colored is not False:
    _formLegend(target.ax, cS, faceColors(cS, faceColor, colored, colormap))
  return h


# every face of every shape as a polygon in space, shape by shape
def _facePolygons(cS, V):
  return [V[k, f[f >= 0]] for k in range(len(cS)) for f in cS.F]


# ------------------------------------------------------------------ inside the shape

def plotInnerFace(cS, N, faceColor="blue", faceAlpha=1.0, edgeColor="k", lineWidth=0.8, displayName=None, **kw: Unpack[PlotKeywords]):
  """plotInnerFace(cS, N): the section of the shape by the lattice plane N through its centre"""
  from matplotlib.colors import to_rgba
  N = asVector3d(N)
  kind, target = _target(cS, kw)
  polys = []
  for n in (N.flatten() if N.ndim > 0 else [N]):
    pts = cS.innerFace(n).data.reshape(-1, 3)
    if len(pts) >= 3:
      polys.append(pts)
  if not polys:
    return None
  return _draw(kind, target, polys, [to_rgba(faceColor)] * len(polys), edgeColor, lineWidth, faceAlpha, displayName)


# an arrow from the centre of the shape along d, up to where d leaves the shape
def _arrow(kind, target, cS, d, color, lineWidth, label=None):
  u = cS._direction(d)
  c = cS.V.data.reshape(-1, 3).mean(axis=0)
  a = 0.95 * cS.exitDistance(u) * u
  if kind == "map":
    p, q = target.plane.to2d(c) @ target.layout.T, target.plane.to2d(c + a) @ target.layout.T
    h = target.ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color, lw=lineWidth), zorder=5)
  else:
    h = target.ax.quiver(c[0], c[1], c[2], a[0], a[1], a[2], color=color, linewidth=lineWidth, arrow_length_ratio=0.25)
  if label is not None:
    h.set_label(label)
  return h


def plotInnerDirection(cS, d, faceColor="blue", arrowWidth=None, lineWidth=2.0, displayName=None, **kw: Unpack[PlotKeywords]):
  """plotInnerDirection(cS, d): the crystal direction d as an arrow inside the shape"""
  d = asVector3d(d)
  kind, target = _target(cS, kw)
  lw = lineWidth if arrowWidth is None else 40 * arrowWidth
  hs = [_arrow(kind, target, cS, v, faceColor, lw, displayName) for v in (d.flatten() if d.ndim > 0 else [d])]
  return hs[0] if len(hs) == 1 else hs


def plotSlipSystem(cS, sS, faceColor="red", faceAlpha=0.5, edgeColor="k", lineWidth=0.8, displayName=None, label=None, **kw: Unpack[PlotKeywords]):
  """plotSlipSystem(cS, sS), plot(cS, sS, label='b'): the slip plane through the centre of the
  shape with the slip direction as an arrow in it, the arrow named at its tip"""
  kind, target = _target(cS, kw)
  h = plotInnerFace(cS, sS.n, faceColor=faceColor, faceAlpha=faceAlpha, edgeColor=edgeColor, lineWidth=lineWidth, displayName=displayName,
                    parent=target.ax)
  for k in range(len(sS)):
    _arrow(kind, target, cS, sS.b[k], faceColor, 2.0)
    if label is not None and kind != "map":
      u = cS._direction(sS.b[k])
      tip = cS.V.data.reshape(-1, 3).mean(axis=0) + 1.05 * cS.exitDistance(u) * u
      target.ax.text(*tip, label, ha="center", va="center")
  return h


# ------------------------------------------------------------------ planes

def plotPlane(plane, faceColor=None, faceAlpha=0.5, edgeColor="k", lineWidth=0.8, displayName=None, **kw: Unpack[PlotKeywords]):
  """plot(plane): planes in space, each clipped by the bounds of the axes"""
  from matplotlib.colors import to_rgba
  kind, target = _target(plane, kw)
  if kind == "map":
    raise ValueError("a plane is drawn in space, not into a map")
  ax = target.ax
  lo = np.array([ax.get_xlim()[0], ax.get_ylim()[0], ax.get_zlim()[0]])
  hi = np.array([ax.get_xlim()[1], ax.get_ylim()[1], ax.get_zlim()[1]])
  if faceColor is None:
    faceColor = SHAPE_COLOR
  fc = np.asarray(faceColor, dtype=float) if not isinstance(faceColor, str) else None
  perPlane = fc is not None and fc.ndim == 2 and len(fc) == plane.size
  polys, colors = [], []
  for k in range(plane.size):
    pts = plane[k].clip(lo, hi).data.reshape(-1, 3)
    if len(pts) >= 3:
      polys.append(pts)
      colors.append(fc[k] if perPlane else to_rgba(faceColor))
  if not polys:
    return None
  colors = np.array([to_rgba(c) for c in colors])
  h = _draw(kind, target, polys, colors, edgeColor, lineWidth, faceAlpha, displayName)
  # the plane fills the box it was cut by, so the box stays
  ax.set_xlim(lo[0], hi[0])
  ax.set_ylim(lo[1], hi[1])
  ax.set_zlim(lo[2], hi[2])
  return h


# ------------------------------------------------------------------ into a map

def plotShapesOnMap(grains, cS, faceColor=None, faceAlpha=1.0, edgeColor="k", lineWidth=0.5, colored=None, colormap="viridis",
                    scale=1.0, micronbar=None, refFrame=True, displayName=None, **kw: Unpack[PlotKeywords]):
  """plot(grains, cS), plot(ebsd, cS): the crystal shape of every grain of the shape's phase
  drawn into the map, rotated by its mean orientation, scaled by the square root of its
  area, at its centroid, its front faces as polygons; on an EBSD map one shape per
  measurement at the shape's own size"""
  from .mapplot import _finish, _splitMap, _target as mapTarget
  mapKw = _splitMap(kw)
  mP = mapTarget(grains, mapKw)
  s, ids = shapesOnMap(grains, cS, scale)
  if s is None:
    return None
  fc = faceColor
  if fc is not None and not isinstance(fc, str) and np.ndim(fc) == 2 and len(fc) == grains.size and grains.size != cS.numFaces:
    fc = np.asarray(fc, dtype=float)[ids]
  colors = faceColors(s, fc, colored, colormap)
  polys, shape, face = shapeOutlines(s, mP.plane, mP.layout)
  polys = [mP.plane.from2d(p @ np.linalg.inv(mP.layout).T) for p in polys]
  h = _draw("map", mP, polys, colors[shape, face], edgeColor, lineWidth, faceAlpha, displayName)
  if colored is not None and colored is not False:
    _formLegend(mP.ax, s, colors)
  _finish(mP, grains, micronbar, refFrame, False, None)
  return h


__all__ = ["plotInnerFace", "plotInnerDirection", "plotSlipSystem", "faceColors", "plotPlane"]
