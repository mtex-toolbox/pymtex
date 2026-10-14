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
from .mapplot import _currentMap
from .mtexfigure import gcm
from .orientationkeys import ipfColorKey
from .space import shapeOutlines, shapesOnMap
from .spaceplot import _keys, _setBox, _spaceAxes

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
  axesKw = _keys(kw)
  parent = axesKw.get("parent")
  if parent is not None and hasattr(parent, "_mtexMapPlot"):
    return "map", parent._mtexMapPlot
  if parent is not None and hasattr(parent, "_mtexSpacePlot"):
    return "space", parent._mtexSpacePlot
  if parent is None:
    fig = gcm()
    mP = _currentMap() if fig is not None and fig.hold else None
    if mP is not None:
      return "map", mP
  return "space", _spaceAxes(cS.frame, **axesKw)


# the axes fitted around what is drawn, at equal scale, widened when it held something
def _fit(sP, P, held):
  P = np.asarray(P, dtype=float).reshape(-1, 3)
  lo, hi = columnMin(P), columnMax(P)
  pad = 0.05 * max(float(np.max(hi - lo)), 1e-12)
  _setBox(sP.ax, lo - pad, hi + pad, held)


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
  h = target.shapes
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
  """the colour of every face of every crystal shape, as plot(cS) paints them

    C = faceColors(cS)                                       # the colour of the mineral, or LightSkyBlue
    C = faceColors(cS, 'red')                                # one colour
    C = faceColors(cS, cS.faceArea, colormap='parula')       # a value per face through a colour map
    C = faceColors(cS, colored=True)                         # a colour per form
    C = faceColors(cS, colored=ipfHSVKey(cS.CS))             # the key's colour of every normal

  faceColor is one colour, one RGB or RGBA row per face or per shape, or a value per face,
  mapped through the colour map from its smallest to its largest value. colored overrides it.

  Input
    cS - crystalShape, one or a list of n shapes
  Output
    C - (n, number of faces, 4) array of RGBA colours
  Keywords
    faceColor - one colour, one per face, one per shape, or a value per face
    colored - True for a colour per form, or a direction or ipf key colouring the normals
    colormap - the colour map of the values per face
  See also
    crystalShape.plot, ipfHSVKey"""
  from matplotlib.colors import Normalize, to_rgba
  n, nF = len(cS), cS.numFaces
  if colored:
    if colored is True:
      forms = np.array([to_rgba(FORM_COLORS[k % len(FORM_COLORS)]) for k in range(len(cS.N))])
    else:
      key = colored.dirMap if isinstance(colored, ipfColorKey) else colored
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
  """a crystal shape drawn as the polygons of its faces, in a 3D axes or onto a held map

    plot(cS)                                                     # the shape in a 3D axes
    plot(cS, faceColor='red', faceAlpha=0.5, edgeColor='k')      # one colour, translucent
    plot(cS, faceColor=cS.faceArea)                              # a value per face through the colour map
    plot(cS, colored=True)                                       # a colour per form, with a legend
    plot(cS, colored=ipfHSVKey(cS.CS))                           # the colours of a key of the normals
    plot(x, y, cS)                                               # at positions, plot(x, y, z, cS) or plot(xy, cS)
    plot(cS, Miller(1, 1, 0, cS.CS, 'hkl'))                      # the section by a lattice plane
    plot(cS, sS)                                                 # a slip system inside the shape

  A list of shapes, ori * cS for instance, is drawn at once; with positions every shape is
  moved there first. Drawn while a map is held, the shapes are seen along the normal of the
  map at their positions in its plane, as plot(grains, cS) puts them.

  By default the faces take the colour of the mineral, or LightSkyBlue. A lattice plane after
  the shape hands over to plotInnerFace, a slip system to plotSlipSystem. The keywords of the
  axes, parent and the like, are taken as by every plot; MATLAB's legendSpacing is not
  provided.

  Input
    cS - crystalShape, one or a list
    x, y, z - the positions of the shapes, or xy an (n, 2) or (n, 3) array of them
  Output
    h - the matplotlib collection of the faces
  Keywords
    faceColor - one colour, one per face, one per shape, or a value per face mapped through
        the colour map
    faceAlpha - the opacity of the faces
    edgeColor - the colour of the edges
    lineWidth - the width of the edges
    colored - True gives every form its own colour with a legend; a direction or ipf key
        colours the faces by their normals
    colormap - the colour map of the values per face
    displayName - the label of the shape in a legend
  See also
    crystalShape, crystalShape.plotInnerFace, crystalShape.plotInnerDirection,
    crystalShape.plotSlipSystem, faceColors, CrystalShapes"""
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
  if colored:
    _formLegend(target.ax, cS, faceColors(cS, faceColor, colored, colormap))
  return h


# every face of every shape as a polygon in space, shape by shape
def _facePolygons(cS, V):
  return [V[k, f[f >= 0]] for k in range(len(cS)) for f in cS.F]


# ------------------------------------------------------------------ inside the shape

def plotInnerFace(cS, N, faceColor="blue", faceAlpha=1.0, edgeColor="k", lineWidth=0.8, displayName=None, **kw: Unpack[PlotKeywords]):
  """the section of a crystal shape by a lattice plane through its centre

    plotInnerFace(cS, N)                                                # the plane N inside the shape
    plotInnerFace(cS, N, faceColor='red', faceAlpha=0.5, displayName='(101)')  # coloured, for a legend

  The plane is drawn as the polygon where it cuts the shape, in the axes of the shape, so
  that it shows inside a translucent shape drawn before: plot(cS, faceAlpha=0.2), then
  plotInnerFace with the figure held. Several planes give several polygons.

  Input
    cS - crystalShape
    N - Miller, the lattice plane, one or several
  Output
    h - the matplotlib collection of the sections, None when the plane misses the shape
  Keywords
    faceColor - the colour of the section
    faceAlpha - its opacity
    edgeColor - the colour of its edges
    lineWidth - the width of its edges
    displayName - its label in a legend
  See also
    crystalShape.plotInnerDirection, crystalShape.plotSlipSystem, crystalShape.plot, CrystalShapes"""
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
  """a crystal direction drawn as an arrow from the centre of a crystal shape to its surface

    plotInnerDirection(cS, d)                                                     # the direction d inside the shape
    plotInnerDirection(cS, d, faceColor='blue', arrowWidth=0.05, displayName='[101]')  # coloured, for a legend

  The arrow runs from the centre of the shape along d up to where d leaves it, so that it
  shows inside a translucent shape drawn before. Several directions give several arrows.

  Input
    cS - crystalShape
    d - Miller, the crystal direction, one or several
  Output
    h - the arrow, or a list of them
  Keywords
    faceColor - the colour of the arrow
    arrowWidth - MATLAB's width of the arrow, which replaces lineWidth: 0.05 gives two points
    lineWidth - the width of the arrow in points
    displayName - its label in a legend
  See also
    crystalShape.plotInnerFace, crystalShape.plotSlipSystem, crystalShape.plot, CrystalShapes"""
  d = asVector3d(d)
  kind, target = _target(cS, kw)
  lw = lineWidth if arrowWidth is None else 40 * arrowWidth
  hs = [_arrow(kind, target, cS, v, faceColor, lw, displayName) for v in (d.flatten() if d.ndim > 0 else [d])]
  return hs[0] if len(hs) == 1 else hs


def plotSlipSystem(cS, sS, faceColor="red", faceAlpha=0.5, edgeColor="k", lineWidth=0.8, displayName=None, label=None, **kw: Unpack[PlotKeywords]):
  """a slip system drawn inside a crystal shape: the slip plane through the centre with the
  slip direction as an arrow in it

    plot(cS, sS)                                     # the slip plane and direction, red
    plot(cS, sS, faceColor='blue')                   # another colour
    plotSlipSystem(cS, sS, label='b')                # the arrow named at its tip

  The plane is the section by the slip plane, as plotInnerFace draws it, the arrow the slip
  direction from the centre to the surface; several systems are drawn together. Draw the
  shape translucent first, plot(cS, faceAlpha=0.2), and hold the figure.

  Input
    cS - crystalShape
    sS - slipSystem
  Output
    h - the matplotlib collection of the slip planes
  Keywords
    faceColor - the colour of the plane and the arrow
    faceAlpha - the opacity of the plane
    edgeColor - the colour of the edges of the plane
    lineWidth - the width of the edges of the plane
    displayName - the label of the plane in a legend
    label - the text at the tip of the arrow (not on a map)
  Example
  Draw two pyramidal slip systems in a translucent magnesium crystal. The blue plane and
  arrow show an a-type system; the red ones show a c+a-type system. The lattice parameters
  are those of Mg-Magnesium.cif in the MATLAB example, in angstroms.

    from mtex import crystalFrame, crystalShape, slipSystem, plot, hold
    from matplotlib.pyplot import show

    cs = crystalFrame('6/mmm', [3.20927, 3.20927, 5.21033], mineral='Magnesium')
    cS = crystalShape.hex(cs)
    aSlip = slipSystem.pyramidalA(cs)
    caSlip = slipSystem.pyramidal2CA(cs)

    plot(cS, faceAlpha=0.2)
    hold(True)
    plot(cS, aSlip, faceColor='blue')
    plot(cS, caSlip, faceColor='red')
    hold(False)
    show()

  Keep the first plot with hold(True) so that the slip planes and arrows are added to the
  same figure.
  See also
    crystalShape.plotInnerFace, crystalShape.plotInnerDirection, slipSystem, crystalShape.plot,
    CrystalShapes"""
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
  """planes in space drawn as polygons, each clipped by the bounds of the 3D axes

    plot(plane)                                       # into the current box, translucent
    plot(plane, faceColor='red', faceAlpha=0.3)       # coloured

  Every plane is cut by the box of the current axes and drawn as the polygon that remains;
  the box stays as it was, so draw what defines it first. A plane cannot be drawn into a map.

  Input
    plane - plane3d, one or several
  Output
    h - the matplotlib collection of the polygons, None when no plane meets the box
  Keywords
    faceColor - one colour, or one per plane
    faceAlpha - the opacity of the polygons
    edgeColor - the colour of their edges
    lineWidth - the width of their edges
    displayName - the label in a legend
  See also
    plane3d, crystalShape.plot, plot3d"""
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
  if colored:
    _formLegend(mP.ax, s, colors)
  _finish(mP, grains, micronbar, refFrame, False, None)
  return h


__all__ = ["plotInnerFace", "plotInnerDirection", "plotSlipSystem", "faceColors", "plotPlane"]
