# plot(grains3d), plot(grain3Boundary), plot(EBSD3): a scene3d held by an mtexFigure, shown in
# its axes as the picture PyVista renders offscreen, so that hold, setCamera and the colour bar
# act on it as on any MTEX figure; show3d opens it in a window, exportHTML writes it as an
# interactive page. Without PyVista the scene is drawn on matplotlib's 3D axes.
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from ..geometry._array import MtexArray
from ..geometry.session import getMTEXpref
from .keywords import refuseVariants
from .mtexfigure import _freshAxes, gcm
from .render3d import drawMatplotlib, hasPyVista, renderImage, writeHTML
from .scene3d import arrows3d, boundaryMesh, ellipsoids3d, lines3d, points3d, grainMesh, picture3d, scene3d, volumeSlices

if TYPE_CHECKING:
  from typing import Unpack

  from .keywords import PlotKeywords

def is3dMap(x):
  """whether x is a map of a volume: 3D grains, their boundary or an EBSD3 map"""
  from ..maps3d.ebsd3 import EBSD3
  from ..maps3d.grain3boundary import grain3Boundary
  from ..maps3d.grain3d import grain3d
  return isinstance(x, (grain3d, grain3Boundary, EBSD3))


def _plot3dMap(x, *args, **kw):
  """plot(x, ...) for a map of a volume: the EBSD3 map, the grain3Boundary or the 3D grains drawn
  as a scene3d"""
  from ..maps3d.ebsd3 import EBSD3
  from ..maps3d.grain3boundary import grain3Boundary
  if isinstance(x, EBSD3):
    return _plotEBSD3(x, *args, **kw)
  if isinstance(x, grain3Boundary):
    return _plotBoundary3d(x, *args, **kw)
  return _plotGrains3d(x, *args, **kw)


# the options of a 3D plot, their spelling checked
def _options(kw):
  refuseVariants(kw)
  # the scale bar and the side of a boundary plot mean nothing in 3D
  for k in ("micronbar", "upper", "lower"):
    kw.pop(k, None)
  if kw.pop("lineStyle", None) == "none":
    kw["edgeColor"] = None
  return kw


def _plotGrains3d(grains, color=None, how2plot=None, figSize=None, **kw: Unpack[PlotKeywords]):
  """the 3D grains drawn by their faces, coloured by phase, by a value or an orientation per
  grain, or in one colour; MATLAB's grain3d.plot

    plot(grains)                                            # coloured by phase
    plot(grains, grains.volume)                             # coloured by a value per grain
    plot(grains, grains.meanOrientation, lineStyle='none')  # by the orientation, no edges
    plot(grains, faceColor='r', faceAlpha=0.5)              # one translucent colour

  Only the outer faces of the set are drawn, each in the colour of the grain it bounds; a
  face between two grains of the set is inside and is drawn grey only where the faces are
  translucent. Values are mapped by the colour map, parula as MATLAB's patches, and
  orientations by an ipfColorKey of their phase.

  The options of the drawing are keywords too: faceColor, faceAlpha, edgeColor, edgeAlpha,
  lineWidth, colormap, colorRange and ipfDirection; lineStyle='none' draws no edges.
  The plot is a scene3d held by the mtexFigure, shown as the picture PyVista renders offscreen
  (matplotlib's 3D axes without PyVista), so hold, setCamera and mtexColorbar act on it as on
  any MTEX figure; show3d opens it in a window and exportHTML writes it as an interactive
  page.

  Input
    grains - grain3d
    color - a value, an orientation or an RGB colour per grain; by default the phase colours
  Output
    fig - mtexFigure
  Keywords
    how2plot - the plottingConvention of the view, by default plottingConvention.default3D
    figSize - the size of a new figure, 'small', 'normal', 'large', ..., or (width, height)
        in inches
  See also
    grain3Boundary.plot, EBSD3.plot, grain2d.plot, plotEllipsoid, Grains3DOperations"""
  kw = _options(kw)
  # a patch takes the colour map of MATLAB's axes, parula, not MTEX's default
  kw.setdefault("colormap", "parula")
  return _draw(grainMesh(grains, color, **kw), how2plot, figSize, grains.frame)


def _plotBoundary3d(gB, color=None, how2plot=None, figSize=None, **kw: Unpack[PlotKeywords]):
  """the faces of a 3D grain boundary, light blue, in one colour, or coloured by a value or a
  colour per face; MATLAB's grain3Boundary.plot

    plot(gB)                                  # light blue faces
    plot(gB, gB.area)                         # coloured by a value per face
    plot(gB, faceColor='r', edgeAlpha=0.2)    # one colour, faint edges

  Values are mapped by the colour map, parula as MATLAB's patches. The options of the
  drawing are keywords too: faceColor, faceAlpha, edgeColor, edgeAlpha, lineWidth, colormap
  and colorRange; lineStyle='none' draws no edges. The plot is a scene3d held by the mtexFigure, shown as the picture PyVista renders offscreen
  (matplotlib's 3D axes without PyVista), so hold, setCamera and mtexColorbar act on it as on
  any MTEX figure; show3d opens it in a window and exportHTML writes it as an interactive
  page.

  Input
    gB - grain3Boundary
    color - a value or an RGB colour per face
  Output
    fig - mtexFigure
  Keywords
    how2plot - the plottingConvention of the view, by default plottingConvention.default3D
    figSize - the size of a new figure, 'small', 'normal', 'large', ..., or (width, height)
        in inches
  See also
    grain3d.plot, grain3Boundary.quiver, Grains3DOperations"""
  kw = _options(kw)
  kw.setdefault("colormap", "parula")
  return _draw(boundaryMesh(gB, color, **kw), how2plot, figSize, gB.frame)


def _plotEBSD3(ebsd, color=None, how2plot=None, figSize=None, at=None, **kw: Unpack[PlotKeywords]):
  """the voxels of a volume in three orthogonal slice planes, coloured by phase, by a value, a
  colour or an orientation per voxel; MATLAB's EBSD3.plot

    plot(ebsd)                                   # coloured by phase
    plot(ebsd, ebsd.prop['ConfidenceIndex'])     # coloured by a value per voxel
    plot(ebsd['indexed'], ebsd['indexed'].orientations)  # coloured by the orientation
    plot(ebsd, at=(5, None, 10))                 # two slices at the voxels given

  The slices are normal to the three axes of the voxel lattice, through the middle of the
  volume unless at gives the voxel index of each. A voxel not indexed or without a value is
  white. Orientations are coloured by an ipfColorKey of their phase, values by the colour
  map. The options of the drawing are keywords too: faceColor, colormap, colorRange and
  ipfDirection. The plot is a scene3d held by the mtexFigure, shown as the picture PyVista renders offscreen
  (matplotlib's 3D axes without PyVista), so hold, setCamera and mtexColorbar act on it as on
  any MTEX figure; show3d opens it in a window and exportHTML writes it as an interactive
  page.

  MATLAB's options of a planar map, the region, the points and the exact unit cells, have no
  meaning here; the slices of one plane are drawn as a map by plot of ebsd.slice.

  Input
    ebsd - EBSD3
    color - a value, an RGB colour or an orientation per voxel; by default the phase colours
  Output
    fig - mtexFigure
  Keywords
    how2plot - the plottingConvention of the view, by default plottingConvention.default3D
    figSize - the size of a new figure, 'small', 'normal', 'large', ..., or (width, height)
        in inches
    at - (i, j, k), the voxel index of the slice normal to each axis of the lattice, None
        for a slice left out; by default the middle
  See also
    EBSD3.slice, grain3d.plot, EBSD.plot, EBSD3Plotting"""
  kw = _options(kw)
  for k in ("faceAlpha", "edgeColor", "edgeAlpha", "lineWidth"):
    kw.pop(k, None)
  return _draw(volumeSlices(ebsd, color, at=at, **kw), how2plot, figSize, ebsd.frame)


def _quiver3dMap(x, d, color=(0.0, 0.0, 1.0), autoScaleFactor=1.0, scaling=True, lineWidth=1.0, antipodal=None, how2plot=None, figSize=None, **kw: Unpack[PlotKeywords]):
  """arrows at the centroids of 3D grains or of their boundary faces; MATLAB's
  grain3Boundary.quiver and grain3d.quiver

    quiver(gB, gB.N, color='r')                                # the face normals in red
    quiver(gB, gB.N, antipodal=True, autoScaleFactor=2)        # axes, twice as long
    quiver(grains, grains.meanOrientation * Miller(0, 0, 1, cs))  # a crystal direction per grain

  A grain's arrow is as long as its direction, a face's a fifth of the face's diameter,
  both times autoScaleFactor. Antipodal directions, axes, are drawn as lines through the
  centroid without heads. The plot is a scene3d held by the mtexFigure, shown as the picture PyVista renders offscreen
  (matplotlib's 3D axes without PyVista), so hold, setCamera and mtexColorbar act on it as on
  any MTEX figure; show3d opens it in a window and exportHTML writes it as an interactive
  page.

  Input
    x - grain3d or grain3Boundary
    d - vector3d, one direction per grain or face
  Output
    fig - mtexFigure
  Keywords
    color - the RGB colour of the arrows, one for all or one per arrow
    autoScaleFactor - the factor of the arrow lengths
    scaling - scale the arrows as above; False gives every arrow a quarter of its
        direction's length
    lineWidth - the width of the shafts
    antipodal - draw axes, lines without heads; by default d.antipodal
    how2plot - the plottingConvention of the view, by default plottingConvention.default3D
    figSize - the size of a new figure, 'small', 'normal', 'large', ..., or (width, height)
        in inches
  See also
    grain3Boundary.plot, grain3d.plot, quiver3d"""
  from ..maps3d.grain3boundary import grain3Boundary
  kw = _options(kw)
  P = np.asarray(x.centroid.data).reshape(-1, 3)
  D = np.asarray(d.data if isinstance(d, MtexArray) else d, dtype=float).reshape(-1, 3)
  if isinstance(x, grain3Boundary) and scaling:
    D = 0.2 * np.asarray(x.diameter).reshape(-1, 1) * D / np.linalg.norm(D, axis=1, keepdims=True)
  D = D * (autoScaleFactor if scaling else 0.25)
  anti = bool(getattr(d, "antipodal", False)) if antipodal is None else antipodal
  return _draw(arrows3d(P, D, color, lineWidth, antipodal=anti), how2plot, figSize, x.frame)


def _quiverVectors(v, d, arrowSize=0.05, color=(0.0, 0.0, 1.0), lineWidth=2.0, how2plot=None, figSize=None, headSize=0.25, **kw):
  """quiver3d(v, d): arrows at the positions v along d times arrowSize in a 3D scene, as
  MATLAB's vector3d/quiver3; an antipodal d points both ways without heads"""
  kw = _options(kw)
  P = np.asarray(v.data).reshape(-1, 3)
  D = arrowSize * np.asarray(d.data, dtype=float).reshape(-1, 3)
  return _draw(arrows3d(P, D, color, lineWidth, antipodal=bool(getattr(d, "antipodal", False)), headSize=headSize), how2plot, figSize, v.frame)


def plotEllipsoid(center, a, b, c, faceColor="lightblue", edgeAlpha=0.25, how2plot=None, figSize=None, **kw: Unpack[PlotKeywords]):
  """ellipsoids with the half axes a, b, c about their centres, in a 3D scene; MATLAB's
  plotEllipsoid

    a, b, c = grains.principalComponents()
    plotEllipsoid(grains.centroid, a, b, c, faceColor='r')     # the ellipsoids of the grains
    plotEllipsoid(grains.centroid, a, b, c, faceColor=rgb, edgeAlpha=0)  # a colour each, no edges

  Every ellipsoid is a surface on a grid of 24 longitudes and 12 latitudes. With hold on it
  joins the grains of the scene drawn before. The plot is a scene3d held by the mtexFigure, shown as the picture PyVista renders offscreen
  (matplotlib's 3D axes without PyVista), so hold, setCamera and mtexColorbar act on it as on
  any MTEX figure; show3d opens it in a window and exportHTML writes it as an interactive
  page.

  Input
    center - vector3d, the centres
    a - vector3d, the first half axis of every ellipsoid
    b - vector3d, the second half axis
    c - vector3d, the third half axis
  Output
    fig - mtexFigure
  Keywords
    faceColor - the colour of the ellipsoids, one for all or an (n, 3) array of RGB values
        one per ellipsoid
    edgeAlpha - the opacity of the grid lines on the surfaces
    how2plot - the plottingConvention of the view, by default plottingConvention.default3D
    figSize - the size of a new figure, 'small', 'normal', 'large', ..., or (width, height)
        in inches
  See also
    grain3d.principalComponents, grain3d.centroid, grain3d.plot"""
  kw = _options(kw)
  faceColor = kw.pop("faceColor", faceColor)
  def data(v):
    return np.asarray(v.data if isinstance(v, MtexArray) else v, dtype=float)
  fr = getattr(center, "frame", None)
  return _draw(ellipsoids3d(data(center), data(a), data(b), data(c), faceColor, kw.pop("edgeAlpha", edgeAlpha)), how2plot, figSize, fr)


# MATLAB's colour order, the colour of the n-th set of markers in one scene
COLOR_ORDER = [(0.0, 0.447, 0.741), (0.85, 0.325, 0.098), (0.929, 0.694, 0.125), (0.494, 0.184, 0.556), (0.466, 0.674, 0.188),
               (0.301, 0.745, 0.933), (0.635, 0.078, 0.184)]


def _plotPoints3d(v, color=None, markerSize=6.0, **kw):
  """plot(v) into a held 3D scene: the vectors as markers at their positions, each set in the
  next colour of MATLAB's colour order"""
  kw = _options(kw)
  s = gcs()
  if color is None:
    color = kw.pop("markerFaceColor", None)
  if color is None:
    k = sum(isinstance(it, points3d) for it in s.items) if s is not None else 0
    color = COLOR_ORDER[k % len(COLOR_ORDER)]
  return _draw(points3d(np.asarray(v.data).reshape(-1, 3), color, markerSize), None, None, v.frame)


def line3(x, y, z, color="k", lineWidth=0.5, **kw: Unpack[PlotKeywords]):
  """line(x, y, z): a polyline in a 3D scene, NaN separating its pieces, as MATLAB's line"""
  refuseVariants(kw)
  P = np.column_stack([np.asarray(c, dtype=float).reshape(-1) for c in (x, y, z)])
  ok = np.all(np.isfinite(P), axis=1)
  seg = ok[:-1] & ok[1:]
  segments = np.stack([P[:-1][seg], P[1:][seg]], axis=1)
  return _draw(lines3d(segments, color, lineWidth), None, None, None)


def _scatter3(x, y, z, s=36, c=None, *args, **kw):
  """_scatter3(x, y, z, s, c): markers of area s in a 3D scene, in one colour or one each, as
  MATLAB's scatter3; 'filled' is accepted and markers are always filled"""
  c = refuseVariants(kw).pop("color", c)
  P = np.column_stack([np.asarray(v, dtype=float).reshape(-1) for v in (x, y, z)])
  size = float(np.sqrt(np.mean(s)))
  if c is None:
    c = COLOR_ORDER[0]
  elif isinstance(c, str) and len(c) == 1:
    c = {"b": "blue", "r": "red", "g": "green", "k": "black", "w": "white", "c": "cyan", "m": "magenta", "y": "yellow"}[c]
  return _draw(points3d(P, c, size), None, None, None)


# ------------------------------------------------------------------ the figure

# the axes a figure draws into now: the one drawn into last
def _currentAxes(fig):
  return None if fig is None else fig._axesDrawnInto()


def gcs(fig=None):
  """the scene of the current axes of a figure, the 3D scene a plot of a volume, of 3D grains
  or of maps in several planes draws into; None when the axes hold none

    plot(grains3); s = gcs()                          # the scene of the current figure
    s = gcs(fig)                                      # of another figure

  Output
    s - scene3d, its items the meshes, lines and pictures drawn
  Keywords
    fig - the mtexFigure, by default the current one
  See also
    show3d, exportHTML, setCamera"""
  return getattr(_currentAxes(gcm() if fig is None else fig), "_mtexScene", None)


def _draw(item, how2plot, figSize, frame):
  fig = gcm()
  s = gcs(fig) if fig is not None and fig.hold else None
  # a held map turns into a scene the 3D item joins
  if s is None and fig is not None and fig.hold and getattr(_currentAxes(fig), "_mtexMapSources", None):
    s = toScene(fig, how2plot)
  if s is None:
    fig, ax = _freshAxes(figSize)
    s = scene3d(how2plot)
    ax._mtexScene = s
    fig.current = ax
  elif how2plot is not None:
    s.how2plot = how2plot
  for it in item if isinstance(item, list) else [item]:
    s.add(it)
  refresh(fig)
  return fig


def _backend():
  b = getMTEXpref("plot3dBackend", None)
  if b is None:
    return "pyvista" if hasPyVista() else "matplotlib"
  return b


# the part of a picture that is not white, with a margin of 3 %
def _crop(img):
  ink = np.any(img[..., :3] < 250, axis=-1)
  if not ink.any():
    return img
  rows, cols = np.flatnonzero(ink.any(axis=1)), np.flatnonzero(ink.any(axis=0))
  pad = int(0.03 * max(img.shape[:2]))
  r0, r1 = max(rows[0] - pad, 0), min(rows[-1] + pad + 1, img.shape[0])
  c0, c1 = max(cols[0] - pad, 0), min(cols[-1] + pad + 1, img.shape[1])
  return img[r0:r1, c0:c1]


class _scenePlot:
  def __init__(self, mappable):
    self.mappable = mappable


# the axes in the place of another, flat or 3D, the scene carried over
def _replaceAxes(fig, ax, projection):
  spec = ax.get_subplotspec()
  s = ax._mtexScene
  k = fig.children.index(ax)
  ax.remove()
  new = fig.fig.add_subplot(spec, projection=projection, computed_zorder=False) if projection else fig.fig.add_subplot(spec)
  new._mtexScene = s
  fig.children[k] = new
  return new


def refresh(fig=None, ax=None):
  """draw the scene of an axes again, after a change of its items or its camera"""
  fig = gcm() if fig is None else fig
  ax = _currentAxes(fig) if ax is None else ax
  s = ax._mtexScene
  if _backend() == "pyvista":
    if getattr(ax, "name", "") == "3d":
      ax = _replaceAxes(fig, ax, None)
    ax.cla()
    # a square picture cropped to what it shows; a figure of one panel takes its shape
    single = len(fig.children) == 1
    if single and not hasattr(fig, "sceneWidth"):
      fig.sceneWidth = float(fig.fig.get_size_inches()[0])
    w, h = fig.fig.get_size_inches() / np.array([fig.ncols, fig.nrows])
    n = int((fig.sceneWidth if single else max(w, h)) * fig.fig.dpi * 1.5)
    img = _crop(renderImage(s, (n, n)))
    if single and not fig.fixedSize:
      fig.fig.set_size_inches(fig.sceneWidth, fig.sceneWidth * img.shape[0] / img.shape[1])
    ax.imshow(img, interpolation="antialiased")
    ax.set_axis_off()
  else:
    if getattr(ax, "name", "") != "3d":
      ax = _replaceAxes(fig, ax, "3d")
    ax.cla()
    drawMatplotlib(s, ax)
  fig.current = ax
  scenes = [c._mtexScene for c in fig.children if getattr(c, "_mtexScene", None) is not None]
  fig.fig._mtexScene = scenes[0] if scenes else None
  fig.fig._mtexScenes = scenes
  fig.fig._mtexSceneVersion = getattr(fig.fig, "_mtexSceneVersion", 0) + 1
  # the values of the last item coloured by them are what a colour bar shows
  ax._mtexScenePlot = None
  for it in reversed(s.items):
    if getattr(it, "values", None) is not None and it.colorRange is not None:
      import matplotlib.cm as cm
      from matplotlib.colors import Normalize

      from .colormaps import mtexColormap
      sm = cm.ScalarMappable(Normalize(*it.colorRange), mtexColormap(it.colormap))
      sm.set_array(np.asarray(it.values)[np.isfinite(it.values)])
      ax._mtexScenePlot = _scenePlot(sm)
      break
  return fig


# ------------------------------------------------------------------ maps in space

def mapPicture(sources, pixels=1600):
  """the picture of two dimensional map plots on the rectangle of the plane they lie in, to be
  shown in a 3D scene

    pic = mapPicture([(plotEBSD, ebsdSlice, (), {})])  # the phase map of a slice in space
    pic = mapPicture(sources, pixels=400)             # a coarser picture

  The plots are drawn by their own commands, one over the other, into an axes filling a
  figure pyplot does not know, so that every option of the map plots holds in space too;
  white, where nothing was measured, is transparent. The picture keeps its sources, so that a
  map in the same plane is drawn into it rather than beside it. A held figure does this
  itself when a map of another plane is plotted into it.

  Input
    sources - a list of (plotFun, map, args, kw), the plot commands of maps in one plane
  Output
    pic - picture3d, the image with the corners it spans in space
  Keywords
    pixels - the length of the picture's longer side in pixels
  See also
    gcs, setCamera, EBSD3.slice"""
  from matplotlib.backends.backend_agg import FigureCanvasAgg
  from matplotlib.figure import Figure

  from .maps import mapExtent, mapLayout, planeOf
  plotFun, x, args, kw = sources[0]
  xlo, xhi, ylo, yhi = mapExtent(x, mapLayout(kw.get("how2plot"), planeOf(x)))
  w, h = xhi - xlo, yhi - ylo
  scale = pixels / max(w, h)
  fig = Figure(figsize=(max(w * scale, 2) / 100, max(h * scale, 2) / 100), dpi=100)
  fig.patch.set_alpha(0)
  ax = fig.add_axes([0, 0, 1, 1])
  ax.patch.set_alpha(0)
  for plotFun, x, args, kw in sources:
    kw = dict(kw)
    for k in ("parent", "figSize", "colorbar"):
      kw.pop(k, None)
    kw["micronbar"] = False
    if plotFun.__name__ != "plotBoundary":
      kw.setdefault("legend", False)
    plotFun(x, *args, parent=ax, **kw)
  ax.set_axis_off()
  canvas = FigureCanvasAgg(fig)
  canvas.draw()
  rgba = np.asarray(canvas.buffer_rgba(), dtype=float)[::-1] / 255
  # white is what a map paints where nothing was measured, and in space it hides what lies behind
  rgba[np.all(rgba[..., :3] == 1, axis=-1), 3] = 0
  mP = ax._mtexMapPlot
  xlo, xhi = ax.get_xlim()
  ylo, yhi = ax.get_ylim()
  screen = np.array([[xlo, ylo], [xhi, ylo], [xlo, yhi], [xhi, yhi]])
  # the screen is the plane's coordinates under the layout
  plane2d = np.linalg.solve(np.asarray(mP.layout, dtype=float), screen.T).T
  pic = picture3d(rgba, mP.plane.from2d(plane2d))
  pic.sources, pic.plane = list(sources), mP.plane
  return pic


# a map into a scene: into the picture of its plane when the scene has one, else a new picture
def _addMap(s, source):
  from .maps import planeOf
  plane = planeOf(source[1])
  for k, it in enumerate(s.items):
    if isinstance(it, picture3d) and getattr(it, "plane", None) is not None and _samePlane(plane, it.plane):
      s.items[k] = mapPicture(it.sources + [source])
      return
  s.add(mapPicture([source]))


def mapInto3d(plotFun, x, args, kw):
  """a plot of a map into a held figure: into its scene, or, when the map lies in another plane
  than the maps the figure shows, the figure turned into a scene of all of them; None when the
  map is drawn as a map"""
  fig = gcm()
  if fig is None or not fig.hold:
    return None
  s = gcs(fig)
  source = (plotFun, x, tuple(args), dict(kw))
  if s is not None:
    _addMap(s, source)
    return refresh(fig)
  ax = _currentAxes(fig)
  mP = getattr(ax, "_mtexMapPlot", None)
  sources = getattr(ax, "_mtexMapSources", None)
  if mP is None or not sources:
    return None
  from .maps import planeOf
  if _samePlane(planeOf(x), mP.plane):
    return None
  _addMap(toScene(fig), source)
  return refresh(fig)


def _samePlane(a, b):
  n1, n2 = a.normal, b.normal
  parallel = abs(abs(float(n1 @ n2)) - 1) < 1e-9
  # the offsets of the two planes along the normal agree, at the scale of their origins
  offset = abs(float(a.o @ n1) - float(b.o @ n1) * np.sign(n1 @ n2))
  return parallel and offset < 1e-9 * max(1, float(np.abs(a.o).max()))


def toScene(fig=None, how2plot=None):
  """the map plot of a figure's current axes as a scene of the pictures of every map drawn
  into it, held by that axes, cleared of the map"""
  fig = gcm() if fig is None else fig
  ax = _currentAxes(fig)
  s = scene3d(how2plot)
  for source in ax._mtexMapSources:
    _addMap(s, source)
  ax.cla()
  ax._mtexMapPlot = ax._mtexMapSources = None
  ax._mtexScene = s
  return s


def recordMap(ax, plotFun, x, args, kw):
  """record the maps drawn into an axes, so that the axes can be drawn again as a scene"""
  if ax is None or getattr(ax, "_mtexMapPlot", None) is None:
    return
  if getattr(ax, "_mtexMapSources", None) is None:
    ax._mtexMapSources = []
  ax._mtexMapSources.append((plotFun, x, tuple(args), dict(kw)))


def show3d(fig=None, window_size=(900, 900)):
  """the scene of the current figure in an interactive PyVista window, rotated and zoomed with
  the mouse

    show3d()                                          # the scene of the current figure
    show3d(fig, window_size=(600, 400))               # of another figure, in a smaller window

  Output
    p - the pyvista Plotter of the window
  Keywords
    fig - the mtexFigure holding the scene, by default the current one
    window_size - the size of the window in pixels
  See also
    exportHTML, gcs, setCamera"""
  from .render3d import pyvistaPlotter
  s = gcs(fig)
  if s is None:
    raise ValueError("the current figure holds no 3D scene")
  p = pyvistaPlotter(s, window_size=window_size)
  p.show()
  return p


def exportHTML(fname, fig=None, title="MTEX 3D plot"):
  """the scene of the current figure written as an interactive three.js page, which a browser
  rotates and zooms

    exportHTML('grains.html')                         # the scene of the current figure
    exportHTML('grains.html', fig=fig, title='Iron grains')  # of another figure, with a title

  Input
    fname - the name of the HTML file
  Output
    fname - the file written
  Keywords
    fig - the mtexFigure holding the scene, by default the current one
    title - the title of the page
  See also
    show3d, gcs, saveFigure"""
  s = gcs(fig)
  if s is None:
    raise ValueError("the current figure holds no 3D scene")
  return writeHTML(s, fname, title)


__all__ = ["show3d", "exportHTML", "is3dMap", "gcs", "mapPicture", "plotEllipsoid"]
