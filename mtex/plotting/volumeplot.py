# plot(grains3d), plot(grain3Boundary), plot(EBSD3): a scene3d held by an mtexFigure, shown in
# its axes as the picture PyVista renders offscreen, so that hold, setCamera and the colour bar
# act on it as on any MTEX figure; show3d opens it in a window, exportHTML writes it as an
# interactive page. Without PyVista the scene is drawn on matplotlib's 3D axes.
import numpy as np

from ..geometry.session import getMTEXpref
from .mtexfigure import gcm, mtexFigure
from .render3d import drawMatplotlib, hasPyVista, renderImage, writeHTML
from .scene3d import arrows3d, boundaryMesh, grainMesh, picture3d, scene3d, volumeSlices

__all__ = ["plotGrains3d", "plotBoundary3d", "plotEBSD3", "show3d", "exportHTML", "is3dMap", "plot3dMap", "gcs", "mapPicture", "quiver3d"]


def is3dMap(x):
  from ..maps3d.ebsd3 import EBSD3
  from ..maps3d.grain3boundary import grain3Boundary
  from ..maps3d.grain3d import grain3d
  return isinstance(x, (grain3d, grain3Boundary, EBSD3))


def plot3dMap(x, *args, **kw):
  from ..maps3d.ebsd3 import EBSD3
  from ..maps3d.grain3boundary import grain3Boundary
  if isinstance(x, EBSD3):
    return plotEBSD3(x, *args, **kw)
  if isinstance(x, grain3Boundary):
    return plotBoundary3d(x, *args, **kw)
  return plotGrains3d(x, *args, **kw)


# MATLAB's spellings of the options
def _options(kw):
  for old, new in (("FaceColor", "faceColor"), ("facecolor", "faceColor"), ("FaceAlpha", "faceAlpha"), ("facealpha", "faceAlpha"),
                   ("EdgeAlpha", "edgeAlpha"), ("edgealpha", "edgeAlpha"), ("EdgeColor", "edgeColor"), ("edgecolor", "edgeColor"),
                   ("LineWidth", "lineWidth"), ("linewidth", "lineWidth"), ("LineStyle", "lineStyle"), ("linestyle", "lineStyle")):
    if old in kw:
      kw[new] = kw.pop(old)
  # the scale bar and the side of a boundary plot mean nothing in 3D
  for k in ("micronbar", "micronBar", "MicronBar", "upper", "lower"):
    kw.pop(k, None)
  if kw.pop("lineStyle", None) == "none":
    kw["edgeColor"] = None
  return kw


def plotGrains3d(grains, color=None, how2plot=None, figSize=None, **kw):
  """plot(grains), plot(grains, values), plot(grains, ori), plot(grains, faceColor=...)

  The outer faces of 3D grains coloured by phase, by a value or an orientation per grain, or
  in one colour, as MATLAB's grain3d/plot. Options faceAlpha, edgeColor, edgeAlpha,
  lineWidth, colormap, colorRange, ipfDirection; lineStyle='none' draws no edges.
  """
  kw = _options(kw)
  return _draw(grainMesh(grains, color, **kw), how2plot, figSize, grains.frame)


def plotBoundary3d(gB, color=None, how2plot=None, figSize=None, **kw):
  """plot(gB), plot(gB, values), plot(gB, faceColor=...): the faces of a 3D grain boundary"""
  kw = _options(kw)
  return _draw(boundaryMesh(gB, color, **kw), how2plot, figSize, gB.frame)


def plotEBSD3(ebsd, color=None, how2plot=None, figSize=None, at=None, **kw):
  """plot(ebsd), plot(ebsd, values), plot(ebsd, ori): three orthogonal slices through the
  voxels of a volume, as MATLAB's EBSD3/plot shows it in slice planes; at=(i, j, k) the voxel
  index of the three slices, None for a slice left out, the middle by default"""
  kw = _options(kw)
  for k in ("faceAlpha", "edgeColor", "edgeAlpha", "lineWidth"):
    kw.pop(k, None)
  return _draw(volumeSlices(ebsd, color, at=at, **kw), how2plot, figSize, ebsd.frame)


def quiver3d(x, d, color=(0.0, 0.0, 1.0), autoScaleFactor=1.0, noScaling=False, lineWidth=1.0, antipodal=None, how2plot=None, figSize=None, **kw):
  """quiver(grains, dir), quiver(gB, dir): arrows at the centroids of 3D grains or of their
  boundary faces, as MATLAB's; a grain's arrow as long as the direction, a face's a fifth of
  the face's diameter, both times autoScaleFactor"""
  from ..maps3d.grain3boundary import grain3Boundary
  kw = _options(kw)
  color = kw.pop("Color", color)
  P = np.asarray(x.centroid.data).reshape(-1, 3)
  D = np.asarray(d.data if hasattr(d, "data") else d, dtype=float).reshape(-1, 3)
  if isinstance(x, grain3Boundary) and not noScaling:
    D = 0.2 * np.asarray(x.diameter).reshape(-1, 1) * D / np.linalg.norm(D, axis=1, keepdims=True)
  D = D * (autoScaleFactor if not noScaling else 0.25)
  anti = bool(getattr(d, "antipodal", False)) if antipodal is None else antipodal
  return _draw(arrows3d(P, D, color, lineWidth, antipodal=anti), how2plot, figSize, x.frame)


# ------------------------------------------------------------------ the figure

# the scene of the current figure, None when it holds none
def gcs(fig=None):
  fig = gcm() if fig is None else fig
  return getattr(fig, "scene3d", None)


def _draw(item, how2plot, figSize, frame):
  fig = gcm()
  if fig is None or not fig.hold or gcs(fig) is None:
    from .mtexfigure import figureSize
    fig = mtexFigure(1, figSize=figureSize(figSize, (5.5, 5.5)))
    fig.scene3d = scene3d(how2plot)
  elif how2plot is not None:
    fig.scene3d.how2plot = how2plot
  for it in item if isinstance(item, list) else [item]:
    fig.scene3d.add(it)
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


def refresh(fig=None):
  """draw the scene of a figure again, after a change of its items or its camera"""
  fig = gcm() if fig is None else fig
  s = gcs(fig)
  ax = fig.children[0] if fig.children else None
  if _backend() == "pyvista":
    if ax is None or getattr(ax, "name", "") == "3d":
      if ax is not None:
        ax.remove()
        fig.children.clear()
      ax = fig.nextAxis()
    ax.cla()
    # a square picture, cropped to what it shows, and the figure of its width in that shape
    if not hasattr(fig, "sceneWidth"):
      fig.sceneWidth = float(fig.fig.get_size_inches()[0])
    n = int(fig.sceneWidth * fig.fig.dpi * 1.5)
    img = _crop(renderImage(s, (n, n)))
    fig.fig.set_size_inches(fig.sceneWidth, fig.sceneWidth * img.shape[0] / img.shape[1])
    ax.imshow(img, interpolation="antialiased")
    ax.set_axis_off()
  else:
    if ax is None or getattr(ax, "name", "") != "3d":
      if ax is not None:
        ax.remove()
        fig.children.clear()
      ax = fig.fig.add_subplot(1, 1, 1, projection="3d", computed_zorder=False)
      fig.children.append(ax)
    ax.cla()
    drawMatplotlib(s, ax)
  fig.current = ax
  fig.fig._mtexScene = s
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

# the picture of two dimensional map plots on the rectangle of the plane they lie in: the
# plots drawn by their own commands, one over the other, into an axes filling a figure pyplot
# does not know, so that every option of the map plots holds in space too. The picture keeps
# its sources, so that a map in the same plane is drawn into it rather than beside it.
def mapPicture(sources, pixels=1600):
  from matplotlib.backends.backend_agg import FigureCanvasAgg
  from matplotlib.figure import Figure

  from .maps import mapExtent, mapLayout
  plotFun, x, args, kw = sources[0]
  xlo, xhi, ylo, yhi = mapExtent(x, mapLayout(kw.get("how2plot")))
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


# a plot of a map into a held figure: into its scene, or, when the map lies in another plane
# than the maps the figure shows, the figure turned into a scene of all of them; None when the
# map is drawn as a map
def mapInto3d(plotFun, x, args, kw):
  fig = gcm()
  if fig is None or not fig.hold:
    return None
  if gcs(fig) is not None:
    _addMap(fig.scene3d, (plotFun, x, tuple(args), dict(kw)))
    return refresh(fig)
  ax = fig.children[-1] if fig.children else None
  mP = getattr(ax, "_mtexMapPlot", None)
  sources = getattr(ax, "_mtexMapSources", None)
  if mP is None or not sources:
    return None
  from .maps import planeOf
  if _samePlane(planeOf(x), mP.plane):
    return None
  toScene(fig)
  _addMap(fig.scene3d, (plotFun, x, tuple(args), dict(kw)))
  return refresh(fig)


def _samePlane(a, b):
  n1, n2 = a.normal, b.normal
  return abs(abs(float(n1 @ n2)) - 1) < 1e-9 and abs(float(a.o @ n1) - float(b.o @ n1) * np.sign(n1 @ n2)) < 1e-9 * max(1, float(np.abs(a.o).max()))


# the map plot of a figure's axes as a scene of the pictures of every map drawn into it
def toScene(fig=None, how2plot=None):
  fig = gcm() if fig is None else fig
  ax = fig.children[-1]
  sources = ax._mtexMapSources
  s = scene3d(how2plot)
  for source in sources:
    _addMap(s, source)
  ax.remove()
  fig.children.remove(ax)
  fig.scene3d = s
  return s


# record the maps drawn into an axes, so that the axes can be drawn again as a scene
def recordMap(ax, plotFun, x, args, kw):
  if ax is None or getattr(ax, "_mtexMapPlot", None) is None:
    return
  if getattr(ax, "_mtexMapSources", None) is None:
    ax._mtexMapSources = []
  ax._mtexMapSources.append((plotFun, x, tuple(args), dict(kw)))


def show3d(fig=None, window_size=(900, 900)):
  """show3d(): the scene of the current figure in an interactive PyVista window"""
  from .render3d import pyvistaPlotter
  s = gcs(fig)
  if s is None:
    raise ValueError("the current figure holds no 3D scene")
  p = pyvistaPlotter(s, window_size=window_size)
  p.show()
  return p


def exportHTML(fname, fig=None, title="MTEX 3D plot"):
  """exportHTML('grains.html'): the scene of the current figure as an interactive page"""
  s = gcs(fig)
  if s is None:
    raise ValueError("the current figure holds no 3D scene")
  return writeHTML(s, fname, title)
