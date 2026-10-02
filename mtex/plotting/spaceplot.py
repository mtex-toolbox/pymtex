# the matplotlib drawing of three dimensional plots (MATLAB @vector3d/plot3d, scatter3d,
# @S2Fun/plot3d, @orientationRegion/plot, @orientation/scatter; docs/plans/plotting.md S5)
#
# space.py says what to draw: the sphere as a mesh with its grid and the axes of the frame,
# the surface of an orientation region in axis angle space, the points orientations are.
# matplotlib's 3D axes is the camera; it is imported inside the functions.
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from ..geometry.orientation import orientation
from ..geometry.orientationregion import eulerRegion, fundamentalRegion, orientationRegion
from ..geometry.quaternion import quaternion
from ..geometry.tangent import SO3TangentVector
from ..geometry.representations import quat2euler, quatAngle, quatAxis
from ..geometry.session import getMTEXpref
from ..geometry.units import degree
from ..geometry.vector3d import vector3d
from .colormaps import mtexColormap
from .keywords import patchKeywords, refuseVariants
from .mtexfigure import gcm, mtexFigure, nextAxis
from .space import axisAnglePoints, frameAxes, regionLines, regionSurface, regionWalls, sphereGridLines, sphereMesh
from .sphericalplot import MARKER_COLOR, _isField, _isFunction, _isAxisField, _rangeOf
from ..geometry.display import frameLabel, header

if TYPE_CHECKING:
  from typing import Unpack

  from .keywords import PlotKeywords

SPHERE_COLOR = (0.93, 0.93, 0.93)
GRID3_COLOR = (0.75, 0.75, 0.75)


class spacePlot:
  """a plot in space: the 3D axes, the frame it is drawn in, the mappable of its colours"""

  def __init__(self, ax, frame=None):
    self.ax, self.frame, self.mappable = ax, frame, None
    ax._mtexSpacePlot = self

  def __repr__(self):
    return header("spacePlot", frameLabel(self.frame))


# the camera of MATLAB's two dimensional view: looking down the out of screen axis of the
# plotting convention of the frame, east to the right
def _conventionView(ax, frame):
  from .spherical import _conventionOf
  _camera(ax, _conventionOf(frame))


def _camera(ax, pC):
  o, e = pC.outOfScreen.normalize().data.reshape(3), pC.east.normalize().data.reshape(3)
  elev, azim = np.arcsin(np.clip(o[2], -1, 1)), np.arctan2(o[1], o[0])
  # screen right before the roll, then the roll that turns east onto it
  r0 = np.array([-np.sin(azim), np.cos(azim), 0.0])
  roll = -np.arctan2(np.dot(np.cross(r0, e), o), np.dot(r0, e))
  ax.view_init(elev=elev / degree, azim=azim / degree, roll=roll / degree)
  ax.set_proj_type("ortho")
  # an axis arrow whose tip the sphere hides from this camera is not drawn
  for d, length, artists in getattr(ax, "_mtexAxisArrows", ()):
    visible = np.dot(d, o) > -np.sqrt(1 - 1 / (1 + length) ** 2)
    for a in artists:
      a.set_visible(bool(visible))


def setCamera(how2plot=None, ax=None):
  """setCamera(how2plot): the 3D view of the current axes looking down how2plot.outOfScreen,
  with how2plot.north up; without a convention the default one"""
  import matplotlib.pyplot as plt

  from ..geometry.plottingconvention import plottingConvention
  pC = plottingConvention.default if how2plot is None else how2plot
  # a figure holding a 3D scene is drawn again from the new camera, and a map seen from off
  # its normal becomes a scene of the maps drawn into it
  from .volumeplot import gcs, refresh, toScene
  fig = gcm()
  cur = None if fig is None or not fig.children else (fig.current if fig.current in fig.children else fig.children[-1])
  if ax is None and gcs(fig) is None and getattr(cur, "_mtexMapSources", None):
    n = cur._mtexMapPlot.plane.normal
    if abs(abs(float(pC.outOfScreen.normalize().data.reshape(3) @ n)) - 1) > 1e-9:
      toScene(fig)
  if ax is None and gcs(fig) is not None:
    gcs(fig).how2plot = pC
    return refresh(fig)
  if ax is None:
    fig = gcm()
    ax = fig.children[-1] if fig is not None and fig.children else plt.gca()
  _camera(ax, pC)


# a 3D axes in a new figure, or the current one when it is held, looking at the unit sphere
# or at a shape of the given extent
def _spaceAxes(frame, limits=1.3, decorated=False, figSize=None, parent=None, azimuth=None, elevation=None, title=None):
  if parent is not None:
    ax = parent
  else:
    fig = gcm()
    # a held plot draws into the axes drawn into last, which nextAxis need not have made last
    if fig is not None and fig.hold and fig.children:
      cur = getattr(fig, "current", None)
      cur = cur if cur in fig.children else fig.children[-1]
      if hasattr(cur, "_mtexSpacePlot"):
        return cur._mtexSpacePlot
    # the axes nextAxis prepared, or the first of a fresh newMtexFigure, becomes a 3D axes
    if fig is not None and (fig.pending is not None or (not fig.children and not fig.hold)):
      flat = fig.takeAxis()
      k = fig.children.index(flat)
      spec = flat.get_subplotspec()
      flat.remove()
      ax = fig.fig.add_subplot(spec, projection="3d", computed_zorder=False)
      fig.children[k] = ax
      fig.current = ax
    else:
      from .mtexfigure import figureSize
      fig = mtexFigure(1, figSize=figureSize(figSize, (5.5, 5.5)))
      # the markers are drawn after the region they lie in, whatever matplotlib makes of
      # the depth of one big translucent surface
      ax = fig.fig.add_subplot(1, 1, 1, projection="3d", computed_zorder=False)
      fig.children.append(ax)
  sP = spacePlot(ax, frame)
  ax.set_box_aspect((1, 1, 1))
  lim = (-limits, limits) if np.isscalar(limits) else limits
  ax.set_xlim(*lim)
  ax.set_ylim(*lim)
  ax.set_zlim(*lim)
  if not decorated:
    ax.set_axis_off()
  ax.view_init(elev=22.5 if elevation is None else elevation / degree, azim=-130 if azimuth is None else azimuth / degree)
  if title is not None:
    ax.set_title(title, fontsize=getMTEXpref("FontSize", 13))
  return sP


# the sphere with its grid and the axes of the frame, the ground every plot in space stands
# on; with values the sphere is coloured at its faces, with radial pushed out by them
def _sphere(sP, resolution=5 * degree, values=None, colormap="WhiteJet", colorRange=None, radial=False, grid=True, axisLabels=True,
            grid_res=30 * degree, alpha=1.0):
  from mpl_toolkits.mplot3d.art3d import Poly3DCollection
  V, F = sphereMesh(resolution)
  R = np.abs(values)[:, None] * V if radial else V
  poly = Poly3DCollection(R[F], linewidths=0.2, alpha=alpha)
  if values is None:
    poly.set_facecolor(SPHERE_COLOR)
  else:
    lo, hi = _rangeOf([values]) if colorRange is None else colorRange
    from matplotlib.colors import Normalize
    norm = Normalize(lo, hi)
    cmap = mtexColormap(colormap)
    faceValues = np.mean(values[F], axis=1)
    poly.set_facecolor(cmap(norm(faceValues)))
    poly.set_cmap(cmap)
    poly.set_norm(norm)
    poly.set_array(faceValues)
    sP.mappable = poly
  sP.ax.add_collection3d(poly)
  # the faces take their own colour as their edge, so the seams between them close
  poly.set_edgecolor("face")
  if grid:
    for l in sphereGridLines(grid_res):
      sP.ax.plot(1.003 * l[:, 0], 1.003 * l[:, 1], 1.003 * l[:, 2], color=GRID3_COLOR, linewidth=0.6)
  if axisLabels:
    for d, name in frameAxes(sP.frame):
      sP.ax.text(*(1.15 * d), name, ha="center", va="center", fontsize=getMTEXpref("FontSize", 13))
  return poly


# the Euler box in degrees at equal scale, ticked every 30 degrees along a side up to 180
# and every 60 beyond, as MATLAB ticks it
def _eulerBox(ax, box):
  from matplotlib.ticker import MultipleLocator
  for axis, b, name in zip((ax.xaxis, ax.yaxis, ax.zaxis), box, ("φ1 / °", "Φ / °", "φ2 / °")):
    axis.set_major_locator(MultipleLocator(30 if b <= 180 else 60))
    axis.set_label_text(name)
  ax.set_xlim(0, box[0])
  ax.set_ylim(0, box[1])
  ax.set_zlim(0, box[2])
  ax.set_box_aspect(tuple(box / box.max()))


def _keys(kw):
  axesKw = {k: kw.pop(k) for k in ("figSize", "parent", "azimuth", "elevation", "title") if k in kw}
  return axesKw


def plot3d(x, *args, **kw: Unpack[PlotKeywords]):
  """plot3d(v), plot3d(v, data), plot3d(sF), plot3d(vF), plot3d(oR), plot3d(ori), plot3d(odf)

  Directions as markers on the sphere in space, a function as the sphere coloured, a field
  as arrows on it, the fundamental region of orientation space as its boundary in axis angle
  space, a list of orientations as points in it, and an ODF as its isosurfaces in the Euler
  box.
  """
  if isinstance(x, orientationRegion):
    return plotRegion3d(x, **kw)
  if isinstance(x, SO3TangentVector):
    return quiver3d(x, *args, **kw)
  if isinstance(x, quaternion):
    return scatterOrientations(x, *args, **kw)
  from ..geometry.fibre import fibre
  if isinstance(x, fibre):
    return plotFibre3d(x, *args, **kw)
  if hasattr(x, "_poleFigure") and hasattr(x, "CS") and hasattr(x, "SS"):
    from .mtexfigure import firstOfList
    return plotODF3d(firstOfList(x), *args, **kw)
  if _isField(x):
    return _quiverField3d(x, **kw)
  if _isFunction(x):
    return plotFunction3d(x, **kw)
  return scatter3d(x, *args, **kw)


def scatter3d(v, data=None, markerSize=6, color=MARKER_COLOR, colormap="viridis", colorRange=None, grid=True, axisLabels=True,
              label=None, colorbar=False, **kw: Unpack[PlotKeywords]):
  """scatter3d(v), plot3d(v, data): directions as markers on the sphere in space"""
  v = v if isinstance(v, vector3d) else vector3d(v)
  axesKw = _keys(kw)
  sP = _spaceAxes(v.frame, **axesKw)
  if not sP.ax.collections:
    _sphere(sP, grid=grid, axisLabels=axisLabels)
  P = 1.01 * v.normalize().data.reshape(-1, 3)
  if v.antipodal:
    P = np.concatenate([P, -P])
  if data is None:
    h = sP.ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=markerSize ** 2, c=[color] if isinstance(color, tuple) else color, edgecolors="k",
                      linewidths=0.4, depthshade=False, **kw)
  else:
    vals = np.asarray(data, dtype=float).reshape(-1)
    vals = np.concatenate([vals, vals]) if v.antipodal else vals
    lo, hi = _rangeOf([vals]) if colorRange is None else colorRange
    h = sP.ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=markerSize ** 2, c=vals, cmap=mtexColormap(colormap), vmin=lo, vmax=hi,
                      edgecolors="k", linewidths=0.4, depthshade=False, **kw)
    sP.mappable = h
    if colorbar:
      sP.ax.figure.colorbar(h, ax=sP.ax, shrink=0.6)
  if label is not None:
    texts = [label] * len(P) if isinstance(label, str) else list(label)
    for p, t in zip(P, texts):
      sP.ax.text(*(1.08 * p), str(t), fontsize=getMTEXpref("FontSize", 13) - 2)
  return h


def plotFunction3d(f, resolution=2.5 * degree, colormap="WhiteJet", colorRange=None, colorbar=False, radial=False, grid=False,
                   axisLabels=None, grid_res=30 * degree, how2plot=None, **kw: Unpack[PlotKeywords]):
  """plot3d(sF, radial=True): a function as the sphere coloured, or pushed out by its values,
  seen as the 3D plotting convention says, with the axes of its frame as arrows"""
  axesKw = _keys(kw)
  V, _ = sphereMesh(resolution)
  fr = getattr(f, "frame", None)
  vals = np.real(np.asarray(f(vector3d(V, frame=fr, isNormalized=True)), dtype=complex)).reshape(-1)
  limits = max(1e-12, 1.1 * np.max(np.abs(vals))) if radial else 1.12
  sP = _spaceAxes(fr, limits=limits, **axesKw)
  h = _sphere(sP, resolution, vals, colormap, colorRange, radial, grid=grid, axisLabels=False, grid_res=grid_res)
  # the axes of the frame as arrows standing on the sphere, as MATLAB draws them
  if (not radial) if axisLabels is None else axisLabels:
    for d, name in frameAxes(fr):
      _axisArrow(sP.ax, d, name)
  # the camera of the convention given, else of the default 3D one
  if "azimuth" not in axesKw and "elevation" not in axesKw:
    from ..geometry.plottingconvention import plottingConvention
    pC = plottingConvention(how2plot) if isinstance(how2plot, str) else how2plot
    _camera(sP.ax, plottingConvention.default3D if pC is None else pC)
  if colorbar:
    sP.ax.figure.colorbar(h, ax=sP.ax, shrink=0.6)
  return h


# a black arrow from the unit sphere outwards along d, its name beyond the tip
def _axisArrow(ax, d, name, length=0.15):
  from mpl_toolkits.mplot3d.art3d import Poly3DCollection
  d = np.asarray(d, dtype=float) / np.linalg.norm(d)
  u = np.cross(d, [0, 0, 1.0] if abs(d[2]) < 0.9 else [1.0, 0, 0])
  u /= np.linalg.norm(u)
  w = np.cross(d, u)
  t = np.linspace(0, 2 * np.pi, 17)[:-1]
  ring = np.cos(t)[:, None] * u + np.sin(t)[:, None] * w
  base, tip = d, (1 + length) * d
  shaft = [np.array([base + 0.012 * ring[i], base + 0.012 * ring[i - 1], base + 0.6 * length * d + 0.012 * ring[i - 1],
                     base + 0.6 * length * d + 0.012 * ring[i]]) for i in range(len(t))]
  cone = [np.array([base + 0.6 * length * d + 0.035 * ring[i], base + 0.6 * length * d + 0.035 * ring[i - 1], tip]) for i in range(len(t))]
  body = Poly3DCollection(shaft + cone, facecolor="k", edgecolor="none")
  ax.add_collection3d(body)
  label = ax.text(*((1 + 1.6 * length) * d), name, ha="center", va="center", fontsize=getMTEXpref("FontSize", 13) - 2, family="serif")
  ax._mtexAxisArrows = getattr(ax, "_mtexAxisArrows", []) + [(d, length, (body, label))]


def surf(sF, layout=None, figSize=None, scaling=True, colormap="parula", resolution=5 * degree, edgeColor=(0.1, 0.1, 0.1),
         **kw: Unpack[PlotKeywords]):
  """surf(sF): the function as the sphere pushed out to |f|, coloured by f, the real part of a
  complex one; a list of functions one panel each; a single real function is first scaled
  to (f - mean) / (max - min) + 1 unless scaling=False, as MATLAB's surf does"""
  from matplotlib.colors import Normalize
  from mpl_toolkits.mplot3d.art3d import Poly3DCollection
  from .mtexfigure import figureSize
  refuseVariants(kw)
  fs = [sF] if not getattr(sF, "shape", ()) else [sF[k] for k in np.ndindex(sF.shape)]
  V, F = sphereMesh(resolution)
  fr = getattr(sF, "frame", None)
  values = [np.real(np.asarray(f(vector3d(V, frame=fr, isNormalized=True)), dtype=complex)).reshape(-1) for f in fs]
  if len(fs) == 1 and getattr(sF, "isReal", True) and scaling:
    a, b = values[0].min(), values[0].max()
    if b > a:
      values[0] = (values[0] - np.mean(values[0])) / (b - a) + 1
  fig = gcm()
  if not (fig is not None and not fig.children and not fig.hold) or len(fs) > 1:
    fig = mtexFigure(len(fs), layout=layout, figSize=figureSize(figSize, (3.2 * min(len(fs), 5), 3.2 * -(-len(fs) // 5))))
  cmap = mtexColormap(colormap)
  hs = []
  for vals in values:
    nextAxis(fig=fig)
    sP = _spaceAxes(fr, decorated=True, **_keys(kw))
    R = np.abs(vals)[:, None] * V
    faceValues = np.mean(vals[F], axis=1)
    # a constant function takes the middle of the colour map, not its rounding noise
    lo, hi = vals.min(), vals.max()
    if hi - lo <= 1e-9 * max(abs(hi), abs(lo), 1e-300):
      lo, hi = lo - 1, hi + 1
    poly = Poly3DCollection(R[F], facecolors=cmap(Normalize(lo, hi)(faceValues)), edgecolors=edgeColor, linewidths=0.2)
    # the colour bar reads the values the faces are coloured by
    poly.set_cmap(cmap)
    poly.set_norm(Normalize(lo, hi))
    poly.set_array(faceValues)
    sP.mappable = poly
    sP.ax.add_collection3d(poly)
    lo, hi = R.min(axis=0), R.max(axis=0)
    sP.ax.set_xlim(lo[0], hi[0])
    sP.ax.set_ylim(lo[1], hi[1])
    sP.ax.set_zlim(lo[2], hi[2])
    sP.ax.set_box_aspect(tuple(np.maximum(hi - lo, 1e-9)))
    hs.append(poly)
  return hs if len(hs) > 1 else hs[0]


def _quiverField3d(vF, resolution=15 * degree, color="k", arrowSize=0.75, grid=True, axisLabels=False, **kw):
  """plot3d(vF): a vector or axis field as arrows on the sphere in space"""
  from ..geometry.grids.s2grid import equispacedS2Grid
  axesKw = _keys(kw)
  fr = getattr(vF, "frame", None)
  fig = gcm()
  held = fig is not None and fig.hold and fig.children and hasattr(fig.children[-1], "_mtexSpacePlot")
  sP = _spaceAxes(fr, limits=1.2, **axesKw)
  # a held plot draws the field onto the sphere already there; a new one is seen as the 3D
  # convention says
  if not held:
    _sphere(sP, grid=grid, axisLabels=axisLabels)
    if "azimuth" not in axesKw and "elevation" not in axesKw:
      from ..geometry.plottingconvention import plottingConvention
      _camera(sP.ax, plottingConvention.default3D)
  # the arrows stand on an equispaced grid of the resolution, as MATLAB's, those of the
  # hemisphere facing the camera, which matplotlib would draw through the sphere
  V = equispacedS2Grid(resolution=resolution).data.reshape(-1, 3)
  e, a = np.radians(sP.ax.elev), np.radians(sP.ax.azim)
  V = V[V @ np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)]) > -0.05]
  W = vF(vector3d(V, frame=fr, isNormalized=True)).data.reshape(-1, 3)
  # the longest arrow spans the grid spacing, the rest in proportion
  scale = 0.9 * resolution * arrowSize / max(np.max(np.linalg.norm(W, axis=1)), 1e-12)
  D = scale * W
  P = 1.01 * V
  if _isAxisField(vF):
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    h = Line3DCollection(np.stack([P - D / 2, P + D / 2], axis=1), colors=color, linewidths=1.0)
    sP.ax.add_collection3d(h)
    return h
  return sP.ax.quiver(P[:, 0], P[:, 1], P[:, 2], D[:, 0], D[:, 1], D[:, 2], color=color, arrow_length_ratio=0.3, linewidths=1.0, **kw)


# ------------------------------------------------------------------ orientation space

def _regionLabels(ax):
  ax.set_xlabel("x / °")
  ax.set_ylabel("y / °")
  ax.set_zlabel("z / °")


# the surface of a fundamental region: light grey, so that the markers inside it show
REGION_COLOR = (0.8, 0.8, 0.8)


# the ball of the whole rotation space in axis angle space, the half turns on its surface,
# as a light wire sphere of radius 180 degrees
def _ball(sP, radius=180.0):
  u, w = np.meshgrid(np.linspace(0, 2 * np.pi, 37), np.linspace(0, np.pi, 19))
  X, Y, Z = radius * np.cos(u) * np.sin(w), radius * np.sin(u) * np.sin(w), radius * np.cos(w)
  sP.ax.plot_surface(X, Y, Z, color=SPHERE_COLOR, alpha=0.05, linewidth=0, shade=False)
  sP.ax.plot_wireframe(X, Y, Z, color=GRID3_COLOR, linewidth=0.4)


def plotRegion3d(oR, resolution=5 * degree, color=REGION_COLOR, alpha=0.25, lines=True, lineColor="k", boundaryColor="k",
                 surface=True, **kw: Unpack[PlotKeywords]):
  """plot(oR), plot(oR, boundaryColor='r', surface=False): the fundamental region in axis
  angle space, in degrees, its surface with the parallels and meridians on it and its edges"""
  axesKw = _keys(kw)
  limit = 1.15 * oR.maxAngle() / degree
  fig = gcm()
  held = fig is not None and fig.hold and fig.children and hasattr(fig.children[-1], "_mtexSpacePlot")
  sP = _spaceAxes(oR.frameA, limits=limit, decorated=True, **axesKw)
  # orientations drawn into the region later come in its coordinates
  if not hasattr(sP, "coordinates"):
    sP.coordinates, sP.reduced = "axisAngle", True
  _regionLabels(sP.ax)
  h = _region(sP, oR, resolution, color, alpha, lines, lineColor, boundaryColor, surface)
  # a new plot looks at the region, as MATLAB fits its axes to it
  if not held:
    _fitRegion(sP, oR)
  return h


# the axes fitted to a region of axis angle space and the points in it, at equal scale; a
# region reaching the half turn keeps the ball around it
def _fitRegion(sP, oR, points=None):
  edges = oR.edges()
  if not edges or oR.maxAngle() >= np.pi - 1e-6:
    return
  P = np.concatenate(edges) / degree
  if points is not None and len(points):
    P = np.concatenate([P, points])
  lo, hi = P.min(axis=0), P.max(axis=0)
  pad = 0.05 * float(np.max(hi - lo))
  lo, hi = lo - pad, hi + pad
  sP.ax.set_xlim(lo[0], hi[0])
  sP.ax.set_ylim(lo[1], hi[1])
  sP.ax.set_zlim(lo[2], hi[2])
  sP.ax.set_box_aspect(tuple(hi - lo))


def _region(sP, oR, resolution=5 * degree, color=REGION_COLOR, alpha=0.25, lines=True, lineColor="k", boundaryColor="k",
            surface=True):
  from mpl_toolkits.mplot3d.art3d import Poly3DCollection
  # the edges where two faces of the region meet, as MATLAB outlines it
  hs = [sP.ax.plot(*(e / degree).T, color=boundaryColor, linewidth=1.5, zorder=3)[0] for e in oR.edges()]
  if not surface:
    return hs
  P, F = regionSurface(oR, resolution)
  P = P / degree
  keep = np.all(np.isfinite(P[F]).reshape(len(F), -1), axis=1)
  poly = Poly3DCollection(P[F[keep]], facecolors=color, alpha=alpha, linewidths=0, zorder=1)
  sP.ax.add_collection3d(poly)
  for w in regionWalls(oR):
    pts = w[np.all(np.isfinite(w), axis=1)] / degree
    if len(pts) >= 3:
      sP.ax.add_collection3d(Poly3DCollection([pts], facecolors=color, alpha=alpha, linewidths=0, zorder=1))
  if lines:
    for l in regionLines(oR):
      q = l / degree
      sP.ax.plot(q[:, 0], q[:, 1], q[:, 2], color=lineColor, alpha=0.5, linewidth=0.6, zorder=2)
  return poly


def orientationPoints(o, coordinates="axisAngle", fundamentalRegion=True):
  """the points orientations are, in the coordinates asked for: axis angle and Euler angles in
  degrees, Rodrigues vectors as they are"""
  if coordinates == "Euler":
    if fundamentalRegion:
      a, b, c = o.flatten().projectIntoEulerRegion()
      return np.stack([a, b, c], axis=1) / degree
    a, b, c = quat2euler(o.data.reshape(-1, 4), "Bunge")
    return np.stack([a, b, c], axis=1) / degree
  q = (o.flatten().projectIntoFundamentalRegion() if fundamentalRegion else o.flatten()).data.reshape(-1, 4)
  if coordinates == "axisAngle":
    omega = quatAngle(q)
    return np.where(omega[:, None] < 1e-12, 0.0, omega[:, None] * quatAxis(q)) / degree
  if coordinates == "Rodrigues":
    return q[:, 1:] / np.where(np.abs(q[:, :1]) < 1e-12, 1e-12, q[:, :1])
  raise ValueError("coordinates are 'axisAngle', 'Euler' or 'Rodrigues'")


# the marker options of a scatter in space
def _markerStyle(kw, color, markerSize):
  refuseVariants(kw)
  color = kw.pop("markerFaceColor", color)
  edge = kw.pop("markerEdgeColor", None)
  lw = kw.pop("lineWidth", None)
  style = {"s": markerSize ** 2, "depthshade": False, "zorder": 5}
  name = kw.pop("displayName", None)
  if name is not None:
    style["label"] = name
  if edge is not None:
    style["edgecolors"] = edge
    style["linewidths"] = 1.0 if lw is None else lw
  return color, style


# the flags of the coordinates of orientation space, as MATLAB names them
def _spaceFlags(args, Euler, axisAngle, Rodrigues):
  data, region, complete = None, None, False
  for a in args:
    if isinstance(a, str):
      flags = {"euler": "Euler", "bunge": "Euler", "axisangle": "axisAngle", "rodrigues": "Rodrigues"}
      if a.lower() in flags:
        Euler, axisAngle, Rodrigues = (flags[a.lower()] == x for x in ("Euler", "axisAngle", "Rodrigues"))
      elif a.lower() == "noboundary":
        region = False
      elif a.lower() == "complete":
        complete = True
      elif a.lower() != "filled":
        raise ValueError("the space is 'axisAngle', 'Euler', 'Bunge' or 'Rodrigues'")
    else:
      data = a
  return data, Euler, axisAngle, Rodrigues, region, complete


def scatterOrientations(o, *args, axisAngle=False, Rodrigues=False, Euler=False, region=True, fundamentalRegion=True,
                        ignoreFundamentalRegion=False, points=None, markerSize=5, color=MARKER_COLOR, colormap="viridis",
                        colorRange=None, colorbar=False, all=False, **kw: Unpack[PlotKeywords]):
  """scatter(ori), plot3d(ori), scatter(ori, 'Rodrigues'), scatter(ori, 'Euler'), scatter(ori, data, 'axisAngle'),
  scatter(ori, all=True)

  Orientations as points in axis angle space, in Rodrigues space or in Euler angle space,
  brought into the fundamental region unless ignoreFundamentalRegion, with the boundary of
  the region behind them in axis angle space unless region=False; the space named by a
  positional string as MATLAB names it, 'Bunge' being 'Euler'. A bare rotation has no
  region: the whole Euler box, the ball of axis angle space.
  """
  data, Euler, axisAngle, Rodrigues, noRegion, complete = _spaceFlags(args, Euler, axisAngle, Rodrigues)
  region = region and noRegion is None and not complete
  ignoreFundamentalRegion = ignoreFundamentalRegion or complete
  color, style = _markerStyle(kw, color, markerSize)
  axesKw = _keys(kw)
  # an orientation with its frames scatters in the Euler region by default, a misorientation
  # and a bare rotation in axis angle space, into held axes in their coordinates, as MATLAB
  # does; a list is sampled to 2000 points unless asked otherwise
  isOri = hasattr(o, "CS")
  fig = gcm()
  heldSP = fig.children[-1]._mtexSpacePlot if fig is not None and fig.hold and fig.children and hasattr(fig.children[-1], "_mtexSpacePlot") else None
  if heldSP is not None and getattr(heldSP, "coordinates", None) and not (Euler or axisAngle or Rodrigues):
    coordinates = heldSP.coordinates
  else:
    euler = Euler or (not axisAngle and not Rodrigues and isOri and not o.isMisorientation)
    coordinates = "Rodrigues" if Rodrigues else "Euler" if euler else "axisAngle"
  reduced = isOri and fundamentalRegion and not ignoreFundamentalRegion
  if heldSP is not None and hasattr(heldSP, "reduced"):
    reduced = heldSP.reduced
  region = region and isOri and not ignoreFundamentalRegion
  # orientations kept as they are are looked at closely, without the region
  zoom = isOri and ignoreFundamentalRegion and not complete
  o = o.flatten()
  # a value or an RGB row per orientation
  data = None if data is None else np.asarray(data).reshape(o.size, -1)
  # an orientation that is not there, a NaN, is left out with its value
  keep = ~np.isnan(o.data).any(axis=-1)
  if not np.all(keep):
    o = o[keep]
    data = None if data is None else data[keep]
  points = o.size if all else 2000 if points is None else points
  if points < o.size:
    print(f"plot {points} random orientations out of {o.size} given orientations")
    take = np.sort(np.random.default_rng(0).permutation(o.size)[:points])
    o = o[take]
    data = None if data is None else data[take]
  P = orientationPoints(o, coordinates, reduced)
  oR = fundamentalRegion_(o) if isOri else None
  frame = o.CS if isOri else None
  fig = gcm()
  held = fig is not None and fig.hold and fig.children and hasattr(fig.children[-1], "_mtexSpacePlot")
  if coordinates == "Euler":
    bounds = np.array(eulerRegion(o.CS, o.SS) if reduced else (2 * np.pi, np.pi, 2 * np.pi)) / degree
    sP = _spaceAxes(frame, limits=(0, float(bounds.max())), decorated=True, **axesKw)
    # held axes keep their box
    if not held:
      _eulerBox(sP.ax, bounds)
  else:
    finite = P[np.all(np.isfinite(P), axis=1)]
    bound = float(np.max(np.abs(finite))) if len(finite) else 1.0
    if coordinates == "axisAngle" and (region or not reduced) and not zoom:
      bound = max(bound, oR.maxAngle() / degree if region else 180.0)
    sP = _spaceAxes(frame, limits=1.1 * bound, decorated=True, **axesKw)
    if zoom and not held and len(finite):
      lo, hi = finite.min(axis=0), finite.max(axis=0)
      pad = 0.05 * max(float(np.max(hi - lo)), 1e-9)
      sP.ax.set_xlim(lo[0] - pad, hi[0] + pad)
      sP.ax.set_ylim(lo[1] - pad, hi[1] + pad)
      sP.ax.set_zlim(lo[2] - pad, hi[2] + pad)
      sP.ax.set_box_aspect(tuple(hi - lo + 2 * pad))
    if coordinates == "axisAngle":
      _regionLabels(sP.ax)
      if region:
        _region(sP, oR)
        if not held:
          _fitRegion(sP, oR, finite)
      elif not reduced and noRegion is None and not zoom:
        _ball(sP)
    else:
      sP.ax.set_xlabel("r1")
      sP.ax.set_ylabel("r2")
      sP.ax.set_zlabel("r3")
  # the coordinates of the axes, for what is drawn into them later
  sP.coordinates, sP.reduced = coordinates, reduced
  if data is None:
    # a further set in held axes takes the next colour of the order, as MATLAB's held scatter
    k = getattr(sP, "markerSets", 0)
    sP.markerSets = k + 1
    if color is MARKER_COLOR:
      # markers of no colour of their own are open circles, as MATLAB draws orientations
      from .colormaps import MTEX_COLORS
      color = tuple(MTEX_COLORS[k % len(MTEX_COLORS)])
      style.setdefault("edgecolors", color)
      h = sP.ax.scatter(P[:, 0], P[:, 1], P[:, 2], facecolors="none", **style)
    else:
      h = sP.ax.scatter(P[:, 0], P[:, 1], P[:, 2], color=color, **style)
  elif data.shape[1] == 3:
    h = sP.ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=np.clip(np.nan_to_num(data.astype(float), nan=1.0), 0, 1), **style)
  else:
    vals = np.asarray(data, dtype=float).reshape(-1)
    lo, hi = _rangeOf([vals]) if colorRange is None else colorRange
    h = sP.ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=vals, cmap=mtexColormap(colormap), vmin=lo, vmax=hi, **style)
    sP.mappable = h
    if colorbar:
      sP.ax.figure.colorbar(h, ax=sP.ax, shrink=0.6)
  return h


def quiver3d(t, *args, lineWidth=1.5, color=None, normalize=False, axisAngle=False, Rodrigues=False, Euler=False, maxHeadSize=None, **kw: Unpack[PlotKeywords]):
  """quiver3d(t): a tangent vector as an arrow at its reference in orientation space, in the
  coordinates of the held axes or axis angle space; the arrow ends where the vector, scaled
  to the longest of the list, points to, as MATLAB draws it; quiver3d(vF) a field on the
  sphere as arrows in space; quiver3d(v, d) arrows at the positions v along d in a 3D scene
  (MATLAB's quiver3)"""
  # arrows at positions in space go into a 3D scene
  if isinstance(t, vector3d) and args and isinstance(args[0], vector3d):
    from .volumeplot import _quiverVectors
    if color is not None:
      kw["color"] = color
    return _quiverVectors(t, args[0], lineWidth=lineWidth if lineWidth != 1.5 else 2.0, **kw)
  # a field on the sphere is drawn as arrows on the sphere in space
  if _isField(t):
    return _quiverField3d(t, color="k" if color is None else color, **kw)
  if type(t).__name__.startswith("SO3VectorField"):
    return _quiverField(t, args, lineWidth, color, normalize, axisAngle, Rodrigues, Euler, kw)
  color = "b" if color is None else color
  _, Euler, axisAngle, Rodrigues, _, _ = _spaceFlags(args, Euler, axisAngle, Rodrigues)
  refuseVariants(kw)
  axesKw = _keys(kw)
  ref = t.oriRef
  isOri = hasattr(ref, "CS")
  fig = gcm()
  held = fig is not None and fig.hold and fig.children and hasattr(fig.children[-1], "_mtexSpacePlot")
  if held:
    sP = fig.children[-1]._mtexSpacePlot
    coordinates, reduced = getattr(sP, "coordinates", "axisAngle"), getattr(sP, "reduced", isOri)
  else:
    coordinates = "Rodrigues" if Rodrigues else "Euler" if Euler else "axisAngle"
    reduced = isOri
    o = ref if ref.ndim else ref.reshape(1)
    scatterOrientations(o, coordinates, markerSize=0.1, **axesKw)
    sP = gcm().children[-1]._mtexSpacePlot
  scale = 1.0 if normalize else float(np.max(t.norm()))
  t = t.normalize() if normalize else t / max(scale, 1e-12)
  P1 = orientationPoints(t.oriRef, coordinates, reduced)
  P2 = orientationPoints(t.exp(), coordinates, reduced)
  D = P2 - P1
  return sP.ax.quiver(P1[:, 0], P1[:, 1], P1[:, 2], D[:, 0], D[:, 1], D[:, 2], color=color, linewidths=lineWidth, arrow_length_ratio=0.25, **kw)


# a field on SO(3): its vectors on an equispaced grid of the fundamental region, drawn in the
# direction they move the orientations and scaled so that the longest spans nine tenths of
# the grid spacing, as MATLAB's quiver scales them
def _quiverField(F, args, lineWidth, color, normalize, axisAngle, Rodrigues, Euler, kw):
  from scipy.spatial import cKDTree
  from ..geometry.grids.so3grid import equispacedSO3Grid
  from ..geometry.orientation import orientation
  color = "b" if color is None else color
  res = kw.pop("resolution", 10 * degree)
  refuseVariants(kw)
  _, Euler, axisAngle, Rodrigues, _, _ = _spaceFlags(args, Euler, axisAngle, Rodrigues)
  coordinates = "Rodrigues" if Rodrigues else "Euler" if Euler else "axisAngle"
  axesKw = _keys(kw)
  g = equispacedSO3Grid(F.CS, F.SS, resolution=res)
  o = orientation._fromRotation(g.data.reshape(-1, 4), (F.CS, F.SS)).projectIntoFundamentalRegion()
  t = F.eval(o)
  fig = gcm()
  held = fig is not None and fig.hold and fig.children and hasattr(fig.children[-1], "_mtexSpacePlot")
  if not held:
    scatterOrientations(o, coordinates, markerSize=0.1, **axesKw)
  sP = gcm().children[-1]._mtexSpacePlot
  coordinates = getattr(sP, "coordinates", coordinates)
  lengths = np.asarray(t.norm(), dtype=float).reshape(-1)
  # the direction by a small step from the representatives in the region, both ends
  # unreduced so that a step over the boundary is not folded back
  step = 1e-4 / max(float(np.max(lengths)), 1e-12)
  P1 = orientationPoints(o, coordinates, False)
  P2 = orientationPoints((t * step).exp(), coordinates, False)
  D = (P2 - P1) / step
  if normalize:
    D = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)
  spacing = float(np.median(cKDTree(P1).query(P1, k=2)[0][:, 1])) if len(P1) > 1 else 1.0
  D = 0.9 * spacing * D / max(float(np.max(np.linalg.norm(D, axis=1))), 1e-12)
  return sP.ax.quiver(P1[:, 0], P1[:, 1], P1[:, 2], D[:, 0], D[:, 1], D[:, 2], color=color, linewidths=lineWidth,
                      arrow_length_ratio=0.3, **kw)


def arrow3d(v, faceColor="k", anchor=None, arrowWidth=None, edgeColor="none", labeled=False, label=None, **kw: Unpack[PlotKeywords]):
  """arrow3d(v): solid arrows from the origin, or from anchor, to the tips v, into the held
  axes or a new figure in space, the axes widened to hold them (MATLAB @vector3d/arrow3d)"""
  from matplotlib.colors import to_rgba
  from mpl_toolkits.mplot3d.art3d import Poly3DCollection
  from .sphericalplot import _nameOf
  v = v if isinstance(v, vector3d) else vector3d(v)
  refuseVariants(kw)
  if labeled and label is None:
    label = [_nameOf(v.flatten()[i]) for i in range(v.size)]
  axesKw = _keys(kw)
  fig = gcm()
  held = fig is not None and fig.hold and fig.children and hasattr(fig.children[-1], "_mtexSpacePlot")
  sP = _spaceAxes(v.frame, **axesKw)
  # fresh axes look down on the screen plane, as MATLAB's new axes do
  if not held and "azimuth" not in axesKw and "elevation" not in axesKw:
    _conventionView(sP.ax, v.frame)
  D = v.data.reshape(-1, 3)
  base = np.zeros_like(D) if anchor is None else np.broadcast_to((anchor.data if isinstance(anchor, vector3d) else np.asarray(anchor, dtype=float)).reshape(-1, 3), D.shape)
  scale = max(float(np.max(np.linalg.norm(D, axis=1))), 1e-12)
  t = np.linspace(0, 2 * np.pi, 25)[:-1]
  faces = []
  for b, d in zip(base, D):
    L = float(np.linalg.norm(d))
    if L < 1e-12:
      continue
    # MATLAB's arrowWidth is the radius of the tail relative to the length, the head 2.5
    # times as wide and a tenth of the length long
    if arrowWidth is None:
      r, head, wide = 0.035 * scale, min(0.2 * scale, 0.5 * L), 2.2
    else:
      r, head, wide = float(arrowWidth) * L, 0.1 * L, 2.5
    n = d / L
    u = np.cross(n, [0, 0, 1.0] if abs(n[2]) < 0.9 else [1.0, 0, 0])
    u /= np.linalg.norm(u)
    w = np.cross(n, u)
    ring = np.cos(t)[:, None] * u + np.sin(t)[:, None] * w
    shaftEnd = b + n * (L - head)
    tip = b + d
    c0, c1, c2 = b + r * ring, shaftEnd + r * ring, shaftEnd + wide * r * ring
    k = np.arange(len(t))
    j = (k + 1) % len(t)
    faces += [[c0[i], c0[j[i]], c1[j[i]], c1[i]] for i in k]
    faces += [[c1[i], c1[j[i]], c2[j[i]], c2[i]] for i in k]
    faces += [[c2[i], c2[j[i]], tip] for i in k]
    faces.append(list(c0[::-1]))
  poly = Poly3DCollection(faces, facecolors=to_rgba(faceColor), shade=True, **patchKeywords(kw))
  if edgeColor != "none":
    poly.set_edgecolor(edgeColor)
  sP.ax.add_collection3d(poly)
  if label is not None:
    texts = [label] * len(D) if isinstance(label, str) else list(label)
    for b, d, t in zip(base, D, texts):
      tip = b + 1.12 * d
      sP.ax.text(tip[0], tip[1], tip[2], str(t), fontsize=getMTEXpref("FontSize", 13), ha="center", va="center")
  # the axes fit every arrow and label drawn so far, at equal scale, as MATLAB's do
  pts = np.concatenate([base, base + (1.25 if label is not None else 1.0) * D])
  lo, hi = pts.min(axis=0) - 0.1 * scale, pts.max(axis=0) + 0.1 * scale
  if held:
    cur = np.array([sP.ax.get_xlim(), sP.ax.get_ylim(), sP.ax.get_zlim()])
    lo, hi = np.minimum(lo, cur[:, 0]), np.maximum(hi, cur[:, 1])
  sP.ax.set_xlim(lo[0], hi[0])
  sP.ax.set_ylim(lo[1], hi[1])
  sP.ax.set_zlim(lo[2], hi[2])
  # arrows alone fill the axes; joining a held plot they leave its framing
  sP.ax.set_box_aspect(tuple(hi - lo), zoom=1.0 if held else 1.3)
  return poly


def plotFibre3d(f, *args, lineWidth=1.5, color="k", axisAngle=False, Rodrigues=False, Euler=False, fundamentalRegion=True,
                ignoreFundamentalRegion=False, region=True, points=1000, **kw: Unpack[PlotKeywords]):
  """plot(f): a fibre as its curve through orientation space, in the coordinates of the held
  axes or Euler angles, as MATLAB draws it"""
  _, Euler, axisAngle, Rodrigues, noRegion, complete = _spaceFlags(args, Euler, axisAngle, Rodrigues)
  refuseVariants(kw)
  color = kw.pop("lineColor", color)
  coordinates = "Rodrigues" if Rodrigues else "axisAngle" if axisAngle else "Euler"
  o = f.flatten().orientation(points) if f.ndim > 0 else f.orientation(points)
  reduced = fundamentalRegion and not ignoreFundamentalRegion and not complete
  P = orientationPoints(o, coordinates, reduced)
  axesKw = _keys(kw)
  fig = gcm()
  held = fig is not None and fig.hold and fig.children and hasattr(fig.children[-1], "_mtexSpacePlot")
  if coordinates == "Euler" or held:
    sP = _spaceAxes(o.CS, limits=(0, float(np.nanmax(np.abs(P)))), decorated=True, **axesKw)
    if not held:
      bounds = np.array(eulerRegion(o.CS, o.SS) if reduced else (2 * np.pi, np.pi, 2 * np.pi)) / degree
      _eulerBox(sP.ax, bounds)
  else:
    # the ball of axis angle space, or the fundamental region with its boundary
    oR = fundamentalRegion_(o)
    bound = float(np.nanmax(np.abs(P)))
    if coordinates == "axisAngle":
      bound = max(bound, (oR.maxAngle() if reduced else np.pi) / degree)
    sP = _spaceAxes(o.CS, limits=1.1 * bound, decorated=True, **axesKw)
    if coordinates == "axisAngle":
      _regionLabels(sP.ax)
      if region and noRegion is None:
        _region(sP, oR) if reduced else _ball(sP)
    else:
      sP.ax.set_xlabel("r1")
      sP.ax.set_ylabel("r2")
      sP.ax.set_zlabel("r3")
  sP.coordinates, sP.reduced = coordinates, reduced
  # a curve that wraps through the region is drawn without the jumps
  jump = np.linalg.norm(np.diff(P, axis=0), axis=1) > 5 * np.nanmedian(np.linalg.norm(np.diff(P, axis=0), axis=1))
  Q = P.copy()
  Q[1:][jump] = np.nan
  label = kw.pop("displayName", None)
  return sP.ax.plot(Q[:, 0], Q[:, 1], Q[:, 2], linewidth=lineWidth, color=color, label=label, **kw)[0]


def plotODF3d(odf, *args, surf3=10, resolution=5 * degree, colormap="parula", axisAngle=False, Rodrigues=False, Euler=False, **kw: Unpack[PlotKeywords]):
  """plot3d(odf), plot3d(odf, 'axisAngle'): the isosurfaces of an ODF at surf3 levels from zero
  to its maximum, over the Euler box of its frames or inside its fundamental region in axis
  angle space, the higher the less transparent (MATLAB @SO3Fun/plot3d)"""
  from matplotlib.cm import ScalarMappable
  from matplotlib.colors import Normalize
  from mpl_toolkits.mplot3d.art3d import Poly3DCollection
  from .space import isosurface
  _, Euler, axisAngle, Rodrigues, noRegion, _ = _spaceFlags(args, Euler, axisAngle, Rodrigues)
  if Rodrigues:
    raise ValueError("the isosurfaces of an ODF are drawn in Euler angle or axis angle space")
  axesKw = _keys(kw)
  if axisAngle:
    # a cube of axis angle vectors about the region, the values outside the region zero
    oR = fundamentalRegion(odf.CS, odf.SS, antipodal=odf.antipodal)
    w = oR.maxAngle()
    axes = [np.linspace(-w, w, int(np.ceil(2 * w / resolution)) + 1)] * 3
    E = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    omega = np.linalg.norm(E, axis=-1)
    n = E / np.where(omega > 0, omega, 1)[..., None]
    q = np.concatenate([np.cos(omega / 2)[..., None], np.sin(omega / 2)[..., None] * n], axis=-1)
    ori = orientation._fromRotation(q.reshape(-1, 4), (odf.CS, odf.SS), odf.antipodal)
    inside = (omega.reshape(-1) <= w) & oR.checkInside(ori)
    V = np.zeros(inside.size)
    V[inside] = np.real(np.asarray(odf(ori[inside]))).reshape(-1)
    V = V.reshape(omega.shape)
    sP = _spaceAxes(odf.CS, limits=1.1 * w / degree, decorated=True, **axesKw)
    _regionLabels(sP.ax)
    if noRegion is None:
      _region(sP, oR)
    sP.coordinates = "axisAngle"
  else:
    bounds = np.array(eulerRegion(odf.CS, odf.SS))
    axes = [np.arange(0, b + 1e-9, resolution) for b in bounds]
    E = np.meshgrid(*axes, indexing="ij")
    V = np.real(np.asarray(odf(orientation.byEuler(E[0], E[1], E[2], "Bunge", odf.CS, odf.SS)))).reshape(E[0].shape)
    sP = _spaceAxes(odf.CS, limits=(0, float(bounds.max() / degree)), decorated=True, **axesKw)
    _eulerBox(sP.ax, bounds / degree)
    sP.coordinates = "Euler"
  sP.reduced = True
  top = float(np.max(V))
  levels = np.linspace(0, top, int(surf3)) if np.isscalar(surf3) else np.asarray(surf3, dtype=float)
  deg = [a / degree for a in axes]
  hs, hLevels = [], []
  for v in levels:
    T = isosurface(*deg, V, v)
    if len(T) == 0:
      continue
    h = Poly3DCollection(T, edgecolors="none")
    sP.ax.add_collection3d(h)
    hs.append(h)
    hLevels.append(v)
  # a level takes its colour from the map, its opacity from its height
  sP.mappable = ScalarMappable(Normalize(0, top), mtexColormap(colormap))

  def refill(colormap=None, colorRange=None):
    if colormap is not None:
      sP.mappable.set_cmap(mtexColormap(colormap) if isinstance(colormap, str) else colormap)
    if colorRange is not None:
      sP.mappable.set_clim(*colorRange)
    lo, hi = sP.mappable.get_clim()
    for h, v in zip(hs, hLevels):
      rgba = list(sP.mappable.to_rgba(v))
      rgba[3] = float(np.clip((v - lo) / (hi - lo), 0, 1)) if hi > lo else 0.0
      h.set_facecolor([rgba])

  sP.refill = refill
  refill()
  return hs


def fundamentalRegion_(o):
  """the orientation region of the frames of orientations o, their exchange symmetry included"""
  return fundamentalRegion(o.CS, o.SS, antipodal=o.antipodal)


__all__ = ["plot3d", "scatter3d", "spacePlot", "arrow3d", "quiver3d", "surf", "setCamera"]
